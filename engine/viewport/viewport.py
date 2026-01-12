"""
Viewport System

Manages viewport areas with:
- Render-to-texture (always - this is non-negotiable for compositing)
- Async extraction via snapshots (thread-safe)
- Multi-camera sub-viewports with scissors
- Picking with (camera_index, entity_id) results
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, NamedTuple
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import time
import math

from engine.core.frame import FrameState
from engine.core.snapshot import (
    GraphSnapshot, SnapshotBuilder, DirtyFlags,
    CameraViewport, SnapshotCamera
)
from engine.render.commands import CommandList, CmdClear, merge_command_lists
from engine.render.world import build_render_commands
from engine.render.targets import RenderTargetPool, RenderTargetSpec, RenderTarget


# =============================================================================
# Panel Layout
# =============================================================================

@dataclass
class PanelRect:
    """Rectangle for a panel in screen space (top-left origin)."""
    id: str
    x: int
    y: int
    w: int
    h: int
    
    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < (self.x + self.w) and self.y <= py < (self.y + self.h)


class AreaLayout:
    """Manages panel layout within a window."""
    
    def __init__(self):
        self.window_w = 0
        self.window_h = 0
        self.current_panels: Dict[str, PanelRect] = {}
    
    def set_window_size(self, w: int, h: int):
        self.window_w, self.window_h = w, h
    
    def compute_panels(self, panels: List[PanelRect]) -> Dict[str, PanelRect]:
        self.current_panels = {p.id: p for p in panels}
        return self.current_panels


# =============================================================================
# Camera
# =============================================================================

@dataclass
class Camera:
    """
    Mutable camera state (for interactive editing).
    Gets snapshotted to immutable SnapshotCamera for rendering.
    """
    name: str
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    fov_y_deg: float = 60.0
    near: float = 0.05
    far: float = 200.0


# =============================================================================
# Pick Result
# =============================================================================

class PickResult(NamedTuple):
    """Result of a picking operation."""
    entity_id: int
    camera_index: int
    local_x: int  # X within camera viewport
    local_y: int  # Y within camera viewport


# =============================================================================
# Shared Thread Pool
# =============================================================================

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="render_worker")


# =============================================================================
# Viewport Area
# =============================================================================

class ViewportArea:
    """
    A viewport area that renders to its own texture.
    
    Key invariants:
    - Always renders to texture (never directly to screen)
    - Async extraction via immutable snapshots
    - Multi-camera support with sub-viewports
    - Picking returns (camera_index, entity_id)
    """
    
    def __init__(self, area_id: str, graph, cameras: List[Camera]):
        self.area_id = area_id
        self.graph = graph  # EntityNode - the live, mutable graph
        self.cameras = cameras
        
        # Render target
        self.surface_rt: Optional[RenderTarget] = None
        self._rt_spec: Optional[RenderTargetSpec] = None
        
        # Async extraction state
        self._latest_cmdlists: Optional[List[CommandList]] = None
        self._pending_future: Optional[Future] = None
        self._camera_viewports: List[CameraViewport] = []
        
        # Dirty tracking
        self._dirty_flags = DirtyFlags.ALL
        self._last_extract_t = 0.0
        self._min_extract_interval = 1.0 / 60.0  # Cap at 60 extracts/sec
        
        # Interaction state
        self._dragging = False
        self._last_mouse_local: Optional[Tuple[float, float]] = None
        self._active_camera_index = -1  # Which camera is being orbited (-1 = last)
        
        # Debug stats
        self._last_snapshot_id = 0
        self._extract_count = 0
    
    # -------------------------------------------------------------------------
    # Dirty Tracking
    # -------------------------------------------------------------------------
    
    def mark_dirty(self, reason: str = "unknown"):
        """Mark viewport as needing re-extraction."""
        flag_map = {
            "graph": DirtyFlags.GRAPH,
            "transform": DirtyFlags.TRANSFORM,
            "material": DirtyFlags.MATERIAL,
            "camera": DirtyFlags.CAMERA,
            "resize": DirtyFlags.RESIZE,
        }
        self._dirty_flags |= flag_map.get(reason, DirtyFlags.ALL)
    
    def is_dirty(self) -> bool:
        return self._dirty_flags != DirtyFlags.NONE
    
    def clear_dirty(self):
        self._dirty_flags = DirtyFlags.NONE
    
    # -------------------------------------------------------------------------
    # Surface Management
    # -------------------------------------------------------------------------
    
    def ensure_surface(self, w: int, h: int, pool: RenderTargetPool, want_picking: bool = False):
        """Ensure render target exists and is correct size."""
        spec = RenderTargetSpec(w=w, h=h, depth=True, picking=want_picking)
        
        needs_new = (
            self._rt_spec is None or
            spec.w != self._rt_spec.w or
            spec.h != self._rt_spec.h or
            spec.picking != self._rt_spec.picking
        )
        
        if needs_new:
            if self.surface_rt is not None:
                pool.release(self.surface_rt)
            self.surface_rt = pool.acquire(spec)
            self._rt_spec = spec
            self.mark_dirty("resize")
    
    # -------------------------------------------------------------------------
    # Async Extraction (Core Architecture)
    # -------------------------------------------------------------------------
    
    def kick_extract_if_needed(self, frame: FrameState, panel_rect: PanelRect):
        """
        Kick off async extraction if needed.
        
        This is the key async entry point:
        1. Check dirty flags
        2. Take a SNAPSHOT of the graph (freezes current state)
        3. Submit snapshot to worker thread for command generation
        4. Worker produces CommandLists from the frozen snapshot
        """
        # Don't extract if not dirty
        if not self.is_dirty():
            return
        
        # Don't stack up pending work
        if self._pending_future and not self._pending_future.done():
            return
        
        # Rate limit extractions
        now = time.perf_counter()
        if now - self._last_extract_t < self._min_extract_interval:
            return
        
        # === CRITICAL: Take snapshot on main thread ===
        # This freezes the graph state. Workers only see this frozen data.
        snapshot = SnapshotBuilder.from_graph(
            graph_root=self.graph,
            cameras=self.cameras,
            viewport_w=int(panel_rect.w),
            viewport_h=int(panel_rect.h),
            dirty_flags=self._dirty_flags,
        )
        
        # Clear dirty flags now that we've captured state
        self.clear_dirty()
        self._last_extract_t = now
        self._extract_count += 1
        
        # Submit to worker thread - worker gets immutable snapshot
        self._pending_future = _executor.submit(
            _worker_build_commands,
            snapshot,
        )
    
    def render_if_ready(self, renderer):
        """
        Check for completed extraction and render if ready.
        Called from GL thread.
        """
        if not self.surface_rt:
            return
        
        # Check if async work completed
        if self._pending_future and self._pending_future.done():
            try:
                result = self._pending_future.result()
                self._latest_cmdlists = result.cmdlists
                self._camera_viewports = result.camera_viewports
                self._last_snapshot_id = result.snapshot_id
            except Exception as e:
                print(f"[ViewportArea {self.area_id}] Extract error: {e}")
            self._pending_future = None
        
        # Render using latest commands
        if not self._latest_cmdlists:
            return
        
        # Merge per-camera command lists with a clear at the start
        clear = CmdClear(0.12, 0.13, 0.16, 1.0, depth=True, clear_pick=True)
        merged = merge_command_lists(self._latest_cmdlists, prepend_clear=clear)
        
        # Execute to render target
        renderer.execute_to_target(merged, self.surface_rt)
    
    # -------------------------------------------------------------------------
    # Picking
    # -------------------------------------------------------------------------
    
    def pick_at(self, renderer, local_x: float, local_y: float) -> Optional[PickResult]:
        """
        Pick entity at local coordinates.
        
        Returns (entity_id, camera_index) or None.
        """
        if not self.surface_rt or self.surface_rt.pick is None:
            return None
        
        ix, iy = int(local_x), int(local_y)
        
        # Find which camera viewport contains this point
        camera_index = self._find_camera_at(ix, iy)
        
        # Read pick ID from GPU
        entity_id = renderer.read_pick_id(self.surface_rt, ix, iy)
        
        if entity_id == 0:
            return None
        
        # Calculate local position within camera viewport
        cam_local_x, cam_local_y = ix, iy
        if camera_index >= 0 and camera_index < len(self._camera_viewports):
            vp = self._camera_viewports[camera_index]
            cam_local_x = ix - vp.x
            cam_local_y = iy - vp.y
        
        return PickResult(
            entity_id=entity_id,
            camera_index=camera_index,
            local_x=cam_local_x,
            local_y=cam_local_y,
        )
    
    def _find_camera_at(self, x: int, y: int) -> int:
        """Find which camera viewport contains point (x, y)."""
        for vp in self._camera_viewports:
            if vp.contains(x, y):
                return vp.camera_index
        return 0  # Default to first camera
    
    # -------------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------------
    
    def handle_pointer(self, kind: str, x: float, y: float, dx: float, dy: float, button=None):
        """Handle mouse/pointer events."""
        if kind == "press":
            self._dragging = True
            self._last_mouse_local = (x, y)
            # Determine which camera to orbit
            self._active_camera_index = self._find_camera_at(int(x), int(y))
        
        elif kind == "release":
            self._dragging = False
            self._active_camera_index = -1
        
        elif kind == "move" and self._dragging and self._last_mouse_local:
            lx, ly = self._last_mouse_local
            self._orbit(x - lx, y - ly)
            self._last_mouse_local = (x, y)
            self.mark_dirty("camera")
    
    def handle_wheel(self, y_offset: float, rect: PanelRect):
        """Handle mouse wheel for zoom."""
        self._zoom(y_offset * 120.0)
        self.mark_dirty("camera")
    
    def _get_active_camera(self) -> Optional[Camera]:
        """Get the camera being manipulated."""
        if not self.cameras:
            return None
        
        if 0 <= self._active_camera_index < len(self.cameras):
            return self.cameras[self._active_camera_index]
        
        # Default to last camera
        return self.cameras[-1]
    
    def _orbit(self, dx: float, dy: float):
        """Orbit the active camera around its target."""
        cam = self._get_active_camera()
        if not cam:
            return
        
        eye = cam.eye - cam.target
        r = float(np.linalg.norm(eye) + 1e-6)
        theta = math.atan2(float(eye[0]), float(eye[2]))
        phi = math.acos(max(-1.0, min(1.0, float(eye[1]) / r)))
        
        theta += dx * 0.01
        phi = max(0.15, min(3.0, phi + dy * 0.01))
        
        cam.eye = cam.target + np.array([
            r * math.sin(phi) * math.sin(theta),
            r * math.cos(phi),
            r * math.sin(phi) * math.cos(theta),
        ], dtype=np.float32)
    
    def _zoom(self, wheel_delta: float):
        """Zoom the active camera."""
        cam = self._get_active_camera()
        if not cam:
            return
        
        direction = cam.eye - cam.target
        r = float(np.linalg.norm(direction) + 1e-6)
        r *= (0.90 if wheel_delta > 0 else 1.10)
        r = max(0.1, min(1000.0, r))  # Clamp zoom range
        
        cam.eye = cam.target + (direction / max(1e-6, np.linalg.norm(direction))) * r
    
    # -------------------------------------------------------------------------
    # Debug Info
    # -------------------------------------------------------------------------
    
    def get_debug_info(self) -> Dict:
        """Get debug information about this viewport."""
        return {
            "area_id": self.area_id,
            "dirty_flags": str(self._dirty_flags),
            "extract_count": self._extract_count,
            "last_snapshot_id": self._last_snapshot_id,
            "camera_count": len(self.cameras),
            "has_pending_work": self._pending_future is not None and not self._pending_future.done(),
            "has_commands": self._latest_cmdlists is not None,
            "rt_size": (self._rt_spec.w, self._rt_spec.h) if self._rt_spec else None,
        }


# =============================================================================
# Worker Function (runs on thread pool)
# =============================================================================

@dataclass
class ExtractResult:
    """Result from async command extraction."""
    cmdlists: List[CommandList]
    camera_viewports: List[CameraViewport]
    snapshot_id: int


def _worker_build_commands(snapshot: GraphSnapshot) -> ExtractResult:
    """
    Worker function that builds CommandLists from a snapshot.
    
    This runs on a worker thread. It ONLY accesses the immutable snapshot,
    never the live graph or any GL objects.
    """
    cmdlists, viewports = build_render_commands(
        snapshot,
        use_instancing=True,
        instancing_threshold=4,
    )
    
    return ExtractResult(
        cmdlists=cmdlists,
        camera_viewports=viewports,
        snapshot_id=snapshot.snapshot_id,
    )

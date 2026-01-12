from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np
import time

from engine.core.frame import FrameState
from engine.render.commands import CommandList, CmdClear, CmdSetViewport, CmdSetScissor, CmdDrawMesh
from engine.render.targets import RenderTargetPool, RenderTargetSpec, RenderTarget
from engine.scene.graph import RenderItem, EntityNode

@dataclass
class PanelRect:
    id: str
    x: int
    y: int
    w: int
    h: int
    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < (self.x + self.w) and self.y <= py < (self.y + self.h)

class AreaLayout:
    def __init__(self):
        self.window_w = 0
        self.window_h = 0
        self.current_panels: Dict[str, PanelRect] = {}
    def set_window_size(self, w: int, h: int):
        self.window_w, self.window_h = w, h
    def compute_panels(self, panels: List[PanelRect]) -> Dict[str, PanelRect]:
        self.current_panels = {p.id: p for p in panels}
        return self.current_panels

@dataclass
class Camera:
    name: str
    eye: np.ndarray
    target: np.ndarray
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32))
    fov_y_deg: float = 60.0
    near: float = 0.05
    far: float = 200.0
    def view(self) -> np.ndarray:
        return look_at(self.eye, self.target, self.up)
    def proj(self, aspect: float) -> np.ndarray:
        return perspective(self.fov_y_deg, aspect, self.near, self.far)

_executor = ThreadPoolExecutor(max_workers=4)

class ViewportArea:
    def __init__(self, area_id: str, graph: EntityNode, cameras: List[Camera]):
        self.area_id = area_id
        self.graph = graph
        self.cameras = cameras

        self.surface_rt: Optional[RenderTarget] = None
        self._rt_spec: Optional[RenderTargetSpec] = None

        self._latest_cmdlists: Optional[List[CommandList]] = None
        self._pending_future: Optional[Future] = None
        self._camera_viewports: List[Tuple[int,int,int,int]] = []

        self._dirty = True
        self._last_extract_t = 0.0
        self._min_extract_interval = 1.0 / 60.0

        # interaction
        self._dragging = False
        self._last_mouse_local = None

    def mark_dirty(self, reason: str = "unknown"):
        self._dirty = True

    def ensure_surface(self, w: int, h: int, pool: RenderTargetPool, want_picking: bool = False):
        spec = RenderTargetSpec(w=w, h=h, depth=True, picking=want_picking)
        if self._rt_spec is None or (spec.w, spec.h, spec.picking) != (self._rt_spec.w, self._rt_spec.h, self._rt_spec.picking):
            if self.surface_rt is not None:
                pool.release(self.surface_rt)
            self.surface_rt = pool.acquire(spec)
            self._rt_spec = spec
            self.mark_dirty("resize")

    def kick_extract_if_needed(self, frame: FrameState, panel_rect: PanelRect):
        if not self._dirty:
            return
        if self._pending_future and not self._pending_future.done():
            return
        now = time.perf_counter()
        if now - self._last_extract_t < self._min_extract_interval:
            return

        self._dirty = False
        self._last_extract_t = now

        items = self.graph.extract_render_items()
        w = int(panel_rect.w)
        h = int(panel_rect.h)
        cams = [
            Camera(
                name=c.name,
                eye=c.eye.copy(),
                target=c.target.copy(),
                up=c.up.copy(),
                fov_y_deg=c.fov_y_deg,
                near=c.near,
                far=c.far,
            )
            for c in (self.cameras or [])
        ]
        self._pending_future = _executor.submit(build_cmdlists_from_items, items, cams, w, h)

    def render_if_ready(self, renderer):
        if not self.surface_rt:
            return
        if self._pending_future and self._pending_future.done():
            try:
                cmdlists, camera_viewports = self._pending_future.result()
                self._latest_cmdlists = cmdlists
                self._camera_viewports = camera_viewports
            except Exception as e:
                print("Extract error:", e)
            self._pending_future = None

        if not self._latest_cmdlists:
            return

        merged = CommandList()
        merged.add(CmdClear(0.12, 0.13, 0.16, 1.0, depth=True, clear_pick=True))
        for cl in self._latest_cmdlists:
            merged.commands.extend(cl.commands)
        renderer.execute_to_target(merged, self.surface_rt)

    def handle_pointer(self, kind: str, x: float, y: float, dx: float, dy: float, button=None):
        if kind == "press":
            self._dragging = True
            self._last_mouse_local = (x, y)
        elif kind == "release":
            self._dragging = False
        elif kind == "move" and self._dragging and self._last_mouse_local:
            lx, ly = self._last_mouse_local
            self._orbit(x - lx, y - ly)
            self._last_mouse_local = (x, y)
            self.mark_dirty("camera")

    def handle_wheel(self, y_offset: float, rect: PanelRect):
        self._zoom(y_offset * 120.0)
        self.mark_dirty("camera")

    def _orbit(self, dx: float, dy: float):
        if not self.cameras:
            return
        cam = self.cameras[-1]
        eye = cam.eye - cam.target
        r = float(np.linalg.norm(eye) + 1e-6)
        theta = math_atan2(eye[0], eye[2])
        phi = math_acos_clamped(eye[1] / r)
        theta += dx * 0.01
        phi = np.clip(phi + dy * 0.01, 0.15, 3.0)
        cam.eye = cam.target + np.array([
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi),
            r * np.sin(phi) * np.cos(theta),
        ], dtype=np.float32)

    def _zoom(self, wheel_delta: float):
        if not self.cameras:
            return
        cam = self.cameras[-1]
        direction = cam.eye - cam.target
        r = float(np.linalg.norm(direction) + 1e-6)
        r *= (0.90 if wheel_delta > 0 else 1.10)
        cam.eye = cam.target + (direction / max(1e-6, np.linalg.norm(direction))) * r

    def pick_at(self, renderer, local_x: float, local_y: float) -> Optional[int]:
        if not self.surface_rt or self.surface_rt.pick is None:
            return None
        eid = renderer.read_pick_id(self.surface_rt, int(local_x), int(local_y))
        return eid if eid != 0 else None

def build_cmdlists_from_items(items: List[RenderItem], cameras: List[Camera], w: int, h: int):
    viewports = compute_camera_viewports(len(cameras), w, h)
    cmdlists: List[CommandList] = []

    for cam, vp in zip(cameras, viewports):
        vx, vy, vw, vh = vp
        aspect = max(1e-6, vw / max(1.0, float(vh)))
        V = cam.view()
        P = cam.proj(aspect)

        cl = CommandList()
        gl_y = h - (vy + vh)
        cl.add(CmdSetViewport(vx, gl_y, vw, vh))
        cl.add(CmdSetScissor(vx, gl_y, vw, vh, enabled=True))

        for it in items:
            mvp = (P @ V @ it.world).astype(np.float32)
            cl.add(CmdDrawMesh(
                pipeline_id=it.pipeline_id,
                mesh_id=it.mesh_id,
                uniforms={"u_mvp": mvp, "u_color": it.color},
                entity_id=it.entity_id
            ))

        cl.add(CmdSetScissor(0, 0, 0, 0, enabled=False))
        cmdlists.append(cl)

    return cmdlists, viewports

def compute_camera_viewports(n: int, w: int, h: int) -> List[Tuple[int,int,int,int]]:
    if n <= 1:
        return [(0, 0, w, h)]
    if n == 2:
        return [(0, 0, w//2, h), (w//2, 0, w - w//2, h)]
    if n <= 4:
        hw, hh = w//2, h//2
        vps = [(0, 0, hw, hh), (hw, 0, w-hw, hh), (0, hh, hw, h-hh), (hw, hh, w-hw, h-hh)]
        return vps[:n]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = w // cols
    cell_h = h // rows
    out = []
    for i in range(n):
        c = i % cols
        r = i // cols
        x = c * cell_w
        y = r * cell_h
        ww = cell_w if c < cols-1 else (w - x)
        hh = cell_h if r < rows-1 else (h - y)
        out.append((x, y, ww, hh))
    return out

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-6)
    u = up / (np.linalg.norm(up) + 1e-6)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-6)
    u2 = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u2
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye[:3]
    return M @ T

def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / max(1e-6, aspect)
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = (2 * far * near) / (near - far)
    M[3, 2] = -1.0
    return M

def math_atan2(y, x):
    import math
    return math.atan2(y, x)

def math_acos_clamped(v):
    import math
    return math.acos(max(-1.0, min(1.0, float(v))))

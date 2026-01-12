"""
Graph Snapshotting System (RCU-style)

This module provides thread-safe scene graph extraction through immutable snapshots.
Workers ONLY operate on GraphSnapshot objects, never on the live EntityGraph.

Key principles:
1. Snapshots are immutable once created
2. Snapshot creation happens on the main/sim thread (owns the graph)
3. Workers receive snapshot handles and produce CommandLists
4. No locks needed during worker traversal - data is frozen
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, FrozenSet
from enum import Flag, auto
import numpy as np
import threading
import time


class DirtyFlags(Flag):
    """Reasons why a viewport needs re-extraction."""
    NONE = 0
    GRAPH = auto()      # Entity added/removed/reparented
    TRANSFORM = auto()  # Any transform changed
    MATERIAL = auto()   # Color/shader/texture changed
    CAMERA = auto()     # Camera moved or settings changed
    RESIZE = auto()     # Viewport dimensions changed
    
    # Convenience combinations
    SCENE = GRAPH | TRANSFORM | MATERIAL
    ALL = GRAPH | TRANSFORM | MATERIAL | CAMERA | RESIZE


@dataclass(frozen=True)
class SnapshotRenderItem:
    """
    Immutable render item extracted from the scene graph.
    Contains all data needed to emit draw commands.
    """
    entity_id: int
    pipeline_id: str
    mesh_id: str
    world_matrix: bytes  # Stored as bytes for true immutability (4x4 float32)
    color: Tuple[float, float, float, float]
    sort_key: int = 0  # For front-to-back or material sorting
    
    def get_world_matrix(self) -> np.ndarray:
        """Reconstruct numpy array from frozen bytes."""
        return np.frombuffer(self.world_matrix, dtype=np.float32).reshape(4, 4).copy()


@dataclass(frozen=True)
class SnapshotCamera:
    """Immutable camera state for command building."""
    index: int
    name: str
    eye: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov_y_deg: float
    near: float
    far: float
    
    def get_eye(self) -> np.ndarray:
        return np.array(self.eye, dtype=np.float32)
    
    def get_target(self) -> np.ndarray:
        return np.array(self.target, dtype=np.float32)
    
    def get_up(self) -> np.ndarray:
        return np.array(self.up, dtype=np.float32)


@dataclass(frozen=True)
class CameraViewport:
    """Camera's sub-region within the viewport texture."""
    camera_index: int
    x: int
    y: int
    w: int
    h: int
    
    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < (self.x + self.w) and self.y <= py < (self.y + self.h)


@dataclass(frozen=True)
class GraphSnapshot:
    """
    Complete immutable snapshot of everything needed to render a frame.
    
    Created on main thread, consumed by worker threads.
    Once created, this object is completely thread-safe to read.
    """
    snapshot_id: int
    timestamp: float
    viewport_w: int
    viewport_h: int
    items: Tuple[SnapshotRenderItem, ...]
    cameras: Tuple[SnapshotCamera, ...]
    camera_viewports: Tuple[CameraViewport, ...]
    dirty_flags: DirtyFlags  # What triggered this snapshot
    
    def __post_init__(self):
        # Validate immutability at creation time
        assert isinstance(self.items, tuple)
        assert isinstance(self.cameras, tuple)
        assert isinstance(self.camera_viewports, tuple)


class SnapshotBuilder:
    """
    Builds GraphSnapshot objects from live scene data.
    
    This class is NOT thread-safe - it should only be used from the main thread.
    Its purpose is to efficiently extract and freeze scene state.
    """
    
    _next_snapshot_id = 0
    _snapshot_lock = threading.Lock()
    
    @classmethod
    def _get_next_id(cls) -> int:
        with cls._snapshot_lock:
            sid = cls._next_snapshot_id
            cls._next_snapshot_id += 1
            return sid
    
    @staticmethod
    def from_graph(
        graph_root,  # EntityNode
        cameras: List,  # List[Camera]
        viewport_w: int,
        viewport_h: int,
        dirty_flags: DirtyFlags = DirtyFlags.ALL
    ) -> GraphSnapshot:
        """
        Extract an immutable snapshot from a live scene graph.
        
        This MUST be called from the main/sim thread while the graph is stable.
        The returned snapshot can then be safely passed to worker threads.
        """
        timestamp = time.perf_counter()
        snapshot_id = SnapshotBuilder._get_next_id()
        
        # Extract render items with world matrices
        items = SnapshotBuilder._extract_items(graph_root)
        
        # Snapshot camera states
        snapshot_cameras = SnapshotBuilder._snapshot_cameras(cameras)
        
        # Compute camera sub-viewports
        camera_viewports = SnapshotBuilder._compute_camera_viewports(
            len(cameras), viewport_w, viewport_h
        )
        
        return GraphSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            items=tuple(items),
            cameras=tuple(snapshot_cameras),
            camera_viewports=tuple(camera_viewports),
            dirty_flags=dirty_flags,
        )
    
    @staticmethod
    def _extract_items(root) -> List[SnapshotRenderItem]:
        """Traverse graph and extract frozen render items."""
        from engine.scene.graph import compose_matrix
        
        items: List[SnapshotRenderItem] = []
        
        def visitor(node, world_m: np.ndarray):
            if node.mesh is None:
                return
            
            # Freeze the world matrix as bytes
            world_bytes = world_m.astype(np.float32).tobytes()
            
            # Freeze color as tuple
            color = tuple(float(c) for c in node.mesh.color[:4])
            
            # Compute sort key (could be depth, material ID, etc.)
            # For now, just use entity_id for stable sorting
            sort_key = node.entity_id
            
            items.append(SnapshotRenderItem(
                entity_id=node.entity_id,
                pipeline_id=node.mesh.pipeline_id,
                mesh_id=node.mesh.mesh_id,
                world_matrix=world_bytes,
                color=color,
                sort_key=sort_key,
            ))
        
        # Use the graph's traverse method
        root.traverse(visitor)
        return items
    
    @staticmethod
    def _snapshot_cameras(cameras: List) -> List[SnapshotCamera]:
        """Create immutable camera snapshots."""
        result = []
        for i, cam in enumerate(cameras):
            result.append(SnapshotCamera(
                index=i,
                name=cam.name,
                eye=tuple(float(v) for v in cam.eye[:3]),
                target=tuple(float(v) for v in cam.target[:3]),
                up=tuple(float(v) for v in cam.up[:3]),
                fov_y_deg=float(cam.fov_y_deg),
                near=float(cam.near),
                far=float(cam.far),
            ))
        return result
    
    @staticmethod
    def _compute_camera_viewports(n: int, w: int, h: int) -> List[CameraViewport]:
        """Compute sub-viewport rects for multi-camera layouts."""
        if n <= 1:
            return [CameraViewport(0, 0, 0, w, h)]
        if n == 2:
            return [
                CameraViewport(0, 0, 0, w // 2, h),
                CameraViewport(1, w // 2, 0, w - w // 2, h),
            ]
        if n <= 4:
            hw, hh = w // 2, h // 2
            vps = [
                CameraViewport(0, 0, 0, hw, hh),
                CameraViewport(1, hw, 0, w - hw, hh),
                CameraViewport(2, 0, hh, hw, h - hh),
                CameraViewport(3, hw, hh, w - hw, h - hh),
            ]
            return vps[:n]
        
        # General grid layout
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        cell_w = w // cols
        cell_h = h // rows
        result = []
        for i in range(n):
            c = i % cols
            r = i // cols
            x = c * cell_w
            y = r * cell_h
            vw = cell_w if c < cols - 1 else (w - x)
            vh = cell_h if r < rows - 1 else (h - y)
            result.append(CameraViewport(i, x, y, vw, vh))
        return result


# Convenience type aliases
SnapshotHandle = GraphSnapshot  # In the future, could be a reference-counted handle

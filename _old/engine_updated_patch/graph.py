from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np

_next_entity_id = 1
def next_entity_id() -> int:
    global _next_entity_id
    eid = _next_entity_id
    _next_entity_id += 1
    return eid

@dataclass
class Transform:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    rot_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))

@dataclass
class MeshRenderer:
    mesh_id: str
    pipeline_id: str
    color: np.ndarray

@dataclass(frozen=True)
class RenderItem:
    entity_id: int
    pipeline_id: str
    mesh_id: str
    world: np.ndarray
    color: np.ndarray

class EntityNode:
    def __init__(self, name: str):
        self.name = name
        self.entity_id = next_entity_id()
        self.children: List[EntityNode] = []
        self.parent: Optional[EntityNode] = None

        self.transform: Optional[Transform] = None
        self.mesh: Optional[MeshRenderer] = None

        self._demo_spin_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._demo_spin_t = 0.0

    def add_child(self, node: "EntityNode"):
        node.parent = self
        self.children.append(node)

    def traverse(self, fn: Callable[["EntityNode", np.ndarray], None], parent_m: Optional[np.ndarray]=None):
        if parent_m is None:
            parent_m = np.eye(4, dtype=np.float32)
        local = compose_matrix(self.transform) if self.transform else np.eye(4, dtype=np.float32)
        world = parent_m @ local
        fn(self, world)
        for ch in self.children:
            ch.traverse(fn, world)

    def extract_render_items(self) -> List[RenderItem]:
        items: List[RenderItem] = []
        def visitor(node: EntityNode, world_m: np.ndarray):
            if node.mesh is None:
                return
            items.append(RenderItem(
                entity_id=node.entity_id,
                pipeline_id=node.mesh.pipeline_id,
                mesh_id=node.mesh.mesh_id,
                world=world_m.astype(np.float32),
                color=node.mesh.color.astype(np.float32),
            ))
        self.traverse(visitor)
        return items

    def apply_demo_spin(self, dt: float):
        if self.transform is not None:
            self._demo_spin_t += dt
            self.transform.rot_euler += self._demo_spin_axis * (dt * 1.1)
        for ch in self.children:
            ch.apply_demo_spin(dt)

def compose_matrix(t: Transform) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = t.pos[:3]

    rx, ry, rz = float(t.rot_euler[0]), float(t.rot_euler[1]), float(t.rot_euler[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1,0,0,0],[0,cx,-sx,0],[0,sx,cx,0],[0,0,0,1]], dtype=np.float32)
    Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0,0],[sz,cz,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

    R = Rz @ Ry @ Rx
    S = np.eye(4, dtype=np.float32)
    S[0,0], S[1,1], S[2,2] = float(t.scale[0]), float(t.scale[1]), float(t.scale[2])
    return T @ (R @ S)

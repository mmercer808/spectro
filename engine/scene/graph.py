"""
Scene Graph

Entity hierarchy for scene authoring.
This is the MUTABLE side - for editing scenes.
Rendering uses immutable snapshots extracted from this.

Key separation:
- EntityNode: Mutable, for authoring
- GraphSnapshot: Immutable, for rendering (see snapshot.py)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
import numpy as np
import threading

# Thread-safe entity ID generation
_entity_id_lock = threading.Lock()
_next_entity_id = 1


def next_entity_id() -> int:
    """Generate a unique entity ID. Thread-safe."""
    global _next_entity_id
    with _entity_id_lock:
        eid = _next_entity_id
        _next_entity_id += 1
        return eid


# =============================================================================
# Components
# =============================================================================

@dataclass
class Transform:
    """Transform component - position, rotation, scale."""
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    rot_euler: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0], dtype=np.float32))
    
    def copy(self) -> "Transform":
        """Create a deep copy."""
        return Transform(
            pos=self.pos.copy(),
            rot_euler=self.rot_euler.copy(),
            scale=self.scale.copy(),
        )


@dataclass
class MeshRenderer:
    """Mesh renderer component - references mesh and material."""
    mesh_id: str
    pipeline_id: str
    color: np.ndarray
    
    # Future: texture_ids, material_params, etc.
    
    def copy(self) -> "MeshRenderer":
        """Create a deep copy."""
        return MeshRenderer(
            mesh_id=self.mesh_id,
            pipeline_id=self.pipeline_id,
            color=self.color.copy(),
        )


# =============================================================================
# Render Item (for extraction)
# =============================================================================

@dataclass(frozen=True)
class RenderItem:
    """
    Extracted render data from an entity.
    This is what gets passed to the rendering pipeline.
    """
    entity_id: int
    pipeline_id: str
    mesh_id: str
    world: np.ndarray  # World transform matrix
    color: np.ndarray


# =============================================================================
# Entity Node
# =============================================================================

class EntityNode:
    """
    A node in the scene graph hierarchy.
    
    This is the mutable authoring representation.
    For rendering, use extract_render_items() or snapshot via SnapshotBuilder.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.entity_id = next_entity_id()
        self.children: List[EntityNode] = []
        self.parent: Optional[EntityNode] = None
        
        # Components (optional)
        self.transform: Optional[Transform] = None
        self.mesh: Optional[MeshRenderer] = None
        
        # Tags/metadata
        self.tags: set = set()
        self.user_data: Dict[str, Any] = {}
        
        # Demo animation state (for testing)
        self._demo_spin_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._demo_spin_t = 0.0
    
    # -------------------------------------------------------------------------
    # Hierarchy
    # -------------------------------------------------------------------------
    
    def add_child(self, node: "EntityNode"):
        """Add a child node."""
        if node.parent is not None:
            node.parent.remove_child(node)
        node.parent = self
        self.children.append(node)
    
    def remove_child(self, node: "EntityNode"):
        """Remove a child node."""
        if node in self.children:
            self.children.remove(node)
            node.parent = None
    
    def get_root(self) -> "EntityNode":
        """Get the root of this hierarchy."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node
    
    def find_by_name(self, name: str) -> Optional["EntityNode"]:
        """Find a descendant by name."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_by_name(name)
            if found:
                return found
        return None
    
    def find_by_id(self, entity_id: int) -> Optional["EntityNode"]:
        """Find a descendant by entity ID."""
        if self.entity_id == entity_id:
            return self
        for child in self.children:
            found = child.find_by_id(entity_id)
            if found:
                return found
        return None
    
    # -------------------------------------------------------------------------
    # Traversal
    # -------------------------------------------------------------------------
    
    def traverse(
        self, 
        fn: Callable[["EntityNode", np.ndarray], None], 
        parent_m: Optional[np.ndarray] = None
    ):
        """
        Traverse the hierarchy depth-first, computing world matrices.
        
        Args:
            fn: Callback receiving (node, world_matrix)
            parent_m: Parent's world matrix (identity if None)
        """
        if parent_m is None:
            parent_m = np.eye(4, dtype=np.float32)
        
        # Compute local matrix
        local = compose_matrix(self.transform) if self.transform else np.eye(4, dtype=np.float32)
        world = parent_m @ local
        
        # Call visitor
        fn(self, world)
        
        # Recurse to children
        for child in self.children:
            child.traverse(fn, world)
    
    def extract_render_items(self) -> List[RenderItem]:
        """
        Extract all renderable items from this hierarchy.
        
        Returns a list of RenderItems ready for rendering.
        Note: For thread-safe async extraction, use SnapshotBuilder instead.
        """
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
    
    # -------------------------------------------------------------------------
    # Demo Animation
    # -------------------------------------------------------------------------
    
    def apply_demo_spin(self, dt: float):
        """Apply demo spinning animation (for testing)."""
        if self.transform is not None:
            self._demo_spin_t += dt
            self.transform.rot_euler += self._demo_spin_axis * (dt * 1.1)
        
        for child in self.children:
            child.apply_demo_spin(dt)
    
    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        return f"EntityNode(name={self.name!r}, id={self.entity_id}, children={len(self.children)})"
    
    def print_tree(self, indent: int = 0):
        """Print hierarchy for debugging."""
        prefix = "  " * indent
        mesh_info = f" [{self.mesh.mesh_id}]" if self.mesh else ""
        print(f"{prefix}{self.name} (id={self.entity_id}){mesh_info}")
        for child in self.children:
            child.print_tree(indent + 1)


# =============================================================================
# Matrix Utilities
# =============================================================================

def compose_matrix(t: Transform) -> np.ndarray:
    """
    Compose a 4x4 transformation matrix from Transform component.
    
    Order: Translation * Rotation * Scale
    Rotation order: Z * Y * X (common for games)
    """
    # Translation
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = t.pos[:3]
    
    # Rotation (Euler angles)
    rx, ry, rz = float(t.rot_euler[0]), float(t.rot_euler[1]), float(t.rot_euler[2])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([
        [1, 0, 0, 0],
        [0, cx, -sx, 0],
        [0, sx, cx, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    Ry = np.array([
        [cy, 0, sy, 0],
        [0, 1, 0, 0],
        [-sy, 0, cy, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    Rz = np.array([
        [cz, -sz, 0, 0],
        [sz, cz, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    R = Rz @ Ry @ Rx
    
    # Scale
    S = np.eye(4, dtype=np.float32)
    S[0, 0] = float(t.scale[0])
    S[1, 1] = float(t.scale[1])
    S[2, 2] = float(t.scale[2])
    
    return T @ R @ S


def decompose_matrix(m: np.ndarray) -> Transform:
    """
    Decompose a 4x4 matrix back to Transform.
    Note: Only works for matrices without shear.
    """
    # Extract translation
    pos = m[:3, 3].copy()
    
    # Extract scale (length of each column)
    sx = float(np.linalg.norm(m[:3, 0]))
    sy = float(np.linalg.norm(m[:3, 1]))
    sz = float(np.linalg.norm(m[:3, 2]))
    scale = np.array([sx, sy, sz], dtype=np.float32)
    
    # Extract rotation matrix (normalize columns)
    R = m[:3, :3].copy()
    if sx > 1e-6:
        R[:, 0] /= sx
    if sy > 1e-6:
        R[:, 1] /= sy
    if sz > 1e-6:
        R[:, 2] /= sz
    
    # Convert rotation matrix to Euler angles
    # This assumes ZYX order
    if abs(R[2, 0]) < 0.99999:
        ry = -np.arcsin(R[2, 0])
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        rz = 0.0
        if R[2, 0] < 0:
            ry = np.pi / 2
            rx = np.arctan2(R[0, 1], R[0, 2])
        else:
            ry = -np.pi / 2
            rx = np.arctan2(-R[0, 1], -R[0, 2])
    
    rot_euler = np.array([rx, ry, rz], dtype=np.float32)
    
    return Transform(pos=pos, rot_euler=rot_euler, scale=scale)

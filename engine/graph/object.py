# engine/graph/object.py
"""
GraphObject - Scene graph node for 2D rendering.

A GraphObject represents a transformable entity that can hold:
- Child GraphObjects (hierarchy)
- DrawNodes (visual primitives)
- Animations (property tweens)

Design:
- Objects live in a GraphSpace (flat registry with optional hierarchy)
- Each object has a local Transform2D
- World transform computed on demand and cached
- Dirty flag propagates to children when transform changes
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, TYPE_CHECKING
from enum import Enum, auto

from engine.core.math3d import Vec2, Mat3

if TYPE_CHECKING:
    from engine.graph.nodes import DrawNode
    from engine.graph.space import GraphSpace



# =============================================================================
# Transform2D
# =============================================================================

class Transform2D:
@dataclass
    position: Vec2
    rotation: float
    scale: Vec2
    anchor: Vec2

    def to_mat3(self) -> Mat3:
        """Convert to 3x3 transformation matrix."""
        # Order: translate to anchor -> scale -> rotate -> translate to position
        c = math.cos(self.rotation)
        s = math.sin(self.rotation)
        sx, sy = self.scale.x, self.scale.y
        ax, ay = self.anchor.x, self.anchor.y
        px, py = self.position.x, self.position.y

        # Combined matrix: T(pos) * R * S * T(-anchor)
        # Expanding this out:
        return Mat3((
            sx * c,  -sy * s,  px - ax * sx * c + ay * sy * s,
            sx * s,   sy * c,  py - ax * sx * s - ay * sy * c,
            0.0,      0.0,     1.0
        ))

    def transform_point(self, point: Vec2) -> Vec2:
        """Transform a point from local to parent space."""
        mat = self.to_mat3()
        # Treat Vec2 as Vec3 with z=1 for affine transform
        x = mat[0, 0] * point.x + mat[0, 1] * point.y + mat[0, 2]
        y = mat[1, 0] * point.x + mat[1, 1] * point.y + mat[1, 2]
        return Vec2(x, y)

    def lerp(self, other: Transform2D, t: float) -> Transform2D:
        """Interpolate between transforms."""
        return Transform2D(
            position=self.position.lerp(other.position, t),
            rotation=self.rotation + (other.rotation - self.rotation) * t,
            scale=self.scale.lerp(other.scale, t),
            anchor=self.anchor.lerp(other.anchor, t)
        )

    @staticmethod
    def identity() -> Transform2D:
        return Transform2D()


# =============================================================================
# GraphObject
# =============================================================================

class GraphObject:
    """
    A node in the 2D scene graph.

    GraphObjects form a hierarchy (parent/children) and can have DrawNodes
    attached for rendering. Each object has a local transform that combines
    with its parent's world transform.

    Attributes:
        id: Unique identifier (assigned by GraphSpace)
        name: Human-readable name
        transform: Local 2D transform
        visible: Whether this object and its children render
        opacity: Alpha multiplier (inherited by children)
        z_index: Draw order within layer
        parent: Parent GraphObject (None if root or orphan)
        children: Child GraphObjects
        draw_nodes: Attached visual primitives
    """

    __slots__ = (
        'id', 'name', 'transform', 'visible', 'opacity', 'z_index',
        'parent', 'children', 'draw_nodes', 'tags', 'user_data',
        '_world_transform', '_world_transform_dirty', '_space',
        '_world_opacity', '_animations'
    )

    def __init__(
        self,
        name: str = "",
        transform: Transform2D = None,
        visible: bool = True,
        opacity: float = 1.0,
        z_index: int = 0
    ):
        self.id: str = ""  # Set by GraphSpace
        self.name: str = name
        self.transform: Transform2D = transform or Transform2D()
        self.visible: bool = visible
        self.opacity: float = opacity
        self.z_index: int = z_index

        self.parent: Optional[GraphObject] = None
        self.children: List[GraphObject] = []
        self.draw_nodes: List[DrawNode] = []

        self.tags: set = set()
        self.user_data: Dict[str, Any] = {}

        # Cached world transform
        self._world_transform: Optional[Mat3] = None
        self._world_transform_dirty: bool = True
        self._world_opacity: float = 1.0

        # Animation state
        self._animations: List[PropertyAnimation] = []

        # Reference to owning space
        self._space: Optional[GraphSpace] = None

    # -------------------------------------------------------------------------
    # Hierarchy
    # -------------------------------------------------------------------------

    def add_child(self, child: GraphObject) -> GraphObject:
        """Add a child object."""
        if child.parent is not None:
            child.parent.remove_child(child)

        child.parent = self
        self.children.append(child)
        child._mark_transform_dirty()

        # If we're in a space, add child to it too
        if self._space is not None:
            self._space._register_object(child)

        return child

    def remove_child(self, child: GraphObject) -> bool:
        """Remove a child object."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            return True
        return False

    def remove_from_parent(self):
        """Remove this object from its parent."""
        if self.parent is not None:
            self.parent.remove_child(self)

    def get_root(self) -> GraphObject:
        """Get the root of this hierarchy."""
        obj = self
        while obj.parent is not None:
            obj = obj.parent
        return obj

    def iter_descendants(self):
        """Iterate all descendants depth-first."""
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def iter_ancestors(self):
        """Iterate ancestors from parent to root."""
        obj = self.parent
        while obj is not None:
            yield obj
            obj = obj.parent

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    @property
    def world_transform(self) -> Mat3:
        """Get the world transform matrix (cached)."""
        if self._world_transform_dirty:
            self._recompute_world_transform()
        return self._world_transform

    @property
    def world_position(self) -> Vec2:
        """Get world-space position."""
        wt = self.world_transform
        return Vec2(wt[0, 2], wt[1, 2])

    @property
    def world_opacity(self) -> float:
        """Get effective opacity (this * ancestors)."""
        if self._world_transform_dirty:
            self._recompute_world_transform()
        return self._world_opacity

    def _recompute_world_transform(self):
        """Recompute cached world transform."""
        local = self.transform.to_mat3()

        if self.parent is not None:
            parent_world = self.parent.world_transform
            self._world_transform = parent_world @ local
            self._world_opacity = self.parent.world_opacity * self.opacity
        else:
            self._world_transform = local
            self._world_opacity = self.opacity

        self._world_transform_dirty = False

    def _mark_transform_dirty(self):
        """Mark this object's transform as dirty and propagate to children."""
        if not self._world_transform_dirty:
            self._world_transform_dirty = True
            for child in self.children:
                child._mark_transform_dirty()

    def set_position(self, x: float, y: float):
        """Set local position."""
        self.transform.position = Vec2(x, y)
        self._mark_transform_dirty()

    def set_rotation(self, radians: float):
        """Set local rotation."""
        self.transform.rotation = radians
        self._mark_transform_dirty()

    def set_scale(self, sx: float, sy: float = None):
        """Set local scale."""
        if sy is None:
            sy = sx
        self.transform.scale = Vec2(sx, sy)
        self._mark_transform_dirty()

    def local_to_world(self, point: Vec2) -> Vec2:
        """Transform a point from local to world space."""
        wt = self.world_transform
        x = wt[0, 0] * point.x + wt[0, 1] * point.y + wt[0, 2]
        y = wt[1, 0] * point.x + wt[1, 1] * point.y + wt[1, 2]
        return Vec2(x, y)

    def world_to_local(self, point: Vec2) -> Vec2:
        """Transform a point from world to local space."""
        # For a proper implementation, we'd need matrix inverse
        # This is a simplified version for common cases
        wt = self.world_transform
        # 2D affine inverse
        det = wt[0, 0] * wt[1, 1] - wt[0, 1] * wt[1, 0]
        if abs(det) < 1e-10:
            return point

        inv_det = 1.0 / det
        # Translate point by -translation first
        px = point.x - wt[0, 2]
        py = point.y - wt[1, 2]
        # Then apply inverse rotation/scale
        x = inv_det * (wt[1, 1] * px - wt[0, 1] * py)
        y = inv_det * (-wt[1, 0] * px + wt[0, 0] * py)
        return Vec2(x, y)

    # -------------------------------------------------------------------------
    # Draw Nodes
    # -------------------------------------------------------------------------

    def add_draw_node(self, node: DrawNode) -> DrawNode:
        """Attach a draw node to this object."""
        node.owner = self
        self.draw_nodes.append(node)
        return node

    def remove_draw_node(self, node: DrawNode) -> bool:
        """Remove a draw node."""
        if node in self.draw_nodes:
            self.draw_nodes.remove(node)
            node.owner = None
            return True
        return False

    def clear_draw_nodes(self):
        """Remove all draw nodes."""
        for node in self.draw_nodes:
            node.owner = None
        self.draw_nodes.clear()

    # -------------------------------------------------------------------------
    # Animation
    # -------------------------------------------------------------------------

    def animate(
        self,
        property_path: str,
        target: Any,
        duration: float,
        easing: Callable[[float], float] = None,
        delay: float = 0.0,
        on_complete: Callable = None
    ) -> PropertyAnimation:
        """
        Animate a property to a target value.

        Args:
            property_path: Dot-separated path, e.g. "transform.position.x"
            target: Target value
            duration: Animation duration in seconds
            easing: Easing function (default: linear)
            delay: Delay before animation starts
            on_complete: Callback when animation finishes

        Returns:
            PropertyAnimation handle
        """
        from engine.core.math3d import ease_linear

        anim = PropertyAnimation(
            target_obj=self,
            property_path=property_path,
            target_value=target,
            duration=duration,
            easing=easing or ease_linear,
            delay=delay,
            on_complete=on_complete
        )
        self._animations.append(anim)
        return anim

    def stop_animations(self, property_path: str = None):
        """Stop animations, optionally filtered by property."""
        if property_path is None:
            self._animations.clear()
        else:
            self._animations = [
                a for a in self._animations
                if a.property_path != property_path
            ]

    def update(self, dt: float):
        """
        Update animations and children.

        Args:
            dt: Delta time in seconds
        """
        # Update animations
        completed = []
        for anim in self._animations:
            if anim.update(dt):
                completed.append(anim)

        for anim in completed:
            self._animations.remove(anim)
            if anim.on_complete:
                anim.on_complete()

        # Update children
        for child in self.children:
            child.update(dt)

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def find_by_name(self, name: str) -> Optional[GraphObject]:
        """Find a descendant by name."""
        for child in self.children:
            if child.name == name:
                return child
            found = child.find_by_name(name)
            if found:
                return found
        return None

    def find_by_tag(self, tag: str) -> List[GraphObject]:
        """Find all descendants with a given tag."""
        result = []
        if tag in self.tags:
            result.append(self)
        for child in self.children:
            result.extend(child.find_by_tag(tag))
        return result

    def __repr__(self) -> str:
        return f"GraphObject(id={self.id!r}, name={self.name!r})"


# =============================================================================
# Property Animation
# =============================================================================

@dataclass
class PropertyAnimation:
    """
    Animates a single property of a GraphObject.
    """
    target_obj: GraphObject
    property_path: str
    target_value: Any
    duration: float
    easing: Callable[[float], float]
    delay: float = 0.0
    on_complete: Callable = None

    _elapsed: float = field(default=0.0, init=False)
    _start_value: Any = field(default=None, init=False)
    _started: bool = field(default=False, init=False)

    def update(self, dt: float) -> bool:
        """
        Update animation state.

        Returns:
            True if animation completed this frame
        """
        self._elapsed += dt

        # Handle delay
        if self._elapsed < self.delay:
            return False

        # Initialize start value on first update past delay
        if not self._started:
            self._start_value = self._get_property()
            self._started = True

        # Calculate progress
        active_time = self._elapsed - self.delay
        t = min(active_time / self.duration, 1.0) if self.duration > 0 else 1.0
        eased_t = self.easing(t)

        # Interpolate and set value
        value = self._interpolate(self._start_value, self.target_value, eased_t)
        self._set_property(value)

        # Mark transform dirty if animating transform properties
        if self.property_path.startswith("transform"):
            self.target_obj._mark_transform_dirty()

        return t >= 1.0

    def _get_property(self) -> Any:
        """Get current property value."""
        parts = self.property_path.split('.')
        obj = self.target_obj
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _set_property(self, value: Any):
        """Set property value."""
        parts = self.property_path.split('.')
        obj = self.target_obj
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def _interpolate(self, start: Any, end: Any, t: float) -> Any:
        """Interpolate between values based on type."""
        if isinstance(start, (int, float)):
            return start + (end - start) * t
        elif isinstance(start, Vec2):
            return start.lerp(end, t)
        elif hasattr(start, 'lerp'):
            return start.lerp(end, t)
        else:
            # No interpolation, snap at end
            return end if t >= 1.0 else start

# engine/graph/space.py
"""
GraphSpace - Container for 2D scene graph.

A GraphSpace manages a collection of GraphObjects and organizes them
into layers for rendering. It provides:

- Flat object registry with unique IDs
- Layer-based organization for draw ordering
- Spatial indexing for culling and picking
- Dirty tracking for efficient updates
- Coordinate space transformation

Design:
- Objects can exist in multiple layers (multi-pass rendering)
- Layers are sorted by z_order for draw sequence
- Each layer can have its own spatial index (quadtree)
- Collect phase gathers DrawNodes into a DrawBatch
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Callable, Iterator, Tuple, TYPE_CHECKING
)
from enum import Enum, auto

from engine.core.math3d import Vec2, Mat3
from engine.core.signal import SignalBridge, SignalEmitter

if TYPE_CHECKING:
    from engine.graph.object import GraphObject
    from engine.graph.nodes import DrawNode
    from engine.graph.coordinates import CoordinateSpace
    from engine.ui.draw import DrawBatch


# =============================================================================
# Signals
# =============================================================================

SIGNAL_OBJECT_ADDED = 'graph.object_added'
SIGNAL_OBJECT_REMOVED = 'graph.object_removed'
SIGNAL_LAYER_ADDED = 'graph.layer_added'
SIGNAL_LAYER_REMOVED = 'graph.layer_removed'
SIGNAL_SPACE_DIRTY = 'graph.space_dirty'


# =============================================================================
# Spatial Index (QuadTree)
# =============================================================================

@dataclass
class Rect:
    """Axis-aligned bounding box."""
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Vec2:
        return Vec2(self.x + self.width / 2, self.y + self.height / 2)

    def contains_point(self, p: Vec2) -> bool:
        return self.x <= p.x <= self.right and self.y <= p.y <= self.bottom

    def intersects(self, other: Rect) -> bool:
        return not (
            self.right < other.x or
            other.right < self.x or
            self.bottom < other.y or
            other.bottom < self.y
        )

    def union(self, other: Rect) -> Rect:
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        r = max(self.right, other.right)
        b = max(self.bottom, other.bottom)
        return Rect(x, y, r - x, b - y)

    @staticmethod
    def from_points(p1: Vec2, p2: Vec2) -> Rect:
        x = min(p1.x, p2.x)
        y = min(p1.y, p2.y)
        return Rect(x, y, abs(p2.x - p1.x), abs(p2.y - p1.y))


class QuadTree:
    """
    Spatial index for efficient visibility culling and picking.

    Stores object IDs (not objects directly) for cache-friendliness.
    """

    MAX_OBJECTS = 8
    MAX_DEPTH = 6

    def __init__(self, bounds: Rect, depth: int = 0):
        self.bounds = bounds
        self.depth = depth
        self.objects: List[Tuple[str, Rect]] = []  # (object_id, bounds)
        self.children: Optional[List[QuadTree]] = None

    def insert(self, object_id: str, bounds: Rect) -> bool:
        """Insert an object with its bounding box."""
        if not self.bounds.intersects(bounds):
            return False

        if self.children is not None:
            for child in self.children:
                child.insert(object_id, bounds)
            return True

        self.objects.append((object_id, bounds))

        if len(self.objects) > self.MAX_OBJECTS and self.depth < self.MAX_DEPTH:
            self._subdivide()

        return True

    def query(self, region: Rect) -> Set[str]:
        """Find all object IDs intersecting a region."""
        result = set()

        if not self.bounds.intersects(region):
            return result

        for obj_id, bounds in self.objects:
            if bounds.intersects(region):
                result.add(obj_id)

        if self.children is not None:
            for child in self.children:
                result.update(child.query(region))

        return result

    def query_point(self, point: Vec2) -> Set[str]:
        """Find all object IDs containing a point."""
        result = set()

        if not self.bounds.contains_point(point):
            return result

        for obj_id, bounds in self.objects:
            if bounds.contains_point(point):
                result.add(obj_id)

        if self.children is not None:
            for child in self.children:
                result.update(child.query_point(point))

        return result

    def clear(self):
        """Clear all objects from the tree."""
        self.objects.clear()
        self.children = None

    def _subdivide(self):
        """Split into four children."""
        x, y = self.bounds.x, self.bounds.y
        hw, hh = self.bounds.width / 2, self.bounds.height / 2

        self.children = [
            QuadTree(Rect(x, y, hw, hh), self.depth + 1),
            QuadTree(Rect(x + hw, y, hw, hh), self.depth + 1),
            QuadTree(Rect(x, y + hh, hw, hh), self.depth + 1),
            QuadTree(Rect(x + hw, y + hh, hw, hh), self.depth + 1),
        ]

        # Re-insert objects into children
        for obj_id, bounds in self.objects:
            for child in self.children:
                child.insert(obj_id, bounds)

        self.objects.clear()


# =============================================================================
# Layer
# =============================================================================

class Layer:
    """
    A render layer containing GraphObjects.

    Layers provide:
    - Coarse z-ordering (layers render in z_order sequence)
    - Spatial indexing per layer
    - Independent dirty tracking
    - Optional render target (for post-processing)
    """

    def __init__(
        self,
        name: str,
        z_order: int = 0,
        visible: bool = True,
        bounds: Rect = None
    ):
        self.name = name
        self.z_order = z_order
        self.visible = visible

        # Objects in this layer (by ID)
        self.object_ids: Set[str] = set()

        # Spatial index
        self._bounds = bounds or Rect(-10000, -10000, 20000, 20000)
        self._spatial_index: Optional[QuadTree] = None
        self._spatial_dirty = True

        # Dirty tracking
        self._dirty_regions: List[Rect] = []
        self._fully_dirty = True

        # Optional render target for layer compositing
        self.render_target: Optional[str] = None  # Target ID
        self.blend_mode: str = "normal"
        self.opacity: float = 1.0

    def add_object(self, object_id: str):
        """Add an object to this layer."""
        self.object_ids.add(object_id)
        self._spatial_dirty = True
        self._fully_dirty = True

    def remove_object(self, object_id: str):
        """Remove an object from this layer."""
        self.object_ids.discard(object_id)
        self._spatial_dirty = True
        self._fully_dirty = True

    def mark_dirty(self, region: Rect = None):
        """Mark a region as needing redraw."""
        if region is None:
            self._fully_dirty = True
        else:
            self._dirty_regions.append(region)

    def clear_dirty(self):
        """Clear dirty state after render."""
        self._dirty_regions.clear()
        self._fully_dirty = False

    def query_visible(self, viewport: Rect) -> Set[str]:
        """Get object IDs visible in viewport."""
        if self._spatial_dirty:
            # Can't use spatial index, return all
            return self.object_ids.copy()

        if self._spatial_index is None:
            return self.object_ids.copy()

        return self._spatial_index.query(viewport)

    def rebuild_spatial_index(self, get_bounds: Callable[[str], Optional[Rect]]):
        """Rebuild the spatial index with current object bounds."""
        self._spatial_index = QuadTree(self._bounds)

        for obj_id in self.object_ids:
            bounds = get_bounds(obj_id)
            if bounds is not None:
                self._spatial_index.insert(obj_id, bounds)

        self._spatial_dirty = False


# =============================================================================
# GraphSpace
# =============================================================================

class GraphSpace(SignalEmitter):
    """
    Container for a 2D scene graph.

    GraphSpace manages:
    - Object registry (flat ID -> object mapping)
    - Layer management
    - Coordinate space transformation
    - Update/render cycle
    """

    def __init__(
        self,
        name: str = "main",
        coordinate_space: CoordinateSpace = None,
        bridge: SignalBridge = None
    ):
        self.name = name

        # Object registry
        self._objects: Dict[str, GraphObject] = {}
        self._root: Optional[GraphObject] = None

        # Layers
        self._layers: Dict[str, Layer] = {}
        self._layer_order: List[str] = []  # Sorted by z_order

        # Create default layer
        self.add_layer(Layer("default", z_order=0))

        # Coordinate space
        self._coordinate_space = coordinate_space

        # View transform (camera)
        self._view_transform: Mat3 = Mat3.identity()
        self._viewport: Rect = Rect(0, 0, 800, 600)

        # Dirty tracking
        self._dirty = True

        # Signal bridge
        if bridge:
            self.bind_bridge(bridge)

    # -------------------------------------------------------------------------
    # Object Management
    # -------------------------------------------------------------------------

    def add(
        self,
        obj: GraphObject,
        layer: str = "default",
        parent: GraphObject = None
    ) -> GraphObject:
        """
        Add an object to the space.

        Args:
            obj: The GraphObject to add
            layer: Layer name (default: "default")
            parent: Optional parent object

        Returns:
            The added object (for chaining)
        """
        # Generate unique ID if not set
        if not obj.id:
            obj.id = str(uuid.uuid4())[:8]

        # Register object
        self._objects[obj.id] = obj
        obj._space = self

        # Set up hierarchy
        if parent is not None:
            parent.add_child(obj)
        elif self._root is None:
            self._root = obj

        # Add to layer
        if layer in self._layers:
            self._layers[layer].add_object(obj.id)

        # Register all descendants too
        for descendant in obj.iter_descendants():
            self._register_object(descendant, layer)

        self._dirty = True
        self.emit(SIGNAL_OBJECT_ADDED, obj)

        return obj

    def _register_object(self, obj: GraphObject, layer: str = "default"):
        """Internal: register an object (called when added to hierarchy)."""
        if not obj.id:
            obj.id = str(uuid.uuid4())[:8]

        self._objects[obj.id] = obj
        obj._space = self

        if layer in self._layers:
            self._layers[layer].add_object(obj.id)

    def remove(self, obj: GraphObject) -> bool:
        """Remove an object from the space."""
        if obj.id not in self._objects:
            return False

        # Remove from all layers
        for layer in self._layers.values():
            layer.remove_object(obj.id)

        # Remove from hierarchy
        obj.remove_from_parent()

        # Remove descendants
        for descendant in list(obj.iter_descendants()):
            self._objects.pop(descendant.id, None)
            descendant._space = None

        # Remove object
        del self._objects[obj.id]
        obj._space = None

        if self._root is obj:
            self._root = None

        self._dirty = True
        self.emit(SIGNAL_OBJECT_REMOVED, obj)

        return True

    def get(self, object_id: str) -> Optional[GraphObject]:
        """Get an object by ID."""
        return self._objects.get(object_id)

    def find_by_name(self, name: str) -> Optional[GraphObject]:
        """Find an object by name."""
        for obj in self._objects.values():
            if obj.name == name:
                return obj
        return None

    def find_by_tag(self, tag: str) -> List[GraphObject]:
        """Find all objects with a tag."""
        return [obj for obj in self._objects.values() if tag in obj.tags]

    def query(self, predicate: Callable[[GraphObject], bool]) -> List[GraphObject]:
        """Find objects matching a predicate."""
        return [obj for obj in self._objects.values() if predicate(obj)]

    # -------------------------------------------------------------------------
    # Layer Management
    # -------------------------------------------------------------------------

    def add_layer(self, layer: Layer) -> Layer:
        """Add a layer to the space."""
        self._layers[layer.name] = layer
        self._update_layer_order()
        self.emit(SIGNAL_LAYER_ADDED, layer)
        return layer

    def remove_layer(self, name: str) -> bool:
        """Remove a layer (objects are not removed, just unassigned)."""
        if name in self._layers:
            layer = self._layers.pop(name)
            self._layer_order.remove(name)
            self.emit(SIGNAL_LAYER_REMOVED, layer)
            return True
        return False

    def get_layer(self, name: str) -> Optional[Layer]:
        """Get a layer by name."""
        return self._layers.get(name)

    def _update_layer_order(self):
        """Sort layers by z_order."""
        self._layer_order = sorted(
            self._layers.keys(),
            key=lambda n: self._layers[n].z_order
        )

    def move_to_layer(self, obj: GraphObject, layer_name: str):
        """Move an object to a different layer."""
        # Remove from current layers
        for layer in self._layers.values():
            layer.remove_object(obj.id)

        # Add to new layer
        if layer_name in self._layers:
            self._layers[layer_name].add_object(obj.id)

    # -------------------------------------------------------------------------
    # View / Coordinate Space
    # -------------------------------------------------------------------------

    def set_viewport(self, x: float, y: float, width: float, height: float):
        """Set the visible viewport rectangle."""
        self._viewport = Rect(x, y, width, height)
        self._dirty = True

    def set_view_transform(self, transform: Mat3):
        """Set the view (camera) transform."""
        self._view_transform = transform
        self._dirty = True

    def screen_to_world(self, screen_pos: Vec2) -> Vec2:
        """Convert screen coordinates to world coordinates."""
        # Apply inverse view transform
        # For now, simple offset inverse
        vt = self._view_transform
        det = vt[0, 0] * vt[1, 1] - vt[0, 1] * vt[1, 0]
        if abs(det) < 1e-10:
            return screen_pos

        inv_det = 1.0 / det
        px = screen_pos.x - vt[0, 2]
        py = screen_pos.y - vt[1, 2]
        x = inv_det * (vt[1, 1] * px - vt[0, 1] * py)
        y = inv_det * (-vt[1, 0] * px + vt[0, 0] * py)
        return Vec2(x, y)

    def world_to_screen(self, world_pos: Vec2) -> Vec2:
        """Convert world coordinates to screen coordinates."""
        vt = self._view_transform
        x = vt[0, 0] * world_pos.x + vt[0, 1] * world_pos.y + vt[0, 2]
        y = vt[1, 0] * world_pos.x + vt[1, 1] * world_pos.y + vt[1, 2]
        return Vec2(x, y)

    # -------------------------------------------------------------------------
    # Update / Collect
    # -------------------------------------------------------------------------

    def update(self, dt: float):
        """
        Update all objects (animations, etc).

        Args:
            dt: Delta time in seconds
        """
        # Update from root (hierarchy order)
        if self._root is not None:
            self._root.update(dt)
        else:
            # No hierarchy, update all
            for obj in self._objects.values():
                if obj.parent is None:
                    obj.update(dt)

    def collect(self, batch: DrawBatch, viewport: Rect = None):
        """
        Collect visible DrawNodes into a DrawBatch.

        Args:
            batch: DrawBatch to fill
            viewport: Visible region (uses current viewport if None)
        """
        if viewport is None:
            viewport = self._viewport

        # Process layers in order
        for layer_name in self._layer_order:
            layer = self._layers[layer_name]

            if not layer.visible:
                continue

            # Get visible objects
            visible_ids = layer.query_visible(viewport)

            # Collect draw nodes from visible objects
            for obj_id in visible_ids:
                obj = self._objects.get(obj_id)
                if obj is None or not obj.visible:
                    continue

                self._collect_object(obj, batch, layer)

        self._dirty = False

    def _collect_object(self, obj: GraphObject, batch: DrawBatch, layer: Layer):
        """Collect DrawNodes from a single object."""
        world_transform = obj.world_transform
        world_opacity = obj.world_opacity * layer.opacity

        for node in obj.draw_nodes:
            if not node.visible:
                continue

            effective_opacity = node.effective_opacity * layer.opacity
            z_index = obj.z_index + node.z_offset + layer.z_order * 1000

            # Add to batch based on node type
            self._add_node_to_batch(node, batch, world_transform, effective_opacity, z_index)

    def _add_node_to_batch(
        self,
        node: DrawNode,
        batch: DrawBatch,
        transform: Mat3,
        opacity: float,
        z_index: int
    ):
        """Convert a DrawNode to batch commands."""
        # Import here to avoid circular import
        from engine.graph.nodes import (
            RectNode, CircleNode, ArcNode, LineNode, TextNode,
            ImageNode, PointsNode, GridLinesNode
        )
        from engine.ui.draw import DrawQuad, DrawLine, DrawText

        if isinstance(node, RectNode):
            # Transform corners
            p0 = self._transform_point(transform, Vec2(node.x, node.y))
            p1 = self._transform_point(transform, Vec2(node.x + node.width, node.y + node.height))

            if node.fill:
                color = node.fill.to_tuple()
                color = (color[0], color[1], color[2], color[3] * opacity)
                batch.add_quad(DrawQuad(
                    x=p0.x, y=p0.y,
                    w=p1.x - p0.x, h=p1.y - p0.y,
                    color=color,
                    radius=node.corner_radius,
                    z_index=z_index
                ))

        elif isinstance(node, LineNode):
            if len(node.points) >= 2:
                color = node.stroke.color.to_tuple()
                color = (color[0], color[1], color[2], color[3] * opacity)

                for i in range(len(node.points) - 1):
                    p0 = self._transform_point(transform, node.points[i])
                    p1 = self._transform_point(transform, node.points[i + 1])
                    batch.add_line(DrawLine(
                        x0=p0.x, y0=p0.y,
                        x1=p1.x, y1=p1.y,
                        color=color,
                        width=node.stroke.width,
                        z_index=z_index
                    ))

                if node.closed and len(node.points) > 2:
                    p0 = self._transform_point(transform, node.points[-1])
                    p1 = self._transform_point(transform, node.points[0])
                    batch.add_line(DrawLine(
                        x0=p0.x, y0=p0.y,
                        x1=p1.x, y1=p1.y,
                        color=color,
                        width=node.stroke.width,
                        z_index=z_index
                    ))

        elif isinstance(node, TextNode):
            p = self._transform_point(transform, Vec2(node.x, node.y))
            color = node.color.to_tuple()
            color = (color[0], color[1], color[2], color[3] * opacity)
            batch.add_text(DrawText(
                text=node.text,
                x=p.x, y=p.y,
                color=color,
                font_size=node.font_size,
                font_id=node.font_family,
                z_index=z_index
            ))

        elif isinstance(node, GridLinesNode):
            # Vertical lines
            if node.vertical_x:
                color = node.vertical_color.to_tuple()
                color = (color[0], color[1], color[2], color[3] * opacity)
                for x in node.vertical_x:
                    p0 = self._transform_point(transform, Vec2(x, node.vertical_y0))
                    p1 = self._transform_point(transform, Vec2(x, node.vertical_y1))
                    batch.add_line(DrawLine(
                        x0=p0.x, y0=p0.y,
                        x1=p1.x, y1=p1.y,
                        color=color,
                        width=node.vertical_width,
                        z_index=z_index
                    ))

            # Horizontal lines
            if node.horizontal_y:
                color = node.horizontal_color.to_tuple()
                color = (color[0], color[1], color[2], color[3] * opacity)
                for y in node.horizontal_y:
                    p0 = self._transform_point(transform, Vec2(node.horizontal_x0, y))
                    p1 = self._transform_point(transform, Vec2(node.horizontal_x1, y))
                    batch.add_line(DrawLine(
                        x0=p0.x, y0=p0.y,
                        x1=p1.x, y1=p1.y,
                        color=color,
                        width=node.horizontal_width,
                        z_index=z_index
                    ))

        # TODO: Handle CircleNode, ArcNode, ImageNode, PointsNode, PathNode
        # These require additional batch command types or shader support

    def _transform_point(self, transform: Mat3, point: Vec2) -> Vec2:
        """Apply matrix transform to a point."""
        x = transform[0, 0] * point.x + transform[0, 1] * point.y + transform[0, 2]
        y = transform[1, 0] * point.x + transform[1, 1] * point.y + transform[1, 2]
        return Vec2(x, y)

    # -------------------------------------------------------------------------
    # Picking
    # -------------------------------------------------------------------------

    def pick(self, screen_pos: Vec2, layer_name: str = None) -> List[GraphObject]:
        """
        Find objects at a screen position.

        Args:
            screen_pos: Position in screen coordinates
            layer_name: Optional layer to restrict search

        Returns:
            List of objects under the point, front-to-back order
        """
        world_pos = self.screen_to_world(screen_pos)
        results = []

        layers = [self._layers[layer_name]] if layer_name else [
            self._layers[n] for n in reversed(self._layer_order)
        ]

        for layer in layers:
            if not layer.visible:
                continue

            candidates = layer.query_visible(Rect(
                world_pos.x - 1, world_pos.y - 1, 2, 2
            ))

            for obj_id in candidates:
                obj = self._objects.get(obj_id)
                if obj is None or not obj.visible:
                    continue

                # Check if point is inside any draw node
                local_pos = obj.world_to_local(world_pos)
                for node in obj.draw_nodes:
                    bounds_min, bounds_max = node.get_bounds()
                    if (bounds_min.x <= local_pos.x <= bounds_max.x and
                        bounds_min.y <= local_pos.y <= bounds_max.y):
                        results.append(obj)
                        break

        return results

    # -------------------------------------------------------------------------
    # Iteration
    # -------------------------------------------------------------------------

    def __iter__(self) -> Iterator[GraphObject]:
        """Iterate all objects."""
        return iter(self._objects.values())

    def __len__(self) -> int:
        """Number of objects."""
        return len(self._objects)

    @property
    def objects(self) -> Dict[str, GraphObject]:
        """Read-only access to object registry."""
        return self._objects

    @property
    def root(self) -> Optional[GraphObject]:
        """Get root object."""
        return self._root

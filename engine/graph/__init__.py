# engine/graph/__init__.py
"""
Graph Module - 2D Scene Graph System

This module provides a 2D scene graph for building animated, interactive
UI and visualizations. Key components:

- GraphObject: Scene node with transform, hierarchy, and draw nodes
- DrawNode: Visual primitives (rect, circle, line, text, etc.)
- GraphSpace: Container managing objects, layers, and rendering
- CoordinateSpace: Different coordinate systems (Cartesian, Polar, 3D projection)

Example usage:

    from engine.graph import GraphSpace, GraphObject, RectNode, Color

    # Create a space
    space = GraphSpace("main")

    # Create an object with a rectangle
    obj = GraphObject(name="my_rect")
    obj.set_position(100, 100)
    obj.add_draw_node(RectNode(
        width=50, height=50,
        fill=Color.from_hex("#ff6b6b")
    ))

    # Add to space
    space.add(obj)

    # Animate
    obj.animate("transform.position.x", 200, duration=0.5)

    # In render loop:
    space.update(dt)
    space.collect(batch)
"""

# Core object types
from engine.graph.object import (
    GraphObject,
    Transform2D,
    PropertyAnimation,
)

# Draw nodes
from engine.graph.nodes import (
    # Base
    DrawNode,
    Color,
    StrokeStyle,
    StrokeCap,
    StrokeJoin,

    # Primitives
    RectNode,
    CircleNode,
    ArcNode,
    LineNode,
    PathNode,
    PathCommand,
    PathSegment,
    TextNode,
    TextAlign,
    TextBaseline,
    ImageNode,

    # Optimized nodes
    PointsNode,
    GridLinesNode,
)

# Space and layers
from engine.graph.space import (
    GraphSpace,
    Layer,
    Rect,
    QuadTree,

    # Signals
    SIGNAL_OBJECT_ADDED,
    SIGNAL_OBJECT_REMOVED,
    SIGNAL_LAYER_ADDED,
    SIGNAL_LAYER_REMOVED,
    SIGNAL_SPACE_DIRTY,
)

# Coordinate spaces
from engine.graph.coordinates import (
    CoordinateSpace,
    Cartesian2D,
    Origin,
    Polar,
    Projected3D,
    BeatSpace,
)

# Style tokens
from engine.graph.style import (
    StyleTokens,
    DARK_THEME,
    LIGHT_THEME,
    HIGH_CONTRAST_THEME,
    StyleUBO,
)

# Procedural renderer
from engine.graph.renderer import (
    ProceduralRenderer,
    ComponentKind,
    UIInstance,
    pack_instances,
)

# Text rendering
from engine.graph.text import (
    FontAtlas,
    TextRenderer,
    TextAlign,
    TextBaseline,
    GlyphMetrics,
    layout_text,
)

__all__ = [
    # Object
    'GraphObject',
    'Transform2D',
    'PropertyAnimation',

    # Nodes
    'DrawNode',
    'Color',
    'StrokeStyle',
    'StrokeCap',
    'StrokeJoin',
    'RectNode',
    'CircleNode',
    'ArcNode',
    'LineNode',
    'PathNode',
    'PathCommand',
    'PathSegment',
    'TextNode',
    'TextAlign',
    'TextBaseline',
    'ImageNode',
    'PointsNode',
    'GridLinesNode',

    # Space
    'GraphSpace',
    'Layer',
    'Rect',
    'QuadTree',
    'SIGNAL_OBJECT_ADDED',
    'SIGNAL_OBJECT_REMOVED',
    'SIGNAL_LAYER_ADDED',
    'SIGNAL_LAYER_REMOVED',
    'SIGNAL_SPACE_DIRTY',

    # Coordinates
    'CoordinateSpace',
    'Cartesian2D',
    'Origin',
    'Polar',
    'Projected3D',
    'BeatSpace',

    # Style
    'StyleTokens',
    'DARK_THEME',
    'LIGHT_THEME',
    'HIGH_CONTRAST_THEME',
    'StyleUBO',

    # Renderer
    'ProceduralRenderer',
    'ComponentKind',
    'UIInstance',
    'pack_instances',

    # Text
    'FontAtlas',
    'TextRenderer',
    'TextAlign',
    'TextBaseline',
    'GlyphMetrics',
    'layout_text',
]

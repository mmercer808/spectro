# engine/graph/nodes.py
"""
DrawNode - Visual primitives attached to GraphObjects.

Each DrawNode type represents a specific visual primitive:
- RectNode: Rectangle with optional corner radius and border
- CircleNode: Circle or ellipse with fill and stroke
- ArcNode: Partial arc (for phase indicators)
- LineNode: Line segment or polyline
- PathNode: Complex paths with bezier curves
- TextNode: Text rendering
- ImageNode: Textured quad

Design:
- DrawNodes are pure data (no GL resources)
- They specify what to draw in local coordinates
- The renderer transforms them using the owner's world matrix
- Changes mark the owner's space as dirty
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, TYPE_CHECKING
from enum import Enum, auto
from abc import ABC, abstractmethod

from engine.core.math3d import Vec2

if TYPE_CHECKING:
    from engine.graph.object import GraphObject


# =============================================================================
# Color Type
# =============================================================================

@dataclass
class Color:
    """RGBA color with 0-1 range components."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    def with_alpha(self, alpha: float) -> Color:
        return Color(self.r, self.g, self.b, alpha)

    def lerp(self, other: Color, t: float) -> Color:
        return Color(
            self.r + (other.r - self.r) * t,
            self.g + (other.g - self.g) * t,
            self.b + (other.b - self.b) * t,
            self.a + (other.a - self.a) * t
        )

    @staticmethod
    def from_hex(hex_str: str) -> Color:
        """Parse hex color like '#ff6b6b' or 'ff6b6b'."""
        hex_str = hex_str.lstrip('#')
        if len(hex_str) == 6:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            return Color(r, g, b, 1.0)
        elif len(hex_str) == 8:
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = int(hex_str[6:8], 16) / 255.0
            return Color(r, g, b, a)
        raise ValueError(f"Invalid hex color: {hex_str}")

    @staticmethod
    def from_rgb(r: int, g: int, b: int, a: int = 255) -> Color:
        """Create from 0-255 RGB values."""
        return Color(r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    # Common colors
    @staticmethod
    def white() -> Color: return Color(1.0, 1.0, 1.0, 1.0)
    @staticmethod
    def black() -> Color: return Color(0.0, 0.0, 0.0, 1.0)
    @staticmethod
    def transparent() -> Color: return Color(0.0, 0.0, 0.0, 0.0)
    @staticmethod
    def red() -> Color: return Color(1.0, 0.0, 0.0, 1.0)
    @staticmethod
    def green() -> Color: return Color(0.0, 1.0, 0.0, 1.0)
    @staticmethod
    def blue() -> Color: return Color(0.0, 0.0, 1.0, 1.0)


# =============================================================================
# Stroke Style
# =============================================================================

class StrokeCap(Enum):
    BUTT = auto()
    ROUND = auto()
    SQUARE = auto()


class StrokeJoin(Enum):
    MITER = auto()
    ROUND = auto()
    BEVEL = auto()


@dataclass
class StrokeStyle:
    """Style for stroked paths."""
    color: Color = field(default_factory=Color.white)
    width: float = 1.0
    cap: StrokeCap = StrokeCap.BUTT
    join: StrokeJoin = StrokeJoin.MITER
    dash_array: Optional[List[float]] = None
    dash_offset: float = 0.0


# =============================================================================
# Base DrawNode
# =============================================================================

class DrawNode(ABC):
    """
    Base class for visual primitives.

    DrawNodes are attached to GraphObjects and rendered in the owner's
    local coordinate space. The renderer applies the world transform.
    """

    __slots__ = ('owner', 'visible', 'opacity', 'z_offset', 'blend_mode')

    def __init__(
        self,
        visible: bool = True,
        opacity: float = 1.0,
        z_offset: int = 0
    ):
        self.owner: Optional[GraphObject] = None
        self.visible: bool = visible
        self.opacity: float = opacity
        self.z_offset: int = z_offset  # Added to owner's z_index
        self.blend_mode: str = "normal"  # normal, add, multiply

    @property
    def effective_opacity(self) -> float:
        """Get opacity multiplied by owner's world opacity."""
        if self.owner is not None:
            return self.opacity * self.owner.world_opacity
        return self.opacity

    @abstractmethod
    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        """Get local-space bounding box (min, max)."""
        pass


# =============================================================================
# RectNode
# =============================================================================

@dataclass
class RectNode(DrawNode):
    """
    Rectangle primitive with optional rounded corners and border.

    Coordinates are in local space, relative to owner's transform.
    """
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    fill: Optional[Color] = field(default_factory=Color.white)
    stroke: Optional[StrokeStyle] = None
    corner_radius: float = 0.0  # All corners, or use corner_radii for individual
    corner_radii: Optional[Tuple[float, float, float, float]] = None  # TL, TR, BR, BL

    def __post_init__(self):
        super().__init__()

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        return (Vec2(self.x, self.y), Vec2(self.x + self.width, self.y + self.height))


# =============================================================================
# CircleNode
# =============================================================================

@dataclass
class CircleNode(DrawNode):
    """
    Circle or ellipse primitive.
    """
    cx: float = 0.0  # Center X
    cy: float = 0.0  # Center Y
    radius: float = 50.0  # Radius (or X radius if ellipse)
    radius_y: Optional[float] = None  # Y radius for ellipse
    fill: Optional[Color] = field(default_factory=Color.white)
    stroke: Optional[StrokeStyle] = None

    def __post_init__(self):
        super().__init__()

    @property
    def ry(self) -> float:
        return self.radius_y if self.radius_y is not None else self.radius

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        ry = self.ry
        return (
            Vec2(self.cx - self.radius, self.cy - ry),
            Vec2(self.cx + self.radius, self.cy + ry)
        )


# =============================================================================
# ArcNode
# =============================================================================

@dataclass
class ArcNode(DrawNode):
    """
    Arc primitive for partial circles (e.g., phase indicators).

    Angles are in radians, with 0 at 3 o'clock, increasing counter-clockwise.
    """
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 50.0
    start_angle: float = 0.0  # Start angle in radians
    end_angle: float = 3.14159  # End angle in radians
    stroke: StrokeStyle = field(default_factory=lambda: StrokeStyle(width=2.0))
    fill: Optional[Color] = None  # If set, fills the pie slice

    def __post_init__(self):
        super().__init__()

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        # Conservative bounds
        return (
            Vec2(self.cx - self.radius, self.cy - self.radius),
            Vec2(self.cx + self.radius, self.cy + self.radius)
        )


# =============================================================================
# LineNode
# =============================================================================

@dataclass
class LineNode(DrawNode):
    """
    Line segment or polyline.
    """
    points: List[Vec2] = field(default_factory=list)
    stroke: StrokeStyle = field(default_factory=lambda: StrokeStyle(width=1.0))
    closed: bool = False  # Connect last point to first

    def __post_init__(self):
        super().__init__()

    @staticmethod
    def segment(x0: float, y0: float, x1: float, y1: float,
                color: Color = None, width: float = 1.0) -> LineNode:
        """Create a simple line segment."""
        return LineNode(
            points=[Vec2(x0, y0), Vec2(x1, y1)],
            stroke=StrokeStyle(color=color or Color.white(), width=width)
        )

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        if not self.points:
            return (Vec2(0, 0), Vec2(0, 0))
        min_x = min(p.x for p in self.points)
        min_y = min(p.y for p in self.points)
        max_x = max(p.x for p in self.points)
        max_y = max(p.y for p in self.points)
        return (Vec2(min_x, min_y), Vec2(max_x, max_y))


# =============================================================================
# PathNode
# =============================================================================

class PathCommand(Enum):
    MOVE = auto()
    LINE = auto()
    QUAD = auto()  # Quadratic bezier
    CUBIC = auto()  # Cubic bezier
    ARC = auto()
    CLOSE = auto()


@dataclass
class PathSegment:
    """A single segment in a path."""
    command: PathCommand
    points: List[Vec2] = field(default_factory=list)
    # For arcs
    radius: float = 0.0
    large_arc: bool = False
    sweep: bool = False


@dataclass
class PathNode(DrawNode):
    """
    Complex path with bezier curves and arcs.
    """
    segments: List[PathSegment] = field(default_factory=list)
    fill: Optional[Color] = None
    stroke: Optional[StrokeStyle] = None
    fill_rule: str = "nonzero"  # nonzero or evenodd

    def __post_init__(self):
        super().__init__()

    def move_to(self, x: float, y: float) -> PathNode:
        self.segments.append(PathSegment(PathCommand.MOVE, [Vec2(x, y)]))
        return self

    def line_to(self, x: float, y: float) -> PathNode:
        self.segments.append(PathSegment(PathCommand.LINE, [Vec2(x, y)]))
        return self

    def quad_to(self, cx: float, cy: float, x: float, y: float) -> PathNode:
        self.segments.append(PathSegment(PathCommand.QUAD, [Vec2(cx, cy), Vec2(x, y)]))
        return self

    def cubic_to(self, c1x: float, c1y: float, c2x: float, c2y: float,
                 x: float, y: float) -> PathNode:
        self.segments.append(PathSegment(
            PathCommand.CUBIC,
            [Vec2(c1x, c1y), Vec2(c2x, c2y), Vec2(x, y)]
        ))
        return self

    def close(self) -> PathNode:
        self.segments.append(PathSegment(PathCommand.CLOSE))
        return self

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        all_points = []
        for seg in self.segments:
            all_points.extend(seg.points)
        if not all_points:
            return (Vec2(0, 0), Vec2(0, 0))
        min_x = min(p.x for p in all_points)
        min_y = min(p.y for p in all_points)
        max_x = max(p.x for p in all_points)
        max_y = max(p.y for p in all_points)
        return (Vec2(min_x, min_y), Vec2(max_x, max_y))


# =============================================================================
# TextNode
# =============================================================================

class TextAlign(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class TextBaseline(Enum):
    TOP = auto()
    MIDDLE = auto()
    ALPHABETIC = auto()
    BOTTOM = auto()


@dataclass
class TextNode(DrawNode):
    """
    Text rendering primitive.
    """
    text: str = ""
    x: float = 0.0
    y: float = 0.0
    color: Color = field(default_factory=Color.white)
    font_size: float = 14.0
    font_family: str = "default"
    font_weight: str = "normal"  # normal, bold
    align: TextAlign = TextAlign.LEFT
    baseline: TextBaseline = TextBaseline.ALPHABETIC
    max_width: Optional[float] = None  # Wrap or scale to fit

    def __post_init__(self):
        super().__init__()

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        # Approximate bounds - real bounds need font metrics
        approx_width = len(self.text) * self.font_size * 0.6
        approx_height = self.font_size
        return (Vec2(self.x, self.y), Vec2(self.x + approx_width, self.y + approx_height))


# =============================================================================
# ImageNode
# =============================================================================

@dataclass
class ImageNode(DrawNode):
    """
    Textured quad for images or sprites.
    """
    texture_id: str = ""  # Reference to texture in resource registry
    x: float = 0.0
    y: float = 0.0
    width: float = 100.0
    height: float = 100.0
    uv_rect: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # u0, v0, u1, v1
    tint: Color = field(default_factory=Color.white)  # Multiplied with texture

    def __post_init__(self):
        super().__init__()

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        return (Vec2(self.x, self.y), Vec2(self.x + self.width, self.y + self.height))


# =============================================================================
# PointsNode (for scatter plots, hit markers, etc.)
# =============================================================================

@dataclass
class PointsNode(DrawNode):
    """
    Multiple points rendered as circles or sprites.
    Efficient for many small markers (instanced rendering).
    """
    positions: List[Vec2] = field(default_factory=list)
    colors: Optional[List[Color]] = None  # Per-point colors, or None for uniform
    size: float = 4.0
    shape: str = "circle"  # circle, square, diamond
    color: Color = field(default_factory=Color.white)  # Uniform color if colors is None

    def __post_init__(self):
        super().__init__()

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        if not self.positions:
            return (Vec2(0, 0), Vec2(0, 0))
        half = self.size / 2
        min_x = min(p.x for p in self.positions) - half
        min_y = min(p.y for p in self.positions) - half
        max_x = max(p.x for p in self.positions) + half
        max_y = max(p.y for p in self.positions) + half
        return (Vec2(min_x, min_y), Vec2(max_x, max_y))


# =============================================================================
# GridLinesNode (optimized for many parallel lines)
# =============================================================================

@dataclass
class GridLinesNode(DrawNode):
    """
    Optimized node for rendering many parallel lines (grid, waveform ruler).
    Uses instanced rendering when above threshold.
    """
    # Vertical lines at these X positions
    vertical_x: List[float] = field(default_factory=list)
    vertical_y0: float = 0.0
    vertical_y1: float = 100.0
    vertical_color: Color = field(default_factory=lambda: Color(0.3, 0.3, 0.3, 1.0))
    vertical_width: float = 1.0

    # Horizontal lines at these Y positions
    horizontal_y: List[float] = field(default_factory=list)
    horizontal_x0: float = 0.0
    horizontal_x1: float = 100.0
    horizontal_color: Color = field(default_factory=lambda: Color(0.3, 0.3, 0.3, 1.0))
    horizontal_width: float = 1.0

    def __post_init__(self):
        super().__init__()

    def get_bounds(self) -> Tuple[Vec2, Vec2]:
        min_x = min(self.vertical_x) if self.vertical_x else self.horizontal_x0
        max_x = max(self.vertical_x) if self.vertical_x else self.horizontal_x1
        min_y = min(self.horizontal_y) if self.horizontal_y else self.vertical_y0
        max_y = max(self.horizontal_y) if self.horizontal_y else self.vertical_y1
        return (Vec2(min_x, min_y), Vec2(max_x, max_y))

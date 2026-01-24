# engine/graph/coordinates.py
"""
Coordinate Spaces - Different ways to interpret 2D positions.

Coordinate spaces transform logical coordinates to screen coordinates:

- Cartesian2D: Standard X/Y pixel space with configurable origin
- Polar: Angle/radius from a center point (for circular displays)
- Projected3D: 3D points projected to 2D via camera matrices

Each space provides bidirectional transformation between its native
coordinate system and screen pixels.
"""

from __future__ import annotations
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Optional
from enum import Enum, auto

from engine.core.math3d import Vec2, Vec3, Vec4, Mat3, Mat4


# =============================================================================
# Base Coordinate Space
# =============================================================================

class CoordinateSpace(ABC):
    """
    Abstract base for coordinate space transformations.

    Coordinate spaces define how logical coordinates map to screen pixels.
    """

    @abstractmethod
    def to_screen(self, *args) -> Vec2:
        """Convert from space coordinates to screen pixels."""
        pass

    @abstractmethod
    def from_screen(self, screen: Vec2) -> Tuple:
        """Convert from screen pixels to space coordinates."""
        pass

    @abstractmethod
    def get_transform_matrix(self) -> Mat3:
        """Get the transformation matrix for this space."""
        pass


# =============================================================================
# Cartesian2D
# =============================================================================

class Origin(Enum):
    """Origin position for Cartesian coordinate space."""
    TOP_LEFT = auto()      # Standard screen coordinates
    CENTER = auto()        # Origin at center of viewport
    BOTTOM_LEFT = auto()   # Math/OpenGL style


@dataclass
class Cartesian2D(CoordinateSpace):
    """
    Standard 2D Cartesian coordinate space.

    Supports:
    - Configurable origin (top-left, center, bottom-left)
    - Scale (zoom)
    - Offset (pan)
    - Y-flip for different conventions

    Attributes:
        origin: Where (0,0) is located
        viewport_width: Width in pixels
        viewport_height: Height in pixels
        scale: Zoom factor (pixels per unit)
        offset: Pan offset in units
        flip_y: Whether Y increases upward
    """
    origin: Origin = Origin.TOP_LEFT
    viewport_width: float = 800.0
    viewport_height: float = 600.0
    scale: Vec2 = field(default_factory=lambda: Vec2(1.0, 1.0))
    offset: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    flip_y: bool = False

    def to_screen(self, x: float, y: float) -> Vec2:
        """Convert Cartesian coordinates to screen pixels."""
        # Apply offset
        x = x - self.offset.x
        y = y - self.offset.y

        # Apply scale
        x = x * self.scale.x
        y = y * self.scale.y

        # Apply Y flip if needed
        if self.flip_y:
            y = -y

        # Apply origin offset
        if self.origin == Origin.CENTER:
            x += self.viewport_width / 2
            y += self.viewport_height / 2
        elif self.origin == Origin.BOTTOM_LEFT:
            y = self.viewport_height - y

        return Vec2(x, y)

    def from_screen(self, screen: Vec2) -> Tuple[float, float]:
        """Convert screen pixels to Cartesian coordinates."""
        x, y = screen.x, screen.y

        # Reverse origin offset
        if self.origin == Origin.CENTER:
            x -= self.viewport_width / 2
            y -= self.viewport_height / 2
        elif self.origin == Origin.BOTTOM_LEFT:
            y = self.viewport_height - y

        # Reverse Y flip
        if self.flip_y:
            y = -y

        # Reverse scale
        if self.scale.x != 0:
            x = x / self.scale.x
        if self.scale.y != 0:
            y = y / self.scale.y

        # Reverse offset
        x = x + self.offset.x
        y = y + self.offset.y

        return (x, y)

    def get_transform_matrix(self) -> Mat3:
        """Get the 2D transformation matrix."""
        # Build: translate(-offset) * scale * flip * origin_offset
        tx, ty = -self.offset.x, -self.offset.y
        sx, sy = self.scale.x, self.scale.y

        if self.flip_y:
            sy = -sy

        ox, oy = 0.0, 0.0
        if self.origin == Origin.CENTER:
            ox = self.viewport_width / 2
            oy = self.viewport_height / 2
        elif self.origin == Origin.BOTTOM_LEFT:
            oy = self.viewport_height

        return Mat3((
            sx,  0.0, tx * sx + ox,
            0.0, sy,  ty * sy + oy,
            0.0, 0.0, 1.0
        ))

    def set_viewport(self, width: float, height: float):
        """Update viewport dimensions."""
        self.viewport_width = width
        self.viewport_height = height

    def pan(self, dx: float, dy: float):
        """Pan the view by delta in screen pixels."""
        self.offset = Vec2(
            self.offset.x - dx / self.scale.x,
            self.offset.y - dy / self.scale.y
        )

    def zoom(self, factor: float, center: Vec2 = None):
        """
        Zoom by a factor, optionally around a center point.

        Args:
            factor: Zoom multiplier (>1 zooms in, <1 zooms out)
            center: Screen position to zoom around (default: viewport center)
        """
        if center is None:
            center = Vec2(self.viewport_width / 2, self.viewport_height / 2)

        # Convert center to world coordinates before zoom
        world_center = self.from_screen(center)

        # Apply zoom
        self.scale = Vec2(self.scale.x * factor, self.scale.y * factor)

        # Adjust offset so world_center stays at screen center
        new_screen = self.to_screen(world_center[0], world_center[1])
        dx = center.x - new_screen.x
        dy = center.y - new_screen.y
        self.offset = Vec2(
            self.offset.x + dx / self.scale.x,
            self.offset.y + dy / self.scale.y
        )


# =============================================================================
# Polar
# =============================================================================

@dataclass
class Polar(CoordinateSpace):
    """
    Polar coordinate space for circular displays.

    Perfect for beat circles, phase indicators, radial menus.

    Coordinates:
    - angle: Radians from the start angle (default: 0 = 3 o'clock)
    - radius: Distance from center

    Attributes:
        center: Center point in screen pixels
        radius_scale: Pixels per unit radius
        start_angle: Angle offset (0 = right, PI/2 = top)
        clockwise: Direction of increasing angle
    """
    center: Vec2 = field(default_factory=lambda: Vec2(400.0, 300.0))
    radius_scale: float = 1.0  # Pixels per unit
    start_angle: float = -math.pi / 2  # Start at top (12 o'clock)
    clockwise: bool = True

    def to_screen(self, angle: float, radius: float) -> Vec2:
        """Convert polar (angle, radius) to screen pixels."""
        # Adjust angle for start offset and direction
        if self.clockwise:
            angle = self.start_angle - angle
        else:
            angle = self.start_angle + angle

        # Convert to Cartesian
        x = self.center.x + math.cos(angle) * radius * self.radius_scale
        y = self.center.y + math.sin(angle) * radius * self.radius_scale

        return Vec2(x, y)

    def from_screen(self, screen: Vec2) -> Tuple[float, float]:
        """Convert screen pixels to polar (angle, radius)."""
        dx = screen.x - self.center.x
        dy = screen.y - self.center.y

        radius = math.sqrt(dx * dx + dy * dy) / self.radius_scale
        angle = math.atan2(dy, dx)

        # Adjust for start angle and direction
        if self.clockwise:
            angle = self.start_angle - angle
        else:
            angle = angle - self.start_angle

        # Normalize angle to [0, 2*PI)
        angle = angle % (2 * math.pi)

        return (angle, radius)

    def get_transform_matrix(self) -> Mat3:
        """
        Polar doesn't have a simple linear transform matrix.
        Returns identity - actual transform is nonlinear.
        """
        return Mat3.identity()

    def angle_to_screen_direction(self, angle: float) -> Vec2:
        """Get the screen-space direction for an angle."""
        if self.clockwise:
            angle = self.start_angle - angle
        else:
            angle = self.start_angle + angle

        return Vec2(math.cos(angle), math.sin(angle))

    def arc_points(
        self,
        start_angle: float,
        end_angle: float,
        radius: float,
        segments: int = 32
    ) -> list:
        """Generate screen points for an arc."""
        points = []
        delta = (end_angle - start_angle) / segments

        for i in range(segments + 1):
            angle = start_angle + delta * i
            points.append(self.to_screen(angle, radius))

        return points


# =============================================================================
# Projected3D
# =============================================================================

@dataclass
class Projected3D(CoordinateSpace):
    """
    3D-to-2D projection coordinate space.

    Projects 3D world coordinates to 2D screen pixels using
    standard view and projection matrices.

    Useful for:
    - Rendering 3D scene objects in a 2D overlay
    - 3D UI elements
    - Perspective-aware 2D effects
    """
    view: Mat4 = field(default_factory=Mat4.identity)
    projection: Mat4 = field(default_factory=Mat4.identity)
    viewport_width: float = 800.0
    viewport_height: float = 600.0

    # Cached combined matrix
    _view_proj: Mat4 = field(default=None, init=False, repr=False)
    _dirty: bool = field(default=True, init=False, repr=False)

    def _update_matrix(self):
        """Update cached view-projection matrix."""
        if self._dirty:
            self._view_proj = self.projection @ self.view
            self._dirty = False

    def set_view(self, view: Mat4):
        """Set the view matrix."""
        self.view = view
        self._dirty = True

    def set_projection(self, projection: Mat4):
        """Set the projection matrix."""
        self.projection = projection
        self._dirty = True

    def set_perspective(
        self,
        fov_y: float,
        aspect: float = None,
        near: float = 0.1,
        far: float = 1000.0
    ):
        """Set up perspective projection."""
        if aspect is None:
            aspect = self.viewport_width / self.viewport_height
        self.projection = Mat4.perspective(fov_y, aspect, near, far)
        self._dirty = True

    def set_orthographic(
        self,
        left: float = None,
        right: float = None,
        bottom: float = None,
        top: float = None,
        near: float = -1.0,
        far: float = 1.0
    ):
        """Set up orthographic projection."""
        if left is None:
            left = 0
        if right is None:
            right = self.viewport_width
        if bottom is None:
            bottom = self.viewport_height
        if top is None:
            top = 0

        self.projection = Mat4.ortho(left, right, bottom, top, near, far)
        self._dirty = True

    def look_at(self, eye: Vec3, target: Vec3, up: Vec3 = None):
        """Set view matrix to look at a target."""
        self.view = Mat4.look_at(eye, target, up)
        self._dirty = True

    def to_screen(self, x: float, y: float, z: float) -> Vec2:
        """Project 3D point to screen pixels."""
        self._update_matrix()

        # Transform point
        p = self._view_proj @ Vec4.point(x, y, z)

        # Perspective divide
        if abs(p.w) < 1e-10:
            return Vec2(0, 0)

        ndc_x = p.x / p.w
        ndc_y = p.y / p.w

        # NDC to screen (NDC is -1 to 1)
        screen_x = (ndc_x + 1) * 0.5 * self.viewport_width
        screen_y = (1 - ndc_y) * 0.5 * self.viewport_height  # Flip Y

        return Vec2(screen_x, screen_y)

    def to_screen_vec3(self, point: Vec3) -> Vec2:
        """Project Vec3 to screen pixels."""
        return self.to_screen(point.x, point.y, point.z)

    def from_screen(self, screen: Vec2, depth: float = 0.0) -> Tuple[float, float, float]:
        """
        Unproject screen point to 3D (requires depth).

        Args:
            screen: Screen position in pixels
            depth: Normalized depth (0 = near, 1 = far)

        Returns:
            (x, y, z) world coordinates
        """
        self._update_matrix()

        # Screen to NDC
        ndc_x = (screen.x / self.viewport_width) * 2 - 1
        ndc_y = 1 - (screen.y / self.viewport_height) * 2  # Flip Y
        ndc_z = depth * 2 - 1  # Map [0,1] to [-1,1]

        # Inverse view-projection
        inv = self._view_proj.inverse()
        p = inv @ Vec4(ndc_x, ndc_y, ndc_z, 1.0)

        if abs(p.w) < 1e-10:
            return (0.0, 0.0, 0.0)

        return (p.x / p.w, p.y / p.w, p.z / p.w)

    def get_transform_matrix(self) -> Mat3:
        """
        3D projection is 4x4, not 3x3.
        Returns a 2D approximation (identity).
        """
        return Mat3.identity()

    def get_ray(self, screen: Vec2) -> Tuple[Vec3, Vec3]:
        """
        Get a ray from screen point into the scene.

        Returns:
            (origin, direction) of the ray
        """
        near = self.from_screen(screen, 0.0)
        far = self.from_screen(screen, 1.0)

        origin = Vec3(*near)
        direction = Vec3(
            far[0] - near[0],
            far[1] - near[1],
            far[2] - near[2]
        ).normalized()

        return (origin, direction)


# =============================================================================
# Beat Space (specialized for music timing)
# =============================================================================

@dataclass
class BeatSpace(CoordinateSpace):
    """
    Coordinate space where X is measured in beats.

    Integrates with TimeCamera for beat-synchronized displays.

    Coordinates:
    - x: Beat number (0.0 = start, 4.0 = start of bar 2 in 4/4)
    - y: Standard pixel Y

    Attributes:
        left_beat: Leftmost visible beat
        pixels_per_beat: Horizontal scale
        viewport_width: Width in pixels
        viewport_height: Height in pixels
    """
    left_beat: float = 0.0
    pixels_per_beat: float = 100.0
    viewport_width: float = 800.0
    viewport_height: float = 600.0

    def to_screen(self, beat: float, y: float) -> Vec2:
        """Convert (beat, y) to screen pixels."""
        x = (beat - self.left_beat) * self.pixels_per_beat
        return Vec2(x, y)

    def from_screen(self, screen: Vec2) -> Tuple[float, float]:
        """Convert screen pixels to (beat, y)."""
        beat = self.left_beat + screen.x / self.pixels_per_beat
        return (beat, screen.y)

    def get_transform_matrix(self) -> Mat3:
        """Get transformation matrix."""
        return Mat3((
            self.pixels_per_beat, 0.0, -self.left_beat * self.pixels_per_beat,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ))

    def beat_to_x(self, beat: float) -> float:
        """Convert beat to X pixel position."""
        return (beat - self.left_beat) * self.pixels_per_beat

    def x_to_beat(self, x: float) -> float:
        """Convert X pixel position to beat."""
        return self.left_beat + x / self.pixels_per_beat

    def get_visible_range(self) -> Tuple[float, float]:
        """Get the range of visible beats."""
        start = self.left_beat
        end = self.left_beat + self.viewport_width / self.pixels_per_beat
        return (start, end)

    def scroll_to_beat(self, beat: float, anchor: float = 0.5):
        """
        Scroll so that a beat is at a specific horizontal position.

        Args:
            beat: Target beat
            anchor: Position in viewport (0=left, 0.5=center, 1=right)
        """
        offset_pixels = self.viewport_width * anchor
        self.left_beat = beat - offset_pixels / self.pixels_per_beat

    def zoom(self, factor: float, anchor_x: float = None):
        """
        Zoom by factor, keeping anchor point fixed.

        Args:
            factor: Zoom multiplier (>1 zooms in)
            anchor_x: Screen X to keep fixed (default: center)
        """
        if anchor_x is None:
            anchor_x = self.viewport_width / 2

        # Beat at anchor before zoom
        anchor_beat = self.x_to_beat(anchor_x)

        # Apply zoom
        self.pixels_per_beat *= factor

        # Adjust left_beat to keep anchor_beat at anchor_x
        self.left_beat = anchor_beat - anchor_x / self.pixels_per_beat

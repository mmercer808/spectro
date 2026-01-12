"""
Style System

Flat CSS-like styles without cascading or selectors.
Each widget has its own Style instance.

Design principles:
- No inheritance/cascading (explicit is better)
- No string parsing (direct values)
- Immutable after creation (use replace() for variants)
- All measurements in pixels (no units parsing)
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Tuple, Optional, Union
import numpy as np


# =============================================================================
# Color
# =============================================================================

# Color can be:
# - Tuple of 3-4 floats (RGB or RGBA, 0.0-1.0)
# - None (transparent/inherit)
Color = Optional[Tuple[float, ...]]


def color_rgba(c: Color) -> Tuple[float, float, float, float]:
    """Normalize color to RGBA tuple."""
    if c is None:
        return (0.0, 0.0, 0.0, 0.0)
    if len(c) == 3:
        return (c[0], c[1], c[2], 1.0)
    return (c[0], c[1], c[2], c[3])


def color_to_array(c: Color) -> np.ndarray:
    """Convert color to numpy array."""
    return np.array(color_rgba(c), dtype=np.float32)


def hex_to_color(hex_str: str) -> Color:
    """Convert hex string to color. Supports #RGB, #RGBA, #RRGGBB, #RRGGBBAA."""
    h = hex_str.lstrip('#')
    if len(h) == 3:
        r, g, b = int(h[0], 16) / 15, int(h[1], 16) / 15, int(h[2], 16) / 15
        return (r, g, b, 1.0)
    elif len(h) == 4:
        r, g, b, a = int(h[0], 16) / 15, int(h[1], 16) / 15, int(h[2], 16) / 15, int(h[3], 16) / 15
        return (r, g, b, a)
    elif len(h) == 6:
        r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
        return (r, g, b, 1.0)
    elif len(h) == 8:
        r, g, b, a = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, int(h[6:8], 16) / 255
        return (r, g, b, a)
    raise ValueError(f"Invalid hex color: {hex_str}")


# =============================================================================
# Edge Insets (padding, margin)
# =============================================================================

@dataclass(frozen=True)
class EdgeInsets:
    """
    Insets for padding/margin (top, right, bottom, left).
    Follows CSS order: top, right, bottom, left.
    """
    top: float = 0.0
    right: float = 0.0
    bottom: float = 0.0
    left: float = 0.0

    @staticmethod
    def all(value: float) -> EdgeInsets:
        """Same value on all sides."""
        return EdgeInsets(value, value, value, value)

    @staticmethod
    def symmetric(vertical: float = 0.0, horizontal: float = 0.0) -> EdgeInsets:
        """Symmetric vertical and horizontal."""
        return EdgeInsets(vertical, horizontal, vertical, horizontal)

    @staticmethod
    def only(top: float = 0.0, right: float = 0.0, bottom: float = 0.0, left: float = 0.0) -> EdgeInsets:
        """Explicit sides."""
        return EdgeInsets(top, right, bottom, left)

    @property
    def horizontal(self) -> float:
        """Total horizontal inset."""
        return self.left + self.right

    @property
    def vertical(self) -> float:
        """Total vertical inset."""
        return self.top + self.bottom


# =============================================================================
# Border
# =============================================================================

@dataclass(frozen=True)
class Border:
    """Border style."""
    width: float = 1.0
    color: Color = (0.5, 0.5, 0.5, 1.0)
    radius: float = 0.0  # Corner radius


# =============================================================================
# Shadow
# =============================================================================

@dataclass(frozen=True)
class Shadow:
    """Box shadow."""
    offset_x: float = 0.0
    offset_y: float = 4.0
    blur: float = 8.0
    spread: float = 0.0
    color: Color = (0.0, 0.0, 0.0, 0.3)


# =============================================================================
# Size Value
# =============================================================================

@dataclass(frozen=True)
class SizeValue:
    """
    Size value that can be:
    - Fixed pixels
    - Percentage of parent
    - Auto (content-based)
    - Fill (expand to available)
    """
    value: float = 0.0
    unit: str = "auto"  # "px", "pct", "auto", "fill"

    @staticmethod
    def px(value: float) -> SizeValue:
        return SizeValue(value, "px")

    @staticmethod
    def pct(value: float) -> SizeValue:
        return SizeValue(value, "pct")

    @staticmethod
    def auto() -> SizeValue:
        return SizeValue(0.0, "auto")

    @staticmethod
    def fill() -> SizeValue:
        return SizeValue(0.0, "fill")

    def resolve(self, available: float, content: float = 0.0) -> float:
        """Resolve to actual pixels given available space and content size."""
        if self.unit == "px":
            return self.value
        elif self.unit == "pct":
            return available * (self.value / 100.0)
        elif self.unit == "fill":
            return available
        else:  # auto
            return content

    def is_fixed(self) -> bool:
        return self.unit == "px"

    def is_flexible(self) -> bool:
        return self.unit in ("pct", "fill")


# =============================================================================
# Style
# =============================================================================

@dataclass(frozen=True)
class Style:
    """
    Complete style for a widget.

    Immutable - use dataclasses.replace() or .with_*() methods for variants.
    All measurements in pixels (no unit parsing).
    """

    # --- Box Model ---
    width: SizeValue = field(default_factory=SizeValue.auto)
    height: SizeValue = field(default_factory=SizeValue.auto)
    min_width: Optional[float] = None
    min_height: Optional[float] = None
    max_width: Optional[float] = None
    max_height: Optional[float] = None

    padding: EdgeInsets = field(default_factory=lambda: EdgeInsets.all(0))
    margin: EdgeInsets = field(default_factory=lambda: EdgeInsets.all(0))

    # --- Visual ---
    background: Color = None
    border: Optional[Border] = None
    shadow: Optional[Shadow] = None
    opacity: float = 1.0

    # --- Text ---
    font_size: float = 14.0
    font_color: Color = (1.0, 1.0, 1.0, 1.0)
    font_weight: str = "normal"  # "normal", "bold"
    text_align: str = "left"  # "left", "center", "right"
    line_height: float = 1.4

    # --- Flex Item ---
    flex_grow: float = 0.0
    flex_shrink: float = 1.0
    align_self: str = "auto"  # "auto", "start", "end", "center", "stretch"

    # --- Interaction ---
    cursor: str = "default"  # "default", "pointer", "text", "move", "resize"
    pointer_events: bool = True  # False to pass through

    # -------------------------------------------------------------------------
    # Builder Methods
    # -------------------------------------------------------------------------

    def with_size(self, width: SizeValue = None, height: SizeValue = None) -> Style:
        """Return new Style with updated size."""
        return replace(
            self,
            width=width if width is not None else self.width,
            height=height if height is not None else self.height,
        )

    def with_padding(self, padding: EdgeInsets) -> Style:
        """Return new Style with updated padding."""
        return replace(self, padding=padding)

    def with_margin(self, margin: EdgeInsets) -> Style:
        """Return new Style with updated margin."""
        return replace(self, margin=margin)

    def with_background(self, color: Color) -> Style:
        """Return new Style with updated background."""
        return replace(self, background=color)

    def with_border(self, border: Optional[Border]) -> Style:
        """Return new Style with updated border."""
        return replace(self, border=border)

    def with_text(
        self,
        size: float = None,
        color: Color = None,
        weight: str = None,
        align: str = None,
    ) -> Style:
        """Return new Style with updated text properties."""
        return replace(
            self,
            font_size=size if size is not None else self.font_size,
            font_color=color if color is not None else self.font_color,
            font_weight=weight if weight is not None else self.font_weight,
            text_align=align if align is not None else self.text_align,
        )

    def with_flex(self, grow: float = None, shrink: float = None) -> Style:
        """Return new Style with updated flex properties."""
        return replace(
            self,
            flex_grow=grow if grow is not None else self.flex_grow,
            flex_shrink=shrink if shrink is not None else self.flex_shrink,
        )


# =============================================================================
# Theme (collection of named styles)
# =============================================================================

@dataclass
class Theme:
    """
    Collection of named styles for consistent theming.
    Not a cascade - just a dictionary of reusable styles.
    """

    # Colors
    bg_primary: Color = (0.12, 0.12, 0.14, 1.0)
    bg_secondary: Color = (0.18, 0.18, 0.20, 1.0)
    bg_tertiary: Color = (0.22, 0.22, 0.25, 1.0)

    fg_primary: Color = (1.0, 1.0, 1.0, 1.0)
    fg_secondary: Color = (0.7, 0.7, 0.7, 1.0)
    fg_muted: Color = (0.5, 0.5, 0.5, 1.0)

    accent: Color = (0.35, 0.55, 0.95, 1.0)
    accent_hover: Color = (0.45, 0.65, 1.0, 1.0)

    border_color: Color = (0.3, 0.3, 0.35, 1.0)

    # Spacing
    spacing_xs: float = 4.0
    spacing_sm: float = 8.0
    spacing_md: float = 12.0
    spacing_lg: float = 16.0
    spacing_xl: float = 24.0

    # Border radius
    radius_sm: float = 4.0
    radius_md: float = 8.0
    radius_lg: float = 12.0

    # Font sizes
    font_xs: float = 10.0
    font_sm: float = 12.0
    font_md: float = 14.0
    font_lg: float = 18.0
    font_xl: float = 24.0

    def panel_style(self) -> Style:
        """Style for a panel container."""
        return Style(
            background=self.bg_primary,
            border=Border(width=1, color=self.border_color, radius=self.radius_md),
            padding=EdgeInsets.all(self.spacing_md),
        )

    def button_style(self) -> Style:
        """Style for a button."""
        return Style(
            background=self.accent,
            border=Border(width=0, color=None, radius=self.radius_sm),
            padding=EdgeInsets.symmetric(self.spacing_sm, self.spacing_md),
            font_color=self.fg_primary,
            font_size=self.font_md,
            text_align="center",
            cursor="pointer",
        )

    def label_style(self) -> Style:
        """Style for a label."""
        return Style(
            font_color=self.fg_primary,
            font_size=self.font_md,
        )

    def title_bar_style(self) -> Style:
        """Style for a panel title bar."""
        return Style(
            background=self.bg_tertiary,
            padding=EdgeInsets.symmetric(self.spacing_sm, self.spacing_md),
            font_color=self.fg_primary,
            font_size=self.font_md,
            font_weight="bold",
        )


# Default dark theme
DEFAULT_THEME = Theme()

"""
Container Widgets

Basic layout containers:
- Container: Generic flex container
- Spacer: Flexible empty space
- Divider: Visual separator line
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING

from engine.ui.widget import Widget
from engine.ui.style import Style, SizeValue, EdgeInsets, Color
from engine.ui.layout import FlexLayout, LayoutDirection, Rect, Constraints

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext


class Container(Widget):
    """
    Generic flex container.

    Just a Widget with convenient constructor for common patterns.
    """

    def __init__(
        self,
        children: List[Widget] = None,
        direction: LayoutDirection = LayoutDirection.COLUMN,
        gap: float = 0,
        padding: float = 0,
        background: Color = None,
        style: Style = None,
        **kwargs,
    ):
        # Build style
        if style is None:
            style = Style(
                padding=EdgeInsets.all(padding) if isinstance(padding, (int, float)) else padding,
                background=background,
            )

        # Build flex layout
        flex = FlexLayout(direction=direction, gap=gap)

        super().__init__(
            style=style,
            flex=flex,
            children=children,
            **kwargs,
        )


class Row(Container):
    """Horizontal flex container."""

    def __init__(self, children: List[Widget] = None, gap: float = 0, **kwargs):
        super().__init__(
            children=children,
            direction=LayoutDirection.ROW,
            gap=gap,
            **kwargs,
        )


class Column(Container):
    """Vertical flex container."""

    def __init__(self, children: List[Widget] = None, gap: float = 0, **kwargs):
        super().__init__(
            children=children,
            direction=LayoutDirection.COLUMN,
            gap=gap,
            **kwargs,
        )


class Spacer(Widget):
    """
    Flexible empty space.

    Grows to fill available space (flex_grow=1 by default).
    """

    def __init__(self, flex_grow: float = 1.0, min_size: float = 0):
        style = Style(
            flex_grow=flex_grow,
            min_width=min_size,
            min_height=min_size,
        )
        super().__init__(style=style)

    def measure(self, constraints: Constraints):
        # Return minimum size, flex will grow it
        return (
            self.style.min_width or 0,
            self.style.min_height or 0,
        )

    def draw(self, ctx: DrawContext):
        # Nothing to draw
        pass


class Divider(Widget):
    """
    Visual separator line.

    Orientation is inferred from parent flex direction.
    """

    def __init__(
        self,
        thickness: float = 1.0,
        color: Color = (0.3, 0.3, 0.35, 1.0),
        margin: float = 8.0,
    ):
        self._thickness = thickness
        self._color = color

        style = Style(
            margin=EdgeInsets.all(margin),
        )
        super().__init__(style=style)

    def measure(self, constraints: Constraints):
        # Determine orientation from parent
        if self.parent and hasattr(self.parent, 'flex'):
            if self.parent.flex.is_row():
                # Vertical divider in a row
                return (self._thickness, constraints.max_h)
            else:
                # Horizontal divider in a column
                return (constraints.max_w, self._thickness)
        return (constraints.max_w, self._thickness)

    def draw(self, ctx: DrawContext):
        if not self._layout:
            return

        rect = self._layout.rect
        ctx.draw_rect(rect, self._color)

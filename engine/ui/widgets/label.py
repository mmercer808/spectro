"""
Label Widget

Text display with styling.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from engine.ui.widget import Widget
from engine.ui.style import Style, Color, SizeValue
from engine.ui.layout import Rect, Constraints

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext


class Label(Widget):
    """
    Text label.

    Displays text with configurable style.
    Does not wrap text (single line).
    """

    def __init__(
        self,
        text: str = "",
        color: Color = (1.0, 1.0, 1.0, 1.0),
        font_size: float = 14.0,
        bold: bool = False,
        align: str = "left",
        style: Style = None,
    ):
        self._text = text

        if style is None:
            style = Style(
                font_color=color,
                font_size=font_size,
                font_weight="bold" if bold else "normal",
                text_align=align,
            )

        super().__init__(style=style)

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    def measure(self, constraints: Constraints):
        """Measure text size."""
        # Simple estimation: ~0.6 * font_size per character width
        # A real implementation would use font metrics
        char_width = self.style.font_size * 0.6
        text_width = len(self._text) * char_width
        text_height = self.style.font_size * self.style.line_height

        # Apply style constraints
        width = text_width + self.style.padding.horizontal
        height = text_height + self.style.padding.vertical

        if self.style.width.is_fixed():
            width = self.style.width.resolve(constraints.max_w)
        if self.style.height.is_fixed():
            height = self.style.height.resolve(constraints.max_h)

        return constraints.constrain(width, height)

    def draw(self, ctx: DrawContext):
        if not self._layout or not self._text:
            return

        rect = self._layout.rect
        style = self.style

        # Draw background if set
        if style.background is not None:
            ctx.draw_rect(rect, style.background)

        # Draw text
        content = self._layout.content_rect
        ctx.draw_text_in_rect(
            self._text,
            content,
            style.font_color,
            font_size=style.font_size,
            align=style.text_align,
            valign="center",
        )


class Heading(Label):
    """Large heading text."""

    def __init__(self, text: str = "", level: int = 1, **kwargs):
        sizes = {1: 24.0, 2: 20.0, 3: 18.0, 4: 16.0, 5: 14.0, 6: 12.0}
        font_size = sizes.get(level, 24.0)
        super().__init__(text=text, font_size=font_size, bold=True, **kwargs)


class Caption(Label):
    """Small caption text."""

    def __init__(self, text: str = "", color: Color = (0.6, 0.6, 0.6, 1.0), **kwargs):
        super().__init__(text=text, color=color, font_size=11.0, **kwargs)

"""
Button Widget

Clickable button with hover/press states.
"""

from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING
from dataclasses import replace

from engine.ui.widget import Widget, WidgetState, Event, EventType
from engine.ui.style import Style, Color, Border, EdgeInsets, SizeValue
from engine.ui.layout import Rect, Constraints

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext


class Button(Widget):
    """
    Clickable button.

    Features:
    - Text label
    - Hover/press visual feedback
    - Optional icon (future)
    - Disabled state
    """

    def __init__(
        self,
        text: str = "",
        on_click: Callable[[Event], None] = None,
        style: Style = None,
        hover_style: Style = None,
        pressed_style: Style = None,
        disabled_style: Style = None,
    ):
        self._text = text

        # Default styles
        if style is None:
            style = Style(
                background=(0.35, 0.55, 0.95, 1.0),
                border=Border(width=0, radius=4),
                padding=EdgeInsets.symmetric(8, 16),
                font_color=(1.0, 1.0, 1.0, 1.0),
                font_size=14.0,
                text_align="center",
                cursor="pointer",
            )

        # State variants (auto-generate if not provided)
        self._base_style = style
        self._hover_style = hover_style or self._make_hover_style(style)
        self._pressed_style = pressed_style or self._make_pressed_style(style)
        self._disabled_style = disabled_style or self._make_disabled_style(style)

        super().__init__(style=style, on_click=on_click)

    def _make_hover_style(self, base: Style) -> Style:
        """Generate hover style from base."""
        if base.background:
            # Lighten background
            r, g, b, a = base.background
            hover_bg = (
                min(1.0, r + 0.1),
                min(1.0, g + 0.1),
                min(1.0, b + 0.1),
                a,
            )
            return replace(base, background=hover_bg)
        return base

    def _make_pressed_style(self, base: Style) -> Style:
        """Generate pressed style from base."""
        if base.background:
            # Darken background
            r, g, b, a = base.background
            pressed_bg = (
                max(0.0, r - 0.1),
                max(0.0, g - 0.1),
                max(0.0, b - 0.1),
                a,
            )
            return replace(base, background=pressed_bg)
        return base

    def _make_disabled_style(self, base: Style) -> Style:
        """Generate disabled style from base."""
        # Desaturate and dim
        if base.background:
            r, g, b, a = base.background
            avg = (r + g + b) / 3
            disabled_bg = (avg * 0.5, avg * 0.5, avg * 0.5, a * 0.5)
            return replace(
                base,
                background=disabled_bg,
                font_color=(0.5, 0.5, 0.5, 0.5),
            )
        return base

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def current_style(self) -> Style:
        """Get style based on current state."""
        state = self.state
        if state == WidgetState.DISABLED:
            return self._disabled_style
        elif state == WidgetState.PRESSED:
            return self._pressed_style
        elif state == WidgetState.HOVERED:
            return self._hover_style
        return self._base_style

    def measure(self, constraints: Constraints):
        """Measure button size based on text."""
        style = self._base_style

        # Estimate text width
        char_width = style.font_size * 0.6
        text_width = len(self._text) * char_width
        text_height = style.font_size * style.line_height

        width = text_width + style.padding.horizontal
        height = text_height + style.padding.vertical

        # Apply fixed sizes if specified
        if style.width.is_fixed():
            width = style.width.resolve(constraints.max_w)
        if style.height.is_fixed():
            height = style.height.resolve(constraints.max_h)

        # Apply min sizes
        if style.min_width:
            width = max(style.min_width, width)
        if style.min_height:
            height = max(style.min_height, height)

        return constraints.constrain(width, height)

    def draw(self, ctx: DrawContext):
        if not self._layout:
            return

        rect = self._layout.rect
        style = self.current_style

        # Draw background
        if style.background:
            radius = style.border.radius if style.border else 0
            ctx.draw_rect(rect, style.background, radius=radius)

        # Draw border
        if style.border and style.border.width > 0:
            ctx.draw_rect_outline(
                rect,
                style.border.color,
                width=style.border.width,
                radius=style.border.radius,
            )

        # Draw text
        if self._text:
            content = self._layout.content_rect
            ctx.draw_text_in_rect(
                self._text,
                content,
                style.font_color,
                font_size=style.font_size,
                align="center",
                valign="center",
            )


class IconButton(Button):
    """
    Button with an icon.

    Icon is specified by name (requires icon font or atlas).
    """

    def __init__(
        self,
        icon: str = "",
        text: str = "",
        icon_size: float = 16.0,
        **kwargs,
    ):
        self._icon = icon
        self._icon_size = icon_size
        super().__init__(text=text, **kwargs)

    def draw(self, ctx: DrawContext):
        # For now, just draw as regular button
        # Icon rendering would require icon font/atlas
        super().draw(ctx)

        # TODO: Draw icon
        # if self._icon and self._layout:
        #     ctx.draw_icon(self._icon, x, y, self._icon_size)


class ToggleButton(Button):
    """
    Toggle button with on/off state.
    """

    def __init__(
        self,
        text: str = "",
        toggled: bool = False,
        on_toggle: Callable[[bool], None] = None,
        toggled_style: Style = None,
        **kwargs,
    ):
        self._toggled = toggled
        self._on_toggle = on_toggle
        self._toggled_style = toggled_style

        super().__init__(text=text, **kwargs)

        # Override click handler
        self.on(EventType.POINTER_UP, self._handle_toggle)

    @property
    def toggled(self) -> bool:
        return self._toggled

    @toggled.setter
    def toggled(self, value: bool):
        self._toggled = value

    def _handle_toggle(self, event: Event):
        self._toggled = not self._toggled
        if self._on_toggle:
            self._on_toggle(self._toggled)

    @property
    def current_style(self) -> Style:
        """Get style including toggle state."""
        if self._toggled and self._toggled_style:
            return self._toggled_style
        return super().current_style

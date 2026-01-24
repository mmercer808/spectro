"""
Panel Widget

Titled container with window chrome (title bar, close button, etc.)
"""

from __future__ import annotations
from typing import List, Optional, Callable, TYPE_CHECKING

from engine.ui.widget import Widget, Event, EventType
from engine.ui.style import Style, Color, Border, EdgeInsets, SizeValue
from engine.ui.layout import FlexLayout, LayoutDirection, Rect, Constraints, Align

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext


class TitleBar(Widget):
    """
    Panel title bar.

    Features:
    - Title text
    - Draggable for moving panel
    - Close/minimize/maximize buttons (optional)
    """

    def __init__(
        self,
        title: str = "",
        closable: bool = True,
        on_close: Callable[[], None] = None,
        on_drag: Callable[[float, float], None] = None,
        style: Style = None,
    ):
        self._title = title
        self._closable = closable
        self._on_close = on_close
        self._on_drag = on_drag

        # Drag state
        self._dragging = False
        self._drag_start_x = 0.0
        self._drag_start_y = 0.0

        if style is None:
            style = Style(
                background=(0.22, 0.22, 0.25, 1.0),
                padding=EdgeInsets.symmetric(6, 10),
                font_color=(1.0, 1.0, 1.0, 1.0),
                font_size=13.0,
                font_weight="bold",
                cursor="move",
            )

        super().__init__(style=style)

        # Register drag handlers
        self.on(EventType.POINTER_DOWN, self._on_pointer_down)
        self.on(EventType.POINTER_MOVE, self._on_pointer_move)
        self.on(EventType.POINTER_UP, self._on_pointer_up)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, value: str):
        self._title = value

    def _on_pointer_down(self, event: Event):
        self._dragging = True
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_pointer_move(self, event: Event):
        if self._dragging and self._on_drag:
            dx = event.x - self._drag_start_x
            dy = event.y - self._drag_start_y
            self._on_drag(dx, dy)
            # Reset drag start (continuous drag)
            self._drag_start_x = event.x
            self._drag_start_y = event.y

    def _on_pointer_up(self, event: Event):
        self._dragging = False

    def measure(self, constraints: Constraints):
        style = self.style

        # Title bar height based on font + padding
        height = style.font_size * 1.5 + style.padding.vertical

        # Close button space
        button_space = 24 if self._closable else 0

        # Title width estimation
        title_width = len(self._title) * style.font_size * 0.6

        width = title_width + button_space + style.padding.horizontal
        width = max(100, width)  # Minimum width

        return constraints.constrain(width, height)

    def draw(self, ctx: DrawContext):
        if not self._layout:
            return

        rect = self._layout.rect
        local_rect = Rect(0, 0, rect.w, rect.h)
        style = self.style

        ctx.push_offset(rect.x, rect.y)
        try:
            # Draw background
            if style.background:
                ctx.draw_rect(local_rect, style.background)

            # Draw title text
            content = self._layout.content_rect
            ctx.draw_text_in_rect(
                self._title,
                content,
                style.font_color,
                font_size=style.font_size,
                align="left",
                valign="center",
            )

            # Draw close button
            if self._closable:
                btn_size = 16
                btn_x = local_rect.right - btn_size - style.padding.right
                btn_y = local_rect.y + (local_rect.h - btn_size) / 2
                btn_rect = Rect(btn_x, btn_y, btn_size, btn_size)

                # Simple X icon
                ctx.draw_line(
                    btn_x + 4, btn_y + 4,
                    btn_x + btn_size - 4, btn_y + btn_size - 4,
                    (0.7, 0.7, 0.7, 1.0),
                    width=2,
                )
                ctx.draw_line(
                    btn_x + btn_size - 4, btn_y + 4,
                    btn_x + 4, btn_y + btn_size - 4,
                    (0.7, 0.7, 0.7, 1.0),
                    width=2,
                )
        finally:
            ctx.pop_offset()


class Panel(Widget):
    """
    Titled panel with optional chrome.

    Features:
    - Title bar with drag support
    - Content area with children
    - Optional close button
    - Border and shadow
    """

    def __init__(
        self,
        title: str = "",
        children: List[Widget] = None,
        closable: bool = True,
        on_close: Callable[[], None] = None,
        draggable: bool = True,
        width: float = None,
        height: float = None,
        style: Style = None,
        content_style: Style = None,
    ):
        self._on_close_callback = on_close
        self._draggable = draggable

        # Panel position (for floating panels)
        self._pos_x = 0.0
        self._pos_y = 0.0

        if style is None:
            style = Style(
                background=(0.14, 0.14, 0.16, 1.0),
                border=Border(width=1, color=(0.3, 0.3, 0.35, 1.0), radius=6),
                width=SizeValue.px(width) if width else SizeValue.auto(),
                height=SizeValue.px(height) if height else SizeValue.auto(),
            )

        # Use column layout
        flex = FlexLayout(direction=LayoutDirection.COLUMN, gap=0)

        super().__init__(style=style, flex=flex)

        # Create title bar
        self._title_bar = TitleBar(
            title=title,
            closable=closable,
            on_close=self._handle_close,
            on_drag=self._handle_drag if draggable else None,
        )
        self.add_child(self._title_bar)

        # Create content container
        if content_style is None:
            content_style = Style(
                padding=EdgeInsets.all(10),
                flex_grow=1.0,
            )

        self._content = Widget(
            style=content_style,
            flex=FlexLayout(direction=LayoutDirection.COLUMN, gap=8),
        )
        self.add_child(self._content)

        # Add children to content
        if children:
            for child in children:
                self._content.add_child(child)

    @property
    def title(self) -> str:
        return self._title_bar.title

    @title.setter
    def title(self, value: str):
        self._title_bar.title = value

    @property
    def content(self) -> Widget:
        """Get content container to add children."""
        return self._content

    def add_content(self, child: Widget):
        """Add a child to the content area."""
        self._content.add_child(child)

    def _handle_close(self):
        if self._on_close_callback:
            self._on_close_callback()

    def _handle_drag(self, dx: float, dy: float):
        """Handle title bar drag."""
        self._pos_x += dx
        self._pos_y += dy
        # Position update is handled by window manager

    def draw(self, ctx: DrawContext):
        if not self._layout:
            return

        rect = self._layout.rect
        local_rect = Rect(0, 0, rect.w, rect.h)
        style = self.style

        ctx.push_offset(rect.x, rect.y)
        try:
            # Draw shadow (simplified)
            # A real shadow would use blur shader or multiple rects

            # Draw background
            if style.background:
                radius = style.border.radius if style.border else 0
                ctx.draw_rect(local_rect, style.background, radius=radius)

            # Draw border
            if style.border and style.border.width > 0:
                ctx.draw_rect_outline(
                    local_rect,
                    style.border.color,
                    width=style.border.width,
                    radius=style.border.radius,
                )

            # Draw children (title bar and content)
            for child in self._children:
                child.draw(ctx)
        finally:
            ctx.pop_offset()


class FloatingPanel(Panel):
    """
    Panel that floats at an absolute position.

    Used for toolbars, palettes, detached panels.
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._pos_x = x
        self._pos_y = y

    @property
    def position(self):
        return (self._pos_x, self._pos_y)

    def set_position(self, x: float, y: float):
        self._pos_x = x
        self._pos_y = y

    def layout(self, rect: Rect):
        """Override layout to use absolute position."""
        # Measure to get preferred size
        w, h = self.measure(Constraints.loose(rect.w, rect.h))

        # Use absolute position
        absolute_rect = Rect(self._pos_x, self._pos_y, w, h)
        super().layout(absolute_rect)

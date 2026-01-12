"""
Viewport3D Widget

Widget that displays a 3D viewport's render target.
Bridges the UI system with the 3D rendering pipeline.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from engine.ui.widget import Widget, Event, EventType
from engine.ui.style import Style, Border
from engine.ui.layout import Rect, Constraints

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext
    from engine.viewport.viewport import ViewportArea


class Viewport3D(Widget):
    """
    3D viewport display widget.

    Displays the render target from a ViewportArea.
    Forwards input events to the ViewportArea for camera control.
    """

    def __init__(
        self,
        area: ViewportArea = None,
        style: Style = None,
    ):
        self._area = area

        if style is None:
            style = Style(
                flex_grow=1.0,
                border=Border(width=1, color=(0.2, 0.2, 0.25, 1.0), radius=0),
            )

        super().__init__(style=style)

        # Forward pointer events to viewport
        self.on(EventType.POINTER_DOWN, self._forward_pointer)
        self.on(EventType.POINTER_MOVE, self._forward_pointer)
        self.on(EventType.POINTER_UP, self._forward_pointer)
        self.on(EventType.SCROLL, self._forward_scroll)

    @property
    def area(self) -> Optional[ViewportArea]:
        return self._area

    @area.setter
    def area(self, value: ViewportArea):
        self._area = value

    def _forward_pointer(self, event: Event):
        """Forward pointer event to ViewportArea."""
        if not self._area:
            return

        kind_map = {
            EventType.POINTER_DOWN: "press",
            EventType.POINTER_UP: "release",
            EventType.POINTER_MOVE: "move",
        }
        kind = kind_map.get(event.type)
        if kind:
            self._area.handle_pointer(
                kind,
                event.x,
                event.y,
                event.delta_x,
                event.delta_y,
                button=event.button,
            )

    def _forward_scroll(self, event: Event):
        """Forward scroll event to ViewportArea."""
        if not self._area or not self._layout:
            return

        # Create a minimal PanelRect for the wheel handler
        from engine.viewport.viewport import PanelRect
        rect = self._layout.rect
        panel_rect = PanelRect(
            id=self._area.area_id,
            x=int(rect.x),
            y=int(rect.y),
            w=int(rect.w),
            h=int(rect.h),
        )
        self._area.handle_wheel(event.delta_y, panel_rect)

    def measure(self, constraints: Constraints):
        """Viewports want to fill available space by default."""
        style = self.style

        # Default to filling available space
        width = constraints.max_w
        height = constraints.max_h

        if style.width.is_fixed():
            width = style.width.resolve(constraints.max_w)
        if style.height.is_fixed():
            height = style.height.resolve(constraints.max_h)

        if style.min_width:
            width = max(style.min_width, width)
        if style.min_height:
            height = max(style.min_height, height)

        return constraints.constrain(width, height)

    def draw(self, ctx: DrawContext):
        """
        Draw the viewport.

        The actual texture blitting is handled by the UIRenderer,
        which knows how to blit render targets.
        """
        if not self._layout:
            return

        rect = self._layout.rect
        style = self.style

        # Draw placeholder background if no viewport
        if not self._area or not self._area.surface_rt:
            ctx.draw_rect(rect, (0.1, 0.1, 0.12, 1.0))

            # Draw "No Viewport" text
            ctx.draw_text_in_rect(
                "No Viewport",
                rect,
                (0.4, 0.4, 0.4, 1.0),
                font_size=14,
                align="center",
                valign="center",
            )
        else:
            # Draw the viewport texture
            # This is a special draw call that the renderer handles
            ctx.draw_rect(
                rect,
                (1.0, 1.0, 1.0, 1.0),  # Tint (white = no tint)
                texture_id=f"viewport:{self._area.area_id}",
                uv=(0, 0, 1, 1),
            )

        # Draw border
        if style.border and style.border.width > 0:
            ctx.draw_rect_outline(
                rect,
                style.border.color,
                width=style.border.width,
                radius=style.border.radius,
            )

    def get_viewport_rect(self) -> Optional[Rect]:
        """Get the screen-space rect for render target sizing."""
        return self._layout.rect if self._layout else None


class ViewportToolbar(Widget):
    """
    Toolbar overlay for a 3D viewport.

    Contains camera controls, shading mode, grid toggle, etc.
    """

    def __init__(
        self,
        style: Style = None,
    ):
        if style is None:
            style = Style(
                background=(0.12, 0.12, 0.14, 0.9),
                padding=EdgeInsets.symmetric(4, 8),
            )

        from engine.ui.layout import FlexLayout, LayoutDirection
        flex = FlexLayout(direction=LayoutDirection.ROW, gap=4)

        super().__init__(style=style, flex=flex)

        # Add toolbar buttons (placeholder)
        # In a real implementation, these would be IconButtons
        from engine.ui.widgets.label import Label
        self.add_child(Label(text="Viewport Tools", font_size=11))


# Import EdgeInsets at module level for ViewportToolbar
from engine.ui.style import EdgeInsets

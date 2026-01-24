"""
Widget Base Class

Core widget tree with:
- Measure/layout/draw phases
- Event propagation
- Hit testing
- Focus management
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any, Tuple, TYPE_CHECKING
from enum import Enum, auto

from engine.ui.style import Style, DEFAULT_THEME
from engine.ui.layout import (
    FlexLayout, LayoutDirection, Rect, Constraints,
    layout_flex, measure_flex,
)

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext


# =============================================================================
# Layout Result
# =============================================================================

@dataclass
class LayoutResult:
    """Result of layout pass, stored on widget."""
    rect: Rect  # Position and size in parent coordinates
    content_rect: Rect  # Inner rect after padding/border


# =============================================================================
# Event Types
# =============================================================================

class EventType(Enum):
    POINTER_DOWN = auto()
    POINTER_UP = auto()
    POINTER_MOVE = auto()
    POINTER_ENTER = auto()
    POINTER_LEAVE = auto()
    SCROLL = auto()
    KEY_DOWN = auto()
    KEY_UP = auto()
    FOCUS = auto()
    BLUR = auto()


@dataclass
class Event:
    """UI event."""
    type: EventType
    x: float = 0.0  # Position in widget-local coordinates
    y: float = 0.0
    button: int = 0  # Mouse button (1=left, 2=middle, 3=right)
    delta_x: float = 0.0  # Scroll delta
    delta_y: float = 0.0
    key: str = ""  # Key name
    modifiers: int = 0  # Modifier flags

    # Propagation control
    _stopped: bool = field(default=False, repr=False)
    _prevented: bool = field(default=False, repr=False)

    def stop_propagation(self):
        """Stop event from bubbling to parent."""
        self._stopped = True

    def prevent_default(self):
        """Prevent default behavior."""
        self._prevented = True

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def prevented(self) -> bool:
        return self._prevented


# Event handler signature
EventHandler = Callable[[Event], None]


# =============================================================================
# Widget State
# =============================================================================

class WidgetState(Enum):
    """Interactive state for styling."""
    NORMAL = auto()
    HOVERED = auto()
    PRESSED = auto()
    FOCUSED = auto()
    DISABLED = auto()


# =============================================================================
# Widget
# =============================================================================

class Widget:
    """
    Base class for all UI widgets.

    Lifecycle:
    1. measure(constraints) -> (width, height) - compute preferred size
    2. layout(rect) -> set position and layout children
    3. draw(ctx) -> emit draw commands
    4. handle_event(event) -> process input

    Tree structure:
    - parent: Optional[Widget]
    - children: List[Widget]
    """

    def __init__(
        self,
        style: Style = None,
        flex: FlexLayout = None,
        children: List[Widget] = None,
        on_click: EventHandler = None,
        on_pointer_down: EventHandler = None,
        on_pointer_up: EventHandler = None,
        on_pointer_move: EventHandler = None,
        on_scroll: EventHandler = None,
    ):
        # Style and layout
        self.style = style or Style()
        self.flex = flex or FlexLayout()

        # Tree
        self.parent: Optional[Widget] = None
        self._children: List[Widget] = []
        if children:
            for child in children:
                self.add_child(child)

        # Layout result (set during layout pass)
        self._layout: Optional[LayoutResult] = None

        # State
        self._state = WidgetState.NORMAL
        self._hovered = False
        self._pressed = False
        self._focused = False
        self._enabled = True
        self._visible = True

        # Event handlers
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        if on_click:
            self.on(EventType.POINTER_UP, on_click)
        if on_pointer_down:
            self.on(EventType.POINTER_DOWN, on_pointer_down)
        if on_pointer_up:
            self.on(EventType.POINTER_UP, on_pointer_up)
        if on_pointer_move:
            self.on(EventType.POINTER_MOVE, on_pointer_move)
        if on_scroll:
            self.on(EventType.SCROLL, on_scroll)

        # User data
        self.data: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Tree Management
    # -------------------------------------------------------------------------

    @property
    def children(self) -> List[Widget]:
        return self._children

    def add_child(self, child: Widget):
        """Add a child widget."""
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self._children.append(child)

    def remove_child(self, child: Widget):
        """Remove a child widget."""
        if child in self._children:
            self._children.remove(child)
            child.parent = None

    def clear_children(self):
        """Remove all children."""
        for child in self._children:
            child.parent = None
        self._children.clear()

    def get_root(self) -> Widget:
        """Get the root of the widget tree."""
        w = self
        while w.parent is not None:
            w = w.parent
        return w

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    @property
    def rect(self) -> Optional[Rect]:
        """Get layout rect (None if not laid out yet)."""
        return self._layout.rect if self._layout else None

    @property
    def content_rect(self) -> Optional[Rect]:
        """Get content rect (after padding)."""
        return self._layout.content_rect if self._layout else None

    def measure(self, constraints: Constraints) -> Tuple[float, float]:
        """
        Measure preferred size given constraints.

        Override in subclasses for custom measurement.
        Default: measure children with flex layout.
        """
        style = self.style

        # Start with style-defined sizes
        content_w = 0.0
        content_h = 0.0

        # Measure children if any
        if self._children:
            content_w, content_h = measure_flex(
                self._children, self.flex, constraints
            )

        # Add padding
        total_w = content_w + style.padding.horizontal
        total_h = content_h + style.padding.vertical

        # Resolve style sizes
        if style.width.is_fixed():
            total_w = style.width.resolve(constraints.max_w, total_w)
        elif style.width.is_flexible():
            total_w = style.width.resolve(constraints.max_w, total_w)

        if style.height.is_fixed():
            total_h = style.height.resolve(constraints.max_h, total_h)
        elif style.height.is_flexible():
            total_h = style.height.resolve(constraints.max_h, total_h)

        # Apply min/max constraints
        if style.min_width is not None:
            total_w = max(style.min_width, total_w)
        if style.min_height is not None:
            total_h = max(style.min_height, total_h)
        if style.max_width is not None:
            total_w = min(style.max_width, total_w)
        if style.max_height is not None:
            total_h = min(style.max_height, total_h)

        return constraints.constrain(total_w, total_h)

    def layout(self, rect: Rect):
        """
        Perform layout within the given rect.

        Sets self._layout and recursively lays out children.
        """
        style = self.style

        # Compute content rect in local coordinates (inside padding)
        content = Rect(
            x=style.padding.left,
            y=style.padding.top,
            w=max(0, rect.w - style.padding.horizontal),
            h=max(0, rect.h - style.padding.vertical),
        )

        self._layout = LayoutResult(rect=rect, content_rect=content)

        # Layout children with flex
        if self._children:
            child_rects = layout_flex(self._children, content, self.flex)
            for child, child_rect in zip(self._children, child_rects):
                child.layout(child_rect)

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def draw(self, ctx: DrawContext):
        """
        Draw this widget and its children.

        Override in subclasses for custom drawing.
        Default: draw background/border, then children.
        """
        if not self._visible or not self._layout:
            return

        rect = self._layout.rect
        local_rect = Rect(0, 0, rect.w, rect.h)
        style = self.style

        ctx.push_offset(rect.x, rect.y)
        try:
            # Draw background
            if style.background is not None:
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

            # Draw children
            for child in self._children:
                child.draw(ctx)
        finally:
            ctx.pop_offset()

    # -------------------------------------------------------------------------
    # Hit Testing
    # -------------------------------------------------------------------------

    def hit_test(self, x: float, y: float) -> Optional[Widget]:
        """
        Find the deepest widget at the given point.

        Returns None if point is outside this widget.
        Coordinates are in parent space.
        """
        if not self._visible or not self._layout:
            return None

        rect = self._layout.rect

        # Check if point is inside
        if not rect.contains(x, y):
            return None

        # Check if pointer events are disabled
        if not self.style.pointer_events:
            return None

        # Convert to local coordinates
        local_x = x - rect.x
        local_y = y - rect.y

        # Check children in reverse order (top-most first)
        for child in reversed(self._children):
            hit = child.hit_test(local_x, local_y)
            if hit is not None:
                return hit

        return self

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on(self, event_type: EventType, handler: EventHandler):
        """Register an event handler."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: EventType, handler: EventHandler):
        """Unregister an event handler."""
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def emit(self, event: Event):
        """Emit an event to registered handlers."""
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            if event.stopped:
                break
            handler(event)

    def handle_event(self, event: Event) -> bool:
        """
        Handle an event, with bubbling.

        Returns True if event was handled.
        """
        # Dispatch to handlers
        self.emit(event)

        # Bubble to parent if not stopped
        if not event.stopped and self.parent:
            # Translate coordinates to parent space
            if self._layout:
                event.x += self._layout.rect.x
                event.y += self._layout.rect.y
            self.parent.handle_event(event)

        return event.stopped

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    @property
    def state(self) -> WidgetState:
        """Get current interactive state."""
        if not self._enabled:
            return WidgetState.DISABLED
        if self._pressed:
            return WidgetState.PRESSED
        if self._focused:
            return WidgetState.FOCUSED
        if self._hovered:
            return WidgetState.HOVERED
        return WidgetState.NORMAL

    def set_hovered(self, hovered: bool):
        """Set hover state."""
        if self._hovered != hovered:
            self._hovered = hovered
            if hovered:
                self.emit(Event(EventType.POINTER_ENTER))
            else:
                self.emit(Event(EventType.POINTER_LEAVE))

    def set_pressed(self, pressed: bool):
        """Set pressed state."""
        self._pressed = pressed

    def set_focused(self, focused: bool):
        """Set focus state."""
        if self._focused != focused:
            self._focused = focused
            if focused:
                self.emit(Event(EventType.FOCUS))
            else:
                self.emit(Event(EventType.BLUR))

    def set_enabled(self, enabled: bool):
        """Set enabled state."""
        self._enabled = enabled

    def set_visible(self, visible: bool):
        """Set visibility."""
        self._visible = visible

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def visible(self) -> bool:
        return self._visible

    # -------------------------------------------------------------------------
    # Debug
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        name = self.__class__.__name__
        rect = self._layout.rect if self._layout else None
        return f"{name}(rect={rect}, children={len(self._children)})"

    def print_tree(self, indent: int = 0):
        """Print widget tree for debugging."""
        prefix = "  " * indent
        print(f"{prefix}{self}")
        for child in self._children:
            child.print_tree(indent + 1)

    def get_absolute_rect(self) -> Optional[Rect]:
        """Get this widget's rect in window coordinates."""
        if self._layout is None:
            return None

        x = self._layout.rect.x
        y = self._layout.rect.y
        w = self._layout.rect.w
        h = self._layout.rect.h

        parent = self.parent
        while parent is not None and parent._layout is not None:
            x += parent._layout.rect.x
            y += parent._layout.rect.y
            parent = parent.parent

        return Rect(x, y, w, h)


# =============================================================================
# Root Widget
# =============================================================================

class RootWidget(Widget):
    """
    Special root widget that fills the window.

    Handles:
    - Window resize
    - Global event dispatch
    - Focus management
    """

    def __init__(self, width: int = 800, height: int = 600, **kwargs):
        super().__init__(**kwargs)
        self._window_width = width
        self._window_height = height
        self._focused_widget: Optional[Widget] = None
        self._hovered_widget: Optional[Widget] = None
        self._captured_widget: Optional[Widget] = None  # For drag

    def set_window_size(self, width: int, height: int):
        """Update window size and re-layout."""
        self._window_width = width
        self._window_height = height
        self.do_layout()

    def do_layout(self):
        """Perform full layout pass."""
        rect = Rect(0, 0, self._window_width, self._window_height)
        self.layout(rect)

    def dispatch_pointer_event(
        self,
        event_type: EventType,
        x: float,
        y: float,
        button: int = 0,
        delta_x: float = 0,
        delta_y: float = 0,
    ):
        """Dispatch a pointer event to the appropriate widget."""
        # Handle capture (during drag)
        if self._captured_widget:
            target = self._captured_widget
            # Convert to widget-local coords
            local_x, local_y = self._to_local(target, x, y)
            event = Event(
                type=event_type,
                x=local_x,
                y=local_y,
                button=button,
                delta_x=delta_x,
                delta_y=delta_y,
            )
            target.handle_event(event)

            # Release capture on pointer up
            if event_type == EventType.POINTER_UP:
                self._captured_widget = None

            return

        # Hit test
        target = self.hit_test(x, y)

        # Update hover state
        if target != self._hovered_widget:
            if self._hovered_widget:
                self._hovered_widget.set_hovered(False)
            if target:
                target.set_hovered(True)
            self._hovered_widget = target

        if target is None:
            return

        # Create event in widget-local coordinates
        local_x, local_y = self._to_local(target, x, y)
        event = Event(
            type=event_type,
            x=local_x,
            y=local_y,
            button=button,
            delta_x=delta_x,
            delta_y=delta_y,
        )

        # Handle focus on click
        if event_type == EventType.POINTER_DOWN:
            self.set_focus(target)
            target.set_pressed(True)
            # Start capture for drag
            self._captured_widget = target

        elif event_type == EventType.POINTER_UP:
            if target:
                target.set_pressed(False)

        target.handle_event(event)

    def dispatch_scroll_event(self, x: float, y: float, delta_x: float, delta_y: float):
        """Dispatch scroll event."""
        target = self.hit_test(x, y)
        if target:
            local_x, local_y = self._to_local(target, x, y)
            event = Event(
                type=EventType.SCROLL,
                x=local_x,
                y=local_y,
                delta_x=delta_x,
                delta_y=delta_y,
            )
            target.handle_event(event)

    def dispatch_key_event(self, event_type: EventType, key: str, modifiers: int = 0):
        """Dispatch key event to focused widget."""
        if self._focused_widget:
            event = Event(type=event_type, key=key, modifiers=modifiers)
            self._focused_widget.handle_event(event)

    def set_focus(self, widget: Optional[Widget]):
        """Set keyboard focus to widget."""
        if self._focused_widget == widget:
            return

        if self._focused_widget:
            self._focused_widget.set_focused(False)

        self._focused_widget = widget

        if widget:
            widget.set_focused(True)

    def _to_local(self, widget: Widget, x: float, y: float) -> Tuple[float, float]:
        """Convert window coordinates to widget-local coordinates."""
        # Walk up the tree, accumulating offsets
        offsets_x, offsets_y = 0.0, 0.0
        w = widget
        while w is not None and w._layout is not None:
            offsets_x += w._layout.rect.x
            offsets_y += w._layout.rect.y
            w = w.parent

        return (x - offsets_x, y - offsets_y)

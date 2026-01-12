"""
Window Manager

Manages panel layout, docking, and focus.

Features:
- Dockable panels with splitters
- Floating panels
- Tab groups
- Focus management
- Layout serialization
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union, Any
from enum import Enum, auto
import json

from engine.ui.widget import Widget, RootWidget, Event, EventType
from engine.ui.layout import Rect, FlexLayout, LayoutDirection, Constraints
from engine.ui.style import Style, SizeValue, EdgeInsets, Border, Color
from engine.ui.widgets.panel import Panel, FloatingPanel


# =============================================================================
# Dock Types
# =============================================================================

class DockSide(Enum):
    """Side for docking."""
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    CENTER = auto()  # Tab


class SplitDirection(Enum):
    """Direction of a split."""
    HORIZONTAL = auto()  # Side by side
    VERTICAL = auto()    # Stacked


# =============================================================================
# Dock Node (Tree Structure)
# =============================================================================

@dataclass
class DockNode:
    """
    Node in the dock tree.

    Can be:
    - Leaf: Contains a panel
    - Split: Contains two children with a splitter
    - Tabs: Contains multiple panels in tabs
    """
    pass


@dataclass
class DockLeaf(DockNode):
    """Leaf node containing a single panel."""
    panel: Panel
    weight: float = 1.0  # Relative size weight


@dataclass
class DockSplit(DockNode):
    """Split node with two children."""
    direction: SplitDirection
    children: List[DockNode] = field(default_factory=list)
    split_ratio: float = 0.5  # 0-1, position of splitter

    def __post_init__(self):
        if not self.children:
            self.children = []


@dataclass
class DockTabs(DockNode):
    """Tab container with multiple panels."""
    panels: List[Panel] = field(default_factory=list)
    active_index: int = 0

    def __post_init__(self):
        if not self.panels:
            self.panels = []


# =============================================================================
# Splitter Widget
# =============================================================================

class Splitter(Widget):
    """
    Draggable splitter between dock areas.
    """

    def __init__(
        self,
        direction: SplitDirection,
        on_drag: callable = None,
        thickness: float = 6,
    ):
        self._direction = direction
        self._on_drag = on_drag
        self._thickness = thickness
        self._dragging = False

        # Style
        is_horizontal = direction == SplitDirection.HORIZONTAL
        style = Style(
            width=SizeValue.px(thickness) if is_horizontal else SizeValue.fill(),
            height=SizeValue.fill() if is_horizontal else SizeValue.px(thickness),
            background=(0.2, 0.2, 0.22, 1.0),
            cursor="ew-resize" if is_horizontal else "ns-resize",
        )

        super().__init__(style=style)

        self.on(EventType.POINTER_DOWN, self._on_pointer_down)
        self.on(EventType.POINTER_MOVE, self._on_pointer_move)
        self.on(EventType.POINTER_UP, self._on_pointer_up)
        self.on(EventType.POINTER_ENTER, self._on_enter)
        self.on(EventType.POINTER_LEAVE, self._on_leave)

    def _on_pointer_down(self, event: Event):
        self._dragging = True
        event.stop_propagation()

    def _on_pointer_move(self, event: Event):
        if self._dragging and self._on_drag:
            if self._direction == SplitDirection.HORIZONTAL:
                self._on_drag(event.delta_x)
            else:
                self._on_drag(event.delta_y)

    def _on_pointer_up(self, event: Event):
        self._dragging = False

    def _on_enter(self, event: Event):
        self.style = self.style.with_background((0.35, 0.55, 0.95, 1.0))

    def _on_leave(self, event: Event):
        self.style = self.style.with_background((0.2, 0.2, 0.22, 1.0))

    def draw(self, ctx):
        if not self._layout:
            return
        ctx.draw_rect(self._layout.rect, self.style.background)


# =============================================================================
# Dock Container Widget
# =============================================================================

class DockContainer(Widget):
    """
    Widget that renders a DockNode tree.

    Handles:
    - Recursive layout of dock tree
    - Splitter interaction
    - Tab bar for tabbed panels
    """

    def __init__(self, root: DockNode = None):
        self._dock_root = root

        style = Style(flex_grow=1.0)
        super().__init__(style=style)

        self._rebuild_widgets()

    @property
    def dock_root(self) -> Optional[DockNode]:
        return self._dock_root

    @dock_root.setter
    def dock_root(self, value: DockNode):
        self._dock_root = value
        self._rebuild_widgets()

    def _rebuild_widgets(self):
        """Rebuild widget tree from dock tree."""
        self.clear_children()

        if self._dock_root is None:
            return

        self._build_node(self._dock_root, self)

    def _build_node(self, node: DockNode, parent: Widget):
        """Recursively build widgets from dock node."""
        if isinstance(node, DockLeaf):
            parent.add_child(node.panel)

        elif isinstance(node, DockSplit):
            # Create container with appropriate flex direction
            is_horizontal = node.direction == SplitDirection.HORIZONTAL
            direction = LayoutDirection.ROW if is_horizontal else LayoutDirection.COLUMN

            container = Widget(
                style=Style(flex_grow=1.0),
                flex=FlexLayout(direction=direction, gap=0),
            )

            # Build children with weights
            for i, child in enumerate(node.children):
                # Wrapper with weight
                weight = 1.0
                if isinstance(child, DockLeaf):
                    weight = child.weight

                wrapper = Widget(
                    style=Style(flex_grow=weight),
                    flex=FlexLayout(direction=LayoutDirection.COLUMN),
                )
                self._build_node(child, wrapper)
                container.add_child(wrapper)

                # Add splitter between children (not after last)
                if i < len(node.children) - 1:
                    splitter = Splitter(
                        node.direction,
                        on_drag=lambda d, n=node, idx=i: self._handle_splitter_drag(n, idx, d),
                    )
                    container.add_child(splitter)

            parent.add_child(container)

        elif isinstance(node, DockTabs):
            # Create tab container
            tab_container = TabContainer(node.panels, node.active_index)
            parent.add_child(tab_container)

    def _handle_splitter_drag(self, split: DockSplit, index: int, delta: float):
        """Handle splitter drag to adjust split ratio."""
        # Calculate new ratio based on delta
        # This is simplified - real impl would track actual sizes
        sensitivity = 0.005
        split.split_ratio = max(0.1, min(0.9, split.split_ratio + delta * sensitivity))
        self._rebuild_widgets()


# =============================================================================
# Tab Container
# =============================================================================

class TabBar(Widget):
    """Tab bar for selecting panels."""

    def __init__(
        self,
        tabs: List[str],
        active: int = 0,
        on_select: callable = None,
    ):
        self._tabs = tabs
        self._active = active
        self._on_select = on_select

        style = Style(
            background=(0.16, 0.16, 0.18, 1.0),
            height=SizeValue.px(28),
        )
        flex = FlexLayout(direction=LayoutDirection.ROW, gap=0)

        super().__init__(style=style, flex=flex)

        self._rebuild_tabs()

    def _rebuild_tabs(self):
        self.clear_children()

        for i, title in enumerate(self._tabs):
            is_active = i == self._active
            tab = self._create_tab(title, i, is_active)
            self.add_child(tab)

    def _create_tab(self, title: str, index: int, active: bool) -> Widget:
        """Create a tab button."""
        bg = (0.22, 0.22, 0.25, 1.0) if active else (0.16, 0.16, 0.18, 1.0)

        tab = Widget(
            style=Style(
                background=bg,
                padding=EdgeInsets.symmetric(4, 12),
                font_color=(1.0, 1.0, 1.0, 1.0) if active else (0.7, 0.7, 0.7, 1.0),
                font_size=12,
                cursor="pointer",
            ),
        )

        # Store title for drawing
        tab.data["title"] = title
        tab.data["index"] = index

        # Click handler
        def on_click(event, idx=index):
            self._active = idx
            if self._on_select:
                self._on_select(idx)
            self._rebuild_tabs()

        tab.on(EventType.POINTER_UP, on_click)

        return tab

    def draw(self, ctx):
        if not self._layout:
            return

        # Draw background
        ctx.draw_rect(self._layout.rect, self.style.background)

        # Draw tabs
        for child in self._children:
            if child._layout:
                rect = child._layout.rect
                ctx.draw_rect(rect, child.style.background)

                # Draw tab title
                title = child.data.get("title", "")
                ctx.draw_text_in_rect(
                    title,
                    rect,
                    child.style.font_color,
                    font_size=child.style.font_size,
                    align="center",
                    valign="center",
                )


class TabContainer(Widget):
    """Container with tab bar and content."""

    def __init__(self, panels: List[Panel], active: int = 0):
        self._panels = panels
        self._active = active

        style = Style(flex_grow=1.0)
        flex = FlexLayout(direction=LayoutDirection.COLUMN, gap=0)

        super().__init__(style=style, flex=flex)

        # Tab bar
        titles = [p.title for p in panels]
        self._tab_bar = TabBar(titles, active, on_select=self._on_tab_select)
        self.add_child(self._tab_bar)

        # Content area
        self._content = Widget(
            style=Style(flex_grow=1.0),
            flex=FlexLayout(direction=LayoutDirection.COLUMN),
        )
        self.add_child(self._content)

        self._update_content()

    def _on_tab_select(self, index: int):
        self._active = index
        self._update_content()

    def _update_content(self):
        self._content.clear_children()
        if 0 <= self._active < len(self._panels):
            self._content.add_child(self._panels[self._active])


# =============================================================================
# Window Manager
# =============================================================================

class WindowManager:
    """
    Manages the entire UI layout.

    Features:
    - Main dock area
    - Floating panels
    - Focus tracking
    - Layout save/restore
    """

    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height

        # Root widget
        self._root = RootWidget(width, height)

        # Main dock container
        self._dock_container = DockContainer()
        self._root.add_child(self._dock_container)

        # Floating panels (rendered on top)
        self._floating: List[FloatingPanel] = []

        # Focus
        self._focused_panel: Optional[Panel] = None

    @property
    def root(self) -> RootWidget:
        return self._root

    def set_size(self, width: int, height: int):
        """Update window size."""
        self._width = width
        self._height = height
        self._root.set_window_size(width, height)

    def do_layout(self):
        """Perform full layout pass."""
        self._root.do_layout()

    # -------------------------------------------------------------------------
    # Dock Management
    # -------------------------------------------------------------------------

    def set_dock_root(self, node: DockNode):
        """Set the root dock node."""
        self._dock_container.dock_root = node

    def dock(self, panel: Panel, target: Panel = None, side: DockSide = DockSide.CENTER):
        """
        Dock a panel.

        If target is None, docks to the root.
        """
        # Simplified implementation - just set as root for now
        # Full implementation would find target and create appropriate split
        leaf = DockLeaf(panel=panel)
        self._dock_container.dock_root = leaf

    def undock(self, panel: Panel) -> FloatingPanel:
        """Undock a panel and make it floating."""
        floating = FloatingPanel(
            title=panel.title,
            x=100,
            y=100,
        )
        # Copy children
        for child in panel.content.children:
            floating.add_content(child)

        self._floating.append(floating)
        return floating

    # -------------------------------------------------------------------------
    # Floating Panels
    # -------------------------------------------------------------------------

    def add_floating(self, panel: FloatingPanel):
        """Add a floating panel."""
        self._floating.append(panel)

    def remove_floating(self, panel: FloatingPanel):
        """Remove a floating panel."""
        if panel in self._floating:
            self._floating.remove(panel)

    def bring_to_front(self, panel: FloatingPanel):
        """Bring a floating panel to the front."""
        if panel in self._floating:
            self._floating.remove(panel)
            self._floating.append(panel)

    # -------------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------------

    def handle_pointer_event(
        self,
        event_type: EventType,
        x: float,
        y: float,
        button: int = 0,
        dx: float = 0,
        dy: float = 0,
    ):
        """Route pointer event through the UI."""
        # Check floating panels first (in reverse for z-order)
        for panel in reversed(self._floating):
            if panel.rect and panel.rect.contains(x, y):
                local_x = x - panel.rect.x
                local_y = y - panel.rect.y
                event = Event(
                    type=event_type,
                    x=local_x,
                    y=local_y,
                    button=button,
                    delta_x=dx,
                    delta_y=dy,
                )
                panel.handle_event(event)
                return

        # Otherwise, route to root
        self._root.dispatch_pointer_event(event_type, x, y, button, dx, dy)

    def handle_scroll_event(self, x: float, y: float, dx: float, dy: float):
        """Handle scroll event."""
        self._root.dispatch_scroll_event(x, y, dx, dy)

    def handle_key_event(self, event_type: EventType, key: str, modifiers: int = 0):
        """Handle key event."""
        self._root.dispatch_key_event(event_type, key, modifiers)

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def draw(self, ctx):
        """Draw the entire UI."""
        # Draw docked content
        self._root.draw(ctx)

        # Draw floating panels on top
        for panel in self._floating:
            panel.layout(Rect(0, 0, self._width, self._height))
            panel.draw(ctx)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def save_layout(self) -> dict:
        """Save current layout to dict."""
        return {
            "dock": self._serialize_node(self._dock_container.dock_root),
            "floating": [
                {
                    "title": p.title,
                    "x": p._pos_x,
                    "y": p._pos_y,
                }
                for p in self._floating
            ],
        }

    def _serialize_node(self, node: DockNode) -> Optional[dict]:
        """Serialize a dock node."""
        if node is None:
            return None

        if isinstance(node, DockLeaf):
            return {
                "type": "leaf",
                "panel_id": node.panel.title,  # Use title as ID
                "weight": node.weight,
            }
        elif isinstance(node, DockSplit):
            return {
                "type": "split",
                "direction": node.direction.name,
                "ratio": node.split_ratio,
                "children": [self._serialize_node(c) for c in node.children],
            }
        elif isinstance(node, DockTabs):
            return {
                "type": "tabs",
                "panels": [p.title for p in node.panels],
                "active": node.active_index,
            }

        return None

    def load_layout(self, data: dict, panel_factory: callable):
        """
        Load layout from dict.

        Args:
            data: Layout dict from save_layout()
            panel_factory: Function(panel_id) -> Panel to create panels
        """
        if "dock" in data and data["dock"]:
            self._dock_container.dock_root = self._deserialize_node(
                data["dock"], panel_factory
            )

        # TODO: Restore floating panels

    def _deserialize_node(self, data: dict, panel_factory: callable) -> Optional[DockNode]:
        """Deserialize a dock node."""
        if data is None:
            return None

        node_type = data.get("type")

        if node_type == "leaf":
            panel = panel_factory(data.get("panel_id", ""))
            return DockLeaf(panel=panel, weight=data.get("weight", 1.0))

        elif node_type == "split":
            direction = SplitDirection[data.get("direction", "HORIZONTAL")]
            children = [
                self._deserialize_node(c, panel_factory)
                for c in data.get("children", [])
            ]
            return DockSplit(
                direction=direction,
                children=[c for c in children if c],
                split_ratio=data.get("ratio", 0.5),
            )

        elif node_type == "tabs":
            panels = [panel_factory(pid) for pid in data.get("panels", [])]
            return DockTabs(
                panels=panels,
                active_index=data.get("active", 0),
            )

        return None

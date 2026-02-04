"""
UI System

Minimal CSS-like styling with flexbox layout, rendered in GL.

Components:
- style: Flat style dataclass (no cascading, no selectors)
- layout: Flexbox-inspired layout engine
- widget: Base widget class and tree
- draw: Batched rendering via CommandList
- widgets/: Concrete widget implementations
- window_manager: Docking, floating, focus management

Example usage:

    from engine.ui import WindowManager, Panel, Label, Button, Viewport3D
    from engine.ui import DockSplit, DockLeaf, SplitDirection

    # Create window manager
    wm = WindowManager(1280, 720)

    # Create panels
    viewport_panel = Panel(title="3D View", children=[
        Viewport3D(area=my_viewport_area)
    ])
    props_panel = Panel(title="Properties", children=[
        Label(text="Object: Cube"),
        Button(text="Apply", on_click=handle_apply),
    ])

    # Set up docking layout
    wm.set_dock_root(DockSplit(
        direction=SplitDirection.HORIZONTAL,
        split_ratio=0.75,
        children=[
            DockLeaf(panel=viewport_panel),
            DockLeaf(panel=props_panel),
        ]
    ))

    # In render loop:
    wm.set_size(window_width, window_height)
    wm.do_layout()
    ctx = DrawContext(window_width, window_height)
    wm.draw(ctx)
    ui_renderer.render(ctx.finalize(), window_width, window_height)
"""

from engine.ui.style import (
    Style, Color, EdgeInsets, Border, Shadow, SizeValue,
    Theme, DEFAULT_THEME, hex_to_color,
)
from engine.ui.layout import (
    FlexLayout, LayoutDirection, Align, Justify,
    Rect, Constraints,
)
from engine.ui.widget import (
    Widget, RootWidget, LayoutResult,
    Event, EventType, EventHandler, WidgetState,
)
from engine.ui.draw import DrawContext, DrawBatch, UIRenderer
from engine.ui.renderer import SimpleUIRenderer
from engine.ui.widgets import (
    Container, Spacer, Divider,
    Panel, TitleBar, FloatingPanel,
    Label, Heading, Caption,
    Button, IconButton, ToggleButton,
    Viewport3D,
    # Sequencer widgets
    TransportBarWidget,
    SequencerGridWidget,
    LaunchpadGridWidget,
    WaveformWidget,
    SequencerEvent,
    SequencerLane,
    CellListener,
    SequencerController,
)
from engine.ui.widgets.container import Row, Column
from engine.ui.window_manager import (
    WindowManager,
    DockNode, DockLeaf, DockSplit, DockTabs,
    DockSide, SplitDirection,
)

__all__ = [
    # Style
    "Style", "Color", "EdgeInsets", "Border", "Shadow", "SizeValue",
    "Theme", "DEFAULT_THEME", "hex_to_color",
    # Layout
    "FlexLayout", "LayoutDirection", "Align", "Justify",
    "Rect", "Constraints",
    # Widget
    "Widget", "RootWidget", "LayoutResult",
    "Event", "EventType", "EventHandler", "WidgetState",
    # Draw
    "DrawContext", "DrawBatch", "UIRenderer", "SimpleUIRenderer",
    # Core Widgets
    "Container", "Row", "Column", "Spacer", "Divider",
    "Panel", "TitleBar", "FloatingPanel",
    "Label", "Heading", "Caption",
    "Button", "IconButton", "ToggleButton",
    "Viewport3D",
    # Sequencer Widgets
    "TransportBarWidget",
    "SequencerGridWidget",
    "LaunchpadGridWidget",
    "WaveformWidget",
    "SequencerEvent",
    "SequencerLane",
    "CellListener",
    "SequencerController",
    # Window Manager
    "WindowManager",
    "DockNode", "DockLeaf", "DockSplit", "DockTabs",
    "DockSide", "SplitDirection",
]

"""
Built-in Widgets

Core widget library:
- Container: Generic flex container (Row, Column)
- Panel: Titled container with chrome
- Label: Text display (Heading, Caption)
- Button: Clickable button (IconButton, ToggleButton)
- Viewport3D: 3D viewport display
- Spacer: Flexible spacing
- Divider: Visual separator

Sequencer widgets:
- TransportBarWidget: Play/pause, beat indicators, BPM
- SequencerGridWidget: Timeline grid with lanes
- LaunchpadGridWidget: 8x8 pad grid
- WaveformWidget: Audio waveform display
"""

from engine.ui.widgets.container import Container, Row, Column, Spacer, Divider
from engine.ui.widgets.panel import Panel, TitleBar, FloatingPanel
from engine.ui.widgets.label import Label, Heading, Caption
from engine.ui.widgets.button import Button, IconButton, ToggleButton
from engine.ui.widgets.viewport import Viewport3D
from engine.ui.widgets.sequencer import (
    TransportBarWidget,
    SequencerGridWidget,
    LaunchpadGridWidget,
    WaveformWidget,
    SequencerEvent,
    SequencerLane,
    CellListener,
    SequencerController,
)

__all__ = [
    # Core widgets
    "Container",
    "Row",
    "Column",
    "Spacer",
    "Divider",
    "Panel",
    "TitleBar",
    "FloatingPanel",
    "Label",
    "Heading",
    "Caption",
    "Button",
    "IconButton",
    "ToggleButton",
    "Viewport3D",
    # Sequencer widgets
    "TransportBarWidget",
    "SequencerGridWidget",
    "LaunchpadGridWidget",
    "WaveformWidget",
    "SequencerEvent",
    "SequencerLane",
    "CellListener",
    "SequencerController",
]

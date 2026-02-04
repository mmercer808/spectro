"""
Sequencer Widgets

MIDI-aware UI components for the SPECTRO sequencer:
- TransportBarWidget: Play/pause, beat indicators, BPM
- SequencerGridWidget: Timeline grid with lanes, synced via TimeCamera
- LaunchpadGridWidget: 8x8 pad grid mirroring hardware
- WaveformWidget: Audio waveform visualization

All widgets integrate with:
- engine/core/signal.py SignalBridge for events
- engine/time/transport.py Transport for playback state
- engine/time/camera.py TimeCamera for beat↔pixel sync
- engine/midi/ for MIDI input/output

Callback patterns:
- Widget callbacks use Optional[Callable] pattern
- Callbacks receive relevant state and UI Event where applicable
- Return values can signal event consumption (True = consumed)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple, Set, TYPE_CHECKING
from enum import Enum, auto
import numpy as np

from engine.ui.widget import Widget, Event, EventType, EventHandler
from engine.ui.style import Style, Color, Border, EdgeInsets, SizeValue
from engine.ui.layout import FlexLayout, LayoutDirection, Rect, Constraints

if TYPE_CHECKING:
    from engine.ui.draw import DrawContext
    from engine.time.transport import Transport, TransportState
    from engine.time.camera import TimeCamera
    from engine.core.signal import SignalBridge


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class SequencerEvent:
    """An event on the sequencer timeline."""
    id: int
    beat: float
    lane: int
    duration: float
    velocity: int
    sample_name: str
    color: Tuple[float, float, float, float] = (1.0, 0.5, 0.2, 1.0)
    fired: bool = False

    @property
    def end_beat(self) -> float:
        return self.beat + self.duration


@dataclass
class SequencerLane:
    """A lane (row) in the sequencer grid."""
    index: int
    name: str
    sample_name: str
    color: Tuple[float, float, float, float]
    events: List[SequencerEvent] = field(default_factory=list)


# =============================================================================
# TRANSPORT BAR WIDGET
# =============================================================================

class TransportBarWidget(Widget):
    """
    Transport controls widget.

    Displays:
    - Play/Pause/Stop buttons
    - Beat indicators (boxes for time signature)
    - BPM display (clickable to edit)
    - Beat counter (bar:beat)
    - Loop toggle

    Integrates with:
    - Transport for playback state
    - SignalBridge for SIGNAL_TRANSPORT_CHANGED, SIGNAL_PLAY, SIGNAL_PAUSE

    Callbacks:
    - on_play: () -> None
        Called when play button clicked while paused.

    - on_pause: () -> None
        Called when pause button clicked while playing.

    - on_stop: () -> None
        Called when stop button clicked (pause + seek to 0).

    - on_seek: (beat: float) -> None
        Called when seeking to a specific beat.

    - on_bpm_change: (bpm: float) -> None
        Called when BPM value is changed.

    - on_beat: (beat: int, bar: int, phase: float) -> None
        Called on each beat crossing during playback.
        phase is 0-1 position within beat.

    - on_bar: (bar: int) -> None
        Called on each bar crossing (beat 1 of bar).

    - on_loop_toggle: (enabled: bool, start: float, end: float) -> None
        Called when loop mode is toggled or range changes.
    """

    def __init__(
        self,
        transport: 'Transport' = None,
        signals: 'SignalBridge' = None,
        on_play: Callable[[], None] = None,
        on_pause: Callable[[], None] = None,
        on_stop: Callable[[], None] = None,
        on_seek: Callable[[float], None] = None,
        on_bpm_change: Callable[[float], None] = None,
        on_beat: Callable[[int, int, float], None] = None,
        on_bar: Callable[[int], None] = None,
        on_loop_toggle: Callable[[bool, float, float], None] = None,
        style: Style = None,
        height: float = 56,
    ):
        self._transport = transport
        self._signals = signals
        self._on_play = on_play
        self._on_pause = on_pause
        self._on_stop = on_stop
        self._on_seek = on_seek
        self._on_bpm_change = on_bpm_change
        self._on_beat = on_beat
        self._on_bar = on_bar
        self._on_loop_toggle = on_loop_toggle

        # Cached state for rendering
        self._playing = False
        self._playhead_beat = 0.0
        self._bpm = 120.0
        self._beats_per_bar = 4
        self._loop_enabled = False
        self._loop_start = 0.0
        self._loop_end = 4.0

        # Track last beat/bar for callbacks
        self._last_beat = -1
        self._last_bar = -1

        if style is None:
            style = Style(
                background=(0.12, 0.13, 0.15, 1.0),
                height=SizeValue.px(height),
                padding=EdgeInsets.symmetric(12, 16),
            )

        super().__init__(style=style)

        # Register click handler for play button area
        self.on(EventType.POINTER_DOWN, self._on_pointer_down)

        # Connect to transport signals if provided
        if signals:
            from engine.core.signal import SIGNAL_TRANSPORT_CHANGED
            signals.connect(SIGNAL_TRANSPORT_CHANGED, self._on_transport_changed)

    def bind_transport(self, transport: 'Transport'):
        """Bind to a Transport instance for state updates."""
        self._transport = transport
        self._sync_from_transport()

    def _sync_from_transport(self):
        """Sync cached state from transport."""
        if self._transport:
            self._playing = self._transport.playing
            self._playhead_beat = self._transport.playhead_beat
            self._bpm = self._transport.bpm
            if self._transport.time_sig:
                self._beats_per_bar = self._transport.time_sig.beats_per_bar

    def _on_transport_changed(self, state: 'TransportState'):
        """Signal callback when transport state changes."""
        self._playing = state.playing
        self._playhead_beat = state.playhead_beat
        self._bpm = state.bpm
        self._beats_per_bar = state.time_sig.beats_per_bar if state.time_sig else 4

    def _on_pointer_down(self, event: Event):
        """Handle click on transport bar."""
        if not self._layout:
            return

        # Button layout: [Play 48px] [Stop 48px] [gap] [beat indicators] [BPM] [Loop toggle]
        play_btn_end = 48
        stop_btn_end = 96
        beat_start = 112
        beat_end = beat_start + self._beats_per_bar * 38 + 10
        bpm_start = beat_end + 20
        bpm_end = bpm_start + 70
        loop_start = bpm_end + 90  # After counter

        # Play button
        if event.x < play_btn_end:
            if self._playing:
                if self._on_pause:
                    self._on_pause()
                elif self._transport:
                    self._transport.pause()
            else:
                if self._on_play:
                    self._on_play()
                elif self._transport:
                    self._transport.play()
            event.stop_propagation()
            return

        # Stop button
        if event.x < stop_btn_end:
            if self._on_stop:
                self._on_stop()
            elif self._transport:
                self._transport.pause()
                self._transport.seek(0)
            event.stop_propagation()
            return

        # BPM area (future: open BPM editor)
        if bpm_start <= event.x < bpm_end:
            # For now, just cycle through common BPM values
            common_bpms = [80, 90, 100, 110, 120, 130, 140, 150, 160]
            try:
                idx = common_bpms.index(int(self._bpm))
                new_bpm = common_bpms[(idx + 1) % len(common_bpms)]
            except ValueError:
                new_bpm = 120
            if self._on_bpm_change:
                self._on_bpm_change(float(new_bpm))
            elif self._transport:
                self._transport.set_bpm(new_bpm)
            event.stop_propagation()
            return

        # Loop toggle area
        if event.x >= loop_start:
            self._loop_enabled = not self._loop_enabled
            if self._on_loop_toggle:
                self._on_loop_toggle(self._loop_enabled, self._loop_start, self._loop_end)
            event.stop_propagation()
            return

    def update(self):
        """Call each frame to sync state and fire beat/bar callbacks."""
        self._sync_from_transport()

        # Fire beat/bar callbacks
        if self._playing:
            current_beat = int(self._playhead_beat)
            current_bar = current_beat // self._beats_per_bar

            # Beat callback
            if current_beat != self._last_beat:
                self._last_beat = current_beat
                phase = self._playhead_beat - current_beat
                if self._on_beat:
                    self._on_beat(current_beat % self._beats_per_bar, current_bar, phase)

            # Bar callback
            if current_bar != self._last_bar:
                self._last_bar = current_bar
                if self._on_bar:
                    self._on_bar(current_bar)

    def set_loop(self, enabled: bool, start: float = None, end: float = None):
        """Set loop state programmatically."""
        self._loop_enabled = enabled
        if start is not None:
            self._loop_start = start
        if end is not None:
            self._loop_end = end
        if self._on_loop_toggle:
            self._on_loop_toggle(self._loop_enabled, self._loop_start, self._loop_end)

    def measure(self, constraints: Constraints) -> Tuple[float, float]:
        """Transport bar fills width, fixed height."""
        h = self.style.height.resolve(constraints.max_h, 56)
        return constraints.constrain(constraints.max_w, h)

    def draw(self, ctx: 'DrawContext'):
        if not self._layout:
            return

        rect = self._layout.rect
        style = self.style

        ctx.push_offset(rect.x, rect.y)
        try:
            local = Rect(0, 0, rect.w, rect.h)

            # Background
            if style.background:
                ctx.draw_rect(local, style.background)

            btn_size = 32
            btn_y = (rect.h - btn_size) / 2

            # Play/Pause button
            btn_x = 12
            play_color = (0.3, 0.85, 0.4, 1.0) if self._playing else (0.5, 0.5, 0.55, 1.0)
            ctx.draw_rect(Rect(btn_x, btn_y, btn_size, btn_size), play_color, radius=4)
            # Draw play/pause icon indication
            if self._playing:
                # Pause bars
                ctx.draw_rect(Rect(btn_x + 10, btn_y + 8, 4, 16), (1.0, 1.0, 1.0, 0.9))
                ctx.draw_rect(Rect(btn_x + 18, btn_y + 8, 4, 16), (1.0, 1.0, 1.0, 0.9))
            else:
                # Play triangle (simplified as rectangle for now)
                ctx.draw_rect(Rect(btn_x + 11, btn_y + 8, 12, 16), (1.0, 1.0, 1.0, 0.9))

            # Stop button
            stop_x = 52
            ctx.draw_rect(Rect(stop_x, btn_y, btn_size, btn_size), (0.35, 0.35, 0.4, 1.0), radius=4)
            ctx.draw_rect(Rect(stop_x + 8, btn_y + 8, 16, 16), (0.9, 0.9, 0.9, 0.9))

            # Beat indicators (boxes for time signature)
            beat_in_bar = int(self._playhead_beat) % self._beats_per_bar
            beat_start_x = 100
            for i in range(self._beats_per_bar):
                bx = beat_start_x + i * 38
                active = i == beat_in_bar and self._playing
                color = (0.45, 0.65, 0.95, 1.0) if active else (0.22, 0.24, 0.28, 1.0)
                ctx.draw_rect(Rect(bx, btn_y, btn_size, btn_size), color, radius=4)
                # Beat number
                ctx.draw_text(str(i + 1), bx + 12, btn_y + 8, (0.7, 0.7, 0.7, 1.0), font_size=12)

            # BPM display
            bpm_x = beat_start_x + self._beats_per_bar * 38 + 20
            ctx.draw_rect(Rect(bpm_x, btn_y + 2, 70, btn_size - 4), (0.18, 0.19, 0.22, 1.0), radius=3)
            ctx.draw_text(f"{self._bpm:.0f} BPM", bpm_x + 8, btn_y + 8, (0.8, 0.8, 0.8, 1.0), font_size=12)

            # Beat counter (bar:beat)
            counter_x = bpm_x + 80
            current_bar = int(self._playhead_beat) // self._beats_per_bar + 1
            current_beat = int(self._playhead_beat) % self._beats_per_bar + 1
            ctx.draw_rect(Rect(counter_x, btn_y + 2, 70, btn_size - 4), (0.15, 0.16, 0.19, 1.0), radius=3)
            ctx.draw_text(f"{current_bar}:{current_beat}", counter_x + 8, btn_y + 8, (0.9, 0.9, 0.9, 1.0), font_size=14)

            # Loop toggle
            loop_x = counter_x + 80
            loop_color = (0.4, 0.7, 0.4, 1.0) if self._loop_enabled else (0.25, 0.25, 0.28, 1.0)
            ctx.draw_rect(Rect(loop_x, btn_y + 2, 50, btn_size - 4), loop_color, radius=3)
            ctx.draw_text("LOOP", loop_x + 8, btn_y + 8, (0.9, 0.9, 0.9, 1.0), font_size=11)

        finally:
            ctx.pop_offset()


# =============================================================================
# SEQUENCER GRID WIDGET
# =============================================================================

class SequencerGridWidget(Widget):
    """
    Timeline grid with lanes for sequencer events.

    Features:
    - Beat grid lines synced via TimeCamera
    - Multiple lanes (rows) for different instruments
    - Events displayed as colored rectangles
    - Playhead line
    - Click to add/remove events
    - Scroll to zoom, drag to pan
    - Event drag/resize

    Integrates with:
    - TimeCamera for beat↔pixel conversion (SHARED instance for sync)
    - Transport for playhead position
    - SignalBridge for SIGNAL_MIDI_PAD, SIGNAL_TRANSPORT_CHANGED

    Callbacks:
    - on_cell_click: (lane: int, beat: float, event: Event) -> None
        Called when clicking on empty grid cell.

    - on_event_click: (event: SequencerEvent, ui_event: Event) -> None
        Called when clicking on an existing event.

    - on_event_drag: (event: SequencerEvent, new_beat: float, new_lane: int) -> None
        Called during event drag operation with new position.

    - on_event_resize: (event: SequencerEvent, new_duration: float) -> None
        Called during event resize operation.

    - on_selection_change: (selected: Set[int]) -> None
        Called when selection changes. Receives set of event IDs.

    - on_lane_click: (lane: int, event: Event) -> None
        Called when clicking on lane label area.

    - on_zoom: (px_per_beat: float, center_beat: float) -> None
        Called when zoom level changes.

    - on_scroll: (left_beat: float, event: Event) -> None
        Called when scrolling/panning horizontally.

    - on_playhead_click: (beat: float, event: Event) -> None
        Called when clicking in playhead/ruler area.

    - on_event_double_click: (event: SequencerEvent, ui_event: Event) -> None
        Called on double-click on an event (for editing).
    """

    # Layout constants
    LANE_HEIGHT = 48
    LANE_GAP = 4
    LABEL_WIDTH = 80
    RULER_HEIGHT = 24

    def __init__(
        self,
        time_camera: 'TimeCamera' = None,
        transport: 'Transport' = None,
        signals: 'SignalBridge' = None,
        lanes: List[SequencerLane] = None,
        on_cell_click: Callable[[int, float, Event], None] = None,
        on_event_click: Callable[[SequencerEvent, Event], None] = None,
        on_event_drag: Callable[[SequencerEvent, float, int], None] = None,
        on_event_resize: Callable[[SequencerEvent, float], None] = None,
        on_selection_change: Callable[[Set[int]], None] = None,
        on_lane_click: Callable[[int, Event], None] = None,
        on_zoom: Callable[[float, float], None] = None,
        on_scroll: Callable[[float, Event], None] = None,
        on_playhead_click: Callable[[float, Event], None] = None,
        on_event_double_click: Callable[[SequencerEvent, Event], None] = None,
        style: Style = None,
    ):
        self._time_camera = time_camera
        self._transport = transport
        self._signals = signals
        self._lanes = lanes or []
        self._on_cell_click = on_cell_click
        self._on_event_click = on_event_click
        self._on_event_drag = on_event_drag
        self._on_event_resize = on_event_resize
        self._on_selection_change = on_selection_change
        self._on_lane_click = on_lane_click
        self._on_zoom = on_zoom
        self._on_scroll = on_scroll
        self._on_playhead_click = on_playhead_click
        self._on_event_double_click = on_event_double_click

        # Cached state
        self._playhead_beat = 0.0
        self._playing = False

        # Selection state
        self._selected_events: Set[int] = set()

        # Drag state
        self._drag_event: Optional[SequencerEvent] = None
        self._drag_start_beat: float = 0.0
        self._drag_start_lane: int = 0
        self._drag_mode: Optional[str] = None  # 'move', 'resize_end'

        # Double-click detection
        self._last_click_time: float = 0.0
        self._last_click_pos: Tuple[float, float] = (0, 0)
        self._double_click_threshold: float = 0.3  # seconds

        if style is None:
            style = Style(
                background=(0.06, 0.065, 0.075, 1.0),
                flex_grow=1.0,
            )

        super().__init__(style=style)

        # Event handlers
        self.on(EventType.POINTER_DOWN, self._on_pointer_down)
        self.on(EventType.POINTER_MOVE, self._on_pointer_move)
        self.on(EventType.POINTER_UP, self._on_pointer_up)
        self.on(EventType.SCROLL, self._on_scroll_event)

        # Connect signals
        if signals:
            from engine.core.signal import SIGNAL_TRANSPORT_CHANGED
            signals.connect(SIGNAL_TRANSPORT_CHANGED, self._on_transport_changed)

    def bind_time_camera(self, camera: 'TimeCamera'):
        """Bind to a TimeCamera for beat↔pixel sync."""
        self._time_camera = camera

    def bind_transport(self, transport: 'Transport'):
        """Bind to Transport for playhead."""
        self._transport = transport

    def set_lanes(self, lanes: List[SequencerLane]):
        """Set the lanes to display."""
        self._lanes = lanes

    def add_lane(self, lane: SequencerLane):
        """Add a lane."""
        self._lanes.append(lane)

    def _on_transport_changed(self, state: 'TransportState'):
        """Update cached transport state."""
        self._playhead_beat = state.playhead_beat
        self._playing = state.playing

    def _on_pointer_down(self, event: Event):
        """Handle click on grid."""
        import time
        current_time = time.time()

        if not self._time_camera:
            return

        grid_x = self.LABEL_WIDTH
        local_x = event.x - grid_x
        local_y = event.y

        # Click in label area
        if local_x < 0:
            lane_idx = self._y_to_lane(local_y)
            if lane_idx is not None and lane_idx < len(self._lanes):
                if self._on_lane_click:
                    self._on_lane_click(lane_idx, event)
            event.stop_propagation()
            return

        # Click in ruler area (top RULER_HEIGHT pixels)
        if local_y < self.RULER_HEIGHT:
            beat = self._time_camera.px_to_beat(local_x)
            if self._on_playhead_click:
                self._on_playhead_click(beat, event)
            event.stop_propagation()
            return

        # Find lane
        lane_idx = self._y_to_lane(local_y - self.RULER_HEIGHT)
        if lane_idx is None or lane_idx >= len(self._lanes):
            return

        # Convert to beat
        beat = self._time_camera.px_to_beat(local_x)

        # Check if clicking on existing event
        lane = self._lanes[lane_idx]
        for seq_event in lane.events:
            if seq_event.beat <= beat < seq_event.end_beat:
                # Check for double-click
                is_double = (
                    current_time - self._last_click_time < self._double_click_threshold and
                    abs(event.x - self._last_click_pos[0]) < 5 and
                    abs(event.y - self._last_click_pos[1]) < 5
                )

                if is_double and self._on_event_double_click:
                    self._on_event_double_click(seq_event, event)
                else:
                    # Check if clicking near end for resize
                    event_end_px = self._time_camera.beat_to_px(seq_event.end_beat) + grid_x
                    if abs(event.x - event_end_px) < 8:
                        self._drag_mode = 'resize_end'
                    else:
                        self._drag_mode = 'move'

                    self._drag_event = seq_event
                    self._drag_start_beat = beat
                    self._drag_start_lane = lane_idx

                    # Update selection
                    if seq_event.id not in self._selected_events:
                        self._selected_events = {seq_event.id}
                        if self._on_selection_change:
                            self._on_selection_change(self._selected_events)

                    if self._on_event_click:
                        self._on_event_click(seq_event, event)

                self._last_click_time = current_time
                self._last_click_pos = (event.x, event.y)
                event.stop_propagation()
                return

        # Click on empty cell - clear selection
        if self._selected_events:
            self._selected_events = set()
            if self._on_selection_change:
                self._on_selection_change(self._selected_events)

        if self._on_cell_click:
            self._on_cell_click(lane_idx, beat, event)

        self._last_click_time = current_time
        self._last_click_pos = (event.x, event.y)
        event.stop_propagation()

    def _on_pointer_move(self, event: Event):
        """Handle drag operations."""
        if not self._drag_event or not self._time_camera:
            return

        grid_x = self.LABEL_WIDTH
        local_x = event.x - grid_x
        beat = self._time_camera.px_to_beat(local_x)
        lane_idx = self._y_to_lane(event.y - self.RULER_HEIGHT)

        if self._drag_mode == 'move':
            if self._on_event_drag and lane_idx is not None:
                self._on_event_drag(self._drag_event, beat, lane_idx)
        elif self._drag_mode == 'resize_end':
            new_duration = max(0.25, beat - self._drag_event.beat)
            if self._on_event_resize:
                self._on_event_resize(self._drag_event, new_duration)

        event.stop_propagation()

    def _on_pointer_up(self, event: Event):
        """End drag operation."""
        self._drag_event = None
        self._drag_mode = None

    def _on_scroll_event(self, event: Event):
        """Handle scroll for zoom/pan."""
        if not self._time_camera:
            return

        if event.delta_y != 0:
            # Zoom centered on mouse position
            center_px = event.x - self.LABEL_WIDTH
            center_beat = self._time_camera.px_to_beat(center_px)
            self._time_camera.zoom(event.delta_y, center_px)

            if self._on_zoom:
                self._on_zoom(self._time_camera._px_per_beat, center_beat)
            event.stop_propagation()

        if hasattr(event, 'delta_x') and event.delta_x != 0:
            # Horizontal scroll for panning
            self._time_camera.pan(event.delta_x)
            if self._on_scroll:
                self._on_scroll(self._time_camera._left_beat, event)
            event.stop_propagation()

    def select_events(self, event_ids: Set[int]):
        """Programmatically set selection."""
        if event_ids != self._selected_events:
            self._selected_events = event_ids
            if self._on_selection_change:
                self._on_selection_change(self._selected_events)

    def clear_selection(self):
        """Clear all selected events."""
        if self._selected_events:
            self._selected_events = set()
            if self._on_selection_change:
                self._on_selection_change(self._selected_events)

    def _y_to_lane(self, y: float) -> Optional[int]:
        """Convert Y coordinate to lane index (y should be relative to lanes area, not ruler)."""
        if not self._layout:
            return None

        num_lanes = len(self._lanes)
        if num_lanes == 0:
            return None

        # Lanes start after ruler
        lanes_area_h = self._layout.rect.h - self.RULER_HEIGHT
        total_lane_h = num_lanes * (self.LANE_HEIGHT + self.LANE_GAP)
        start_y = max(0, (lanes_area_h - total_lane_h) / 2)

        rel_y = y - start_y
        if rel_y < 0:
            return None

        lane_idx = int(rel_y / (self.LANE_HEIGHT + self.LANE_GAP))
        return lane_idx if lane_idx < num_lanes else None

    def _lane_to_y(self, lane_idx: int) -> float:
        """Convert lane index to Y coordinate (returns absolute Y including ruler)."""
        if not self._layout:
            return self.RULER_HEIGHT

        lanes_area_h = self._layout.rect.h - self.RULER_HEIGHT
        num_lanes = len(self._lanes)
        total_lane_h = num_lanes * (self.LANE_HEIGHT + self.LANE_GAP)
        start_y = max(0, (lanes_area_h - total_lane_h) / 2)

        return self.RULER_HEIGHT + start_y + lane_idx * (self.LANE_HEIGHT + self.LANE_GAP)

    def update(self):
        """Call each frame to sync state."""
        if self._transport:
            self._playhead_beat = self._transport.playhead_beat
            self._playing = self._transport.playing

    def draw(self, ctx: 'DrawContext'):
        if not self._layout:
            return

        rect = self._layout.rect
        style = self.style

        ctx.push_offset(rect.x, rect.y)
        try:
            local = Rect(0, 0, rect.w, rect.h)
            grid_x = self.LABEL_WIDTH
            grid_w = rect.w - grid_x

            # Background
            if style.background:
                ctx.draw_rect(local, style.background)

            # Ruler background
            ctx.draw_rect(Rect(grid_x, 0, grid_w, self.RULER_HEIGHT), (0.08, 0.085, 0.095, 1.0))

            # Lane labels background
            ctx.draw_rect(Rect(0, self.RULER_HEIGHT, grid_x, rect.h - self.RULER_HEIGHT), (0.09, 0.095, 0.11, 1.0))

            # Beat grid lines and ruler markings
            if self._time_camera:
                for beat in self._time_camera.iter_beat_positions():
                    px = self._time_camera.beat_to_px(beat) + grid_x
                    if grid_x <= px <= rect.w:
                        is_bar = int(beat) % 4 == 0
                        color = (0.22, 0.23, 0.27, 1.0) if is_bar else (0.13, 0.14, 0.16, 1.0)
                        line_w = 2 if is_bar else 1
                        # Grid line (below ruler)
                        ctx.draw_rect(Rect(px, self.RULER_HEIGHT, line_w, rect.h - self.RULER_HEIGHT), color)
                        # Ruler tick
                        tick_h = 12 if is_bar else 6
                        ctx.draw_rect(Rect(px, self.RULER_HEIGHT - tick_h, 1, tick_h), (0.5, 0.5, 0.55, 1.0))
                        # Bar number in ruler
                        if is_bar:
                            bar_num = int(beat) // 4 + 1
                            ctx.draw_text(str(bar_num), px + 4, 4, (0.6, 0.6, 0.65, 1.0), font_size=10)

            # Draw lanes
            for lane in self._lanes:
                lane_y = self._lane_to_y(lane.index)

                # Lane label background
                label_color = (*lane.color[:3], 0.3)
                ctx.draw_rect(Rect(4, lane_y, grid_x - 8, self.LANE_HEIGHT), label_color, radius=4)

                # Lane label text
                ctx.draw_text(lane.name, 8, lane_y + 14, (0.9, 0.9, 0.9, 1.0), font_size=12)

                # Lane row background
                ctx.draw_rect(Rect(grid_x, lane_y, grid_w, self.LANE_HEIGHT), (0.08, 0.085, 0.10, 0.5))

                # Events
                if self._time_camera:
                    for event in lane.events:
                        px = self._time_camera.beat_to_px(event.beat) + grid_x
                        event_w = event.duration * self._time_camera._px_per_beat
                        if px + event_w > grid_x and px < rect.w:
                            alpha = 0.6 + (event.velocity / 127.0) * 0.4
                            color = (*event.color[:3], alpha)

                            event_rect = Rect(
                                max(grid_x, px), lane_y + 3,
                                max(6, event_w - 2), self.LANE_HEIGHT - 6
                            )
                            ctx.draw_rect(event_rect, color, radius=3)

                            # Selection highlight
                            if event.id in self._selected_events:
                                ctx.draw_rect_outline(
                                    Rect(event_rect.x - 1, event_rect.y - 1,
                                         event_rect.w + 2, event_rect.h + 2),
                                    (1.0, 1.0, 1.0, 0.8),
                                    width=2,
                                    radius=4,
                                )

                            # Resize handle (right edge)
                            if event.id in self._selected_events:
                                handle_x = event_rect.x + event_rect.w - 4
                                ctx.draw_rect(
                                    Rect(handle_x, event_rect.y + 4, 3, event_rect.h - 8),
                                    (1.0, 1.0, 1.0, 0.5),
                                )

            # Playhead (spans full height including ruler)
            if self._time_camera:
                px = self._time_camera.beat_to_px(self._playhead_beat) + grid_x
                if grid_x <= px <= rect.w:
                    # Playhead line
                    ctx.draw_rect(Rect(px - 1, 0, 3, rect.h), (1.0, 0.45, 0.25, 0.9))
                    # Playhead head in ruler
                    ctx.draw_rect(Rect(px - 5, 0, 11, 8), (1.0, 0.45, 0.25, 1.0), radius=2)

        finally:
            ctx.pop_offset()


# =============================================================================
# LAUNCHPAD GRID WIDGET
# =============================================================================

class LaunchpadGridWidget(Widget):
    """
    8x8 pad grid visualization mirroring Launchpad hardware.

    Features:
    - 8x8 grid of pads with colors
    - Click/release to trigger pads (velocity sensitive based on position)
    - Syncs with LaunchpadController LED state
    - Optional beat position highlight
    - Row/column headers optional

    Integrates with:
    - SignalBridge for SIGNAL_MIDI_PAD, SIGNAL_MIDI_NOTE_ON/OFF
    - LaunchpadController for LED colors

    Callbacks:
    - on_pad_click: (row: int, col: int, velocity: int, event: Event) -> None
        Called when pad is pressed. Velocity is 1-127 based on click position.

    - on_pad_release: (row: int, col: int, event: Event) -> None
        Called when pad is released (mouse up).

    - on_pad_hold: (row: int, col: int, duration: float) -> None
        Called periodically while pad is held.

    - on_row_click: (row: int, event: Event) -> None
        Called when clicking on row header (if shown).

    - on_column_click: (col: int, event: Event) -> None
        Called when clicking on column header (if shown).

    - on_aftertouch: (row: int, col: int, pressure: int) -> None
        Called when pressure changes on held pad (simulated via mouse Y).
    """

    PAD_SIZE = 40
    PAD_GAP = 4
    HEADER_SIZE = 20

    # Color palette (matches LaunchpadColor)
    COLORS = {
        'off': (0.15, 0.15, 0.18, 1.0),
        'white': (0.9, 0.9, 0.9, 1.0),
        'red': (1.0, 0.2, 0.2, 1.0),
        'orange': (1.0, 0.5, 0.1, 1.0),
        'yellow': (1.0, 0.9, 0.1, 1.0),
        'lime': (0.6, 1.0, 0.2, 1.0),
        'green': (0.2, 0.9, 0.3, 1.0),
        'cyan': (0.2, 0.9, 0.9, 1.0),
        'blue': (0.2, 0.4, 1.0, 1.0),
        'purple': (0.6, 0.2, 1.0, 1.0),
        'magenta': (1.0, 0.2, 0.8, 1.0),
        'pink': (1.0, 0.5, 0.7, 1.0),
    }

    def __init__(
        self,
        signals: 'SignalBridge' = None,
        on_pad_click: Callable[[int, int, int, Event], None] = None,
        on_pad_release: Callable[[int, int, Event], None] = None,
        on_pad_hold: Callable[[int, int, float], None] = None,
        on_row_click: Callable[[int, Event], None] = None,
        on_column_click: Callable[[int, Event], None] = None,
        on_aftertouch: Callable[[int, int, int], None] = None,
        rows: int = 8,
        cols: int = 8,
        show_headers: bool = False,
        style: Style = None,
    ):
        self._signals = signals
        self._on_pad_click = on_pad_click
        self._on_pad_release = on_pad_release
        self._on_pad_hold = on_pad_hold
        self._on_row_click = on_row_click
        self._on_column_click = on_column_click
        self._on_aftertouch = on_aftertouch
        self._rows = rows
        self._cols = cols
        self._show_headers = show_headers

        # Pad colors (8x8 grid of color names)
        self._pad_colors: List[List[str]] = [
            ['off'] * cols for _ in range(rows)
        ]

        # Highlight column (for playhead)
        self._highlight_col: Optional[int] = None
        self._highlight_row: Optional[int] = None

        # Pressed pad state
        self._pressed_pad: Optional[Tuple[int, int]] = None
        self._press_start_time: float = 0.0
        self._last_pressure: int = 0

        header_offset = self.HEADER_SIZE if show_headers else 0
        total_w = cols * (self.PAD_SIZE + self.PAD_GAP) + self.PAD_GAP + header_offset
        total_h = rows * (self.PAD_SIZE + self.PAD_GAP) + self.PAD_GAP + header_offset

        if style is None:
            style = Style(
                background=(0.08, 0.08, 0.10, 1.0),
                width=SizeValue.px(total_w),
                height=SizeValue.px(total_h),
                border=Border(width=2, color=(0.25, 0.25, 0.3, 1.0), radius=8),
            )

        super().__init__(style=style)

        self.on(EventType.POINTER_DOWN, self._on_pointer_down)
        self.on(EventType.POINTER_UP, self._on_pointer_up)
        self.on(EventType.POINTER_MOVE, self._on_pointer_move)

        # Connect to pad signal
        if signals:
            from engine.core.signal import SIGNAL_MIDI_PAD
            signals.connect(SIGNAL_MIDI_PAD, self._on_midi_pad)

    def set_pad_color(self, row: int, col: int, color: str):
        """Set a pad's color by name."""
        if 0 <= row < self._rows and 0 <= col < self._cols:
            self._pad_colors[row][col] = color

    def set_all_colors(self, colors: List[List[str]]):
        """Set all pad colors at once."""
        self._pad_colors = colors

    def set_highlight_column(self, col: Optional[int]):
        """Set column to highlight (for playhead)."""
        self._highlight_col = col

    def set_highlight_row(self, row: Optional[int]):
        """Set row to highlight."""
        self._highlight_row = row

    def clear(self):
        """Clear all pad colors to off."""
        self._pad_colors = [['off'] * self._cols for _ in range(self._rows)]

    def _on_midi_pad(self, row: int, col: int, velocity: int):
        """Handle MIDI pad signal to flash pad."""
        # Visual feedback handled externally via set_pad_color
        pass

    def _on_pointer_down(self, event: Event):
        """Handle click on grid."""
        import time

        header_offset = self.HEADER_SIZE if self._show_headers else 0

        # Check header clicks
        if self._show_headers:
            # Column header
            if event.y < self.HEADER_SIZE:
                col = int((event.x - header_offset - self.PAD_GAP) / (self.PAD_SIZE + self.PAD_GAP))
                if 0 <= col < self._cols and self._on_column_click:
                    self._on_column_click(col, event)
                event.stop_propagation()
                return

            # Row header
            if event.x < self.HEADER_SIZE:
                row = int((event.y - header_offset - self.PAD_GAP) / (self.PAD_SIZE + self.PAD_GAP))
                if 0 <= row < self._rows and self._on_row_click:
                    self._on_row_click(row, event)
                event.stop_propagation()
                return

        row, col = self._pos_to_pad(event.x - header_offset, event.y - header_offset)
        if row is not None and col is not None:
            # Calculate velocity based on position within pad (center = max)
            pad_x = (event.x - header_offset - self.PAD_GAP) % (self.PAD_SIZE + self.PAD_GAP)
            pad_y = (event.y - header_offset - self.PAD_GAP) % (self.PAD_SIZE + self.PAD_GAP)
            center_dist = ((pad_x - self.PAD_SIZE/2)**2 + (pad_y - self.PAD_SIZE/2)**2)**0.5
            max_dist = self.PAD_SIZE * 0.7
            velocity = int(127 * (1 - min(center_dist / max_dist, 1.0) * 0.5))
            velocity = max(64, min(127, velocity))

            self._pressed_pad = (row, col)
            self._press_start_time = time.time()
            self._last_pressure = velocity

            if self._on_pad_click:
                self._on_pad_click(row, col, velocity, event)
            event.stop_propagation()

    def _on_pointer_up(self, event: Event):
        """Handle pad release."""
        if self._pressed_pad:
            row, col = self._pressed_pad
            if self._on_pad_release:
                self._on_pad_release(row, col, event)
            self._pressed_pad = None
            event.stop_propagation()

    def _on_pointer_move(self, event: Event):
        """Handle aftertouch simulation via mouse position."""
        if self._pressed_pad:
            header_offset = self.HEADER_SIZE if self._show_headers else 0
            row, col = self._pressed_pad

            # Simulate aftertouch based on Y position relative to pad
            pad_y = self.PAD_GAP + row * (self.PAD_SIZE + self.PAD_GAP)
            rel_y = (event.y - header_offset - pad_y) / self.PAD_SIZE
            pressure = int(127 * max(0, min(1, rel_y)))

            if pressure != self._last_pressure and self._on_aftertouch:
                self._on_aftertouch(row, col, pressure)
                self._last_pressure = pressure

    def update(self):
        """Call each frame for hold callbacks."""
        import time
        if self._pressed_pad and self._on_pad_hold:
            row, col = self._pressed_pad
            duration = time.time() - self._press_start_time
            self._on_pad_hold(row, col, duration)

    def _pos_to_pad(self, x: float, y: float) -> Tuple[Optional[int], Optional[int]]:
        """Convert position to pad row/col."""
        col = int((x - self.PAD_GAP) / (self.PAD_SIZE + self.PAD_GAP))
        row = int((y - self.PAD_GAP) / (self.PAD_SIZE + self.PAD_GAP))

        if 0 <= row < self._rows and 0 <= col < self._cols:
            return row, col
        return None, None

    def measure(self, constraints: Constraints) -> Tuple[float, float]:
        header_offset = self.HEADER_SIZE if self._show_headers else 0
        w = self._cols * (self.PAD_SIZE + self.PAD_GAP) + self.PAD_GAP + header_offset
        h = self._rows * (self.PAD_SIZE + self.PAD_GAP) + self.PAD_GAP + header_offset
        return constraints.constrain(w, h)

    def draw(self, ctx: 'DrawContext'):
        if not self._layout:
            return

        rect = self._layout.rect
        style = self.style
        header_offset = self.HEADER_SIZE if self._show_headers else 0

        ctx.push_offset(rect.x, rect.y)
        try:
            local = Rect(0, 0, rect.w, rect.h)

            # Background
            if style.background:
                radius = style.border.radius if style.border else 0
                ctx.draw_rect(local, style.background, radius=radius)

            # Draw headers if enabled
            if self._show_headers:
                # Column headers
                for col in range(self._cols):
                    hx = header_offset + self.PAD_GAP + col * (self.PAD_SIZE + self.PAD_GAP)
                    ctx.draw_text(str(col + 1), hx + self.PAD_SIZE // 3, 4, (0.6, 0.6, 0.65, 1.0), font_size=10)

                # Row headers
                for row in range(self._rows):
                    hy = header_offset + self.PAD_GAP + row * (self.PAD_SIZE + self.PAD_GAP)
                    ctx.draw_text(str(row + 1), 4, hy + self.PAD_SIZE // 3, (0.6, 0.6, 0.65, 1.0), font_size=10)

            # Draw pads
            for row in range(self._rows):
                for col in range(self._cols):
                    px = header_offset + self.PAD_GAP + col * (self.PAD_SIZE + self.PAD_GAP)
                    py = header_offset + self.PAD_GAP + row * (self.PAD_SIZE + self.PAD_GAP)

                    color_name = self._pad_colors[row][col]
                    color = self.COLORS.get(color_name, self.COLORS['off'])

                    # Highlight column (playhead position)
                    if col == self._highlight_col:
                        color = tuple(min(1.0, c * 1.3) for c in color[:3]) + (color[3],)

                    # Highlight row
                    if row == self._highlight_row:
                        color = tuple(min(1.0, c * 1.2) for c in color[:3]) + (color[3],)

                    # Pressed pad visual feedback
                    is_pressed = self._pressed_pad == (row, col)
                    if is_pressed:
                        # Darken slightly when pressed
                        color = tuple(c * 0.7 for c in color[:3]) + (color[3],)

                    ctx.draw_rect(Rect(px, py, self.PAD_SIZE, self.PAD_SIZE), color, radius=4)

                    # Pressed outline
                    if is_pressed:
                        ctx.draw_rect_outline(
                            Rect(px, py, self.PAD_SIZE, self.PAD_SIZE),
                            (1.0, 1.0, 1.0, 0.6),
                            width=2,
                            radius=4,
                        )

            # Border
            if style.border and style.border.width > 0:
                ctx.draw_rect_outline(
                    local,
                    style.border.color,
                    width=style.border.width,
                    radius=style.border.radius,
                )

        finally:
            ctx.pop_offset()


# =============================================================================
# WAVEFORM WIDGET
# =============================================================================

class WaveformWidget(Widget):
    """
    Audio waveform visualization.

    Features:
    - Displays waveform envelope (min/max peaks)
    - Syncs view with TimeCamera
    - Playhead line
    - Region selection with drag
    - Click to seek
    - Scroll to zoom

    Integrates with:
    - TimeCamera for beat↔pixel sync
    - Transport for playhead
    - engine/audio/waveform.py for data

    Data format:
    - waveform_data: numpy array of samples (float32, -1 to 1) or
      pre-computed peaks array

    Callbacks:
    - on_click: (beat: float, event: Event) -> None
        Called when clicking on waveform (for seeking).

    - on_region_select: (start_beat: float, end_beat: float) -> None
        Called when drag-selecting a region.

    - on_region_clear: () -> None
        Called when selection is cleared.

    - on_zoom: (px_per_beat: float, center_beat: float) -> None
        Called when zoom level changes.

    - on_scroll: (left_beat: float) -> None
        Called when panning/scrolling.

    - on_marker_add: (beat: float, event: Event) -> bool
        Called on double-click to add marker. Return True if handled.

    - on_cue_point: (beat: float, label: str) -> None
        Called when cue point is triggered.
    """

    def __init__(
        self,
        time_camera: 'TimeCamera' = None,
        transport: 'Transport' = None,
        waveform_data: np.ndarray = None,
        samples_per_beat: float = 100,
        total_beats: float = 16,
        on_click: Callable[[float, Event], None] = None,
        on_region_select: Callable[[float, float], None] = None,
        on_region_clear: Callable[[], None] = None,
        on_zoom: Callable[[float, float], None] = None,
        on_scroll: Callable[[float], None] = None,
        on_marker_add: Callable[[float, Event], bool] = None,
        on_cue_point: Callable[[float, str], None] = None,
        style: Style = None,
        height: float = 80,
    ):
        self._time_camera = time_camera
        self._transport = transport
        self._waveform_data = waveform_data
        self._samples_per_beat = samples_per_beat
        self._total_beats = total_beats

        # Callbacks
        self._on_click = on_click
        self._on_region_select = on_region_select
        self._on_region_clear = on_region_clear
        self._on_zoom = on_zoom
        self._on_scroll = on_scroll
        self._on_marker_add = on_marker_add
        self._on_cue_point = on_cue_point

        # Cached state
        self._playhead_beat = 0.0

        # Selection state
        self._selecting = False
        self._selection_start: Optional[float] = None
        self._selection_end: Optional[float] = None
        self._drag_start_x: float = 0

        # Double-click detection
        self._last_click_time: float = 0
        self._last_click_pos: Tuple[float, float] = (0, 0)

        # Markers/cue points
        self._markers: List[Tuple[float, str]] = []

        if style is None:
            style = Style(
                background=(0.06, 0.07, 0.09, 1.0),
                height=SizeValue.px(height),
            )

        super().__init__(style=style)

        # Event handlers
        self.on(EventType.POINTER_DOWN, self._on_pointer_down)
        self.on(EventType.POINTER_MOVE, self._on_pointer_move)
        self.on(EventType.POINTER_UP, self._on_pointer_up)
        self.on(EventType.SCROLL, self._on_scroll_event)

    def bind_time_camera(self, camera: 'TimeCamera'):
        """Bind to TimeCamera for view sync."""
        self._time_camera = camera

    def bind_transport(self, transport: 'Transport'):
        """Bind to Transport for playhead."""
        self._transport = transport

    def set_waveform(self, data: np.ndarray, samples_per_beat: float = 100, total_beats: float = 16):
        """Set waveform data to display."""
        self._waveform_data = data
        self._samples_per_beat = samples_per_beat
        self._total_beats = total_beats

    def add_marker(self, beat: float, label: str = ""):
        """Add a marker/cue point."""
        self._markers.append((beat, label))
        self._markers.sort(key=lambda m: m[0])

    def clear_markers(self):
        """Remove all markers."""
        self._markers = []

    def set_selection(self, start: Optional[float], end: Optional[float]):
        """Set selection range programmatically."""
        self._selection_start = start
        self._selection_end = end
        if start is not None and end is not None and self._on_region_select:
            self._on_region_select(min(start, end), max(start, end))

    def clear_selection(self):
        """Clear selection."""
        self._selection_start = None
        self._selection_end = None
        if self._on_region_clear:
            self._on_region_clear()

    def _on_pointer_down(self, event: Event):
        """Handle click on waveform."""
        import time

        if not self._time_camera:
            return

        current_time = time.time()
        beat = self._time_camera.px_to_beat(event.x)

        # Check for double-click (add marker)
        is_double = (
            current_time - self._last_click_time < 0.3 and
            abs(event.x - self._last_click_pos[0]) < 5 and
            abs(event.y - self._last_click_pos[1]) < 5
        )

        if is_double and self._on_marker_add:
            if self._on_marker_add(beat, event):
                self._last_click_time = 0
                event.stop_propagation()
                return

        # Start selection or click
        self._selecting = True
        self._selection_start = beat
        self._selection_end = beat
        self._drag_start_x = event.x

        # Single click callback
        if self._on_click:
            self._on_click(beat, event)

        self._last_click_time = current_time
        self._last_click_pos = (event.x, event.y)
        event.stop_propagation()

    def _on_pointer_move(self, event: Event):
        """Handle drag for selection."""
        if not self._selecting or not self._time_camera:
            return

        beat = self._time_camera.px_to_beat(event.x)
        self._selection_end = beat

        # Only trigger callback if dragged beyond threshold
        if abs(event.x - self._drag_start_x) > 5 and self._on_region_select:
            start = min(self._selection_start, self._selection_end)
            end = max(self._selection_start, self._selection_end)
            self._on_region_select(start, end)

    def _on_pointer_up(self, event: Event):
        """End selection drag."""
        if self._selecting:
            # If barely moved, clear selection
            if abs(event.x - self._drag_start_x) < 5:
                self._selection_start = None
                self._selection_end = None
                if self._on_region_clear:
                    self._on_region_clear()
            self._selecting = False

    def _on_scroll_event(self, event: Event):
        """Handle scroll for zoom/pan."""
        if not self._time_camera:
            return

        if event.delta_y != 0:
            center_beat = self._time_camera.px_to_beat(event.x)
            self._time_camera.zoom(event.delta_y, event.x)

            if self._on_zoom:
                self._on_zoom(self._time_camera._px_per_beat, center_beat)
            event.stop_propagation()

        if hasattr(event, 'delta_x') and event.delta_x != 0:
            self._time_camera.pan(event.delta_x)
            if self._on_scroll:
                self._on_scroll(self._time_camera._left_beat)
            event.stop_propagation()

    def update(self):
        """Sync state from transport and check cue points."""
        if self._transport:
            old_beat = self._playhead_beat
            self._playhead_beat = self._transport.playhead_beat

            # Check if we crossed any markers
            if self._on_cue_point and self._transport.playing:
                for beat, label in self._markers:
                    if old_beat < beat <= self._playhead_beat:
                        self._on_cue_point(beat, label)

    def measure(self, constraints: Constraints) -> Tuple[float, float]:
        h = self.style.height.resolve(constraints.max_h, 80)
        return constraints.constrain(constraints.max_w, h)

    def draw(self, ctx: 'DrawContext'):
        if not self._layout:
            return

        rect = self._layout.rect
        style = self.style

        ctx.push_offset(rect.x, rect.y)
        try:
            local = Rect(0, 0, rect.w, rect.h)

            # Background
            if style.background:
                ctx.draw_rect(local, style.background)

            # Selection region
            if self._selection_start is not None and self._selection_end is not None and self._time_camera:
                start = min(self._selection_start, self._selection_end)
                end = max(self._selection_start, self._selection_end)
                start_px = self._time_camera.beat_to_px(start)
                end_px = self._time_camera.beat_to_px(end)
                if end_px > 0 and start_px < rect.w:
                    ctx.draw_rect(
                        Rect(max(0, start_px), 0, min(rect.w, end_px) - max(0, start_px), rect.h),
                        (0.3, 0.5, 0.8, 0.25)
                    )

            # Draw waveform
            if self._waveform_data is not None and len(self._waveform_data) > 0 and self._time_camera:
                self._draw_waveform(ctx, rect.w, rect.h)

            # Markers
            if self._time_camera:
                for beat, label in self._markers:
                    px = self._time_camera.beat_to_px(beat)
                    if 0 <= px <= rect.w:
                        ctx.draw_rect(Rect(px - 1, 0, 2, rect.h), (1.0, 0.8, 0.2, 0.8))
                        if label:
                            ctx.draw_text(label, px + 4, 4, (1.0, 0.8, 0.2, 1.0), font_size=10)

            # Playhead
            if self._time_camera:
                px = self._time_camera.beat_to_px(self._playhead_beat)
                if 0 <= px <= rect.w:
                    ctx.draw_rect(Rect(px - 1, 0, 3, rect.h), (1.0, 0.45, 0.25, 0.9))

        finally:
            ctx.pop_offset()

    def _draw_waveform(self, ctx: 'DrawContext', width: float, height: float):
        """Draw waveform envelope."""
        # Get visible beat range from camera
        start_beat = max(0, self._time_camera._left_beat)
        end_beat = self._time_camera._left_beat + (width / self._time_camera._px_per_beat)

        # Convert to sample indices
        start_idx = int(start_beat * self._samples_per_beat)
        end_idx = int(end_beat * self._samples_per_beat)

        if start_idx >= len(self._waveform_data):
            return

        # Get visible portion of waveform
        visible = self._waveform_data[start_idx:min(end_idx, len(self._waveform_data))]
        if len(visible) == 0:
            return

        # Downsample to pixel resolution
        num_pixels = int(width)
        step = max(1, len(visible) // num_pixels)

        center_y = height / 2
        half_h = height / 2 - 4

        # Draw waveform bars
        for i in range(0, min(len(visible), num_pixels * step), step):
            chunk = visible[i:i + step]
            if len(chunk) == 0:
                continue

            min_v = float(np.min(chunk))
            max_v = float(np.max(chunk))

            px = i // step
            top = center_y + max_v * half_h
            bottom = center_y + min_v * half_h
            bar_h = max(1, top - bottom)

            ctx.draw_rect(Rect(px, bottom, 1, bar_h), (0.3, 0.5, 0.8, 0.8))


# =============================================================================
# CELL LISTENER (for sequencer callbacks)
# =============================================================================

class CellListener:
    """
    Listener interface for sequencer cell events.

    Used to connect sequencer grid events to external systems (audio, MIDI, etc).

    Callbacks:
    - on_trigger: (lane: int, beat: float, event: SequencerEvent, transport: TransportState) -> None
        Called when playback reaches an event's start beat.
        Use this to trigger audio samples, MIDI notes, etc.

    - on_release: (lane: int, beat: float, event: SequencerEvent, transport: TransportState) -> None
        Called when playback reaches an event's end beat.
        Use this to send note-off or stop sustained sounds.

    - on_edit: (lane: int, beat: float, added: bool, event: SequencerEvent | None) -> None
        Called when an event is added (added=True) or removed (added=False).

    - on_move: (event: SequencerEvent, old_lane: int, old_beat: float) -> None
        Called when an event is moved to a new position.

    - on_resize: (event: SequencerEvent, old_duration: float) -> None
        Called when an event's duration changes.

    - on_copy: (events: List[SequencerEvent]) -> None
        Called when events are copied to clipboard.

    - on_paste: (events: List[SequencerEvent], target_lane: int, target_beat: float) -> None
        Called when events are pasted from clipboard.

    - on_delete: (events: List[SequencerEvent]) -> None
        Called when events are deleted.

    - on_velocity_change: (event: SequencerEvent, old_velocity: int) -> None
        Called when an event's velocity changes.

    - on_quantize: (events: List[SequencerEvent], grid_size: float) -> None
        Called when events are quantized to grid.
    """

    def __init__(
        self,
        on_trigger: Callable[[int, float, SequencerEvent, 'TransportState'], None] = None,
        on_release: Callable[[int, float, SequencerEvent, 'TransportState'], None] = None,
        on_edit: Callable[[int, float, bool, Optional[SequencerEvent]], None] = None,
        on_move: Callable[[SequencerEvent, int, float], None] = None,
        on_resize: Callable[[SequencerEvent, float], None] = None,
        on_copy: Callable[[List[SequencerEvent]], None] = None,
        on_paste: Callable[[List[SequencerEvent], int, float], None] = None,
        on_delete: Callable[[List[SequencerEvent]], None] = None,
        on_velocity_change: Callable[[SequencerEvent, int], None] = None,
        on_quantize: Callable[[List[SequencerEvent], float], None] = None,
    ):
        self.on_trigger = on_trigger
        self.on_release = on_release
        self.on_edit = on_edit
        self.on_move = on_move
        self.on_resize = on_resize
        self.on_copy = on_copy
        self.on_paste = on_paste
        self.on_delete = on_delete
        self.on_velocity_change = on_velocity_change
        self.on_quantize = on_quantize

    def trigger(self, lane: int, beat: float, event: SequencerEvent, transport: 'TransportState'):
        """Fire trigger callback if set."""
        if self.on_trigger:
            self.on_trigger(lane, beat, event, transport)

    def release(self, lane: int, beat: float, event: SequencerEvent, transport: 'TransportState'):
        """Fire release callback if set."""
        if self.on_release:
            self.on_release(lane, beat, event, transport)

    def edit(self, lane: int, beat: float, added: bool, event: Optional[SequencerEvent]):
        """Fire edit callback if set."""
        if self.on_edit:
            self.on_edit(lane, beat, added, event)

    def move(self, event: SequencerEvent, old_lane: int, old_beat: float):
        """Fire move callback if set."""
        if self.on_move:
            self.on_move(event, old_lane, old_beat)

    def resize(self, event: SequencerEvent, old_duration: float):
        """Fire resize callback if set."""
        if self.on_resize:
            self.on_resize(event, old_duration)

    def copy(self, events: List[SequencerEvent]):
        """Fire copy callback if set."""
        if self.on_copy:
            self.on_copy(events)

    def paste(self, events: List[SequencerEvent], target_lane: int, target_beat: float):
        """Fire paste callback if set."""
        if self.on_paste:
            self.on_paste(events, target_lane, target_beat)

    def delete(self, events: List[SequencerEvent]):
        """Fire delete callback if set."""
        if self.on_delete:
            self.on_delete(events)

    def velocity_change(self, event: SequencerEvent, old_velocity: int):
        """Fire velocity change callback if set."""
        if self.on_velocity_change:
            self.on_velocity_change(event, old_velocity)

    def quantize(self, events: List[SequencerEvent], grid_size: float):
        """Fire quantize callback if set."""
        if self.on_quantize:
            self.on_quantize(events, grid_size)


# =============================================================================
# SEQUENCER CONTROLLER (Wiring Helper)
# =============================================================================

class SequencerController:
    """
    Helper class to wire together sequencer widgets with transport/MIDI.

    Provides common callback implementations and state management for:
    - Transport control (play/pause/stop/seek)
    - Event editing (add/remove/move/resize)
    - Playback triggering
    - MIDI input handling
    - Clipboard operations

    Usage:
        controller = SequencerController(transport, signals)

        # Wire to widgets
        transport_bar = TransportBarWidget(
            on_play=controller.play,
            on_pause=controller.pause,
            on_stop=controller.stop,
            on_bpm_change=controller.set_bpm,
        )

        grid = SequencerGridWidget(
            on_cell_click=controller.toggle_event,
            on_event_drag=controller.move_event,
            on_playhead_click=controller.seek,
        )

        launchpad = LaunchpadGridWidget(
            on_pad_click=controller.on_pad,
        )

        # In update loop:
        controller.update(dt)
    """

    def __init__(
        self,
        transport: 'Transport' = None,
        signals: 'SignalBridge' = None,
        lanes: List[SequencerLane] = None,
        cell_listener: CellListener = None,
    ):
        self._transport = transport
        self._signals = signals
        self._lanes = lanes or []
        self._cell_listener = cell_listener or CellListener()

        # Event ID counter
        self._next_event_id = 1

        # Clipboard
        self._clipboard: List[SequencerEvent] = []

        # Track fired events for release callbacks
        self._fired_events: Set[int] = set()

        # Connect to MIDI signals
        if signals:
            try:
                from engine.core.signal import SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_NOTE_OFF
                signals.connect(SIGNAL_MIDI_NOTE_ON, self._on_midi_note_on)
                signals.connect(SIGNAL_MIDI_NOTE_OFF, self._on_midi_note_off)
            except ImportError:
                pass

    def bind_transport(self, transport: 'Transport'):
        """Bind to transport."""
        self._transport = transport

    def bind_lanes(self, lanes: List[SequencerLane]):
        """Bind to lanes list."""
        self._lanes = lanes

    def set_cell_listener(self, listener: CellListener):
        """Set the cell listener for callbacks."""
        self._cell_listener = listener

    # -------------------------------------------------------------------------
    # Transport Control
    # -------------------------------------------------------------------------

    def play(self):
        """Start playback."""
        if self._transport:
            self._transport.play()

    def pause(self):
        """Pause playback."""
        if self._transport:
            self._transport.pause()

    def stop(self):
        """Stop playback and seek to start."""
        if self._transport:
            self._transport.pause()
            self._transport.seek(0)
            self._fired_events.clear()

    def seek(self, beat: float, event: Event = None):
        """Seek to beat position."""
        if self._transport:
            self._transport.seek(beat)
            self._fired_events.clear()

    def set_bpm(self, bpm: float):
        """Set BPM."""
        if self._transport:
            self._transport.set_bpm(bpm)

    # -------------------------------------------------------------------------
    # Event Editing
    # -------------------------------------------------------------------------

    def toggle_event(self, lane_idx: int, beat: float, event: Event = None):
        """Toggle event at position (add if empty, remove if exists)."""
        if lane_idx >= len(self._lanes):
            return

        lane = self._lanes[lane_idx]
        quantized_beat = self._quantize_beat(beat)

        # Check if event exists at this position
        for seq_event in lane.events:
            if seq_event.beat <= quantized_beat < seq_event.end_beat:
                # Remove event
                lane.events.remove(seq_event)
                self._cell_listener.edit(lane_idx, quantized_beat, False, seq_event)
                return

        # Add new event
        new_event = SequencerEvent(
            id=self._next_event_id,
            beat=quantized_beat,
            lane=lane_idx,
            duration=1.0,
            velocity=100,
            sample_name=lane.sample_name,
            color=lane.color,
        )
        self._next_event_id += 1
        lane.events.append(new_event)
        self._cell_listener.edit(lane_idx, quantized_beat, True, new_event)

    def add_event(self, lane_idx: int, beat: float, duration: float = 1.0, velocity: int = 100) -> Optional[SequencerEvent]:
        """Add event at position."""
        if lane_idx >= len(self._lanes):
            return None

        lane = self._lanes[lane_idx]
        quantized_beat = self._quantize_beat(beat)

        new_event = SequencerEvent(
            id=self._next_event_id,
            beat=quantized_beat,
            lane=lane_idx,
            duration=duration,
            velocity=velocity,
            sample_name=lane.sample_name,
            color=lane.color,
        )
        self._next_event_id += 1
        lane.events.append(new_event)
        self._cell_listener.edit(lane_idx, quantized_beat, True, new_event)
        return new_event

    def remove_event(self, seq_event: SequencerEvent):
        """Remove an event."""
        if seq_event.lane < len(self._lanes):
            lane = self._lanes[seq_event.lane]
            if seq_event in lane.events:
                lane.events.remove(seq_event)
                self._cell_listener.edit(seq_event.lane, seq_event.beat, False, seq_event)

    def move_event(self, seq_event: SequencerEvent, new_beat: float, new_lane: int):
        """Move event to new position."""
        old_lane = seq_event.lane
        old_beat = seq_event.beat

        # Remove from old lane
        if old_lane < len(self._lanes):
            old_lane_obj = self._lanes[old_lane]
            if seq_event in old_lane_obj.events:
                old_lane_obj.events.remove(seq_event)

        # Update event
        seq_event.beat = self._quantize_beat(new_beat)
        seq_event.lane = new_lane

        # Add to new lane
        if new_lane < len(self._lanes):
            self._lanes[new_lane].events.append(seq_event)

        self._cell_listener.move(seq_event, old_lane, old_beat)

    def resize_event(self, seq_event: SequencerEvent, new_duration: float):
        """Resize event."""
        old_duration = seq_event.duration
        seq_event.duration = max(0.25, new_duration)
        self._cell_listener.resize(seq_event, old_duration)

    def _quantize_beat(self, beat: float, grid: float = 0.25) -> float:
        """Quantize beat to grid."""
        return round(beat / grid) * grid

    # -------------------------------------------------------------------------
    # Playback Triggering
    # -------------------------------------------------------------------------

    def update(self, dt: float = 0):
        """Call each frame to check for event triggers."""
        if not self._transport or not self._transport.playing:
            return

        playhead = self._transport.playhead_beat
        state = self._transport.get_state() if hasattr(self._transport, 'get_state') else None

        for lane in self._lanes:
            for seq_event in lane.events:
                # Check for trigger (event start)
                if seq_event.id not in self._fired_events:
                    if seq_event.beat <= playhead < seq_event.beat + 0.1:
                        self._fired_events.add(seq_event.id)
                        seq_event.fired = True
                        self._cell_listener.trigger(lane.index, seq_event.beat, seq_event, state)

                # Check for release (event end)
                elif seq_event.fired and playhead >= seq_event.end_beat:
                    seq_event.fired = False
                    self._cell_listener.release(lane.index, seq_event.end_beat, seq_event, state)

    # -------------------------------------------------------------------------
    # MIDI Input
    # -------------------------------------------------------------------------

    def on_pad(self, row: int, col: int, velocity: int, event: Event = None):
        """Handle pad press (from LaunchpadGridWidget)."""
        # Map pad to lane/beat
        lane_idx = row
        beat = float(col)

        if velocity > 0:
            self.toggle_event(lane_idx, beat, event)

    def _on_midi_note_on(self, channel: int, note: int, velocity: int):
        """Handle MIDI note on."""
        # Default mapping: note → lane, use current beat
        lane_idx = note % len(self._lanes) if self._lanes else 0
        beat = self._transport.playhead_beat if self._transport else 0
        self.add_event(lane_idx, beat, velocity=velocity)

    def _on_midi_note_off(self, channel: int, note: int, velocity: int):
        """Handle MIDI note off."""
        pass  # Note offs not used for sequencer input

    # -------------------------------------------------------------------------
    # Clipboard
    # -------------------------------------------------------------------------

    def copy_selected(self, selected_ids: Set[int]):
        """Copy selected events to clipboard."""
        self._clipboard = []
        for lane in self._lanes:
            for seq_event in lane.events:
                if seq_event.id in selected_ids:
                    # Clone event
                    self._clipboard.append(SequencerEvent(
                        id=0,  # New ID assigned on paste
                        beat=seq_event.beat,
                        lane=seq_event.lane,
                        duration=seq_event.duration,
                        velocity=seq_event.velocity,
                        sample_name=seq_event.sample_name,
                        color=seq_event.color,
                    ))
        self._cell_listener.copy(self._clipboard)

    def paste(self, target_lane: int, target_beat: float):
        """Paste clipboard at position."""
        if not self._clipboard:
            return

        # Find min beat in clipboard for offset
        min_beat = min(e.beat for e in self._clipboard)
        min_lane = min(e.lane for e in self._clipboard)

        pasted = []
        for template in self._clipboard:
            new_event = SequencerEvent(
                id=self._next_event_id,
                beat=target_beat + (template.beat - min_beat),
                lane=target_lane + (template.lane - min_lane),
                duration=template.duration,
                velocity=template.velocity,
                sample_name=template.sample_name,
                color=template.color,
            )
            self._next_event_id += 1

            if new_event.lane < len(self._lanes):
                self._lanes[new_event.lane].events.append(new_event)
                pasted.append(new_event)

        self._cell_listener.paste(pasted, target_lane, target_beat)

    def delete_selected(self, selected_ids: Set[int]):
        """Delete selected events."""
        deleted = []
        for lane in self._lanes:
            to_remove = [e for e in lane.events if e.id in selected_ids]
            for seq_event in to_remove:
                lane.events.remove(seq_event)
                deleted.append(seq_event)
        self._cell_listener.delete(deleted)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SequencerEvent',
    'SequencerLane',
    'TransportBarWidget',
    'SequencerGridWidget',
    'LaunchpadGridWidget',
    'WaveformWidget',
    'CellListener',
    'SequencerController',
]

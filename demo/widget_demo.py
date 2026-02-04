"""
SPECTRO Widget Demo

Demonstrates the UI widget system with MIDI-aware components:
- TransportBarWidget: Play/pause/stop, beat indicators, BPM, loop toggle
- SequencerGridWidget: Timeline with lanes, event editing
- LaunchpadGridWidget: 8x8 pad mirror with velocity
- WaveformWidget: Audio waveform with region selection

Extended callbacks demonstrated:
- Transport: on_play, on_pause, on_stop, on_bpm_change, on_beat, on_bar, on_loop_toggle
- Grid: on_cell_click, on_event_click, on_event_drag, on_zoom, on_playhead_click
- Launchpad: on_pad_click, on_pad_release, on_aftertouch
- Waveform: on_click, on_region_select, on_marker_add

This version uses the proper widget tree system with SimpleUIRenderer.

Run:
    python demo/widget_demo.py
"""

from __future__ import annotations
import sys
import time
import queue
from pathlib import Path
from typing import Optional, Set

import numpy as np
import moderngl_window as mglw

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Engine imports
from engine.time.transport import Transport
from engine.time.camera import TimeCamera, TimeCameraMode
from engine.core.signal import SignalBridge, SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_PAD
from engine.audio.engine import AudioEngine
from engine.audio.waveform import generate_beat_pattern_waveform
from engine.buffers_v2 import (
    EventDispatcher, MidiRingBuffer, AudioRingBuffer,
    MidiEventType, ExecutionContext, InputDevice
)

# UI system
from engine.ui.widget import RootWidget, Event, EventType
from engine.ui.style import Style, SizeValue, EdgeInsets
from engine.ui.layout import FlexLayout, LayoutDirection, Rect, Constraints
from engine.ui.draw import DrawContext
from engine.ui.renderer import SimpleUIRenderer
from engine.ui.widgets import (
    Container, Column, Row, Spacer,
    TransportBarWidget,
    SequencerGridWidget,
    LaunchpadGridWidget,
    WaveformWidget,
    SequencerEvent,
    SequencerLane,
    CellListener,
    SequencerController,
)

# Optional MIDI
try:
    from engine.midi.manager import MidiManager
    from engine.midi.launchpad import LaunchpadController, LaunchpadColor
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    LaunchpadColor = None


# =============================================================================
# SEQUENCER MODEL
# =============================================================================

class Sequencer:
    """Sequencer data model with lanes and events."""

    def __init__(self):
        self._next_id = 1
        self.lanes = []
        self.cell_listeners = []
        self._setup_default_lanes()

    def _setup_default_lanes(self):
        defaults = [
            ("Kick", "kick", (1.0, 0.3, 0.2, 1.0)),
            ("Snare", "snare", (1.0, 0.6, 0.2, 1.0)),
            ("HiHat", "hihat", (1.0, 0.9, 0.2, 1.0)),
            ("Clap", "clap", (0.4, 1.0, 0.3, 1.0)),
        ]
        for i, (name, sample, color) in enumerate(defaults):
            self.lanes.append(SequencerLane(
                index=i, name=name, sample_name=sample, color=color
            ))

    def add_event(self, lane_idx: int, beat: float, velocity: int = 100,
                  duration: float = 0.25) -> Optional[SequencerEvent]:
        if lane_idx < 0 or lane_idx >= len(self.lanes):
            return None
        lane = self.lanes[lane_idx]
        event = SequencerEvent(
            id=self._next_id,
            beat=beat,
            lane=lane_idx,
            duration=duration,
            velocity=velocity,
            sample_name=lane.sample_name,
            color=lane.color,
        )
        self._next_id += 1
        lane.events.append(event)
        self._dispatch_edit(lane_idx, beat, True, event)
        return event

    def toggle_event(self, lane_idx: int, beat: float, velocity: int = 100,
                     duration: float = 0.25) -> Optional[SequencerEvent]:
        """Toggle event at position."""
        if lane_idx < 0 or lane_idx >= len(self.lanes):
            return None
        lane = self.lanes[lane_idx]

        # Check for existing event
        for event in lane.events:
            if abs(event.beat - beat) < 0.1:
                lane.events.remove(event)
                self._dispatch_edit(lane_idx, beat, False, event)
                return None

        return self.add_event(lane_idx, beat, velocity, duration)

    def get_events_in_range(self, start: float, end: float):
        result = []
        for lane in self.lanes:
            for event in lane.events:
                if event.beat < end and event.beat + event.duration > start:
                    result.append(event)
        return result

    def reset_fired(self):
        for lane in self.lanes:
            for event in lane.events:
                event.fired = False

    def add_cell_listener(self, listener: CellListener):
        self.cell_listeners.append(listener)

    def _dispatch_edit(self, lane: int, beat: float, added: bool, event):
        for listener in self.cell_listeners:
            if listener.on_edit:
                listener.on_edit(lane, beat, added, event)

    def dispatch_trigger(self, event, transport):
        for listener in self.cell_listeners:
            if listener.on_trigger:
                listener.on_trigger(event.lane, event.beat, event, transport)


# =============================================================================
# DEMO APP
# =============================================================================

class WidgetDemo(mglw.WindowConfig):
    """Demo using the widget system for MIDI visualization."""

    gl_version = (3, 3)
    title = "SPECTRO Widget Demo"
    window_size = (1280, 720)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA

        # Core systems
        self.signals = SignalBridge()
        self.transport = Transport(bpm=120.0)
        self.time_camera = TimeCamera(mode=TimeCameraMode.FOLLOW_PLAYHEAD)
        self.sequencer = Sequencer()

        # Audio
        self.audio = AudioEngine(
            transport=self.transport,
            signals=self.signals,
            sample_rate=44100,
            buffer_size=512,
        )
        self.audio.load_default_sounds()

        # Event dispatcher
        self.midi_buffer = MidiRingBuffer(capacity=4096)
        self.audio_buffer = AudioRingBuffer(capacity_samples=65536)
        self.dispatcher = EventDispatcher(self.midi_buffer, self.audio_buffer)
        self.dispatcher.set_bpm(self.transport.bpm)

        # Keyboard input
        self.keyboard_device = InputDevice("Keyboard")
        self.keyboard_device.connect(self.midi_buffer, self.audio_buffer)

        # MIDI queue
        self._midi_queue = queue.Queue()
        self.midi_manager = None
        self.launchpad = None

        if MIDI_AVAILABLE:
            self._init_midi()

        # Generate test waveform
        pattern = {"kick": [0, 4, 8, 12], "snare": [4, 12], "hihat": list(range(0, 16, 2))}
        self.waveform_data = generate_beat_pattern_waveform(pattern, bars=4, bpm=120.0)

        # Build widget tree
        self._build_widget_tree()

        # Setup callbacks
        self._setup_callbacks()

        # Start audio
        self.audio.start()
        self.audio_buffer.write_silence(44100)

        # Rendering
        w, h = self.window_size
        self.draw_ctx = DrawContext(w, h)
        self.ui_renderer = SimpleUIRenderer(self.ctx)

        self.last_time = time.perf_counter()

        print("\n=== SPECTRO Widget Demo (Extended Callbacks) ===")
        print("Keys: 1-4 = Drums | SPACE = Play/Pause | R = Reset | Arrows = Seek")
        print("Grid: Click = toggle | Drag = move | Edge = resize | Label = preview")
        print("Waveform: Click = seek | Drag = select region | Double-click = marker")
        print("Transport: Play/Pause/Stop | BPM (click to cycle) | Loop toggle")
        print("=" * 60 + "\n")

    def _init_midi(self):
        """Initialize MIDI."""
        self.signals.connect(SIGNAL_MIDI_NOTE_ON, self._on_midi_note)
        self.midi_manager = MidiManager(self.signals)
        if self.midi_manager.is_available:
            if self.midi_manager.connect("Launchpad"):
                print("[MIDI] Launchpad connected")
                self.launchpad = LaunchpadController(self.midi_manager)
                self.signals.connect(SIGNAL_MIDI_PAD, self._on_pad)
            else:
                print("[MIDI] No Launchpad found")

    def _build_widget_tree(self):
        """Build the UI widget tree."""
        w, h = self.window_size

        # Root widget
        self.root = RootWidget(w, h)

        # Main container (fills entire window)
        main_style = Style(
            background=(0.07, 0.08, 0.10, 1.0),
            flex_grow=1.0,
        )
        main_flex = FlexLayout(direction=LayoutDirection.COLUMN, gap=0)
        main_container = Container(
            style=main_style,
            direction=LayoutDirection.COLUMN,
            gap=0,
        )
        self.root.add_child(main_container)

        # Transport bar at top (with extended callbacks)
        self.transport_widget = TransportBarWidget(
            transport=self.transport,
            signals=self.signals,
            on_play=self._on_transport_play,
            on_pause=self._on_transport_pause,
            on_stop=self._on_transport_stop,
            on_bpm_change=self._on_bpm_change,
            on_beat=self._on_beat_callback,
            on_bar=self._on_bar_callback,
            on_loop_toggle=self._on_loop_toggle,
            height=56,
        )
        main_container.add_child(self.transport_widget)

        # Content area (waveform + sequencer + launchpad)
        content_style = Style(flex_grow=1.0)
        content_flex = FlexLayout(direction=LayoutDirection.ROW, gap=0)
        content_container = Container(
            style=content_style,
            direction=LayoutDirection.ROW,
            gap=8,
            padding=8,
        )
        main_container.add_child(content_container)

        # Left side: waveform + sequencer (column)
        left_style = Style(flex_grow=1.0)
        left_container = Container(
            style=left_style,
            direction=LayoutDirection.COLUMN,
            gap=8,
        )
        content_container.add_child(left_container)

        # Waveform (with extended callbacks)
        self.waveform_widget = WaveformWidget(
            time_camera=self.time_camera,
            transport=self.transport,
            waveform_data=self.waveform_data,
            samples_per_beat=100,
            total_beats=16,
            on_click=self._on_waveform_click,
            on_region_select=self._on_waveform_region_select,
            on_marker_add=self._on_waveform_marker_add,
            on_zoom=self._on_waveform_zoom,
            height=80,
        )
        left_container.add_child(self.waveform_widget)

        # Sequencer grid (with extended callbacks)
        self.sequencer_widget = SequencerGridWidget(
            time_camera=self.time_camera,
            transport=self.transport,
            signals=self.signals,
            lanes=self.sequencer.lanes,
            on_cell_click=self._on_grid_cell_click,
            on_event_click=self._on_grid_event_click,
            on_event_drag=self._on_grid_event_drag,
            on_event_resize=self._on_grid_event_resize,
            on_selection_change=self._on_grid_selection_change,
            on_lane_click=self._on_grid_lane_click,
            on_zoom=self._on_grid_zoom,
            on_playhead_click=self._on_grid_playhead_click,
        )
        left_container.add_child(self.sequencer_widget)

        # Right side: Launchpad grid (with extended callbacks)
        self.launchpad_widget = LaunchpadGridWidget(
            signals=self.signals,
            on_pad_click=self._on_launchpad_widget_click,
            on_pad_release=self._on_launchpad_widget_release,
            on_pad_hold=self._on_launchpad_widget_hold,
            on_aftertouch=self._on_launchpad_aftertouch,
            rows=4,  # 4 rows for 4 lanes
            cols=8,
            show_headers=True,
        )
        content_container.add_child(self.launchpad_widget)

        # Set panel size for time camera
        self.time_camera.set_panel_size(float(w - 400), float(h - 150))

        # Initial layout
        self.root.do_layout()

    def _setup_callbacks(self):
        """Wire up event callbacks."""

        def on_midi_input(ctx: ExecutionContext):
            event = ctx.event
            lane_idx = event.note % len(self.sequencer.lanes)
            seq_event = self.sequencer.add_event(
                lane_idx, ctx.transport.beat, event.velocity, 0.25
            )
            if seq_event:
                lane = self.sequencer.lanes[lane_idx]
                print(f"[+] {lane.name} @ {seq_event.beat:.2f}")
                self.audio.trigger(lane.sample_name, event.velocity / 127.0)
                self._update_launchpad_colors()

        self.dispatcher.register(
            callback=on_midi_input,
            event_types={MidiEventType.NOTE_ON},
            name="midi_to_sequencer"
        )

        def on_beat(beat: int, transport):
            if not self.transport.playing:
                return
            window = 0.1
            current = self.transport.playhead_beat
            events = self.sequencer.get_events_in_range(current - window, current + window)
            for event in events:
                if not event.fired and event.beat <= current:
                    event.fired = True
                    self.audio.trigger(event.sample_name, event.velocity / 127.0)
                    self.sequencer.dispatch_trigger(event, transport)

        self.dispatcher.on_beat(on_beat)
        self.transport.on_loop_callbacks.append(lambda s: self.sequencer.reset_fired())

        # Cell listener for launchpad LED updates
        self.sequencer.add_cell_listener(CellListener(
            on_edit=lambda lane, beat, added, event: self._update_launchpad_colors()
        ))

    def _on_midi_note(self, note: int, velocity: int, channel: int):
        """MIDI note callback."""
        self._midi_queue.put((note, velocity))

    def _on_pad(self, row: int, col: int, velocity: int):
        """Launchpad pad callback."""
        if velocity == 0:
            return
        lane_idx = row % len(self.sequencer.lanes)
        beat = float(col)
        event = self.sequencer.toggle_event(lane_idx, beat, velocity, 0.25)
        if event:
            lane = self.sequencer.lanes[lane_idx]
            self.audio.trigger(lane.sample_name, velocity / 127.0)
            print(f"[LP] {lane.name} @ {beat:.0f}")
        self._update_launchpad_colors()

    # =========================================================================
    # TRANSPORT CALLBACKS
    # =========================================================================

    def _on_transport_play(self):
        """Transport play callback."""
        self.transport.play()
        print("[Transport] Play")

    def _on_transport_pause(self):
        """Transport pause callback."""
        self.transport.pause()
        print("[Transport] Pause")

    def _on_transport_stop(self):
        """Transport stop callback."""
        self.transport.pause()
        self.transport.seek(0)
        self.sequencer.reset_fired()
        print("[Transport] Stop -> 0")

    def _on_bpm_change(self, bpm: float):
        """BPM changed callback."""
        self.transport.set_bpm(bpm)
        self.dispatcher._bpm = bpm
        print(f"[Transport] BPM = {bpm:.0f}")

    def _on_beat_callback(self, beat: int, bar: int, phase: float):
        """Beat crossing callback (from transport widget)."""
        pass  # Could trigger metronome click or visual flash

    def _on_bar_callback(self, bar: int):
        """Bar crossing callback."""
        pass  # Could trigger loop reset visual

    def _on_loop_toggle(self, enabled: bool, start: float, end: float):
        """Loop toggle callback."""
        print(f"[Transport] Loop={'ON' if enabled else 'OFF'} ({start:.1f}-{end:.1f})")
        # Could set transport loop range here

    # =========================================================================
    # WAVEFORM CALLBACKS
    # =========================================================================

    def _on_waveform_click(self, beat: float, event: Event):
        """Click on waveform to seek."""
        self.transport.seek(beat)
        print(f"[Waveform] Seek -> {beat:.2f}")

    def _on_waveform_region_select(self, start: float, end: float):
        """Waveform region selected."""
        print(f"[Waveform] Region: {start:.2f} - {end:.2f}")
        # Could set loop points or selection

    def _on_waveform_marker_add(self, beat: float, event: Event) -> bool:
        """Double-click on waveform to add marker."""
        self.waveform_widget.add_marker(beat, f"M{int(beat)}")
        print(f"[Waveform] Marker @ {beat:.2f}")
        return True

    def _on_waveform_zoom(self, px_per_beat: float, center_beat: float):
        """Waveform zoom callback."""
        pass  # Could sync other views

    # =========================================================================
    # GRID CALLBACKS
    # =========================================================================

    def _on_grid_cell_click(self, lane: int, beat: float, event: Event):
        """Click on empty grid cell."""
        beat = int(beat)
        seq_event = self.sequencer.toggle_event(lane, float(beat), 100, 0.25)
        if seq_event:
            lane_obj = self.sequencer.lanes[lane]
            self.audio.trigger(lane_obj.sample_name, 0.8)
            print(f"[Grid] {lane_obj.name} @ {beat}")
        self._update_launchpad_colors()

    def _on_grid_event_click(self, event: SequencerEvent, ui_event: Event):
        """Click on existing event to remove it."""
        lane = self.sequencer.lanes[event.lane]
        if event in lane.events:
            lane.events.remove(event)
            self.sequencer._dispatch_edit(event.lane, event.beat, False, event)
            print(f"[Grid] Removed {lane.name} @ {event.beat:.0f}")
        self._update_launchpad_colors()

    def _on_grid_event_drag(self, event: SequencerEvent, new_beat: float, new_lane: int):
        """Event being dragged."""
        # Update event position
        old_lane = event.lane
        old_beat = event.beat

        # Remove from old lane
        if old_lane < len(self.sequencer.lanes):
            old_lane_obj = self.sequencer.lanes[old_lane]
            if event in old_lane_obj.events:
                old_lane_obj.events.remove(event)

        # Update position
        event.beat = int(new_beat)  # Quantize to beat
        event.lane = new_lane

        # Add to new lane
        if new_lane < len(self.sequencer.lanes):
            self.sequencer.lanes[new_lane].events.append(event)
            lane_obj = self.sequencer.lanes[new_lane]
            print(f"[Grid] Move {lane_obj.name} -> beat {event.beat}")

        self._update_launchpad_colors()

    def _on_grid_event_resize(self, event: SequencerEvent, new_duration: float):
        """Event being resized."""
        event.duration = max(0.25, new_duration)
        print(f"[Grid] Resize -> {event.duration:.2f} beats")

    def _on_grid_selection_change(self, selected: Set[int]):
        """Selection changed."""
        if selected:
            print(f"[Grid] Selected {len(selected)} event(s)")

    def _on_grid_lane_click(self, lane: int, event: Event):
        """Click on lane label."""
        if lane < len(self.sequencer.lanes):
            lane_obj = self.sequencer.lanes[lane]
            self.audio.trigger(lane_obj.sample_name, 1.0)
            print(f"[Grid] Preview {lane_obj.name}")

    def _on_grid_zoom(self, px_per_beat: float, center_beat: float):
        """Grid zoom callback."""
        pass  # Already handled by time camera

    def _on_grid_playhead_click(self, beat: float, event: Event):
        """Click in playhead/ruler area."""
        self.transport.seek(beat)
        print(f"[Grid] Seek -> {beat:.2f}")

    def _on_launchpad_widget_click(self, row: int, col: int, velocity: int, event: Event):
        """Click on launchpad widget (with velocity)."""
        if row < len(self.sequencer.lanes):
            beat = float(col)
            seq_event = self.sequencer.toggle_event(row, beat, velocity, 0.25)
            if seq_event:
                lane = self.sequencer.lanes[row]
                self.audio.trigger(lane.sample_name, velocity / 127.0)
                print(f"[LP Widget] {lane.name} @ {beat:.0f} vel={velocity}")
            self._update_launchpad_colors()

    def _on_launchpad_widget_release(self, row: int, col: int, event: Event):
        """Launchpad pad release."""
        pass  # Could trigger note-off for sustained samples

    def _on_launchpad_widget_hold(self, row: int, col: int, duration: float):
        """Launchpad pad held."""
        pass  # Could show hold duration indicator

    def _on_launchpad_aftertouch(self, row: int, col: int, pressure: int):
        """Launchpad aftertouch simulation."""
        pass  # Could modulate velocity or filter

    def _update_launchpad_colors(self):
        """Update launchpad widget colors from sequencer state."""
        color_map = ['red', 'orange', 'yellow', 'green']

        for row, lane in enumerate(self.sequencer.lanes):
            if row >= 4:
                break
            for col in range(8):
                has_event = any(int(e.beat) == col for e in lane.events)
                color = color_map[row] if has_event else 'off'
                self.launchpad_widget.set_pad_color(row, col, color)

        # Highlight playhead column
        playhead_col = int(self.transport.playhead_beat) % 8
        self.launchpad_widget.set_highlight_column(playhead_col if self.transport.playing else None)

    # =========================================================================
    # RENDER
    # =========================================================================

    def on_render(self, t: float, frame_time: float):
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_time)
        self.last_time = now

        # Update systems
        self.transport.update(dt)
        self.dispatcher._bpm = self.transport.bpm
        if self.transport.playing and not self.dispatcher.playing:
            self.dispatcher.play()
        elif not self.transport.playing and self.dispatcher.playing:
            self.dispatcher.pause()
        self.dispatcher.process_frame(dt)
        self.time_camera.update(dt, self.transport.playhead_beat)

        # Drain MIDI queue
        while True:
            try:
                note, vel = self._midi_queue.get_nowait()
                lane = note % len(self.sequencer.lanes)
                lane_obj = self.sequencer.lanes[lane]
                ev = self.sequencer.add_event(lane, self.transport.playhead_beat, vel, 0.25)
                if ev:
                    self.audio.trigger(lane_obj.sample_name, vel / 127.0)
                    print(f"[MIDI] {lane_obj.name} @ {ev.beat:.2f}")
                self._update_launchpad_colors()
            except queue.Empty:
                break

        # Update widgets
        self.transport_widget.update()
        self.sequencer_widget.update()
        self.waveform_widget.update()
        self.launchpad_widget.update()
        self._update_launchpad_colors()

        # Render
        w, h = self.wnd.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.07, 0.08, 0.10, 1.0)

        # Update root widget size if window resized
        if w != self.root._window_width or h != self.root._window_height:
            self.root.set_window_size(w, h)
            self.draw_ctx = DrawContext(w, h)

        # Draw widget tree to batch
        self.draw_ctx.clear()
        self.root.draw(self.draw_ctx)
        batch = self.draw_ctx.finalize()

        # Render the batch
        self.ui_renderer.render(batch, w, h)

    # =========================================================================
    # INPUT
    # =========================================================================

    def key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return

        key_to_note = {
            self.wnd.keys.NUMBER_1: 0,
            self.wnd.keys.NUMBER_2: 1,
            self.wnd.keys.NUMBER_3: 2,
            self.wnd.keys.NUMBER_4: 3,
        }

        if key in key_to_note:
            self.keyboard_device.note_on(key_to_note[key], velocity=100)
        elif key == self.wnd.keys.SPACE:
            self.transport.toggle()
            print(f"[>] {'Playing' if self.transport.playing else 'Paused'}")
        elif key == self.wnd.keys.R:
            self.transport.stop()
            self.sequencer.reset_fired()
            print("[>] Reset")
        elif key == self.wnd.keys.LEFT:
            self.transport.seek_by_bars(-1)
        elif key == self.wnd.keys.RIGHT:
            self.transport.seek_by_bars(1)

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        self.root.dispatch_pointer_event(EventType.POINTER_MOVE, x, y)

    def mouse_press_event(self, x: int, y: int, button: int):
        self.root.dispatch_pointer_event(EventType.POINTER_DOWN, x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.root.dispatch_pointer_event(EventType.POINTER_UP, x, y, button)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        x, y = self.wnd.mouse
        self.root.dispatch_scroll_event(x, y, x_offset, y_offset)

    def close(self):
        self.audio.stop()
        self.ui_renderer.release()


if __name__ == "__main__":
    mglw.run_window_config(WidgetDemo)

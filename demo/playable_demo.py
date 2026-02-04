"""
SPECTRO Playable Demo - Sequencer + Transport

A minimal playable demo with:
- Transport bar (play/pause, beat indicators, BPM)
- Sequencer grid (4 lanes, events on timeline)
- Audio playback (drum samples)
- Keyboard input (1-4 = drums)
- Launchpad/MIDI input (optional)

Run:
    python demo/playable_demo.py

Controls:
    1-4     Trigger drums (Kick, Snare, HiHat, Clap)
    SPACE   Play / Pause
    R       Reset to beat 0
    L/R     Seek by bar
    Scroll  Zoom timeline
"""

from __future__ import annotations
import sys
import time
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Callable

import numpy as np
import moderngl_window as mglw

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# =============================================================================
# ENGINE IMPORTS - Use existing engine components
# =============================================================================

from engine.time.transport import Transport, TimeSignature
from engine.time.camera import TimeCamera, TimeCameraMode
from engine.core.signal import SignalBridge, SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_PAD
from engine.audio.engine import AudioEngine
from engine.buffers_v2 import (
    EventDispatcher, MidiRingBuffer, AudioRingBuffer,
    MidiEvent, MidiEventType, ExecutionContext, InputDevice
)

# WebSocket + Logging (engine/ws/)
from engine.ws import ResponseLog, WSClient, DarkScheduler

# Text Rendering (engine/graph/)
from engine.graph.text import TextRenderer, TextBaseline

# Waveform utilities (engine/audio/)
from engine.audio.waveform import (
    extract_waveform,
    generate_test_waveform,
    generate_beat_pattern_waveform,
)

# Optional MIDI / Launchpad
try:
    from engine.midi.manager import MidiManager
    from engine.midi.launchpad import LaunchpadController, LaunchpadColor, LaunchpadMapper
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    LaunchpadColor = None


# =============================================================================
# SEQUENCER DATA MODEL
# =============================================================================

@dataclass
class SequencerEvent:
    """An event on the timeline."""
    id: int
    beat: float
    lane: int
    duration: float
    velocity: int
    sample_name: str
    color: tuple = (1.0, 0.5, 0.2, 1.0)
    fired: bool = False

    @property
    def end_beat(self) -> float:
        return self.beat + self.duration


class SequencerLane:
    """A horizontal lane in the sequencer."""
    def __init__(self, index: int, name: str, sample_name: str, color: tuple):
        self.index = index
        self.name = name
        self.sample_name = sample_name
        self.color = color
        self.events: List[SequencerEvent] = []


class Sequencer:
    """Sequencer with lanes and events."""
    def __init__(self):
        self._next_id = 1
        self.lanes: List[SequencerLane] = []
        self.cell_listeners: List[CellListener] = []
        self._setup_default_lanes()

    def _setup_default_lanes(self):
        defaults = [
            ("Kick", "kick", (1.0, 0.3, 0.2, 1.0)),
            ("Snare", "snare", (1.0, 0.6, 0.2, 1.0)),
            ("HiHat", "hihat", (1.0, 0.9, 0.2, 1.0)),
            ("Clap", "clap", (0.4, 1.0, 0.3, 1.0)),
        ]
        for i, (name, sample, color) in enumerate(defaults):
            self.lanes.append(SequencerLane(i, name, sample, color))

    def add_event(self, lane_index: int, beat: float, velocity: int = 100,
                  duration: float = 0.25) -> Optional[SequencerEvent]:
        if lane_index < 0 or lane_index >= len(self.lanes):
            return None
        lane = self.lanes[lane_index]
        event = SequencerEvent(
            id=self._next_id,
            beat=beat,
            lane=lane_index,
            duration=duration,
            velocity=velocity,
            sample_name=lane.sample_name,
            color=lane.color,
        )
        self._next_id += 1
        lane.events.append(event)
        self._dispatch_edit(lane_index, beat, added=True, event=event)
        return event

    def remove_event(self, event_id: int) -> bool:
        """Remove an event by ID. Returns True if found and removed."""
        for lane in self.lanes:
            for event in lane.events:
                if event.id == event_id:
                    lane.events.remove(event)
                    self._dispatch_edit(event.lane, event.beat, added=False, event=event)
                    return True
        return False

    def get_event_at(self, lane_index: int, beat: float, tolerance: float = 0.1) -> Optional[SequencerEvent]:
        """Find an event at a specific lane/beat position."""
        if lane_index < 0 or lane_index >= len(self.lanes):
            return None
        for event in self.lanes[lane_index].events:
            if abs(event.beat - beat) < tolerance:
                return event
        return None

    def toggle_event(self, lane_index: int, beat: float, velocity: int = 100,
                     duration: float = 0.25) -> Optional[SequencerEvent]:
        """Toggle event at position: remove if exists, add if not."""
        existing = self.get_event_at(lane_index, beat)
        if existing:
            self.remove_event(existing.id)
            return None
        return self.add_event(lane_index, beat, velocity, duration)

    def get_events_in_range(self, start_beat: float, end_beat: float) -> List[SequencerEvent]:
        result = []
        for lane in self.lanes:
            for event in lane.events:
                if event.beat < end_beat and event.end_beat > start_beat:
                    result.append(event)
        return result

    def reset_fired_flags(self):
        for lane in self.lanes:
            for event in lane.events:
                event.fired = False

    def add_cell_listener(self, listener: 'CellListener') -> None:
        """Register a listener for cell trigger/edit events."""
        self.cell_listeners.append(listener)

    def _dispatch_edit(self, lane: int, beat: float, added: bool, event: Optional[SequencerEvent]) -> None:
        """Notify listeners of edit (add/remove)."""
        for listener in self.cell_listeners:
            if listener.on_edit:
                listener.on_edit(lane, beat, added, event)

    def dispatch_trigger(self, event: SequencerEvent, transport) -> None:
        """Notify listeners of playback trigger."""
        for listener in self.cell_listeners:
            if listener.on_trigger:
                listener.on_trigger(event.lane, event.beat, event, transport)


class CellListener:
    """
    Listener for sequencer cell events.

    on_trigger: Called when playback reaches an event.
        Signature: (lane: int, beat: float, event: SequencerEvent, transport) -> None

    on_edit: Called when an event is added or removed.
        Signature: (lane: int, beat: float, added: bool, event: Optional[SequencerEvent]) -> None
    """
    def __init__(self,
                 on_trigger: Optional[Callable] = None,
                 on_edit: Optional[Callable] = None):
        self.on_trigger = on_trigger
        self.on_edit = on_edit


# =============================================================================
# LAYOUT
# =============================================================================

TRANSPORT_HEIGHT = 56
WAVEFORM_HEIGHT = 80
LANE_HEIGHT = 48
LANE_GAP = 4
LANE_LABEL_WIDTH = 80
RIGHT_PANEL_WIDTH = 320


# =============================================================================
# DEMO APP
# =============================================================================

class PlayableDemo(mglw.WindowConfig):
    """Minimal playable demo: sequencer + transport + audio."""

    gl_version = (4, 3)  # 4.3 for TextRenderer SSBO
    title = "SPECTRO Playable Demo"
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

        # Keyboard as input device
        self.keyboard_device = InputDevice("Keyboard")
        self.keyboard_device.connect(self.midi_buffer, self.audio_buffer)

        # Optional components (stubs show how to enable)
        self.output_log: Optional[ResponseLog] = None
        self.ws_client: Optional[WSClient] = None
        self.text_renderer: Optional[TextRenderer] = None
        self.waveform_data: Optional[np.ndarray] = None
        self.launchpad: Optional[LaunchpadController] = None

        # MIDI / Launchpad
        self._midi_queue: queue.Queue = queue.Queue()
        self.midi_manager = None
        if MIDI_AVAILABLE:
            self._init_midi()

        # Wire callbacks
        self._setup_callbacks()

        # Rendering
        self._build_quad_pipeline()
        w, h = self.window_size
        seq_height = h - TRANSPORT_HEIGHT
        self.time_camera.set_panel_size(float(w - LANE_LABEL_WIDTH), float(seq_height))

        # Start audio
        self.audio.start()
        self.audio_buffer.write_silence(44100)

        self.last_time = time.perf_counter()

        print("\n=== SPECTRO Playable Demo ===")
        print("Keys: 1-4 = Drums | SPACE = Play/Pause | R = Reset | Scroll = Zoom")
        print("=" * 32 + "\n")

    # =========================================================================
    # INITIALIZATION: OPTIONAL COMPONENTS
    # =========================================================================

    def _init_midi(self) -> None:
        """
        Initialize MIDI input via MidiManager and optional Launchpad.

        Uses engine/midi/manager.py MidiManager for MIDI routing and
        engine/midi/launchpad.py LaunchpadController for grid LED feedback.
        """
        self.signals.connect(SIGNAL_MIDI_NOTE_ON, self._on_midi_note)
        self.midi_manager = MidiManager(self.signals)

        if self.midi_manager.is_available:
            # Try to connect Launchpad
            if self.midi_manager.connect("Launchpad"):
                print("[MIDI] Launchpad connected")
                # Create LaunchpadController for LED control
                self.launchpad = LaunchpadController(self.midi_manager)
                self.signals.connect(SIGNAL_MIDI_PAD, self._on_launchpad_pad)
                # Add cell listener for LED updates
                self.sequencer.add_cell_listener(CellListener(
                    on_edit=self._on_sequencer_edit_for_leds
                ))
            else:
                print("[MIDI] No Launchpad found")

    def _init_output_panel(self, enable_tee: bool = True) -> None:
        """
        Initialize the output panel using engine/ws/response_log.py ResponseLog
        and engine/graph/text.py TextRenderer.

        Args:
            enable_tee: If True, tee stdout/stderr into the log.
        """
        self.output_log = ResponseLog(max_lines=500)
        self.output_log.append("[SPECTRO] Output panel initialized")

        # TextRenderer for GPU-accelerated text (requires GL 4.3)
        self.text_renderer = TextRenderer(self.ctx)

        if enable_tee:
            # Tee stdout/stderr - use TeeOutput pattern from run_demo.py
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = _TeeStream(sys.stdout, self.output_log, "[stdout]")
            sys.stderr = _TeeStream(sys.stderr, self.output_log, "[stderr]")

    def _init_websocket(self, url: Optional[str] = None) -> None:
        """
        Initialize WebSocket client using engine/ws/client.py WSClient.

        Messages are routed to self.output_log if initialized.

        Args:
            url: WebSocket URL, or reads from SPECTRO_WS_URL env var.
        """
        import os
        self.ws_client = WSClient(response_log=self.output_log)
        ws_url = url or os.environ.get("SPECTRO_WS_URL", "")
        if ws_url:
            self.ws_client.connect(ws_url)
            if self.output_log:
                self.output_log.append(f"[WS] Connecting to {ws_url}")
        else:
            if self.output_log:
                self.output_log.append("[WS] No URL; set SPECTRO_WS_URL to connect")

    def _init_waveform(self, audio_path: Optional[str] = None) -> None:
        """
        Initialize waveform display using engine/audio/waveform.py utilities.

        Args:
            audio_path: Path to audio file, or None for test waveform.
        """
        if audio_path:
            # Load from file (requires audio loading - use librosa or similar)
            # samples, sr = load_audio(audio_path)
            # self.waveform_data = extract_waveform(samples, sr, self.transport.bpm)
            pass
        else:
            # Generate test waveform with beat pattern
            pattern = {
                "kick": [0, 4, 8, 12],
                "snare": [4, 12],
                "hihat": [0, 2, 4, 6, 8, 10, 12, 14],
            }
            self.waveform_data = generate_beat_pattern_waveform(
                pattern, bars=4, bpm=self.transport.bpm
            )

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    def _shutdown_output_panel(self) -> None:
        """Restore stdout/stderr if tee'd."""
        if hasattr(self, '_original_stdout'):
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr

    def _shutdown_websocket(self) -> None:
        """Disconnect WebSocket client."""
        if self.ws_client:
            self.ws_client.disconnect()

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _setup_callbacks(self):
        """Wire MIDI input to sequencer + audio."""

        def on_midi_input(ctx: ExecutionContext):
            event = ctx.event
            lane_index = event.note % len(self.sequencer.lanes)
            seq_event = self.sequencer.add_event(
                lane_index=lane_index,
                beat=ctx.transport.beat,
                velocity=event.velocity,
                duration=0.25,
            )
            if seq_event:
                lane = self.sequencer.lanes[lane_index]
                print(f"[+] {lane.name} @ {seq_event.beat:.2f}")
                self.audio.trigger(lane.sample_name, event.velocity / 127.0)

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
        self.transport.on_loop_callbacks.append(lambda s: self.sequencer.reset_fired_flags())

    def _on_midi_note(self, note: int, velocity: int, channel: int):
        """MIDI callback (from MidiManager thread) - queue for main thread."""
        self._midi_queue.put((note, velocity))

    def _on_launchpad_pad(self, row: int, col: int, velocity: int):
        """
        Handle Launchpad pad press for grid-based note entry.
        Uses engine/midi/launchpad.py LaunchpadMapper for coordinate mapping.
        """
        if velocity == 0:
            return

        # Map pad to lane/beat
        lane_index = row % len(self.sequencer.lanes)
        beats_per_page = 8
        page_start = int(self.transport.playhead_beat // beats_per_page) * beats_per_page
        beat = float(page_start + col)

        # Toggle event
        event = self.sequencer.toggle_event(lane_index, beat, velocity, 0.25)
        if event:
            # New event added - play sound
            lane = self.sequencer.lanes[lane_index]
            self.audio.trigger(lane.sample_name, velocity / 127.0)
            print(f"[LP] {lane.name} @ {beat:.0f}")
        else:
            print(f"[LP] Removed @ lane {lane_index}, beat {beat:.0f}")

        self._update_launchpad_leds()

    def _on_sequencer_edit_for_leds(self, lane: int, beat: float, added: bool,
                                     event: Optional[SequencerEvent]) -> None:
        """Cell listener callback to update Launchpad LEDs on edit."""
        self._update_launchpad_leds()

    def _update_launchpad_leds(self) -> None:
        """
        Update Launchpad LEDs to reflect sequencer state.
        Uses LaunchpadController.set_pad_color() and LaunchpadColor enum.
        """
        if not self.launchpad:
            return

        # Map lane colors to LaunchpadColor
        lane_colors = [
            LaunchpadColor.RED,     # Kick
            LaunchpadColor.ORANGE,  # Snare
            LaunchpadColor.YELLOW,  # HiHat
            LaunchpadColor.GREEN,   # Clap
        ]

        beats_per_page = 8
        page_start = int(self.transport.playhead_beat // beats_per_page) * beats_per_page
        playhead_col = int(self.transport.playhead_beat) % beats_per_page

        for lane_idx, lane in enumerate(self.sequencer.lanes):
            if lane_idx >= 4:
                break
            for col in range(beats_per_page):
                beat = page_start + col
                has_event = any(int(e.beat) == beat for e in lane.events)
                is_playhead = col == playhead_col and self.transport.playing

                if has_event:
                    color = lane_colors[lane_idx]
                elif is_playhead:
                    color = LaunchpadColor.WHITE
                else:
                    color = LaunchpadColor.OFF

                self.launchpad.set_pad_color(lane_idx, col, color)

    # =========================================================================
    # RENDERING SETUP
    # =========================================================================

    def _build_quad_pipeline(self):
        self.quad_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            uniform vec2 u_offset;
            uniform vec2 u_size;
            uniform vec2 u_window;
            void main() {
                vec2 px = u_offset + in_pos * u_size;
                vec2 ndc = (px / u_window) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform vec4 u_color;
            out vec4 fragColor;
            void main() { fragColor = u_color; }
            """
        )
        quad_verts = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype="f4")
        self.quad_vbo = self.ctx.buffer(quad_verts)
        self.quad_vao = self.ctx.vertex_array(self.quad_prog, [(self.quad_vbo, "2f", "in_pos")])

    def _rect(self, x: float, y: float, w: float, h: float, color: tuple, win: tuple):
        self.quad_prog["u_offset"].value = (x, y)
        self.quad_prog["u_size"].value = (w, h)
        self.quad_prog["u_color"].value = color
        self.quad_prog["u_window"].value = win
        self.quad_vao.render()

    # =========================================================================
    # RENDER LOOP
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

        # Drain MIDI queue (from MidiManager thread)
        while True:
            try:
                note, vel = self._midi_queue.get_nowait()
                lane = note % len(self.sequencer.lanes)
                lane_obj = self.sequencer.lanes[lane]
                ev = self.sequencer.add_event(lane, self.transport.playhead_beat, vel, 0.25)
                if ev:
                    self.audio.trigger(lane_obj.sample_name, vel / 127.0)
                    print(f"[LP] {lane_obj.name} @ {ev.beat:.2f}")
            except queue.Empty:
                break

        # Drain output log queue if enabled
        if self.output_log:
            self.output_log.drain()

        # Update Launchpad LEDs (playhead moves)
        if self.launchpad and self.transport.playing:
            self._update_launchpad_leds()

        # Render
        w, h = self.wnd.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.07, 0.08, 0.10, 1.0)

        # Calculate layout based on enabled components
        layout = self._calculate_layout(w, h)

        self._draw_transport(w, h, layout)
        if self.waveform_data is not None:
            self._draw_waveform(w, h, layout)
        self._draw_sequencer(w, h, layout)
        if self.output_log and self.text_renderer:
            self._draw_output_panel(w, h, layout)

    def _calculate_layout(self, w: int, h: int) -> dict:
        """Calculate region bounds based on enabled components."""
        has_waveform = self.waveform_data is not None
        has_output = self.output_log is not None

        content_w = w - (RIGHT_PANEL_WIDTH if has_output else 0)
        waveform_h = WAVEFORM_HEIGHT if has_waveform else 0

        return {
            'content_width': content_w,
            'transport_y': h - TRANSPORT_HEIGHT,
            'waveform_y': h - TRANSPORT_HEIGHT - waveform_h,
            'waveform_h': waveform_h,
            'seq_height': h - TRANSPORT_HEIGHT - waveform_h,
            'output_x': content_w,
            'output_w': RIGHT_PANEL_WIDTH if has_output else 0,
        }

    # =========================================================================
    # RENDER: COMPONENTS
    # =========================================================================

    def _draw_transport(self, w: int, h: int, layout: dict):
        """Transport bar at top."""
        cw = layout['content_width']
        y = layout['transport_y']
        self._rect(0, y, cw, TRANSPORT_HEIGHT, (0.12, 0.13, 0.15, 1.0), (w, h))

        # Play/pause indicator
        play_color = (0.3, 0.85, 0.4, 1.0) if self.transport.playing else (0.85, 0.35, 0.35, 1.0)
        self._rect(16, y + 12, 32, 32, play_color, (w, h))

        # Beat indicators (4 boxes for 4/4)
        beat_in_bar = int(self.transport.playhead_beat) % 4
        for i in range(4):
            bx = 64 + i * 38
            active = i == beat_in_bar and self.transport.playing
            bc = (0.45, 0.65, 0.95, 1.0) if active else (0.22, 0.24, 0.28, 1.0)
            self._rect(bx, y + 12, 32, 32, bc, (w, h))

        # BPM display background
        self._rect(230, y + 14, 70, 28, (0.18, 0.19, 0.22, 1.0), (w, h))

        # Beat counter background
        self._rect(310, y + 14, 90, 28, (0.15, 0.16, 0.19, 1.0), (w, h))

    def _draw_waveform(self, w: int, h: int, layout: dict):
        """
        Draw waveform visualization using data from engine/audio/waveform.py.
        Renders as filled envelope (min/max peaks).
        """
        cw = layout['content_width']
        y = layout['waveform_y']
        wh = layout['waveform_h']

        # Background
        self._rect(0, y, cw, wh, (0.06, 0.07, 0.09, 1.0), (w, h))

        if self.waveform_data is None or len(self.waveform_data) == 0:
            return

        # Get visible beat range
        start_beat = max(0, self.time_camera._left_beat)
        end_beat = self.time_camera._left_beat + (cw / self.time_camera._px_per_beat)

        # Sample waveform data for visible range
        samples_per_beat = len(self.waveform_data) / 16  # Assuming 4 bars = 16 beats
        start_idx = int(start_beat * samples_per_beat)
        end_idx = int(end_beat * samples_per_beat)

        if start_idx >= len(self.waveform_data):
            return

        visible_data = self.waveform_data[start_idx:min(end_idx, len(self.waveform_data))]
        if len(visible_data) == 0:
            return

        # Downsample to pixels
        num_pixels = int(cw)
        step = max(1, len(visible_data) // num_pixels)

        center_y = y + wh / 2
        half_h = wh / 2 - 4

        for i in range(0, min(len(visible_data), num_pixels * step), step):
            chunk = visible_data[i:i + step]
            if len(chunk) == 0:
                continue
            min_v = float(np.min(chunk))
            max_v = float(np.max(chunk))

            px = (i // step)
            top = center_y + max_v * half_h
            bottom = center_y + min_v * half_h
            height = max(1, top - bottom)

            self._rect(px, bottom, 1, height, (0.3, 0.5, 0.8, 0.8), (w, h))

    def _draw_sequencer(self, w: int, h: int, layout: dict):
        """Sequencer grid with lanes."""
        cw = layout['content_width']
        seq_height = layout['seq_height']
        grid_x = LANE_LABEL_WIDTH
        grid_w = cw - LANE_LABEL_WIDTH

        # Background
        self._rect(0, 0, cw, seq_height, (0.06, 0.065, 0.075, 1.0), (w, h))

        # Lane labels background
        self._rect(0, 0, LANE_LABEL_WIDTH, seq_height, (0.09, 0.095, 0.11, 1.0), (w, h))

        # Beat grid lines
        for beat in self.time_camera.iter_beat_positions():
            px = self.time_camera.beat_to_px(beat) + grid_x
            if grid_x <= px <= cw:
                is_bar = int(beat) % 4 == 0
                color = (0.22, 0.23, 0.27, 1.0) if is_bar else (0.13, 0.14, 0.16, 1.0)
                line_w = 2 if is_bar else 1
                self._rect(px, 0, line_w, seq_height, color, (w, h))

        # Lanes
        num_lanes = len(self.sequencer.lanes)
        total_lane_height = num_lanes * (LANE_HEIGHT + LANE_GAP)
        lanes_start_y = (seq_height - total_lane_height) // 2

        for lane in self.sequencer.lanes:
            lane_y = lanes_start_y + lane.index * (LANE_HEIGHT + LANE_GAP)

            # Lane label bg
            label_color = (*lane.color[:3], 0.3)
            self._rect(4, lane_y, LANE_LABEL_WIDTH - 8, LANE_HEIGHT, label_color, (w, h))

            # Lane row bg
            self._rect(grid_x, lane_y, grid_w, LANE_HEIGHT, (0.08, 0.085, 0.10, 0.5), (w, h))

            # Events
            for event in lane.events:
                px = self.time_camera.beat_to_px(event.beat) + grid_x
                event_w = event.duration * self.time_camera._px_per_beat
                if px + event_w > grid_x and px < cw:
                    alpha = 0.6 + (event.velocity / 127.0) * 0.4
                    color = (*event.color[:3], alpha)
                    self._rect(max(grid_x, px), lane_y + 3, max(6, event_w - 2), LANE_HEIGHT - 6, color, (w, h))

        # Playhead
        px = self.time_camera.beat_to_px(self.transport.playhead_beat) + grid_x
        if grid_x <= px <= cw:
            self._rect(px - 1, 0, 3, seq_height, (1.0, 0.45, 0.25, 0.9), (w, h))

    def _draw_output_panel(self, w: int, h: int, layout: dict):
        """
        Draw output panel using engine/ws/response_log.py ResponseLog
        and engine/graph/text.py TextRenderer.
        """
        x = layout['output_x']
        panel_w = layout['output_w']

        # Background
        self._rect(x, 0, panel_w, h, (0.06, 0.05, 0.08, 1.0), (w, h))
        self._rect(x + 4, 4, panel_w - 8, h - 8, (0.08, 0.07, 0.10, 1.0), (w, h))

        if not self.text_renderer or not self.output_log:
            return

        font = self.text_renderer.default_font
        line_height = int(font.size * 1.2)
        margin = 8
        max_visible = max(1, (h - 2 * margin) // line_height)

        self.text_renderer.begin_frame()
        lines = self.output_log.get_lines()
        visible = lines[-max_visible:]

        for i, line in enumerate(visible):
            text_y = h - margin - (i + 1) * line_height
            self.text_renderer.draw_text(
                line[:200] if len(line) > 200 else line,
                x=margin,
                y=text_y,
                baseline=TextBaseline.TOP,
                color=(0.75, 0.78, 0.82, 1.0),
                max_width=panel_w - 16,
            )

        # Render with panel viewport
        prev_vp = self.ctx.viewport
        self.ctx.viewport = (int(x), 0, int(panel_w), h)
        self.text_renderer.render(panel_w, h)
        self.ctx.viewport = prev_vp

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
            state = "Playing" if self.transport.playing else "Paused"
            print(f"[>] {state}")
        elif key == self.wnd.keys.R:
            self.transport.stop()
            self.sequencer.reset_fired_flags()
            print("[>] Reset")
        elif key == self.wnd.keys.LEFT:
            self.transport.seek_by_bars(-1)
        elif key == self.wnd.keys.RIGHT:
            self.transport.seek_by_bars(1)

    def mouse_scroll_event(self, x_offset, y_offset):
        x, y = self.wnd.mouse
        self.time_camera.zoom(y_offset, x)

    def close(self):
        self.audio.stop()
        self._shutdown_output_panel()
        self._shutdown_websocket()


# =============================================================================
# HELPER: TEE STREAM (for stdout/stderr capture)
# =============================================================================

class _TeeStream:
    """File-like that writes to original stream and pushes lines to ResponseLog."""

    def __init__(self, stream, log: ResponseLog, prefix: str):
        self._stream = stream
        self._log = log
        self._prefix = prefix
        self._buf = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._stream.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                self._log.push_from_queue(f"{self._prefix} {line}")
        return len(data)

    def flush(self):
        self._stream.flush()
        if self._buf.strip():
            self._log.push_from_queue(f"{self._prefix} {self._buf.strip()}")
            self._buf = ""

    def __getattr__(self, name):
        return getattr(self._stream, name)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mglw.run_window_config(PlayableDemo)

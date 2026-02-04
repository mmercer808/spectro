"""
SPECTRO Display Demo

Layout:
- Left panel: Tracks view (SyncRegion, WaveformRegion, LanesRegion). Lanes show
  what's playing; Launchpad input is echoed into the sequencer as lane events.
- Right panel: Output echo — all output (stdout, stderr, WS) rendered here.
  The text renderer runs inside the right panel viewport only.

WebSocket client runs in a separate thread from the start.
Launchpad (or MIDI) input is echoed to the sequencer and shown as lanes on the left.

Run from project root:
    python demo/run_demo.py

Controls:
- SPACE: Play / Pause
- R: Reset to beat 0
- Left/Right: Seek by bar
- Scroll: Zoom timeline
- Launchpad pads: add events to lanes (row -> lane)
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is on path when run as python demo/run_demo.py
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))

import time
import queue
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import moderngl_window as mglw

from engine.time.transport import Transport, TimeSignature
from engine.time.camera import TimeCamera, TimeCameraMode
from engine.ws import ResponseLog, WSClient, DarkScheduler
from engine.graph.text import TextRenderer, TextBaseline
from engine.core.signal import SignalBridge, SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_CONNECTED, SIGNAL_MIDI_DISCONNECTED
from engine.midi.manager import MidiManager


# =============================================================================
# SEQUENCER DATA (display-only: empty lanes for layout)
# =============================================================================

@dataclass
class SequencerEvent:
    id: int
    beat: float
    lane: int
    duration: float
    velocity: int
    sample_name: str
    color: Tuple[float, float, float, float] = (1.0, 0.5, 0.2, 1.0)

    @property
    def end_beat(self) -> float:
        return self.beat + self.duration


class SequencerLane:
    def __init__(self, index: int, name: str, sample_name: str, color: tuple):
        self.index = index
        self.name = name
        self.sample_name = sample_name
        self.color = color
        self.events: List[SequencerEvent] = []


class Sequencer:
    def __init__(self):
        self.lanes: List[SequencerLane] = []
        defaults = [
            ("Kick", "kick", (1.0, 0.3, 0.2, 1.0)),
            ("Snare", "snare", (1.0, 0.6, 0.2, 1.0)),
            ("HiHat", "hihat", (1.0, 0.9, 0.2, 1.0)),
            ("Clap", "clap", (0.4, 1.0, 0.3, 1.0)),
        ]
        for i, (name, sample, color) in enumerate(defaults):
            self.lanes.append(SequencerLane(i, name, sample, color))


# =============================================================================
# LAYOUT CONSTANTS
# =============================================================================
# Left panel: tracks (sync + waveform + lanes). Right panel: output echo only.

SYNC_REGION_HEIGHT = 56       # Transport bar
WAVEFORM_REGION_HEIGHT = 80   # Waveform strip
LANES_TOP = SYNC_REGION_HEIGHT + WAVEFORM_REGION_HEIGHT
LANE_HEIGHT = 40
LANE_GAP = 5
RIGHT_PANEL_WIDTH = 320       # Right panel: echo ALL output (stdout, stderr, WS); renderer inside this panel
DEFAULT_BEAT_DURATION = 0.25  # One 16th at 120 BPM for pad hits


# =============================================================================
# STDOUT/STDERR TEE -> Output log (so right panel echoes everything)
# =============================================================================

class TeeOutput:
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
# DEMO APP
# =============================================================================

class SpectroDisplayDemo(mglw.WindowConfig):
    """SPECTRO main views only — no audio."""

    gl_version = (4, 3)  # 4.3 for graph TextRenderer (SSBO)
    title = "SPECTRO — Display Demo (no audio)"
    window_size = (1280, 720)
    resource_dir = "."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA

        self.transport = Transport(bpm=120.0)
        self.time_camera = TimeCamera(mode=TimeCameraMode.FOLLOW_PLAYHEAD)
        self.sequencer = Sequencer()

        self._build_quad_pipeline()
        self._text_renderer = TextRenderer(self.ctx)
        w, h = self.window_size
        self._content_width = w - RIGHT_PANEL_WIDTH
        self.time_camera.set_panel_size(float(self._content_width), float(h - SYNC_REGION_HEIGHT - WAVEFORM_REGION_HEIGHT))

        # --- Unified output log (right panel echoes ALL: stdout, stderr, WS) ---
        self.output_log = ResponseLog(max_lines=500)
        self.output_log.append("[SPECTRO] left = tracks view, right = output echo")
        # Tee stdout/stderr into output log so right panel shows everything
        self._real_stdout = sys.stdout
        self._real_stderr = sys.stderr
        sys.stdout = TeeOutput(sys.stdout, self.output_log, "[stdout]")
        sys.stderr = TeeOutput(sys.stderr, self.output_log, "[stderr]")

        # --- WebSocket: separate thread from the start ---
        self.ws_client = WSClient(response_log=self.output_log)
        self.dark_scheduler = DarkScheduler(self.ws_client, self.output_log)
        self.dark_scheduler.start()
        self.output_log.append("[WS] client ready; thread starts on connect")
        import os
        ws_url = os.environ.get("SPECTRO_WS_URL", "")
        if ws_url:
            self.ws_client.connect(ws_url)
        else:
            self.output_log.append("[WS] no SPECTRO_WS_URL; set to connect")

        # --- Launchpad (MIDI) -> sequencer: echo input to lanes; left panel shows what's playing ---
        self._midi_to_sequencer: queue.Queue = queue.Queue()
        self._next_event_id = 0
        self.signals = SignalBridge()
        self.signals.connect(SIGNAL_MIDI_NOTE_ON, self._on_midi_note_on)
        self.midi_manager = MidiManager(self.signals)
        if self.midi_manager.is_available:
            if self.midi_manager.connect("Launchpad"):
                self.output_log.append("[LP] Launchpad connected; pad hits -> sequencer lanes")
            else:
                self.output_log.append("[LP] no Launchpad found; connect one and restart")
        else:
            self.output_log.append("[LP] rtmidi not available; pip install python-rtmidi")

        self.last_time = time.perf_counter()
        self.frame_id = 0

        print("\n=== SPECTRO Display Demo ===")
        print("Left = tracks (lanes). Right = output echo. LP pads -> lanes.")
        print("SPACE = Play/Pause  R = Reset  Left/Right = Seek  Scroll = Zoom\n")

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
        quad_verts = np.array([
            0, 0,  1, 0,  1, 1,
            0, 0,  1, 1,  0, 1,
        ], dtype="f4")
        self.quad_vbo = self.ctx.buffer(quad_verts)
        self.quad_vao = self.ctx.vertex_array(
            self.quad_prog, [(self.quad_vbo, "2f", "in_pos")]
        )

    def _rect(self, x: float, y: float, w: float, h: float, color: tuple, win: tuple):
        self.quad_prog["u_offset"].value = (x, y)
        self.quad_prog["u_size"].value = (w, h)
        self.quad_prog["u_color"].value = color
        self.quad_prog["u_window"].value = win
        self.quad_vao.render()

    def _on_midi_note_on(self, note: int, velocity: int, channel: int):
        """Called from MIDI thread; enqueue for main thread to add to sequencer."""
        self._midi_to_sequencer.put((note, velocity))

    def on_render(self, t: float, frame_time: float):
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_time)
        self.last_time = now
        self.frame_id += 1

        self.transport.update(dt)
        self.time_camera.update(dt, self.transport.playhead_beat)

        w, h = self.wnd.size
        self._content_width = w - RIGHT_PANEL_WIDTH
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)

        # Drain output log queue (WS, tee) so new lines appear in log
        self.output_log.drain()

        # Drain MIDI -> sequencer: echo Launchpad (or any MIDI) into lanes (left panel)
        num_lanes = len(self.sequencer.lanes)
        while True:
            try:
                note, vel = self._midi_to_sequencer.get_nowait()
            except queue.Empty:
                break
            lane = note % num_lanes
            lane_obj = self.sequencer.lanes[lane]
            self._next_event_id += 1
            ev = SequencerEvent(
                id=self._next_event_id,
                beat=self.transport.playhead_beat,
                lane=lane,
                duration=DEFAULT_BEAT_DURATION,
                velocity=min(127, max(0, vel)),
                sample_name=lane_obj.sample_name,
                color=lane_obj.color,
            )
            lane_obj.events.append(ev)
            self.output_log.append(f"[LP] note {note} vel {vel} -> lane {lane} @ beat {ev.beat:.2f}")

        self._draw_sync_region(w, h)
        self._draw_waveform_region(w, h)
        self._draw_lanes_region(w, h)
        self._draw_playhead(w, h)
        self._draw_right_panel(w, h)

    def _draw_sync_region(self, w: int, h: int):
        """SyncRegion: transport bar at top."""
        bar_h = SYNC_REGION_HEIGHT
        self._rect(0, h - bar_h, w, bar_h, (0.12, 0.12, 0.14, 1.0), (w, h))

        play_color = (0.3, 0.8, 0.3, 1.0) if self.transport.playing else (0.8, 0.3, 0.3, 1.0)
        self._rect(20, h - bar_h + 12, 32, 32, play_color, (w, h))

        beat_in_bar = int(self.transport.playhead_beat) % 4
        for i in range(4):
            bx = 70 + i * 36
            bc = (0.4, 0.6, 0.9, 1.0) if i == beat_in_bar else (0.2, 0.25, 0.3, 1.0)
            self._rect(bx, h - bar_h + 12, 30, 30, bc, (w, h))

        bpm_y = h - bar_h + 18
        self._rect(220, bpm_y, 80, 20, (0.18, 0.2, 0.22, 1.0), (w, h))

    def _draw_waveform_region(self, w: int, h: int):
        """WaveformRegion: placeholder strip (waveform later)."""
        y = h - SYNC_REGION_HEIGHT - WAVEFORM_REGION_HEIGHT
        cw = self._content_width
        self._rect(0, y, cw, WAVEFORM_REGION_HEIGHT, (0.06, 0.07, 0.09, 1.0), (w, h))
        self._rect(10, y + 20, cw - 20, 40, (0.1, 0.12, 0.15, 0.6), (w, h))

    def _draw_lanes_region(self, w: int, h: int):
        """LanesRegion: timeline grid + lane rows."""
        timeline_y = 0
        timeline_h = h - SYNC_REGION_HEIGHT - WAVEFORM_REGION_HEIGHT
        cw = self._content_width

        self._rect(0, timeline_y, cw, timeline_h, (0.06, 0.07, 0.08, 1.0), (w, h))

        for beat in self.time_camera.iter_beat_positions():
            px = self.time_camera.beat_to_px(beat)
            if 0 <= px <= cw:
                is_bar = int(beat) % 4 == 0
                color = (0.25, 0.25, 0.3, 1.0) if is_bar else (0.15, 0.15, 0.18, 1.0)
                line_w = 2 if is_bar else 1
                self._rect(px, timeline_y, line_w, timeline_h, color, (w, h))

        for lane in self.sequencer.lanes:
            lane_y = timeline_y + lane.index * (LANE_HEIGHT + LANE_GAP) + 8
            self._rect(0, lane_y, cw, LANE_HEIGHT, (0.1, 0.1, 0.12, 0.5), (w, h))
            for event in lane.events:
                px = self.time_camera.beat_to_px(event.beat)
                event_w = event.duration * self.time_camera._px_per_beat
                if px + event_w > 0 and px < cw:
                    alpha = 0.5 + (event.velocity / 127.0) * 0.5
                    color = (*event.color[:3], alpha)
                    self._rect(px, lane_y + 2, max(4, event_w - 2), LANE_HEIGHT - 4, color, (w, h))

    def _draw_playhead(self, w: int, h: int):
        timeline_h = h - SYNC_REGION_HEIGHT - WAVEFORM_REGION_HEIGHT
        cw = self._content_width
        px = self.time_camera.beat_to_px(self.transport.playhead_beat)
        if 0 <= px <= cw:
            self._rect(px - 1, 0, 3, timeline_h, (1.0, 0.4, 0.2, 0.9), (w, h))

    def _draw_right_panel(self, w: int, h: int):
        """Right panel: echo ALL output. Renderer runs inside this panel viewport only."""
        x_left = w - RIGHT_PANEL_WIDTH
        # Background (drawn in full-window coords)
        self._rect(x_left, 0, RIGHT_PANEL_WIDTH, h, (0.06, 0.05, 0.08, 1.0), (w, h))
        self._rect(x_left + 4, 4, RIGHT_PANEL_WIDTH - 8, h - 8, (0.08, 0.07, 0.10, 1.0), (w, h))

        font = self._text_renderer.default_font
        line_height = int(font.size * 1.2)
        margin = 8
        max_width = RIGHT_PANEL_WIDTH - 16
        max_visible = max(1, (h - 2 * margin) // line_height)

        # Text in panel-relative coords so renderer (inside panel viewport) is correct
        self._text_renderer.begin_frame()
        lines = self.output_log.get_lines()
        visible = lines[-max_visible:]
        for i, line in enumerate(visible):
            y_top = h - margin - (i + 1) * line_height
            self._text_renderer.draw_text(
                line[:200] if len(line) > 200 else line,
                x=margin,
                y=y_top,
                baseline=TextBaseline.TOP,
                color=(0.75, 0.78, 0.82, 1.0),
                max_width=max_width,
            )
        # Renderer inside right panel only: viewport = panel, then restore
        prev_vp = self.ctx.viewport
        self.ctx.viewport = (x_left, 0, RIGHT_PANEL_WIDTH, h)
        self._text_renderer.render(RIGHT_PANEL_WIDTH, h)
        self.ctx.viewport = prev_vp

    def key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        if key == self.wnd.keys.SPACE:
            self.transport.toggle()
        elif key == self.wnd.keys.R:
            self.transport.stop()
        elif key == self.wnd.keys.LEFT:
            self.transport.seek_by_bars(-1)
        elif key == self.wnd.keys.RIGHT:
            self.transport.seek_by_bars(1)

    def mouse_scroll_event(self, x_offset, y_offset):
        x, y = self.wnd.mouse
        self.time_camera.zoom(y_offset, x)


if __name__ == "__main__":
    mglw.run_window_config(SpectroDisplayDemo)

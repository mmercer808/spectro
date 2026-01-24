# SPECTRO Playable Demo - Implementation Plan

## Reference
Based on: `concepts/sequencer_hybrid.html`

---

## Target Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ HEADER: SPECTRO                        ● MIDI Connected  [Pads View]    │
├─────────────────────────────────────────────────────────────────────────┤
│ TRANSPORT: [■][▶][●][⟲] | BPM 120 ═══○═══ | Loop 1:1→5:1 | A B C D | ⏱  │
├─────────────────────────────────────────────────────────────────────────┤
│ MASTER WAVEFORM (DJ-style output display)                               │
│ ▼ playhead                                                              │
│ ╭──────────────────────────────────────────────────────────────────────╮│
│ │  ≋≋≋≋≋≋≋≋█████████≋≋≋≋≋≋≋≋≋≋███≋≋≋≋≋≋≋≋≋≋≋≋████████≋≋≋≋≋≋≋≋≋≋≋≋≋≋  ││
│ ╰──────────────────────────────────────────────────────────────────────╯│
├──────────────┬──────────────────────────────────────────────────────────┤
│ TRACKS       │  |1   |2   |3   |4   |5   |6   |7   |8   |...           │
│              │  ····:····:····:····:····:····:····:····:               │
│ ▓ Kick    MS │  [==]         [==]         [==]         [==]            │
│ ▓ Snare   MS │            [==]                     [==]                │
│ ▓ Hi-Hat  MS │     [=] [=] [=] [=] [=] [=] [=] [=]                     │
│ ▓ Bass    MS │  [======]       [==]    [==========]                    │
│              │              ▼                                           │
│ [+ Add]      │           playhead                                       │
└──────────────┴──────────────────────────────────────────────────────────┘
```

---

## What Already Exists

| Component | File | Status |
|-----------|------|--------|
| Transport (play/pause/stop/BPM) | `engine/time/transport.py` | DONE |
| TimeCamera (beat↔pixel sync) | `engine/time/camera.py` | DONE |
| Widget base (events, layout) | `engine/ui/widget.py` | DONE |
| Button widget | `engine/ui/widgets/button.py` | DONE |
| Panel widget | `engine/ui/widgets/panel.py` | DONE |
| Container (flex layout) | `engine/ui/widgets/container.py` | DONE |
| DrawContext (rects, lines, text) | `engine/ui/draw.py` | DONE |
| SignalBridge (events) | `engine/core/signal.py` | DONE |
| MIDI Manager | `engine/midi/manager.py` | DONE |
| Launchpad Controller | `engine/midi/launchpad.py` | DONE |

---

## What Needs To Be Built

### New Widgets (6 total)

| Widget | Purpose |
|--------|---------|
| `TransportBar` | Play/Stop/Record/Loop buttons, BPM, Loop range, Pattern, Timecode |
| `Slider` | Horizontal/vertical draggable slider (for BPM, volume) |
| `Toggle` | On/off switch (for metronome, mute, solo) |
| `WaveformDisplay` | Master output waveform (DJ-style, scrolls with playhead) |
| `TrackList` | Left panel with track names, colors, M/S buttons |
| `SequencerGrid` | Main grid with note blocks, timeline, playhead |

---

## PHASE 1: Core Widgets

### 1.1 Slider Widget

**File:** `engine/ui/widgets/slider.py`

```python
class Slider(Widget):
    """Draggable slider for continuous values."""

    def __init__(
        self,
        min_value: float = 0.0,
        max_value: float = 1.0,
        value: float = 0.5,
        orientation: str = "horizontal",  # or "vertical"
        on_change: Callable[[float], None] = None
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.orientation = orientation
        self.on_change = on_change
        self._dragging = False

    def draw(self, ctx: DrawContext):
        # Track background
        ctx.draw_rect(self.rect, COLORS.SLIDER_TRACK, radius=2)

        # Fill
        fill_ratio = (self.value - self.min_value) / (self.max_value - self.min_value)
        if self.orientation == "horizontal":
            fill_width = self.rect.width * fill_ratio
            ctx.draw_rect(Rect(self.rect.x, self.rect.y, fill_width, self.rect.height),
                         COLORS.SLIDER_FILL, radius=2)
            # Thumb
            thumb_x = self.rect.x + fill_width - 7
            ctx.draw_circle(thumb_x + 7, self.rect.center_y, 7, COLORS.SLIDER_THUMB)

    def handle_pointer_down(self, event):
        self._dragging = True
        self._update_from_pointer(event.x, event.y)

    def handle_pointer_move(self, event):
        if self._dragging:
            self._update_from_pointer(event.x, event.y)
```

### 1.2 Toggle Widget

**File:** `engine/ui/widgets/toggle.py`

```python
class Toggle(Widget):
    """On/off toggle switch."""

    def __init__(self, value: bool = False, on_change: Callable[[bool], None] = None):
        self.value = value
        self.on_change = on_change

    def draw(self, ctx: DrawContext):
        bg_color = COLORS.TOGGLE_ON if self.value else COLORS.TOGGLE_OFF
        ctx.draw_rect(self.rect, bg_color, radius=self.rect.height / 2)

        # Knob position
        knob_x = self.rect.right - 9 if self.value else self.rect.x + 9
        ctx.draw_circle(knob_x, self.rect.center_y, 8, COLORS.TOGGLE_KNOB)

    def handle_pointer_down(self, event):
        self.value = not self.value
        if self.on_change:
            self.on_change(self.value)
```

---

## PHASE 2: Transport Bar

**File:** `engine/ui/widgets/transport_bar.py`

### Layout

```
┌────────────────────────────────────────────────────────────────────────┐
│ [■][▶][●][⟲] │ BPM 120 ═══○═══ │ Loop 1:1 → 5:1 [Set] │ A B C D │ ⏱   │
└────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class TransportBar(Widget):
    """Full transport control bar matching HTML mockup."""

    def __init__(self, transport: Transport, signals: SignalBridge):
        super().__init__()
        self.transport = transport
        self.signals = signals

        # Sub-widgets
        self.btn_stop = Button("■", on_click=self._stop)
        self.btn_play = Button("▶", on_click=self._toggle_play)
        self.btn_record = Button("●", on_click=self._toggle_record)
        self.btn_loop = Button("⟲", on_click=self._toggle_loop)

        self.bpm_slider = Slider(min_value=60, max_value=200, value=120,
                                  on_change=self._set_bpm)

        self.pattern_buttons = [Button(c) for c in "ABCD"]
        self.active_pattern = 0

        self.metro_toggle = Toggle(value=True)

        # State
        self.recording = False

    def draw(self, ctx: DrawContext):
        # Background
        ctx.draw_rect(self.rect, COLORS.TRANSPORT_BG)

        # Transport buttons group
        x = 20
        for btn in [self.btn_stop, self.btn_play, self.btn_record, self.btn_loop]:
            btn.rect = Rect(x, 8, 38, 38)
            btn.draw(ctx)
            x += 48

        # BPM section
        x += 20
        ctx.draw_text("BPM", x, 12, COLORS.LABEL, font_size=11)
        ctx.draw_text(f"{self.transport.bpm:.0f}", x, 28, COLORS.ACCENT, font_size=18)
        x += 60
        self.bpm_slider.rect = Rect(x, 22, 120, 10)
        self.bpm_slider.draw(ctx)
        x += 140

        # Loop section
        ctx.draw_text("Loop", x, 12, COLORS.LABEL, font_size=11)
        loop_text = f"{self.transport.format_bar_beat(self.transport.loop_start or 0)}"
        loop_text += f" → {self.transport.format_bar_beat(self.transport.loop_end or 16)}"
        ctx.draw_text(loop_text, x, 28, COLORS.TEXT, font_size=12)
        x += 120

        # Pattern buttons
        for i, btn in enumerate(self.pattern_buttons):
            btn.rect = Rect(x + i * 36, 12, 32, 32)
            btn.active = (i == self.active_pattern)
            btn.draw(ctx)
        x += 160

        # Metronome
        ctx.draw_text("Metro", x, 12, COLORS.LABEL, font_size=11)
        self.metro_toggle.rect = Rect(x, 28, 40, 20)
        self.metro_toggle.draw(ctx)
        x += 60

        # Timecode (right-aligned)
        timecode = self.transport.format_time()
        ctx.draw_text(timecode, self.rect.right - 100, 20, COLORS.ACCENT, font_size=16)
```

---

## PHASE 3: Master Waveform Display

**File:** `engine/ui/widgets/waveform_display.py`

This is the DJ-style waveform showing the mixed audio output.

```python
class WaveformDisplay(Widget):
    """
    Master waveform display - shows audio output like a DJ deck.
    Scrolls with playhead, shows amplitude over time.
    """

    def __init__(self, time_camera: TimeCamera, transport: Transport):
        super().__init__()
        self.time_camera = time_camera  # SHARED instance
        self.transport = transport

        # Waveform data: list of (beat, amplitude) or numpy array
        self.waveform_data: Optional[np.ndarray] = None
        self.samples_per_beat = 100  # Resolution

    def set_waveform(self, data: np.ndarray):
        """Set waveform data (amplitude values 0-1)."""
        self.waveform_data = data

    def draw(self, ctx: DrawContext):
        # Background
        ctx.draw_rect(self.rect, COLORS.WAVEFORM_BG, radius=4)

        if self.waveform_data is None:
            # Draw placeholder
            ctx.draw_text("No audio loaded", self.rect.center_x, self.rect.center_y,
                         COLORS.TEXT_DIM, align="center")
            return

        # Get visible beat range
        left_beat, right_beat = self.time_camera.get_visible_range()

        # Draw waveform
        center_y = self.rect.center_y
        half_height = self.rect.height * 0.4

        points = []
        for px in range(int(self.rect.width)):
            beat = self.time_camera.px_to_beat(self.rect.x + px)
            sample_idx = int(beat * self.samples_per_beat)

            if 0 <= sample_idx < len(self.waveform_data):
                amplitude = self.waveform_data[sample_idx]
            else:
                amplitude = 0

            y = center_y - amplitude * half_height
            points.append((self.rect.x + px, y))

        # Draw as filled area (mirrored)
        for px, (x, y_top) in enumerate(points):
            y_bottom = center_y + (center_y - y_top)
            ctx.draw_line(x, y_top, x, y_bottom, COLORS.WAVEFORM_FILL, width=1)

        # Draw playhead
        playhead_px = self.time_camera.beat_to_px(self.transport.playhead_beat)
        if self.rect.x <= playhead_px <= self.rect.right:
            ctx.draw_line(playhead_px, self.rect.y, playhead_px, self.rect.bottom,
                         COLORS.PLAYHEAD, width=2)

        # Draw beat grid
        for beat in self.time_camera.iter_bar_beats():
            px = self.time_camera.beat_to_px(beat)
            if self.rect.x <= px <= self.rect.right:
                ctx.draw_line(px, self.rect.y, px, self.rect.bottom,
                             COLORS.GRID_BAR, width=1)
```

---

## PHASE 4: Track List Panel

**File:** `engine/ui/widgets/track_list.py`

Left sidebar showing track names with mute/solo controls.

```python
@dataclass
class Track:
    """Track data model."""
    name: str
    color: tuple  # RGBA
    muted: bool = False
    solo: bool = False

class TrackList(Widget):
    """Left panel with track names and M/S controls."""

    TRACK_HEIGHT = 44

    def __init__(self, tracks: List[Track], on_track_change: Callable = None):
        super().__init__()
        self.tracks = tracks
        self.on_track_change = on_track_change

    def draw(self, ctx: DrawContext):
        # Header with zoom buttons
        ctx.draw_rect(Rect(self.rect.x, self.rect.y, self.rect.width, 50),
                     COLORS.PANEL_HEADER)
        ctx.draw_text("−", self.rect.x + 50, self.rect.y + 25, COLORS.TEXT)
        ctx.draw_text("+", self.rect.x + 90, self.rect.y + 25, COLORS.TEXT)

        # Track rows
        y = self.rect.y + 50
        for track in self.tracks:
            self._draw_track_row(ctx, track, y)
            y += self.TRACK_HEIGHT

    def _draw_track_row(self, ctx: DrawContext, track: Track, y: float):
        # Color bar
        ctx.draw_rect(Rect(self.rect.x + 12, y + 8, 4, 28), track.color, radius=2)

        # Track name
        ctx.draw_text(track.name, self.rect.x + 24, y + 16, COLORS.TEXT, font_size=13)

        # Mute button
        m_color = COLORS.MUTE_ACTIVE if track.muted else COLORS.BUTTON_INACTIVE
        ctx.draw_rect(Rect(self.rect.x + 100, y + 10, 24, 24), m_color, radius=3)
        ctx.draw_text("M", self.rect.x + 107, y + 16, COLORS.TEXT, font_size=10)

        # Solo button
        s_color = COLORS.SOLO_ACTIVE if track.solo else COLORS.BUTTON_INACTIVE
        ctx.draw_rect(Rect(self.rect.x + 128, y + 10, 24, 24), s_color, radius=3)
        ctx.draw_text("S", self.rect.x + 135, y + 16, COLORS.TEXT, font_size=10)
```

---

## PHASE 5: Sequencer Grid

**File:** `engine/ui/widgets/sequencer_grid.py`

Main grid area with timeline, note blocks, and playhead.

```python
@dataclass
class NoteBlock:
    """A note/event in the sequencer."""
    track_index: int
    start_beat: float
    end_beat: float
    velocity: float = 1.0

class SequencerGrid(Widget):
    """Main sequencer grid with timeline and note blocks."""

    TRACK_HEIGHT = 44
    TIMELINE_HEIGHT = 50

    def __init__(self, time_camera: TimeCamera, transport: Transport, tracks: List[Track]):
        super().__init__()
        self.time_camera = time_camera  # SHARED
        self.transport = transport
        self.tracks = tracks
        self.notes: List[NoteBlock] = []

    def draw(self, ctx: DrawContext):
        self._draw_timeline(ctx)
        self._draw_grid(ctx)
        self._draw_notes(ctx)
        self._draw_playhead(ctx)
        self._draw_loop_markers(ctx)

    def _draw_timeline(self, ctx: DrawContext):
        """Draw beat/bar markers at top."""
        y = self.rect.y
        ctx.draw_rect(Rect(self.rect.x, y, self.rect.width, self.TIMELINE_HEIGHT),
                     COLORS.TIMELINE_BG)

        # Beat numbers and tick marks
        for beat in self.time_camera.iter_beat_positions():
            px = self.time_camera.beat_to_px(beat)
            if not (self.rect.x <= px <= self.rect.right):
                continue

            beat_num = int(beat) + 1
            is_bar = (beat % 4 == 0)

            # Beat number
            color = COLORS.TEXT if is_bar else COLORS.TEXT_DIM
            ctx.draw_text(str(beat_num), px, y + 12, color, font_size=12)

            # Tick marks
            tick_height = 10 if is_bar else 5
            ctx.draw_line(px, y + 35, px, y + 35 + tick_height, COLORS.GRID_TICK)

    def _draw_grid(self, ctx: DrawContext):
        """Draw grid lines and track backgrounds."""
        y = self.rect.y + self.TIMELINE_HEIGHT

        # Track rows
        for i, track in enumerate(self.tracks):
            row_y = y + i * self.TRACK_HEIGHT
            # Alternating background
            bg = COLORS.GRID_ROW_ALT if i % 2 else COLORS.GRID_ROW
            ctx.draw_rect(Rect(self.rect.x, row_y, self.rect.width, self.TRACK_HEIGHT), bg)

        # Vertical grid lines
        for beat in self.time_camera.iter_beat_positions():
            px = self.time_camera.beat_to_px(beat)
            if not (self.rect.x <= px <= self.rect.right):
                continue

            is_bar = (beat % 4 == 0)
            color = COLORS.GRID_BAR if is_bar else COLORS.GRID_BEAT
            ctx.draw_line(px, y, px, y + len(self.tracks) * self.TRACK_HEIGHT, color)

    def _draw_notes(self, ctx: DrawContext):
        """Draw note blocks."""
        y_base = self.rect.y + self.TIMELINE_HEIGHT

        for note in self.notes:
            if not self.time_camera.is_range_visible(note.start_beat, note.end_beat):
                continue

            track = self.tracks[note.track_index]
            y = y_base + note.track_index * self.TRACK_HEIGHT + 2

            x0 = self.time_camera.beat_to_px(note.start_beat)
            x1 = self.time_camera.beat_to_px(note.end_beat)
            width = max(x1 - x0, 4)  # Minimum width

            # Note block with track color border
            ctx.draw_rect(Rect(x0, y, width, self.TRACK_HEIGHT - 4),
                         COLORS.NOTE_BG, radius=4)
            ctx.draw_rect_outline(Rect(x0, y, width, self.TRACK_HEIGHT - 4),
                                  track.color, width=2, radius=4)

            # Mini waveform inside (decorative)
            # ... draw waveform pattern

    def _draw_playhead(self, ctx: DrawContext):
        """Draw playhead line."""
        px = self.time_camera.beat_to_px(self.transport.playhead_beat)
        if self.rect.x <= px <= self.rect.right:
            # Triangle at top
            ctx.draw_rect(Rect(px - 6, self.rect.y, 12, 10), COLORS.PLAYHEAD, radius=0)
            # Vertical line
            ctx.draw_line(px, self.rect.y, px, self.rect.bottom, COLORS.PLAYHEAD, width=2)

    def _draw_loop_markers(self, ctx: DrawContext):
        """Draw loop start/end markers."""
        if self.transport.loop_start is not None:
            px = self.time_camera.beat_to_px(self.transport.loop_start)
            ctx.draw_line(px, self.rect.y, px, self.rect.bottom, COLORS.LOOP_MARKER, width=2)

        if self.transport.loop_end is not None:
            px = self.time_camera.beat_to_px(self.transport.loop_end)
            ctx.draw_line(px, self.rect.y, px, self.rect.bottom, COLORS.LOOP_MARKER, width=2)

    # Mouse handlers for scrolling/zooming
    def handle_scroll(self, event):
        if event.ctrl:
            # Zoom
            self.time_camera.zoom(event.delta_y * 0.1, anchor_px=event.x)
        else:
            # Pan
            self.time_camera.scroll_by_px(event.delta_x * 20)
```

---

## PHASE 6: Main Demo Application

**File:** `playable_demo.py`

```python
"""
SPECTRO Playable Demo
Run: python playable_demo.py
"""

import moderngl_window as mglw
from engine.core.signal import SignalBridge
from engine.time.transport import Transport
from engine.time.camera import TimeCamera
from engine.midi import MidiManager, LaunchpadController
from engine.ui.widgets.transport_bar import TransportBar
from engine.ui.widgets.waveform_display import WaveformDisplay
from engine.ui.widgets.track_list import TrackList, Track
from engine.ui.widgets.sequencer_grid import SequencerGrid, NoteBlock


class PlayableDemo(mglw.WindowConfig):
    title = "SPECTRO"
    window_size = (1400, 800)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Core systems
        self.signals = SignalBridge()
        self.transport = Transport(bpm=120)
        self.time_camera = TimeCamera(width=1200)  # SHARED

        # MIDI (optional)
        self.midi = MidiManager(self.signals)
        if self.midi.list_input_ports():
            self.midi.connect("Launchpad")
            self.launchpad = LaunchpadController(self.midi, self.signals)
            self.launchpad.bind_transport(self.transport)

        # Tracks (4 initial)
        self.tracks = [
            Track("Kick", (1.0, 0.29, 0.17, 1.0)),    # #ff4b2b
            Track("Snare", (0.22, 0.94, 0.49, 1.0)),  # #38ef7d
            Track("Hi-Hat", (0.0, 0.95, 1.0, 1.0)),   # #00f2fe
            Track("Bass", (0.29, 0.0, 0.88, 1.0)),    # #4a00e0
        ]

        # Demo notes
        self.notes = self._create_demo_pattern()

        # UI Widgets
        self.transport_bar = TransportBar(self.transport, self.signals)
        self.waveform_display = WaveformDisplay(self.time_camera, self.transport)
        self.track_list = TrackList(self.tracks)
        self.sequencer = SequencerGrid(self.time_camera, self.transport, self.tracks)
        self.sequencer.notes = self.notes

        # Generate fake waveform for demo
        self._generate_demo_waveform()

    def _create_demo_pattern(self) -> List[NoteBlock]:
        """Create a simple 4-bar pattern."""
        notes = []

        # Kick on 1, 5, 9, 13 (every bar)
        for bar in range(4):
            beat = bar * 4
            notes.append(NoteBlock(0, beat, beat + 0.5))

        # Snare on 5, 13 (beats 2 and 4 of 4-bar phrase)
        notes.append(NoteBlock(1, 4, 4.5))
        notes.append(NoteBlock(1, 12, 12.5))

        # Hi-hats on off-beats
        for beat in range(16):
            if beat % 2 == 1:
                notes.append(NoteBlock(2, beat, beat + 0.25))

        # Bass - longer notes
        notes.append(NoteBlock(3, 0, 2))
        notes.append(NoteBlock(3, 4, 5))
        notes.append(NoteBlock(3, 8, 12))

        return notes

    def _generate_demo_waveform(self):
        """Generate fake waveform data for demo."""
        import numpy as np
        samples = 16 * 100  # 16 beats * 100 samples/beat
        t = np.linspace(0, 16, samples)
        # Mix of frequencies to simulate music
        wave = (np.sin(t * 2 * np.pi * 2) * 0.3 +
                np.sin(t * 2 * np.pi * 4) * 0.2 +
                np.random.random(samples) * 0.1)
        wave = np.abs(wave)
        wave = wave / wave.max()
        self.waveform_display.set_waveform(wave)

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.04, 0.04, 0.04)

        # Update transport
        state = self.transport.update(frame_time)

        # Update time camera to follow playhead
        if self.transport.playing:
            self.time_camera.update(state.playhead_beat)

        # Layout
        w, h = self.window_size

        # Transport bar: top 56px
        self.transport_bar.rect = Rect(0, 0, w, 56)
        self.transport_bar.draw(self.draw_ctx)

        # Waveform display: below transport, 80px tall
        self.waveform_display.rect = Rect(0, 56, w, 80)
        self.waveform_display.draw(self.draw_ctx)

        # Track list: left side, 180px wide
        track_top = 136
        self.track_list.rect = Rect(0, track_top, 180, h - track_top)
        self.track_list.draw(self.draw_ctx)

        # Sequencer: rest of space
        self.sequencer.rect = Rect(180, track_top, w - 180, h - track_top)
        self.sequencer.draw(self.draw_ctx)

        # Render all UI
        self.ui_renderer.render(self.draw_batch)

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return

        if key == self.wnd.keys.SPACE:
            self.transport.toggle()
        elif key == self.wnd.keys.HOME:
            self.transport.seek_to_beat(0)
        elif key == self.wnd.keys.LEFT:
            self.transport.seek_by_bars(-1)
        elif key == self.wnd.keys.RIGHT:
            self.transport.seek_by_bars(1)
        elif key == self.wnd.keys.UP:
            self.transport.set_bpm(self.transport.bpm + 5)
        elif key == self.wnd.keys.DOWN:
            self.transport.set_bpm(self.transport.bpm - 5)


if __name__ == "__main__":
    PlayableDemo.run()
```

---

## Color Palette

```python
# colors.py
class COLORS:
    # Backgrounds
    BG_DARK = (0.04, 0.04, 0.04, 1.0)
    PANEL_BG = (0.1, 0.1, 0.1, 1.0)
    PANEL_HEADER = (0.12, 0.12, 0.12, 1.0)
    TRANSPORT_BG = (0.1, 0.1, 0.1, 1.0)
    TIMELINE_BG = (0.1, 0.1, 0.1, 1.0)
    WAVEFORM_BG = (0.08, 0.08, 0.08, 1.0)

    # Grid
    GRID_ROW = (0.09, 0.09, 0.09, 1.0)
    GRID_ROW_ALT = (0.08, 0.08, 0.08, 1.0)
    GRID_BEAT = (0.15, 0.15, 0.15, 1.0)
    GRID_BAR = (0.17, 0.17, 0.17, 1.0)
    GRID_TICK = (0.2, 0.2, 0.2, 1.0)

    # Playhead & Markers
    PLAYHEAD = (0.0, 1.0, 0.53, 1.0)  # #00ff88
    LOOP_MARKER = (1.0, 1.0, 1.0, 0.6)

    # Text
    TEXT = (0.88, 0.88, 0.88, 1.0)
    TEXT_DIM = (0.4, 0.4, 0.4, 1.0)
    LABEL = (0.4, 0.4, 0.4, 1.0)
    ACCENT = (0.0, 0.83, 1.0, 1.0)  # #00d4ff

    # Buttons
    BUTTON_BG = (0.17, 0.17, 0.17, 1.0)
    BUTTON_HOVER = (0.23, 0.23, 0.23, 1.0)
    BUTTON_ACTIVE = (0.0, 0.83, 1.0, 1.0)
    BUTTON_INACTIVE = (0.23, 0.23, 0.23, 1.0)

    # Mute/Solo
    MUTE_ACTIVE = (0.0, 0.83, 1.0, 1.0)
    SOLO_ACTIVE = (0.0, 0.83, 1.0, 1.0)

    # Notes
    NOTE_BG = (0.17, 0.17, 0.17, 1.0)

    # Waveform
    WAVEFORM_FILL = (0.0, 0.83, 1.0, 0.6)

    # Slider
    SLIDER_TRACK = (0.17, 0.17, 0.17, 1.0)
    SLIDER_FILL = (0.0, 0.83, 1.0, 1.0)
    SLIDER_THUMB = (0.0, 0.83, 1.0, 1.0)

    # Toggle
    TOGGLE_ON = (0.0, 0.83, 1.0, 1.0)
    TOGGLE_OFF = (0.17, 0.17, 0.17, 1.0)
    TOGGLE_KNOB = (0.04, 0.04, 0.04, 1.0)

    # Track colors (from mockup)
    TRACK_KICK = (1.0, 0.29, 0.17, 1.0)   # #ff4b2b
    TRACK_SNARE = (0.22, 0.94, 0.49, 1.0) # #38ef7d
    TRACK_HIHAT = (0.0, 0.95, 1.0, 1.0)   # #00f2fe
    TRACK_BASS = (0.29, 0.0, 0.88, 1.0)   # #4a00e0
```

---

## Cursor AI Instructions

Copy this to Cursor:

```
You are implementing a playable demo for SPECTRO based on the HTML mockup in concepts/sequencer_hybrid.html.

READ THESE FILES FIRST:
- PLAYABLE_DEMO_PLAN.md (this file) - Full implementation spec
- concepts/sequencer_hybrid.html - Visual reference
- engine/time/transport.py - Transport API
- engine/time/camera.py - TimeCamera API (CRITICAL)
- engine/ui/widget.py - Base Widget class

IMPLEMENTATION ORDER:

1. Create engine/ui/widgets/slider.py
   - Horizontal slider with draggable thumb
   - on_change callback

2. Create engine/ui/widgets/toggle.py
   - Boolean toggle switch

3. Create engine/ui/widgets/transport_bar.py
   - Matches HTML mockup layout exactly
   - Uses Slider for BPM
   - Uses Toggle for metronome

4. Create engine/ui/widgets/waveform_display.py
   - DJ-style waveform display
   - Scrolls with TimeCamera (SHARED instance)
   - Shows playhead

5. Create engine/ui/widgets/track_list.py
   - Track dataclass
   - Left panel with track names, colors, M/S buttons

6. Create engine/ui/widgets/sequencer_grid.py
   - NoteBlock dataclass
   - Timeline at top
   - Grid with note blocks
   - Playhead and loop markers
   - Mouse scroll = pan, Ctrl+scroll = zoom

7. Create playable_demo.py
   - Wire everything together
   - 4 tracks: Kick, Snare, Hi-Hat, Bass
   - Demo pattern
   - Keyboard shortcuts: Space, Home, Left/Right, Up/Down

CRITICAL RULES:
- ALL time-synced widgets share ONE TimeCamera instance
- Use Transport callbacks for state sync
- Match HTML mockup colors (#00d4ff accent, #00ff88 playhead)
- 4 tracks initially, not 8
```

---

## Summary

| Phase | Widget | Purpose |
|-------|--------|---------|
| 1 | Slider, Toggle | Basic input widgets |
| 2 | TransportBar | Play/Stop/BPM/Loop controls |
| 3 | WaveformDisplay | Master DJ-style waveform output |
| 4 | TrackList | Left panel with M/S buttons |
| 5 | SequencerGrid | Main grid with notes |
| 6 | playable_demo.py | Wire everything together |

Total: **6 new widget files + 1 main application file**

# SPECTRO Engine - TODO List

## Overview

The engine needs these systems to support the playable demo:

1. **Audio Playback** - Play WAV/MP3, get waveform data, output to speakers
2. **Live MIDI Recording** - Capture Launchpad hits with timestamps
3. **Waveform Rendering** - Draw audio waveform scrolling with playhead
4. **Sample Triggering** - Play drum samples on beat

---

## Priority 1: Audio System

### What's Missing

There is NO audio playback in the engine. Need:

```
engine/audio/
├── __init__.py
├── device.py       # Audio device enumeration and output
├── player.py       # Audio file playback with position sync
├── waveform.py     # Extract waveform data from audio
├── sampler.py      # Trigger one-shot samples (drums)
└── mixer.py        # Mix multiple audio sources
```

### Required Functionality

```python
class AudioPlayer:
    """Play audio files synchronized to Transport."""

    def load(self, path: str) -> bool:
        """Load WAV/MP3 file."""
        pass

    def play(self):
        """Start playback from current transport position."""
        pass

    def stop(self):
        """Stop playback."""
        pass

    def get_waveform(self, samples_per_beat: int = 100) -> np.ndarray:
        """Get amplitude data for visualization."""
        pass

    def get_position_beats(self) -> float:
        """Current playback position in beats (synced to Transport)."""
        pass


class Sampler:
    """Trigger one-shot samples."""

    def load_sample(self, name: str, path: str):
        """Load a sample into a slot."""
        pass

    def trigger(self, name: str, velocity: float = 1.0):
        """Play a sample immediately."""
        pass


class AudioMixer:
    """Mix audio player + sampler output."""

    def add_source(self, source: AudioSource):
        pass

    def get_output_waveform(self) -> np.ndarray:
        """Get mixed output for visualization."""
        pass
```

### Recommended Libraries

- **miniaudio** - Simple, cross-platform audio playback
- **sounddevice** - NumPy-friendly audio I/O
- **pydub** - Audio file loading (WAV, MP3)

### Install

```bash
pip install miniaudio sounddevice pydub numpy
```

---

## Priority 2: Live MIDI Recording

### What Exists

- `engine/midi/manager.py` - MidiManager receives MIDI, emits signals
- `engine/midi/launchpad.py` - LaunchpadController maps pads to grid

### What's Missing

Need to **record** MIDI events with beat-accurate timestamps:

```python
# engine/midi/recorder.py

@dataclass
class RecordedNote:
    """A MIDI note captured during recording."""
    beat: float          # When it was played (in beats)
    note: int            # MIDI note number
    velocity: int        # 0-127
    channel: int
    duration: float = 0  # Set on note-off

    # For display
    row: int = 0         # Launchpad row
    col: int = 0         # Launchpad col
    color: tuple = None  # Track color


class MidiRecorder:
    """Records MIDI input with beat-accurate timestamps."""

    def __init__(self, transport: Transport, signals: SignalBridge):
        self.transport = transport
        self.signals = signals
        self.recording = False
        self.notes: List[RecordedNote] = []
        self._active_notes: Dict[int, RecordedNote] = {}  # note -> RecordedNote

        # Subscribe to MIDI signals
        signals.connect(SIGNAL_MIDI_NOTE_ON, self._on_note_on)
        signals.connect(SIGNAL_MIDI_NOTE_OFF, self._on_note_off)
        signals.connect(SIGNAL_MIDI_PAD, self._on_pad)

    def start_recording(self):
        """Start capturing MIDI."""
        self.recording = True
        self.notes.clear()

    def stop_recording(self):
        """Stop capturing."""
        self.recording = False

    def _on_note_on(self, note: int, velocity: int, channel: int):
        if not self.recording:
            return

        beat = self.transport.playhead_beat
        recorded = RecordedNote(
            beat=beat,
            note=note,
            velocity=velocity,
            channel=channel
        )
        self.notes.append(recorded)
        self._active_notes[note] = recorded

    def _on_note_off(self, note: int, velocity: int, channel: int):
        if note in self._active_notes:
            recorded = self._active_notes.pop(note)
            recorded.duration = self.transport.playhead_beat - recorded.beat

    def _on_pad(self, row: int, col: int, velocity: int):
        """Record Launchpad pads with grid position."""
        if not self.recording or velocity == 0:
            return

        beat = self.transport.playhead_beat
        recorded = RecordedNote(
            beat=beat,
            note=row * 10 + col,  # Encode grid position
            velocity=velocity,
            channel=0,
            row=row,
            col=col,
            color=LAUNCHPAD_COLORS.get(row, (0.5, 0.5, 0.5, 1.0))
        )
        self.notes.append(recorded)

    def get_notes_in_range(self, start_beat: float, end_beat: float) -> List[RecordedNote]:
        """Get notes visible in beat range."""
        return [n for n in self.notes if start_beat <= n.beat <= end_beat]
```

---

## Priority 3: Sequencer Display Updates

### Current State

`SequencerGrid` in the plan draws pre-defined `NoteBlock` objects.

### Needed Changes

1. **Accept live notes** from MidiRecorder
2. **Separate tracks** for audio vs Launchpad input
3. **Auto-scroll** with playhead (waveform moves left, playhead stays fixed)

```python
class SequencerGrid(Widget):
    """Updated to show live MIDI recording."""

    def __init__(self, time_camera: TimeCamera, transport: Transport,
                 midi_recorder: MidiRecorder):
        self.time_camera = time_camera
        self.transport = transport
        self.midi_recorder = midi_recorder

        # Track definitions
        self.audio_track = Track("Audio", (0.0, 0.83, 1.0, 1.0))  # Cyan
        self.launchpad_track = Track("Launchpad", (1.0, 0.5, 0.0, 1.0))  # Orange

    def draw(self, ctx: DrawContext):
        self._draw_timeline(ctx)
        self._draw_grid(ctx)

        # Draw audio track events (pre-programmed)
        self._draw_audio_events(ctx)

        # Draw live Launchpad notes from recorder
        self._draw_live_notes(ctx)

        self._draw_playhead(ctx)

    def _draw_live_notes(self, ctx: DrawContext):
        """Draw notes from MidiRecorder."""
        left, right = self.time_camera.get_visible_range()
        notes = self.midi_recorder.get_notes_in_range(left, right)

        for note in notes:
            x = self.time_camera.beat_to_px(note.beat)
            y = self._get_launchpad_track_y()

            # Small colored block for each hit
            color = note.color or self.launchpad_track.color
            ctx.draw_rect(Rect(x - 4, y + 4, 8, 36), color, radius=2)
```

---

## Priority 4: Waveform Display Mode

### Scrolling Behavior

Two modes:

1. **Playhead moves** (current plan) - Waveform static, playhead moves right
2. **Waveform scrolls** (DJ style) - Playhead fixed center, waveform moves left

For the demo, you want **mode 2**:

```python
class WaveformDisplay(Widget):
    """DJ-style waveform - playhead fixed, waveform scrolls left."""

    PLAYHEAD_POSITION = 0.3  # Playhead at 30% from left

    def draw(self, ctx: DrawContext):
        ctx.draw_rect(self.rect, COLORS.WAVEFORM_BG, radius=4)

        if self.waveform_data is None:
            return

        # Playhead is FIXED at 30% of width
        playhead_x = self.rect.x + self.rect.width * self.PLAYHEAD_POSITION

        # Calculate which beats are visible based on playhead position
        current_beat = self.transport.playhead_beat
        px_per_beat = 50  # Zoom level

        # Draw waveform relative to fixed playhead
        for px in range(int(self.rect.width)):
            # Convert pixel to beat (playhead is at current_beat)
            beat_offset = (px - (self.rect.width * self.PLAYHEAD_POSITION)) / px_per_beat
            beat = current_beat + beat_offset

            # Get amplitude at this beat
            sample_idx = int(beat * self.samples_per_beat)
            if 0 <= sample_idx < len(self.waveform_data):
                amplitude = self.waveform_data[sample_idx]
            else:
                amplitude = 0

            # Draw vertical line
            y_top = self.rect.center_y - amplitude * self.rect.height * 0.4
            y_bot = self.rect.center_y + amplitude * self.rect.height * 0.4

            # Color: past = dimmer, future = brighter
            if px < playhead_x - self.rect.x:
                color = COLORS.WAVEFORM_PAST
            else:
                color = COLORS.WAVEFORM_FUTURE

            ctx.draw_line(self.rect.x + px, y_top, self.rect.x + px, y_bot, color)

        # Draw fixed playhead line
        ctx.draw_line(playhead_x, self.rect.y, playhead_x, self.rect.bottom,
                     COLORS.PLAYHEAD, width=2)
```

---

## File Structure Summary

```
engine/
├── audio/                    # NEW - Audio system
│   ├── __init__.py
│   ├── device.py            # Audio output device
│   ├── player.py            # WAV/MP3 playback synced to Transport
│   ├── waveform.py          # Extract waveform from audio
│   ├── sampler.py           # One-shot sample triggering
│   └── mixer.py             # Mix sources
│
├── midi/                     # EXISTS - needs additions
│   ├── manager.py           # EXISTS
│   ├── launchpad.py         # EXISTS
│   └── recorder.py          # NEW - Live MIDI recording
│
├── time/                     # EXISTS
│   ├── transport.py         # EXISTS
│   └── camera.py            # EXISTS
│
└── core/                     # EXISTS
    └── signal.py            # EXISTS
```

---

## Implementation Order for AI

### Phase 1: Audio Foundation
1. Create `engine/audio/device.py` - Audio output using sounddevice
2. Create `engine/audio/player.py` - Load and play audio files
3. Create `engine/audio/waveform.py` - Extract waveform data
4. Wire player to Transport for sync

### Phase 2: MIDI Recording
5. Create `engine/midi/recorder.py` - MidiRecorder class
6. Update `engine/midi/launchpad.py` - Add recording callbacks
7. Test: play Launchpad, see notes captured with timestamps

### Phase 3: Sample Playback
8. Create `engine/audio/sampler.py` - Trigger drum samples
9. Load kick/snare/hihat samples
10. Wire Transport beat callbacks to trigger samples

### Phase 4: Integration
11. Update WaveformDisplay for DJ-style scrolling
12. Update SequencerGrid to show live notes
13. Wire everything in playable_demo.py

---

## Dependencies to Add

```bash
pip install sounddevice numpy pydub
```

Or add to requirements.txt:
```
sounddevice>=0.4.6
numpy>=1.24.0
pydub>=0.25.1
python-rtmidi>=1.5.0
```

---

## Quick Test Script

```python
# test_audio.py
import sounddevice as sd
import numpy as np

# Generate a simple kick drum sound
duration = 0.1
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration))
freq = 60 * np.exp(-t * 30)  # Pitch drops
kick = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)
kick = (kick * 0.5).astype(np.float32)

print("Playing kick...")
sd.play(kick, sample_rate)
sd.wait()
print("Done!")
```

---

## Notes for AI Implementation

1. **Transport is the master clock** - Audio player must sync to `Transport.playhead_beat`
2. **SignalBridge is the event bus** - All MIDI events go through it
3. **TimeCamera is shared** - Waveform and Sequencer use same instance
4. **Thread safety** - Audio runs on separate thread, use queues for communication
5. **Latency** - Keep audio buffer small (256-512 samples) for responsive playback

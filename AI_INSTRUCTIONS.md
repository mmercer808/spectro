# Instructions for AI Assistant - Engine Improvements

## Your Role

You are improving the base engine for SPECTRO, a real-time audio visualization and MIDI sequencer. Focus on **engine systems**, not UI widgets.

---

## Context Files to Read First

Read these files to understand the architecture:

1. `CLAUDE.md` - Project overview and conventions
2. `ENGINE_TODO.md` - Detailed todo list with code snippets
3. `engine/core/signal.py` - Event system (SignalBridge)
4. `engine/time/transport.py` - Playback control (Transport)
5. `engine/midi/manager.py` - MIDI input handling

---

## What Exists

| System | Status | Files |
|--------|--------|-------|
| Transport | DONE | `engine/time/transport.py` |
| TimeCamera | DONE | `engine/time/camera.py` |
| SignalBridge | DONE | `engine/core/signal.py` |
| MIDI Input | DONE | `engine/midi/manager.py` |
| Launchpad | DONE | `engine/midi/launchpad.py` |
| Audio | NOT STARTED | needs `engine/audio/` |
| MIDI Recording | NOT STARTED | needs `engine/midi/recorder.py` |

---

## Your Tasks (Priority Order)

### Task 1: Audio Player

Create `engine/audio/player.py`:

```python
class AudioPlayer:
    """Play audio files synchronized to Transport."""

    def __init__(self, transport: Transport):
        self.transport = transport
        self.audio_data: np.ndarray = None
        self.sample_rate: int = 44100
        self.playing = False

    def load(self, path: str) -> bool:
        """Load WAV file and extract waveform."""
        pass

    def play(self):
        """Start playback at current transport position."""
        pass

    def stop(self):
        """Stop playback."""
        pass

    def get_waveform(self, samples_per_beat: int = 100) -> np.ndarray:
        """Get amplitude envelope for visualization."""
        pass
```

**Requirements:**
- Use `sounddevice` for audio output
- Sync playback position to `transport.playhead_beat`
- Extract waveform data for visualization
- Handle Transport callbacks (play/pause/stop/seek)

### Task 2: MIDI Recorder

Create `engine/midi/recorder.py`:

```python
class MidiRecorder:
    """Records MIDI input with beat-accurate timestamps."""

    def __init__(self, transport: Transport, signals: SignalBridge):
        pass

    def start(self):
        """Begin recording."""
        pass

    def stop(self) -> List[RecordedNote]:
        """Stop and return recorded notes."""
        pass

    @property
    def notes(self) -> List[RecordedNote]:
        """Get all recorded notes."""
        pass
```

**Requirements:**
- Subscribe to `SIGNAL_MIDI_NOTE_ON`, `SIGNAL_MIDI_NOTE_OFF`, `SIGNAL_MIDI_PAD`
- Timestamp each note with `transport.playhead_beat`
- Store Launchpad row/col for grid display
- Track note-on/note-off for duration calculation

### Task 3: Sample Triggering

Create `engine/audio/sampler.py`:

```python
class Sampler:
    """Trigger one-shot drum samples."""

    def load(self, name: str, path: str):
        """Load a sample into a slot."""
        pass

    def trigger(self, name: str, velocity: float = 1.0):
        """Play sample immediately."""
        pass
```

**Requirements:**
- Load WAV files into memory
- Mix with main audio output
- Low latency triggering
- Velocity scaling

### Task 4: Audio Mixer

Create `engine/audio/mixer.py`:

```python
class AudioMixer:
    """Mix multiple audio sources."""

    def add_source(self, name: str, source):
        pass

    def remove_source(self, name: str):
        pass

    def get_output(self) -> np.ndarray:
        """Get mixed output buffer."""
        pass

    def get_output_waveform(self) -> np.ndarray:
        """Get RMS envelope of output for visualization."""
        pass
```

---

## Architecture Rules

1. **Transport is master clock**
   ```python
   # Audio position = transport position
   beat = self.transport.playhead_beat
   sample_position = int(beat * self.samples_per_beat)
   ```

2. **Events go through SignalBridge**
   ```python
   # Emit events
   signals.emit(SIGNAL_AUDIO_POSITION, beat)

   # Subscribe to events
   signals.connect(SIGNAL_TRANSPORT_CHANGED, self._on_transport_changed)
   ```

3. **Thread safety for audio**
   - Audio callback runs on separate thread
   - Use `queue.Queue` for thread-safe communication
   - Never access Transport directly from audio thread

4. **Immutable snapshots**
   - Use `TransportState` (frozen dataclass) for thread-safe state passing
   - Don't mutate shared objects from audio thread

---

## New Signals to Add

Add to `engine/core/signal.py`:

```python
# Audio signals
SIGNAL_AUDIO_LOADED = 'audio_loaded'      # (path, duration_beats)
SIGNAL_AUDIO_POSITION = 'audio_position'  # (beat,)
SIGNAL_AUDIO_FINISHED = 'audio_finished'  # ()

# Recording signals
SIGNAL_RECORDING_STARTED = 'recording_started'
SIGNAL_RECORDING_STOPPED = 'recording_stopped'
SIGNAL_NOTE_RECORDED = 'note_recorded'    # (RecordedNote,)
```

---

## File Structure to Create

```
engine/audio/
├── __init__.py
├── device.py       # sounddevice wrapper
├── player.py       # AudioPlayer
├── sampler.py      # Sampler
├── mixer.py        # AudioMixer
└── waveform.py     # Waveform extraction utilities

engine/midi/
├── recorder.py     # MidiRecorder (NEW)
└── ... (existing files)
```

---

## Dependencies

```bash
pip install sounddevice numpy
```

For MP3 support (optional):
```bash
pip install pydub
```

---

## Testing

After implementation, this should work:

```python
from engine.core.signal import SignalBridge
from engine.time.transport import Transport
from engine.audio import AudioPlayer, Sampler, AudioMixer
from engine.midi import MidiManager, MidiRecorder

# Setup
signals = SignalBridge()
transport = Transport(bpm=120)

# Audio
player = AudioPlayer(transport)
player.load("assets/beat.wav")

sampler = Sampler()
sampler.load("kick", "assets/kick.wav")
sampler.load("snare", "assets/snare.wav")

mixer = AudioMixer()
mixer.add_source("player", player)
mixer.add_source("sampler", sampler)

# MIDI
midi = MidiManager(signals)
midi.connect("Launchpad")
recorder = MidiRecorder(transport, signals)

# Start
recorder.start()
transport.play()
player.play()

# ... user plays Launchpad ...

# Later
transport.stop()
recorder.stop()
print(f"Recorded {len(recorder.notes)} notes")
```

---

## Do NOT Modify

- `engine/ui/` - UI widgets (separate task)
- `engine/time/transport.py` - Already complete
- `engine/time/camera.py` - Already complete
- `playable_demo.py` - Will be updated after engine work

---

## Code Style

- Type hints on all public methods
- Docstrings on classes and public methods
- Follow patterns in existing engine code
- No emojis
- Keep it simple - minimal abstractions

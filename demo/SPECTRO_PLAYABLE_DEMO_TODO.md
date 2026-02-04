# SPECTRO Playable Demo — Complete TODO List

> **Goal**: A working demo with Transport, Scrolling Waveform Timeline (DJ-style), Sequencer Grid, MIDI Input, and Audio Playback.
> 
> **Vision Reference**: Traktor/Serato/VirtualDJ parallel waveform view — constantly scrolling timeline where you can see the waveform of what's playing and insert MIDI events that trigger sounds.

---

## Demo Layout

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  HEADER: MIDI Status ● | View Switch                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│  TRANSPORT: [▶] [■] [⏮]  |  120 BPM [━━━━●━━━━━]  |  Loop: 1:1 to 2:4  |  Metro │
├─────────────────────────────────────────────────────────────────────────────────┤
│  WAVEFORM TIMELINE (constantly scrolling, DJ-style)                             │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓███████▓▓▓▓▓▓▓▓▓████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  │    │    │    │    ▼    │    │    │    │    │    │    │    │    │    │    │   │
│  1    2    3    4  PLAYHEAD 6   7    8    9    10   11   12   13   14   15   16  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  SEQUENCER GRID (scrolls in sync with timeline)                                 │
│  ┌─────┬────────────────────────────────────────────────────────────────────────┤
│  │Kick │ ■    ■    ■    ■    ■    ■    ■    ■    ■    ■    ■    ■              │
│  │Snare│      ■         ■         ■         ■         ■         ■              │
│  │HiHat│ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪ ▪    │
│  │Clap │           ▪                   ▪                   ▪                   │
│  └─────┴────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Current State Assessment

Based on SPECTRO_DEVELOPMENT_PLAN_5.md:

### ✅ EXISTS (in codebase)
- Scene graph with RCU-style snapshots (`graph.py`, `snapshot.py`)
- CommandList system for 3D (`commands.py`)
- Renderer with picking support (`renderer.py`)
- Render target pooling (`targets.py`)
- Upload queue with versioning (`uploader.py`)
- Flexbox layout engine (`layout.py`)
- CSS-like style system (`style.py`)
- UI widgets (partial)

### ❌ NOT IMPLEMENTED
- TimeCamera (beat ↔ pixel mapping)
- Transport / TransportState
- 2D drawing commands (rects, lines, waveform)
- GraphPanel system
- MIDI input handling
- Audio synthesis/playback
- Waveform rendering

---

## Phase 0: Demo Shortcut Decision

**Option A: Full Engine Path** (weeks of work)
- Implement all planned architecture
- Use ModernGL + PySide6
- Production-ready but slow

**Option B: Rapid Prototype Path** (days of work) ⬅️ **RECOMMENDED FOR DEMO**
- Use existing Python audio libs (pygame, sounddevice)
- Use existing MIDI libs (mido, rtmidi)
- Simpler rendering (pygame or basic Qt)
- Prove the concept, then port to full engine

**This TODO assumes Option B** — get something playable ASAP, then polish.

---

## Phase 1: Core Time Infrastructure (Day 1)

### 1.1 Transport State
| ID | Task | Status |
|----|------|--------|
| 1.1.1 | Create `TransportState` dataclass (playing, bpm, beat_position, loop_start, loop_end) | ⬜ |
| 1.1.2 | Create `Transport` class with play/pause/stop/toggle methods | ⬜ |
| 1.1.3 | Implement `Transport.update(dt)` — advance beat based on BPM | ⬜ |
| 1.1.4 | Implement loop wraparound logic | ⬜ |
| 1.1.5 | Add `on_beat` callback system for event triggering | ⬜ |
| 1.1.6 | Add `on_bar` callback for phrase-level events | ⬜ |
| 1.1.7 | Unit tests for Transport timing accuracy | ⬜ |

### 1.2 TimeCamera
| ID | Task | Status |
|----|------|--------|
| 1.2.1 | Create `TimeCamera` class with `px_per_beat`, `left_beat` | ⬜ |
| 1.2.2 | Implement `beat_to_px(beat) -> float` | ⬜ |
| 1.2.3 | Implement `px_to_beat(px) -> float` | ⬜ |
| 1.2.4 | Implement `update(playhead_beat, panel_width)` for follow mode | ⬜ |
| 1.2.5 | Add `FOLLOW_PLAYHEAD` mode (playhead stays centered) | ⬜ |
| 1.2.6 | Add `FREE_SCROLL` mode (user controls scroll) | ⬜ |
| 1.2.7 | Implement zoom (change `px_per_beat`, keep anchor) | ⬜ |
| 1.2.8 | Unit tests for beat↔pixel inverse relationship | ⬜ |

---

## Phase 2: Audio Playback System (Day 1-2)

### 2.1 Sound Sample Management
| ID | Task | Status |
|----|------|--------|
| 2.1.1 | Create `SampleBank` class to hold loaded sounds | ⬜ |
| 2.1.2 | Load drum samples (kick, snare, hi-hat, clap, etc.) | ⬜ |
| 2.1.3 | Support WAV file loading via soundfile/scipy | ⬜ |
| 2.1.4 | Normalize samples to consistent volume | ⬜ |
| 2.1.5 | Map MIDI note numbers to samples (GM drum map) | ⬜ |

### 2.2 Audio Engine
| ID | Task | Status |
|----|------|--------|
| 2.2.1 | Choose audio backend (sounddevice recommended) | ⬜ |
| 2.2.2 | Create `AudioEngine` class with audio callback | ⬜ |
| 2.2.3 | Implement sample mixing in callback | ⬜ |
| 2.2.4 | Create `trigger_sound(sample_id, velocity)` method | ⬜ |
| 2.2.5 | Implement polyphonic playback (multiple sounds at once) | ⬜ |
| 2.2.6 | Add volume envelope (simple attack/release) | ⬜ |
| 2.2.7 | Connect to Transport — trigger sounds at beat positions | ⬜ |

### 2.3 Real-time Waveform Buffer
| ID | Task | Status |
|----|------|--------|
| 2.3.1 | Create ring buffer for recent audio output | ⬜ |
| 2.3.2 | Capture audio callback output into ring buffer | ⬜ |
| 2.3.3 | Implement decimation for display (peak finding) | ⬜ |
| 2.3.4 | Provide `get_waveform_data(start_beat, end_beat)` | ⬜ |

---

## Phase 3: MIDI Input System (Day 2)

### 3.1 MIDI Connection
| ID | Task | Status |
|----|------|--------|
| 3.1.1 | Choose MIDI library (mido + rtmidi-python recommended) | ⬜ |
| 3.1.2 | List available MIDI input devices | ⬜ |
| 3.1.3 | Connect to selected MIDI device | ⬜ |
| 3.1.4 | Handle connection/disconnection gracefully | ⬜ |
| 3.1.5 | Display MIDI connection status indicator | ⬜ |

### 3.2 MIDI Processing
| ID | Task | Status |
|----|------|--------|
| 3.2.1 | Create `MidiEvent` dataclass (note, velocity, timestamp) | ⬜ |
| 3.2.2 | Create `MidiInputHandler` class | ⬜ |
| 3.2.3 | Parse incoming MIDI messages (note on/off) | ⬜ |
| 3.2.4 | Quantize MIDI input to grid (optional, configurable) | ⬜ |
| 3.2.5 | Emit callback on MIDI note received | ⬜ |

### 3.3 MIDI to Sequencer Integration
| ID | Task | Status |
|----|------|--------|
| 3.3.1 | On MIDI note: immediately trigger sound (play-through) | ⬜ |
| 3.3.2 | On MIDI note: insert event into sequencer at current beat | ⬜ |
| 3.3.3 | Handle note velocity for volume variation | ⬜ |
| 3.3.4 | Map MIDI notes to sequencer lanes (configurable) | ⬜ |

---

## Phase 4: Sequencer Data Model (Day 2-3)

### 4.1 Event Data Structure
| ID | Task | Status |
|----|------|--------|
| 4.1.1 | Create `SequencerEvent` dataclass (beat, lane, velocity, duration) | ⬜ |
| 4.1.2 | Create `SequencerLane` class (name, events list, color, MIDI note) | ⬜ |
| 4.1.3 | Create `Sequence` class (lanes, length_beats, time_signature) | ⬜ |
| 4.1.4 | Implement `add_event(beat, lane, velocity)` | ⬜ |
| 4.1.5 | Implement `remove_event(event)` | ⬜ |
| 4.1.6 | Implement `get_events_in_range(start_beat, end_beat)` | ⬜ |
| 4.1.7 | Implement `get_events_at_beat(beat, tolerance)` | ⬜ |

### 4.2 Playback Integration
| ID | Task | Status |
|----|------|--------|
| 4.2.1 | Connect Sequence to Transport's `on_beat` callback | ⬜ |
| 4.2.2 | Fire events slightly ahead (look-ahead scheduling) | ⬜ |
| 4.2.3 | Trigger AudioEngine when sequencer event fires | ⬜ |
| 4.2.4 | Support looping — wrap events properly | ⬜ |

---

## Phase 5: Rendering System (Day 3-4)

### 5.1 Choose Rendering Approach
| ID | Task | Status |
|----|------|--------|
| 5.1.1 | **Decision**: pygame / PySide6 QPainter / ModernGL | ⬜ |
| 5.1.2 | Set up window with chosen framework | ⬜ |
| 5.1.3 | Establish 60 FPS render loop | ⬜ |
| 5.1.4 | Handle window resize | ⬜ |

### 5.2 Grid Rendering
| ID | Task | Status |
|----|------|--------|
| 5.2.1 | Create `GridBuilder` class | ⬜ |
| 5.2.2 | Draw vertical beat lines (subdivisions based on zoom) | ⬜ |
| 5.2.3 | Draw bar lines (thicker, different color) | ⬜ |
| 5.2.4 | Draw beat numbers at top | ⬜ |
| 5.2.5 | Adaptive subdivision (show 16ths when zoomed in, quarters when out) | ⬜ |

### 5.3 Waveform Timeline Panel
| ID | Task | Status |
|----|------|--------|
| 5.3.1 | Create `WaveformPanel` class | ⬜ |
| 5.3.2 | Get waveform data from AudioEngine's ring buffer | ⬜ |
| 5.3.3 | Map waveform samples to pixels using TimeCamera | ⬜ |
| 5.3.4 | Draw waveform as filled polygon or line strip | ⬜ |
| 5.3.5 | Color-code waveform by frequency content (optional, later) | ⬜ |
| 5.3.6 | Draw playhead line | ⬜ |
| 5.3.7 | Scroll with TimeCamera in FOLLOW mode | ⬜ |

### 5.4 Sequencer Grid Panel
| ID | Task | Status |
|----|------|--------|
| 5.4.1 | Create `SequencerPanel` class | ⬜ |
| 5.4.2 | Draw lane labels on left side | ⬜ |
| 5.4.3 | Draw horizontal lane dividers | ⬜ |
| 5.4.4 | Draw events as colored rectangles | ⬜ |
| 5.4.5 | Event color based on lane | ⬜ |
| 5.4.6 | Event brightness based on velocity | ⬜ |
| 5.4.7 | Draw playhead line (same position as waveform) | ⬜ |
| 5.4.8 | Scroll in sync with WaveformPanel (shared TimeCamera!) | ⬜ |

### 5.5 Transport UI
| ID | Task | Status |
|----|------|--------|
| 5.5.1 | Create `TransportBar` class | ⬜ |
| 5.5.2 | Draw Play/Pause button | ⬜ |
| 5.5.3 | Draw Stop button | ⬜ |
| 5.5.4 | Draw BPM display | ⬜ |
| 5.5.5 | Draw BPM slider/input | ⬜ |
| 5.5.6 | Draw loop start/end controls | ⬜ |
| 5.5.7 | Draw metronome toggle | ⬜ |
| 5.5.8 | Draw current beat/bar display | ⬜ |

---

## Phase 6: Input Handling (Day 4)

### 6.1 Mouse Input
| ID | Task | Status |
|----|------|--------|
| 6.1.1 | Click on sequencer grid → toggle event at that position | ⬜ |
| 6.1.2 | Click-drag to create/delete multiple events | ⬜ |
| 6.1.3 | Click on transport buttons | ⬜ |
| 6.1.4 | Scroll wheel on panels → zoom in/out | ⬜ |
| 6.1.5 | Middle-click drag → pan the timeline | ⬜ |
| 6.1.6 | Click on timeline → seek to that position | ⬜ |

### 6.2 Keyboard Input
| ID | Task | Status |
|----|------|--------|
| 6.2.1 | Space bar → toggle play/pause | ⬜ |
| 6.2.2 | Enter → stop and return to start | ⬜ |
| 6.2.3 | +/- → adjust BPM | ⬜ |
| 6.2.4 | Arrow keys → nudge playhead | ⬜ |
| 6.2.5 | Z → undo last action | ⬜ |
| 6.2.6 | Delete → remove selected events | ⬜ |

---

## Phase 7: Integration & Polish (Day 5)

### 7.1 Wire Everything Together
| ID | Task | Status |
|----|------|--------|
| 7.1.1 | Create main `App` class that owns all systems | ⬜ |
| 7.1.2 | Initialize: Window, AudioEngine, Transport, Sequence, MidiInput | ⬜ |
| 7.1.3 | Connect MIDI → Sequencer → AudioEngine pipeline | ⬜ |
| 7.1.4 | Connect Transport → TimeCamera → Panels pipeline | ⬜ |
| 7.1.5 | Main loop: poll input → update → render | ⬜ |

### 7.2 Default Content
| ID | Task | Status |
|----|------|--------|
| 7.2.1 | Create default drum kit samples (kick, snare, hihat, clap) | ⬜ |
| 7.2.2 | Create default 4-lane sequencer (mapped to GM drums) | ⬜ |
| 7.2.3 | Pre-populate with simple 4/4 beat pattern | ⬜ |
| 7.2.4 | Default BPM: 120, Loop: 1 bar | ⬜ |

### 7.3 Visual Polish
| ID | Task | Status |
|----|------|--------|
| 7.3.1 | Dark theme colors (match mockups) | ⬜ |
| 7.3.2 | Gradient fills on events (like mockup) | ⬜ |
| 7.3.3 | Smooth playhead animation | ⬜ |
| 7.3.4 | Visual feedback on MIDI input (flash) | ⬜ |
| 7.3.5 | Visual feedback on event trigger (pulse) | ⬜ |

### 7.4 Stability
| ID | Task | Status |
|----|------|--------|
| 7.4.1 | Handle audio underruns gracefully | ⬜ |
| 7.4.2 | Handle MIDI device disconnection | ⬜ |
| 7.4.3 | Proper cleanup on exit | ⬜ |
| 7.4.4 | Performance profiling (target 60 FPS) | ⬜ |

---

## Summary: Critical Path to Playable Demo

**Minimum Viable Demo** (in order of implementation):

1. **Transport + TimeCamera** — Can't do anything without time
2. **AudioEngine + SampleBank** — Need to hear something
3. **Sequencer data model** — Store and play events
4. **Basic rendering** — See what's happening
5. **MIDI input** — Play along and record

**Can skip for initial demo:**
- Zoom (use fixed zoom)
- Free scroll mode (always follow playhead)
- Mouse event editing (MIDI-only input)
- Metronome (rely on drum pattern)
- Fancy waveform (rectangle representation is fine)

---

## Suggested File Structure

```
spectro_demo/
├── main.py                    # Entry point
├── core/
│   ├── __init__.py
│   ├── transport.py           # Transport, TransportState
│   └── time_camera.py         # TimeCamera
├── audio/
│   ├── __init__.py
│   ├── engine.py              # AudioEngine
│   ├── sample_bank.py         # SampleBank
│   └── ring_buffer.py         # RingBuffer for waveform capture
├── midi/
│   ├── __init__.py
│   └── input_handler.py       # MidiInputHandler
├── sequencer/
│   ├── __init__.py
│   ├── event.py               # SequencerEvent
│   ├── lane.py                # SequencerLane
│   └── sequence.py            # Sequence
├── ui/
│   ├── __init__.py
│   ├── app.py                 # Main application window
│   ├── transport_bar.py       # Transport controls
│   ├── waveform_panel.py      # Waveform timeline
│   ├── sequencer_panel.py     # Sequencer grid
│   └── grid_builder.py        # Beat/bar line generation
├── assets/
│   └── samples/
│       ├── kick.wav
│       ├── snare.wav
│       ├── hihat.wav
│       └── clap.wav
└── tests/
    ├── test_transport.py
    ├── test_time_camera.py
    └── test_sequencer.py
```

---

## Dependencies

```bash
# Core
pip install numpy

# Audio
pip install sounddevice
pip install soundfile

# MIDI
pip install mido
pip install python-rtmidi

# GUI (choose one)
pip install pygame           # Option A: simplest
pip install PySide6          # Option B: more polished
# or use existing ModernGL setup for Option C
```

---

## Quick Start Pseudocode

```python
# main.py
class SpectroDemo:
    def __init__(self):
        # Audio
        self.sample_bank = SampleBank()
        self.sample_bank.load("kick", "assets/samples/kick.wav")
        self.sample_bank.load("snare", "assets/samples/snare.wav")
        self.sample_bank.load("hihat", "assets/samples/hihat.wav")
        
        self.audio_engine = AudioEngine(self.sample_bank)
        self.audio_engine.start()
        
        # Time
        self.transport = Transport(bpm=120)
        self.time_camera = TimeCamera(px_per_beat=100)
        
        # Sequencer
        self.sequence = Sequence(length_beats=8)
        self.sequence.add_lane("Kick", midi_note=36)
        self.sequence.add_lane("Snare", midi_note=38)
        self.sequence.add_lane("HiHat", midi_note=42)
        
        # MIDI
        self.midi_input = MidiInputHandler()
        self.midi_input.on_note = self.on_midi_note
        
        # Connect transport to sequencer
        self.transport.on_beat = self.check_events
    
    def on_midi_note(self, note, velocity):
        # Immediate playback
        sample = self.get_sample_for_note(note)
        self.audio_engine.trigger(sample, velocity)
        
        # Insert into sequencer
        current_beat = self.transport.beat_position
        lane = self.get_lane_for_note(note)
        self.sequence.add_event(current_beat, lane, velocity)
    
    def check_events(self, beat):
        events = self.sequence.get_events_at_beat(beat)
        for event in events:
            sample = self.sample_bank.get(event.lane.sample_id)
            self.audio_engine.trigger(sample, event.velocity)
    
    def update(self, dt):
        self.transport.update(dt)
        self.time_camera.update(self.transport.beat_position)
    
    def render(self):
        # Waveform panel
        self.waveform_panel.draw(self.time_camera, self.audio_engine.ring_buffer)
        
        # Sequencer panel  
        self.sequencer_panel.draw(self.time_camera, self.sequence)
        
        # Transport bar
        self.transport_bar.draw(self.transport)
```

---

## Next Steps

1. **Create repository structure** with folders above
2. **Start with Transport + TimeCamera** — unit test them thoroughly
3. **Add AudioEngine** — verify you can trigger sounds
4. **Add Sequencer** — verify transport fires events
5. **Add basic window** — see something on screen
6. **Add MIDI** — play along!
7. **Polish rendering** — make it look good

---

*This is a working document. Update status markers as you progress.*
*Last updated: January 2025*

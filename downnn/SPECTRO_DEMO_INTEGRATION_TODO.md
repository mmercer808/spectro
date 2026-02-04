# SPECTRO Demo Integration TODO

## Summary of What Exists (Verified from Uploaded Code)

### ✅ COMPLETE - Ready to Wire
| System | File | Status |
|--------|------|--------|
| Transport | `engine/time/transport.py` | Full playback, BPM, loop, callbacks |
| TimeCamera | `engine/time/camera.py` | Beat↔px, zoom, follow, animations |
| SignalBridge | `engine/core/signal.py` | Observer hub, all signal types |
| Scene/Entity | `engine/core/scene.py` | Unified 3D space (X=time, Y=freq, Z=intensity) |
| SceneManager | `engine/core/manager.py` | Top-level coordinator |
| MidiManager | `engine/midi/manager.py` | rtmidi, device connect, send/receive |
| LaunchpadController | `engine/midi/launchpad.py` | Grid mapping, LED feedback |
| MidiRecorder | `engine/midi/recorder.py` | Beat-accurate timestamps |
| Sampler | `engine/audio/sampler.py` | One-shots, voice management |
| AudioEngine | `engine/audio/engine.py` | sounddevice output, beat patterns |
| EventDispatcher | `buffers_v2.py` | **LATE-BINDING** event firing |
| MidiRingBuffer | `buffers_v2.py` | Lock-free circular buffer |
| AudioRingBuffer | `buffers_v2.py` | Sample-accurate playhead |
| CommandList | `engine/render/commands.py` | Thread-safe render commands |
| Renderer | `engine/render/renderer.py` | GL executor with picking |
| RenderTargetPool | `engine/render/targets.py` | Pooled FBOs |
| ViewportArea | `engine/viewport/viewport.py` | Multi-camera, async extraction |
| UI Widgets | `engine/ui/widgets/*.py` | Button, Label, Panel, Container |

### ⚠️ NEEDS WIRING (The Demo App Does This)
- Connect EventDispatcher ↔ Transport sync
- Connect MIDI input → Entity creation → Audio trigger
- Connect TimeCamera to visual panels
- Implement DrawContext for 2D rendering

---

## The Core Signal Flow (What the Demo Demonstrates)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT PHASE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Keyboard/Launchpad                                                    │
│          │                                                              │
│          ▼                                                              │
│   InputDevice.note_on(note, velocity)                                   │
│          │                                                              │
│          ▼                                                              │
│   MidiRingBuffer.write(MidiEvent)  ← "pinball loaded in chute"         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DISPATCH PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   EventDispatcher.process_frame(dt)                                     │
│          │                                                              │
│          ├── Advance AudioRingBuffer playhead                           │
│          │                                                              │
│          ├── Scan MidiRingBuffer for events at playhead                 │
│          │                                                              │
│          ▼                                                              │
│   === LATE BINDING HAPPENS HERE ===                                     │
│                                                                         │
│   ExecutionContext assembled with:                                      │
│   - event: MidiEvent (the raw data)                                     │
│   - transport: TransportSnapshot.capture() ← FRESH STATE                │
│   - timing_error_ms: how early/late the hit was                         │
│                                                                         │
│          │                                                              │
│          ▼                                                              │
│   Fire registered callbacks with ExecutionContext                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENTITY CREATION                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   on_midi_input(ctx: ExecutionContext):                                 │
│          │                                                              │
│          ├── lane_index = ctx.event.note % num_lanes                    │
│          │                                                              │
│          ├── sequencer.add_event(                                       │
│          │       lane_index,                                            │
│          │       beat=ctx.transport.beat,  ← FROM LATE-BOUND CONTEXT    │
│          │       velocity=ctx.event.velocity                            │
│          │   )                                                          │
│          │                                                              │
│          │   Returns: SequencerEvent (the ENTITY)                       │
│          │      - id: unique                                            │
│          │      - beat: X position on timeline                          │
│          │      - lane: Y position (row)                                │
│          │      - duration: width                                       │
│          │      - velocity: intensity                                   │
│          │      - color: for rendering                                  │
│          │                                                              │
│          ▼                                                              │
│   audio.trigger(sample_name, velocity)  ← IMMEDIATE FEEDBACK            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RENDER PHASE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   on_render(t, frame_time):                                             │
│          │                                                              │
│          ├── transport.update(dt)           # Advance playhead          │
│          ├── dispatcher.process_frame(dt)   # Fire events               │
│          ├── time_camera.update(dt, beat)   # Sync view                 │
│          │                                                              │
│          ├── _draw_timeline()               # Beat grid lines           │
│          │      └── time_camera.beat_to_px(beat)                        │
│          │                                                              │
│          ├── _draw_sequencer_grid()         # Event rectangles          │
│          │      └── For each SequencerEvent:                            │
│          │            px = time_camera.beat_to_px(event.beat)           │
│          │            draw_rect(px, lane_y, width, height, color)       │
│          │                                                              │
│          └── _draw_playhead()               # Current position          │
│                 └── time_camera.beat_to_px(transport.playhead_beat)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PLAYBACK PHASE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   on_beat(beat, transport):  ← Called by dispatcher                     │
│          │                                                              │
│          ├── events = sequencer.get_events_in_range(                    │
│          │       current_beat - window,                                 │
│          │       current_beat + window                                  │
│          │   )                                                          │
│          │                                                              │
│          ├── For each event not yet fired:                              │
│          │      if event.beat <= current_beat:                          │
│          │          event.fired = True                                  │
│          │          audio.trigger(event.sample_name)                    │
│          │                                                              │
│          └── (Loop restart resets all fired flags)                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Immediate TODO (To Make Demo Runnable)

### Phase 1: Verify Imports (30 min)
- [ ] Ensure all engine modules are importable
- [ ] Fix any circular import issues
- [ ] Create `__init__.py` files if missing

### Phase 2: Test Basic Flow (1 hour)
- [ ] Run `spectro_demo_app.py`
- [ ] Verify keyboard 1-4 triggers sound
- [ ] Verify events appear on timeline
- [ ] Verify playhead moves when playing
- [ ] Verify playback triggers recorded events

### Phase 3: Add Launchpad Support (1 hour)
```python
# In SpectroDemo.__init__:
from engine.midi.manager import MidiManager
from engine.midi.launchpad import LaunchpadController

self.midi_manager = MidiManager(self.signals)
self.launchpad = LaunchpadController(self.midi_manager, self.signals)

# Connect to Launchpad (if available)
if self.midi_manager.list_input_ports():
    self.midi_manager.connect("Launchpad")  # Partial match
    
# Wire pad events to same flow
self.signals.connect(SIGNAL_MIDI_PAD, self._on_pad)

def _on_pad(self, row, col, velocity):
    if velocity > 0:
        # Map row to lane, emit to buffer
        self.keyboard_device.note_on(row, velocity)
```

### Phase 4: Add Waveform Display (2 hours)
```python
def _draw_waveform(self, w, h):
    """Draw audio output waveform above timeline."""
    waveform_y = 10
    waveform_h = 50
    
    # Get recent audio from AudioEngine
    samples = self.audio.get_output_waveform(num_samples=w)
    
    # Draw as vertical bars
    for i, amp in enumerate(samples):
        bar_h = amp * waveform_h
        self._draw_rect(
            i, waveform_y + (waveform_h - bar_h) / 2,
            1, bar_h,
            (0.3, 0.7, 0.9, 0.8),
            (w, h)
        )
```

### Phase 5: Polish (2 hours)
- [ ] Add BPM display/control
- [ ] Add loop region markers
- [ ] Add velocity display on events
- [ ] Add lane labels
- [ ] Scroll/pan with mouse drag

---

## Project Knowledge Recommendations

### Files to Add to Project Knowledge
These are the **core** files that define the architecture:

```
Priority 1 (Essential):
├── buffers_v2.py              # Late-binding event system
├── engine/time/transport.py   # Transport state
├── engine/time/camera.py      # TimeCamera
├── engine/core/signal.py      # SignalBridge
└── engine/core/scene.py       # Entity system

Priority 2 (Reference):
├── engine/audio/engine.py     # AudioEngine
├── engine/audio/sampler.py    # Sampler
├── engine/midi/manager.py     # MidiManager
└── engine/midi/launchpad.py   # LaunchpadController

Priority 3 (Deep Dive):
├── engine/render/commands.py  # CommandList
├── engine/render/renderer.py  # GL executor
├── engine/core/snapshot.py    # RCU snapshots
└── engine/render/world.py     # Batching
```

### Folder Structure (Suggested)
```
/mnt/skills/user/spectro/
├── SKILL.md                   # How to work with SPECTRO
├── ARCHITECTURE.md            # System overview
├── SIGNAL_FLOW.md            # This document
└── core_files/
    ├── buffers_v2.py
    ├── transport.py
    ├── camera.py
    └── signal.py
```

---

## Quick Reference: Key APIs

### Transport
```python
transport.play() / .pause() / .stop() / .toggle()
transport.seek_to_beat(beat) / .seek_to_bar(bar)
transport.set_bpm(bpm)
transport.on_beat_callbacks.append(fn)
transport.on_bar_callbacks.append(fn)
state = transport.update(dt)  # Returns TransportState
```

### TimeCamera
```python
px = time_camera.beat_to_px(beat)
beat = time_camera.px_to_beat(px)
time_camera.zoom(delta, anchor_px)
time_camera.animate_to_beat(beat)
time_camera.update(dt, playhead_beat)
for beat in time_camera.iter_beat_positions(): ...
```

### EventDispatcher (Late-Binding)
```python
dispatcher.register(
    callback=fn,           # fn(ctx: ExecutionContext)
    event_types={MidiEventType.NOTE_ON},
    name="handler_name"
)
dispatcher.on_beat(fn)     # fn(beat, transport_snapshot)
dispatcher.play() / .pause()
dispatcher.process_frame(dt)  # Returns List[ExecutionContext]
```

### InputDevice
```python
device = InputDevice("name")
device.connect(midi_buffer, audio_buffer)
device.note_on(note, velocity, channel=0)
device.note_off(note, channel=0)
device.control_change(cc, value, channel=0)
```

---

## What Success Looks Like

1. **Press key 1** → Kick sound plays, orange rect appears at playhead position
2. **Press SPACE** → Playhead starts moving
3. **Playhead crosses event** → Kick sound plays again
4. **Scroll wheel** → Timeline zooms in/out
5. **Connect Launchpad** → Pads trigger corresponding lanes with LED feedback
6. **Loop enabled** → Events replay each loop, fired flags reset

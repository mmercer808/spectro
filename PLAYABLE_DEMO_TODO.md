# SPECTRO Playable Demo TODO

> **Goal**: Play Launchpad → see sequence in a 4/4 grid → hear audio (when wired).  
> **Layout**: Panel above for 3D/overlay view; sequencer grid below; each grid square has handler listeners for its note.

---

## 1. Sequencer Grid (4/4, Cell Listeners)

### 1.1 Grid Model (4/4)

- **Time signature**: 4 beats per bar; grid step = 1 beat (or ¼ beat for 16th).
- **Layout**: Rows = lanes (Kick, Snare, HiHat, Clap, …). Columns = beat slots in 4/4 (e.g. 4 columns per bar, or 16 for 16ths).
- **Each cell** = one slot `(lane_index, beat_index)` where `beat_index` is in beats (e.g. 0, 1, 2, 3 for bar 0; 4, 5, 6, 7 for bar 1).

**Data model** (align with existing `SequencerEvent`):

- `SequencerEvent`: `beat` (float), `lane` (int), `duration`, `velocity`, `sample_name`, `color`.
- **Grid → events**: Each “square” in the grid is either empty or has an event at `beat = bar * 4 + step` (step 0..3 per bar for quarter notes).

### 1.2 Cell Handler Listeners (Per-Note in Sequence)

- **Each sequencer cell** (or each event) should support **listeners** so that:
  - **On trigger** (playback): when playhead hits that beat, the handler runs (e.g. `audio.trigger(sample_name, velocity)`).
  - **On edit** (place/delete): when user or Launchpad places/deletes a note, the handler runs (e.g. update model, LED feedback).
- **Concrete tasks**:
  - [ ] Add optional `on_trigger` (and optionally `on_remove`) to `SequencerEvent` or to a **cell delegate** keyed by `(lane, beat)`.
  - [ ] When playback fires an event (in `on_beat` / dispatcher path), call that event’s (or cell’s) handler before/after `audio.trigger`.
  - [ ] When placing/removing an event (MIDI/Launchpad or mouse), invoke the cell’s edit handler (e.g. for LED or UI feedback).

**Minimal API** (conceptual):

```text
# Per-cell handler (e.g. for a grid square at lane L, beat B)
def on_cell_trigger(lane: int, beat: float, event: SequencerEvent, transport_snapshot):
    audio.trigger(event.sample_name, event.velocity / 127.0)

def on_cell_edit(lane: int, beat: float, added: bool, event: Optional[SequencerEvent]):
    if launchpad:
        launchpad.set_pad_color(lane, beat_to_col(beat), color if added else OFF)
```

- [ ] Implement a **SequencerGrid** or extend **Sequencer** so each cell (lane, beat slot) can register listeners; or use a single “sequence listener” that dispatches by (lane, beat).

---

## 2. How to Queue Audio Work / Events / Edits / Notes

Audio isn’t working yet; below is how events and notes are queued and executed.

### 2.1 Two Paths: Live Input vs Playback

| Path | Queue / source | Who consumes | Audio |
|------|----------------|-------------|--------|
| **Live input** (Launchpad/keyboard) | `MidiRingBuffer` | `EventDispatcher.process_frame(dt)` | Callback calls `audio.trigger()` immediately |
| **Playback** (timeline) | Sequencer events (in range of playhead) | Transport `on_beat` (or dispatcher beat callback) | Scan `get_events_in_range(current_beat ± ε)`, call `audio.trigger()` for each |

There is **no separate “audio command queue”** today: triggers are immediate. For “queue” we mean: (1) **MIDI queue** = `MidiRingBuffer`; (2) **Timeline queue** = sequencer events; (3) optional future **scheduled-audio queue** (trigger at sample X).

### 2.2 Live Input Queue (MIDI → Audio)

```text
Launchpad/Keyboard
    → InputDevice.note_on(note, velocity)
    → MidiRingBuffer.write(MidiEvent)   ← “queued” here
    → (each frame) EventDispatcher.process_frame(dt)
        → AudioRingBuffer.read(n)       ← advances playhead
        → MidiRingBuffer.pop_ready()     ← events whose time ≤ playhead
        → for each event: build ExecutionContext (late-binding transport state)
        → fire callbacks(context)
            → your callback: sequencer.add_event(...); audio.trigger(...)
```

So: **queue = MidiRingBuffer**. Work is “queued” as MIDI events; “processed” when `process_frame` runs and playhead has reached the event time.

### 2.3 Playback Queue (Timeline Notes → Audio)

- **Source of truth**: `Sequencer.get_events_in_range(start_beat, end_beat)`.
- **Consumer**: A beat callback (e.g. `dispatcher.on_beat(fn)` or `transport.on_beat_callbacks`) that:
  - Gets current beat from transport.
  - Calls `sequencer.get_events_in_range(current_beat - ε, current_beat + ε)`.
  - For each event not yet fired: mark fired, call `audio.trigger(event.sample_name, velocity)` (and optional **cell trigger listener**).
- **Loop**: On loop restart, reset `event.fired` (or equivalent) so events fire again.

So: **queue = sequencer events on the timeline**. No separate audio queue; “scheduling” is by beat position and firing in the beat callback.

### 2.4 Optional: Explicit Audio Schedule Queue (Future)

If you want **sample-accurate** or **look-ahead** scheduling:

- [ ] Add an **AudioScheduleQueue**: list of `(trigger_sample_index, sample_name, velocity)`.
- [ ] Each frame (or on beat): compute `current_sample = transport.playhead_samples`; append triggers for the next N ms (e.g. next 50 ms) from sequencer events.
- [ ] Audio callback (or a separate consumer) pops triggers where `trigger_sample_index <= current_sample` and calls `sampler.trigger(...)`.

For the playable demo, **beat callback + immediate trigger** is enough; the “queue” is the sequencer + beat callback.

---

## 3. Launchpad → Sequence in Grid

- **Launchpad**: Use existing `engine/midi/launchpad.py` (`LaunchpadController`, `LaunchpadMapper`). Connect via `MidiManager`; subscribe to pad events (e.g. `SIGNAL_MIDI_PAD` or `on_pad`).
- **Map pad (row, col) to grid**:
  - Rows 1–4 (or 0–3) = lanes (Kick, Snare, HiHat, Clap).
  - Cols 0–15 (or 0–7 × 2 bars) = beat slots in 4/4 (e.g. col = beat_in_loop % 16 for 16th, or % 4 for quarter).
- **On pad press**: 
  - Compute `(lane_index, beat_index)` from (row, col).
  - Either **toggle** event at that cell (add/remove) or **add** with default velocity.
  - Push to sequencer: `sequencer.add_event(lane_index, beat=beat_index, velocity=...)` or remove.
  - Optionally inject into `MidiRingBuffer` (or call the same path as MIDI input) so live play-through and recording stay in sync.
- **LED feedback**: For each pad, set color from sequencer state: event at (lane, beat) → ON color; no event → OFF. Refresh on sequencer change or on beat (e.g. dim current playhead column).

Tasks:

- [ ] Wire `MidiManager` + `LaunchpadController` in the demo app.
- [ ] Map Launchpad grid (row, col) → (lane_index, beat_in_loop).
- [ ] On pad: update sequencer (add/toggle/remove event); optionally write to `MidiRingBuffer` for dispatcher.
- [ ] Implement LED refresh: from sequencer state + playhead (e.g. `set_pad_color(row, col, ...)`).

---

## 4. Panel Above for 3D / Overlay View

- **Panel above**: A **top panel** (widget) that can show:
  - A **3D view** (free camera: orbit/pan to view anything),
  - Or **any other view** “dumped” into it (waveform, lanes, analyzer, etc.).
- **Widget view first**: The panel is a **widget** (e.g. `Viewport3D` or a generic “overlay” widget) owned by the layout. It’s the same size/position as the panel.
- **3D view free to view anything**: The 3D viewport is not tied to one scene; it can show:
  - The main 3D scene (e.g. shared scene graph),
  - A copy of the waveform as 3D geometry,
  - Lanes as 3D strips,
  - Or any “dump” of another view as a texture/quad. So “dump any view into that above view” = render another view to a texture and show it in the 3D panel (e.g. as a quad or in the scene).
- **Overlay**: “Overlay any view” = the same panel can show a 3D scene with an overlay (e.g. 2D UI or a rendered 2D view on top). So the panel supports:
  - Background: 3D or a “dumped” view texture.
  - Optional overlay: another view or UI.

Tasks:

- [ ] Add a **top panel** in the layout (above transport + waveform + sequencer) that hosts a single widget (e.g. `Viewport3D` or `OverlayView`).
- [ ] Implement **OverlayView** (or extend Viewport3D): can set “source” to 3D scene, or to a texture produced by another view (waveform, lanes, etc.).
- [ ] 3D camera: free orbit/pan (no fixed binding to one scene).
- [ ] “Dump view into panel”: render waveform/lanes to a texture; bind that texture to the top panel’s view (e.g. fullscreen quad in 3D or 2D overlay).

---

## 5. Dispatcher Equation (Graph)

The **dispatcher** is the function that advances time, pops ready MIDI events, and fires callbacks with a **late-bound** context. Conceptually it’s an equation:

```text
context = f(event, transport, time)
```

where `transport` and `time` are read **at execution time**, not at registration time.

### 5.1 Equation (One Frame)

```text
process_frame(dt) =
    advance(playhead, dt)
    → ready_events = pop_ready(MidiRingBuffer)
    → for each event in ready_events:
        context = ExecutionContext(
            event,
            transport = TransportSnapshot.capture(dispatcher),
            fire_time, latency_ms, timing_error_ms, ...
        )
        fire(callbacks, context)
    → check_beat_bar()
    → fire(beat_callbacks, current_beat, transport)
    → return fired_contexts
```

So:

- **Inputs**: `dt`, `MidiRingBuffer`, `AudioRingBuffer` (for playhead), registered callbacks, beat/bar callbacks.
- **Collection solver**: The dispatcher needs a **collection solver** for whatever **length** (loop/sequence in beats), **bar length** (brlength = beats per bar), and **tempo** (BPM). That solver converts samples ↔ beats, advances playhead, and computes `current_bar`, `beat_in_loop`, etc. See [docs/dispatcher_equation.md](docs/dispatcher_equation.md) § Collection Solver.
- **Output**: List of `ExecutionContext` that were fired; side effects = callbacks run (e.g. sequencer + audio trigger).

### 5.2 Graph (Flow Diagram)

```text
                    ┌─────────────────────────────────────────────────────────────┐
                    │                    DISPATCHER(Δt)                            │
                    └─────────────────────────────────────────────────────────────┘
                                               │
     ┌─────────────────────────────────────────┼─────────────────────────────────────────┐
     │                                         ▼                                         │
     │   advance(playhead)  ←── AudioRingBuffer.read(frame_samples)                      │
     │         │                     │                                                   │
     │         │                     └──→ MidiRingBuffer.advance_playhead(samples)        │
     │         ▼                                                                          │
     │   ready_events ← MidiRingBuffer.pop_ready()                                        │
     │         │                                                                          │
     │         ▼                                                                          │
     │   for each event in ready_events:                                                  │
     │         │                                                                          │
     │         ├──→ context = ExecutionContext(                                          │
     │         │         event,                                                          │
     │         │         transport = TransportSnapshot.capture(dispatcher),  ← LATE BIND  │
     │         │         fire_time, latency_ms, timing_error_ms, ...                      │
     │         │     )                                                                    │
     │         │                                                                          │
     │         └──→ fire(registered_callbacks, context)                                  │
     │                     │                                                              │
     │                     ├──→ e.g. sequencer.add_event(...)                             │
     │                     └──→ e.g. audio.trigger(...)                                   │
     │                                                                                    │
     │   _check_boundaries()                                                              │
     │         │                                                                          │
     │         ├──→ on beat: fire(beat_callbacks, current_beat, transport)               │
     │         │              └──→ e.g. sequencer.get_events_in_range() → audio.trigger   │
     │         └──→ on bar:  fire(bar_callbacks, current_bar, transport)                 │
     │                                                                                    │
     └───────────────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Audio “Queue” in the Equation

- **Live**: `event ∈ ready_events` → `fire(callbacks, context)` → one callback does `audio.trigger(...)`. So “queue” = `MidiRingBuffer`; “work” = callback.
- **Playback**: `beat_callbacks` run with `(current_beat, transport)`. One callback does `events = sequencer.get_events_in_range(...)` then for each event `audio.trigger(...)`. So “queue” = sequencer events; “work” = beat callback.

So the **dispatcher equation** ties together: **input buffer (MIDI) + playhead (audio) + late-bound context + callbacks**. No separate “audio queue” unless we add a scheduled-trigger queue later.

---

**Mermaid graph**: See [docs/dispatcher_equation.md](docs/dispatcher_equation.md) for a renderable Mermaid diagram of the dispatcher flow.

## 6. Task Checklist (Summary)

| Area | Task |
|------|------|
| **4/4 grid** | Sequencer grid with 4/4; cells = (lane, beat); optional 16th subdivision. |
| **Cell listeners** | Per-cell (or per-event) trigger and edit handlers; wire to playback and LED. |
| **Audio queue** | Use MidiRingBuffer for live; sequencer + on_beat for playback; document; optional AudioScheduleQueue later. |
| **Launchpad** | Wire MidiManager + LaunchpadController; map pads → (lane, beat); toggle/add events; LED from sequencer + playhead. |
| **3D overlay panel** | Top panel widget; 3D view or “dumped” view; free camera; overlay support. |
| **Dispatcher graph** | Document equation and flow (this doc §5). |

---

## 7. File References

- **EventDispatcher / buffers**: `engine/buffers_v2.py` — `EventDispatcher.process_frame`, `ExecutionContext`, `MidiRingBuffer`, `AudioRingBuffer`.
- **Transport**: `engine/time/transport.py` — `Transport`, `TransportState`, `on_beat_callbacks`.
- **Launchpad**: `engine/midi/launchpad.py` — `LaunchpadController`, `LaunchpadMapper`, `set_pad_color`.
- **Audio**: `engine/audio/engine.py` — `AudioEngine.trigger`; `engine/audio/sampler.py` — `Sampler.trigger`.
- **Demo**: `spectro_demo_app.py` (full), `demo/run_demo.py` (display-only).
- **Integration flow**: `SPECTRO_DEMO_INTEGRATION_TODO.md` — signal flow and wiring.

---

*Last updated: Feb 2025*

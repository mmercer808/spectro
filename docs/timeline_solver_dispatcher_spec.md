# Timeline — Solver — Dispatcher — Event Manager (Formal Spec)

Formalization of the four-layer architecture that drives the quad stream. The same pipeline runs over an internal BPM clock or **over an analyzed audio sample** (e.g. heavenfalls.mp3).

---

## 1. Layer Definitions

### 1.1 Timeline

**Role:** Single source of truth for **position** and **length** in time. Everything downstream is keyed by this.

**Interface (conceptual):**

| Symbol / accessor | Type | Meaning |
|-------------------|------|---------|
| `playhead_samples` | int | Current position in samples (monotonic or wrapped by length). |
| `playhead_beats` | float | Current position in beats (derived or native). |
| `length_beats` | float | Total length of the timeline in beats (0 = unbounded). |
| `length_samples` | int | Total length in samples (optional; can derive from length_beats and tempo). |
| `playing` | bool | Whether the timeline is advancing. |

**Implementations:**

| Implementation | playhead | length | Use case |
|----------------|----------|--------|----------|
| **Internal** | Advance by `dt * sr` each frame; beat = samples × tempo/(60×sr) | `loop_length_beats` (fixed) | Sequencer/loop without an audio file. |
| **Audio file (analyzed)** | File position (samples/sec) → beats via analyzed or user BPM | File duration in beats (from analysis) | Run over a backing track (e.g. heavenfalls.mp3). |

**Invariant:** All consumers (solver, dispatcher, event manager, quad stream) read **only** from the timeline (and solver outputs). No other source of “current time” or “length.”

---

### 1.2 Solver (Collection Solver)

**Role:** Given timeline position and length, plus **tempo** and **bar length**, compute derived time quantities used for collection and loop-aware logic.

**Parameters (inputs):**

| Symbol | Meaning | Example |
|--------|---------|---------|
| `length` | Timeline length in beats (from Timeline) | 16, or file duration in beats |
| `brlength` | Beats per bar (e.g. 4 for 4/4) | 4 |
| `tempo` | BPM — **user-provided (proper fixation)**; do not hardcode | 120 |
| `sr` | Sample rate (Hz) | 44100 |

**Tempo:** The user provides the proper fixation on tempo (BPM). The solver uses it; it is not assumed or hardcoded in production. Optional: tempo can also come from analyzed sample (e.g. beat detection). **Standard of record and fixation of tempo on the beats:** See [sourcebook/tempo_fixation_standard.md](sourcebook/tempo_fixation_standard.md).

**Outputs (derived):**

| Symbol | Formula / meaning |
|--------|--------------------|
| `samples_per_beat` | `sr * 60 / tempo` |
| `playhead_beats` | `playhead_samples * tempo / (60 * sr)` (if not provided by timeline) |
| `current_bar` | `floor(playhead_beats / brlength)` |
| `current_beat_in_bar` | `playhead_beats mod brlength` |
| `beat_in_loop` | `playhead_beats mod length` when length > 0, else `playhead_beats` |
| `bars_in_loop` | `length / brlength` when length > 0 |
| `bar_in_loop` | `current_bar mod bars_in_loop` when length > 0 |

**In code:** `EventDispatcher` and `AudioRingBuffer` (engine/buffers_v2.py) hold tempo, sr, brlength, loop length; `TransportSnapshot.capture(dispatcher)` exposes beat, beat_in_loop, bar_in_loop, bars_in_loop. See docs/dispatcher_equation.md § Collection Solver.

**Invariant:** Solver is stateless given (playhead, length, tempo, brlength, sr). No hidden state.

---

### 1.3 Dispatcher

**Role:** Advance the timeline (or receive advance from it), **collect** ready events from the MIDI buffer (timestamp ≤ playhead), and **fire** callbacks with a **late-bound** context (TransportSnapshot captured at fire time).

**Interface:**

| Operation | Meaning |
|-----------|---------|
| `process_frame(dt)` | Advance timeline by `dt` (or sync to timeline); pop ready MIDI events; for each event build `ExecutionContext(transport=TransportSnapshot.capture(self))` and fire registered callbacks; check beat/bar boundaries and fire beat/bar callbacks. |
| `register(callback, ...)` | Register a callback for MIDI events (receives `ExecutionContext`). |
| `on_beat(callback)` | Register callback for beat boundaries (receives `beat, transport_snapshot`). |
| `on_bar(callback)` | Register callback for bar boundaries. |
| `play()` / `pause()` / `stop()` | Control timeline advance. |
| `set_bpm`, `set_beats_per_bar`, `set_loop_length` | Configure solver parameters. |

**Invariant:** Context passed to callbacks is **always** built at execution time (TransportSnapshot.capture(dispatcher)). No stale closure over time.

---

### 1.4 Event Manager

**Role:** The component that **registers** and **routes** events: it connects the dispatcher’s callbacks to application logic (sequencer, audio, UI, automation). It is the “event manager” in the sense of “who subscribes to what.”

**Responsibilities:**

- Register with the dispatcher: MIDI → entity creation + live audio trigger.
- Register beat/bar callbacks: sequencer playback (get_events_in_range → audio.trigger), cell listeners, LED/UI updates.
- Optionally: route other signals (e.g. transport play/stop, view changes).

**Interface (conceptual):**

| Operation | Meaning |
|-----------|---------|
| `wire_midi_to_sequencer_and_audio(dispatcher, sequencer, audio)` | Register a callback on dispatcher that, given ExecutionContext, adds a sequencer event and triggers audio. |
| `wire_beat_to_sequencer_playback(dispatcher, sequencer, audio)` | Register a beat callback that calls `sequencer.get_events_in_range(beat_in_loop ± ε)` and triggers audio for each. |
| (Optional) `wire_bar_to_automation(dispatcher, ...)` | Bar callback for automation or reset. |

**In code:** This is the “wiring” layer in the demo app (e.g. spectro_demo_app.py `_setup_callbacks`). It can be a dedicated class (EventManager) or inline registration.

**Invariant:** Event manager does not define time or length; it only subscribes to the dispatcher and uses TransportSnapshot (and solver outputs) from the context.

---

## 2. Data Flow (Formal)

```
Timeline (playhead_samples, length_beats, playing)
    │
    ▼
Solver (length, brlength, tempo, sr) → beat_in_loop, bar_in_loop, samples_per_beat, ...
    │
    ▼
Dispatcher (process_frame(dt))
    │
    ├── advance timeline (or read from timeline)
    ├── pop_ready(MidiRingBuffer)  ← events where timestamp_samples ≤ playhead_samples
    ├── for each event: ExecutionContext(transport=TransportSnapshot.capture(dispatcher)) → fire(callbacks)
    └── check_beat_bar → fire(beat_callbacks, bar_callbacks)
    │
    ▼
Event Manager (registered callbacks)
    │
    ├── Stream 1: MIDI → sequencer.add_event + audio.trigger
    ├── Stream 2: beat → sequencer.get_events_in_range → audio.trigger, cell listeners
    ├── Stream 3: waveform view (same playhead/length)
    └── Stream 4: automation / extra lane (same beat_in_loop)
```

**Timeline** is the only producer of position/length. **Solver** is pure derivation. **Dispatcher** advances (or syncs to) timeline and fires. **Event manager** is the set of subscriptions that implement the quad stream.

---

## 3. Over an Analyzed Sample (e.g. heavenfalls.mp3)

The sample (audio file) is **analyzed** to drive the timeline and solver. The same dispatcher and event manager then run **over** that analyzed sample.

### 3.1 What “analyzed” means

| Analysis output | Required? | Use |
|-----------------|-----------|-----|
| **Duration** | Yes | Timeline length in seconds → `length_beats = duration_sec * (tempo/60)`. |
| **Sample rate** | Yes | For sample-accurate position and solver (samples_per_beat, etc.). |
| **Tempo (BPM)** | Optional | If omitted, user sets BPM; then length_beats = duration_sec * (bpm/60). If analyzed (e.g. from beat detection), use it for solver and length_beats. |
| **Beat grid** | Optional | List of beat times (seconds or samples). Enables playhead_beats from file position by mapping position → nearest beat or interpolated beat. |

So: **minimum** = duration + sample rate (+ user BPM). **Richer** = duration + sample rate + analyzed tempo and/or beat grid.

### 3.2 Analysis pipeline (conceptual)

1. **Load file** (e.g. heavenfalls.mp3) → decode to PCM, get `duration_sec`, `sr`, `num_samples`.
2. **Analyze (optional):**
   - **Tempo:** Use beat-detection (e.g. math/beat_detection_oscillator_theory.md, onset + oscillator agreement) to estimate BPM; or use a library (librosa, aubio, etc.).
   - **Beat grid:** Optionally compute beat positions (times in seconds or sample indices) so the timeline can snap or align to actual beats in the file.
3. **Set timeline (audio-file mode):**
   - `length_samples = num_samples`, `length_beats = duration_sec * (bpm/60)` (with BPM from analysis or user).
   - Each frame: `playhead_samples = file_position_samples` (from audio playback or scrubber), `playhead_beats = playhead_samples * (bpm/60) / sr` or from beat grid lookup.
4. **Configure solver:** `set_bpm(bpm)`, `set_beats_per_bar(brlength)`, `set_loop_length(length_beats)` (or length from file).
5. **Run dispatcher + event manager** as usual; quad stream (MIDI, sequencer, waveform, +1) now runs **over** the analyzed sample.

### 3.3 Can we do that?

Yes, in stages:

1. **Without beat detection (minimal):**
   - Load heavenfalls.mp3 (e.g. with soundfile, pydub, or ffmpeg) to get duration and sample rate.
   - User sets BPM (or we use a default).
   - `length_beats = duration_sec * (bpm/60)`.
   - Playback: drive `playhead_samples` from the file playback position (e.g. from an audio file player that reports position).
   - Dispatcher/solver/event manager use that playhead and length; waveform stream shows the file; sequencer and MIDI run in beat time over the file.

2. **With tempo analysis:**
   - Run an offline tempo estimator on the decoded PCM (e.g. onset strength + autocorrelation, or the oscillator-based method in math/beat_detection_oscillator_theory.md).
   - Use estimated BPM for solver and for `length_beats`.
   - Rest same as above.

3. **With beat grid (optional):**
   - Run beat tracking to get a list of beat times (seconds or samples).
   - Map `playhead_samples` → `playhead_beats` by lookup/interpolation in the beat grid instead of linear `samples * (bpm/60) / sr`. Gives alignment to actual beats in the track.

**Implementation notes:**

- **Loading MP3:** Use `soundfile` (if libsndfile built with MP3), or `pydub` (ffmpeg), or decode to WAV and load with `soundfile`/`scipy.io.wavfile`. Engine already uses `soundfile` in sampler; waveform extraction can share the same load path.
- **Duration:** From decoded length and sample rate: `duration_sec = num_samples / sr`.
- **Tempo/beat analysis:** Either integrate the math in `math/beat_detection_oscillator_theory.md` (STFT → onset → oscillator banks → tempo/beats), or use an existing library (e.g. librosa.beat.beat_track, aubio tempo) for a first version.
- **Timeline driver for file:** A small “file timeline” driver that: (a) loads the file and runs analysis (duration, optional BPM/beats), (b) exposes `playhead_samples` (from playback or scrub), `length_beats`, `playing`, (c) is used to sync the dispatcher (e.g. by pushing file samples into the same AudioRingBuffer and advancing read pointer from file position, or by setting a separate playhead that the dispatcher reads). The existing quad-stream wiring then runs over that timeline.

### 3.4 Summary

- **Timeline** can be internal (BPM + loop) or **audio file** (position + length from file).
- **Analyzed sample** = file loaded + duration + sr + (optional) tempo and/or beat grid.
- **Formal pipeline:** Timeline (from analyzed sample) → Solver → Dispatcher → Event Manager → quad stream. Same four layers; only the timeline implementation and the analysis step (for file) are new.
- **Yes, we can do that:** Start with load + duration + user BPM; add tempo/beat analysis when needed; wire a file-timeline driver into the existing dispatcher/solver/event-manager stack.

---

## 4. File References

| Layer | Code | Doc |
|-------|------|-----|
| Timeline | Internal: engine/buffers_v2.py (AudioRingBuffer, EventDispatcher). File: to be added. | docs/dispatcher_equation.md § Quad stream over any sample |
| Solver | engine/buffers_v2.py (EventDispatcher, AudioRingBuffer, TransportSnapshot) | docs/dispatcher_equation.md § Collection Solver |
| Dispatcher | engine/buffers_v2.py (EventDispatcher.process_frame, TransportSnapshot.capture) | docs/dispatcher_equation.md |
| Event manager | spectro_demo_app.py _setup_callbacks; or dedicated EventManager class | SPECTRO_DEMO_INTEGRATION_TODO.md |
| Beat detection (analysis) | math/beat_detection_oscillator_theory.md, math/comprehensive_beat_detection_math.md | math/CLAUDE.md |
| Waveform | engine/audio/waveform.py (extract_waveform, etc.) | — |

---

**Clubs (links and bouncing elements):** See `docs/clubs/index.json` and the `.json` club files in `docs/clubs/` for multiple elements that reference each other (timeline, solver, dispatcher, events, analysis), including external links (librosa, aubio, soundfile, pydub). Tempo is noted as user-provided in `solver_club.json`.

*Last updated: Feb 2025*

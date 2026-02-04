# Spectro Sequencer Methodology + TODO
Date: 2026-01-24  
Project: https://github.com/mmercer808/spectro

This document captures the **methodology** (how the pipeline works), the **TODO list**, and an **outline** for implementing and using a **block of space = period of section** (e.g., 1 bar, 4 bars, verse = 16 bars).

---

## 0) Definitions

### Timeline time vs audio time
- **Timeline time**: musical time used for editing/history (seconds initially; later ticks/PPQ + tempo map).
- **Audio time**: sample index used for circular buffer scheduling.

### Core artifacts
- **EventLog**: append-only record of input intent (DomainOps + EditCommands).
- **Snapshot**: current resolved state (notes/clips/lanes) derived from EventLog.
- **Diff**: the minimal summary of changes after a commit (created/updated/deleted + dirty time range + version).
- **DecisionRecord**: trace record with decision coordinates: x=time, y=lane/track, plus why/result.

---

## 1) Methodology (the pipeline that does not lie)

### 1.1 The three-stage buffer model
You don’t have “one queue”. You have *stacked buffers* with explicit responsibility:

1) **InputQueue (raw)**  
   Accepts everything: live MIDI, file playback cues, UI edits, AI edits, parameters.

2) **PendingOps (normalized)**  
   A staging buffer of DomainOps/EditCommands ready for deterministic application.

3) **EventWheel (scheduled, circular)**  
   A ring of future-due events aligned to audio blocks (or time slots in simulation).

Additional supporting buffers:
- **DecisionBuffer**: ring of DecisionRecords for debugging/inspection.
- **LateBuffer**: events that missed their slot (debug).

### 1.2 Where to drain the bus
Drain the bus in the app’s authoritative tick (e.g., `EngineApp.on_render()`), **after dt/frame creation** and **before** simulation/render planning.

Order per tick:
1) drain bus
2) normalize → PendingOps
3) commit once → Diff
4) observers run after commit (call/response)
5) scheduler ensures lookahead window filled (writes EventWheel)
6) engine continues (render, etc.)

### 1.3 The “switch to main mix”
The switch is **not scheduling**. The switch is **consumption**:
- Planning: `scheduler.ensure_window(now, now+lookahead)`
- Mixing: `audio_sink.consume_block(sample_start, block_n)` reads EventWheel and applies events at offsets.

### 1.4 Simulation space (no side effects)
Run the same pipeline, swap only the sink:
- `AudioSink`: produces sound / alters mixer
- `NullSink`: records fired events and derived state only

This makes the app “not realize it’s music”; it just processes timed events.

---

## 2) The block of space = period of section

### 2.1 What “block of space” means
A **block** is a *bounded time region* used for:
- committing in musically meaningful chunks
- scheduling in lookahead windows
- caching / invalidation (dirty ranges)
- “verse”/“phrase” semantics

Examples:
- 1 bar (tight lookahead)
- 4 bars (phrase)
- 8 bars (phrase)
- 16 bars (verse)

### 2.2 Two distinct block types (don’t mix them)
You need both:

#### A) Scheduling block (lookahead window)
Used to fill the EventWheel (future).
- Usually **1 bar** to **4 bars** lookahead.
- Updated continuously: every tick/block.

#### B) Commit section block (checkpoint)
Used to stabilize history and cache.
- Usually **4/8/16 bars** or “Section boundaries” (Intro/Verse/Chorus).
- Occurs when you cross a boundary, or on demand.

**Rule:** Scheduling blocks run frequently; commit blocks run less often.

### 2.3 How to implement blocks (minimal)
Start with seconds-based time and a fixed BPM. Later upgrade to ticks/PPQ.

Let:
- `spb = 60 / bpm` seconds per beat
- `beats_per_bar = 4` (initial)
- `bar_seconds = spb * beats_per_bar`

Then:
- Current bar index: `bar = floor(now / bar_seconds)`
- Bar start time: `bar_t0 = bar * bar_seconds`
- Bar end time: `bar_t1 = bar_t0 + bar_seconds`

#### Scheduling window (1 bar)
- `w0 = now`
- `w1 = bar_t1` (end of current bar)  
  OR `w1 = now + 1 * bar_seconds` (rolling)

#### Section period (N bars)
- `section_len_bars = 4 or 8 or 16`
- `section_index = floor(bar / section_len_bars)`
- `section_t0 = section_index * section_len_bars * bar_seconds`
- `section_t1 = section_t0 + section_len_bars * bar_seconds`

### 2.4 How to *utilize* the section period
Use it in three places:

1) **Commit cadence**  
   - When you enter a new section: `commit_section(previous_section)` (or just commit once and tag records).

2) **Cache & invalidation boundaries**  
   - Dirty ranges snap outward to section boundaries for efficient rebuild:
     - dirty_range = expand_to_section(dirty_range)

3) **Storage / pattern library**  
   - “MIDI pattern” is an EventLog slice + metadata tied to a section:
     - `pattern = eventlog[ops_in_section]`
     - `pattern.meta = {bpm, bars, tags, instrument_map}`

---

## 3) TODO list (implementation sequence)

### 3.1 Drain / commit boundary
- [ ] Identify the authoritative tick in the app (e.g., `EngineApp.on_render`).
- [ ] Add `bus.drain()` call after dt/frame creation.
- [ ] Normalize drained events into DomainOps/EditCommands.
- [ ] Push into sequencer pending ops.
- [ ] Call `sequencer.commit()` once per tick (or per boundary).

### 3.2 “Switch to main mix” sink interface
- [ ] Provide `NullSink` (simulation) + `Aud- [ ] Implement `EventSink` interface: `handle_edge_event(...)` and/or `consume_block(...)`.
ioSink` (real).

### 3.3 Circular buffer MIDI scheduling (EventWheel)
- [ ] Implement sample-index EventWheel:
  - slot = sample_index % R
- [ ] Implement timeline→sample mapping:
  - sample = floor(t_seconds * sample_rate)
- [ ] Scheduler compiles lookahead windows into EdgeEvents and writes to wheel.
- [ ] Consumer processes wheel buckets per audio block.

### 3.4 Decision coordinates (x/y trace)
- [ ] Define `DecisionRecord`:
  - x=time (sec/ticks/sample), y=lane/track, rule, result IDs
- [ ] Add `DecisionBuffer` ring.
- [ ] Instrument:
  - after drain, after commit (Diff), after compile window, after wheel write, after consume.

### 3.5 Pattern saving (abstract contract)
- [ ] Define `PatternSource` contract:
  - export/import EventLog slices
  - store meta: section_len_bars, bpm, tags
- [ ] Save/load patterns from disk (JSON/MsgPack) including:
  - eventlog slice + section metadata + tags

### 3.6 Section blocks
- [ ] Implement section math (bar_seconds, section_t0/t1).
- [ ] Add “commit on section boundary” option:
  - every 4/8/16 bars
- [ ] Expand dirty ranges to section boundaries for rebuild.

### 3.7 Sound + visual “qualities” tags (round-trip)
- [ ] Define tag vocabulary (e.g., punch, warm, bright; glow, snap, flow).
- [ ] Map tags → audio parameter packs (EQ, envelope, saturation).
- [ ] Map tags → visual parameter packs (bloom, easing, jitter).
- [ ] Store tags in EventLog/Snapshot so they round-trip.

---

## 4) Outline: building and using the “block of space == period of section”

### Step 1 — Choose the smallest useful block
- Start with **1 bar** scheduling lookahead:
  - goal: keep wheel filled for the next bar.
- Choose **4 bars** commit period:
  - goal: phrase checkpoints.

### Step 2 — Add the math helpers
Implement helpers:
- `bar_index(now)`
- `bar_bounds(now)`
- `section_bounds(now, section_len_bars)`

### Step 3 — Scheduling uses the block end
On each tick:
- compute `w1 = end_of_current_bar` (or `now + 1 bar`)
- compile schedule for `[now, w1]`
- write to EventWheel

### Step 4 — Commit uses section boundaries
When `section_index` changes:
- call `commit()` (and optionally “seal” the prior section)
- snapshot/cache indices per section
- emit DecisionRecords

### Step 5 — Pattern library = section slices
To save a pattern:
- choose a section `[section_t0, section_t1]`
- export EventLog ops whose times fall inside
- save with metadata: bpm, section_len_bars, tags

To load a pattern:
- import ops and time-shift to target section
- push ops into PendingOps and commit

### Step 6 — Simulation mode validates everything
Run with `NullSink`:
- verify events fire at correct times
- verify section boundaries cause the right commits
- verify DecisionRecords produce readable x/y trace

---

## 5) Acceptance checklist
- [ ] Bus drain happens once per tick at the defined boundary.
- [ ] EventLog and Snapshot stay consistent across replay.
- [ ] Lookahead scheduling fills exactly 1 bar (configurable).
- [ ] Section commits occur at N-bar boundaries (configurable).
- [ ] NullSink simulation produces the same fired event stream as AudioSink (minus sound).
- [ ] DecisionRecords show x=time and y=lane for commit/schedule/fire actions.

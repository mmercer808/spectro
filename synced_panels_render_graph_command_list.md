# Synced Sequencer + Wave Panels — 3D Graph Panels, TimeCamera, Render Graph, Command List

This document expands the design for a **3-panel** UI:

1. **Sync/Metronome Row** (free draw + overlay; does *not* scroll)
2. **Sequencer Panel** (free draw 2D/3D graph; *scrolls*)
3. **Wave Panel** (free draw 2D/3D graph; *scrolls in sync with sequencer*)

Key requirement: **SequencerPanel and WavePanel share a single time-moving state** so bar/beat alignment stays pixel-perfect while scrolling.

---

## 0) Mental Model

You are building a window that hosts *mini rendering scenes* (panels). Each panel is a clipped viewport that owns:

- A **Scene** (graph coordinate system; can be 3D even if rendered as “2D” primitives)
- An optional **Overlay** (HUD UI in panel-space)
- A **ViewModel cache** (decimation, bins, instanced geometry, etc.)

The only thing that must be shared between Sequencer and Wave panels is the **TimeCamera**.

---

## 1) The Shared TimeCamera (the “time-moving state”)

### Purpose
To guarantee the invariant:

> SequencerPanel.screenX(beat) == WavePanel.screenX(beat)

### Core fields
- `mode`: `"follow_playhead" | "free_scroll" | "snap_to_bars"`
- `left_beat`: the musical time at the left edge of the viewport
- `window_beats`: width of the visible window in beats (zoom)
- `px_per_beat`: derived from panel width and `window_beats`
- `playhead_beat`: current playhead position in beats
- `playhead_screen_x_mode`: `"fixed" | "moving"`
  - if `"fixed"`, playhead stays at e.g. 30% of panel width and content scrolls
- `follow_strength`: 0..1 smoothing toward target scroll position
- `user_scroll_active`: bool; prevents follow from fighting user drag
- `scroll_velocity`: for inertial scroll (optional)
- `snap_grid`: e.g. 1 beat or 4 beats (bar) when mode is snap-to-bars

### Mapping function (single source of truth)
Given a beat timestamp `b`:

- `x_px = (b - left_beat) * px_per_beat`

Both panels use this *exact* mapping for:
- bar lines
- beat lines
- event placement
- hit markers
- waveform / curve samples

### Updating left_beat
If playhead is fixed at `playhead_x_px` (e.g. 30% width), you compute a target left beat:

- `target_left = playhead_beat - playhead_x_px / px_per_beat`

Then you move toward it:

- `left_beat = lerp(left_beat, target_left, follow_strength)`

If the user is dragging:
- `left_beat -= drag_dx / px_per_beat`
- set `user_scroll_active = True` while dragging, then decay / release.

---

## 2) Panel as a 3D Graph Coordinate System

Each panel’s inner draw area is a **graph space with axes**:

- **X axis = Time** (beats)
- **Y axis = Lane/value**
  - sequencer: lane index / track row
  - wave: amplitude/value per lane or multiple stacked lanes
- **Z axis = Depth** (history layers, emphasis, parallax)

### Why keep Z?
Even if you render mostly 2D, Z gives you:
- history trails behind current (z < 0)
- “active” items forward (z > 0)
- subtle parallax/camera zoom later

### Camera
Use an **orthographic camera** per panel (recommended):
- consistent measurement
- easy alignment
- predictable with UI overlays

The camera uses **TimeCamera** to set its view transform:
- view translates by `-left_beat` in world X
- projection scales X by `px_per_beat`

---

## 3) Sync Panel (Metronome Row) — free draw + overlay

### Behavior
- Does **not** scroll; no TimeCamera needed.
- Uses `TransportState` directly (phase in beat/bar).

### Two-layer design (clean separation)
1. **Base**: pure metronome display
   - rings, tick marks, phase arc/hand
2. **Overlay**: sync diagnostics
   - lock arc, drift dots, pulse glow, small arrows, confidence indicator

### Key state inputs
- `transport.playheadBeat`
- `transport.phaseInBeat` / `phaseInBar`
- `sync.driftMs`, `sync.lockConfidence`
- `pulseEnvelope` (decays after beat boundaries)

---

## 4) Sequencer Panel — synced scroll, discrete musical events

### Behavior
- Uses **TimeCamera** (shared with WavePanel).
- Draws musical grid + discrete events.

### Scene content (3D graph)
- bar & beat grid lines (world X)
- events as quads/rectangles at (beat, lane)
- optional: velocity → height or brightness
- optional: note duration → width in X

### Overlay content (2D HUD)
- selection rectangles
- hover highlights
- edit handles
- playhead (if you choose fixed playhead overlay style)
- lane labels pinned to left edge (panel-space)

### Key state inputs
- `timeCamera.left_beat, window_beats, px_per_beat`
- `seq.events[]` (note onsets/durations/velocity)
- `seq.selection`, `seq.hover`
- `transport.playheadBeat`

---

## 5) Wave Panel — synced scroll, continuous signals + markers

### Behavior
- Uses **TimeCamera** shared with SequencerPanel.
- Draws continuous data (waveforms/curves) plus markers.

### Scene content (3D graph)
- bar & beat grid lines (world X)
- waveform lines (line strips) per lane
- groove curve (smoothed line)
- timing error curve (line or bars)
- hit markers (instanced vertical lines/sprites)

### Overlay content (2D HUD)
- fixed playhead line (recommended)
- crosshair and readouts
- lane titles / scales
- tooltip boxes

### The critical performance rule
Waveforms should be drawn from **decimated caches**:
- For each visible window, build a per-pixel min/max envelope
- Update incrementally as new samples arrive
- Recompute only when zoom changes materially

Key state inputs:
- `audioRingBuffer` / `curveRingBuffer`
- `derivedMinMaxCache` (per zoom level)
- `hitMarkers` timestamps (beats)
- `timeCamera` mapping for X

---

## 6) Shared Scrolling Semantics (two viable styles)

Both keep panels synced via TimeCamera.

### A) Fixed playhead, scrolling content (recommended)
- Playhead drawn in overlay at fixed X (e.g. 30% width)
- TimeCamera.left_beat changes each frame to keep playhead centered at that X
- Bar/beat grid and events “move” leftward under the playhead

### B) Moving playhead, fixed content
- TimeCamera.left_beat changes only via user scroll
- Playhead is a world-space object moving across both panels

Most training/feedback tools prefer **A** for readability.

---

## 7) Render Graph (Frame Graph)

Think “engine-style” graph nodes:

1. **InputNode**
   - poll MIDI/audio
   - push into ring buffers
2. **TransportNode**
   - advance playhead (beats)
   - compute phase, beat boundaries
3. **SyncNode**
   - compute drift, lock confidence, pulse envelopes
4. **TimeCameraNode**
   - update left_beat based on mode + playhead + user scroll
5. **ViewModelNode**
   - Sequencer: update event batches if dirty
   - Wave: update decimation caches / markers batches
6. **LayoutNode**
   - compute panel rects and clip/scissor
7. **RenderNode**
   - execute command list (see end)

---

## 8) State Properties That Drive Visual Change (label list)

### Global (shared)
- `transport.playing`
- `transport.playheadBeat`
- `transport.bpm`, `timeSig`
- `timeCamera.left_beat`
- `timeCamera.window_beats` (zoom)
- `timeCamera.mode`, `user_scroll_active`
- `ui.theme` (rare)

### SyncPanel
- `sync.lockConfidence`
- `sync.driftMs`
- `pulseEnvelope` per beat subdivision

### SequencerPanel
- `seq.events` (dirty flags)
- `seq.selection`, `seq.hover`
- `seq.editMode` (draw/erase/drag)

### WavePanel
- `audioRingBuffer.writeIndex`
- `wave.decimationCache` (dirty flags)
- `hitMarkers` updates
- `cursorCrosshair` (mouse move)

---

## 9) Practical “Panel = Free Draw” Implementation Detail

Each panel provides:
- `rect` (viewport)
- `clipRect` (scissor)
- `panelCamera` (orthographic; uses TimeCamera mapping in X)
- `draw(sceneCtx)` (graph primitives)
- `drawOverlay(overlayCtx)` (2D primitives, text)

The window orchestrates:
- layout
- per-panel viewport/scissor
- pass ordering

---

## 10) Command List Sequence (end-to-end)

Below is a concrete **command list** you can implement using your engine’s style (render items / command buffers). The idea is to record a list of commands each frame and execute them in order.

### Data structures (conceptual)
- `CmdSetRenderTarget(rt)`
- `CmdClear(color, depth)`
- `CmdSetViewport(x,y,w,h)`
- `CmdSetScissor(x,y,w,h)`
- `CmdSetPipeline(pipelineId)`  (blend/depth/shader)
- `CmdBindResources(bindGroupId)` (UBOs, textures, buffers)
- `CmdDraw(meshId, first, count)` or instanced variant
- `CmdDrawText(textRunId)`
- `CmdPushDebugLabel(name)` / `CmdPopDebugLabel()`

### Passes and order
**Pass 0 — Frame setup**
1. `SetRenderTarget(Backbuffer)`
2. `Clear(color=bg, depth=1.0)`

**Pass 1 — UI chrome (panel backplates/borders)**
3. `SetPipeline(UI_CHROME)`
4. For each panel in z-order:
   - `SetViewport(panel.rect)`
   - `SetScissor(panel.clip)`
   - `BindResources(themeUBO)`
   - `Draw(roundedRectMesh, ...)`

**Pass 2 — SyncPanel Scene**
5. `PushLabel("SyncPanel")`
6. `SetViewport(syncRect)` + `SetScissor(syncRect)`
7. `SetPipeline(SYNC_BASE_2D)`
8. `BindResources(syncUBO)`
9. `Draw(ringsMesh/instancedArcs, ...)`
10. `SetPipeline(SYNC_OVERLAY_2D)`
11. `BindResources(syncOverlayUBO)`
12. `Draw(lockArcMesh/dots, ...)`
13. `PopLabel()`

**Pass 3 — SequencerPanel Scene (3D graph in panel)**
14. `PushLabel("SequencerPanel")`
15. `SetViewport(seqRect)` + `SetScissor(seqRect)`
16. `SetPipeline(GRID_LINES)`
17. `BindResources(timeCameraUBO + seqGridUBO)`
18. `Draw(gridLineMesh, ...)`
19. `SetPipeline(SEQUENCER_EVENTS)`
20. `BindResources(timeCameraUBO + seqEventsBuffer)`
21. `DrawInstanced(eventQuadMesh, instanceCount=...)`
22. `SetPipeline(SEQUENCER_OVERLAY_2D)`
23. `BindResources(seqOverlayUBO)`
24. `Draw(selectionRects/playheadOverlay, ...)`
25. `PopLabel()`

**Pass 4 — WavePanel Scene (3D graph in panel)**
26. `PushLabel("WavePanel")`
27. `SetViewport(waveRect)` + `SetScissor(waveRect)`
28. `SetPipeline(GRID_LINES)`
29. `BindResources(timeCameraUBO + waveGridUBO)`
30. `Draw(gridLineMesh, ...)`
31. `SetPipeline(WAVE_LINES)`
32. `BindResources(timeCameraUBO + waveLineVBO)`
33. `Draw(lineStripMesh, ...)` (or multiple lanes)
34. `SetPipeline(HIT_MARKERS)`
35. `BindResources(timeCameraUBO + markerInstances)`
36. `DrawInstanced(markerMesh, instanceCount=...)`
37. `SetPipeline(WAVE_OVERLAY_2D)`
38. `BindResources(waveOverlayUBO)`
39. `Draw(playhead/crosshair/readouts, ...)`
40. `PopLabel()`

**Pass 5 — Text**
41. `SetPipeline(TEXT_ALPHA)`
42. `BindResources(fontAtlas + textUBO)`
43. For each text run (panel titles, BPM, ms readouts):
   - `SetViewport(targetPanelRect)` + `SetScissor(targetPanelRect)`
   - `DrawText(textRunId)`

**Pass 6 — Present**
44. `Present()`

### Notes
- The **TimeCameraUBO** is bound in both Sequencer and Wave passes so they share scroll/zoom.
- Bar/beat grid lines are drawn in both panels using the same mapping, guaranteeing alignment.
- Overlays can be 2D pipelines with no depth testing for simplicity.
- If you want 3D depth layers, enable depth only for the panel scene passes (not overlays).

---

## 11) Implementation Hint: One Builder Produces Both Panels’ Grid
To avoid mismatched grid math, build grid lines from a shared helper:

- `GridBuilder.build(timeCamera, rectWidth) -> gridLineInstances`

Then both panels call it and draw the same line positions.

---

## 12) What to implement first (minimal but correct)
1. Implement `TimeCamera` and the mapping `beat -> x_px`
2. Implement panel rects + scissor/viewport
3. Draw bar/beat grid lines in both panels (verify alignment)
4. Add a fixed playhead overlay line in both panels
5. Add sequencer events quads
6. Add wave line rendering using decimated data

Once those work, everything else layers on safely.

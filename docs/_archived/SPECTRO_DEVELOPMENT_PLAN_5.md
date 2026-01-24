# Spectro UI Engine — Development Plan

> **Purpose**: This document serves as the authoritative implementation guide for Claude Code / Cursor.
> Commit this to the repository root. Reference it when starting work sessions.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Component Specifications](#3-component-specifications)
4. [Implementation Phases](#4-implementation-phases)
5. [Detailed Task List](#5-detailed-task-list)
6. [Code Templates](#6-code-templates)
7. [Testing Strategy](#7-testing-strategy)
8. [Worked Example](#8-worked-example)
9. [Critical Invariants](#9-critical-invariants)
10. [File Structure](#10-file-structure)

---

## 1. Project Overview

### 1.1 What We're Building

A **time-synchronized panel system** for audio visualization and music production tools:

- **SyncPanel**: Metronome/phase display (no scroll)
- **SequencerPanel**: Discrete musical events on a grid (scrolls)
- **WavePanel**: Continuous waveform + hit markers (scrolls in sync)

**Key Requirement**: SequencerPanel and WavePanel share a single `TimeCamera` so bar/beat alignment stays pixel-perfect while scrolling.

### 1.2 Design Principles

1. **Immutable snapshots for rendering** — Mutable graph for authoring, frozen snapshots for render thread
2. **Pure-data command lists** — No GL objects in commands; thread-safe by construction
3. **Single source of truth for time** — `TimeCamera` is THE mapping from beats to pixels
4. **Instanced drawing** — Batch similar primitives for GPU efficiency

### 1.3 Existing Codebase

The engine already has:
- ✅ Scene graph with RCU-style snapshots (`graph.py`, `snapshot.py`)
- ✅ CommandList system for 3D (`commands.py`)
- ✅ Renderer with picking support (`renderer.py`)
- ✅ Render target pooling (`targets.py`)
- ✅ Upload queue with versioning (`uploader.py`)
- ✅ Flexbox layout engine (`layout.py`)
- ✅ CSS-like style system (`style.py`)
- ⚠️ UI widgets (partial — need `widget.py`, `draw.py` review)
- ❌ TimeCamera (not implemented)
- ❌ Transport state (not implemented)
- ❌ 2D drawing commands (not implemented)
- ❌ GraphPanel system (not implemented)

---

## 2. Architecture

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              APPLICATION                                     │
│  main.py — owns Transport, AudioEngine, creates WindowManager               │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRAME LOOP (per frame)                            │
│                                                                              │
│  1. Input     →  2. Transport  →  3. TimeCamera  →  4. Panels  →  5. Render │
│  (poll)          (advance)        (update scroll)    (draw)        (execute)│
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UI LAYER                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        WindowManager                                 │    │
│  │  ┌─────────────┐  ┌─────────────────────────────────────────────┐   │    │
│  │  │  SyncPanel  │  │              DockSplit                      │   │    │
│  │  │ (metronome) │  │  ┌──────────────────┐ ┌──────────────────┐  │   │    │
│  │  │             │  │  │  SequencerPanel  │ │    WavePanel     │  │   │    │
│  │  │ NO SCROLL   │  │  │   (scrolls)      │ │   (scrolls)      │  │   │    │
│  │  └─────────────┘  │  │                  │ │                  │  │   │    │
│  │                   │  │  ◄─── TimeCamera shared ───►          │  │   │    │
│  │                   │  └──────────────────┘ └──────────────────┘  │   │    │
│  │                   └─────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RENDER PIPELINE                                    │
│                                                                              │
│   DrawBatch   ──▶  UIRenderer  ──▶  GL                                      │
│   (2D cmds)        (executes)       (screen)                                │
│                                                                              │
│   Snapshot ──▶ CommandList ──▶ Renderer ──▶ RenderTarget                    │
│   (3D scene)   (3D cmds)       (executes)   (texture)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Transport.update(dt)
       │
       ▼
TransportState (immutable snapshot)
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
TimeCamera.update(playhead)         SyncPanel.update(transport)
       │
       ├──────────────────────────────────┐
       ▼                                  ▼
SequencerPanel.draw(ctx)           WavePanel.draw(ctx)
       │                                  │
       │    ┌─────────────────────────────┘
       ▼    ▼
   GridBuilder.build(time_camera, rect)  ← SHARED, guarantees alignment
       │
       ▼
   DrawBatch (2D commands)
       │
       ▼
   UIRenderer.execute(batch)
       │
       ▼
   Screen
```

### 2.3 Key Insight: Why TimeCamera Must Be Shared

```python
# WRONG: Each panel computes its own mapping
sequencer_x = (beat - sequencer_left) * sequencer_px_per_beat
wave_x = (beat - wave_left) * wave_px_per_beat
# These can drift apart!

# RIGHT: Single TimeCamera, single mapping
x = time_camera.beat_to_px(beat)
# Both panels use this exact function
```

---

## 3. Component Specifications

### 3.1 TimeCamera

**File**: `engine/time/camera.py`

**Purpose**: Single source of truth for beat ↔ pixel mapping.

**State**:
```python
@dataclass
class TimeCamera:
    # Core state
    mode: TimeCameraMode          # FOLLOW_PLAYHEAD | FREE_SCROLL | SNAP_TO_BARS
    left_beat: float              # Beat at left edge of viewport
    window_beats: float           # How many beats visible (zoom level)
    
    # Derived (computed each frame)
    _panel_width_px: float        # Set by layout
    _px_per_beat: float           # panel_width / window_beats
    
    # Interaction
    user_scroll_active: bool      # True while dragging
    scroll_velocity: float        # For inertia
    
    # Config
    config: TimeCameraConfig      # Playhead ratio, follow strength, etc.
```

**Core Methods**:
```python
def beat_to_px(self, beat: float) -> float:
    """THE canonical mapping. All panels use this."""
    return (beat - self.left_beat) * self._px_per_beat

def px_to_beat(self, px: float) -> float:
    """Inverse mapping."""
    return self.left_beat + (px / self._px_per_beat)

def update(self, dt: float, playhead_beat: float, panel_width_px: float, frame_id: int):
    """Called each frame to update scroll position."""
    
def begin_drag(self, mouse_x: float): ...
def update_drag(self, mouse_x: float): ...
def end_drag(self, mouse_x: float, velocity: float): ...
def zoom(self, delta: float, anchor_px: float): ...
```

**Modes**:
- `FOLLOW_PLAYHEAD`: Content scrolls, playhead stays at fixed screen position (30%)
- `FREE_SCROLL`: User controls scroll, playhead moves across screen
- `SNAP_TO_BARS`: Like follow, but jumps in bar-sized increments

---

### 3.2 Transport

**File**: `engine/time/transport.py`

**Purpose**: Playback control and timing.

**State**:
```python
class Transport:
    playing: bool
    playhead_beat: float
    playhead_time: float          # Seconds
    bpm: float
    time_sig: TimeSignature       # (numerator, denominator)
    
    # Looping
    loop_start: Optional[float]
    loop_end: Optional[float]
    loop_enabled: bool
```

**Immutable Snapshot** (created each frame for thread safety):
```python
@dataclass(frozen=True)
class TransportState:
    playing: bool
    playhead_beat: float
    playhead_time: float
    bpm: float
    time_sig: TimeSignature
    loop_start: Optional[float]
    loop_end: Optional[float]
    loop_enabled: bool
    
    # Derived
    phase_in_beat: float          # 0.0 - 1.0
    phase_in_bar: float           # 0.0 - 1.0
    current_bar: int
    current_beat_in_bar: int
```

---

### 3.3 2D Drawing Commands

**File**: `engine/render/commands_2d.py`

**Commands**:
```python
# Clipping
Cmd2DSetClip(x, y, w, h, enabled)

# Primitives
Cmd2DRect(x, y, w, h, color, corner_radius, border_width, border_color)
Cmd2DLine(x0, y0, x1, y1, color, width)
Cmd2DLineStrip(points: bytes, point_count, color, width, closed)
Cmd2DCircle(cx, cy, radius, fill_color, stroke_color, stroke_width)
Cmd2DArc(cx, cy, radius, start_angle, end_angle, color, width)

# Instanced (for efficiency)
Cmd2DRectsInstanced(instance_data: bytes, instance_count, corner_radius)
Cmd2DLinesInstanced(instance_data: bytes, instance_count)

# Text
Cmd2DText(text, x, y, color, font_size, font_id, align, baseline)

# Texture
Cmd2DBlit(texture_key, dst_x, dst_y, dst_w, dst_h, src_uv, tint)

# Waveform (optimized)
Cmd2DWaveform(envelope_data: bytes, sample_count, x_start, x_scale, y_center, y_scale, fill_color, stroke_color)
```

**DrawBatch** (container):
```python
@dataclass
class DrawBatch:
    commands: List[Command2D]
    panel_id: Optional[str]
    z_order: int
    
    def rect(self, x, y, w, h, color, **kwargs): ...
    def line(self, x0, y0, x1, y1, color, width=1.0): ...
    def text(self, text, x, y, color, **kwargs): ...
    def vertical_lines(self, x_positions, y0, y1, color, width=1.0): ...
```

---

### 3.4 GridBuilder

**File**: `engine/ui/panels/grid_builder.py`

**Purpose**: Generate grid lines for any panel using TimeCamera.

**CRITICAL**: Both SequencerPanel and WavePanel use THE SAME GridBuilder instance to guarantee pixel-perfect alignment.

```python
class GridBuilder:
    def __init__(self, config: GridConfig): ...
    
    def build(
        self,
        time_camera: TimeCamera,
        rect: Rect,
        beats_per_bar: float = 4.0,
    ) -> DrawBatch:
        """Generate grid line commands."""
        # 1. Determine subdivision level based on zoom
        # 2. Collect bar line X positions
        # 3. Collect beat line X positions
        # 4. Collect subdivision line X positions
        # 5. Emit instanced line commands (back to front)
```

**GridConfig**:
```python
@dataclass
class GridConfig:
    bar_color: Color4 = (0.4, 0.4, 0.45, 1.0)
    bar_width: float = 1.5
    beat_color: Color4 = (0.25, 0.25, 0.3, 1.0)
    beat_width: float = 1.0
    subdivision_color: Color4 = (0.18, 0.18, 0.22, 0.5)
    subdivision_width: float = 0.5
    
    # Zoom thresholds: (px_per_beat, subdivision_level)
    subdivision_thresholds: List[Tuple[float, int]] = [
        (100.0, 4),   # 16ths when >= 100 px/beat
        (50.0, 2),    # 8ths when >= 50 px/beat
        (20.0, 1),    # Quarters when >= 20 px/beat
        (0.0, 0),     # Only bars otherwise
    ]
```

---

### 3.5 GraphPanel Base Class

**File**: `engine/ui/panels/graph_panel.py`

**Purpose**: Base class for time-synchronized panels.

```python
class GraphPanel(Widget, ABC):
    def __init__(
        self,
        time_camera: TimeCamera,          # SHARED reference
        grid_config: GridConfig = None,
        playhead_config: PlayheadConfig = None,
    ): ...
    
    # Widget interface
    def measure(self, constraints) -> Tuple[float, float]: ...
    def layout(self, rect: Rect): ...
    def draw(self, ctx: DrawContext): ...
    
    # Drawing sequence (in draw()):
    # 1. Background
    # 2. Set clip to content area
    # 3. Grid lines (via GridBuilder)
    # 4. draw_content() — subclass implements
    # 5. Playhead
    # 6. draw_overlay() — subclass implements
    # 7. Clear clip
    
    @abstractmethod
    def draw_content(self, ctx: DrawContext): ...
    
    def draw_overlay(self, ctx: DrawContext): ...  # Optional override
    
    # Input handling
    def on_pointer_down(self, x, y, button) -> bool: ...
    def on_pointer_move(self, x, y, dx, dy) -> bool: ...
    def on_pointer_up(self, x, y, button) -> bool: ...
    def on_scroll(self, x, y, dx, dy) -> bool: ...
```

---

### 3.6 SequencerPanel

**File**: `engine/ui/panels/sequencer_panel.py`

```python
class SequencerPanel(GraphPanel):
    lanes: List[SequencerLane]
    events: List[SequencerEvent]
    selected_ids: set[int]
    
    def draw_content(self, ctx):
        # 1. Draw lane backgrounds (alternating)
        # 2. Collect visible events
        # 3. Build instanced rect data
        # 4. Emit Cmd2DRectsInstanced
    
    def draw_overlay(self, ctx):
        # 1. Hover highlight
        # 2. Selection outlines

@dataclass
class SequencerEvent:
    id: int
    beat: float
    duration: float
    lane: int
    velocity: float
    selected: bool = False

@dataclass
class SequencerLane:
    id: int
    name: str
    color: Color4
    height: float = 24.0
```

---

### 3.7 WavePanel

**File**: `engine/ui/panels/wave_panel.py`

```python
class WavePanel(GraphPanel):
    waveform: WaveformCache
    hit_markers: List[HitMarker]
    
    def draw_content(self, ctx):
        # 1. Center line
        # 2. Waveform (via Cmd2DWaveform)
        # 3. Hit markers
    
    def draw_overlay(self, ctx):
        # 1. Crosshair at hover position
        # 2. Beat readout

@dataclass
class WaveformCache:
    sample_rate: float
    bpm: float
    raw_buffer: np.ndarray
    decimation_cache: dict[int, np.ndarray]  # samples_per_px -> envelope
    
    def get_envelope(self, samples_per_px: int) -> np.ndarray: ...

@dataclass
class HitMarker:
    beat: float
    confidence: float = 1.0
    color: Color4 = (1.0, 0.8, 0.2, 1.0)
```

---

### 3.8 SyncPanel (Metronome)

**File**: `engine/ui/panels/sync_panel.py`

```python
class SyncPanel(Widget):  # NOT GraphPanel — no TimeCamera
    """
    Metronome display. Does NOT scroll.
    Uses TransportState directly for phase.
    """
    
    def draw(self, ctx):
        # 1. Background ring
        # 2. Beat tick marks (4 or 8)
        # 3. Phase arc/hand
        # 4. Beat number indicator
        # 5. Pulse glow on beat boundaries
```

---

## 4. Implementation Phases

### Phase 1: Core Time Infrastructure (Week 1, Days 1-3)

**Goal**: TimeCamera and Transport working with unit tests.

**Deliverables**:
- [ ] `engine/time/__init__.py`
- [ ] `engine/time/camera.py` — TimeCamera class
- [ ] `engine/time/transport.py` — Transport and TransportState
- [ ] `tests/test_time_camera.py`
- [ ] `tests/test_transport.py`

**Acceptance Criteria**:
- `beat_to_px()` and `px_to_beat()` are exact inverses
- FOLLOW_PLAYHEAD mode smoothly tracks playhead
- Drag and zoom work correctly
- Transport advances correctly with looping

---

### Phase 2: 2D Drawing System (Week 1, Days 4-5)

**Goal**: 2D commands rendering to screen.

**Deliverables**:
- [ ] `engine/render/commands_2d.py` — All 2D command types
- [ ] `engine/render/shaders/rect.glsl` — Rounded rect shader
- [ ] `engine/render/shaders/line.glsl` — Line shader
- [ ] `engine/render/shaders/waveform.glsl` — Waveform fill shader
- [ ] `engine/ui/renderer_2d.py` — 2D command executor

**Acceptance Criteria**:
- Can draw filled rects with corner radius
- Can draw lines and line strips
- Instanced draws work for many rects/lines
- Scissor clipping works

---

### Phase 3: GraphPanel System (Week 2, Days 1-3)

**Goal**: Working synchronized panels.

**Deliverables**:
- [ ] `engine/ui/panels/__init__.py`
- [ ] `engine/ui/panels/grid_builder.py`
- [ ] `engine/ui/panels/graph_panel.py`
- [ ] `engine/ui/panels/sequencer_panel.py`
- [ ] `engine/ui/panels/wave_panel.py`
- [ ] `engine/ui/panels/sync_panel.py`

**Acceptance Criteria**:
- Grid lines align EXACTLY between SequencerPanel and WavePanel
- Playhead position is identical in both panels
- Scroll and zoom affect both panels simultaneously
- Events render correctly in SequencerPanel

---

### Phase 4: Widget Integration (Week 2, Days 4-5)

**Goal**: Panels working in UI framework.

**Deliverables**:
- [ ] Update `engine/ui/widget.py` — Add GraphPanel support
- [ ] Update `engine/ui/draw.py` — DrawContext with DrawBatch
- [ ] Update `engine/ui/renderer.py` — Execute 2D batches
- [ ] Update `engine/ui/window_manager.py` — Route input to panels

**Acceptance Criteria**:
- Panels can be added to DockLayout
- Mouse events route correctly to panels
- Keyboard shortcuts work (space = play/pause)

---

### Phase 5: Audio Integration (Week 3)

**Goal**: Real waveform display.

**Deliverables**:
- [ ] `engine/audio/__init__.py`
- [ ] `engine/audio/buffer.py` — Ring buffer
- [ ] `engine/audio/waveform.py` — WaveformCache with decimation
- [ ] `engine/audio/engine.py` — Audio playback (stub or real)

**Acceptance Criteria**:
- Waveform renders at multiple zoom levels
- Decimation cache updates incrementally
- No visual glitches during playback

---

### Phase 6: Demo Application (Week 3)

**Goal**: Working spectro demo.

**Deliverables**:
- [ ] `examples/spectro_demo.py`
- [ ] Transport control UI (play/pause/stop buttons)
- [ ] BPM and time signature display
- [ ] `docs/ARCHITECTURE.md`

**Acceptance Criteria**:
- Three panels visible: Sync, Sequencer, Wave
- Playback works with visual sync
- Can scroll and zoom
- Performance is smooth (60 FPS)

---

## 5. Detailed Task List

### 5.1 Phase 1: Time Infrastructure

| ID | Task | File | Status |
|----|------|------|--------|
| 1.1 | Create time module | `engine/time/__init__.py` | ⬜ |
| 1.2 | Define TimeCameraMode enum | `engine/time/camera.py` | ⬜ |
| 1.3 | Define TimeCameraConfig dataclass | `engine/time/camera.py` | ⬜ |
| 1.4 | Implement TimeCamera.beat_to_px() | `engine/time/camera.py` | ⬜ |
| 1.5 | Implement TimeCamera.px_to_beat() | `engine/time/camera.py` | ⬜ |
| 1.6 | Implement TimeCamera.update() | `engine/time/camera.py` | ⬜ |
| 1.7 | Implement TimeCamera._update_follow_mode() | `engine/time/camera.py` | ⬜ |
| 1.8 | Implement TimeCamera._update_snap_mode() | `engine/time/camera.py` | ⬜ |
| 1.9 | Implement TimeCamera.begin_drag() | `engine/time/camera.py` | ⬜ |
| 1.10 | Implement TimeCamera.update_drag() | `engine/time/camera.py` | ⬜ |
| 1.11 | Implement TimeCamera.end_drag() | `engine/time/camera.py` | ⬜ |
| 1.12 | Implement TimeCamera.zoom() | `engine/time/camera.py` | ⬜ |
| 1.13 | Implement TimeCamera.get_visible_bar_lines() | `engine/time/camera.py` | ⬜ |
| 1.14 | Implement TimeCamera.get_visible_beat_lines() | `engine/time/camera.py` | ⬜ |
| 1.15 | Define TimeSignature dataclass | `engine/time/transport.py` | ⬜ |
| 1.16 | Define TransportState frozen dataclass | `engine/time/transport.py` | ⬜ |
| 1.17 | Implement Transport class | `engine/time/transport.py` | ⬜ |
| 1.18 | Implement Transport.update() | `engine/time/transport.py` | ⬜ |
| 1.19 | Implement Transport looping logic | `engine/time/transport.py` | ⬜ |
| 1.20 | Implement Transport.on_beat() callback | `engine/time/transport.py` | ⬜ |
| 1.21 | Write test: beat_to_px / px_to_beat inverse | `tests/test_time_camera.py` | ⬜ |
| 1.22 | Write test: follow mode tracking | `tests/test_time_camera.py` | ⬜ |
| 1.23 | Write test: drag scroll | `tests/test_time_camera.py` | ⬜ |
| 1.24 | Write test: zoom with anchor | `tests/test_time_camera.py` | ⬜ |
| 1.25 | Write test: transport advance | `tests/test_transport.py` | ⬜ |
| 1.26 | Write test: transport looping | `tests/test_transport.py` | ⬜ |

### 5.2 Phase 2: 2D Drawing

| ID | Task | File | Status |
|----|------|------|--------|
| 2.1 | Define Color4 type | `engine/render/commands_2d.py` | ⬜ |
| 2.2 | Define Cmd2DSetClip | `engine/render/commands_2d.py` | ⬜ |
| 2.3 | Define Cmd2DRect | `engine/render/commands_2d.py` | ⬜ |
| 2.4 | Define Cmd2DLine | `engine/render/commands_2d.py` | ⬜ |
| 2.5 | Define Cmd2DLineStrip | `engine/render/commands_2d.py` | ⬜ |
| 2.6 | Define Cmd2DCircle | `engine/render/commands_2d.py` | ⬜ |
| 2.7 | Define Cmd2DArc | `engine/render/commands_2d.py` | ⬜ |
| 2.8 | Define Cmd2DRectsInstanced | `engine/render/commands_2d.py` | ⬜ |
| 2.9 | Define Cmd2DLinesInstanced | `engine/render/commands_2d.py` | ⬜ |
| 2.10 | Define Cmd2DText | `engine/render/commands_2d.py` | ⬜ |
| 2.11 | Define Cmd2DBlit | `engine/render/commands_2d.py` | ⬜ |
| 2.12 | Define Cmd2DWaveform | `engine/render/commands_2d.py` | ⬜ |
| 2.13 | Implement DrawBatch class | `engine/render/commands_2d.py` | ⬜ |
| 2.14 | Implement DrawBatch.vertical_lines() helper | `engine/render/commands_2d.py` | ⬜ |
| 2.15 | Write rect vertex shader | `engine/render/shaders/rect.vert` | ⬜ |
| 2.16 | Write rect fragment shader (rounded corners) | `engine/render/shaders/rect.frag` | ⬜ |
| 2.17 | Write line vertex shader | `engine/render/shaders/line.vert` | ⬜ |
| 2.18 | Write line fragment shader | `engine/render/shaders/line.frag` | ⬜ |
| 2.19 | Write instanced rect shader | `engine/render/shaders/rect_instanced.vert` | ⬜ |
| 2.20 | Write instanced line shader | `engine/render/shaders/line_instanced.vert` | ⬜ |
| 2.21 | Write waveform shader | `engine/render/shaders/waveform.vert` | ⬜ |
| 2.22 | Write waveform fragment shader | `engine/render/shaders/waveform.frag` | ⬜ |
| 2.23 | Create UIRenderer2D class | `engine/ui/renderer_2d.py` | ⬜ |
| 2.24 | Implement UIRenderer2D._setup_pipelines() | `engine/ui/renderer_2d.py` | ⬜ |
| 2.25 | Implement UIRenderer2D.execute(batch) | `engine/ui/renderer_2d.py` | ⬜ |
| 2.26 | Implement Cmd2DSetClip execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.27 | Implement Cmd2DRect execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.28 | Implement Cmd2DLine execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.29 | Implement Cmd2DRectsInstanced execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.30 | Implement Cmd2DLinesInstanced execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.31 | Implement Cmd2DWaveform execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.32 | Write visual test for 2D primitives | `tests/test_2d_visual.py` | ⬜ |

### 5.3 Phase 3: GraphPanel System

| ID | Task | File | Status |
|----|------|------|--------|
| 3.1 | Create panels module | `engine/ui/panels/__init__.py` | ⬜ |
| 3.2 | Define GridConfig dataclass | `engine/ui/panels/grid_builder.py` | ⬜ |
| 3.3 | Implement GridBuilder.get_subdivision_level() | `engine/ui/panels/grid_builder.py` | ⬜ |
| 3.4 | Implement GridBuilder.build() | `engine/ui/panels/grid_builder.py` | ⬜ |
| 3.5 | Define PlayheadConfig dataclass | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.6 | Implement GraphPanel.__init__() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.7 | Implement GraphPanel.measure() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.8 | Implement GraphPanel.layout() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.9 | Implement GraphPanel.draw() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.10 | Implement GraphPanel._draw_playhead() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.11 | Implement GraphPanel.update() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.12 | Implement GraphPanel.on_pointer_down() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.13 | Implement GraphPanel.on_pointer_move() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.14 | Implement GraphPanel.on_pointer_up() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.15 | Implement GraphPanel.on_scroll() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.16 | Define SequencerEvent dataclass | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.17 | Define SequencerLane dataclass | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.18 | Implement SequencerPanel.__init__() | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.19 | Implement SequencerPanel._compute_lane_rects() | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.20 | Implement SequencerPanel.draw_content() | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.21 | Implement SequencerPanel.draw_overlay() | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.22 | Implement SequencerPanel._get_visible_events() | `engine/ui/panels/sequencer_panel.py` | ⬜ |
| 3.23 | Define HitMarker dataclass | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.24 | Define WaveformCache dataclass | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.25 | Implement WaveformCache.get_envelope() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.26 | Implement WaveformCache._compute_decimation() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.27 | Implement WavePanel.__init__() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.28 | Implement WavePanel.draw_content() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.29 | Implement WavePanel._draw_waveform() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.30 | Implement WavePanel._draw_hit_markers() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.31 | Implement WavePanel.draw_overlay() | `engine/ui/panels/wave_panel.py` | ⬜ |
| 3.32 | Implement SyncPanel.__init__() | `engine/ui/panels/sync_panel.py` | ⬜ |
| 3.33 | Implement SyncPanel.draw() | `engine/ui/panels/sync_panel.py` | ⬜ |
| 3.34 | Write test: grid alignment between panels | `tests/test_panel_sync.py` | ⬜ |
| 3.35 | Write test: playhead alignment | `tests/test_panel_sync.py` | ⬜ |

### 5.4 Phase 4: Widget Integration

| ID | Task | File | Status |
|----|------|------|--------|
| 4.1 | Add DrawBatch to DrawContext | `engine/ui/draw.py` | ⬜ |
| 4.2 | Update Widget base for GraphPanel | `engine/ui/widget.py` | ⬜ |
| 4.3 | Integrate UIRenderer2D with UIRenderer | `engine/ui/renderer.py` | ⬜ |
| 4.4 | Add GraphPanel input routing | `engine/ui/window_manager.py` | ⬜ |
| 4.5 | Add keyboard shortcut handling | `engine/ui/input.py` | ⬜ |
| 4.6 | Implement space bar = play/pause | `engine/ui/input.py` | ⬜ |
| 4.7 | Add focus tracking for panels | `engine/ui/window_manager.py` | ⬜ |
| 4.8 | Test panels in DockLayout | `tests/test_ui_integration.py` | ⬜ |

### 5.5 Phase 5: Audio Integration

| ID | Task | File | Status |
|----|------|------|--------|
| 5.1 | Create audio module | `engine/audio/__init__.py` | ⬜ |
| 5.2 | Implement RingBuffer class | `engine/audio/buffer.py` | ⬜ |
| 5.3 | Move WaveformCache to audio module | `engine/audio/waveform.py` | ⬜ |
| 5.4 | Implement incremental decimation update | `engine/audio/waveform.py` | ⬜ |
| 5.5 | Create AudioEngine stub | `engine/audio/engine.py` | ⬜ |
| 5.6 | Implement audio file loading | `engine/audio/loader.py` | ⬜ |
| 5.7 | Integrate AudioEngine with Transport | `engine/audio/engine.py` | ⬜ |
| 5.8 | Test waveform at multiple zoom levels | `tests/test_waveform.py` | ⬜ |

### 5.6 Phase 6: Demo Application

| ID | Task | File | Status |
|----|------|------|--------|
| 6.1 | Create spectro_demo.py scaffold | `examples/spectro_demo.py` | ⬜ |
| 6.2 | Set up three-panel layout | `examples/spectro_demo.py` | ⬜ |
| 6.3 | Create shared TimeCamera instance | `examples/spectro_demo.py` | ⬜ |
| 6.4 | Wire Transport to panels | `examples/spectro_demo.py` | ⬜ |
| 6.5 | Add transport control buttons | `examples/spectro_demo.py` | ⬜ |
| 6.6 | Add BPM display | `examples/spectro_demo.py` | ⬜ |
| 6.7 | Add time signature display | `examples/spectro_demo.py` | ⬜ |
| 6.8 | Profile and identify bottlenecks | N/A | ⬜ |
| 6.9 | Optimize hot paths | N/A | ⬜ |
| 6.10 | Write ARCHITECTURE.md | `docs/ARCHITECTURE.md` | ⬜ |
| 6.11 | Update README.md | `README.md` | ⬜ |

---

## 6. Code Templates

### 6.1 TimeCamera Template

```python
# engine/time/camera.py

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class TimeCameraMode(Enum):
    FOLLOW_PLAYHEAD = auto()
    FREE_SCROLL = auto()
    SNAP_TO_BARS = auto()


@dataclass
class TimeCameraConfig:
    playhead_screen_ratio: float = 0.3
    follow_strength: float = 0.15
    scroll_decay: float = 0.92
    snap_grid_beats: float = 4.0
    min_window_beats: float = 1.0
    max_window_beats: float = 64.0
    zoom_sensitivity: float = 0.1


@dataclass
class TimeCamera:
    mode: TimeCameraMode = TimeCameraMode.FOLLOW_PLAYHEAD
    left_beat: float = 0.0
    window_beats: float = 16.0
    
    _panel_width_px: float = 800.0
    _px_per_beat: float = 50.0
    
    user_scroll_active: bool = False
    scroll_velocity: float = 0.0
    _drag_start_left_beat: float = 0.0
    _drag_start_x: float = 0.0
    
    config: TimeCameraConfig = field(default_factory=TimeCameraConfig)
    _last_update_frame: int = -1
    
    def beat_to_px(self, beat: float) -> float:
        """Convert beat to pixel X offset within panel."""
        return (beat - self.left_beat) * self._px_per_beat
    
    def px_to_beat(self, px: float) -> float:
        """Convert pixel X offset to beat."""
        return self.left_beat + (px / self._px_per_beat)
    
    @property
    def right_beat(self) -> float:
        return self.left_beat + self.window_beats
    
    @property
    def px_per_beat(self) -> float:
        return self._px_per_beat
    
    def update(self, dt: float, playhead_beat: float, panel_width_px: float, frame_id: int):
        """Called each frame."""
        self._panel_width_px = max(1.0, panel_width_px)
        self._px_per_beat = self._panel_width_px / max(0.001, self.window_beats)
        
        if self._last_update_frame == frame_id:
            return
        self._last_update_frame = frame_id
        
        if self.mode == TimeCameraMode.FOLLOW_PLAYHEAD and not self.user_scroll_active:
            self._update_follow_mode(playhead_beat)
        
        # Inertial scroll
        if not self.user_scroll_active and abs(self.scroll_velocity) > 0.01:
            self.left_beat += self.scroll_velocity * dt
            self.scroll_velocity *= self.config.scroll_decay
    
    def _update_follow_mode(self, playhead_beat: float):
        target_px = self._panel_width_px * self.config.playhead_screen_ratio
        target_left = playhead_beat - (target_px / self._px_per_beat)
        alpha = self.config.follow_strength
        self.left_beat = self.left_beat + (target_left - self.left_beat) * alpha
    
    def begin_drag(self, mouse_x: float):
        self.user_scroll_active = True
        self._drag_start_left_beat = self.left_beat
        self._drag_start_x = mouse_x
        self.scroll_velocity = 0.0
    
    def update_drag(self, mouse_x: float):
        if not self.user_scroll_active:
            return
        dx = mouse_x - self._drag_start_x
        delta_beats = -dx / self._px_per_beat
        self.left_beat = self._drag_start_left_beat + delta_beats
    
    def end_drag(self, mouse_x: float, velocity_px_per_sec: float = 0.0):
        self.user_scroll_active = False
        self.scroll_velocity = -velocity_px_per_sec / self._px_per_beat
    
    def zoom(self, delta: float, anchor_px: float):
        anchor_beat = self.px_to_beat(anchor_px)
        factor = 1.0 - delta * self.config.zoom_sensitivity
        new_window = self.window_beats * factor
        new_window = max(self.config.min_window_beats,
                        min(self.config.max_window_beats, new_window))
        self.window_beats = new_window
        self._px_per_beat = self._panel_width_px / max(0.001, self.window_beats)
        self.left_beat = anchor_beat - (anchor_px / self._px_per_beat)
```

### 6.2 DrawBatch Template

```python
# engine/render/commands_2d.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import numpy as np

Color4 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Cmd2DSetClip:
    x: int
    y: int
    w: int
    h: int
    enabled: bool = True


@dataclass(frozen=True)
class Cmd2DRect:
    x: float
    y: float
    w: float
    h: float
    color: Color4
    corner_radius: float = 0.0
    border_width: float = 0.0
    border_color: Optional[Color4] = None


@dataclass(frozen=True)
class Cmd2DLine:
    x0: float
    y0: float
    x1: float
    y1: float
    color: Color4
    width: float = 1.0


@dataclass(frozen=True)
class Cmd2DLinesInstanced:
    instance_data: bytes
    instance_count: int
    
    @staticmethod
    def from_vertical_lines(
        x_positions: List[float],
        y0: float,
        y1: float,
        color: Color4,
        width: float = 1.0
    ) -> "Cmd2DLinesInstanced":
        n = len(x_positions)
        # Layout: [x0, y0, x1, y1, r, g, b, a, width]
        data = np.zeros((n, 9), dtype=np.float32)
        for i, x in enumerate(x_positions):
            data[i] = [x, y0, x, y1, *color, width]
        return Cmd2DLinesInstanced(instance_data=data.tobytes(), instance_count=n)


@dataclass(frozen=True)
class Cmd2DRectsInstanced:
    instance_data: bytes
    instance_count: int
    corner_radius: float = 0.0


@dataclass(frozen=True)
class Cmd2DText:
    text: str
    x: float
    y: float
    color: Color4
    font_size: float = 14.0
    align: str = "left"


@dataclass(frozen=True)
class Cmd2DWaveform:
    envelope_data: bytes
    sample_count: int
    x_start: float
    x_scale: float
    y_center: float
    y_scale: float
    fill_color: Color4
    stroke_color: Optional[Color4] = None
    stroke_width: float = 1.0


Command2D = Union[
    Cmd2DSetClip, Cmd2DRect, Cmd2DLine, Cmd2DLinesInstanced,
    Cmd2DRectsInstanced, Cmd2DText, Cmd2DWaveform,
]


@dataclass
class DrawBatch:
    commands: List[Command2D] = field(default_factory=list)
    panel_id: Optional[str] = None
    
    def add(self, cmd: Command2D):
        self.commands.append(cmd)
    
    def rect(self, x, y, w, h, color, **kwargs):
        self.add(Cmd2DRect(x, y, w, h, color, **kwargs))
    
    def line(self, x0, y0, x1, y1, color, width=1.0):
        self.add(Cmd2DLine(x0, y0, x1, y1, color, width))
    
    def text(self, text, x, y, color, **kwargs):
        self.add(Cmd2DText(text, x, y, color, **kwargs))
    
    def vertical_lines(self, x_positions, y0, y1, color, width=1.0):
        if x_positions:
            self.add(Cmd2DLinesInstanced.from_vertical_lines(
                x_positions, y0, y1, color, width
            ))
```

### 6.3 GridBuilder Template

```python
# engine/ui/panels/grid_builder.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from engine.ui.layout import Rect
from engine.render.commands_2d import DrawBatch, Color4
from engine.time.camera import TimeCamera


@dataclass
class GridConfig:
    bar_color: Color4 = (0.4, 0.4, 0.45, 1.0)
    bar_width: float = 1.5
    beat_color: Color4 = (0.25, 0.25, 0.3, 1.0)
    beat_width: float = 1.0
    subdivision_color: Color4 = (0.18, 0.18, 0.22, 0.5)
    subdivision_width: float = 0.5
    subdivision_thresholds: List[Tuple[float, int]] = field(default_factory=lambda: [
        (100.0, 4), (50.0, 2), (20.0, 1), (0.0, 0),
    ])
    
    def get_subdivision_level(self, px_per_beat: float) -> int:
        for threshold, level in self.subdivision_thresholds:
            if px_per_beat >= threshold:
                return level
        return 0


class GridBuilder:
    def __init__(self, config: GridConfig = None):
        self.config = config or GridConfig()
    
    def build(
        self,
        time_camera: TimeCamera,
        rect: Rect,
        beats_per_bar: float = 4.0,
    ) -> DrawBatch:
        batch = DrawBatch()
        cfg = self.config
        
        subdiv_level = cfg.get_subdivision_level(time_camera.px_per_beat)
        
        bar_xs: List[float] = []
        beat_xs: List[float] = []
        subdiv_xs: List[float] = []
        
        margin = 1.0
        left = time_camera.left_beat - margin
        right = time_camera.right_beat + margin
        
        # Bar lines
        first_bar = int(left / beats_per_bar)
        last_bar = int(right / beats_per_bar) + 1
        for bar in range(first_bar, last_bar + 1):
            beat = bar * beats_per_bar
            x = time_camera.beat_to_px(beat)
            if 0 <= x <= rect.w:
                bar_xs.append(rect.x + x)
        
        # Beat lines
        if subdiv_level >= 1:
            first_beat = int(left)
            last_beat = int(right) + 1
            for beat in range(first_beat, last_beat + 1):
                if beat % beats_per_bar == 0:
                    continue
                x = time_camera.beat_to_px(float(beat))
                if 0 <= x <= rect.w:
                    beat_xs.append(rect.x + x)
        
        # Subdivision lines
        if subdiv_level >= 2:
            step = 1.0 / subdiv_level
            subdiv = int(left / step) * step
            while subdiv <= right:
                if subdiv % 1.0 != 0:
                    x = time_camera.beat_to_px(subdiv)
                    if 0 <= x <= rect.w:
                        subdiv_xs.append(rect.x + x)
                subdiv += step
        
        y0, y1 = rect.y, rect.y + rect.h
        
        if subdiv_xs:
            batch.vertical_lines(subdiv_xs, y0, y1, cfg.subdivision_color, cfg.subdivision_width)
        if beat_xs:
            batch.vertical_lines(beat_xs, y0, y1, cfg.beat_color, cfg.beat_width)
        if bar_xs:
            batch.vertical_lines(bar_xs, y0, y1, cfg.bar_color, cfg.bar_width)
        
        return batch
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

**TimeCamera**:
```python
def test_beat_to_px_inverse():
    cam = TimeCamera(left_beat=4.0, window_beats=16.0)
    cam._panel_width_px = 800.0
    cam._px_per_beat = 50.0
    
    for beat in [0.0, 4.0, 8.5, 12.0, 20.0]:
        px = cam.beat_to_px(beat)
        recovered = cam.px_to_beat(px)
        assert abs(recovered - beat) < 1e-6

def test_follow_mode_converges():
    cam = TimeCamera(mode=TimeCameraMode.FOLLOW_PLAYHEAD)
    cam._panel_width_px = 800.0
    playhead = 10.0
    
    for _ in range(100):
        cam.update(dt=0.016, playhead_beat=playhead, panel_width_px=800.0, frame_id=_)
    
    # Playhead should be at 30% of screen
    playhead_px = cam.beat_to_px(playhead)
    expected_px = 800.0 * 0.3
    assert abs(playhead_px - expected_px) < 1.0
```

**Grid Alignment**:
```python
def test_grid_alignment_between_panels():
    time_camera = TimeCamera(left_beat=4.0, window_beats=16.0)
    time_camera._panel_width_px = 800.0
    time_camera._px_per_beat = 50.0
    
    builder = GridBuilder()
    
    rect1 = Rect(x=0, y=0, w=800, h=200)
    rect2 = Rect(x=0, y=200, w=800, h=200)
    
    batch1 = builder.build(time_camera, rect1, beats_per_bar=4.0)
    batch2 = builder.build(time_camera, rect2, beats_per_bar=4.0)
    
    # Extract X positions from line commands
    xs1 = extract_line_x_positions(batch1)
    xs2 = extract_line_x_positions(batch2)
    
    assert xs1 == xs2, "Grid lines must align exactly"
```

### 7.2 Visual Tests

Create a visual test that renders both panels and overlays them to verify alignment:

```python
def test_visual_panel_alignment():
    # Render SequencerPanel to texture A
    # Render WavePanel to texture B
    # Composite with difference blend mode
    # Assert: grid lines produce zero difference (black)
```

### 7.3 Performance Tests

```python
def test_draw_batch_performance():
    batch = DrawBatch()
    
    # Add 1000 events
    for i in range(1000):
        batch.rect(i * 10, 0, 8, 20, (0.5, 0.5, 0.5, 1.0))
    
    # Should complete in < 1ms
    start = time.perf_counter()
    renderer.execute(batch)
    elapsed = time.perf_counter() - start
    
    assert elapsed < 0.001
```

---

## 8. Worked Example

### Frame at Beat 8.5, 120 BPM

**Setup**:
```
Transport:
  bpm = 120
  playhead_beat = 8.5
  playing = True

TimeCamera:
  mode = FOLLOW_PLAYHEAD
  left_beat = 4.98 (smoothly tracking)
  window_beats = 16.0
  panel_width = 800px
  px_per_beat = 50.0
```

**Calculations**:

1. **Visible range**: beats 4.98 to 20.98

2. **Bar lines** (every 4 beats):
   - Beat 4: `(4 - 4.98) * 50 = -49px` (off screen)
   - Beat 8: `(8 - 4.98) * 50 = 151px` ✓
   - Beat 12: `(12 - 4.98) * 50 = 351px` ✓
   - Beat 16: `(16 - 4.98) * 50 = 551px` ✓
   - Beat 20: `(20 - 4.98) * 50 = 751px` ✓

3. **Playhead position**:
   - `(8.5 - 4.98) * 50 = 176px`
   - This is ~22% across the 800px panel
   - Follow mode will keep adjusting until playhead reaches 30% (240px)

4. **Event at beat 8, duration 0.5**:
   - x = `(8 - 4.98) * 50 = 151px`
   - w = `0.5 * 50 = 25px`
   - Rect: (151, lane_y, 25, lane_h)

**Visual Result**:
```
  0px              151px    176px             351px             551px             751px    800px
  │                  │        │                 │                 │                 │        │
  │    beat 5,6,7    │ BAR 8  │ PLAYHEAD        │ BAR 12          │ BAR 16          │ BAR 20 │
  │    │  │  │       │████████│▼                │                 │                 │        │
  │    │  │  │       │ EVENT  │                 │                 │                 │        │
```

---

## 9. Critical Invariants

### 9.1 MUST Hold At All Times

1. **Single TimeCamera instance** shared between SequencerPanel and WavePanel

2. **beat_to_px() is THE mapping** — no panel may compute its own

3. **GridBuilder.build()** called with SAME TimeCamera for both panels

4. **DrawBatch contains no GL objects** — only pure data

5. **TransportState is immutable** — created fresh each frame

### 9.2 Performance Targets

- 60 FPS with 1000 visible events
- < 1ms for grid generation
- < 2ms for waveform decimation lookup
- Instanced draws for > 4 similar items

### 9.3 Thread Safety

- TransportState is frozen dataclass — safe to read from any thread
- TimeCamera is single-threaded (main thread only)
- DrawBatch is created and consumed on main thread
- Upload queue (existing) handles async GPU uploads

---

## 10. File Structure

```
spectro/
├── engine/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── frame.py                 # FrameState (existing)
│   │   └── snapshot.py              # GraphSnapshot (existing)
│   ├── time/                        # NEW
│   │   ├── __init__.py
│   │   ├── camera.py                # TimeCamera
│   │   └── transport.py             # Transport, TransportState
│   ├── render/
│   │   ├── __init__.py
│   │   ├── commands.py              # 3D commands (existing)
│   │   ├── commands_2d.py           # NEW: 2D commands
│   │   ├── renderer.py              # 3D renderer (existing)
│   │   ├── resources.py             # ResourceRegistry (existing)
│   │   ├── targets.py               # RenderTargetPool (existing)
│   │   ├── uploader.py              # UploadQueue (existing)
│   │   ├── world.py                 # RenderWorld (existing)
│   │   └── shaders/                 # NEW
│   │       ├── rect.vert
│   │       ├── rect.frag
│   │       ├── line.vert
│   │       ├── line.frag
│   │       ├── rect_instanced.vert
│   │       ├── line_instanced.vert
│   │       ├── waveform.vert
│   │       └── waveform.frag
│   ├── scene/
│   │   ├── __init__.py
│   │   └── graph.py                 # EntityNode (existing)
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── widget.py                # Widget base (existing, needs update)
│   │   ├── draw.py                  # DrawContext (existing, needs update)
│   │   ├── layout.py                # Flexbox layout (existing)
│   │   ├── style.py                 # Style system (existing)
│   │   ├── renderer.py              # UIRenderer (existing, needs update)
│   │   ├── renderer_2d.py           # NEW: 2D command executor
│   │   ├── window_manager.py        # WindowManager (existing)
│   │   └── panels/                  # NEW
│   │       ├── __init__.py
│   │       ├── grid_builder.py
│   │       ├── graph_panel.py
│   │       ├── sequencer_panel.py
│   │       ├── wave_panel.py
│   │       └── sync_panel.py
│   ├── audio/                       # NEW
│   │   ├── __init__.py
│   │   ├── buffer.py                # RingBuffer
│   │   ├── waveform.py              # WaveformCache
│   │   └── engine.py                # AudioEngine stub
│   └── viewport/
│       └── viewport.py              # ViewportArea (existing)
├── examples/
│   ├── app_mglw.py                  # 3D demo (existing)
│   ├── ui_demo.py                   # UI demo (existing)
│   └── spectro_demo.py              # NEW: Main spectro demo
├── tests/
│   ├── test_time_camera.py          # NEW
│   ├── test_transport.py            # NEW
│   ├── test_panel_sync.py           # NEW
│   ├── test_2d_visual.py            # NEW
│   └── test_waveform.py             # NEW
├── docs/
│   ├── synced_panels_render_graph_command_list.md  # (existing)
│   └── ARCHITECTURE.md              # NEW
├── SPECTRO_DEVELOPMENT_PLAN.md      # THIS FILE
└── README.md
```

---

## Appendix A: Quick Reference

### Beat ↔ Pixel Conversion

```python
# Beat to pixel (within panel)
px = (beat - time_camera.left_beat) * time_camera.px_per_beat

# Pixel to beat
beat = time_camera.left_beat + (px / time_camera.px_per_beat)

# Check if beat is visible
visible = time_camera.left_beat <= beat <= time_camera.right_beat
```

### Frame Loop Pseudocode

```python
def render_frame(dt):
    # 1. Update transport
    transport_state = transport.update(dt)
    
    # 2. Update time camera
    time_camera.update(
        dt=dt,
        playhead_beat=transport_state.playhead_beat,
        panel_width_px=panel_width,
        frame_id=frame_id,
    )
    
    # 3. Update panels
    sync_panel.update(transport_state)
    sequencer_panel.update(transport_state, frame_id, dt)
    wave_panel.update(transport_state, frame_id, dt)
    
    # 4. Draw
    ctx = DrawContext(window_width, window_height)
    window_manager.draw(ctx)
    batch = ctx.finalize()
    
    # 5. Execute
    ui_renderer.execute(batch)
```

### Creating a New Panel Type

```python
class MyPanel(GraphPanel):
    def __init__(self, time_camera: TimeCamera, **kwargs):
        super().__init__(time_camera, **kwargs)
        # Custom init
    
    def draw_content(self, ctx: DrawContext):
        # Draw panel-specific content
        # Grid and playhead are handled by base class
        pass
    
    def draw_overlay(self, ctx: DrawContext):
        # Optional: draw overlay elements
        pass
```

---

## Appendix B: Shader Snippets

### Instanced Line Vertex Shader

```glsl
#version 330

// Per-vertex
in vec2 in_pos;  // Unit line: (0,0) to (1,0)

// Per-instance
in vec4 in_line;    // x0, y0, x1, y1
in vec4 in_color;   // r, g, b, a
in float in_width;

uniform vec2 u_resolution;

out vec4 v_color;

void main() {
    vec2 p0 = in_line.xy;
    vec2 p1 = in_line.zw;
    
    vec2 dir = normalize(p1 - p0);
    vec2 normal = vec2(-dir.y, dir.x) * in_width * 0.5;
    
    vec2 pos = mix(p0, p1, in_pos.x) + normal * (in_pos.y * 2.0 - 1.0);
    
    // To NDC
    vec2 ndc = (pos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}
```

### Waveform Vertex Shader

```glsl
#version 330

in float in_index;  // 0, 1, 2, 3, ... (vertex index within strip)

uniform sampler1D u_envelope;  // R = min, G = max
uniform float u_x_start;
uniform float u_x_scale;
uniform float u_y_center;
uniform float u_y_scale;
uniform vec2 u_resolution;
uniform int u_sample_count;

out float v_alpha;

void main() {
    int sample_idx = int(in_index) / 2;
    bool is_max = (int(in_index) % 2) == 1;
    
    vec2 envelope = texelFetch(u_envelope, sample_idx, 0).rg;
    float value = is_max ? envelope.g : envelope.r;
    
    float x = u_x_start + float(sample_idx) * u_x_scale;
    float y = u_y_center + value * u_y_scale;
    
    vec2 ndc = (vec2(x, y) / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_alpha = 1.0;
}
```

---

*End of Development Plan*

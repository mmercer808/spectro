# SPECTRO UI Engine — Consolidated Specification v6.0

> **Purpose**: Single authoritative implementation guide.
> **Status**: Consolidates Architecture v2, Development Plan 5, and Gap Analysis.
> **Last Updated**: 2025-01-13

---

## Quick Reference: What's New in v6.0

1. **Double-buffered DrawBatch** — Widgets schedule renders, batch accumulates in back buffer
2. **FreeType-based text rendering** — Avoids Windows CP1252 encoding issues
3. **Complete class definitions** — All gaps from Gap Analysis addressed
4. **Explicit font loading path** — TTF files can be placed anywhere

---

## 1. Project Overview

### 1.1 What We're Building

A **time-synchronized panel system** for audio visualization:

- **SyncRegion**: Transport controls + metronome (no scroll)
- **WaveformRegion**: Continuous waveform (scrolls)  
- **LanesRegion**: Discrete events on grid (scrolls in sync)

**Key Requirement**: WaveformRegion and LanesRegion share a single `TimeCamera` for pixel-perfect alignment.

### 1.2 Design Principles

1. **Immutable snapshots** — Mutable graph for authoring, frozen snapshots for render
2. **Pure-data command lists** — No GL objects in commands; thread-safe
3. **Single time source** — `TimeCamera` is THE mapping from beats to pixels
4. **Double-buffered batches** — Widgets add to back buffer, swap before render
5. **FreeType for fonts** — Direct TTF loading, no system font dependencies

---

## 2. Architecture

### 2.1 Ownership Hierarchy

```
Application
    └── PanelManager (owns everything)
            ├── Transport
            ├── TimeCamera (shared with Panel3D)
            ├── Panel3D
            │       ├── SyncRegion (reads TransportState, no scroll)
            │       ├── WaveformRegion (reads TimeCamera)
            │       └── LanesRegion (reads TimeCamera)
            │
            └── RenderScheduler
                    ├── batch_back (widgets add here)
                    ├── batch_front (being rendered)
                    └── UIRenderer2D
                            └── FontManager (FreeType-based)
```

### 2.2 Double-Buffered Rendering Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAME LOOP                                    │
│                                                                  │
│  1. INPUT ──────► poll_input()                                  │
│                                                                  │
│  2. UPDATE ─────► transport.update(dt) → TransportState         │
│                   time_camera.update(playhead)                  │
│                   panel.update(transport_state)                 │
│                                                                  │
│  3. COLLECT ────► batch_back.clear()                            │
│                   panel.draw(batch_back)                        │
│                   [widgets add commands to batch_back]          │
│                                                                  │
│  4. SWAP ───────► batch_front, batch_back = batch_back, batch_front
│                                                                  │
│  5. RENDER ─────► ui_renderer.execute(batch_front)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why double-buffer?**
- Widgets can add to batch_back outside main loop (async events)
- Clean separation: collection vs execution
- Matches RCU-style snapshot pattern already in codebase

### 2.3 Panel Layout

```
┌──────────────────────────────────────────────────────────────┐
│  SyncRegion (15%)  │ [▶][■] BPM:120 │ ⭕ Beat │ ⭕ Bar      │
├──────────────────────────────────────────────────────────────┤
│  WaveformRegion (25%)                                        │
│  ▓▓▓▓▓▓▓▓▓▓███████▓▓▓▓▓▓▓▓▓████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  │
│  │    │    │    ▼    │    │    │    │    │    │             │
├──────────────────────────────────────────────────────────────┤
│  LanesRegion (60%)                                           │
│  Lane 0 │░░░░│████│░░░░░░│████│░░░░│████│░░░░│              │
│  Lane 1 │████│░░░░│████│░░░░░░│████│░░░░│░░░░│              │
│         │    │    │    ▼    │    │    │    │                │
│                    PLAYHEAD                                  │
└──────────────────────────────────────────────────────────────┘
              ◄──── Shared TimeCamera ────►
```

---

## 3. Key Components

### 3.1 TimeCamera (Complete)

**File**: `engine/time/camera.py`

Core state:
- `left_beat: float` — Beat at left edge
- `window_beats: float` — Visible beat range (zoom)
- `mode: TimeCameraMode` — FOLLOW_PLAYHEAD | FREE_SCROLL | SNAP_TO_BARS

Essential methods:
```python
def beat_to_px(self, beat: float) -> float:
    """THE canonical mapping."""
    return (beat - self.left_beat) * self._px_per_beat

def px_to_beat(self, px: float) -> float:
    """Inverse mapping."""
    return self.left_beat + (px / self._px_per_beat)

def is_beat_visible(self, beat: float) -> bool
def get_visible_range(self) -> Tuple[float, float]
def snap_to_grid(self, beat: float, subdivision: int = 4) -> float
def animate_to_beat(self, beat: float, duration: float = 0.3)
def begin_drag(self, mouse_x: float)
def update_drag(self, mouse_x: float)
def end_drag(self, mouse_x: float, velocity: float)
def zoom(self, delta: float, anchor_px: float)
```

### 3.2 Transport (Complete)

**File**: `engine/time/transport.py`

Mutable `Transport` class with:
- `play()`, `pause()`, `stop()`, `toggle()`
- `seek_to_beat()`, `seek_to_time()`, `seek_by_bars()`
- `set_bpm()`, `set_time_signature()`
- Callback lists: `on_beat_callbacks`, `on_bar_callbacks`, `on_loop_callbacks`

Immutable `TransportState` frozen dataclass with derived properties:
- `phase_in_beat`, `phase_in_bar`
- `current_bar`, `current_beat_in_bar`
- `seconds_per_beat`, `beats_per_second`

### 3.3 DrawBatch (Complete)

**File**: `engine/render/commands_2d.py`

Pure-data command container with convenience methods:
```python
batch.rect(x, y, w, h, color, corner_radius=0, border_width=0)
batch.line(x0, y0, x1, y1, color, width=1)
batch.circle(cx, cy, radius, fill_color, stroke_color)
batch.arc(cx, cy, radius, start_angle, end_angle, color, width)
batch.text(text, x, y, color, font_size=14, align="left")
batch.vertical_lines(x_positions, y0, y1, color, width)
batch.horizontal_lines(y_positions, x0, x1, color, width)
batch.polyline(points, color, width, closed=False)
batch.set_clip(x, y, w, h)
batch.clear_clip()
```

Instanced commands for efficiency (auto-selected when >4 items):
- `Cmd2DRectsInstanced`
- `Cmd2DLinesInstanced`
- `Cmd2DTrianglesInstanced`

### 3.4 RenderScheduler (New)

**File**: `engine/render/renderer_2d.py`

```python
class RenderScheduler:
    def __init__(self, ctx: moderngl.Context):
        self.ui_renderer = UIRenderer2D(ctx)
        self._batch_back = DrawBatch()
        self._batch_front = DrawBatch()
    
    def get_back_buffer(self) -> DrawBatch:
        """Widgets add commands here."""
        return self._batch_back
    
    def swap_and_execute(self):
        """Swap buffers and render front."""
        self._batch_front, self._batch_back = self._batch_back, self._batch_front
        self.ui_renderer.execute(self._batch_front)
        self._batch_back.clear()
```

---

## 4. Text Rendering System

### 4.1 Design Decision: FreeType via freetype-py

**Why not system fonts?**
- Windows CP1252 encoding issues with PyQt font handling
- Inconsistent rendering across platforms
- No control over glyph metrics

**Solution**: Use `freetype-py` to load TTF files directly and rasterize to texture atlas.

### 4.2 Components

**File**: `engine/ui/text/font.py`

```python
import freetype

@dataclass
class GlyphInfo:
    char: str
    atlas_x: int          # Position in atlas texture
    atlas_y: int
    width: int            # Glyph bitmap size
    height: int
    bearing_x: float      # Offset from cursor to glyph origin
    bearing_y: float
    advance: float        # How far to move cursor after glyph

class FontAtlas:
    """Packed glyph texture atlas for a single font/size."""
    
    def __init__(self, ctx: moderngl.Context, font_path: str, size: int):
        self.face = freetype.Face(font_path)
        self.face.set_pixel_sizes(0, size)
        self.size = size
        self.glyphs: Dict[str, GlyphInfo] = {}
        self.texture: moderngl.Texture = None
        
        self._build_atlas(ctx)
    
    def _build_atlas(self, ctx):
        """Rasterize ASCII + extended chars into texture."""
        # Determine atlas size (power of 2)
        # Render each glyph, pack into atlas
        # Upload to GPU texture
        pass

class FontManager:
    """Load and cache fonts from arbitrary paths."""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._atlases: Dict[Tuple[str, int], FontAtlas] = {}
        self._default_font_path: Optional[str] = None
    
    def set_default_font(self, path: str):
        """Set default font (TTF can be anywhere)."""
        self._default_font_path = path
    
    def get_font(self, path: str, size: int) -> FontAtlas:
        """Get or create font atlas."""
        key = (path, size)
        if key not in self._atlases:
            self._atlases[key] = FontAtlas(self.ctx, path, size)
        return self._atlases[key]
    
    def get_default(self, size: int) -> FontAtlas:
        """Get default font at specified size."""
        if self._default_font_path is None:
            raise FontNotFoundError("No default font set")
        return self.get_font(self._default_font_path, size)
```

### 4.3 Text Layout

```python
class TextLayout:
    """Convert text to positioned glyphs for rendering."""
    
    def layout(self, text: str, atlas: FontAtlas, 
               max_width: float = None) -> List[PositionedGlyph]:
        """Layout text, optionally with word wrap."""
        pass
    
    def measure(self, text: str, atlas: FontAtlas) -> Tuple[float, float]:
        """Measure text dimensions without rendering."""
        width = sum(atlas.glyphs[c].advance for c in text if c in atlas.glyphs)
        height = atlas.size
        return (width, height)
```

### 4.4 Integration

```python
# In UIRenderer2D._exec_text():
def _exec_text(self, cmd: Cmd2DText):
    atlas = self._font_manager.get_default(int(cmd.font_size))
    
    # Layout glyphs
    layout = TextLayout()
    glyphs = layout.layout(cmd.text, atlas)
    
    # Build instanced quad data
    instance_data = self._build_text_instances(glyphs, cmd.x, cmd.y, cmd.color)
    
    # Render with text shader (samples from atlas texture)
    prog = self._get_program("text")
    prog['u_atlas'].value = 0
    atlas.texture.use(0)
    # ... render instanced quads
```

---

## 5. Input Handling

### 5.1 InputState

**File**: `engine/ui/input.py`

```python
@dataclass
class InputState:
    mouse_pos: Tuple[float, float] = (0, 0)
    mouse_delta: Tuple[float, float] = (0, 0)
    mouse_buttons: Tuple[bool, bool, bool] = (False, False, False)
    mouse_just_pressed: Tuple[bool, bool, bool] = (False, False, False)
    mouse_just_released: Tuple[bool, bool, bool] = (False, False, False)
    scroll_delta: Tuple[float, float] = (0, 0)
    keys_pressed: Set[str] = field(default_factory=set)
    keys_just_pressed: Set[str] = field(default_factory=set)
    modifiers: Set[str] = field(default_factory=set)  # "Ctrl", "Shift", "Alt"

class MouseButtons:
    LEFT = 0
    MIDDLE = 1
    RIGHT = 2

class KeyBindings:
    """Default key bindings."""
    PLAY_PAUSE = "Space"
    STOP = "Escape"
    SEEK_START = "Home"
    SEEK_END = "End"
    ZOOM_IN = "="
    ZOOM_OUT = "-"
```

### 5.2 Hit Testing and Focus

```python
class WindowManager:
    def hit_test(self, x: float, y: float) -> Optional[Widget]:
        """Find topmost widget at position."""
        return self._hit_test_recursive(self.root, x, y)
    
    def _hit_test_recursive(self, widget, x, y) -> Optional[Widget]:
        if not widget.point_in_widget(x, y):
            return None
        
        # Check children in reverse order (topmost first)
        for child in reversed(widget.children):
            result = self._hit_test_recursive(child, x, y)
            if result:
                return result
        
        return widget if widget.interactive else None
```

---

## 6. Implementation Phases

### Phase 1: Core Time Infrastructure (3 days)
- TimeCamera with all methods
- Transport with callbacks
- Unit tests for time math

### Phase 2: 2D Drawing System (2 days)
- All 2D command types
- RenderScheduler with double buffer
- Shaders: rect, line, circle, arc
- **Text system with FreeType**

### Phase 3: Panel System (3 days)
- GraphPanel base class
- GridBuilder
- SyncRegion, WaveformRegion, LanesRegion
- Playhead rendering

### Phase 4: Widget Integration (2 days)
- Input routing through WindowManager
- Focus management
- Global shortcuts

### Phase 5: Audio Integration (1 week)
- WaveformCache with decimation
- AudioEngine stub

### Phase 6: Demo Application (1 week)
- Complete working demo
- Performance optimization

### Phase 7: Error Handling & Polish
- Custom exceptions
- Validation
- Configuration files

---

## 7. Task List Summary

### Phase 1 (26 tasks → 34 tasks with Gap Analysis additions)
| ID | Task | Status |
|----|------|--------|
| 1.1-1.26 | Original TimeCamera/Transport tasks | ⬜ |
| 1.27 | Add scroll limits | ⬜ |
| 1.28 | Add visibility queries | ⬜ |
| 1.29 | Add snap_to_grid() | ⬜ |
| 1.30 | Add animate_to_beat() | ⬜ |
| 1.31 | Transport play/pause/stop | ⬜ |
| 1.32 | Transport seek methods | ⬜ |
| 1.33 | Transport callbacks | ⬜ |
| 1.34 | TransportState derived properties | ⬜ |

### Phase 2 (32 tasks → 44 tasks)
| ID | Task | Status |
|----|------|--------|
| 2.1-2.32 | Original 2D drawing tasks | ⬜ |
| 2.33 | FontAtlas with FreeType | ⬜ |
| 2.34 | FontManager class | ⬜ |
| 2.35 | TextLayout class | ⬜ |
| 2.36 | Text shaders | ⬜ |
| 2.37 | Cmd2DText execution | ⬜ |
| 2.38-2.44 | Additional shapes (circle, arc, gradient, etc.) | ⬜ |

### Phase 3-7
See detailed task lists in original documents.

---

## 8. File Structure

```
spectro/
├── engine/
│   ├── __init__.py
│   ├── core/
│   │   ├── errors.py              # Custom exceptions
│   │   └── types.py               # Color4, Rect, etc.
│   ├── time/
│   │   ├── __init__.py
│   │   ├── camera.py              # TimeCamera
│   │   └── transport.py           # Transport, TransportState
│   ├── render/
│   │   ├── __init__.py
│   │   ├── commands_2d.py         # All 2D commands, DrawBatch
│   │   ├── renderer_2d.py         # RenderScheduler, UIRenderer2D
│   │   └── shaders/
│   │       ├── rect.vert/frag
│   │       ├── line.vert/frag
│   │       ├── line_instanced.vert/frag
│   │       ├── circle.vert/frag
│   │       ├── arc.vert/frag
│   │       ├── text.vert/frag
│   │       └── waveform.vert/frag
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── panel_manager.py       # PanelManager
│   │   ├── input.py               # InputState, KeyBindings
│   │   ├── text/
│   │   │   ├── font.py            # FontAtlas, FontManager (FreeType)
│   │   │   └── layout.py          # TextLayout
│   │   └── panels/
│   │       ├── __init__.py
│   │       ├── graph_panel.py     # GraphPanel base
│   │       ├── grid_builder.py    # GridBuilder
│   │       ├── panel_3d.py        # Panel3D container
│   │       ├── sync_region.py
│   │       ├── waveform_region.py
│   │       └── lanes_region.py
│   └── audio/
│       ├── __init__.py
│       ├── waveform.py            # WaveformCache
│       └── engine.py              # AudioEngine stub
├── assets/
│   └── fonts/
│       └── FiraMono-Regular.ttf   # Default font (or any TTF)
├── examples/
│   └── spectro_demo.py
└── tests/
    ├── test_time_camera.py
    ├── test_transport.py
    ├── test_draw_batch.py
    ├── test_text.py
    └── test_panel_sync.py
```

---

## 9. Critical Invariants

### Thread Safety
- `TransportState` is frozen dataclass — safe to read from any thread
- `DrawBatch` contains no GL objects — safe to build off main thread
- `TimeCamera` is single-threaded (main thread only)

### Performance Targets
- 60 FPS with 1000 visible events
- < 1ms for grid generation
- < 2ms for waveform decimation lookup
- Use instanced draws for > 4 similar items

### Pixel-Perfect Sync
- ALL scrolling panels use same TimeCamera instance
- GridBuilder uses TimeCamera.beat_to_px() exclusively
- Playhead position computed identically in all regions

---

## 10. Migration Notes

### From Original Documents

This document consolidates:
- `SPECTRO_ARCHITECTURE_v2.md` → Sections 2, 3
- `SPECTRO_DEVELOPMENT_PLAN_5.md` → Sections 1, 3, 6, 7, 8, 9
- `SPECTRO_DEVELOPMENT_PLAN_GAP_ANALYSIS.md` → Integrated throughout

### Key Additions from Gap Analysis
1. All missing TimeCamera methods (visibility, animation, snap)
2. All missing Transport methods (play/pause/stop/seek, callbacks)
3. Complete text rendering system with FreeType
4. DrawContext → DrawBatch clarification (use DrawBatch directly)
5. UIRenderer2D complete dispatch table
6. Input handling classes
7. Error types
8. Phase 7 for polish

---

*End of Consolidated Specification*

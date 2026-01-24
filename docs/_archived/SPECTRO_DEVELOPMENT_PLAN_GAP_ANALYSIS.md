# SPECTRO_DEVELOPMENT_PLAN v5 — Gap Analysis

> This document identifies **missing details, undefined interfaces, and edge cases** in the current development plan.

---

## Executive Summary

The plan is **well-structured** with good architecture diagrams, task lists, and code templates. However, there are gaps in:

1. **Input handling specifics** (key bindings, mouse button mapping)
2. **UIRenderer2D internals** (shader compilation, buffer management, state tracking)
3. **DrawContext integration** (how it interfaces with existing UI system)
4. **Text rendering** (font loading, glyph atlas, metrics)
5. **Error handling and validation**
6. **Integration seams** (how new code connects to existing modules)
7. **Missing 2D command types** (circles, arcs, gradients)
8. **Configuration serialization**
9. **Hot reload / live editing support**

---

## 1. Missing Class/Function Definitions

### 1.1 TimeCamera — Missing Methods

The plan defines core methods but omits:

```python
# MISSING: Visibility queries
def is_beat_visible(self, beat: float) -> bool: ...
def get_visible_range(self) -> Tuple[float, float]: ...

# MISSING: Quantization helpers  
def snap_to_grid(self, beat: float, subdivision: int = 4) -> float: ...
def nearest_beat(self, px: float) -> float: ...
def nearest_bar(self, px: float) -> float: ...

# MISSING: Scroll limits
min_left_beat: float = -4.0  # Allow negative for pre-roll
max_right_beat: float = 1000.0  # Prevent infinite scroll

# MISSING: Animation state
_target_left_beat: float  # For smooth scrolling
_animation_progress: float
def animate_to_beat(self, beat: float, duration: float = 0.3): ...

# MISSING: Frame-rate independent smoothing
def _lerp_with_dt(self, current: float, target: float, factor: float, dt: float) -> float: ...
```

### 1.2 Transport — Missing Methods

```python
# MISSING: Playback control
def play(self): ...
def pause(self): ...
def stop(self): ...  # Stop + reset to start or loop start
def toggle(self): ...

# MISSING: Seeking
def seek_to_beat(self, beat: float): ...
def seek_to_time(self, seconds: float): ...
def seek_by_bars(self, delta: int): ...

# MISSING: Tempo changes
def set_bpm(self, bpm: float): ...
def tap_tempo(self, timestamp: float): ...  # For live BPM detection

# MISSING: Time signature
def set_time_signature(self, numerator: int, denominator: int): ...

# MISSING: Callbacks/signals
on_beat_callbacks: List[Callable[[int, TransportState], None]]
on_bar_callbacks: List[Callable[[int, TransportState], None]]
on_loop_callbacks: List[Callable[[TransportState], None]]
on_stop_callbacks: List[Callable[[TransportState], None]]

# MISSING: Derived properties on TransportState
@property
def seconds_per_beat(self) -> float:
    return 60.0 / self.bpm

@property  
def beats_per_second(self) -> float:
    return self.bpm / 60.0

@property
def bar_duration_beats(self) -> float:
    return self.time_sig.numerator

@property
def bar_duration_seconds(self) -> float:
    return self.bar_duration_beats * self.seconds_per_beat
```

### 1.3 DrawBatch — Missing Methods

```python
# MISSING: Convenience methods
def fill_rect(self, x, y, w, h, color): ...  # Alias for rect with no border
def stroke_rect(self, x, y, w, h, color, width=1.0): ...  # Border only
def rounded_rect(self, x, y, w, h, color, radius, **kwargs): ...

# MISSING: Arc/circle support (defined in commands but no helper)
def circle(self, cx, cy, radius, fill_color, stroke_color=None, stroke_width=1.0): ...
def arc(self, cx, cy, radius, start_angle, end_angle, color, width=1.0): ...

# MISSING: Polyline support
def polyline(self, points: List[Tuple[float, float]], color, width=1.0, closed=False): ...

# MISSING: Horizontal lines helper (for lane dividers)
def horizontal_lines(self, y_positions, x0, x1, color, width=1.0): ...

# MISSING: Batch merge
def extend(self, other: "DrawBatch"): ...

# MISSING: Statistics
def command_count(self) -> int: ...
def estimated_draw_calls(self) -> int: ...

# MISSING: Clear/reset
def clear(self): ...
```

### 1.4 GridBuilder — Missing Features

```python
# MISSING: Label generation
def build_with_labels(
    self,
    time_camera: TimeCamera,
    rect: Rect,
    beats_per_bar: float = 4.0,
    show_bar_numbers: bool = True,
    show_beat_numbers: bool = False,
) -> Tuple[DrawBatch, List[GridLabel]]:
    """Return grid lines and text labels separately."""

@dataclass
class GridLabel:
    text: str
    x: float
    y: float
    level: str  # "bar", "beat", "subdivision"

# MISSING: Custom grid intervals (for triplets, odd time signatures)
def build_custom(
    self,
    time_camera: TimeCamera,
    rect: Rect,
    intervals: List[float],  # [4.0, 1.0, 0.5, 0.25] for standard
    colors: List[Color4],
    widths: List[float],
) -> DrawBatch: ...

# MISSING: Alternating backgrounds (for lanes)
def build_alternating_backgrounds(
    self,
    time_camera: TimeCamera,
    rect: Rect,
    beats_per_bar: float,
    color_a: Color4,
    color_b: Color4,
) -> DrawBatch: ...
```

---

## 2. Missing 2D Command Types

### 2.1 Commands Defined But Without Execution Logic

The plan defines these commands but **does not specify UIRenderer2D execution**:

```python
# Cmd2DCircle — needs arc tessellation or SDF shader
# Cmd2DArc — partial circle
# Cmd2DLineStrip — polyline with optional closure
# Cmd2DText — requires font system (see Section 3)
# Cmd2DBlit — texture sampling
```

### 2.2 Missing Command Types

```python
# Gradients (useful for waveform fill)
@dataclass(frozen=True)
class Cmd2DGradientRect:
    x: float
    y: float
    w: float
    h: float
    color_top: Color4
    color_bottom: Color4
    corner_radius: float = 0.0

# Triangle (for playhead, markers)
@dataclass(frozen=True)
class Cmd2DTriangle:
    x0: float
    y0: float
    x1: float
    y1: float
    x2: float
    y2: float
    color: Color4

# Triangles instanced (for many markers)
@dataclass(frozen=True)
class Cmd2DTrianglesInstanced:
    instance_data: bytes  # [x0,y0,x1,y1,x2,y2,r,g,b,a] per instance
    instance_count: int

# Bezier curve (for smooth connections)
@dataclass(frozen=True)
class Cmd2DBezier:
    points: bytes  # Control points
    point_count: int
    color: Color4
    width: float = 1.0
    segments: int = 32  # Tessellation quality

# Clear region (for partial updates)
@dataclass(frozen=True)
class Cmd2DClear:
    x: int
    y: int
    w: int
    h: int
    color: Color4
```

---

## 3. Text Rendering — MAJOR GAP

The plan mentions `Cmd2DText` but provides **no implementation details**.

### 3.1 Required Components

```python
# Font system needs:
class FontAtlas:
    """Packed glyph texture atlas."""
    texture: moderngl.Texture
    glyphs: Dict[str, GlyphInfo]  # char -> metrics
    
@dataclass
class GlyphInfo:
    char: str
    x: int  # Atlas position
    y: int
    w: int  # Glyph size
    h: int
    bearing_x: float  # Offset from cursor
    bearing_y: float
    advance: float  # How much to move cursor after this glyph

class FontManager:
    """Load and cache fonts."""
    def load_font(self, path: str, size: int) -> FontAtlas: ...
    def get_font(self, font_id: str) -> FontAtlas: ...
    
class TextLayout:
    """Convert text string to positioned glyphs."""
    def layout(self, text: str, font: FontAtlas, max_width: float = None) -> List[PositionedGlyph]: ...
    def measure(self, text: str, font: FontAtlas) -> Tuple[float, float]: ...

@dataclass
class PositionedGlyph:
    glyph: GlyphInfo
    x: float
    y: float
```

### 3.2 Missing Task Items for Text

Add to Phase 2:

| ID | Task | File | Status |
|----|------|------|--------|
| 2.33 | Define GlyphInfo dataclass | `engine/ui/text/font.py` | ⬜ |
| 2.34 | Implement FontAtlas class | `engine/ui/text/font.py` | ⬜ |
| 2.35 | Implement FontManager class | `engine/ui/text/font.py` | ⬜ |
| 2.36 | Load default font (FiraMono or similar) | `engine/ui/text/font.py` | ⬜ |
| 2.37 | Write text vertex shader | `engine/render/shaders/text.vert` | ⬜ |
| 2.38 | Write text fragment shader (SDF) | `engine/render/shaders/text.frag` | ⬜ |
| 2.39 | Implement TextLayout class | `engine/ui/text/layout.py` | ⬜ |
| 2.40 | Implement Cmd2DText execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.41 | Add instanced text rendering | `engine/ui/renderer_2d.py` | ⬜ |
| 2.42 | Test text at multiple sizes | `tests/test_text.py` | ⬜ |

---

## 4. UIRenderer2D — Missing Implementation Details

### 4.1 Class Structure Not Specified

```python
class UIRenderer2D:
    """Execute 2D drawing commands."""
    
    # MISSING: Constructor internals
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Shader programs
        self._rect_program: moderngl.Program = None
        self._rect_instanced_program: moderngl.Program = None
        self._line_program: moderngl.Program = None
        self._line_instanced_program: moderngl.Program = None
        self._waveform_program: moderngl.Program = None
        self._text_program: moderngl.Program = None
        
        # Geometry buffers
        self._unit_quad_vao: moderngl.VertexArray = None
        self._unit_line_vao: moderngl.VertexArray = None
        
        # Instance buffers (reused each frame)
        self._rect_instance_buffer: moderngl.Buffer = None
        self._line_instance_buffer: moderngl.Buffer = None
        self._max_instances: int = 10000
        
        # State tracking
        self._current_clip: Optional[Tuple[int,int,int,int]] = None
        self._viewport_size: Tuple[int, int] = (800, 600)
        
        # Font system
        self._font_manager: FontManager = None
        self._text_batch_buffer: moderngl.Buffer = None
        
        self._setup_pipelines()
    
    # MISSING: Shader loading
    def _setup_pipelines(self):
        shader_dir = Path(__file__).parent.parent / "render" / "shaders"
        
        # Load each shader pair
        self._rect_program = self._load_program(
            shader_dir / "rect.vert",
            shader_dir / "rect.frag"
        )
        # ... etc
    
    def _load_program(self, vert_path: Path, frag_path: Path) -> moderngl.Program:
        with open(vert_path) as f:
            vert_src = f.read()
        with open(frag_path) as f:
            frag_src = f.read()
        return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
    
    # MISSING: Buffer management
    def _ensure_instance_buffer_capacity(self, count: int, buffer: moderngl.Buffer, stride: int):
        required = count * stride
        if buffer.size < required:
            buffer.orphan(max(required, buffer.size * 2))
    
    # MISSING: Clip stack (for nested clipping)
    _clip_stack: List[Tuple[int,int,int,int]]
    
    def push_clip(self, x, y, w, h):
        self._clip_stack.append((x, y, w, h))
        self._apply_clip(x, y, w, h)
    
    def pop_clip(self):
        self._clip_stack.pop()
        if self._clip_stack:
            self._apply_clip(*self._clip_stack[-1])
        else:
            self.ctx.scissor = None
    
    def _apply_clip(self, x, y, w, h):
        # Convert to GL coordinates (bottom-left origin)
        gl_y = self._viewport_size[1] - y - h
        self.ctx.scissor = (x, gl_y, w, h)
```

### 4.2 Command Dispatch Pattern

```python
def execute(self, batch: DrawBatch):
    """Execute all commands in a batch."""
    for cmd in batch.commands:
        self._dispatch(cmd)

def _dispatch(self, cmd: Command2D):
    """Route command to appropriate handler."""
    # MISSING: This dispatch table
    handlers = {
        Cmd2DSetClip: self._exec_set_clip,
        Cmd2DRect: self._exec_rect,
        Cmd2DLine: self._exec_line,
        Cmd2DLinesInstanced: self._exec_lines_instanced,
        Cmd2DRectsInstanced: self._exec_rects_instanced,
        Cmd2DText: self._exec_text,
        Cmd2DWaveform: self._exec_waveform,
        Cmd2DCircle: self._exec_circle,
        Cmd2DArc: self._exec_arc,
        Cmd2DLineStrip: self._exec_line_strip,
        Cmd2DBlit: self._exec_blit,
    }
    
    handler = handlers.get(type(cmd))
    if handler:
        handler(cmd)
    else:
        raise ValueError(f"Unknown command type: {type(cmd)}")
```

---

## 5. DrawContext Integration — Missing Details

### 5.1 How DrawContext Connects to DrawBatch

The plan mentions updating `draw.py` but doesn't show the interface:

```python
# MISSING: DrawContext class definition
class DrawContext:
    """Accumulates drawing commands during widget traversal."""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._batches: List[DrawBatch] = []
        self._current_batch: DrawBatch = DrawBatch()
        self._transform_stack: List[Transform2D] = []
        self._clip_stack: List[Rect] = []
    
    # MISSING: Transform support
    def push_transform(self, x: float, y: float):
        """Push translation onto stack."""
        ...
    
    def pop_transform(self):
        """Pop transform from stack."""
        ...
    
    def _apply_transform(self, x: float, y: float) -> Tuple[float, float]:
        """Apply current transform to point."""
        ...
    
    # MISSING: Clip integration
    def push_clip(self, rect: Rect):
        self._clip_stack.append(rect)
        self._current_batch.add(Cmd2DSetClip(
            int(rect.x), int(rect.y), 
            int(rect.w), int(rect.h), 
            enabled=True
        ))
    
    def pop_clip(self):
        self._clip_stack.pop()
        if self._clip_stack:
            r = self._clip_stack[-1]
            self._current_batch.add(Cmd2DSetClip(int(r.x), int(r.y), int(r.w), int(r.h), enabled=True))
        else:
            self._current_batch.add(Cmd2DSetClip(0, 0, 0, 0, enabled=False))
    
    # MISSING: Z-ordering
    def begin_layer(self, z_order: int):
        """Start a new batch at given z-order."""
        self._current_batch = DrawBatch(z_order=z_order)
    
    def end_layer(self):
        """Finish current batch and add to list."""
        self._batches.append(self._current_batch)
        self._current_batch = DrawBatch()
    
    # MISSING: Finalization
    def finalize(self) -> List[DrawBatch]:
        """Return all batches sorted by z-order."""
        if self._current_batch.commands:
            self._batches.append(self._current_batch)
        return sorted(self._batches, key=lambda b: b.z_order)
    
    # Drawing methods (delegate to current batch)
    def rect(self, x, y, w, h, color, **kwargs):
        tx, ty = self._apply_transform(x, y)
        self._current_batch.rect(tx, ty, w, h, color, **kwargs)
    
    def line(self, x0, y0, x1, y1, color, width=1.0):
        tx0, ty0 = self._apply_transform(x0, y0)
        tx1, ty1 = self._apply_transform(x1, y1)
        self._current_batch.line(tx0, ty0, tx1, ty1, color, width)
    
    # ... etc
```

### 5.2 Widget Base Class Updates

```python
# MISSING: Updated Widget interface
class Widget(ABC):
    # Existing...
    
    # NEW: Drawing method signature
    @abstractmethod
    def draw(self, ctx: DrawContext):
        """Draw this widget using the provided context."""
        pass
    
    # MISSING: Focus state
    focused: bool = False
    
    # MISSING: Dirty tracking
    _dirty: bool = True
    
    def mark_dirty(self):
        self._dirty = True
        if self.parent:
            self.parent.mark_dirty()
    
    # MISSING: Hit testing refinement
    def point_in_widget(self, x: float, y: float) -> bool:
        """Check if point is inside this widget's bounds."""
        return (self.rect.x <= x < self.rect.x + self.rect.w and
                self.rect.y <= y < self.rect.y + self.rect.h)
```

---

## 6. Input Handling — Missing Specifications

### 6.1 Keyboard Bindings Not Defined

```python
# MISSING: Key binding constants
class KeyBindings:
    # Transport
    PLAY_PAUSE = "Space"
    STOP = "Escape"
    REWIND = "Home"
    FORWARD = "End"
    
    # Navigation
    SCROLL_LEFT = "Left"
    SCROLL_RIGHT = "Right"
    SCROLL_LEFT_BAR = "Ctrl+Left"
    SCROLL_RIGHT_BAR = "Ctrl+Right"
    
    # Zoom
    ZOOM_IN = "Ctrl+Plus"
    ZOOM_OUT = "Ctrl+Minus"
    ZOOM_FIT = "Ctrl+0"
    
    # Selection
    SELECT_ALL = "Ctrl+A"
    DESELECT = "Escape"
    DELETE = "Delete"
    
    # Undo
    UNDO = "Ctrl+Z"
    REDO = "Ctrl+Shift+Z"
```

### 6.2 Mouse Button Mapping

```python
# MISSING: Mouse button semantics
class MouseButtons:
    LEFT = 0    # Select, drag
    MIDDLE = 1  # Pan (scroll)
    RIGHT = 2   # Context menu

# MISSING: Input state tracking
@dataclass
class InputState:
    mouse_x: float = 0.0
    mouse_y: float = 0.0
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    buttons_down: Set[int] = field(default_factory=set)
    keys_down: Set[str] = field(default_factory=set)
    modifiers: Set[str] = field(default_factory=set)  # "Ctrl", "Shift", "Alt"
    scroll_x: float = 0.0
    scroll_y: float = 0.0
```

### 6.3 GraphPanel Input Handling Details

```python
# MISSING: Detailed input handling in GraphPanel
class GraphPanel(Widget, ABC):
    # ... existing ...
    
    # Input state
    _dragging: bool = False
    _drag_button: int = -1
    _hover_beat: Optional[float] = None
    _last_mouse_x: float = 0.0
    _last_mouse_y: float = 0.0
    _drag_velocity_samples: List[Tuple[float, float]] = []  # (time, x) pairs
    
    def on_pointer_down(self, x: float, y: float, button: int) -> bool:
        if not self.point_in_widget(x, y):
            return False
        
        if button == MouseButtons.LEFT:
            # Start selection or item interaction
            return self._handle_left_click(x, y)
        elif button == MouseButtons.MIDDLE:
            # Start panning
            self.time_camera.begin_drag(x - self.rect.x)
            self._dragging = True
            self._drag_button = button
            return True
        elif button == MouseButtons.RIGHT:
            # Context menu
            return self._handle_right_click(x, y)
        return False
    
    def on_pointer_move(self, x: float, y: float, dx: float, dy: float) -> bool:
        local_x = x - self.rect.x
        self._hover_beat = self.time_camera.px_to_beat(local_x)
        
        if self._dragging:
            if self._drag_button == MouseButtons.MIDDLE:
                self.time_camera.update_drag(local_x)
                # Record velocity sample for inertia
                now = time.perf_counter()
                self._drag_velocity_samples.append((now, local_x))
                # Keep only recent samples
                self._drag_velocity_samples = [
                    s for s in self._drag_velocity_samples 
                    if now - s[0] < 0.1
                ]
            return True
        return False
    
    def on_pointer_up(self, x: float, y: float, button: int) -> bool:
        if self._dragging and button == self._drag_button:
            if button == MouseButtons.MIDDLE:
                # Calculate release velocity
                velocity = self._calculate_drag_velocity()
                self.time_camera.end_drag(x - self.rect.x, velocity)
            self._dragging = False
            self._drag_button = -1
            return True
        return False
    
    def _calculate_drag_velocity(self) -> float:
        """Calculate velocity from recent drag samples."""
        if len(self._drag_velocity_samples) < 2:
            return 0.0
        samples = self._drag_velocity_samples
        dt = samples[-1][0] - samples[0][0]
        if dt < 0.001:
            return 0.0
        dx = samples[-1][1] - samples[0][1]
        return dx / dt
    
    def on_scroll(self, x: float, y: float, dx: float, dy: float) -> bool:
        if not self.point_in_widget(x, y):
            return False
        
        local_x = x - self.rect.x
        
        # Ctrl+scroll = zoom, regular scroll = pan
        if "Ctrl" in self._modifiers:
            self.time_camera.zoom(dy, local_x)
        else:
            # Convert scroll to beats
            scroll_beats = dy * 0.5  # Adjust sensitivity
            self.time_camera.left_beat += scroll_beats
        
        return True
    
    # MISSING: Abstract methods for subclass-specific interaction
    @abstractmethod
    def _handle_left_click(self, x: float, y: float) -> bool:
        """Handle left click for selection/interaction."""
        pass
    
    def _handle_right_click(self, x: float, y: float) -> bool:
        """Handle right click for context menu."""
        return False  # Override in subclass
```

---

## 7. Missing Integration Points

### 7.1 How Does renderer_2d.py Connect to renderer.py?

The plan shows two renderers but doesn't explain integration:

```python
# MISSING: UIRenderer updates
class UIRenderer:
    """Existing UI renderer — needs update."""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        # EXISTING: 3D rendering setup
        self._3d_renderer = Renderer(ctx)
        
        # NEW: 2D rendering
        self._2d_renderer = UIRenderer2D(ctx)
    
    def render_frame(self, batches: List[DrawBatch], viewports: List[Viewport3D]):
        """Render a complete frame."""
        
        # 1. Render 3D viewports to textures
        for viewport in viewports:
            self._3d_renderer.render(viewport)
        
        # 2. Execute 2D batches (sorted by z-order)
        for batch in sorted(batches, key=lambda b: b.z_order):
            self._2d_renderer.execute(batch)
```

### 7.2 How Does WindowManager Route Input?

```python
# MISSING: WindowManager input routing
class WindowManager:
    # ... existing ...
    
    _focused_widget: Optional[Widget] = None
    _hovered_widget: Optional[Widget] = None
    _capture_widget: Optional[Widget] = None  # Widget capturing all input (during drag)
    
    def on_mouse_move(self, x: float, y: float):
        dx = x - self._last_mouse_x
        dy = y - self._last_mouse_y
        self._last_mouse_x = x
        self._last_mouse_y = y
        
        if self._capture_widget:
            self._capture_widget.on_pointer_move(x, y, dx, dy)
            return
        
        # Hit test to find hovered widget
        widget = self._hit_test(x, y)
        
        if widget != self._hovered_widget:
            if self._hovered_widget:
                self._hovered_widget.on_pointer_leave()
            self._hovered_widget = widget
            if widget:
                widget.on_pointer_enter()
        
        if widget:
            widget.on_pointer_move(x, y, dx, dy)
    
    def on_mouse_button(self, button: int, pressed: bool, x: float, y: float):
        if pressed:
            widget = self._hit_test(x, y)
            if widget:
                if widget.on_pointer_down(x, y, button):
                    self._capture_widget = widget
                    self._focused_widget = widget
        else:
            if self._capture_widget:
                self._capture_widget.on_pointer_up(x, y, button)
                self._capture_widget = None
    
    def on_key(self, key: str, pressed: bool, modifiers: Set[str]):
        # First try focused widget
        if self._focused_widget:
            if pressed:
                if self._focused_widget.on_key_down(key, modifiers):
                    return
            else:
                if self._focused_widget.on_key_up(key, modifiers):
                    return
        
        # Then try global shortcuts
        self._handle_global_shortcut(key, pressed, modifiers)
    
    def _hit_test(self, x: float, y: float) -> Optional[Widget]:
        """Find topmost widget at position."""
        # Traverse widget tree in reverse draw order
        return self._hit_test_recursive(self.root, x, y)
    
    def _hit_test_recursive(self, widget: Widget, x: float, y: float) -> Optional[Widget]:
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

## 8. Missing File Structure Details

### 8.1 Files Not Listed in Structure

```
spectro/
├── engine/
│   ├── ui/
│   │   ├── text/                    # MISSING: Text rendering module
│   │   │   ├── __init__.py
│   │   │   ├── font.py              # FontAtlas, FontManager
│   │   │   └── layout.py            # TextLayout
│   │   ├── input.py                 # MISSING: Input handling
│   │   └── focus.py                 # MISSING: Focus management
│   ├── render/
│   │   └── shaders/
│   │       ├── text.vert            # MISSING: Text shaders
│   │       ├── text.frag
│   │       ├── circle.vert          # MISSING: Circle shaders
│   │       ├── circle.frag
│   │       └── common.glsl          # MISSING: Shared shader utilities
│   └── util/                        # MISSING: Utilities
│       ├── __init__.py
│       ├── color.py                 # Color utilities
│       └── math.py                  # Math utilities
├── assets/                          # MISSING: Assets folder
│   └── fonts/
│       └── FiraMono-Regular.ttf     # Default font
└── config/                          # MISSING: Configuration
    └── default.toml
```

---

## 9. Missing Tests

### 9.1 Additional Test Cases Needed

```python
# test_time_camera.py — MISSING TESTS
def test_scroll_limits(): ...
def test_animate_to_beat(): ...
def test_snap_to_grid(): ...
def test_visibility_queries(): ...
def test_frame_rate_independence(): ...

# test_transport.py — MISSING TESTS
def test_seek_operations(): ...
def test_tap_tempo(): ...
def test_callbacks_fire(): ...
def test_time_signature_changes(): ...

# test_panel_sync.py — MISSING TESTS
def test_multiple_panels_share_camera(): ...
def test_zoom_preserves_alignment(): ...
def test_scroll_preserves_alignment(): ...

# test_renderer_2d.py — MISSING FILE
def test_clip_stack(): ...
def test_instanced_batching_threshold(): ...
def test_buffer_resize(): ...
def test_shader_uniform_binding(): ...

# test_input.py — MISSING FILE
def test_drag_velocity_calculation(): ...
def test_modifier_tracking(): ...
def test_focus_traversal(): ...

# test_text.py — MISSING FILE
def test_font_loading(): ...
def test_text_measurement(): ...
def test_text_alignment(): ...
def test_text_clipping(): ...
```

---

## 10. Performance Considerations Not Addressed

### 10.1 Missing Details

1. **When to batch vs. when to draw immediately**
   - Threshold for instancing (plan says "> 4 items" but doesn't specify where this check happens)

2. **Buffer allocation strategy**
   - Pre-allocate? Grow on demand? Pool?

3. **Draw call minimization**
   - Sort commands by shader/texture before dispatch?

4. **Dirty region tracking**
   - Partial updates for static content?

5. **GPU memory budget**
   - What's the limit for instance buffers?
   - What happens when exceeded?

---

## 11. Error Handling Not Specified

```python
# MISSING: Error handling throughout

# Shader compilation failures
def _load_program(self, vert_path, frag_path):
    try:
        ...
    except moderngl.Error as e:
        logging.error(f"Shader compilation failed: {e}")
        raise ShaderCompilationError(f"Failed to compile {vert_path}") from e

# Font loading failures
def load_font(self, path: str, size: int):
    if not Path(path).exists():
        raise FontNotFoundError(f"Font not found: {path}")
    ...

# Buffer overflow
def _ensure_capacity(self, count: int, buffer: moderngl.Buffer, stride: int):
    required = count * stride
    if required > MAX_BUFFER_SIZE:
        raise BufferOverflowError(f"Required {required} bytes exceeds max {MAX_BUFFER_SIZE}")
    ...

# Invalid beat values
def beat_to_px(self, beat: float) -> float:
    if not math.isfinite(beat):
        raise ValueError(f"Invalid beat value: {beat}")
    ...
```

---

## 12. Suggested Task List Additions

Add these to the existing task list:

### Phase 1 Additions

| ID | Task | File | Status |
|----|------|------|--------|
| 1.27 | Add scroll limits to TimeCamera | `engine/time/camera.py` | ⬜ |
| 1.28 | Add visibility queries | `engine/time/camera.py` | ⬜ |
| 1.29 | Add snap_to_grid() | `engine/time/camera.py` | ⬜ |
| 1.30 | Add animate_to_beat() | `engine/time/camera.py` | ⬜ |
| 1.31 | Add Transport.play/pause/stop | `engine/time/transport.py` | ⬜ |
| 1.32 | Add Transport.seek_to_beat() | `engine/time/transport.py` | ⬜ |
| 1.33 | Add Transport callbacks | `engine/time/transport.py` | ⬜ |
| 1.34 | Add derived properties to TransportState | `engine/time/transport.py` | ⬜ |

### Phase 2 Additions

| ID | Task | File | Status |
|----|------|------|--------|
| 2.33 | Implement FontAtlas class | `engine/ui/text/font.py` | ⬜ |
| 2.34 | Implement FontManager class | `engine/ui/text/font.py` | ⬜ |
| 2.35 | Implement TextLayout class | `engine/ui/text/layout.py` | ⬜ |
| 2.36 | Write text shaders | `engine/render/shaders/text.*` | ⬜ |
| 2.37 | Implement Cmd2DText execution | `engine/ui/renderer_2d.py` | ⬜ |
| 2.38 | Add Cmd2DCircle support | `engine/render/commands_2d.py` | ⬜ |
| 2.39 | Add Cmd2DArc support | `engine/render/commands_2d.py` | ⬜ |
| 2.40 | Add Cmd2DLineStrip support | `engine/render/commands_2d.py` | ⬜ |
| 2.41 | Implement circle shader | `engine/render/shaders/circle.*` | ⬜ |
| 2.42 | Add DrawBatch convenience methods | `engine/render/commands_2d.py` | ⬜ |
| 2.43 | Add Cmd2DGradientRect | `engine/render/commands_2d.py` | ⬜ |
| 2.44 | Add Cmd2DTriangle/Instanced | `engine/render/commands_2d.py` | ⬜ |

### Phase 3 Additions

| ID | Task | File | Status |
|----|------|------|--------|
| 3.36 | Add GridBuilder.build_with_labels() | `engine/ui/panels/grid_builder.py` | ⬜ |
| 3.37 | Add GridBuilder alternating backgrounds | `engine/ui/panels/grid_builder.py` | ⬜ |
| 3.38 | Add playhead triangle marker | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.39 | Implement GraphPanel._handle_left_click() | `engine/ui/panels/graph_panel.py` | ⬜ |
| 3.40 | Add drag velocity calculation | `engine/ui/panels/graph_panel.py` | ⬜ |

### Phase 4 Additions

| ID | Task | File | Status |
|----|------|------|--------|
| 4.9 | Define KeyBindings class | `engine/ui/input.py` | ⬜ |
| 4.10 | Define MouseButtons class | `engine/ui/input.py` | ⬜ |
| 4.11 | Define InputState dataclass | `engine/ui/input.py` | ⬜ |
| 4.12 | Implement focus traversal | `engine/ui/focus.py` | ⬜ |
| 4.13 | Implement capture widget | `engine/ui/window_manager.py` | ⬜ |
| 4.14 | Implement hit testing | `engine/ui/window_manager.py` | ⬜ |
| 4.15 | Add global shortcuts | `engine/ui/window_manager.py` | ⬜ |

### New Phase: Error Handling & Polish

| ID | Task | File | Status |
|----|------|------|--------|
| 7.1 | Add ShaderCompilationError | `engine/core/errors.py` | ⬜ |
| 7.2 | Add FontNotFoundError | `engine/core/errors.py` | ⬜ |
| 7.3 | Add BufferOverflowError | `engine/core/errors.py` | ⬜ |
| 7.4 | Add validation to TimeCamera | `engine/time/camera.py` | ⬜ |
| 7.5 | Add validation to Transport | `engine/time/transport.py` | ⬜ |
| 7.6 | Add shader hot-reload support | `engine/ui/renderer_2d.py` | ⬜ |
| 7.7 | Add configuration file support | `engine/config.py` | ⬜ |

---

## 13. Summary: Critical Missing Items

### Must Have (Blocking)

1. **Text rendering system** — Cannot display bar numbers, beat labels, or any text UI without this
2. **DrawContext class definition** — How widgets actually draw is undefined
3. **UIRenderer2D internals** — Shader loading, buffer management, dispatch table
4. **WindowManager input routing** — Hit testing, focus, capture

### Should Have (Important)

1. **Transport control methods** — play/pause/stop/seek
2. **TimeCamera animation** — Smooth scrolling
3. **Circle/arc rendering** — SyncPanel needs these
4. **Error handling** — Graceful failures

### Nice to Have (Polish)

1. **Hot reload** — Faster iteration
2. **Configuration files** — Persistable settings
3. **Gradient rects** — Visual polish
4. **Bezier curves** — Smooth connections

---

*This gap analysis should be addressed before implementation begins. Update SPECTRO_DEVELOPMENT_PLAN.md with missing items.*

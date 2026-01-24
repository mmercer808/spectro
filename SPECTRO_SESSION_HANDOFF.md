# SPECTRO Engine Session Handoff

> **For next Claude session**: Read this first, then SPECTRO_SPEC_CONSOLIDATED.md and SPECTRO_IMPLEMENTATION_PLAN.md

---

## What Was Built This Session

We created the foundational engine layer (~3000 lines) with unified 3D coordinate space and signal-based architecture.

### Files Created

```
engine/
├── __init__.py                 # Main package entry
├── core/
│   ├── __init__.py            
│   ├── math3d.py              # 3D math library (850 lines)
│   ├── signal.py              # Observer pattern hub (300 lines)
│   ├── scene.py               # Unified space + entities (400 lines)
│   └── manager.py             # Top-level coordinator (300 lines)
└── time/
    ├── __init__.py
    ├── camera.py              # TimeCamera with Mat4 (600 lines)
    └── transport.py           # Playback state (500 lines)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      SceneManager                                │
│  - Owns everything, orchestrates frame updates                  │
│  - save()/load() for project serialization                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Scene     │  │  TimeCamera  │  │  Transport   │          │
│  │              │  │              │  │              │          │
│  │ Unified 3D   │  │ beat ↔ pixel │  │ play/pause   │          │
│  │ space with   │  │ mapping +    │  │ seek, BPM    │          │
│  │ entities     │  │ Mat4 support │  │ time sig     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                      │
│                    SignalBridge                                  │
│              (routes events between systems)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete Signature Reference

### math3d.py

```python
# === Vectors ===
@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0
    def __add__(self, other: Vec2) -> Vec2
    def __sub__(self, other: Vec2) -> Vec2
    def __mul__(self, scalar: float) -> Vec2
    def __neg__(self) -> Vec2
    def dot(self, other: Vec2) -> float
    def length(self) -> float
    def length_squared(self) -> float
    def normalized(self) -> Vec2
    def lerp(self, other: Vec2, t: float) -> Vec2
    def to_tuple(self) -> Tuple[float, float]
    @staticmethod
    def from_tuple(t: Tuple[float, float]) -> Vec2

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    def __add__(self, other: Vec3) -> Vec3
    def __sub__(self, other: Vec3) -> Vec3
    def __mul__(self, scalar: float) -> Vec3
    def __neg__(self) -> Vec3
    def dot(self, other: Vec3) -> float
    def cross(self, other: Vec3) -> Vec3
    def length(self) -> float
    def length_squared(self) -> float
    def normalized(self) -> Vec3
    def lerp(self, other: Vec3, t: float) -> Vec3
    def to_tuple(self) -> Tuple[float, float, float]
    def xy(self) -> Vec2
    @staticmethod
    def from_tuple(t: Tuple[float, float, float]) -> Vec3
    @staticmethod
    def unit_x() -> Vec3
    @staticmethod
    def unit_y() -> Vec3
    @staticmethod
    def unit_z() -> Vec3

@dataclass
class Vec4:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    def __add__(self, other: Vec4) -> Vec4
    def __sub__(self, other: Vec4) -> Vec4
    def __mul__(self, scalar: float) -> Vec4
    def dot(self, other: Vec4) -> float
    def xyz(self) -> Vec3
    def to_vec3(self) -> Vec3  # perspective divide
    def to_tuple(self) -> Tuple[float, float, float, float]
    @staticmethod
    def from_vec3(v: Vec3, w: float = 1.0) -> Vec4
    @staticmethod
    def point(x: float, y: float, z: float) -> Vec4
    @staticmethod
    def direction(x: float, y: float, z: float) -> Vec4

# === Matrices ===
class Mat3:
    m: Tuple[float, ...]  # 9 values, row-major
    def __init__(self, values: Tuple[float, ...] = None)
    def __getitem__(self, idx: Tuple[int, int]) -> float
    def __matmul__(self, other: Union[Mat3, Vec3]) -> Union[Mat3, Vec3]
    def transpose(self) -> Mat3
    def determinant(self) -> float
    def to_tuple(self) -> Tuple[float, ...]
    def to_list_column_major(self) -> list  # for GPU upload
    @staticmethod
    def identity() -> Mat3
    @staticmethod
    def scale(sx: float, sy: float) -> Mat3
    @staticmethod
    def translate(tx: float, ty: float) -> Mat3
    @staticmethod
    def rotate(angle: float) -> Mat3

class Mat4:
    m: Tuple[float, ...]  # 16 values, row-major
    def __init__(self, values: Tuple[float, ...] = None)
    def __getitem__(self, idx: Tuple[int, int]) -> float
    def __matmul__(self, other: Union[Mat4, Vec4, Vec3]) -> Union[Mat4, Vec4, Vec3]
    def transpose(self) -> Mat4
    def to_tuple(self) -> Tuple[float, ...]
    def to_list_column_major(self) -> list  # for GPU upload
    def to_mat3(self) -> Mat3
    def inverse(self) -> Mat4
    @staticmethod
    def identity() -> Mat4
    @staticmethod
    def scale(sx: float, sy: float = None, sz: float = None) -> Mat4
    @staticmethod
    def translate(tx: float, ty: float, tz: float) -> Mat4
    @staticmethod
    def translate_vec(v: Vec3) -> Mat4
    @staticmethod
    def rotate_x(angle: float) -> Mat4
    @staticmethod
    def rotate_y(angle: float) -> Mat4
    @staticmethod
    def rotate_z(angle: float) -> Mat4
    @staticmethod
    def rotate_axis(axis: Vec3, angle: float) -> Mat4
    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3 = None) -> Mat4
    @staticmethod
    def ortho(left: float, right: float, bottom: float, top: float, near: float, far: float) -> Mat4
    @staticmethod
    def perspective(fov_y: float, aspect: float, near: float, far: float) -> Mat4

# === Quaternion ===
@dataclass
class Quat:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    def __mul__(self, other: Quat) -> Quat
    def conjugate(self) -> Quat
    def length(self) -> float
    def normalized(self) -> Quat
    def rotate_vec(self, v: Vec3) -> Vec3
    def to_mat4(self) -> Mat4
    def slerp(self, other: Quat, t: float) -> Quat
    @staticmethod
    def identity() -> Quat
    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> Quat
    @staticmethod
    def from_euler(pitch: float, yaw: float, roll: float) -> Quat

# === Transform ===
@dataclass
class Transform:
    position: Vec3 = None
    rotation: Quat = None
    scale: Vec3 = None
    def to_mat4(self) -> Mat4
    def lerp(self, other: Transform, t: float) -> Transform

# === Easing Functions ===
def ease_linear(t: float) -> float
def ease_in_quad(t: float) -> float
def ease_out_quad(t: float) -> float
def ease_in_out_quad(t: float) -> float
def ease_in_cubic(t: float) -> float
def ease_out_cubic(t: float) -> float
def ease_in_out_cubic(t: float) -> float
def ease_in_elastic(t: float) -> float
def ease_out_elastic(t: float) -> float
def ease_out_bounce(t: float) -> float

# === Utilities ===
def clamp(value: float, min_val: float, max_val: float) -> float
def lerp(a: float, b: float, t: float) -> float
def inverse_lerp(a: float, b: float, value: float) -> float
def remap(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float
def smoothstep(edge0: float, edge1: float, x: float) -> float
def deg_to_rad(degrees: float) -> float
def rad_to_deg(radians: float) -> float
```

### signal.py

```python
# === Signal Constants ===
SIGNAL_DT = 'dt'
SIGNAL_TRANSPORT_CHANGED = 'transport_changed'
SIGNAL_SEEK = 'seek'
SIGNAL_PLAY = 'play'
SIGNAL_PAUSE = 'pause'
SIGNAL_STOP = 'stop'
SIGNAL_BPM_CHANGED = 'bpm_changed'
SIGNAL_TIME_SIG_CHANGED = 'time_sig_changed'
SIGNAL_VIEW_CHANGED = 'view_changed'
SIGNAL_ZOOM_CHANGED = 'zoom_changed'
SIGNAL_SCROLL = 'scroll'
SIGNAL_POINTER_DOWN = 'pointer_down'
SIGNAL_POINTER_MOVE = 'pointer_move'
SIGNAL_POINTER_UP = 'pointer_up'
SIGNAL_KEY_DOWN = 'key_down'
SIGNAL_KEY_UP = 'key_up'
SIGNAL_WHEEL = 'wheel'
SIGNAL_ENTITY_ADDED = 'entity_added'
SIGNAL_ENTITY_REMOVED = 'entity_removed'
SIGNAL_ENTITY_CHANGED = 'entity_changed'
SIGNAL_SELECTION_CHANGED = 'selection_changed'
SIGNAL_DIRTY = 'dirty'
SIGNAL_RESIZE = 'resize'

# === Connection Handle ===
@dataclass
class Connection:
    signal: str
    callback_id: int
    bridge: SignalBridge = None
    def disconnect(self)

# === Signal Bridge ===
class SignalBridge:
    def connect(self, signal: str, handler: Callable) -> Connection
    def connect_weak(self, signal: str, obj: object, method_name: str) -> Connection
    def disconnect_all(self, signal: str = None)
    def emit(self, signal: str, *args, **kwargs)
    def emit_later(self, signal: str, *args, **kwargs)
    def block(self, signal: str)
    def unblock(self, signal: str)
    def is_connected(self, signal: str) -> bool

# === Debugger ===
class SignalDebugger:
    def __init__(self, bridge: SignalBridge)
    def watch(self, signal: str)
    def unwatch(self, signal: str)
    def watch_all(self, enabled: bool = True)
    def detach(self)

# === Mixins ===
class SignalEmitter:
    def bind_bridge(self, bridge: SignalBridge)
    def emit(self, signal: str, *args, **kwargs)
    def connect(self, signal: str, handler: Callable) -> Optional[Connection]

class SignalReceiver:
    def bind_bridge(self, bridge: SignalBridge)
    def subscribe(self, signal: str, handler: Callable)
    def unsubscribe_all(self)

# === Decorator ===
def on_signal(bridge: SignalBridge, signal: str)
```

### scene.py

```python
# === Entity Types ===
class EntityType(Enum):
    GENERIC = auto()
    AUDIO_CLIP = auto()
    MIDI_EVENT = auto()
    MARKER = auto()
    REGION = auto()
    AUTOMATION = auto()

# === Entity ===
@dataclass
class Entity:
    id: str  # UUID
    entity_type: EntityType = EntityType.GENERIC
    name: str = ""
    position: Vec3  # x=beat, y=freq, z=intensity
    extent: Vec3    # duration, freq_range, intensity_range
    color: Tuple[float, float, float, float]
    visible: bool = True
    selected: bool = False
    metadata: Dict[str, Any]
    
    # Properties
    @property beat -> float
    @property end_beat -> float
    @property duration_beats -> float
    @property frequency -> float
    @property intensity -> float
    
    def contains_beat(self, beat: float) -> bool
    def overlaps_range(self, start_beat: float, end_beat: float) -> bool
    def to_dict(self) -> dict
    @staticmethod
    def from_dict(data: dict) -> Entity

# === Specialized Entities ===
@dataclass
class AudioClipEntity(Entity):
    file_path: str
    sample_rate: int
    channels: int
    source_start_sample: int
    source_end_sample: int

@dataclass
class MidiEventEntity(Entity):
    note: int
    velocity: int
    channel: int

@dataclass
class MarkerEntity(Entity):
    marker_type: str
    label: str

# === Scene Bounds ===
@dataclass
class SceneBounds:
    time_min: float = 0.0
    time_max: float = 1000.0
    freq_min: float = 20.0
    freq_max: float = 20000.0
    intensity_min: float = 0.0
    intensity_max: float = 1.0

# === Scene ===
class Scene:
    def __init__(self, bridge: SignalBridge = None)
    def bind_bridge(self, bridge: SignalBridge)
    
    # Entity Management
    def add(self, entity: Entity) -> Entity
    def remove(self, entity_id: str) -> Optional[Entity]
    def get(self, entity_id: str) -> Optional[Entity]
    def update(self, entity: Entity)
    def clear(self)
    
    # Queries
    def all(self) -> Iterator[Entity]
    def by_type(self, entity_type: EntityType) -> Iterator[Entity]
    def in_time_range(self, start_beat: float, end_beat: float) -> Iterator[Entity]
    def at_beat(self, beat: float) -> Iterator[Entity]
    def in_box(self, min_pos: Vec3, max_pos: Vec3) -> Iterator[Entity]
    def visible(self) -> Iterator[Entity]
    @property count -> int
    
    # Selection
    def select(self, entity_id: str, add_to_selection: bool = False)
    def deselect(self, entity_id: str)
    def deselect_all(self)
    def selected(self) -> Iterator[Entity]
    @property selection_count -> int
    
    # Serialization
    def to_dict(self) -> dict
    def from_dict(self, data: dict)
    def save(self, path: str)
    def load(self, path: str)
```

### manager.py

```python
@dataclass
class SceneManagerConfig:
    default_bpm: float = 120.0
    default_time_sig_numerator: int = 4
    default_time_sig_denominator: int = 4
    auto_follow: bool = True

class SceneManager:
    # Owned systems
    bridge: SignalBridge
    scene: Scene
    transport: Transport
    time_camera: TimeCamera
    
    def __init__(self, config: SceneManagerConfig = None)
    
    # Frame Loop
    def update(self, dt: float = None)
    def resize(self, width: int, height: int)
    
    # Transport Control
    def play(self)
    def pause(self)
    def stop(self)
    def toggle_playback(self)
    def seek(self, beat: float)
    def set_bpm(self, bpm: float)
    @property is_playing -> bool
    @property current_beat -> float
    @property bpm -> float
    
    # View Control
    def scroll_to_beat(self, beat: float, animate: bool = True)
    def zoom_in(self)
    def zoom_out(self)
    def zoom_to_fit_selection(self)
    def set_follow_mode(self, enabled: bool)
    
    # Serialization
    def to_dict(self) -> Dict[str, Any]
    def from_dict(self, data: Dict[str, Any])
    def save(self, path: str)
    def load(self, path: str)
    
    # Info
    @property frame_id -> int
    def get_stats(self) -> Dict[str, Any]
```

### camera.py

```python
class TimeCameraMode(Enum):
    FREE_SCROLL = auto()
    FOLLOW_PLAYHEAD = auto()
    SNAP_TO_BARS = auto()

@dataclass
class TimeCameraConfig:
    playhead_ratio: float = 0.3
    follow_strength: float = 8.0
    min_px_per_beat: float = 10.0
    max_px_per_beat: float = 500.0
    min_window_beats: float = 1.0
    max_window_beats: float = 256.0
    inertia_friction: float = 0.92
    inertia_threshold: float = 0.1
    snap_threshold_px: float = 5.0
    default_animation_duration: float = 0.3

@dataclass
class TimeCamera(SignalEmitter):
    left_beat: float = 0.0
    window_beats: float = 16.0
    mode: TimeCameraMode
    config: TimeCameraConfig
    
    # Core Mapping
    @property _px_per_beat -> float
    def beat_to_px(self, beat: float) -> float
    def px_to_beat(self, px: float) -> float
    def beat_to_screen(self, beat: float) -> Vec2
    def screen_to_beat(self, screen_pos: Vec2) -> float
    
    # Visibility
    def is_beat_visible(self, beat: float) -> bool
    def is_range_visible(self, start_beat: float, end_beat: float) -> bool
    def get_visible_range(self) -> Tuple[float, float]
    @property right_beat -> float
    @property center_beat -> float
    
    # Grid Snapping
    def snap_to_grid(self, beat: float, subdivision: int = 4) -> float
    def nearest_beat(self, px: float) -> float
    def nearest_bar(self, px: float, beats_per_bar: float = 4.0) -> float
    def snap_px_to_grid(self, px: float, subdivision: int = 4) -> float
    
    # Grid Generation
    def iter_bar_beats(self, beats_per_bar: float = 4.0) -> Iterator[float]
    def iter_beat_positions(self) -> Iterator[float]
    def iter_subdivision_beats(self, subdivision: int = 4) -> Iterator[float]
    def get_visible_bar_lines_px(self, beats_per_bar: float = 4.0) -> List[float]
    def get_visible_beat_lines_px(self, beats_per_bar: float = 4.0) -> List[float]
    
    # Matrix / Transform
    def get_time_matrix(self) -> Mat4
    def get_inverse_time_matrix(self) -> Mat4
    def get_view_projection(self, ortho_height: float = None) -> Mat4
    
    # Drag / Pan
    def begin_drag(self, mouse_x: float)
    def update_drag(self, mouse_x: float)
    def end_drag(self, mouse_x: float, velocity_px_per_sec: float = 0.0)
    def cancel_drag(self)
    
    # Zoom
    def zoom(self, delta: float, anchor_px: float)
    def zoom_to_fit(self, start_beat: float, end_beat: float, padding: float = 0.1)
    def set_zoom_level(self, px_per_beat: float)
    
    # Animation
    def animate_to_beat(self, target_beat: float, duration: float = None)
    def animate_to_center_on(self, beat: float, duration: float = None)
    def jump_to_beat(self, beat: float)
    
    # Frame Update
    def update(self, dt: float, playhead_beat: float = None)
    
    # Signal Bridge
    def bind(self, bridge: SignalBridge)
    
    # Layout
    def set_panel_size(self, width: float, height: float)
```

### transport.py

```python
@dataclass(frozen=True)
class TimeSignature:
    numerator: int = 4
    denominator: int = 4
    @property beats_per_bar -> float

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
    
    @property phase_in_beat -> float
    @property phase_in_bar -> float
    @property current_bar -> int
    @property current_beat_in_bar -> int
    @property seconds_per_beat -> float
    @property beats_per_second -> float
    @property bar_duration_beats -> float
    @property bar_duration_seconds -> float

class Transport:
    playing: bool
    playhead_beat: float
    playhead_time: float
    bpm: float
    time_sig: TimeSignature
    loop_start: Optional[float]
    loop_end: Optional[float]
    loop_enabled: bool
    
    def __init__(self, bpm: float = 120.0, time_sig: TimeSignature = None)
    
    # Playback Control
    def play(self)
    def pause(self)
    def stop(self)
    def toggle(self)
    
    # Seeking
    def seek_to_beat(self, beat: float)
    def seek_to_time(self, seconds: float)
    def seek_by_bars(self, delta: int)
    def seek_to_bar(self, bar: int)
    def seek_to_next_bar(self)
    def seek_to_previous_bar(self)
    
    # Configuration
    def set_bpm(self, bpm: float)
    def set_time_signature(self, numerator: int, denominator: int)
    def tap_tempo(self, tap_times: List[float]) -> float
    
    # Looping
    def set_loop(self, start: float, end: float)
    def clear_loop(self)
    def toggle_loop(self)
    
    # Frame Update
    def update(self, dt: float) -> TransportState
    
    # Callbacks
    def on_beat(self, callback: BeatCallback)
    def on_bar(self, callback: BarCallback)
    def on_loop(self, callback: LoopCallback)
    
    # Conversion
    def beat_to_time(self, beat: float) -> float
    def time_to_beat(self, time: float) -> float
    def beat_to_bar(self, beat: float) -> int
    def bar_to_beat(self, bar: int) -> float
    def format_time(self, beat: float = None) -> str
    def format_bar_beat(self, beat: float = None) -> str
```

---

## Next Steps

1. **Copy engine/ folder** to your spectro repo
2. **Run existing tests**: `python -m pytest tests/test_time_camera.py -v`
3. **Continue with WP2**: 2D Drawing Foundation (DrawBatch, UIRenderer2D, shaders)

### Immediate Tasks (Session 2.1)
- Create `engine/render/commands_2d.py` with Cmd2D* dataclasses
- Create `engine/render/renderer_2d.py` with DrawBatch
- Basic rect/line shaders

---

## Loading Signatures in Your Environment

To auto-generate this signature list from code:

```python
# scripts/extract_signatures.py
import ast
import inspect

def extract_signatures(filepath):
    """Extract class and function signatures from Python file."""
    with open(filepath) as f:
        tree = ast.parse(f.read())
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            print(f"class {node.name}:")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = [a.arg for a in item.args.args]
                    print(f"    def {item.name}({', '.join(args)})")
        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            args = [a.arg for a in node.args.args]
            print(f"def {node.name}({', '.join(args)})")

# Usage:
# python scripts/extract_signatures.py engine/core/math3d.py
```

Or use `pydoc` / `help()`:
```python
from engine import TimeCamera
help(TimeCamera)
```

Or generate stubs with `stubgen`:
```bash
pip install mypy
stubgen engine/ -o stubs/
```

This creates `.pyi` stub files with just signatures - perfect for reference.

---

*Session ended: TimeCamera + math3d + signal + scene + manager complete*
*Next: WP2 - 2D Drawing Foundation*

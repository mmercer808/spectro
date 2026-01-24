# SPECTRO Engine Handoff - Session Complete

## What Was Built This Session

Created foundational engine architecture (~3000 lines):

```
engine/
├── __init__.py
├── core/
│   ├── math3d.py    # Vec2/3/4, Mat3/4, Quat, Transform, easing
│   ├── signal.py    # SignalBridge observer pattern
│   ├── scene.py     # Scene (unified 3D space), Entity system
│   └── manager.py   # SceneManager coordinator
└── time/
    ├── camera.py    # TimeCamera with 3D matrix support
    └── transport.py # Transport playback control
```

## Key Architecture Decisions

1. **Unified 3D Space**: Scene coordinates are (X=beats, Y=frequency, Z=intensity)
2. **SignalBridge**: Central event routing - components emit/subscribe to signals
3. **TimeCamera provides Mat4**: `get_time_matrix()` for shader uniforms
4. **Immutable TransportState**: Thread-safe snapshots for render thread

## Next Steps

- [ ] Write tests for new components (signal, scene, manager)
- [ ] WP2: 2D Drawing Foundation (DrawBatch, UIRenderer2D)
- [ ] WP3: Text Rendering (FreeType integration)

---

# COMPLETE API SIGNATURES

## engine/time/camera.py

```python
class TimeCameraMode(Enum):
    FREE_SCROLL
    FOLLOW_PLAYHEAD
    SNAP_TO_BARS

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
    mode: TimeCameraMode = TimeCameraMode.FOLLOW_PLAYHEAD
    _panel_width_px: float = 800.0
    _panel_height_px: float = 600.0
    min_left_beat: float = -4.0
    max_right_beat: float = 10000.0
    config: TimeCameraConfig
    
    # Core Mapping
    def _px_per_beat(self) -> float
    def beat_to_px(self, beat: float) -> float
    def px_to_beat(self, px: float) -> float
    def beat_to_screen(self, beat: float) -> Vec2
    def screen_to_beat(self, screen_pos: Vec2) -> float
    
    # Visibility
    def is_beat_visible(self, beat: float) -> bool
    def is_range_visible(self, start_beat: float, end_beat: float) -> bool
    def get_visible_range(self) -> Tuple[float, float]
    def right_beat(self) -> float
    def center_beat(self) -> float
    
    # Grid/Snapping
    def snap_to_grid(self, beat: float, subdivision: int = 4) -> float
    def nearest_beat(self, px: float) -> float
    def nearest_bar(self, px: float, beats_per_bar: float = 4.0) -> float
    def snap_px_to_grid(self, px: float, subdivision: int = 4) -> float
    
    # Grid Line Generation
    def iter_bar_beats(self, beats_per_bar: float = 4.0) -> Iterator[float]
    def iter_beat_positions(self) -> Iterator[float]
    def iter_subdivision_beats(self, subdivision: int = 4) -> Iterator[float]
    def get_visible_bar_lines_px(self, beats_per_bar: float = 4.0) -> List[float]
    def get_visible_beat_lines_px(self, beats_per_bar: float = 4.0) -> List[float]
    
    # Matrix/Transform (for 3D/shader integration)
    def get_time_matrix(self) -> Mat4
    def get_inverse_time_matrix(self) -> Mat4
    def get_view_projection(self, ortho_height: float = None) -> Mat4
    
    # Drag/Pan
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
    
    # Signal Integration
    def bind(self, bridge: SignalBridge)
    def set_panel_size(self, width: float, height: float)
```

## engine/time/transport.py

```python
@dataclass(frozen=True)
class TimeSignature:
    numerator: int = 4
    denominator: int = 4
    def beats_per_bar(self) -> float

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
    
    def phase_in_beat(self) -> float
    def phase_in_bar(self) -> float
    def current_bar(self) -> int
    def current_beat_in_bar(self) -> int
    def seconds_per_beat(self) -> float
    def beats_per_second(self) -> float
    def bar_duration_beats(self) -> float
    def bar_duration_seconds(self) -> float

class Transport:
    def __init__(self, bpm: float = 120.0, time_sig: TimeSignature = None)
    
    def play(self)
    def pause(self)
    def stop(self)
    def toggle(self)
    
    def seek_to_beat(self, beat: float)
    def seek_to_time(self, seconds: float)
    def seek_by_bars(self, delta: int)
    def seek_to_bar(self, bar: int)
    def seek_to_next_bar(self)
    def seek_to_previous_bar(self)
    
    def set_bpm(self, bpm: float)
    def set_time_signature(self, numerator: int, denominator: int)
    def tap_tempo(self, tap_times: List[float]) -> float
    
    def set_loop(self, start: float, end: float)
    def clear_loop(self)
    def toggle_loop(self)
    
    def update(self, dt: float) -> TransportState
    
    def on_beat(self, callback: BeatCallback)
    def on_bar(self, callback: BarCallback)
    def on_loop(self, callback: LoopCallback)
    
    def beat_to_time(self, beat: float) -> float
    def time_to_beat(self, time: float) -> float
    def beat_to_bar(self, beat: float) -> int
    def bar_to_beat(self, bar: int) -> float
    def format_time(self, beat: float = None) -> str
    def format_bar_beat(self, beat: float = None) -> str
```

## engine/core/math3d.py

```python
@dataclass
class Vec2:
    x: float = 0.0
    y: float = 0.0
    def dot(self, other: Vec2) -> float
    def length(self) -> float
    def normalized(self) -> Vec2
    def lerp(self, other: Vec2, t: float) -> Vec2
    def to_tuple(self) -> Tuple[float, float]
    @staticmethod
    def from_tuple(t) -> Vec2

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    def dot(self, other: Vec3) -> float
    def cross(self, other: Vec3) -> Vec3
    def length(self) -> float
    def normalized(self) -> Vec3
    def lerp(self, other: Vec3, t: float) -> Vec3
    def to_tuple(self) -> Tuple[float, float, float]
    def xy(self) -> Vec2
    @staticmethod
    def from_tuple(t) -> Vec3
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
    def dot(self, other: Vec4) -> float
    def xyz(self) -> Vec3
    def to_vec3(self) -> Vec3
    def to_tuple(self) -> Tuple[float, float, float, float]
    @staticmethod
    def from_vec3(v: Vec3, w: float = 1.0) -> Vec4
    @staticmethod
    def point(x, y, z) -> Vec4
    @staticmethod
    def direction(x, y, z) -> Vec4

class Mat3:
    def __init__(self, values: Tuple[float, ...] = None)
    def __matmul__(self, other) -> Union[Mat3, Vec3]
    def transpose(self) -> Mat3
    def determinant(self) -> float
    def to_list_column_major(self) -> list
    @staticmethod
    def identity() -> Mat3
    @staticmethod
    def scale(sx, sy) -> Mat3
    @staticmethod
    def translate(tx, ty) -> Mat3
    @staticmethod
    def rotate(angle) -> Mat3

class Mat4:
    def __init__(self, values: Tuple[float, ...] = None)
    def __matmul__(self, other) -> Union[Mat4, Vec4, Vec3]
    def transpose(self) -> Mat4
    def inverse(self) -> Mat4
    def to_list_column_major(self) -> list
    def to_mat3(self) -> Mat3
    @staticmethod
    def identity() -> Mat4
    @staticmethod
    def scale(sx, sy=None, sz=None) -> Mat4
    @staticmethod
    def translate(tx, ty, tz) -> Mat4
    @staticmethod
    def translate_vec(v: Vec3) -> Mat4
    @staticmethod
    def rotate_x(angle) -> Mat4
    @staticmethod
    def rotate_y(angle) -> Mat4
    @staticmethod
    def rotate_z(angle) -> Mat4
    @staticmethod
    def rotate_axis(axis: Vec3, angle) -> Mat4
    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3 = None) -> Mat4
    @staticmethod
    def ortho(left, right, bottom, top, near, far) -> Mat4
    @staticmethod
    def perspective(fov_y, aspect, near, far) -> Mat4

@dataclass
class Quat:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0
    def __mul__(self, other: Quat) -> Quat
    def conjugate(self) -> Quat
    def normalized(self) -> Quat
    def rotate_vec(self, v: Vec3) -> Vec3
    def to_mat4(self) -> Mat4
    def slerp(self, other: Quat, t: float) -> Quat
    @staticmethod
    def identity() -> Quat
    @staticmethod
    def from_axis_angle(axis: Vec3, angle: float) -> Quat
    @staticmethod
    def from_euler(pitch, yaw, roll) -> Quat

@dataclass
class Transform:
    position: Vec3 = None
    rotation: Quat = None
    scale: Vec3 = None
    def to_mat4(self) -> Mat4
    def lerp(self, other: Transform, t: float) -> Transform

# Easing
def ease_linear(t) -> float
def ease_in_quad(t) -> float
def ease_out_quad(t) -> float
def ease_in_out_quad(t) -> float
def ease_in_cubic(t) -> float
def ease_out_cubic(t) -> float
def ease_in_out_cubic(t) -> float
def ease_in_elastic(t) -> float
def ease_out_elastic(t) -> float
def ease_out_bounce(t) -> float

# Utilities
def clamp(value, min_val, max_val) -> float
def lerp(a, b, t) -> float
def inverse_lerp(a, b, value) -> float
def remap(value, in_min, in_max, out_min, out_max) -> float
def smoothstep(edge0, edge1, x) -> float
def deg_to_rad(degrees) -> float
def rad_to_deg(radians) -> float
```

## engine/core/signal.py

```python
# Signal Constants
SIGNAL_DT = 'dt'
SIGNAL_TRANSPORT_CHANGED = 'transport_changed'
SIGNAL_SEEK = 'seek'
SIGNAL_PLAY = 'play'
SIGNAL_PAUSE = 'pause'
SIGNAL_STOP = 'stop'
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
SIGNAL_DIRTY = 'dirty'
SIGNAL_RESIZE = 'resize'

@dataclass
class Connection:
    def disconnect(self)

class SignalBridge:
    def connect(self, signal: str, handler: Callable) -> Connection
    def connect_weak(self, signal: str, obj: object, method_name: str) -> Connection
    def disconnect_all(self, signal: str = None)
    def emit(self, signal: str, *args, **kwargs)
    def block(self, signal: str)
    def unblock(self, signal: str)
    def is_connected(self, signal: str) -> bool

class SignalEmitter:  # mixin
    def bind_bridge(self, bridge: SignalBridge)
    def emit(self, signal: str, *args, **kwargs)

class SignalReceiver:  # mixin
    def bind_bridge(self, bridge: SignalBridge)
    def subscribe(self, signal: str, handler: Callable)
    def unsubscribe_all(self)
```

## engine/core/scene.py

```python
class EntityType(Enum):
    GENERIC, AUDIO_CLIP, MIDI_EVENT, MARKER, REGION, AUTOMATION

@dataclass
class Entity:
    id: str
    entity_type: EntityType = EntityType.GENERIC
    name: str = ""
    position: Vec3  # (beat, frequency, intensity)
    extent: Vec3 = Vec3(1, 1, 1)
    color: Tuple[float, float, float, float]
    visible: bool = True
    selected: bool = False
    metadata: Dict[str, Any]
    
    def beat(self) -> float
    def end_beat(self) -> float
    def duration_beats(self) -> float
    def frequency(self) -> float
    def intensity(self) -> float
    def contains_beat(self, beat: float) -> bool
    def overlaps_range(self, start_beat: float, end_beat: float) -> bool
    def to_dict(self) -> dict
    @staticmethod
    def from_dict(data: dict) -> Entity

class Scene:
    def __init__(self, bridge: SignalBridge = None)
    def bind_bridge(self, bridge: SignalBridge)
    
    def add(self, entity: Entity) -> Entity
    def remove(self, entity_id: str) -> Optional[Entity]
    def get(self, entity_id: str) -> Optional[Entity]
    def update(self, entity: Entity)
    def clear(self)
    
    def all(self) -> Iterator[Entity]
    def by_type(self, entity_type: EntityType) -> Iterator[Entity]
    def in_time_range(self, start_beat: float, end_beat: float) -> Iterator[Entity]
    def at_beat(self, beat: float) -> Iterator[Entity]
    def in_box(self, min_pos: Vec3, max_pos: Vec3) -> Iterator[Entity]
    def visible(self) -> Iterator[Entity]
    def count(self) -> int
    
    def select(self, entity_id: str, add_to_selection: bool = False)
    def deselect(self, entity_id: str)
    def deselect_all(self)
    def selected(self) -> Iterator[Entity]
    
    def to_dict(self) -> dict
    def from_dict(self, data: dict)
    def save(self, path: str)
    def load(self, path: str)
```

## engine/core/manager.py

```python
@dataclass
class SceneManagerConfig:
    default_bpm: float = 120.0
    default_time_sig_numerator: int = 4
    default_time_sig_denominator: int = 4
    auto_follow: bool = True

class SceneManager:
    bridge: SignalBridge
    scene: Scene
    transport: Transport
    time_camera: TimeCamera
    
    def __init__(self, config: SceneManagerConfig = None)
    
    def update(self, dt: float = None)
    def resize(self, width: int, height: int)
    
    def play(self)
    def pause(self)
    def stop(self)
    def toggle_playback(self)
    def seek(self, beat: float)
    def set_bpm(self, bpm: float)
    def is_playing(self) -> bool
    def current_beat(self) -> float
    def bpm(self) -> float
    
    def scroll_to_beat(self, beat: float, animate: bool = True)
    def zoom_in(self)
    def zoom_out(self)
    def zoom_to_fit_selection(self)
    def set_follow_mode(self, enabled: bool)
    
    def to_dict(self) -> Dict[str, Any]
    def from_dict(self, data: Dict[str, Any])
    def save(self, path: str)
    def load(self, path: str)
    
    def frame_id(self) -> int
    def get_stats(self) -> Dict[str, Any]
```

---

## Usage Example

```python
from engine import SceneManager, Entity, Vec3

manager = SceneManager()

entity = Entity(
    name="Kick",
    position=Vec3(0.0, 80.0, 0.8),  # beat, freq, intensity
    extent=Vec3(0.25, 40.0, 0.2),
)
manager.scene.add(entity)

manager.play()
manager.set_bpm(128.0)

while running:
    manager.update()
    matrix = manager.time_camera.get_time_matrix()
    # render...

manager.save("project.spectro")
```

---

## Files Location

All files in `/mnt/user-data/outputs/engine/` - copy to spectro repo.

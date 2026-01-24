# engine/core/__init__.py
"""Core module - foundational types and systems."""

from .math3d import (
    Vec2, Vec3, Vec4,
    Mat3, Mat4,
    Quat,
    Transform,
    lerp, clamp, remap, smoothstep,
    ease_linear, ease_in_quad, ease_out_quad, ease_in_out_quad,
    ease_in_cubic, ease_out_cubic, ease_in_out_cubic,
    ease_in_elastic, ease_out_elastic, ease_out_bounce,
    deg_to_rad, rad_to_deg,
)

from .signal import (
    SignalBridge,
    Connection,
    SignalEmitter,
    SignalReceiver,
    SignalDebugger,
    on_signal,
    SIGNAL_DT,
    SIGNAL_TRANSPORT_CHANGED,
    SIGNAL_SEEK,
    SIGNAL_PLAY,
    SIGNAL_PAUSE,
    SIGNAL_STOP,
    SIGNAL_VIEW_CHANGED,
    SIGNAL_ZOOM_CHANGED,
    SIGNAL_SCROLL,
    SIGNAL_POINTER_DOWN,
    SIGNAL_POINTER_MOVE,
    SIGNAL_POINTER_UP,
    SIGNAL_KEY_DOWN,
    SIGNAL_KEY_UP,
    SIGNAL_WHEEL,
    SIGNAL_ENTITY_ADDED,
    SIGNAL_ENTITY_REMOVED,
    SIGNAL_ENTITY_CHANGED,
    SIGNAL_SELECTION_CHANGED,
    SIGNAL_DIRTY,
    SIGNAL_RESIZE,
)

from .scene import (
    Scene,
    SceneBounds,
    Entity,
    EntityType,
    AudioClipEntity,
    MidiEventEntity,
    MarkerEntity,
)

from .manager import (
    SceneManager,
    SceneManagerConfig,
)

__all__ = [
    'Vec2', 'Vec3', 'Vec4',
    'Mat3', 'Mat4',
    'Quat',
    'Transform',
    'lerp', 'clamp', 'remap', 'smoothstep',
    'SignalBridge',
    'Connection',
    'SignalEmitter',
    'SignalReceiver',
    'Scene',
    'SceneBounds',
    'Entity',
    'EntityType',
    'SceneManager',
    'SceneManagerConfig',
]

# engine/__init__.py
"""
SPECTRO Engine - Time-synchronized visualization system.
"""

from .core import (
    Vec2, Vec3, Vec4,
    Mat3, Mat4,
    Quat,
    Transform,
    lerp, clamp, remap, smoothstep,
    SignalBridge,
    SignalEmitter,
    SignalReceiver,
    Scene,
    Entity,
    EntityType,
    SceneManager,
    SceneManagerConfig,
)

from .time import (
    TimeCamera,
    TimeCameraMode,
    TimeCameraConfig,
    Transport,
    TransportState,
    TimeSignature,
)

__version__ = '0.1.0'

__all__ = [
    'SceneManager',
    'SceneManagerConfig',
    'Scene',
    'Entity',
    'EntityType',
    'SignalBridge',
    'SignalEmitter',
    'SignalReceiver',
    'Vec2', 'Vec3', 'Vec4',
    'Mat3', 'Mat4',
    'Quat',
    'Transform',
    'lerp', 'clamp', 'remap', 'smoothstep',
    'TimeCamera',
    'TimeCameraMode',
    'TimeCameraConfig',
    'Transport',
    'TransportState',
    'TimeSignature',
]

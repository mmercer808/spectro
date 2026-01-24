# engine/__init__.py
"""
SPECTRO Engine - Time-synchronized visualization system.

Core components:
- SceneManager: Top-level coordinator
- Scene: Unified 3D space (time × frequency × intensity)  
- TimeCamera: View transformation for time axis
- Transport: Playback state and control
- SignalBridge: Event routing system
"""

from .core import (
    # Math
    Vec2, Vec3, Vec4,
    Mat3, Mat4,
    Quat,
    Transform,
    lerp, clamp, remap, smoothstep,
    
    # Signals
    SignalBridge,
    SignalEmitter,
    SignalReceiver,
    
    # Scene
    Scene,
    Entity,
    EntityType,
    
    # Manager
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
    # Core
    'SceneManager',
    'SceneManagerConfig',
    'Scene',
    'Entity',
    'EntityType',
    'SignalBridge',
    'SignalEmitter',
    'SignalReceiver',
    
    # Math
    'Vec2', 'Vec3', 'Vec4',
    'Mat3', 'Mat4',
    'Quat',
    'Transform',
    'lerp', 'clamp', 'remap', 'smoothstep',
    
    # Time
    'TimeCamera',
    'TimeCameraMode',
    'TimeCameraConfig',
    'Transport',
    'TransportState',
    'TimeSignature',
]

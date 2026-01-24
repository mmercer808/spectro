# engine/core/manager.py
"""
SceneManager - Top-level coordinator.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import time as time_module

from .scene import Scene
from .signal import SignalBridge, SIGNAL_DT, SIGNAL_TRANSPORT_CHANGED, SIGNAL_RESIZE
from ..time.camera import TimeCamera, TimeCameraMode
from ..time.transport import Transport, TransportState


@dataclass
class SceneManagerConfig:
    default_bpm: float = 120.0
    default_time_sig_numerator: int = 4
    default_time_sig_denominator: int = 4
    auto_follow: bool = True


class SceneManager:
    """Top-level coordinator for the SPECTRO engine."""
    
    def __init__(self, config: SceneManagerConfig = None):
        self.config = config or SceneManagerConfig()
        
        self.bridge = SignalBridge()
        self.scene = Scene(self.bridge)
        self.transport = Transport(bpm=self.config.default_bpm)
        self.time_camera = TimeCamera(
            mode=TimeCameraMode.FOLLOW_PLAYHEAD if self.config.auto_follow 
                 else TimeCameraMode.FREE_SCROLL
        )
        
        self._connect_signals()
        
        self._transport_state: Optional[TransportState] = None
        self._frame_id: int = 0
        self._last_time: float = time_module.perf_counter()
        self._window_width: int = 800
        self._window_height: int = 600
    
    def _connect_signals(self):
        self.bridge.connect(SIGNAL_TRANSPORT_CHANGED, self._on_transport_changed)
        self.time_camera.bind(self.bridge)
    
    def _on_transport_changed(self, state: TransportState):
        self._transport_state = state
        self.time_camera._playhead_beat = state.playhead_beat
    
    def update(self, dt: float = None):
        self._frame_id += 1
        
        if dt is None:
            now = time_module.perf_counter()
            dt = now - self._last_time
            self._last_time = now
        
        self.bridge.emit(SIGNAL_DT, dt)
        
        self._transport_state = self.transport.update(dt)
        self.bridge.emit(SIGNAL_TRANSPORT_CHANGED, self._transport_state)
    
    def resize(self, width: int, height: int):
        self._window_width = width
        self._window_height = height
        self.time_camera.set_panel_size(float(width), float(height))
        self.bridge.emit(SIGNAL_RESIZE, width, height)
    
    def play(self):
        self.transport.play()
    
    def pause(self):
        self.transport.pause()
    
    def stop(self):
        self.transport.stop()
    
    def toggle_playback(self):
        self.transport.toggle()
    
    def seek(self, beat: float):
        self.transport.seek_to_beat(beat)
    
    def set_bpm(self, bpm: float):
        self.transport.set_bpm(bpm)
    
    @property
    def is_playing(self) -> bool:
        return self.transport.playing
    
    @property
    def current_beat(self) -> float:
        return self.transport.playhead_beat
    
    @property
    def bpm(self) -> float:
        return self.transport.bpm
    
    def scroll_to_beat(self, beat: float, animate: bool = True):
        if animate:
            self.time_camera.animate_to_beat(beat)
        else:
            self.time_camera.jump_to_beat(beat)
    
    def zoom_in(self):
        center_x = self._window_width / 2.0
        self.time_camera.zoom(1.0, center_x)
    
    def zoom_out(self):
        center_x = self._window_width / 2.0
        self.time_camera.zoom(-1.0, center_x)
    
    def zoom_to_fit_selection(self):
        selected = list(self.scene.selected())
        if not selected:
            return
        
        min_beat = min(e.beat for e in selected)
        max_beat = max(e.end_beat for e in selected)
        self.time_camera.zoom_to_fit(min_beat, max_beat)
    
    def set_follow_mode(self, enabled: bool):
        self.time_camera.mode = (
            TimeCameraMode.FOLLOW_PLAYHEAD if enabled 
            else TimeCameraMode.FREE_SCROLL
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': '1.0',
            'scene': self.scene.to_dict(),
            'transport': {
                'bpm': self.transport.bpm,
                'playhead_beat': self.transport.playhead_beat,
                'time_sig_numerator': self.transport.time_sig.numerator,
                'time_sig_denominator': self.transport.time_sig.denominator,
                'loop_enabled': self.transport.loop_enabled,
                'loop_start': self.transport.loop_start,
                'loop_end': self.transport.loop_end,
            },
            'view': {
                'left_beat': self.time_camera.left_beat,
                'window_beats': self.time_camera.window_beats,
                'mode': self.time_camera.mode.name,
            }
        }
    
    def from_dict(self, data: Dict[str, Any]):
        self.scene.from_dict(data.get('scene', {}))
        
        transport_data = data.get('transport', {})
        self.transport.set_bpm(transport_data.get('bpm', 120.0))
        self.transport.seek_to_beat(transport_data.get('playhead_beat', 0.0))
        self.transport.set_time_signature(
            transport_data.get('time_sig_numerator', 4),
            transport_data.get('time_sig_denominator', 4)
        )
        self.transport.loop_enabled = transport_data.get('loop_enabled', False)
        self.transport.loop_start = transport_data.get('loop_start')
        self.transport.loop_end = transport_data.get('loop_end')
        
        view_data = data.get('view', {})
        self.time_camera.left_beat = view_data.get('left_beat', 0.0)
        self.time_camera.window_beats = view_data.get('window_beats', 16.0)
        mode_name = view_data.get('mode', 'FOLLOW_PLAYHEAD')
        self.time_camera.mode = TimeCameraMode[mode_name]
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.from_dict(data)
    
    @property
    def frame_id(self) -> int:
        return self._frame_id
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'frame_id': self._frame_id,
            'entity_count': self.scene.count,
            'selection_count': self.scene.selection_count,
            'bpm': self.transport.bpm,
            'playhead_beat': self.transport.playhead_beat,
            'playing': self.transport.playing,
            'view_left': self.time_camera.left_beat,
            'view_right': self.time_camera.right_beat,
            'px_per_beat': self.time_camera._px_per_beat,
        }

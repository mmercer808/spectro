# engine/time/camera.py
"""
TimeCamera - View transformation for the time (beat) axis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Iterator
from enum import Enum, auto
import math

from ..core.math3d import Vec2, Mat4, lerp, clamp, ease_out_cubic
from ..core.signal import (
    SignalBridge, SignalEmitter,
    SIGNAL_VIEW_CHANGED, SIGNAL_ZOOM_CHANGED, SIGNAL_SCROLL, SIGNAL_DT
)


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
    """View transformation for the time axis."""
    
    left_beat: float = 0.0
    window_beats: float = 16.0
    mode: TimeCameraMode = TimeCameraMode.FOLLOW_PLAYHEAD
    _panel_width_px: float = 800.0
    _panel_height_px: float = 600.0
    min_left_beat: float = -4.0
    max_right_beat: float = 10000.0
    config: TimeCameraConfig = field(default_factory=TimeCameraConfig)
    
    _dragging: bool = False
    _drag_start_px: float = 0.0
    _drag_start_beat: float = 0.0
    _scroll_velocity: float = 0.0
    
    _animating: bool = False
    _anim_start_beat: float = 0.0
    _anim_target_beat: float = 0.0
    _anim_elapsed: float = 0.0
    _anim_duration: float = 0.3
    
    _playhead_beat: float = 0.0
    _bridge: SignalBridge = None
    
    @property
    def _px_per_beat(self) -> float:
        if self.window_beats == 0:
            return 50.0
        return self._panel_width_px / self.window_beats
    
    def beat_to_px(self, beat: float) -> float:
        return (beat - self.left_beat) * self._px_per_beat
    
    def px_to_beat(self, px: float) -> float:
        return self.left_beat + (px / self._px_per_beat)
    
    def beat_to_screen(self, beat: float) -> Vec2:
        return Vec2(self.beat_to_px(beat), 0.0)
    
    def screen_to_beat(self, screen_pos: Vec2) -> float:
        return self.px_to_beat(screen_pos.x)
    
    def is_beat_visible(self, beat: float) -> bool:
        return beat >= self.left_beat and beat < self.left_beat + self.window_beats
    
    def is_range_visible(self, start_beat: float, end_beat: float) -> bool:
        return start_beat < self.right_beat and end_beat > self.left_beat
    
    def get_visible_range(self) -> Tuple[float, float]:
        return (self.left_beat, self.left_beat + self.window_beats)
    
    @property
    def right_beat(self) -> float:
        return self.left_beat + self.window_beats
    
    @property
    def center_beat(self) -> float:
        return self.left_beat + self.window_beats / 2.0
    
    def snap_to_grid(self, beat: float, subdivision: int = 4) -> float:
        grid_size = 1.0 / subdivision
        return round(beat / grid_size) * grid_size
    
    def nearest_beat(self, px: float) -> float:
        return round(self.px_to_beat(px))
    
    def nearest_bar(self, px: float, beats_per_bar: float = 4.0) -> float:
        beat = self.px_to_beat(px)
        return round(beat / beats_per_bar) * beats_per_bar
    
    def snap_px_to_grid(self, px: float, subdivision: int = 4) -> float:
        beat = self.px_to_beat(px)
        snapped = self.snap_to_grid(beat, subdivision)
        snapped_px = self.beat_to_px(snapped)
        
        if abs(snapped_px - px) <= self.config.snap_threshold_px:
            return snapped_px
        return px
    
    def iter_bar_beats(self, beats_per_bar: float = 4.0) -> Iterator[float]:
        first_bar = math.floor(self.left_beat / beats_per_bar) * beats_per_bar
        beat = first_bar
        while beat <= self.right_beat:
            if beat >= self.left_beat:
                yield beat
            beat += beats_per_bar
    
    def iter_beat_positions(self) -> Iterator[float]:
        first = math.floor(self.left_beat)
        beat = first
        while beat <= self.right_beat:
            if beat >= self.left_beat:
                yield beat
            beat += 1.0
    
    def iter_subdivision_beats(self, subdivision: int = 4) -> Iterator[float]:
        step = 1.0 / subdivision
        first = math.floor(self.left_beat / step) * step
        beat = first
        while beat <= self.right_beat:
            if beat >= self.left_beat:
                yield beat
            beat += step
    
    def get_visible_bar_lines_px(self, beats_per_bar: float = 4.0) -> List[float]:
        return [self.beat_to_px(b) for b in self.iter_bar_beats(beats_per_bar)]
    
    def get_visible_beat_lines_px(self, beats_per_bar: float = 4.0) -> List[float]:
        lines = []
        for beat in self.iter_beat_positions():
            if beat % beats_per_bar != 0:
                lines.append(self.beat_to_px(beat))
        return lines
    
    def get_time_matrix(self) -> Mat4:
        scale_x = self._px_per_beat
        offset_x = -self.left_beat * scale_x
        
        return Mat4((
            scale_x, 0.0, 0.0, offset_x,
            0.0,     1.0, 0.0, 0.0,
            0.0,     0.0, 1.0, 0.0,
            0.0,     0.0, 0.0, 1.0
        ))
    
    def get_inverse_time_matrix(self) -> Mat4:
        inv_scale = 1.0 / self._px_per_beat if self._px_per_beat != 0 else 1.0
        offset_beat = self.left_beat
        
        return Mat4((
            inv_scale, 0.0, 0.0, offset_beat,
            0.0,       1.0, 0.0, 0.0,
            0.0,       0.0, 1.0, 0.0,
            0.0,       0.0, 0.0, 1.0
        ))
    
    def get_view_projection(self, ortho_height: float = None) -> Mat4:
        if ortho_height is None:
            ortho_height = self._panel_height_px
        
        proj = Mat4.ortho(
            left=0, 
            right=self._panel_width_px,
            bottom=ortho_height, 
            top=0,
            near=-1, 
            far=1
        )
        
        return proj @ self.get_time_matrix()
    
    def begin_drag(self, mouse_x: float):
        self._dragging = True
        self._drag_start_px = mouse_x
        self._drag_start_beat = self.left_beat
        self._scroll_velocity = 0.0
        self._animating = False
    
    def update_drag(self, mouse_x: float):
        if not self._dragging:
            return
        
        dx = mouse_x - self._drag_start_px
        delta_beats = dx / self._px_per_beat
        new_left = self._drag_start_beat - delta_beats
        
        new_left = clamp(new_left, self.min_left_beat, 
                        self.max_right_beat - self.window_beats)
        
        self.left_beat = new_left
        self._emit_view_changed()
    
    def end_drag(self, mouse_x: float, velocity_px_per_sec: float = 0.0):
        self._dragging = False
        self._scroll_velocity = -velocity_px_per_sec / self._px_per_beat
    
    def cancel_drag(self):
        self._dragging = False
        self._scroll_velocity = 0.0
    
    def zoom(self, delta: float, anchor_px: float):
        anchor_beat = self.px_to_beat(anchor_px)
        
        factor = 1.1 if delta < 0 else 0.9
        new_window = self.window_beats * factor
        
        new_window = clamp(new_window, 
                          self.config.min_window_beats,
                          self.config.max_window_beats)
        
        new_px_per_beat = self._panel_width_px / new_window
        if new_px_per_beat < self.config.min_px_per_beat:
            new_window = self._panel_width_px / self.config.min_px_per_beat
        elif new_px_per_beat > self.config.max_px_per_beat:
            new_window = self._panel_width_px / self.config.max_px_per_beat
        
        self.window_beats = new_window
        self.left_beat = anchor_beat - (anchor_px / self._px_per_beat)
        
        self.left_beat = clamp(self.left_beat, self.min_left_beat,
                              self.max_right_beat - self.window_beats)
        
        self._emit_view_changed()
        if self._bridge:
            self._bridge.emit(SIGNAL_ZOOM_CHANGED, self._px_per_beat)
    
    def zoom_to_fit(self, start_beat: float, end_beat: float, padding: float = 0.1):
        duration = end_beat - start_beat
        if duration <= 0:
            return
        
        padded_duration = duration * (1 + padding * 2)
        padded_start = start_beat - duration * padding
        
        self.window_beats = clamp(padded_duration,
                                 self.config.min_window_beats,
                                 self.config.max_window_beats)
        self.left_beat = padded_start
        
        self._emit_view_changed()
    
    def set_zoom_level(self, px_per_beat: float):
        px_per_beat = clamp(px_per_beat, 
                           self.config.min_px_per_beat,
                           self.config.max_px_per_beat)
        self.window_beats = self._panel_width_px / px_per_beat
        self._emit_view_changed()
    
    def animate_to_beat(self, target_beat: float, duration: float = None):
        if duration is None:
            duration = self.config.default_animation_duration
        
        self._animating = True
        self._anim_start_beat = self.left_beat
        self._anim_target_beat = clamp(target_beat, self.min_left_beat,
                                       self.max_right_beat - self.window_beats)
        self._anim_elapsed = 0.0
        self._anim_duration = duration
    
    def animate_to_center_on(self, beat: float, duration: float = None):
        target_left = beat - self.window_beats / 2.0
        self.animate_to_beat(target_left, duration)
    
    def jump_to_beat(self, beat: float):
        self._animating = False
        self.left_beat = clamp(beat, self.min_left_beat,
                              self.max_right_beat - self.window_beats)
        self._emit_view_changed()
    
    def update(self, dt: float, playhead_beat: float = None):
        if playhead_beat is not None:
            self._playhead_beat = playhead_beat
        
        if self._animating:
            self._update_animation(dt)
        elif not self._dragging and self.mode == TimeCameraMode.FOLLOW_PLAYHEAD:
            self._update_follow_mode(dt)
        elif not self._dragging and self.mode == TimeCameraMode.SNAP_TO_BARS:
            self._update_snap_mode(dt)
        
        if not self._dragging and abs(self._scroll_velocity) > self.config.inertia_threshold:
            self.left_beat += self._scroll_velocity * dt
            self._scroll_velocity *= self.config.inertia_friction
            
            self.left_beat = clamp(self.left_beat, self.min_left_beat,
                                  self.max_right_beat - self.window_beats)
            
            self._emit_view_changed()
        elif abs(self._scroll_velocity) <= self.config.inertia_threshold:
            self._scroll_velocity = 0.0
    
    def _update_animation(self, dt: float):
        self._anim_elapsed += dt
        
        if self._anim_elapsed >= self._anim_duration:
            self.left_beat = self._anim_target_beat
            self._animating = False
        else:
            t = self._anim_elapsed / self._anim_duration
            t = ease_out_cubic(t)
            self.left_beat = lerp(self._anim_start_beat, self._anim_target_beat, t)
        
        self._emit_view_changed()
    
    def _update_follow_mode(self, dt: float):
        target_left = self._playhead_beat - (self.window_beats * self.config.playhead_ratio)
        target_left = clamp(target_left, self.min_left_beat,
                           self.max_right_beat - self.window_beats)
        
        diff = target_left - self.left_beat
        if abs(diff) > 0.001:
            strength = self.config.follow_strength * dt
            self.left_beat += diff * min(strength, 1.0)
            self._emit_view_changed()
    
    def _update_snap_mode(self, dt: float, beats_per_bar: float = 4.0):
        playhead_px = self.beat_to_px(self._playhead_beat)
        threshold_px = self._panel_width_px * 0.7
        
        if playhead_px > threshold_px:
            next_bar = (math.floor(self._playhead_beat / beats_per_bar) + 1) * beats_per_bar
            target = next_bar - self.window_beats * self.config.playhead_ratio
            self.animate_to_beat(target)
    
    def bind(self, bridge: SignalBridge):
        self._bridge = bridge
        self.bind_bridge(bridge)
        bridge.connect(SIGNAL_DT, self._on_dt)
    
    def _on_dt(self, dt: float):
        self.update(dt, self._playhead_beat)
    
    def _emit_view_changed(self):
        if self._bridge:
            self._bridge.emit(SIGNAL_VIEW_CHANGED, self)
            self._bridge.emit(SIGNAL_SCROLL, self.left_beat)
    
    def set_panel_size(self, width: float, height: float):
        self._panel_width_px = width
        self._panel_height_px = height

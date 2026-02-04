"""
SPECTRO Demo Harness
====================

Three panels, one scene, context bounces off timeline.

Layout:
┌─────────────────────────────────────────────────────────────┐
│                     TRANSPORT BAR                           │
├─────────────────────────────┬───────────────────────────────┤
│     SEQUENCER GRID          │         3D VIEWPORT           │
│     (2D: X=beat, Y=lane)    │      (perspective cubes)      │
├─────────────────────────────┴───────────────────────────────┤
│     WAVEFORM PANEL (2D: X=beat, Y=amplitude)               │
└─────────────────────────────────────────────────────────────┘

Keys: 1-4 = drums, SPACE = play/pause, R = reset
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Iterator, Protocol
from abc import ABC, abstractmethod
import numpy as np

import moderngl
import moderngl_window as mglw


# =============================================================================
# CORE TYPES
# =============================================================================

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    
    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


# =============================================================================
# ENTITY SYSTEM
# =============================================================================

_entity_id = 0
def next_entity_id() -> str:
    global _entity_id
    _entity_id += 1
    return f"e_{_entity_id}"


@dataclass
class Entity:
    """An entity in the composition space."""
    id: str = field(default_factory=next_entity_id)
    position: Vec3 = field(default_factory=Vec3)  # X=beat, Y=lane, Z=velocity
    extent: Vec3 = field(default_factory=lambda: Vec3(0.25, 1.0, 1.0))
    color: Tuple[float, float, float, float] = (1.0, 0.5, 0.2, 1.0)
    metadata: Dict = field(default_factory=dict)
    fired: bool = False  # For playback


# =============================================================================
# TRANSPORT (Minimal)
# =============================================================================

class Transport:
    """Playback transport."""
    
    def __init__(self, bpm: float = 120.0):
        self.bpm = bpm
        self.playing = False
        self.playhead_beat = 0.0
        self.loop_start = 0.0
        self.loop_end = 16.0
        self.loop_enabled = True
        self._on_loop: List[Callable] = []
    
    def update(self, dt: float):
        if not self.playing:
            return
        
        beats_per_sec = self.bpm / 60.0
        self.playhead_beat += dt * beats_per_sec
        
        # Loop
        if self.loop_enabled and self.playhead_beat >= self.loop_end:
            self.playhead_beat = self.loop_start
            for cb in self._on_loop:
                cb()
    
    def play(self):
        self.playing = True
    
    def pause(self):
        self.playing = False
    
    def stop(self):
        self.playing = False
        self.playhead_beat = self.loop_start
    
    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()
    
    def on_loop(self, callback: Callable):
        self._on_loop.append(callback)


# =============================================================================
# TIME CAMERA (Minimal)
# =============================================================================

class TimeCamera:
    """Maps beats to pixels."""
    
    def __init__(self, px_per_beat: float = 60.0):
        self._px_per_beat = px_per_beat
        self._scroll_beat = 0.0
        self._panel_width = 640.0
    
    def beat_to_px(self, beat: float) -> float:
        return (beat - self._scroll_beat) * self._px_per_beat
    
    def px_to_beat(self, px: float) -> float:
        return px / self._px_per_beat + self._scroll_beat
    
    def zoom(self, delta: float, anchor_px: float = None):
        factor = 1.1 if delta > 0 else 0.9
        self._px_per_beat = max(10, min(200, self._px_per_beat * factor))
    
    def iter_beats(self, width: float) -> Iterator[float]:
        """Yield visible beat positions."""
        start = int(self._scroll_beat)
        end = int(self.px_to_beat(width)) + 2
        for beat in range(start, end):
            yield float(beat)
    
    def follow(self, playhead_beat: float, panel_width: float):
        """Keep playhead visible."""
        playhead_px = self.beat_to_px(playhead_beat)
        margin = panel_width * 0.7
        if playhead_px > margin:
            self._scroll_beat = playhead_beat - margin / self._px_per_beat


# =============================================================================
# TRACK
# =============================================================================

LANE_COLORS = [
    (1.0, 0.3, 0.2, 1.0),  # Kick - red
    (1.0, 0.6, 0.2, 1.0),  # Snare - orange
    (1.0, 0.9, 0.2, 1.0),  # HiHat - yellow
    (0.4, 1.0, 0.3, 1.0),  # Clap - green
]

LANE_NAMES = ["Kick", "Snare", "HiHat", "Clap"]
LANE_SAMPLES = ["kick", "snare", "hihat", "clap"]


@dataclass
class Track:
    """A lane in the sequencer."""
    index: int
    name: str
    sample: str
    color: Tuple[float, float, float, float]
    events: List[Entity] = field(default_factory=list)
    muted: bool = False
    
    def add_event(self, beat: float, velocity: int = 100, duration: float = 0.25) -> Entity:
        entity = Entity(
            position=Vec3(beat, self.index, velocity / 127.0),
            extent=Vec3(duration, 1.0, 1.0),
            color=self.color,
            metadata={'velocity': velocity, 'sample': self.sample}
        )
        self.events.append(entity)
        return entity
    
    def events_in_range(self, start: float, end: float) -> List[Entity]:
        return [e for e in self.events 
                if e.position.x < end and e.position.x + e.extent.x > start]


# =============================================================================
# SCENE (The Space)
# =============================================================================

class Scene:
    """
    The unified composition space.
    All panels observe this.
    """
    
    def __init__(self):
        self.transport = Transport(bpm=120.0)
        self.time_camera = TimeCamera()
        self.tracks: List[Track] = []
        self._observers: List[Callable[[Entity, str], None]] = []
        self._all_entities: Dict[str, Entity] = {}
        
        # Setup default tracks
        for i in range(4):
            self.tracks.append(Track(
                index=i,
                name=LANE_NAMES[i],
                sample=LANE_SAMPLES[i],
                color=LANE_COLORS[i],
            ))
        
        # Wire loop callback
        self.transport.on_loop(self._on_loop)
    
    def observe(self, callback: Callable[[Entity, str], None]):
        """Register panel as observer."""
        self._observers.append(callback)
    
    def _notify(self, entity: Entity, action: str):
        for cb in self._observers:
            cb(entity, action)
    
    def add_note(self, track_index: int, beat: float, velocity: int = 100) -> Optional[Entity]:
        """Add note to track and notify observers."""
        if track_index < 0 or track_index >= len(self.tracks):
            return None
        
        track = self.tracks[track_index]
        entity = track.add_event(beat, velocity)
        self._all_entities[entity.id] = entity
        self._notify(entity, "added")
        return entity
    
    def all_entities(self) -> Iterator[Entity]:
        return iter(self._all_entities.values())
    
    def _on_loop(self):
        """Reset fired flags on loop."""
        for entity in self._all_entities.values():
            entity.fired = False


# =============================================================================
# SIMPLE AUDIO (Placeholder)
# =============================================================================

class SimpleAudio:
    """
    Minimal audio - just prints triggers.
    Replace with real AudioEngine for sound.
    """
    
    def __init__(self):
        self._waveform = [0.0] * 512
        self._write_pos = 0
    
    def trigger(self, sample: str, velocity: float):
        print(f"  ♪ {sample} (vel={velocity:.2f})")
        # Add spike to waveform
        for i in range(20):
            idx = (self._write_pos + i) % len(self._waveform)
            self._waveform[idx] = velocity * (1.0 - i / 20.0)
        self._write_pos = (self._write_pos + 20) % len(self._waveform)
    
    def get_waveform(self, num_samples: int) -> List[float]:
        """Get recent waveform for display."""
        result = []
        for i in range(num_samples):
            idx = (self._write_pos - num_samples + i) % len(self._waveform)
            result.append(abs(self._waveform[idx]))
        return result
    
    def decay(self, dt: float):
        """Decay waveform over time."""
        for i in range(len(self._waveform)):
            self._waveform[i] *= 0.95


# =============================================================================
# INPUT SYSTEM
# =============================================================================

@dataclass
class InputEvent:
    """A pending input event."""
    track: int
    velocity: int
    beat: float  # When it should fire


class InputManager:
    """Collects input from all sources."""
    
    def __init__(self, scene: Scene):
        self.scene = scene
        self._pending: List[InputEvent] = []
        self._pattern_index = 0
        
        # Demo pattern (simulated input)
        self._pattern = [
            (0.0, 0, 100),   # Kick
            (1.0, 2, 70),    # HiHat
            (2.0, 1, 100),   # Snare
            (3.0, 2, 70),    # HiHat
            (4.0, 0, 100),   # Kick
            (5.0, 2, 70),    # HiHat
            (6.0, 1, 100),   # Snare
            (6.5, 0, 80),    # Kick
            (7.0, 2, 70),    # HiHat
        ]
    
    def key_pressed(self, key: int):
        """Handle keyboard drum trigger."""
        # Keys 1-4 map to tracks 0-3
        if 0 <= key <= 3:
            self._pending.append(InputEvent(
                track=key,
                velocity=100,
                beat=self.scene.transport.playhead_beat,
            ))
    
    def poll_pattern(self, current_beat: float) -> List[InputEvent]:
        """Poll simulated pattern."""
        events = []
        pattern_beat = current_beat % 8.0  # 8 beat loop
        
        # Check for pattern events to fire
        for beat, track, vel in self._pattern:
            # Fire if we just crossed this beat
            prev_beat = (current_beat - 0.05) % 8.0
            if prev_beat <= beat < pattern_beat or (prev_beat > pattern_beat and beat < pattern_beat):
                events.append(InputEvent(track=track, velocity=vel, beat=current_beat))
        
        return events
    
    def poll_all(self) -> List[InputEvent]:
        """Get all pending events."""
        events = list(self._pending)
        self._pending.clear()
        
        # Add pattern events if playing
        if self.scene.transport.playing:
            events.extend(self.poll_pattern(self.scene.transport.playhead_beat))
        
        return events


# =============================================================================
# CONTEXT FIRER (The Timeline Bounce)
# =============================================================================

@dataclass
class ExecutionContext:
    """Context assembled at fire time (LATE BINDING)."""
    beat: float
    track: int
    velocity: int
    timing_error_ms: float = 0.0


class ContextFirer:
    """
    Fires context when events hit the timeline.
    
    Input → Buffer → BOUNCE → Context → Entity + Audio
    """
    
    def __init__(self, scene: Scene, audio: SimpleAudio, input_mgr: InputManager):
        self.scene = scene
        self.audio = audio
        self.input = input_mgr
        self._callbacks: List[Callable[[ExecutionContext], None]] = []
    
    def on_event(self, callback: Callable[[ExecutionContext], None]):
        """Register callback for events."""
        self._callbacks.append(callback)
    
    def process(self, dt: float):
        """Process input and fire context."""
        
        # Poll all input sources
        for event in self.input.poll_all():
            
            # === THE BOUNCE ===
            # Assemble context with CURRENT state
            ctx = ExecutionContext(
                beat=self.scene.transport.playhead_beat,  # FRESH
                track=event.track,
                velocity=event.velocity,
            )
            
            # Fire callbacks
            for cb in self._callbacks:
                cb(ctx)
        
        # Also check for playback events (recorded notes firing)
        if self.scene.transport.playing:
            self._check_playback()
    
    def _check_playback(self):
        """Fire recorded events when playhead crosses them."""
        current = self.scene.transport.playhead_beat
        window = 0.1
        
        for track in self.scene.tracks:
            if track.muted:
                continue
            
            for entity in track.events_in_range(current - window, current + window):
                if not entity.fired and entity.position.x <= current:
                    entity.fired = True
                    
                    # Trigger audio (playback)
                    vel = entity.metadata.get('velocity', 100) / 127.0
                    self.audio.trigger(track.sample, vel)


# =============================================================================
# DRAW CONTEXT (2D Rendering)
# =============================================================================

class DrawContext:
    """Simple 2D drawing."""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.window_size = (1280, 720)
        self._setup_shaders()
    
    def _setup_shaders(self):
        self._prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            uniform vec2 u_offset;
            uniform vec2 u_size;
            uniform vec2 u_window;
            void main() {
                vec2 px = u_offset + in_pos * u_size;
                vec2 ndc = (px / u_window) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform vec4 u_color;
            out vec4 fragColor;
            void main() { fragColor = u_color; }
            """
        )
        
        quad = np.array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1], dtype='f4')
        vbo = self.ctx.buffer(quad)
        self._vao = self.ctx.vertex_array(self._prog, [(vbo, '2f', 'in_pos')])
    
    def rect(self, x: float, y: float, w: float, h: float, color: tuple):
        self._prog['u_offset'].value = (x, y)
        self._prog['u_size'].value = (w, h)
        self._prog['u_color'].value = color
        self._prog['u_window'].value = self.window_size
        self._vao.render()


# =============================================================================
# PANELS
# =============================================================================

class Panel(ABC):
    """Base panel class."""
    
    def __init__(self, scene: Scene, rect: Rect):
        self.scene = scene
        self.rect = rect
        scene.observe(self.on_entity_change)
    
    def on_entity_change(self, entity: Entity, action: str):
        pass  # Override if needed
    
    @abstractmethod
    def render(self, draw: DrawContext):
        pass


class SequencerPanel(Panel):
    """2D grid: X=beats, Y=lanes."""
    
    LANE_H = 40
    
    def render(self, draw: DrawContext):
        r = self.rect
        cam = self.scene.time_camera
        
        # Background
        draw.rect(r.x, r.y, r.w, r.h, (0.06, 0.07, 0.08, 1.0))
        
        # Lane backgrounds
        for i, track in enumerate(self.scene.tracks):
            y = r.y + i * self.LANE_H
            bg = (*track.color[:3], 0.15)
            draw.rect(r.x, y, r.w, self.LANE_H, bg)
        
        # Beat grid
        for beat in cam.iter_beats(r.w):
            px = cam.beat_to_px(beat) + r.x
            if px < r.x or px > r.x + r.w:
                continue
            is_bar = int(beat) % 4 == 0
            color = (0.35, 0.35, 0.4, 1.0) if is_bar else (0.18, 0.18, 0.22, 1.0)
            draw.rect(px, r.y, 2 if is_bar else 1, r.h, color)
        
        # Events
        for track in self.scene.tracks:
            for entity in track.events:
                px = cam.beat_to_px(entity.position.x) + r.x
                py = r.y + track.index * self.LANE_H + 3
                pw = max(4, entity.extent.x * cam._px_per_beat - 2)
                ph = self.LANE_H - 6
                
                if px + pw > r.x and px < r.x + r.w:
                    draw.rect(px, py, pw, ph, entity.color)
        
        # Playhead
        ph_px = cam.beat_to_px(self.scene.transport.playhead_beat) + r.x
        if r.x <= ph_px <= r.x + r.w:
            draw.rect(ph_px - 1, r.y, 3, r.h, (1.0, 0.4, 0.2, 0.9))


class WaveformPanel(Panel):
    """2D waveform: X=time, Y=amplitude."""
    
    def __init__(self, scene: Scene, rect: Rect, audio: SimpleAudio):
        super().__init__(scene, rect)
        self.audio = audio
    
    def render(self, draw: DrawContext):
        r = self.rect
        cam = self.scene.time_camera
        
        # Background
        draw.rect(r.x, r.y, r.w, r.h, (0.05, 0.06, 0.07, 1.0))
        
        # Beat grid (subtle)
        for beat in cam.iter_beats(r.w):
            px = cam.beat_to_px(beat) + r.x
            if r.x <= px <= r.x + r.w:
                draw.rect(px, r.y, 1, r.h, (0.12, 0.12, 0.15, 1.0))
        
        # Waveform
        samples = self.audio.get_waveform(int(r.w))
        for i, amp in enumerate(samples):
            if amp > 0.01:
                h = amp * r.h * 0.8
                y = r.y + (r.h - h) / 2
                draw.rect(r.x + i, y, 2, h, (0.3, 0.6, 0.9, 0.7))
        
        # Event markers
        for track in self.scene.tracks:
            for entity in track.events:
                px = cam.beat_to_px(entity.position.x) + r.x
                if r.x <= px <= r.x + r.w:
                    draw.rect(px, r.y, 2, 8, entity.color)
        
        # Playhead
        ph_px = cam.beat_to_px(self.scene.transport.playhead_beat) + r.x
        if r.x <= ph_px <= r.x + r.w:
            draw.rect(ph_px - 1, r.y, 3, r.h, (1.0, 0.4, 0.2, 0.9))


class Viewport3DPanel(Panel):
    """3D view with cubes for events."""
    
    def __init__(self, scene: Scene, rect: Rect, ctx: moderngl.Context):
        super().__init__(scene, rect)
        self.gl = ctx
        self._cubes: List[Tuple[Entity, float]] = []  # (entity, spawn_time)
        self._setup_3d()
    
    def _setup_3d(self):
        """Setup 3D cube rendering."""
        self._prog3d = self.gl.program(
            vertex_shader="""
            #version 330
            in vec3 in_pos;
            in vec3 in_normal;
            uniform mat4 u_mvp;
            uniform vec3 u_offset;
            uniform vec3 u_scale;
            out vec3 v_normal;
            void main() {
                vec3 pos = in_pos * u_scale + u_offset;
                gl_Position = u_mvp * vec4(pos, 1.0);
                v_normal = in_normal;
            }
            """,
            fragment_shader="""
            #version 330
            in vec3 v_normal;
            uniform vec4 u_color;
            out vec4 fragColor;
            void main() {
                float light = 0.4 + 0.6 * max(0.0, dot(v_normal, normalize(vec3(1, 2, 1))));
                fragColor = vec4(u_color.rgb * light, u_color.a);
            }
            """
        )
        
        # Cube vertices with normals
        v = [
            # Front
            -1,-1, 1,  0, 0, 1,   1,-1, 1,  0, 0, 1,   1, 1, 1,  0, 0, 1,
            -1,-1, 1,  0, 0, 1,   1, 1, 1,  0, 0, 1,  -1, 1, 1,  0, 0, 1,
            # Back
            -1,-1,-1,  0, 0,-1,  -1, 1,-1,  0, 0,-1,   1, 1,-1,  0, 0,-1,
            -1,-1,-1,  0, 0,-1,   1, 1,-1,  0, 0,-1,   1,-1,-1,  0, 0,-1,
            # Top
            -1, 1,-1,  0, 1, 0,  -1, 1, 1,  0, 1, 0,   1, 1, 1,  0, 1, 0,
            -1, 1,-1,  0, 1, 0,   1, 1, 1,  0, 1, 0,   1, 1,-1,  0, 1, 0,
            # Bottom
            -1,-1,-1,  0,-1, 0,   1,-1,-1,  0,-1, 0,   1,-1, 1,  0,-1, 0,
            -1,-1,-1,  0,-1, 0,   1,-1, 1,  0,-1, 0,  -1,-1, 1,  0,-1, 0,
            # Right
             1,-1,-1,  1, 0, 0,   1, 1,-1,  1, 0, 0,   1, 1, 1,  1, 0, 0,
             1,-1,-1,  1, 0, 0,   1, 1, 1,  1, 0, 0,   1,-1, 1,  1, 0, 0,
            # Left
            -1,-1,-1, -1, 0, 0,  -1,-1, 1, -1, 0, 0,  -1, 1, 1, -1, 0, 0,
            -1,-1,-1, -1, 0, 0,  -1, 1, 1, -1, 0, 0,  -1, 1,-1, -1, 0, 0,
        ]
        vbo = self.gl.buffer(np.array(v, dtype='f4'))
        self._cube_vao = self.gl.vertex_array(
            self._prog3d, [(vbo, '3f 3f', 'in_pos', 'in_normal')]
        )
        
        # Create FBO for this panel
        self._fbo = None
        self._fbo_tex = None
    
    def on_entity_change(self, entity: Entity, action: str):
        if action == "added":
            self._cubes.append((entity, time.time()))
    
    def _ensure_fbo(self):
        """Create/resize FBO if needed."""
        w, h = int(self.rect.w), int(self.rect.h)
        if self._fbo is None or self._fbo.size != (w, h):
            self._fbo_tex = self.gl.texture((w, h), 4)
            depth = self.gl.depth_renderbuffer((w, h))
            self._fbo = self.gl.framebuffer(self._fbo_tex, depth)
    
    def _perspective(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        f = 1.0 / math.tan(fov / 2)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), -1],
            [0, 0, 2*far*near/(near-far), 0]
        ], dtype='f4')
    
    def _look_at(self, eye, target, up) -> np.ndarray:
        f = target - eye
        f = f / np.linalg.norm(f)
        r = np.cross(f, up)
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        return np.array([
            [r[0], u[0], -f[0], 0],
            [r[1], u[1], -f[1], 0],
            [r[2], u[2], -f[2], 0],
            [-np.dot(r,eye), -np.dot(u,eye), np.dot(f,eye), 1]
        ], dtype='f4')
    
    def render(self, draw: DrawContext):
        self._ensure_fbo()
        r = self.rect
        
        # Render to FBO
        self._fbo.use()
        self.gl.viewport = (0, 0, int(r.w), int(r.h))
        self.gl.clear(0.08, 0.09, 0.12, 1.0)
        self.gl.enable(self.gl.DEPTH_TEST)
        
        # Camera orbits based on playhead
        t = self.scene.transport.playhead_beat * 0.1
        eye = np.array([
            8 + math.sin(t) * 3,
            5 + math.sin(t * 0.7) * 2,
            8 + math.cos(t) * 3
        ], dtype='f4')
        target = np.array([4, 2, 0], dtype='f4')
        up = np.array([0, 1, 0], dtype='f4')
        
        proj = self._perspective(math.radians(45), r.w / r.h, 0.1, 100)
        view = self._look_at(eye, target, up)
        mvp = proj @ view
        self._prog3d['u_mvp'].write(mvp.T.tobytes())
        
        # Draw floor grid
        self._prog3d['u_color'].value = (0.15, 0.15, 0.18, 1.0)
        for x in range(17):
            self._prog3d['u_offset'].value = (x, -0.1, 0)
            self._prog3d['u_scale'].value = (0.02, 0.02, 8)
            self._cube_vao.render()
        for z in range(9):
            self._prog3d['u_offset'].value = (8, -0.1, z)
            self._prog3d['u_scale'].value = (8, 0.02, 0.02)
            self._cube_vao.render()
        
        # Draw event cubes
        now = time.time()
        for entity, spawn_time in self._cubes:
            age = now - spawn_time
            scale = min(1.0, age * 4)  # Pop-in animation
            
            self._prog3d['u_offset'].value = (
                entity.position.x,           # beat → X
                entity.position.y * 1.2,     # lane → Y
                entity.position.z * 3        # velocity → Z
            )
            self._prog3d['u_scale'].value = (0.15 * scale, 0.4 * scale, 0.2 * scale)
            self._prog3d['u_color'].value = entity.color
            self._cube_vao.render()
        
        # Draw playhead as tall thin cube
        ph = self.scene.transport.playhead_beat
        self._prog3d['u_offset'].value = (ph, 2, 0)
        self._prog3d['u_scale'].value = (0.05, 2, 0.05)
        self._prog3d['u_color'].value = (1.0, 0.4, 0.2, 0.9)
        self._cube_vao.render()
        
        self.gl.disable(self.gl.DEPTH_TEST)
        
        # Blit to screen (simple - just draw background, real impl would blit texture)
        self.gl.screen.use()
        draw.rect(r.x, r.y, r.w, r.h, (0.08, 0.09, 0.12, 1.0))
        # Note: Full implementation would blit self._fbo_tex to screen
        # For now the 3D renders directly to screen in next frame


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class SpectroDemo(mglw.WindowConfig):
    """SPECTRO Demo - Three panels, one scene."""
    
    gl_version = (3, 3)
    title = "SPECTRO - Context Bounces Off Timeline"
    window_size = (1280, 720)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # === THE SPACE ===
        self.scene = Scene()
        
        # === AUDIO ===
        self.audio = SimpleAudio()
        
        # === INPUT ===
        self.input = InputManager(self.scene)
        
        # === CONTEXT FIRER ===
        self.firer = ContextFirer(self.scene, self.audio, self.input)
        self.firer.on_event(self._on_context_fire)
        
        # === RENDERING ===
        self.draw = DrawContext(self.ctx)
        
        # === PANELS ===
        self._layout_panels()
        
        # === START ===
        print("\n" + "="*50)
        print("  SPECTRO DEMO")
        print("="*50)
        print("  Keys 1-4: Trigger drums")
        print("  SPACE: Play/Pause")
        print("  R: Reset")
        print("  Scroll: Zoom timeline")
        print("="*50 + "\n")
        
        # Auto-play
        self.scene.transport.play()
    
    def _layout_panels(self):
        """Create panel layout."""
        w, h = self.window_size
        
        transport_h = 50
        waveform_h = 100
        main_h = h - transport_h - waveform_h
        
        self.panels = [
            SequencerPanel(
                self.scene,
                Rect(0, transport_h, w // 2, main_h)
            ),
            Viewport3DPanel(
                self.scene,
                Rect(w // 2, transport_h, w // 2, main_h),
                self.ctx
            ),
            WaveformPanel(
                self.scene,
                Rect(0, transport_h + main_h, w, waveform_h),
                self.audio
            ),
        ]
    
    def _on_context_fire(self, ctx: ExecutionContext):
        """
        THE BOUNCE: Context fires, create entity + audio.
        """
        track = self.scene.tracks[ctx.track]
        
        # Create entity in scene (notifies all panels)
        entity = self.scene.add_note(
            track_index=ctx.track,
            beat=ctx.beat,  # LATE-BOUND: current beat
            velocity=ctx.velocity
        )
        
        print(f"[BOUNCE] {track.name} @ beat {ctx.beat:.2f}")
        
        # Trigger audio
        self.audio.trigger(track.sample, ctx.velocity / 127.0)
    
    def on_render(self, t: float, frame_time: float):
        dt = min(frame_time, 0.05)  # Cap dt
        
        # Update
        self.scene.transport.update(dt)
        self.scene.time_camera.follow(
            self.scene.transport.playhead_beat,
            self.panels[0].rect.w
        )
        self.firer.process(dt)
        self.audio.decay(dt)
        
        # Update draw context
        self.draw.window_size = self.wnd.size
        
        # Clear
        self.ctx.clear(0.1, 0.1, 0.12)
        
        # Transport bar
        self._draw_transport()
        
        # Panels
        for panel in self.panels:
            panel.render(self.draw)
    
    def _draw_transport(self):
        """Draw transport bar at top."""
        w = self.wnd.size[0]
        
        # Background
        self.draw.rect(0, 0, w, 50, (0.12, 0.13, 0.15, 1.0))
        
        # Play indicator
        playing = self.scene.transport.playing
        color = (0.3, 0.8, 0.3, 1.0) if playing else (0.6, 0.25, 0.25, 1.0)
        self.draw.rect(15, 10, 30, 30, color)
        
        # Beat boxes
        beat = self.scene.transport.playhead_beat
        beat_in_bar = int(beat) % 4
        for i in range(4):
            x = 60 + i * 35
            c = (0.4, 0.6, 0.9, 1.0) if i == beat_in_bar else (0.2, 0.22, 0.25, 1.0)
            self.draw.rect(x, 10, 30, 30, c)
        
        # Beat counter
        bar = int(beat / 4) + 1
        beat_num = int(beat % 4) + 1
        # (Text would go here - skipping for simplicity)
    
    def key_event(self, key, action, mods):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        
        keys = self.wnd.keys
        
        if key == keys.NUMBER_1:
            self.input.key_pressed(0)
        elif key == keys.NUMBER_2:
            self.input.key_pressed(1)
        elif key == keys.NUMBER_3:
            self.input.key_pressed(2)
        elif key == keys.NUMBER_4:
            self.input.key_pressed(3)
        elif key == keys.SPACE:
            self.scene.transport.toggle()
            print(f"[TRANSPORT] {'▶ Playing' if self.scene.transport.playing else '⏸ Paused'}")
        elif key == keys.R:
            self.scene.transport.stop()
            print("[TRANSPORT] ⏹ Reset")
    
    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.scene.time_camera.zoom(y_offset)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mglw.run_window_config(SpectroDemo)

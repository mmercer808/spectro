"""
SPECTRO Playable Demo v1
========================
pip install moderngl moderngl-window numpy
python spectro_v1.py

Keys: 1-4 = drums, SPACE = play/pause, R = reset, scroll = zoom
"""

import time, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import moderngl
import moderngl_window as mglw

# === DATA ===
@dataclass
class Note:
    beat: float
    lane: int
    velocity: float
    color: Tuple[float,float,float,float]
    fired: bool = False

COLORS = [(1,.3,.2,1), (1,.6,.2,1), (1,.9,.2,1), (.4,1,.3,1)]
NAMES = ["Kick", "Snare", "HiHat", "Clap"]

class Scene:
    def __init__(self):
        self.bpm = 120.0
        self.playing = True
        self.beat = 0.0
        self.notes: List[Note] = []
        self.scroll = 0.0
        self.px_per_beat = 50.0
        self.waveform = [0.0] * 400
        self._pattern = [(0,0), (1,2), (2,1), (3,2), (4,0), (5,2), (6,1), (6.5,0), (7,2)]
        self._pattern_fired = set()
    
    def update(self, dt):
        if not self.playing: return
        self.beat += dt * self.bpm / 60
        if self.beat >= 8: 
            self.beat = 0
            self._pattern_fired.clear()
            for n in self.notes: n.fired = False
        # Auto-scroll
        px = (self.beat - self.scroll) * self.px_per_beat
        if px > 400: self.scroll = self.beat - 400 / self.px_per_beat
        # Pattern
        for i, (b, lane) in enumerate(self._pattern):
            if i not in self._pattern_fired and b <= self.beat:
                self._pattern_fired.add(i)
                self.add_note(lane, b, 0.8)
        # Playback
        for n in self.notes:
            if not n.fired and n.beat <= self.beat:
                n.fired = True
                self._trigger(n.lane, n.velocity)
        # Decay waveform
        for i in range(len(self.waveform)): self.waveform[i] *= 0.92
    
    def add_note(self, lane, beat, vel=1.0):
        self.notes.append(Note(beat, lane, vel, COLORS[lane]))
        self._trigger(lane, vel)
    
    def _trigger(self, lane, vel):
        print(f"♪ {NAMES[lane]}")
        for i in range(15): self.waveform[(int(self.beat*20)+i) % len(self.waveform)] = vel
    
    def beat_to_px(self, b): return (b - self.scroll) * self.px_per_beat

class App(mglw.WindowConfig):
    gl_version = (3,3)
    title = "SPECTRO v1"
    window_size = (1000, 600)
    
    def __init__(self, **kw):
        super().__init__(**kw)
        self.scene = Scene()
        self._prog = self.ctx.program(
            vertex_shader="#version 330\nin vec2 p;uniform vec4 r;uniform vec2 w;void main(){vec2 x=r.xy+p*r.zw;gl_Position=vec4(x/w*2.0-1.0,0,1);gl_Position.y*=-1;}",
            fragment_shader="#version 330\nuniform vec4 c;out vec4 f;void main(){f=c;}"
        )
        self._vao = self.ctx.vertex_array(self._prog, [(self.ctx.buffer(np.array([0,0,1,0,1,1,0,0,1,1,0,1],dtype='f4')), '2f', 'p')])
        print("\n[SPECTRO] 1-4=drums SPACE=play/pause R=reset\n")
    
    def rect(self, x, y, w, h, col):
        self._prog['r'].value = (x,y,w,h)
        self._prog['w'].value = self.wnd.size
        self._prog['c'].value = col
        self._vao.render()
    
    def on_render(self, t, dt):
        self.scene.update(min(dt, 0.05))
        W, H = self.wnd.size
        self.ctx.clear(0.08, 0.09, 0.1)
        
        # Transport bar
        self.rect(0, 0, W, 40, (0.12,0.12,0.14,1))
        self.rect(10, 8, 24, 24, (0.3,0.8,0.3,1) if self.scene.playing else (0.7,0.3,0.3,1))
        for i in range(4):
            c = (0.4,0.6,0.9,1) if int(self.scene.beat)%4==i else (0.2,0.2,0.25,1)
            self.rect(50+i*30, 8, 24, 24, c)
        
        # Sequencer (left)
        sy, sh = 45, H-145
        self.rect(0, sy, W//2-5, sh, (0.06,0.07,0.08,1))
        for i in range(4): self.rect(0, sy+i*sh//4, W//2-5, sh//4-2, (*COLORS[i][:3],0.1))
        for b in range(int(self.scene.scroll), int(self.scene.scroll)+20):
            px = self.scene.beat_to_px(b)
            if 0 <= px < W//2: self.rect(px, sy, 2 if b%4==0 else 1, sh, (0.3,0.3,0.35,1) if b%4==0 else (0.15,0.15,0.18,1))
        for n in self.scene.notes:
            px = self.scene.beat_to_px(n.beat)
            if 0 <= px < W//2: self.rect(px, sy+n.lane*sh//4+3, max(4,self.scene.px_per_beat*0.2), sh//4-6, n.color)
        px = self.scene.beat_to_px(self.scene.beat)
        if 0 <= px < W//2: self.rect(px-1, sy, 3, sh, (1,0.4,0.2,0.9))
        
        # 3D area (right) - simple beat visualization
        self.rect(W//2, sy, W//2, sh, (0.07,0.08,0.1,1))
        for i in range(9): self.rect(W//2+20, sy+i*sh//8, W//2-40, 1, (0.15,0.15,0.18,1))
        for n in self.scene.notes:
            x = W//2+30 + (n.beat/8)*(W//2-60)
            y = sy+10 + n.lane*(sh//4-5)
            s = 8 + n.velocity*12
            self.rect(x-s/2, y, s, s, (*n.color[:3], 0.8))
        # Playhead in 3D
        px3d = W//2+30 + (self.scene.beat/8)*(W//2-60)
        self.rect(px3d-1, sy, 3, sh, (1,0.4,0.2,0.7))
        
        # Waveform (bottom)
        wy = H-95
        self.rect(0, wy, W, 90, (0.05,0.06,0.07,1))
        for i, v in enumerate(self.scene.waveform[:W]):
            if v > 0.02:
                h = v * 70
                self.rect(i, wy+45-h/2, 2, h, (0.3,0.6,0.9,0.7))
        px = self.scene.beat_to_px(self.scene.beat)
        if 0 <= px < W: self.rect(px-1, wy, 3, 90, (1,0.4,0.2,0.8))
    
    def key_event(self, key, action, mods):
        if action != self.wnd.keys.ACTION_PRESS: return
        k = self.wnd.keys
        if key == k.NUMBER_1: self.scene.add_note(0, self.scene.beat)
        elif key == k.NUMBER_2: self.scene.add_note(1, self.scene.beat)
        elif key == k.NUMBER_3: self.scene.add_note(2, self.scene.beat)
        elif key == k.NUMBER_4: self.scene.add_note(3, self.scene.beat)
        elif key == k.SPACE: self.scene.playing = not self.scene.playing
        elif key == k.R: self.scene.beat = 0; self.scene.notes.clear(); self.scene._pattern_fired.clear()
    
    def mouse_scroll_event(self, x, y):
        self.scene.px_per_beat = max(20, min(150, self.scene.px_per_beat * (1.1 if y > 0 else 0.9)))

if __name__ == "__main__": mglw.run_window_config(App)

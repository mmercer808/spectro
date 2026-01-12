"""
Engine Seed v3 - Stress Test

Tests the engine with many entities to verify:
- Snapshot system handles large graphs
- Batching/instancing works correctly
- Async extraction stays responsive
"""

from __future__ import annotations
import time
import random
from typing import List

import numpy as np
import moderngl_window as mglw

from engine.core.frame import FrameState
from engine.render.targets import RenderTargetPool
from engine.render.resources import ResourceRegistry
from engine.render.renderer import Renderer
from engine.scene.graph import EntityNode, Transform, MeshRenderer
from engine.viewport.viewport import ViewportArea, Camera, AreaLayout, PanelRect


class StressTestApp(mglw.WindowConfig):
    """Stress test with many cubes."""
    
    gl_version = (3, 3)
    title = "Engine Seed v3 — Stress Test (1000 cubes)"
    window_size = (1280, 720)
    resource_dir = "."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        
        # Core systems
        self.pool = RenderTargetPool(self.ctx)
        self.registry = ResourceRegistry(self.ctx)
        self.registry.bootstrap_defaults()
        self.renderer = Renderer(self.ctx, self.registry)
        
        self.layout = AreaLayout()
        self.areas: List[ViewportArea] = []
        
        self.last_t = time.perf_counter()
        self.frame_id = 0
        
        # Create stress test viewport
        self._setup_stress_test()
        
        self._last_mouse = (0.0, 0.0)
        
        # Stats
        self._frame_times: List[float] = []
    
    def _setup_stress_test(self):
        """Create scene with many cubes."""
        
        # Create a scene with N cubes arranged in a grid
        num_cubes = 1000
        graph = create_cube_field(num_cubes)
        
        cameras = [
            Camera(
                name="Persp",
                eye=np.array([30.0, 20.0, 30.0], dtype=np.float32),
                target=np.array([0, 0, 0], dtype=np.float32),
                far=500.0,
            ),
        ]
        
        area = ViewportArea("main", graph, cameras=cameras)
        self.areas = [area]
        self.layout.set_window_size(*self.window_size)
        
        print(f"[StressTest] Created scene with {num_cubes} cubes")
    
    def render(self, time_delta: float):
        frame_start = time.perf_counter()
        
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_t)
        self.last_t = now
        self.frame_id += 1
        frame = FrameState(frame_id=self.frame_id, dt=dt, t=now)
        
        w, h = self.wnd.size
        self.layout.set_window_size(w, h)
        
        panels = self.layout.compute_panels([
            PanelRect(id="main", x=10, y=10, w=w - 20, h=h - 20),
        ])
        
        # Update animations
        for area in self.areas:
            area.graph.apply_demo_spin(dt)
            area.mark_dirty("transform")
        
        # Async extraction
        for area in self.areas:
            rect = panels[area.area_id]
            area.ensure_surface(rect.w, rect.h, self.pool, want_picking=True)
            area.kick_extract_if_needed(frame, rect)
        
        # Render
        for area in self.areas:
            area.render_if_ready(self.renderer)
        
        # Composite
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)
        self.renderer.composite_panels(panels, self.areas, window_size=(w, h))
        
        # Stats
        frame_time = time.perf_counter() - frame_start
        self._frame_times.append(frame_time)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)
        
        if self.frame_id % 60 == 0:
            avg_ms = np.mean(self._frame_times) * 1000
            max_ms = np.max(self._frame_times) * 1000
            print(f"[Frame {self.frame_id}] avg: {avg_ms:.2f}ms, max: {max_ms:.2f}ms, FPS: {1000/avg_ms:.1f}")
    
    def mouse_position_event(self, x, y, dx, dy):
        self._last_mouse = (x, y)
        self._route_mouse("move", x, y, dx, dy)
    
    def mouse_press_event(self, x, y, button):
        self._last_mouse = (x, y)
        self._route_mouse("press", x, y, 0, 0, button=button)
    
    def mouse_release_event(self, x, y, button):
        self._route_mouse("release", x, y, 0, 0, button=button)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        x, y = self._last_mouse
        for area in self.areas:
            rect = self.layout.current_panels.get(area.area_id)
            if rect and rect.contains(x, y):
                area.handle_wheel(y_offset, rect)
                break
    
    def _route_mouse(self, kind, x, y, dx, dy, button=None):
        for area in self.areas:
            rect = self.layout.current_panels.get(area.area_id)
            if rect and rect.contains(x, y):
                area.handle_pointer(kind, x - rect.x, y - rect.y, dx, dy, button=button)
                
                if kind == "press" and button == 1:
                    result = area.pick_at(self.renderer, x - rect.x, y - rect.y)
                    if result:
                        print(f"[PICK] entity_id={result.entity_id}")
                break


def create_cube_field(n: int) -> EntityNode:
    """Create a grid of cubes."""
    root = EntityNode("root")
    
    # Arrange in a grid
    side = int(np.ceil(n ** (1/3)))  # Cube root for 3D arrangement
    spacing = 2.5
    
    colors = [
        (0.95, 0.35, 0.35, 1.0),  # Red
        (0.35, 0.95, 0.35, 1.0),  # Green
        (0.35, 0.35, 0.95, 1.0),  # Blue
        (0.95, 0.95, 0.35, 1.0),  # Yellow
        (0.95, 0.35, 0.95, 1.0),  # Magenta
        (0.35, 0.95, 0.95, 1.0),  # Cyan
    ]
    
    count = 0
    for x in range(side):
        for y in range(side):
            for z in range(side):
                if count >= n:
                    break
                
                cube = EntityNode(f"cube_{count}")
                cube.transform = Transform()
                
                # Position
                cx = (x - side / 2) * spacing
                cy = (y - side / 2) * spacing
                cz = (z - side / 2) * spacing
                cube.transform.pos = np.array([cx, cy, cz], dtype=np.float32)
                
                # Random scale variation
                scale = 0.3 + random.random() * 0.4
                cube.transform.scale = np.array([scale, scale, scale], dtype=np.float32)
                
                # Random initial rotation
                cube.transform.rot_euler = np.array([
                    random.random() * 3.14,
                    random.random() * 3.14,
                    random.random() * 3.14,
                ], dtype=np.float32)
                
                # Color based on position
                color = colors[(x + y + z) % len(colors)]
                cube.mesh = MeshRenderer(
                    mesh_id="cube",
                    pipeline_id="lit_color",
                    color=np.array(color, dtype=np.float32),
                )
                
                # Random spin axis
                cube._demo_spin_axis = np.array([
                    random.random() - 0.5,
                    random.random() - 0.5,
                    random.random() - 0.5,
                ], dtype=np.float32)
                cube._demo_spin_axis /= np.linalg.norm(cube._demo_spin_axis) + 1e-6
                
                root.add_child(cube)
                count += 1
    
    return root


if __name__ == "__main__":
    mglw.run_window_config(StressTestApp)

"""
Engine Seed v3 - moderngl-window Host

Standalone demo using moderngl-window.
Demonstrates:
- Thread-safe async extraction via snapshots
- Multi-camera viewports with picking
- Render-to-texture + compositing
"""

from __future__ import annotations
import time
from typing import List

import numpy as np
import moderngl_window as mglw

from engine.core.frame import FrameState
from engine.render.targets import RenderTargetPool
from engine.render.resources import ResourceRegistry
from engine.render.renderer import Renderer
from engine.scene.graph import EntityNode, Transform, MeshRenderer
from engine.viewport.viewport import ViewportArea, Camera, AreaLayout, PanelRect


class EngineApp(mglw.WindowConfig):
    """Main application window."""
    
    gl_version = (3, 3)
    title = "Engine Seed v3 — moderngl-window host"
    window_size = (1280, 720)
    resource_dir = "."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)    
        
        # Enable depth test and blending
        self.ctx.enable(self.ctx.DEPTH_TEST)
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        
        # Create core systems
        self.pool = RenderTargetPool(self.ctx)
        self.registry = ResourceRegistry(self.ctx)
        self.registry.bootstrap_defaults()
        self.renderer = Renderer(self.ctx, self.registry)
        
        # Layout manager
        self.layout = AreaLayout()
        self.areas: List[ViewportArea] = []

        # Timing
        self.last_t = time.perf_counter()
        self.frame_id = 0
        
        # Create demo viewports
        self._setup_demo_viewports()
        
        # Input tracking
        self._last_mouse = (0.0, 0.0)
    
    def _setup_demo_viewports(self):
        """Set up demo scene with two viewport areas."""
        
        # Left area: Single cube, Y-axis spin
        graph_a = demo_scene_graph(spin_axis=(0.0, 1.0, 0.0), color=(0.35, 0.55, 0.95, 1.0))
        cams_a = [
            Camera(
                name="CamA1",
                eye=np.array([2.5, 1.5, 2.5], dtype=np.float32),
                target=np.array([0, 0, 0], dtype=np.float32),
            ),
            Camera(
                name="CamA2",
                eye=np.array([-2.5, 1.0, 2.5], dtype=np.float32),
                target=np.array([0, 0, 0], dtype=np.float32),
            ),
        ]
        area_a = ViewportArea("left", graph_a, cameras=cams_a)
        
        # Right area: Single cube, X-axis spin, 4 cameras
        graph_b = demo_scene_graph(spin_axis=(1.0, 0.0, 0.0), color=(0.95, 0.55, 0.35, 1.0))
        cams_b = [
            Camera(name="Top", eye=np.array([0.0, 4.0, 0.01], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="Front", eye=np.array([0.0, 0.0, 4.0], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="Side", eye=np.array([4.0, 0.0, 0.0], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="Persp", eye=np.array([2.8, 1.6, 2.8], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
        ]
        area_b = ViewportArea("right", graph_b, cameras=cams_b)
        
        self.areas = [area_a, area_b]
        self.layout.set_window_size(*self.window_size)
    
    def on_render(self, t: float, frame_time: float):
        """Main render loop."""
        
        # Update timing
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_t)
        self.last_t = now
        self.frame_id += 1
        frame = FrameState(frame_id=self.frame_id, dt=dt, t=now)
        
        # Update layout
        w, h = self.wnd.size
        self.layout.set_window_size(w, h)
        
        # Compute panel rects
        panels = self.layout.compute_panels([
            PanelRect(id="left", x=10, y=10, w=(w // 2) - 15, h=h - 20),
            PanelRect(id="right", x=(w // 2) + 5, y=10, w=(w // 2) - 15, h=h - 20),
        ])
        
        # Update scenes (demo animation)
        for area in self.areas:
            area.graph.apply_demo_spin(dt)
            area.mark_dirty("transform")
        
        # Kick async extraction for each area
        for area in self.areas:
            rect = panels[area.area_id]
            area.ensure_surface(rect.w, rect.h, self.pool, want_picking=True)
            area.kick_extract_if_needed(frame, rect)
        
        # Render viewports (consumes completed async work)
        for area in self.areas:
            area.render_if_ready(self.renderer)
        
        # Composite to screen
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)
        self.renderer.composite_panels(panels, self.areas, window_size=(w, h))
    
    # -------------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------------
    
    def mouse_position_event(self, x, y, dx, dy):
        self._last_mouse = (x, y)
        self._route_mouse("move", x, y, dx, dy)
    
    def mouse_press_event(self, x, y, button):
        self._last_mouse = (x, y)
        self._route_mouse("press", x, y, 0, 0, button=button)
    
    def mouse_release_event(self, x, y, button):
        self._last_mouse = (x, y)
        self._route_mouse("release", x, y, 0, 0, button=button)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        x, y = self._last_mouse
        for area in self.areas:
            rect = self.layout.current_panels.get(area.area_id)
            if rect and rect.contains(x, y):
                area.handle_wheel(y_offset, rect)
                break
    
    def _route_mouse(self, kind, x, y, dx, dy, button=None):
        """Route mouse event to appropriate viewport area."""
        for area in self.areas:
            rect = self.layout.current_panels.get(area.area_id)
            if rect and rect.contains(x, y):
                area.handle_pointer(kind, x - rect.x, y - rect.y, dx, dy, button=button)
                
                # Handle picking on left click press
                if kind == "press" and button == 1:
                    result = area.pick_at(self.renderer, x - rect.x, y - rect.y)
                    if result:
                        print(f"[PICK] area={area.area_id} entity_id={result.entity_id} camera={result.camera_index}")
                break


def demo_scene_graph(spin_axis=(0.0, 1.0, 0.0), color=(0.35, 0.55, 0.95, 1.0)) -> EntityNode:
    """Create a simple demo scene with a spinning cube."""
    root = EntityNode("root")
    
    cube = EntityNode("cube")
    cube.transform = Transform()
    cube.transform.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    cube.mesh = MeshRenderer(
        mesh_id="cube",
        pipeline_id="lit_color",
        color=np.array(color, dtype=np.float32),
    )
    cube._demo_spin_axis = np.array(spin_axis, dtype=np.float32)
    
    root.add_child(cube)
    return root


if __name__ == "__main__":
    mglw.run_window_config(EngineApp)

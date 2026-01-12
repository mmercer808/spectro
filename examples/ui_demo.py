"""
UI System Demo

Demonstrates the UI framework integrated with the 3D engine.
Shows:
- Window manager with dockable panels
- 3D viewport widget
- Interactive buttons and labels
- Flexbox layout
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
from engine.viewport.viewport import ViewportArea, Camera

# UI imports
from engine.ui import (
    WindowManager, DrawContext, UIRenderer,
    Panel, Label, Button, Heading, Spacer, Divider,
    Row, Column, Viewport3D,
    DockSplit, DockLeaf, SplitDirection,
    EventType,
)


class UIDemoApp(mglw.WindowConfig):
    """Demo application with UI system."""

    gl_version = (3, 3)
    title = "UI System Demo"
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

        # Timing
        self.frame_id = 0

        # Create 3D scene
        self._setup_scene()

        # Create UI
        self._setup_ui()

    def _setup_scene(self):
        """Create the 3D scene."""
        # Scene graph with spinning cube
        self.graph = self._create_demo_graph()

        # Cameras
        self.cameras = [
            Camera(
                name="Main",
                eye=np.array([3.0, 2.0, 3.0], dtype=np.float32),
                target=np.array([0, 0, 0], dtype=np.float32),
            ),
        ]

        # Viewport area
        self.viewport_area = ViewportArea("main", self.graph, self.cameras)

    def _create_demo_graph(self) -> EntityNode:
        """Create demo scene graph."""
        root = EntityNode("root")

        cube = EntityNode("cube")
        cube.transform = Transform()
        cube.mesh = MeshRenderer(
            mesh_id="cube",
            pipeline_id="lit_color",
            color=np.array([0.35, 0.55, 0.95, 1.0], dtype=np.float32),
        )
        cube._demo_spin_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        root.add_child(cube)
        return root

    def _setup_ui(self):
        """Create the UI layout."""
        w, h = self.window_size

        # Window manager
        self.wm = WindowManager(w, h)

        # UI renderer
        self.ui_renderer = UIRenderer(self.ctx, self.registry)

        # State for UI
        self.click_count = 0
        self.status_label = Label(text="Ready")

        # Create 3D viewport panel
        viewport_panel = Panel(
            title="3D Viewport",
            closable=False,
            children=[
                Viewport3D(area=self.viewport_area),
            ],
        )

        # Create properties panel
        props_panel = Panel(
            title="Properties",
            width=280,
            children=[
                Heading(text="Object", level=3),
                Label(text="Name: Cube"),
                Label(text="Type: Mesh"),
                Divider(),

                Heading(text="Transform", level=3),
                Row(gap=8, children=[
                    Label(text="Position:"),
                    Label(text="0, 0, 0"),
                ]),
                Row(gap=8, children=[
                    Label(text="Rotation:"),
                    Label(text="0, 0, 0"),
                ]),
                Row(gap=8, children=[
                    Label(text="Scale:"),
                    Label(text="1, 1, 1"),
                ]),
                Divider(),

                Heading(text="Actions", level=3),
                Button(
                    text="Click Me!",
                    on_click=self._on_button_click,
                ),
                Spacer(min_size=8),
                Button(
                    text="Reset Camera",
                    on_click=self._on_reset_camera,
                ),
                Spacer(flex_grow=1),

                Divider(),
                self.status_label,
            ],
        )

        # Set up dock layout: viewport on left, properties on right
        self.wm.set_dock_root(DockSplit(
            direction=SplitDirection.HORIZONTAL,
            split_ratio=0.78,
            children=[
                DockLeaf(panel=viewport_panel),
                DockLeaf(panel=props_panel),
            ],
        ))

    def _on_button_click(self, event):
        """Handle button click."""
        self.click_count += 1
        self.status_label.text = f"Clicked {self.click_count} times"
        print(f"[UI] Button clicked! Count: {self.click_count}")

    def _on_reset_camera(self, event):
        """Reset camera to default position."""
        cam = self.cameras[0]
        cam.eye = np.array([3.0, 2.0, 3.0], dtype=np.float32)
        cam.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.viewport_area.mark_dirty("camera")
        self.status_label.text = "Camera reset"

    def on_render(self, time: float, frame_time: float):
        """Main render loop."""
        dt = max(1e-6, frame_time)
        self.frame_id += 1
        frame = FrameState(frame_id=self.frame_id, dt=dt, t=time)

        w, h = self.wnd.size

        # Update UI layout
        self.wm.set_size(w, h)
        self.wm.do_layout()

        # Get viewport rect from UI for sizing render target
        # Find the Viewport3D widget and get its rect
        viewport_rect = self._get_viewport_rect()
        if viewport_rect:
            vw, vh = int(viewport_rect.w), int(viewport_rect.h)
            if vw > 0 and vh > 0:
                # Update scene animation
                self.graph.apply_demo_spin(dt)
                self.viewport_area.mark_dirty("transform")

                # Ensure render target
                self.viewport_area.ensure_surface(vw, vh, self.pool, want_picking=True)

                # Create a panel rect for extraction
                from engine.viewport.viewport import PanelRect
                panel_rect = PanelRect(
                    id="main",
                    x=int(viewport_rect.x),
                    y=int(viewport_rect.y),
                    w=vw,
                    h=vh,
                )

                # Kick extraction and render
                self.viewport_area.kick_extract_if_needed(frame, panel_rect)
                self.viewport_area.render_if_ready(self.renderer)

        # Clear screen
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)

        # Draw UI
        self.ctx.disable(self.ctx.DEPTH_TEST)
        draw_ctx = DrawContext(w, h)
        self.wm.draw(draw_ctx)
        batch = draw_ctx.finalize()

        # Blit viewport texture manually (since UIRenderer doesn't handle viewport textures yet)
        if viewport_rect and self.viewport_area.surface_rt:
            self._blit_viewport(viewport_rect)

        # Render UI elements
        self.ui_renderer.render(batch, w, h)

        self.ctx.enable(self.ctx.DEPTH_TEST)

    def _get_viewport_rect(self):
        """Find the Viewport3D widget and get its rect."""
        def find_viewport(widget):
            if isinstance(widget, Viewport3D):
                return widget.rect
            for child in widget.children:
                result = find_viewport(child)
                if result:
                    return result
            return None

        return find_viewport(self.wm.root)

    def _blit_viewport(self, rect):
        """Blit the 3D viewport texture to screen."""
        rt = self.viewport_area.surface_rt
        if not rt or not rt.color:
            return

        # Use the renderer's compositor (simplified)
        # In a full implementation, UIRenderer would handle this
        from engine.viewport.viewport import PanelRect

        panel_rect = PanelRect(
            id="main",
            x=int(rect.x),
            y=int(rect.y),
            w=int(rect.w),
            h=int(rect.h),
        )

        w, h = self.wnd.size
        self.renderer.composite_panels(
            {"main": panel_rect},
            [self.viewport_area],
            window_size=(w, h),
        )

    # -------------------------------------------------------------------------
    # Input Handling
    # -------------------------------------------------------------------------

    def mouse_position_event(self, x, y, dx, dy):
        self.wm.handle_pointer_event(EventType.POINTER_MOVE, x, y, 0, dx, dy)

    def mouse_press_event(self, x, y, button):
        self.wm.handle_pointer_event(EventType.POINTER_DOWN, x, y, button)

    def mouse_release_event(self, x, y, button):
        self.wm.handle_pointer_event(EventType.POINTER_UP, x, y, button)

    def mouse_scroll_event(self, x_offset, y_offset):
        # Get current mouse position (approximate)
        self.wm.handle_scroll_event(0, 0, x_offset, y_offset)


if __name__ == "__main__":
    mglw.run_window_config(UIDemoApp)

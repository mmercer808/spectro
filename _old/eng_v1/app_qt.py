from __future__ import annotations
import time
from typing import List

import numpy as np
from PySide6 import QtWidgets, QtCore, QtOpenGLWidgets
import moderngl

from engine.core.frame import FrameState
from engine.render.targets import RenderTargetPool
from engine.render.resources import ResourceRegistry
from engine.render.renderer import Renderer
from engine.scene.graph import EntityNode, Transform, MeshRenderer
from engine.viewport.viewport import ViewportArea, Camera, AreaLayout, PanelRect


class GLHost(QtOpenGLWidgets.QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.ctx: moderngl.Context | None = None
        self.pool: RenderTargetPool | None = None
        self.registry: ResourceRegistry | None = None
        self.renderer: Renderer | None = None

        self.last_t = time.perf_counter()
        self.frame_id = 0

        self.layout = AreaLayout()
        self.areas: List[ViewportArea] = []

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)
        self.setMouseTracking(True)

    def initializeGL(self):
        self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.pool = RenderTargetPool(self.ctx)
        self.registry = ResourceRegistry(self.ctx)
        self.registry.bootstrap_defaults()
        self.renderer = Renderer(self.ctx, self.registry)

        graph_a = demo_scene_graph(spin_axis=(0.0, 1.0, 0.0))
        cams_a = [
            Camera(name="CamA1", eye=np.array([2.5, 1.5, 2.5], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="CamA2", eye=np.array([-2.5, 1.0, 2.5], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
        ]
        area_a = ViewportArea("left", graph_a, cameras=cams_a)

        graph_b = demo_scene_graph(spin_axis=(1.0, 0.0, 0.0))
        cams_b = [
            Camera(name="Top",  eye=np.array([0.0, 4.0, 0.01], dtype=np.float32), target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="Front",eye=np.array([0.0, 0.0, 4.0], dtype=np.float32),  target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="Side", eye=np.array([4.0, 0.0, 0.0], dtype=np.float32),  target=np.array([0, 0, 0], dtype=np.float32)),
            Camera(name="Persp",eye=np.array([2.8, 1.6, 2.8], dtype=np.float32),  target=np.array([0, 0, 0], dtype=np.float32)),
        ]
        area_b = ViewportArea("right", graph_b, cameras=cams_b)

        self.areas = [area_a, area_b]

    def resizeGL(self, w: int, h: int):
        self.layout.set_window_size(w, h)

    def paintGL(self):
        if not (self.ctx and self.pool and self.registry and self.renderer):
            return
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_t)
        self.last_t = now
        self.frame_id += 1
        frame = FrameState(frame_id=self.frame_id, dt=dt, t=now)

        w, h = self.width(), self.height()
        panels = self.layout.compute_panels([
            PanelRect(id="left",  x=10, y=10, w=(w//2)-15, h=h-20),
            PanelRect(id="right", x=(w//2)+5, y=10, w=(w//2)-15, h=h-20),
        ])

        for area in self.areas:
            area.graph.apply_demo_spin(dt)
            area.mark_dirty("graph")

        for area in self.areas:
            rect = panels[area.area_id]
            area.ensure_surface(rect.w, rect.h, self.pool, want_picking=True)
            area.kick_extract_if_needed(frame, rect)

        for area in self.areas:
            area.render_if_ready(self.renderer)

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)
        self.renderer.composite_panels(panels, self.areas, window_size=(w, h))

    def mousePressEvent(self, e): self._route_mouse(e, "press")
    def mouseReleaseEvent(self, e): self._route_mouse(e, "release")
    def mouseMoveEvent(self, e): self._route_mouse(e, "move")

    def wheelEvent(self, e):
        x, y = e.position().x(), e.position().y()
        for area in self.areas:
            rect = self.layout.current_panels.get(area.area_id)
            if rect and rect.contains(x, y):
                area.handle_wheel(e.angleDelta().y() / 120.0, rect)
                break

    def _route_mouse(self, e, kind: str):
        if not self.renderer:
            return
        x, y = e.position().x(), e.position().y()
        for area in self.areas:
            rect = self.layout.current_panels.get(area.area_id)
            if rect and rect.contains(x, y):
                area.handle_pointer(kind, x-rect.x, y-rect.y, 0, 0, button=int(e.button()))
                if kind == "press" and int(e.button()) == 1:
                    ent = area.pick_at(self.renderer, x-rect.x, y-rect.y)
                    if ent:
                        print(f"[PICK] area={area.area_id} entity_id={ent}")
                break


def demo_scene_graph(spin_axis=(0.0, 1.0, 0.0)) -> EntityNode:
    root = EntityNode("root")
    cube = EntityNode("cube")
    cube.transform = Transform()
    cube.transform.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    cube.mesh = MeshRenderer(mesh_id="cube", pipeline_id="lit_color", color=np.array([0.35, 0.55, 0.95, 1.0], dtype=np.float32))
    cube._demo_spin_axis = np.array(spin_axis, dtype=np.float32)
    root.add_child(cube)
    return root


def main():
    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    gl = GLHost()
    win.setCentralWidget(gl)
    win.resize(1200, 720)
    win.setWindowTitle("Engine Seed v2 — Qt host")
    win.show()
    app.exec()


if __name__ == "__main__":
    main()

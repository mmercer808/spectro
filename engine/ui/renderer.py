"""
Standalone UI Renderer

Renders DrawBatch directly to screen without requiring ResourceRegistry.
Designed for use with the widget system in demos and standalone apps.
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import moderngl
    from engine.ui.draw import DrawBatch, DrawQuad, DrawLine, DrawText


class SimpleUIRenderer:
    """
    Standalone UI renderer that renders DrawBatch to GPU.

    Unlike UIRenderer, this doesn't require ResourceRegistry - it creates
    and manages its own shaders and buffers.

    Usage:
        renderer = SimpleUIRenderer(ctx)

        # Each frame:
        draw_ctx.clear()
        root_widget.draw(draw_ctx)
        batch = draw_ctx.finalize()
        renderer.render(batch, screen_width, screen_height)
    """

    def __init__(self, ctx: 'moderngl.Context'):
        self.ctx = ctx

        # Shaders
        self._quad_prog = None
        self._line_prog = None

        # Buffers
        self._quad_vbo = None
        self._quad_vao = None
        self._quad_capacity = 0

        self._line_vbo = None
        self._line_vao = None
        self._line_capacity = 0

        self._initialized = False

    def _ensure_initialized(self):
        """Create GPU resources on first use."""
        if self._initialized:
            return

        # Quad shader (for rectangles)
        self._quad_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            in vec4 in_color;

            out vec2 v_uv;
            out vec4 v_color;

            uniform vec2 u_screen_size;

            void main() {
                vec2 ndc = (in_pos / u_screen_size) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_uv = in_uv;
                v_color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec2 v_uv;
            in vec4 v_color;
            out vec4 frag_color;

            uniform sampler2D u_texture;
            uniform bool u_use_texture;

            void main() {
                if (u_use_texture) {
                    frag_color = texture(u_texture, v_uv) * v_color;
                } else {
                    frag_color = v_color;
                }
            }
            """
        )

        # Line shader
        self._line_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec4 in_color;
            out vec4 v_color;
            uniform vec2 u_screen_size;

            void main() {
                vec2 ndc = (in_pos / u_screen_size) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
                v_color = in_color;
            }
            """,
            fragment_shader="""
            #version 330
            in vec4 v_color;
            out vec4 frag_color;
            void main() { frag_color = v_color; }
            """
        )

        self._initialized = True

    def _ensure_quad_buffer(self, count: int):
        """Ensure quad VBO can hold count quads."""
        needed = count * 6
        if self._quad_capacity >= needed and self._quad_vbo is not None:
            return

        new_capacity = max(needed, self._quad_capacity * 2, 256)
        byte_size = new_capacity * 32  # 8 floats * 4 bytes

        if self._quad_vbo:
            self._quad_vbo.release()

        self._quad_vbo = self.ctx.buffer(reserve=byte_size, dynamic=True)
        self._quad_capacity = new_capacity

        self._quad_vao = self.ctx.vertex_array(
            self._quad_prog,
            [(self._quad_vbo, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
        )

    def _ensure_line_buffer(self, count: int):
        """Ensure line VBO can hold count lines."""
        needed = count * 2
        if self._line_capacity >= needed and self._line_vbo is not None:
            return

        new_capacity = max(needed, self._line_capacity * 2, 256)
        byte_size = new_capacity * 24  # 6 floats * 4 bytes

        if self._line_vbo:
            self._line_vbo.release()

        self._line_vbo = self.ctx.buffer(reserve=byte_size, dynamic=True)
        self._line_capacity = new_capacity

        self._line_vao = self.ctx.vertex_array(
            self._line_prog,
            [(self._line_vbo, "2f 4f", "in_pos", "in_color")],
        )

    def render(self, batch: 'DrawBatch', screen_width: int, screen_height: int):
        """
        Render a DrawBatch to the screen.

        Args:
            batch: DrawBatch containing quads, lines, and text commands.
            screen_width: Window width in pixels.
            screen_height: Window height in pixels.
        """
        self._ensure_initialized()

        if batch.quads:
            self._render_quads(batch.quads, screen_width, screen_height)

        if batch.lines:
            self._render_lines(batch.lines, screen_width, screen_height)

        # Text rendering would require font atlas integration
        # For now, text commands are collected but not rendered

    def _render_quads(self, quads: List['DrawQuad'], sw: int, sh: int):
        """Render quad batch."""
        self._ensure_quad_buffer(len(quads))

        # Build vertex data: pos(2f) + uv(2f) + color(4f)
        vertices = np.zeros((len(quads) * 6, 8), dtype=np.float32)

        for i, q in enumerate(quads):
            x0, y0 = q.x, q.y
            x1, y1 = q.x + q.w, q.y + q.h
            u0, v0, u1, v1 = q.uv
            c = q.color

            base = i * 6
            # Triangle 1
            vertices[base + 0] = [x0, y0, u0, v0, *c]
            vertices[base + 1] = [x1, y0, u1, v0, *c]
            vertices[base + 2] = [x1, y1, u1, v1, *c]
            # Triangle 2
            vertices[base + 3] = [x0, y0, u0, v0, *c]
            vertices[base + 4] = [x1, y1, u1, v1, *c]
            vertices[base + 5] = [x0, y1, u0, v1, *c]

        self._quad_vbo.write(vertices.tobytes())

        self._quad_prog["u_screen_size"].value = (sw, sh)
        self._quad_prog["u_use_texture"].value = False

        self._quad_vao.render(mode=self.ctx.TRIANGLES, vertices=len(quads) * 6)

    def _render_lines(self, lines: List['DrawLine'], sw: int, sh: int):
        """Render line batch."""
        self._ensure_line_buffer(len(lines))

        vertices = np.zeros((len(lines) * 2, 6), dtype=np.float32)

        for i, ln in enumerate(lines):
            base = i * 2
            vertices[base + 0] = [ln.x0, ln.y0, *ln.color]
            vertices[base + 1] = [ln.x1, ln.y1, *ln.color]

        self._line_vbo.write(vertices.tobytes())

        self._line_prog["u_screen_size"].value = (sw, sh)

        self._line_vao.render(mode=self.ctx.LINES, vertices=len(lines) * 2)

    def release(self):
        """Release GPU resources."""
        if self._quad_vbo:
            self._quad_vbo.release()
        if self._quad_vao:
            self._quad_vao.release()
        if self._line_vbo:
            self._line_vbo.release()
        if self._line_vao:
            self._line_vao.release()
        if self._quad_prog:
            self._quad_prog.release()
        if self._line_prog:
            self._line_prog.release()

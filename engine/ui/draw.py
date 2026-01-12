"""
Draw Context

Batched rendering of UI primitives via CommandList.

Design:
- Collects draw calls during widget.draw() phase
- Batches quads by texture/shader for efficiency
- Emits CommandList at end for renderer execution

Primitives:
- Filled rectangles (with optional corner radius)
- Rectangle outlines
- Text (requires font atlas)
- Textured quads
- Lines
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from engine.ui.layout import Rect
from engine.ui.style import Color, color_rgba, color_to_array


# =============================================================================
# Draw Commands (internal representation)
# =============================================================================

@dataclass
class DrawQuad:
    """A single quad to draw."""
    x: float
    y: float
    w: float
    h: float
    color: Tuple[float, float, float, float]
    radius: float = 0.0  # Corner radius
    texture_id: Optional[str] = None
    uv: Tuple[float, float, float, float] = (0, 0, 1, 1)  # u0, v0, u1, v1
    z_index: int = 0


@dataclass
class DrawLine:
    """A line segment."""
    x0: float
    y0: float
    x1: float
    y1: float
    color: Tuple[float, float, float, float]
    width: float = 1.0
    z_index: int = 0


@dataclass
class DrawText:
    """Text to draw."""
    text: str
    x: float
    y: float
    color: Tuple[float, float, float, float]
    font_size: float = 14.0
    font_id: str = "default"
    align: str = "left"  # left, center, right
    z_index: int = 0


# =============================================================================
# Draw Batch
# =============================================================================

@dataclass
class DrawBatch:
    """
    Collection of draw commands, sorted for efficient rendering.

    After building, call finalize() to sort and prepare for rendering.
    """
    quads: List[DrawQuad] = field(default_factory=list)
    lines: List[DrawLine] = field(default_factory=list)
    texts: List[DrawText] = field(default_factory=list)

    _finalized: bool = False

    def add_quad(self, quad: DrawQuad):
        self.quads.append(quad)
        self._finalized = False

    def add_line(self, line: DrawLine):
        self.lines.append(line)
        self._finalized = False

    def add_text(self, text: DrawText):
        self.texts.append(text)
        self._finalized = False

    def finalize(self):
        """Sort commands by z-index and batch by texture."""
        self.quads.sort(key=lambda q: (q.z_index, q.texture_id or ""))
        self.lines.sort(key=lambda l: l.z_index)
        self.texts.sort(key=lambda t: (t.z_index, t.font_id))
        self._finalized = True

    def clear(self):
        """Clear all commands."""
        self.quads.clear()
        self.lines.clear()
        self.texts.clear()
        self._finalized = False

    @property
    def quad_count(self) -> int:
        return len(self.quads)

    @property
    def total_vertices(self) -> int:
        """Total vertices needed (quads = 4 verts each, or 6 for triangles)."""
        return len(self.quads) * 6 + len(self.lines) * 2


# =============================================================================
# Draw Context
# =============================================================================

class DrawContext:
    """
    Context for drawing UI elements.

    Passed to Widget.draw() to collect draw commands.
    Maintains a transform stack for nested widgets.
    """

    def __init__(self, window_width: int, window_height: int):
        self.window_width = window_width
        self.window_height = window_height

        # Current batch
        self.batch = DrawBatch()

        # Transform stack (just translation for now)
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._offset_stack: List[Tuple[float, float]] = []

        # Z-index counter (auto-increment for draw order)
        self._z_index = 0

        # Clip stack
        self._clip_stack: List[Rect] = []
        self._current_clip: Optional[Rect] = None

    # -------------------------------------------------------------------------
    # Transform Stack
    # -------------------------------------------------------------------------

    def push_offset(self, x: float, y: float):
        """Push translation offset."""
        self._offset_stack.append((self._offset_x, self._offset_y))
        self._offset_x += x
        self._offset_y += y

    def pop_offset(self):
        """Pop translation offset."""
        if self._offset_stack:
            self._offset_x, self._offset_y = self._offset_stack.pop()

    def _transform(self, x: float, y: float) -> Tuple[float, float]:
        """Apply current transform to point."""
        return (x + self._offset_x, y + self._offset_y)

    # -------------------------------------------------------------------------
    # Clipping
    # -------------------------------------------------------------------------

    def push_clip(self, rect: Rect):
        """Push clip rectangle."""
        # Transform rect
        tx, ty = self._transform(rect.x, rect.y)
        clip = Rect(tx, ty, rect.w, rect.h)

        # Intersect with current clip
        if self._current_clip:
            clip = self._intersect_rects(self._current_clip, clip)

        self._clip_stack.append(self._current_clip)
        self._current_clip = clip

    def pop_clip(self):
        """Pop clip rectangle."""
        if self._clip_stack:
            self._current_clip = self._clip_stack.pop()

    def _intersect_rects(self, a: Rect, b: Rect) -> Rect:
        """Intersect two rectangles."""
        x = max(a.x, b.x)
        y = max(a.y, b.y)
        r = min(a.right, b.right)
        bot = min(a.bottom, b.bottom)
        return Rect(x, y, max(0, r - x), max(0, bot - y))

    def _is_clipped(self, rect: Rect) -> bool:
        """Check if rect is completely outside clip region."""
        if self._current_clip is None:
            return False
        c = self._current_clip
        return (
            rect.right <= c.x or
            rect.x >= c.right or
            rect.bottom <= c.y or
            rect.y >= c.bottom
        )

    # -------------------------------------------------------------------------
    # Drawing Primitives
    # -------------------------------------------------------------------------

    def draw_rect(
        self,
        rect: Rect,
        color: Color,
        radius: float = 0.0,
        texture_id: str = None,
        uv: Tuple[float, float, float, float] = None,
    ):
        """Draw a filled rectangle."""
        tx, ty = self._transform(rect.x, rect.y)
        transformed = Rect(tx, ty, rect.w, rect.h)

        if self._is_clipped(transformed):
            return

        self.batch.add_quad(DrawQuad(
            x=tx,
            y=ty,
            w=rect.w,
            h=rect.h,
            color=color_rgba(color),
            radius=radius,
            texture_id=texture_id,
            uv=uv or (0, 0, 1, 1),
            z_index=self._z_index,
        ))
        self._z_index += 1

    def draw_rect_outline(
        self,
        rect: Rect,
        color: Color,
        width: float = 1.0,
        radius: float = 0.0,
    ):
        """Draw a rectangle outline."""
        # For simplicity, draw as 4 lines (ignoring radius for now)
        # A proper implementation would use a rounded rect shader
        tx, ty = self._transform(rect.x, rect.y)
        c = color_rgba(color)
        w, h = rect.w, rect.h

        # Top
        self.batch.add_line(DrawLine(tx, ty, tx + w, ty, c, width, self._z_index))
        # Right
        self.batch.add_line(DrawLine(tx + w, ty, tx + w, ty + h, c, width, self._z_index))
        # Bottom
        self.batch.add_line(DrawLine(tx + w, ty + h, tx, ty + h, c, width, self._z_index))
        # Left
        self.batch.add_line(DrawLine(tx, ty + h, tx, ty, c, width, self._z_index))

        self._z_index += 1

    def draw_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: Color,
        width: float = 1.0,
    ):
        """Draw a line segment."""
        tx0, ty0 = self._transform(x0, y0)
        tx1, ty1 = self._transform(x1, y1)

        self.batch.add_line(DrawLine(
            x0=tx0, y0=ty0,
            x1=tx1, y1=ty1,
            color=color_rgba(color),
            width=width,
            z_index=self._z_index,
        ))
        self._z_index += 1

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        color: Color,
        font_size: float = 14.0,
        font_id: str = "default",
        align: str = "left",
    ):
        """Draw text."""
        tx, ty = self._transform(x, y)

        self.batch.add_text(DrawText(
            text=text,
            x=tx,
            y=ty,
            color=color_rgba(color),
            font_size=font_size,
            font_id=font_id,
            align=align,
            z_index=self._z_index,
        ))
        self._z_index += 1

    def draw_text_in_rect(
        self,
        text: str,
        rect: Rect,
        color: Color,
        font_size: float = 14.0,
        font_id: str = "default",
        align: str = "left",
        valign: str = "center",
    ):
        """Draw text within a rectangle with alignment."""
        # Calculate position based on alignment
        x = rect.x
        y = rect.y

        if align == "center":
            x = rect.x + rect.w / 2
        elif align == "right":
            x = rect.x + rect.w

        if valign == "center":
            y = rect.y + rect.h / 2 - font_size / 2
        elif valign == "bottom":
            y = rect.y + rect.h - font_size

        self.draw_text(text, x, y, color, font_size, font_id, align)

    # -------------------------------------------------------------------------
    # Finalization
    # -------------------------------------------------------------------------

    def finalize(self) -> DrawBatch:
        """Finalize and return the draw batch."""
        self.batch.finalize()
        return self.batch

    def clear(self):
        """Clear the context for next frame."""
        self.batch.clear()
        self._z_index = 0
        self._offset_x = 0.0
        self._offset_y = 0.0
        self._offset_stack.clear()
        self._clip_stack.clear()
        self._current_clip = None


# =============================================================================
# Batch Renderer
# =============================================================================

class UIRenderer:
    """
    Renders DrawBatch to GPU using the engine's command list system.

    Responsibilities:
    - Convert DrawBatch to vertex data
    - Create/update GPU buffers
    - Emit render commands

    This integrates with the existing engine rendering pipeline.
    """

    def __init__(self, ctx, registry):
        """
        Initialize UI renderer.

        Args:
            ctx: moderngl context
            registry: ResourceRegistry for shaders/meshes
        """
        self.ctx = ctx
        self.registry = registry

        # GPU resources (created on first use)
        self._quad_vbo = None
        self._quad_vao = None
        self._quad_capacity = 0

        self._line_vbo = None
        self._line_vao = None
        self._line_capacity = 0

        self._initialized = False

    def _ensure_initialized(self):
        """Create GPU resources if needed."""
        if self._initialized:
            return

        # Register UI shader if not present
        if "ui_quad" not in self.registry.pipelines:
            self._create_ui_pipeline()

        self._initialized = True

    def _create_ui_pipeline(self):
        """Create the UI rendering pipeline."""
        # Simple 2D quad shader
        vs = """
        #version 330
        in vec2 in_pos;
        in vec2 in_uv;
        in vec4 in_color;

        out vec2 v_uv;
        out vec4 v_color;

        uniform vec2 u_screen_size;

        void main() {
            // Convert pixel coords to NDC
            vec2 ndc = (in_pos / u_screen_size) * 2.0 - 1.0;
            ndc.y = -ndc.y;  // Flip Y for top-left origin
            gl_Position = vec4(ndc, 0.0, 1.0);
            v_uv = in_uv;
            v_color = in_color;
        }
        """

        fs = """
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

        prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.registry.pipelines["ui_quad"] = prog

        # Line shader
        vs_line = """
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
        """

        fs_line = """
        #version 330
        in vec4 v_color;
        out vec4 frag_color;

        void main() {
            frag_color = v_color;
        }
        """

        prog_line = self.ctx.program(vertex_shader=vs_line, fragment_shader=fs_line)
        self.registry.pipelines["ui_line"] = prog_line

    def _ensure_quad_buffer(self, count: int):
        """Ensure quad VBO can hold count quads."""
        needed = count * 6  # 6 vertices per quad (2 triangles)
        if self._quad_capacity >= needed and self._quad_vbo is not None:
            return

        # Vertex format: pos(2f) + uv(2f) + color(4f) = 32 bytes
        new_capacity = max(needed, self._quad_capacity * 2, 256)
        byte_size = new_capacity * 32

        if self._quad_vbo:
            self._quad_vbo.release()

        self._quad_vbo = self.ctx.buffer(reserve=byte_size, dynamic=True)
        self._quad_capacity = new_capacity

        # Recreate VAO
        prog = self.registry.pipelines["ui_quad"]
        self._quad_vao = self.ctx.vertex_array(
            prog,
            [(self._quad_vbo, "2f 2f 4f", "in_pos", "in_uv", "in_color")],
        )

    def _ensure_line_buffer(self, count: int):
        """Ensure line VBO can hold count lines."""
        needed = count * 2  # 2 vertices per line
        if self._line_capacity >= needed and self._line_vbo is not None:
            return

        new_capacity = max(needed, self._line_capacity * 2, 256)
        # Vertex format: pos(2f) + color(4f) = 24 bytes
        byte_size = new_capacity * 24

        if self._line_vbo:
            self._line_vbo.release()

        self._line_vbo = self.ctx.buffer(reserve=byte_size, dynamic=True)
        self._line_capacity = new_capacity

        prog = self.registry.pipelines["ui_line"]
        self._line_vao = self.ctx.vertex_array(
            prog,
            [(self._line_vbo, "2f 4f", "in_pos", "in_color")],
        )

    def render(self, batch: DrawBatch, screen_width: int, screen_height: int):
        """
        Render a DrawBatch to the screen.

        Should be called after 3D rendering, with depth test disabled.
        """
        self._ensure_initialized()

        # Render quads
        if batch.quads:
            self._render_quads(batch.quads, screen_width, screen_height)

        # Render lines
        if batch.lines:
            self._render_lines(batch.lines, screen_width, screen_height)

        # TODO: Render text (requires font atlas)

    def _render_quads(self, quads: List[DrawQuad], sw: int, sh: int):
        """Render quad batch."""
        self._ensure_quad_buffer(len(quads))

        # Build vertex data
        # Format: pos(2f) + uv(2f) + color(4f)
        vertices = np.zeros((len(quads) * 6, 8), dtype=np.float32)

        for i, q in enumerate(quads):
            x0, y0 = q.x, q.y
            x1, y1 = q.x + q.w, q.y + q.h
            u0, v0, u1, v1 = q.uv
            c = q.color

            # Two triangles per quad
            base = i * 6
            # Triangle 1: top-left, top-right, bottom-right
            vertices[base + 0] = [x0, y0, u0, v0, *c]
            vertices[base + 1] = [x1, y0, u1, v0, *c]
            vertices[base + 2] = [x1, y1, u1, v1, *c]
            # Triangle 2: top-left, bottom-right, bottom-left
            vertices[base + 3] = [x0, y0, u0, v0, *c]
            vertices[base + 4] = [x1, y1, u1, v1, *c]
            vertices[base + 5] = [x0, y1, u0, v1, *c]

        self._quad_vbo.write(vertices.tobytes())

        # Draw
        prog = self.registry.pipelines["ui_quad"]
        prog["u_screen_size"].value = (sw, sh)
        prog["u_use_texture"].value = False

        self._quad_vao.render(mode=self.ctx.TRIANGLES, vertices=len(quads) * 6)

    def _render_lines(self, lines: List[DrawLine], sw: int, sh: int):
        """Render line batch."""
        self._ensure_line_buffer(len(lines))

        # Format: pos(2f) + color(4f)
        vertices = np.zeros((len(lines) * 2, 6), dtype=np.float32)

        for i, ln in enumerate(lines):
            base = i * 2
            vertices[base + 0] = [ln.x0, ln.y0, *ln.color]
            vertices[base + 1] = [ln.x1, ln.y1, *ln.color]

        self._line_vbo.write(vertices.tobytes())

        prog = self.registry.pipelines["ui_line"]
        prog["u_screen_size"].value = (sw, sh)

        self._line_vao.render(mode=self.ctx.LINES, vertices=len(lines) * 2)

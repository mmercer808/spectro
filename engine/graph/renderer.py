# engine/graph/renderer.py
"""
Procedural UI Renderer - GPU-accelerated 2D rendering.

Renders UI components as instanced quads with procedural fragment shaders.
Each component type (dial, circle, arc, rect, wave) has its own shader
that draws the component analytically using SDF and smoothstep AA.

Pipeline:
1. Collect visible nodes into batches by component kind
2. Pack instance data into SSBO (rect, colors, params)
3. Draw instanced quads - vertex shader expands to screen rect
4. Fragment shader draws procedurally

Instance Layout (5 x vec4 = 20 floats per instance):
    [0] rect:   x, y, w, h (pixels)
    [1] color1: r, g, b, a (accent/fill)
    [2] color2: r, g, b, a (stroke)
    [3] params0: varies by kind
    [4] params1: varies by kind
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto

if TYPE_CHECKING:
    import moderngl

from engine.graph.style import StyleTokens, DARK_THEME
from engine.graph.text import TextRenderer, FontAtlas, TextAlign, TextBaseline


# =============================================================================
# Component Kinds
# =============================================================================

class ComponentKind(Enum):
    RECT = auto()
    CIRCLE = auto()
    DIAL = auto()      # Ring with needle and arcs
    ARC = auto()       # Partial arc
    WAVE = auto()      # Waveform line
    LINE = auto()      # Simple line


# =============================================================================
# Instance Data
# =============================================================================

STRIDE_FLOATS = 20  # 5 vec4 per instance


@dataclass
class UIInstance:
    """Instance data for one UI component."""
    kind: ComponentKind
    rect: Tuple[float, float, float, float]  # x, y, w, h
    color1: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    color2: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.1)
    params0: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    params1: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def pack_instances(instances: List[UIInstance]) -> np.ndarray:
    """Pack instances into a numpy array for GPU upload."""
    if not instances:
        return np.array([], dtype=np.float32)

    out = np.zeros((len(instances), STRIDE_FLOATS), dtype=np.float32)
    for i, inst in enumerate(instances):
        out[i, 0:4] = inst.rect
        out[i, 4:8] = inst.color1
        out[i, 8:12] = inst.color2
        out[i, 12:16] = inst.params0
        out[i, 16:20] = inst.params1
    return out.ravel()


# =============================================================================
# Shaders
# =============================================================================

VERTEX_SHADER = """
#version 430

in vec2 in_vert;  // Unit quad vertex position [0,1]

uniform vec2 u_resolution;

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

out vec2 v_uv;
flat out int v_instance;

void main() {
    v_uv = in_vert;
    v_instance = gl_InstanceID;

    // Read instance rect: x, y, w, h in pixels
    vec4 rect = data[v_instance * 5 + 0];

    // Expand unit quad to instance rect
    vec2 px = vec2(rect.x, rect.y) + in_vert * vec2(rect.z, rect.w);

    // Convert to NDC
    vec2 ndc = (px / u_resolution) * 2.0 - 1.0;
    ndc.y *= -1.0;  // Flip Y for top-left origin

    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

FRAG_RECT = """
#version 430

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;
flat in int v_instance;
out vec4 fragColor;

void main() {
    int base = v_instance * 5;
    vec4 rect = data[base + 0];
    vec4 fill = data[base + 1];
    vec4 stroke = data[base + 2];
    vec4 params = data[base + 3];

    float corner_radius = params.x;
    float stroke_width = params.y;
    float opacity = params.z;

    // Local coords in pixels from center
    vec2 size = vec2(rect.z, rect.w);
    vec2 p = (v_uv - 0.5) * size;

    // Rounded rect SDF
    vec2 q = abs(p) - size * 0.5 + corner_radius;
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - corner_radius;

    // Fill
    float fill_mask = 1.0 - smoothstep(-1.0, 0.0, d);

    // Stroke
    float stroke_mask = 1.0 - smoothstep(0.0, 1.0, abs(d) - stroke_width * 0.5);

    vec3 col = fill.rgb * fill.a * fill_mask;
    col = mix(col, stroke.rgb, stroke_mask * stroke.a);

    fragColor = vec4(col, fill_mask * opacity);
}
"""

FRAG_CIRCLE = """
#version 430

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;
flat in int v_instance;
out vec4 fragColor;

void main() {
    int base = v_instance * 5;
    vec4 rect = data[base + 0];
    vec4 fill = data[base + 1];
    vec4 stroke = data[base + 2];
    vec4 params = data[base + 3];

    float stroke_width = params.x;
    float opacity = params.y;
    float glow = params.z;

    // Local coords [-1, 1]
    vec2 p = v_uv * 2.0 - 1.0;
    float r = length(p);

    // Circle SDF (radius 1)
    float d = r - 1.0;

    // Fill (inside circle)
    float fill_mask = 1.0 - smoothstep(-0.02, 0.0, d);

    // Stroke (ring at edge)
    float ring_inner = 1.0 - stroke_width;
    float ring = smoothstep(ring_inner - 0.02, ring_inner, r)
               * (1.0 - smoothstep(1.0, 1.0 + 0.02, r));

    // Glow (soft outer halo)
    float glow_mask = (1.0 - smoothstep(1.0, 1.0 + 0.15, r)) * glow;

    vec3 col = fill.rgb * fill.a * fill_mask;
    col += stroke.rgb * ring * stroke.a;
    col += fill.rgb * glow_mask * 0.5;

    float alpha = max(fill_mask * fill.a, max(ring * stroke.a, glow_mask * 0.3));
    fragColor = vec4(col, alpha * opacity);
}
"""

FRAG_DIAL = """
#version 430

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;
flat in int v_instance;
out vec4 fragColor;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

void main() {
    int base = v_instance * 5;
    vec4 rect = data[base + 0];
    vec4 accent = data[base + 1];
    vec4 stroke = data[base + 2];
    vec4 params0 = data[base + 3];
    vec4 params1 = data[base + 4];  // arcs (4 quarter lengths)

    float needle_angle = params0.x;
    float opacity = params0.y;
    float glow = params0.z;
    float ring_r = params0.w;
    if (ring_r == 0.0) ring_r = 0.78;

    // Local coords [-1, 1]
    vec2 p = v_uv * 2.0 - 1.0;
    float r = length(p);
    float ang = atan(p.y, p.x);
    if (ang < 0.0) ang += TAU;

    // Ring parameters
    float ring_t = 0.06;
    float ring_mask = smoothstep(ring_r - ring_t - 0.01, ring_r - ring_t, r)
                    * (1.0 - smoothstep(ring_r + ring_t, ring_r + ring_t + 0.01, r));

    // Quarter arcs (each spans PI/2)
    float q = PI * 0.5;
    float arc_mask = 0.0;
    for (int k = 0; k < 4; k++) {
        float a0 = float(k) * q;
        float a1 = a0 + q * params1[k];
        float in_arc = step(a0, ang) * step(ang, a1);
        arc_mask = max(arc_mask, in_arc);
    }

    // Crosshair lines
    float vx = 1.0 - smoothstep(0.0, 0.015, abs(p.x));
    float vy = 1.0 - smoothstep(0.0, 0.015, abs(p.y));
    float cross = max(vx, vy) * 0.2 * (1.0 - smoothstep(0.15, 0.2, r));

    // Needle
    vec2 dir = vec2(cos(needle_angle), sin(needle_angle));
    float proj = dot(p, dir);
    vec2 perp = p - dir * proj;
    float d_needle = length(perp);
    float needle_mask = smoothstep(0.025, 0.0, d_needle)
                      * smoothstep(-0.15, 0.3, proj)
                      * (1.0 - smoothstep(0.85, 0.9, proj));

    // Build color
    vec3 col = vec3(0.06);

    // Ring base stroke
    col += ring_mask * stroke.rgb * 0.5;

    // Ring accent (where arcs are active)
    col += ring_mask * arc_mask * accent.rgb * 1.2;

    // Needle
    col += needle_mask * accent.rgb * 1.3;

    // Crosshair
    col += cross * stroke.rgb;

    // Glow on ring
    float glow_ring = smoothstep(ring_r + ring_t + 0.08, ring_r + ring_t, r)
                    * ring_mask * arc_mask;
    col += glow_ring * accent.rgb * glow;

    // Center dot
    float center_dot = 1.0 - smoothstep(0.04, 0.06, r);
    col += center_dot * accent.rgb * 0.4;

    fragColor = vec4(col, opacity);
}
"""

FRAG_ARC = """
#version 430

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;
flat in int v_instance;
out vec4 fragColor;

const float PI = 3.14159265359;
const float TAU = 6.28318530718;

void main() {
    int base = v_instance * 5;
    vec4 rect = data[base + 0];
    vec4 accent = data[base + 1];
    vec4 stroke = data[base + 2];
    vec4 params0 = data[base + 3];

    float start_angle = params0.x;
    float end_angle = params0.y;
    float thickness = params0.z;
    float opacity = params0.w;

    if (thickness == 0.0) thickness = 0.08;
    if (opacity == 0.0) opacity = 1.0;

    // Local coords [-1, 1]
    vec2 p = v_uv * 2.0 - 1.0;
    float r = length(p);
    float ang = atan(p.y, p.x);
    if (ang < 0.0) ang += TAU;

    // Normalize angles
    float a0 = mod(start_angle, TAU);
    float a1 = mod(end_angle, TAU);

    // Check if angle is in arc
    float in_arc;
    if (a0 <= a1) {
        in_arc = step(a0, ang) * step(ang, a1);
    } else {
        in_arc = step(a0, ang) + step(ang, a1);
        in_arc = min(in_arc, 1.0);
    }

    // Ring mask
    float ring_r = 0.85;
    float ring_mask = smoothstep(ring_r - thickness - 0.01, ring_r - thickness, r)
                    * (1.0 - smoothstep(ring_r + thickness, ring_r + thickness + 0.01, r));

    // Combine
    float mask = ring_mask * in_arc;

    // Soft glow
    float glow = smoothstep(ring_r + thickness + 0.1, ring_r + thickness, r) * in_arc * 0.3;

    vec3 col = accent.rgb * mask + accent.rgb * glow;

    fragColor = vec4(col, (mask + glow * 0.5) * opacity * accent.a);
}
"""

FRAG_LINE = """
#version 430

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;
flat in int v_instance;
out vec4 fragColor;

void main() {
    int base = v_instance * 5;
    vec4 rect = data[base + 0];
    vec4 color = data[base + 1];
    vec4 params = data[base + 3];

    float line_width = params.x;
    float opacity = params.y;

    if (line_width == 0.0) line_width = 2.0;
    if (opacity == 0.0) opacity = 1.0;

    // For a horizontal line in the rect
    vec2 size = vec2(rect.z, rect.w);
    float d = abs(v_uv.y - 0.5) * size.y;
    float mask = 1.0 - smoothstep(line_width * 0.5 - 0.5, line_width * 0.5 + 0.5, d);

    fragColor = vec4(color.rgb, mask * opacity * color.a);
}
"""


# =============================================================================
# Procedural Renderer
# =============================================================================

class ProceduralRenderer:
    """
    GPU renderer for procedural UI components.

    Uses instanced quad rendering with SSBO for instance data.
    Each component kind has its own fragment shader.
    """

    def __init__(self, ctx: moderngl.Context):
        """
        Initialize renderer.

        Args:
            ctx: moderngl context
        """
        self.ctx = ctx
        self._style = DARK_THEME

        # Create unit quad VBO (just vertex positions, used as UVs too)
        quad_data = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ], dtype='f4')
        self._quad_vbo = ctx.buffer(quad_data.tobytes())

        # Instance SSBO (dynamically sized)
        self._instance_buffer = ctx.buffer(reserve=STRIDE_FLOATS * 4 * 256)
        self._instance_capacity = 256

        # Compile shaders
        self._programs: Dict[ComponentKind, any] = {}
        self._vaos: Dict[ComponentKind, any] = {}

        self._compile_program(ComponentKind.RECT, FRAG_RECT)
        self._compile_program(ComponentKind.CIRCLE, FRAG_CIRCLE)
        self._compile_program(ComponentKind.DIAL, FRAG_DIAL)
        self._compile_program(ComponentKind.ARC, FRAG_ARC)
        self._compile_program(ComponentKind.LINE, FRAG_LINE)

        # Batched instances by kind
        self._batches: Dict[ComponentKind, List[UIInstance]] = {
            k: [] for k in ComponentKind
        }

        # Text renderer
        self._text_renderer = TextRenderer(ctx)

    def _compile_program(self, kind: ComponentKind, frag_src: str):
        """Compile a shader program for a component kind."""
        prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=frag_src
        )
        self._programs[kind] = prog

        vao = self.ctx.vertex_array(
            prog,
            [(self._quad_vbo, '2f', 'in_vert')]
        )
        self._vaos[kind] = vao

    def set_style(self, style: StyleTokens):
        """Set the style tokens."""
        self._style = style

    def begin_frame(self):
        """Clear batches for new frame."""
        for batch in self._batches.values():
            batch.clear()
        self._text_renderer.begin_frame()

    def add_rect(
        self,
        x: float, y: float, w: float, h: float,
        fill: Tuple[float, ...] = None,
        stroke: Tuple[float, ...] = None,
        corner_radius: float = 0.0,
        stroke_width: float = 1.0,
        opacity: float = 1.0
    ):
        """Add a rectangle to the batch."""
        self._batches[ComponentKind.RECT].append(UIInstance(
            kind=ComponentKind.RECT,
            rect=(x, y, w, h),
            color1=fill or (0.2, 0.2, 0.25, 1.0),
            color2=stroke or self._style.stroke_primary,
            params0=(corner_radius, stroke_width, opacity, 0.0)
        ))

    def add_circle(
        self,
        cx: float, cy: float, radius: float,
        fill: Tuple[float, ...] = None,
        stroke: Tuple[float, ...] = None,
        stroke_width: float = 0.1,
        opacity: float = 1.0,
        glow: float = 0.0
    ):
        """Add a circle to the batch."""
        self._batches[ComponentKind.CIRCLE].append(UIInstance(
            kind=ComponentKind.CIRCLE,
            rect=(cx - radius, cy - radius, radius * 2, radius * 2),
            color1=fill or (0.0, 0.0, 0.0, 0.0),
            color2=stroke or self._style.accent_teal,
            params0=(stroke_width, opacity, glow, 0.0)
        ))

    def add_dial(
        self,
        cx: float, cy: float, radius: float,
        needle_angle: float = 0.0,
        arcs: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        accent: Tuple[float, ...] = None,
        stroke: Tuple[float, ...] = None,
        opacity: float = 1.0,
        glow: float = 0.5,
        ring_radius: float = 0.78
    ):
        """
        Add a dial (ring with needle and quarter arcs).

        Args:
            cx, cy: Center position
            radius: Outer radius
            needle_angle: Needle angle in radians
            arcs: Length of each quarter arc (0-1), starting from right going CCW
            accent: Accent color
            stroke: Stroke color
            opacity: Overall opacity
            glow: Glow intensity
            ring_radius: Ring radius as fraction of component (0-1)
        """
        self._batches[ComponentKind.DIAL].append(UIInstance(
            kind=ComponentKind.DIAL,
            rect=(cx - radius, cy - radius, radius * 2, radius * 2),
            color1=accent or self._style.accent_teal,
            color2=stroke or self._style.stroke_primary,
            params0=(needle_angle, opacity, glow, ring_radius),
            params1=arcs
        ))

    def add_arc(
        self,
        cx: float, cy: float, radius: float,
        start_angle: float,
        end_angle: float,
        color: Tuple[float, ...] = None,
        thickness: float = 0.08,
        opacity: float = 1.0
    ):
        """Add an arc to the batch."""
        self._batches[ComponentKind.ARC].append(UIInstance(
            kind=ComponentKind.ARC,
            rect=(cx - radius, cy - radius, radius * 2, radius * 2),
            color1=color or self._style.accent_teal,
            color2=(0.0, 0.0, 0.0, 0.0),
            params0=(start_angle, end_angle, thickness, opacity)
        ))

    def add_line(
        self,
        x0: float, y0: float, x1: float, y1: float,
        color: Tuple[float, ...] = None,
        width: float = 2.0,
        opacity: float = 1.0
    ):
        """Add a line to the batch."""
        # Create rect that encompasses the line
        x = min(x0, x1)
        y = min(y0, y1) - width
        w = abs(x1 - x0) or 1.0
        h = abs(y1 - y0) + width * 2 or width * 2

        self._batches[ComponentKind.LINE].append(UIInstance(
            kind=ComponentKind.LINE,
            rect=(x, y, w, h),
            color1=color or self._style.stroke_primary,
            params0=(width, opacity, 0.0, 0.0)
        ))

    # -------------------------------------------------------------------------
    # Text Rendering
    # -------------------------------------------------------------------------

    def load_font(self, path: str, size: int = 24) -> FontAtlas:
        """
        Load a font from file.

        Args:
            path: Path to TTF/OTF font file
            size: Font size in pixels

        Returns:
            FontAtlas for use with add_text()
        """
        return FontAtlas(self.ctx, path, size)

    @property
    def default_font(self) -> FontAtlas:
        """Get the default font."""
        return self._text_renderer.default_font

    def add_text(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[float, ...] = None,
        font: FontAtlas = None,
        align: str = 'left',
        baseline: str = 'alphabetic',
        opacity: float = 1.0,
        max_width: float = None
    ):
        """
        Add text to the render batch.

        Args:
            text: Text string to render
            x, y: Position
            color: RGBA color (default: text_primary from style)
            font: Font to use (default: built-in font)
            align: 'left', 'center', or 'right'
            baseline: 'top', 'middle', 'alphabetic', or 'bottom'
            opacity: Overall opacity
            max_width: Word wrap width (None = no wrap)
        """
        if not text:
            return

        self._text_renderer.draw_text(
            text=text,
            x=x,
            y=y,
            color=color or self._style.text_primary,
            font=font,
            align=align,
            baseline=baseline,
            opacity=opacity,
            max_width=max_width
        )

    def render(self, width: int, height: int):
        """
        Render all batched instances.

        Args:
            width: Viewport width
            height: Viewport height
        """
        # Render shapes first
        for kind in ComponentKind:
            batch = self._batches[kind]
            if not batch:
                continue

            self._render_batch(kind, batch, width, height)

        # Render text on top
        self._text_renderer.render(width, height)

    def _render_batch(
        self,
        kind: ComponentKind,
        instances: List[UIInstance],
        width: int,
        height: int
    ):
        """Render a batch of instances of the same kind."""
        # Ensure buffer capacity
        needed = len(instances) * STRIDE_FLOATS * 4
        if needed > self._instance_capacity * STRIDE_FLOATS * 4:
            new_cap = max(len(instances), self._instance_capacity * 2)
            self._instance_buffer = self.ctx.buffer(reserve=new_cap * STRIDE_FLOATS * 4)
            self._instance_capacity = new_cap

        # Pack and upload
        packed = pack_instances(instances)
        self._instance_buffer.write(packed.tobytes())

        # Bind SSBO
        self._instance_buffer.bind_to_storage_buffer(binding=0)

        # Get program and set uniforms
        prog = self._programs[kind]
        if 'u_resolution' in prog:
            prog['u_resolution'].value = (width, height)

        # Render
        vao = self._vaos[kind]
        vao.render(instances=len(instances))

    def end_frame(self):
        """Finalize frame."""
        self._text_renderer.end_frame()

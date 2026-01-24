# engine/graph/text.py
"""
Text Rendering - Font atlas and text layout system.

Uses Pillow for font loading and glyph rasterization, then renders
text via instanced quads sampling from a font atlas texture.

Architecture:
- FontAtlas: Packs glyphs into a texture, provides UV lookups
- TextLayout: Converts text strings to positioned glyph quads
- TextRenderer: GPU resources and rendering

Usage:
    font = FontAtlas(ctx, "arial.ttf", size=24)
    renderer.draw_text("Hello", x=100, y=100, font=font, color=(1,1,1,1))
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from pathlib import Path
import numpy as np

try:
    from PIL import Image, ImageFont, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

if TYPE_CHECKING:
    import moderngl


# =============================================================================
# Glyph Data
# =============================================================================

@dataclass
class GlyphMetrics:
    """Metrics for a single glyph."""
    char: str
    width: int          # Glyph width in pixels
    height: int         # Glyph height in pixels
    bearing_x: int      # Offset from cursor to left edge
    bearing_y: int      # Offset from baseline to top
    advance: int        # Cursor advance after this glyph

    # Atlas position
    atlas_x: int = 0
    atlas_y: int = 0

    # UV coordinates (normalized 0-1)
    uv_x0: float = 0.0
    uv_y0: float = 0.0
    uv_x1: float = 0.0
    uv_y1: float = 0.0


# =============================================================================
# Font Atlas
# =============================================================================

class FontAtlas:
    """
    Texture atlas containing rendered glyphs.

    Generates a texture with all ASCII printable characters plus
    common symbols. Provides UV lookup for text rendering.
    """

    # Default character set (ASCII printable + common symbols)
    DEFAULT_CHARS = (
        " !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
        "°±²³´µ¶·¸¹º»¼½¾¿×÷"
        "–—''""•…™©®"
    )

    def __init__(
        self,
        ctx: moderngl.Context,
        font_path: str = None,
        size: int = 24,
        chars: str = None,
        padding: int = 2
    ):
        """
        Create a font atlas.

        Args:
            ctx: ModernGL context
            font_path: Path to TTF/OTF font file (None = default font)
            size: Font size in pixels
            chars: Characters to include (None = DEFAULT_CHARS)
            padding: Padding between glyphs in atlas
        """
        if not HAS_PIL:
            raise RuntimeError("Pillow is required for text rendering")

        self.ctx = ctx
        self.size = size
        self.padding = padding
        self._chars = chars or self.DEFAULT_CHARS

        # Load font
        self._font = self._load_font(font_path, size)

        # Glyph data
        self._glyphs: Dict[str, GlyphMetrics] = {}
        self._atlas_width = 0
        self._atlas_height = 0

        # GPU texture
        self._texture: Optional[moderngl.Texture] = None

        # Build atlas
        self._build_atlas()

    def _load_font(self, font_path: str, size: int):
        """Load a font file or use default."""
        if font_path and Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

        # Try common system fonts
        system_fonts = [
            "C:/Windows/Fonts/consola.ttf",   # Consolas (Windows)
            "C:/Windows/Fonts/arial.ttf",      # Arial (Windows)
            "C:/Windows/Fonts/segoeui.ttf",    # Segoe UI (Windows)
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
            "/System/Library/Fonts/Menlo.ttc",  # macOS
        ]

        for path in system_fonts:
            if Path(path).exists():
                return ImageFont.truetype(path, size)

        # Fall back to default (bitmap) font
        return ImageFont.load_default()

    def _build_atlas(self):
        """Render all glyphs and pack into atlas texture."""
        # First pass: measure all glyphs
        glyph_images = {}
        max_height = 0

        for char in self._chars:
            # Get glyph bounding box
            bbox = self._font.getbbox(char)
            if bbox is None:
                continue

            left, top, right, bottom = bbox
            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                # Space or zero-width character
                width = max(width, 1)
                height = max(height, self.size)

            # Render glyph to image
            img = Image.new('L', (width + 4, height + 4), 0)
            draw = ImageDraw.Draw(img)
            draw.text((-left + 2, -top + 2), char, font=self._font, fill=255)

            # Get advance width
            advance = int(self._font.getlength(char))

            glyph_images[char] = img
            self._glyphs[char] = GlyphMetrics(
                char=char,
                width=width,
                height=height,
                bearing_x=-left,
                bearing_y=top,
                advance=advance
            )

            max_height = max(max_height, height + 4)

        # Calculate atlas size (simple row packing)
        row_width = 0
        row_height = max_height + self.padding
        num_rows = 1

        for char, img in glyph_images.items():
            if row_width + img.width + self.padding > 1024:
                # Start new row
                row_width = 0
                num_rows += 1
            row_width += img.width + self.padding

        # Atlas dimensions (power of 2 preferred)
        self._atlas_width = 1024
        self._atlas_height = self._next_pow2(num_rows * row_height)

        # Create atlas image
        atlas = Image.new('L', (self._atlas_width, self._atlas_height), 0)

        # Second pass: pack glyphs into atlas
        x, y = self.padding, self.padding

        for char, img in glyph_images.items():
            if x + img.width + self.padding > self._atlas_width:
                x = self.padding
                y += row_height

            # Paste glyph
            atlas.paste(img, (x, y))

            # Update glyph metrics with atlas position
            glyph = self._glyphs[char]
            glyph.atlas_x = x
            glyph.atlas_y = y

            # Calculate UVs (normalized)
            glyph.uv_x0 = x / self._atlas_width
            glyph.uv_y0 = y / self._atlas_height
            glyph.uv_x1 = (x + img.width) / self._atlas_width
            glyph.uv_y1 = (y + img.height) / self._atlas_height

            x += img.width + self.padding

        # Upload to GPU
        atlas_data = atlas.tobytes()
        self._texture = self.ctx.texture(
            (self._atlas_width, self._atlas_height),
            1,  # Single channel (red)
            atlas_data
        )
        self._texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)

        # Add space character if missing
        if ' ' not in self._glyphs:
            self._glyphs[' '] = GlyphMetrics(
                char=' ',
                width=0, height=0,
                bearing_x=0, bearing_y=0,
                advance=self.size // 3
            )

    def _next_pow2(self, n: int) -> int:
        """Round up to next power of 2."""
        return 1 << (n - 1).bit_length()

    def get_glyph(self, char: str) -> Optional[GlyphMetrics]:
        """Get metrics for a character."""
        return self._glyphs.get(char)

    @property
    def texture(self) -> moderngl.Texture:
        """Get the atlas texture."""
        return self._texture

    @property
    def line_height(self) -> int:
        """Get line height."""
        return int(self.size * 1.2)

    def measure_text(self, text: str) -> Tuple[int, int]:
        """Measure text dimensions."""
        width = 0
        height = self.line_height

        for char in text:
            glyph = self._glyphs.get(char)
            if glyph:
                width += glyph.advance

        return (width, height)


# =============================================================================
# Text Layout
# =============================================================================

@dataclass
class GlyphQuad:
    """A positioned glyph for rendering."""
    x: float            # Screen X
    y: float            # Screen Y
    width: float        # Quad width
    height: float       # Quad height
    uv_x0: float        # Texture UV left
    uv_y0: float        # Texture UV top
    uv_x1: float        # Texture UV right
    uv_y1: float        # Texture UV bottom


class TextAlign:
    LEFT = 'left'
    CENTER = 'center'
    RIGHT = 'right'


class TextBaseline:
    TOP = 'top'
    MIDDLE = 'middle'
    ALPHABETIC = 'alphabetic'
    BOTTOM = 'bottom'


def layout_text(
    text: str,
    x: float,
    y: float,
    font: FontAtlas,
    align: str = TextAlign.LEFT,
    baseline: str = TextBaseline.ALPHABETIC,
    max_width: float = None,
    line_spacing: float = 1.2
) -> List[GlyphQuad]:
    """
    Convert text to positioned glyph quads.

    Args:
        text: Text string to layout
        x, y: Position (interpretation depends on align/baseline)
        font: FontAtlas to use
        align: Horizontal alignment (left, center, right)
        baseline: Vertical alignment
        max_width: Wrap text if wider than this
        line_spacing: Line height multiplier

    Returns:
        List of GlyphQuad for rendering
    """
    quads = []
    lines = text.split('\n')

    # Calculate total height
    line_height = int(font.size * line_spacing)
    total_height = len(lines) * line_height

    # Adjust Y based on baseline
    if baseline == TextBaseline.TOP:
        start_y = y
    elif baseline == TextBaseline.MIDDLE:
        start_y = y - total_height / 2
    elif baseline == TextBaseline.BOTTOM:
        start_y = y - total_height
    else:  # ALPHABETIC
        start_y = y - font.size

    current_y = start_y

    for line in lines:
        # Word wrap if needed
        if max_width:
            wrapped_lines = _word_wrap(line, font, max_width)
        else:
            wrapped_lines = [line]

        for wrapped in wrapped_lines:
            line_width = font.measure_text(wrapped)[0]

            # Adjust X based on alignment
            if align == TextAlign.CENTER:
                current_x = x - line_width / 2
            elif align == TextAlign.RIGHT:
                current_x = x - line_width
            else:  # LEFT
                current_x = x

            # Layout each character
            for char in wrapped:
                glyph = font.get_glyph(char)
                if glyph is None:
                    glyph = font.get_glyph('?')
                if glyph is None:
                    continue

                if glyph.width > 0 and glyph.height > 0:
                    # Calculate quad position
                    qx = current_x + glyph.bearing_x
                    qy = current_y + glyph.bearing_y

                    quads.append(GlyphQuad(
                        x=qx,
                        y=qy,
                        width=glyph.width + 4,  # Include padding
                        height=glyph.height + 4,
                        uv_x0=glyph.uv_x0,
                        uv_y0=glyph.uv_y0,
                        uv_x1=glyph.uv_x1,
                        uv_y1=glyph.uv_y1
                    ))

                current_x += glyph.advance

            current_y += line_height

    return quads


def _word_wrap(text: str, font: FontAtlas, max_width: float) -> List[str]:
    """Wrap text to fit within max_width."""
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0
    space_width = font.measure_text(' ')[0]

    for word in words:
        word_width = font.measure_text(word)[0]

        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width + space_width
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_width = word_width + space_width

    if current_line:
        lines.append(' '.join(current_line))

    return lines if lines else ['']


# =============================================================================
# Text Shader
# =============================================================================

TEXT_VERTEX_SHADER = """
#version 430

in vec2 in_vert;

uniform vec2 u_resolution;

layout(std430, binding = 0) buffer Instances {
    vec4 data[];  // 4 vec4 per glyph: rect, uv, color, params
};

out vec2 v_uv;
flat out int v_instance;

void main() {
    v_instance = gl_InstanceID;

    // Read glyph rect: x, y, w, h
    vec4 rect = data[v_instance * 4 + 0];
    vec4 uv_rect = data[v_instance * 4 + 1];

    // Interpolate UV within glyph
    v_uv = mix(uv_rect.xy, uv_rect.zw, in_vert);

    // Expand quad to glyph rect
    vec2 px = rect.xy + in_vert * rect.zw;

    // To NDC
    vec2 ndc = (px / u_resolution) * 2.0 - 1.0;
    ndc.y *= -1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

TEXT_FRAGMENT_SHADER = """
#version 430

uniform sampler2D u_atlas;

layout(std430, binding = 0) buffer Instances {
    vec4 data[];
};

in vec2 v_uv;
flat in int v_instance;

out vec4 fragColor;

void main() {
    // Read color
    vec4 color = data[v_instance * 4 + 2];
    vec4 params = data[v_instance * 4 + 3];

    float opacity = params.x;

    // Sample atlas (single channel)
    float alpha = texture(u_atlas, v_uv).r;

    // Apply color
    fragColor = vec4(color.rgb, alpha * color.a * opacity);
}
"""


# =============================================================================
# Text Renderer
# =============================================================================

@dataclass
class TextInstance:
    """Instance data for one glyph quad."""
    rect: Tuple[float, float, float, float]      # x, y, w, h
    uv_rect: Tuple[float, float, float, float]   # u0, v0, u1, v1
    color: Tuple[float, float, float, float]     # r, g, b, a
    params: Tuple[float, float, float, float]    # opacity, _, _, _


class TextRenderer:
    """
    GPU text renderer using font atlas.

    Renders text as instanced quads, one quad per glyph.
    """

    STRIDE_FLOATS = 16  # 4 vec4 per glyph

    def __init__(self, ctx: moderngl.Context):
        """Initialize text renderer."""
        self.ctx = ctx

        # Default font (lazy loaded)
        self._default_font: Optional[FontAtlas] = None

        # Unit quad VBO
        quad_data = np.array([
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 0.0,
            1.0, 1.0,
            0.0, 1.0,
        ], dtype='f4')
        self._quad_vbo = ctx.buffer(quad_data.tobytes())

        # Instance buffer
        self._instance_buffer = ctx.buffer(reserve=self.STRIDE_FLOATS * 4 * 1024)
        self._instance_capacity = 1024

        # Compile shader
        self._program = ctx.program(
            vertex_shader=TEXT_VERTEX_SHADER,
            fragment_shader=TEXT_FRAGMENT_SHADER
        )
        self._vao = ctx.vertex_array(
            self._program,
            [(self._quad_vbo, '2f', 'in_vert')]
        )

        # Batch of glyph instances
        self._instances: List[TextInstance] = []
        self._current_font: Optional[FontAtlas] = None

    @property
    def default_font(self) -> FontAtlas:
        """Get or create default font."""
        if self._default_font is None:
            self._default_font = FontAtlas(self.ctx, size=18)
        return self._default_font

    def load_font(self, path: str, size: int = 24) -> FontAtlas:
        """Load a font from file."""
        return FontAtlas(self.ctx, path, size)

    def begin_frame(self):
        """Clear instances for new frame."""
        self._instances.clear()
        self._current_font = None

    def draw_text(
        self,
        text: str,
        x: float,
        y: float,
        color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        font: FontAtlas = None,
        align: str = TextAlign.LEFT,
        baseline: str = TextBaseline.ALPHABETIC,
        opacity: float = 1.0,
        max_width: float = None
    ):
        """
        Add text to render batch.

        Args:
            text: Text to render
            x, y: Position
            color: RGBA color
            font: Font to use (None = default)
            align: Horizontal alignment
            baseline: Vertical alignment
            opacity: Overall opacity
            max_width: Wrap text if wider
        """
        if not text:
            return

        font = font or self.default_font

        # Track font changes (for multi-font support in future)
        if self._current_font is None:
            self._current_font = font
        elif self._current_font != font:
            # Would need to flush and switch textures
            # For now, just use the new font
            self._current_font = font

        # Layout text to glyphs
        quads = layout_text(
            text, x, y, font,
            align=align,
            baseline=baseline,
            max_width=max_width
        )

        # Convert to instances
        for quad in quads:
            self._instances.append(TextInstance(
                rect=(quad.x, quad.y, quad.width, quad.height),
                uv_rect=(quad.uv_x0, quad.uv_y0, quad.uv_x1, quad.uv_y1),
                color=color,
                params=(opacity, 0.0, 0.0, 0.0)
            ))

    def render(self, width: int, height: int):
        """Render all batched text."""
        if not self._instances or self._current_font is None:
            return

        # Ensure buffer capacity
        needed = len(self._instances) * self.STRIDE_FLOATS * 4
        if needed > self._instance_capacity * self.STRIDE_FLOATS * 4:
            new_cap = max(len(self._instances), self._instance_capacity * 2)
            self._instance_buffer = self.ctx.buffer(reserve=new_cap * self.STRIDE_FLOATS * 4)
            self._instance_capacity = new_cap

        # Pack instances
        packed = self._pack_instances()
        self._instance_buffer.write(packed.tobytes())

        # Bind resources
        self._instance_buffer.bind_to_storage_buffer(binding=0)
        self._current_font.texture.use(location=0)

        # Set uniforms
        self._program['u_resolution'].value = (width, height)
        if 'u_atlas' in self._program:
            self._program['u_atlas'].value = 0

        # Render
        self._vao.render(instances=len(self._instances))

    def _pack_instances(self) -> np.ndarray:
        """Pack glyph instances into array."""
        out = np.zeros((len(self._instances), self.STRIDE_FLOATS), dtype=np.float32)

        for i, inst in enumerate(self._instances):
            out[i, 0:4] = inst.rect
            out[i, 4:8] = inst.uv_rect
            out[i, 8:12] = inst.color
            out[i, 12:16] = inst.params

        return out.ravel()

    def end_frame(self):
        """Finalize frame."""
        pass

# engine/graph/style.py
"""
Style Tokens - Design system for consistent theming.

Style tokens define the visual vocabulary: colors, strokes, glows, spacing.
They are not component-specific; components reference tokens semantically.

The GPU reads tokens from a uniform buffer, enabling instant theme switching
by updating a single buffer.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
import struct

# Type alias for RGBA color (0-1 range)
Vec4 = Tuple[float, float, float, float]


@dataclass(frozen=True)
class StyleTokens:
    """
    Immutable style token set.

    Colors are RGBA tuples in 0-1 range.
    """
    # Background colors
    bg_app: Vec4 = (0.04, 0.04, 0.06, 1.0)      # Main app background
    bg_panel: Vec4 = (0.08, 0.08, 0.10, 1.0)    # Panel background
    bg_well: Vec4 = (0.06, 0.06, 0.08, 1.0)     # Inset/well background

    # Stroke colors
    stroke_primary: Vec4 = (1.0, 1.0, 1.0, 0.12)
    stroke_secondary: Vec4 = (1.0, 1.0, 1.0, 0.06)
    stroke_grid: Vec4 = (1.0, 1.0, 1.0, 0.04)

    # Text colors
    text_primary: Vec4 = (0.85, 0.84, 0.82, 1.0)
    text_secondary: Vec4 = (0.60, 0.62, 0.65, 1.0)
    text_muted: Vec4 = (0.40, 0.42, 0.45, 1.0)

    # Beat accent colors (1, 2, 3, 4)
    accent_beat1: Vec4 = (1.0, 0.42, 0.42, 1.0)   # #ff6b6b - Red
    accent_beat2: Vec4 = (1.0, 0.85, 0.24, 1.0)   # #ffd93d - Yellow
    accent_beat3: Vec4 = (0.42, 0.80, 0.47, 1.0)  # #6bcb77 - Green
    accent_beat4: Vec4 = (0.30, 0.59, 1.0, 1.0)   # #4d96ff - Blue

    # Additional accents
    accent_teal: Vec4 = (0.68, 0.91, 0.87, 1.0)
    accent_gold: Vec4 = (0.89, 0.70, 0.42, 1.0)
    accent_green: Vec4 = (0.58, 0.69, 0.51, 1.0)
    accent_peri: Vec4 = (0.63, 0.65, 0.78, 1.0)   # Periwinkle
    accent_cyan: Vec4 = (0.0, 0.83, 1.0, 1.0)     # #00d4ff

    # Glow parameters
    glow_core: float = 0.55
    glow_halo: float = 0.18

    # Geometry
    corner_radius: float = 4.0
    ring_thickness: float = 0.06
    line_width: float = 1.0

    def get_beat_color(self, beat_index: int) -> Vec4:
        """Get accent color for beat 1-4 (wraps)."""
        colors = [self.accent_beat1, self.accent_beat2,
                  self.accent_beat3, self.accent_beat4]
        return colors[beat_index % 4]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def with_overrides(self, **kwargs) -> StyleTokens:
        """Create new tokens with overridden values."""
        data = self.to_dict()
        data.update(kwargs)
        return StyleTokens(**data)


# Default dark theme
DARK_THEME = StyleTokens()

# Alternative themes
LIGHT_THEME = StyleTokens(
    bg_app=(0.94, 0.94, 0.96, 1.0),
    bg_panel=(0.98, 0.98, 0.99, 1.0),
    bg_well=(0.90, 0.90, 0.92, 1.0),
    stroke_primary=(0.0, 0.0, 0.0, 0.15),
    stroke_secondary=(0.0, 0.0, 0.0, 0.08),
    text_primary=(0.15, 0.15, 0.18, 1.0),
    text_secondary=(0.45, 0.45, 0.50, 1.0),
)

HIGH_CONTRAST_THEME = StyleTokens(
    bg_app=(0.0, 0.0, 0.0, 1.0),
    bg_panel=(0.05, 0.05, 0.05, 1.0),
    stroke_primary=(1.0, 1.0, 1.0, 0.25),
    glow_core=0.8,
    glow_halo=0.3,
)


# =============================================================================
# Style UBO for GPU
# =============================================================================

class StyleUBO:
    """
    Manages GPU uniform buffer for style tokens.

    Layout (std140):
        vec4 bg_app
        vec4 bg_panel
        vec4 bg_well
        vec4 stroke_primary
        vec4 stroke_secondary
        vec4 text_primary
        vec4 text_secondary
        vec4 accent_beat1
        vec4 accent_beat2
        vec4 accent_beat3
        vec4 accent_beat4
        vec4 accent_teal
        vec4 accent_gold
        vec4 params  (glow_core, glow_halo, corner_radius, ring_thickness)
    """

    BINDING = 1  # UBO binding point

    def __init__(self, ctx):
        """
        Create style UBO.

        Args:
            ctx: moderngl context
        """
        self.ctx = ctx
        # 14 vec4s = 56 floats = 224 bytes
        self._buffer = ctx.buffer(reserve=224)
        self._tokens: StyleTokens = DARK_THEME

    @property
    def tokens(self) -> StyleTokens:
        return self._tokens

    def set_tokens(self, tokens: StyleTokens):
        """Update tokens and upload to GPU."""
        self._tokens = tokens
        self._upload()

    def _upload(self):
        """Pack and upload tokens to GPU buffer."""
        t = self._tokens
        data = struct.pack(
            '56f',
            # Colors (14 vec4s)
            *t.bg_app,
            *t.bg_panel,
            *t.bg_well,
            *t.stroke_primary,
            *t.stroke_secondary,
            *t.text_primary,
            *t.text_secondary,
            *t.accent_beat1,
            *t.accent_beat2,
            *t.accent_beat3,
            *t.accent_beat4,
            *t.accent_teal,
            *t.accent_gold,
            # Params
            t.glow_core, t.glow_halo, t.corner_radius, t.ring_thickness,
        )
        self._buffer.write(data)

    def bind(self):
        """Bind UBO to its binding point."""
        self._buffer.bind_to_uniform_block(self.BINDING)

    @property
    def buffer(self):
        return self._buffer

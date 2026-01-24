# examples/graph_demo.py
"""
Graph Demo - Animated beat circles using the procedural renderer.

Demonstrates:
- ProceduralRenderer with dial, circle, arc components
- Beat-synchronized animation
- Style tokens / theming

Run with:
    python examples/graph_demo.py
"""

import math
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import moderngl_window as mglw
from moderngl_window.conf import settings

from engine.graph.renderer import ProceduralRenderer, ComponentKind
from engine.graph.style import DARK_THEME, StyleTokens


class GraphDemo(mglw.WindowConfig):
    """Demo window showing animated beat circles."""

    gl_version = (4, 3)  # Need 4.3 for SSBO
    title = "SPECTRO Graph Demo - Beat Circles"
    window_size = (1200, 700)
    aspect_ratio = None
    resizable = True
    vsync = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create renderer
        self.renderer = ProceduralRenderer(self.ctx)
        self.renderer.set_style(DARK_THEME)

        # Animation state
        self.time = 0.0
        self.bpm = 120.0
        self.beat = 0.0

        # Beat colors
        self.beat_colors = [
            DARK_THEME.accent_beat1,
            DARK_THEME.accent_beat2,
            DARK_THEME.accent_beat3,
            DARK_THEME.accent_beat4,
        ]

    def on_render(self, time: float, frame_time: float):
        """Render frame."""
        # Update time and beat
        self.time = time
        beats_per_second = self.bpm / 60.0
        self.beat = (time * beats_per_second) % 4.0

        # Clear
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)

        # Enable blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA
        )

        # Get window size
        w, h = self.wnd.size

        # Begin frame
        self.renderer.begin_frame()

        # Draw background panels
        self._draw_panels(w, h)

        # Draw sync circles (row 1)
        self._draw_sync_circles(w, h)

        # Draw beat circles with histograms (row 1 right)
        self._draw_beat_circles(w, h)

        # Draw dials (row 2)
        self._draw_dials(w, h)

        # Draw phase arcs
        self._draw_phase_arcs(w, h)

        # Draw text labels
        self._draw_labels(w, h)

        # Render all
        self.renderer.render(w, h)
        self.renderer.end_frame()

    def _draw_panels(self, w: int, h: int):
        """Draw background panels."""
        # Main panels
        self.renderer.add_rect(
            40, 40, 520, 200,
            fill=(0.06, 0.06, 0.08, 1.0),
            stroke=(1.0, 1.0, 1.0, 0.08),
            corner_radius=8,
            stroke_width=1
        )

        self.renderer.add_rect(
            580, 40, w - 620, 200,
            fill=(0.06, 0.06, 0.08, 1.0),
            stroke=(1.0, 1.0, 1.0, 0.08),
            corner_radius=8,
            stroke_width=1
        )

        self.renderer.add_rect(
            40, 260, w - 80, 280,
            fill=(0.06, 0.06, 0.08, 1.0),
            stroke=(1.0, 1.0, 1.0, 0.08),
            corner_radius=8,
            stroke_width=1
        )

    def _draw_sync_circles(self, w: int, h: int):
        """Draw 4 sync circles showing bar phase."""
        cx_start = 140
        spacing = 110
        cy = 140
        radius = 45

        for i in range(4):
            cx = cx_start + i * spacing
            color = self.beat_colors[i]

            # Current beat phase (0-1 within this beat)
            beat_index = int(self.beat)
            phase_in_beat = self.beat - beat_index

            # This circle is "active" if we're on this beat
            is_active = (beat_index == i)

            # Ring opacity based on whether active
            ring_opacity = 1.0 if is_active else 0.4

            # Draw ring
            self.renderer.add_circle(
                cx, cy, radius,
                fill=(0.0, 0.0, 0.0, 0.0),
                stroke=(*color[:3], ring_opacity),
                stroke_width=0.06,
                glow=0.4 if is_active else 0.1
            )

            # Draw phase dot (rotating around circle)
            if is_active:
                # Dot rotates from top (-PI/2) clockwise
                angle = -math.pi / 2 + phase_in_beat * 2 * math.pi
                dot_x = cx + math.cos(angle) * (radius - 5)
                dot_y = cy + math.sin(angle) * (radius - 5)

                self.renderer.add_circle(
                    dot_x, dot_y, 6,
                    fill=color,
                    stroke=(0.0, 0.0, 0.0, 0.0),
                    glow=0.8
                )

            # Beat number label position
            # (text not implemented yet, but we mark the spot)

    def _draw_beat_circles(self, w: int, h: int):
        """Draw individual beat circles with hit visualization."""
        cx_start = 680
        spacing = 140
        cy = 100
        radius = 35

        for i in range(4):
            cx = cx_start + i * spacing
            color = self.beat_colors[i]

            # Draw ring
            self.renderer.add_circle(
                cx, cy, radius,
                fill=(0.0, 0.0, 0.0, 0.0),
                stroke=(*color[:3], 0.6),
                stroke_width=0.05,
                glow=0.2
            )

            # Simulate some hit dots with fade
            # In real app, these come from timing data
            num_dots = 3
            for j in range(num_dots):
                # Fake hit positions
                hit_time = (self.time * 0.3 + i * 0.7 + j * 0.4) % 1.0
                hit_angle = -math.pi / 2 + hit_time * 0.3  # Slight variation

                dot_x = cx + math.cos(hit_angle) * (radius - 3)
                dot_y = cy + math.sin(hit_angle) * (radius - 3)

                # Fade based on "age"
                age = (j + 1) / num_dots
                opacity = 1.0 - age * 0.7

                self.renderer.add_circle(
                    dot_x, dot_y, 4 - j,
                    fill=(*color[:3], opacity),
                    stroke=(0.0, 0.0, 0.0, 0.0),
                    glow=0.3 * opacity
                )

            # Draw histogram below (simplified as rects)
            hist_y = cy + radius + 20
            hist_w = 60
            hist_h = 35

            self.renderer.add_rect(
                cx - hist_w / 2, hist_y, hist_w, hist_h,
                fill=(0.04, 0.04, 0.06, 1.0),
                stroke=(1.0, 1.0, 1.0, 0.1),
                corner_radius=2
            )

            # Histogram bars (fake data)
            bar_count = 6
            bar_w = hist_w / bar_count - 2
            for b in range(bar_count):
                # Random-ish height based on beat and bar index
                bar_h = (math.sin(self.time + i + b * 0.5) * 0.5 + 0.5) * (hist_h - 4)
                bar_x = cx - hist_w / 2 + 2 + b * (bar_w + 2)
                bar_y = hist_y + hist_h - 2 - bar_h

                bar_opacity = 0.3 + (math.sin(self.time * 2 + b) * 0.5 + 0.5) * 0.6

                self.renderer.add_rect(
                    bar_x, bar_y, bar_w, bar_h,
                    fill=(*color[:3], bar_opacity),
                    corner_radius=1
                )

    def _draw_dials(self, w: int, h: int):
        """Draw dial components with needle animation."""
        dials = [
            {"cx": 200, "cy": 400, "color": DARK_THEME.accent_teal, "speed": 1.2},
            {"cx": 400, "cy": 400, "color": DARK_THEME.accent_gold, "speed": 0.7},
            {"cx": 600, "cy": 400, "color": DARK_THEME.accent_green, "speed": 1.5},
            {"cx": 800, "cy": 400, "color": DARK_THEME.accent_peri, "speed": 0.9},
        ]

        for i, dial in enumerate(dials):
            # Animate needle
            needle_angle = math.sin(self.time * dial["speed"]) * math.pi * 0.8

            # Animate arc lengths
            base_arc = 0.25
            arcs = tuple(
                base_arc + 0.2 * math.sin(self.time * (0.5 + j * 0.3) + i)
                for j in range(4)
            )

            self.renderer.add_dial(
                dial["cx"], dial["cy"], 80,
                needle_angle=needle_angle,
                arcs=arcs,
                accent=dial["color"],
                glow=0.5,
                opacity=1.0
            )

    def _draw_phase_arcs(self, w: int, h: int):
        """Draw phase indicator arcs."""
        # Beat phase arc
        beat_phase = self.beat % 1.0
        beat_index = int(self.beat) % 4

        cx, cy = w - 150, 400
        radius = 60

        # Background ring
        self.renderer.add_circle(
            cx, cy, radius,
            fill=(0.0, 0.0, 0.0, 0.0),
            stroke=(1.0, 1.0, 1.0, 0.1),
            stroke_width=0.08
        )

        # Phase arc
        start = -math.pi / 2
        end = start + beat_phase * 2 * math.pi

        self.renderer.add_arc(
            cx, cy, radius,
            start, end,
            color=self.beat_colors[beat_index],
            thickness=0.1,
            opacity=1.0
        )

        # Bar phase arc (outer)
        bar_phase = self.beat / 4.0
        bar_end = start + bar_phase * 2 * math.pi

        self.renderer.add_arc(
            cx, cy, radius + 20,
            start, bar_end,
            color=DARK_THEME.accent_cyan,
            thickness=0.06,
            opacity=0.8
        )

    def _draw_labels(self, w: int, h: int):
        """Draw text labels."""
        # Panel titles
        self.renderer.add_text(
            "SYNC CIRCLES",
            x=60, y=55,
            color=DARK_THEME.text_secondary,
            align='left',
            baseline='top'
        )

        self.renderer.add_text(
            "BEAT TIMING",
            x=600, y=55,
            color=DARK_THEME.text_secondary,
            align='left',
            baseline='top'
        )

        self.renderer.add_text(
            "QUARTER DIALS",
            x=60, y=275,
            color=DARK_THEME.text_secondary,
            align='left',
            baseline='top'
        )

        # BPM display
        self.renderer.add_text(
            f"BPM: {self.bpm:.0f}",
            x=460, y=55,
            color=DARK_THEME.text_primary,
            align='right',
            baseline='top'
        )

        # Beat number labels under sync circles
        cx_start = 140
        spacing = 110
        for i in range(4):
            cx = cx_start + i * spacing
            beat_index = int(self.beat) % 4

            # Beat number
            color = self.beat_colors[i] if i == beat_index else DARK_THEME.text_muted
            self.renderer.add_text(
                f"{i + 1}",
                x=cx, y=200,
                color=color,
                align='center',
                baseline='top'
            )

        # Current beat display
        beat_index = int(self.beat) % 4
        beat_phase = self.beat % 1.0

        self.renderer.add_text(
            f"Beat {beat_index + 1}",
            x=w - 150, y=480,
            color=self.beat_colors[beat_index],
            align='center',
            baseline='top'
        )

        self.renderer.add_text(
            f"{beat_phase * 100:.0f}%",
            x=w - 150, y=500,
            color=DARK_THEME.text_secondary,
            align='center',
            baseline='top'
        )

        # Time display
        minutes = int(self.time) // 60
        seconds = int(self.time) % 60
        ms = int((self.time % 1) * 100)

        self.renderer.add_text(
            f"{minutes:02d}:{seconds:02d}.{ms:02d}",
            x=w - 60, y=55,
            color=DARK_THEME.text_primary,
            align='right',
            baseline='top'
        )

        # Bar counter
        bar = int(self.beat / 4) + 1
        bar_beat = int(self.beat % 4) + 1

        self.renderer.add_text(
            f"Bar {bar} . {bar_beat}",
            x=w - 150, y=550,
            color=DARK_THEME.accent_cyan,
            align='center',
            baseline='top'
        )

        # Dial labels
        dial_names = ["Q1", "Q2", "Q3", "Q4"]
        dial_positions = [200, 400, 600, 800]
        for i, (name, dx) in enumerate(zip(dial_names, dial_positions)):
            self.renderer.add_text(
                name,
                x=dx, y=500,
                color=DARK_THEME.text_muted,
                align='center',
                baseline='top'
            )


# Need to import moderngl for constants
import moderngl


if __name__ == "__main__":
    # Configure settings
    settings.WINDOW['class'] = 'moderngl_window.context.pyglet.Window'

    # Run
    GraphDemo.run()

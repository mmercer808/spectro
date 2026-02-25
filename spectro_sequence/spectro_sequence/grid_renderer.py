"""
GridRenderer - Shader-based rendering for the sequence grid.

Reads from SequenceTensor (as texture) and renders the visual grid.
Can emit events back to CPU when playhead crosses cells.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


# =============================================================================
# SHADER SOURCES
# =============================================================================

GRID_VERTEX_SHADER = """
#version 330 core

in vec2 in_pos;
in vec2 in_uv;

out vec2 v_uv;

uniform vec2 u_resolution;    // Window size
uniform vec2 u_offset;        // Panel offset in pixels
uniform vec2 u_size;          // Panel size in pixels

void main() {
    // Transform to panel coordinates
    vec2 px = u_offset + in_pos * u_size;
    
    // To NDC
    vec2 ndc = (px / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for screen coords
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
"""

GRID_FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

// Grid state texture (RGBA = active, velocity, timing, subdivision)
uniform sampler2D u_grid_state;
uniform vec2 u_grid_size;      // (cols, rows)

// Playhead
uniform float u_playhead;      // Current beat position
uniform float u_playhead_width; // Width in beats

// Visual settings
uniform vec4 u_color_inactive;
uniform vec4 u_color_active;
uniform vec4 u_color_playing;
uniform vec4 u_grid_line_color;
uniform float u_grid_line_width;
uniform float u_cell_padding;

// Row colors (optional)
uniform vec4 u_row_colors[8];
uniform int u_use_row_colors;

// Cell we're in
vec2 get_cell() {
    return floor(v_uv * u_grid_size);
}

// Position within cell (0-1)
vec2 get_cell_uv() {
    return fract(v_uv * u_grid_size);
}

// Sample grid state at cell
vec4 sample_state(vec2 cell) {
    vec2 uv = (cell + 0.5) / u_grid_size;
    return texture(u_grid_state, uv);
}

// Is this cell under the playhead?
bool is_playing(vec2 cell) {
    float beat = cell.x;
    return abs(beat - u_playhead) < u_playhead_width;
}

// Draw a rounded rectangle (SDF)
float rounded_rect(vec2 uv, vec2 half_size, float radius) {
    vec2 d = abs(uv - 0.5) - half_size + radius;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0) - radius;
}

void main() {
    vec2 cell = get_cell();
    vec2 cell_uv = get_cell_uv();
    int row = int(cell.y);
    int col = int(cell.x);
    
    // Bounds check
    if (cell.x < 0.0 || cell.x >= u_grid_size.x || 
        cell.y < 0.0 || cell.y >= u_grid_size.y) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Sample state
    vec4 state = sample_state(cell);
    float active = state.r;
    float velocity = state.g;
    float timing = state.b;
    float subdivision = state.a;
    
    // Base color (background)
    vec4 color = u_color_inactive;
    
    // Alternate row shading
    if (mod(cell.y, 2.0) < 1.0) {
        color.rgb *= 0.95;
    }
    
    // Grid lines
    float line_x = step(cell_uv.x, u_grid_line_width) + step(1.0 - u_grid_line_width, cell_uv.x);
    float line_y = step(cell_uv.y, u_grid_line_width) + step(1.0 - u_grid_line_width, cell_uv.y);
    float grid_line = max(line_x, line_y);
    
    // Bar lines (every 4 beats) - thicker
    float bar_line = 0.0;
    if (mod(cell.x, 4.0) < 0.01) {
        bar_line = step(cell_uv.x, u_grid_line_width * 3.0);
    }
    
    // Cell content area (with padding)
    float pad = u_cell_padding;
    vec2 content_uv = (cell_uv - pad) / (1.0 - 2.0 * pad);
    bool in_content = all(greaterThan(content_uv, vec2(0.0))) && 
                      all(lessThan(content_uv, vec2(1.0)));
    
    if (active > 0.5 && in_content) {
        // Active cell - draw pad
        float rect = rounded_rect(content_uv, vec2(0.42), 0.12);
        
        if (rect < 0.0) {
            // Inside the pad
            vec4 pad_color;
            
            // Use row color if enabled
            if (u_use_row_colors > 0 && row < 8) {
                pad_color = u_row_colors[row];
            } else {
                pad_color = u_color_active;
            }
            
            // Velocity affects brightness
            pad_color.rgb *= 0.4 + velocity * 0.6;
            
            // Timing offset visualization (subtle hue shift)
            if (abs(timing) > 0.05) {
                pad_color.r += timing * 0.1;
                pad_color.b -= timing * 0.1;
            }
            
            // Playing highlight
            if (is_playing(cell)) {
                pad_color = mix(pad_color, u_color_playing, 0.5);
                // Pulse effect
                pad_color.rgb *= 1.0 + sin(u_playhead * 6.28) * 0.1;
            }
            
            // Edge highlight
            float edge = smoothstep(-0.02, 0.0, rect);
            pad_color.rgb = mix(pad_color.rgb * 1.3, pad_color.rgb, edge);
            
            color = pad_color;
        }
    }
    
    // Draw subdivision grid if subdivision > 1
    if (subdivision > 1.5 && in_content) {
        int sub_level = int(subdivision);
        float sub_x = content_uv.x * float(sub_level);
        float sub_uv = fract(sub_x);
        
        // Sub-cell separator lines
        if (sub_uv < 0.08 && sub_x > 0.5) {
            color = mix(color, u_grid_line_color, 0.4);
        }
    }
    
    // Playhead column highlight (subtle)
    if (abs(cell.x - floor(u_playhead)) < 0.5) {
        color.rgb *= 1.1;
    }
    
    // Apply grid lines
    color = mix(color, u_grid_line_color, grid_line * 0.3);
    color = mix(color, u_grid_line_color * 1.5, bar_line * 0.6);
    
    // Playhead line (bright vertical line)
    float playhead_x = fract(u_playhead) * (1.0 - 2.0 * u_grid_line_width) + u_grid_line_width;
    float in_playhead_col = step(abs(cell.x - floor(u_playhead)), 0.5);
    float playhead_dist = abs(cell_uv.x - playhead_x);
    float playhead_line = 1.0 - smoothstep(0.0, 0.03, playhead_dist);
    
    if (in_playhead_col > 0.5) {
        color = mix(color, u_color_playing, playhead_line * 0.9);
    }
    
    fragColor = color;
}
"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GridColors:
    """Color scheme for grid rendering."""
    inactive: Tuple[float, float, float, float] = (0.10, 0.11, 0.13, 1.0)
    active: Tuple[float, float, float, float] = (0.9, 0.5, 0.2, 1.0)
    playing: Tuple[float, float, float, float] = (1.0, 0.85, 0.3, 1.0)
    grid_line: Tuple[float, float, float, float] = (0.22, 0.23, 0.27, 1.0)
    
    # Per-row colors (for drum machine style)
    row_colors: List[Tuple[float, float, float, float]] = field(default_factory=lambda: [
        (1.0, 0.3, 0.2, 1.0),   # Row 0: Kick (red-orange)
        (1.0, 0.6, 0.2, 1.0),   # Row 1: Snare (orange)
        (1.0, 0.9, 0.2, 1.0),   # Row 2: Hi-hat closed (yellow)
        (0.4, 1.0, 0.3, 1.0),   # Row 3: Hi-hat open (green)
        (0.3, 0.8, 1.0, 1.0),   # Row 4: Tom high (cyan)
        (0.4, 0.5, 1.0, 1.0),   # Row 5: Tom mid (blue)
        (0.7, 0.4, 1.0, 1.0),   # Row 6: Tom low (purple)
        (1.0, 0.4, 0.8, 1.0),   # Row 7: Clap/rim (pink)
    ])
    
    use_row_colors: bool = True


@dataclass
class GridEvent:
    """Event emitted when playhead crosses an active cell."""
    type: int          # 1 = trigger
    row: int
    col: int
    velocity: float
    subdivision: int
    timestamp: float = 0.0


# =============================================================================
# RENDERER CLASS
# =============================================================================

class GridRenderer:
    """
    Renders the sequence grid using shaders.
    
    The shader reads from a texture containing the grid state (RGBA channels)
    and renders a visual drum machine grid. Events are detected on the CPU
    when the playhead crosses active cells.
    
    Usage:
        renderer = GridRenderer(ctx, grid)
        
        # Each frame:
        renderer.set_playhead(transport.playhead_beat)
        renderer.render(x, y, width, height, window_size)
        
        # Check for triggered events:
        events = renderer.poll_events()
        for event in events:
            audio.trigger(event.row, event.velocity)
    """
    
    def __init__(
        self,
        ctx,  # ModernGL context
        grid,  # SequenceTensor
        colors: Optional[GridColors] = None
    ):
        self.ctx = ctx
        self.grid = grid
        self.colors = colors or GridColors()
        
        # Compile shader program
        self._program = ctx.program(
            vertex_shader=GRID_VERTEX_SHADER,
            fragment_shader=GRID_FRAGMENT_SHADER
        )
        
        # Create fullscreen quad geometry
        vertices = np.array([
            # pos       uv
            0.0, 0.0,   0.0, 1.0,
            1.0, 0.0,   1.0, 1.0,
            1.0, 1.0,   1.0, 0.0,
            0.0, 0.0,   0.0, 1.0,
            1.0, 1.0,   1.0, 0.0,
            0.0, 1.0,   0.0, 0.0,
        ], dtype='f4')
        
        self._vbo = ctx.buffer(vertices)
        self._vao = ctx.vertex_array(
            self._program,
            [(self._vbo, '2f 2f', 'in_pos', 'in_uv')]
        )
        
        # Create grid state texture
        self._texture = ctx.texture(
            size=(grid.cols, grid.rows),
            components=4,
            dtype='f4'
        )
        self._texture.filter = (ctx.NEAREST, ctx.NEAREST)
        
        # Playhead state
        self._playhead = 0.0
        self._last_playhead = 0.0
        self._timestamp = 0.0
        
        # Pending events (triggered cells)
        self._pending_events: List[GridEvent] = []
        
        # Initial upload
        self._upload_grid()
    
    def _upload_grid(self):
        """Upload grid state to GPU texture."""
        data = self.grid.to_rgba_bytes()
        self._texture.write(data)
        self.grid.clear_dirty()
    
    def set_playhead(self, beat: float, timestamp: float = 0.0):
        """
        Update playhead position and detect crossings.
        
        Args:
            beat: Current playhead position in beats
            timestamp: Current time (for event timestamps)
        """
        self._last_playhead = self._playhead
        self._playhead = beat
        self._timestamp = timestamp
        
        # Detect events (CPU-side)
        self._detect_crossings()
    
    def _detect_crossings(self):
        """Detect cells that the playhead crossed since last update."""
        last = self._last_playhead
        current = self._playhead
        
        if last == current:
            return
        
        cols_to_check = []
        
        if current < last:
            # Wrapped around - check end of grid then start
            for col in range(int(last) + 1, self.grid.cols):
                cols_to_check.append(col)
            for col in range(0, int(current) + 1):
                cols_to_check.append(col)
        else:
            # Normal forward motion
            for col in range(int(last) + 1, int(current) + 1):
                cols_to_check.append(col)
            
            # Also check current column if we just entered it
            if int(current) != int(last):
                if int(current) not in cols_to_check:
                    cols_to_check.append(int(current))
        
        # Check each column for active cells
        for col in cols_to_check:
            if 0 <= col < self.grid.cols:
                self._check_column(col)
    
    def _check_column(self, col: int):
        """Check a column for active cells and create trigger events."""
        for row in range(self.grid.rows):
            if self.grid.is_active(row, col):
                velocity = self.grid.get_velocity(row, col)
                subdiv = self.grid.get_subdivision_level(row, col)
                
                event = GridEvent(
                    type=1,  # TRIGGER
                    row=row,
                    col=col,
                    velocity=velocity,
                    subdivision=subdiv,
                    timestamp=self._timestamp
                )
                self._pending_events.append(event)
    
    def poll_events(self) -> List[GridEvent]:
        """Get and clear pending trigger events."""
        events = self._pending_events
        self._pending_events = []
        return events
    
    def render(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        window_size: Tuple[float, float]
    ):
        """
        Render the grid.
        
        Args:
            x, y: Position in window (pixels)
            width, height: Size in pixels
            window_size: (window_width, window_height)
        """
        # Upload grid state if dirty
        if self.grid.is_dirty():
            self._upload_grid()
        
        # Bind texture
        self._texture.use(0)
        
        # Set uniforms
        prog = self._program
        prog['u_grid_state'] = 0
        prog['u_resolution'] = window_size
        prog['u_offset'] = (x, y)
        prog['u_size'] = (width, height)
        prog['u_grid_size'] = (float(self.grid.cols), float(self.grid.rows))
        prog['u_playhead'] = self._playhead
        prog['u_playhead_width'] = 0.5
        
        # Colors
        prog['u_color_inactive'] = self.colors.inactive
        prog['u_color_active'] = self.colors.active
        prog['u_color_playing'] = self.colors.playing
        prog['u_grid_line_color'] = self.colors.grid_line
        prog['u_grid_line_width'] = 0.015
        prog['u_cell_padding'] = 0.06
        
        # Row colors
        prog['u_use_row_colors'] = 1 if self.colors.use_row_colors else 0
        if self.colors.use_row_colors:
            for i, color in enumerate(self.colors.row_colors[:8]):
                prog[f'u_row_colors[{i}]'] = color
        
        # Draw
        self._vao.render()
    
    def resize_grid(self, rows: int, cols: int):
        """Handle grid dimension change."""
        self._texture.release()
        self._texture = self.ctx.texture(
            size=(cols, rows),
            components=4,
            dtype='f4'
        )
        self._texture.filter = (self.ctx.NEAREST, self.ctx.NEAREST)
    
    def release(self):
        """Release GPU resources."""
        self._vao.release()
        self._vbo.release()
        self._texture.release()

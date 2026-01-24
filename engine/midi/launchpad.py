"""
LaunchpadController - Grid controller mapping and LED feedback for Launchpad devices.

Supports Launchpad Mini, X, MK3, and classic models.
"""

from __future__ import annotations
from typing import Optional, Callable, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from .manager import MidiManager
    from ..time.transport import Transport
    from ..core.signal import SignalBridge


# =============================================================================
# Launchpad Color Palette
# =============================================================================

class LaunchpadColor:
    """Common Launchpad velocity colors (Launchpad Mini MK3 / X)."""
    OFF = 0
    WHITE = 3
    RED = 5
    RED_DIM = 7
    ORANGE = 9
    YELLOW = 13
    LIME = 17
    GREEN = 21
    GREEN_DIM = 23
    CYAN = 37
    BLUE = 45
    BLUE_DIM = 47
    PURPLE = 53
    MAGENTA = 57
    PINK = 57


# =============================================================================
# Grid Mapper
# =============================================================================

@dataclass
class PadEvent:
    """A pad press/release event."""
    row: int
    col: int
    velocity: int
    is_side_button: bool = False

    @property
    def is_pressed(self) -> bool:
        return self.velocity > 0


class LaunchpadMapper:
    """
    Maps Launchpad MIDI notes to grid (row, col) positions.

    Launchpad layout varies by model:
    - Mini/X/MK3: rows of 10 (notes 11-18 = row 1, cols 0-7)
    - Classic: rows of 16

    The 8x8 main grid uses rows 1-8, cols 0-7.
    Row 0 is the top control row (CC messages on some models).
    Column 8 is side buttons.
    """

    def __init__(self, model: str = "mini"):
        self.model = model
        self.rows = 8
        self.cols = 8

        # Model-specific note mapping
        if model in ("mini", "x", "mk3"):
            self._row_offset = 10  # Modern Launchpads
            self._first_row = 1   # Notes start at 11
        else:
            self._row_offset = 16  # Classic Launchpad
            self._first_row = 0

    def note_to_grid(self, note: int) -> Tuple[int, int, bool]:
        """
        Convert MIDI note to (row, col, is_side_button).

        Returns:
            (row, col, is_side_button) where row/col are 0-7 for main grid
        """
        raw_row = note // self._row_offset
        raw_col = note % self._row_offset

        # Adjust for first row offset
        grid_row = raw_row - self._first_row

        # Side buttons are column 8
        is_side = raw_col == 8

        # Clamp to valid grid
        if raw_col > 7:
            raw_col = 8  # Mark as side button column

        return (grid_row, raw_col, is_side)

    def grid_to_note(self, row: int, col: int) -> int:
        """Convert (row, col) to MIDI note for LED control."""
        actual_row = row + self._first_row
        return actual_row * self._row_offset + col


# =============================================================================
# Launchpad Controller
# =============================================================================

class LaunchpadController:
    """
    Bidirectional Launchpad controller with LED feedback.

    Subscribes to MIDI note events, converts to grid coordinates,
    and emits SIGNAL_MIDI_PAD. Also provides LED control methods.

    Usage:
        launchpad = LaunchpadController(midi_manager, signals)
        launchpad.bind_transport(transport)  # For play/stop button mapping
        launchpad.on_pad(lambda row, col, vel: print(f"Pad {row},{col}"))
    """

    def __init__(self, midi: MidiManager, signals: SignalBridge, model: str = "mini"):
        from ..core.signal import SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_PAD

        self.midi = midi
        self.signals = signals
        self.mapper = LaunchpadMapper(model)
        self._transport: Optional[Transport] = None

        # Callbacks
        self._on_pad_callbacks: List[Callable[[int, int, int], None]] = []

        # Grid state for LED tracking
        self._grid_colors = [[0] * 8 for _ in range(8)]

        # Subscribe to MIDI notes
        signals.connect(SIGNAL_MIDI_NOTE_ON, self._handle_note_on)

        # Store signal constant
        self._SIG_PAD = SIGNAL_MIDI_PAD

    def bind_transport(self, transport: Transport):
        """Bind to transport for play/stop button feedback."""
        self._transport = transport
        transport.on_play_callbacks.append(self._on_transport_play)
        transport.on_stop_callbacks.append(self._on_transport_stop)
        transport.on_beat_callbacks.append(self._on_beat)

    def _handle_note_on(self, note: int, velocity: int, channel: int):
        """Handle incoming MIDI note and convert to grid event."""
        row, col, is_side = self.mapper.note_to_grid(note)

        # Validate grid bounds
        if not (0 <= row < 8 and 0 <= col < 8):
            return

        # Emit grid signal
        self.signals.emit(self._SIG_PAD, row, col, velocity)

        # Call registered callbacks
        for cb in self._on_pad_callbacks:
            cb(row, col, velocity)

        # Handle transport control (row 0 by default)
        if row == 0 and velocity > 0 and self._transport:
            self._handle_transport_button(col)

    def _handle_transport_button(self, col: int):
        """Map top row buttons to transport controls."""
        if not self._transport:
            return

        if col == 0:
            # Play/Pause toggle
            self._transport.toggle()
        elif col == 1:
            # Stop
            self._transport.stop()
        elif col == 2:
            # Rewind to start
            self._transport.seek_to_beat(0)
        elif col in (3, 4, 5, 6, 7):
            # Jump to bar (col-2)
            bar = col - 2
            beat = bar * self._transport.time_signature[0]
            self._transport.seek_to_beat(beat)

    def _on_transport_play(self):
        """Light play button when playing."""
        self.set_pad_color(0, 0, LaunchpadColor.GREEN)

    def _on_transport_stop(self):
        """Dim play button when stopped."""
        self.set_pad_color(0, 0, LaunchpadColor.GREEN_DIM)

    def _on_beat(self, beat: int):
        """Flash beat indicator on transport row."""
        col = beat % 8
        if col >= 3:  # Bar position indicators
            # Brief flash (would need async handling for proper pulse)
            pass

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_pad(self, callback: Callable[[int, int, int], None]):
        """
        Register pad callback: fn(row, col, velocity).

        Row 0 is bottom, row 7 is top.
        Col 0 is left, col 7 is right.
        """
        self._on_pad_callbacks.append(callback)

    # =========================================================================
    # LED Control
    # =========================================================================

    def set_pad_color(self, row: int, col: int, color: int):
        """Set a single pad LED color."""
        note = self.mapper.grid_to_note(row, col)
        self.midi.send_note_on(note, color)
        if 0 <= row < 8 and 0 <= col < 8:
            self._grid_colors[row][col] = color

    def set_row_color(self, row: int, color: int):
        """Set all pads in a row to the same color."""
        for col in range(8):
            self.set_pad_color(row, col, color)

    def set_column_color(self, col: int, color: int):
        """Set all pads in a column to the same color."""
        for row in range(8):
            self.set_pad_color(row, col, color)

    def clear_all(self):
        """Turn off all pad LEDs."""
        for row in range(8):
            for col in range(8):
                self.set_pad_color(row, col, LaunchpadColor.OFF)

    def set_grid_colors(self, colors: List[List[int]]):
        """Set entire grid at once. colors[row][col] = velocity color."""
        for row in range(min(8, len(colors))):
            for col in range(min(8, len(colors[row]))):
                self.set_pad_color(row, col, colors[row][col])

    # =========================================================================
    # Patterns
    # =========================================================================

    def show_startup_pattern(self):
        """Display a startup animation pattern."""
        # Simple diagonal pattern
        for i in range(8):
            self.set_pad_color(i, i, LaunchpadColor.CYAN)
            if i > 0:
                self.set_pad_color(i - 1, i - 1, LaunchpadColor.BLUE_DIM)

    def show_transport_row(self):
        """Light up transport control row with appropriate colors."""
        # Play - green
        self.set_pad_color(0, 0, LaunchpadColor.GREEN_DIM)
        # Stop - red
        self.set_pad_color(0, 1, LaunchpadColor.RED_DIM)
        # Rewind - yellow
        self.set_pad_color(0, 2, LaunchpadColor.YELLOW)
        # Bar markers - white dim
        for col in range(3, 8):
            self.set_pad_color(0, col, LaunchpadColor.WHITE)

"""
MidiRecorder - Records MIDI input with beat-accurate timestamps.

Captures Launchpad pads and other MIDI notes with timing relative to Transport.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..time.transport import Transport
    from ..core.signal import SignalBridge


@dataclass
class RecordedNote:
    """A MIDI note captured during recording."""
    beat: float              # When it was played (in beats)
    note: int                # MIDI note number
    velocity: int            # 0-127
    channel: int = 0
    duration: float = 0.25   # Default duration (set on note-off)

    # Launchpad grid info (if from pad)
    row: int = -1            # -1 if not a pad
    col: int = -1
    is_pad: bool = False

    # Visual info
    color: tuple = (1.0, 0.5, 0.0, 1.0)  # Default orange

    def __post_init__(self):
        # Clamp velocity
        self.velocity = max(0, min(127, self.velocity))


# Color palette for Launchpad rows
PAD_ROW_COLORS = [
    (1.0, 0.3, 0.2, 1.0),   # Row 0: Red
    (1.0, 0.6, 0.2, 1.0),   # Row 1: Orange
    (1.0, 0.9, 0.2, 1.0),   # Row 2: Yellow
    (0.4, 1.0, 0.3, 1.0),   # Row 3: Green
    (0.2, 0.9, 0.9, 1.0),   # Row 4: Cyan
    (0.3, 0.5, 1.0, 1.0),   # Row 5: Blue
    (0.7, 0.3, 1.0, 1.0),   # Row 6: Purple
    (1.0, 0.4, 0.8, 1.0),   # Row 7: Pink
]


class MidiRecorder:
    """
    Records MIDI input with beat-accurate timestamps.

    Subscribes to SignalBridge for MIDI events and records notes
    with timestamps from Transport.

    Usage:
        recorder = MidiRecorder(transport, signals)
        recorder.start()
        # ... user plays Launchpad ...
        recorder.stop()
        for note in recorder.notes:
            print(f"Beat {note.beat}: Note {note.note}")
    """

    def __init__(self, transport: Transport, signals: SignalBridge):
        from ..core.signal import (
            SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_NOTE_OFF, SIGNAL_MIDI_PAD
        )

        self.transport = transport
        self.signals = signals

        self._notes: List[RecordedNote] = []
        self._active_notes: Dict[int, RecordedNote] = {}  # note -> RecordedNote
        self._recording = False

        # Callbacks for real-time notification
        self._on_note_callbacks: List[Callable[[RecordedNote], None]] = []

        # Subscribe to MIDI signals
        signals.connect(SIGNAL_MIDI_NOTE_ON, self._on_note_on)
        signals.connect(SIGNAL_MIDI_NOTE_OFF, self._on_note_off)
        signals.connect(SIGNAL_MIDI_PAD, self._on_pad)

        # Store signal names for reference
        self._SIG_NOTE_ON = SIGNAL_MIDI_NOTE_ON
        self._SIG_NOTE_OFF = SIGNAL_MIDI_NOTE_OFF
        self._SIG_PAD = SIGNAL_MIDI_PAD

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def notes(self) -> List[RecordedNote]:
        return self._notes

    def start(self):
        """Start recording MIDI input."""
        self._recording = True
        print("MidiRecorder: Started recording")

    def stop(self):
        """Stop recording."""
        self._recording = False

        # Close any active notes
        for note, recorded in self._active_notes.items():
            recorded.duration = self.transport.playhead_beat - recorded.beat
            recorded.duration = max(0.1, recorded.duration)  # Minimum duration

        self._active_notes.clear()
        print(f"MidiRecorder: Stopped ({len(self._notes)} notes recorded)")

    def clear(self):
        """Clear all recorded notes."""
        self._notes.clear()
        self._active_notes.clear()

    def on_note(self, callback: Callable[[RecordedNote], None]):
        """Register callback for real-time note notification."""
        self._on_note_callbacks.append(callback)

    def _on_note_on(self, note: int, velocity: int, channel: int):
        """Handle note-on from SignalBridge."""
        if not self._recording or velocity == 0:
            return

        beat = self.transport.playhead_beat

        recorded = RecordedNote(
            beat=beat,
            note=note,
            velocity=velocity,
            channel=channel,
            is_pad=False,
            color=(0.5, 0.5, 0.5, 1.0),  # Gray for non-pad notes
        )

        self._notes.append(recorded)
        self._active_notes[note] = recorded

        # Notify callbacks
        for cb in self._on_note_callbacks:
            cb(recorded)

    def _on_note_off(self, note: int, velocity: int, channel: int):
        """Handle note-off - set duration of note."""
        if note in self._active_notes:
            recorded = self._active_notes.pop(note)
            recorded.duration = max(0.1, self.transport.playhead_beat - recorded.beat)

    def _on_pad(self, row: int, col: int, velocity: int):
        """Handle Launchpad pad press."""
        if not self._recording or velocity == 0:
            return

        beat = self.transport.playhead_beat

        # Get color for this row
        color = PAD_ROW_COLORS[row % len(PAD_ROW_COLORS)]

        # Encode row/col as note number for tracking
        pad_note = row * 10 + col

        recorded = RecordedNote(
            beat=beat,
            note=pad_note,
            velocity=velocity,
            channel=0,
            row=row,
            col=col,
            is_pad=True,
            color=color,
            duration=0.25,  # Pads get fixed short duration
        )

        self._notes.append(recorded)

        # Notify callbacks
        for cb in self._on_note_callbacks:
            cb(recorded)

    def get_notes_in_range(self, start_beat: float, end_beat: float) -> List[RecordedNote]:
        """Get notes within a beat range (for display)."""
        return [
            n for n in self._notes
            if start_beat <= n.beat < end_beat
        ]

    def get_notes_since(self, beat: float) -> List[RecordedNote]:
        """Get all notes after a given beat."""
        return [n for n in self._notes if n.beat >= beat]

    def get_pad_notes(self) -> List[RecordedNote]:
        """Get only Launchpad pad notes."""
        return [n for n in self._notes if n.is_pad]

    def quantize(self, subdivision: int = 4):
        """Quantize all notes to grid."""
        grid_size = 1.0 / subdivision
        for note in self._notes:
            note.beat = round(note.beat / grid_size) * grid_size

    def simulate_pad(self, row: int, col: int, velocity: int = 100):
        """Simulate a pad press (for testing without hardware)."""
        if self._recording:
            self._on_pad(row, col, velocity)

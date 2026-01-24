"""
SPECTRO MIDI Module
===================

MIDI input/output handling with Launchpad grid controller support.

Quick Start:
    from engine.core.signal import SignalBridge, SIGNAL_MIDI_NOTE_ON
    from engine.midi import MidiManager, LaunchpadController

    signals = SignalBridge()
    midi = MidiManager(signals)

    # List available devices
    print(midi.list_input_ports())

    # Connect to Launchpad (partial name match)
    midi.connect("Launchpad")

    # Set up grid controller
    launchpad = LaunchpadController(midi, signals)
    launchpad.on_pad(lambda row, col, vel: print(f"Pad ({row}, {col}) vel={vel}"))

Dependencies:
    pip install python-rtmidi
"""

from .manager import MidiManager, MidiMessage, MidiMessageType, RTMIDI_AVAILABLE
from .launchpad import LaunchpadController, LaunchpadMapper, LaunchpadColor, PadEvent
from .recorder import MidiRecorder, RecordedNote, PAD_ROW_COLORS

__all__ = [
    'MidiManager',
    'MidiMessage',
    'MidiMessageType',
    'LaunchpadController',
    'LaunchpadMapper',
    'LaunchpadColor',
    'PadEvent',
    'MidiRecorder',
    'RecordedNote',
    'PAD_ROW_COLORS',
    'RTMIDI_AVAILABLE',
]

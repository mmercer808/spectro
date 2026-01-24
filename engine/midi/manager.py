"""
MidiManager - Central MIDI coordinator for SPECTRO.

Owns MIDI input/output lifecycle and routes events through SignalBridge.
"""

from __future__ import annotations
from typing import Optional, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
import threading
import time

if TYPE_CHECKING:
    from ..core.signal import SignalBridge

try:
    import rtmidi
    RTMIDI_AVAILABLE = True
except ImportError:
    RTMIDI_AVAILABLE = False


# =============================================================================
# MIDI Message Types
# =============================================================================

class MidiMessageType:
    NOTE_OFF = 0x80
    NOTE_ON = 0x90
    POLY_AFTERTOUCH = 0xA0
    CONTROL_CHANGE = 0xB0
    PROGRAM_CHANGE = 0xC0
    CHANNEL_AFTERTOUCH = 0xD0
    PITCH_BEND = 0xE0
    SYSEX = 0xF0
    CLOCK = 0xF8
    START = 0xFA
    CONTINUE = 0xFB
    STOP = 0xFC


@dataclass
class MidiMessage:
    """Parsed MIDI message."""
    msg_type: int
    channel: int
    data1: int
    data2: int
    timestamp: float

    @property
    def note(self) -> int:
        return self.data1

    @property
    def velocity(self) -> int:
        return self.data2

    @property
    def cc_number(self) -> int:
        return self.data1

    @property
    def cc_value(self) -> int:
        return self.data2

    def is_note_on(self) -> bool:
        return self.msg_type == MidiMessageType.NOTE_ON and self.velocity > 0

    def is_note_off(self) -> bool:
        return (self.msg_type == MidiMessageType.NOTE_OFF or
                (self.msg_type == MidiMessageType.NOTE_ON and self.velocity == 0))


# =============================================================================
# MIDI Manager
# =============================================================================

class MidiManager:
    """
    Central MIDI coordinator that routes events through SignalBridge.

    Usage:
        from engine.core.signal import SignalBridge, SIGNAL_MIDI_NOTE_ON
        from engine.midi import MidiManager

        signals = SignalBridge()
        signals.connect(SIGNAL_MIDI_NOTE_ON, lambda n, v, c: print(f"Note {n}"))

        midi = MidiManager(signals)
        midi.connect("Launchpad")  # Partial name match
    """

    def __init__(self, signals: SignalBridge):
        from ..core.signal import (
            SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_NOTE_OFF,
            SIGNAL_MIDI_CC, SIGNAL_MIDI_CLOCK,
            SIGNAL_MIDI_CONNECTED, SIGNAL_MIDI_DISCONNECTED
        )

        self.signals = signals
        self._midi_in: Optional[rtmidi.MidiIn] = None
        self._midi_out: Optional[rtmidi.MidiOut] = None
        self._port_name: Optional[str] = None
        self._running = False

        # Signal constants for internal use
        self._SIG_NOTE_ON = SIGNAL_MIDI_NOTE_ON
        self._SIG_NOTE_OFF = SIGNAL_MIDI_NOTE_OFF
        self._SIG_CC = SIGNAL_MIDI_CC
        self._SIG_CLOCK = SIGNAL_MIDI_CLOCK
        self._SIG_CONNECTED = SIGNAL_MIDI_CONNECTED
        self._SIG_DISCONNECTED = SIGNAL_MIDI_DISCONNECTED

        # Initialize rtmidi if available
        if RTMIDI_AVAILABLE:
            self._midi_in = rtmidi.MidiIn()
            self._midi_out = rtmidi.MidiOut()

    @property
    def is_available(self) -> bool:
        """Check if MIDI is available."""
        return RTMIDI_AVAILABLE

    @property
    def is_connected(self) -> bool:
        """Check if a MIDI device is connected."""
        return self._port_name is not None

    @property
    def port_name(self) -> Optional[str]:
        """Get the name of the connected port."""
        return self._port_name

    def list_input_ports(self) -> List[str]:
        """List available MIDI input ports."""
        if not RTMIDI_AVAILABLE or not self._midi_in:
            return []
        ports = []
        for i in range(self._midi_in.get_port_count()):
            ports.append(self._midi_in.get_port_name(i))
        return ports

    def list_output_ports(self) -> List[str]:
        """List available MIDI output ports."""
        if not RTMIDI_AVAILABLE or not self._midi_out:
            return []
        ports = []
        for i in range(self._midi_out.get_port_count()):
            ports.append(self._midi_out.get_port_name(i))
        return ports

    def connect(self, port_name: str) -> bool:
        """
        Connect to MIDI device by name (partial match).

        Args:
            port_name: Full or partial name to match against available ports

        Returns:
            True if connection successful
        """
        if not RTMIDI_AVAILABLE:
            print("MidiManager: python-rtmidi not installed")
            return False

        # Find matching input port
        in_ports = self.list_input_ports()
        in_port_idx = None
        for i, name in enumerate(in_ports):
            if port_name.lower() in name.lower():
                in_port_idx = i
                break

        if in_port_idx is None:
            print(f"MidiManager: No input port matching '{port_name}'")
            return False

        # Find matching output port
        out_ports = self.list_output_ports()
        out_port_idx = None
        for i, name in enumerate(out_ports):
            if port_name.lower() in name.lower():
                out_port_idx = i
                break

        # Close existing connections
        self.disconnect()

        # Open input
        try:
            self._midi_in.open_port(in_port_idx)
            self._port_name = in_ports[in_port_idx]
            self._midi_in.set_callback(self._midi_callback)
            self._running = True
            print(f"MidiManager: Opened input '{self._port_name}'")
        except Exception as e:
            print(f"MidiManager: Error opening input: {e}")
            return False

        # Open output if found
        if out_port_idx is not None:
            try:
                self._midi_out.open_port(out_port_idx)
                print(f"MidiManager: Opened output '{out_ports[out_port_idx]}'")
            except Exception as e:
                print(f"MidiManager: Warning - output port error: {e}")

        self.signals.emit(self._SIG_CONNECTED, self._port_name)
        return True

    def connect_by_index(self, input_port: int, output_port: int = -1) -> bool:
        """Connect to MIDI device by port index."""
        if not RTMIDI_AVAILABLE:
            return False

        self.disconnect()

        try:
            self._midi_in.open_port(input_port)
            self._port_name = self._midi_in.get_port_name(input_port)
            self._midi_in.set_callback(self._midi_callback)
            self._running = True
            print(f"MidiManager: Opened input '{self._port_name}'")
        except Exception as e:
            print(f"MidiManager: Error opening input port {input_port}: {e}")
            return False

        if output_port >= 0:
            try:
                self._midi_out.open_port(output_port)
            except Exception as e:
                print(f"MidiManager: Warning - output port error: {e}")

        self.signals.emit(self._SIG_CONNECTED, self._port_name)
        return True

    def disconnect(self):
        """Disconnect current MIDI device."""
        old_name = self._port_name
        self._running = False

        if self._midi_in and self._midi_in.is_port_open():
            self._midi_in.cancel_callback()
            self._midi_in.close_port()

        if self._midi_out and self._midi_out.is_port_open():
            self._midi_out.close_port()

        self._port_name = None

        if old_name:
            self.signals.emit(self._SIG_DISCONNECTED, old_name)

    def _midi_callback(self, event, data=None):
        """Called by rtmidi when a MIDI message arrives."""
        message_data, delta_time = event
        self._process_message(message_data, time.time())

    def _process_message(self, data: List[int], timestamp: float):
        """Parse and dispatch a MIDI message through SignalBridge."""
        if not data:
            return

        status = data[0]

        # Handle system realtime messages (no channel)
        if status == MidiMessageType.CLOCK:
            self.signals.emit(self._SIG_CLOCK)
            return
        if status in (MidiMessageType.START, MidiMessageType.CONTINUE, MidiMessageType.STOP):
            return

        msg_type = status & 0xF0
        channel = status & 0x0F
        data1 = data[1] if len(data) > 1 else 0
        data2 = data[2] if len(data) > 2 else 0

        msg = MidiMessage(
            msg_type=msg_type,
            channel=channel,
            data1=data1,
            data2=data2,
            timestamp=timestamp
        )

        # Dispatch through SignalBridge
        if msg.is_note_on():
            self.signals.emit(self._SIG_NOTE_ON, msg.note, msg.velocity, msg.channel)
        elif msg.is_note_off():
            self.signals.emit(self._SIG_NOTE_OFF, msg.note, msg.velocity, msg.channel)
        elif msg_type == MidiMessageType.CONTROL_CHANGE:
            self.signals.emit(self._SIG_CC, msg.cc_number, msg.cc_value, msg.channel)

    # =========================================================================
    # MIDI Output
    # =========================================================================

    def send_note_on(self, note: int, velocity: int = 127, channel: int = 0):
        """Send note-on message."""
        self._send([0x90 | channel, note, velocity])

    def send_note_off(self, note: int, velocity: int = 0, channel: int = 0):
        """Send note-off message."""
        self._send([0x80 | channel, note, velocity])

    def send_cc(self, cc: int, value: int, channel: int = 0):
        """Send control change message."""
        self._send([0xB0 | channel, cc, value])

    def _send(self, message: List[int]):
        """Send raw MIDI message."""
        if self._midi_out and self._midi_out.is_port_open():
            self._midi_out.send_message(message)

    # =========================================================================
    # Simulation (for testing without hardware)
    # =========================================================================

    def simulate_note_on(self, note: int, velocity: int = 127, channel: int = 0):
        """Simulate a note-on message (for testing)."""
        self.signals.emit(self._SIG_NOTE_ON, note, velocity, channel)

    def simulate_note_off(self, note: int, velocity: int = 0, channel: int = 0):
        """Simulate a note-off message (for testing)."""
        self.signals.emit(self._SIG_NOTE_OFF, note, velocity, channel)

    def simulate_cc(self, cc: int, value: int, channel: int = 0):
        """Simulate a CC message (for testing)."""
        self.signals.emit(self._SIG_CC, cc, value, channel)

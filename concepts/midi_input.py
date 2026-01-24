"""
DR BEAT MIDI INPUT
==================

Real MIDI hardware input using python-rtmidi.
Supports note on/off, CC, and Launchpad grid mapping.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
from enum import Enum, auto
import threading
import time

try:
    import rtmidi
    RTMIDI_AVAILABLE = True
except ImportError:
    RTMIDI_AVAILABLE = False
    print("Warning: python-rtmidi not available. MIDI will be simulated.")


# ============================================================================
# MIDI MESSAGE TYPES
# ============================================================================

class MidiMessageType(Enum):
    """MIDI message types"""
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
    """Parsed MIDI message"""
    msg_type: MidiMessageType
    channel: int
    data1: int  # Note number or CC number
    data2: int  # Velocity or CC value
    timestamp: float
    raw: bytes
    
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


# ============================================================================
# MIDI INPUT
# ============================================================================

class MidiInput:
    """
    Real MIDI input handler using python-rtmidi.
    
    Usage:
        midi = MidiInput()
        ports = midi.list_ports()
        midi.open_port(0)  # or midi.open_port_by_name("Launchpad")
        midi.on_note_on(lambda note, vel, ch: print(f"Note {note}"))
        midi.start()
    """
    
    def __init__(self):
        self.midi_in = None
        self.port_name: Optional[str] = None
        self.running = False
        
        # Callbacks
        self._on_note_on: List[Callable[[int, int, int], None]] = []
        self._on_note_off: List[Callable[[int, int, int], None]] = []
        self._on_cc: List[Callable[[int, int, int], None]] = []
        self._on_message: List[Callable[[MidiMessage], None]] = []
        
        # For polling mode
        self._poll_thread: Optional[threading.Thread] = None
        
        if RTMIDI_AVAILABLE:
            self.midi_in = rtmidi.MidiIn()
    
    def list_ports(self) -> List[str]:
        """List available MIDI input ports"""
        if not RTMIDI_AVAILABLE or not self.midi_in:
            return ["[Simulated MIDI]"]
        
        ports = []
        for i in range(self.midi_in.get_port_count()):
            ports.append(self.midi_in.get_port_name(i))
        return ports
    
    def open_port(self, port: int) -> bool:
        """Open a MIDI port by index"""
        if not RTMIDI_AVAILABLE or not self.midi_in:
            print(f"MidiInput: Simulated port {port}")
            self.port_name = f"Simulated Port {port}"
            return True
        
        try:
            if self.midi_in.is_port_open():
                self.midi_in.close_port()
            
            self.midi_in.open_port(port)
            self.port_name = self.midi_in.get_port_name(port)
            print(f"MidiInput: Opened '{self.port_name}'")
            return True
        except Exception as e:
            print(f"MidiInput: Error opening port {port}: {e}")
            return False
    
    def open_port_by_name(self, name: str) -> bool:
        """Open a MIDI port by name (partial match)"""
        ports = self.list_ports()
        for i, port_name in enumerate(ports):
            if name.lower() in port_name.lower():
                return self.open_port(i)
        print(f"MidiInput: No port matching '{name}' found")
        return False
    
    def open_virtual(self, name: str = "DR BEAT") -> bool:
        """Open a virtual MIDI port (for software routing)"""
        if not RTMIDI_AVAILABLE or not self.midi_in:
            print("MidiInput: Virtual ports not available")
            return False
        
        try:
            self.midi_in.open_virtual_port(name)
            self.port_name = name
            print(f"MidiInput: Opened virtual port '{name}'")
            return True
        except Exception as e:
            print(f"MidiInput: Error opening virtual port: {e}")
            return False
    
    def close(self):
        """Close the MIDI port"""
        self.running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=1.0)
            self._poll_thread = None
        
        if RTMIDI_AVAILABLE and self.midi_in and self.midi_in.is_port_open():
            self.midi_in.close_port()
        
        self.port_name = None
    
    def start(self, use_callback: bool = True):
        """Start receiving MIDI messages"""
        self.running = True
        
        if not RTMIDI_AVAILABLE:
            return
        
        if use_callback:
            # Use rtmidi callback
            self.midi_in.set_callback(self._midi_callback)
        else:
            # Use polling thread
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
    
    def stop(self):
        """Stop receiving MIDI messages"""
        self.running = False
        if RTMIDI_AVAILABLE and self.midi_in:
            self.midi_in.cancel_callback()
    
    def _midi_callback(self, event, data=None):
        """Called by rtmidi when a MIDI message arrives"""
        message_data, delta_time = event
        self._process_message(message_data, time.time())
    
    def _poll_loop(self):
        """Polling loop for MIDI messages"""
        while self.running:
            if self.midi_in:
                msg = self.midi_in.get_message()
                if msg:
                    message_data, delta_time = msg
                    self._process_message(message_data, time.time())
            time.sleep(0.001)  # 1ms poll interval
    
    def _process_message(self, data: List[int], timestamp: float):
        """Parse and dispatch a MIDI message"""
        if not data:
            return
        
        status = data[0]
        msg_type_value = status & 0xF0
        channel = status & 0x0F
        
        try:
            msg_type = MidiMessageType(msg_type_value)
        except ValueError:
            # Unknown message type
            return
        
        data1 = data[1] if len(data) > 1 else 0
        data2 = data[2] if len(data) > 2 else 0
        
        msg = MidiMessage(
            msg_type=msg_type,
            channel=channel,
            data1=data1,
            data2=data2,
            timestamp=timestamp,
            raw=bytes(data)
        )
        
        # Dispatch to specific callbacks
        if msg.is_note_on():
            for cb in self._on_note_on:
                cb(msg.note, msg.velocity, msg.channel)
        elif msg.is_note_off():
            for cb in self._on_note_off:
                cb(msg.note, msg.velocity, msg.channel)
        elif msg_type == MidiMessageType.CONTROL_CHANGE:
            for cb in self._on_cc:
                cb(msg.cc_number, msg.cc_value, msg.channel)
        
        # Dispatch to generic callbacks
        for cb in self._on_message:
            cb(msg)
    
    # Callback registration
    
    def on_note_on(self, callback: Callable[[int, int, int], None]):
        """Register note-on callback: fn(note, velocity, channel)"""
        self._on_note_on.append(callback)
    
    def on_note_off(self, callback: Callable[[int, int, int], None]):
        """Register note-off callback: fn(note, velocity, channel)"""
        self._on_note_off.append(callback)
    
    def on_cc(self, callback: Callable[[int, int, int], None]):
        """Register CC callback: fn(cc_number, cc_value, channel)"""
        self._on_cc.append(callback)
    
    def on_message(self, callback: Callable[[MidiMessage], None]):
        """Register generic message callback"""
        self._on_message.append(callback)
    
    # Simulated input (for testing without hardware)
    
    def simulate_note_on(self, note: int, velocity: int = 127, channel: int = 0):
        """Simulate a note-on message"""
        data = [0x90 | channel, note, velocity]
        self._process_message(data, time.time())
    
    def simulate_note_off(self, note: int, velocity: int = 0, channel: int = 0):
        """Simulate a note-off message"""
        data = [0x80 | channel, note, velocity]
        self._process_message(data, time.time())
    
    def simulate_cc(self, cc: int, value: int, channel: int = 0):
        """Simulate a CC message"""
        data = [0xB0 | channel, cc, value]
        self._process_message(data, time.time())


# ============================================================================
# LAUNCHPAD MAPPER
# ============================================================================

class LaunchpadMapper:
    """
    Maps Launchpad MIDI notes to grid positions.
    
    Launchpad layout:
    - Notes 0-7 are row 0 (bottom)
    - Notes 16-23 are row 1
    - etc. (10-based offset for Launchpad Mini/X)
    
    This mapper supports multiple Launchpad models.
    """
    
    def __init__(self, midi_input: MidiInput, model: str = "mini"):
        self.midi = midi_input
        self.model = model
        
        # Grid dimensions
        self.rows = 8
        self.cols = 8
        
        # Callbacks
        self._on_pad: List[Callable[[int, int, int], None]] = []
        
        # Wire up MIDI
        self.midi.on_note_on(self._handle_note)
        
        # Model-specific note mapping
        if model in ("mini", "x", "mk3"):
            # Modern Launchpads use rows of 10
            self._note_offset = 10
        else:
            # Classic Launchpad uses rows of 16
            self._note_offset = 16
    
    def note_to_grid(self, note: int) -> Tuple[int, int]:
        """Convert MIDI note to (row, col)"""
        row = note // self._note_offset
        col = note % self._note_offset
        if col > 7:
            col = -1  # Side buttons
        return (row, col)
    
    def grid_to_note(self, row: int, col: int) -> int:
        """Convert (row, col) to MIDI note"""
        return row * self._note_offset + col
    
    def _handle_note(self, note: int, velocity: int, channel: int):
        """Handle incoming note and convert to grid"""
        row, col = self.note_to_grid(note)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            for cb in self._on_pad:
                cb(row, col, velocity)
    
    def on_pad(self, callback: Callable[[int, int, int], None]):
        """Register pad callback: fn(row, col, velocity)"""
        self._on_pad.append(callback)


# ============================================================================
# MIDI OUTPUT (for lighting Launchpad, etc.)
# ============================================================================

class MidiOutput:
    """MIDI output for lighting Launchpad pads, etc."""
    
    def __init__(self):
        self.midi_out = None
        self.port_name: Optional[str] = None
        
        if RTMIDI_AVAILABLE:
            self.midi_out = rtmidi.MidiOut()
    
    def list_ports(self) -> List[str]:
        """List available MIDI output ports"""
        if not RTMIDI_AVAILABLE or not self.midi_out:
            return ["[Simulated MIDI]"]
        
        ports = []
        for i in range(self.midi_out.get_port_count()):
            ports.append(self.midi_out.get_port_name(i))
        return ports
    
    def open_port(self, port: int) -> bool:
        """Open a MIDI port by index"""
        if not RTMIDI_AVAILABLE or not self.midi_out:
            return True
        
        try:
            if self.midi_out.is_port_open():
                self.midi_out.close_port()
            
            self.midi_out.open_port(port)
            self.port_name = self.midi_out.get_port_name(port)
            print(f"MidiOutput: Opened '{self.port_name}'")
            return True
        except Exception as e:
            print(f"MidiOutput: Error opening port {port}: {e}")
            return False
    
    def open_port_by_name(self, name: str) -> bool:
        """Open a MIDI port by name (partial match)"""
        ports = self.list_ports()
        for i, port_name in enumerate(ports):
            if name.lower() in port_name.lower():
                return self.open_port(i)
        return False
    
    def close(self):
        """Close the MIDI port"""
        if RTMIDI_AVAILABLE and self.midi_out and self.midi_out.is_port_open():
            self.midi_out.close_port()
        self.port_name = None
    
    def send_note_on(self, note: int, velocity: int = 127, channel: int = 0):
        """Send note-on message"""
        self._send([0x90 | channel, note, velocity])
    
    def send_note_off(self, note: int, velocity: int = 0, channel: int = 0):
        """Send note-off message"""
        self._send([0x80 | channel, note, velocity])
    
    def send_cc(self, cc: int, value: int, channel: int = 0):
        """Send control change message"""
        self._send([0xB0 | channel, cc, value])
    
    def _send(self, message: List[int]):
        """Send raw MIDI message"""
        if RTMIDI_AVAILABLE and self.midi_out and self.midi_out.is_port_open():
            self.midi_out.send_message(message)
    
    def set_pad_color(self, row: int, col: int, color: int, model: str = "mini"):
        """Set Launchpad pad color"""
        if model in ("mini", "x", "mk3"):
            note = row * 10 + col
        else:
            note = row * 16 + col
        self.send_note_on(note, color)
    
    def clear_pads(self, model: str = "mini"):
        """Clear all pad colors"""
        for row in range(8):
            for col in range(8):
                self.set_pad_color(row, col, 0, model)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo MIDI input"""
    print("=" * 60)
    print("DR BEAT MIDI INPUT DEMO")
    print("=" * 60)
    
    midi = MidiInput()
    
    # List available ports
    ports = midi.list_ports()
    print("\nAvailable MIDI input ports:")
    for i, port in enumerate(ports):
        print(f"  {i}: {port}")
    
    if not ports:
        print("No MIDI ports available")
        return
    
    # Register callbacks
    def on_note(note, velocity, channel):
        print(f"Note ON: {note} (vel={velocity}, ch={channel})")
    
    def on_note_off(note, velocity, channel):
        print(f"Note OFF: {note} (ch={channel})")
    
    def on_cc(cc, value, channel):
        print(f"CC: {cc} = {value} (ch={channel})")
    
    midi.on_note_on(on_note)
    midi.on_note_off(on_note_off)
    midi.on_cc(on_cc)
    
    # Try to open first port
    if RTMIDI_AVAILABLE:
        if midi.open_port(0):
            midi.start()
            
            print("\nListening for MIDI... Press Ctrl+C to stop\n")
            
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nStopping...")
            
            midi.stop()
            midi.close()
    else:
        print("\nSimulating MIDI input...")
        
        # Simulate some notes
        for note in [36, 38, 42, 46]:
            midi.simulate_note_on(note, 100)
            time.sleep(0.2)
            midi.simulate_note_off(note)
            time.sleep(0.1)
    
    print("Done!")


if __name__ == "__main__":
    demo()


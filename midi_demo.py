"""
MIDI Demo - Test MIDI input with Launchpad support.

Run: python midi_demo.py
"""

import time
from engine.core.signal import SignalBridge, SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_PAD
from engine.midi import MidiManager, LaunchpadController, RTMIDI_AVAILABLE

def main():
    print("=" * 60)
    print("SPECTRO MIDI DEMO")
    print("=" * 60)

    if not RTMIDI_AVAILABLE:
        print("\nWARNING: python-rtmidi not installed")
        print("Install with: pip install python-rtmidi")
        print("\nRunning in simulation mode...\n")

    # Create signal bridge
    signals = SignalBridge()

    # Create MIDI manager
    midi = MidiManager(signals)

    # List available ports
    print("\nAvailable MIDI Input Ports:")
    ports = midi.list_input_ports()
    if not ports:
        print("  (none found)")
    for i, port in enumerate(ports):
        print(f"  {i}: {port}")

    print("\nAvailable MIDI Output Ports:")
    out_ports = midi.list_output_ports()
    if not out_ports:
        print("  (none found)")
    for i, port in enumerate(out_ports):
        print(f"  {i}: {port}")

    # Set up event handlers
    def on_note(note, velocity, channel):
        print(f"  Note ON: {note:3d}  vel={velocity:3d}  ch={channel}")

    def on_pad(row, col, velocity):
        print(f"  PAD: row={row} col={col} vel={velocity}")

    signals.connect(SIGNAL_MIDI_NOTE_ON, on_note)
    signals.connect(SIGNAL_MIDI_PAD, on_pad)

    # Try to connect to Launchpad
    connected = False
    for port in ports:
        if "launchpad" in port.lower():
            print(f"\nConnecting to: {port}")
            if midi.connect(port):
                connected = True
                break

    if not connected and ports:
        print(f"\nNo Launchpad found. Connecting to first port: {ports[0]}")
        midi.connect_by_index(0)
        connected = midi.is_connected

    # Set up Launchpad controller
    if connected:
        launchpad = LaunchpadController(midi, signals)
        launchpad.show_transport_row()
        print("\nLaunchpad transport row lit. Play a pad!")

    print("\n" + "-" * 60)
    print("Listening for MIDI... Press Ctrl+C to stop")
    print("-" * 60 + "\n")

    if not RTMIDI_AVAILABLE:
        # Simulate some input for testing
        print("Simulating MIDI input...\n")
        for note in [36, 38, 42, 46]:
            midi.simulate_note_on(note, 100, 0)
            time.sleep(0.2)
        print("\nSimulation complete.")
        return

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")

    midi.disconnect()
    print("Done!")


if __name__ == "__main__":
    main()

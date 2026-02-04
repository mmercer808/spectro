"""
SPECTRO Demo App - Main Integration Harness

This wires together all existing systems to demonstrate the core flow:
    MIDI Input → Entity Creation → Timeline → Audio Trigger → Visual Display

Based on existing codebase:
- engine/time/transport.py (Transport, TransportState)
- engine/time/camera.py (TimeCamera)
- engine/audio/engine.py (AudioEngine, Sampler)
- engine/midi/manager.py (MidiManager)
- engine/core/signal.py (SignalBridge)
- engine/core/scene.py (Scene, Entity)
- buffers_v2.py (EventDispatcher, MidiRingBuffer, AudioRingBuffer, ExecutionContext)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np

import moderngl
import moderngl_window as mglw

# === IMPORTS FROM YOUR EXISTING ENGINE ===
# (Adjust paths based on your actual structure)

from engine.core.signal import (
    SignalBridge, 
    SIGNAL_MIDI_NOTE_ON, SIGNAL_MIDI_PAD,
    SIGNAL_TRANSPORT_CHANGED, SIGNAL_DT
)
from engine.time.transport import Transport, TransportState, TimeSignature
from engine.time.camera import TimeCamera, TimeCameraMode
from engine.audio.engine import AudioEngine
from engine.audio.sampler import Sampler, generate_drum_samples

# From engine.buffers_v2
from engine.buffers_v2 import (
    EventDispatcher, MidiRingBuffer, AudioRingBuffer,
    MidiEvent, MidiEventType, ExecutionContext, InputDevice
)


# =============================================================================
# SEQUENCER DATA MODEL
# =============================================================================

@dataclass
class SequencerEvent:
    """An event on the timeline - the ENTITY created from MIDI input."""
    id: int
    beat: float           # X position (time)
    lane: int             # Y position (row/instrument)
    duration: float       # Width in beats
    velocity: int         # Intensity (0-127)
    sample_name: str      # What sound to trigger
    color: tuple = (1.0, 0.5, 0.2, 1.0)  # RGBA for display
    fired: bool = False   # Has this been triggered during playback?
    
    @property
    def end_beat(self) -> float:
        return self.beat + self.duration


class SequencerLane:
    """A horizontal lane (row) in the sequencer."""
    def __init__(self, index: int, name: str, sample_name: str, color: tuple):
        self.index = index
        self.name = name
        self.sample_name = sample_name
        self.color = color
        self.events: List[SequencerEvent] = []


class Sequencer:
    """
    The sequencer data model.
    
    Holds lanes and events. Events are created from MIDI input
    and trigger audio during playback.
    """
    def __init__(self):
        self._next_id = 1
        self.lanes: List[SequencerLane] = []
        self._setup_default_lanes()
    
    def _setup_default_lanes(self):
        """Create default drum lanes."""
        defaults = [
            ("Kick", "kick", (1.0, 0.3, 0.2, 1.0)),
            ("Snare", "snare", (1.0, 0.6, 0.2, 1.0)),
            ("HiHat", "hihat", (1.0, 0.9, 0.2, 1.0)),
            ("Clap", "clap", (0.4, 1.0, 0.3, 1.0)),
        ]
        for i, (name, sample, color) in enumerate(defaults):
            self.lanes.append(SequencerLane(i, name, sample, color))
    
    def add_event(self, lane_index: int, beat: float, velocity: int = 100, 
                  duration: float = 0.25) -> SequencerEvent:
        """
        Create a new event on the timeline.
        
        This is called when MIDI input arrives - the ENTITY CREATION step.
        """
        if lane_index < 0 or lane_index >= len(self.lanes):
            return None
        
        lane = self.lanes[lane_index]
        event = SequencerEvent(
            id=self._next_id,
            beat=beat,
            lane=lane_index,
            duration=duration,
            velocity=velocity,
            sample_name=lane.sample_name,
            color=lane.color,
        )
        self._next_id += 1
        lane.events.append(event)
        return event
    
    def get_events_in_range(self, start_beat: float, end_beat: float) -> List[SequencerEvent]:
        """Get all events overlapping a beat range."""
        result = []
        for lane in self.lanes:
            for event in lane.events:
                if event.beat < end_beat and event.end_beat > start_beat:
                    result.append(event)
        return result
    
    def reset_fired_flags(self):
        """Reset all fired flags (for loop restart)."""
        for lane in self.lanes:
            for event in lane.events:
                event.fired = False


# =============================================================================
# DEMO APP
# =============================================================================

class SpectroDemo(mglw.WindowConfig):
    """
    Main SPECTRO demo application.
    
    Demonstrates the complete flow:
    1. MIDI input received (keyboard or Launchpad)
    2. Entity (SequencerEvent) created and placed on timeline
    3. During playback, events trigger audio
    4. Visual display shows timeline, waveform, events
    """
    
    gl_version = (3, 3)
    title = "SPECTRO Demo - MIDI → Timeline → Audio"
    window_size = (1280, 720)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # === CORE SYSTEMS ===
        self.signals = SignalBridge()
        self.transport = Transport(bpm=120.0)
        self.time_camera = TimeCamera(mode=TimeCameraMode.FOLLOW_PLAYHEAD)
        self.sequencer = Sequencer()
        
        # === AUDIO ===
        self.audio = AudioEngine(
            transport=self.transport,
            signals=self.signals,
            sample_rate=44100,
            buffer_size=512,
        )
        self.audio.load_default_sounds()
        
        # === EVENT SYSTEM (from buffers_v2.py) ===
        self.midi_buffer = MidiRingBuffer(capacity=4096)
        self.audio_buffer = AudioRingBuffer(capacity_samples=65536)
        self.dispatcher = EventDispatcher(self.midi_buffer, self.audio_buffer)
        self.dispatcher.set_bpm(self.transport.bpm)
        
        # === INPUT DEVICE (keyboard acts as drum pad) ===
        self.keyboard_device = InputDevice("Keyboard")
        self.keyboard_device.connect(self.midi_buffer, self.audio_buffer)
        
        # === WIRE UP CALLBACKS ===
        self._setup_callbacks()
        
        # === RENDERING STATE ===
        self._setup_rendering()
        
        # === TIMING ===
        self.last_time = time.perf_counter()
        self.frame_id = 0
        
        # Start audio
        self.audio.start()
        
        # Fill audio buffer
        self.audio_buffer.write_silence(44100)
        
        print("\n=== SPECTRO Demo ===")
        print("Keys: 1-4 = Drums, SPACE = Play/Pause, R = Reset")
        print("=====================\n")
    
    def _setup_callbacks(self):
        """Wire up the event system callbacks."""
        
        # === THE CRITICAL CALLBACK: MIDI → Entity → Audio ===
        def on_midi_input(ctx: ExecutionContext):
            """
            Called when MIDI input arrives.
            
            This is where:
            1. We receive the ExecutionContext with FRESH transport state
            2. Create an entity (SequencerEvent) on the timeline
            3. Immediately trigger audio (for real-time feedback)
            """
            event = ctx.event
            
            # Map note to lane (simple mapping: notes 0-3 = lanes 0-3)
            lane_index = event.note % len(self.sequencer.lanes)
            
            # === STEP 1: Create entity on timeline ===
            seq_event = self.sequencer.add_event(
                lane_index=lane_index,
                beat=ctx.transport.beat,  # Use CURRENT beat from late-bound context
                velocity=event.velocity,
                duration=0.25,
            )
            
            if seq_event:
                lane = self.sequencer.lanes[lane_index]
                print(f"[ENTITY] {lane.name} @ beat {seq_event.beat:.2f} "
                      f"(timing: {ctx.timing_error_ms:+.1f}ms)")
                
                # === STEP 2: Immediate audio feedback ===
                self.audio.trigger(lane.sample_name, event.velocity / 127.0)
        
        # Register with dispatcher (LATE BINDING - no params captured now)
        self.dispatcher.register(
            callback=on_midi_input,
            event_types={MidiEventType.NOTE_ON},
            name="midi_to_entity"
        )
        
        # === BEAT CALLBACK: Check for scheduled events ===
        def on_beat(beat: int, transport):
            """
            Called on each beat.
            
            During playback, we scan for events near the playhead
            and trigger their audio.
            """
            if not self.transport.playing:
                return
            
            # Look for events in a small window around current beat
            window = 0.1  # beats
            current = self.transport.playhead_beat
            events = self.sequencer.get_events_in_range(
                current - window, 
                current + window
            )
            
            for event in events:
                if not event.fired and event.beat <= current:
                    event.fired = True
                    self.audio.trigger(event.sample_name, event.velocity / 127.0)
                    print(f"[PLAY] {event.sample_name} @ beat {event.beat:.2f}")
        
        self.dispatcher.on_beat(on_beat)
        
        # === LOOP CALLBACK: Reset fired flags ===
        self.transport.on_loop_callbacks.append(
            lambda state: self.sequencer.reset_fired_flags()
        )
    
    def _setup_rendering(self):
        """Set up basic 2D rendering for timeline/sequencer display."""
        self.ctx.enable(self.ctx.BLEND)
        self.ctx.blend_func = self.ctx.SRC_ALPHA, self.ctx.ONE_MINUS_SRC_ALPHA
        
        # Simple quad shader for drawing rectangles
        self.quad_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            uniform vec2 u_offset;
            uniform vec2 u_size;
            uniform vec2 u_window;
            void main() {
                vec2 px = u_offset + in_pos * u_size;
                vec2 ndc = (px / u_window) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform vec4 u_color;
            out vec4 fragColor;
            void main() {
                fragColor = u_color;
            }
            """
        )
        
        # Unit quad
        quad_verts = np.array([
            0, 0,  1, 0,  1, 1,
            0, 0,  1, 1,  0, 1,
        ], dtype='f4')
        self.quad_vbo = self.ctx.buffer(quad_verts)
        self.quad_vao = self.ctx.vertex_array(
            self.quad_prog, [(self.quad_vbo, '2f', 'in_pos')]
        )
        
        # Update TimeCamera with window size
        w, h = self.window_size
        self.time_camera.set_panel_size(float(w), float(h - 150))  # Leave room for transport bar
    
    # =========================================================================
    # RENDER LOOP
    # =========================================================================
    
    def on_render(self, t: float, frame_time: float):
        """Main render loop."""
        # === UPDATE TIMING ===
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_time)
        self.last_time = now
        self.frame_id += 1
        
        # === UPDATE SYSTEMS ===
        
        # Transport update (advances playhead if playing)
        self.transport.update(dt)
        
        # Sync dispatcher with transport
        self.dispatcher._bpm = self.transport.bpm
        if self.transport.playing and not self.dispatcher.playing:
            self.dispatcher.play()
        elif not self.transport.playing and self.dispatcher.playing:
            self.dispatcher.pause()
        
        # Process dispatcher (fires callbacks with late-bound context)
        self.dispatcher.process_frame(dt)
        
        # Update time camera (for follow mode)
        self.time_camera.update(dt, self.transport.playhead_beat)
        
        # === RENDER ===
        w, h = self.wnd.size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.08, 0.09, 0.11, 1.0)
        
        # Draw regions
        self._draw_transport_bar(w, h)
        self._draw_timeline(w, h)
        self._draw_sequencer_grid(w, h)
        self._draw_playhead(w, h)
    
    def _draw_rect(self, x, y, w, h, color, window_size):
        """Helper to draw a rectangle."""
        self.quad_prog['u_offset'].value = (x, y)
        self.quad_prog['u_size'].value = (w, h)
        self.quad_prog['u_color'].value = color
        self.quad_prog['u_window'].value = window_size
        self.quad_vao.render()
    
    def _draw_transport_bar(self, w, h):
        """Draw transport controls at bottom."""
        bar_h = 50
        y = h - bar_h
        
        # Background
        self._draw_rect(0, y, w, bar_h, (0.12, 0.12, 0.14, 1.0), (w, h))
        
        # Play state indicator
        color = (0.3, 0.8, 0.3, 1.0) if self.transport.playing else (0.8, 0.3, 0.3, 1.0)
        self._draw_rect(20, y + 10, 30, 30, color, (w, h))
        
        # Beat indicator boxes
        beat_in_bar = int(self.transport.playhead_beat) % 4
        for i in range(4):
            bx = 70 + i * 35
            bc = (0.4, 0.6, 0.9, 1.0) if i == beat_in_bar else (0.2, 0.25, 0.3, 1.0)
            self._draw_rect(bx, y + 10, 30, 30, bc, (w, h))
    
    def _draw_timeline(self, w, h):
        """Draw beat grid lines."""
        timeline_y = 60
        timeline_h = h - 160  # Leave room for transport bar and header
        
        # Background
        self._draw_rect(0, timeline_y, w, timeline_h, (0.06, 0.07, 0.08, 1.0), (w, h))
        
        # Beat lines
        for beat in self.time_camera.iter_beat_positions():
            px = self.time_camera.beat_to_px(beat)
            if 0 <= px <= w:
                is_bar = int(beat) % 4 == 0
                color = (0.25, 0.25, 0.3, 1.0) if is_bar else (0.15, 0.15, 0.18, 1.0)
                line_w = 2 if is_bar else 1
                self._draw_rect(px, timeline_y, line_w, timeline_h, color, (w, h))
    
    def _draw_sequencer_grid(self, w, h):
        """Draw sequencer events."""
        timeline_y = 60
        lane_h = 40
        
        for lane in self.sequencer.lanes:
            lane_y = timeline_y + lane.index * (lane_h + 5) + 10
            
            # Lane background
            self._draw_rect(0, lane_y, w, lane_h, (0.1, 0.1, 0.12, 0.5), (w, h))
            
            # Events
            for event in lane.events:
                px = self.time_camera.beat_to_px(event.beat)
                event_w = event.duration * self.time_camera._px_per_beat
                
                if px + event_w > 0 and px < w:
                    # Event rect with velocity-based alpha
                    alpha = 0.5 + (event.velocity / 127.0) * 0.5
                    color = (*event.color[:3], alpha)
                    self._draw_rect(px, lane_y + 2, max(4, event_w - 2), lane_h - 4, color, (w, h))
    
    def _draw_playhead(self, w, h):
        """Draw the playhead line."""
        timeline_y = 60
        timeline_h = h - 160
        
        px = self.time_camera.beat_to_px(self.transport.playhead_beat)
        if 0 <= px <= w:
            self._draw_rect(px - 1, timeline_y, 3, timeline_h, (1.0, 0.4, 0.2, 0.9), (w, h))
    
    # =========================================================================
    # INPUT HANDLING
    # =========================================================================
    
    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action != self.wnd.keys.ACTION_PRESS:
            return
        
        # Number keys 1-4 = drum triggers
        key_to_note = {
            self.wnd.keys.NUMBER_1: 0,  # Kick
            self.wnd.keys.NUMBER_2: 1,  # Snare
            self.wnd.keys.NUMBER_3: 2,  # HiHat
            self.wnd.keys.NUMBER_4: 3,  # Clap
        }
        
        if key in key_to_note:
            # === THIS IS THE INPUT → BUFFER FLOW ===
            # The callback registered above will handle entity creation + audio
            self.keyboard_device.note_on(key_to_note[key], velocity=100)
        
        elif key == self.wnd.keys.SPACE:
            self.transport.toggle()
            print(f"[TRANSPORT] {'Playing' if self.transport.playing else 'Paused'}")
        
        elif key == self.wnd.keys.R:
            self.transport.stop()
            self.sequencer.reset_fired_flags()
            print("[TRANSPORT] Reset")
        
        elif key == self.wnd.keys.LEFT:
            self.transport.seek_by_bars(-1)
        
        elif key == self.wnd.keys.RIGHT:
            self.transport.seek_by_bars(1)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        """Zoom timeline with scroll wheel."""
        x, y = self.wnd.mouse
        self.time_camera.zoom(y_offset, x)
    
    def close(self):
        """Cleanup on exit."""
        self.audio.stop()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mglw.run_window_config(SpectroDemo)

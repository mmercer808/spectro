"""
Drum Sequencer Integration - Connects all components into a working system.

This module provides the high-level DrumSequencer class that integrates:
- SequenceTensor (grid state)
- GridRenderer (shader visualization)
- GridOverlay (gesture capture)
- OnsetEngine (audio analysis)
- Timeline connection (playhead sync)

Flow:
    AUDIO FILE                           USER GESTURES
         │                                     │
         ▼                                     ▼
    OnsetEngine ──────────────────────→ GridOverlay
         │                                     │
         │         ┌───────────────┐           │
         └────────→│ SequenceTensor│←──────────┘
                   │   (RGBA grid) │
                   └───────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         GridRenderer   Timeline    SignalBridge
              │            │            │
              ▼            ▼            ▼
           SHADER      PLAYHEAD     AUDIO ENGINE
         (visual)      (sync)       (triggers)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any

import numpy as np

# Internal imports
from sequence_tensor import SequenceTensor, RhythmSet, Channel
from onset_engine import OnsetEngine, BeatSlicer, KeyDetector
from grid_overlay import GridOverlay, GestureRouter
from grid_renderer import GridRenderer, GridColors, GridEvent


@dataclass
class DrumLane:
    """Configuration for a drum lane/row."""
    name: str
    sample_name: str
    color: tuple  # RGBA
    midi_note: int = 36  # Default kick
    frequency_band: int = 1  # For onset classification


class DrumSequencer:
    """
    Complete drum sequencer with tensor-based grid, shader rendering,
    gesture input, and audio analysis.
    
    Usage:
        # Setup
        sequencer = DrumSequencer(ctx, rows=8, cols=16)
        sequencer.set_signal_bridge(bridge)
        sequencer.set_audio_callback(lambda row, vel: audio.trigger(row, vel))
        
        # Get widgets for layout
        overlay = sequencer.get_overlay()
        parent_widget.layout().addWidget(overlay)
        
        # Each frame
        sequencer.update(dt, transport.playhead_beat)
        sequencer.render(x, y, width, height, window_size)
        
        # Load audio for analysis
        sequencer.analyze_audio(audio_data, bpm=120)
    """
    
    # Default drum kit lanes
    DEFAULT_LANES = [
        DrumLane("Kick", "kick", (1.0, 0.3, 0.2, 1.0), midi_note=36, frequency_band=1),
        DrumLane("Snare", "snare", (1.0, 0.6, 0.2, 1.0), midi_note=38, frequency_band=3),
        DrumLane("Closed HH", "hihat_closed", (1.0, 0.9, 0.2, 1.0), midi_note=42, frequency_band=4),
        DrumLane("Open HH", "hihat_open", (0.4, 1.0, 0.3, 1.0), midi_note=46, frequency_band=5),
        DrumLane("Tom High", "tom_high", (0.3, 0.8, 1.0, 1.0), midi_note=50, frequency_band=3),
        DrumLane("Tom Mid", "tom_mid", (0.4, 0.5, 1.0, 1.0), midi_note=47, frequency_band=2),
        DrumLane("Tom Low", "tom_low", (0.7, 0.4, 1.0, 1.0), midi_note=43, frequency_band=2),
        DrumLane("Clap", "clap", (1.0, 0.4, 0.8, 1.0), midi_note=39, frequency_band=3),
    ]
    
    def __init__(
        self,
        ctx,  # ModernGL context
        rows: int = 8,
        cols: int = 16,
        bpm: float = 120.0,
        sample_rate: int = 44100,
        lanes: Optional[List[DrumLane]] = None
    ):
        self.ctx = ctx
        self.rows = rows
        self.cols = cols
        self._bpm = bpm
        self.sample_rate = sample_rate
        
        # Lanes configuration
        self.lanes = lanes or self.DEFAULT_LANES[:rows]
        
        # Core tensor grid
        self.grid = SequenceTensor(rows=rows, cols=cols)
        
        # Shader renderer
        self.renderer = GridRenderer(ctx, self.grid, self._create_colors())
        
        # Gesture overlay (created on demand via get_overlay)
        self._overlay: Optional[GridOverlay] = None
        
        # Audio analysis engine
        self.onset_engine = OnsetEngine(sample_rate=sample_rate)
        self.onset_engine.set_bpm(bpm)
        
        # Configure onset engine band mapping
        for i, lane in enumerate(self.lanes):
            if lane.frequency_band in self.onset_engine.band_to_row:
                self.onset_engine.band_to_row[lane.frequency_band] = i
        
        # Beat slicer for sample extraction
        self.slicer = BeatSlicer(sample_rate=sample_rate)
        self.slicer.set_bpm(bpm)
        
        # Key detector for harmonic info
        self.key_detector = KeyDetector(sample_rate=sample_rate)
        
        # External connections
        self._signal_bridge = None
        self._audio_callback: Optional[Callable[[int, float], None]] = None
        
        # Playback state
        self._playhead = 0.0
        self._playing = False
        self._loop_length = float(cols)  # Default: loop whole grid
        
        # Timing
        self._last_update = time.perf_counter()
    
    def _create_colors(self) -> GridColors:
        """Create color scheme from lane definitions."""
        row_colors = [lane.color for lane in self.lanes]
        # Pad to 8 if needed
        while len(row_colors) < 8:
            row_colors.append((0.5, 0.5, 0.5, 1.0))
        
        return GridColors(
            row_colors=row_colors,
            use_row_colors=True
        )
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def bpm(self) -> float:
        return self._bpm
    
    @bpm.setter
    def bpm(self, value: float):
        self._bpm = max(20.0, min(300.0, value))
        self.onset_engine.set_bpm(self._bpm)
        self.slicer.set_bpm(self._bpm)
    
    @property
    def playhead(self) -> float:
        return self._playhead
    
    @property
    def playing(self) -> bool:
        return self._playing
    
    # =========================================================================
    # EXTERNAL CONNECTIONS
    # =========================================================================
    
    def set_signal_bridge(self, bridge):
        """Connect to SignalBridge for event routing."""
        self._signal_bridge = bridge
        
        # Connect grid events
        if self._overlay:
            self._overlay.router.signal_bridge = bridge
    
    def set_audio_callback(self, callback: Callable[[int, float], None]):
        """
        Set callback for audio triggering.
        
        Callback signature: callback(row: int, velocity: float)
        """
        self._audio_callback = callback
    
    def get_overlay(self, parent=None) -> GridOverlay:
        """
        Get or create the gesture overlay widget.
        
        This should be added to your layout, positioned over the renderer.
        """
        if self._overlay is None:
            self._overlay = GridOverlay(
                grid=self.grid,
                rows=self.rows,
                cols=self.cols,
                signal_bridge=self._signal_bridge,
                parent=parent
            )
        return self._overlay
    
    # =========================================================================
    # PLAYBACK CONTROL
    # =========================================================================
    
    def play(self):
        """Start playback."""
        self._playing = True
        self._emit('seq.play')
    
    def pause(self):
        """Pause playback."""
        self._playing = False
        self._emit('seq.pause')
    
    def stop(self):
        """Stop and reset to beginning."""
        self._playing = False
        self._playhead = 0.0
        self._emit('seq.stop')
    
    def toggle(self):
        """Toggle play/pause."""
        if self._playing:
            self.pause()
        else:
            self.play()
    
    def seek(self, beat: float):
        """Seek to specific beat position."""
        self._playhead = beat % self._loop_length
        self._emit('seq.seek', beat=self._playhead)
    
    def set_loop_length(self, beats: float):
        """Set loop length in beats."""
        self._loop_length = max(1.0, beats)
    
    # =========================================================================
    # UPDATE & RENDER
    # =========================================================================
    
    def update(self, dt: float = None, external_playhead: float = None):
        """
        Update sequencer state.
        
        Args:
            dt: Delta time (if None, computed internally)
            external_playhead: If provided, sync to external transport
        """
        now = time.perf_counter()
        if dt is None:
            dt = now - self._last_update
        self._last_update = now
        
        # Sync to external transport or advance internally
        if external_playhead is not None:
            self._playhead = external_playhead % self._loop_length
        elif self._playing:
            beats_per_second = self._bpm / 60.0
            self._playhead += dt * beats_per_second
            self._playhead = self._playhead % self._loop_length
        
        # Update renderer and detect triggers
        self.renderer.set_playhead(self._playhead, now)
        
        # Process triggered events
        events = self.renderer.poll_events()
        for event in events:
            self._on_trigger(event)
    
    def render(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        window_size: tuple
    ):
        """Render the grid."""
        self.renderer.render(x, y, width, height, window_size)
    
    def _on_trigger(self, event: GridEvent):
        """Handle a cell trigger event."""
        row = event.row
        velocity = event.velocity
        
        # Get lane info
        if row < len(self.lanes):
            lane = self.lanes[row]
            sample_name = lane.sample_name
        else:
            sample_name = f"row_{row}"
        
        # Call audio callback
        if self._audio_callback:
            self._audio_callback(row, velocity)
        
        # Emit signal
        self._emit('seq.trigger',
                  row=row,
                  col=event.col,
                  velocity=velocity,
                  sample=sample_name)
    
    # =========================================================================
    # GRID OPERATIONS
    # =========================================================================
    
    def toggle_cell(self, row: int, col: int) -> bool:
        """Toggle a cell. Returns new state."""
        return self.grid.toggle(row, col)
    
    def set_cell(self, row: int, col: int, active: bool, velocity: float = 0.8):
        """Set cell state."""
        self.grid.set_active(row, col, active)
        if active:
            self.grid.set_velocity(row, col, velocity)
    
    def clear(self):
        """Clear entire grid."""
        self.grid.clear()
        self._emit('seq.cleared')
    
    def clear_row(self, row: int):
        """Clear a single row."""
        self.grid.clear_row(row)
    
    def fill_pattern(self, pattern_name: str):
        """Fill grid with a preset pattern."""
        self.grid.clear()
        
        patterns = {
            'four_on_floor': self._pattern_four_on_floor,
            'breakbeat': self._pattern_breakbeat,
            'hiphop': self._pattern_hiphop,
        }
        
        if pattern_name in patterns:
            patterns[pattern_name]()
            self._emit('seq.pattern_loaded', pattern=pattern_name)
    
    def _pattern_four_on_floor(self):
        """Classic four-on-floor pattern."""
        # Kick on every beat
        for col in range(0, self.cols, 1):
            if col % 4 == 0:
                self.grid.set_active(0, col, True)
                self.grid.set_velocity(0, col, 1.0)
        
        # Snare on 2 and 4
        for col in range(0, self.cols, 4):
            self.grid.set_active(1, col + 2, True)
            self.grid.set_velocity(1, col + 2, 0.9)
        
        # Hi-hat on every 8th
        for col in range(0, self.cols, 1):
            if col % 2 == 0:
                self.grid.set_active(2, col, True)
                self.grid.set_velocity(2, col, 0.6)
    
    def _pattern_breakbeat(self):
        """Breakbeat pattern."""
        # Kick pattern
        kicks = [0, 2, 5, 8, 10, 13]
        for col in kicks:
            if col < self.cols:
                self.grid.set_active(0, col, True)
                self.grid.set_velocity(0, col, 0.9)
        
        # Snare pattern
        snares = [4, 12]
        for col in snares:
            if col < self.cols:
                self.grid.set_active(1, col, True)
                self.grid.set_velocity(1, col, 1.0)
        
        # Hi-hat pattern
        for col in range(0, self.cols, 2):
            self.grid.set_active(2, col, True)
            self.grid.set_velocity(2, col, 0.5 + 0.3 * (col % 4 == 0))
    
    def _pattern_hiphop(self):
        """Hip-hop style pattern."""
        # Kick
        self.grid.set_active(0, 0, True)
        self.grid.set_velocity(0, 0, 1.0)
        self.grid.set_active(0, 6, True)
        self.grid.set_velocity(0, 6, 0.8)
        self.grid.set_active(0, 8, True)
        self.grid.set_velocity(0, 8, 0.9)
        
        # Snare
        self.grid.set_active(1, 4, True)
        self.grid.set_velocity(1, 4, 1.0)
        self.grid.set_active(1, 12, True)
        self.grid.set_velocity(1, 12, 1.0)
        
        # Hi-hat with swing
        for col in range(0, self.cols, 2):
            self.grid.set_active(2, col, True)
            self.grid.set_velocity(2, col, 0.6)
            # Swing timing
            self.grid.set_timing(2, col, 0.1 if col % 4 == 2 else 0.0)
    
    # =========================================================================
    # AUDIO ANALYSIS
    # =========================================================================
    
    def analyze_audio(
        self,
        audio: np.ndarray,
        bpm: Optional[float] = None,
        merge: bool = False
    ) -> dict:
        """
        Analyze audio and populate grid with detected onsets.
        
        Args:
            audio: Mono audio data
            bpm: BPM (if None, will attempt detection)
            merge: If True, add to existing grid. If False, clear first.
            
        Returns:
            Analysis metadata dict
        """
        return self.onset_engine.analyze_and_populate(
            audio=audio,
            grid=self.grid,
            bpm=bpm,
            detect_bpm=bpm is None,
            merge=merge
        )
    
    def detect_key(self, audio: np.ndarray) -> tuple:
        """
        Detect musical key from audio.
        
        Returns:
            (key_name, mode, confidence)
        """
        return self.key_detector.detect_key(audio)
    
    def slice_audio(
        self,
        audio: np.ndarray,
        mode: str = 'beat'
    ) -> list:
        """
        Slice audio at beat boundaries.
        
        Args:
            audio: Audio data
            mode: 'beat', 'transient', or beat length like '0.25'
            
        Returns:
            List of BeatSlice objects
        """
        if mode == 'beat':
            return self.slicer.slice_by_beats(audio, beats_per_slice=1.0)
        elif mode == 'transient':
            return self.slicer.slice_by_transients(audio)
        else:
            try:
                beats = float(mode)
                return self.slicer.slice_by_beats(audio, beats_per_slice=beats)
            except ValueError:
                return self.slicer.slice_by_beats(audio)
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> dict:
        """Serialize sequencer state."""
        return {
            'rows': self.rows,
            'cols': self.cols,
            'bpm': self._bpm,
            'loop_length': self._loop_length,
            'grid': self.grid.to_dict(),
            'lanes': [
                {
                    'name': lane.name,
                    'sample_name': lane.sample_name,
                    'color': lane.color,
                    'midi_note': lane.midi_note,
                }
                for lane in self.lanes
            ]
        }
    
    def from_dict(self, data: dict):
        """Load sequencer state from dict."""
        self._bpm = data.get('bpm', 120.0)
        self._loop_length = data.get('loop_length', self.cols)
        
        if 'grid' in data:
            self.grid = SequenceTensor.from_dict(data['grid'])
        
        self.onset_engine.set_bpm(self._bpm)
        self.slicer.set_bpm(self._bpm)
    
    # =========================================================================
    # INTERNAL
    # =========================================================================
    
    def _emit(self, signal_name: str, **kwargs):
        """Emit signal through bridge if available."""
        if self._signal_bridge:
            self._signal_bridge.emit(signal_name, **kwargs)
    
    def release(self):
        """Release resources."""
        self.renderer.release()


# =============================================================================
# EXAMPLE INTEGRATION
# =============================================================================

def create_demo_sequencer(ctx, width: int = 800, height: int = 400):
    """
    Create a demo sequencer for testing.
    
    Returns:
        (sequencer, overlay_widget)
    """
    sequencer = DrumSequencer(ctx, rows=8, cols=16, bpm=120)
    
    # Fill with a pattern
    sequencer.fill_pattern('four_on_floor')
    
    # Create overlay
    overlay = sequencer.get_overlay()
    
    return sequencer, overlay

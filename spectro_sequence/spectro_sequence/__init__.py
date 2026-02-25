"""
SPECTRO Sequence Grid Module

A tensor-based drum sequencer with:
- GPU-rendered grid via shaders
- Qt overlay for gesture input  
- Audio onset detection for auto-population
- Bidirectional shader ↔ CPU event flow

Architecture:
                                                              
    ┌─────────────────────────────────────────────────────────┐
    │                    DrumSequencer                         │
    │                                                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │SequenceTensor│  │ GridRenderer│  │ GridOverlay │     │
    │  │             │  │             │  │             │     │
    │  │ RGBA tensor │←→│  Shader     │  │  Qt gestures│     │
    │  │ subdivisions│  │  playhead   │  │  router     │     │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
    │         │                │                │             │
    │         └────────────────┴────────────────┘             │
    │                          │                              │
    │  ┌─────────────┐  ┌──────┴──────┐  ┌─────────────┐     │
    │  │ OnsetEngine │  │ SignalBridge│  │ AudioEngine │     │
    │  │             │  │             │  │             │     │
    │  │ librosa/    │→ │ events      │→ │ triggers    │     │
    │  │ aubio       │  │ routing     │  │             │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    │                                                          │
    └─────────────────────────────────────────────────────────┘

Components:

1. SequenceTensor (sequence_tensor.py)
   - Grid state as (4, rows, cols) tensor
   - RGBA channels: active, velocity, timing, subdivision
   - Sparse subdivision tensors for cells with subdivisions
   - Dirty tracking for efficient GPU upload

2. OnsetEngine (onset_engine.py)  
   - Audio onset detection (librosa/aubio)
   - Beat synchronization and quantization
   - Rhythm set detection (swing, triplets, etc.)
   - Grid population from audio analysis

3. GridOverlay (grid_overlay.py)
   - Invisible Qt widget layer over shader
   - Gesture recognition (tap, drag, long-press)
   - Routes gestures through GestureRouter
   - Connects to SignalBridge

4. GridRenderer (grid_renderer.py)
   - Shader-based grid visualization
   - Reads from SequenceTensor texture
   - Playhead crossing detection
   - Event emission to CPU

5. DrumSequencer (drum_sequencer.py)
   - High-level integration class
   - Combines all components
   - Playback control
   - Pattern presets

Usage:
    from spectro_sequence import DrumSequencer
    
    # Create sequencer with ModernGL context
    sequencer = DrumSequencer(ctx, rows=8, cols=16, bpm=120)
    
    # Get overlay for layout
    overlay = sequencer.get_overlay(parent_widget)
    layout.addWidget(overlay)
    
    # Set audio callback
    sequencer.set_audio_callback(lambda row, vel: audio.play(row, vel))
    
    # Main loop
    sequencer.update(dt, transport.playhead_beat)
    sequencer.render(x, y, width, height, window_size)
    
    # Load audio
    sequencer.analyze_audio(audio_data, bpm=120)
"""

from .sequence_tensor import (
    SequenceTensor,
    Subdivision,
    RhythmSet,
    Channel,
    ShaderEventType,
    ShaderEvent,
    ShaderEventQueue,
    quantize_to_subdivision,
    detect_rhythm_set,
)

from .onset_engine import (
    OnsetEngine,
    Onset,
    GridEvent,
    FrequencyBand,
    BeatSlicer,
    BeatSlice,
    KeyDetector,
)

from .grid_overlay import (
    GridOverlay,
    GestureCell,
    GestureRouter,
    GestureType,
    Gesture,
    GridShaderBridge,
)

from .grid_renderer import (
    GridRenderer,
    GridColors,
    GridEvent,
)

from .drum_sequencer import (
    DrumSequencer,
    DrumLane,
    create_demo_sequencer,
)

__all__ = [
    # Core tensor
    'SequenceTensor',
    'Subdivision', 
    'RhythmSet',
    'Channel',
    
    # Shader events
    'ShaderEventType',
    'ShaderEvent', 
    'ShaderEventQueue',
    
    # Quantization
    'quantize_to_subdivision',
    'detect_rhythm_set',
    
    # Onset detection
    'OnsetEngine',
    'Onset',
    'FrequencyBand',
    'BeatSlicer',
    'BeatSlice',
    'KeyDetector',
    
    # UI overlay
    'GridOverlay',
    'GestureCell',
    'GestureRouter', 
    'GestureType',
    'Gesture',
    'GridShaderBridge',
    
    # Rendering
    'GridRenderer',
    'GridColors',
    'GridEvent',
    
    # Integration
    'DrumSequencer',
    'DrumLane',
    'create_demo_sequencer',
]

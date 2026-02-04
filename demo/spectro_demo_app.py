"""
SPECTRO Demo App - Main Integration Harness
============================================

CURRENT FLOW (Phase 1 - MIDI Triggering):
    MIDI Input → Entity Creation → Timeline → Audio Trigger → Visual Display

FUTURE FLOW (Phase 2 - Audio Analysis):
    Audio File → Beat Slice → Key Detection → Entity with Harmonic Metadata →
    Timeline Placement (with harmonic validation) → Playback (pitch-aware mixing)

=============================================================================
ARCHITECTURE OVERVIEW
=============================================================================

The system has three conceptual layers:

    ┌─────────────────────────────────────────────────────────────────────┐
    │  INPUT LAYER                                                        │
    │  ───────────                                                        │
    │  - MIDI devices (Launchpad, keyboards)                              │
    │  - Audio files (WAV, MP3 → beat slicing)                            │
    │  - Generated sequences (algorithmic composition)                    │
    │                                                                     │
    │  FUTURE: Audio input will be sliced at transients, each slice      │
    │  analyzed for key/chord content via chroma vectors. This creates   │
    │  "harmonic fingerprints" that travel with the entity.              │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ENTITY LAYER (The Timeline)                                        │
    │  ──────────────────────────                                         │
    │  - SequencerEvent: positioned in beat-space (X-axis)                │
    │  - Carries metadata: velocity, duration, color                      │
    │  - FUTURE: Also carries harmonic_info (key, chord, chroma)          │
    │                                                                     │
    │  The unified 3D space from engine/core/scene.py:                    │
    │    X = Time (beats)                                                 │
    │    Y = Frequency / Pitch / Lane                                     │
    │    Z = Intensity / Velocity / Amplitude                             │
    │                                                                     │
    │  HAPPY ACCIDENT POTENTIAL: The Y-axis as frequency means we can    │
    │  literally visualize harmonic relationships spatially. Notes that  │
    │  are harmonically related (3:2 ratio = perfect fifth) could be     │
    │  shown at proportional Y positions, revealing the geometry of      │
    │  chord progressions.                                                │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │  OUTPUT LAYER                                                       │
    │  ────────────                                                       │
    │  - AudioEngine: triggers samples, mixes output                      │
    │  - Visual: timeline, waveform, sequencer grid                       │
    │  - FUTURE: Pitch-shifted playback based on harmonic context         │
    │                                                                     │
    │  COMPLICATION: Real-time pitch shifting while maintaining timing   │
    │  is non-trivial. Phase vocoder or granular synthesis needed.       │
    │  Could be a "happy accident" if tempo-stretching artifacts become  │
    │  an aesthetic choice (see: vaporwave, chopped-and-screwed).        │
    └─────────────────────────────────────────────────────────────────────┘

=============================================================================
THE LATE-BINDING EVENT SYSTEM (from buffers_v2.py)
=============================================================================

This is the KEY ARCHITECTURAL INSIGHT:

    Traditional approach (EARLY binding):
        callback registered → parameters captured at registration time
        PROBLEM: By the time callback fires, state is STALE

    SPECTRO approach (LATE binding):
        callback registered → NO parameters captured
        event fires → ExecutionContext assembled with FRESH state
        callback receives current transport position, timing error, etc.

Why this matters for audio analysis:

    When a beat slice is placed on the timeline, we don't know what the
    harmonic context will be at playback time. Other slices might be added,
    the key center might shift, the user might transpose a section.

    With late binding:
        - Slice registered with its intrinsic key (e.g., "C major chord")
        - At playback time, ExecutionContext includes:
            - Current harmonic context (what key are we in NOW?)
            - Surrounding slices (what's playing simultaneously?)
            - Suggested pitch shift to fit context
        - Callback can decide: play as-is, pitch shift, or flag conflict

    FUTURE ExecutionContext additions:
        - harmonic_context: HarmonicSnapshot (current key center, chord)
        - suggested_transposition: int (semitones to shift for compatibility)
        - harmonic_tension: float (0.0 = consonant, 1.0 = dissonant)
        - nearby_slices: List[SequencerEvent] (for polyphonic awareness)

=============================================================================
BEAT SLICING + KEY DETECTION (Future Integration Points)
=============================================================================

The flow for audio file analysis:

    1. LOAD AUDIO FILE
       │
       ├── Decode to PCM (soundfile, librosa)
       ├── Detect tempo (beat tracking)
       └── Identify transients (onset detection)
       
    2. SLICE AT BEAT BOUNDARIES
       │
       │   COMPLICATION: Beat boundaries vs transient boundaries
       │   - Musical beats may not align with audio transients
       │   - Drum hits have attack → sustain → decay
       │   - Slicing mid-sustain creates artifacts
       │   
       │   HAPPY ACCIDENT: "Wrong" slice points can create glitchy
       │   textures. The error becomes a feature. Consider exposing
       │   slice offset as a creative parameter.
       │
       └── Output: List of audio chunks with beat positions
       
    3. ANALYZE EACH SLICE FOR HARMONIC CONTENT
       │
       ├── Compute chroma vector (12-bin pitch class histogram)
       │   - Uses STFT → magnitude spectrum → fold into octave
       │   - Result: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B] weights
       │
       ├── Match against chord templates
       │   - Major: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0] (root-3rd-5th)
       │   - Minor: [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
       │   - Dominant 7: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
       │   - etc.
       │
       ├── Estimate key center (Krumhansl-Schmuckler algorithm)
       │   - Compare chroma to major/minor key profiles
       │   - Highest correlation = most likely key
       │
       │   COMPLICATION: Ambiguous slices (e.g., just a kick drum)
       │   - Percussive sounds have weak pitch content
       │   - Solution: Mark as "rhythmic" vs "harmonic" slice
       │   - Rhythmic slices don't participate in key validation
       │
       └── Output: HarmonicInfo attached to each slice
       
    4. CREATE ENTITIES WITH HARMONIC METADATA
       │
       │   SequencerEvent now includes:
       │   - detected_key: Key (e.g., Key.C_MAJOR)
       │   - detected_chord: Chord (e.g., Chord.MAJOR)
       │   - chroma_vector: np.array (12 floats, for comparison)
       │   - is_harmonic: bool (vs purely rhythmic)
       │   - confidence: float (how sure are we about the detection)
       │
       └── Entity ready for timeline placement
       
    5. HARMONIC VALIDATION ON PLACEMENT
       │
       │   When placing a slice on the timeline:
       │   
       │   a) Check what's already there at that beat position
       │   b) Compare chroma vectors for harmonic compatibility
       │   c) Options:
       │      - Compatible: place as-is
       │      - Incompatible but close: suggest transposition
       │      - Highly dissonant: warn user (or auto-reject)
       │   
       │   HAPPY ACCIDENT: Dissonance isn't always bad. Jazz uses it
       │   intentionally. Consider a "tension" slider that controls
       │   how much harmonic clash is allowed. At 100%, anything goes.
       │   At 0%, only perfect consonance allowed.
       │
       │   CONNECTION TO THEORY: This is where the "attractor fields"
       │   concept applies. Each key has an attractor basin. Slices
       │   are pulled toward compatible keys. The visualization could
       │   show these fields as color gradients on the timeline.
       │
       └── Entity placed (possibly with transposition metadata)
       
    6. PLAYBACK WITH PITCH AWARENESS
       │
       │   On playback, the late-binding system kicks in:
       │   
       │   ExecutionContext includes:
       │   - event.harmonic_info (the slice's intrinsic key)
       │   - context.current_key (what key we're "in" right now)
       │   - context.transposition (semitones to shift)
       │   
       │   Audio callback:
       │   - If transposition != 0: pitch shift the audio
       │   - Trigger the (possibly shifted) sample
       │   
       │   COMPLICATION: Pitch shifting in real-time
       │   - Phase vocoder preserves timing but smears transients
       │   - Granular synthesis can be glitchy
       │   - Resampling changes duration
       │   
       │   SOLUTION: Pre-compute shifted versions at load time
       │   - Store slice at -12 to +12 semitone variants
       │   - Playback just selects the right variant
       │   - Memory vs CPU tradeoff
       │
       └── Audio output (harmonically coherent mix)

=============================================================================
THE 3:2 RATIO INSIGHT (Pitch-Rhythm Duality)
=============================================================================

From your theoretical work:

    In PITCH space: 3:2 ratio = perfect fifth (C to G)
    In RHYTHM space: 3:2 ratio = polyrhythm (3 against 2)

This duality suggests:

    1. VISUAL REPRESENTATION
       - Y-axis (pitch) and X-axis (time) are fundamentally related
       - A perfect fifth interval could be visualized as a specific
         slope on the timeline (rise/run = 3/2)
       - Chord progressions become geometric shapes

    2. HARMONIC RHYTHM
       - The rate of chord changes has rhythmic implications
       - Fast harmonic rhythm = energetic
       - Slow harmonic rhythm = stable, ambient
       - Could visualize this as "harmonic tempo" alongside BPM

    3. POLYRHYTHMIC HARMONY
       - What if pitch relationships WERE rhythmic relationships?
       - A C major chord (C-E-G) has frequency ratios ~4:5:6
       - Interpreted rhythmically: 4 against 5 against 6
       - The "sound" of a chord becomes the "feel" of a polyrhythm

    HAPPY ACCIDENT POTENTIAL: Generate rhythms FROM chords
    - Analyze slice's chord
    - Convert frequency ratios to rhythmic ratios
    - Use as a swing/groove template
    - Harmony literally becomes rhythm

=============================================================================
DJ MIXING APPLICATION (The Original Use Case)
=============================================================================

For DJ mixing, the key detection enables:

    1. PHRASE BOUNDARY DETECTION
       - Key changes often occur at phrase boundaries
       - Detect key shifts to find verse/chorus transitions
       - "Mix point" = end of phrase in compatible key

    2. HARMONIC MIXING
       - Camelot wheel / Circle of fifths compatibility
       - C major compatible with: G major, F major, A minor
       - Show compatible next tracks
       - Auto-suggest mix points

    3. SPECTRAL COMPLEMENT
       - Two tracks mixed should fill different frequency ranges
       - Track A heavy in low-mids + Track B heavy in highs = good
       - Track A + Track A (same spectrum) = muddy
       - Visualize as "negative space" fitting together

    FUTURE FEATURE: "Harmonic Autopilot"
    - Analyze both decks
    - Compute optimal mix point (phrase boundary + key match)
    - Compute optimal pitch adjustment for Deck B
    - One-button harmonic mix

=============================================================================
EXISTING CODEBASE INTEGRATION
=============================================================================

Files that will be extended:

    engine/audio/sampler.py
        ADD: pitch_shift(semitones) method
        ADD: slice loading with chroma analysis
        
    engine/core/scene.py  
        EXTEND: Entity with harmonic_info field
        ADD: HarmonicInfo dataclass
        
    buffers_v2.py
        EXTEND: ExecutionContext with harmonic_context
        ADD: HarmonicSnapshot (like TransportSnapshot but for harmony)
        
    engine/time/camera.py
        ADD: frequency_to_y() for pitch visualization
        ADD: key_color_map for harmonic coloring

New files needed:

    engine/audio/slicer.py
        - beat_slice(audio, bpm) → List[Slice]
        - detect_transients(audio) → List[int] (sample positions)
        
    engine/audio/harmony.py
        - compute_chroma(audio) → np.array (12 floats)
        - detect_key(chroma) → Key
        - detect_chord(chroma) → Chord
        - harmonic_distance(key1, key2) → float
        
    engine/audio/pitch.py
        - pitch_shift(audio, semitones) → audio
        - time_stretch(audio, factor) → audio

=============================================================================
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

# From buffers_v2.py
from buffers_v2 import (
    EventDispatcher, MidiRingBuffer, AudioRingBuffer,
    MidiEvent, MidiEventType, ExecutionContext, InputDevice
)


# =============================================================================
# HARMONIC INFO (Future: populated by audio analysis)
# =============================================================================

@dataclass
class HarmonicInfo:
    """
    Harmonic metadata for a slice/event.
    
    CURRENT: Placeholder, populated with defaults
    FUTURE: Populated by chroma analysis of audio slices
    
    The chroma_vector is a 12-element array representing pitch class
    distribution: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
    
    Values are normalized weights (0.0 to 1.0) indicating how much
    of each pitch class is present in the audio.
    
    Example chroma vectors:
        C major chord:  [1.0, 0, 0, 0, 0.8, 0, 0, 0.9, 0, 0, 0, 0]
        A minor chord:  [0.8, 0, 0, 0, 0.9, 0, 0, 0, 0, 1.0, 0, 0]
        Pure kick drum: [0.1, 0.1, 0.1, 0.1, ...] (flat, no clear pitch)
    
    COMPLICATION: Percussive sounds have weak/ambiguous pitch content.
    The is_harmonic flag distinguishes pitched vs unpitched material.
    Rhythmic slices bypass harmonic validation entirely.
    """
    detected_key: str = "unknown"      # e.g., "C major", "A minor"
    detected_chord: str = "unknown"    # e.g., "maj", "min", "dom7"
    root_note: int = 0                 # MIDI note number of root (60 = C4)
    chroma_vector: tuple = tuple([0.0] * 12)  # Immutable for hashing
    is_harmonic: bool = True           # False for pure percussion
    confidence: float = 0.0            # 0.0 = guess, 1.0 = certain
    
    # FUTURE: These enable harmonic mixing decisions
    # 
    # compatible_keys: List[str]       # Keys this can mix with
    # tension_level: float             # 0.0 = resolved, 1.0 = tense
    # suggested_transpositions: Dict[str, int]  # key → semitones to shift


# =============================================================================
# SEQUENCER DATA MODEL
# =============================================================================

@dataclass
class SequencerEvent:
    """
    An event on the timeline - the ENTITY created from MIDI input or audio slice.
    
    COORDINATE SYSTEM (from engine/core/scene.py):
        beat     → X axis (time in beats)
        lane     → Y axis (could map to frequency for harmonic viz)
        velocity → Z axis (intensity)
    
    CURRENT: Created from MIDI input, triggers pre-loaded samples
    FUTURE: Created from audio slices, carries harmonic metadata
    
    LIFECYCLE:
        1. Created (MIDI input or audio slice)
        2. Placed on timeline (beat position determined)
        3. Validated (harmonic compatibility checked) [FUTURE]
        4. Rendered (drawn on sequencer grid)
        5. Fired (audio triggered when playhead crosses)
        6. Reset (fired flag cleared on loop)
    
    HAPPY ACCIDENT: The 'fired' flag creates interesting possibilities
    for one-shot vs retriggerable behavior. What if some events only
    fire once per session? Or fire with decreasing velocity each time?
    Could create "decaying" patterns that evolve over loops.
    """
    id: int
    beat: float           # X position (time in beats)
    lane: int             # Y position (row/instrument)  
    duration: float       # Width in beats
    velocity: int         # Intensity (0-127), maps to Z axis
    sample_name: str      # What sound to trigger
    color: tuple = (1.0, 0.5, 0.2, 1.0)  # RGBA for display
    fired: bool = False   # Has this been triggered during playback?
    
    # === HARMONIC METADATA (Future: from audio analysis) ===
    harmonic_info: HarmonicInfo = None  # Populated when slicing audio
    transposition: int = 0              # Semitones to shift on playback
    
    # === SOURCE TRACKING ===
    # Useful for knowing where an event came from
    source_type: str = "midi"           # "midi", "audio_slice", "generated"
    source_file: str = ""               # Path to audio file if sliced
    source_start_sample: int = 0        # Sample offset in source file
    source_end_sample: int = 0          # For reconstructing the slice
    
    def __post_init__(self):
        if self.harmonic_info is None:
            self.harmonic_info = HarmonicInfo()
    
    @property
    def end_beat(self) -> float:
        return self.beat + self.duration
    
    @property
    def is_harmonic(self) -> bool:
        """Does this event participate in harmonic validation?"""
        return self.harmonic_info.is_harmonic
    
    # FUTURE: Methods for harmonic comparison
    #
    # def harmonic_distance(self, other: 'SequencerEvent') -> float:
    #     """Compute harmonic distance (0 = identical, 1 = maximally different)"""
    #     # Compare chroma vectors via cosine similarity
    #     pass
    #
    # def compatible_with(self, key: str) -> bool:
    #     """Check if this event is compatible with a target key"""
    #     pass
    #
    # def suggest_transposition(self, target_key: str) -> int:
    #     """Suggest semitones to shift to fit target key"""
    #     pass


class SequencerLane:
    """
    A horizontal lane (row) in the sequencer.
    
    CURRENT: Simple drum lane with associated sample
    FUTURE: Could represent:
        - A frequency band (for spectral slicing)
        - A key/mode (for harmonic organization)
        - A source track (for multi-track mixing)
    
    HAPPY ACCIDENT IDEA: What if lanes weren't fixed to instruments,
    but to PITCH CLASSES? Lane 0 = C, Lane 1 = C#, etc.
    Events would auto-sort to lanes based on detected root note.
    The visual layout would literally show harmonic content.
    
    CONNECTION TO 3:2 INSIGHT: If Y-axis is pitch (frequency),
    then vertical spacing between events shows intervals.
    A perfect fifth (3:2) would be ~7 lanes apart (semitones).
    Chords become visual shapes. Inversions rotate the shape.
    """
    def __init__(self, index: int, name: str, sample_name: str, color: tuple):
        self.index = index
        self.name = name
        self.sample_name = sample_name
        self.color = color
        self.events: List[SequencerEvent] = []
        
        # === FUTURE: Lane-level harmonic context ===
        # 
        # self.key_center: str = None        # If lane has a fixed key
        # self.pitch_class: int = None       # If lane represents a pitch (0-11)
        # self.frequency_range: tuple = None # If lane is a frequency band (low_hz, high_hz)
        # self.accepts_harmonic: bool = True # False for pure rhythm lanes


class Sequencer:
    """
    The sequencer data model.
    
    Holds lanes and events. Events are created from MIDI input (now)
    or audio slices (future) and trigger audio during playback.
    
    ARCHITECTURE NOTE: This is separate from engine/core/scene.py Scene
    for simplicity, but they share the same conceptual space:
        X = beat (time)
        Y = lane (pitch/frequency)
        Z = velocity (intensity)
    
    FUTURE INTEGRATION: Sequencer could BE a Scene, with SequencerEvents
    as Entities. This would enable:
        - Unified picking (click on event to select)
        - Unified rendering (same pipeline for 2D and 3D views)
        - Spatial queries (find all events in beat range)
    
    For now, keeping them separate avoids complexity.
    """
    def __init__(self):
        self._next_id = 1
        self.lanes: List[SequencerLane] = []
        self._setup_default_lanes()
        
        # === FUTURE: Global harmonic context ===
        #
        # self.key_center: str = "C major"    # Current key for the project
        # self.key_changes: List[tuple] = []  # (beat, new_key) for modulations
        # self.harmonic_tolerance: float = 0.5  # How much dissonance allowed
    
    def _setup_default_lanes(self):
        """
        Create default drum lanes.
        
        FUTURE: Could auto-generate lanes from audio analysis:
            - Detect instruments in audio file
            - Create lane per detected instrument
            - Or: Create lane per detected pitch class
        """
        defaults = [
            ("Kick", "kick", (1.0, 0.3, 0.2, 1.0)),      # Red
            ("Snare", "snare", (1.0, 0.6, 0.2, 1.0)),    # Orange
            ("HiHat", "hihat", (1.0, 0.9, 0.2, 1.0)),    # Yellow
            ("Clap", "clap", (0.4, 1.0, 0.3, 1.0)),      # Green
            # FUTURE: Add melodic lanes
            # ("Bass", "bass", (0.3, 0.5, 1.0, 1.0)),    # Blue - harmonic
            # ("Lead", "lead", (0.8, 0.3, 1.0, 1.0)),    # Purple - harmonic
        ]
        for i, (name, sample, color) in enumerate(defaults):
            self.lanes.append(SequencerLane(i, name, sample, color))
    
    def add_event(self, lane_index: int, beat: float, velocity: int = 100, 
                  duration: float = 0.25, harmonic_info: HarmonicInfo = None,
                  source_type: str = "midi") -> SequencerEvent:
        """
        Create a new event on the timeline.
        
        This is called when:
            1. MIDI input arrives (current) - the ENTITY CREATION step
            2. Audio file is sliced (future) - each slice becomes an event
            3. Algorithmically generated (future) - procedural composition
        
        HARMONIC VALIDATION (Future):
            Before adding, check compatibility with existing events:
            
            existing = self.get_events_in_range(beat - 0.5, beat + 0.5)
            for e in existing:
                if e.is_harmonic and harmonic_info.is_harmonic:
                    distance = harmonic_distance(e.harmonic_info, harmonic_info)
                    if distance > self.harmonic_tolerance:
                        # Options:
                        # 1. Reject placement
                        # 2. Suggest transposition
                        # 3. Warn user but allow
                        pass
        
        COMPLICATION: What about intentional dissonance?
        Jazz, experimental music, sound design all use "wrong" notes.
        Solution: User-controllable tolerance + "force place" option.
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
            harmonic_info=harmonic_info or HarmonicInfo(),
            source_type=source_type,
        )
        self._next_id += 1
        lane.events.append(event)
        return event
    
    # === FUTURE: Audio slice integration ===
    #
    # def add_slices_from_audio(self, audio_path: str, bpm: float) -> List[SequencerEvent]:
    #     """
    #     Slice an audio file and add events for each slice.
    #     
    #     FLOW:
    #         1. Load audio file
    #         2. Detect beats/transients
    #         3. Slice at boundaries
    #         4. Analyze each slice for harmonic content
    #         5. Create SequencerEvent for each slice
    #         6. Auto-assign to lanes (by pitch class or instrument)
    #     
    #     HAPPY ACCIDENT: Overlapping slices create layering.
    #     If slices aren't quantized perfectly, they might overlap,
    #     creating phasing effects or rhythmic complexity.
    #     """
    #     from engine.audio.slicer import beat_slice
    #     from engine.audio.harmony import compute_chroma, detect_key, detect_chord
    #     
    #     slices = beat_slice(audio_path, bpm)
    #     events = []
    #     
    #     for slice_data in slices:
    #         chroma = compute_chroma(slice_data.audio)
    #         key = detect_key(chroma)
    #         chord = detect_chord(chroma)
    #         
    #         harmonic = HarmonicInfo(
    #             detected_key=key,
    #             detected_chord=chord,
    #             chroma_vector=tuple(chroma),
    #             is_harmonic=(key != "unknown"),
    #             confidence=slice_data.confidence,
    #         )
    #         
    #         # Auto-assign lane based on pitch or instrument
    #         lane_index = self._lane_for_slice(slice_data, harmonic)
    #         
    #         event = self.add_event(
    #             lane_index=lane_index,
    #             beat=slice_data.beat,
    #             velocity=int(slice_data.amplitude * 127),
    #             duration=slice_data.duration_beats,
    #             harmonic_info=harmonic,
    #             source_type="audio_slice",
    #         )
    #         event.source_file = audio_path
    #         event.source_start_sample = slice_data.start_sample
    #         event.source_end_sample = slice_data.end_sample
    #         
    #         events.append(event)
    #     
    #     return events
    
    def get_events_in_range(self, start_beat: float, end_beat: float) -> List[SequencerEvent]:
        """
        Get all events overlapping a beat range.
        
        Used for:
            1. Rendering (which events are visible?)
            2. Playback (which events should fire?)
            3. Harmonic validation (what's playing at this time?)
        """
        result = []
        for lane in self.lanes:
            for event in lane.events:
                if event.beat < end_beat and event.end_beat > start_beat:
                    result.append(event)
        return result
    
    def reset_fired_flags(self):
        """
        Reset all fired flags (for loop restart).
        
        FUTURE: Could have different reset behaviors:
            - Reset all (current)
            - Reset only specific lanes
            - Decay: reduce velocity each loop instead of reset
            - One-shot: never reset, event only fires once ever
        """
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
        """
        Wire up the event system callbacks.
        
        KEY CONCEPT: Late Binding
        ─────────────────────────
        Callbacks are registered NOW but parameters are assembled LATER.
        
        When on_midi_input fires, the ExecutionContext contains:
            - event: The raw MIDI data
            - transport: FRESH snapshot of transport state
            - timing_error_ms: How early/late the hit was
        
        This "late binding" is critical because:
            1. We don't know what beat we'll be on when the event fires
            2. Other events might have changed the harmonic context
            3. The user might have changed BPM/tempo mid-session
        
        FUTURE: ExecutionContext will also include:
            - harmonic_context: Current key, chord, nearby harmonic events
            - suggested_transposition: If current event clashes harmonically
        """
        
        # =====================================================================
        # THE CRITICAL CALLBACK: MIDI → Entity → Audio
        # =====================================================================
        
        def on_midi_input(ctx: ExecutionContext):
            """
            Called when MIDI input arrives.
            
            This is where the magic happens:
            
            1. We receive ExecutionContext with FRESH transport state
               (late-bound, not stale from registration time)
               
            2. Create an entity (SequencerEvent) on the timeline
               - Position is ctx.transport.beat (CURRENT beat)
               - NOT the beat when callback was registered
               
            3. Immediately trigger audio (real-time feedback)
               - User hears their input instantly
               - The event is also stored for playback
            
            TIMING ACCURACY (Guitar Hero style):
            ─────────────────────────────────────
            ctx.timing_error_ms tells us how early/late the hit was:
                - Negative = early (hit before the beat)
                - Positive = late (hit after the beat)
                - Zero = perfect
            
            FUTURE USE: Could display timing feedback:
                - Green flash for tight hits (< 20ms)
                - Yellow for okay (20-50ms)
                - Red for sloppy (> 50ms)
            
            FUTURE USE: Quantization
                - If timing_error is small, snap to grid
                - If large, preserve human feel
                - User-controllable quantize strength
            
            HARMONIC CONTEXT (Future):
            ──────────────────────────
            When adding audio slices (not just MIDI triggers), we'll also:
                1. Check ctx.harmonic_context for current key
                2. Compare slice's detected_key with context
                3. If incompatible:
                    a. Suggest transposition
                    b. Apply auto-transpose if enabled
                    c. Or warn user
            """
            event = ctx.event
            
            # Map note to lane (simple mapping: notes 0-3 = lanes 0-3)
            # FUTURE: Map by pitch class, or by detected instrument
            lane_index = event.note % len(self.sequencer.lanes)
            
            # === STEP 1: Create entity on timeline ===
            #
            # The beat position comes from ctx.transport.beat
            # This is the LATE-BOUND value - captured at fire time
            #
            # If we had captured the beat at registration time,
            # we'd always place events at beat 0 (or wherever we were
            # when we started). Late binding fixes this.
            
            seq_event = self.sequencer.add_event(
                lane_index=lane_index,
                beat=ctx.transport.beat,  # <-- LATE BOUND: current beat
                velocity=event.velocity,
                duration=0.25,
                source_type="midi",
            )
            
            if seq_event:
                lane = self.sequencer.lanes[lane_index]
                
                # Log with timing info
                # FUTURE: Use timing_error for visual feedback
                timing_str = f"{ctx.timing_error_ms:+.1f}ms"
                print(f"[ENTITY] {lane.name} @ beat {seq_event.beat:.2f} "
                      f"(timing: {timing_str})")
                
                # === STEP 2: Immediate audio feedback ===
                #
                # User hears their input right away.
                # The event is ALSO stored in the sequencer for playback.
                #
                # This means:
                #   - Live jamming: sound plays immediately
                #   - On loop playback: sound plays again when playhead crosses
                #
                # FUTURE: If this is an audio slice with transposition:
                #   velocity = event.velocity / 127.0
                #   transposed_sample = get_transposed(sample_name, seq_event.transposition)
                #   self.audio.trigger(transposed_sample, velocity)
                
                self.audio.trigger(lane.sample_name, event.velocity / 127.0)
        
        # Register with dispatcher
        # NOTE: No parameters captured here! Just the function reference.
        self.dispatcher.register(
            callback=on_midi_input,
            event_types={MidiEventType.NOTE_ON},
            name="midi_to_entity"
        )
        
        # =====================================================================
        # BEAT CALLBACK: Check for scheduled events during playback
        # =====================================================================
        
        def on_beat(beat: int, transport):
            """
            Called on each beat by the dispatcher.
            
            During playback, we scan for events near the playhead
            and trigger their audio. This is the "playback" part
            of the sequencer.
            
            LOOK-AHEAD WINDOW:
            ──────────────────
            We don't wait for exact beat == event.beat because:
                1. Floating point: beats are floats, exact equality is unreliable
                2. Frame timing: we might skip past an event between frames
                3. Latency compensation: we might want to fire slightly early
            
            Instead, we use a small window and fire events that:
                - Haven't fired yet (event.fired == False)
                - Are at or before the current playhead
            
            FUTURE: Look-ahead for latency compensation
            ──────────────────────────────────────────
            Audio systems have latency (buffer size, driver, hardware).
            To compensate, we could:
                1. Fire events EARLY by the latency amount
                2. Schedule them for future playback
                3. Audio arrives at speakers exactly on beat
            
            Example:
                audio_latency_ms = 20
                audio_latency_beats = audio_latency_ms / (60000 / bpm)
                lookahead = current_beat + audio_latency_beats
                # Fire events up to 'lookahead', not just 'current_beat'
            
            FUTURE: Pitch-shifted playback
            ─────────────────────────────
            For audio slices with transposition:
                if event.transposition != 0:
                    shifted_sample = get_pitched(event.sample_name, event.transposition)
                    self.audio.trigger(shifted_sample, velocity)
            """
            if not self.transport.playing:
                return
            
            # Look for events in a small window around current beat
            window = 0.1  # beats (adjustable for feel)
            current = self.transport.playhead_beat
            events = self.sequencer.get_events_in_range(
                current - window, 
                current + window
            )
            
            for event in events:
                if not event.fired and event.beat <= current:
                    event.fired = True
                    
                    # Trigger audio
                    # FUTURE: Apply transposition if event.transposition != 0
                    velocity = event.velocity / 127.0
                    self.audio.trigger(event.sample_name, velocity)
                    
                    print(f"[PLAY] {event.sample_name} @ beat {event.beat:.2f}")
        
        self.dispatcher.on_beat(on_beat)
        
        # =====================================================================
        # LOOP CALLBACK: Reset state on loop restart
        # =====================================================================
        
        self.transport.on_loop_callbacks.append(
            lambda state: self._on_loop_restart(state)
        )
    
    def _on_loop_restart(self, state):
        """
        Called when the transport loops back to the start.
        
        CURRENT: Just reset fired flags so events can fire again.
        
        FUTURE POSSIBILITIES:
        ─────────────────────
        1. DECAY: Reduce velocity each loop (events fade out)
           for event in all_events:
               event.velocity = int(event.velocity * 0.9)
               if event.velocity < 10:
                   remove(event)
        
        2. MUTATION: Slightly randomize timing each loop
           for event in all_events:
               event.beat += random.uniform(-0.05, 0.05)
        
        3. EVOLUTION: Add variations based on harmonic analysis
           - Occasionally transpose an event
           - Add passing tones
           - Create fills based on pattern analysis
        
        4. ACCUMULATION: Don't reset, let events pile up
           - Each loop adds new events
           - Creates dense, layered textures
           - Good for ambient/drone generation
        
        These could be mix-and-match modes the user selects.
        """
        self.sequencer.reset_fired_flags()
        print("[LOOP] Reset fired flags")
    
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

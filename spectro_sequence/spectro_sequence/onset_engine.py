"""
OnsetEngine - Audio analysis pipeline for onset detection → grid population.

Onsets detected from audio are quantized and mapped to the SequenceTensor,
with subdivision levels determined by the rhythmic content.

Flow:
    Audio File → Onset Detection → Classification → Quantization → Timeline → Tensor
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
from enum import IntEnum

# Optional audio analysis libraries
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = None
    HAS_LIBROSA = False

try:
    import aubio
    HAS_AUBIO = True
except ImportError:
    aubio = None
    HAS_AUBIO = False

from sequence_tensor import (
    SequenceTensor, 
    Subdivision, 
    RhythmSet,
    quantize_to_subdivision,
    detect_rhythm_set,
    Channel
)


class FrequencyBand(IntEnum):
    """Frequency bands for instrument classification."""
    SUB_BASS = 0      # 20-60 Hz (sub, 808)
    BASS = 1          # 60-250 Hz (kick, bass)
    LOW_MID = 2       # 250-500 Hz (low toms, snare body)
    MID = 3           # 500-2000 Hz (snare crack, vocals)
    HIGH_MID = 4      # 2000-6000 Hz (hi-hats, cymbals)
    HIGH = 5          # 6000-20000 Hz (air, sizzle)


@dataclass
class Onset:
    """Raw detection from audio analysis."""
    sample_pos: int           # Sample position in audio
    time_seconds: float       # Time in seconds
    strength: float           # Detection strength (0-1)
    frequency_band: FrequencyBand = FrequencyBand.MID
    spectral_centroid: float = 0.0  # For instrument classification
    
    # Computed after BPM sync
    beat_position: float = 0.0      # Position in beats
    quantized: Optional[Subdivision] = None


@dataclass
class GridEvent:
    """Event ready to be placed in the grid."""
    beat: float               # Beat position (integer part = column)
    row: int                  # Grid row (instrument/frequency band)
    velocity: float           # 0-1
    subdivision: int          # Subdivision level
    sub_index: int            # Position within subdivision
    source_onset: Optional[Onset] = None  # Original onset for reference


class OnsetEngine:
    """
    Audio analysis engine for onset detection and grid population.
    
    Usage:
        engine = OnsetEngine(sample_rate=44100)
        engine.set_bpm(120.0)
        
        # Analyze audio
        onsets = engine.detect_onsets(audio_data)
        
        # Convert to grid events
        events = engine.onsets_to_events(onsets)
        
        # Populate grid
        engine.populate_grid(grid, events)
    """
    
    def __init__(
        self,
        sample_rate: int = 44100,
        hop_size: int = 512,
        onset_threshold: float = 0.3,
        min_onset_gap_ms: float = 30.0
    ):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.onset_threshold = onset_threshold
        self.min_onset_gap_samples = int(min_onset_gap_ms * sample_rate / 1000)
        
        # BPM sync
        self._bpm = 120.0
        self._samples_per_beat = self._compute_samples_per_beat()
        
        # Band mapping: which rows correspond to which frequency bands
        self.band_to_row: dict[FrequencyBand, int] = {
            FrequencyBand.BASS: 0,       # Kick
            FrequencyBand.LOW_MID: 1,    # Snare body
            FrequencyBand.MID: 2,        # Snare/clap
            FrequencyBand.HIGH_MID: 3,   # Hi-hat closed
            FrequencyBand.HIGH: 4,       # Hi-hat open / cymbal
        }
        
        # Subdivision candidates for quantization
        self.subdivision_candidates = [1, 2, 3, 4, 6, 8, 12, 16]
    
    def _compute_samples_per_beat(self) -> float:
        return (60.0 / self._bpm) * self.sample_rate
    
    @property
    def bpm(self) -> float:
        return self._bpm
    
    @bpm.setter
    def bpm(self, value: float):
        self._bpm = max(20.0, min(300.0, value))
        self._samples_per_beat = self._compute_samples_per_beat()
    
    def set_bpm(self, bpm: float):
        """Set BPM for beat synchronization."""
        self.bpm = bpm
    
    # =========================================================================
    # ONSET DETECTION
    # =========================================================================
    
    def detect_onsets(self, audio: np.ndarray, method: str = 'librosa') -> List[Onset]:
        """
        Detect onsets in audio data.
        
        Args:
            audio: Mono audio array (float32, -1 to 1)
            method: 'librosa' or 'aubio'
            
        Returns:
            List of Onset objects with sample positions and strengths
        """
        if method == 'librosa' and HAS_LIBROSA:
            return self._detect_librosa(audio)
        elif method == 'aubio' and HAS_AUBIO:
            return self._detect_aubio(audio)
        else:
            # Fallback: simple energy-based detection
            return self._detect_energy(audio)
    
    def _detect_librosa(self, audio: np.ndarray) -> List[Onset]:
        """Onset detection using librosa."""
        if not HAS_LIBROSA:
            return []
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_size,
            backtrack=True,
            units='frames'
        )
        
        # Onset strength for velocity
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_size
        )
        
        # Spectral centroid for classification
        centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_size
        )[0]
        
        onsets = []
        last_sample = -self.min_onset_gap_samples
        
        for frame in onset_frames:
            sample_pos = frame * self.hop_size
            
            # Enforce minimum gap
            if sample_pos - last_sample < self.min_onset_gap_samples:
                continue
            last_sample = sample_pos
            
            # Get strength and centroid at this frame
            strength = float(onset_env[frame]) if frame < len(onset_env) else 0.5
            cent = float(centroid[frame]) if frame < len(centroid) else 1000.0
            
            # Classify frequency band
            band = self._classify_band(cent)
            
            onset = Onset(
                sample_pos=sample_pos,
                time_seconds=sample_pos / self.sample_rate,
                strength=min(1.0, strength / (onset_env.max() + 1e-6)),
                frequency_band=band,
                spectral_centroid=cent
            )
            onsets.append(onset)
        
        return onsets
    
    def _detect_aubio(self, audio: np.ndarray) -> List[Onset]:
        """Onset detection using aubio."""
        if not HAS_AUBIO:
            return []
        
        # Create onset detector
        onset_detector = aubio.onset(
            "default",
            self.hop_size * 2,
            self.hop_size,
            self.sample_rate
        )
        onset_detector.set_threshold(self.onset_threshold)
        
        onsets = []
        last_sample = -self.min_onset_gap_samples
        
        # Process in blocks
        for i in range(0, len(audio) - self.hop_size, self.hop_size):
            block = audio[i:i + self.hop_size].astype(np.float32)
            
            if onset_detector(block):
                sample_pos = onset_detector.get_last()
                
                # Enforce minimum gap
                if sample_pos - last_sample < self.min_onset_gap_samples:
                    continue
                last_sample = sample_pos
                
                onset = Onset(
                    sample_pos=sample_pos,
                    time_seconds=sample_pos / self.sample_rate,
                    strength=float(onset_detector.get_descriptor()),
                    frequency_band=FrequencyBand.MID  # Would need spectral analysis
                )
                onsets.append(onset)
        
        return onsets
    
    def _detect_energy(self, audio: np.ndarray) -> List[Onset]:
        """Simple energy-based onset detection (fallback)."""
        # Compute energy in frames
        frame_size = self.hop_size
        n_frames = len(audio) // frame_size
        energy = np.zeros(n_frames)
        
        for i in range(n_frames):
            frame = audio[i * frame_size:(i + 1) * frame_size]
            energy[i] = np.sum(frame ** 2)
        
        # Normalize
        energy = energy / (energy.max() + 1e-6)
        
        # Compute energy difference (onset = sudden increase)
        diff = np.diff(energy, prepend=0)
        diff = np.maximum(0, diff)  # Only positive changes
        
        # Find peaks above threshold
        threshold = self.onset_threshold * diff.max()
        peak_frames = np.where(diff > threshold)[0]
        
        onsets = []
        last_sample = -self.min_onset_gap_samples
        
        for frame in peak_frames:
            sample_pos = frame * frame_size
            
            if sample_pos - last_sample < self.min_onset_gap_samples:
                continue
            last_sample = sample_pos
            
            onset = Onset(
                sample_pos=sample_pos,
                time_seconds=sample_pos / self.sample_rate,
                strength=float(diff[frame] / (diff.max() + 1e-6)),
                frequency_band=FrequencyBand.MID
            )
            onsets.append(onset)
        
        return onsets
    
    def _classify_band(self, centroid: float) -> FrequencyBand:
        """Classify frequency band from spectral centroid."""
        if centroid < 100:
            return FrequencyBand.SUB_BASS
        elif centroid < 300:
            return FrequencyBand.BASS
        elif centroid < 600:
            return FrequencyBand.LOW_MID
        elif centroid < 2500:
            return FrequencyBand.MID
        elif centroid < 7000:
            return FrequencyBand.HIGH_MID
        else:
            return FrequencyBand.HIGH
    
    # =========================================================================
    # BEAT SYNCHRONIZATION & QUANTIZATION
    # =========================================================================
    
    def sync_to_beats(self, onsets: List[Onset]) -> List[Onset]:
        """
        Convert onset sample positions to beat positions.
        Modifies onsets in place and returns them.
        """
        for onset in onsets:
            onset.beat_position = onset.sample_pos / self._samples_per_beat
        return onsets
    
    def quantize_onsets(
        self,
        onsets: List[Onset],
        candidates: List[int] = None
    ) -> List[Onset]:
        """
        Quantize onsets to grid subdivisions.
        Modifies onsets in place.
        """
        if candidates is None:
            candidates = self.subdivision_candidates
        
        for onset in onsets:
            frac = onset.beat_position - int(onset.beat_position)
            onset.quantized = quantize_to_subdivision(frac, candidates)
        
        return onsets
    
    def detect_swing(self, onsets: List[Onset]) -> Optional[float]:
        """
        Analyze onsets to detect swing amount.
        
        Returns:
            Swing ratio (0.5 = straight, 0.67 = triplet swing) or None
        """
        # Collect fractional positions for off-beats
        off_beat_positions = []
        
        for onset in onsets:
            frac = onset.beat_position - int(onset.beat_position)
            # Look at positions near 0.5 (off-beats)
            if 0.4 < frac < 0.7:
                off_beat_positions.append(frac)
        
        if len(off_beat_positions) < 3:
            return None
        
        avg_position = np.mean(off_beat_positions)
        
        # 0.5 = straight, 0.67 = full triplet swing
        return avg_position
    
    # =========================================================================
    # GRID POPULATION
    # =========================================================================
    
    def onsets_to_events(self, onsets: List[Onset]) -> List[GridEvent]:
        """
        Convert analyzed onsets to GridEvents ready for tensor population.
        """
        events = []
        
        for onset in onsets:
            if onset.quantized is None:
                # Not quantized yet - use raw position
                col = int(onset.beat_position)
                frac = onset.beat_position - col
                subdiv = quantize_to_subdivision(frac)
            else:
                col = int(onset.beat_position)
                subdiv = onset.quantized
            
            # Map frequency band to row
            row = self.band_to_row.get(onset.frequency_band, 2)
            
            event = GridEvent(
                beat=onset.beat_position,
                row=row,
                velocity=onset.strength,
                subdivision=subdiv.level,
                sub_index=subdiv.index,
                source_onset=onset
            )
            events.append(event)
        
        return events
    
    def populate_grid(
        self,
        grid: SequenceTensor,
        events: List[GridEvent],
        merge: bool = True
    ):
        """
        Populate a SequenceTensor from GridEvents.
        
        Args:
            grid: Target tensor
            events: Events to add
            merge: If True, add to existing. If False, clear first.
        """
        if not merge:
            grid.clear()
        
        for event in events:
            col = int(event.beat) % grid.cols
            row = event.row % grid.rows
            
            # Set basic cell
            grid.set_active(row, col, True)
            grid.set_velocity(row, col, event.velocity)
            
            # Handle subdivision
            if event.subdivision > 1:
                # Need subdivision for this cell
                current_subdiv = grid.get_subdivision_level(row, col)
                if event.subdivision > current_subdiv:
                    grid.set_subdivision(row, col, event.subdivision)
                
                # Set the specific sub-cell
                grid.set_subdivision_cell(
                    row, col,
                    event.sub_index,
                    active=True,
                    velocity=event.velocity
                )
    
    # =========================================================================
    # HIGH-LEVEL PIPELINE
    # =========================================================================
    
    def analyze_and_populate(
        self,
        audio: np.ndarray,
        grid: SequenceTensor,
        bpm: Optional[float] = None,
        detect_bpm: bool = True,
        merge: bool = True
    ) -> dict:
        """
        Full pipeline: audio → analysis → grid population.
        
        Args:
            audio: Mono audio data
            grid: Target SequenceTensor
            bpm: BPM to use (if None and detect_bpm=True, will estimate)
            detect_bpm: Whether to estimate BPM from audio
            merge: Whether to merge with existing grid content
            
        Returns:
            Analysis metadata dict
        """
        metadata = {}
        
        # BPM detection/setting
        if bpm is not None:
            self.set_bpm(bpm)
        elif detect_bpm and HAS_LIBROSA:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            self.set_bpm(float(tempo))
            metadata['detected_bpm'] = float(tempo)
        
        metadata['bpm'] = self.bpm
        
        # Detect onsets
        onsets = self.detect_onsets(audio)
        metadata['onset_count'] = len(onsets)
        
        # Sync to beats
        self.sync_to_beats(onsets)
        
        # Detect swing
        swing = self.detect_swing(onsets)
        metadata['swing'] = swing
        
        # Quantize
        self.quantize_onsets(onsets)
        
        # Convert to events
        events = self.onsets_to_events(onsets)
        metadata['event_count'] = len(events)
        
        # Populate grid
        self.populate_grid(grid, events, merge=merge)
        
        return metadata


# =============================================================================
# BEAT SLICER
# =============================================================================

@dataclass
class BeatSlice:
    """A slice of audio aligned to beat boundaries."""
    start_sample: int
    end_sample: int
    start_beat: float
    end_beat: float
    audio: Optional[np.ndarray] = None
    
    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample
    
    @property
    def duration_beats(self) -> float:
        return self.end_beat - self.start_beat


class BeatSlicer:
    """
    Slice audio at beat boundaries for sample triggering.
    
    Used to chop loops into individual hits that can be triggered
    from the sequencer grid.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self._bpm = 120.0
        self._samples_per_beat = self._compute_samples_per_beat()
    
    def _compute_samples_per_beat(self) -> float:
        return (60.0 / self._bpm) * self.sample_rate
    
    def set_bpm(self, bpm: float):
        self._bpm = max(20.0, min(300.0, bpm))
        self._samples_per_beat = self._compute_samples_per_beat()
    
    def slice_by_beats(
        self,
        audio: np.ndarray,
        beats_per_slice: float = 1.0,
        include_audio: bool = True
    ) -> List[BeatSlice]:
        """
        Slice audio into chunks of specified beat length.
        """
        slices = []
        samples_per_slice = int(beats_per_slice * self._samples_per_beat)
        
        beat = 0.0
        for start in range(0, len(audio) - samples_per_slice, samples_per_slice):
            end = start + samples_per_slice
            
            slice_obj = BeatSlice(
                start_sample=start,
                end_sample=end,
                start_beat=beat,
                end_beat=beat + beats_per_slice,
                audio=audio[start:end].copy() if include_audio else None
            )
            slices.append(slice_obj)
            beat += beats_per_slice
        
        return slices
    
    def slice_at_onsets(
        self,
        audio: np.ndarray,
        onsets: List[Onset],
        include_audio: bool = True
    ) -> List[BeatSlice]:
        """
        Slice audio at onset positions.
        Each slice runs from one onset to the next.
        """
        if not onsets:
            return []
        
        slices = []
        
        for i in range(len(onsets)):
            start = onsets[i].sample_pos
            end = onsets[i + 1].sample_pos if i + 1 < len(onsets) else len(audio)
            
            slice_obj = BeatSlice(
                start_sample=start,
                end_sample=end,
                start_beat=onsets[i].beat_position,
                end_beat=onsets[i + 1].beat_position if i + 1 < len(onsets) else onsets[i].beat_position + 1,
                audio=audio[start:end].copy() if include_audio else None
            )
            slices.append(slice_obj)
        
        return slices
    
    def slice_by_transients(
        self,
        audio: np.ndarray,
        threshold: float = 0.3,
        include_audio: bool = True
    ) -> List[BeatSlice]:
        """
        Slice audio at transient/onset positions detected automatically.
        """
        engine = OnsetEngine(sample_rate=self.sample_rate)
        engine.set_bpm(self._bpm)
        
        onsets = engine.detect_onsets(audio)
        engine.sync_to_beats(onsets)
        
        return self.slice_at_onsets(audio, onsets, include_audio)


# =============================================================================
# KEY DETECTION (for harmonic mixing)
# =============================================================================

class KeyDetector:
    """
    Detect musical key from audio for harmonic mixing.
    Adds harmonic metadata to events for intelligent mixing.
    """
    
    # Chroma to key mapping (simplified)
    KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def detect_key(self, audio: np.ndarray) -> Tuple[str, str, float]:
        """
        Detect key from audio.
        
        Returns:
            (key_name, mode, confidence)
            e.g., ('C', 'major', 0.85)
        """
        if not HAS_LIBROSA:
            return ('C', 'major', 0.0)
        
        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
        
        # Average over time
        chroma_mean = np.mean(chroma, axis=1)
        
        # Key profiles (Krumhansl-Kessler)
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        # Correlate with all possible keys
        best_key = 0
        best_mode = 'major'
        best_corr = -1
        
        for shift in range(12):
            # Rotate chroma to this key
            rotated = np.roll(chroma_mean, -shift)
            
            # Correlate with major/minor profiles
            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]
            
            if major_corr > best_corr:
                best_corr = major_corr
                best_key = shift
                best_mode = 'major'
            
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = shift
                best_mode = 'minor'
        
        return (self.KEY_NAMES[best_key], best_mode, float(best_corr))
    
    def harmonic_compatibility(self, key1: str, mode1: str, key2: str, mode2: str) -> float:
        """
        Calculate harmonic compatibility between two keys (Camelot wheel style).
        Returns 0-1 where 1 = perfect match, 0 = clash.
        """
        # Convert to pitch class
        pc1 = self.KEY_NAMES.index(key1.upper().replace('♯', '#').replace('♭', 'b'))
        pc2 = self.KEY_NAMES.index(key2.upper().replace('♯', '#').replace('♭', 'b'))
        
        # Semitone difference
        diff = (pc2 - pc1) % 12
        
        # Compatible intervals (Camelot wheel logic)
        if mode1 == mode2:
            # Same mode: 0, 5, 7 semitones are compatible
            if diff in [0, 5, 7]:
                return 1.0
            elif diff in [2, 10]:
                return 0.7
            else:
                return 0.3
        else:
            # Different modes: relative major/minor (3 semitones)
            if mode1 == 'major' and diff == 9:  # Relative minor
                return 0.9
            elif mode1 == 'minor' and diff == 3:  # Relative major
                return 0.9
            else:
                return 0.4

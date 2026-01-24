"""
Sampler - Load and trigger one-shot samples.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import os

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class Sample:
    """A loaded audio sample."""
    name: str
    data: np.ndarray  # Mono or stereo float32, shape (samples,) or (samples, 2)
    sample_rate: int
    channels: int
    duration: float  # In seconds

    @property
    def num_samples(self) -> int:
        return self.data.shape[0]


@dataclass
class PlayingVoice:
    """A sample currently playing."""
    sample: Sample
    position: int = 0
    velocity: float = 1.0
    done: bool = False


class Sampler:
    """
    Simple sampler for triggering one-shot sounds.

    Usage:
        sampler = Sampler()
        sampler.load("kick", "assets/kick.wav")
        sampler.trigger("kick", velocity=0.8)

        # In audio callback:
        output = sampler.process(num_frames)
    """

    def __init__(self, sample_rate: int = 44100, max_voices: int = 32):
        self.sample_rate = sample_rate
        self.max_voices = max_voices

        self._samples: Dict[str, Sample] = {}
        self._voices: List[PlayingVoice] = []

    def load(self, name: str, path: str) -> bool:
        """Load a WAV file into a sample slot."""
        if not os.path.exists(path):
            print(f"Sampler: File not found: {path}")
            return False

        try:
            if SOUNDFILE_AVAILABLE:
                data, sr = sf.read(path, dtype='float32')
            else:
                # Fallback: generate a simple sound
                print(f"Sampler: soundfile not available, generating placeholder for {name}")
                data, sr = self._generate_placeholder(name)

            # Convert to float32
            if data.dtype != np.float32:
                data = data.astype(np.float32)

            # Ensure 2D (samples, channels)
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]

            # Resample if needed
            if sr != self.sample_rate:
                data = self._resample(data, sr, self.sample_rate)

            sample = Sample(
                name=name,
                data=data,
                sample_rate=self.sample_rate,
                channels=channels,
                duration=len(data) / self.sample_rate
            )
            self._samples[name] = sample
            print(f"Sampler: Loaded '{name}' ({sample.duration:.2f}s)")
            return True

        except Exception as e:
            print(f"Sampler: Error loading {path}: {e}")
            return False

    def load_from_array(self, name: str, data: np.ndarray, sample_rate: int = None):
        """Load a sample from a numpy array."""
        sr = sample_rate or self.sample_rate
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1]

        sample = Sample(
            name=name,
            data=data,
            sample_rate=sr,
            channels=channels,
            duration=len(data) / sr
        )
        self._samples[name] = sample

    def trigger(self, name: str, velocity: float = 1.0) -> bool:
        """Trigger a sample to play."""
        sample = self._samples.get(name)
        if sample is None:
            return False

        # Remove oldest voice if at capacity
        if len(self._voices) >= self.max_voices:
            self._voices.pop(0)

        voice = PlayingVoice(sample=sample, velocity=velocity)
        self._voices.append(voice)
        return True

    def stop_all(self):
        """Stop all playing voices."""
        self._voices.clear()

    def process(self, num_frames: int) -> np.ndarray:
        """
        Process audio and return mixed output.

        Args:
            num_frames: Number of samples to generate

        Returns:
            Stereo float32 array of shape (num_frames, 2)
        """
        output = np.zeros((num_frames, 2), dtype=np.float32)

        # Process each voice
        voices_to_remove = []
        for voice in self._voices:
            if voice.done:
                voices_to_remove.append(voice)
                continue

            sample = voice.sample
            remaining = sample.num_samples - voice.position
            to_copy = min(remaining, num_frames)

            if to_copy <= 0:
                voice.done = True
                voices_to_remove.append(voice)
                continue

            # Get sample data
            chunk = sample.data[voice.position:voice.position + to_copy]

            # Convert mono to stereo if needed
            if sample.channels == 1:
                if chunk.ndim == 1:
                    chunk = np.column_stack([chunk, chunk])

            # Apply velocity and mix
            output[:to_copy] += chunk * voice.velocity

            voice.position += to_copy

            if voice.position >= sample.num_samples:
                voice.done = True

        # Remove finished voices
        for voice in voices_to_remove:
            if voice in self._voices:
                self._voices.remove(voice)

        # Clip to prevent clipping
        np.clip(output, -1.0, 1.0, out=output)

        return output

    def _resample(self, data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Simple linear resampling."""
        if src_rate == dst_rate:
            return data

        ratio = dst_rate / src_rate
        new_length = int(len(data) * ratio)

        if data.ndim == 1:
            indices = np.linspace(0, len(data) - 1, new_length)
            return np.interp(indices, np.arange(len(data)), data).astype(np.float32)
        else:
            result = np.zeros((new_length, data.shape[1]), dtype=np.float32)
            indices = np.linspace(0, len(data) - 1, new_length)
            for ch in range(data.shape[1]):
                result[:, ch] = np.interp(indices, np.arange(len(data)), data[:, ch])
            return result

    def _generate_placeholder(self, name: str) -> Tuple[np.ndarray, int]:
        """Generate a placeholder sound when file loading fails."""
        sr = self.sample_rate
        duration = 0.1

        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        if "kick" in name.lower():
            # Kick: pitch drop + exponential decay
            freq = 60 * np.exp(-t * 30)
            data = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)
        elif "snare" in name.lower():
            # Snare: noise + tone
            noise = np.random.randn(len(t)).astype(np.float32) * 0.3
            tone = np.sin(2 * np.pi * 200 * t) * 0.5
            data = (noise + tone) * np.exp(-t * 15)
        elif "hat" in name.lower() or "hihat" in name.lower():
            # Hi-hat: filtered noise
            noise = np.random.randn(len(t)).astype(np.float32)
            data = noise * np.exp(-t * 40) * 0.3
        else:
            # Generic click
            data = np.sin(2 * np.pi * 440 * t) * np.exp(-t * 30)

        data = (data * 0.7).astype(np.float32)
        return data, sr

    @property
    def sample_names(self) -> List[str]:
        """Get list of loaded sample names."""
        return list(self._samples.keys())

    @property
    def voice_count(self) -> int:
        """Get number of currently playing voices."""
        return len(self._voices)


def generate_drum_samples(sampler: Sampler):
    """Generate basic drum samples for testing."""
    sr = sampler.sample_rate

    # Kick
    t = np.linspace(0, 0.15, int(sr * 0.15), dtype=np.float32)
    freq = 60 * np.exp(-t * 25)
    kick = np.sin(2 * np.pi * freq * t) * np.exp(-t * 15) * 0.8
    sampler.load_from_array("kick", kick)

    # Snare
    t = np.linspace(0, 0.12, int(sr * 0.12), dtype=np.float32)
    noise = np.random.randn(len(t)).astype(np.float32) * 0.4
    tone = np.sin(2 * np.pi * 180 * t) * 0.4
    snare = (noise + tone) * np.exp(-t * 18) * 0.7
    sampler.load_from_array("snare", snare)

    # Hi-hat
    t = np.linspace(0, 0.05, int(sr * 0.05), dtype=np.float32)
    hihat = np.random.randn(len(t)).astype(np.float32) * np.exp(-t * 60) * 0.3
    sampler.load_from_array("hihat", hihat)

    # Clap
    t = np.linspace(0, 0.1, int(sr * 0.1), dtype=np.float32)
    clap = np.random.randn(len(t)).astype(np.float32) * np.exp(-t * 25) * 0.5
    sampler.load_from_array("clap", clap)

    print("Sampler: Generated drum samples (kick, snare, hihat, clap)")

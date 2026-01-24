"""
Waveform utilities - Extract and generate waveform data for visualization.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def extract_waveform(
    audio_data: np.ndarray,
    sample_rate: int,
    samples_per_beat: int = 100,
    bpm: float = 120.0,
) -> np.ndarray:
    """
    Extract waveform envelope from audio data.

    Args:
        audio_data: Audio samples (mono or stereo)
        sample_rate: Sample rate of audio
        samples_per_beat: Output resolution (samples per beat)
        bpm: Tempo for beat calculation

    Returns:
        Float32 array of amplitude values (0-1) per beat sample
    """
    # Convert to mono
    if audio_data.ndim > 1:
        mono = np.mean(audio_data, axis=1)
    else:
        mono = audio_data

    # Calculate beat duration in samples
    samples_per_sec = sample_rate
    beats_per_sec = bpm / 60.0
    audio_samples_per_beat = samples_per_sec / beats_per_sec

    # Total beats in audio
    total_samples = len(mono)
    total_beats = total_samples / audio_samples_per_beat
    output_length = int(total_beats * samples_per_beat)

    if output_length <= 0:
        return np.zeros(1, dtype=np.float32)

    # Downsample with RMS envelope
    result = np.zeros(output_length, dtype=np.float32)
    chunk_size = max(1, total_samples // output_length)

    for i in range(output_length):
        start = int(i * chunk_size)
        end = min(start + chunk_size, total_samples)
        if start < end:
            chunk = mono[start:end]
            result[i] = np.sqrt(np.mean(chunk ** 2))  # RMS

    # Normalize to 0-1
    max_val = np.max(result)
    if max_val > 0:
        result /= max_val

    return result


def generate_sine_wave(
    frequency: float,
    duration: float,
    sample_rate: int = 44100,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a sine wave."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    return (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)


def generate_test_waveform(
    num_beats: int = 16,
    samples_per_beat: int = 100,
    bpm: float = 120.0,
) -> np.ndarray:
    """
    Generate a test waveform with beat-aligned transients.

    Creates a waveform that has peaks on beat positions,
    useful for testing visualization.
    """
    total_samples = num_beats * samples_per_beat
    result = np.zeros(total_samples, dtype=np.float32)

    # Base noise
    result += np.random.random(total_samples).astype(np.float32) * 0.1

    # Add peaks on beats
    for beat in range(num_beats):
        beat_pos = beat * samples_per_beat
        # Transient: sharp attack, exponential decay
        for i in range(min(samples_per_beat, total_samples - beat_pos)):
            decay = np.exp(-i / (samples_per_beat * 0.2))
            result[beat_pos + i] += decay * 0.6

        # Extra peak on kick beats (0, 4, 8, 12)
        if beat % 4 == 0:
            for i in range(min(samples_per_beat // 2, total_samples - beat_pos)):
                decay = np.exp(-i / (samples_per_beat * 0.1))
                result[beat_pos + i] += decay * 0.3

    # Normalize
    max_val = np.max(result)
    if max_val > 0:
        result /= max_val

    return result


def generate_beat_pattern_waveform(
    pattern: dict,
    num_beats: int = 16,
    samples_per_beat: int = 100,
) -> np.ndarray:
    """
    Generate waveform from a beat pattern.

    Args:
        pattern: Dict mapping names to beat lists, e.g., {"kick": [0, 4, 8, 12]}
        num_beats: Total beats to generate
        samples_per_beat: Resolution

    Returns:
        Waveform array
    """
    total_samples = num_beats * samples_per_beat
    result = np.zeros(total_samples, dtype=np.float32)

    # Instrument characteristics
    characteristics = {
        "kick": {"attack": 0.05, "decay": 0.15, "amplitude": 1.0},
        "snare": {"attack": 0.02, "decay": 0.1, "amplitude": 0.8},
        "hihat": {"attack": 0.01, "decay": 0.03, "amplitude": 0.4},
        "clap": {"attack": 0.02, "decay": 0.08, "amplitude": 0.6},
    }

    for name, beats in pattern.items():
        char = characteristics.get(name, {"attack": 0.02, "decay": 0.1, "amplitude": 0.5})

        for beat in beats:
            if beat >= num_beats:
                continue

            beat_pos = int(beat * samples_per_beat)
            decay_samples = int(char["decay"] * samples_per_beat * 4)

            for i in range(min(decay_samples, total_samples - beat_pos)):
                t = i / (samples_per_beat * 4)
                envelope = np.exp(-t / char["decay"]) * char["amplitude"]
                result[beat_pos + i] += envelope

    # Add subtle noise floor
    result += np.random.random(total_samples).astype(np.float32) * 0.02

    # Normalize to 0-1
    max_val = np.max(result)
    if max_val > 0:
        result /= max_val

    return result

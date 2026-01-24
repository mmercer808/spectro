"""
SPECTRO Audio System
====================

Audio playback, sampling, and waveform extraction.

Quick Start:
    from engine.audio import AudioEngine, Sampler

    audio = AudioEngine()
    audio.sampler.load("kick", "assets/kick.wav")
    audio.start()
    audio.sampler.trigger("kick")
"""

from .engine import AudioEngine
from .sampler import Sampler, Sample
from .waveform import extract_waveform, generate_sine_wave

__all__ = [
    'AudioEngine',
    'Sampler',
    'Sample',
    'extract_waveform',
    'generate_sine_wave',
]

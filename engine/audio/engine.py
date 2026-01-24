"""
AudioEngine - Main audio output coordinator.

Manages audio device, sampler, and synchronization with Transport.
"""

from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING
import threading
import queue
import numpy as np

from .sampler import Sampler, generate_drum_samples

if TYPE_CHECKING:
    from ..time.transport import Transport
    from ..core.signal import SignalBridge

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Audio will be silent.")


class AudioEngine:
    """
    Main audio engine coordinating output and timing.

    Usage:
        from engine.audio import AudioEngine
        from engine.time.transport import Transport

        transport = Transport(bpm=120)
        audio = AudioEngine(transport)
        audio.start()

        # Drums will play on beat callbacks
        transport.play()
    """

    def __init__(
        self,
        transport: Transport = None,
        signals: SignalBridge = None,
        sample_rate: int = 44100,
        buffer_size: int = 512,
    ):
        self.transport = transport
        self.signals = signals
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Create sampler
        self.sampler = Sampler(sample_rate=sample_rate)

        # Audio stream
        self._stream: Optional[sd.OutputStream] = None
        self._running = False

        # For waveform visualization
        self._output_buffer = np.zeros((sample_rate, 2), dtype=np.float32)
        self._output_write_pos = 0
        self._output_lock = threading.Lock()

        # Beat pattern (which samples to play on which beats)
        self._beat_pattern: dict = {}
        self._last_beat = -1

        # Wire transport callbacks
        if transport:
            transport.on_beat_callbacks.append(self._on_beat)

    def start(self) -> bool:
        """Start audio output."""
        if not SOUNDDEVICE_AVAILABLE:
            print("AudioEngine: sounddevice not available")
            return False

        if self._running:
            return True

        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                channels=2,
                dtype='float32',
                callback=self._audio_callback,
            )
            self._stream.start()
            self._running = True
            print(f"AudioEngine: Started (sr={self.sample_rate}, buf={self.buffer_size})")
            return True
        except Exception as e:
            print(f"AudioEngine: Failed to start: {e}")
            return False

    def stop(self):
        """Stop audio output."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("AudioEngine: Stopped")

    def _audio_callback(self, outdata: np.ndarray, frames: int,
                        time_info, status):
        """Audio callback - runs on audio thread."""
        if status:
            print(f"AudioEngine: {status}")

        # Get sampler output
        output = self.sampler.process(frames)

        # Copy to output
        outdata[:] = output

        # Store for visualization (thread-safe)
        with self._output_lock:
            end_pos = self._output_write_pos + frames
            if end_pos <= len(self._output_buffer):
                self._output_buffer[self._output_write_pos:end_pos] = output
            else:
                # Wrap around
                first_part = len(self._output_buffer) - self._output_write_pos
                self._output_buffer[self._output_write_pos:] = output[:first_part]
                self._output_buffer[:frames - first_part] = output[first_part:]
            self._output_write_pos = end_pos % len(self._output_buffer)

    def _on_beat(self, beat: int, state):
        """Called on each beat by Transport."""
        # Check beat pattern
        pattern_beat = beat % len(self._beat_pattern) if self._beat_pattern else -1

        for sample_name, beats in self._beat_pattern.items():
            if pattern_beat in beats:
                self.sampler.trigger(sample_name)

    def set_pattern(self, pattern: dict):
        """
        Set the beat pattern for auto-triggering samples.

        Args:
            pattern: Dict mapping sample names to list of beats.
                     e.g., {"kick": [0, 4, 8, 12], "snare": [4, 12]}
        """
        self._beat_pattern = pattern

    def trigger(self, sample_name: str, velocity: float = 1.0):
        """Manually trigger a sample."""
        self.sampler.trigger(sample_name, velocity)

    def get_output_waveform(self, num_samples: int = 1000) -> np.ndarray:
        """
        Get recent output waveform for visualization.

        Returns:
            Mono float32 array of RMS values.
        """
        with self._output_lock:
            # Get recent audio from ring buffer
            start = max(0, self._output_write_pos - num_samples * 4)
            if start < self._output_write_pos:
                data = self._output_buffer[start:self._output_write_pos].copy()
            else:
                # Wrapped
                data = np.concatenate([
                    self._output_buffer[start:],
                    self._output_buffer[:self._output_write_pos]
                ])

        if len(data) == 0:
            return np.zeros(num_samples, dtype=np.float32)

        # Convert to mono
        if data.ndim > 1:
            mono = (data[:, 0] + data[:, 1]) / 2
        else:
            mono = data

        # Downsample to requested size
        if len(mono) > num_samples:
            # Simple decimation with RMS
            chunk_size = len(mono) // num_samples
            result = np.zeros(num_samples, dtype=np.float32)
            for i in range(num_samples):
                chunk = mono[i * chunk_size:(i + 1) * chunk_size]
                if len(chunk) > 0:
                    result[i] = np.sqrt(np.mean(chunk ** 2))  # RMS
            return result
        else:
            return np.abs(mono)

    def load_default_sounds(self):
        """Load default drum samples."""
        generate_drum_samples(self.sampler)

    def set_default_pattern(self):
        """Set a basic 4/4 beat pattern."""
        self.set_pattern({
            "kick": [0, 4, 8, 12],      # Four on the floor
            "snare": [4, 12],           # 2 and 4
            "hihat": [2, 6, 10, 14],    # Off-beats
        })

    @property
    def is_running(self) -> bool:
        return self._running

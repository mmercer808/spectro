"""
spectro_bufferkit.py

A cleaned-up, easier-to-embed version of the earlier design.

What changed vs spectro_stream_api.py / lib_mt:
- Fewer concepts exposed up-front: you get ONE object: `Engine`
- Ergonomic sub-APIs: engine.tracks / engine.overlays / engine.context
- Clear separation of concerns:
    - Core primitives: GlobalClock, TimelineRingBuffer, EventWheel
    - Engine runtime: Mixer + threads + backend sync
    - Friendly façade: TrackAPI, OverlayAPI, ContextAPI
- Drop-in friendly:
    - No singleton required
    - Optional lazy default engine if you want it
    - Context manager support: `with Engine(...) as eng:`
- Designed to be used in other code with minimal wiring:
    - In a real audio callback: `eng.pull(frames).data`
    - In your main loop: `eng.sync()` (align to device cursor)
    - In UI: `eng.context.snapshot()` / `eng.context.read_audio(...)`

-----------------------------------------------------------------------------
MINIMAL USAGE (SIM)
-----------------------------------------------------------------------------
from spectro_bufferkit import Engine, make_sine

eng = Engine(sample_rate=44100, channels=2).enable_sim_audio()
eng.tracks.loop("a", make_sine(220, 0.5, 44100, 2), gain=0.2)
eng.tracks.loop("b", make_sine(330, 0.25, 44100, 2), gain=0.15)

eng.start()                 # starts mixer keep-filled thread
for _ in range(10):
    eng.sync()              # in a real app: call each frame
    block = eng.pull(512)   # in real audio: called by callback
    print(eng.context.snapshot())

eng.stop()

-----------------------------------------------------------------------------
MINIMAL USAGE (REAL AUDIO via sounddevice, optional)
-----------------------------------------------------------------------------
# pip install sounddevice
from spectro_bufferkit import Engine
eng = Engine(sample_rate=48000, channels=2)
eng.start_audio("sounddevice", blocksize=512)  # creates device backend + starts stream
eng.start()                                    # starts mixer keep-filled
...                                             # UI/main loop calls eng.sync()
eng.stop()

-----------------------------------------------------------------------------

-----------------------------------------------------------------------------
WAV LOADING (new)
-----------------------------------------------------------------------------
This module now includes standard WAV/MP3 loading helpers
MP3 support is optional and implemented via a layered loader (soundfile / pydub+ffmpeg / librosa).
 that return float32 numpy
arrays suitable for looping and for overlay writes.

Key functions:
- load_wav(path, target_sr=None, target_channels=None) -> np.ndarray
    Loads PCM WAV via stdlib `wave` (8/16/24/32-bit int) and returns float32 in [-1, 1].
    If target_sr is provided and differs, a simple linear resampler is applied.
    If target_channels is provided, mono<->stereo adaptation is applied.

Convenience APIs:
- eng.tracks.loop_wav(name, wav_path, gain=..., target_seconds=20.0, bpm=120.0, beat_div=8)
    Loads a WAV, optionally tiles it to roughly `target_seconds` (default 20s),
    and creates a looping track.

- eng.overlays.audio_wav(abs_start, wav_path, ...)
    Loads a WAV and writes it into the timeline ring at abs_start.

.
Beat-grid sizing note:
- A "beat" at bpm has duration beat_sec = 60/bpm.
- A \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
      frames_per_subbeat = sample_rate * beat_sec / beat_div
  These helpers expose bpm/beat_div so you can align loops/tiles to a grid if desired.

WHY THIS IS EASIER TO EMBED
-----------------------------------------------------------------------------
- You can ignore buses/groups/events at first and just schedule loop tracks.
- Overlay API accepts any time-tagged data (events) in the same sample timeline.
- The engine self-polices buffer memory and keeps a standard "lookahead" filled region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol
import threading
import time
import numpy as np


# =============================================================================
# Backend contract (stable)
# =============================================================================

class AudioBackend(Protocol):
    sample_rate: int
    channels: int
    def is_running(self) -> bool: ...
    def get_playhead_samples(self) -> int: ...
    def get_stream_time_seconds(self) -> float: ...
    def get_xrun_count(self) -> int: ...
    def get_latency_seconds(self) -> float: ...
    def get_status_string(self) -> str: ...


class NullAudioBackend:
    """Simulation backend (device cursor is advanced by our SimAudioDriver)."""
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._running = False
        self._playhead = 0
        self._t0 = time.perf_counter()
        self._lock = threading.Lock()
        self._xruns = 0

    def start(self):
        with self._lock:
            self._running = True
            self._playhead = 0
        self._t0 = time.perf_counter()

    def stop(self):
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_playhead_samples(self) -> int:
        with self._lock:
            return self._playhead

    def advance_simulated(self, n_frames: int) -> int:
        with self._lock:
            self._playhead += int(n_frames)
            return self._playhead

    def get_stream_time_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def get_xrun_count(self) -> int:
        return self._xruns

    def get_latency_seconds(self) -> float:
        return 0.0

    def get_status_string(self) -> str:
        return "NullAudioBackend(sim)"


class SoundDeviceBackend:
    """
    Optional real backend via sounddevice (PortAudio).
    Starts an OutputStream and pulls audio via `audio_pull(frames)->np.ndarray`.
    """
    def __init__(self,
                 sample_rate: int,
                 channels: int,
                 audio_pull: Callable[[int], np.ndarray],
                 blocksize: int = 512,
                 device: Optional[int] = None):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.device = device
        self._pull = audio_pull

        self._running = False
        self._playhead = 0
        self._xruns = 0
        self._t0 = time.perf_counter()
        self._latency = 0.0
        self._lock = threading.Lock()
        self._stream = None

    def start(self):
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            raise RuntimeError("sounddevice is not installed; pip install sounddevice") from e

        def callback(outdata, frames, time_info, status):
            if status:
                with self._lock:
                    self._xruns += 1
            block = self._pull(frames)
            if block.ndim == 1:
                block = block.reshape(-1, 1)
            if block.shape[1] != self.channels:
                if block.shape[1] == 1 and self.channels == 2:
                    block = np.column_stack([block[:, 0], block[:, 0]])
                else:
                    tmp = np.zeros((frames, self.channels), dtype=np.float32)
                    c = min(block.shape[1], self.channels)
                    tmp[:, :c] = block[:frames, :c]
                    block = tmp
            outdata[:] = block.astype(np.float32, copy=False)
            with self._lock:
                self._playhead += int(frames)

        import sounddevice as sd  # type: ignore
        self._t0 = time.perf_counter()
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            device=self.device,
            callback=callback,
        )
        self._latency = float(self._stream.latency) if hasattr(self._stream, "latency") else 0.0
        self._stream.start()
        with self._lock:
            self._running = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._running = False

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_playhead_samples(self) -> int:
        with self._lock:
            return self._playhead

    def get_stream_time_seconds(self) -> float:
        return time.perf_counter() - self._t0

    def get_xrun_count(self) -> int:
        with self._lock:
            return self._xruns

    def get_latency_seconds(self) -> float:
        return float(self._latency)

    def get_status_string(self) -> str:
        return "SoundDeviceBackend(portaudio)"


# =============================================================================
# Core time + buffer primitives
# =============================================================================

class GlobalClock:
    """Absolute sample timeline with backend sync and drift checking."""
    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self._playhead = 0
        self._lock = threading.Lock()

    def get(self) -> int:
        with self._lock:
            return self._playhead

    def set(self, sample_pos: int) -> None:
        with self._lock:
            self._playhead = int(sample_pos)

    def advance(self, frames: int) -> int:
        with self._lock:
            self._playhead += int(frames)
            return self._playhead

    def sync(self, backend: AudioBackend) -> int:
        cur = int(backend.get_playhead_samples())
        self.set(cur)
        return cur

    def drift_seconds(self, backend: AudioBackend) -> float:
        return (self.get() - int(backend.get_playhead_samples())) / float(self.sample_rate)


@dataclass(frozen=True)
class PlayheadSnapshot:
    playhead_samples: int
    sample_rate: int

    @property
    def seconds(self) -> float:
        return self.playhead_samples / float(self.sample_rate)


@dataclass
class BufferSpan:
    abs_start: int
    data: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.data.shape[0])


class TimelineRingBuffer:
    """
    Timeline-addressed circular audio memory.

    Self-policing:
    - tracks filled_until and consumed_until
    - write_at() refuses overwrite of unread region unless allow_overwrite=True
    """
    def __init__(self, capacity_frames: int, channels: int):
        cap = 1 << (int(capacity_frames) - 1).bit_length()
        self.capacity = cap
        self.mask = cap - 1
        self.channels = int(channels)

        self._ring = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self._filled_until = 0
        self._consumed_until = 0
        self._lock = threading.Lock()

    def filled_until(self) -> int:
        with self._lock:
            return self._filled_until

    def consumed_until(self) -> int:
        with self._lock:
            return self._consumed_until

    def unread_frames(self) -> int:
        with self._lock:
            return max(0, self._filled_until - self._consumed_until)

    def write_at(self, abs_start: int, block: np.ndarray, allow_overwrite: bool = False) -> int:
        abs_start = int(abs_start)
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        if block.shape[1] != self.channels:
            if block.shape[1] == 1 and self.channels == 2:
                block = np.column_stack([block[:, 0], block[:, 0]])
            else:
                tmp = np.zeros((block.shape[0], self.channels), dtype=np.float32)
                c = min(block.shape[1], self.channels)
                tmp[:, :c] = block[:, :c]
                block = tmp
        frames = int(block.shape[0])
        if frames <= 0:
            return 0

        with self._lock:
            abs_end = abs_start + frames

            if not allow_overwrite:
                unread0 = self._consumed_until
                unread1 = self._filled_until
                if abs_start < unread1 and abs_end > unread0:
                    if abs_end <= unread1:
                        return 0
                    clip = unread1 - abs_start
                    abs_start = unread1
                    block = block[clip:]
                    frames = int(block.shape[0])
                    abs_end = abs_start + frames

                unread_after = max(self._filled_until, abs_end) - self._consumed_until
                if unread_after > (self.capacity - 8):
                    max_end = self._consumed_until + (self.capacity - 8)
                    if abs_start >= max_end:
                        return 0
                    block = block[: max_end - abs_start]
                    frames = int(block.shape[0])
                    abs_end = abs_start + frames

            # write with wrap
            i0 = abs_start & self.mask
            end = i0 + frames
            if end <= self.capacity:
                self._ring[i0:end] = block
            else:
                first = self.capacity - i0
                self._ring[i0:] = block[:first]
                self._ring[:(end & self.mask)] = block[first:]

            if abs_end > self._filled_until:
                self._filled_until = abs_end
            return frames

    def read_at(self, abs_start: int, frames: int) -> BufferSpan:
        abs_start = int(abs_start)
        frames = int(max(0, frames))
        out = np.zeros((frames, self.channels), dtype=np.float32)
        if frames == 0:
            return BufferSpan(abs_start=abs_start, data=out)

        with self._lock:
            i0 = abs_start & self.mask
            end = i0 + frames
            if end <= self.capacity:
                out[:] = self._ring[i0:end]
            else:
                first = self.capacity - i0
                out[:first] = self._ring[i0:]
                out[first:] = self._ring[:(end & self.mask)]
        return BufferSpan(abs_start=abs_start, data=out)

    def consume(self, playhead: int, frames: int) -> BufferSpan:
        playhead = int(playhead)
        frames = int(max(0, frames))
        with self._lock:
            frames = min(frames, max(0, self._filled_until - playhead))
        span = self.read_at(playhead, frames)
        with self._lock:
            new_cons = playhead + span.frames
            if new_cons > self._consumed_until:
                self._consumed_until = new_cons
        return span


@dataclass(frozen=True)
class TimedEvent:
    when_samples: int
    kind: str
    payload: Any


class EventWheel:
    """Sample-indexed bucket ring for non-audio overlays (MIDI/automation/markers)."""
    def __init__(self, slots: int = 65536):
        cap = 1 << (int(slots) - 1).bit_length()
        self.capacity = cap
        self.mask = cap - 1
        self._buckets: List[List[TimedEvent]] = [[] for _ in range(self.capacity)]
        self._lock = threading.Lock()

    def post(self, ev: TimedEvent) -> None:
        idx = int(ev.when_samples) & self.mask
        with self._lock:
            self._buckets[idx].append(ev)

    def pop_exact_sample(self, s: int) -> List[TimedEvent]:
        idx = int(s) & self.mask
        with self._lock:
            bucket = self._buckets[idx]
            if not bucket:
                return []
            out, keep = [], []
            for ev in bucket:
                if ev.when_samples == s:
                    out.append(ev)
                else:
                    keep.append(ev)
            self._buckets[idx] = keep
        return out


# =============================================================================
# Mixing model
# =============================================================================

@dataclass
class MixContext:
    ph: PlayheadSnapshot
    abs_start: int
    abs_end: int
    scratch: np.ndarray


class RenderSource(Protocol):
    def render(self, ctx: MixContext) -> None: ...


class LoopTrack:
    """Simple looping clip track."""
    def __init__(self, clip: np.ndarray, gain: float = 1.0):
        if clip.ndim == 1:
            clip = clip.reshape(-1, 1)
        self.clip = clip.astype(np.float32, copy=False)
        self.clip_frames = int(self.clip.shape[0])
        self.gain = float(gain)
        self.start_sample = 0

    def render(self, ctx: MixContext) -> None:
        frames = ctx.abs_end - ctx.abs_start
        if frames <= 0 or self.clip_frames <= 0:
            return
        rel = (np.arange(frames, dtype=np.int64) + (ctx.abs_start - self.start_sample)) % self.clip_frames
        chunk = self.clip[rel]
        if chunk.shape[1] == 1 and ctx.scratch.shape[1] == 2:
            chunk = np.column_stack([chunk[:, 0], chunk[:, 0]])
        ctx.scratch[:frames] += chunk * self.gain


class Mixer:
    """Fill-ahead mixer; one writer into output ring."""
    def __init__(self, clock: GlobalClock, out: TimelineRingBuffer, channels: int):
        self.clock = clock
        self.out = out
        self.channels = int(channels)
        self.master_gain = 1.0
        self._lock = threading.RLock()
        self.sources: List[RenderSource] = []

    def add(self, src: RenderSource) -> None:
        with self._lock:
            self.sources.append(src)

    def fill_ahead(self, lookahead_frames: int, max_chunk: int = 2048) -> PlayheadSnapshot:
        ph = PlayheadSnapshot(self.clock.get(), self.clock.sample_rate)
        target = ph.playhead_samples + int(lookahead_frames)
        filled = self.out.filled_until()
        s0 = max(filled, ph.playhead_samples)

        with self._lock:
            sources = list(self.sources)

        while s0 < target:
            s1 = min(target, s0 + int(max_chunk))
            frames = s1 - s0
            scratch = np.zeros((frames, self.channels), dtype=np.float32)
            ctx = MixContext(ph=ph, abs_start=s0, abs_end=s1, scratch=scratch)

            for src in sources:
                src.render(ctx)

            scratch *= self.master_gain
            wrote = self.out.write_at(s0, scratch, allow_overwrite=False)
            if wrote <= 0:
                break
            s0 += wrote

        return ph


# =============================================================================
# Threads
# =============================================================================

class MixThread(threading.Thread):
    def __init__(self, engine: "Engine", tick_hz: float):
        super().__init__(daemon=True)
        self.engine = engine
        self.tick_hz = float(tick_hz)
        self._stop = threading.Event()

    def run(self):
        period = 1.0 / max(1.0, self.tick_hz)
        while not self._stop.is_set():
            self.engine.tick()
            time.sleep(period)

    def stop(self):
        self._stop.set()


class SimAudioDriver(threading.Thread):
    """Simulates device pull: sync -> pull -> advance backend cursor."""
    def __init__(self, engine: "Engine", backend: NullAudioBackend, block_frames: int, tick_hz: float):
        super().__init__(daemon=True)
        self.engine = engine
        self.backend = backend
        self.block_frames = int(block_frames)
        self.tick_hz = float(tick_hz)
        self._stop = threading.Event()

    def run(self):
        self.backend.start()
        period = 1.0 / max(1.0, self.tick_hz)
        while not self._stop.is_set():
            self.engine.sync()
            span = self.engine.pull(self.block_frames)
            self.backend.advance_simulated(span.frames)
            time.sleep(period)
        self.backend.stop()

    def stop(self):
        self._stop.set()


# =============================================================================
# Friendly façade APIs
# =============================================================================

@dataclass(frozen=True)
class EngineSnapshot:
    playhead_samples: int
    playhead_seconds: float
    filled_until: int
    consumed_until: int
    unread_frames: int
    backend_running: bool
    backend_status: str
    drift_seconds: float
    xruns: int
    latency_seconds: float


class ContextAPI:
    def __init__(self, engine: "Engine"):
        self.engine = engine

    def snapshot(self) -> EngineSnapshot:
        ph = PlayheadSnapshot(self.engine.clock.get(), self.engine.clock.sample_rate)
        filled = self.engine.out.filled_until()
        consumed = self.engine.out.consumed_until()
        unread = self.engine.out.unread_frames()
        running = bool(self.engine.backend and self.engine.backend.is_running())
        status = self.engine.backend.get_status_string() if self.engine.backend else "no-backend"
        drift = self.engine.clock.drift_seconds(self.engine.backend) if self.engine.backend else 0.0
        xruns = self.engine.backend.get_xrun_count() if self.engine.backend else 0
        lat = self.engine.backend.get_latency_seconds() if self.engine.backend else 0.0
        return EngineSnapshot(
            playhead_samples=ph.playhead_samples,
            playhead_seconds=ph.seconds,
            filled_until=filled,
            consumed_until=consumed,
            unread_frames=unread,
            backend_running=running,
            backend_status=status,
            drift_seconds=drift,
            xruns=xruns,
            latency_seconds=lat,
        )

    def read_audio(self, abs_start: int, frames: int) -> BufferSpan:
        return self.engine.out.read_at(abs_start, frames)


class TrackAPI:
    def __init__(self, engine: "Engine"):
        self.engine = engine
        self._lock = threading.Lock()
        self._tracks: Dict[str, LoopTrack] = {}


    def loop_wav(self,
                 name: str,
                 wav_path: str,
                 gain: float = 1.0,
                 target_seconds: float = 20.0,
                 bpm: float = 120.0,
                 beat_div: int = 8) -> "LoopTrack":
        """
        Load a WAV file, optionally tile it to ~target_seconds (default ~20s),
        and create a looping track.

        This keeps the engine easy to embed: "give me a wav file and loop it."
        """
        clip = load_wav(wav_path, target_sr=self.engine.clock.sample_rate, target_channels=self.engine.out.channels)
        if target_seconds and target_seconds > 0:
            clip = _tile_to_seconds(clip, sr=self.engine.clock.sample_rate, target_seconds=target_seconds, bpm=bpm, beat_div=beat_div)
        return self.loop(name, clip, gain=gain)


    def loop_audio(self,
                   name: str,
                   path: str,
                   gain: float = 1.0,
                   target_seconds: float = 20.0,
                   bpm: float = 120.0,
                   beat_div: int = 8) -> "LoopTrack":
        """
        Like loop_wav(), but supports MP3 too (via load_audio()).
        Keeps embedding simple: loop("file.wav") or loop("file.mp3").
        """
        clip = load_audio(path, target_sr=self.engine.clock.sample_rate, target_channels=self.engine.out.channels)
        if target_seconds and target_seconds > 0:
            clip = _tile_to_seconds(clip, sr=self.engine.clock.sample_rate, target_seconds=target_seconds, bpm=bpm, beat_div=beat_div)
        return self.loop(name, clip, gain=gain)
    def loop(self, name: str, clip: np.ndarray, gain: float = 1.0) -> LoopTrack:
        """Add a looping in-memory clip."""

        tr = LoopTrack(clip=clip, gain=gain)
        with self._lock:
            self._tracks[name] = tr
        self.engine.mixer.add(tr)
        return tr


    def wav_loop(self,
                 name: str,
                 wav_path: str,
                 gain: float = 1.0,
                 repeat_seconds: float = 20.0) -> LoopTrack:
        """
        Load a WAV from disk (resampled to engine sample rate), then loop it as a track.

        This is the easiest "drop a song/loop into the engine" call:

            eng.tracks.wav_loop("song", "C:/music/loop.wav", gain=0.8, repeat_seconds=20.0)

        repeat_seconds:
          The clip is truncated or padded-by-repeating to exactly this length so it loops cleanly.
        """
        clip, sr = load_wav_clip(
            wav_path,
            target_sr=self.engine.clock.sample_rate,
            target_channels=self.engine.out.channels,
            repeat_seconds=repeat_seconds,
        )
        # sr should now match engine sample rate
        return self.loop(name=name, clip=clip, gain=gain)

    def gain(self, name: str, value: float) -> None:
        with self._lock:
            tr = self._tracks.get(name)
        if tr:
            tr.gain = float(value)

    def get(self, name: str) -> Optional[LoopTrack]:
        with self._lock:
            return self._tracks.get(name)

    def list(self) -> List[str]:
        with self._lock:
            return list(self._tracks.keys())


class OverlayAPI:
    def __init__(self, engine: "Engine"):
        self.engine = engine


    def audio_wav(self,
                  abs_start: int,
                  wav_path: str,
                  target_seconds: Optional[float] = None,
                  bpm: float = 120.0,
                  beat_div: int = 8) -> int:
        """
        Load a WAV and overlay-write it into the engine output ring at abs_start.

        If target_seconds is provided, the clip is tiled to that duration before writing
        (useful for "repeat ~20 seconds in the buffer" style overlays).
        """
        clip = load_wav(wav_path, target_sr=self.engine.clock.sample_rate, target_channels=self.engine.out.channels)
        if target_seconds is not None and target_seconds > 0:
            clip = _tile_to_seconds(clip, sr=self.engine.clock.sample_rate, target_seconds=float(target_seconds), bpm=bpm, beat_div=beat_div)
        return self.audio(abs_start, clip)


    def audio_file(self,
                   abs_start: int,
                   path: str,
                   target_seconds: Optional[float] = None,
                   bpm: float = 120.0,
                   beat_div: int = 8) -> int:
        """
        Like audio_wav(), but supports MP3 too (via load_audio()).
        """
        clip = load_audio(path, target_sr=self.engine.clock.sample_rate, target_channels=self.engine.out.channels)
        if target_seconds is not None and target_seconds > 0:
            clip = _tile_to_seconds(clip, sr=self.engine.clock.sample_rate, target_seconds=float(target_seconds), bpm=bpm, beat_div=beat_div)
        return self.audio(abs_start, clip)
    def event(self, when_samples: int, kind: str, payload: Any) -> None:
        self.engine.events.post(TimedEvent(int(when_samples), str(kind), payload))

    def audio(self, abs_start: int, block: np.ndarray) -> int:
        return self.engine.out.write_at(int(abs_start), block, allow_overwrite=False)


# =============================================================================
# The ONE object most users should touch: Engine
# =============================================================================

class Engine:
    """
    Engine is the simplest embedding surface:
    - start()/stop() handles threads
    - sync() is the main-loop playhead alignment hook
    - pull(frames) is the audio callback pull
    - tracks/overlays/context are easy sub-APIs
    """
    def __init__(self,
                 sample_rate: int = 44100,
                 channels: int = 2,
                 capacity_frames: int = 262144,
                 lookahead_frames: Optional[int] = None,
                 backend: Optional[AudioBackend] = None):
        self.backend = backend
        self.clock = GlobalClock(sample_rate=sample_rate)
        self.out = TimelineRingBuffer(capacity_frames=capacity_frames, channels=channels)
        self.events = EventWheel(slots=65536)
        self.mixer = Mixer(self.clock, self.out, channels=channels)
        self.lookahead_frames = int(lookahead_frames) if lookahead_frames is not None else int(sample_rate // 4)

        self.tracks = TrackAPI(self)
        self.overlays = OverlayAPI(self)
        self.context = ContextAPI(self)

        self._mix_thread: Optional[MixThread] = None
        self._sim_audio: Optional[SimAudioDriver] = None

    # context manager convenience
    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self, mix_tick_hz: float = 240.0) -> "Engine":
        if not self._mix_thread or not self._mix_thread.is_alive():
            self._mix_thread = MixThread(self, tick_hz=mix_tick_hz)
            self._mix_thread.start()
        return self

    def stop(self) -> None:
        if self._mix_thread:
            self._mix_thread.stop()
        if self._sim_audio:
            self._sim_audio.stop()
        if isinstance(self.backend, SoundDeviceBackend):
            self.backend.stop()

    # main-loop sync hook
    def sync(self) -> int:
        if not self.backend:
            return self.clock.get()
        return self.clock.sync(self.backend)

    # the mixing tick (called by MixThread)
    def tick(self) -> PlayheadSnapshot:
        return self.mixer.fill_ahead(self.lookahead_frames)

    # audio callback pull
    def pull(self, frames: int) -> BufferSpan:
        ph = self.clock.get()
        span = self.out.consume(ph, int(frames))
        self.clock.advance(span.frames)
        return span

    # helpers
    def enable_sim_audio(self, block_frames: int = 512, tick_hz: float = 200.0) -> "Engine":
        if not isinstance(self.backend, NullAudioBackend):
            self.backend = NullAudioBackend(sample_rate=self.clock.sample_rate, channels=self.out.channels)
        if not self._sim_audio or not self._sim_audio.is_alive():
            self._sim_audio = SimAudioDriver(self, self.backend, block_frames=block_frames, tick_hz=tick_hz)
            self._sim_audio.start()
        return self

    def start_audio(self, kind: str = "sounddevice", blocksize: int = 512) -> "Engine":
        if kind != "sounddevice":
            raise ValueError(f"Unsupported backend kind: {kind}")
        self.backend = SoundDeviceBackend(
            sample_rate=self.clock.sample_rate,
            channels=self.out.channels,
            audio_pull=lambda n: self.pull(n).data,
            blocksize=blocksize,
        )
        self.backend.start()
        return self


# Optional lazy default engine (only if you want it)
_default_engine: Optional[Engine] = None
_default_lock = threading.Lock()

def default_engine(sample_rate: int = 44100, channels: int = 2) -> Engine:
    global _default_engine
    with _default_lock:
        if _default_engine is None:
            _default_engine = Engine(sample_rate=sample_rate, channels=channels)
        return _default_engine




# =============================================================================
# WAV loading utilities (drop-in, dependency-light)
# =============================================================================
# Goal:
# - Load a WAV from disk into a numpy float32 clip (frames, channels)
# - Optionally resample to engine sample_rate (simple linear; upgrade later)
# - Optionally enforce "repeat window" (e.g., 20s) so it loops cleanly
#
# Notes:
# - Uses Python stdlib `wave` (PCM 16/24/32-bit, mono/stereo)
# - 24-bit is handled via manual unpack
# - If you want faster / more formats (mp3, flac), swap to `soundfile` later.
#
# Beat-friendly heuristic:
# - "sample_rate * at least 8x a good sample for a beat" means keep enough frames
#   to represent musical timing with good resolution.
#   In practice the engine works in samples already; the important part is:
#     use the device sample_rate, keep clips long enough (>= multiple beats/bars),
#     and loop them.
#
# TODO (next iterations):
# - High-quality resampling (sinc / polyphase)
# - Auto-detect BPM / bar length from metadata or analysis
# - Time-stretching / pitch correction for tempo match
# - Streaming for long files (disk reader feeding a timeline ring)
#

import wave
import struct


def _pcm24_to_int32(raw: bytes) -> np.ndarray:
    """Convert little-endian packed 24-bit PCM to int32 array."""
    # raw length should be multiple of 3
    a = np.frombuffer(raw, dtype=np.uint8)
    a = a.reshape(-1, 3)
    # assemble little endian to signed 32
    x = (a[:, 0].astype(np.int32) |
         (a[:, 1].astype(np.int32) << 8) |
         (a[:, 2].astype(np.int32) << 16))
    # sign extension for 24-bit
    sign = x & 0x800000
    x = x - (sign << 1)
    return x


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """
    Load WAV file into float32 numpy array in range [-1, 1].

    Returns:
        (audio, sample_rate)
        audio shape: (frames, channels)
    """
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        # unsigned 8-bit
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:
        x = _pcm24_to_int32(raw).astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if ch > 1:
        x = x.reshape(-1, ch)
    else:
        x = x.reshape(-1, 1)

    return x, int(sr)


def _linear_resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Simple linear resample (good enough to start; replace later).

    audio: (frames, channels)
    """
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)

    frames, ch = audio.shape
    duration = frames / float(src_sr)
    dst_frames = int(round(duration * dst_sr))
    if dst_frames <= 1:
        return np.zeros((0, ch), dtype=np.float32)

    src_t = np.linspace(0.0, duration, num=frames, endpoint=False, dtype=np.float32)
    dst_t = np.linspace(0.0, duration, num=dst_frames, endpoint=False, dtype=np.float32)

    out = np.zeros((dst_frames, ch), dtype=np.float32)
    for c in range(ch):
        out[:, c] = np.interp(dst_t, src_t, audio[:, c]).astype(np.float32)
    return out


def load_wav_clip(path: str,
                  target_sr: Optional[int] = None,
                  target_channels: int = 2,
                  repeat_seconds: Optional[float] = 20.0) -> tuple[np.ndarray, int]:
    """
    Load a wav, adapt to target sample rate/channels, optionally truncate/loop-window.

    repeat_seconds:
      If set, we will truncate or pad the clip to exactly repeat_seconds long
      so you can loop cleanly inside the ring.

    Returns:
      (clip, sr) where clip is float32 (frames, channels)
    """
    audio, sr = load_wav(path)

    # resample if requested
    if target_sr is not None:
        audio = _linear_resample(audio, sr, target_sr)
        sr = int(target_sr)

    # channel adapt
    if audio.shape[1] != target_channels:
        if audio.shape[1] == 1 and target_channels == 2:
            audio = np.column_stack([audio[:, 0], audio[:, 0]]).astype(np.float32)
        else:
            tmp = np.zeros((audio.shape[0], target_channels), dtype=np.float32)
            c = min(audio.shape[1], target_channels)
            tmp[:, :c] = audio[:, :c]
            audio = tmp

    # enforce repeat window length if requested
    if repeat_seconds is not None:
        want_frames = int(round(float(repeat_seconds) * sr))
        if want_frames > 0:
            if audio.shape[0] >= want_frames:
                audio = audio[:want_frames]
            else:
                # pad by wrapping (repeat the audio to reach want_frames)
                reps = int(np.ceil(want_frames / max(1, audio.shape[0])))
                audio = np.tile(audio, (reps, 1))[:want_frames]

    return audio.astype(np.float32, copy=False), sr


class WavClip:
    """
    Convenience wrapper: load once, then feed into a LoopTrack.

    This is intentionally small: it gives you a stable way to reuse loaded clips
    across multiple tracks or engines.
    """
    def __init__(self, path: str, engine_sample_rate: int, channels: int = 2, repeat_seconds: float = 20.0):
        self.path = path
        self.clip, self.sample_rate = load_wav_clip(
            path, target_sr=engine_sample_rate, target_channels=channels, repeat_seconds=repeat_seconds
        )

    @property
    def frames(self) -> int:
        return int(self.clip.shape[0])




# =============================================================================
# Common audio utility functions (small, dependency-free)
# =============================================================================
# Thought process:
# - Keep the buffer/mixer core minimal, but ship a few "batteries" that almost every
#   audio workflow needs: gain<->dB, RMS/peak, normalization, beat-grid frame math.
# - These utilities are pure numpy/math and safe to reuse across your app.

def db_to_gain(db: float) -> float:
    """Convert decibels to linear gain."""
    return float(10.0 ** (float(db) / 20.0))

def gain_to_db(gain: float, floor_db: float = -120.0) -> float:
    """Convert linear gain to decibels with a floor to avoid -inf."""
    g = max(float(gain), 10.0 ** (float(floor_db) / 20.0))
    return float(20.0 * np.log10(g))

def rms(x: np.ndarray) -> float:
    """Root-mean-square level of a signal."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))

def peak(x: np.ndarray) -> float:
    """Peak absolute value of a signal."""
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x.astype(np.float64))))

def normalize(x: np.ndarray, target_peak: float = 0.98) -> np.ndarray:
    """Scale audio so peak reaches target_peak (returns a copy)."""
    p = peak(x)
    if p <= 1e-12:
        return x.astype(np.float32, copy=True)
    g = float(target_peak) / p
    y = (x.astype(np.float32, copy=True) * g).clip(-1.0, 1.0)
    return y

def frames_per_beat(sample_rate: int, bpm: float) -> int:
    """Frames per quarter-note beat at bpm."""
    return int(round(int(sample_rate) * (60.0 / float(bpm))))

def frames_per_subbeat(sample_rate: int, bpm: float, beat_div: int = 8) -> int:
    """Frames per sub-beat (e.g., beat_div=8 -> 8th-note resolution)."""
    beat_div = max(1, int(beat_div))
    return int(round(frames_per_beat(sample_rate, bpm) / beat_div))


# =============================================================================
# WAV loading helpers (stdlib wave + numpy)
# =============================================================================

def _linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """
    Simple linear resampler (good enough for dev/preview; replace with quality resampler later).
    x: (frames, channels) float32
    """
    if src_sr == dst_sr:
        return x
    src_sr = int(src_sr); dst_sr = int(dst_sr)
    n_src = int(x.shape[0])
    if n_src <= 1:
        return x
    n_dst = max(1, int(round(n_src * (dst_sr / float(src_sr)))))
    # Interpolate each channel
    t_src = np.linspace(0.0, 1.0, n_src, endpoint=True, dtype=np.float64)
    t_dst = np.linspace(0.0, 1.0, n_dst, endpoint=True, dtype=np.float64)
    out = np.empty((n_dst, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(t_dst, t_src, x[:, c].astype(np.float64)).astype(np.float32)
    return out

def _adapt_channels(x: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Adapt mono<->stereo (and basic trunc/pad for other counts).
    x: (frames, ch)
    """
    target_channels = int(target_channels)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    ch = int(x.shape[1])
    if ch == target_channels:
        return x
    if ch == 1 and target_channels == 2:
        return np.column_stack([x[:, 0], x[:, 0]]).astype(np.float32, copy=False)
    if ch == 2 and target_channels == 1:
        return x.mean(axis=1, keepdims=True).astype(np.float32, copy=False)
    # general: truncate or pad with zeros
    out = np.zeros((x.shape[0], target_channels), dtype=np.float32)
    c = min(ch, target_channels)
    out[:, :c] = x[:, :c]
    return out

def load_wav(path: str, target_sr: Optional[int] = None, target_channels: Optional[int] = None) -> np.ndarray:
    """
    Load a PCM WAV file using only the Python standard library.

    Returns:
      float32 numpy array of shape (frames, channels) in [-1, 1].

    Supported PCM widths:
      - 8-bit unsigned
      - 16-bit signed
      - 24-bit signed
      - 32-bit signed

    If target_sr is provided and differs, applies _linear_resample().
    If target_channels is provided, applies _adapt_channels().
    """
    import wave

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        src_sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    # Decode to int32 then scale to float32
    if sampwidth == 1:
        # 8-bit unsigned
        data_u8 = np.frombuffer(raw, dtype=np.uint8)
        data = (data_u8.astype(np.int16) - 128).astype(np.int32)
        scale = 128.0
    elif sampwidth == 2:
        data_i16 = np.frombuffer(raw, dtype=np.int16)
        data = data_i16.astype(np.int32)
        scale = 32768.0
    elif sampwidth == 3:
        # 24-bit little-endian packed: convert to int32
        b = np.frombuffer(raw, dtype=np.uint8)
        # reshape to (samples, 3)
        b = b.reshape(-1, 3)
        data = (b[:, 0].astype(np.int32) |
                (b[:, 1].astype(np.int32) << 8) |
                (b[:, 2].astype(np.int32) << 16))
        # sign extend 24-bit to 32-bit
        sign_bit = 1 << 23
        data = (data ^ sign_bit) - sign_bit
        scale = float(1 << 23)
    elif sampwidth == 4:
        data_i32 = np.frombuffer(raw, dtype=np.int32)
        data = data_i32.astype(np.int32)
        scale = float(1 << 31)
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if data.size == 0:
        x = np.zeros((0, int(target_channels or n_channels)), dtype=np.float32)
        return x

    # interleaved -> (frames, channels)
    data = data.reshape(-1, n_channels)
    x = (data.astype(np.float32) / float(scale)).clip(-1.0, 1.0)

    if target_sr is not None and int(target_sr) != int(src_sr):
        x = _linear_resample(x, src_sr=int(src_sr), dst_sr=int(target_sr))

    if target_channels is not None:
        x = _adapt_channels(x, int(target_channels))

    return x.astype(np.float32, copy=False)

def _tile_to_seconds(x: np.ndarray, sr: int, target_seconds: float, bpm: float = 120.0, beat_div: int = 8) -> np.ndarray:
    """
    Tile/loop a clip so its length is roughly target_seconds.
    Also optionally snap tile length to a beat-grid multiple:
      subbeat_frames = sr * (60/bpm) / beat_div

    This helps you build "20 seconds or so" of repeated material for:
      - analysis visuals
      - quick looping
      - pre-filled overlays
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    sr = int(sr)
    if sr <= 0:
        return x
    n = int(x.shape[0])
    if n <= 0:
        return x

    target_frames = max(1, int(round(float(target_seconds) * sr)))

    # Snap target_frames to beat-grid multiples if bpm/beat_div provided sensibly
    bpm = float(bpm)
    beat_div = int(max(1, beat_div))
    if bpm > 0:
        subbeat = int(round(sr * (60.0 / bpm) / beat_div))
        subbeat = max(1, subbeat)
        target_frames = int(round(target_frames / subbeat)) * subbeat
        target_frames = max(subbeat, target_frames)

    reps = (target_frames + n - 1) // n
    y = np.tile(x, (reps, 1))[:target_frames]
    return y.astype(np.float32, copy=False)

# =============================================================================
# MP3 loading helpers (optional dependencies)
# =============================================================================
# Thought process:
# - WAV is easy with stdlib `wave`.
# - MP3 is *not* in the Python stdlib. Different environments support it differently.
# - So we implement a "layered" loader that tries, in order:
#     1) soundfile (libsndfile) if installed and MP3-enabled on your system
#     2) pydub + ffmpeg if installed (most flexible)
#     3) librosa (audioread) if installed (also depends on ffmpeg/gstreamer)
# - The API stays simple: load_audio(path, target_sr, target_channels) -> float32 (frames, channels).
#
# Notes:
# - If you ship this as a library, you can keep MP3 optional to avoid forcing ffmpeg installs.
# - For production-quality resampling/decoding, you may later standardize on a single dependency
#   (e.g., ffmpeg decode to float32) and/or a higher quality resampler.

def _ext_lower(path: str) -> str:
    return (str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else "")

def load_mp3(path: str, target_sr: Optional[int] = None, target_channels: Optional[int] = None) -> np.ndarray:
    """
    Load an MP3 using optional dependencies.

    Returns:
      float32 numpy array (frames, channels) in [-1, 1].

    Tries (in order):
      1) soundfile (libsndfile)         -> fast, simple if MP3 supported in your libsndfile build
      2) pydub (ffmpeg)                 -> very common, robust format support via ffmpeg
      3) librosa (audioread backend)    -> convenient, but depends on external decoders too

    If none work, raises RuntimeError with install guidance.
    """
    # 1) soundfile
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        x = data.astype(np.float32, copy=False)
        if target_sr is not None and int(target_sr) != int(sr):
            x = _linear_resample(x, src_sr=int(sr), dst_sr=int(target_sr))
        if target_channels is not None:
            x = _adapt_channels(x, int(target_channels))
        return x
    except Exception:
        pass

    # 2) pydub + ffmpeg
    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(path, format="mp3")
        sr = seg.frame_rate
        ch = seg.channels
        # pydub provides samples as array of ints
        samples = np.array(seg.get_array_of_samples())
        # interleaved -> (frames, channels)
        if ch > 1:
            samples = samples.reshape(-1, ch)
        else:
            samples = samples.reshape(-1, 1)

        # sample_width (bytes) tells scaling
        sw = seg.sample_width
        if sw == 1:
            scale = 128.0
        elif sw == 2:
            scale = 32768.0
        elif sw == 3:
            scale = float(1 << 23)
        elif sw == 4:
            scale = float(1 << 31)
        else:
            scale = float(np.max(np.abs(samples)) or 1.0)

        x = (samples.astype(np.float32) / float(scale)).clip(-1.0, 1.0)
        if target_sr is not None and int(target_sr) != int(sr):
            x = _linear_resample(x, src_sr=int(sr), dst_sr=int(target_sr))
        if target_channels is not None:
            x = _adapt_channels(x, int(target_channels))
        return x.astype(np.float32, copy=False)
    except Exception:
        pass

    # 3) librosa (audioread)
    try:
        import librosa  # type: ignore
        y, sr = librosa.load(path, sr=None, mono=False, dtype=np.float32)
        # librosa returns (n,) mono or (ch, n) multi
        if y.ndim == 1:
            x = y.reshape(-1, 1)
        else:
            x = np.ascontiguousarray(y.T)
        if target_sr is not None and int(target_sr) != int(sr):
            x = _linear_resample(x, src_sr=int(sr), dst_sr=int(target_sr))
        if target_channels is not None:
            x = _adapt_channels(x, int(target_channels))
        return x.astype(np.float32, copy=False)
    except Exception:
        pass

    raise RuntimeError(
        "MP3 loading failed. Install one of:\n"
        "  - soundfile (and an MP3-enabled libsndfile build)\n"
        "  - pydub + ffmpeg (recommended for Windows)\n"
        "  - librosa + ffmpeg/gstreamer\n"
        "Then retry load_mp3()."
    )

def load_audio(path: str, target_sr: Optional[int] = None, target_channels: Optional[int] = None) -> np.ndarray:
    """
    Unified audio loader. Supports:
      - WAV via stdlib wave (always available)
      - MP3 via optional dependencies (soundfile / pydub / librosa)

    Returns float32 (frames, channels) in [-1, 1].
    """
    ext = _ext_lower(path)
    if ext == "wav":
        return load_wav(path, target_sr=target_sr, target_channels=target_channels)
    if ext == "mp3":
        return load_mp3(path, target_sr=target_sr, target_channels=target_channels)
    # fallback: try soundfile first for other formats if installed
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        x = data.astype(np.float32, copy=False)
        if target_sr is not None and int(target_sr) != int(sr):
            x = _linear_resample(x, src_sr=int(sr), dst_sr=int(target_sr))
        if target_channels is not None:
            x = _adapt_channels(x, int(target_channels))
        return x
    except Exception:
        raise ValueError(f"Unsupported audio extension '{ext}'. Use WAV or MP3, or install soundfile for more formats.")


# Utility
def make_sine(freq: float, seconds: float, sr: int, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    x = np.sin(2.0 * np.pi * float(freq) * t).astype(np.float32)
    if channels == 2:
        return np.column_stack([x, x])
    return x.reshape(-1, 1)


# =============================================================================
# TODO / next extensions (kept close to code so structure is implicit)
# =============================================================================
# 1) Transport + Song Model
#    - Engine.transport: play/pause/seek/loop regions
#    - Song: sections, bars/beats mapping -> absolute sample conversion
#
# 2) MIDI + Sequencer Overlay
#    - Convert MIDI events to TimedEvent(kind="midi", payload=...)
#    - A SynthSource that renders voices during Mixer.fill_ahead() using EventWheel
#
# 3) Bus / Group Routing (drums/bass/vocals)
#    - eng.buses.create("drums"), route tracks to buses, bus gain/pan
#
# 4) FX (non-real-time first)
#    - Per-track offline FX renders to overlay audio blocks
#    - Real-time FX later with careful RT constraints
#
# 5) Better resampling
#    - Swap _linear_resample with a higher quality resampler when needed



# --- Signature ---
SIGNATURE = "Spectro BufferKit — GPT-5.2 Thinking (precision + playhead discipline)"

def demo():
    eng = Engine(sample_rate=44100, channels=2).enable_sim_audio().start(mix_tick_hz=400.0)
    eng.tracks.loop("a", make_sine(220, 0.5, 44100, 2), gain=0.2)
    eng.tracks.loop("b", make_sine(330, 0.25, 44100, 2), gain=0.15)

    for i in range(8):
        eng.sync()
        snap = eng.context.snapshot()
        eng.overlays.event(snap.playhead_samples + 2000, "marker", {"i": i})
        print(snap)
        time.sleep(0.05)

    eng.stop()

if __name__ == "__main__":
    demo()

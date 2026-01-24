"""
SPECTRO Input Buffers v2
========================

Circular buffers for MIDI and Audio with LATE-BINDING event firing.

KEY INSIGHT: Callbacks are registered early, but parameters are assembled
and passed at EXECUTION TIME by the event loop manager.

The flow:
    1. REGISTRATION: callback stored, no parameters bound
    2. EVENT ARRIVES: written to ring buffer with raw data
    3. EXECUTION: manager assembles ExecutionContext with CURRENT state
    4. FIRE: callback receives context with fresh parameters

This ensures callbacks always see current state, not stale captured values.

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Optional, Tuple, Callable, Iterator, 
    Dict, Any, Protocol, TYPE_CHECKING
)
from enum import Enum, auto
import numpy as np
import time as time_module
import threading

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


# =============================================================================
# PART 1: Core Types
# =============================================================================

class MidiEventType(Enum):
    """MIDI event types."""
    NOTE_ON = auto()
    NOTE_OFF = auto()
    CONTROL_CHANGE = auto()
    PROGRAM_CHANGE = auto()
    PITCH_BEND = auto()
    AFTERTOUCH = auto()
    SYSEX = auto()


@dataclass(slots=True)
class MidiEvent:
    """
    Raw MIDI event data.
    
    This is the PINBALL - the thing that sits in the buffer waiting to fire.
    It contains only the raw MIDI data, not execution context.
    """
    event_type: MidiEventType
    channel: int
    data1: int      # note number, controller number
    data2: int      # velocity, controller value
    timestamp_samples: int
    timestamp_beats: float = 0.0
    device_id: int = 0
    flags: int = 0
    
    # Flags
    FLAG_LINKED = 0x01
    FLAG_PROCESSED = 0x02
    FLAG_GENERATED = 0x04
    
    @property
    def note(self) -> int:
        return self.data1
    
    @property
    def velocity(self) -> int:
        return self.data2
    
    @property
    def controller(self) -> int:
        return self.data1
    
    @property
    def value(self) -> int:
        return self.data2
    
    @property
    def is_processed(self) -> bool:
        return bool(self.flags & self.FLAG_PROCESSED)
    
    @property
    def is_linked(self) -> bool:
        return bool(self.flags & self.FLAG_LINKED)
    
    def mark_processed(self):
        self.flags |= self.FLAG_PROCESSED


@dataclass
class TransportSnapshot:
    """
    Immutable snapshot of transport state at a moment in time.
    
    This is captured at EXECUTION TIME and passed to callbacks.
    """
    playing: bool
    beat: float
    bar: int
    beat_in_bar: int
    phase_in_beat: float      # 0.0 - 1.0, where in the beat we are
    phase_in_bar: float       # 0.0 - 1.0, where in the bar we are
    bpm: float
    sample_rate: int
    playhead_samples: int
    time_seconds: float
    
    @classmethod
    def capture(cls, sync: 'EventDispatcher') -> 'TransportSnapshot':
        """Capture current state from sync controller."""
        beat = sync.playhead_beats
        beats_per_bar = sync._beats_per_bar
        bar = int(beat / beats_per_bar)
        beat_in_bar = int(beat % beats_per_bar)
        
        return cls(
            playing=sync._playing,
            beat=beat,
            bar=bar,
            beat_in_bar=beat_in_bar,
            phase_in_beat=beat % 1.0,
            phase_in_bar=(beat % beats_per_bar) / beats_per_bar,
            bpm=sync._bpm,
            sample_rate=sync._sample_rate,
            playhead_samples=sync.playhead_samples,
            time_seconds=sync.playhead_samples / sync._sample_rate
        )


@dataclass
class ExecutionContext:
    """
    The context passed to callbacks at EXECUTION TIME.
    
    This is assembled fresh by the event loop manager when firing.
    Contains:
    - The raw MIDI event
    - Current transport state (captured at fire time)
    - Timing information
    - Device information
    
    This is the LATE-BOUND PARAMETERS - gathered at execution, not registration.
    """
    # The raw event data
    event: MidiEvent
    
    # Transport state at fire time (FRESH, not stale)
    transport: TransportSnapshot
    
    # Timing
    fire_time: float          # perf_counter timestamp when fired
    latency_samples: int      # how late we are (event timestamp vs playhead)
    latency_ms: float         # latency in milliseconds
    
    # Device info
    device_id: int
    device_name: str
    
    # For Guitar Hero style: timing accuracy
    timing_error_beats: float  # negative = early, positive = late
    timing_error_ms: float
    
    # For linked events
    is_chain_head: bool = True
    chain_depth: int = 0
    parent_event: Optional[MidiEvent] = None


# =============================================================================
# PART 2: Callback Registration (NO parameters bound here)
# =============================================================================

# The callback signature - receives ExecutionContext, not raw event
EventCallback = Callable[[ExecutionContext], None]


@dataclass
class RegisteredCallback:
    """
    A callback registered but NOT YET BOUND to parameters.
    
    At registration time, we only store:
    - The function to call
    - What event types trigger it
    - Optional filters
    
    NO state/parameters are captured here.
    """
    callback: EventCallback
    event_types: set  # Which MidiEventType(s) trigger this
    channels: Optional[set] = None  # Filter by channel (None = all)
    devices: Optional[set] = None   # Filter by device_id (None = all)
    notes: Optional[set] = None     # Filter by note number (None = all)
    priority: int = 0               # Higher = fires first
    enabled: bool = True
    name: str = ""                  # For debugging
    
    def matches(self, event: MidiEvent) -> bool:
        """Check if this callback should fire for the given event."""
        if not self.enabled:
            return False
        if event.event_type not in self.event_types:
            return False
        if self.channels is not None and event.channel not in self.channels:
            return False
        if self.devices is not None and event.device_id not in self.devices:
            return False
        if self.notes is not None and event.data1 not in self.notes:
            return False
        return True


# =============================================================================
# PART 3: MIDI Ring Buffer (The Pinball Chute)
# =============================================================================

class MidiRingBuffer:
    """
    Lock-free circular buffer for MIDI events.
    
    Events are loaded here (pinballs in the chute).
    When the playhead reaches them, they're ready to release.
    
    This buffer does NOT fire callbacks - it just stores events.
    The EventDispatcher handles firing with proper context.
    """
    
    def __init__(self, capacity: int = 4096):
        # Power of 2 for fast modulo
        self.capacity = 1 << (capacity - 1).bit_length()
        self._mask = self.capacity - 1
        
        # Pre-allocated slots
        self._events: List[Optional[MidiEvent]] = [None] * self.capacity
        
        # Read/write positions
        self._write_pos = 0
        self._read_pos = 0
        
        # Playhead (in samples)
        self._playhead_samples = 0
        
        # Device registry
        self._devices: Dict[int, str] = {}
        self._next_device_id = 0
        
        # Linked events: source_idx -> [target_indices]
        self._links: Dict[int, List[int]] = {}
    
    def write(self, event: MidiEvent) -> int:
        """
        Write event to buffer.
        
        Returns the slot index, or -1 if buffer full.
        """
        next_write = (self._write_pos + 1) & self._mask
        if next_write == self._read_pos:
            return -1  # Full
        
        slot = self._write_pos
        self._events[slot] = event
        self._write_pos = next_write
        
        return slot
    
    def pop_ready(self) -> List[Tuple[int, MidiEvent]]:
        """
        Pop events that are ready (timestamp <= playhead).
        
        Returns list of (slot_index, event) tuples.
        The slot_index is needed for link resolution.
        
        NOTE: Events that are LINKED TO (i.e., children in a chain) are NOT
        returned here. They will fire through their parent's chain traversal.
        This prevents double-firing of linked events.
        """
        ready = []
        
        # Build set of slots that are link targets (children)
        link_targets = set[Any]()
        for targets in self._links.values():
            link_targets.update(targets)
        
        while self._read_pos != self._write_pos:
            event = self._events[self._read_pos]
            slot = self._read_pos
            
            if event is None:
                self._read_pos = (self._read_pos + 1) & self._mask
                continue
            
            if event.timestamp_samples > self._playhead_samples:
                break  # Not ready yet
            
            if not event.is_processed:
                # Skip if this is a link target - it fires through its parent
                if slot not in link_targets:
                    ready.append((slot, event))
                    event.mark_processed()
            
            self._read_pos = (self._read_pos + 1) & self._mask
        
        return ready
    
    def get_linked(self, slot: int) -> List[Tuple[int, MidiEvent]]:
        """Get events linked to the given slot."""
        linked = []
        if slot in self._links:
            for target_slot in self._links[slot]:
                event = self._events[target_slot]
                if event and not event.is_processed:
                    linked.append((target_slot, event))
        return linked
    
    def link(self, source_slot: int, target_slot: int):
        """Link two events (source fires -> target fires too)."""
        if source_slot not in self._links:
            self._links[source_slot] = []
        self._links[source_slot].append(target_slot)
        
        # Mark source as having links
        source = self._events[source_slot]
        if source:
            source.flags |= MidiEvent.FLAG_LINKED
    
    def advance_playhead(self, samples: int):
        """Set playhead to absolute sample position."""
        self._playhead_samples = samples
    
    def register_device(self, name: str) -> int:
        """Register a device, returns device_id."""
        device_id = self._next_device_id
        self._next_device_id += 1
        self._devices[device_id] = name
        return device_id
    
    def get_device_name(self, device_id: int) -> str:
        """Get device name by ID."""
        return self._devices.get(device_id, f"Unknown({device_id})")
    
    def clear(self):
        """Clear buffer and reset."""
        self._read_pos = self._write_pos
        self._links.clear()
        for i in range(self.capacity):
            if self._events[i]:
                self._events[i].flags &= ~MidiEvent.FLAG_PROCESSED
    
    @property
    def available(self) -> int:
        if self._write_pos >= self._read_pos:
            return self._write_pos - self._read_pos
        return self.capacity - self._read_pos + self._write_pos
    
    @property
    def playhead_samples(self) -> int:
        return self._playhead_samples


# =============================================================================
# PART 4: Audio Ring Buffer
# =============================================================================

class AudioRingBuffer:
    """
    Circular buffer for audio samples.
    
    Syncs with MIDI buffer via sample position.
    When audio advances, MIDI playhead advances too.
    """
    
    def __init__(self, capacity_samples: int = 65536, channels: int = 2):
        self.capacity = 1 << (capacity_samples - 1).bit_length()
        self._mask = self.capacity - 1
        self.channels = channels
        
        self._samples = np.zeros((self.capacity, channels), dtype=np.float32)
        
        self._write_pos = 0
        self._read_pos = 0
        self._playhead_samples = 0
        
        self._sample_rate = 44100
        self._bpm = 120.0
        
        self._midi_buffer: Optional[MidiRingBuffer] = None
    
    def write(self, samples: np.ndarray) -> int:
        """Write samples to buffer. Returns frames written."""
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
            if self.channels == 2:
                samples = np.column_stack([samples, samples])
        
        n_frames = len(samples)
        space = self.free_space
        
        if n_frames > space:
            n_frames = space
            samples = samples[:n_frames]
        
        if n_frames == 0:
            return 0
        
        end_pos = self._write_pos + n_frames
        
        if end_pos <= self.capacity:
            self._samples[self._write_pos:end_pos] = samples
        else:
            first_part = self.capacity - self._write_pos
            self._samples[self._write_pos:] = samples[:first_part]
            self._samples[:end_pos & self._mask] = samples[first_part:]
        
        self._write_pos = end_pos & self._mask
        return n_frames
    
    def write_silence(self, n_frames: int) -> int:
        """Write silence."""
        return self.write(np.zeros((n_frames, self.channels), dtype=np.float32))
    
    def read(self, n_frames: int) -> np.ndarray:
        """Read samples, advance playhead, sync MIDI buffer."""
        available = self.available
        
        if n_frames > available:
            n_frames = available
        
        if n_frames == 0:
            return np.zeros((0, self.channels), dtype=np.float32)
        
        end_pos = self._read_pos + n_frames
        
        if end_pos <= self.capacity:
            result = self._samples[self._read_pos:end_pos].copy()
        else:
            first_part = self.capacity - self._read_pos
            result = np.vstack([
                self._samples[self._read_pos:],
                self._samples[:end_pos & self._mask]
            ])
        
        self._read_pos = end_pos & self._mask
        self._playhead_samples += n_frames
        
        # SYNC: advance MIDI buffer's playhead
        if self._midi_buffer:
            self._midi_buffer.advance_playhead(self._playhead_samples)
        
        return result
    
    def sync_to_midi(self, midi_buffer: MidiRingBuffer):
        """Link to MIDI buffer for sync."""
        self._midi_buffer = midi_buffer
    
    def set_tempo(self, bpm: float):
        self._bpm = bpm
    
    def set_sample_rate(self, sample_rate: int):
        self._sample_rate = sample_rate
    
    def samples_to_beats(self, samples: int) -> float:
        seconds = samples / self._sample_rate
        return seconds * (self._bpm / 60.0)
    
    def beats_to_samples(self, beats: float) -> int:
        seconds = beats / (self._bpm / 60.0)
        return int(seconds * self._sample_rate)
    
    def seek(self, sample: int):
        """Seek to sample position."""
        self._playhead_samples = sample
        self._read_pos = 0
        self._write_pos = 0
        if self._midi_buffer:
            self._midi_buffer.advance_playhead(sample)
    
    def seek_to_beat(self, beat: float):
        """Seek to beat position."""
        self.seek(self.beats_to_samples(beat))
    
    @property
    def available(self) -> int:
        if self._write_pos >= self._read_pos:
            return self._write_pos - self._read_pos
        return self.capacity - self._read_pos + self._write_pos
    
    @property
    def free_space(self) -> int:
        return self.capacity - self.available - 1
    
    @property
    def playhead_samples(self) -> int:
        return self._playhead_samples
    
    @property
    def playhead_beats(self) -> float:
        return self.samples_to_beats(self._playhead_samples)


# =============================================================================
# PART 4.5: Loopback Audio Wrapper
# =============================================================================

class LoopbackAudioBuffer:
    """
    Wrapper that captures loopback audio into an AudioRingBuffer.

    This uses sounddevice InputStream (WASAPI loopback when available)
    and writes incoming samples into a circular buffer for later reads.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        channels: int = 2,
        capacity_samples: int = 131072,
        device: Optional[int | str] = None,
        blocksize: int = 1024,
        loopback: bool = True,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.device = device
        self.loopback = loopback

        self.buffer = AudioRingBuffer(
            capacity_samples=capacity_samples,
            channels=channels,
        )
        self.buffer.set_sample_rate(sample_rate)

        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start capturing audio into the ring buffer."""
        if not SOUNDDEVICE_AVAILABLE:
            print("LoopbackAudioBuffer: sounddevice not available")
            return False

        if self._stream:
            return True

        extra_settings = None
        if self.loopback:
            wasapi_settings = getattr(sd, "WasapiSettings", None)
            if wasapi_settings:
                extra_settings = sd.WasapiSettings(loopback=True)

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=self._audio_callback,
                extra_settings=extra_settings,
            )
            self._stream.start()
            return True
        except Exception as exc:
            print(f"LoopbackAudioBuffer: failed to start: {exc}")
            self._stream = None
            return False

    def stop(self):
        """Stop capturing audio."""
        if not self._stream:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Audio callback from sounddevice."""
        if status:
            print(f"LoopbackAudioBuffer: {status}")
        if frames <= 0:
            return
        with self._lock:
            self.buffer.write(indata)

    def read(self, n_frames: int) -> np.ndarray:
        """Thread-safe read from the ring buffer."""
        with self._lock:
            return self.buffer.read(n_frames)

    def get_buffer(self) -> AudioRingBuffer:
        """Return the underlying ring buffer."""
        return self.buffer

# =============================================================================
# PART 5: Event Dispatcher (THE MANAGER - assembles context at execution time)
# =============================================================================

class EventDispatcher:
    """
    The event loop manager that FIRES callbacks with LATE-BOUND parameters.
    
    This is where the magic happens:
    1. Callbacks are registered (no parameters bound)
    2. Events arrive in the MIDI buffer
    3. When ready, dispatcher:
       a. Captures current TransportSnapshot
       b. Assembles ExecutionContext with FRESH state
       c. Fires callback with context
    
    The callback receives parameters that are current AT FIRE TIME,
    not stale values captured at registration.
    """
    
    def __init__(self, midi_buffer: MidiRingBuffer, audio_buffer: AudioRingBuffer):
        self.midi_buffer = midi_buffer
        self.audio_buffer = audio_buffer
        
        # Link buffers
        audio_buffer.sync_to_midi(midi_buffer)
        
        # Transport state
        self._playing = False
        self._bpm = 120.0
        self._sample_rate = 44100
        self._beats_per_bar = 4
        
        # Registered callbacks (NO parameters bound)
        self._callbacks: List[RegisteredCallback] = []
        
        # Special callbacks for beat/bar (also late-bound)
        self._beat_callbacks: List[Callable[[int, TransportSnapshot], None]] = []
        self._bar_callbacks: List[Callable[[int, TransportSnapshot], None]] = []
        
        # Tracking
        self._last_beat = -1
        self._last_bar = -1
    
    # -------------------------------------------------------------------------
    # Callback Registration (NO parameters bound here)
    # -------------------------------------------------------------------------
    
    def register(
        self,
        callback: EventCallback,
        event_types: set = None,
        channels: set = None,
        devices: set = None,
        notes: set = None,
        priority: int = 0,
        name: str = ""
    ) -> RegisteredCallback:
        """
        Register a callback for MIDI events.
        
        IMPORTANT: No parameters are bound here. The callback will receive
        an ExecutionContext with FRESH state when it fires.
        
        Args:
            callback: Function that takes ExecutionContext
            event_types: Which event types trigger this (default: all)
            channels: Filter by MIDI channel (default: all)
            devices: Filter by device_id (default: all)
            notes: Filter by note number (default: all)
            priority: Higher fires first
            name: For debugging
            
        Returns:
            The RegisteredCallback (can be used to unregister)
        """
        if event_types is None:
            event_types = set(MidiEventType)
        
        reg = RegisteredCallback(
            callback=callback,
            event_types=event_types,
            channels=channels,
            devices=devices,
            notes=notes,
            priority=priority,
            name=name
        )
        
        self._callbacks.append(reg)
        # Sort by priority (highest first)
        self._callbacks.sort(key=lambda c: -c.priority)
        
        return reg
    
    def unregister(self, reg: RegisteredCallback):
        """Remove a registered callback."""
        if reg in self._callbacks:
            self._callbacks.remove(reg)
    
    def on_beat(self, callback: Callable[[int, TransportSnapshot], None]):
        """Register beat callback. Receives (beat_number, transport_snapshot)."""
        self._beat_callbacks.append(callback)
    
    def on_bar(self, callback: Callable[[int, TransportSnapshot], None]):
        """Register bar callback. Receives (bar_number, transport_snapshot)."""
        self._bar_callbacks.append(callback)
    
    # -------------------------------------------------------------------------
    # Transport Control
    # -------------------------------------------------------------------------
    
    def play(self):
        self._playing = True
    
    def pause(self):
        self._playing = False
    
    def stop(self):
        self._playing = False
        self.seek_to_beat(0.0)
    
    def seek_to_beat(self, beat: float):
        self.audio_buffer.seek_to_beat(beat)
        self._last_beat = int(beat) - 1
        self._last_bar = int(beat / self._beats_per_bar) - 1
    
    def set_bpm(self, bpm: float):
        self._bpm = bpm
        self.audio_buffer.set_tempo(bpm)
    
    # -------------------------------------------------------------------------
    # The Main Event Loop
    # -------------------------------------------------------------------------
    
    def process_frame(self, dt: float) -> List[ExecutionContext]:
        """
        Process one frame.
        
        This is where LATE BINDING happens:
        1. Advance audio (which advances MIDI playhead)
        2. Pop ready MIDI events
        3. For each event:
           a. Capture CURRENT transport state
           b. Assemble ExecutionContext with FRESH parameters
           c. Fire matching callbacks with context
        
        Returns list of ExecutionContexts that were fired (for external use).
        """
        if not self._playing:
            return []
        
        fired_contexts = []
        fire_time = time_module.perf_counter()
        
        # Convert dt to samples and advance audio
        frame_samples = int(dt * self._sample_rate)
        if self.audio_buffer.available < frame_samples:
            self.audio_buffer.write_silence(frame_samples)
        
        self.audio_buffer.read(frame_samples)
        
        # Pop ready events from MIDI buffer
        ready_events = self.midi_buffer.pop_ready()
        
        # Process each event
        for slot, event in ready_events:
            # Fire this event and any linked events
            contexts = self._fire_event_chain(slot, event, fire_time)
            fired_contexts.extend(contexts)
        
        # Check beat/bar boundaries
        self._check_boundaries(fire_time)
        
        return fired_contexts
    
    def _fire_event_chain(
        self,
        slot: int,
        event: MidiEvent,
        fire_time: float,
        chain_depth: int = 0,
        parent: Optional[MidiEvent] = None
    ) -> List[ExecutionContext]:
        """
        Fire an event and its linked events.
        
        LATE BINDING: ExecutionContext is assembled HERE with current state.
        """
        contexts = []
        
        # ===== LATE BINDING: Assemble ExecutionContext with FRESH state =====
        
        transport = TransportSnapshot.capture(self)
        
        # Calculate timing error (for Guitar Hero accuracy)
        event_beat = event.timestamp_beats
        current_beat = transport.beat
        timing_error_beats = current_beat - event_beat  # positive = late
        timing_error_ms = timing_error_beats * (60000.0 / self._bpm)
        
        # Calculate latency
        latency_samples = self.playhead_samples - event.timestamp_samples
        latency_ms = (latency_samples / self._sample_rate) * 1000.0
        
        context = ExecutionContext(
            event=event,
            transport=transport,
            fire_time=fire_time,
            latency_samples=latency_samples,
            latency_ms=latency_ms,
            device_id=event.device_id,
            device_name=self.midi_buffer.get_device_name(event.device_id),
            timing_error_beats=timing_error_beats,
            timing_error_ms=timing_error_ms,
            is_chain_head=(parent is None),
            chain_depth=chain_depth,
            parent_event=parent
        )
        
        # ===== Fire matching callbacks with the FRESH context =====
        
        for reg in self._callbacks:
            if reg.matches(event):
                try:
                    reg.callback(context)
                except Exception as e:
                    print(f"Callback error ({reg.name}): {e}")
        
        contexts.append(context)
        
        # ===== Fire linked events (recursively) =====
        
        linked = self.midi_buffer.get_linked(slot)
        for linked_slot, linked_event in linked:
            linked_event.mark_processed()
            child_contexts = self._fire_event_chain(
                linked_slot, 
                linked_event, 
                fire_time,
                chain_depth=chain_depth + 1,
                parent=event
            )
            contexts.extend(child_contexts)
        
        return contexts
    
    def _check_boundaries(self, fire_time: float):
        """Check and fire beat/bar callbacks with fresh state."""
        current_beat = int(self.playhead_beats)
        current_bar = int(self.playhead_beats / self._beats_per_bar)
        
        if current_beat > self._last_beat:
            self._last_beat = current_beat
            # Late binding: capture fresh state for beat callbacks
            transport = TransportSnapshot.capture(self)
            for callback in self._beat_callbacks:
                try:
                    callback(current_beat, transport)
                except Exception as e:
                    print(f"Beat callback error: {e}")
        
        if current_bar > self._last_bar:
            self._last_bar = current_bar
            transport = TransportSnapshot.capture(self)
            for callback in self._bar_callbacks:
                try:
                    callback(current_bar, transport)
                except Exception as e:
                    print(f"Bar callback error: {e}")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def playhead_samples(self) -> int:
        return self.audio_buffer.playhead_samples
    
    @property
    def playhead_beats(self) -> float:
        return self.audio_buffer.playhead_beats
    
    @property
    def playing(self) -> bool:
        return self._playing


# =============================================================================
# PART 6: Input Device
# =============================================================================

class InputDevice:
    """
    Base class for input devices.
    
    Devices emit events into the MIDI buffer.
    The EventDispatcher fires callbacks with late-bound context.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.device_id: int = -1
        self._buffer: Optional[MidiRingBuffer] = None
        self._audio: Optional[AudioRingBuffer] = None
    
    def connect(self, midi_buffer: MidiRingBuffer, audio_buffer: AudioRingBuffer) -> int:
        """Connect device to buffers. Returns device_id."""
        self._buffer = midi_buffer
        self._audio = audio_buffer
        self.device_id = midi_buffer.register_device(self.name)
        return self.device_id
    
    def emit(self, event_type: MidiEventType, channel: int, data1: int, data2: int):
        """Emit an event from this device."""
        if not self._buffer or not self._audio:
            return
        
        event = MidiEvent(
            event_type=event_type,
            channel=channel,
            data1=data1,
            data2=data2,
            timestamp_samples=self._audio.playhead_samples,
            timestamp_beats=self._audio.playhead_beats,
            device_id=self.device_id
        )
        
        self._buffer.write(event)
    
    def note_on(self, note: int, velocity: int, channel: int = 0):
        """Convenience: emit note on."""
        self.emit(MidiEventType.NOTE_ON, channel, note, velocity)
    
    def note_off(self, note: int, channel: int = 0):
        """Convenience: emit note off."""
        self.emit(MidiEventType.NOTE_OFF, channel, note, 0)
    
    def control_change(self, controller: int, value: int, channel: int = 0):
        """Convenience: emit control change."""
        self.emit(MidiEventType.CONTROL_CHANGE, channel, controller, value)


# =============================================================================
# PART 7: Example / Test
# =============================================================================

def example():
    """
    Demonstrates late-binding event firing.
    """
    print("=== SPECTRO Event System v2 ===\n")
    
    # Create buffers
    midi = MidiRingBuffer(capacity=4096)
    audio = AudioRingBuffer(capacity_samples=65536)
    
    # Create dispatcher (THE MANAGER)
    dispatcher = EventDispatcher(midi, audio)
    dispatcher.set_bpm(120.0)
    
    # Create a device
    drum_pad = InputDevice("Drum Pad")
    drum_pad.connect(midi, audio)
    
    # =========================================================================
    # Register callbacks - NO PARAMETERS BOUND YET
    # =========================================================================
    
    def on_drum_hit(ctx: ExecutionContext):
        """
        This callback receives ExecutionContext with FRESH state.
        
        The transport snapshot, timing error, etc. are all captured
        at FIRE TIME, not at registration time.
        """
        print(f"DRUM HIT!")
        print(f"  Note: {ctx.event.note}, Velocity: {ctx.event.velocity}")
        print(f"  Current beat: {ctx.transport.beat:.3f}")
        print(f"  Timing error: {ctx.timing_error_ms:+.1f}ms")
        print(f"  Fire time: {ctx.fire_time:.6f}")
        print()
    
    dispatcher.register(
        callback=on_drum_hit,
        event_types={MidiEventType.NOTE_ON},
        name="drum_handler"
    )
    
    def on_beat(beat: int, transport: TransportSnapshot):
        """Beat callback also gets fresh transport snapshot."""
        print(f"BEAT {beat} | bar={transport.bar} phase={transport.phase_in_bar:.2f}")
    
    dispatcher.on_beat(on_beat)
    
    # =========================================================================
    # Simulate playback
    # =========================================================================
    
    print("Starting playback...\n")
    
    # Fill audio buffer
    audio.write_silence(44100)
    
    # Hit some drums at different times
    drum_pad.note_on(36, 100)  # Kick
    
    dispatcher.play()
    
    # Process frames
    for frame in range(30):
        contexts = dispatcher.process_frame(1/60)  # 60fps
        
        # Hit another drum mid-playback
        if frame == 15:
            drum_pad.note_on(38, 80)  # Snare
    
    print(f"\nFinal playhead: {dispatcher.playhead_beats:.3f} beats")
    print("=== Done ===")


if __name__ == "__main__":
    example()

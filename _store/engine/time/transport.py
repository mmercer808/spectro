# engine/time/transport.py
"""
Transport - Playback state and control.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, List


@dataclass(frozen=True)
class TimeSignature:
    numerator: int = 4
    denominator: int = 4
    
    @property
    def beats_per_bar(self) -> float:
        return self.numerator * (4 / self.denominator)
    
    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class TransportState:
    """Immutable snapshot of transport state."""
    playing: bool
    playhead_beat: float
    playhead_time: float
    bpm: float
    time_sig: TimeSignature
    loop_start: Optional[float]
    loop_end: Optional[float]
    loop_enabled: bool
    
    @property
    def phase_in_beat(self) -> float:
        return self.playhead_beat % 1.0
    
    @property
    def phase_in_bar(self) -> float:
        beats_per_bar = self.time_sig.beats_per_bar
        return (self.playhead_beat % beats_per_bar) / beats_per_bar
    
    @property
    def current_bar(self) -> int:
        return int(self.playhead_beat / self.time_sig.beats_per_bar)
    
    @property
    def current_beat_in_bar(self) -> int:
        return int(self.playhead_beat % self.time_sig.beats_per_bar)
    
    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self.bpm
    
    @property
    def beats_per_second(self) -> float:
        return self.bpm / 60.0
    
    @property
    def bar_duration_beats(self) -> float:
        return self.time_sig.beats_per_bar
    
    @property
    def bar_duration_seconds(self) -> float:
        return self.bar_duration_beats * self.seconds_per_beat


BeatCallback = Callable[[int, TransportState], None]
BarCallback = Callable[[int, TransportState], None]
LoopCallback = Callable[[TransportState], None]
StateCallback = Callable[[TransportState], None]


class Transport:
    """Mutable transport control."""
    
    def __init__(self, bpm: float = 120.0, time_sig: TimeSignature = None):
        self.playing: bool = False
        self.playhead_beat: float = 0.0
        self.playhead_time: float = 0.0
        self.bpm: float = bpm
        self.time_sig: TimeSignature = time_sig or TimeSignature()
        self.loop_start: Optional[float] = None
        self.loop_end: Optional[float] = None
        self.loop_enabled: bool = False
        
        self.on_beat_callbacks: List[BeatCallback] = []
        self.on_bar_callbacks: List[BarCallback] = []
        self.on_loop_callbacks: List[LoopCallback] = []
        self.on_play_callbacks: List[StateCallback] = []
        self.on_pause_callbacks: List[StateCallback] = []
        self.on_stop_callbacks: List[StateCallback] = []
        self.on_seek_callbacks: List[StateCallback] = []
        
        self._last_beat: int = 0
        self._last_bar: int = 0
    
    def play(self):
        if not self.playing:
            self.playing = True
            state = self._snapshot()
            for callback in self.on_play_callbacks:
                callback(state)
    
    def pause(self):
        if self.playing:
            self.playing = False
            state = self._snapshot()
            for callback in self.on_pause_callbacks:
                callback(state)
    
    def stop(self):
        self.playing = False
        
        if self.loop_enabled and self.loop_start is not None:
            self.playhead_beat = self.loop_start
        else:
            self.playhead_beat = 0.0
        
        self.playhead_time = self.playhead_beat * (60.0 / self.bpm)
        self._last_beat = int(self.playhead_beat)
        self._last_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
        
        state = self._snapshot()
        for callback in self.on_stop_callbacks:
            callback(state)
    
    def toggle(self):
        if self.playing:
            self.pause()
        else:
            self.play()
    
    def seek_to_beat(self, beat: float):
        self.playhead_beat = max(0.0, beat)
        self.playhead_time = self.playhead_beat * (60.0 / self.bpm)
        self._last_beat = int(self.playhead_beat)
        self._last_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
        
        state = self._snapshot()
        for callback in self.on_seek_callbacks:
            callback(state)
    
    def seek_to_time(self, seconds: float):
        self.playhead_time = max(0.0, seconds)
        self.playhead_beat = self.playhead_time * (self.bpm / 60.0)
        self._last_beat = int(self.playhead_beat)
        self._last_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
        
        state = self._snapshot()
        for callback in self.on_seek_callbacks:
            callback(state)
    
    def seek_by_bars(self, delta: int):
        new_beat = self.playhead_beat + (delta * self.time_sig.beats_per_bar)
        self.seek_to_beat(new_beat)
    
    def seek_to_bar(self, bar: int):
        beat = bar * self.time_sig.beats_per_bar
        self.seek_to_beat(beat)
    
    def seek_to_next_bar(self):
        current_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
        self.seek_to_bar(current_bar + 1)
    
    def seek_to_previous_bar(self):
        beats_per_bar = self.time_sig.beats_per_bar
        current_bar = int(self.playhead_beat / beats_per_bar)
        beat_in_bar = self.playhead_beat % beats_per_bar
        
        if beat_in_bar < 0.1:
            self.seek_to_bar(max(0, current_bar - 1))
        else:
            self.seek_to_bar(current_bar)
    
    def set_bpm(self, bpm: float):
        self.bpm = max(20.0, min(bpm, 999.0))
        self.playhead_time = self.playhead_beat * (60.0 / self.bpm)
    
    def set_time_signature(self, numerator: int, denominator: int):
        self.time_sig = TimeSignature(numerator, denominator)
        self._last_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
    
    def tap_tempo(self, tap_times: List[float]) -> float:
        if len(tap_times) < 2:
            return self.bpm
        
        intervals = []
        for i in range(1, len(tap_times)):
            intervals.append(tap_times[i] - tap_times[i-1])
        
        avg_interval = sum(intervals) / len(intervals)
        
        if avg_interval > 0:
            new_bpm = 60.0 / avg_interval
            self.set_bpm(new_bpm)
            return new_bpm
        
        return self.bpm
    
    def set_loop(self, start: float, end: float):
        self.loop_start = min(start, end)
        self.loop_end = max(start, end)
    
    def clear_loop(self):
        self.loop_start = None
        self.loop_end = None
        self.loop_enabled = False
    
    def toggle_loop(self):
        self.loop_enabled = not self.loop_enabled
    
    def update(self, dt: float) -> TransportState:
        if self.playing:
            beats_delta = dt * (self.bpm / 60.0)
            self.playhead_beat += beats_delta
            self.playhead_time += dt
            
            if self.loop_enabled and self.loop_end is not None:
                if self.playhead_beat >= self.loop_end:
                    self.playhead_beat = self.loop_start or 0.0
                    self.playhead_time = self.playhead_beat * (60.0 / self.bpm)
                    
                    self._last_beat = int(self.playhead_beat)
                    self._last_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
                    
                    state = self._snapshot()
                    for callback in self.on_loop_callbacks:
                        callback(state)
                    return state
            
            current_beat = int(self.playhead_beat)
            current_bar = int(self.playhead_beat / self.time_sig.beats_per_bar)
            
            state = self._snapshot()
            
            if current_beat != self._last_beat:
                for callback in self.on_beat_callbacks:
                    callback(current_beat, state)
                self._last_beat = current_beat
            
            if current_bar != self._last_bar:
                for callback in self.on_bar_callbacks:
                    callback(current_bar, state)
                self._last_bar = current_bar
            
            return state
        
        return self._snapshot()
    
    def _snapshot(self) -> TransportState:
        return TransportState(
            playing=self.playing,
            playhead_beat=self.playhead_beat,
            playhead_time=self.playhead_time,
            bpm=self.bpm,
            time_sig=self.time_sig,
            loop_start=self.loop_start,
            loop_end=self.loop_end,
            loop_enabled=self.loop_enabled,
        )
    
    def on_beat(self, callback: BeatCallback):
        self.on_beat_callbacks.append(callback)
        return callback
    
    def on_bar(self, callback: BarCallback):
        self.on_bar_callbacks.append(callback)
        return callback
    
    def on_loop(self, callback: LoopCallback):
        self.on_loop_callbacks.append(callback)
        return callback
    
    def beat_to_time(self, beat: float) -> float:
        return beat * (60.0 / self.bpm)
    
    def time_to_beat(self, time: float) -> float:
        return time * (self.bpm / 60.0)
    
    def beat_to_bar(self, beat: float) -> int:
        return int(beat / self.time_sig.beats_per_bar)
    
    def bar_to_beat(self, bar: int) -> float:
        return bar * self.time_sig.beats_per_bar
    
    def format_time(self, beat: float = None) -> str:
        if beat is None:
            beat = self.playhead_beat
        seconds = self.beat_to_time(beat)
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"
    
    def format_bar_beat(self, beat: float = None) -> str:
        if beat is None:
            beat = self.playhead_beat
        bar = int(beat / self.time_sig.beats_per_bar) + 1
        beat_in_bar = int(beat % self.time_sig.beats_per_bar) + 1
        return f"{bar}:{beat_in_bar}"

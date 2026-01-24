# engine/core/signal.py
"""
SignalBridge - Observer pattern hub for routing events between components.
"""

from __future__ import annotations
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass
from weakref import ref
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Signal Types
# =============================================================================

SIGNAL_DT = 'dt'
SIGNAL_TRANSPORT_CHANGED = 'transport_changed'
SIGNAL_SEEK = 'seek'
SIGNAL_PLAY = 'play'
SIGNAL_PAUSE = 'pause'
SIGNAL_STOP = 'stop'
SIGNAL_BPM_CHANGED = 'bpm_changed'
SIGNAL_TIME_SIG_CHANGED = 'time_sig_changed'
SIGNAL_VIEW_CHANGED = 'view_changed'
SIGNAL_ZOOM_CHANGED = 'zoom_changed'
SIGNAL_SCROLL = 'scroll'
SIGNAL_POINTER_DOWN = 'pointer_down'
SIGNAL_POINTER_MOVE = 'pointer_move'
SIGNAL_POINTER_UP = 'pointer_up'
SIGNAL_KEY_DOWN = 'key_down'
SIGNAL_KEY_UP = 'key_up'
SIGNAL_WHEEL = 'wheel'
SIGNAL_ENTITY_ADDED = 'entity_added'
SIGNAL_ENTITY_REMOVED = 'entity_removed'
SIGNAL_ENTITY_CHANGED = 'entity_changed'
SIGNAL_SELECTION_CHANGED = 'selection_changed'
SIGNAL_DIRTY = 'dirty'
SIGNAL_RESIZE = 'resize'

# MIDI Signals
SIGNAL_MIDI_NOTE_ON = 'midi_note_on'        # (note, velocity, channel)
SIGNAL_MIDI_NOTE_OFF = 'midi_note_off'      # (note, velocity, channel)
SIGNAL_MIDI_CC = 'midi_cc'                  # (cc_number, value, channel)
SIGNAL_MIDI_PAD = 'midi_pad'                # (row, col, velocity) - Grid controllers
SIGNAL_MIDI_CLOCK = 'midi_clock'            # () - 24 PPQ clock tick
SIGNAL_MIDI_CONNECTED = 'midi_connected'    # (port_name,)
SIGNAL_MIDI_DISCONNECTED = 'midi_disconnected'  # (port_name,)


# =============================================================================
# Connection Handle
# =============================================================================

@dataclass
class Connection:
    """Handle to a signal connection."""
    signal: str
    callback_id: int
    bridge: SignalBridge = None
    
    def disconnect(self):
        if self.bridge:
            self.bridge._remove_connection(self.signal, self.callback_id)
            self.bridge = None


# =============================================================================
# Signal Bridge
# =============================================================================

class SignalBridge:
    """Central hub for signal routing."""
    
    def __init__(self):
        self._connections: Dict[str, Dict[int, Callable]] = {}
        self._next_id: int = 0
        self._blocked: set = set()
        self._emit_depth: int = 0
        self._pending_removes: List[tuple] = []
        
    def connect(self, signal: str, handler: Callable) -> Connection:
        if signal not in self._connections:
            self._connections[signal] = {}
        
        callback_id = self._next_id
        self._next_id += 1
        
        self._connections[signal][callback_id] = handler
        
        return Connection(signal=signal, callback_id=callback_id, bridge=self)
    
    def connect_weak(self, signal: str, obj: object, method_name: str) -> Connection:
        weak_obj = ref(obj)
        
        def weak_handler(*args, **kwargs):
            strong = weak_obj()
            if strong is not None:
                method = getattr(strong, method_name, None)
                if method:
                    method(*args, **kwargs)
        
        return self.connect(signal, weak_handler)
    
    def disconnect_all(self, signal: str = None):
        if signal:
            self._connections.pop(signal, None)
        else:
            self._connections.clear()
    
    def emit(self, signal: str, *args, **kwargs):
        if signal in self._blocked:
            return
        
        handlers = self._connections.get(signal, {})
        if not handlers:
            return
        
        self._emit_depth += 1
        
        try:
            for callback_id, handler in list(handlers.items()):
                try:
                    handler(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Signal handler error [{signal}]: {e}")
        finally:
            self._emit_depth -= 1
            
            if self._emit_depth == 0 and self._pending_removes:
                for sig, cid in self._pending_removes:
                    self._do_remove(sig, cid)
                self._pending_removes.clear()
    
    def emit_later(self, signal: str, *args, **kwargs):
        self.emit(signal, *args, **kwargs)
    
    def block(self, signal: str):
        self._blocked.add(signal)
    
    def unblock(self, signal: str):
        self._blocked.discard(signal)
    
    def is_connected(self, signal: str) -> bool:
        return bool(self._connections.get(signal))
    
    def _remove_connection(self, signal: str, callback_id: int):
        if self._emit_depth > 0:
            self._pending_removes.append((signal, callback_id))
        else:
            self._do_remove(signal, callback_id)
    
    def _do_remove(self, signal: str, callback_id: int):
        if signal in self._connections:
            self._connections[signal].pop(callback_id, None)


# =============================================================================
# Signal Debugger
# =============================================================================

class SignalDebugger:
    """Debug wrapper that logs all signal activity."""
    
    def __init__(self, bridge: SignalBridge):
        self.bridge = bridge
        self._original_emit = bridge.emit
        self._watched: set = set()
        self._watch_all: bool = False
        bridge.emit = self._debug_emit
    
    def watch(self, signal: str):
        self._watched.add(signal)
    
    def unwatch(self, signal: str):
        self._watched.discard(signal)
    
    def watch_all(self, enabled: bool = True):
        self._watch_all = enabled
    
    def _debug_emit(self, signal: str, *args, **kwargs):
        if self._watch_all or signal in self._watched:
            args_str = ', '.join(repr(a) for a in args)
            kwargs_str = ', '.join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ', '.join(filter(None, [args_str, kwargs_str]))
            logger.debug(f"SIGNAL: {signal}({all_args})")
        
        self._original_emit(signal, *args, **kwargs)
    
    def detach(self):
        self.bridge.emit = self._original_emit


# =============================================================================
# Convenience
# =============================================================================

def on_signal(bridge: SignalBridge, signal: str):
    """Decorator to connect a function to a signal."""
    def decorator(func):
        bridge.connect(signal, func)
        return func
    return decorator


class SignalEmitter:
    """Mixin class for objects that emit signals."""
    
    _bridge: SignalBridge = None
    
    def bind_bridge(self, bridge: SignalBridge):
        self._bridge = bridge
    
    def emit(self, signal: str, *args, **kwargs):
        if self._bridge:
            self._bridge.emit(signal, *args, **kwargs)
    
    def connect(self, signal: str, handler: Callable) -> Optional[Connection]:
        if self._bridge:
            return self._bridge.connect(signal, handler)
        return None


class SignalReceiver:
    """Mixin class for objects that receive signals."""
    
    _bridge: SignalBridge = None
    _connections: List[Connection] = None
    
    def bind_bridge(self, bridge: SignalBridge):
        self._bridge = bridge
        self._connections = []
    
    def subscribe(self, signal: str, handler: Callable):
        if self._bridge:
            conn = self._bridge.connect(signal, handler)
            if self._connections is not None:
                self._connections.append(conn)
    
    def unsubscribe_all(self):
        if self._connections:
            for conn in self._connections:
                conn.disconnect()
            self._connections.clear()

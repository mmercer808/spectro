"""
DR BEAT GRAPH UI
================

A hybrid architecture combining:
- GraphObject for sequencer logic (message passing, pipelines, prototype inheritance)
- SceneGraph for UI rendering (depth-based ordering, components, event-driven)

This creates a complete finger drumming practice application UI.

Run: python dr_beat_graph_ui.py

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Set, Callable, Any, Tuple, 
    Iterator, TypeVar, Generic, Union
)
from collections import defaultdict, deque
from enum import Enum, auto
from abc import ABC, abstractmethod
import time
import uuid
import os
import sys


# ============================================================================
# PART 1: EVENT SYSTEM (from scene_graph_v3.py)
# ============================================================================

class Event:
    """Base event class for state changes"""
    __slots__ = ('source', 'timestamp', 'handled')
    
    def __init__(self, source: Any):
        self.source = source
        self.timestamp: float = 0.0
        self.handled: bool = False


@dataclass
class PropertyChangedEvent(Event):
    """Emitted when a property changes"""
    property_name: str = ""
    old_value: Any = None
    new_value: Any = None
    
    def __init__(self, source: Any, prop: str, old: Any, new: Any):
        super().__init__(source)
        self.property_name = prop
        self.old_value = old
        self.new_value = new


class ChildAddedEvent(Event):
    """Emitted when a child is added"""
    __slots__ = ('child', 'index')
    
    def __init__(self, source: Any, child: Any, index: int = -1):
        super().__init__(source)
        self.child = child
        self.index = index


class ChildRemovedEvent(Event):
    """Emitted when a child is removed"""
    __slots__ = ('child',)
    
    def __init__(self, source: Any, child: Any):
        super().__init__(source)
        self.child = child


class TransformChangedEvent(Event):
    """Emitted when transform changes"""
    pass


class EventBus:
    """
    Central event dispatcher with batch processing.
    Events are queued and processed together to prevent cascades.
    """
    
    def __init__(self):
        self._subscribers: Dict[type, List[Callable]] = defaultdict(list)
        self._queue: List[Event] = []
        self._processing = False
        self._time = 0.0
        self._frame = 0
        self.debug = False
    
    def subscribe(self, event_type: type, callback: Callable[[Event], None]) -> None:
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: type, callback: Callable) -> None:
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
    
    def emit(self, event: Event) -> None:
        event.timestamp = self._time
        self._queue.append(event)
        if self.debug:
            print(f"  [EVENT] {event}")
    
    def process(self) -> int:
        if self._processing:
            return 0
        self._processing = True
        count = 0
        while self._queue:
            event = self._queue.pop(0)
            self._dispatch(event)
            count += 1
        self._processing = False
        self._frame += 1
        return count
    
    def _dispatch(self, event: Event) -> None:
        for callback in self._subscribers[type(event)]:
            try:
                callback(event)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    def set_time(self, t: float) -> None:
        self._time = t


# ============================================================================
# PART 2: SPATIAL TYPES
# ============================================================================

@dataclass
class Transform:
    """3D Transform - z is used for depth ordering in 2D"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    rotation: float = 0.0
    
    def copy(self) -> Transform:
        return Transform(self.x, self.y, self.z, self.scale_x, self.scale_y, self.rotation)


@dataclass
class Bounds:
    """Axis-aligned bounding box"""
    min_x: float = 0
    min_y: float = 0
    max_x: float = 0
    max_y: float = 0
    
    @property
    def width(self) -> float:
        return self.max_x - self.min_x
    
    @property
    def height(self) -> float:
        return self.max_y - self.min_y
    
    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y
    
    def intersects(self, other: Bounds) -> bool:
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                   self.max_y < other.min_y or self.min_y > other.max_y)


# ============================================================================
# PART 3: SCENE GRAPH NODE
# ============================================================================

class Node:
    """
    Scene graph node with depth-based draw ordering.
    
    Draw order = world_z + depth_bias
    Lower values drawn first (behind), higher drawn last (in front).
    """
    
    _id_counter = 0
    
    def __init__(self, bus: EventBus, name: str = ""):
        Node._id_counter += 1
        self.id = Node._id_counter
        self.name = name or f"node_{self.id}"
        self._bus = bus
        
        # Hierarchy
        self._parent: Optional[Node] = None
        self._children: List[Node] = []
        
        # Transform
        self._local = Transform()
        self._world = Transform()
        self._size = (0.0, 0.0)
        
        # Depth for draw ordering
        self._depth_bias = 0.0
        
        # Visibility
        self._visible = True
        self._opacity = 1.0
        
        # Components (domain data)
        self._components: Dict[type, Any] = {}
        
        # Tags
        self._tags: Set[str] = set()
        
        # Dirty flag
        self._dirty = True
    
    # --- Hierarchy ---
    
    @property
    def parent(self) -> Optional[Node]:
        return self._parent
    
    @property
    def children(self) -> List[Node]:
        return list(self._children)
    
    def add_child(self, child: Node, index: int = -1) -> None:
        if child._parent is not None:
            child._parent.remove_child(child)
        child._parent = self
        if index < 0:
            self._children.append(child)
            index = len(self._children) - 1
        else:
            self._children.insert(index, child)
        child._mark_dirty()
        self._bus.emit(ChildAddedEvent(self, child, index))
    
    def remove_child(self, child: Node) -> bool:
        if child not in self._children:
            return False
        self._children.remove(child)
        child._parent = None
        self._bus.emit(ChildRemovedEvent(self, child))
        return True
    
    # --- Transform ---
    
    @property
    def local_transform(self) -> Transform:
        return self._local
    
    @property
    def world_transform(self) -> Transform:
        if self._dirty:
            self._update_world()
        return self._world
    
    def set_position(self, x: float, y: float, z: Optional[float] = None) -> None:
        changed = False
        if self._local.x != x:
            self._local.x = x
            changed = True
        if self._local.y != y:
            self._local.y = y
            changed = True
        if z is not None and self._local.z != z:
            self._local.z = z
            changed = True
        if changed:
            self._mark_dirty()
            self._bus.emit(TransformChangedEvent(self))
    
    def set_size(self, w: float, h: float) -> None:
        if self._size != (w, h):
            old = self._size
            self._size = (w, h)
            self._bus.emit(PropertyChangedEvent(self, 'size', old, (w, h)))
    
    @property
    def size(self) -> Tuple[float, float]:
        return self._size
    
    @property
    def width(self) -> float:
        return self._size[0]
    
    @property
    def height(self) -> float:
        return self._size[1]
    
    def set_depth_bias(self, bias: float) -> None:
        if self._depth_bias != bias:
            old = self._depth_bias
            self._depth_bias = bias
            self._bus.emit(PropertyChangedEvent(self, 'depth_bias', old, bias))
    
    @property
    def effective_depth(self) -> float:
        return self.world_transform.z + self._depth_bias
    
    @property
    def bounds(self) -> Bounds:
        wt = self.world_transform
        return Bounds(wt.x, wt.y, wt.x + self._size[0], wt.y + self._size[1])
    
    @property
    def world_x(self) -> float:
        return self.world_transform.x
    
    @property
    def world_y(self) -> float:
        return self.world_transform.y
    
    def _mark_dirty(self) -> None:
        self._dirty = True
        for child in self._children:
            child._mark_dirty()
    
    def _update_world(self) -> None:
        if self._parent:
            pw = self._parent.world_transform
            self._world.x = pw.x + self._local.x * pw.scale_x
            self._world.y = pw.y + self._local.y * pw.scale_y
            self._world.z = pw.z + self._local.z
            self._world.scale_x = pw.scale_x * self._local.scale_x
            self._world.scale_y = pw.scale_y * self._local.scale_y
        else:
            self._world = self._local.copy()
        self._dirty = False
    
    # --- Visibility ---
    
    @property
    def visible(self) -> bool:
        return self._visible
    
    @visible.setter
    def visible(self, v: bool) -> None:
        if self._visible != v:
            self._visible = v
            self._bus.emit(PropertyChangedEvent(self, 'visible', not v, v))
    
    @property
    def opacity(self) -> float:
        return self._opacity
    
    @opacity.setter
    def opacity(self, v: float) -> None:
        self._opacity = max(0.0, min(1.0, v))
    
    # --- Components ---
    
    def add_component(self, comp: Any) -> None:
        self._components[type(comp)] = comp
    
    def get_component(self, t: type) -> Optional[Any]:
        return self._components.get(t)
    
    def has_component(self, t: type) -> bool:
        return t in self._components
    
    # --- Tags ---
    
    def add_tag(self, tag: str) -> None:
        self._tags.add(tag)
    
    def has_tag(self, tag: str) -> bool:
        return tag in self._tags
    
    # --- Traversal ---
    
    def traverse(self) -> Iterator[Node]:
        yield self
        for child in self._children:
            yield from child.traverse()
    
    def find_at(self, x: float, y: float) -> Optional[Node]:
        """Find deepest node at position"""
        if not self._visible:
            return None
        # Check children in reverse (top-most first)
        for child in reversed(self._children):
            found = child.find_at(x, y)
            if found:
                return found
        # Check self
        if self.bounds.contains(x, y):
            return self
        return None
    
    def __repr__(self):
        return f"{self.name}(depth={self.effective_depth:.1f})"


# ============================================================================
# PART 4: VISUAL COMPONENTS (domain data attached to nodes)
# ============================================================================

@dataclass
class VisualStyle:
    """Visual styling for widgets"""
    background: str = "#2a2a2a"
    border_color: str = "#4a4a4a"
    border_width: float = 1.0
    border_radius: float = 4.0
    text_color: str = "#e0e0e0"
    opacity: float = 1.0
    
    def lerp(self, other: VisualStyle, t: float) -> VisualStyle:
        """Linear interpolation between styles"""
        return VisualStyle(
            background=other.background if t > 0.5 else self.background,
            border_color=other.border_color if t > 0.5 else self.border_color,
            border_width=self.border_width + (other.border_width - self.border_width) * t,
            border_radius=self.border_radius + (other.border_radius - self.border_radius) * t,
            text_color=other.text_color if t > 0.5 else self.text_color,
            opacity=self.opacity + (other.opacity - self.opacity) * t
        )


@dataclass
class ClipVisual:
    """Visual data for a clip"""
    color: Tuple[float, float, float] = (1.0, 0.3, 0.2)
    waveform: Optional[List[float]] = None
    selected: bool = False
    triggered: bool = False
    trigger_time: float = 0.0


@dataclass
class TrackVisual:
    """Visual data for a track"""
    name: str = "Track"
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    muted: bool = False
    solo: bool = False


@dataclass
class GridVisual:
    """Visual data for the grid"""
    tracks: int = 4
    beats: int = 16
    track_height: float = 40
    beat_width: float = 60
    show_beat_numbers: bool = True


@dataclass
class PlayheadVisual:
    """Visual data for playhead"""
    beat: float = 0.0
    color: Tuple[float, float, float] = (0.0, 1.0, 0.5)
    width: float = 2.0


@dataclass
class TransportVisual:
    """Visual data for transport controls"""
    playing: bool = False
    recording: bool = False
    bpm: int = 120
    loop_enabled: bool = True


# ============================================================================
# PART 5: MESSAGE AND GRAPHOBJECT SYSTEM (from graph_object.py)
# ============================================================================

class MessageType(Enum):
    """Types of messages"""
    CALL = auto()
    RETURN = auto()
    EVENT = auto()
    QUERY = auto()
    COMMAND = auto()
    SIGNAL = auto()


@dataclass
class Message:
    """Messages flow through the graph"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    msg_type: MessageType = MessageType.SIGNAL
    sender: Optional[str] = None
    target: Optional[str] = None
    name: str = ""
    args: Tuple = ()
    kwargs: Dict = field(default_factory=dict)
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    history: List[str] = field(default_factory=list)
    
    def reply(self, payload: Any) -> Message:
        return Message(
            msg_type=MessageType.RETURN,
            sender=self.target,
            target=self.sender,
            payload=payload
        )


class Namespace:
    """Object state with prototype inheritance"""
    
    def __init__(self, name: str = "", prototype: Optional[Namespace] = None):
        self._name = name
        self._data: Dict[str, Any] = {}
        self._prototype = prototype
    
    def __getattr__(self, key: str) -> Any:
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        data = object.__getattribute__(self, '_data')
        if key in data:
            return data[key]
        proto = object.__getattribute__(self, '_prototype')
        if proto:
            return getattr(proto, key)
        raise AttributeError(f"'{self._name}' has no '{key}'")
    
    def __setattr__(self, key: str, value: Any):
        if key.startswith('_'):
            object.__setattr__(self, key, value)
        else:
            self._data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default
    
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        if self._prototype:
            result.update(self._prototype.to_dict())
        result.update(self._data)
        return result


class CallbackQueue:
    """Priority queue of callbacks"""
    
    def __init__(self):
        self._callbacks: List[Tuple[int, Callable]] = []
        self._seq = 0
    
    def add(self, callback: Callable, priority: int = 50) -> None:
        self._seq += 1
        self._callbacks.append((priority, self._seq, callback))
        self._callbacks.sort(key=lambda x: (x[0], x[1]))
    
    def execute(self, *args, **kwargs) -> List[Any]:
        results = []
        for _, _, cb in self._callbacks:
            try:
                results.append(cb(*args, **kwargs))
            except Exception as e:
                results.append(e)
        return results
    
    def clear(self) -> None:
        self._callbacks.clear()


class Stage:
    """A single stage in a pipeline"""
    
    def __init__(self, name: str, process: Callable[[Message], Message]):
        self.name = name
        self._process = process
        self.enabled = True
    
    def execute(self, msg: Message) -> Message:
        if not self.enabled:
            return msg
        msg.history.append(self.name)
        return self._process(msg)


class Pipeline:
    """Sequence of stages that process messages"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.stages: List[Stage] = []
    
    def add_stage(self, name: str, process: Callable[[Message], Message]) -> Pipeline:
        self.stages.append(Stage(name, process))
        return self
    
    def execute(self, msg: Message) -> Message:
        for stage in self.stages:
            msg = stage.execute(msg)
        return msg
    
    def __call__(self, msg: Message) -> Message:
        return self.execute(msg)


class GraphObject:
    """
    Unified object with state, links, pipelines, and events.
    """
    
    _id_counter = 0
    
    def __init__(self, name: str = "", prototype: Optional[GraphObject] = None):
        GraphObject._id_counter += 1
        self.id = GraphObject._id_counter
        self.name = name or f"obj_{self.id}"
        
        # State with prototype inheritance
        proto_ns = prototype.state if prototype else None
        self.state = Namespace(self.name, prototype=proto_ns)
        
        # Pipelines as methods
        self.pipelines: Dict[str, Pipeline] = {}
        
        # Event callbacks
        self.events: Dict[str, CallbackQueue] = {}
        
        # Prototype reference
        self.prototype = prototype
        
        # Message inbox
        self.inbox: deque = deque()
    
    def __getattr__(self, key: str) -> Any:
        if key in ('id', 'name', 'state', 'pipelines', 'events', 'prototype', 'inbox'):
            return object.__getattribute__(self, key)
        
        pipelines = object.__getattribute__(self, 'pipelines')
        if key in pipelines:
            return lambda *args, **kwargs: self._call_pipeline(key, *args, **kwargs)
        
        prototype = object.__getattribute__(self, 'prototype')
        if prototype and key in prototype.pipelines:
            return lambda *args, **kwargs: self._call_inherited(prototype, key, *args, **kwargs)
        
        state = object.__getattribute__(self, 'state')
        return getattr(state, key)
    
    def __setattr__(self, key: str, value: Any):
        if key in ('id', 'name', 'state', 'pipelines', 'events', 'prototype', 'inbox'):
            object.__setattr__(self, key, value)
        else:
            self.state._data[key] = value
    
    def define(self, name: str) -> Pipeline:
        pipeline = Pipeline(name)
        self.pipelines[name] = pipeline
        return pipeline
    
    def _call_pipeline(self, name: str, *args, **kwargs) -> Any:
        msg = Message(
            msg_type=MessageType.CALL,
            sender=self.name,
            target=self.name,
            name=name,
            args=args,
            kwargs=kwargs,
            payload={'self': self}
        )
        result = self.pipelines[name].execute(msg)
        return result.payload
    
    def _call_inherited(self, proto: GraphObject, name: str, *args, **kwargs) -> Any:
        msg = Message(
            msg_type=MessageType.CALL,
            sender=self.name,
            target=self.name,
            name=name,
            args=args,
            kwargs=kwargs,
            payload={'self': self}
        )
        result = proto.pipelines[name].execute(msg)
        return result.payload
    
    def on(self, event: str, callback: Callable, priority: int = 50) -> None:
        if event not in self.events:
            self.events[event] = CallbackQueue()
        self.events[event].add(callback, priority)
    
    def emit(self, event: str, *args, **kwargs) -> List[Any]:
        if event in self.events:
            return self.events[event].execute(*args, **kwargs)
        return []
    
    def create_child(self, name: str = "") -> GraphObject:
        return GraphObject(name or f"{self.name}_child", prototype=self)


class Graph:
    """Container for all GraphObjects"""
    
    def __init__(self, name: str = "world"):
        self.name = name
        self.objects: Dict[int, GraphObject] = {}
        self.by_name: Dict[str, GraphObject] = {}
        self.events: Dict[str, CallbackQueue] = {}
    
    def create(self, name: str = "", prototype: Optional[GraphObject] = None) -> GraphObject:
        obj = GraphObject(name, prototype)
        self.objects[obj.id] = obj
        self.by_name[obj.name] = obj
        return obj
    
    def get(self, name: str) -> Optional[GraphObject]:
        return self.by_name.get(name)
    
    def on(self, event: str, callback: Callable, priority: int = 50) -> None:
        if event not in self.events:
            self.events[event] = CallbackQueue()
        self.events[event].add(callback, priority)
    
    def emit(self, event: str, *args, **kwargs) -> List[Any]:
        if event in self.events:
            return self.events[event].execute(*args, **kwargs)
        return []


# ============================================================================
# PART 6: SEQUENCER GRAPHOBJECTS
# ============================================================================

def create_sequencer_prototypes(graph: Graph) -> Dict[str, GraphObject]:
    """Create prototype objects for sequencer entities"""
    
    # --- Clip Prototype ---
    clip_proto = graph.create("ClipProto")
    clip_proto.sample_name = ""
    clip_proto.start_beat = 0.0
    clip_proto.duration = 1.0
    clip_proto.velocity = 100
    clip_proto.color = (1.0, 0.3, 0.2)
    
    def trigger_stage(msg: Message) -> Message:
        self = msg.payload['self']
        velocity = msg.args[0] if msg.args else self.state.get('velocity', 100)
        self.emit('triggered', self, velocity)
        msg.payload = {'triggered': True, 'velocity': velocity}
        return msg
    
    clip_proto.define("trigger").add_stage("fire", trigger_stage)
    
    # --- Track Prototype ---
    track_proto = graph.create("TrackProto")
    track_proto.name = "Track"
    track_proto.clips = []
    track_proto.muted = False
    track_proto.solo = False
    track_proto.volume = 1.0
    track_proto.color = (0.5, 0.5, 0.5)
    
    def add_clip_stage(msg: Message) -> Message:
        self = msg.payload['self']
        clip = msg.args[0] if msg.args else None
        if clip:
            clips = list(self.state.get('clips', []))
            clips.append(clip)
            self.clips = clips
            self.emit('clip_added', self, clip)
        msg.payload = clip
        return msg
    
    def mute_stage(msg: Message) -> Message:
        self = msg.payload['self']
        mute = msg.args[0] if msg.args else not self.state.get('muted', False)
        self.muted = mute
        self.emit('mute_changed', self, mute)
        msg.payload = mute
        return msg
    
    track_proto.define("add_clip").add_stage("add", add_clip_stage)
    track_proto.define("toggle_mute").add_stage("mute", mute_stage)
    
    # --- Pattern Prototype ---
    pattern_proto = graph.create("PatternProto")
    pattern_proto.name = "Pattern A"
    pattern_proto.tracks = []
    pattern_proto.length_beats = 16
    pattern_proto.bpm = 120
    
    def add_track_stage(msg: Message) -> Message:
        self = msg.payload['self']
        track = msg.args[0] if msg.args else None
        if track:
            tracks = list(self.state.get('tracks', []))
            tracks.append(track)
            self.tracks = tracks
            self.emit('track_added', self, track)
        msg.payload = track
        return msg
    
    def get_clips_at_beat_stage(msg: Message) -> Message:
        self = msg.payload['self']
        beat = msg.args[0] if msg.args else 0
        clips = []
        for track in self.state.get('tracks', []):
            if track.state.get('muted', False):
                continue
            for clip in track.state.get('clips', []):
                clip_start = clip.state.get('start_beat', 0)
                clip_dur = clip.state.get('duration', 1)
                if clip_start <= beat < clip_start + clip_dur:
                    clips.append(clip)
        msg.payload = clips
        return msg
    
    pattern_proto.define("add_track").add_stage("add", add_track_stage)
    pattern_proto.define("get_clips_at_beat").add_stage("find", get_clips_at_beat_stage)
    
    # --- Sequencer Prototype ---
    seq_proto = graph.create("SequencerProto")
    seq_proto.patterns = {}
    seq_proto.current_pattern = None
    seq_proto.playing = False
    seq_proto.current_beat = 0.0
    seq_proto.loop_start = 0
    seq_proto.loop_end = 16
    seq_proto.bpm = 120
    seq_proto.last_beat_int = -1
    
    def play_stage(msg: Message) -> Message:
        self = msg.payload['self']
        self.playing = True
        self.emit('play')
        msg.payload = True
        return msg
    
    def stop_stage(msg: Message) -> Message:
        self = msg.payload['self']
        self.playing = False
        self.current_beat = 0.0
        self.last_beat_int = -1
        self.emit('stop')
        msg.payload = False
        return msg
    
    def tick_stage(msg: Message) -> Message:
        self = msg.payload['self']
        dt = msg.args[0] if msg.args else 0.0
        
        if not self.state.get('playing', False):
            msg.payload = None
            return msg
        
        bpm = self.state.get('bpm', 120)
        beats_per_sec = bpm / 60.0
        current = self.state.get('current_beat', 0.0)
        current += beats_per_sec * dt
        
        loop_end = self.state.get('loop_end', 16)
        loop_start = self.state.get('loop_start', 0)
        if current >= loop_end:
            current = loop_start + (current - loop_end)
        
        self.current_beat = current
        
        # Check for beat triggers
        beat_int = int(current)
        last_beat = self.state.get('last_beat_int', -1)
        
        triggered_clips = []
        if beat_int != last_beat:
            self.last_beat_int = beat_int
            pattern = self.state.get('current_pattern')
            if pattern:
                clips = pattern.get_clips_at_beat(beat_int)
                for clip in clips:
                    clip.trigger()
                    triggered_clips.append(clip)
        
        self.emit('tick', current, triggered_clips)
        msg.payload = {'beat': current, 'triggered': triggered_clips}
        return msg
    
    def set_bpm_stage(msg: Message) -> Message:
        self = msg.payload['self']
        bpm = msg.args[0] if msg.args else 120
        self.bpm = max(30, min(300, bpm))
        self.emit('bpm_changed', self.state.get('bpm'))
        msg.payload = self.state.get('bpm')
        return msg
    
    seq_proto.define("play").add_stage("start", play_stage)
    seq_proto.define("stop").add_stage("halt", stop_stage)
    seq_proto.define("tick").add_stage("advance", tick_stage)
    seq_proto.define("set_bpm").add_stage("update", set_bpm_stage)
    
    return {
        'clip': clip_proto,
        'track': track_proto,
        'pattern': pattern_proto,
        'sequencer': seq_proto
    }


# ============================================================================
# PART 6B: MIDI ROUTER GRAPHOBJECT
# ============================================================================

def create_midi_router(graph: Graph) -> GraphObject:
    """Create MidiRouter for handling MIDI input"""
    
    router = graph.create("MidiRouter")
    router.note_handlers = {}  # note -> handler
    router.channel_handlers = {}  # channel -> handler
    router.grid_handlers = []  # For Launchpad-style grid callbacks
    router.last_note = -1
    router.last_velocity = 0
    
    def map_note_stage(msg: Message) -> Message:
        """Map a MIDI note to a handler"""
        self = msg.payload['self']
        note = msg.args[0] if len(msg.args) > 0 else 0
        handler = msg.args[1] if len(msg.args) > 1 else None
        if handler:
            handlers = dict(self.state.get('note_handlers', {}))
            handlers[note] = handler
            self.note_handlers = handlers
        msg.payload = note
        return msg
    
    def map_channel_stage(msg: Message) -> Message:
        """Map a MIDI channel to a handler"""
        self = msg.payload['self']
        channel = msg.args[0] if len(msg.args) > 0 else 0
        handler = msg.args[1] if len(msg.args) > 1 else None
        if handler:
            handlers = dict(self.state.get('channel_handlers', {}))
            handlers[channel] = handler
            self.channel_handlers = handlers
        msg.payload = channel
        return msg
    
    def on_grid_press_stage(msg: Message) -> Message:
        """Register a grid press handler (for Launchpad)"""
        self = msg.payload['self']
        handler = msg.args[0] if msg.args else None
        if handler:
            handlers = list(self.state.get('grid_handlers', []))
            handlers.append(handler)
            self.grid_handlers = handlers
        return msg
    
    def process_midi_stage(msg: Message) -> Message:
        """Process incoming MIDI message"""
        self = msg.payload['self']
        note = msg.args[0] if len(msg.args) > 0 else 0
        velocity = msg.args[1] if len(msg.args) > 1 else 127
        channel = msg.args[2] if len(msg.args) > 2 else 0
        
        self.last_note = note
        self.last_velocity = velocity
        
        # Check note handlers
        note_handlers = self.state.get('note_handlers', {})
        if note in note_handlers:
            note_handlers[note](note, velocity, channel)
        
        # Check channel handlers
        channel_handlers = self.state.get('channel_handlers', {})
        if channel in channel_handlers:
            channel_handlers[channel](note, velocity, channel)
        
        # Grid handlers (Launchpad: note -> row, col)
        grid_handlers = self.state.get('grid_handlers', [])
        if grid_handlers:
            row = note // 8
            col = note % 8
            for handler in grid_handlers:
                handler(row, col, velocity)
        
        self.emit('midi_received', note, velocity, channel)
        msg.payload = {'note': note, 'velocity': velocity, 'channel': channel}
        return msg
    
    router.define("map_note").add_stage("map", map_note_stage)
    router.define("map_channel").add_stage("map", map_channel_stage)
    router.define("on_grid_press").add_stage("register", on_grid_press_stage)
    router.define("process_midi").add_stage("route", process_midi_stage)
    
    return router


class LaunchpadMapper:
    """Maps Launchpad grid to clips - convenience wrapper"""
    
    def __init__(self, midi_router: GraphObject, grid: Optional[Any] = None):
        self.router = midi_router
        self.grid = grid
        self._setup_grid_mapping()
    
    def _setup_grid_mapping(self):
        """Setup default Launchpad grid mapping"""
        def on_grid(row: int, col: int, velocity: int):
            if self.grid and velocity > 0:
                # Find clip at row, col and trigger it
                for track_node in self.grid.track_nodes:
                    if track_node.track_index == row:
                        for clip_node in track_node.clip_nodes:
                            clip_start = clip_node.clip_object.state.get('start_beat', 0) if clip_node.clip_object else 0
                            if int(clip_start) == col:
                                clip_node.trigger(velocity)
                                break
        
        self.router.on_grid_press(on_grid)
    
    def note_to_grid(self, note: int) -> Tuple[int, int]:
        """Convert MIDI note to grid position"""
        return (note // 8, note % 8)
    
    def grid_to_note(self, row: int, col: int) -> int:
        """Convert grid position to MIDI note"""
        return row * 8 + col


# ============================================================================
# PART 6C: UNDO/REDO ACTION SYSTEM
# ============================================================================

class Action(ABC):
    """Base class for undoable actions"""
    
    @abstractmethod
    def execute(self) -> None:
        """Perform the action"""
        pass
    
    @abstractmethod
    def undo(self) -> None:
        """Reverse the action"""
        pass
    
    @property
    def description(self) -> str:
        return self.__class__.__name__


class ActionStack:
    """Manages undo/redo stack"""
    
    def __init__(self, max_size: int = 50):
        self.undo_stack: List[Action] = []
        self.redo_stack: List[Action] = []
        self.max_size = max_size
    
    def execute(self, action: Action) -> None:
        """Execute action and add to undo stack"""
        action.execute()
        self.undo_stack.append(action)
        self.redo_stack.clear()  # Clear redo after new action
        
        # Limit stack size
        if len(self.undo_stack) > self.max_size:
            self.undo_stack.pop(0)
    
    def undo(self) -> Optional[Action]:
        """Undo last action"""
        if not self.undo_stack:
            return None
        action = self.undo_stack.pop()
        action.undo()
        self.redo_stack.append(action)
        return action
    
    def redo(self) -> Optional[Action]:
        """Redo last undone action"""
        if not self.redo_stack:
            return None
        action = self.redo_stack.pop()
        action.execute()
        self.undo_stack.append(action)
        return action
    
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0
    
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0
    
    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()


class MoveClipAction(Action):
    """Action for moving a clip"""
    
    def __init__(self, clip_node: Any, old_track: int, old_beat: float, 
                 new_track: int, new_beat: float):
        self.clip_node = clip_node
        self.old_track = old_track
        self.old_beat = old_beat
        self.new_track = new_track
        self.new_beat = new_beat
    
    def execute(self) -> None:
        # Move to new position
        if self.clip_node.clip_object:
            self.clip_node.clip_object.start_beat = self.new_beat
    
    def undo(self) -> None:
        # Move back to old position
        if self.clip_node.clip_object:
            self.clip_node.clip_object.start_beat = self.old_beat
    
    @property
    def description(self) -> str:
        return f"Move clip to beat {self.new_beat}"


class DeleteClipAction(Action):
    """Action for deleting a clip"""
    
    def __init__(self, clip_node: Any, track_node: Any, parent_grid: Any):
        self.clip_node = clip_node
        self.track_node = track_node
        self.parent_grid = parent_grid
        self.clip_object = clip_node.clip_object
        self.beat = clip_node.clip_object.state.get('start_beat', 0) if clip_node.clip_object else 0
    
    def execute(self) -> None:
        # Remove from UI
        if self.clip_node in self.track_node.clip_nodes:
            self.track_node.clip_nodes.remove(self.clip_node)
        self.clip_node.visible = False
    
    def undo(self) -> None:
        # Restore to UI
        if self.clip_node not in self.track_node.clip_nodes:
            self.track_node.clip_nodes.append(self.clip_node)
        self.clip_node.visible = True
    
    @property
    def description(self) -> str:
        return "Delete clip"


class ToggleMuteAction(Action):
    """Action for toggling track mute"""
    
    def __init__(self, track_node: Any):
        self.track_node = track_node
        self.was_muted = track_node.track_visual.muted
    
    def execute(self) -> None:
        self.track_node.track_visual.muted = not self.was_muted
        if self.track_node.track_object:
            self.track_node.track_object.muted = not self.was_muted
    
    def undo(self) -> None:
        self.track_node.track_visual.muted = self.was_muted
        if self.track_node.track_object:
            self.track_node.track_object.muted = self.was_muted


class ToggleSoloAction(Action):
    """Action for toggling track solo"""
    
    def __init__(self, track_node: Any):
        self.track_node = track_node
        self.was_solo = track_node.track_visual.solo
    
    def execute(self) -> None:
        self.track_node.track_visual.solo = not self.was_solo
        if self.track_node.track_object:
            self.track_node.track_object.solo = not self.was_solo
    
    def undo(self) -> None:
        self.track_node.track_visual.solo = self.was_solo
        if self.track_node.track_object:
            self.track_node.track_object.solo = self.was_solo


# ============================================================================
# PART 6D: SPATIAL INDEX FOR EFFICIENT HIT TESTING
# ============================================================================

class SpatialIndex:
    """
    Grid-based spatial index for fast queries.
    Used for efficient hit testing with many objects.
    """
    
    def __init__(self, cell_size: float = 100):
        self.cell_size = cell_size
        self._cells: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        self._bounds: Dict[int, Bounds] = {}
        self._nodes: Dict[int, Node] = {}
    
    def _cell_key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))
    
    def insert(self, node: Node) -> None:
        """Insert or update a node"""
        obj_id = node.id
        bounds = node.bounds
        
        # Remove old entry if exists
        if obj_id in self._bounds:
            self.remove(node)
        
        self._bounds[obj_id] = bounds
        self._nodes[obj_id] = node
        
        # Add to all overlapping cells
        min_cell = self._cell_key(bounds.min_x, bounds.min_y)
        max_cell = self._cell_key(bounds.max_x, bounds.max_y)
        
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                self._cells[(cx, cy)].add(obj_id)
    
    def remove(self, node: Node) -> None:
        """Remove a node"""
        obj_id = node.id
        if obj_id not in self._bounds:
            return
        
        bounds = self._bounds[obj_id]
        min_cell = self._cell_key(bounds.min_x, bounds.min_y)
        max_cell = self._cell_key(bounds.max_x, bounds.max_y)
        
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                self._cells[(cx, cy)].discard(obj_id)
        
        del self._bounds[obj_id]
        del self._nodes[obj_id]
    
    def query_point(self, x: float, y: float) -> List[Node]:
        """Find all nodes containing this point"""
        cell = self._cell_key(x, y)
        results = []
        
        for obj_id in self._cells.get(cell, set()):
            bounds = self._bounds.get(obj_id)
            if bounds and bounds.contains(x, y):
                node = self._nodes.get(obj_id)
                if node:
                    results.append(node)
        
        return results
    
    def query_region(self, region: Bounds) -> List[Node]:
        """Find all nodes intersecting this region"""
        min_cell = self._cell_key(region.min_x, region.min_y)
        max_cell = self._cell_key(region.max_x, region.max_y)
        
        candidates: Set[int] = set()
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cy in range(min_cell[1], max_cell[1] + 1):
                candidates.update(self._cells.get((cx, cy), set()))
        
        results = []
        for obj_id in candidates:
            bounds = self._bounds.get(obj_id)
            if bounds and bounds.intersects(region):
                node = self._nodes.get(obj_id)
                if node:
                    results.append(node)
        
        return results
    
    def update_all(self, nodes: List[Node]) -> None:
        """Rebuild index from node list"""
        self._cells.clear()
        self._bounds.clear()
        self._nodes.clear()
        for node in nodes:
            if node.visible:
                self.insert(node)


# ============================================================================
# PART 7: THEME SYSTEM
# ============================================================================

class WidgetState(Enum):
    """Widget interaction states"""
    NORMAL = auto()
    HOVER = auto()
    PRESSED = auto()
    FOCUSED = auto()
    DISABLED = auto()
    ACTIVE = auto()
    DRAGGING = auto()


class Theme:
    """Visual theme for the application"""
    
    # Core colors
    BACKGROUND = "#0a0a0a"
    PANEL = "#1a1a1a"
    WIDGET = "#2a2a2a"
    HOVER = "#3a3a3a"
    PRESSED = "#4a4a4a"
    BORDER = "#4a4a4a"
    
    # Accent colors
    ACCENT = "#00d4ff"
    PLAYHEAD = "#00ff88"
    RECORDING = "#ff4444"
    
    # Text colors
    TEXT = "#e0e0e0"
    TEXT_DIM = "#666666"
    
    # Track colors
    TRACK_COLORS = [
        (1.0, 0.29, 0.17),   # Kick - red
        (0.22, 0.94, 0.49),  # Snare - green
        (0.0, 0.95, 0.99),   # HiHat - cyan
        (1.0, 0.88, 0.25),   # Clap - yellow
        (0.29, 0.0, 0.88),   # Bass - purple
        (0.96, 0.34, 0.42),  # Lead - pink
        (0.0, 0.83, 0.53),   # Perc - teal
        (1.0, 0.6, 0.0),     # FX - orange
    ]
    
    def __init__(self):
        self._styles: Dict[Tuple[str, WidgetState], VisualStyle] = {}
        self._init_default_styles()
    
    def _init_default_styles(self):
        # Button styles
        self.set("button", WidgetState.NORMAL, VisualStyle(
            background=self.WIDGET, border_color=self.BORDER))
        self.set("button", WidgetState.HOVER, VisualStyle(
            background=self.HOVER, border_color=self.ACCENT))
        self.set("button", WidgetState.PRESSED, VisualStyle(
            background=self.PRESSED, border_color=self.ACCENT))
        
        # Toggle styles
        self.set("toggle", WidgetState.NORMAL, VisualStyle(
            background=self.WIDGET, border_color=self.BORDER))
        self.set("toggle", WidgetState.ACTIVE, VisualStyle(
            background=self.ACCENT, border_color=self.ACCENT, text_color="#000000"))
        
        # Slider styles
        self.set("slider", WidgetState.NORMAL, VisualStyle(
            background=self.PANEL, border_color=self.BORDER))
        self.set("slider", WidgetState.DRAGGING, VisualStyle(
            background=self.PANEL, border_color=self.ACCENT))
        
        # Clip styles
        self.set("clip", WidgetState.NORMAL, VisualStyle(
            background=self.WIDGET, border_color=self.BORDER, border_radius=4))
        self.set("clip", WidgetState.HOVER, VisualStyle(
            background=self.HOVER, border_color=self.ACCENT, border_radius=4))
        self.set("clip", WidgetState.PRESSED, VisualStyle(
            background=self.PRESSED, border_color=self.ACCENT, border_radius=4))
    
    def set(self, widget_type: str, state: WidgetState, style: VisualStyle):
        self._styles[(widget_type, state)] = style
    
    def get(self, widget_type: str, state: WidgetState) -> VisualStyle:
        key = (widget_type, state)
        if key in self._styles:
            return self._styles[key]
        # Fallback to NORMAL
        fallback = (widget_type, WidgetState.NORMAL)
        if fallback in self._styles:
            return self._styles[fallback]
        return VisualStyle()
    
    def get_track_color(self, index: int) -> Tuple[float, float, float]:
        return self.TRACK_COLORS[index % len(self.TRACK_COLORS)]


DEFAULT_THEME = Theme()


# ============================================================================
# PART 8: WIDGET NODES (Interactive SceneGraph Nodes)
# ============================================================================

class WidgetNode(Node):
    """Base widget with state machine and interaction handling"""
    
    WIDGET_TYPE = "widget"
    
    def __init__(self, bus: EventBus, name: str = "", theme: Theme = None):
        super().__init__(bus, name)
        self.theme = theme or DEFAULT_THEME
        self._state = WidgetState.NORMAL
        self._interactive = True
        self._focusable = True
        self._draggable = False
        self._accepts_drop = False  # Can receive dropped items
        
        # Callbacks
        self._on_click: List[Callable] = []
        self._on_double_click: List[Callable] = []
        self._on_state_change: List[Callable] = []
        self._on_drop: List[Callable] = []  # Called when item dropped on this
    
    @property
    def widget_state(self) -> WidgetState:
        return self._state
    
    @widget_state.setter
    def widget_state(self, new_state: WidgetState):
        if self._state != new_state:
            old = self._state
            self._state = new_state
            for cb in self._on_state_change:
                cb(self, old, new_state)
            self._bus.emit(PropertyChangedEvent(self, 'widget_state', old, new_state))
    
    @property
    def visual(self) -> VisualStyle:
        return self.theme.get(self.WIDGET_TYPE, self._state)
    
    def on_click(self, callback: Callable) -> None:
        self._on_click.append(callback)
    
    def on_double_click(self, callback: Callable) -> None:
        """Register double-click handler"""
        self._on_double_click.append(callback)
    
    def on_state_change(self, callback: Callable) -> None:
        self._on_state_change.append(callback)
    
    def on_drop(self, callback: Callable) -> None:
        """Register drop handler - called when item dropped on this widget"""
        self._on_drop.append(callback)
    
    def accepts_drop(self, dragged_widget: 'WidgetNode') -> bool:
        """Check if this widget can accept the dragged widget"""
        return self._accepts_drop
    
    def handle_drop(self, dragged_widget: 'WidgetNode', x: float, y: float) -> bool:
        """Handle a drop event - return True if accepted"""
        if not self.accepts_drop(dragged_widget):
            return False
        for cb in self._on_drop:
            cb(self, dragged_widget, x, y)
        return True
    
    # Input handlers (called by EventDispatcher)
    
    def on_mouse_enter(self):
        if self._state == WidgetState.NORMAL:
            self.widget_state = WidgetState.HOVER
    
    def on_mouse_leave(self):
        if self._state in (WidgetState.HOVER, WidgetState.PRESSED):
            self.widget_state = WidgetState.NORMAL
    
    def on_mouse_down(self, x: float, y: float, button: int = 0):
        self.widget_state = WidgetState.PRESSED
    
    def on_mouse_up(self, x: float, y: float, button: int = 0):
        if self._state == WidgetState.PRESSED:
            # Trigger click
            for cb in self._on_click:
                cb(self)
            self.widget_state = WidgetState.HOVER
    
    def on_mouse_double_click(self, x: float, y: float, button: int = 0):
        """Handle double-click event"""
        for cb in self._on_double_click:
            cb(self, x, y)
    
    def on_drag_start(self, x: float, y: float):
        if self._draggable:
            self.widget_state = WidgetState.DRAGGING
    
    def on_drag(self, x: float, y: float, dx: float, dy: float):
        pass
    
    def on_drag_end(self, x: float, y: float):
        if self._state == WidgetState.DRAGGING:
            self.widget_state = WidgetState.NORMAL
    
    def on_key_down(self, key: str) -> bool:
        return False
    
    def on_key_up(self, key: str) -> bool:
        return False


class ButtonNode(WidgetNode):
    """Clickable button widget"""
    
    WIDGET_TYPE = "button"
    
    def __init__(self, bus: EventBus, name: str = "", text: str = "", theme: Theme = None):
        super().__init__(bus, name, theme)
        self.text = text
        self.set_size(80, 30)


class ToggleNode(WidgetNode):
    """Toggle button with on/off state"""
    
    WIDGET_TYPE = "toggle"
    
    def __init__(self, bus: EventBus, name: str = "", text: str = "", theme: Theme = None):
        super().__init__(bus, name, theme)
        self.text = text
        self._active = False
        self._on_toggle: List[Callable] = []
        self.set_size(60, 30)
    
    @property
    def active(self) -> bool:
        return self._active
    
    @active.setter
    def active(self, v: bool):
        if self._active != v:
            self._active = v
            for cb in self._on_toggle:
                cb(self, v)
    
    @property
    def widget_state(self) -> WidgetState:
        if self._active:
            return WidgetState.ACTIVE
        return self._state
    
    def on_toggle(self, callback: Callable) -> None:
        self._on_toggle.append(callback)
    
    def on_mouse_up(self, x: float, y: float, button: int = 0):
        if self._state == WidgetState.PRESSED:
            self.active = not self._active
            for cb in self._on_click:
                cb(self)
            self._state = WidgetState.HOVER


class SliderNode(WidgetNode):
    """Draggable slider widget"""
    
    WIDGET_TYPE = "slider"
    
    def __init__(self, bus: EventBus, name: str = "", 
                 min_val: float = 0, max_val: float = 100, theme: Theme = None):
        super().__init__(bus, name, theme)
        self._draggable = True
        self.min_val = min_val
        self.max_val = max_val
        self._value = min_val
        self._on_change: List[Callable] = []
        self.set_size(120, 20)
    
    @property
    def value(self) -> float:
        return self._value
    
    @value.setter
    def value(self, v: float):
        v = max(self.min_val, min(self.max_val, v))
        if self._value != v:
            self._value = v
            for cb in self._on_change:
                cb(self, v)
    
    @property
    def normalized(self) -> float:
        range_val = self.max_val - self.min_val
        if range_val == 0:
            return 0
        return (self._value - self.min_val) / range_val
    
    def on_change(self, callback: Callable) -> None:
        self._on_change.append(callback)
    
    def on_drag(self, x: float, y: float, dx: float, dy: float):
        # Update value based on x position
        local_x = x - self.world_x
        normalized = max(0, min(1, local_x / self.width))
        self.value = self.min_val + normalized * (self.max_val - self.min_val)


class LabelNode(Node):
    """Text label (non-interactive)"""
    
    def __init__(self, bus: EventBus, name: str = "", text: str = ""):
        super().__init__(bus, name)
        self.text = text
        self.text_color = Theme.TEXT
        self.set_size(100, 20)


class PanelNode(WidgetNode):
    """Container panel"""
    
    WIDGET_TYPE = "panel"
    
    def __init__(self, bus: EventBus, name: str = "", theme: Theme = None):
        super().__init__(bus, name, theme)
        self._interactive = False


# ============================================================================
# PART 9: SEQUENCER WIDGET NODES
# ============================================================================

class ClipNode(WidgetNode):
    """Clip widget - triggerable, selectable, draggable"""
    
    WIDGET_TYPE = "clip"
    
    def __init__(self, bus: EventBus, name: str = "", theme: Theme = None):
        super().__init__(bus, name, theme)
        self._draggable = True
        self.add_component(ClipVisual())
        
        # Reference to GraphObject clip
        self.clip_object: Optional[GraphObject] = None
        
        # Callbacks
        self._on_trigger: List[Callable] = []
        self._on_select: List[Callable] = []
        
        # Selection
        self._selected = False
    
    @property
    def clip_visual(self) -> ClipVisual:
        return self.get_component(ClipVisual)
    
    @property
    def selected(self) -> bool:
        return self._selected
    
    @selected.setter
    def selected(self, v: bool):
        if self._selected != v:
            self._selected = v
            self.clip_visual.selected = v
            for cb in self._on_select:
                cb(self, v)
    
    def on_trigger(self, callback: Callable) -> None:
        self._on_trigger.append(callback)
    
    def on_select(self, callback: Callable) -> None:
        self._on_select.append(callback)
    
    def trigger(self, velocity: int = 100):
        """Trigger the clip"""
        self.clip_visual.triggered = True
        self.clip_visual.trigger_time = time.time()
        for cb in self._on_trigger:
            cb(self, velocity)
        if self.clip_object:
            self.clip_object.trigger(velocity)
    
    def on_mouse_up(self, x: float, y: float, button: int = 0):
        if self._state == WidgetState.PRESSED:
            self.trigger()
            self.selected = True
            for cb in self._on_click:
                cb(self)
            self.widget_state = WidgetState.HOVER


class TrackNode(WidgetNode):
    """Track widget - contains clips, has mute/solo"""
    
    WIDGET_TYPE = "track"
    HEADER_WIDTH = 80  # Width of track header area
    
    def __init__(self, bus: EventBus, name: str = "", track_index: int = 0, theme: Theme = None):
        super().__init__(bus, name, theme)
        self.track_index = track_index
        self.add_component(TrackVisual(name=name, color=theme.get_track_color(track_index) if theme else (0.5, 0.5, 0.5)))
        
        # Reference to GraphObject track
        self.track_object: Optional[GraphObject] = None
        
        # Child clips
        self.clip_nodes: List[ClipNode] = []
        
        # Track accepts clip drops
        self._accepts_drop = True
        self._interactive = True
        
        # Callbacks for mute/solo
        self._on_mute: List[Callable] = []
        self._on_solo: List[Callable] = []
    
    @property
    def track_visual(self) -> TrackVisual:
        return self.get_component(TrackVisual)
    
    def on_mute(self, callback: Callable) -> None:
        self._on_mute.append(callback)
    
    def on_solo(self, callback: Callable) -> None:
        self._on_solo.append(callback)
    
    def toggle_mute(self) -> None:
        """Toggle mute state"""
        tv = self.track_visual
        tv.muted = not tv.muted
        if self.track_object:
            self.track_object.muted = tv.muted
        for cb in self._on_mute:
            cb(self, tv.muted)
    
    def toggle_solo(self) -> None:
        """Toggle solo state"""
        tv = self.track_visual
        tv.solo = not tv.solo
        if self.track_object:
            self.track_object.solo = tv.solo
        for cb in self._on_solo:
            cb(self, tv.solo)
    
    def accepts_drop(self, dragged_widget: WidgetNode) -> bool:
        """Accept clip drops"""
        return isinstance(dragged_widget, ClipNode)
    
    def handle_drop(self, dragged_widget: WidgetNode, x: float, y: float) -> bool:
        """Handle clip drop - move clip to this track"""
        if not isinstance(dragged_widget, ClipNode):
            return False
        # Calculate beat position
        local_x = x - self.world_x
        beat = int(local_x / 60)  # Assume 60px per beat
        # Move clip to this track (would trigger action in real app)
        for cb in self._on_drop:
            cb(self, dragged_widget, x, y)
        return True
    
    def add_clip_node(self, clip_node: ClipNode, beat: float) -> None:
        self.add_child(clip_node)
        self.clip_nodes.append(clip_node)


class GridNode(WidgetNode):
    """Grid widget - contains tracks"""
    
    WIDGET_TYPE = "grid"
    
    def __init__(self, bus: EventBus, name: str = "", 
                 tracks: int = 4, beats: int = 16, theme: Theme = None):
        super().__init__(bus, name, theme)
        self.add_component(GridVisual(tracks=tracks, beats=beats))
        
        self.track_nodes: List[TrackNode] = []
        self._on_cell_click: List[Callable] = []
        
        self._interactive = False  # Children handle clicks
    
    @property
    def grid_visual(self) -> GridVisual:
        return self.get_component(GridVisual)
    
    def on_cell_click(self, callback: Callable) -> None:
        self._on_cell_click.append(callback)
    
    def add_track_node(self, track_node: TrackNode) -> None:
        gv = self.grid_visual
        track_node.set_position(0, len(self.track_nodes) * gv.track_height)
        track_node.set_size(gv.beats * gv.beat_width, gv.track_height)
        self.add_child(track_node)
        self.track_nodes.append(track_node)
    
    def beat_to_x(self, beat: float) -> float:
        return beat * self.grid_visual.beat_width
    
    def track_to_y(self, track: int) -> float:
        return track * self.grid_visual.track_height
    
    def cell_at(self, x: float, y: float) -> Tuple[int, int]:
        """Get (track, beat) at position"""
        gv = self.grid_visual
        local_x = x - self.world_x
        local_y = y - self.world_y
        track = int(local_y / gv.track_height)
        beat = int(local_x / gv.beat_width)
        return (track, beat)


class PlayheadNode(Node):
    """Playhead position indicator"""
    
    def __init__(self, bus: EventBus, name: str = "playhead"):
        super().__init__(bus, name)
        self.add_component(PlayheadVisual())
        self.set_size(2, 160)
        self.set_depth_bias(100)  # Always on top
        
        self.pixels_per_beat = 60.0
    
    @property
    def playhead_visual(self) -> PlayheadVisual:
        return self.get_component(PlayheadVisual)
    
    @property
    def beat(self) -> float:
        return self.playhead_visual.beat
    
    @beat.setter
    def beat(self, v: float):
        self.playhead_visual.beat = v
        self.set_position(v * self.pixels_per_beat, self.local_transform.y)


class TransportNode(WidgetNode):
    """Transport controls - play/stop/record/BPM/Loop"""
    
    WIDGET_TYPE = "transport"
    
    def __init__(self, bus: EventBus, name: str = "transport", theme: Theme = None):
        super().__init__(bus, name, theme)
        self.add_component(TransportVisual())
        self.set_size(500, 40)  # Wider to fit loop controls
        
        self._interactive = False  # Children handle clicks
        
        # Create child buttons
        self.play_btn = ToggleNode(bus, "play_btn", "▶", theme)
        self.play_btn.set_position(10, 5)
        self.play_btn.set_size(40, 30)
        self.add_child(self.play_btn)
        
        self.stop_btn = ButtonNode(bus, "stop_btn", "■", theme)
        self.stop_btn.set_position(55, 5)
        self.stop_btn.set_size(40, 30)
        self.add_child(self.stop_btn)
        
        self.record_btn = ToggleNode(bus, "rec_btn", "●", theme)
        self.record_btn.set_position(100, 5)
        self.record_btn.set_size(40, 30)
        self.add_child(self.record_btn)
        
        # Loop toggle button
        self.loop_btn = ToggleNode(bus, "loop_btn", "⟳", theme)
        self.loop_btn.set_position(145, 5)
        self.loop_btn.set_size(40, 30)
        self.loop_btn.active = True  # Loop enabled by default
        self.add_child(self.loop_btn)
        
        self.bpm_slider = SliderNode(bus, "bpm_slider", 30, 300, theme)
        self.bpm_slider.set_position(200, 10)
        self.bpm_slider.value = 120
        self.add_child(self.bpm_slider)
        
        self.bpm_label = LabelNode(bus, "bpm_label", "120 BPM")
        self.bpm_label.set_position(330, 10)
        self.add_child(self.bpm_label)
        
        # Loop region controls
        self.loop_start = 0
        self.loop_end = 16
        self.loop_label = LabelNode(bus, "loop_label", "1-16")
        self.loop_label.set_position(410, 10)
        self.add_child(self.loop_label)
        
        # Callbacks
        self._on_play: List[Callable] = []
        self._on_stop: List[Callable] = []
        self._on_record: List[Callable] = []
        self._on_bpm_change: List[Callable] = []
        self._on_loop_toggle: List[Callable] = []
        self._on_loop_change: List[Callable] = []
        
        # Wire up internal callbacks
        self.play_btn.on_toggle(self._handle_play_toggle)
        self.stop_btn.on_click(self._handle_stop_click)
        self.record_btn.on_toggle(self._handle_record_toggle)
        self.loop_btn.on_toggle(self._handle_loop_toggle)
        self.bpm_slider.on_change(self._handle_bpm_change)
    
    @property
    def transport_visual(self) -> TransportVisual:
        return self.get_component(TransportVisual)
    
    def on_play(self, callback: Callable) -> None:
        self._on_play.append(callback)
    
    def on_stop(self, callback: Callable) -> None:
        self._on_stop.append(callback)
    
    def on_record(self, callback: Callable) -> None:
        self._on_record.append(callback)
    
    def on_bpm_change(self, callback: Callable) -> None:
        self._on_bpm_change.append(callback)
    
    def on_loop_toggle(self, callback: Callable) -> None:
        self._on_loop_toggle.append(callback)
    
    def on_loop_change(self, callback: Callable) -> None:
        self._on_loop_change.append(callback)
    
    def set_loop_region(self, start: int, end: int) -> None:
        """Set loop region (in beats)"""
        self.loop_start = max(0, start)
        self.loop_end = max(self.loop_start + 1, end)
        self.loop_label.text = f"{self.loop_start + 1}-{self.loop_end}"
        self.transport_visual.loop_enabled = self.loop_btn.active
        for cb in self._on_loop_change:
            cb(self.loop_start, self.loop_end)
    
    def _handle_play_toggle(self, widget, active: bool):
        self.transport_visual.playing = active
        for cb in self._on_play:
            cb(active)
    
    def _handle_stop_click(self, widget):
        self.play_btn.active = False
        self.transport_visual.playing = False
        for cb in self._on_stop:
            cb()
    
    def _handle_record_toggle(self, widget, active: bool):
        self.transport_visual.recording = active
        for cb in self._on_record:
            cb(active)
    
    def _handle_loop_toggle(self, widget, active: bool):
        self.transport_visual.loop_enabled = active
        for cb in self._on_loop_toggle:
            cb(active)
    
    def _handle_bpm_change(self, widget, value: float):
        bpm = int(value)
        self.transport_visual.bpm = bpm
        self.bpm_label.text = f"{bpm} BPM"
        for cb in self._on_bpm_change:
            cb(bpm)


# ============================================================================
# PART 10: EVENT DISPATCHER
# ============================================================================

class EventDispatcher:
    """Routes input events to widgets with double-click and drop support"""
    
    DOUBLE_CLICK_TIME = 0.3  # Max seconds between clicks for double-click
    
    def __init__(self, root: Node):
        self.root = root
        self.focused: Optional[WidgetNode] = None
        self.hovered: Optional[WidgetNode] = None
        self.dragging: Optional[WidgetNode] = None
        self.drag_start_pos: Tuple[float, float] = (0, 0)
        self.last_mouse_pos: Tuple[float, float] = (0, 0)
        
        self.drag_threshold = 5.0
        self._potential_drag = False
        
        # Double-click detection
        self._last_click_time = 0.0
        self._last_click_widget: Optional[WidgetNode] = None
        self._last_click_pos: Tuple[float, float] = (0, 0)
        
        # Multi-selection support
        self.selection: List[WidgetNode] = []
        self.shift_held = False
        self.ctrl_held = False
        
        # Spatial index for faster hit testing
        self.spatial_index: Optional[SpatialIndex] = None
    
    def set_spatial_index(self, index: SpatialIndex) -> None:
        """Set spatial index for faster hit testing"""
        self.spatial_index = index
    
    def mouse_move(self, x: float, y: float) -> Optional[Node]:
        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        self.last_mouse_pos = (x, y)
        
        # Handle dragging
        if self.dragging:
            self.dragging.on_drag(x, y, dx, dy)
            return self.dragging
        
        # Check for drag start
        if self._potential_drag and self.focused and isinstance(self.focused, WidgetNode):
            dist = ((x - self.drag_start_pos[0])**2 + (y - self.drag_start_pos[1])**2)**0.5
            if dist > self.drag_threshold and self.focused._draggable:
                self.dragging = self.focused
                self.dragging.on_drag_start(x, y)
                return self.dragging
        
        # Find widget at position
        found = self._find_widget_at(x, y)
        
        # Update hover state
        if found != self.hovered:
            if self.hovered:
                self.hovered.on_mouse_leave()
            self.hovered = found
            if self.hovered:
                self.hovered.on_mouse_enter()
        
        return found
    
    def mouse_down(self, x: float, y: float, button: int = 0) -> Optional[Node]:
        self.drag_start_pos = (x, y)
        self._potential_drag = True
        
        found = self._find_widget_at(x, y)
        
        # Handle selection with modifiers
        if found and isinstance(found, WidgetNode):
            if self.ctrl_held:
                # Toggle selection
                if found in self.selection:
                    self.selection.remove(found)
                else:
                    self.selection.append(found)
            elif self.shift_held:
                # Range selection (add to selection)
                if found not in self.selection:
                    self.selection.append(found)
            else:
                # Single selection (unless clicking already selected)
                if found not in self.selection:
                    self.selection = [found]
        elif not self.ctrl_held and not self.shift_held:
            # Click on empty space clears selection
            self.selection = []
        
        # Update focus
        if found != self.focused:
            self.focused = found
        
        if found:
            found.on_mouse_down(x, y, button)
        
        return found
    
    def mouse_up(self, x: float, y: float, button: int = 0) -> Optional[Node]:
        self._potential_drag = False
        current_time = time.time()
        
        # Handle drop
        if self.dragging:
            drop_target = self._find_widget_at(x, y)
            if drop_target and drop_target != self.dragging:
                if isinstance(drop_target, WidgetNode):
                    drop_target.handle_drop(self.dragging, x, y)
            self.dragging.on_drag_end(x, y)
            self.dragging = None
            return self.focused
        
        found = self._find_widget_at(x, y)
        
        if found:
            # Check for double-click
            is_double_click = (
                self._last_click_widget == found and
                current_time - self._last_click_time < self.DOUBLE_CLICK_TIME and
                abs(x - self._last_click_pos[0]) < 5 and
                abs(y - self._last_click_pos[1]) < 5
            )
            
            if is_double_click:
                found.on_mouse_double_click(x, y, button)
                self._last_click_widget = None  # Reset to prevent triple-click
            else:
                found.on_mouse_up(x, y, button)
                self._last_click_widget = found
                self._last_click_time = current_time
                self._last_click_pos = (x, y)
        
        return found
    
    def key_down(self, key: str, modifiers: Dict[str, bool] = None) -> bool:
        """Handle key down with modifier tracking"""
        modifiers = modifiers or {}
        self.shift_held = modifiers.get('shift', False)
        self.ctrl_held = modifiers.get('ctrl', False)
        
        if self.focused:
            return self.focused.on_key_down(key)
        return False
    
    def key_up(self, key: str, modifiers: Dict[str, bool] = None) -> bool:
        """Handle key up with modifier tracking"""
        modifiers = modifiers or {}
        self.shift_held = modifiers.get('shift', False)
        self.ctrl_held = modifiers.get('ctrl', False)
        
        if self.focused:
            return self.focused.on_key_up(key)
        return False
    
    def set_modifiers(self, shift: bool = False, ctrl: bool = False) -> None:
        """Update modifier key state"""
        self.shift_held = shift
        self.ctrl_held = ctrl
    
    def clear_selection(self) -> None:
        """Clear all selections"""
        self.selection = []
    
    def select_all(self, widgets: List[WidgetNode]) -> None:
        """Select all provided widgets"""
        self.selection = list(widgets)
    
    def _find_widget_at(self, x: float, y: float) -> Optional[WidgetNode]:
        """Find deepest interactive widget at position"""
        def find_recursive(node: Node) -> Optional[WidgetNode]:
            if not node.visible:
                return None
            # Check children in reverse (top-most first)
            for child in reversed(node.children):
                found = find_recursive(child)
                if found:
                    return found
            # Check self
            if isinstance(node, WidgetNode) and node._interactive:
                if node.bounds.contains(x, y):
                    return node
            return None
        
        return find_recursive(self.root)


# ============================================================================
# PART 11: RENDER CONTEXT
# ============================================================================

class RenderContext(ABC):
    """Abstract rendering interface"""
    
    @abstractmethod
    def begin(self) -> None:
        pass
    
    @abstractmethod
    def end(self) -> None:
        pass
    
    @abstractmethod
    def draw_rect(self, x: float, y: float, w: float, h: float, 
                  fill: str, stroke: str = "", stroke_width: float = 1,
                  radius: float = 0) -> None:
        pass
    
    @abstractmethod
    def draw_text(self, x: float, y: float, text: str, 
                  color: str = "#ffffff", size: int = 12) -> None:
        pass
    
    @abstractmethod
    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                  color: str, width: float = 1) -> None:
        pass
    
    @abstractmethod
    def draw_waveform(self, x: float, y: float, w: float, h: float,
                      data: List[float], color: str) -> None:
        """Draw waveform visualization"""
        pass


class ConsoleRenderContext(RenderContext):
    """ASCII rendering for testing"""
    
    TRACK_CHARS = ['█', '▓', '▒', '░', '▪', '▫', '◆', '◇']
    
    def __init__(self, width: int = 80, height: int = 24):
        self.width = width
        self.height = height
        self.scale_x = 10.0
        self.scale_y = 10.0
        self.chars: List[List[str]] = []
        self.depths: List[List[float]] = []
        self.output_lines: List[str] = []
    
    def begin(self) -> None:
        self.chars = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.depths = [[float('-inf') for _ in range(self.width)] for _ in range(self.height)]
        self.output_lines = []
    
    def end(self) -> None:
        self.output_lines = [''.join(row) for row in self.chars]
    
    def get_output(self) -> str:
        return '\n'.join(self.output_lines)
    
    def draw_rect(self, x: float, y: float, w: float, h: float,
                  fill: str, stroke: str = "", stroke_width: float = 1,
                  radius: float = 0, depth: float = 0, char: str = '█') -> None:
        x1 = max(0, int(x / self.scale_x))
        y1 = max(0, int(y / self.scale_y))
        x2 = min(self.width - 1, int((x + w) / self.scale_x))
        y2 = min(self.height - 1, int((y + h) / self.scale_y))
        
        for cy in range(y1, y2 + 1):
            for cx in range(x1, x2 + 1):
                if 0 <= cy < self.height and 0 <= cx < self.width:
                    if self.depths[cy][cx] < depth:
                        self.chars[cy][cx] = char
                        self.depths[cy][cx] = depth
    
    def draw_text(self, x: float, y: float, text: str,
                  color: str = "#ffffff", size: int = 12, depth: float = 0) -> None:
        cx = int(x / self.scale_x)
        cy = int(y / self.scale_y)
        for i, ch in enumerate(text):
            px = cx + i
            if 0 <= cy < self.height and 0 <= px < self.width:
                if self.depths[cy][px] < depth:
                    self.chars[cy][px] = ch
                    self.depths[cy][px] = depth
    
    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                  color: str, width: float = 1, depth: float = 0) -> None:
        # Vertical line (playhead)
        if abs(x1 - x2) < 1:
            cx = int(x1 / self.scale_x)
            cy1 = int(min(y1, y2) / self.scale_y)
            cy2 = int(max(y1, y2) / self.scale_y)
            for cy in range(cy1, cy2 + 1):
                if 0 <= cy < self.height and 0 <= cx < self.width:
                    if self.depths[cy][cx] < depth:
                        self.chars[cy][cx] = '│'
                        self.depths[cy][cx] = depth
        # Horizontal line
        elif abs(y1 - y2) < 1:
            cy = int(y1 / self.scale_y)
            cx1 = int(min(x1, x2) / self.scale_x)
            cx2 = int(max(x1, x2) / self.scale_x)
            for cx in range(cx1, cx2 + 1):
                if 0 <= cy < self.height and 0 <= cx < self.width:
                    if self.depths[cy][cx] < depth:
                        self.chars[cy][cx] = '─'
                        self.depths[cy][cx] = depth
    
    def draw_waveform(self, x: float, y: float, w: float, h: float,
                      data: List[float], color: str, depth: float = 0) -> None:
        """Draw waveform as ASCII visualization"""
        if not data:
            return
        
        x1 = max(0, int(x / self.scale_x))
        y1 = max(0, int(y / self.scale_y))
        x2 = min(self.width - 1, int((x + w) / self.scale_x))
        y2 = min(self.height - 1, int((y + h) / self.scale_y))
        
        width_chars = x2 - x1
        height_chars = y2 - y1
        
        if width_chars <= 0 or height_chars <= 0:
            return
        
        # Waveform characters by intensity
        wave_chars = '▁▂▃▄▅▆▇█'
        
        # Sample waveform data
        for cx in range(x1, x2 + 1):
            # Map character x to data index
            data_idx = int((cx - x1) / width_chars * len(data))
            data_idx = min(data_idx, len(data) - 1)
            
            # Get amplitude (0-1)
            amplitude = abs(data[data_idx]) if data_idx < len(data) else 0
            amplitude = min(1.0, max(0.0, amplitude))
            
            # Map to character
            char_idx = int(amplitude * (len(wave_chars) - 1))
            char = wave_chars[char_idx]
            
            # Draw at bottom of region
            cy = y2
            if 0 <= cy < self.height and 0 <= cx < self.width:
                if self.depths[cy][cx] < depth:
                    self.chars[cy][cx] = char
                    self.depths[cy][cx] = depth


# ============================================================================
# PART 12: SCENE GRAPH MANAGER
# ============================================================================

class SceneGraph:
    """Scene graph with depth-sorted rendering"""
    
    def __init__(self):
        self.bus = EventBus()
        self.root = Node(self.bus, "root")
        self._nodes: Dict[int, Node] = {self.root.id: self.root}
    
    def create_node(self, name: str = "", parent: Node = None) -> Node:
        node = Node(self.bus, name)
        self._nodes[node.id] = node
        (parent or self.root).add_child(node)
        return node
    
    def create_widget(self, widget_class: type, name: str = "", 
                      parent: Node = None, **kwargs) -> WidgetNode:
        widget = widget_class(self.bus, name, **kwargs)
        self._nodes[widget.id] = widget
        (parent or self.root).add_child(widget)
        return widget
    
    def destroy_node(self, node: Node) -> None:
        for child in list(node.children):
            self.destroy_node(child)
        if node.parent:
            node.parent.remove_child(node)
        self._nodes.pop(node.id, None)
    
    def get_render_order(self) -> List[Node]:
        """Get nodes sorted by effective_depth for rendering"""
        visible = [n for n in self._nodes.values() if n.visible]
        return sorted(visible, key=lambda n: (n.effective_depth, n.id))
    
    def update(self) -> int:
        return self.bus.process()
    
    def find_by_component(self, comp_type: type) -> List[Node]:
        return [n for n in self._nodes.values() if n.has_component(comp_type)]


# ============================================================================
# PART 13: UI BRIDGE (Connects GraphObject logic to SceneGraph UI)
# ============================================================================

class UIBridge:
    """Bridges GraphObject sequencer logic to SceneGraph UI"""
    
    def __init__(self, graph: Graph, scene: SceneGraph, theme: Theme):
        self.graph = graph
        self.scene = scene
        self.theme = theme
        
        # Mappings
        self.clip_map: Dict[int, ClipNode] = {}  # GraphObject ID -> ClipNode
        self.track_map: Dict[int, TrackNode] = {}
        
        # Selected clip
        self.selected_clip: Optional[ClipNode] = None
    
    def create_clip_node(self, clip_obj: GraphObject, track_index: int, 
                         beat: float, duration: float = 1.0) -> ClipNode:
        """Create a ClipNode linked to a GraphObject clip"""
        clip_node = ClipNode(self.scene.bus, f"clip_{clip_obj.id}", self.theme)
        
        # Link to GraphObject
        clip_node.clip_object = clip_obj
        
        # Set visual properties
        color = self.theme.get_track_color(track_index)
        clip_node.clip_visual.color = color
        
        # Position based on beat
        gv = GridVisual()  # Default grid visual for sizing
        clip_node.set_position(beat * gv.beat_width + 2, 2)
        clip_node.set_size(duration * gv.beat_width - 4, gv.track_height - 4)
        
        # Store mapping
        self.clip_map[clip_obj.id] = clip_node
        
        # Wire up selection
        def on_select(node, selected):
            if selected:
                if self.selected_clip and self.selected_clip != node:
                    self.selected_clip.selected = False
                self.selected_clip = node
        
        clip_node.on_select(on_select)
        
        return clip_node
    
    def create_track_node(self, track_obj: GraphObject, track_index: int) -> TrackNode:
        """Create a TrackNode linked to a GraphObject track"""
        track_node = TrackNode(self.scene.bus, f"track_{track_index}", track_index, self.theme)
        
        # Link to GraphObject
        track_node.track_object = track_obj
        
        # Update visual from track state
        track_node.track_visual.name = track_obj.state.get('name', f'Track {track_index + 1}')
        track_node.track_visual.muted = track_obj.state.get('muted', False)
        track_node.track_visual.solo = track_obj.state.get('solo', False)
        
        # Store mapping
        self.track_map[track_obj.id] = track_node
        
        return track_node
    
    def sync_playhead(self, playhead_node: PlayheadNode, sequencer: GraphObject):
        """Update playhead position from sequencer state"""
        beat = sequencer.state.get('current_beat', 0.0)
        playhead_node.beat = beat
    
    def update_clip_triggers(self, dt: float):
        """Fade out triggered clip visuals"""
        for clip_node in self.clip_map.values():
            cv = clip_node.clip_visual
            if cv.triggered:
                elapsed = time.time() - cv.trigger_time
                if elapsed > 0.2:  # Fade after 200ms
                    cv.triggered = False


# ============================================================================
# PART 14: DR BEAT APPLICATION
# ============================================================================

class DrBeatApp:
    """Main application combining all systems"""
    
    def __init__(self):
        # Theme
        self.theme = Theme()
        
        # Logic layer (GraphObject)
        self.graph = Graph("dr_beat")
        self.prototypes = create_sequencer_prototypes(self.graph)
        
        # UI layer (SceneGraph)
        self.scene = SceneGraph()
        
        # Bridge
        self.bridge = UIBridge(self.graph, self.scene, self.theme)
        
        # Event dispatcher
        self.dispatcher = EventDispatcher(self.scene.root)
        
        # Undo/Redo action stack
        self.action_stack = ActionStack()
        
        # MIDI router (optional)
        self.midi_router = create_midi_router(self.graph)
        
        # Create sequencer
        self.sequencer = self.graph.create("sequencer", self.prototypes['sequencer'])
        self.sequencer.bpm = 120
        self.sequencer.loop_start = 0
        self.sequencer.loop_end = 16
        
        # Create pattern
        self.pattern = self.graph.create("pattern_a", self.prototypes['pattern'])
        self.pattern.name = "Pattern A"
        self.pattern.length_beats = 16
        self.sequencer.current_pattern = self.pattern
        
        # UI elements
        self.root_panel: PanelNode = None
        self.transport: TransportNode = None
        self.grid: GridNode = None
        self.playhead: PlayheadNode = None
        
        # Timing
        self.last_update = time.time()
        
        # Build UI
        self._build_ui()
    
    def _build_ui(self):
        """Build the UI hierarchy"""
        # Root panel
        self.root_panel = self.scene.create_widget(
            PanelNode, "root_panel", theme=self.theme)
        self.root_panel.set_position(0, 0)
        self.root_panel.set_size(800, 400)
        self.root_panel.set_depth_bias(-1000)
        
        # Transport
        self.transport = TransportNode(self.scene.bus, "transport", self.theme)
        self.transport.set_position(10, 10)
        self.root_panel.add_child(self.transport)
        self.scene._nodes[self.transport.id] = self.transport
        
        # Wire transport to sequencer
        self.transport.on_play(lambda playing: self.sequencer.play() if playing else None)
        self.transport.on_stop(lambda: self.sequencer.stop())
        self.transport.on_bpm_change(lambda bpm: self.sequencer.set_bpm(bpm))
        
        # Grid
        self.grid = GridNode(self.scene.bus, "grid", tracks=4, beats=16, theme=self.theme)
        self.grid.set_position(10, 60)
        gv = self.grid.grid_visual
        self.grid.set_size(gv.beats * gv.beat_width, gv.tracks * gv.track_height)
        self.root_panel.add_child(self.grid)
        self.scene._nodes[self.grid.id] = self.grid
        
        # Playhead
        self.playhead = PlayheadNode(self.scene.bus, "playhead")
        self.playhead.set_position(0, 0)
        self.playhead.set_size(2, gv.tracks * gv.track_height)
        self.playhead.pixels_per_beat = gv.beat_width
        self.grid.add_child(self.playhead)
        self.scene._nodes[self.playhead.id] = self.playhead
        
        # Create tracks and clips
        self._create_sample_pattern()
    
    def _create_sample_pattern(self):
        """Create a sample drum pattern"""
        track_names = ["Kick", "Snare", "HiHat", "Clap"]
        
        # Pattern: basic 4/4 drum beat
        patterns = [
            [0, 4, 8, 12],           # Kick on 1, 2, 3, 4
            [4, 12],                  # Snare on 2, 4
            [0, 2, 4, 6, 8, 10, 12, 14],  # HiHat on every 8th
            [4, 12],                  # Clap with snare
        ]
        
        for track_idx, (name, beats) in enumerate(zip(track_names, patterns)):
            # Create GraphObject track
            track_obj = self.graph.create(name, self.prototypes['track'])
            track_obj.name = name
            track_obj.color = self.theme.get_track_color(track_idx)
            self.pattern.add_track(track_obj)
            
            # Create UI track
            track_node = self.bridge.create_track_node(track_obj, track_idx)
            self.grid.add_track_node(track_node)
            
            # Register in scene
            self.scene._nodes[track_node.id] = track_node
            
            # Create clips
            for beat in beats:
                # GraphObject clip
                clip_obj = self.graph.create(f"{name}_{beat}", self.prototypes['clip'])
                clip_obj.sample_name = name
                clip_obj.start_beat = beat
                clip_obj.duration = 1.0
                clip_obj.velocity = 100
                track_obj.add_clip(clip_obj)
                
                # UI clip
                clip_node = self.bridge.create_clip_node(clip_obj, track_idx, beat, 1.0)
                track_node.add_clip_node(clip_node, beat)
                
                # Wire trigger callback
                clip_obj.on('triggered', lambda c, v, node=clip_node: self._on_clip_triggered(node, v))
                
                # Register in scene
                self.scene._nodes[clip_node.id] = clip_node
    
    def _on_clip_triggered(self, clip_node: ClipNode, velocity: int):
        """Handle clip trigger from sequencer"""
        clip_node.clip_visual.triggered = True
        clip_node.clip_visual.trigger_time = time.time()
    
    def update(self, dt: float = None):
        """Update game state"""
        if dt is None:
            now = time.time()
            dt = now - self.last_update
            self.last_update = now
        
        # Update sequencer
        self.sequencer.tick(dt)
        
        # Sync playhead
        self.bridge.sync_playhead(self.playhead, self.sequencer)
        
        # Update clip visuals
        self.bridge.update_clip_triggers(dt)
        
        # Process UI events
        self.scene.update()
    
    def render(self, ctx: RenderContext):
        """Render the UI"""
        ctx.begin()
        
        # Get nodes in depth order
        nodes = self.scene.get_render_order()
        
        for node in nodes:
            if not node.visible:
                continue
            
            self._render_node(ctx, node)
        
        ctx.end()
    
    def _render_node(self, ctx: RenderContext, node: Node):
        """Render a single node"""
        depth = node.effective_depth
        bounds = node.bounds
        
        # Render based on node type
        if isinstance(node, ClipNode):
            self._render_clip(ctx, node, depth)
        elif isinstance(node, TrackNode):
            self._render_track(ctx, node, depth)
        elif isinstance(node, GridNode):
            self._render_grid(ctx, node, depth)
        elif isinstance(node, PlayheadNode):
            self._render_playhead(ctx, node, depth)
        elif isinstance(node, TransportNode):
            self._render_transport(ctx, node, depth)
        elif isinstance(node, ButtonNode):
            self._render_button(ctx, node, depth)
        elif isinstance(node, ToggleNode):
            self._render_toggle(ctx, node, depth)
        elif isinstance(node, SliderNode):
            self._render_slider(ctx, node, depth)
        elif isinstance(node, LabelNode):
            self._render_label(ctx, node, depth)
        elif isinstance(node, PanelNode):
            self._render_panel(ctx, node, depth)
    
    def _render_clip(self, ctx: RenderContext, node: ClipNode, depth: float):
        cv = node.clip_visual
        style = node.visual
        
        char = '█' if cv.triggered else ('▓' if cv.selected else '░')
        
        if isinstance(ctx, ConsoleRenderContext):
            ctx.draw_rect(node.world_x, node.world_y, node.width, node.height,
                         style.background, style.border_color, depth=depth, char=char)
    
    def _render_track(self, ctx: RenderContext, node: TrackNode, depth: float):
        """Render track header with name, mute/solo indicators"""
        tv = node.track_visual
        
        if isinstance(ctx, ConsoleRenderContext):
            # Track name (first 6 chars)
            name = tv.name[:6].ljust(6)
            
            # Mute indicator
            mute_char = 'M' if tv.muted else '.'
            
            # Solo indicator
            solo_char = 'S' if tv.solo else '.'
            
            # Build header string
            header = f"{name}{mute_char}{solo_char}"
            
            # Draw at left side of track
            ctx.draw_text(node.world_x - 80, node.world_y + 5, header, depth=depth + 1)
    
    def _render_grid(self, ctx: RenderContext, node: GridNode, depth: float):
        gv = node.grid_visual
        
        if isinstance(ctx, ConsoleRenderContext):
            # Draw horizontal lines
            for t in range(gv.tracks + 1):
                y = node.world_y + t * gv.track_height
                ctx.draw_line(node.world_x, y, node.world_x + gv.beats * gv.beat_width, y,
                             Theme.BORDER, depth=depth - 10)
            
            # Draw vertical lines
            for b in range(gv.beats + 1):
                x = node.world_x + b * gv.beat_width
                ctx.draw_line(x, node.world_y, x, node.world_y + gv.tracks * gv.track_height,
                             Theme.BORDER, depth=depth - 10)
    
    def _render_playhead(self, ctx: RenderContext, node: PlayheadNode, depth: float):
        pv = node.playhead_visual
        
        if isinstance(ctx, ConsoleRenderContext):
            ctx.draw_line(node.world_x, node.world_y, 
                         node.world_x, node.world_y + node.height,
                         Theme.PLAYHEAD, depth=depth)
    
    def _render_transport(self, ctx: RenderContext, node: TransportNode, depth: float):
        # Transport background
        if isinstance(ctx, ConsoleRenderContext):
            tv = node.transport_visual
            status = "▶" if tv.playing else "■"
            bpm_text = f"{status} {tv.bpm} BPM"
            ctx.draw_text(node.world_x, node.world_y, bpm_text, depth=depth)
    
    def _render_button(self, ctx: RenderContext, node: ButtonNode, depth: float):
        style = node.visual
        if isinstance(ctx, ConsoleRenderContext):
            ctx.draw_text(node.world_x, node.world_y, f"[{node.text}]", depth=depth)
    
    def _render_toggle(self, ctx: RenderContext, node: ToggleNode, depth: float):
        style = node.visual
        if isinstance(ctx, ConsoleRenderContext):
            prefix = "●" if node.active else "○"
            ctx.draw_text(node.world_x, node.world_y, f"{prefix}{node.text}", depth=depth)
    
    def _render_slider(self, ctx: RenderContext, node: SliderNode, depth: float):
        if isinstance(ctx, ConsoleRenderContext):
            bar_width = 10
            filled = int(node.normalized * bar_width)
            bar = '█' * filled + '░' * (bar_width - filled)
            ctx.draw_text(node.world_x, node.world_y, f"[{bar}]", depth=depth)
    
    def _render_label(self, ctx: RenderContext, node: LabelNode, depth: float):
        if isinstance(ctx, ConsoleRenderContext):
            ctx.draw_text(node.world_x, node.world_y, node.text, depth=depth)
    
    def _render_panel(self, ctx: RenderContext, node: PanelNode, depth: float):
        """Render panel background"""
        style = node.visual
        
        if isinstance(ctx, ConsoleRenderContext):
            # Draw panel background as a filled rectangle
            ctx.draw_rect(
                node.world_x, node.world_y, 
                node.width, node.height,
                fill=style.background,
                depth=depth,
                char='░'  # Light fill for background
            )
    
    # Input routing
    
    def mouse_move(self, x: float, y: float):
        self.dispatcher.mouse_move(x, y)
    
    def mouse_down(self, x: float, y: float, button: int = 0):
        self.dispatcher.mouse_down(x, y, button)
    
    def mouse_up(self, x: float, y: float, button: int = 0):
        self.dispatcher.mouse_up(x, y, button)
    
    def key_down(self, key: str, modifiers: Dict[str, bool] = None):
        """Handle keyboard shortcuts
        
        Shortcuts:
        - Space: Play/Pause toggle
        - R: Toggle record mode
        - L: Toggle loop
        - 1-4: Select track (and focus)
        - M: Mute selected track
        - S: Solo selected track
        - Delete: Delete selected clip
        - Ctrl+Z: Undo
        - Ctrl+Shift+Z / Ctrl+Y: Redo
        - Ctrl+A: Select all clips
        - Escape: Clear selection
        """
        modifiers = modifiers or {}
        ctrl = modifiers.get('ctrl', False)
        shift = modifiers.get('shift', False)
        
        # Update dispatcher modifiers
        self.dispatcher.set_modifiers(shift=shift, ctrl=ctrl)
        
        # Ctrl+Z: Undo
        if key == 'z' and ctrl and not shift:
            if self.action_stack.can_undo():
                self.action_stack.undo()
            return True
        
        # Ctrl+Shift+Z or Ctrl+Y: Redo
        if (key == 'z' and ctrl and shift) or (key == 'y' and ctrl):
            if self.action_stack.can_redo():
                self.action_stack.redo()
            return True
        
        # Ctrl+A: Select all clips
        if key == 'a' and ctrl:
            all_clips = list(self.bridge.clip_map.values())
            self.dispatcher.select_all(all_clips)
            return True
        
        # Space: Play/Pause toggle
        if key == ' ':
            self.transport.play_btn.active = not self.transport.play_btn.active
            return True
        
        # R: Toggle record
        if key == 'r' and not ctrl:
            self.transport.record_btn.active = not self.transport.record_btn.active
            return True
        
        # L: Toggle loop
        if key == 'l' and not ctrl:
            self.transport.loop_btn.active = not self.transport.loop_btn.active
            return True
        
        # 1-4: Select track
        if key in ('1', '2', '3', '4'):
            track_idx = int(key) - 1
            if track_idx < len(self.grid.track_nodes):
                track = self.grid.track_nodes[track_idx]
                self.dispatcher.focused = track
                return True
        
        # M: Mute focused/selected track
        if key == 'm' and not ctrl:
            focused = self.dispatcher.focused
            if isinstance(focused, TrackNode):
                self.action_stack.execute(ToggleMuteAction(focused))
            elif isinstance(focused, ClipNode):
                # Find parent track
                parent = focused.parent
                while parent and not isinstance(parent, TrackNode):
                    parent = parent.parent
                if isinstance(parent, TrackNode):
                    self.action_stack.execute(ToggleMuteAction(parent))
            return True
        
        # S: Solo focused/selected track
        if key == 's' and not ctrl:
            focused = self.dispatcher.focused
            if isinstance(focused, TrackNode):
                self.action_stack.execute(ToggleSoloAction(focused))
            elif isinstance(focused, ClipNode):
                parent = focused.parent
                while parent and not isinstance(parent, TrackNode):
                    parent = parent.parent
                if isinstance(parent, TrackNode):
                    self.action_stack.execute(ToggleSoloAction(parent))
            return True
        
        # Delete: Delete selected clip(s)
        if key == 'Delete' or key == 'Backspace':
            for widget in list(self.dispatcher.selection):
                if isinstance(widget, ClipNode):
                    parent = widget.parent
                    while parent and not isinstance(parent, TrackNode):
                        parent = parent.parent
                    if isinstance(parent, TrackNode):
                        self.action_stack.execute(
                            DeleteClipAction(widget, parent, self.grid))
            self.dispatcher.clear_selection()
            return True
        
        # Escape: Clear selection
        if key == 'Escape':
            self.dispatcher.clear_selection()
            return True
        
        return self.dispatcher.key_down(key, modifiers)


# ============================================================================
# PART 15: DEMO / MAIN
# ============================================================================

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def demo_interactive():
    """Interactive demo with real-time updates"""
    print("=" * 70)
    print("  DR BEAT GRAPH UI - Interactive Demo")
    print("=" * 70)
    print()
    print("Controls:")
    print("  SPACE - Play/Stop")
    print("  Q - Quit")
    print("  Click on clips to trigger them")
    print()
    print("Press ENTER to start...")
    input()
    
    # Create app
    app = DrBeatApp()
    ctx = ConsoleRenderContext(width=80, height=20)
    ctx.scale_x = 12
    ctx.scale_y = 10
    
    # Start playing
    app.transport.play_btn.active = True
    app.sequencer.play()
    
    running = True
    frame = 0
    
    try:
        while running:
            clear_screen()
            
            # Update
            app.update()
            
            # Render
            app.render(ctx)
            
            # Display
            print("╔" + "═" * 78 + "╗")
            print("║  DR BEAT - Finger Drumming Practice" + " " * 40 + "║")
            print("╠" + "═" * 78 + "╣")
            
            # Transport status
            tv = app.transport.transport_visual
            status = "▶ PLAYING" if tv.playing else "■ STOPPED"
            beat = app.sequencer.state.get('current_beat', 0)
            print(f"║  {status}  BPM: {tv.bpm}  Beat: {beat:5.1f}  " + " " * 35 + "║")
            print("╠" + "═" * 78 + "╣")
            
            # Grid
            for line in ctx.get_output().split('\n')[:16]:
                print(f"║  {line:<76}║")
            
            print("╠" + "═" * 78 + "╣")
            print("║  [SPACE] Play/Stop   [Q] Quit" + " " * 47 + "║")
            print("╚" + "═" * 78 + "╝")
            
            # Check for input (non-blocking would require platform-specific code)
            # For demo, just run for a bit then stop
            frame += 1
            if frame > 200:  # ~6 seconds at 30fps
                running = False
            
            time.sleep(1/30)  # ~30 FPS
            
    except KeyboardInterrupt:
        pass
    
    print("\nDemo complete!")


def demo_static():
    """Static demo showing the system architecture"""
    print("=" * 70)
    print("  DR BEAT GRAPH UI - Architecture Demo")
    print("=" * 70)
    
    # Create app
    app = DrBeatApp()
    
    print("\n1. GRAPH (Logic Layer)")
    print("-" * 40)
    print(f"   Graph: {app.graph.name}")
    print(f"   Objects: {len(app.graph.objects)}")
    for name, obj in list(app.graph.by_name.items())[:10]:
        state_keys = list(obj.state._data.keys())[:3]
        print(f"   - {name}: {state_keys}")
    
    print("\n2. SCENE GRAPH (UI Layer)")
    print("-" * 40)
    print(f"   Nodes: {len(app.scene._nodes)}")
    for node in app.scene.get_render_order()[:10]:
        depth = node.effective_depth
        print(f"   - {node.name}: depth={depth:.1f}, pos=({node.world_x:.0f},{node.world_y:.0f})")
    
    print("\n3. BRIDGE (Connections)")
    print("-" * 40)
    print(f"   Clip mappings: {len(app.bridge.clip_map)}")
    print(f"   Track mappings: {len(app.bridge.track_map)}")
    
    print("\n4. SEQUENCER STATE")
    print("-" * 40)
    print(f"   BPM: {app.sequencer.state.get('bpm')}")
    print(f"   Playing: {app.sequencer.state.get('playing')}")
    pattern_name = app.pattern.state.get('name') or app.pattern.name
    print(f"   Pattern: {pattern_name}")
    print(f"   Tracks: {len(app.pattern.state.get('tracks', []))}")
    
    print("\n5. RENDER OUTPUT")
    print("-" * 40)
    
    ctx = ConsoleRenderContext(width=70, height=12)
    ctx.scale_x = 12
    ctx.scale_y = 12
    
    # Simulate a few beats
    for beat in [0, 4, 8]:
        app.playhead.beat = beat
        app.render(ctx)
        print(f"\n   Beat {beat}:")
        for line in ctx.get_output().split('\n'):
            if line.strip():
                print(f"   {line}")
    
    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


def main():
    """Main entry point"""
    # Fix Windows console encoding
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    
    print()
    print("DR BEAT GRAPH UI")
    print("================")
    print()
    print("Choose demo mode:")
    print("  1. Static architecture demo")
    print("  2. Interactive playback demo")
    print()
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2":
        demo_interactive()
    else:
        demo_static()


if __name__ == "__main__":
    main()


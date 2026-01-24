"""
GRAPH OBJECT SYSTEM
===================

Everything is unified:
- Every OBJECT is a NODE on a graph
- Every OPERATION is a PIPELINE (stages at nodes)
- Every COMMUNICATION is a MESSAGE
- Every STATE is a NAMESPACE

This is a general-purpose "object" system built on graphs.

Think of it like:
- JavaScript's prototype chain + 
- Smalltalk's message passing +
- Unix pipes +
- Actor model

"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    List, Dict, Set, Optional, Callable, Any, TypeVar, Generic,
    Iterator, Tuple, Union, Type
)
from enum import Enum, auto
from collections import deque
import time
import uuid
import heapq
from contextlib import contextmanager


# ============================================================================
# PART 1: MESSAGE (How objects communicate)
# ============================================================================

class MessageType(Enum):
    """Types of messages"""
    CALL = auto()       # Method call
    RETURN = auto()     # Return value
    EVENT = auto()      # Event notification
    QUERY = auto()      # Ask for data
    COMMAND = auto()    # Tell to do something
    SIGNAL = auto()     # Generic signal
    ERROR = auto()      # Error occurred


@dataclass
class Message:
    """
    Messages flow through the graph.
    
    Everything communicates via messages:
    - Method calls are messages
    - Events are messages
    - Data transfer is messages
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    msg_type: MessageType = MessageType.SIGNAL
    
    # Routing
    sender: Optional[str] = None
    target: Optional[str] = None
    
    # Content
    name: str = ""           # Method/event name
    args: Tuple = ()         # Positional arguments
    kwargs: Dict = field(default_factory=dict)  # Keyword arguments
    payload: Any = None      # Arbitrary data
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None  # For request/response
    
    # Chain for pipeline processing
    history: List[str] = field(default_factory=list)
    
    def reply(self, payload: Any) -> Message:
        """Create reply message"""
        return Message(
            msg_type=MessageType.RETURN,
            sender=self.target,
            target=self.sender,
            reply_to=self.id,
            payload=payload
        )
    
    def forward(self, new_target: str) -> Message:
        """Forward message to new target"""
        new_msg = Message(
            msg_type=self.msg_type,
            sender=self.target,  # Current target becomes sender
            target=new_target,
            name=self.name,
            args=self.args,
            kwargs=self.kwargs,
            payload=self.payload,
            history=self.history + [self.target or ""]
        )
        return new_msg
    
    def __repr__(self):
        return f"Message({self.msg_type.name}: {self.name or self.payload})"


class MessageFactory:
    """
    Factory for creating messages.
    
    Centralizes message creation with defaults.
    """
    
    def __init__(self, default_sender: str = ""):
        self.default_sender = default_sender
        self.message_count = 0
    
    def call(self, target: str, method: str, *args, **kwargs) -> Message:
        """Create method call message"""
        self.message_count += 1
        return Message(
            msg_type=MessageType.CALL,
            sender=self.default_sender,
            target=target,
            name=method,
            args=args,
            kwargs=kwargs
        )
    
    def event(self, name: str, payload: Any = None) -> Message:
        """Create event message"""
        self.message_count += 1
        return Message(
            msg_type=MessageType.EVENT,
            sender=self.default_sender,
            name=name,
            payload=payload
        )
    
    def query(self, target: str, what: str) -> Message:
        """Create query message"""
        self.message_count += 1
        return Message(
            msg_type=MessageType.QUERY,
            sender=self.default_sender,
            target=target,
            name=what
        )
    
    def command(self, target: str, action: str, *args, **kwargs) -> Message:
        """Create command message"""
        self.message_count += 1
        return Message(
            msg_type=MessageType.COMMAND,
            sender=self.default_sender,
            target=target,
            name=action,
            args=args,
            kwargs=kwargs
        )
    
    def signal(self, payload: Any = None) -> Message:
        """Create simple signal"""
        self.message_count += 1
        return Message(
            msg_type=MessageType.SIGNAL,
            sender=self.default_sender,
            payload=payload
        )


# ============================================================================
# PART 2: NAMESPACE (Object state)
# ============================================================================

class Namespace:
    """
    Stores object state with prototype inheritance.
    
    Like JavaScript's prototype chain.
    """
    
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
    
    def __contains__(self, key: str) -> bool:
        if key in self._data:
            return True
        if self._prototype:
            return key in self._prototype
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default
    
    def keys(self) -> List[str]:
        own = list(self._data.keys())
        if self._prototype:
            inherited = [k for k in self._prototype.keys() if k not in own]
            return own + inherited
        return own
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: self.get(k) for k in self.keys()}
    
    def create_child(self, name: str = "") -> Namespace:
        return Namespace(name, prototype=self)


# ============================================================================
# PART 3: CALLBACK QUEUE
# ============================================================================

@dataclass(order=True)
class QueuedCallback:
    """A callback in the queue"""
    priority: int
    sequence: int
    callback: Callable = field(compare=False)
    name: str = field(default="", compare=False)


class CallbackQueue:
    """
    Priority queue of callbacks.
    
    Callbacks execute in priority order when triggered.
    Can be one-shot (removed after execution) or persistent.
    """
    
    def __init__(self, persistent: bool = True):
        self._queue: List[QueuedCallback] = []
        self._sequence = 0
        self.persistent = persistent  # If True, callbacks stay after execution
    
    def add(self, callback: Callable, priority: int = 50, name: str = "") -> QueuedCallback:
        """Add callback to queue"""
        self._sequence += 1
        qc = QueuedCallback(priority, self._sequence, callback, name)
        heapq.heappush(self._queue, qc)
        return qc
    
    def execute(self, *args, **kwargs) -> List[Any]:
        """Execute all callbacks in priority order"""
        results = []
        
        # Sort callbacks by priority
        callbacks = sorted(self._queue, key=lambda x: (x.priority, x.sequence))
        
        if not self.persistent:
            # One-shot: clear queue
            self._queue.clear()
        
        for qc in callbacks:
            try:
                result = qc.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    def execute_once(self, *args, **kwargs) -> List[Any]:
        """Execute all callbacks and remove them"""
        results = []
        callbacks = []
        
        while self._queue:
            callbacks.append(heapq.heappop(self._queue))
        
        for qc in callbacks:
            try:
                result = qc.callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(e)
        
        return results
    
    def peek(self) -> Optional[QueuedCallback]:
        return self._queue[0] if self._queue else None
    
    def clear(self) -> None:
        self._queue.clear()
    
    def __len__(self):
        return len(self._queue)
    
    def __bool__(self):
        return len(self._queue) > 0


# ============================================================================
# PART 4: LINK (Graph edges)
# ============================================================================

class LinkType(Enum):
    """Types of links between nodes"""
    CHILD = auto()      # Parent-child (tree structure)
    REFERENCE = auto()  # Reference to another object
    PIPE = auto()       # Data flows through
    EVENT = auto()      # Event propagation
    PROTOTYPE = auto()  # Prototype inheritance
    DEPENDENCY = auto() # Depends on


@dataclass
class Link:
    """
    Connection between two nodes.
    
    Links form the graph structure.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    link_type: LinkType = LinkType.REFERENCE
    
    source: Optional[GraphObject] = None
    target: Optional[GraphObject] = None
    
    name: str = ""          # Named relationship
    weight: float = 1.0     # For weighted graphs
    bidirectional: bool = False
    
    # For pipes
    transform: Optional[Callable] = None  # Transform data as it flows
    
    def traverse(self, message: Message) -> Optional[Message]:
        """Pass message through link"""
        if self.transform:
            message.payload = self.transform(message.payload)
        return message
    
    def __repr__(self):
        src = self.source.name if self.source else "?"
        tgt = self.target.name if self.target else "?"
        return f"Link({src} --{self.link_type.name}--> {tgt})"


# ============================================================================
# PART 5: PIPELINE STAGE (Operations as graph nodes)
# ============================================================================

class Stage:
    """
    A single stage in a pipeline.
    
    Operations on objects are pipelines.
    Each stage is a processing step.
    """
    
    def __init__(self, 
                 name: str = "",
                 process: Optional[Callable[[Message], Message]] = None):
        self.name = name
        self._process = process or (lambda m: m)
        self.next_stages: List[Stage] = []
        self.enabled = True
        
        # Stats
        self.executions = 0
    
    def execute(self, message: Message) -> Message:
        """Execute this stage"""
        if not self.enabled:
            return message
        
        message.history.append(self.name)
        self.executions += 1
        return self._process(message)
    
    def then(self, next_stage: Stage) -> Stage:
        """Chain to next stage (returns next for fluent API)"""
        self.next_stages.append(next_stage)
        return next_stage
    
    def __repr__(self):
        return f"Stage({self.name})"


class Pipeline:
    """
    A sequence of stages that process messages.
    
    Methods on objects are pipelines.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self.stages: List[Stage] = []
        self.input_queue = CallbackQueue()
        self.output_queue = CallbackQueue()
    
    def add_stage(self, 
                  name: str,
                  process: Callable[[Message], Message]) -> Stage:
        """Add a stage to the pipeline"""
        stage = Stage(name, process)
        if self.stages:
            self.stages[-1].then(stage)
        self.stages.append(stage)
        return stage
    
    def execute(self, message: Message) -> Message:
        """Execute entire pipeline"""
        # Pre-processing callbacks
        self.input_queue.execute(message)
        
        # Execute stages
        current = message
        for stage in self.stages:
            current = stage.execute(current)
        
        # Post-processing callbacks
        self.output_queue.execute(current)
        
        return current
    
    def __call__(self, message: Message) -> Message:
        return self.execute(message)
    
    def __repr__(self):
        return f"Pipeline({self.name}, stages={len(self.stages)})"


# ============================================================================
# PART 6: GRAPH OBJECT (The unified object)
# ============================================================================

class GraphObject:
    """
    THE UNIFIED OBJECT
    
    Every object in the system is a GraphObject:
    - Lives on a graph (has links to other objects)
    - Has state (namespace with prototype inheritance)
    - Has operations (pipelines attached as methods)
    - Communicates via messages
    - Has callback queues for events
    
    This unifies:
    - OOP (objects with state and methods)
    - Graph theory (nodes and edges)
    - Dataflow (pipelines)
    - Events (callbacks)
    - Prototypes (inheritance)
    """
    
    _id_counter = 0
    
    def __init__(self, 
                 name: str = "",
                 prototype: Optional[GraphObject] = None):
        GraphObject._id_counter += 1
        self.id = GraphObject._id_counter
        self.name = name or f"obj_{self.id}"
        
        # STATE: Namespace with prototype inheritance
        proto_ns = prototype.state if prototype else None
        self.state = Namespace(self.name, prototype=proto_ns)
        
        # GRAPH: Links to other objects
        self.links: Dict[str, Link] = {}  # Named links
        self.incoming: List[Link] = []    # Links pointing to this
        self.outgoing: List[Link] = []    # Links from this
        
        # OPERATIONS: Pipelines as methods
        self.pipelines: Dict[str, Pipeline] = {}
        
        # EVENTS: Callback queues for different events
        self.events: Dict[str, CallbackQueue] = {}
        
        # MESSAGE: Factory for creating messages
        self.messages = MessageFactory(self.name)
        
        # PROTOTYPE: For creating child objects
        self.prototype = prototype
        
        # INBOX: Messages waiting to be processed
        self.inbox: deque = deque()
    
    # --- State Access ---
    
    def __getattr__(self, key: str) -> Any:
        """Access state or pipelines"""
        # Check if it's a core attribute
        if key in ('id', 'name', 'state', 'links', 'incoming', 'outgoing',
                   'pipelines', 'events', 'messages', 'prototype', 'inbox'):
            return object.__getattribute__(self, key)
        
        # Check own pipelines (methods)
        pipelines = object.__getattribute__(self, 'pipelines')
        if key in pipelines:
            # Return bound method-like callable
            return lambda *args, **kwargs: self._call_pipeline(key, *args, **kwargs)
        
        # Check prototype's pipelines (inherited methods)
        prototype = object.__getattribute__(self, 'prototype')
        if prototype and key in prototype.pipelines:
            # Return callable that uses prototype's pipeline but with self
            return lambda *args, **kwargs: self._call_inherited_pipeline(prototype, key, *args, **kwargs)
        
        # Check state
        state = object.__getattribute__(self, 'state')
        return getattr(state, key)
    
    def __setattr__(self, key: str, value: Any):
        """Set state"""
        if key in ('id', 'name', 'state', 'links', 'incoming', 'outgoing',
                   'pipelines', 'events', 'messages', 'prototype', 'inbox'):
            object.__setattr__(self, key, value)
        else:
            self.state._data[key] = value
    
    # --- Pipeline/Method Operations ---
    
    def define(self, name: str) -> Pipeline:
        """
        Define a new operation (method) as a pipeline.
        
        Usage:
            obj.define("greet").add_stage("say_hi", lambda m: ...)
        """
        pipeline = Pipeline(name)
        self.pipelines[name] = pipeline
        return pipeline
    
    def _call_pipeline(self, name: str, *args, **kwargs) -> Any:
        """Execute a pipeline (method call)"""
        if name not in self.pipelines:
            raise AttributeError(f"'{self.name}' has no method '{name}'")
        
        # Create message for this call
        msg = Message(
            msg_type=MessageType.CALL,
            sender=self.name,
            target=self.name,
            name=name,
            args=args,
            kwargs=kwargs,
            payload={'self': self}
        )
        
        # Execute pipeline
        result = self.pipelines[name].execute(msg)
        return result.payload
    
    def _call_inherited_pipeline(self, prototype: GraphObject, name: str, *args, **kwargs) -> Any:
        """Execute an inherited pipeline (from prototype)"""
        if name not in prototype.pipelines:
            raise AttributeError(f"'{prototype.name}' has no method '{name}'")
        
        # Create message with SELF as this object (not prototype)
        msg = Message(
            msg_type=MessageType.CALL,
            sender=self.name,
            target=self.name,
            name=name,
            args=args,
            kwargs=kwargs,
            payload={'self': self}  # Key: self is THIS object
        )
        
        # Execute prototype's pipeline
        result = prototype.pipelines[name].execute(msg)
        return result.payload
    
    # --- Graph Operations ---
    
    def link_to(self, 
                other: GraphObject, 
                link_type: LinkType = LinkType.REFERENCE,
                name: str = "") -> Link:
        """Create link to another object"""
        link = Link(
            link_type=link_type,
            source=self,
            target=other,
            name=name or other.name
        )
        
        self.links[link.name] = link
        self.outgoing.append(link)
        other.incoming.append(link)
        
        return link
    
    def get_linked(self, name: str) -> Optional[GraphObject]:
        """Get linked object by name"""
        if name in self.links:
            return self.links[name].target
        return None
    
    def children(self) -> List[GraphObject]:
        """Get child objects"""
        return [l.target for l in self.outgoing if l.link_type == LinkType.CHILD]
    
    def parent(self) -> Optional[GraphObject]:
        """Get parent object"""
        for link in self.incoming:
            if link.link_type == LinkType.CHILD:
                return link.source
        return None
    
    # --- Event Operations ---
    
    def on(self, event: str, callback: Callable, priority: int = 50) -> None:
        """Subscribe to event"""
        if event not in self.events:
            self.events[event] = CallbackQueue()
        self.events[event].add(callback, priority, event)
    
    def emit(self, event: str, *args, **kwargs) -> List[Any]:
        """Emit event"""
        if event in self.events:
            return self.events[event].execute(*args, **kwargs)
        return []
    
    def emit_to_children(self, event: str, *args, **kwargs) -> Dict[str, List]:
        """Emit event to all children"""
        results = {}
        for child in self.children():
            results[child.name] = child.emit(event, *args, **kwargs)
        return results
    
    # --- Message Operations ---
    
    def send(self, target: GraphObject, method: str, *args, **kwargs) -> Message:
        """Send message to another object"""
        msg = self.messages.call(target.name, method, *args, **kwargs)
        target.receive(msg)
        return msg
    
    def receive(self, message: Message) -> None:
        """Receive a message"""
        self.inbox.append(message)
    
    def process_inbox(self) -> List[Any]:
        """Process all messages in inbox"""
        results = []
        while self.inbox:
            msg = self.inbox.popleft()
            result = self._handle_message(msg)
            results.append(result)
        return results
    
    def _handle_message(self, msg: Message) -> Any:
        """Handle a single message"""
        if msg.msg_type == MessageType.CALL:
            # Method call
            if msg.name in self.pipelines:
                return self._call_pipeline(msg.name, *msg.args, **msg.kwargs)
        elif msg.msg_type == MessageType.EVENT:
            # Event
            return self.emit(msg.name, msg.payload)
        elif msg.msg_type == MessageType.QUERY:
            # Query state
            return self.state.get(msg.name)
        
        return None
    
    # --- Prototype Operations ---
    
    def create_child(self, name: str = "") -> GraphObject:
        """Create child object (prototype inheritance)"""
        child = GraphObject(name or f"{self.name}_child", prototype=self)
        self.link_to(child, LinkType.CHILD, child.name)
        return child
    
    def clone(self) -> GraphObject:
        """Clone this object"""
        new_obj = GraphObject(f"{self.name}_clone", prototype=self.prototype)
        # Copy state
        for key in self.state._data:
            new_obj.state._data[key] = self.state._data[key]
        # Copy pipeline definitions (not execution state)
        for name, pipeline in self.pipelines.items():
            new_obj.pipelines[name] = pipeline
        return new_obj
    
    def __repr__(self):
        return f"GraphObject({self.id}: {self.name}, state={list(self.state._data.keys())}, methods={list(self.pipelines.keys())})"


# ============================================================================
# PART 7: GRAPH (The world)
# ============================================================================

class Graph:
    """
    The graph that holds all objects.
    
    This is "the world" - all objects live here.
    """
    
    def __init__(self, name: str = "world"):
        self.name = name
        
        # All objects
        self.objects: Dict[int, GraphObject] = {}
        self.by_name: Dict[str, GraphObject] = {}
        
        # Root objects (entry points)
        self.roots: List[GraphObject] = []
        
        # Global namespace (inherited by all)
        self.globals = Namespace("globals")
        
        # Global event bus
        self.events: Dict[str, CallbackQueue] = {}
        
        # Message factory
        self.messages = MessageFactory("graph")
        
        # Stats
        self.tick_count = 0
    
    def create(self, name: str = "", prototype: Optional[GraphObject] = None) -> GraphObject:
        """Create new object in graph"""
        obj = GraphObject(name, prototype)
        self.objects[obj.id] = obj
        self.by_name[obj.name] = obj
        return obj
    
    def add(self, obj: GraphObject) -> GraphObject:
        """Add existing object to graph"""
        self.objects[obj.id] = obj
        self.by_name[obj.name] = obj
        return obj
    
    def get(self, name: str) -> Optional[GraphObject]:
        """Get object by name"""
        return self.by_name.get(name)
    
    def remove(self, obj: GraphObject) -> None:
        """Remove object from graph"""
        if obj.id in self.objects:
            del self.objects[obj.id]
        if obj.name in self.by_name:
            del self.by_name[obj.name]
    
    # --- Traversal ---
    
    def traverse_dfs(self, start: GraphObject) -> Iterator[GraphObject]:
        """Depth-first traversal"""
        visited = set()
        stack = [start]
        
        while stack:
            obj = stack.pop()
            if obj.id in visited:
                continue
            visited.add(obj.id)
            yield obj
            
            for link in reversed(obj.outgoing):
                if link.target.id not in visited:
                    stack.append(link.target)
    
    def traverse_bfs(self, start: GraphObject) -> Iterator[GraphObject]:
        """Breadth-first traversal"""
        visited = set()
        queue = deque([start])
        
        while queue:
            obj = queue.popleft()
            if obj.id in visited:
                continue
            visited.add(obj.id)
            yield obj
            
            for link in obj.outgoing:
                if link.target.id not in visited:
                    queue.append(link.target)
    
    def find(self, predicate: Callable[[GraphObject], bool]) -> List[GraphObject]:
        """Find objects matching predicate"""
        return [obj for obj in self.objects.values() if predicate(obj)]
    
    # --- Events ---
    
    def on(self, event: str, callback: Callable, priority: int = 50) -> None:
        """Global event subscription"""
        if event not in self.events:
            self.events[event] = CallbackQueue()
        self.events[event].add(callback, priority)
    
    def emit(self, event: str, *args, **kwargs) -> List[Any]:
        """Global event emission"""
        if event in self.events:
            return self.events[event].execute(*args, **kwargs)
        return []
    
    def broadcast(self, event: str, *args, **kwargs) -> Dict[str, List]:
        """Broadcast event to all objects"""
        results = {}
        for obj in self.objects.values():
            results[obj.name] = obj.emit(event, *args, **kwargs)
        return results
    
    # --- Tick (process all inboxes) ---
    
    def tick(self) -> Dict[str, List]:
        """Process one tick - all objects process their inboxes"""
        self.tick_count += 1
        results = {}
        for obj in self.objects.values():
            results[obj.name] = obj.process_inbox()
        return results
    
    def __repr__(self):
        return f"Graph({self.name}, objects={len(self.objects)})"


# ============================================================================
# PART 8: FACTORY FOR COMMON PATTERNS
# ============================================================================

class ObjectFactory:
    """
    Factory for creating common object patterns.
    """
    
    def __init__(self, graph: Graph):
        self.graph = graph
    
    def entity(self, name: str, **initial_state) -> GraphObject:
        """Create basic entity with state"""
        obj = self.graph.create(name)
        for key, value in initial_state.items():
            setattr(obj, key, value)
        return obj
    
    def actor(self, name: str) -> GraphObject:
        """
        Create actor (processes messages asynchronously).
        
        An actor:
        - Has inbox for messages
        - Processes messages one at a time
        - Can send messages to other actors
        """
        obj = self.graph.create(name)
        obj.state.actor = True
        
        # Define receive behavior
        obj.define("receive").add_stage("handle", lambda m: m)
        
        return obj
    
    def component(self, name: str, entity: GraphObject) -> GraphObject:
        """
        Create component attached to entity.
        
        Component pattern:
        - Component is owned by entity
        - Can access entity's state
        """
        comp = entity.create_child(name)
        comp.state.is_component = True
        comp.state.owner = entity
        return comp
    
    def pipeline_node(self, name: str) -> GraphObject:
        """
        Create pipeline processing node.
        
        Has input/output and processes data flowing through.
        """
        obj = self.graph.create(name)
        obj.state.is_pipeline_node = True
        
        # Input/output queues
        obj.state.input_buffer = []
        obj.state.output_buffer = []
        
        # Define process operation
        def process_stage(msg):
            # Process input to output
            data = msg.payload.get('data')
            obj.state.output_buffer.append(data)
            return msg
        
        obj.define("process").add_stage("transform", process_stage)
        
        return obj
    
    def state_machine(self, name: str, states: List[str], initial: str) -> GraphObject:
        """
        Create finite state machine.
        """
        obj = self.graph.create(name)
        obj.state.is_fsm = True
        obj.state.states = states
        obj.state.current_state = initial
        obj.state.transitions = {}
        
        def transition_stage(msg):
            event = msg.name
            current = obj.state.current_state
            key = (current, event)
            
            if key in obj.state.transitions:
                new_state = obj.state.transitions[key]
                obj.state.current_state = new_state
                obj.emit("state_changed", current, new_state)
                msg.payload = new_state
            return msg
        
        obj.define("transition").add_stage("check", transition_stage)
        
        return obj
    
    def observable(self, name: str, initial_value: Any = None) -> GraphObject:
        """
        Create observable value (reactive).
        
        Notifies observers when value changes.
        """
        obj = self.graph.create(name)
        obj.state._value = initial_value
        obj.state.observers = []
        
        def get_stage(msg):
            msg.payload = obj.state._value
            return msg
        
        def set_stage(msg):
            old_value = obj.state._value
            new_value = msg.args[0] if msg.args else msg.payload
            obj.state._value = new_value
            
            if old_value != new_value:
                obj.emit("changed", old_value, new_value)
            
            msg.payload = new_value
            return msg
        
        obj.define("get").add_stage("read", get_stage)
        obj.define("set").add_stage("write", set_stage)
        
        return obj


# ============================================================================
# EXAMPLES
# ============================================================================

def example_basic():
    """Basic GraphObject usage"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic GraphObject")
    print("=" * 60)
    
    # Create graph (the world)
    world = Graph("my_world")
    
    # Create an object
    person = world.create("alice")
    person.full_name = "Alice Smith"
    person.age = 30
    person.job = "Engineer"
    
    print(f"\nCreated: {person}")
    print(f"  State: {person.state.to_dict()}")
    
    # Define a method as pipeline
    def greet_stage(msg):
        self = msg.payload.get('self')
        name = self.state.get('full_name', 'Unknown')
        greeting = f"Hello, I'm {name}!"
        msg.payload = greeting
        return msg
    
    person.define("greet").add_stage("say_hello", greet_stage)
    
    # Call the method
    result = person.greet()
    print(f"  person.greet() = {result}")


def example_inheritance():
    """Prototype inheritance"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Prototype Inheritance")
    print("=" * 60)
    
    world = Graph()
    
    # Create prototype
    animal = world.create("Animal")
    animal.species = "Unknown"
    animal.legs = 4
    
    def speak_stage(msg):
        self = msg.payload.get('self')
        sound = self.state.get('sound', '...')
        obj_name = self.name  # GraphObject.name attribute
        msg.payload = f"{obj_name} says: {sound}"
        return msg
    
    animal.define("speak").add_stage("make_sound", speak_stage)
    
    # Create child (inherits from animal)
    dog = animal.create_child("dog")
    dog.species = "Canis familiaris"
    dog.sound = "Woof!"
    
    cat = animal.create_child("cat")
    cat.species = "Felis catus"
    cat.sound = "Meow!"
    
    print(f"\nAnimal: {animal.state.to_dict()}")
    print(f"Dog: species={dog.species}, legs={dog.legs}, sound={dog.sound}")
    print(f"Cat: species={cat.species}, legs={cat.legs}, sound={cat.sound}")
    
    print(f"\ndog.speak() = {dog.speak()}")
    print(f"cat.speak() = {cat.speak()}")


def example_graph_links():
    """Graph relationships"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Graph Relationships")
    print("=" * 60)
    
    world = Graph()
    
    # Create objects
    company = world.create("TechCorp")
    company.industry = "Technology"
    
    ceo = world.create("CEO")
    ceo.title = "Chief Executive Officer"
    
    dev_team = world.create("DevTeam")
    dev_team.size = 10
    
    alice = world.create("Alice")
    alice.role = "Developer"
    
    bob = world.create("Bob")
    bob.role = "Designer"
    
    # Create relationships
    company.link_to(ceo, LinkType.CHILD, "leader")
    company.link_to(dev_team, LinkType.CHILD, "development")
    dev_team.link_to(alice, LinkType.CHILD, "member_1")
    dev_team.link_to(bob, LinkType.CHILD, "member_2")
    alice.link_to(bob, LinkType.REFERENCE, "collaborates_with")
    
    print(f"\nGraph structure:")
    print(f"  {company.name}")
    for link in company.outgoing:
        print(f"    └── {link.name}: {link.target.name}")
        for sublink in link.target.outgoing:
            print(f"        └── {sublink.name}: {sublink.target.name}")
    
    # Traverse
    print(f"\nDFS from company:")
    for obj in world.traverse_dfs(company):
        print(f"  -> {obj.name}")


def example_events():
    """Event system"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Events")
    print("=" * 60)
    
    world = Graph()
    
    button = world.create("button")
    button.label = "Click Me"
    button.clicks = 0
    
    # Subscribe to events
    button.on("click", lambda: print("  Handler 1: Button clicked!"))
    button.on("click", lambda: print("  Handler 2: Updating count..."))
    button.on("click", lambda: setattr(button, 'clicks', button.clicks + 1), priority=10)
    
    print(f"\nButton: {button.name}, clicks={button.clicks}")
    
    print("\nSimulating clicks:")
    for i in range(3):
        print(f"\nClick {i+1}:")
        button.emit("click")
        print(f"  Total clicks: {button.clicks}")


def example_messaging():
    """Message passing"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Message Passing")
    print("=" * 60)
    
    world = Graph()
    
    # Create actors
    sender = world.create("sender")
    receiver = world.create("receiver")
    
    # Define receive handler
    def handle_receive(msg):
        print(f"  {receiver.name} received: {msg.name}({msg.args})")
        msg.payload = "OK"
        return msg
    
    receiver.define("ping").add_stage("respond", handle_receive)
    
    print(f"\nActors: {sender.name}, {receiver.name}")
    
    # Send messages
    print("\nSending messages:")
    sender.send(receiver, "ping", "Hello!")
    sender.send(receiver, "ping", "How are you?")
    
    # Process
    print("\nProcessing receiver's inbox:")
    results = receiver.process_inbox()
    print(f"  Results: {results}")


def example_pipeline():
    """Data pipeline"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Data Pipeline")
    print("=" * 60)
    
    world = Graph()
    factory = ObjectFactory(world)
    
    # Create pipeline nodes
    source = factory.pipeline_node("source")
    transform = factory.pipeline_node("transform")
    sink = factory.pipeline_node("sink")
    
    # Connect them
    source.link_to(transform, LinkType.PIPE)
    transform.link_to(sink, LinkType.PIPE)
    
    # Custom transform logic
    def double_stage(msg):
        data = msg.payload.get('data', 0)
        msg.payload['data'] = data * 2
        return msg
    
    transform.define("process").add_stage("double", double_stage)
    
    print(f"\nPipeline: {source.name} -> {transform.name} -> {sink.name}")
    
    # Process data
    print("\nProcessing data [1, 2, 3, 4, 5]:")
    for value in [1, 2, 3, 4, 5]:
        msg = Message(payload={'data': value})
        result = transform.pipelines['process'].execute(msg)  # Execute pipeline directly
        print(f"  {value} -> {result.payload['data']}")


def example_state_machine():
    """Finite state machine"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: State Machine")
    print("=" * 60)
    
    world = Graph()
    factory = ObjectFactory(world)
    
    # Create FSM
    traffic_light = factory.state_machine(
        "traffic_light",
        states=["red", "yellow", "green"],
        initial="red"
    )
    
    # Define transitions
    traffic_light.state.transitions = {
        ("red", "timer"): "green",
        ("green", "timer"): "yellow",
        ("yellow", "timer"): "red",
    }
    
    # Subscribe to state changes
    traffic_light.on("state_changed", 
                    lambda old, new: print(f"  Light changed: {old} -> {new}"))
    
    print(f"\nTraffic Light FSM")
    print(f"  States: {traffic_light.state.states}")
    print(f"  Initial: {traffic_light.state.current_state}")
    
    # Trigger transitions
    print("\nSimulating timer events:")
    for i in range(6):
        traffic_light.transition("timer")


def example_observable():
    """Observable/reactive pattern"""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Observable/Reactive")
    print("=" * 60)
    
    world = Graph()
    factory = ObjectFactory(world)
    
    # Create observable
    counter = factory.observable("counter", 0)
    
    # Subscribe to changes
    counter.on("changed", lambda old, new: print(f"  Value changed: {old} -> {new}"))
    
    print(f"\nObservable counter")
    print(f"  Initial: {counter.get()}")
    
    print("\nSetting values:")
    counter.set(1)
    counter.set(5)
    counter.set(5)  # No change, no event
    counter.set(10)


def demo():
    """Run all examples"""
    print("=" * 60)
    print("GRAPH OBJECT SYSTEM - EXAMPLES")
    print("=" * 60)
    print("""
    Everything is unified:
    
    ┌─────────────────────────────────────────────────────────┐
    │                    GRAPH OBJECT                         │
    │                                                         │
    │  ┌───────────────┐  ┌───────────────┐                  │
    │  │   NAMESPACE   │  │    LINKS      │                  │
    │  │   (state)     │  │   (graph)     │                  │
    │  │               │  │               │                  │
    │  │  .name        │  │  .outgoing[]  │                  │
    │  │  .age         │  │  .incoming[]  │                  │
    │  │  .data        │  │  .children()  │                  │
    │  └───────────────┘  └───────────────┘                  │
    │                                                         │
    │  ┌───────────────┐  ┌───────────────┐                  │
    │  │   PIPELINES   │  │    EVENTS     │                  │
    │  │   (methods)   │  │  (callbacks)  │                  │
    │  │               │  │               │                  │
    │  │  .greet()     │  │  .on("click") │                  │
    │  │  .process()   │  │  .emit("x")   │                  │
    │  │  .update()    │  │               │                  │
    │  └───────────────┘  └───────────────┘                  │
    │                                                         │
    │  ┌───────────────┐  ┌───────────────┐                  │
    │  │   MESSAGES    │  │   PROTOTYPE   │                  │
    │  │   (inbox)     │  │ (inheritance) │                  │
    │  │               │  │               │                  │
    │  │  .send()      │  │  .create_     │                  │
    │  │  .receive()   │  │    child()    │                  │
    │  │               │  │               │                  │
    │  └───────────────┘  └───────────────┘                  │
    │                                                         │
    └─────────────────────────────────────────────────────────┘
    """)
    
    example_basic()
    example_inheritance()
    example_graph_links()
    example_events()
    example_messaging()
    example_pipeline()
    example_state_machine()
    example_observable()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    USE THIS FOR:
    
    1. Game Entities
       - Objects with state, behaviors, relationships
       - Component systems
       - Event handling
    
    2. UI Systems
       - Components with props
       - Event bubbling
       - State management
    
    3. Data Processing
       - Pipeline nodes
       - Transform chains
       - Message passing
    
    4. State Machines
       - FSM pattern built-in
       - Transition events
    
    5. Reactive Systems
       - Observable values
       - Change notifications
    
    6. Actor Systems
       - Async message passing
       - Inbox processing
    
    7. Scene Graphs
       - Hierarchical objects
       - Traversal built-in
    
    8. Entity-Component Systems
       - Compose behaviors
       - Share via prototypes
    """)


if __name__ == "__main__":
    demo()

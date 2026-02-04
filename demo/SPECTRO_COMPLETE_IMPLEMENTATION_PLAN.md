# SPECTRO Complete Implementation Plan
## Everything Not Yet Built

### The Core Model

```
INPUT (real or simulated)
    │
    ▼
SPACE (the composition itself)
    │
    ▼
TIMELINE (where context bounces)
    │
    ▼
OUTPUT (audio + visual)
```

**Key insight:** The app IS a simulation. Real MIDI is just one input source. The space contains the composition. Events fire context when they hit the timeline.

---

## PART 1: What EXISTS vs What NEEDS BUILDING

### ✅ EXISTS (Ready to Use)

| Component | File | Status |
|-----------|------|--------|
| Transport | `engine/time/transport.py` | Complete |
| TimeCamera | `engine/time/camera.py` | Complete |
| SignalBridge | `engine/core/signal.py` | Complete |
| Scene/Entity | `engine/core/scene.py` | Complete |
| MidiManager | `engine/midi/manager.py` | Complete |
| LaunchpadController | `engine/midi/launchpad.py` | Complete |
| MidiRecorder | `engine/midi/recorder.py` | Complete |
| Sampler | `engine/audio/sampler.py` | Complete |
| AudioEngine | `engine/audio/engine.py` | Complete |
| EventDispatcher | `buffers_v2.py` | Complete |
| MidiRingBuffer | `buffers_v2.py` | Complete |
| AudioRingBuffer | `buffers_v2.py` | Complete |
| CommandList | `engine/render/commands.py` | Complete |
| Renderer | `engine/render/renderer.py` | Complete |
| RenderTargetPool | `engine/render/targets.py` | Complete |
| ViewportArea | `engine/viewport/viewport.py` | Complete |
| Graph/Snapshot | `engine/core/snapshot.py` | Complete |
| Style/Theme | `engine/graph/style.py` | Complete |
| Widget Base | `engine/ui/widget.py` | Partial |

### ❌ NEEDS BUILDING

| Component | Purpose | Priority |
|-----------|---------|----------|
| **SpectroScene** | Unified space with observer pattern | P0 |
| **InputSimulator** | Generate events without hardware | P0 |
| **Track** | Container for events on one lane | P0 |
| **Panel Base** | Abstract panel with projection | P1 |
| **SequencerPanel** | 2D X-Y grid view | P1 |
| **WaveformPanel** | 2D X-Z amplitude view | P1 |
| **Viewport3DPanel** | 3D perspective view | P1 |
| **DrawContext** | 2D rendering implementation | P1 |
| **ContextFirer** | Fires context when playhead crosses events | P0 |
| **HarmonicAnalyzer** | Chroma/key detection (future) | P2 |
| **BeatSlicer** | Audio → slices (future) | P2 |
| **PitchShifter** | Transpose audio (future) | P2 |

---

## PART 2: The Simulation Model

### 2.1 Everything is Input

```python
class InputSource(Protocol):
    """Any source of events - real or simulated."""
    
    def poll(self) -> List[MidiEvent]:
        """Get pending events from this source."""
        ...
    
    def is_realtime(self) -> bool:
        """True if events arrive unpredictably (hardware)."""
        ...
```

### 2.2 Input Sources

```python
# === REAL HARDWARE ===
class MidiInputSource(InputSource):
    """Wraps MidiManager for real MIDI input."""
    def __init__(self, midi_manager: MidiManager):
        self.midi = midi_manager
        self._pending: List[MidiEvent] = []
        # MidiManager routes through SignalBridge
        # We collect events for polling
    
    def is_realtime(self) -> bool:
        return True

# === KEYBOARD (real but simple) ===
class KeyboardInputSource(InputSource):
    """Maps keyboard keys to MIDI events."""
    KEY_MAP = {
        '1': (0, 100),  # note 0, velocity 100
        '2': (1, 100),
        '3': (2, 100),
        '4': (3, 100),
        'q': (4, 80),
        'w': (5, 80),
        'e': (6, 80),
        'r': (7, 80),
    }
    
    def on_key(self, key: str, pressed: bool):
        if pressed and key in self.KEY_MAP:
            note, vel = self.KEY_MAP[key]
            self._pending.append(MidiEvent(
                event_type=MidiEventType.NOTE_ON,
                channel=0,
                data1=note,
                data2=vel,
                timestamp_samples=self._current_sample,
                timestamp_beats=self._current_beat,
            ))
    
    def is_realtime(self) -> bool:
        return True  # User presses keys in real time

# === SIMULATED (patterns, sequences, algorithms) ===
class PatternInputSource(InputSource):
    """
    Generates events from a pattern.
    
    This is the key: you can define a composition as a PatternInputSource
    and "play" it through the same pipeline as real MIDI.
    """
    def __init__(self, pattern: List[Tuple[float, int, int]]):
        # pattern = [(beat, note, velocity), ...]
        self.pattern = sorted(pattern, key=lambda x: x[0])
        self._index = 0
        self._loop_length = 16.0  # beats
    
    def poll(self, current_beat: float) -> List[MidiEvent]:
        """Return events that should fire at current beat."""
        events = []
        
        # Handle looping
        pattern_beat = current_beat % self._loop_length
        
        while self._index < len(self.pattern):
            beat, note, vel = self.pattern[self._index]
            if beat <= pattern_beat:
                events.append(MidiEvent(
                    event_type=MidiEventType.NOTE_ON,
                    channel=0,
                    data1=note,
                    data2=vel,
                    timestamp_beats=current_beat,
                    flags=MidiEvent.FLAG_GENERATED,  # Mark as simulated
                ))
                self._index += 1
            else:
                break
        
        # Reset on loop
        if pattern_beat < 0.1 and self._index >= len(self.pattern):
            self._index = 0
        
        return events
    
    def is_realtime(self) -> bool:
        return False  # Events are pre-determined

# === ALGORITHMIC (generative) ===
class GenerativeInputSource(InputSource):
    """
    Generates events algorithmically.
    
    Could implement:
    - Euclidean rhythms
    - Markov chains
    - L-systems
    - Neural network output
    """
    def __init__(self, algorithm: Callable[[float], List[MidiEvent]]):
        self.algorithm = algorithm
    
    def poll(self, current_beat: float) -> List[MidiEvent]:
        return self.algorithm(current_beat)
    
    def is_realtime(self) -> bool:
        return False
```

### 2.3 InputManager Collects All Sources

```python
class InputManager:
    """
    Collects events from all input sources.
    
    The app doesn't care WHERE events come from.
    Real MIDI, keyboard, patterns, algorithms - all the same.
    """
    def __init__(self):
        self.sources: List[InputSource] = []
        self._pending: List[MidiEvent] = []
    
    def add_source(self, source: InputSource):
        self.sources.append(source)
    
    def poll_all(self, current_beat: float) -> List[MidiEvent]:
        """Gather events from all sources."""
        events = []
        for source in self.sources:
            events.extend(source.poll(current_beat))
        return events
    
    def feed_to_buffer(self, buffer: MidiRingBuffer, current_beat: float):
        """Poll all sources and write to ring buffer."""
        for event in self.poll_all(current_beat):
            buffer.write(event)
```

---

## PART 3: The Space (Composition Model)

### 3.1 Track = Event Container

```python
@dataclass
class Track:
    """
    A track holds events for one instrument/lane.
    
    Tracks observe the timeline and fire events when crossed.
    Tracks can also BE input sources (play back recorded events).
    """
    id: str
    name: str
    index: int  # Lane position
    color: tuple
    sample_name: str  # What sound to trigger
    events: List[Entity] = field(default_factory=list)
    
    # Track state
    muted: bool = False
    solo: bool = False
    volume: float = 1.0
    pan: float = 0.0  # -1 left, +1 right
    
    # FUTURE: Harmonic info
    # key_center: str = None  # If track has fixed key
    # transpose: int = 0      # Semitones offset
    
    def add_event(self, beat: float, velocity: int, duration: float = 0.25) -> Entity:
        """Add event to this track."""
        entity = Entity(
            entity_type=EntityType.MIDI_EVENT,
            position=Vec3(beat, self.index, velocity / 127.0),
            extent=Vec3(duration, 1.0, 1.0),
            color=self.color,
            metadata={
                'track_id': self.id,
                'sample_name': self.sample_name,
                'velocity': velocity,
            }
        )
        self.events.append(entity)
        return entity
    
    def events_in_range(self, start: float, end: float) -> List[Entity]:
        """Get events overlapping beat range."""
        return [e for e in self.events 
                if e.position.x < end and e.position.x + e.extent.x > start]
    
    def as_input_source(self) -> InputSource:
        """
        Return this track as an input source.
        
        This is how playback works: the track becomes a source
        that emits its events at the right times.
        """
        return TrackPlaybackSource(self)


class TrackPlaybackSource(InputSource):
    """Plays back events from a track."""
    
    def __init__(self, track: Track):
        self.track = track
        self._fired: Set[str] = set()  # Event IDs already fired
    
    def poll(self, current_beat: float) -> List[MidiEvent]:
        if self.track.muted:
            return []
        
        events = []
        for entity in self.track.events_in_range(current_beat - 0.05, current_beat + 0.05):
            if entity.id not in self._fired and entity.position.x <= current_beat:
                self._fired.add(entity.id)
                events.append(MidiEvent(
                    event_type=MidiEventType.NOTE_ON,
                    channel=0,
                    data1=self.track.index,  # Note = lane
                    data2=entity.metadata['velocity'],
                    timestamp_beats=entity.position.x,
                    device_id=-1,  # Internal playback
                ))
        return events
    
    def reset(self):
        """Reset for loop restart."""
        self._fired.clear()
    
    def is_realtime(self) -> bool:
        return False
```

### 3.2 SpectroScene = The Composition Space

```python
class SpectroScene:
    """
    The unified composition space.
    
    Contains:
    - Tracks (collections of events)
    - Transport (playback state)
    - TimeCamera (view transformation)
    - Observer pattern (panels watch this)
    
    The scene IS the composition. Saving the scene saves the song.
    """
    
    def __init__(self, bridge: SignalBridge = None):
        self.bridge = bridge or SignalBridge()
        
        # Core systems
        self.transport = Transport(bpm=120.0)
        self.time_camera = TimeCamera()
        self.time_camera.bind(self.bridge)
        
        # Tracks (ordered by index)
        self.tracks: List[Track] = []
        self._setup_default_tracks()
        
        # All entities (flat list for rendering)
        self._entities: Dict[str, Entity] = {}
        
        # Observers (panels register here)
        self._observers: List[Callable[[Entity, str], None]] = []
        
        # Input management
        self.input_manager = InputManager()
        
        # Wire transport callbacks
        self.transport.on_loop_callbacks.append(self._on_loop)
    
    def _setup_default_tracks(self):
        """Create default drum tracks."""
        defaults = [
            ("kick", "Kick", (1.0, 0.3, 0.2, 1.0)),
            ("snare", "Snare", (1.0, 0.6, 0.2, 1.0)),
            ("hihat", "HiHat", (1.0, 0.9, 0.2, 1.0)),
            ("clap", "Clap", (0.4, 1.0, 0.3, 1.0)),
        ]
        for i, (sample, name, color) in enumerate(defaults):
            self.tracks.append(Track(
                id=f"track_{i}",
                name=name,
                index=i,
                color=color,
                sample_name=sample,
            ))
    
    # === OBSERVER PATTERN ===
    
    def observe(self, callback: Callable[[Entity, str], None]):
        """Register observer for entity changes."""
        self._observers.append(callback)
    
    def _notify(self, entity: Entity, action: str):
        """Notify all observers."""
        for callback in self._observers:
            callback(entity, action)
    
    # === ENTITY MANAGEMENT ===
    
    def add_entity(self, entity: Entity) -> Entity:
        """Add entity and notify observers."""
        self._entities[entity.id] = entity
        self._notify(entity, "added")
        return entity
    
    def remove_entity(self, entity_id: str):
        """Remove entity and notify observers."""
        entity = self._entities.pop(entity_id, None)
        if entity:
            self._notify(entity, "removed")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)
    
    def all_entities(self) -> Iterator[Entity]:
        return iter(self._entities.values())
    
    def entities_by_type(self, etype: EntityType) -> Iterator[Entity]:
        return (e for e in self._entities.values() if e.entity_type == etype)
    
    # === TRACK OPERATIONS ===
    
    def get_track(self, index: int) -> Optional[Track]:
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return None
    
    def add_note(self, track_index: int, beat: float, velocity: int = 100,
                 duration: float = 0.25) -> Optional[Entity]:
        """
        Add a note to a track.
        
        This is the main entry point for creating events.
        """
        track = self.get_track(track_index)
        if not track:
            return None
        
        entity = track.add_event(beat, velocity, duration)
        return self.add_entity(entity)
    
    # === PLAYBACK ===
    
    def setup_playback_sources(self):
        """Add all tracks as input sources for playback."""
        for track in self.tracks:
            source = track.as_input_source()
            self.input_manager.add_source(source)
    
    def _on_loop(self, state):
        """Reset playback sources on loop."""
        for source in self.input_manager.sources:
            if hasattr(source, 'reset'):
                source.reset()
    
    # === SERIALIZATION ===
    
    def to_dict(self) -> dict:
        """Serialize scene (save project)."""
        return {
            'bpm': self.transport.bpm,
            'tracks': [
                {
                    'id': t.id,
                    'name': t.name,
                    'sample_name': t.sample_name,
                    'color': t.color,
                    'events': [
                        {
                            'beat': e.position.x,
                            'velocity': e.metadata.get('velocity', 100),
                            'duration': e.extent.x,
                        }
                        for e in t.events
                    ]
                }
                for t in self.tracks
            ]
        }
    
    def from_dict(self, data: dict):
        """Deserialize scene (load project)."""
        self.transport.set_bpm(data.get('bpm', 120.0))
        # ... restore tracks and events
```

---

## PART 4: Context Firing (The Timeline Bounce)

### 4.1 ContextFirer = The Dispatcher Wrapper

```python
class ContextFirer:
    """
    Fires context when events cross the timeline.
    
    This wraps EventDispatcher and integrates with SpectroScene.
    
    Flow:
        1. Poll all input sources (real + simulated)
        2. Write events to MidiRingBuffer
        3. Process dispatcher (scans buffer at playhead)
        4. Dispatcher assembles ExecutionContext (LATE BINDING)
        5. Callbacks fire with fresh context
        6. Entities created in scene
        7. Audio triggered
    """
    
    def __init__(self, scene: SpectroScene, audio: AudioEngine):
        self.scene = scene
        self.audio = audio
        
        # Buffers
        self.midi_buffer = MidiRingBuffer(capacity=4096)
        self.audio_buffer = AudioRingBuffer(capacity_samples=65536)
        
        # Dispatcher
        self.dispatcher = EventDispatcher(self.midi_buffer, self.audio_buffer)
        self.dispatcher.set_bpm(scene.transport.bpm)
        
        # Register the main callback
        self.dispatcher.register(
            callback=self._on_event,
            event_types={MidiEventType.NOTE_ON},
            name="context_firer"
        )
        
        # Beat callback
        self.dispatcher.on_beat(self._on_beat)
    
    def _on_event(self, ctx: ExecutionContext):
        """
        THE BOUNCE: Event hits timeline, context fires.
        
        This is where:
        1. We receive FRESH context (late-bound)
        2. Create entity in scene (notifies all panels)
        3. Trigger audio (immediate feedback)
        """
        event = ctx.event
        
        # Determine track from note
        track_index = event.note % len(self.scene.tracks)
        track = self.scene.get_track(track_index)
        
        if not track:
            return
        
        # === THE BOUNCE ===
        # ctx.transport.beat is CURRENT beat (late-bound)
        # This is where we place the entity
        
        entity = self.scene.add_note(
            track_index=track_index,
            beat=ctx.transport.beat,
            velocity=event.velocity,
        )
        
        if entity:
            print(f"[BOUNCE] {track.name} @ beat {ctx.transport.beat:.2f} "
                  f"(timing: {ctx.timing_error_ms:+.1f}ms)")
        
        # Trigger audio
        if not track.muted:
            velocity = event.velocity / 127.0 * track.volume
            self.audio.trigger(track.sample_name, velocity)
    
    def _on_beat(self, beat: int, transport):
        """Beat boundary callback."""
        # Could add beat-aligned effects here
        pass
    
    def process(self, dt: float):
        """
        Main update - call every frame.
        
        1. Sync with transport
        2. Poll input sources
        3. Process dispatcher
        """
        # Sync BPM
        if self.dispatcher._bpm != self.scene.transport.bpm:
            self.dispatcher.set_bpm(self.scene.transport.bpm)
        
        # Sync play state
        if self.scene.transport.playing and not self.dispatcher.playing:
            self.dispatcher.play()
        elif not self.scene.transport.playing and self.dispatcher.playing:
            self.dispatcher.pause()
        
        # Poll input sources → buffer
        self.scene.input_manager.feed_to_buffer(
            self.midi_buffer, 
            self.scene.transport.playhead_beat
        )
        
        # Process (fires callbacks via late binding)
        self.dispatcher.process_frame(dt)
```

---

## PART 5: Panel System (Views into Space)

### 5.1 Panel Base

```python
class Panel(ABC):
    """
    Abstract base for panels.
    
    A panel is a view into the scene.
    It observes entity changes and renders its projection.
    """
    
    def __init__(self, scene: SpectroScene, rect: Rect):
        self.scene = scene
        self.rect = rect
        self.visible = True
        
        # Register as observer
        scene.observe(self.on_entity_change)
    
    @abstractmethod
    def on_entity_change(self, entity: Entity, action: str):
        """Handle entity added/removed/changed."""
        pass
    
    @abstractmethod
    def project(self, position: Vec3) -> Tuple[float, float]:
        """Project 3D position to 2D panel coords."""
        pass
    
    @abstractmethod
    def render(self, ctx: DrawContext):
        """Render panel contents."""
        pass
    
    def update(self, dt: float):
        """Update panel state (override if needed)."""
        pass
    
    def contains(self, x: float, y: float) -> bool:
        """Check if point is inside panel."""
        return (self.rect.x <= x < self.rect.x + self.rect.w and
                self.rect.y <= y < self.rect.y + self.rect.h)
```

### 5.2 DrawContext (2D Rendering)

```python
class DrawContext:
    """
    2D drawing context.
    
    Wraps ModernGL for basic 2D primitives.
    Uses batched rendering for efficiency.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._setup_shaders()
        self._offset_stack: List[Tuple[float, float]] = [(0, 0)]
        
        # Batch buffers
        self._rect_batch: List[tuple] = []
        self._line_batch: List[tuple] = []
    
    def _setup_shaders(self):
        """Create shader programs for 2D drawing."""
        # Rect shader
        self._rect_prog = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_pos;
            uniform vec2 u_offset;
            uniform vec2 u_size;
            uniform vec2 u_window;
            void main() {
                vec2 px = u_offset + in_pos * u_size;
                vec2 ndc = (px / u_window) * 2.0 - 1.0;
                ndc.y = -ndc.y;
                gl_Position = vec4(ndc, 0.0, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            uniform vec4 u_color;
            out vec4 fragColor;
            void main() {
                fragColor = u_color;
            }
            """
        )
        
        # Unit quad
        quad = np.array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1], dtype='f4')
        self._quad_vbo = self.ctx.buffer(quad)
        self._quad_vao = self.ctx.vertex_array(
            self._rect_prog, [(self._quad_vbo, '2f', 'in_pos')]
        )
    
    def push_offset(self, x: float, y: float):
        """Push translation offset onto stack."""
        ox, oy = self._offset_stack[-1]
        self._offset_stack.append((ox + x, oy + y))
    
    def pop_offset(self):
        """Pop translation offset from stack."""
        if len(self._offset_stack) > 1:
            self._offset_stack.pop()
    
    @property
    def offset(self) -> Tuple[float, float]:
        return self._offset_stack[-1]
    
    def draw_rect(self, x: float, y: float, w: float, h: float, 
                  color: tuple, window_size: tuple):
        """Draw a filled rectangle."""
        ox, oy = self.offset
        self._rect_prog['u_offset'].value = (x + ox, y + oy)
        self._rect_prog['u_size'].value = (w, h)
        self._rect_prog['u_color'].value = color
        self._rect_prog['u_window'].value = window_size
        self._quad_vao.render()
    
    def draw_line(self, x1: float, y1: float, x2: float, y2: float,
                  color: tuple, width: float = 1.0):
        """Draw a line (as thin rect for now)."""
        ox, oy = self.offset
        # Simplified: draw as 1px wide rect
        dx, dy = x2 - x1, y2 - y1
        length = (dx*dx + dy*dy) ** 0.5
        # ... actual implementation would rotate rect
        pass
    
    def draw_text(self, text: str, x: float, y: float, color: tuple,
                  font_size: float = 14.0):
        """Draw text (placeholder - needs font atlas)."""
        # For MVP: skip or use simple bitmap font
        pass
```

### 5.3 Concrete Panels

```python
# === SEQUENCER PANEL ===
class SequencerPanel(Panel):
    """2D grid view: X = beats, Y = tracks/lanes."""
    
    LANE_HEIGHT = 40
    
    def project(self, position: Vec3) -> Tuple[float, float]:
        px = self.scene.time_camera.beat_to_px(position.x)
        py = position.y * self.LANE_HEIGHT
        return (px, py)
    
    def on_entity_change(self, entity: Entity, action: str):
        # Could cache visible entities for efficiency
        pass
    
    def render(self, ctx: DrawContext):
        w, h = self.rect.w, self.rect.h
        window = (w, h)
        
        ctx.push_offset(self.rect.x, self.rect.y)
        
        # Background
        ctx.draw_rect(0, 0, w, h, (0.06, 0.07, 0.08, 1.0), window)
        
        # Lane backgrounds
        for i, track in enumerate(self.scene.tracks):
            y = i * self.LANE_HEIGHT
            bg = (*track.color[:3], 0.1)  # Tinted background
            ctx.draw_rect(0, y, w, self.LANE_HEIGHT, bg, window)
        
        # Beat grid
        for beat in self.scene.time_camera.iter_beat_positions():
            px = self.scene.time_camera.beat_to_px(beat)
            is_bar = int(beat) % 4 == 0
            color = (0.3, 0.3, 0.35, 1.0) if is_bar else (0.15, 0.15, 0.18, 1.0)
            ctx.draw_rect(px, 0, 2 if is_bar else 1, h, color, window)
        
        # Events
        for entity in self.scene.entities_by_type(EntityType.MIDI_EVENT):
            px, py = self.project(entity.position)
            ew = entity.extent.x * self.scene.time_camera._px_per_beat
            eh = self.LANE_HEIGHT - 4
            ctx.draw_rect(px, py + 2, max(4, ew), eh, entity.color, window)
        
        # Playhead
        playhead_px = self.scene.time_camera.beat_to_px(
            self.scene.transport.playhead_beat
        )
        ctx.draw_rect(playhead_px - 1, 0, 3, h, (1.0, 0.4, 0.2, 0.9), window)
        
        ctx.pop_offset()


# === WAVEFORM PANEL ===
class WaveformPanel(Panel):
    """2D amplitude view: X = beats, Y = amplitude."""
    
    def __init__(self, scene: SpectroScene, rect: Rect, audio: AudioEngine):
        super().__init__(scene, rect)
        self.audio = audio
    
    def project(self, position: Vec3) -> Tuple[float, float]:
        px = self.scene.time_camera.beat_to_px(position.x)
        py = self.rect.h * (1.0 - position.z)  # Z = intensity → Y
        return (px, py)
    
    def on_entity_change(self, entity: Entity, action: str):
        pass
    
    def render(self, ctx: DrawContext):
        w, h = self.rect.w, self.rect.h
        window = (w, h)
        
        ctx.push_offset(self.rect.x, self.rect.y)
        
        # Background
        ctx.draw_rect(0, 0, w, h, (0.05, 0.06, 0.07, 1.0), window)
        
        # Waveform
        samples = self.audio.get_output_waveform(num_samples=int(w))
        for i, amp in enumerate(samples):
            bar_h = amp * h * 0.8
            y = (h - bar_h) / 2
            ctx.draw_rect(i, y, 1, bar_h, (0.3, 0.6, 0.9, 0.8), window)
        
        # Event markers
        for entity in self.scene.entities_by_type(EntityType.MIDI_EVENT):
            px, _ = self.project(entity.position)
            if 0 <= px <= w:
                ctx.draw_rect(px, 0, 2, 8, entity.color, window)
        
        # Playhead
        playhead_px = self.scene.time_camera.beat_to_px(
            self.scene.transport.playhead_beat
        )
        ctx.draw_rect(playhead_px - 1, 0, 3, h, (1.0, 0.4, 0.2, 0.9), window)
        
        ctx.pop_offset()


# === 3D VIEWPORT PANEL ===
class Viewport3DPanel(Panel):
    """3D perspective view using existing render pipeline."""
    
    def __init__(self, scene: SpectroScene, rect: Rect, 
                 ctx: moderngl.Context, pool: RenderTargetPool, 
                 registry: ResourceRegistry):
        super().__init__(scene, rect)
        
        # 3D infrastructure
        self.graph = EntityNode("root")
        self.viewport = ViewportArea("3d", self.graph, cameras=[
            Camera(
                name="main",
                eye=np.array([8.0, 4.0, 8.0]),
                target=np.array([4.0, 2.0, 0.0]),
            )
        ])
        self.pool = pool
        self.registry = registry
        self.renderer = Renderer(ctx, registry)
        
        # Map scene entities to 3D nodes
        self._entity_nodes: Dict[str, EntityNode] = {}
    
    def on_entity_change(self, entity: Entity, action: str):
        """Sync scene entity to 3D graph."""
        if entity.entity_type != EntityType.MIDI_EVENT:
            return
        
        if action == "added":
            node = EntityNode(f"note_{entity.id}")
            node.transform = Transform(
                pos=np.array([
                    entity.position.x,      # beat → X
                    entity.position.y,      # lane → Y
                    entity.position.z * 2,  # velocity → Z
                ], dtype=np.float32),
                scale=np.array([0.2, 0.8, 0.3], dtype=np.float32),
            )
            node.mesh = MeshRenderer(
                mesh_id="cube",
                pipeline_id="lit_color",
                color=np.array(entity.color, dtype=np.float32),
            )
            self.graph.add_child(node)
            self._entity_nodes[entity.id] = node
        
        elif action == "removed":
            node = self._entity_nodes.pop(entity.id, None)
            if node:
                self.graph.remove_child(node)
    
    def project(self, position: Vec3) -> Tuple[float, float]:
        # 3D projection handled by viewport
        return (0, 0)
    
    def render(self, ctx: DrawContext):
        # Render 3D to texture
        self.viewport.ensure_surface(
            self.rect.w, self.rect.h, self.pool, want_picking=False
        )
        self.viewport.render_if_ready(self.renderer)
        
        # Blit texture to panel rect
        # (handled by compositor or manual blit)
```

---

## PART 6: Main Application

```python
class SpectroApp(mglw.WindowConfig):
    """
    The complete SPECTRO demo application.
    """
    
    gl_version = (3, 3)
    title = "SPECTRO - Space/Composition/Timeline"
    window_size = (1280, 720)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # === THE SPACE ===
        self.scene = SpectroScene()
        
        # === AUDIO ===
        self.audio = AudioEngine(self.scene.transport)
        self.audio.load_default_sounds()
        self.audio.start()
        
        # === CONTEXT FIRER ===
        self.firer = ContextFirer(self.scene, self.audio)
        
        # === INPUT SOURCES ===
        # Keyboard (real-time)
        self.keyboard = KeyboardInputSource()
        self.scene.input_manager.add_source(self.keyboard)
        
        # Demo pattern (simulated)
        demo_pattern = [
            (0.0, 0, 100),   # Kick on 1
            (1.0, 2, 80),    # HiHat on 2
            (2.0, 1, 100),   # Snare on 3
            (3.0, 2, 80),    # HiHat on 4
        ]
        self.scene.input_manager.add_source(PatternInputSource(demo_pattern))
        
        # Track playback sources (for recorded events)
        self.scene.setup_playback_sources()
        
        # === RENDERING ===
        self.draw_ctx = DrawContext(self.ctx)
        self.pool = RenderTargetPool(self.ctx)
        self.registry = ResourceRegistry(self.ctx)
        self.registry.bootstrap_defaults()
        
        # === PANELS ===
        self._setup_panels()
        
        print("\n=== SPECTRO ===")
        print("Keys 1-4: Trigger drums")
        print("SPACE: Play/Pause")
        print("R: Reset")
        print("===============\n")
    
    def _setup_panels(self):
        """Create the three-panel layout."""
        w, h = self.window_size
        
        transport_h = 50
        waveform_h = 120
        main_h = h - transport_h - waveform_h
        
        self.panels = [
            SequencerPanel(
                self.scene, 
                Rect(0, transport_h, w // 2, main_h)
            ),
            Viewport3DPanel(
                self.scene,
                Rect(w // 2, transport_h, w // 2, main_h),
                self.ctx, self.pool, self.registry
            ),
            WaveformPanel(
                self.scene,
                Rect(0, transport_h + main_h, w, waveform_h),
                self.audio
            ),
        ]
    
    def on_render(self, t: float, frame_time: float):
        """Main render loop."""
        dt = frame_time
        
        # Update transport
        self.scene.transport.update(dt)
        
        # Update time camera
        self.scene.time_camera.update(dt, self.scene.transport.playhead_beat)
        
        # Process context firer (polls inputs, fires callbacks)
        self.firer.process(dt)
        
        # Clear screen
        self.ctx.clear(0.08, 0.09, 0.11)
        
        # Render panels
        for panel in self.panels:
            panel.update(dt)
            panel.render(self.draw_ctx)
        
        # Draw transport bar
        self._draw_transport_bar()
    
    def _draw_transport_bar(self):
        """Draw transport controls at top."""
        w, h = self.wnd.size
        window = (w, h)
        
        # Background
        self.draw_ctx.draw_rect(0, 0, w, 50, (0.12, 0.12, 0.14, 1.0), window)
        
        # Play indicator
        playing = self.scene.transport.playing
        color = (0.3, 0.8, 0.3, 1.0) if playing else (0.8, 0.3, 0.3, 1.0)
        self.draw_ctx.draw_rect(20, 10, 30, 30, color, window)
        
        # Beat boxes
        beat_in_bar = int(self.scene.transport.playhead_beat) % 4
        for i in range(4):
            bc = (0.4, 0.6, 0.9, 1.0) if i == beat_in_bar else (0.2, 0.25, 0.3, 1.0)
            self.draw_ctx.draw_rect(70 + i * 35, 10, 30, 30, bc, window)
    
    def key_event(self, key, action, mods):
        """Handle keyboard input."""
        if action != self.wnd.keys.ACTION_PRESS:
            return
        
        # Drum triggers
        key_map = {
            self.wnd.keys.NUMBER_1: '1',
            self.wnd.keys.NUMBER_2: '2',
            self.wnd.keys.NUMBER_3: '3',
            self.wnd.keys.NUMBER_4: '4',
        }
        if key in key_map:
            self.keyboard.on_key(key_map[key], True)
        
        # Transport
        elif key == self.wnd.keys.SPACE:
            self.scene.transport.toggle()
        elif key == self.wnd.keys.R:
            self.scene.transport.stop()


if __name__ == "__main__":
    mglw.run_window_config(SpectroApp)
```

---

## PART 7: Implementation Order

```
PHASE 1: Core (Day 1)
├── [ ] InputSource protocol + KeyboardInputSource
├── [ ] PatternInputSource (simulated input)
├── [ ] Track class with events
├── [ ] SpectroScene with observer pattern
└── [ ] ContextFirer wrapping EventDispatcher

PHASE 2: Rendering (Day 1-2)
├── [ ] DrawContext with rect rendering
├── [ ] Panel base class
├── [ ] SequencerPanel (2D grid)
└── [ ] WaveformPanel (2D amplitude)

PHASE 3: Integration (Day 2)
├── [ ] Main app with layout
├── [ ] Input → Scene → Panels flow
├── [ ] Transport bar UI
└── [ ] Keyboard controls

PHASE 4: 3D View (Day 2-3)
├── [ ] Viewport3DPanel using existing pipeline
├── [ ] Entity → 3D node sync
└── [ ] Camera controls

PHASE 5: Polish (Day 3)
├── [ ] Track mute/solo
├── [ ] Loop region
├── [ ] Save/load project
└── [ ] MIDI device connection
```

---

## The Key Insights

1. **Input is simulation** - Real MIDI is just one source. Patterns, algorithms, playback are all input sources.

2. **Space is composition** - The scene holds everything. Tracks are just organization. Entities are the atoms.

3. **Timeline bounces context** - Events hit the timeline, context is assembled FRESH, callbacks fire with current state.

4. **Panels are projections** - Same space, different views. Add once, see everywhere.

5. **Everything flows through the firer** - Input → Buffer → Dispatcher → Context → Callback → Entity → Audio.

# SPECTRO Playable Demo TODO
## 3 Panels, 1 Scene, Entities Bounce Off Timeline

### The Core Idea

**One Scene. Multiple Views. Context bounces off the timeline.**

```
SCENE (unified 3D space)
├── X = beats (time)
├── Y = pitch/frequency/lane  
├── Z = velocity/intensity/depth
│
├── Contains ALL entities:
│   ├── SequencerEvents (notes, audio slices)
│   ├── Playhead (a moving entity!)
│   ├── Beat markers
│   ├── Waveform points
│   └── 3D objects (cubes, whatever)
│
└── Observed by 3 PANELS (different projections):
    ├── Panel 1: Sequencer Grid (X-Y, top-down, 2D)
    ├── Panel 2: Waveform Timeline (X-Z, side view, 2D)
    └── Panel 3: 3D Viewport (perspective, full 3D)
```

**Add entity to scene → it appears in all panels automatically.**

---

## Phase 1: Unified Scene with Entity System

### 1.1 Extend Scene to be the single source of truth

```python
# In engine/core/scene.py or new file

class SpectroScene(Scene):
    """
    The unified space. All panels observe this.
    """
    def __init__(self, bridge: SignalBridge):
        super().__init__(bridge)
        self.transport = Transport(bpm=120.0)
        self.time_camera = TimeCamera()
        
        # Entity observers - panels register here
        self._observers: List[Callable[[Entity, str], None]] = []
    
    def add(self, entity: Entity) -> Entity:
        """Add entity and notify all observers."""
        result = super().add(entity)
        for observer in self._observers:
            observer(entity, "added")
        return result
    
    def observe(self, callback: Callable[[Entity, str], None]):
        """Register a panel to observe scene changes."""
        self._observers.append(callback)
    
    # Convenience: add sequencer event as entity
    def add_note(self, beat: float, lane: int, velocity: int = 100) -> Entity:
        """Add a note/event to the scene."""
        entity = Entity(
            entity_type=EntityType.MIDI_EVENT,
            position=Vec3(beat, lane, velocity / 127.0),
            extent=Vec3(0.25, 1.0, 1.0),  # duration, lane height, intensity
            color=LANE_COLORS[lane % len(LANE_COLORS)],
        )
        return self.add(entity)
```

### 1.2 Entity created from MIDI bounces through scene

```python
def on_midi_input(ctx: ExecutionContext):
    """MIDI → Scene Entity → All Panels See It"""
    
    # Get FRESH position from late-bound context
    beat = ctx.transport.beat
    lane = ctx.event.note % 4
    velocity = ctx.event.velocity
    
    # Add to SCENE (not directly to sequencer)
    entity = scene.add_note(beat, lane, velocity)
    
    # Scene notifies all observers (panels)
    # Each panel decides how to render this entity
    # - Sequencer panel: draws rectangle at (beat, lane)
    # - Waveform panel: maybe ignores it, or shows marker
    # - 3D panel: spawns a cube at (beat, lane, velocity)
    
    # Trigger audio
    audio.trigger(LANE_SAMPLES[lane], velocity / 127.0)
```

---

## Phase 2: Three Panel Types

### 2.1 Base Panel Class

```python
class Panel:
    """
    Base class for scene observers.
    
    Each panel:
    - Observes the scene for entity changes
    - Has its own projection (how to map 3D → panel coords)
    - Renders its view of the entities
    """
    def __init__(self, scene: SpectroScene, rect: Rect):
        self.scene = scene
        self.rect = rect
        self.entities_in_view: List[Entity] = []
        
        # Register as observer
        scene.observe(self.on_entity_change)
    
    def on_entity_change(self, entity: Entity, action: str):
        """Called when scene changes. Override in subclass."""
        pass
    
    def project(self, entity: Entity) -> Tuple[float, float]:
        """Map entity 3D position to panel 2D coords. Override."""
        raise NotImplementedError
    
    def render(self, ctx):
        """Draw this panel's view. Override."""
        raise NotImplementedError
    
    def update(self, dt: float):
        """Update panel state (e.g., follow playhead)."""
        pass
```

### 2.2 Sequencer Grid Panel (2D, X-Y view)

```python
class SequencerPanel(Panel):
    """
    Top-down view: X = beats, Y = lanes
    
    Shows:
    - Beat grid lines
    - Lane rows  
    - Events as colored rectangles
    - Playhead as vertical line
    """
    def project(self, entity: Entity) -> Tuple[float, float]:
        # X = beat position (through TimeCamera)
        px = self.scene.time_camera.beat_to_px(entity.position.x)
        # Y = lane (entity.position.y is lane index)
        py = entity.position.y * LANE_HEIGHT
        return (px, py)
    
    def render(self, ctx):
        # Draw lane backgrounds
        for i in range(NUM_LANES):
            ctx.draw_rect(0, i * LANE_HEIGHT, self.rect.w, LANE_HEIGHT, LANE_BG[i])
        
        # Draw beat grid
        for beat in self.scene.time_camera.iter_beat_positions():
            px = self.scene.time_camera.beat_to_px(beat)
            ctx.draw_line(px, 0, px, self.rect.h, GRID_COLOR)
        
        # Draw entities (only MIDI_EVENT type)
        for entity in self.scene.by_type(EntityType.MIDI_EVENT):
            if self.is_visible(entity):
                px, py = self.project(entity)
                w = entity.extent.x * self.scene.time_camera._px_per_beat
                h = LANE_HEIGHT - 4
                ctx.draw_rect(px, py + 2, w, h, entity.color)
        
        # Draw playhead
        playhead_px = self.scene.time_camera.beat_to_px(
            self.scene.transport.playhead_beat
        )
        ctx.draw_rect(playhead_px - 1, 0, 3, self.rect.h, PLAYHEAD_COLOR)
```

### 2.3 Waveform Panel (2D, X-Z view)

```python
class WaveformPanel(Panel):
    """
    Side view: X = beats, Y = amplitude (mapped from Z)
    
    Shows:
    - Audio waveform
    - Beat grid
    - Playhead
    - Event markers (vertical ticks where notes hit)
    """
    def __init__(self, scene: SpectroScene, rect: Rect, audio_engine: AudioEngine):
        super().__init__(scene, rect)
        self.audio = audio_engine
    
    def project(self, entity: Entity) -> Tuple[float, float]:
        px = self.scene.time_camera.beat_to_px(entity.position.x)
        # Z (intensity) maps to Y (height in waveform)
        py = self.rect.h * (1.0 - entity.position.z)
        return (px, py)
    
    def render(self, ctx):
        # Draw waveform from audio output
        waveform = self.audio.get_output_waveform(num_samples=self.rect.w)
        for i, amp in enumerate(waveform):
            h = amp * self.rect.h * 0.8
            y = (self.rect.h - h) / 2
            ctx.draw_rect(i, y, 1, h, WAVEFORM_COLOR)
        
        # Draw event markers (small ticks)
        for entity in self.scene.by_type(EntityType.MIDI_EVENT):
            px, py = self.project(entity)
            if 0 <= px <= self.rect.w:
                ctx.draw_rect(px, 0, 2, 10, entity.color)
        
        # Playhead
        playhead_px = self.scene.time_camera.beat_to_px(
            self.scene.transport.playhead_beat
        )
        ctx.draw_rect(playhead_px - 1, 0, 3, self.rect.h, PLAYHEAD_COLOR)
```

### 2.4 3D Viewport Panel (Perspective view)

```python
class Viewport3DPanel(Panel):
    """
    Perspective view into the scene.
    
    Shows:
    - Events as 3D cubes positioned at (beat, lane, velocity)
    - Time flows along X axis
    - Could rotate camera for different angles
    - Guitar Hero style: time coming toward you (Z depth)
    
    Uses existing ViewportArea + render pipeline.
    """
    def __init__(self, scene: SpectroScene, rect: Rect, ctx: moderngl.Context):
        super().__init__(scene, rect)
        
        # Use existing 3D infrastructure
        self.graph = EntityNode("root")
        self.viewport = ViewportArea("3d", self.graph, cameras=[
            Camera(
                name="main",
                eye=np.array([8.0, 4.0, 8.0]),
                target=np.array([4.0, 2.0, 0.0]),
            )
        ])
    
    def on_entity_change(self, entity: Entity, action: str):
        """Sync scene entity to 3D graph."""
        if entity.entity_type == EntityType.MIDI_EVENT:
            if action == "added":
                # Create 3D node for this entity
                node = EntityNode(f"note_{entity.id}")
                node.transform = Transform(
                    pos=np.array([
                        entity.position.x,  # beat → X
                        entity.position.y,  # lane → Y  
                        entity.position.z * 2.0,  # velocity → Z (scaled)
                    ], dtype=np.float32),
                    scale=np.array([0.2, 0.8, 0.3], dtype=np.float32),
                )
                node.mesh = MeshRenderer(
                    mesh_id="cube",
                    pipeline_id="lit_color",
                    color=np.array(entity.color, dtype=np.float32),
                )
                self.graph.add_child(node)
    
    def render(self, renderer: Renderer):
        """Render 3D scene to viewport texture."""
        self.viewport.render_if_ready(renderer)
```

---

## Phase 3: Main App Wiring

### 3.1 Layout

```
┌─────────────────────────────────────────────────────────────┐
│                     TRANSPORT BAR                           │
│  [▶] [■] [⟲]   BPM: 120   |  1:2:3  |  ════════════        │
├─────────────────────────────┬───────────────────────────────┤
│                             │                               │
│     SEQUENCER GRID          │         3D VIEWPORT           │
│     (2D: X=beat, Y=lane)    │      (perspective view)       │
│                             │                               │
│   ┌─┬─┬─┬─┬─┬─┬─┬─┐        │         ◇                     │
│   │■│ │ │■│ │ │ │ │ Kick   │       ◇   ◇                   │
│   ├─┼─┼─┼─┼─┼─┼─┼─┤        │     ◇       ◇                 │
│   │ │ │■│ │ │ │■│ │ Snare  │           ◇                   │
│   ├─┼─┼─┼─┼─┼─┼─┼─┤        │                               │
│   │■│■│■│■│■│■│■│■│ HiHat  │      ▲ (camera)               │
│   └─┴─┴─┴─┴─┴─┴─┴─┘        │                               │
│          │ playhead         │                               │
├──────────┼──────────────────┴───────────────────────────────┤
│          │                                                  │
│     WAVEFORM PANEL (2D: X=beat, Y=amplitude)               │
│   ∼∼∼∼╱╲∼∼∼╱╲∼∼∼∼∼∼∼∼╱╲╱╲∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼∼              │
│          │ playhead                                         │
└──────────┴──────────────────────────────────────────────────┘
```

### 3.2 Main App Structure

```python
class SpectroDemo(mglw.WindowConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # === THE SCENE (single source of truth) ===
        self.scene = SpectroScene(SignalBridge())
        
        # === AUDIO ===
        self.audio = AudioEngine(self.scene.transport)
        self.audio.load_default_sounds()
        self.audio.start()
        
        # === EVENT SYSTEM ===
        self.dispatcher = EventDispatcher(...)
        self._setup_callbacks()  # MIDI → scene.add_note()
        
        # === PANELS (all observe the same scene) ===
        self.panels = [
            SequencerPanel(self.scene, Rect(0, 50, 640, 300)),
            Viewport3DPanel(self.scene, Rect(640, 50, 640, 300), self.ctx),
            WaveformPanel(self.scene, Rect(0, 350, 1280, 150), self.audio),
        ]
        
        # Transport bar at top (not a panel, just UI)
        self.transport_bar = TransportBar(self.scene.transport)
    
    def on_render(self, t, dt):
        # Update systems
        self.scene.transport.update(dt)
        self.dispatcher.process_frame(dt)
        self.scene.time_camera.update(dt, self.scene.transport.playhead_beat)
        
        # Clear
        self.ctx.clear(0.08, 0.09, 0.11)
        
        # Render all panels
        for panel in self.panels:
            panel.update(dt)
            panel.render(self.ctx)  # or self.renderer for 3D
        
        # Transport bar
        self.transport_bar.render(self.ctx)
    
    def key_event(self, key, action, mods):
        if action == PRESS:
            if key in [K_1, K_2, K_3, K_4]:
                # Emit to dispatcher → callback → scene.add_note()
                self.keyboard_device.note_on(key - K_1, 100)
            elif key == K_SPACE:
                self.scene.transport.toggle()
```

---

## Phase 4: The Bounce in Action

```
USER PRESSES KEY "1"
        │
        ▼
keyboard_device.note_on(0, 100)
        │
        ▼
MidiRingBuffer.write(event)
        │
        ▼
dispatcher.process_frame()
        │
        ├── Event reaches playhead position
        │
        ▼
═══════════════════════════════════════════════════════
        │         TIMELINE BOUNCE
        │
        ▼
ExecutionContext assembled with FRESH state:
  - ctx.transport.beat = 4.75 (current position)
  - ctx.timing_error_ms = -12.3 (slightly early)
        │
        ▼
═══════════════════════════════════════════════════════
        │
        ▼
on_midi_input(ctx) callback fires
        │
        ├── scene.add_note(beat=4.75, lane=0, velocity=100)
        │       │
        │       ├── Creates Entity in scene
        │       │
        │       └── Notifies all observers:
        │               │
        │               ├── SequencerPanel: "I'll draw a rect at beat 4.75, lane 0"
        │               │
        │               ├── WaveformPanel: "I'll draw a tick mark at beat 4.75"
        │               │
        │               └── Viewport3DPanel: "I'll spawn a cube at (4.75, 0, 0.78)"
        │
        └── audio.trigger("kick", 0.78)
                │
                └── SOUND PLAYS
```

---

## Deliverable Checklist

- [ ] **SpectroScene** class wrapping Scene + Transport + TimeCamera + observer pattern
- [ ] **Panel base class** with project(), render(), on_entity_change()
- [ ] **SequencerPanel** (2D X-Y grid view)
- [ ] **WaveformPanel** (2D X-Z amplitude view)  
- [ ] **Viewport3DPanel** (3D perspective, uses existing render pipeline)
- [ ] **Main app** with layout, input handling, render loop
- [ ] **MIDI callback** that adds to scene (not directly to panel)
- [ ] **Playback** that fires events when playhead crosses them

---

## Files to Modify/Create

```
MODIFY:
  engine/core/scene.py          - Add observer pattern
  spectro_demo_app.py           - Restructure around unified scene

CREATE:
  engine/ui/panels/base.py      - Panel base class
  engine/ui/panels/sequencer.py - SequencerPanel
  engine/ui/panels/waveform.py  - WaveformPanel  
  engine/ui/panels/viewport3d.py - Viewport3DPanel

EXISTING (use as-is):
  engine/render/*               - 3D pipeline (works)
  engine/viewport/viewport.py   - ViewportArea (works)
  buffers_v2.py                 - EventDispatcher (works)
```

---

## The Key Insight

**The scene is not a data structure. It's a space.**

Entities exist in that space. Panels are windows into it. The timeline is where context bounces. When something happens, it happens IN THE SCENE, and all views update because they're looking at the same space.

This is why your architecture works: you're not synchronizing three separate data stores. You have ONE space, THREE projections.

# SPECTRO Architecture — Detailed Design Document

> **Purpose**: Complete architectural specification with UML diagrams, event flow, 
> rendering pipeline, and branch organization strategy.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Branch Organization Strategy](#2-branch-organization-strategy)
3. [Layout Tree Architecture](#3-layout-tree-architecture)
4. [Event Cascade System](#4-event-cascade-system)
5. [Panel Architecture](#5-panel-architecture)
6. [Timeline Synchronization](#6-timeline-synchronization)
7. [Rendering Pipeline](#7-rendering-pipeline)
8. [Entity-Component Model](#8-entity-component-model)
9. [Text Rendering System](#9-text-rendering-system)
10. [Complete Class Diagrams](#10-complete-class-diagrams)

---

## 1. System Overview

### 1.1 High-Level UML — Main Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    APPLICATION                                           │
│                                                                                          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐    │
│  │   MainLoop   │─────▶│  InputQueue  │─────▶│  EventBus    │─────▶│  LayoutTree  │    │
│  │  (60 FPS)    │      │              │      │              │      │   (Root)     │    │
│  └──────────────┘      └──────────────┘      └──────────────┘      └──────┬───────┘    │
│         │                                           │                      │            │
│         │                                           │                      ▼            │
│         │                                           │              ┌──────────────┐    │
│         │                                           │              │  3D Panel    │    │
│         │                                           │              │  (unified)   │    │
│         │                                           │              └──────┬───────┘    │
│         │                                           │                      │            │
│         ▼                                           ▼                      ▼            │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐    │
│  │  Transport   │─────▶│ MasterClock  │─────▶│  Timelines   │◀────▶│   Entities   │    │
│  │              │      │              │      │  (per panel) │      │ (Components) │    │
│  └──────────────┘      └──────────────┘      └──────────────┘      └──────┬───────┘    │
│                                                                            │            │
│                                                                            ▼            │
│                                                    ┌──────────────────────────────────┐│
│                                                    │         RENDER PIPELINE          ││
│                                                    │  ┌────────┐ ┌────────┐ ┌───────┐ ││
│                                                    │  │Collect │▶│ Batch  │▶│Execute│ ││
│                                                    │  │Commands│ │& Sort  │ │  GL   │ ││
│                                                    │  └────────┘ └────────┘ └───────┘ ││
│                                                    └──────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Key Insight: Two Kinds of "Movement"

There are **two completely different kinds of animation** in your system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TYPE 1: SCROLLING (TimeCamera movement)                                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                    │
│                                                                              │
│  The VIEWPORT moves over STATIONARY data.                                   │
│  Entities don't change position — the camera pans across them.              │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  beat 0    beat 4    beat 8    beat 12   beat 16   beat 20        │     │
│  │  ▓▓▓▓▓     ████████  ▓▓▓       █████     ▓▓▓▓      ████           │     │
│  │            ┌──────────────────────────┐                            │     │
│  │            │     VIEWPORT WINDOW      │◀── TimeCamera.left_beat   │     │
│  │            │    (what user sees)      │                            │     │
│  │            └──────────────────────────┘                            │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  • Entity positions are in BEAT SPACE (never change)                        │
│  • TimeCamera converts beat → pixel for current view                        │
│  • Scrolling = updating TimeCamera.left_beat                                │
│  • Rendering uses: px = (entity.beat - camera.left_beat) * px_per_beat      │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  TYPE 2: ANIMATION (Entity property changes over time)                      │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                       │
│                                                                              │
│  Entity PROPERTIES change over time (position, rotation, scale, alpha).     │
│                                                                              │
│  ┌────────────────┐                                                         │
│  │    Entity      │                                                         │
│  │  ┌──────────┐  │   Timeline: ─────●─────●─────●─────▶ time               │
│  │  │ position │◀─┼───────────────┘                                         │
│  │  │ rotation │◀─┼─────────────────────┘                                   │
│  │  │ scale    │◀─┼───────────────────────────┘                             │
│  │  │ alpha    │  │                                                         │
│  │  └──────────┘  │                                                         │
│  └────────────────┘                                                         │
│                                                                              │
│  • Keyframe interpolation updates entity properties                         │
│  • Used for: glow effects, transitions, 3D orbit, etc.                      │
│  • Independent of scrolling — happens in local entity time                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Branch Organization Strategy

### 2.1 Three Branches

```
main
  │
  ├── feature/layout-tree
  │   │
  │   │   Core layout system, no rendering yet
  │   │   - LayoutNode base class
  │   │   - FlexBox layout algorithm  
  │   │   - Constraint propagation
  │   │   - Event cascade (linked list traversal)
  │   │   - Hit testing
  │   │
  │   └── Deliverable: Layout tree that can be traversed and resized
  │
  ├── feature/panel-system
  │   │
  │   │   3D Panel with timeline + entities
  │   │   - Panel3D class
  │   │   - TimeCamera
  │   │   - Entity + Component model
  │   │   - Local timeline per panel
  │   │   - Sync with master clock
  │   │
  │   └── Deliverable: Single panel with scrolling entities
  │
  └── feature/render-pipeline
      │
      │   Batched rendering system
      │   - DrawBatch / CommandList
      │   - UIRenderer2D
      │   - Text rendering
      │   - Shader management
      │   - GPU buffer pooling
      │
      └── Deliverable: Can render a DrawBatch to screen
```

### 2.2 Merge Order

```
1. feature/render-pipeline → main
   (Need rendering to see anything)

2. feature/layout-tree → main  
   (Need layout to position panels)

3. feature/panel-system → main
   (Panels go into layout, use rendering)
```

### 2.3 Integration Points

```
┌─────────────────────┐
│   Layout Tree       │──────────────────┐
│   (positions)       │                  │
└─────────────────────┘                  │
          │                              │
          │ provides Rect               │
          ▼                              ▼
┌─────────────────────┐         ┌─────────────────────┐
│   Panel System      │────────▶│   Render Pipeline   │
│   (entities)        │ emits   │   (executes)        │
│                     │ DrawBatch                     │
└─────────────────────┘         └─────────────────────┘
```

---

## 3. Layout Tree Architecture

### 3.1 Class Diagram

```
                            ┌─────────────────────────┐
                            │      LayoutNode         │
                            │ ─────────────────────── │
                            │ id: str                 │
                            │ rect: Rect              │
                            │ style: Style            │
                            │ parent: LayoutNode?     │
                            │ children: List[Node]    │
                            │ next_sibling: Node?     │◀── Linked list for traversal
                            │ first_child: Node?      │
                            │ ─────────────────────── │
                            │ measure(constraints)    │
                            │ layout(rect)            │
                            │ handle_event(event)     │
                            │ draw(ctx: DrawContext)  │
                            └───────────┬─────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │    Container      │ │     Splitter      │ │     Panel3D       │
        │ ───────────────── │ │ ───────────────── │ │ ───────────────── │
        │ direction: Axis   │ │ orientation: Axis │ │ timeline: Timeline│
        │ gap: float        │ │ position: float   │ │ camera: TimeCamera│
        │                   │ │ dragging: bool    │ │ entities: List    │
        └───────────────────┘ └───────────────────┘ └───────────────────┘
```

### 3.2 Tree Structure (Your Single 3D Panel Case)

```
Root (Container, vertical)
  │
  ├── Header (Container, horizontal)
  │     ├── TransportControls
  │     └── BPMDisplay
  │
  └── Panel3D ◀── THE MAIN PANEL
        │
        │   Contains multiple "regions" rendered as one:
        │   ┌────────────────────────────────────┐
        │   │  Sync Circles (phase indicators)   │  ← Region 0
        │   ├────────────────────────────────────┤
        │   │  Waveform Timeline                 │  ← Region 1
        │   ├────────────────────────────────────┤
        │   │  3D Viewport with Lanes            │  ← Region 2
        │   └────────────────────────────────────┘
        │
        └── All regions share the SAME TimeCamera
            and render in ONE draw call sequence
```

### 3.3 Linked List Traversal for Events

```python
class LayoutNode:
    """Base class with linked-list pointers for efficient traversal."""
    
    # Tree structure
    parent: Optional["LayoutNode"] = None
    first_child: Optional["LayoutNode"] = None
    next_sibling: Optional["LayoutNode"] = None
    
    # Cached for O(1) access
    last_child: Optional["LayoutNode"] = None
    prev_sibling: Optional["LayoutNode"] = None
    
    def add_child(self, child: "LayoutNode"):
        """Add child, maintaining linked list."""
        child.parent = self
        if self.first_child is None:
            self.first_child = child
            self.last_child = child
        else:
            self.last_child.next_sibling = child
            child.prev_sibling = self.last_child
            self.last_child = child
    
    def traverse_depth_first(self) -> Iterator["LayoutNode"]:
        """Iterate tree without recursion using linked list."""
        node = self
        while node is not None:
            yield node
            
            # Go to first child if exists
            if node.first_child is not None:
                node = node.first_child
            # Otherwise go to next sibling
            elif node.next_sibling is not None:
                node = node.next_sibling
            # Otherwise go up and right
            else:
                while node.parent is not None:
                    node = node.parent
                    if node.next_sibling is not None:
                        node = node.next_sibling
                        break
                else:
                    node = None
    
    def traverse_ancestors(self) -> Iterator["LayoutNode"]:
        """Walk up the tree (for event bubbling)."""
        node = self.parent
        while node is not None:
            yield node
            node = node.parent
```

---

## 4. Event Cascade System

### 4.1 Event Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVENT FLOW                                      │
│                                                                              │
│   Input Source          Phase 1: CAPTURE         Phase 2: BUBBLE            │
│   ────────────          ────────────────         ───────────────            │
│                                                                              │
│   ┌─────────┐           Root ─────────┐          ┌───────── Root            │
│   │ Keyboard│               │         │          │              ▲           │
│   │ Mouse   │───▶      Container      │          │         Container        │
│   │ Touch   │               │         │          │              ▲           │
│   └─────────┘               │         ▼          ▼              │           │
│                         Panel3D ──────●───▶ TARGET ───▶ Panel3D             │
│                                                                              │
│   ● = Event reaches target, then bubbles back up                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Event Types

```python
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any

class EventPhase(Enum):
    CAPTURE = auto()   # Going down the tree
    TARGET = auto()    # At the target node
    BUBBLE = auto()    # Going back up

class EventType(Enum):
    # Pointer events
    POINTER_DOWN = auto()
    POINTER_UP = auto()
    POINTER_MOVE = auto()
    POINTER_ENTER = auto()
    POINTER_LEAVE = auto()
    
    # Scroll/zoom
    SCROLL = auto()
    ZOOM = auto()
    
    # Keyboard
    KEY_DOWN = auto()
    KEY_UP = auto()
    
    # Focus
    FOCUS_IN = auto()
    FOCUS_OUT = auto()
    
    # Transport (custom)
    TRANSPORT_PLAY = auto()
    TRANSPORT_PAUSE = auto()
    TRANSPORT_SEEK = auto()
    TRANSPORT_TICK = auto()  # Every frame while playing
    
    # Layout
    RESIZE = auto()
    
    # Timeline
    BEAT = auto()  # Fires on beat boundaries
    BAR = auto()   # Fires on bar boundaries

@dataclass
class Event:
    type: EventType
    phase: EventPhase = EventPhase.CAPTURE
    target: Optional["LayoutNode"] = None
    current_target: Optional["LayoutNode"] = None
    
    # State
    propagation_stopped: bool = False
    default_prevented: bool = False
    
    # Pointer data (if applicable)
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    button: int = 0
    
    # Key data (if applicable)
    key: str = ""
    modifiers: set = field(default_factory=set)
    
    # Scroll data
    scroll_x: float = 0.0
    scroll_y: float = 0.0
    
    # Transport data
    beat: float = 0.0
    bar: int = 0
    
    # Generic payload
    data: Any = None
    
    def stop_propagation(self):
        self.propagation_stopped = True
    
    def prevent_default(self):
        self.default_prevented = True
```

### 4.3 Event Dispatcher

```python
class EventDispatcher:
    """Dispatches events through the layout tree."""
    
    def __init__(self, root: LayoutNode):
        self.root = root
        self._capture_listeners: Dict[EventType, List[Callable]] = {}
        self._bubble_listeners: Dict[EventType, List[Callable]] = {}
    
    def dispatch(self, event: Event, target: LayoutNode):
        """Full event dispatch cycle: capture → target → bubble."""
        
        event.target = target
        
        # Build path from root to target
        path = list(target.traverse_ancestors())
        path.reverse()  # Now root → ... → parent
        path.append(target)
        
        # CAPTURE PHASE: root → target
        event.phase = EventPhase.CAPTURE
        for node in path[:-1]:  # Exclude target
            if event.propagation_stopped:
                return
            event.current_target = node
            node.handle_event(event)
        
        # TARGET PHASE
        event.phase = EventPhase.TARGET
        event.current_target = target
        target.handle_event(event)
        
        if event.propagation_stopped:
            return
        
        # BUBBLE PHASE: target → root
        event.phase = EventPhase.BUBBLE
        for node in reversed(path[:-1]):  # Exclude target, reverse order
            if event.propagation_stopped:
                return
            event.current_target = node
            node.handle_event(event)
    
    def dispatch_to_tree(self, event: Event):
        """Dispatch event to entire tree (e.g., TRANSPORT_TICK)."""
        event.phase = EventPhase.CAPTURE
        for node in self.root.traverse_depth_first():
            event.current_target = node
            node.handle_event(event)
            if event.propagation_stopped:
                return
```

### 4.4 How Events Schedule Rendering

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EVENT → RENDER SCHEDULING                                │
│                                                                              │
│   1. Event dispatched to Panel3D                                            │
│      │                                                                       │
│      ▼                                                                       │
│   2. Panel3D.handle_event() updates internal state                          │
│      - TimeCamera.left_beat (if scroll)                                     │
│      - Entity positions (if drag)                                           │
│      - Selection state (if click)                                           │
│      │                                                                       │
│      ▼                                                                       │
│   3. Panel3D marks itself dirty                                             │
│      self._needs_redraw = True                                              │
│      │                                                                       │
│      ▼                                                                       │
│   4. Next frame, collect_draw_commands() is called                          │
│      │                                                                       │
│      ▼                                                                       │
│   5. Panel3D emits DrawBatch with all visible entities                      │
│      │                                                                       │
│      ▼                                                                       │
│   6. RenderScheduler receives batch, sorts, executes                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Panel Architecture

### 5.1 The Unified 3D Panel

```python
class Panel3D(LayoutNode):
    """
    Single panel containing all visualization regions:
    - Sync circles (phase indicators)
    - Waveform timeline
    - 3D viewport with lanes
    
    All regions share the same TimeCamera for perfect sync.
    """
    
    def __init__(self):
        super().__init__()
        
        # Shared timeline infrastructure
        self.time_camera = TimeCamera()
        self.timeline = Timeline()  # Local animation timeline
        self.transport_state: Optional[TransportState] = None
        
        # Regions (logical divisions, not separate nodes)
        self.regions: List[PanelRegion] = [
            SyncCirclesRegion(height_ratio=0.15),
            WaveformRegion(height_ratio=0.25),
            LanesRegion(height_ratio=0.60),
        ]
        
        # Entity storage (all entities across all regions)
        self.entities: List[Entity] = []
        self.entity_index: Dict[int, Entity] = {}  # id → entity
        
        # Render state
        self._needs_redraw: bool = True
        self._cached_batch: Optional[DrawBatch] = None
        
        # Selection
        self.selected_ids: Set[int] = set()
        self.hover_id: Optional[int] = None
    
    def layout(self, rect: Rect):
        """Divide rect among regions."""
        self.rect = rect
        y = rect.y
        for region in self.regions:
            h = rect.h * region.height_ratio
            region.rect = Rect(rect.x, y, rect.w, h)
            y += h
        
        # Update camera with new width
        self.time_camera._panel_width_px = rect.w
        self._needs_redraw = True
    
    def handle_event(self, event: Event):
        """Route events to appropriate region or handle globally."""
        
        if event.type == EventType.TRANSPORT_TICK:
            # Update from master clock
            self.transport_state = event.data
            self.time_camera.update(
                dt=1/60,  # Assume 60fps
                playhead_beat=self.transport_state.playhead_beat,
                panel_width_px=self.rect.w,
                frame_id=self.transport_state.frame_id,
            )
            self._needs_redraw = True
            return
        
        if event.type == EventType.SCROLL:
            self.time_camera.zoom(event.scroll_y, event.x - self.rect.x)
            self._needs_redraw = True
            event.stop_propagation()
            return
        
        if event.type in (EventType.POINTER_DOWN, EventType.POINTER_MOVE, EventType.POINTER_UP):
            # Find which region
            for region in self.regions:
                if region.rect.contains(event.x, event.y):
                    region.handle_event(event, self)
                    break
            return
    
    def collect_draw_commands(self) -> DrawBatch:
        """Generate all drawing commands for this panel."""
        
        if not self._needs_redraw and self._cached_batch is not None:
            return self._cached_batch
        
        batch = DrawBatch(panel_id=self.id)
        
        # Background
        batch.rect(self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                   color=(0.1, 0.1, 0.12, 1.0))
        
        # Each region draws itself
        for region in self.regions:
            region.draw(batch, self.time_camera, self.entities, self)
        
        # Playhead (on top of everything)
        playhead_x = self.time_camera.beat_to_px(self.transport_state.playhead_beat)
        batch.line(
            self.rect.x + playhead_x, self.rect.y,
            self.rect.x + playhead_x, self.rect.y + self.rect.h,
            color=(1.0, 0.3, 0.3, 1.0),
            width=2.0
        )
        
        self._cached_batch = batch
        self._needs_redraw = False
        return batch
```

### 5.2 Panel Regions

```python
@dataclass
class PanelRegion(ABC):
    """Logical region within Panel3D."""
    height_ratio: float
    rect: Rect = field(default_factory=lambda: Rect(0,0,0,0))
    
    @abstractmethod
    def draw(self, batch: DrawBatch, camera: TimeCamera, 
             entities: List[Entity], panel: Panel3D): ...
    
    def handle_event(self, event: Event, panel: Panel3D):
        """Override for region-specific interaction."""
        pass


class SyncCirclesRegion(PanelRegion):
    """Phase indicator circles — does NOT scroll with TimeCamera."""
    
    def draw(self, batch: DrawBatch, camera: TimeCamera,
             entities: List[Entity], panel: Panel3D):
        
        ts = panel.transport_state
        if ts is None:
            return
        
        # Draw beat phase circle
        cx = self.rect.x + self.rect.w * 0.25
        cy = self.rect.y + self.rect.h * 0.5
        radius = min(self.rect.w * 0.1, self.rect.h * 0.4)
        
        # Background ring
        batch.circle(cx, cy, radius, 
                     fill_color=(0.2, 0.2, 0.25, 1.0),
                     stroke_color=(0.4, 0.4, 0.45, 1.0),
                     stroke_width=2.0)
        
        # Phase arc
        phase = ts.phase_in_beat * 2 * math.pi - math.pi/2  # Start at top
        batch.arc(cx, cy, radius * 0.8,
                  start_angle=-math.pi/2,
                  end_angle=phase,
                  color=(0.3, 0.8, 0.4, 1.0),
                  width=4.0)


class WaveformRegion(PanelRegion):
    """Waveform display — scrolls with TimeCamera."""
    
    waveform_cache: Optional[WaveformCache] = None
    
    def draw(self, batch: DrawBatch, camera: TimeCamera,
             entities: List[Entity], panel: Panel3D):
        
        # Grid lines (using GridBuilder)
        grid = panel.grid_builder.build(camera, self.rect, beats_per_bar=4.0)
        batch.extend(grid)
        
        # Waveform
        if self.waveform_cache is not None:
            # ... waveform rendering
            pass
        
        # Hit markers from entities
        for entity in entities:
            if entity.region == "waveform" and camera.is_beat_visible(entity.beat):
                x = self.rect.x + camera.beat_to_px(entity.beat)
                # Draw marker
                batch.line(x, self.rect.y, x, self.rect.y + self.rect.h,
                           color=entity.color, width=2.0)


class LanesRegion(PanelRegion):
    """3D lanes viewport — scrolls with TimeCamera, supports depth."""
    
    lane_count: int = 8
    lane_height: float = 24.0
    
    def draw(self, batch: DrawBatch, camera: TimeCamera,
             entities: List[Entity], panel: Panel3D):
        
        # Lane backgrounds (alternating)
        for i in range(self.lane_count):
            y = self.rect.y + i * self.lane_height
            color = (0.12, 0.12, 0.14, 1.0) if i % 2 == 0 else (0.10, 0.10, 0.12, 1.0)
            batch.rect(self.rect.x, y, self.rect.w, self.lane_height, color=color)
        
        # Grid
        grid = panel.grid_builder.build(camera, self.rect, beats_per_bar=4.0)
        batch.extend(grid)
        
        # Collect visible entities for this region
        visible_entities = [
            e for e in entities
            if e.region == "lanes" and camera.is_beat_visible(e.beat)
        ]
        
        # Sort by depth (back to front for alpha blending)
        visible_entities.sort(key=lambda e: e.depth, reverse=True)
        
        # Batch similar entities
        if len(visible_entities) > 4:
            # Use instanced rendering
            instance_data = self._build_instance_data(visible_entities, camera)
            batch.add(Cmd2DRectsInstanced(
                instance_data=instance_data,
                instance_count=len(visible_entities),
                corner_radius=3.0
            ))
        else:
            # Individual draws
            for entity in visible_entities:
                x = self.rect.x + camera.beat_to_px(entity.beat)
                w = entity.duration * camera.px_per_beat
                y = self.rect.y + entity.lane * self.lane_height
                batch.rect(x, y, w, self.lane_height - 2, 
                           color=entity.color, corner_radius=3.0)
```

---

## 6. Timeline Synchronization

### 6.1 Master Clock and Per-Panel Timelines

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TIMELINE HIERARCHY                                   │
│                                                                              │
│                        ┌──────────────────┐                                  │
│                        │   MasterClock    │                                  │
│                        │ ──────────────── │                                  │
│                        │ bpm: float       │                                  │
│                        │ playing: bool    │                                  │
│                        │ beat: float      │                                  │
│                        │ frame_id: int    │                                  │
│                        └────────┬─────────┘                                  │
│                                 │                                            │
│              ┌──────────────────┼──────────────────┐                        │
│              │                  │                  │                        │
│              ▼                  ▼                  ▼                        │
│    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                   │
│    │  Timeline A  │   │  Timeline B  │   │  Timeline C  │                   │
│    │  (Panel 1)   │   │  (Panel 2)   │   │  (UI Anim)   │                   │
│    └──────────────┘   └──────────────┘   └──────────────┘                   │
│                                                                              │
│    Each Timeline can:                                                        │
│    - Follow master clock (synchronized)                                      │
│    - Run independently (for local animations)                                │
│    - Have different time scale (slow-mo, fast-forward)                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation

```python
@dataclass
class MasterClock:
    """Global time source, drives all synchronized timelines."""
    
    bpm: float = 120.0
    playing: bool = False
    beat: float = 0.0
    time_seconds: float = 0.0
    frame_id: int = 0
    
    # Looping
    loop_enabled: bool = False
    loop_start: float = 0.0
    loop_end: float = 16.0
    
    # Callbacks
    _on_beat: List[Callable[[int], None]] = field(default_factory=list)
    _on_bar: List[Callable[[int], None]] = field(default_factory=list)
    
    def tick(self, dt: float) -> TransportState:
        """Advance clock by dt seconds. Call once per frame."""
        self.frame_id += 1
        
        if not self.playing:
            return self._make_state()
        
        old_beat = self.beat
        
        # Advance
        beats_per_second = self.bpm / 60.0
        self.beat += dt * beats_per_second
        self.time_seconds += dt
        
        # Looping
        if self.loop_enabled and self.beat >= self.loop_end:
            self.beat = self.loop_start + (self.beat - self.loop_end)
        
        # Fire beat callbacks
        old_beat_int = int(old_beat)
        new_beat_int = int(self.beat)
        if new_beat_int > old_beat_int:
            for cb in self._on_beat:
                cb(new_beat_int)
        
        # Fire bar callbacks (assuming 4/4)
        old_bar = int(old_beat / 4)
        new_bar = int(self.beat / 4)
        if new_bar > old_bar:
            for cb in self._on_bar:
                cb(new_bar)
        
        return self._make_state()
    
    def _make_state(self) -> TransportState:
        return TransportState(
            playing=self.playing,
            playhead_beat=self.beat,
            playhead_time=self.time_seconds,
            bpm=self.bpm,
            frame_id=self.frame_id,
            phase_in_beat=self.beat % 1.0,
            phase_in_bar=(self.beat % 4.0) / 4.0,
            current_bar=int(self.beat / 4),
            current_beat_in_bar=int(self.beat % 4),
        )


class Timeline:
    """Per-panel or per-entity animation timeline."""
    
    def __init__(self, sync_to_master: bool = True):
        self.sync_to_master = sync_to_master
        self.local_time: float = 0.0
        self.time_scale: float = 1.0
        self.paused: bool = False
        
        # Keyframe tracks
        self.tracks: Dict[str, KeyframeTrack] = {}
    
    def update(self, master_state: TransportState, dt: float):
        """Update timeline based on master clock or local time."""
        if self.sync_to_master:
            # Follow master beat
            self.local_time = master_state.playhead_beat
        else:
            # Independent timeline
            if not self.paused:
                self.local_time += dt * self.time_scale
        
        # Evaluate all keyframe tracks
        for track in self.tracks.values():
            track.evaluate(self.local_time)
    
    def add_track(self, name: str, target: Any, property_name: str) -> "KeyframeTrack":
        """Create a keyframe track for animating a property."""
        track = KeyframeTrack(target, property_name)
        self.tracks[name] = track
        return track


@dataclass
class Keyframe:
    time: float
    value: Any
    easing: str = "linear"  # "linear", "ease_in", "ease_out", "ease_in_out"


class KeyframeTrack:
    """Animates a single property over time."""
    
    def __init__(self, target: Any, property_name: str):
        self.target = target
        self.property_name = property_name
        self.keyframes: List[Keyframe] = []
    
    def add_keyframe(self, time: float, value: Any, easing: str = "linear"):
        kf = Keyframe(time, value, easing)
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.time)
    
    def evaluate(self, time: float):
        """Compute interpolated value and apply to target."""
        if not self.keyframes:
            return
        
        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]
        
        for i, kf in enumerate(self.keyframes):
            if kf.time > time:
                next_kf = kf
                if i > 0:
                    prev_kf = self.keyframes[i - 1]
                break
            prev_kf = kf
        
        # Interpolate
        if prev_kf.time == next_kf.time:
            value = prev_kf.value
        else:
            t = (time - prev_kf.time) / (next_kf.time - prev_kf.time)
            t = self._apply_easing(t, next_kf.easing)
            value = self._lerp(prev_kf.value, next_kf.value, t)
        
        # Apply to target
        setattr(self.target, self.property_name, value)
    
    def _apply_easing(self, t: float, easing: str) -> float:
        if easing == "linear":
            return t
        elif easing == "ease_in":
            return t * t
        elif easing == "ease_out":
            return 1 - (1 - t) ** 2
        elif easing == "ease_in_out":
            return 3 * t**2 - 2 * t**3
        return t
    
    def _lerp(self, a: Any, b: Any, t: float) -> Any:
        """Linear interpolation for various types."""
        if isinstance(a, (int, float)):
            return a + (b - a) * t
        elif isinstance(a, tuple):
            return tuple(self._lerp(ai, bi, t) for ai, bi in zip(a, b))
        return b if t > 0.5 else a
```

---

## 7. Rendering Pipeline

### 7.1 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RENDER PIPELINE                                      │
│                                                                              │
│  PHASE 1: COLLECT                                                           │
│  ────────────────                                                           │
│                                                                              │
│  ┌─────────────┐                                                            │
│  │ LayoutTree  │                                                            │
│  │   Root      │                                                            │
│  └──────┬──────┘                                                            │
│         │ traverse_depth_first()                                            │
│         ▼                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   Node 1    │───▶│   Node 2    │───▶│   Panel3D   │                      │
│  │  draw(ctx)  │    │  draw(ctx)  │    │  draw(ctx)  │                      │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                      │
│         │                 │                   │                              │
│         ▼                 ▼                   ▼                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        DrawContext                                      │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │ │
│  │  │ Batch 0  │ │ Batch 1  │ │ Batch 2  │ │ Batch 3  │ │ Batch 4  │     │ │
│  │  │ z=0      │ │ z=0      │ │ z=10     │ │ z=10     │ │ z=100    │     │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: SORT & BATCH                                                      │
│  ─────────────────────                                                       │
│                                                                              │
│  DrawContext.finalize() → List[DrawBatch] sorted by z_order                 │
│                                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Batch 0  │ │ Batch 1  │ │ Batch 2  │ │ Batch 3  │ │ Batch 4  │          │
│  │ z=0      │ │ z=0      │ │ z=10     │ │ z=10     │ │ z=100    │          │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
│       │            │            │            │            │                  │
│       ▼            ▼            ▼            ▼            ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  RenderScheduler                                                         ││
│  │                                                                          ││
│  │  For each batch:                                                         ││
│  │    - Group commands by type (rects, lines, text, etc.)                  ││
│  │    - Convert to instanced draws where beneficial (>4 items)             ││
│  │    - Emit optimized CommandList                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: EXECUTE                                                           │
│  ───────────────                                                             │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  UIRenderer2D                                                            ││
│  │                                                                          ││
│  │  For each command:                                                       ││
│  │    1. Bind appropriate shader program                                   ││
│  │    2. Set uniforms (resolution, clip rect, etc.)                        ││
│  │    3. Upload instance data to GPU buffer                                ││
│  │    4. Issue draw call                                                   ││
│  │                                                                          ││
│  │  State tracking:                                                         ││
│  │    - Current shader                                                      ││
│  │    - Current clip rect                                                   ││
│  │    - Minimize state changes between draws                               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
│                            ┌──────────────┐                                  │
│                            │  FRAMEBUFFER │                                  │
│                            │   (screen)   │                                  │
│                            └──────────────┘                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Command Batching Logic

```python
class RenderScheduler:
    """Optimizes draw batches before execution."""
    
    # Threshold for switching to instanced rendering
    INSTANCE_THRESHOLD = 4
    
    def optimize(self, batches: List[DrawBatch]) -> List[DrawBatch]:
        """Convert individual draws to instanced where beneficial."""
        
        optimized = []
        
        for batch in batches:
            opt_batch = DrawBatch(panel_id=batch.panel_id, z_order=batch.z_order)
            
            # Group commands by type
            rect_cmds: List[Cmd2DRect] = []
            line_cmds: List[Cmd2DLine] = []
            other_cmds: List[Command2D] = []
            
            for cmd in batch.commands:
                if isinstance(cmd, Cmd2DRect):
                    rect_cmds.append(cmd)
                elif isinstance(cmd, Cmd2DLine):
                    line_cmds.append(cmd)
                else:
                    # Flush accumulated rects/lines before other commands
                    self._flush_rects(opt_batch, rect_cmds)
                    self._flush_lines(opt_batch, line_cmds)
                    rect_cmds.clear()
                    line_cmds.clear()
                    opt_batch.add(cmd)
            
            # Flush remaining
            self._flush_rects(opt_batch, rect_cmds)
            self._flush_lines(opt_batch, line_cmds)
            
            optimized.append(opt_batch)
        
        return optimized
    
    def _flush_rects(self, batch: DrawBatch, rects: List[Cmd2DRect]):
        if len(rects) >= self.INSTANCE_THRESHOLD:
            # Group by corner_radius
            by_radius: Dict[float, List[Cmd2DRect]] = {}
            for r in rects:
                by_radius.setdefault(r.corner_radius, []).append(r)
            
            for radius, group in by_radius.items():
                if len(group) >= self.INSTANCE_THRESHOLD:
                    # Build instance buffer
                    data = np.zeros((len(group), 8), dtype=np.float32)
                    for i, r in enumerate(group):
                        data[i] = [r.x, r.y, r.w, r.h, *r.color]
                    batch.add(Cmd2DRectsInstanced(
                        instance_data=data.tobytes(),
                        instance_count=len(group),
                        corner_radius=radius
                    ))
                else:
                    for r in group:
                        batch.add(r)
        else:
            for r in rects:
                batch.add(r)
    
    def _flush_lines(self, batch: DrawBatch, lines: List[Cmd2DLine]):
        if len(lines) >= self.INSTANCE_THRESHOLD:
            # Build instance buffer
            data = np.zeros((len(lines), 9), dtype=np.float32)
            for i, l in enumerate(lines):
                data[i] = [l.x0, l.y0, l.x1, l.y1, *l.color, l.width]
            batch.add(Cmd2DLinesInstanced(
                instance_data=data.tobytes(),
                instance_count=len(lines)
            ))
        else:
            for l in lines:
                batch.add(l)
```

### 7.3 UIRenderer2D Full Implementation

```python
class UIRenderer2D:
    """Executes 2D drawing commands on GPU."""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._viewport_size = (800, 600)
        
        # Shader programs
        self._programs: Dict[str, moderngl.Program] = {}
        
        # Geometry
        self._unit_quad_vao: moderngl.VertexArray = None
        self._unit_line_vao: moderngl.VertexArray = None
        
        # Instance buffers (grow as needed)
        self._rect_instance_buf: moderngl.Buffer = None
        self._line_instance_buf: moderngl.Buffer = None
        self._instance_buf_size = 10000  # Initial capacity
        
        # Clip stack
        self._clip_stack: List[Tuple[int,int,int,int]] = []
        
        # State tracking for minimal changes
        self._current_program: Optional[moderngl.Program] = None
        
        # Text system
        self._font_manager: FontManager = None
        self._text_renderer: TextRenderer = None
        
        self._setup()
    
    def _setup(self):
        """Initialize shaders, buffers, geometry."""
        
        # Load shaders
        shader_dir = Path(__file__).parent.parent / "render" / "shaders"
        self._programs["rect"] = self._load_program(shader_dir, "rect")
        self._programs["rect_instanced"] = self._load_program(shader_dir, "rect_instanced")
        self._programs["line"] = self._load_program(shader_dir, "line")
        self._programs["line_instanced"] = self._load_program(shader_dir, "line_instanced")
        self._programs["waveform"] = self._load_program(shader_dir, "waveform")
        self._programs["circle"] = self._load_program(shader_dir, "circle")
        self._programs["text"] = self._load_program(shader_dir, "text")
        
        # Unit quad: positions for a [0,1] x [0,1] quad
        quad_verts = np.array([
            0, 0,  1, 0,  1, 1,
            0, 0,  1, 1,  0, 1,
        ], dtype='f4')
        quad_vbo = self.ctx.buffer(quad_verts)
        self._unit_quad_vao = self.ctx.vertex_array(
            self._programs["rect"],
            [(quad_vbo, '2f', 'in_pos')],
        )
        
        # Unit line: positions for a [0,0] to [1,0] line segment
        line_verts = np.array([
            0, -0.5,  1, -0.5,  1, 0.5,
            0, -0.5,  1, 0.5,   0, 0.5,
        ], dtype='f4')
        line_vbo = self.ctx.buffer(line_verts)
        self._unit_line_vao = self.ctx.vertex_array(
            self._programs["line"],
            [(line_vbo, '2f', 'in_pos')],
        )
        
        # Instance buffers
        self._rect_instance_buf = self.ctx.buffer(reserve=self._instance_buf_size * 8 * 4)
        self._line_instance_buf = self.ctx.buffer(reserve=self._instance_buf_size * 9 * 4)
        
        # Font system
        self._font_manager = FontManager(self.ctx)
        self._text_renderer = TextRenderer(self.ctx, self._programs["text"], self._font_manager)
    
    def _load_program(self, shader_dir: Path, name: str) -> moderngl.Program:
        vert_path = shader_dir / f"{name}.vert"
        frag_path = shader_dir / f"{name}.frag"
        
        with open(vert_path) as f:
            vert_src = f.read()
        with open(frag_path) as f:
            frag_src = f.read()
        
        try:
            return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        except moderngl.Error as e:
            raise ShaderCompilationError(f"Failed to compile {name}: {e}") from e
    
    def set_viewport_size(self, width: int, height: int):
        self._viewport_size = (width, height)
    
    def execute(self, batch: DrawBatch):
        """Execute all commands in a batch."""
        for cmd in batch.commands:
            self._dispatch(cmd)
    
    def _dispatch(self, cmd: Command2D):
        """Route command to handler."""
        handlers = {
            Cmd2DSetClip: self._exec_clip,
            Cmd2DRect: self._exec_rect,
            Cmd2DRectsInstanced: self._exec_rects_instanced,
            Cmd2DLine: self._exec_line,
            Cmd2DLinesInstanced: self._exec_lines_instanced,
            Cmd2DCircle: self._exec_circle,
            Cmd2DArc: self._exec_arc,
            Cmd2DText: self._exec_text,
            Cmd2DWaveform: self._exec_waveform,
        }
        
        handler = handlers.get(type(cmd))
        if handler:
            handler(cmd)
        else:
            raise ValueError(f"Unknown command: {type(cmd)}")
    
    def _use_program(self, name: str):
        """Switch shader program if needed."""
        prog = self._programs[name]
        if prog != self._current_program:
            self._current_program = prog
            # Set common uniforms
            if "u_resolution" in prog:
                prog["u_resolution"].value = self._viewport_size
    
    def _exec_clip(self, cmd: Cmd2DSetClip):
        if cmd.enabled:
            # Convert to GL coords (bottom-left origin)
            gl_y = self._viewport_size[1] - cmd.y - cmd.h
            self.ctx.scissor = (cmd.x, gl_y, cmd.w, cmd.h)
            self._clip_stack.append((cmd.x, cmd.y, cmd.w, cmd.h))
        else:
            if self._clip_stack:
                self._clip_stack.pop()
            if self._clip_stack:
                x, y, w, h = self._clip_stack[-1]
                gl_y = self._viewport_size[1] - y - h
                self.ctx.scissor = (x, gl_y, w, h)
            else:
                self.ctx.scissor = None
    
    def _exec_rect(self, cmd: Cmd2DRect):
        self._use_program("rect")
        prog = self._programs["rect"]
        
        prog["u_rect"].value = (cmd.x, cmd.y, cmd.w, cmd.h)
        prog["u_color"].value = cmd.color
        prog["u_corner_radius"].value = cmd.corner_radius
        prog["u_border_width"].value = cmd.border_width
        if cmd.border_color:
            prog["u_border_color"].value = cmd.border_color
        else:
            prog["u_border_color"].value = (0, 0, 0, 0)
        
        self._unit_quad_vao.render()
    
    def _exec_rects_instanced(self, cmd: Cmd2DRectsInstanced):
        self._use_program("rect_instanced")
        prog = self._programs["rect_instanced"]
        
        # Grow buffer if needed
        required = cmd.instance_count * 8 * 4  # 8 floats per instance
        if self._rect_instance_buf.size < required:
            self._rect_instance_buf.orphan(max(required, self._rect_instance_buf.size * 2))
        
        # Upload instance data
        self._rect_instance_buf.write(cmd.instance_data)
        
        prog["u_corner_radius"].value = cmd.corner_radius
        
        # Create VAO with instance buffer
        vao = self.ctx.vertex_array(
            prog,
            [
                (self._unit_quad_vao.vertex_buffer, '2f', 'in_pos'),
                (self._rect_instance_buf, '4f 4f /i', 'in_rect', 'in_color'),
            ],
        )
        vao.render(instances=cmd.instance_count)
    
    def _exec_line(self, cmd: Cmd2DLine):
        self._use_program("line")
        prog = self._programs["line"]
        
        prog["u_line"].value = (cmd.x0, cmd.y0, cmd.x1, cmd.y1)
        prog["u_color"].value = cmd.color
        prog["u_width"].value = cmd.width
        
        self._unit_line_vao.render()
    
    def _exec_lines_instanced(self, cmd: Cmd2DLinesInstanced):
        self._use_program("line_instanced")
        prog = self._programs["line_instanced"]
        
        required = cmd.instance_count * 9 * 4
        if self._line_instance_buf.size < required:
            self._line_instance_buf.orphan(max(required, self._line_instance_buf.size * 2))
        
        self._line_instance_buf.write(cmd.instance_data)
        
        vao = self.ctx.vertex_array(
            prog,
            [
                (self._unit_line_vao.vertex_buffer, '2f', 'in_pos'),
                (self._line_instance_buf, '4f 4f 1f /i', 'in_line', 'in_color', 'in_width'),
            ],
        )
        vao.render(instances=cmd.instance_count)
    
    def _exec_circle(self, cmd: Cmd2DCircle):
        self._use_program("circle")
        prog = self._programs["circle"]
        
        prog["u_center"].value = (cmd.cx, cmd.cy)
        prog["u_radius"].value = cmd.radius
        prog["u_fill_color"].value = cmd.fill_color
        prog["u_stroke_color"].value = cmd.stroke_color or (0,0,0,0)
        prog["u_stroke_width"].value = cmd.stroke_width
        
        # Render a quad that covers the circle bounds
        self._unit_quad_vao.render()
    
    def _exec_arc(self, cmd: Cmd2DArc):
        # Similar to circle but with angle uniforms
        pass
    
    def _exec_text(self, cmd: Cmd2DText):
        self._text_renderer.draw(
            cmd.text, cmd.x, cmd.y, cmd.color,
            font_size=cmd.font_size, align=cmd.align
        )
    
    def _exec_waveform(self, cmd: Cmd2DWaveform):
        self._use_program("waveform")
        prog = self._programs["waveform"]
        
        # Upload envelope to 1D texture
        # ... implementation
        pass
```

---

## 8. Entity-Component Model

### 8.1 Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENTITY-COMPONENT SYSTEM                               │
│                                                                              │
│  ┌─────────────────┐                                                        │
│  │     Entity      │                                                        │
│  │ ─────────────── │                                                        │
│  │ id: int         │                                                        │
│  │ region: str     │◀── "sync", "waveform", "lanes"                        │
│  │ beat: float     │◀── Position in beat-space (for scrolling)             │
│  │ depth: float    │◀── Z-order within region                              │
│  │ visible: bool   │                                                        │
│  │ components: {}  │───────────────────────────────────────┐               │
│  └─────────────────┘                                       │               │
│                                                             │               │
│                    ┌────────────────────────────────────────┘               │
│                    │                                                         │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         COMPONENTS                                       ││
│  │                                                                          ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               ││
│  │  │  Transform    │  │   Renderable  │  │   Selectable  │               ││
│  │  │ ───────────── │  │ ───────────── │  │ ───────────── │               ││
│  │  │ x, y          │  │ color         │  │ selected      │               ││
│  │  │ rotation      │  │ shape         │  │ hovered       │               ││
│  │  │ scale_x, y    │  │ corner_radius │  │ on_select()   │               ││
│  │  │ alpha         │  │               │  │               │               ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘               ││
│  │                                                                          ││
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               ││
│  │  │  Duration     │  │   Lane        │  │   Velocity    │               ││
│  │  │ ───────────── │  │ ───────────── │  │ ───────────── │               ││
│  │  │ beats         │  │ index         │  │ value (0-127) │               ││
│  │  │               │  │               │  │               │               ││
│  │  └───────────────┘  └───────────────┘  └───────────────┘               ││
│  │                                                                          ││
│  │  ┌───────────────┐  ┌───────────────┐                                   ││
│  │  │   Animated    │  │   HitMarker   │                                   ││
│  │  │ ───────────── │  │ ───────────── │                                   ││
│  │  │ timeline      │  │ confidence    │                                   ││
│  │  │ keyframes     │  │ type          │                                   ││
│  │  └───────────────┘  └───────────────┘                                   ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Implementation

```python
from typing import Dict, Any, Optional, Type, TypeVar

T = TypeVar('T', bound='Component')


class Component:
    """Base class for all components."""
    pass


@dataclass
class Transform(Component):
    """Position, rotation, scale."""
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0
    alpha: float = 1.0


@dataclass
class Renderable(Component):
    """Visual appearance."""
    color: Color4 = (0.5, 0.5, 0.5, 1.0)
    shape: str = "rect"  # "rect", "circle", "triangle", "line"
    corner_radius: float = 0.0
    border_color: Optional[Color4] = None
    border_width: float = 0.0


@dataclass
class Selectable(Component):
    """Selection state and callbacks."""
    selected: bool = False
    hovered: bool = False
    on_select: Optional[Callable[["Entity"], None]] = None
    on_deselect: Optional[Callable[["Entity"], None]] = None


@dataclass
class Duration(Component):
    """Time extent (for note-like entities)."""
    beats: float = 1.0


@dataclass
class Lane(Component):
    """Lane assignment (for sequencer)."""
    index: int = 0


@dataclass
class Velocity(Component):
    """MIDI-style velocity."""
    value: int = 100  # 0-127


@dataclass
class Animated(Component):
    """Has keyframe animation."""
    timeline: Optional[Timeline] = None


@dataclass  
class HitMarker(Component):
    """Transient/onset marker."""
    confidence: float = 1.0
    marker_type: str = "transient"  # "transient", "beat", "downbeat", "phrase"


class Entity:
    """Game object with components."""
    
    _next_id: int = 0
    
    def __init__(self, region: str, beat: float = 0.0):
        Entity._next_id += 1
        self.id = Entity._next_id
        self.region = region
        self.beat = beat
        self.depth = 0.0
        self.visible = True
        self._components: Dict[Type[Component], Component] = {}
    
    def add(self, component: Component) -> "Entity":
        """Add component, return self for chaining."""
        self._components[type(component)] = component
        return self
    
    def get(self, component_type: Type[T]) -> Optional[T]:
        """Get component by type."""
        return self._components.get(component_type)
    
    def has(self, component_type: Type[Component]) -> bool:
        """Check if entity has component."""
        return component_type in self._components
    
    def remove(self, component_type: Type[Component]):
        """Remove component."""
        self._components.pop(component_type, None)


# Factory functions for common entity types

def create_note(beat: float, duration: float, lane: int, velocity: int = 100) -> Entity:
    """Create a sequencer note entity."""
    return (Entity("lanes", beat)
        .add(Transform())
        .add(Duration(beats=duration))
        .add(Lane(index=lane))
        .add(Velocity(value=velocity))
        .add(Renderable(
            color=velocity_to_color(velocity),
            shape="rect",
            corner_radius=3.0
        ))
        .add(Selectable()))


def create_hit_marker(beat: float, confidence: float = 1.0) -> Entity:
    """Create a transient/hit marker."""
    return (Entity("waveform", beat)
        .add(HitMarker(confidence=confidence))
        .add(Renderable(
            color=(1.0, 0.8, 0.2, confidence),
            shape="line"
        )))


def create_phase_indicator(phase: float) -> Entity:
    """Create sync circle phase indicator."""
    return (Entity("sync", 0)
        .add(Transform(rotation=phase * 2 * math.pi))
        .add(Animated())
        .add(Renderable(
            color=(0.3, 0.8, 0.4, 1.0),
            shape="arc"
        )))
```

---

## 9. Text Rendering System

### 9.1 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEXT RENDERING SYSTEM                                │
│                                                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │ FontManager │─────▶│  FontAtlas  │─────▶│  Texture    │                  │
│  │             │      │  (per size) │      │  (on GPU)   │                  │
│  └─────────────┘      └──────┬──────┘      └─────────────┘                  │
│                              │                                               │
│                              │ contains                                      │
│                              ▼                                               │
│                       ┌─────────────┐                                        │
│                       │ GlyphInfo[] │                                        │
│                       │ (metrics)   │                                        │
│                       └──────┬──────┘                                        │
│                              │                                               │
│                              │ used by                                       │
│                              ▼                                               │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                  │
│  │  Cmd2DText  │─────▶│ TextLayout  │─────▶│ TextRenderer│                  │
│  │  (command)  │      │ (positions) │      │ (draws)     │                  │
│  └─────────────┘      └─────────────┘      └─────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Implementation

```python
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import freetype  # pip install freetype-py
import numpy as np


@dataclass
class GlyphInfo:
    """Metrics for a single glyph."""
    char: str
    # Position in atlas
    atlas_x: int
    atlas_y: int
    # Size
    width: int
    height: int
    # Offsets from cursor position
    bearing_x: int
    bearing_y: int
    # How far to advance cursor after this glyph
    advance: int


class FontAtlas:
    """Packed texture atlas containing glyphs."""
    
    def __init__(self, ctx: moderngl.Context, font_path: str, font_size: int):
        self.ctx = ctx
        self.font_size = font_size
        self.glyphs: Dict[str, GlyphInfo] = {}
        self.texture: moderngl.Texture = None
        
        self._build(font_path)
    
    def _build(self, font_path: str):
        """Load font and pack glyphs into atlas."""
        
        face = freetype.Face(font_path)
        face.set_pixel_sizes(0, self.font_size)
        
        # Characters to include
        chars = "".join([
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            "0123456789",
            "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
            " ",
        ])
        
        # First pass: calculate atlas size
        total_width = 0
        max_height = 0
        for char in chars:
            face.load_char(char, freetype.FT_LOAD_RENDER)
            glyph = face.glyph
            total_width += glyph.bitmap.width + 2  # +2 for padding
            max_height = max(max_height, glyph.bitmap.rows)
        
        # Calculate atlas dimensions (roughly square)
        atlas_size = int(np.ceil(np.sqrt(total_width * max_height)))
        atlas_size = max(256, 1 << int(np.ceil(np.log2(atlas_size))))  # Power of 2
        
        # Create atlas buffer
        atlas = np.zeros((atlas_size, atlas_size), dtype=np.uint8)
        
        # Second pass: pack glyphs
        cursor_x = 0
        cursor_y = 0
        row_height = 0
        
        for char in chars:
            face.load_char(char, freetype.FT_LOAD_RENDER)
            glyph = face.glyph
            bitmap = glyph.bitmap
            
            # Move to next row if needed
            if cursor_x + bitmap.width + 2 > atlas_size:
                cursor_x = 0
                cursor_y += row_height + 2
                row_height = 0
            
            # Copy glyph to atlas
            if bitmap.width > 0 and bitmap.rows > 0:
                buffer = np.array(bitmap.buffer, dtype=np.uint8).reshape(
                    bitmap.rows, bitmap.width
                )
                atlas[cursor_y:cursor_y+bitmap.rows, 
                      cursor_x:cursor_x+bitmap.width] = buffer
            
            # Store glyph info
            self.glyphs[char] = GlyphInfo(
                char=char,
                atlas_x=cursor_x,
                atlas_y=cursor_y,
                width=bitmap.width,
                height=bitmap.rows,
                bearing_x=glyph.bitmap_left,
                bearing_y=glyph.bitmap_top,
                advance=glyph.advance.x >> 6,  # Convert from 26.6 fixed point
            )
            
            cursor_x += bitmap.width + 2
            row_height = max(row_height, bitmap.rows)
        
        # Upload to GPU
        self.atlas_size = atlas_size
        self.texture = self.ctx.texture((atlas_size, atlas_size), 1, atlas.tobytes())
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)


class FontManager:
    """Manages font loading and caching."""
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._atlases: Dict[Tuple[str, int], FontAtlas] = {}
        self._default_font_path = self._find_default_font()
    
    def _find_default_font(self) -> str:
        """Find a suitable default font."""
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "assets/fonts/FiraMono-Regular.ttf",
        ]
        for path in candidates:
            if Path(path).exists():
                return path
        raise FileNotFoundError("No suitable font found")
    
    def get_atlas(self, font_path: Optional[str] = None, font_size: int = 14) -> FontAtlas:
        """Get or create font atlas."""
        path = font_path or self._default_font_path
        key = (path, font_size)
        
        if key not in self._atlases:
            self._atlases[key] = FontAtlas(self.ctx, path, font_size)
        
        return self._atlases[key]


@dataclass
class PositionedGlyph:
    """Glyph with its screen position."""
    glyph: GlyphInfo
    x: float
    y: float


class TextLayout:
    """Layouts text string into positioned glyphs."""
    
    @staticmethod
    def layout(text: str, atlas: FontAtlas, x: float, y: float,
               align: str = "left", max_width: Optional[float] = None) -> List[PositionedGlyph]:
        """Convert text to positioned glyphs."""
        
        result = []
        cursor_x = x
        
        for char in text:
            if char not in atlas.glyphs:
                char = "?"  # Fallback
            
            glyph = atlas.glyphs.get(char)
            if glyph is None:
                continue
            
            result.append(PositionedGlyph(
                glyph=glyph,
                x=cursor_x + glyph.bearing_x,
                y=y - glyph.bearing_y,  # Y is flipped
            ))
            
            cursor_x += glyph.advance
        
        # Handle alignment
        if align != "left" and result:
            total_width = cursor_x - x
            if align == "center":
                offset = -total_width / 2
            elif align == "right":
                offset = -total_width
            else:
                offset = 0
            
            for pg in result:
                pg.x += offset
        
        return result
    
    @staticmethod
    def measure(text: str, atlas: FontAtlas) -> Tuple[float, float]:
        """Measure text dimensions."""
        width = sum(atlas.glyphs.get(c, atlas.glyphs.get("?")).advance for c in text)
        height = atlas.font_size
        return (width, height)


class TextRenderer:
    """Renders text using instanced quads."""
    
    MAX_CHARS = 1000
    
    def __init__(self, ctx: moderngl.Context, program: moderngl.Program, 
                 font_manager: FontManager):
        self.ctx = ctx
        self.program = program
        self.font_manager = font_manager
        
        # Instance buffer: [x, y, w, h, u0, v0, u1, v1] per char
        self._instance_buf = ctx.buffer(reserve=self.MAX_CHARS * 8 * 4)
        
        # Unit quad
        quad = np.array([0,0, 1,0, 1,1, 0,0, 1,1, 0,1], dtype='f4')
        self._quad_buf = ctx.buffer(quad)
    
    def draw(self, text: str, x: float, y: float, color: Color4,
             font_size: int = 14, align: str = "left"):
        """Draw text string."""
        
        atlas = self.font_manager.get_atlas(font_size=font_size)
        glyphs = TextLayout.layout(text, atlas, x, y, align)
        
        if not glyphs:
            return
        
        # Build instance data
        atlas_size = atlas.atlas_size
        data = np.zeros((len(glyphs), 8), dtype=np.float32)
        
        for i, pg in enumerate(glyphs):
            g = pg.glyph
            # Screen position and size
            data[i, 0] = pg.x
            data[i, 1] = pg.y
            data[i, 2] = g.width
            data[i, 3] = g.height
            # UV coordinates
            data[i, 4] = g.atlas_x / atlas_size
            data[i, 5] = g.atlas_y / atlas_size
            data[i, 6] = (g.atlas_x + g.width) / atlas_size
            data[i, 7] = (g.atlas_y + g.height) / atlas_size
        
        # Upload
        self._instance_buf.write(data.tobytes())
        
        # Bind texture
        atlas.texture.use(0)
        self.program["u_font_atlas"].value = 0
        self.program["u_color"].value = color
        
        # Draw
        vao = self.ctx.vertex_array(
            self.program,
            [
                (self._quad_buf, '2f', 'in_pos'),
                (self._instance_buf, '4f 4f /i', 'in_rect', 'in_uv'),
            ],
        )
        vao.render(instances=len(glyphs))
```

### 9.3 Text Shaders

**text.vert:**
```glsl
#version 330

in vec2 in_pos;

// Per-instance
in vec4 in_rect;  // x, y, w, h
in vec4 in_uv;    // u0, v0, u1, v1

uniform vec2 u_resolution;

out vec2 v_uv;

void main() {
    // Scale and position the quad
    vec2 pos = in_rect.xy + in_pos * in_rect.zw;
    
    // To NDC
    vec2 ndc = (pos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y;
    
    gl_Position = vec4(ndc, 0.0, 1.0);
    
    // Interpolate UV
    v_uv = mix(in_uv.xy, in_uv.zw, in_pos);
}
```

**text.frag:**
```glsl
#version 330

in vec2 v_uv;

uniform sampler2D u_font_atlas;
uniform vec4 u_color;

out vec4 frag_color;

void main() {
    float alpha = texture(u_font_atlas, v_uv).r;
    frag_color = vec4(u_color.rgb, u_color.a * alpha);
}
```

---

## 10. Complete Class Diagrams

### 10.1 Full System Class Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                              SPECTRO SYSTEM                                                      │
│                                                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                          APPLICATION LAYER                                                 │  │
│  │                                                                                                            │  │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                          │  │
│  │  │  Application │────▶│ MasterClock  │────▶│ EventBus     │────▶│ LayoutTree   │                          │  │
│  │  │              │     │              │     │              │     │   (Root)     │                          │  │
│  │  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘                          │  │
│  │                                                                         │                                  │  │
│  └─────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┘  │
│                                                                            │                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┼──────────────────────────────────┐  │
│  │                                          LAYOUT LAYER                   │                                  │  │
│  │                                                                         ▼                                  │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │                                    LayoutNode (linked list tree)                                      │ │  │
│  │  │                                                                                                       │ │  │
│  │  │  ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────────────────────────────────────┐  │ │  │
│  │  │  │ Container │────▶│ Container │────▶│ Splitter  │────▶│               Panel3D                     │  │ │  │
│  │  │  │  (root)   │     │ (header)  │     │           │     │  ┌─────────────────────────────────────┐  │  │ │  │
│  │  │  └───────────┘     └───────────┘     └───────────┘     │  │ Regions:                            │  │  │ │  │
│  │  │                                                         │  │  - SyncCirclesRegion                │  │  │ │  │
│  │  │                                                         │  │  - WaveformRegion                   │  │  │ │  │
│  │  │                                                         │  │  - LanesRegion                      │  │  │ │  │
│  │  │                                                         │  └─────────────────────────────────────┘  │  │ │  │
│  │  │                                                         │  ┌─────────────────────────────────────┐  │  │ │  │
│  │  │                                                         │  │ TimeCamera (shared)                 │  │  │ │  │
│  │  │                                                         │  │ Timeline (local)                    │  │  │ │  │
│  │  │                                                         │  │ Entities[]                          │  │  │ │  │
│  │  │                                                         │  └─────────────────────────────────────┘  │  │ │  │
│  │  │                                                         └───────────────────────────────────────────┘  │ │  │
│  │  └──────────────────────────────────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                                                            │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                          RENDER LAYER                                                      │  │
│  │                                                                                                            │  │
│  │  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │  │
│  │  │ DrawContext  │────▶│  DrawBatch   │────▶│RenderScheduler────▶│ UIRenderer2D │────▶│  Framebuffer │     │  │
│  │  │              │     │  (commands)  │     │  (optimize)  │     │  (execute)   │     │              │     │  │
│  │  └──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘     └──────────────┘     │  │
│  │                                                                         │                                  │  │
│  │                                                         ┌───────────────┼───────────────┐                  │  │
│  │                                                         │               │               │                  │  │
│  │                                                         ▼               ▼               ▼                  │  │
│  │                                                  ┌────────────┐ ┌────────────┐ ┌────────────┐              │  │
│  │                                                  │  Shaders   │ │  Buffers   │ │TextRenderer│              │  │
│  │                                                  │ (programs) │ │ (instance) │ │ (glyphs)   │              │  │
│  │                                                  └────────────┘ └────────────┘ └────────────┘              │  │
│  │                                                                                                            │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                                          ENTITY LAYER                                                      │  │
│  │                                                                                                            │  │
│  │  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │  │
│  │  │  Entity                                                                                               │ │  │
│  │  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │ │  │
│  │  │  │ Transform │ │Renderable │ │Selectable │ │ Duration  │ │   Lane    │ │ Velocity  │ │ Animated  │   │ │  │
│  │  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘ └───────────┘   │ │  │
│  │  └──────────────────────────────────────────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                                                            │  │
│  └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary: What This Document Provides

1. **Branch strategy** — Three focused branches that can be developed somewhat independently
2. **Layout tree** — Linked-list traversal for efficient event dispatch
3. **Event cascade** — Capture → Target → Bubble, with clear scheduling of renders
4. **Single 3D panel** — All regions share TimeCamera, render as one batch
5. **Two kinds of movement** — Scrolling (camera) vs Animation (entity properties)
6. **Timeline sync** — Master clock drives per-panel timelines
7. **Rendering pipeline** — Collect → Sort/Batch → Execute, with instancing
8. **Entity-Component model** — Flexible composition for different entity types
9. **Text rendering** — Complete font atlas → layout → render system
10. **Complete class diagrams** — Visual reference for the entire system

This should provide the clarity needed to organize the project and implement each piece.

# SPECTRO Architecture v2 — Corrected Design

> **Key Corrections:**
> 1. Render pipeline connects to PanelManager (not separate tree branch)
> 2. Transport controls live IN the sync panel (same region)
> 3. Clearer ownership hierarchy

---

## 1. Corrected System Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                    APPLICATION                                           │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │                              LayoutRoot                                              ││
│  │                                  │                                                   ││
│  │                    ┌─────────────┴─────────────┐                                    ││
│  │                    │                           │                                    ││
│  │                    ▼                           ▼                                    ││
│  │           ┌─────────────────┐         ┌─────────────────┐                          ││
│  │           │   SidePanel     │         │   PanelManager  │◀══════════════════════╗  ││
│  │           │  (tools, etc)   │         │                 │                       ║  ││
│  │           └─────────────────┘         │  ┌───────────┐  │    ┌────────────────┐ ║  ││
│  │                                       │  │  Panel3D  │──┼───▶│ RenderScheduler│ ║  ││
│  │                                       │  │           │  │    │                │ ║  ││
│  │                                       │  └───────────┘  │    │  ┌──────────┐  │ ║  ││
│  │                                       │                 │    │  │DrawBatch │  │ ║  ││
│  │                                       │  owns:          │    │  └────┬─────┘  │ ║  ││
│  │                                       │  - TimeCamera   │    │       │        │ ║  ││
│  │                                       │  - MasterClock  │    │       ▼        │ ║  ││
│  │                                       │  - Entities     │    │  ┌──────────┐  │ ║  ││
│  │                                       │                 │    │  │UIRenderer│  │ ║  ││
│  │                                       └────────┬────────┘    │  │    2D    │  │ ║  ││
│  │                                                │             │  └──────────┘  │ ║  ││
│  │                                                │             └────────────────┘ ║  ││
│  │                                                │                                ║  ││
│  │                                                ▼                                ║  ││
│  │                              PanelManager.render_frame()════════════════════════╝  ││
│  │                                                                                    ││
│  └────────────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key insight**: `PanelManager` owns the `RenderScheduler`. When the panel collects draw commands, it hands them directly to its manager's renderer. No tree traversal to find the renderer.

---

## 2. Revised Panel Structure — Transport IN Sync Region

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                     Panel3D                                              │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  REGION 0: SyncRegion (includes transport)                                         │ │
│  │  ══════════════════════════════════════════                                        │ │
│  │                                                                                     │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────────────────────────┐│ │
│  │  │ Transport       │  │  Beat Phase     │  │  Bar Phase                           ││ │
│  │  │ Controls        │  │  Circle         │  │  Circle                              ││ │
│  │  │                 │  │                 │  │                                      ││ │
│  │  │ [▶] [■] [⏮]    │  │    ╭───╮        │  │         ╭───╮                        ││ │
│  │  │                 │  │   ╱  ●  ╲       │  │        ╱     ╲                       ││ │
│  │  │ BPM: 120.0      │  │  │   │   │      │  │       │   ●   │                      ││ │
│  │  │ 4/4             │  │   ╲     ╱       │  │        ╲     ╱                       ││ │
│  │  │                 │  │    ╰───╯        │  │         ╰───╯                        ││ │
│  │  │ Bar: 4          │  │                 │  │                                      ││ │
│  │  │ Beat: 2         │  │  Beat: 2        │  │  Phase: 0.37                         ││ │
│  │  └─────────────────┘  └─────────────────┘  └──────────────────────────────────────┘│ │
│  │                                                                                     │ │
│  │  Does NOT scroll — uses TransportState directly                                    │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  REGION 1: WaveformRegion                                                          │ │
│  │  ════════════════════════                                                          │ │
│  │                                                                                     │ │
│  │  │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │     │ │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█████▓▓▓▓▓▓▓▓▓▓▓▓▓████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓███████████████▓▓▓▓▓██████████████████▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ │
│  │  │    │    │    │    ▼    │    │    │    │    │    │    │    │    │    │    │     │ │
│  │  4    5    6    7  PLAYHEAD 9   10   11   12   13   14   15   16   17   18   19    │ │
│  │                                                                                     │ │
│  │  Scrolls with TimeCamera                                                           │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐ │
│  │  REGION 2: LanesRegion (3D viewport)                                               │ │
│  │  ═══════════════════════════════════                                               │ │
│  │                                                                                     │ │
│  │  Lane 0 │░░░░░░░░░│████████│░░░░░░░░░░░░│██████│░░░░░░░░░│████│░░░░░░░░│          │ │
│  │  Lane 1 │░░░│████████████│░░░░░░│████│░░░░░░░░░░░│██████████│░░░░░░░│             │ │
│  │  Lane 2 │░░░░░░░░░░░░│████│░░░░░░░░░│████████████│░░░░░░│████│░░░░░░│             │ │
│  │  Lane 3 │██████│░░░░░░░░░░░│████████│░░░░░░░░░│██████████████│░░░░░░│             │ │
│  │  ...    │    │    │    ▼    │    │    │    │    │    │    │    │    │             │ │
│  │         4    5    6  PLAYHEAD 8   9    10   11   12   13   14   15   16            │ │
│  │                                                                                     │ │
│  │  Scrolls with TimeCamera (shared with WaveformRegion)                              │ │
│  └────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Ownership and Connection Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 OWNERSHIP HIERARCHY                                      │
│                                                                                          │
│                              ┌──────────────────┐                                        │
│                              │   Application    │                                        │
│                              │                  │                                        │
│                              │  - window        │                                        │
│                              │  - gl_context    │                                        │
│                              └────────┬─────────┘                                        │
│                                       │ owns                                             │
│                                       ▼                                                  │
│                              ┌──────────────────┐                                        │
│                              │   PanelManager   │                                        │
│                              │                  │                                        │
│                              │  - layout_root   │                                        │
│                              │  - renderer ─────┼───────────────────┐                   │
│                              │  - master_clock  │                   │                   │
│                              │  - time_camera ──┼─────────┐         │                   │
│                              └────────┬─────────┘         │         │                   │
│                                       │ owns              │ shared  │ owned             │
│                                       ▼                   │         │                   │
│                              ┌──────────────────┐         │         ▼                   │
│                              │     Panel3D      │         │  ┌──────────────────┐       │
│                              │                  │         │  │  RenderScheduler │       │
│                              │  - sync_region ──┼─────┐   │  │                  │       │
│                              │  - wave_region ──┼───┐ │   │  │  - ui_renderer   │       │
│                              │  - lanes_region ─┼─┐ │ │   │  │  - batch_queue   │       │
│                              │  - entities[]    │ │ │ │   │  └──────────────────┘       │
│                              │  - transport ────┼─┼─┼─┼───┘                             │
│                              └──────────────────┘ │ │ │                                 │
│                                                   │ │ │                                 │
│                    ┌──────────────────────────────┘ │ │                                 │
│                    │  ┌─────────────────────────────┘ │                                 │
│                    │  │  ┌────────────────────────────┘                                 │
│                    ▼  ▼  ▼                                                              │
│              ┌─────────────────────────────────────────────────────────────────────┐    │
│              │                         REGIONS                                      │    │
│              │                                                                      │    │
│              │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐            │    │
│              │  │  SyncRegion   │  │ WaveformRegion│  │  LanesRegion  │            │    │
│              │  │               │  │               │  │               │            │    │
│              │  │ - transport ◀─┼──┼───────────────┼──┼─ NO SCROLL    │            │    │
│              │  │ - circles     │  │ - waveform    │  │ - lanes[]     │            │    │
│              │  │ - bpm_display │  │ - hit_markers │  │ - events[]    │            │    │
│              │  │               │  │               │  │               │            │    │
│              │  │ reads:        │  │ reads:        │  │ reads:        │            │    │
│              │  │ TransportState│  │ TimeCamera ◀──┼──┼─ TimeCamera   │            │    │
│              │  └───────────────┘  └───────────────┘  └───────────────┘            │    │
│              │                            ▲                   ▲                     │    │
│              │                            │                   │                     │    │
│              │                            └─────────┬─────────┘                     │    │
│              │                                      │                               │    │
│              │                              SHARED TimeCamera                       │    │
│              │                              (perfect sync)                          │    │
│              └─────────────────────────────────────────────────────────────────────┘    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow — Frame Loop

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              FRAME LOOP (60 FPS)                                         │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │  STEP 1: INPUT                                                                       ││
│  │  ─────────────────                                                                   ││
│  │                                                                                      ││
│  │  PanelManager.poll_input()                                                          ││
│  │       │                                                                              ││
│  │       ├──▶ Keyboard: Space pressed?  ──▶ transport.toggle_play()                    ││
│  │       ├──▶ Mouse: in SyncRegion?     ──▶ transport button clicks                    ││
│  │       ├──▶ Mouse: in WaveformRegion? ──▶ time_camera.begin_drag() / seek           ││
│  │       └──▶ Mouse: in LanesRegion?    ──▶ entity selection / drag                    ││
│  │                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                       │                                                  │
│                                       ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │  STEP 2: UPDATE                                                                      ││
│  │  ──────────────────                                                                  ││
│  │                                                                                      ││
│  │  PanelManager.update(dt)                                                            ││
│  │       │                                                                              ││
│  │       ├──▶ master_clock.tick(dt)                                                    ││
│  │       │         │                                                                    ││
│  │       │         └──▶ Returns TransportState (immutable snapshot)                    ││
│  │       │                    │                                                         ││
│  │       │                    ▼                                                         ││
│  │       ├──▶ time_camera.update(transport_state.playhead_beat)                        ││
│  │       │                                                                              ││
│  │       └──▶ panel.update(transport_state)                                            ││
│  │                 │                                                                    ││
│  │                 ├──▶ sync_region.update(transport_state)                            ││
│  │                 ├──▶ wave_region.update(transport_state, time_camera)               ││
│  │                 └──▶ lanes_region.update(transport_state, time_camera)              ││
│  │                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                       │                                                  │
│                                       ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │  STEP 3: COLLECT DRAW COMMANDS                                                       ││
│  │  ─────────────────────────────────                                                   ││
│  │                                                                                      ││
│  │  PanelManager.collect_commands()                                                    ││
│  │       │                                                                              ││
│  │       └──▶ panel.collect_draw_commands()                                            ││
│  │                 │                                                                    ││
│  │                 │   ┌────────────────────────────────────────────────────────────┐  ││
│  │                 │   │                    DrawBatch                                │  ││
│  │                 │   │                                                             │  ││
│  │                 ├──▶│  sync_region.draw(batch)                                   │  ││
│  │                 │   │    - transport buttons (rects)                             │  ││
│  │                 │   │    - BPM text                                              │  ││
│  │                 │   │    - phase circles (arcs)                                  │  ││
│  │                 │   │    - beat/bar indicators (text)                            │  ││
│  │                 │   │                                                             │  ││
│  │                 ├──▶│  wave_region.draw(batch, time_camera)                      │  ││
│  │                 │   │    - grid lines (instanced)                                │  ││
│  │                 │   │    - waveform (Cmd2DWaveform)                              │  ││
│  │                 │   │    - hit markers (instanced lines)                         │  ││
│  │                 │   │                                                             │  ││
│  │                 ├──▶│  lanes_region.draw(batch, time_camera)                     │  ││
│  │                 │   │    - lane backgrounds (rects)                              │  ││
│  │                 │   │    - grid lines (instanced)                                │  ││
│  │                 │   │    - events (instanced rects)                              │  ││
│  │                 │   │                                                             │  ││
│  │                 └──▶│  playhead (line, on top)                                   │  ││
│  │                     │                                                             │  ││
│  │                     └────────────────────────────────────────────────────────────┘  ││
│  │                                         │                                            ││
│  │                                         ▼                                            ││
│  │                              renderer.submit(batch)                                  ││
│  │                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                       │                                                  │
│                                       ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐│
│  │  STEP 4: RENDER                                                                      ││
│  │  ──────────────────                                                                  ││
│  │                                                                                      ││
│  │  PanelManager.render()                                                              ││
│  │       │                                                                              ││
│  │       └──▶ renderer.execute_all()                                                   ││
│  │                 │                                                                    ││
│  │                 ├──▶ Sort batches by z_order                                        ││
│  │                 ├──▶ Optimize (convert to instanced where beneficial)               ││
│  │                 └──▶ Execute each command on GPU                                    ││
│  │                           │                                                          ││
│  │                           └──▶ Framebuffer (screen)                                 ││
│  │                                                                                      ││
│  └─────────────────────────────────────────────────────────────────────────────────────┘│
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Class Definitions — Corrected Structure

### 5.1 PanelManager (owns everything)

```python
class PanelManager:
    """
    Central coordinator. Owns the panel, clock, camera, and renderer.
    The render pipeline is directly connected here — no tree traversal needed.
    """
    
    def __init__(self, ctx: moderngl.Context, window_size: Tuple[int, int]):
        # OpenGL context
        self.ctx = ctx
        self.window_size = window_size
        
        # Time infrastructure (owned here, shared with panel)
        self.master_clock = MasterClock()
        self.time_camera = TimeCamera()
        
        # The panel (gets references to clock and camera)
        self.panel = Panel3D(
            master_clock=self.master_clock,
            time_camera=self.time_camera,
        )
        
        # Render pipeline (directly connected)
        self.renderer = RenderScheduler(ctx)
        
        # Current state
        self._transport_state: Optional[TransportState] = None
        self._frame_id: int = 0
    
    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.window_size = (width, height)
        self.panel.layout(Rect(0, 0, width, height))
        self.renderer.set_viewport_size(width, height)
    
    def poll_input(self, input_state: InputState):
        """Process input and route to appropriate handler."""
        
        # Global shortcuts
        if input_state.key_just_pressed("Space"):
            self.master_clock.toggle_play()
            return
        
        # Route to panel
        self.panel.handle_input(input_state)
    
    def update(self, dt: float):
        """Update all state for this frame."""
        self._frame_id += 1
        
        # Advance master clock
        self._transport_state = self.master_clock.tick(dt)
        
        # Update camera (follows playhead or responds to user scroll)
        self.time_camera.update(
            dt=dt,
            playhead_beat=self._transport_state.playhead_beat,
            panel_width_px=self.panel.scrollable_width,
            frame_id=self._frame_id,
        )
        
        # Update panel
        self.panel.update(self._transport_state)
    
    def render(self):
        """Collect and execute draw commands."""
        
        # Collect from panel
        batch = self.panel.collect_draw_commands(self._transport_state, self.time_camera)
        
        # Execute (render pipeline is right here, no indirection)
        self.renderer.execute(batch)
    
    def frame(self, dt: float, input_state: InputState):
        """Complete frame: input → update → render."""
        self.poll_input(input_state)
        self.update(dt)
        self.render()
```

### 5.2 Panel3D (single panel with regions)

```python
class Panel3D:
    """
    The main visualization panel containing all regions.
    Does NOT own TimeCamera or MasterClock — receives references.
    """
    
    def __init__(self, master_clock: MasterClock, time_camera: TimeCamera):
        # References (not owned)
        self.master_clock = master_clock
        self.time_camera = time_camera
        
        # Layout
        self.rect = Rect(0, 0, 800, 600)
        
        # Regions
        self.sync_region = SyncRegion(
            height_ratio=0.15,
            transport=Transport(master_clock),  # Transport lives here
        )
        self.wave_region = WaveformRegion(height_ratio=0.25)
        self.lanes_region = LanesRegion(height_ratio=0.60)
        
        self.regions = [self.sync_region, self.wave_region, self.lanes_region]
        
        # Entities (managed by panel, drawn by regions)
        self.entities: List[Entity] = []
        
        # Grid builder (shared)
        self.grid_builder = GridBuilder()
    
    @property
    def scrollable_width(self) -> float:
        """Width available for scrolling content (wave + lanes)."""
        return self.rect.w
    
    def layout(self, rect: Rect):
        """Divide space among regions."""
        self.rect = rect
        y = rect.y
        
        for region in self.regions:
            h = rect.h * region.height_ratio
            region.layout(Rect(rect.x, y, rect.w, h))
            y += h
    
    def handle_input(self, input_state: InputState):
        """Route input to appropriate region."""
        
        mx, my = input_state.mouse_pos
        
        for region in self.regions:
            if region.rect.contains(mx, my):
                region.handle_input(input_state, self)
                return
    
    def update(self, transport_state: TransportState):
        """Update all regions."""
        for region in self.regions:
            region.update(transport_state, self.time_camera)
    
    def collect_draw_commands(self, transport_state: TransportState, 
                               time_camera: TimeCamera) -> DrawBatch:
        """Generate all draw commands for this panel."""
        
        batch = DrawBatch()
        
        # Background
        batch.rect(self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                   color=(0.08, 0.08, 0.10, 1.0))
        
        # Each region draws
        for region in self.regions:
            region.draw(batch, transport_state, time_camera, self)
        
        # Playhead (spans wave and lanes regions, not sync)
        if transport_state is not None:
            playhead_x = time_camera.beat_to_px(transport_state.playhead_beat)
            playhead_top = self.wave_region.rect.y
            playhead_bottom = self.lanes_region.rect.y + self.lanes_region.rect.h
            
            batch.line(
                self.rect.x + playhead_x, playhead_top,
                self.rect.x + playhead_x, playhead_bottom,
                color=(1.0, 0.3, 0.3, 1.0),
                width=2.0
            )
        
        return batch
```

### 5.3 SyncRegion (with Transport Controls)

```python
class SyncRegion(PanelRegion):
    """
    Contains transport controls AND phase circles.
    Does NOT scroll — reads TransportState directly.
    """
    
    def __init__(self, height_ratio: float, transport: Transport):
        super().__init__(height_ratio)
        self.transport = transport
        
        # Sub-components layout (relative positions within region)
        self.transport_area = Rect(0, 0, 0, 0)  # Left third
        self.beat_circle_area = Rect(0, 0, 0, 0)  # Middle
        self.bar_circle_area = Rect(0, 0, 0, 0)  # Right
        
        # Interactive state
        self.hovered_button: Optional[str] = None
    
    def layout(self, rect: Rect):
        self.rect = rect
        third = rect.w / 3
        
        self.transport_area = Rect(rect.x, rect.y, third, rect.h)
        self.beat_circle_area = Rect(rect.x + third, rect.y, third, rect.h)
        self.bar_circle_area = Rect(rect.x + 2*third, rect.y, third, rect.h)
    
    def handle_input(self, input_state: InputState, panel: Panel3D):
        """Handle transport button clicks."""
        
        mx, my = input_state.mouse_pos
        
        if self.transport_area.contains(mx, my):
            # Check which button
            btn = self._hit_test_button(mx, my)
            self.hovered_button = btn
            
            if input_state.mouse_just_pressed(0) and btn:
                if btn == "play":
                    self.transport.toggle_play()
                elif btn == "stop":
                    self.transport.stop()
                elif btn == "rewind":
                    self.transport.seek_to_beat(0)
        else:
            self.hovered_button = None
    
    def _hit_test_button(self, x: float, y: float) -> Optional[str]:
        """Determine which button is at position."""
        # ... button bounds checking
        pass
    
    def update(self, transport_state: TransportState, time_camera: TimeCamera):
        """SyncRegion doesn't use TimeCamera — only TransportState."""
        pass  # State is read directly in draw()
    
    def draw(self, batch: DrawBatch, transport_state: TransportState,
             time_camera: TimeCamera, panel: Panel3D):
        """Draw transport controls and phase circles."""
        
        # Region background
        batch.rect(self.rect.x, self.rect.y, self.rect.w, self.rect.h,
                   color=(0.12, 0.12, 0.14, 1.0))
        
        # === Transport Controls ===
        self._draw_transport_controls(batch, transport_state)
        
        # === Beat Phase Circle ===
        self._draw_phase_circle(
            batch,
            center_x=self.beat_circle_area.x + self.beat_circle_area.w / 2,
            center_y=self.beat_circle_area.y + self.beat_circle_area.h / 2,
            radius=min(self.beat_circle_area.w, self.beat_circle_area.h) * 0.35,
            phase=transport_state.phase_in_beat if transport_state else 0,
            label=f"Beat: {transport_state.current_beat_in_bar + 1}" if transport_state else "Beat: -",
            divisions=4,
        )
        
        # === Bar Phase Circle ===
        self._draw_phase_circle(
            batch,
            center_x=self.bar_circle_area.x + self.bar_circle_area.w / 2,
            center_y=self.bar_circle_area.y + self.bar_circle_area.h / 2,
            radius=min(self.bar_circle_area.w, self.bar_circle_area.h) * 0.35,
            phase=transport_state.phase_in_bar if transport_state else 0,
            label=f"Bar: {transport_state.current_bar + 1}" if transport_state else "Bar: -",
            divisions=transport_state.time_sig.numerator if transport_state else 4,
        )
    
    def _draw_transport_controls(self, batch: DrawBatch, ts: TransportState):
        """Draw play/stop/rewind buttons and BPM display."""
        
        area = self.transport_area
        button_size = min(area.w * 0.2, area.h * 0.4)
        button_y = area.y + (area.h - button_size) / 2
        spacing = button_size * 0.3
        
        # Play/Pause button
        play_x = area.x + spacing
        is_playing = ts.playing if ts else False
        play_color = (0.3, 0.8, 0.4, 1.0) if is_playing else (0.5, 0.5, 0.5, 1.0)
        batch.rect(play_x, button_y, button_size, button_size,
                   color=play_color, corner_radius=4.0)
        # Icon (triangle or pause bars)
        if is_playing:
            # Pause icon (two bars)
            bar_w = button_size * 0.15
            bar_h = button_size * 0.5
            bar_y = button_y + (button_size - bar_h) / 2
            batch.rect(play_x + button_size*0.25, bar_y, bar_w, bar_h, color=(1,1,1,1))
            batch.rect(play_x + button_size*0.55, bar_y, bar_w, bar_h, color=(1,1,1,1))
        else:
            # Play icon (triangle) - simplified as rect for now
            batch.rect(play_x + button_size*0.3, button_y + button_size*0.25,
                      button_size*0.4, button_size*0.5, color=(1,1,1,1))
        
        # Stop button
        stop_x = play_x + button_size + spacing
        batch.rect(stop_x, button_y, button_size, button_size,
                   color=(0.6, 0.3, 0.3, 1.0), corner_radius=4.0)
        batch.rect(stop_x + button_size*0.25, button_y + button_size*0.25,
                   button_size*0.5, button_size*0.5, color=(1,1,1,1))
        
        # BPM display
        bpm_text = f"BPM: {ts.bpm:.1f}" if ts else "BPM: ---"
        batch.text(bpm_text, area.x + spacing, area.y + area.h - 20,
                   color=(0.8, 0.8, 0.8, 1.0), font_size=14)
        
        # Time signature
        if ts:
            sig_text = f"{ts.time_sig.numerator}/{ts.time_sig.denominator}"
            batch.text(sig_text, area.x + spacing + 80, area.y + area.h - 20,
                       color=(0.6, 0.6, 0.6, 1.0), font_size=14)
    
    def _draw_phase_circle(self, batch: DrawBatch, center_x: float, center_y: float,
                           radius: float, phase: float, label: str, divisions: int):
        """Draw a phase indicator circle."""
        
        # Background ring
        batch.circle(center_x, center_y, radius,
                     fill_color=(0.15, 0.15, 0.18, 1.0),
                     stroke_color=(0.3, 0.3, 0.35, 1.0),
                     stroke_width=2.0)
        
        # Division tick marks
        for i in range(divisions):
            angle = (i / divisions) * 2 * math.pi - math.pi / 2
            inner_r = radius * 0.75
            outer_r = radius * 0.95
            x0 = center_x + math.cos(angle) * inner_r
            y0 = center_y + math.sin(angle) * inner_r
            x1 = center_x + math.cos(angle) * outer_r
            y1 = center_y + math.sin(angle) * outer_r
            batch.line(x0, y0, x1, y1, color=(0.5, 0.5, 0.55, 1.0), width=2.0)
        
        # Phase arc (from top, clockwise)
        end_angle = phase * 2 * math.pi - math.pi / 2
        batch.arc(center_x, center_y, radius * 0.6,
                  start_angle=-math.pi / 2,
                  end_angle=end_angle,
                  color=(0.3, 0.8, 0.5, 1.0),
                  width=6.0)
        
        # Current position dot
        dot_x = center_x + math.cos(end_angle) * radius * 0.6
        dot_y = center_y + math.sin(end_angle) * radius * 0.6
        batch.circle(dot_x, dot_y, 5.0,
                     fill_color=(1.0, 1.0, 1.0, 1.0))
        
        # Label below
        batch.text(label, center_x, center_y + radius + 15,
                   color=(0.7, 0.7, 0.7, 1.0), font_size=12, align="center")
```

---

## 6. Render Pipeline Connection

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                          RENDER PIPELINE CONNECTION                                      │
│                                                                                          │
│                                                                                          │
│     PanelManager                                                                         │
│     ════════════                                                                         │
│          │                                                                               │
│          │  owns                                                                         │
│          │                                                                               │
│          ├───────────────────────┬────────────────────────┐                             │
│          │                       │                        │                             │
│          ▼                       ▼                        ▼                             │
│    ┌───────────┐          ┌───────────┐           ┌─────────────────┐                   │
│    │  Panel3D  │          │TimeCamera │           │ RenderScheduler │                   │
│    │           │          │           │           │                 │                   │
│    │ regions[] │          │ beat↔px   │           │ ┌─────────────┐ │                   │
│    │ entities[]│          │ mapping   │           │ │ UIRenderer2D│ │                   │
│    │           │          │           │           │ │             │ │                   │
│    └─────┬─────┘          └───────────┘           │ │ - shaders   │ │                   │
│          │                      ▲                 │ │ - buffers   │ │                   │
│          │                      │                 │ │ - text      │ │                   │
│          │                      │                 │ └─────────────┘ │                   │
│          │                      │                 └────────▲────────┘                   │
│          │                      │                          │                            │
│          │    ┌─────────────────┘                          │                            │
│          │    │                                            │                            │
│          │    │  regions READ from TimeCamera              │                            │
│          │    │                                            │                            │
│          ▼    ▼                                            │                            │
│    ┌───────────────────────────────────────┐               │                            │
│    │  panel.collect_draw_commands()        │               │                            │
│    │                                        │               │                            │
│    │  for region in regions:               │               │                            │
│    │      region.draw(batch, ts, camera)   │               │                            │
│    │                                        │               │                            │
│    │  return batch ─────────────────────────┼───────────────┘                            │
│    │                                        │      DrawBatch goes                        │
│    └────────────────────────────────────────┘      directly to renderer                  │
│                                                                                          │
│                                                                                          │
│   NO TREE TRAVERSAL — Direct connection from Panel to Renderer                          │
│   Both are owned by PanelManager                                                         │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Summary of Corrections

| Issue | Before | After |
|-------|--------|-------|
| Render pipeline location | Separate tree branch | Owned by PanelManager, directly connected to Panel |
| Transport controls | Unclear location | Inside SyncRegion (same panel as phase circles) |
| TimeCamera ownership | Unclear | Owned by PanelManager, shared reference to Panel |
| MasterClock ownership | Unclear | Owned by PanelManager |
| DrawBatch routing | Tree traversal | Direct: Panel → PanelManager.renderer |
| SyncRegion scrolling | Unclear | Does NOT scroll (reads TransportState directly) |

---

## 8. Revised Branch Strategy

Given the corrected architecture:

```
main
  │
  ├── feature/core-infrastructure
  │   │
  │   │   MasterClock, TimeCamera, Transport, TransportState
  │   │   - No rendering yet
  │   │   - Unit tests for all time math
  │   │
  │   └── Deliverable: Time infrastructure that can be tested standalone
  │
  ├── feature/render-pipeline
  │   │
  │   │   UIRenderer2D, RenderScheduler, DrawBatch, Commands
  │   │   - Shader loading
  │   │   - Buffer management
  │   │   - Text rendering (FontManager, FontAtlas, TextRenderer)
  │   │
  │   └── Deliverable: Can render a DrawBatch to screen
  │
  └── feature/panel-system
      │
      │   PanelManager, Panel3D, Regions
      │   - Depends on both above branches
      │   - Integrates time + rendering
      │   - SyncRegion with transport controls
      │   - WaveformRegion + LanesRegion with scrolling
      │
      └── Deliverable: Complete working panel
```

**Merge order:**
1. `feature/core-infrastructure` → main
2. `feature/render-pipeline` → main  
3. `feature/panel-system` → main (uses both)

This ensures each piece can be developed and tested independently, but the final integration happens in `panel-system` where PanelManager brings everything together.

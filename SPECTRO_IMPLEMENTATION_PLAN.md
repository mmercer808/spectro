# SPECTRO Implementation Plan

> **Purpose**: Concrete work breakdown for Claude Code / Cursor sessions
> **Companion to**: SPECTRO_SPEC_CONSOLIDATED.md

---

## Implementation Strategy

### Approach: Vertical Slices

Rather than building all of Phase 1 before Phase 2, we build **vertical slices** — minimal working features end-to-end. This lets you see progress immediately and catch integration issues early.

**First Target**: A window that shows transport controls and a playhead moving across a grid.

---

## Branch Strategy

```
main
│
├── feature/time-core          ← Week 1: TimeCamera + Transport
│   └── Merge when: beat_to_px/px_to_beat tests pass
│
├── feature/draw-batch         ← Week 1: Commands + RenderScheduler  
│   └── Merge when: Can draw rects/lines to screen
│
├── feature/text-rendering     ← Week 2: FreeType font system
│   └── Merge when: Can render "Hello World" at any position
│
├── feature/grid-panel         ← Week 2: GridBuilder + GraphPanel
│   └── Merge when: Grid lines scroll with TimeCamera
│
├── feature/sync-region        ← Week 3: Transport controls + phase circles
│   └── Merge when: Play/pause works, circles animate
│
├── feature/waveform-region    ← Week 3-4: WaveformCache + display
│   └── Merge when: Audio file loads and displays
│
└── feature/demo-app           ← Week 4: Integration
    └── Merge when: Full demo working
```

---

## Work Packages

### WP1: Time Infrastructure (2-3 sessions)

**Goal**: TimeCamera and Transport with full test coverage.

**Files to create**:
```
engine/time/__init__.py
engine/time/camera.py
engine/time/transport.py
tests/test_time_camera.py
tests/test_transport.py
```

**Session 1.1: TimeCamera Core**
```python
# Deliverables:
- TimeCameraMode enum
- TimeCameraConfig dataclass  
- TimeCamera class with:
  - beat_to_px() / px_to_beat()
  - is_beat_visible() / get_visible_range()
  - snap_to_grid()
  
# Tests:
- test_beat_px_roundtrip()
- test_visible_range()
- test_snap_to_grid()
```

**Session 1.2: TimeCamera Interaction**
```python
# Deliverables:
- begin_drag() / update_drag() / end_drag()
- zoom()
- update() with follow mode
- animate_to_beat()

# Tests:
- test_drag_updates_left_beat()
- test_zoom_preserves_anchor()
- test_follow_mode_tracks_playhead()
- test_animation_completes()
```

**Session 1.3: Transport**
```python
# Deliverables:
- TimeSignature dataclass
- TransportState frozen dataclass with all properties
- Transport class with:
  - play/pause/stop/toggle
  - seek_to_beat/seek_to_time/seek_by_bars
  - update() returning TransportState
  - Callback lists

# Tests:
- test_transport_advances_on_update()
- test_loop_wraps_correctly()
- test_callbacks_fire_on_beat()
- test_state_is_immutable()
```

---

### WP2: 2D Drawing Foundation (2-3 sessions)

**Goal**: Draw rectangles and lines to screen via DrawBatch → UIRenderer2D.

**Files to create**:
```
engine/render/commands_2d.py
engine/render/renderer_2d.py
engine/render/shaders/rect.vert
engine/render/shaders/rect.frag
engine/render/shaders/line.vert
engine/render/shaders/line.frag
engine/render/shaders/line_instanced.vert
engine/render/shaders/line_instanced.frag
engine/core/errors.py
```

**Session 2.1: Command Types + DrawBatch**
```python
# Deliverables:
- All Cmd2D* dataclasses
- DrawBatch class with convenience methods
- Color4 type alias

# Tests:
- test_drawbatch_accumulates_commands()
- test_vertical_lines_uses_instancing_above_threshold()
```

**Session 2.2: RenderScheduler + Basic UIRenderer2D**
```python
# Deliverables:
- RenderScheduler with double-buffer
- UIRenderer2D skeleton with:
  - _setup_geometry() (unit quad, unit line)
  - _dispatch() method
  - _exec_rect() (single rect)
  - _exec_line() (single line)
- Rect shader (rounded corners)
- Line shader

# Tests (visual):
- Draw colored rect at position
- Draw line between two points
```

**Session 2.3: Instanced Rendering**
```python
# Deliverables:
- _exec_lines_instanced()
- _exec_rects_instanced()
- line_instanced shader
- Instance buffer management

# Tests (visual):
- Draw 100 lines efficiently
- Verify single draw call for instanced batch
```

---

### WP3: Text Rendering (2 sessions)

**Goal**: Render text using FreeType + texture atlas.

**Files to create**:
```
engine/ui/text/__init__.py
engine/ui/text/font.py
engine/ui/text/layout.py
engine/render/shaders/text.vert
engine/render/shaders/text.frag
assets/fonts/FiraMono-Regular.ttf  (or your choice)
```

**Session 3.1: FontAtlas + FontManager**
```python
# Deliverables:
- GlyphInfo dataclass
- FontAtlas class:
  - __init__(ctx, font_path, size)
  - _build_atlas() with FreeType
  - Glyph lookup
- FontManager class:
  - set_default_font()
  - get_font()
  - get_default()

# Dependencies:
- pip install freetype-py

# Tests:
- test_font_loads_from_path()
- test_glyph_metrics_reasonable()
```

**Session 3.2: Text Rendering Integration**
```python
# Deliverables:
- TextLayout class with measure() and layout()
- text.vert/frag shaders
- UIRenderer2D._exec_text()

# Tests (visual):
- Render "Hello SPECTRO" at various positions
- Verify alignment modes (left/center/right)
```

---

### WP4: Grid + Panel System (2-3 sessions)

**Goal**: GridBuilder and GraphPanel base class with scrolling grid.

**Files to create**:
```
engine/ui/panels/__init__.py
engine/ui/panels/grid_builder.py
engine/ui/panels/graph_panel.py
engine/core/types.py  (Rect, etc.)
```

**Session 4.1: GridBuilder**
```python
# Deliverables:
- GridConfig dataclass
- GridLabel dataclass
- GridBuilder class:
  - build()
  - build_with_labels()
  - build_alternating_backgrounds()
  - _get_subdivision_level()

# Tests:
- test_bar_lines_at_correct_positions()
- test_subdivision_changes_with_zoom()
```

**Session 4.2: GraphPanel Base**
```python
# Deliverables:
- PlayheadConfig dataclass
- MouseButtons enum
- GraphPanel ABC:
  - layout()
  - draw() orchestration
  - _draw_playhead()
  - Input handlers (pointer down/move/up, scroll)
  - _calculate_drag_velocity()

# Tests:
- test_draw_calls_grid_builder()
- test_middle_click_starts_drag()
```

**Session 4.3: Integration Test**
```python
# Deliverables:
- Simple test panel subclass
- Window that shows scrolling grid
- TimeCamera connected to panel

# Visual verification:
- Grid scrolls when dragging
- Zoom works with scroll wheel
- Playhead visible and positioned correctly
```

---

### WP5: Regions (3-4 sessions)

**Goal**: SyncRegion, WaveformRegion, LanesRegion, Panel3D container.

**Files to create**:
```
engine/ui/panels/sync_region.py
engine/ui/panels/waveform_region.py
engine/ui/panels/lanes_region.py
engine/ui/panels/panel_3d.py
engine/render/shaders/circle.vert
engine/render/shaders/circle.frag
engine/render/shaders/arc.vert
engine/render/shaders/arc.frag
```

**Session 5.1: Circle/Arc Rendering**
```python
# Deliverables:
- Circle shader (SDF-based)
- Arc shader
- UIRenderer2D._exec_circle()
- UIRenderer2D._exec_arc()

# Tests (visual):
- Draw filled circle
- Draw stroked circle
- Draw partial arc
```

**Session 5.2: SyncRegion**
```python
# Deliverables:
- SyncRegion class:
  - layout() with transport/beat/bar areas
  - handle_input() for button clicks
  - draw() with:
    - Transport buttons (play/stop)
    - BPM text
    - Beat phase circle
    - Bar phase circle
  - _draw_phase_circle()
  - _draw_transport_controls()

# Tests (visual):
- Buttons render
- Phase circles animate when transport playing
```

**Session 5.3: Panel3D + Basic WaveformRegion**
```python
# Deliverables:
- Panel3D container:
  - Owns regions
  - layout() divides space
  - handle_input() routes to regions
  - collect_draw_commands()
- WaveformRegion skeleton:
  - draw_content() with placeholder

# Tests (visual):
- Three regions visible
- Input routes correctly
```

**Session 5.4: LanesRegion**
```python
# Deliverables:
- SequencerEvent dataclass
- SequencerLane dataclass
- LanesRegion:
  - draw_content() with lane backgrounds
  - Instanced event rendering
  - draw_overlay() with selection

# Tests (visual):
- Lanes with alternating backgrounds
- Events render as colored blocks
```

---

### WP6: Input + PanelManager (1-2 sessions)

**Goal**: Complete input routing and central coordinator.

**Files to create**:
```
engine/ui/input.py
engine/ui/panel_manager.py
```

**Session 6.1: Input Infrastructure**
```python
# Deliverables:
- InputState dataclass
- MouseButtons class
- KeyBindings class
- PanelManager:
  - poll_input()
  - update()
  - render()
  - frame()

# Tests:
- test_space_toggles_playback()
- test_mouse_routes_to_correct_region()
```

---

### WP7: Audio Integration (2-3 sessions)

**Goal**: Load audio, compute waveform, display in WaveformRegion.

**Files to create**:
```
engine/audio/__init__.py
engine/audio/waveform.py
engine/audio/loader.py
engine/render/shaders/waveform.vert
engine/render/shaders/waveform.frag
```

**Session 7.1: WaveformCache**
```python
# Deliverables:
- WaveformCache class:
  - Load from numpy array
  - Decimation cache at multiple levels
  - get_envelope()

# Dependencies:
- pip install librosa (or soundfile for simpler loading)

# Tests:
- test_decimation_levels_correct()
- test_envelope_min_max_reasonable()
```

**Session 7.2: Waveform Rendering**
```python
# Deliverables:
- waveform.vert/frag shaders
- UIRenderer2D._exec_waveform()
- WaveformRegion.draw_content() complete

# Tests (visual):
- Load audio file
- Waveform displays correctly
- Zoom shows appropriate detail level
```

---

### WP8: Demo Application (1-2 sessions)

**Goal**: Complete working demo.

**Files to create**:
```
examples/spectro_demo.py
```

**Session 8.1: Integration**
```python
# Deliverables:
- Main window setup
- PanelManager integration
- Audio file loading dialog
- All panels working together

# Verification:
- 60 FPS performance
- Grid alignment pixel-perfect between regions
- Transport controls work
- Playhead syncs with audio (if audio playback added)
```

---

## Session Template

Each Claude Code / Cursor session should:

1. **Start**: 
   ```
   Reference SPECTRO_SPEC_CONSOLIDATED.md
   Current work package: WP[N]
   Branch: feature/[name]
   ```

2. **During**:
   - Implement deliverables listed
   - Write tests as specified
   - Commit after each logical chunk

3. **End**:
   - Update this plan with ✅ for completed items
   - Note any blockers or design changes
   - Commit with message: `WP[N]: [description]`

---

## Dependencies

```bash
# Core
pip install moderngl
pip install numpy

# Text rendering
pip install freetype-py

# Audio (for WP7)
pip install librosa
# OR simpler:
pip install soundfile

# Testing
pip install pytest
```

---

## Risk Mitigation

### Risk: Text rendering complexity
**Mitigation**: Start with monospace font only, fixed sizes. Expand later.

### Risk: Shader debugging
**Mitigation**: Keep shaders simple. Use solid colors first, then gradients/effects.

### Risk: Integration issues
**Mitigation**: Test each component against existing codebase immediately after creation.

### Risk: Performance
**Mitigation**: Profile early (Phase 4). Instance batching threshold can be tuned.

---

## Definition of Done

A work package is complete when:
1. All deliverables exist and compile
2. All specified tests pass
3. Visual tests show expected output
4. Code is committed to feature branch
5. No regressions in existing functionality

---

## Progress Tracking

| WP | Name | Sessions | Status | Branch |
|----|------|----------|--------|--------|
| 1 | Time Infrastructure | 3 | ⬜ Not Started | feature/time-core |
| 2 | 2D Drawing Foundation | 3 | ⬜ Not Started | feature/draw-batch |
| 3 | Text Rendering | 2 | ⬜ Not Started | feature/text-rendering |
| 4 | Grid + Panel System | 3 | ⬜ Not Started | feature/grid-panel |
| 5 | Regions | 4 | ⬜ Not Started | feature/regions |
| 6 | Input + PanelManager | 2 | ⬜ Not Started | feature/panel-manager |
| 7 | Audio Integration | 3 | ⬜ Not Started | feature/audio |
| 8 | Demo Application | 2 | ⬜ Not Started | feature/demo-app |

**Total estimated sessions**: 22 (~4-5 weeks at 5 sessions/week)

---

## First Session Kickoff

**Branch**: `feature/time-core`

**Goal**: Complete Session 1.1 (TimeCamera Core)

**Steps**:
1. Create `engine/time/` directory structure
2. Implement TimeCamera with core methods
3. Write roundtrip tests
4. Verify tests pass

**Success criteria**: 
```python
camera = TimeCamera(left_beat=0.0, window_beats=16.0)
camera._panel_width_px = 800.0
camera._px_per_beat = 50.0

assert camera.beat_to_px(8.0) == 400.0
assert camera.px_to_beat(400.0) == 8.0
assert camera.is_beat_visible(8.0) == True
assert camera.is_beat_visible(20.0) == False
```

---

*Ready to begin implementation.*

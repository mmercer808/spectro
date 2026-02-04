# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SPECTRO is a time-synchronized audio visualization engine built with Python and ModernGL. It provides a graphics engine for visualizing waveforms, spectrograms, MIDI events, and musical structure in a grid-based interface synchronized to audio transport.

The key architectural innovation is **pixel-perfect time synchronization** — multiple panels (waveform, lanes, sync controls) share a single `TimeCamera` to ensure perfect alignment as the user zooms and scrolls.

## Running the Application

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix

# SPECTRO display demo (main views, no audio) — run this first
python demo/run_demo.py

# Full SPECTRO demo (with audio, MIDI)
python spectro_demo_app.py

# Engine seed (3D viewports)
python app_mglw.py

# UI system demo (panels, 3D viewport widget)
python ui_demo.py

# Run tests
pytest tests/
pytest tests/test_time_camera.py  # Single test file
```

## Dependencies

Core dependencies: `moderngl`, `moderngl-window`, `numpy`

## Architecture

### Directory Structure

```
engine/
├── core/           # Foundational systems (manager, scene, signal, math3d)
├── time/           # Time/tempo synchronization (TimeCamera, Transport)
├── render/         # GPU rendering (commands, renderer, resources, targets)
├── ui/             # 2D UI system (widgets, layout, draw context)
├── viewport/       # 3D viewport management
├── graph/          # Procedural UI renderer (GraphObject, GraphSpace)
└── scene/          # 3D scene graph
```

### Core Design Patterns

1. **Immutable Snapshots**: Mutable scene graph during authoring, frozen snapshots for rendering. Command lists contain NO GL objects — they're pure data, enabling thread-safe production on worker threads.

2. **Double-Buffered DrawBatch**: Widgets add commands to `batch_back`, frame boundary swaps to `batch_front`, renderer executes `batch_front`. Enables async event handling outside main loop.

3. **Signal/Event Bridge**: Central `SignalBridge` for inter-system communication. Signals include `SIGNAL_DT`, `SIGNAL_TRANSPORT_CHANGED`, `SIGNAL_VIEW_CHANGED`, `SIGNAL_POINTER_*`. Weak references prevent circular dependencies.

4. **Shared TimeCamera**: ALL scrolling panels use the same `TimeCamera` instance. This is THE canonical beat↔pixel mapping. Never create separate time cameras for panels that need to scroll together.

### Key Classes

- **TimeCamera** (`engine/time/camera.py`): View transformation for beat axis. Methods: `beat_to_px()`, `px_to_beat()`, `is_beat_visible()`, `zoom()`, `animate_to_beat()`. Modes: `FOLLOW_PLAYHEAD`, `FREE_SCROLL`, `SNAP_TO_BARS`.

- **Transport** (`engine/time/transport.py`): Mutable playback state (play, pause, seek, BPM). Produces immutable `TransportState` snapshots with derived properties (`phase_in_beat`, `phase_in_bar`).

- **DrawBatch** (`engine/render/commands_2d.py`): Pure-data 2D command container. Methods: `rect()`, `line()`, `circle()`, `text()`, `set_clip()`.

- **ViewportArea** (`engine/viewport/viewport.py`): Manages 3D viewport with cameras, render targets, and picking.

### Rendering Pipeline

```
Transport.update(dt) → TransportState
    ↓
TimeCamera.update(playhead)
    ↓
Panel.draw(batch_back)  # widgets add commands
    ↓
swap(batch_front, batch_back)
    ↓
UIRenderer.execute(batch_front)
```

### Thread Safety Invariants

- `TransportState` is frozen dataclass — safe to read from any thread
- `DrawBatch` contains no GL objects — safe to build off main thread
- `TimeCamera` is single-threaded (main thread only)

## Specification Documents

- `SPECTRO_SPEC_CONSOLIDATED.md` — Complete specification (architecture, component definitions, implementation phases)
- `SPECTRO_IMPLEMENTATION_PLAN.md` — Phased development roadmap
- `docs/PROJECT_HANDOFF.md` — Project context and theory background

## Code Conventions

- Entry points inherit from `mglw.WindowConfig` (moderngl-window)
- Render method is `on_render(self, time: float, frame_time: float)`
- Use `self.ctx` for ModernGL context access
- Prefer instanced draws when rendering >4 similar items

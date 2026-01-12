# Engine Seed v3 — ModernGL Rendering Engine

A production-ready 3D rendering engine foundation built with Python and ModernGL.

## Key Features

- **Thread-Safe Async Extraction**: Scene graph snapshots enable safe multi-threaded command building
- **Render-to-Texture Architecture**: Every viewport renders to texture for compositing flexibility
- **Multi-Camera Viewports**: Sub-viewport support with per-camera picking
- **Upload Queue with Versioning**: Thread-safe GPU resource updates with stale job prevention
- **RenderWorld Batching**: Material buckets and instancing for scalable rendering

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN THREAD                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ EntityGraph  │───>│ Snapshot-    │───>│ ViewportArea │      │
│  │ (mutable)    │    │ Builder      │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │            GraphSnapshot              │               │
│         │            (immutable)                │               │
│         v                   v                   v               │
├─────────────────────────────────────────────────────────────────┤
│                       WORKER THREADS                             │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │ RenderWorld  │───>│ CommandList  │  (no GL objects!)        │
│  │ (batching)   │    │ (pure data)  │                          │
│  └──────────────┘    └──────────────┘                          │
├─────────────────────────────────────────────────────────────────┤
│                        GL THREAD                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ UploadQueue  │───>│ Renderer     │───>│ RenderTarget │      │
│  │ (versioned)  │    │ (execute)    │    │ (texture)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                             │                                    │
│                             v                                    │
│                      ┌──────────────┐                           │
│                      │ Compositor   │───> Screen                │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install moderngl moderngl-window numpy PySide6
```

PySide6 is optional if you only use the moderngl-window host.

## Running

### Standalone (moderngl-window)
```bash
python app_mglw.py
```

### Qt/PySide6 Host
```bash
python app_qt.py
```

### Stress Test (1000 cubes)
```bash
python app_stress_test.py
```

## Project Structure

```
engine_seed_v3/
├── app_mglw.py           # moderngl-window demo
├── app_qt.py             # PySide6/Qt demo
├── app_stress_test.py    # Performance stress test
├── engine/
│   ├── core/
│   │   ├── frame.py      # FrameState (timing)
│   │   └── snapshot.py   # GraphSnapshot, SnapshotBuilder, DirtyFlags
│   ├── render/
│   │   ├── commands.py   # CommandList, draw commands (pure data)
│   │   ├── renderer.py   # GL command execution
│   │   ├── resources.py  # ResourceRegistry (meshes, pipelines)
│   │   ├── targets.py    # RenderTarget, RenderTargetPool
│   │   ├── uploader.py   # UploadQueue with versioning
│   │   └── world.py      # RenderWorld batching layer
│   ├── scene/
│   │   └── graph.py      # EntityNode, Transform, MeshRenderer
│   └── viewport/
│       └── viewport.py   # ViewportArea, Camera, async extraction
```

## Design Principles

### 1. Viewport = Render-to-Texture (Always)

Every viewport renders to a texture, never directly to screen. This enables:
- Compositing viewports anywhere
- Post-processing per viewport
- Future: window docking, picture-in-picture, etc.

### 2. Two-Phase Architecture: Extract → Execute

```python
# Phase 1: Create immutable snapshot (main thread)
snapshot = SnapshotBuilder.from_graph(graph, cameras, w, h)

# Phase 2: Build commands (worker thread - no GL!)
cmdlists = build_render_commands(snapshot)

# Phase 3: Execute (GL thread)
renderer.execute_to_target(merged_cmdlist, target)
```

### 3. CommandLists Contain No GL Objects

Workers produce `CommandList` objects containing only:
- String keys (mesh_id, pipeline_id)
- Numpy arrays (transforms, colors)
- Scalars (counts, flags)

This makes them:
- Thread-safe to create anywhere
- Serializable for debugging/replay
- Testable without GL context

### 4. EntityGraph vs RenderWorld

- **EntityGraph**: Mutable, hierarchical, for authoring
- **RenderWorld**: Immutable, bucketed, for rendering

```python
# Authoring (main thread)
cube = EntityNode("cube")
cube.transform.pos = [1, 2, 3]

# Rendering (worker thread)
snapshot = SnapshotBuilder.from_graph(root, ...)
world = RenderWorld(snapshot)  # Bucketed by material
cmdlists = world.emit_commands_for_camera(cam, viewport)
```

### 5. Upload Queue with Versioning

Async uploads can complete out-of-order. Without versioning:
```
Frame 10: Request upload v10
Frame 11: Request upload v11
Frame 12: v11 completes → GPU has v11
Frame 13: v10 completes → GPU has v10 (WRONG!)
```

With versioning:
```
Frame 10: Request upload v10
Frame 11: Request upload v11
Frame 12: v11 completes → GPU has v11, current_version=11
Frame 13: v10 arrives, v10 < v11 → DROPPED
```

## Extending the Engine

### Adding a New Mesh

```python
# In ResourceRegistry or at runtime
registry.meshes["my_mesh"] = MeshGPU(
    vao_lit=ctx.vertex_array(lit_prog, ...),
    vao_pick=ctx.vertex_array(pick_prog, ...),
    vbo=ctx.buffer(vertices),
    ibo=ctx.buffer(indices),
    index_count=len(indices),
)
```

### Adding a New Shader Pipeline

```python
# In ResourceRegistry._create_default_pipelines()
my_vs = """..."""
my_fs = """..."""
self.pipelines["my_pipeline"] = PipelineGPU(
    program=self.ctx.program(vertex_shader=my_vs, fragment_shader=my_fs)
)
```

### Adding Custom Commands

```python
# In commands.py
@dataclass(frozen=True)
class CmdMyCustomDraw:
    my_param: int
    uniforms: Dict[str, Any]

# Update Command union
Command = Union[..., CmdMyCustomDraw]

# Handle in renderer.py
elif isinstance(cmd, CmdMyCustomDraw):
    self._execute_my_custom_draw(cmd, target)
```

## Future Development (TODO)

1. **RenderGraph**: Full frame graph with automatic resource management
2. **Instanced Picking**: Per-instance entity IDs in batched draws
3. **Shadow Mapping**: Multi-pass rendering with depth targets
4. **Post-Processing**: Screen-space effects pipeline
5. **Scene Serialization**: Save/load scene graphs

## License

MIT License - Use freely for your projects.

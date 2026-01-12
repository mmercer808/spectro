# Engine Seed v2 — ModernGL + moderngl-window host + Async CommandLists + Multi-Camera + Picking (skeleton)

This update adds a **standalone** ModernGL host via `moderngl-window` while keeping the PySide6 host.

### Install
```bash
pip install moderngl moderngl-window numpy PySide6
```
(PySide6 is optional if you only run `app_mglw.py`.)

### Run (standalone, recommended for pure engine work)
```bash
python app_mglw.py
```

### Run (Qt host)
```bash
python app_qt.py
```

### What's new vs v1
- **moderngl-window** host (`app_mglw.py`)
- **Dirty flag + debounce** async extraction per panel/area
- **RenderWorld extraction** (graph -> RenderItems once per extract)
- **Picking pass skeleton** (ID attachment + read pixel on click)

### Where to look
- `engine/viewport/viewport.py` : ViewportArea, dirty async extract, multi-camera sub-viewports, picking
- `engine/scene/graph.py`       : entity graph + stable entity ids + RenderItem extraction
- `engine/render/commands.py`   : CommandList + draw commands
- `engine/render/targets.py`    : RenderTargetSpec with optional pick attachment
- `engine/render/resources.py`  : pipelines + meshes + compositor blitter
- `engine/render/renderer.py`   : executes command lists + reads pick IDs

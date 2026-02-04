# SPECTRO — What Is Buildable

This document lists **runnable artifacts** and how to run them. All commands assume project root and an active venv (e.g. `.venv\Scripts\activate` on Windows).

---

## The buildable (summary)

| Artifact | Command | What you get |
|----------|---------|--------------|
| **Display demo** | `python demo/run_demo.py` | Main UI (SyncRegion, WaveformRegion, LanesRegion) + WS response log panel; no audio. |
| **Full demo** | `python spectro_demo_app.py` | MIDI → timeline → audio; keys 1–4 = drums, SPACE = play/pause. |
| **Engine seed** | `python app_mglw.py` | 3D viewports, picking, engine seed. |
| **UI system demo** | `python ui_demo.py` | WindowManager, panels, 3D viewport widget. |
| **Graph demo** | `python examples/graph_demo.py` | Procedural UI (GraphObject, text, quads); requires GL 4.3. |
| **Tests** | `pytest tests/` | TimeCamera, transport, etc. |

---

## 1. Display demo (recommended for layout + WS)

**Command:** `python demo/run_demo.py`

**What it is:** The main SPECTRO view layout with **no audio or MIDI**. Use it to verify:

- SyncRegion (transport bar: play/pause, BPM, beat/bar)
- WaveformRegion (placeholder strip)
- LanesRegion (timeline grid, lane rows, playhead)
- **Right-hand panel:** append-only WS response log (text rendered in-panel)

**Requirements:**

- Python 3.10+
- `moderngl`, `moderngl-window`, `numpy`
- **OpenGL 4.3** (for graph `TextRenderer` / SSBO)
- **Pillow** (for font atlas used in WS log text)
- Optional: set `SPECTRO_WS_URL` to connect to a WebSocket server; responses appear in the log panel and in the console.

**Controls:** SPACE = play/pause, R = reset, Left/Right = seek by bar, Scroll = zoom.

---

## 2. Full demo (audio + MIDI)

**Command:** `python spectro_demo_app.py`

**What it is:** Full integration: MIDI input, EventDispatcher, Transport, TimeCamera, AudioEngine. Keys 1–4 trigger drums; SPACE = play/pause.

**Requirements:** Same as display demo, plus `sounddevice`, `rtmidi` (or portmidi) if using MIDI.

---

## 3. Engine seed

**Command:** `python app_mglw.py`

**What it is:** 3D viewports, picking, core engine seed. GL 3.3.

---

## 4. UI system demo

**Command:** `python ui_demo.py`

**What it is:** WindowManager, 2D panels, 3D viewport widget. GL 3.3.

---

## 5. Graph demo (procedural UI)

**Command:** `python examples/graph_demo.py`

**What it is:** GraphObject, GraphSpace, text and quad rendering. **Requires GL 4.3.**

---

## 6. Tests

**Command:** `pytest tests/` or `pytest tests/test_time_camera.py`

**What it is:** Unit tests for TimeCamera, transport, etc.

---

## Dependencies (minimal for display demo)

```text
moderngl
moderngl-window
numpy
Pillow
```

OpenGL 4.3 is required for the display demo’s **WS log text** (engine graph `TextRenderer`). On machines without 4.3, the demo would need a fallback (e.g. text omitted or a 3.3-compatible path).

---

## Where things live

- **Display demo:** `demo/run_demo.py`
- **WS stack (response log, client, dark scheduler):** `engine/ws/`
- **Graph text (font atlas, TextRenderer):** `engine/graph/text.py`
- **Time/transport:** `engine/time/`
- **Specs / sourcebook:** `docs/`, `docs/sourcebook/`

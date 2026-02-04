# SPECTRO Demo

## Running the demo

**Display-only (no audio)** — main UI views only:

```bash
# From project root
python demo/run_demo.py
```

**Layout:**  
- **Left panel:** Tracks view — SyncRegion, WaveformRegion, LanesRegion. Lanes show what’s playing; Launchpad pad hits are echoed into the sequencer as lane events.  
- **Right panel:** Output echo — all output (stdout, stderr, WS) is rendered here; the text renderer runs inside this panel’s viewport only.  

WebSocket client runs in a separate thread from the start. Launchpad (or any MIDI) input is echoed to the sequencer and shown as lanes on the left.  

Controls: **SPACE** play/pause, **R** reset, **Left/Right** seek by bar, **Scroll** zoom. **Launchpad pads** → add events to lanes (note % 4 = lane).  
Requires: OpenGL 4.3, Pillow. Optional: `python-rtmidi` for Launchpad, `SPECTRO_WS_URL` for WebSocket.

**Full demo (with audio)** — MIDI → timeline → audio:

```bash
python spectro_demo_app.py
```

Requires `engine.buffers_v2`, Transport, TimeCamera, AudioEngine. Keys 1–4 = drums, SPACE = play/pause.

## Other entry points

- `python app_mglw.py` — Engine seed (3D viewports, picking)
- `python ui_demo.py` — UI system (WindowManager, panels, 3D viewport widget)

## Layout (display demo)

Matches `SPECTRO_SPEC_CONSOLIDATED.md`:

- **SyncRegion** (~15%): Transport bar at bottom of window
- **WaveformRegion** (~25%): Placeholder strip above transport
- **LanesRegion** (~60%): Timeline grid + lane rows + playhead

Audio and waveform data can be hooked up later without changing the view layout.

**Full list of runnable artifacts:** see project root [BUILDABLE.md](../BUILDABLE.md).

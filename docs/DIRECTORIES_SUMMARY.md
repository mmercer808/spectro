# Directories Summary (and Extraneous / Fall‑out)

Keep the **Vercel app** in mind: there is no Vercel config in this repo yet. If you add a Vercel app (e.g. Next.js frontend or serverless API), it usually lives in a subdir such as `web/`, `app/`, or `api/`; Vercel deploys from that subdir or repo root. This repo is currently the **SPECTRO Python engine + demos**; a Vercel app would typically consume it via API or static build. **Vercel app timeline/mix-channel/verses/piano spec:** [VERCEL_APP_TIMELINE_SPEC.md](VERCEL_APP_TIMELINE_SPEC.md) — continuous timeline, add object → mix channel when it matches, schedule sequence (start = now or 0.00), verses/blocks playable regardless, MIDI key changes over the track, record piano, plugin/VST selected then.

---

## 1. Directory summary

| Directory / file | Purpose |
|------------------|--------|
| **engine/** | Core Python engine: audio, core, graph, midi, render, scene, time, ui, viewport, **ws** (WebSocket client, dark scheduler, response log). |
| **demo/** | Display-only and playable demos: `run_demo.py`, harnesses, demo TODOs. |
| **docs/** | Specs, sourcebook, clubs, MAIN_APP_WS_REQUIREMENT, RENDERING_PIPELINE, etc. |
| **docs/sourcebook/** | Standards of record: tempo fixation, improvising standards, required_standards.json. |
| **docs/clubs/** | Club JSONs (timeline, solver, dispatcher, events, analysis) + index. |
| **math/** | Beat-detection and signal-path math (markdown). |
| **tests/** | Pytest tests (e.g. test_time_camera). |
| **spec/** | Separate spec runner (pyproject.toml, main.py, uv). Role TBD vs root. |
| **spectro_demo_app.py** | Main demo app (MIDI → timeline → audio). |
| **app_mglw.py** | Engine seed (3D viewports). |
| **ui_demo.py** | UI system demo (panels, viewport). |
| **SPECTRO_*.md, CLAUDE.md, etc.** | Project and integration docs at root. |

| Directory / file | Purpose |
|------------------|--------|
| **_old/** | Old/duplicate code (86+ files). Archive; not needed for current app. |
| **_store/** | Snapshot/backup of engine + SPECTRO_SESSION_HANDOFF. Archive; can fall out. |
| **downnn/** | Contains only `spectro_demo_app.py` + `SPECTRO_DEMO_INTEGRATION_TODO.md` — **duplicates of root**. |
| **sess/** | Contains only `SPECTRO_DEMO_INTEGRATION_TODO.md` — **duplicate of root**. |
| **concepts/** | Mockups, images, HTML/SVG sequencer views, extractor UI. Reference/concept; could move to `docs/concepts` or `_archived`. |
| **circ-buf/** | Buffer/glue stubs and guide. Overlaps with `engine/buffers_v2.py`; likely legacy. |
| **matts/** | Separate systems (context, graph, signal, live_code, examples). Unclear if used by SPECTRO demos; could be dependency or legacy. |
| **examples/** | `graph_demo.py` only. Keep if referenced; else optional. |

| Root files | Note |
|------------|------|
| **spectro_demo_app copy.py** | Duplicate of `spectro_demo_app.py` → **extraneous**. |
| **files.zip**, **files(26).zip** | Binary archives at root → better in `assets/` or `.gitignore`; often **extraneous** in repo. |
| **beat_detection_math.html**, **beat_detection_math(2).html** | One may be duplicate; **(2)** candidate to remove. |
| **heavenfalls.mp3** | Sample for “over analyzed sample”; keep. |
| **Audio Systems Learning Map (...).pdf**, **moderngl_procedural_ui_paper.pdf** | Reference; optional in `docs/` or `assets/`. |
| **#plan.md** | Planning; keep or move to docs. |
| **midi_demo.py** | Standalone MIDI demo; keep if used. |

---

## 2. Extraneous / fall‑out candidates

These can be removed or archived so the tree stays clear for the main app (and a future Vercel app):

| Entry | Reason |
|-------|--------|
| **downnn/** | Only duplicates of root files; safe to delete or archive. |
| **sess/** | Only duplicate TODO; safe to delete or archive. |
| **spectro_demo_app copy.py** | Duplicate; remove once sure. |
| **_old/** | Archive; exclude from “active” tree or move to `_archived/`. |
| **_store/** | Snapshot; same as above. |
| **beat_detection_math(2).html** | Likely duplicate of `beat_detection_math.html`; remove **(2)** if redundant. |
| **files.zip**, **files(26).zip** | Move to `assets/` or ignore; avoid binary blobs at root. |
| **circ-buf/** | If all behavior lives in `engine/buffers_v2.py`, treat as legacy; move to `_old` or delete. |
| **concepts/** | If only reference, move to `docs/concepts` or `docs/_archived/concepts`. |
| **matts/** | If SPECTRO demos don’t use it, move to `_old` or separate repo; else document as dependency. |
| **engine/spectro.code-workspace** | Workspace file; optional in repo. |
| **engine/__init__ .py** | Space in filename (`__init__ .py`); rename to `__init__.py` if that’s the intent. |

---

## 3. Suggested “active” layout (Vercel‑friendly)

Keep the main app and a potential Vercel app easy to reason about:

- **Root:** Entry points (`spectro_demo_app.py`, `app_mglw.py`, `ui_demo.py`), top-level docs, config.
- **engine/** — SPECTRO Python engine (no change).
- **demo/** — Demos (no change).
- **docs/** — All specs, sourcebook, clubs (no change).
- **math/**, **tests/** — Keep.
- **spec/** — Keep if you use it; otherwise fold into root or remove.
- **web/** or **app/** — (Future) Vercel app (Next.js, etc.) if you add it.
- **assets/** — (Optional) Samples, PDFs, zips, so root stays clean.

After fall‑out: remove or archive **downnn**, **sess**, **spectro_demo_app copy.py**, and optionally **_old** / **_store**; move zips and heavy binaries to **assets/** or ignore; then the repo is a clear “engine + demos + docs,” with room for a Vercel app in a subdir.

---

*Last updated: Feb 2025.*

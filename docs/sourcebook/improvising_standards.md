# Improvising Standards for the App

These standards govern real-time improvisation, timeline, and event flow in SPECTRO. They are **required** for the app; each standard document is **recorded for its entirety** as the canonical source. See [required_standards.json](required_standards.json) for the full manifest.

---

## 1. Scope

**Improvising standards** are the standards of record for:

- **Tempo and beats** — fixation of tempo on the beats; time↔beat conversion.
- **Timeline** — single source of position and length; internal or audio-file.
- **Solver** — collection solver (length, bar length, tempo, sr) and derived quantities.
- **Dispatcher** — late-bound context; process_frame; beat/bar callbacks.
- **Event manager** — wiring of MIDI, sequencer, waveform, automation (quad stream).
- **Over any sample** — running the pipeline over an analyzed audio file (e.g. heavenfalls.mp3).
- **Clubs** — links and bouncing elements (timeline, solver, dispatcher, events, analysis).
- **Main app WS** — accepted WebSocket server for saving additional edits for a project (whichever project); dark scheduling; response text space (append-only).

All of the above are **required**; the documents that define them are listed in the required-standards manifest and are recorded for their entireties.

---

## 2. Required Documents (Recorded for Their Entireties)

| Document | Purpose |
|----------|---------|
| [tempo_fixation_standard.md](tempo_fixation_standard.md) | Fixation of tempo on the beats; standard of record for beat timing; canonical equations. |
| [required_standards.json](required_standards.json) | Manifest of all required standard files; each recorded for its entirety. |
| [../timeline_solver_dispatcher_spec.md](../timeline_solver_dispatcher_spec.md) | Formal spec: Timeline, Solver, Dispatcher, Event Manager; over an analyzed sample. |
| [../dispatcher_equation.md](../dispatcher_equation.md) | Collection solver equations; one-frame equation; quad stream over any sample. |
| [../clubs/index.json](../clubs/index.json) | Clubs index; external links; sourcebook reference. |
| [../clubs/timeline_club.json](../clubs/timeline_club.json) | Timeline club — elements and bounces. |
| [../clubs/solver_club.json](../clubs/solver_club.json) | Solver club — tempo fixation, equations; standard of record link. |
| [../clubs/dispatcher_club.json](../clubs/dispatcher_club.json) | Dispatcher club — equation, code ref, quad stream. |
| [../clubs/events_club.json](../clubs/events_club.json) | Events club — event manager, integration flow. |
| [../clubs/analysis_club.json](../clubs/analysis_club.json) | Analysis club — analyzed sample, duration, tempo, heavenfalls; external links. |
| [../MAIN_APP_WS_REQUIREMENT.md](../MAIN_APP_WS_REQUIREMENT.md) | Main app: accepted WS server for saving project edits; dark scheduling; response text space (append-only). |

These files are **required** for the app. Each is the **full record** of its standard (recorded for its entirety). Do not rely on summaries elsewhere in place of the full document.

---

## 3. Other Improvising Standards (To Be Recorded)

Additional improvising standards may be added and will be:

- Listed in [required_standards.json](required_standards.json).
- Recorded for their entireties (the full document is the standard).
- Referenced from this page.

When adding a new standard, add an entry to the manifest and link it here.

---

## 4. Summary

- **Improvising standards** = tempo fixation, timeline–solver–dispatcher–event manager, quad stream, clubs, analysis.
- **Required** = the documents listed in §2 and in required_standards.json are required for the app.
- **Recorded for their entireties** = the full content of each document is the canonical standard; do not substitute a summary.

---

*Sourcebook — Improvising standards. Last updated: Feb 2025.*

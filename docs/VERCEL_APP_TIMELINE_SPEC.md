# Vercel App: Continuous Timeline, Mix Channel, Verses/Blocks, Piano Recording

Spec for the **Vercel app** (or frontend that consumes SPECTRO): continuous timeline, timeline objects that become the **mix channel** when the playhead matches, sequence scheduling (start = now or 0.00), verses/blocks playable regardless, MIDI key changes over the track, and piano recording with plugin/VST selection (to be selected then).

---

## 1. Continuous Timeline

- The Vercel app uses a **continuous timeline**: one time axis that runs from a **start** (e.g. 0.00 or “now”) and advances with the beat.
- **Start** can be:
  - **time = 0.00** — timeline origin at 0; playhead and all positions are relative to 0.
  - **time = now** — timeline origin at “now”; start is the current moment, and positions are relative to that.
- The timeline is the single source of truth for “where we are”; objects (sequences, verses, blocks) are placed on it and can be scheduled to start at a given time (including start = 0.00 or start = now).

---

## 2. Add Object to Timeline → Mix Channel When It Matches

- **Add an object to the timeline:** The user (or the system) can add **timeline objects** (e.g. a sequence, a verse, a block, a mix preset). Each object has a **position** (time or beat) on the timeline.
- **When it matches up, it is the mix channel:** When the **playhead** (current time) **matches** the object’s position (or enters its range), that object becomes the **active mix channel**. So:
  - The **mix channel** = the timeline object whose position/range the playhead is currently in (or the object that “matches” the current time).
  - The app can use this to switch levels, effects, or arrangement (e.g. verse A mix, verse B mix) automatically as the timeline advances.
- Matching can be: playhead ≥ object.start and playhead < object.end, or playhead at object.start (instant switch), depending on design. The spec is: **match = playhead aligns with the object’s position/range → that object is the mix channel.**

---

## 3. Schedule a Sequence Object: Start = Now or Start = 0.00

- A **sequence object** can be scheduled to **start** at:
  - **start = now** — the sequence begins at the current playhead position (or the moment “now” when scheduled). So “start where we are.”
  - **start = 0.00** — the sequence begins at timeline time 0.00 (wherever “start” is for the project). So “start at the beginning.”
- The same idea applies to **verses/blocks**: they can be scheduled with a start time (now or 0.00 or any fixed time). The important point is: **start** is explicit (now or 0.00 or a value), and the timeline is continuous so that when playhead reaches that start, the object can become active (e.g. mix channel) or begin playing.

---

## 4. Verses/Blocks Playable Regardless; Orchestrate with MIDI

- **Verses/blocks can be played regardless:** Verses or blocks (sections of the arrangement) are **not** locked to a strict linear order. They can be:
  - Triggered or played independently (e.g. “play verse B now”),
  - Scheduled on the timeline (start at 0.00, now, or another time),
  - Used as the **mix channel** when the playhead matches their position/range.
- **Orchestrate with MIDI:** MIDI is used to drive **key changes** (and other harmony/arrangement) **over the track**. So:
  - The **beat** is the underlying grid (tempo fixation; see sourcebook).
  - MIDI events (or a MIDI track) can change key (or patch) at certain positions; the timeline + beat grid stay the reference, and MIDI “orchestrates” the key changes over that.
- So: verses/blocks are playable regardless of strict order; the timeline + beat underlie everything; MIDI orchestrates key (and optionally other parameters) over the track.

---

## 5. Record Piano; Plugin/VST Selection (Then)

- **Record some piano:** The app must support **recording piano** (MIDI or audio). Recorded piano is placed on the timeline (or in a lane) so it plays back in sync with the continuous timeline and beat.
- **Plugin/VST option will be selected then:** The choice of **plugin/VST** (for piano sound, or for the piano track) is **selected later** — i.e. the spec does not fix a particular plugin/VST now; the app should allow:
  - Selecting a plugin/VST for the piano (or for a track) at record time or playback time.
  - Using that selection for playback (and optionally for monitoring during record).
- So: **piano recording** is required; **plugin/VST selection** is a placeholder for “user selects plugin/VST then” (implementation detail to be wired when the option is available).

---

## 6. Summary (Vercel App)

| Requirement | Description |
|-------------|-------------|
| **Continuous timeline** | One timeline; start = 0.00 or now; playhead advances with beat. |
| **Add object to timeline** | Objects (sequences, verses, blocks) have a position/range on the timeline. |
| **Match → mix channel** | When playhead matches an object’s position/range, that object is the **mix channel**. |
| **Schedule sequence** | Sequence (and verses/blocks) can start at **start = now** or **start = 0.00** (or any time). |
| **Verses/blocks playable regardless** | Can be triggered or scheduled independently; not strictly linear order. |
| **MIDI key changes** | MIDI orchestrates key changes over the track; beat is the underlying grid. |
| **Record piano** | Support recording piano (MIDI or audio) on the timeline. |
| **Plugin/VST** | Option to select plugin/VST for piano (or track) — selected then. |

---

## 7. Relation to SPECTRO Engine

- **Timeline:** Same notion as in [timeline_solver_dispatcher_spec.md](timeline_solver_dispatcher_spec.md) — continuous timeline with playhead and length; solver gives beat_in_loop, etc.
- **Tempo/beat:** [sourcebook/tempo_fixation_standard.md](sourcebook/tempo_fixation_standard.md) — BPM fixes the beat grid; MIDI key changes and verses/blocks use that grid.
- **Vercel app** can be a frontend (Next.js, etc.) that:
  - Renders the continuous timeline and timeline objects.
  - Sends “add object,” “schedule sequence (start = now or 0.00),” “set mix channel,” “record piano,” “select plugin/VST” to a backend or to the SPECTRO engine via API/WS.
  - Receives playhead position and match state (which object is mix channel) from the engine or from local state driven by the same timeline.

---

*Last updated: Feb 2025.*

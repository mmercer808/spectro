# Sourcebook: Tempo Fixation and Standard of Record for Beats

This page is the **standard of record** for how tempo is defined and how it fixes the relationship between time and beats in SPECTRO. All timeline, solver, and dispatcher behavior that depends on beat position or length shall follow this standard.

---

## 1. Standard of Record

**Tempo (BPM) is the single authoritative source** for the relationship between:

- Real time (seconds or samples)
- Beat position (beat index, fractional beat)
- Bar position (when used with a fixed beats-per-bar)

No other quantity shall define “where beat N is” in time except as derived from tempo (and sample rate). If tempo is changed, all beat positions are recomputed from the new tempo; there is no separate “beat grid” that overrides tempo unless explicitly designated as a re-fixation (e.g. a user-edited or analyzed beat map that then defines a new effective tempo or a tempo map).

**Scope:** The standard applies over a defined scope: one loop, one audio file, or one session. Within that scope, one tempo (or one tempo map, if extended) is the standard of record.

---

## 2. Fixation of Tempo on the Beats

**Definition:** *Tempo is fixed on the beats* means:

1. **Single BPM per scope**  
   One value of BPM (or one tempo map) applies over the scope (loop, file, or session). That value is the **fixation**: it fixes how real time maps to beat position.

2. **Beat position is derived from tempo**  
   Beat position (in beats, possibly fractional) is defined by:
   - **From time (seconds):**  
     `beat_position = time_seconds × (tempo / 60)`  
     so that one minute of real time equals `tempo` beats.
   - **From samples:**  
     `beat_position = sample_index × (tempo / 60) / sample_rate`  
     equivalently:  
     `beat_position = sample_index / samples_per_beat`  
     where  
     `samples_per_beat = sample_rate × 60 / tempo`.

3. **Inverse (time from beats)**  
   - **Time in seconds:**  
     `time_seconds = beat_position × (60 / tempo)`  
   - **Sample index:**  
     `sample_index = beat_position × samples_per_beat`  
     with `samples_per_beat = sample_rate × 60 / tempo`.

4. **Source of the fixation**  
   The BPM value (the fixation) is **provided by the user** (e.g. from a control, config, or session) or from an **analyzed** source (e.g. tempo detection on an audio file). The system does not assume a default BPM as the standard of record in production; the fixation is explicit.

5. **Re-fixation**  
   Changing BPM (or loading a new tempo map) is a **re-fixation**: from that point on, beat positions are derived from the new tempo. Any previously computed beat positions are not retained as a separate grid unless explicitly stored and designated as the new standard (e.g. an analyzed beat grid used as a tempo map).

---

## 3. Canonical Equations (Standard of Record)

All components that convert between time and beats shall use these definitions. They are the **fixation of tempo on the beats** in equation form.

| Quantity | Formula |
|----------|---------|
| Seconds per beat | `seconds_per_beat = 60 / tempo` |
| Samples per beat | `samples_per_beat = sample_rate × 60 / tempo` |
| Beat position from time (seconds) | `beat = time_seconds × (tempo / 60)` |
| Beat position from samples | `beat = sample_index / samples_per_beat` |
| Time (seconds) from beat | `time_seconds = beat × (60 / tempo)` |
| Sample index from beat | `sample_index = beat × samples_per_beat` |

**Symbols:**  
- `tempo` = BPM (beats per minute), the fixation.  
- `sample_rate` = sample rate in Hz.  
- `beat` = beat position (real number; integer part = beat index, fractional part = phase within the beat).

---

## 4. Length in Beats

When the timeline has a length (e.g. a loop or an audio file):

- **Length in beats** is either:
  - User- or config-specified (e.g. loop length in beats), or  
  - Derived from duration in time using the same fixation:  
    `length_beats = duration_seconds × (tempo / 60)`  
    or  
    `length_beats = length_samples / samples_per_beat`.

- The same tempo (fixation) used for playhead position is used for length. There is one standard of record for tempo over the scope.

---

## 5. Bar and Beat-in-Bar (When Using a Fixed Bar Length)

When a bar length in beats (`brlength`, e.g. 4 for 4/4) is defined:

- `current_bar = floor(beat_position / brlength)`
- `beat_in_bar = beat_position mod brlength`
- `phase_in_beat = beat_position mod 1`
- `phase_in_bar = (beat_position mod brlength) / brlength`

Bar positions are derived from the same beat position that is derived from tempo. No separate “bar clock” overrides the tempo fixation.

---

## 6. References from Other Docs

- **Timeline–solver–dispatcher spec:** Tempo is user-provided (proper fixation); solver uses the canonical equations above. See [../timeline_solver_dispatcher_spec.md](../timeline_solver_dispatcher_spec.md).
- **Collection solver (dispatcher equation):** Same equations; solver parameters include `tempo`. See [../dispatcher_equation.md](../dispatcher_equation.md#collection-solver-length--bar_length--tempo).
- **Clubs:** Solver club and index reference this sourcebook as the standard of record for tempo fixation. See [../clubs/solver_club.json](../clubs/solver_club.json), [../clubs/index.json](../clubs/index.json).

---

## 7. Summary

| Term | Meaning |
|------|---------|
| **Standard of record** | Tempo (BPM) is the single source for time↔beat conversion over the scope. |
| **Fixation of tempo on the beats** | One BPM per scope; beat position = f(tempo, time); user or analysis provides BPM; equations in §3 are canonical. |
| **Re-fixation** | Changing BPM (or tempo map) updates the standard of record from that point on. |

This sourcebook page is the **fixation of tempo on the beats** and the **standard of record** for beat timing. Implementations (timeline, solver, dispatcher, event manager) shall derive beat position and length from tempo according to §3 and §4.

---

*Sourcebook — Tempo fixation and standard of record. Last updated: Feb 2025.*

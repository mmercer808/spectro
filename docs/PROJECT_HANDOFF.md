# Project Handoff: Audio Analysis Tool & Pitch-Rhythm Theory

*Document for continuing this work in future sessions*

---

## Who Is CloudyCadet?

A software developer and graphics programmer with expertise in Python, PySide6/Qt, OpenGL, and audio visualization. They're a DJ who uses Ableton Live, Virtual DJ, and hardware controllers (Launchpad X, Numark controller). They do live stem-separation mixing and have discovered interesting phenomena about how tracks combine.

**Key traits:**
- Thinks visually and geometrically
- Has deep intuition about music from DJ experience but isn't formally trained
- Hears overtones that others filter out (sometimes confusing, but means raw acoustic perception)
- Bounces between projects; has theoretical graphics engine work but no working engine yet
- Has a moderngl setup that works without glitches
- Prefers discovery and exploration over rigid planning

---

## The Big Picture: What We're Building Toward

### The Theory Side (Path B + C from earlier discussion)

We're developing a unified framework connecting:

1. **Continuous pitch space** — Real singing/playing isn't locked to 12-TET. Singers navigate between multiple "attractor" systems (12-TET, just intonation, 53-TET) based on harmonic context.

2. **Rhythm-harmony duality** — Frequency ratios in harmony ARE polyrhythms at a different timescale. A 3:2 ratio is both a perfect fifth (pitch) and a polyrhythm (rhythm). The math is identical.

3. **The "groove" / "pocket"** — CloudyCadet's pendulum/punching bag model suggests groove is about phase relationships at multiple hierarchical levels (beat → bar → phrase → section).

4. **Negative space in audio** — When mixing two tracks, the gaps (frequencies NOT present) matter as much as what's there. Good combinations fill each other's spectral gaps.

### The Tool Side

An audio analyzer that helps:
- Visualize spectral content
- Detect phrase/section boundaries  
- Compare tracks for compatibility
- Eventually: slice drum breaks, detect "magic moments" in mixes

---

## The Logarithmic Piano Insight (Latest Discovery)

CloudyCadet asked a brilliant question: If we scale the piano roll to match a spectrogram, would lower notes be taller (larger Y dimension) and higher notes be shorter?

**Answer: YES.** 

Spectrograms are typically linear in frequency (Hz), but musical pitch is logarithmic (each octave doubles). If we want the piano to match the spectrogram:

```
Linear frequency spectrogram:
0 Hz ─────────────────────────────── 20000 Hz
     [C1 C2 C3 C4 C5 C6 C7 C8]
      ↑   ↑  ↑  ↑ ↑ ↑↑↑↑
      │   │  │  │ └─┴┴┴── Higher octaves compressed
      │   │  └──┴─────── Middle octaves 
      └───┴───────────── Low octaves stretched

The piano keys would be HUGE at the bottom, tiny at the top.
```

Alternatively, use a **log-frequency spectrogram** where each octave gets equal vertical space — then the piano keys are uniform, which is what Melodyne does.

**This is a design choice:** Linear shows acoustic reality (and explains why bass frequencies "dominate"). Log shows musical reality (equal importance per octave).

Could offer both views, or a slider between them.

---

## Technical Context

### CloudyCadet's Setup
- **DAWs:** Ableton Live, Virtual DJ
- **Interface:** MOTU
- **Controllers:** Novation Launchpad X, Numark DJ controller (needs upgrade)
- **Stem separation:** Virtual DJ built-in, RipX (has alignment issues with stem lengths)
- **Programming:** Python, PySide6/Qt, OpenGL/ModernGL
- **Current state:** Has moderngl setup that works, but no complete graphics engine yet

### Proposed Tech Stack for Audio Tool
```
Audio I/O:        librosa or audiofile
FFT/Spectral:     numpy + scipy (or librosa)
Pitch detection:  crepe, pyin, or basic-pitch
Beat detection:   madmom or librosa.beat
GUI:              PySide6
Visualization:    ModernGL (spectrogram as texture)
```

---

## The Audio Analyzer Tool Vision

### Interface Layout
```
┌─────────────────────────────────────────────────────────────────────┐
│  WAVEFORM VIEW                                                      │
│  ▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁▁▂▃▅▆▇█▇▆▅▃▂▁            │
│  ①          ②              ③                ④                       │
├─────────────────────────────────────────────────────────────────────┤
│  SPECTROGRAM VIEW (with optional piano overlay)                     │
│  [Linear/Log toggle]                                                │
│  ░░▒▒▓▓██▓▓▒▒░░░░▒▒▓▓██▓▓▒▒░░░░▒▒▓▓██▓▓▒▒░░░░▒▒▓▓██▓▓▒▒░░        │
│  ░▒▓██████▓▒░░░▒▓██████▓▒░░░░▒▓██████▓▒░░░░▒▓██████▓▒░░░          │
│  ▓████████████▓████████████▓████████████▓████████████▓            │
├─────────────────────────────────────────────────────────────────────┤
│  ANALYSIS MARKERS                                                   │
│  Numbered circles at detected features                              │
│  ① Phrase boundary  ② Energy peak  ③ Chord change  etc.            │
├─────────────────────────────────────────────────────────────────────┤
│  TRANSPORT + INFO                                                   │
│  [Play] [Stop] ──●────── 0:00    BPM: 128  Key: Cm                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Development Phases

**Phase 1: Just See It** (START HERE)
- Load audio file (wav, mp3, etc.)
- Display waveform
- Display spectrogram (linear frequency first, then add log option)
- Basic transport (play, stop, seek)
- BPM and key detection (display only)

**Phase 2: Find Structure**
- Phrase boundary detection
- Transient detection (drum hits)
- Energy contour
- Manual marker system (click to add numbered markers)

**Phase 3: Compare Tracks**
- Load two tracks
- Spectral complement visualization (show where they fill each other's gaps)
- Alignment tools (shift one track in time)
- Compatibility scoring

**Phase 4: Drum Break Slicer** (separate feature, lower priority)
- Detect hits in a drum loop
- Slice at transients
- Rearrange slices
- Export

---

## Key Documents Created

1. **chord_math_cheatsheet.md** — Distilled math from Cubarsi's book on chord transformations, covering modes, groups, drift operators, and polytopes

2. **pitch_rhythm_framework.md** — Working theory document on continuous pitch space, attractors, and rhythm-harmony duality

3. **This document** — Project handoff for continuity

---

## Open Questions to Explore

1. **Attractor strength** — How do we measure/model the "pull" of different tuning systems on a sung note?

2. **The 10-20 Hz gap** — Too fast for rhythm, too slow for pitch. Is this where "groove" lives?

3. **Phase at multiple levels** — How do beat phase, bar phase, and phrase phase interact to create "feel"?

4. **Spectral complement** — Can we predict which tracks will mix well by analyzing their frequency gaps?

5. **The magic moments** — When CloudyCadet mixes two songs and it clicks, what's actually happening? Need data to analyze.

---

## Immediate TODOs for Next Session

1. **Decide on starting point:** Basic spectrogram viewer with ModernGL? Or simpler PySide6-only prototype first?

2. **Get audio loading working:** Pick a library (librosa is full-featured, audiofile is simpler)

3. **Render a spectrogram:** Either as ModernGL texture or QImage

4. **Add waveform view:** Simpler than spectrogram, good warmup

5. **Basic transport:** Play/pause/seek with audio playback synced to visual position

---

## CloudyCadet's Workflow Context

When DJing live with stem separation:
- Takes acapella from one track, drums from another, instruments from a third
- Pitch shifts and BPM matches to get them in the same key/tempo
- **Critical finding:** Aligning phrase start points (verse, chorus, bridge) makes combinations work
- "Magic moments" happen when things align in ways that aren't fully understood yet
- Currently loses these moments because live mixing has momentum and mistakes happen
- Wants to analyze recordings to understand what made the magic

---

## Personality Notes for Next Claude

- CloudyCadet thinks in images and physical metaphors (punching bag, pendulums, negative space)
- Responds well to concrete examples over abstract theory
- Has limited time/usage, so be efficient but thorough
- Is excited about discovery — the fun is finding new things, not just building tools
- Has many projects in flight; this tool should be practical and achievable, not scope-creep into a massive engine
- Trust their intuitions — the "negative space" and "attractor" ideas came from real experience

---

## The Vision (Why This Matters)

We're not just building an audio tool. We're exploring whether:
- Harmony and rhythm are the same phenomenon at different timescales
- "Feel" and "groove" can be understood as phase relationships
- The space between tuning systems is where expressive music lives
- Tools can reveal structure that's hard to hear but easy to see

If we get this right, it could change how CloudyCadet (and others) understand and create music.

---

*Last updated: January 2025*
*Continue this conversation by referencing this document*

# Toward a Unified Framework: Continuous Pitch Space and Rhythm-Harmony Duality

*Working document — conceptual exploration*

---

## Part 1: The Problem with Discrete Systems

### 1.1 What the Book Gives Us

Cubarsi's framework operates entirely within n-TET systems:
- Pitch classes are elements of Zₙ (integers mod n)
- Chords are subsets
- Transformations are group actions

This is powerful for analyzing relationships **within** a system, but it can't express:
- Movement **between** systems
- Continuous pitch (what singers actually do)
- Why certain "wrong" notes sound "right"
- The influence of acoustic context on pitch choice

### 1.2 What Actually Happens

Real pitch space is **continuous**: any frequency is possible. Discrete systems (12-TET, 53-TET, just intonation) are **grids overlaid** on this continuum.

A performed pitch exists at a specific point in the continuum. Its "meaning" depends on:
- Which grid(s) it's near
- The harmonic context (what other pitches are sounding)
- The melodic context (what came before, what's expected next)
- Expressive intent

### 1.3 The Singer's Problem

When a singer chooses a pitch, they're solving an optimization problem (unconsciously):

```
minimize:  Σᵢ wᵢ · distance(pitch, attractor_i)

where attractors include:
  - 12-TET grid points (match the keyboard)
  - Just intonation points (match the overtones)
  - Melodic expectation (match the tune)
  - Expressive targets (bend for emotion)
```

The weights wᵢ shift dynamically based on context.

---

## Part 2: Attractors and Fields

### 2.1 Types of Attractors

**Harmonic Attractors (from sounding pitches)**

When a note sounds, it creates a harmonic series:
```
f, 2f, 3f, 4f, 5f, 6f, 7f, ...
```

Each harmonic is an attractor for other pitches. The strength of attraction might follow:
```
strength(harmonic n) ∝ 1/n  (or 1/n², depending on timbre)
```

When multiple notes sound (a chord), their harmonic series **interfere**:
- Reinforcement where harmonics align (strong attractors)
- Cancellation or ambiguity where they conflict

**Grid Attractors (from tuning systems)**

Each tuning system defines a set of "legal" pitches:
```
12-TET:  {0, 100, 200, 300, ..., 1100} cents
53-TET:  {0, 22.6, 45.3, 67.9, ...} cents
Just:    {0, 112, 204, 316, 386, 498, ...} cents (from C)
```

These are cognitive/cultural attractors — musicians learn to target them.

**Melodic Attractors (from expectation)**

Given a melodic context, certain continuations are expected:
- Leading tone → tonic (strong pull)
- Scalar neighbors (weak pull)
- Pattern completion (variable)

### 2.2 The Attractor Field

At any moment, the pitch continuum has an "attractor field" — a potential landscape where:
- Low points = stable pitches (strong attraction)
- High points = unstable pitches (weak attraction)
- The shape changes dynamically with context

```
Potential
    │
    │    ╱╲      ╱╲        ╱╲
    │   ╱  ╲    ╱  ╲      ╱  ╲
    │  ╱    ╲  ╱    ╲    ╱    ╲
    │ ╱      ╲╱      ╲  ╱      ╲
    │╱                ╲╱        ╲
    └─────────────────────────────── Pitch (cents)
         ^      ^           ^
         │      │           │
      12-TET   JI         12-TET
      + JI   alone        alone
    (strong) (medium)    (medium)
```

### 2.3 Field Dynamics

The field isn't static. As the harmonic context changes:

**Chord change:**
```
C major sounding → field favors {C, E, G} and their just relatives
F major sounding → field shifts to favor {F, A, C}
Transition → field morphs, creating "pathways" for voice leading
```

**Rhythmic position:**
```
Strong beat → field sharper (grid attractors stronger)
Weak beat → field softer (more freedom for expression)
Syncopation → field ambiguous (multiple valid interpretations)
```

---

## Part 3: The Rhythm-Harmony Duality

### 3.1 Frequency Ratios ARE Polyrhythms

This is the key insight. Consider a perfect fifth (3:2 ratio):

**In pitch:** 
- Note A vibrates at 440 Hz
- Note E vibrates at 660 Hz
- Ratio: 660/440 = 3/2

**In rhythm:**
- Pattern A: ● · · ● · · ● · · (period 3)
- Pattern B: ● · ● · ● · ● · ● (period 2)
- Together: a 3:2 polyrhythm

**They're the same mathematical object** at different timescales:
- Pitch: periods of ~1-10 milliseconds (20 Hz - 1000 Hz)
- Rhythm: periods of ~200-2000 milliseconds (0.5 Hz - 5 Hz)

### 3.2 The Tonnetz as a Rhythm Space

If we reinterpret the Tonnetz:
- Instead of pitch classes → rhythm periods
- Instead of intervals → period ratios
- Instead of chords → polyrhythmic cells

A "major triad" in rhythm space:
```
Periods in ratio 4:5:6

  ●···●···●···●···  (period 4)
  ●····●····●····●  (period 5)  
  ●·····●·····●····· (period 6)
```

The "consonance" of this polyrhythm corresponds to the consonance of the major chord!

### 3.3 Pendulums and Phase Space

A pendulum's motion can be described as:
```
θ(t) = A · cos(ωt + φ)

where:
  θ = angle
  A = amplitude
  ω = angular frequency (= 2π/period)
  φ = phase
```

Multiple pendulums create a trajectory in phase space:
```
(θ₁(t), θ₂(t), θ₃(t), ...)
```

**Key insight:** The "shape" of this trajectory depends only on the **ratios** of the frequencies, not the absolute frequencies.

A pendulum system with frequencies in ratio 4:5:6 traces the **same shape** whether those are:
- 4 Hz, 5 Hz, 6 Hz (rhythm range)
- 400 Hz, 500 Hz, 600 Hz (pitch range)

The shape IS the chord. The chord IS the polyrhythm.

### 3.4 Phase and "Inversion"

In harmony, chord "inversions" keep the same pitch classes but change the bass:
- C major: C-E-G (root position)
- C/E: E-G-C (first inversion)
- C/G: G-C-E (second inversion)

In rhythm, changing the **phase** of one oscillator against the others creates analogous variations:

```
Root position:   ●···●···●···●···  
                 ●····●····●····●  
                 ●·····●·····●·····

Phase shifted:   ●···●···●···●···  
                 ··●····●····●····  (shifted by 2)
                 ●·····●·····●·····
```

The "feel" changes even though the periods are identical. This is the rhythmic equivalent of inversion!

---

## Part 4: Toward a Unified Notation

### 4.1 Requirements

A good notation should express:
1. Pitch as continuous, with grid overlays
2. Which grids are "active" (contextually relevant)
3. Rhythm as the same structure at a different timescale
4. Phase relationships
5. Attractor strengths

### 4.2 Sketch: Continuous Pitch with Attractors

```
Pitch point:     p = 386¢
Grid context:    G = {12-TET, Just}
Nearest in each: 12-TET → 400¢ (E), Just → 386¢ (5/4)
Deviation:       δ₁₂ = -14¢, δⱼᵢ = 0¢

Notation:        E↓14 or (5/4)
                 └─ "E, 14 cents flat" or "just major third"
```

**For a sung pitch in context:**
```
Context: C major chord sounding (12-TET piano)
Singer:  Sings 388¢

Analysis:
  - 12-TET target: 400¢ (E)      → deviation -12¢
  - Just target:   386¢ (5:4)    → deviation +2¢
  - Singer is "between" but closer to just

Notation: E[−12|+2] or E⟨just−2⟩
```

### 4.3 Sketch: Unified Pitch-Rhythm

If pitch and rhythm are the same structure:

```
Harmonic object:  ratios {4, 5, 6} with phases {0, 0, 0}

As pitch:  4:5:6 above fundamental f
           → frequencies {4f, 5f, 6f}
           → "major triad in just intonation"

As rhythm: 4:5:6 polyrhythm at tempo t
           → periods {4t, 5t, 6t}
           → "major triad rhythm"

Notation:  ⟨4:5:6⟩ with context determining interpretation
           ⟨4:5:6⟩_pitch or ⟨4:5:6⟩_rhythm if ambiguous
```

**With phase:**
```
⟨4:5:6 | 0:2:0⟩  → second voice offset by 2 units
```

### 4.4 Sketch: Attractor Field Notation

To describe the current "pull" on a pitch:

```
Field at 390¢:
  A(12-TET, 400¢) = 0.6    → pulled toward 400
  A(Just, 386¢)   = 0.3    → pulled toward 386
  A(melodic)      = 0.1    → pulled toward 380 (descending tendency)

Net attractor: 395¢ with strength 0.7

Notation: 390¢ → ⟨395¢, 0.7⟩
          └─ "390 cents, attracted toward 395 with strength 0.7"
```

---

## Part 5: Open Questions

### 5.1 How to Measure Attractor Strength?

- Acoustic: amplitude of harmonics, roughness, beating
- Cognitive: familiarity, expectation, cultural training
- Contextual: what's sounding, what's implied

Can we derive attractor strength from first principles, or is it empirical?

### 5.2 Phase in Harmony

We said phase shift in rhythm ≈ inversion in harmony. But:
- Inversions have distinct harmonic functions
- Phase shifts in pitched sound create different timbres

Is there a deeper connection? When does phase "matter" vs "not matter"?

### 5.3 The Perception Boundary

Pitch: ~20 Hz to ~4000 Hz (melodic perception)
Rhythm: ~0.5 Hz to ~10 Hz (temporal perception)

Between ~10-20 Hz is a perceptual "gap" — too fast for rhythm, too slow for pitch. What happens in this region? Is this where "groove" lives?

### 5.4 Non-Integer Ratios

Just intonation uses integer ratios. 12-TET uses irrational ratios (12th root of 2).

- Do irrational ratios create "fuzzy" attractors?
- Is the slight detuning of 12-TET why it feels "alive" vs just intonation's "purity"?
- Can we model attractor width (not just position)?

### 5.5 Higher Dimensions

The Tonnetz for triads is 2D. For tetrads, 3D. For rhythm:
- 2 oscillators → 2D phase space → Lissajous figures
- 3 oscillators → 3D phase space → space curves
- 4+ → higher dimensional

Is there a natural limit? Do humans perceive relationships in >3 dimensions, or do we project down?

---

## Part 6: Experimental Directions

### 6.1 Capture Singer Deviation in Call-Response

**Setup:**
- Fixed instrumental "call" (piano, guitar, or synth)
- Singer responds with target pitch
- Record and analyze actual pitch (cent deviation)

**Variables:**
- Call tuning: 12-TET vs just intonation vs intentionally detuned
- Harmonic context: unison, fifth, third, seventh
- Rhythmic position: strong beat, weak beat, syncopation

**Hypothesis:** Singer deviation correlates with distance between grid attractors. When 12-TET and just are close (fifths), deviation is small. When far (thirds), deviation is larger and biased toward just.

### 6.2 Polyrhythm as Chord Perception

**Setup:**
- Generate polyrhythms from ratio sets: 4:5:6, 10:12:15, 4:5:6:7
- Play at different tempos (slow enough to perceive rhythm, fast enough to perceive pitch)
- Ask listeners: "Does this feel major? Minor? Dissonant?"

**Hypothesis:** Polyrhythm affect correlates with chord quality of the same ratios. A 4:5:6 polyrhythm should feel "bright" like a major chord.

### 6.3 Attractor Visualization

**Setup:**
- Real-time analysis of pitch input (microphone)
- Display "attractor field" showing:
  - Current pitch as a point
  - Nearby grid points as wells
  - Strength of attraction based on harmonic context

**Use:** Training tool for singers/instrumentalists to see which system they're locking to.

---

## Summary: Where We Are

**Established:**
- Cubarsi gives us rigorous math for discrete systems
- Singers/performers navigate continuous space
- Harmonic context creates attractors
- Rhythm and pitch are mathematically dual (same ratios, different timescales)

**Proposed:**
- Attractor field model for continuous pitch
- Unified notation for pitch-rhythm
- Phase as the rhythmic analog of inversion

**To Explore:**
- Empirical measurement of attractor strength
- The perception boundary between rhythm and pitch
- Higher-dimensional relationships
- Call-response as a test bed

---

*This is a working document. The math isn't rigorous yet — we're finding the right concepts before formalizing.*

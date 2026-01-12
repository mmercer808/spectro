 # Chord Transformations: Mathematical Cheatsheet

*Distilled from Cubarsi's "Chord Transformations in Higher-Dimensional Networks" (2025)*

---

## 1. Foundational Structures

### 1.1 The Ambient Space: n-TET

An **n-TET system** (n-tone equal temperament) lives in **Zₙ** — integers mod n.

```
Notes:     {0, 1, 2, ..., n-1}
Intervals: addition in Zₙ
Octave:    equivalence class (adding n = identity)
```

**Common systems:**
| System | n | Notes |
|--------|---|-------|
| Standard Western | 12 | C, C♯, D, ..., B |
| Quarter-tone | 24 | Microtonal |
| 31-TET | 31 | Better thirds approximation |
| 53-TET | 53 | Near-just intonation |

### 1.2 Modes

A **k-mode** μ is an ordered partition of n into k positive integers:

```
μ = [A₀, A₁, ..., Aₖ₋₁]   where   Σᵢ Aᵢ = n
```

Each Aᵢ is an **interval** (step size in semitones for 12-TET).

**Examples in 12-TET:**
| Mode | Intervals | Musical Name |
|------|-----------|--------------|
| [4, 3, 5] | maj3, min3, P4 | Major triad |
| [3, 4, 5] | min3, maj3, P4 | Minor triad |
| [4, 4, 4] | maj3, maj3, maj3 | Augmented triad |
| [3, 3, 6] | min3, min3, tritone | Diminished triad |
| [4, 3, 3, 2] | — | Dominant 7th |
| [4, 3, 4, 1] | — | Major 7th |

**Notation shorthand:** [A, B, *] means the last interval is determined by n - A - B.

### 1.3 Directed Chords

A **directed chord** is a root note plus a mode:

```
a = a₀ | μ = a₀ | [A₀, A₁, ..., Aₖ₋₁]
```

This generates notes by cumulative addition:

```
Notes: (a₀, a₁, a₂, ..., aₖ₋₁)
where  aᵢ = a₀ + A₀ + A₁ + ... + Aᵢ₋₁  (mod n)
```

**Example:** `0 | [4, 3, 5]` in 12-TET
```
a₀ = 0           → C
a₁ = 0 + 4 = 4   → E  
a₂ = 4 + 3 = 7   → G
(a₃ = 7 + 5 = 12 ≡ 0 → back to C)
```

### 1.4 Chords (Equivalence Classes)

A **chord** {a} is the equivalence class of all rotations of a directed chord:

```
{a} = {a₀, a₁, ..., aₖ₋₁}  (unordered set)
```

Multiple directed chords can represent the same chord:
```
{0 | [4,3,5]} = {4 | [3,5,4]} = {7 | [5,4,3]} = {0, 4, 7}
```

---

## 2. Counting Formulas

### 2.1 Number of k-chords in n-TET

```
NK(n, k) = C(n, k) = n! / (k!(n-k)!)
```

| n-TET | k=3 (triads) | k=4 (tetrads) | k=5 | k=6 |
|-------|--------------|---------------|-----|-----|
| 12 | 220 | 495 | 792 | 924 |
| 24 | 2,024 | 10,626 | 42,504 | 134,596 |
| 53 | 23,426 | 292,825 | 2,869,685 | 22,957,480 |

### 2.2 Number of k-modes in n-TET

```
NM(n, k) = C(n-1, k-1) = (n-1)! / ((k-1)!(n-k)!)
```

| n-TET | k=3 | k=4 | k=5 | k=6 |
|-------|-----|-----|-----|-----|
| 12 | 55 | 165 | 330 | 462 |
| 53 | 1,326 | 22,100 | 270,725 | 2,598,960 |

### 2.3 Mode Classes

A **mode class** is an equivalence class under shifts (cyclic permutations).

For a mode μ with all distinct intervals:
```
Number of shifts: s(μ) = k
Number of mode classes: NM(n,k) / k
```

For modes with repeated intervals, some shifts coincide, reducing the class size.

**12-TET trimode classes:** 19 total
- 18 classes with 3 distinct shifts each
- 1 class with 1 shift: [4, 4, 4] (augmented)

### 2.4 Symmetric Modes

For a mode μ = [A₀, ..., Aₖ₋₁], the **symmetric modes** are all permutations in Sₖ.

Number of mode classes for symmetric modes: k! / k = (k-1)!

| k | Symmetric mode classes |
|---|------------------------|
| 3 | 2 (e.g., [A,B,C] and [A,C,B]) |
| 4 | 6 |
| 5 | 24 |
| 6 | 120 |

---

## 3. Operations on Roots

### 3.1 Translation τᵤ

Shifts the root by u semitones:

```
τᵤ(a₀ | μ) = (a₀ + u) | μ
```

**Properties:**
```
τᵤ ∘ τᵥ = τᵤ₊ᵥ           (composition)
τ₀ = ε (identity)         (identity)
τᵤ⁻¹ = τ₋ᵤ = τₙ₋ᵤ        (inverse)
τᵤ ∘ τᵥ = τᵥ ∘ τᵤ        (commutativity)
```

The translations form a **cyclic group** isomorphic to (Zₙ, +).

### 3.2 Inversion ι

Negates the root:

```
ι(a₀ | μ) = (-a₀) | μ = (n - a₀) | μ
```

**Properties:**
```
ι² = ε                    (involution)
ι ∘ τᵤ = τ₋ᵤ ∘ ι          (non-commutativity!)
```

### 3.3 Combined Group Structure

Translations and inversion together form a **semidirect product**:

```
G = Zₙ ⋊ Z₂
```

This is the **dihedral group** D₂ₙ (symmetries of a regular n-gon).

---

## 4. Operations on Modes

### 4.1 Shift sⁱ

Cyclic permutation of intervals by i positions:

```
s(μ) = s([A₀, A₁, ..., Aₖ₋₁]) = [A₁, A₂, ..., Aₖ₋₁, A₀]
sⁱ(μ) = [Aᵢ, Aᵢ₊₁, ..., Aᵢ₋₁]  (indices mod k)
```

**Properties:**
```
sᵏ = ε                    (period k)
s⁻¹ = sᵏ⁻¹                (inverse)
```

Shifts form a **cyclic group** Zₖ.

### 4.2 Retrogradation r

Reverses the mode:

```
r([A₀, A₁, ..., Aₖ₋₁]) = [Aₖ₋₁, Aₖ₋₂, ..., A₀]
```

**Properties:**
```
r² = ε                    (involution)
r ∘ s = s⁻¹ ∘ r           (non-commutativity)
```

### 4.3 Negative Mode

```
-μ = -[A₀, A₁, ..., Aₖ₋₁] = [-A₀, -A₁, ..., -Aₖ₋₁]
```

where -Aᵢ is computed mod n.

**Key relationship:** -μ = r(μ) when applied to chords.

### 4.4 Transpositions σᵢ

Swap adjacent intervals:

```
σᵢ([..., Aᵢ, Aᵢ₊₁, ...]) = [..., Aᵢ₊₁, Aᵢ, ...]
```

**Properties:**
```
σᵢ² = ε                   (involution)
σᵢσⱼ = σⱼσᵢ  if |i-j| > 1 (distant commute)
σᵢσᵢ₊₁σᵢ = σᵢ₊₁σᵢσᵢ₊₁      (braid relation)
```

The transpositions generate the **symmetric group** Sₖ.

---

## 5. Rotations and the Fundamental Theorem

### 5.1 Chord Rotation Rʲ

A rotation combines a shift with a translation:

```
Rʲ(a₀ | μ) = (a₀ + [μ]₀ʲ) | sʲ(μ)
```

where [μ]₀ʲ = A₀ + A₁ + ... + Aⱼ₋₁ (sum of first j intervals).

**This is crucial:** Rotation changes both root AND mode, but produces the same chord.

### 5.2 Rotation Properties

```
Rᵏ = ε                    (period k)
{Rʲ(a₀ | μ)} = {a₀ | μ}   (same chord)
```

The k rotations of a directed chord are:
```
R⁰(a) = a₀ | [A₀, A₁, ..., Aₖ₋₁]
R¹(a) = a₁ | [A₁, A₂, ..., A₀]
R²(a) = a₂ | [A₂, A₃, ..., A₁]
...
```

---

## 6. The Drift Operators (Main Innovation)

### 6.1 Definition

The **drift** Dᵢ combines translation, shift, and transposition:

```
Dᵢ(a₀ | μ) = τ_Aᵢ ∘ s⁻ⁱ ∘ σᵢ ∘ sⁱ (a₀ | μ)
```

Explicitly:
```
Dᵢ(a₀ | [A₀, ..., Aᵢ, Aᵢ₊₁, ..., Aₖ₋₁]) = (a₀ + Aᵢ) | [A₀, ..., Aᵢ₊₁, Aᵢ, ..., Aₖ₋₁]
```

It does THREE things:
1. Translates root by Aᵢ
2. Swaps intervals Aᵢ and Aᵢ₊₁
3. Changes the mode class (if Aᵢ ≠ Aᵢ₊₁)

### 6.2 Drift Properties

```
Dᵢ² = ε                   (involution)
DᵢDⱼ = DⱼDᵢ  if |i-j| > 1 (distant commute)
DᵢDᵢ₊₁Dᵢ = Dᵢ₊₁DᵢDᵢ₊₁      (braid relation)
```

These are **Coxeter group** relations! The drifts generate Sₖ.

### 6.3 Relationship to Neo-Riemannian P, R, L

For **trichords** in 12-TET with mode [4, 3, 5] (major triad):

| Drift | Effect | Neo-Riemannian | Example |
|-------|--------|----------------|---------|
| D₀ | Swap A₀↔A₁, translate by A₀ | L (Leittonwechsel) | C → Em |
| D₁ | Swap A₁↔A₂, translate by A₁ | R (Relative) | C → Am |
| D₀D₁D₀ | Composite | P (Parallel) | C → Cm |

### 6.4 Edge Transformations

Within a chord cell, drifts move along edges:

```
Dᵢ: moves along edge orthogonal to direction Aᵢ
```

The **interior drifts** D₀, D₁, ..., Dₖ₋₃ stay within the chord cell.
The **exterior drift** Dₖ₋₂ moves to an adjacent (non-congruent) cell.

---

## 7. Geometric Structures

### 7.1 Tonnetz (Tone Network)

A (k-1)-dimensional lattice embedded in a (k-1)-torus.

**Construction:**
- Vertices: notes (pitch classes)
- Edges: intervals between notes
- k-1 independent directions from mode intervals A₀, ..., Aₖ₋₂
- The k-th interval Aₖ₋₁ is dependent (returns to start)

**For trichords (k=3):** 2D triangular lattice on a torus
**For tetrachords (k=4):** 3D lattice on a 3-torus

### 7.2 Simplicial Structure of Chords

A **k-chord** is a **(k-1)-simplex**:

| k | Simplex | Faces | Edges | Vertices |
|---|---------|-------|-------|----------|
| 2 | edge | 1 | 1 | 2 |
| 3 | triangle | 3 | 3 | 3 |
| 4 | tetrahedron | 4 | 6 | 4 |
| 5 | 4-simplex | 5 | 10 | 5 |

General formula for m-faces: C(k, m+1)

### 7.3 Tonal Cell

The **tonal cell** around a root a₀ contains all notes reachable by mode intervals.

**Number of vertices (excluding root):**
```
|Tonal cell| = 2ᵏ - 2
```

| k | Vertices | Shape |
|---|----------|-------|
| 3 | 6 | hexagon |
| 4 | 14 | tetrakis hexahedron |
| 5 | 30 | — |

### 7.4 Chord Cell

The **chord cell** is dual to the tonal cell — vertices are chords, faces are notes.

**Number of chords per cell:**
```
|Chord cell| = k!
```

| k | Chords | Polytope |
|---|--------|----------|
| 3 | 6 | hexagon |
| 4 | 24 | truncated octahedron |
| 5 | 120 | — |

### 7.5 The Truncated Octahedron (k=4)

The chord cell for tetrachords:
- **24 vertices** (chords sharing a root)
- **36 edges** (chord adjacencies)
- **14 faces**: 6 squares + 8 hexagons
- **Tessellates** 3D space (honeycomb)

Square faces: orthogonal to main directions (±A, ±B, ±C)
Hexagonal faces: orthogonal to diagonal directions

---

## 8. Circuits and Paths

### 8.1 Simple Circuit

A **simple circuit** is a closed path visiting each vertex exactly once.

For chord cells, these are **Hamiltonian circuits** on the polytope.

**Tetrachord example (truncated octahedron):**
```
D₂D₁D₀D₁D₀D₁D₂D₁D₂D₁D₀D₁D₀D₁D₂D₁D₂D₁D₀D₁D₀D₁D₂D₁
```

### 8.2 Shortcut Circuit

A circuit that crosses between chord cells (change of tonality).

Condition: Find minimum j where j·Aᵢ ≡ 0 (mod n)

These generalize **maximally smooth cycles** from neo-Riemannian theory.

### 8.3 Voice Leading Distance

Adjacency on the chord network corresponds to **minimal voice leading**:
- Adjacent chords differ by exactly one note
- Graph distance = number of note changes needed

---

## 9. Algebraic Summary

### 9.1 Groups Acting on Directed Chords

| Group | Generators | Action | Structure |
|-------|------------|--------|-----------|
| Translations | τ₁ | On root | Zₙ |
| Inversion | ι | On root | Z₂ |
| Shifts | s | On mode | Zₖ |
| Retrogradation | r | On mode | Z₂ |
| Transpositions | σᵢ | On mode | Sₖ |
| Drifts | Dᵢ | On both | Sₖ (Coxeter presentation) |

### 9.2 Full Symmetry Group

The complete group of chord transformations:
```
G = (Zₙ ⋊ Z₂) × (Sₖ ⋊ Z₂)
```

Where:
- Zₙ ⋊ Z₂ acts on roots (dihedral group)
- Sₖ ⋊ Z₂ acts on modes (hyperoctahedral group)

### 9.3 Key Identities

**Rotation decomposition:**
```
Rʲ = τ[μ]₀ʲ ∘ sʲ
```

**Drift decomposition:**
```
Dᵢ = τ_Aᵢ ∘ s⁻ⁱ ∘ σᵢ ∘ sⁱ
```

**Chord equivalence:**
```
{a₀ | μ} = {a₀ | sʲ(μ)} iff τ[μ]₀ʲ = ε
```

---

## 10. Computational Formulas

### 10.1 Notes from Directed Chord

```python
def chord_notes(root, mode, n):
    """Generate notes from root and mode."""
    notes = [root]
    current = root
    for interval in mode[:-1]:  # Last interval returns to root
        current = (current + interval) % n
        notes.append(current)
    return notes
```

### 10.2 All Rotations

```python
def rotations(root, mode, n):
    """Generate all rotations of a directed chord."""
    k = len(mode)
    result = []
    current_root = root
    current_mode = mode[:]
    
    for _ in range(k):
        result.append((current_root, current_mode[:]))
        # Rotate: new_root = old_root + first_interval
        current_root = (current_root + current_mode[0]) % n
        # Shift mode left
        current_mode = current_mode[1:] + [current_mode[0]]
    
    return result
```

### 10.3 Drift Operator

```python
def drift(root, mode, i, n):
    """Apply drift Dᵢ to directed chord."""
    new_mode = mode[:]
    # Swap intervals at positions i and i+1 (mod k)
    k = len(mode)
    j = (i + 1) % k
    new_mode[i], new_mode[j] = new_mode[j], new_mode[i]
    # Translate root by original interval at position i
    new_root = (root + mode[i]) % n
    return new_root, new_mode
```

### 10.4 Mode Class Enumeration

```python
def mode_classes(n, k):
    """Enumerate all mode classes for n-TET, k-chords."""
    from itertools import combinations_with_replacement
    
    # Generate all modes (compositions of n into k parts)
    def compositions(total, parts):
        if parts == 1:
            yield [total]
        else:
            for i in range(1, total - parts + 2):
                for rest in compositions(total - i, parts - 1):
                    yield [i] + rest
    
    seen = set()
    classes = []
    
    for mode in compositions(n, k):
        # Canonical form: lexicographically smallest rotation
        rotations = [tuple(mode[i:] + mode[:i]) for i in range(k)]
        canonical = min(rotations)
        
        if canonical not in seen:
            seen.add(canonical)
            classes.append(list(canonical))
    
    return classes
```

### 10.5 Adjacency Check

```python
def are_adjacent(chord1, chord2):
    """Check if two chords share k-1 notes."""
    set1, set2 = set(chord1), set(chord2)
    return len(set1 & set2) == len(chord1) - 1
```

---

## 11. Quick Reference Tables

### 11.1 12-TET Triads

| Mode | Class | Name | Example (root 0) |
|------|-------|------|------------------|
| [4,3,5] | μ | Major | {0,4,7} = C |
| [3,5,4] | μ | Major (1st inv) | — |
| [5,4,3] | μ | Major (2nd inv) | — |
| [3,4,5] | ν | Minor | {0,3,7} = Cm |
| [4,5,3] | ν | Minor (1st inv) | — |
| [5,3,4] | ν | Minor (2nd inv) | — |
| [4,4,4] | — | Augmented | {0,4,8} = C+ |
| [3,3,6] | — | Diminished | {0,3,6} = C° |

### 11.2 Drift Actions on Major/Minor

Starting from C major = 0 | [4,3,5]:

| Operation | Result | Notes | Name |
|-----------|--------|-------|------|
| D₀ | 4 \| [3,4,5] | {4,7,11} | E minor |
| D₁ | 3 \| [4,5,3] | {3,7,0} | — |
| D₀D₁ | 7 \| [5,4,3] | {7,0,4} | — |
| D₁D₀ | 7 \| [3,5,4] | {7,10,3} | — |
| D₀D₁D₀ | 0 \| [3,4,5] | {0,3,7} | C minor |

### 11.3 Polytope Summary

| k | Tonal Cell | Chord Cell | Tessellation |
|---|------------|------------|--------------|
| 3 | Hexagon (6) | Hexagon (6) | 2D honeycomb |
| 4 | Tetrakis hex (14) | Truncated oct (24) | 3D bitruncated cubic |
| 5 | 30 vertices | 120 vertices | 4D honeycomb |

---

## Appendix: Notation Summary

| Symbol | Meaning |
|--------|---------|
| n | Number of tones in TET system |
| k | Number of notes in chord |
| Zₙ | Integers mod n |
| μ, ν | Modes |
| [A₀,...,Aₖ₋₁] | Mode as interval array |
| a₀ \| μ | Directed chord (root + mode) |
| {a} | Chord (equivalence class) |
| τᵤ | Translation by u |
| ι | Root inversion |
| s, sⁱ | Mode shift |
| r | Mode retrogradation |
| σᵢ | Transposition (swap i, i+1) |
| Dᵢ | Drift operator |
| Rʲ | Rotation |
| [μ]₀ʲ | Sum of first j intervals |
| Sₖ | Symmetric group on k elements |

---

*Last updated: January 2025*

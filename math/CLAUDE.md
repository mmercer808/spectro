# CLAUDE.md

## Project: Beat Detection via Harmonic Oscillator Matching

---

## Overview

This is a real-time beat detection library using adversarial oscillator banks. The same input sample drives both onset detection and branch modulation — this is a core architectural constraint.

**Do not break the single-sample principle.** Every path through the system must trace back to `x[n]`.

---

## Architecture

```
x[n] → STFT → S[m,k] → O[m] → {tᵢ} → (tₐ,tᵦ) → θₖ → Mₐ,Mᵦ → A(t) → conf(t)
```

### Modules

| Module | Responsibility | Key Equations |
|--------|----------------|---------------|
| `context.py` | Shared state, sample rate, buffer size | — |
| `stft.py` | Short-time Fourier transform | `X[m,k] = Σ x[n+mH]·w[n]·exp(-2πikn/N)` |
| `onset.py` | Spectral flux, peak detection | `O[m] = Σₖ max(0, S[m,k] - S[m-1,k])` |
| `frequency.py` | Spectral centroid, tempo estimation | `C = Σ fₖS[m,k] / Σ S[m,k]` |
| `selector.py` | Call-response point selection | Minimize `L(tₐ,tᵦ)` under constraints |
| `oscillator.py` | Harmonic oscillator banks | `θₖ[n+1] = θₖ[n] + 2πkf₀/fₛ + εₖ·s[n]·sin(θₖ - θ_target)` |
| `mixer.py` | Additive and multiplicative mixing | `cos(α)cos(β) = ½[cos(α-β) + cos(α+β)]` |
| `agreement.py` | Adversarial verification, confidence | `A(t) = cos(θ₁⁽ᴬ⁾ - θ₁⁽ᴮ⁾)` |
| `pipeline.py` | Data flow orchestration | Chains all modules |

---

## Context Object

All modules receive a shared `Context` instance. Never use global state.

```python
@dataclass
class Context:
    sample_rate: int = 44100
    buffer_size: int = 1024
    hop_size: int = 256
    fft_size: int = 2048
    num_harmonics: int = 8
    coupling_strength: float = 0.1
```

Pass `ctx` to every class constructor.

---

## Critical Constraints

### 1. Single Sample Principle
The same `x[n]` that triggers onset detection must modulate the oscillators. No separate buffers, no delayed copies.

### 2. Causality
Modulation cannot precede detection. The inherent latency is `(N-1)/fₛ` seconds from the STFT. Do not add more.

### 3. Harmonic Series
Oscillator banks use integer multiples of the fundamental: `fₖ = k·f₀`. Do not use inharmonic frequencies.

### 4. Phase Initialization
When spawning an oscillator at onset time `tₐ`:
```python
phi_k = -2 * np.pi * k * f0 * t_a
```
This ensures `θₖ(tₐ) = 0`.

### 5. Backbeat Detection
For half-period offset (`t_b - t_a = T/2`), the agreement function must yield `A ≈ -1`. If it doesn't, something is wrong.

---

## Coding Conventions

### Style
- Python 3.10+
- Type hints on all function signatures
- Dataclasses for structured state
- NumPy for signal processing
- No global mutable state

### Naming
- `x` or `signal` for time-domain audio
- `X` for frequency-domain (STFT output)
- `S` for spectrogram (power)
- `O` for onset strength
- `theta` or `θ` for oscillator phase
- `M` for mixer output
- `A` for agreement

### Files
- One class per file for core modules
- Tests in `tests/` mirroring `src/` structure
- Math reference in `docs/signal_path_math.md`

---

## Do NOT

- Use global variables for shared state
- Add latency beyond the STFT window
- Mix sample rates within the pipeline
- Use inharmonic oscillator frequencies
- Forget to pass `ctx` to new classes
- Break the single-sample flow

---

## Testing Strategy

### Unit Tests
Each module gets isolated tests with synthetic signals:
- `test_stft.py` — verify Parseval's theorem holds
- `test_onset.py` — click track should produce clean peaks
- `test_oscillator.py` — phase should wrap correctly at 2π
- `test_mixer.py` — verify product-to-sum identity numerically
- `test_agreement.py` — backbeat signals should yield A ≈ -1

### Integration Tests
- Full pipeline with known BPM input
- Verify detected tempo within ±1 BPM
- Verify phase lock within 10ms

### Synthetic Test Signals
```python
def click_track(bpm, duration, sr=44100):
    """Generate impulse train at given tempo."""
    period_samples = int(60 * sr / bpm)
    n_samples = int(duration * sr)
    signal = np.zeros(n_samples)
    signal[::period_samples] = 1.0
    return signal
```

---

## Implementation Order

1. `context.py` — get shared state working
2. `stft.py` — verify against `scipy.signal.stft`
3. `onset.py` — test with click track
4. `frequency.py` — verify centroid on pure tones
5. `oscillator.py` — single oscillator first, then bank
6. `mixer.py` — test identities numerically
7. `selector.py` — mock onset data first
8. `agreement.py` — test with known phase offsets
9. `pipeline.py` — wire everything together

---

## Key Equations Reference

Keep `docs/signal_path_math.md` loaded when implementing. Every function should trace to an equation in that document.

### Most Important

**Spectral flux:**
```
O[m] = Σₖ max(0, S[m,k] - S[m-1,k])
```

**Oscillator update:**
```
θₖ[n+1] = θₖ[n] + 2πkf₀/fₛ + εₖ·s[n]·sin(θₖ[n] - θ_target)
```

**Mixer identity:**
```
cos(α)cos(β) = ½[cos(α-β) + cos(α+β)]
```

**Agreement:**
```
A(t) = cos(θ₁⁽ᴬ⁾ - θ₁⁽ᴮ⁾)
```

---

## Questions to Ask Before Implementing

1. Does this module receive the Context?
2. Which equation from the math doc does this implement?
3. Does this preserve causality?
4. Is the sample flow unbroken from `x[n]`?
5. Are there tests with synthetic signals?

---

## Example Module Structure

```python
# src/onset.py

from dataclasses import dataclass
import numpy as np
from .context import Context

@dataclass
class OnsetResult:
    times: np.ndarray      # onset times in seconds
    strengths: np.ndarray  # onset strengths O[mᵢ]
    frames: np.ndarray     # onset frame indices

class OnsetDetector:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.threshold = 0.1
        self._prev_spectrum = None
    
    def spectral_flux(self, S: np.ndarray) -> float:
        """
        Compute onset strength for current frame.
        
        Equation: O[m] = Σₖ max(0, S[m,k] - S[m-1,k])
        """
        if self._prev_spectrum is None:
            self._prev_spectrum = S
            return 0.0
        
        diff = S - self._prev_spectrum
        flux = np.sum(np.maximum(0, diff))
        self._prev_spectrum = S.copy()
        return flux
    
    def detect(self, flux_history: np.ndarray) -> OnsetResult:
        """Find peaks in onset strength function."""
        # Implementation here
        pass
```

---

## When Stuck

1. Check the math doc — is the equation implemented correctly?
2. Test with a pure tone or click track — simplest possible input
3. Print intermediate values — where does it diverge from expected?
4. Check phase wrapping — should be modulo 2π
5. Verify sample rate consistency — mixed rates cause subtle bugs

---

*Last updated: January 2026*

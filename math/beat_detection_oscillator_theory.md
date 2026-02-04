# Beat Detection via Harmonic Oscillator Matching

## A Mathematical Framework for Adversarial Branch Detection

---

**Preliminary Draft**  
**Status:** Working document for formalization

---

## 1. Introduction

This document establishes the mathematical foundation for a beat detection system based on harmonic oscillator matching. The core principle: a single input sample simultaneously drives onset detection and branch modulation, ensuring temporal coherence across the detection pipeline.

### 1.1 Design Philosophy

The system treats rhythm as a coupled oscillator problem. Rather than detecting beats as isolated events, we model the underlying periodic structure and use adversarial verification between oscillator banks to establish confidence.

### 1.2 Key Components

1. **Onset Detection** — identifies energy transients in the signal
2. **Frequency Detection** — extracts spectral information at onset points
3. **Harmonic Branching** — spawns oscillator series from detected points
4. **Adversarial Mixing** — two branches verify each other's predictions
5. **Branch Modulation** — the same sample that triggers detection modulates the oscillator functions

---

## 2. Signal Representation

### 2.1 The Input Sample

Let $x[n]$ be the discrete input signal at sample index $n$, with sample rate $f_s$.

**Critical constraint:** The same sample $x[n]$ must be available to both:
- The onset detection subsystem
- The branch modulation subsystem

This ensures zero latency between detection and modulation.

### 2.2 Short-Time Fourier Transform

The time-frequency representation is given by:

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-2\pi i k n / N}$$

where:
- $m$ is the frame index
- $k$ is the frequency bin index
- $H$ is the hop size (frame advance)
- $w[n]$ is the window function (Hann recommended)
- $N$ is the FFT size

The spectrogram magnitude:

$$S[m, k] = |X[m, k]|^2$$

---

## 3. Onset Detection

### 3.1 Spectral Flux

**Definition:** The onset strength function measures the rate of increase in spectral energy.

$$O[m] = \sum_{k=0}^{N/2} \max\left(0, S[m, k] - S[m-1, k]\right)$$

The half-wave rectification (max with 0) ensures we only respond to energy increases, not decreases.

### 3.2 Proof: Relationship Between Onset and Frequency Content

**Theorem 3.1:** The onset strength $O[m]$ is bounded by the total spectral energy change and is maximized when new frequency components appear.

**Proof:**

Let $\Delta S[m, k] = S[m, k] - S[m-1, k]$.

Then:

$$O[m] = \sum_k \max(0, \Delta S[m, k]) \leq \sum_k |\Delta S[m, k]|$$

Equality holds when all $\Delta S[m, k] \geq 0$, i.e., when energy increases across all bins.

For a pure tone onset at frequency $f_0$, the energy concentrates in bin $k_0 = \lfloor f_0 N / f_s \rfloor$:

$$O[m] \approx S[m, k_0] - S[m-1, k_0]$$

Thus onset strength directly reflects the magnitude of new frequency content. $\square$

### 3.3 Peak Detection

Onsets occur at local maxima of $O[m]$:

$$\text{onset at } m \iff O[m] > O[m-1] \land O[m] > O[m+1] \land O[m] > \theta$$

where $\theta$ is the detection threshold.

Let the detected onset times be $\{t_1, t_2, \ldots, t_n\}$ where $t_i = m_i H / f_s$.

---

## 4. Frequency Detection at Onset Points

### 4.1 Spectral Centroid

At each onset $t_i$, compute the spectral centroid:

$$C_i = \frac{\sum_{k=0}^{N/2} f_k \cdot S[m_i, k]}{\sum_{k=0}^{N/2} S[m_i, k]}$$

where $f_k = k f_s / N$ is the frequency of bin $k$.

### 4.2 Proof: Centroid as Weighted Frequency Average

**Theorem 4.1:** The spectral centroid $C_i$ represents the power-weighted mean frequency at onset $t_i$.

**Proof:**

Define the normalized spectral distribution:

$$p_k = \frac{S[m_i, k]}{\sum_j S[m_i, j]}$$

This is a valid probability distribution: $p_k \geq 0$ and $\sum_k p_k = 1$.

Then:

$$C_i = \sum_k f_k \cdot p_k = \mathbb{E}[f]$$

The centroid is the expected value of frequency under the power distribution. $\square$

### 4.3 Base Frequency Estimation

For beat detection, we need the rhythmic frequency (tempo), not the spectral centroid.

**From onset intervals:**

$$\hat{f}_0 = \frac{n-1}{\sum_{i=1}^{n-1}(t_{i+1} - t_i)} = \frac{n-1}{t_n - t_1}$$

This estimates beats per second. Convert to BPM: $\text{BPM} = 60 \hat{f}_0$.

---

## 5. Call-Response Point Selection

### 5.1 Selection Criteria

To spawn adversarial branches, select two onset points $(t_a, t_b)$ satisfying:

**Criterion 1 — Strength Balance:**
$$|O[m_a] - O[m_b]| < \epsilon_s$$

**Criterion 2 — Spectral Contrast:**
$$|C_a - C_b| > \epsilon_c$$

**Criterion 3 — Temporal Relationship:**
$$\left| (t_b - t_a) - \frac{T}{2} \right| < \epsilon_t$$

where $T = 1/\hat{f}_0$ is the estimated beat period.

### 5.2 Interpretation

These criteria select complementary rhythmic elements:
- Similar strength → balanced call and response
- Different spectra → distinct instruments (kick vs snare)
- Half-period separation → backbeat relationship

---

## 6. Harmonic Oscillator Banks

### 6.1 Oscillator Definition

Each branch spawns a bank of harmonically-related oscillators.

**Branch A (the "call"):**

$$\theta_k^{(A)}(t) = 2\pi k f_0 t + \phi_k^{(A)}, \quad k = 1, 2, 3, \ldots, K$$

**Branch B (the "response"):**

$$\theta_k^{(B)}(t) = 2\pi k f_0 t + \phi_k^{(B)}, \quad k = 1, 2, 3, \ldots, K$$

### 6.2 Phase Initialization

Set phases so oscillators peak at their respective onset times:

$$\phi_k^{(A)} = -2\pi k f_0 t_a$$
$$\phi_k^{(B)} = -2\pi k f_0 t_b$$

This ensures $\theta_k^{(A)}(t_a) = 0$ and $\theta_k^{(B)}(t_b) = 0$.

### 6.3 Oscillator Output

The output of oscillator $k$ in branch $A$:

$$x_k^{(A)}(t) = a_k \cos(\theta_k^{(A)}(t))$$

The coefficients $a_k$ control the harmonic weighting.

### 6.4 Proof: Harmonic Series Forms Fourier Basis

**Theorem 6.1:** The harmonic oscillator bank with fundamental $f_0$ spans the space of $f_0$-periodic functions.

**Proof:**

Any $T$-periodic function $g(t)$ where $T = 1/f_0$ can be written:

$$g(t) = \frac{a_0}{2} + \sum_{k=1}^{\infty} \left[ a_k \cos(2\pi k f_0 t) + b_k \sin(2\pi k f_0 t) \right]$$

Our oscillator bank (with appropriate phase shifts) generates exactly these basis functions. $\square$

---

## 7. The Mixer

### 7.1 Bank Summation

**Branch A output:**

$$M_A(t) = \sum_{k=1}^{K} a_k \cos(\theta_k^{(A)}(t))$$

**Branch B output:**

$$M_B(t) = \sum_{k=1}^{K} b_k \cos(\theta_k^{(B)}(t))$$

### 7.2 Additive Mixing

$$M_{add}(t) = M_A(t) + M_B(t)$$

This preserves both rhythmic patterns. The result is still a harmonic series with fundamental $f_0$.

### 7.3 Multiplicative Mixing (Ring Modulation)

$$M_{mult}(t) = M_A(t) \cdot M_B(t)$$

**Theorem 7.1:** Multiplicative mixing of two harmonic series produces sum and difference frequencies.

**Proof:**

Consider a single term:

$$\cos(\theta_k^{(A)}) \cos(\theta_j^{(B)}) = \frac{1}{2}\left[ \cos(\theta_k^{(A)} - \theta_j^{(B)}) + \cos(\theta_k^{(A)} + \theta_j^{(B)}) \right]$$

The difference frequency term:

$$\theta_k^{(A)} - \theta_j^{(B)} = 2\pi(k - j)f_0 t + (\phi_k^{(A)} - \phi_j^{(B)})$$

When $k = j$:

$$\theta_k^{(A)} - \theta_k^{(B)} = \phi_k^{(A)} - \phi_k^{(B)} = -2\pi k f_0 (t_a - t_b)$$

This is a constant phase offset encoding the temporal relationship between call and response. $\square$

### 7.4 Proof: Phase Difference Encodes Beat Structure

**Theorem 7.2:** For backbeat patterns where $t_b - t_a = T/2$, the fundamental phase difference is $\pi$.

**Proof:**

$$\phi_1^{(A)} - \phi_1^{(B)} = -2\pi f_0 t_a + 2\pi f_0 t_b = 2\pi f_0 (t_b - t_a) = 2\pi f_0 \cdot \frac{T}{2} = \pi$$

Therefore $\cos(\phi_1^{(A)} - \phi_1^{(B)}) = \cos(\pi) = -1$.

The multiplicative mixer outputs a negative DC component for perfect backbeat alignment. $\square$

---

## 8. Branch Modulation from Input Sample

### 8.1 The Modulation Requirement

The same sample $x[n]$ that triggers onset detection must modulate the oscillator functions.

**Modulation signal:**

$$s[n] = O[m(n)] \cdot \delta[n - n_{onset}]$$

where $m(n) = \lfloor n / H \rfloor$ maps sample index to frame index, and $\delta$ is the Kronecker delta.

### 8.2 Coupled Oscillator Dynamics

The oscillator phase evolves according to:

$$\frac{d\theta_k^{(A)}}{dt} = 2\pi k f_0 + \epsilon_k \cdot s(t) \cdot \sin(\theta_k^{(A)} - \theta_{target})$$

where $\epsilon_k$ is the coupling strength for harmonic $k$.

### 8.3 Proof: Sample-Driven Modulation Preserves Causality

**Theorem 8.1:** When the onset signal $s(t)$ is derived from sample $x[n]$, the modulation response cannot precede the input.

**Proof:**

The onset strength at frame $m$ depends on samples up to index $n = mH + N - 1$.

The modulation is applied at time $t = n/f_s$.

Since the FFT requires $N$ samples, there is an inherent latency of $(N-1)/f_s$ seconds.

However, within this constraint, the same sample that completes the onset detection immediately modulates the oscillators — there is no additional delay. $\square$

---

## 9. Adversarial Verification

### 9.1 Agreement Function

Define the agreement between branches:

$$A(t) = \cos(\theta_1^{(A)}(t) - \theta_1^{(B)}(t))$$

For stable backbeat: $A(t) \approx -1$.

For in-phase patterns: $A(t) \approx +1$.

### 9.2 Divergence Detection

**Definition:** Branches diverge when:

$$|A(t) - A_{target}| > \epsilon_d$$

where $A_{target}$ is the expected phase relationship.

### 9.3 Confidence Score

Define confidence as the inverse of disagreement:

$$\text{conf}(t) = 1 - \frac{|A(t) - A_{target}|}{2}$$

Range: $[0, 1]$ where 1 indicates perfect agreement.

---

## 10. Discrete-Time Implementation

### 10.1 Oscillator Update

At each sample $n$:

$$\theta_k[n+1] = \theta_k[n] + \frac{2\pi k f_0}{f_s} + \epsilon_k \cdot s[n] \cdot \sin(\theta_k[n] - \theta_{target})$$

### 10.2 Output Computation

$$x_k[n] = a_k \cos(\theta_k[n])$$

$$M[n] = \sum_k x_k[n]$$

### 10.3 Agreement Update

$$A[n] = \cos(\theta_1^{(A)}[n] - \theta_1^{(B)}[n])$$

---

## 11. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Context Wrapper                          │
│  (sample_rate, buffer_size, oscillator_bank, state)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Input Sample x[n]                         │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────┐          ┌──────────────────────┐
│   Onset Detector     │          │   Branch Modulator   │
│   → O[m], {t_i}      │          │   → modulates θ_k    │
└──────────────────────┘          └──────────────────────┘
              │                               │
              ▼                               │
┌──────────────────────┐                      │
│   Point Selector     │                      │
│   → (t_a, t_b)       │                      │
└──────────────────────┘                      │
              │                               │
              ▼                               │
┌──────────────────────────────────────────────────────────────┐
│                    Oscillator Banks                          │
│         Branch A                    Branch B                 │
│     θ_k^(A), x_k^(A)            θ_k^(B), x_k^(B)            │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Mixer                                │
│              M_add = M_A + M_B                               │
│              M_mult = M_A × M_B                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Agreement Evaluator                         │
│                 A(t), conf(t), divergence                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     GUI / Output                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Summary of Equations

| Component | Equation |
|-----------|----------|
| Spectrogram | $S[m,k] = \|X[m,k]\|^2$ |
| Onset strength | $O[m] = \sum_k \max(0, S[m,k] - S[m-1,k])$ |
| Spectral centroid | $C_i = \sum_k f_k S[m_i,k] / \sum_k S[m_i,k]$ |
| Oscillator phase | $\theta_k(t) = 2\pi k f_0 t + \phi_k$ |
| Branch output | $M(t) = \sum_k a_k \cos(\theta_k(t))$ |
| Ring modulation | $\cos A \cos B = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$ |
| Agreement | $A(t) = \cos(\theta_1^{(A)} - \theta_1^{(B)})$ |
| Confidence | $\text{conf}(t) = 1 - \|A(t) - A_{target}\|/2$ |

---

## 13. Open Questions

1. **Optimal harmonic count $K$** — How many harmonics are needed for robust detection?

2. **Coupling strength schedule** — Should $\epsilon_k$ vary with harmonic number?

3. **Multi-tempo handling** — Can we run parallel systems at different $f_0$ estimates?

4. **Feedback topology** — When should mixer output feed back into onset detection?

5. **Training the adversarial network** — What loss function best captures rhythmic agreement?

---

## 14. Next Steps

- [ ] Implement discrete-time oscillator bank
- [ ] Test onset-to-modulation latency
- [ ] Validate phase relationship detection on known backbeat patterns
- [ ] Design adversarial loss function
- [ ] Profile computational cost per sample

---

**Document Version:** 0.1 (Preliminary)  
**Last Updated:** January 2026

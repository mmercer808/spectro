# Signal Path Mathematics

## Complete Equation Reference for Beat Detection System

---

## The Signal Path

```
x[n] → STFT → S[m,k] → O[m] → {tᵢ} → (tₐ,tᵦ) → θₖ → Mₐ,Mᵦ → A(t) → conf(t)
```

Each arrow is a mathematical transformation. This document traces every equation.

---

# STAGE 1: Input Sample

## 1.1 The Discrete Signal

$$x[n], \quad n \in \mathbb{Z}, \quad n \geq 0$$

- $n$ = sample index
- $f_s$ = sample rate (samples/second)
- Time: $t = n / f_s$

**This same $x[n]$ feeds both onset detection AND branch modulation.**

---

# STAGE 2: Time-Frequency Representation

## 2.1 Window Function

Hann window of length $N$:

$$w[n] = \frac{1}{2}\left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right), \quad 0 \leq n < N$$

## 2.2 Short-Time Fourier Transform

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-2\pi i k n / N}$$

**Variables:**
- $m$ = frame index
- $k$ = frequency bin index, $0 \leq k \leq N/2$
- $H$ = hop size (typically $N/4$)

**Frequency of bin $k$:**

$$f_k = \frac{k \cdot f_s}{N}$$

## 2.3 Spectrogram (Power Spectrum)

$$S[m, k] = |X[m, k]|^2 = \text{Re}(X[m,k])^2 + \text{Im}(X[m,k])^2$$

**Units:** Power (amplitude squared)

---

# STAGE 3: Onset Detection

## 3.1 Spectral Flux

$$O[m] = \sum_{k=0}^{N/2} \max\left(0, \; S[m, k] - S[m-1, k]\right)$$

**Why max(0, ·)?** Half-wave rectification — we only care about energy *increases*.

**Expanded form:**

$$O[m] = \sum_{k=0}^{N/2} \begin{cases} S[m,k] - S[m-1,k] & \text{if } S[m,k] > S[m-1,k] \\ 0 & \text{otherwise} \end{cases}$$

## 3.2 Onset Peak Detection

Onset at frame $m$ if:

$$O[m] > O[m-1] \quad \land \quad O[m] > O[m+1] \quad \land \quad O[m] > \theta$$

**Output:** Set of onset times $\{t_1, t_2, \ldots, t_n\}$

$$t_i = \frac{m_i \cdot H}{f_s}$$

## 3.3 Onset Strength at Detection Point

Store the strength with each onset:

$$w_i = O[m_i]$$

This weight is used later for oscillator correlation.

---

# STAGE 4: Frequency Analysis at Onsets

## 4.1 Spectral Centroid

At onset $t_i$ (frame $m_i$):

$$C_i = \frac{\sum_{k=0}^{N/2} f_k \cdot S[m_i, k]}{\sum_{k=0}^{N/2} S[m_i, k]}$$

**Substituting $f_k$:**

$$C_i = \frac{\sum_{k=0}^{N/2} \frac{k f_s}{N} \cdot S[m_i, k]}{\sum_{k=0}^{N/2} S[m_i, k]} = \frac{f_s}{N} \cdot \frac{\sum_{k} k \cdot S[m_i, k]}{\sum_{k} S[m_i, k]}$$

**Units:** Hertz

## 4.2 Tempo Estimation from Onset Intervals

**Inter-onset intervals:**

$$\Delta t_i = t_{i+1} - t_i$$

**Mean interval:**

$$\bar{\Delta t} = \frac{1}{n-1} \sum_{i=1}^{n-1} \Delta t_i = \frac{t_n - t_1}{n-1}$$

**Tempo (beats per second):**

$$f_0 = \frac{1}{\bar{\Delta t}}$$

**Tempo (BPM):**

$$\text{BPM} = 60 \cdot f_0$$

---

# STAGE 5: Call-Response Point Selection

## 5.1 Selection Function

Find pair $(t_a, t_b)$ minimizing:

$$L(t_a, t_b) = \alpha |w_a - w_b| - \beta |C_a - C_b| + \gamma \left|(t_b - t_a) - \frac{T}{2}\right|$$

where $T = 1/f_0$ is the beat period.

**Terms:**
- $|w_a - w_b|$ → strength similarity (minimize)
- $|C_a - C_b|$ → spectral contrast (maximize, hence negative)
- $|(t_b - t_a) - T/2|$ → half-period timing (minimize)

## 5.2 Constraint Form

Alternatively, select $(t_a, t_b)$ satisfying:

$$|w_a - w_b| < \epsilon_s$$

$$|C_a - C_b| > \epsilon_c$$

$$\left|(t_b - t_a) - \frac{T}{2}\right| < \epsilon_t$$

---

# STAGE 6: Harmonic Oscillator Banks

## 6.1 Phase Definition

**Branch A (call):**

$$\theta_k^{(A)}(t) = 2\pi k f_0 t + \phi_k^{(A)}$$

**Branch B (response):**

$$\theta_k^{(B)}(t) = 2\pi k f_0 t + \phi_k^{(B)}$$

**Harmonic index:** $k = 1, 2, 3, \ldots, K$

## 6.2 Phase Initialization

Set phases so oscillators peak at onset times:

$$\phi_k^{(A)} = -2\pi k f_0 t_a$$

$$\phi_k^{(B)} = -2\pi k f_0 t_b$$

**Verification:**

$$\theta_k^{(A)}(t_a) = 2\pi k f_0 t_a - 2\pi k f_0 t_a = 0 \quad \checkmark$$

## 6.3 Oscillator Output

$$x_k^{(A)}(t) = a_k \cos\left(\theta_k^{(A)}(t)\right)$$

$$x_k^{(B)}(t) = b_k \cos\left(\theta_k^{(B)}(t)\right)$$

**Coefficients $a_k, b_k$:** Harmonic amplitudes (user-defined or learned)

## 6.4 Bank Output (Summation)

$$M_A(t) = \sum_{k=1}^{K} a_k \cos\left(\theta_k^{(A)}(t)\right)$$

$$M_B(t) = \sum_{k=1}^{K} b_k \cos\left(\theta_k^{(B)}(t)\right)$$

**Expanded:**

$$M_A(t) = \sum_{k=1}^{K} a_k \cos\left(2\pi k f_0 t + \phi_k^{(A)}\right)$$

---

# STAGE 7: The Mixer

## 7.1 Additive Mixing

$$M_{add}(t) = M_A(t) + M_B(t) = \sum_{k=1}^{K} \left[ a_k \cos(\theta_k^{(A)}) + b_k \cos(\theta_k^{(B)}) \right]$$

**Property:** Result is still a harmonic series with fundamental $f_0$.

## 7.2 Multiplicative Mixing (Ring Modulation)

$$M_{mult}(t) = M_A(t) \cdot M_B(t)$$

**Expansion using product-to-sum identity:**

$$\cos(\alpha)\cos(\beta) = \frac{1}{2}\left[\cos(\alpha - \beta) + \cos(\alpha + \beta)\right]$$

**Single term:**

$$a_k \cos(\theta_k^{(A)}) \cdot b_j \cos(\theta_j^{(B)}) = \frac{a_k b_j}{2} \left[ \cos(\theta_k^{(A)} - \theta_j^{(B)}) + \cos(\theta_k^{(A)} + \theta_j^{(B)}) \right]$$

**Full expansion:**

$$M_{mult}(t) = \sum_{k=1}^{K} \sum_{j=1}^{K} \frac{a_k b_j}{2} \left[ \cos(\theta_k^{(A)} - \theta_j^{(B)}) + \cos(\theta_k^{(A)} + \theta_j^{(B)}) \right]$$

## 7.3 Difference Frequency Analysis

**Phase difference:**

$$\theta_k^{(A)} - \theta_j^{(B)} = 2\pi(k-j)f_0 t + (\phi_k^{(A)} - \phi_j^{(B)})$$

**When $k = j$ (same harmonic):**

$$\theta_k^{(A)} - \theta_k^{(B)} = \phi_k^{(A)} - \phi_k^{(B)} = -2\pi k f_0 (t_a - t_b) = 2\pi k f_0 (t_b - t_a)$$

**This is a constant** — the time-varying terms cancel!

## 7.4 Backbeat Phase Relationship

For backbeat: $t_b - t_a = T/2 = 1/(2f_0)$

**Fundamental ($k=1$):**

$$\phi_1^{(A)} - \phi_1^{(B)} = 2\pi f_0 \cdot \frac{1}{2f_0} = \pi$$

**Therefore:**

$$\cos(\theta_1^{(A)} - \theta_1^{(B)}) = \cos(\pi) = -1$$

---

# STAGE 8: Branch Modulation (Coupling to Input)

## 8.1 Onset Signal

Convert onset detections to a continuous signal:

$$s(t) = \sum_i w_i \cdot \delta(t - t_i)$$

**Discrete form:**

$$s[n] = \begin{cases} w_i & \text{if } n = \lfloor t_i \cdot f_s \rfloor \\ 0 & \text{otherwise} \end{cases}$$

## 8.2 Coupled Oscillator Dynamics

**Continuous time:**

$$\frac{d\theta_k}{dt} = 2\pi k f_0 + \epsilon_k \cdot s(t) \cdot \sin(\theta_k - \theta_{target})$$

**Discrete time update:**

$$\theta_k[n+1] = \theta_k[n] + \frac{2\pi k f_0}{f_s} + \epsilon_k \cdot s[n] \cdot \sin(\theta_k[n] - \theta_{target})$$

**Terms:**
- $2\pi k f_0 / f_s$ = free-running phase increment
- $\epsilon_k \cdot s[n] \cdot \sin(\cdot)$ = phase correction from onset

## 8.3 Phase Correction Dynamics

When an onset arrives ($s[n] = w_i$):

- If $\theta_k < \theta_{target}$: $\sin(\theta_k - \theta_{target}) < 0$ → phase decreases (slows down)
- If $\theta_k > \theta_{target}$: $\sin(\theta_k - \theta_{target}) > 0$ → phase increases (speeds up)

This is **Kuramoto coupling** — oscillators pull toward expected phase.

---

# STAGE 9: Adversarial Agreement

## 9.1 Agreement Function

$$A(t) = \cos\left(\theta_1^{(A)}(t) - \theta_1^{(B)}(t)\right)$$

**Range:** $[-1, +1]$

**Interpretation:**
- $A = +1$: branches in phase (unison)
- $A = -1$: branches in antiphase (backbeat)
- $A = 0$: branches in quadrature (90° offset)

## 9.2 Expected Agreement

For backbeat pattern: $A_{target} = -1$

For unison pattern: $A_{target} = +1$

## 9.3 Divergence Measure

$$D(t) = |A(t) - A_{target}|$$

**Divergence threshold:**

$$\text{diverged} \iff D(t) > \epsilon_d$$

## 9.4 Confidence Score

$$\text{conf}(t) = 1 - \frac{D(t)}{2} = 1 - \frac{|A(t) - A_{target}|}{2}$$

**Range:** $[0, 1]$

---

# STAGE 10: Correlation Feedback

## 10.1 Oscillator-Onset Correlation

How well does branch A match the onset pattern?

$$R_A = \sum_{i=1}^{n} w_i \cdot \cos(\theta_1^{(A)}(t_i))$$

**Optimal when:** Oscillator peaks align with onsets.

## 10.2 Closed-Form Optimal Phase

For fixed $f_0$, the optimal phase $\phi^*$ maximizes:

$$R(\phi) = \sum_i w_i \cos(2\pi f_0 t_i + \phi)$$

**Define:**

$$A = \sum_i w_i \cos(2\pi f_0 t_i)$$

$$B = \sum_i w_i \sin(2\pi f_0 t_i)$$

**Optimal phase:**

$$\phi^* = -\arctan2(B, A)$$

**Maximum correlation:**

$$R_{max} = \sqrt{A^2 + B^2}$$

---

# COMPLETE SIGNAL PATH SUMMARY

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: INPUT                                                             │
│  x[n]                                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: STFT                                                              │
│  X[m,k] = Σ x[n+mH]·w[n]·exp(-2πikn/N)                                     │
│  S[m,k] = |X[m,k]|²                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: ONSET DETECTION                                                   │
│  O[m] = Σₖ max(0, S[m,k] - S[m-1,k])                                       │
│  {t₁, t₂, ..., tₙ} = peak_detect(O[m])                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  STAGE 4: FREQUENCY           │   │  STAGE 8: MODULATION SIGNAL   │
│  Cᵢ = Σ fₖS[mᵢ,k] / Σ S[mᵢ,k]│   │  s[n] = wᵢ·δ[n - nᵢ]         │
│  f₀ = (n-1)/(tₙ - t₁)        │   │                               │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                               │
                    ▼                               │
┌───────────────────────────────┐                   │
│  STAGE 5: POINT SELECTION     │                   │
│  (tₐ, tᵦ) = argmin L(·,·)    │                   │
└───────────────────────────────┘                   │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: OSCILLATOR BANKS                                                  │
│  θₖ⁽ᴬ⁾(t) = 2πkf₀t + φₖ⁽ᴬ⁾        θₖ⁽ᴮ⁾(t) = 2πkf₀t + φₖ⁽ᴮ⁾              │
│  φₖ⁽ᴬ⁾ = -2πkf₀tₐ                  φₖ⁽ᴮ⁾ = -2πkf₀tᵦ                        │
│                                                                             │
│  COUPLING (from Stage 8):                                                   │
│  θₖ[n+1] = θₖ[n] + 2πkf₀/fₛ + εₖ·s[n]·sin(θₖ[n] - θ_target)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 7: MIXER                                                             │
│  Mₐ(t) = Σₖ aₖ cos(θₖ⁽ᴬ⁾)         Mᵦ(t) = Σₖ bₖ cos(θₖ⁽ᴮ⁾)                │
│                                                                             │
│  ADDITIVE:       M_add = Mₐ + Mᵦ                                           │
│  MULTIPLICATIVE: M_mult = Mₐ · Mᵦ                                          │
│                                                                             │
│  cos(α)cos(β) = ½[cos(α-β) + cos(α+β)]                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 9: ADVERSARIAL AGREEMENT                                             │
│  A(t) = cos(θ₁⁽ᴬ⁾ - θ₁⁽ᴮ⁾)                                                 │
│  D(t) = |A(t) - A_target|                                                   │
│  conf(t) = 1 - D(t)/2                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 10: CORRELATION FEEDBACK                                             │
│  Rₐ = Σᵢ wᵢ cos(θ₁⁽ᴬ⁾(tᵢ))                                                 │
│  φ* = -arctan2(B, A)                                                        │
│  R_max = √(A² + B²)                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# QUICK REFERENCE: ALL EQUATIONS

## Input/Transform
| Name | Equation |
|------|----------|
| Sample to time | $t = n/f_s$ |
| Hann window | $w[n] = \frac{1}{2}(1 - \cos(2\pi n/(N-1)))$ |
| STFT | $X[m,k] = \sum_n x[n+mH] \cdot w[n] \cdot e^{-2\pi ikn/N}$ |
| Spectrogram | $S[m,k] = \|X[m,k]\|^2$ |
| Bin frequency | $f_k = kf_s/N$ |

## Onset Detection
| Name | Equation |
|------|----------|
| Spectral flux | $O[m] = \sum_k \max(0, S[m,k] - S[m-1,k])$ |
| Onset time | $t_i = m_i H / f_s$ |
| Onset weight | $w_i = O[m_i]$ |

## Frequency/Tempo
| Name | Equation |
|------|----------|
| Spectral centroid | $C_i = \sum_k f_k S[m_i,k] / \sum_k S[m_i,k]$ |
| Beat period | $T = (t_n - t_1)/(n-1)$ |
| Tempo | $f_0 = 1/T$ |

## Oscillators
| Name | Equation |
|------|----------|
| Phase | $\theta_k(t) = 2\pi kf_0 t + \phi_k$ |
| Initial phase | $\phi_k = -2\pi kf_0 t_{onset}$ |
| Output | $x_k(t) = a_k \cos(\theta_k(t))$ |
| Bank sum | $M(t) = \sum_k a_k \cos(\theta_k(t))$ |
| Discrete update | $\theta_k[n+1] = \theta_k[n] + 2\pi kf_0/f_s$ |
| Coupled update | $+ \epsilon_k s[n] \sin(\theta_k[n] - \theta_{target})$ |

## Mixer
| Name | Equation |
|------|----------|
| Additive | $M_{add} = M_A + M_B$ |
| Multiplicative | $M_{mult} = M_A \cdot M_B$ |
| Product identity | $\cos\alpha\cos\beta = \frac{1}{2}[\cos(\alpha-\beta) + \cos(\alpha+\beta)]$ |
| Phase difference | $\Delta\phi = 2\pi kf_0(t_b - t_a)$ |
| Backbeat ($T/2$ offset) | $\Delta\phi_1 = \pi \Rightarrow \cos(\Delta\phi_1) = -1$ |

## Agreement
| Name | Equation |
|------|----------|
| Agreement | $A(t) = \cos(\theta_1^{(A)} - \theta_1^{(B)})$ |
| Divergence | $D(t) = \|A(t) - A_{target}\|$ |
| Confidence | $\text{conf}(t) = 1 - D(t)/2$ |

## Correlation
| Name | Equation |
|------|----------|
| Correlation | $R = \sum_i w_i \cos(\theta_1(t_i))$ |
| Optimal phase | $\phi^* = -\arctan2(B, A)$ |
| Max correlation | $R_{max} = \sqrt{A^2 + B^2}$ |

---

# KEY IDENTITIES USED

$$e^{i\theta} = \cos\theta + i\sin\theta$$

$$\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}$$

$$\cos\alpha\cos\beta = \frac{1}{2}[\cos(\alpha-\beta) + \cos(\alpha+\beta)]$$

$$\sin\alpha\sin\beta = \frac{1}{2}[\cos(\alpha-\beta) - \cos(\alpha+\beta)]$$

$$a\cos\theta + b\sin\theta = \sqrt{a^2+b^2}\cos(\theta - \arctan2(b,a))$$

---

*End of signal path mathematics reference*

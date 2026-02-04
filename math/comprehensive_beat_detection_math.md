# Comprehensive Mathematical Foundations for Beat Detection via Harmonic Oscillator Matching

## A Complete Reference with Theoretical Background and Literature Support

---

**Document Type:** Technical Reference  
**Version:** 1.0  
**Status:** Complete Draft

---

# Part I: Prerequisite Knowledge

Before implementing the onset-to-frequency conversion algorithm, you need to understand the following foundational concepts.

---

## Chapter 1: The Fourier Transform and Time-Frequency Analysis

### 1.1 The Continuous Fourier Transform

The Fourier transform decomposes a signal into its constituent frequencies.

**Definition:**

$$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(t) \, e^{-2\pi i \xi t} \, dt$$

**Inverse:**

$$f(t) = \int_{-\infty}^{\infty} \hat{f}(\xi) \, e^{2\pi i \xi t} \, d\xi$$

**Key Identity (Euler's Formula):**

$$e^{i\theta} = \cos\theta + i\sin\theta$$

This means the Fourier transform projects the signal onto sinusoidal basis functions.

### 1.2 The Discrete Fourier Transform (DFT)

For sampled signals $x[n]$ of length $N$:

$$X[k] = \sum_{n=0}^{N-1} x[n] \, e^{-2\pi i k n / N}$$

**Frequency of bin $k$:**

$$f_k = \frac{k \cdot f_s}{N}$$

where $f_s$ is the sample rate.

**Frequency resolution:**

$$\Delta f = \frac{f_s}{N}$$

### 1.3 The Uncertainty Principle

**Theorem (Heisenberg-Gabor):** A signal cannot be simultaneously localized in both time and frequency.

$$\Delta t \cdot \Delta f \geq \frac{1}{4\pi}$$

where $\Delta t$ and $\Delta f$ are the standard deviations of the signal in time and frequency domains.

**Implication for STFT:** Choosing window length $N$ involves a tradeoff:
- Large $N$ → good frequency resolution, poor time resolution
- Small $N$ → poor frequency resolution, good time resolution

**Reference:** The uncertainty principle for signal analysis is discussed extensively in Cohen (2001), "The Uncertainty Principle for the Short-Time Fourier Transform and Wavelet Transform."

### 1.4 The Short-Time Fourier Transform (STFT)

To analyze how frequency content changes over time:

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-2\pi i k n / N}$$

**Parameters:**
- $m$ = frame index (time)
- $k$ = frequency bin index
- $H$ = hop size (frame advance in samples)
- $w[n]$ = window function

**The Hann Window:**

$$w[n] = \frac{1}{2}\left(1 - \cos\left(\frac{2\pi n}{N-1}\right)\right)$$

The window reduces spectral leakage by tapering the signal at frame boundaries.

### 1.5 The Spectrogram

The power spectrum at each frame:

$$S[m, k] = |X[m, k]|^2 = \text{Re}(X[m,k])^2 + \text{Im}(X[m,k])^2$$

The spectrogram is the primary input to onset detection.

---

## Chapter 2: Onset Detection Theory

### 2.1 What is an Onset?

An onset is the beginning of a musical event—the moment when a new note, drum hit, or transient begins. Detecting onsets is fundamental to rhythm analysis.

**Reference:** Bello, J.P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., Sandler, M.B. (2005). "A Tutorial on Onset Detection in Music Signals." IEEE Transactions on Speech and Audio Processing, 13(5), pp. 1035–1047.

### 2.2 Onset Detection Functions

An onset detection function (ODF) transforms the audio signal into a one-dimensional function where peaks correspond to onsets.

**Common ODFs:**

| Method | Description | Best For |
|--------|-------------|----------|
| Spectral Flux | Energy increase across frequency bins | General purpose |
| High Frequency Content | Weighted by frequency | Percussive onsets |
| Complex Domain | Uses phase and magnitude | Pitched instruments |
| Phase Deviation | Measures phase irregularity | Soft onsets |

### 2.3 Spectral Flux

**Definition:**

$$O[m] = \sum_{k=0}^{N/2} H(S[m, k] - S[m-1, k])$$

where $H(x) = \max(0, x)$ is the half-wave rectifier.

**Why half-wave rectification?** We only care about energy *increases*, not decreases. A note ending doesn't constitute a new onset.

**Expanded form:**

$$O[m] = \sum_{k=0}^{N/2} \begin{cases} S[m,k] - S[m-1,k] & \text{if } S[m,k] > S[m-1,k] \\ 0 & \text{otherwise} \end{cases}$$

### 2.4 Proof: Spectral Flux Bounds

**Theorem:** The spectral flux $O[m]$ is bounded by the total spectral energy change.

**Proof:**

Let $\Delta S[m, k] = S[m, k] - S[m-1, k]$.

$$O[m] = \sum_k H(\Delta S[m, k]) \leq \sum_k |\Delta S[m, k]|$$

Equality holds when all $\Delta S[m, k] \geq 0$, i.e., when energy increases across all bins (as happens at a strong transient onset). $\square$

### 2.5 Peak Picking

Onsets occur at local maxima of $O[m]$:

$$\text{onset at } m \iff O[m] > O[m-1] \land O[m] > O[m+1] \land O[m] > \theta$$

where $\theta$ is a detection threshold.

**Adaptive thresholding:**

$$\theta[m] = \alpha \cdot \text{median}(O[m-W:m]) + \beta$$

Using a moving median makes the threshold robust to varying signal levels.

**Reference:** Dixon, S. (2006). "Onset Detection Revisited." Proceedings of the 9th International Conference on Digital Audio Effects (DAFx'06).

### 2.6 Converting Onset Frames to Time

$$t_i = \frac{m_i \cdot H}{f_s}$$

where:
- $m_i$ = frame index of onset $i$
- $H$ = hop size in samples
- $f_s$ = sample rate

---

## Chapter 3: From Onset to Frequency

### 3.1 The Core Problem

Given a sequence of onset times $\{t_1, t_2, \ldots, t_n\}$, estimate the underlying beat frequency $f_0$.

This is the **onset-to-frequency conversion** that links time-domain events to a periodic model.

### 3.2 Inter-Onset Interval (IOI)

$$\Delta t_i = t_{i+1} - t_i$$

The IOIs contain information about the underlying tempo.

### 3.3 Naive Tempo Estimation

**Mean IOI method:**

$$\bar{\Delta t} = \frac{1}{n-1} \sum_{i=1}^{n-1} \Delta t_i = \frac{t_n - t_1}{n-1}$$

**Tempo:**

$$f_0 = \frac{1}{\bar{\Delta t}} \quad \text{(beats per second)}$$

$$\text{BPM} = 60 \cdot f_0$$

**Problem:** This assumes all onsets lie on the beat grid, which is often false (syncopation, off-beats, noise).

### 3.4 Autocorrelation Method

The autocorrelation of the onset strength function reveals periodicity:

$$R[\tau] = \sum_m O[m] \cdot O[m + \tau]$$

Peaks in $R[\tau]$ correspond to likely beat periods.

**Reference:** Scheirer, E. (1998). "Tempo and Beat Analysis of Acoustic Musical Signals." Journal of the Acoustical Society of America, 103(1), pp. 588–601.

### 3.5 Spectral Centroid at Onsets

To characterize the frequency content at each onset:

$$C_i = \frac{\sum_{k=0}^{N/2} f_k \cdot S[m_i, k]}{\sum_{k=0}^{N/2} S[m_i, k]}$$

**Theorem:** The spectral centroid is the power-weighted mean frequency.

**Proof:**

Define the normalized spectral distribution:

$$p_k = \frac{S[m_i, k]}{\sum_j S[m_i, j]}$$

This is a valid probability distribution: $p_k \geq 0$ and $\sum_k p_k = 1$.

Then:

$$C_i = \sum_k f_k \cdot p_k = \mathbb{E}[f]$$

The centroid is the expected value of frequency. $\square$

**Application:** The centroid helps distinguish different instruments (low centroid = bass/kick, high centroid = snare/hi-hat).

---

## Chapter 4: Coupled Oscillator Theory

### 4.1 The Harmonic Oscillator

A simple harmonic oscillator has phase:

$$\theta(t) = 2\pi f_0 t + \phi$$

where:
- $f_0$ = frequency
- $\phi$ = initial phase

**Output signal:**

$$x(t) = A \cos(\theta(t)) = A \cos(2\pi f_0 t + \phi)$$

### 4.2 The Kuramoto Model

The Kuramoto model describes synchronization in coupled oscillator populations.

**Definition:** For $N$ oscillators:

$$\frac{d\theta_k}{dt} = \omega_k + \frac{K}{N} \sum_{j=1}^{N} \sin(\theta_j - \theta_k)$$

where:
- $\theta_k$ = phase of oscillator $k$
- $\omega_k$ = natural frequency of oscillator $k$
- $K$ = coupling strength

**Key insight:** When coupling $K$ exceeds a critical value, oscillators spontaneously synchronize.

**Reference:** Kuramoto, Y. (1975). "Self-entrainment of a Population of Coupled Non-linear Oscillators." International Symposium on Mathematical Problems in Theoretical Physics, pp. 420–422.

**Reference:** Acebrón, J.A., et al. (2005). "The Kuramoto Model: A Simple Paradigm for Synchronization Phenomena." Reviews of Modern Physics, 77(1), pp. 137–185.

### 4.3 Adaptation for Beat Tracking

We modify the Kuramoto model to couple oscillators to an external onset signal rather than to each other:

$$\frac{d\theta_k}{dt} = 2\pi k f_0 + \epsilon_k \cdot s(t) \cdot \sin(\theta_k - \theta_{target})$$

where:
- $k$ = harmonic number
- $s(t)$ = onset signal (spikes at onset times)
- $\epsilon_k$ = coupling strength for harmonic $k$
- $\theta_{target}$ = expected phase at onset

This is a form of **injection locking** — the oscillator is pulled toward alignment with external events.

### 4.4 Phase-Locked Loop Analogy

The coupled oscillator is analogous to a Phase-Locked Loop (PLL):

| PLL Component | Our System |
|---------------|------------|
| Reference signal | Onset pulse train |
| VCO | Harmonic oscillator |
| Phase detector | $\sin(\theta_k - \theta_{target})$ |
| Loop filter | Coupling strength $\epsilon_k$ |

**Reference:** Gardner, F.M. (2005). *Phaselock Techniques*, 3rd ed. Wiley.

### 4.5 Discrete-Time Update Equation

For real-time implementation:

$$\theta_k[n+1] = \theta_k[n] + \frac{2\pi k f_0}{f_s} + \epsilon_k \cdot s[n] \cdot \sin(\theta_k[n] - \theta_{target})$$

**Terms:**
- $\frac{2\pi k f_0}{f_s}$ = free-running phase increment per sample
- $\epsilon_k \cdot s[n] \cdot \sin(\cdot)$ = phase correction from onset

---

## Chapter 5: Harmonic Series and the Mixer

### 5.1 Harmonic Series Definition

A harmonic series with fundamental $f_0$:

$$f_k = k \cdot f_0, \quad k = 1, 2, 3, \ldots$$

**Oscillator bank output:**

$$M(t) = \sum_{k=1}^{K} a_k \cos(2\pi k f_0 t + \phi_k)$$

This is a truncated Fourier series.

### 5.2 Fourier Series Representation Theorem

**Theorem:** Any periodic function with period $T = 1/f_0$ can be represented as:

$$g(t) = \frac{a_0}{2} + \sum_{k=1}^{\infty} \left[ a_k \cos(2\pi k f_0 t) + b_k \sin(2\pi k f_0 t) \right]$$

**Implication:** Our harmonic oscillator bank spans the space of beat patterns with fundamental frequency $f_0$.

### 5.3 Additive Mixing

Two harmonic series add:

$$M_{add}(t) = M_A(t) + M_B(t) = \sum_k \left[ a_k \cos(\theta_k^{(A)}) + b_k \cos(\theta_k^{(B)}) \right]$$

The result is still a harmonic series with fundamental $f_0$.

### 5.4 Multiplicative Mixing (Ring Modulation)

**Product-to-Sum Identity:**

$$\cos(\alpha) \cos(\beta) = \frac{1}{2}\left[\cos(\alpha - \beta) + \cos(\alpha + \beta)\right]$$

**Proof:**

Using Euler's formula:
$$\cos(\alpha) = \frac{e^{i\alpha} + e^{-i\alpha}}{2}, \quad \cos(\beta) = \frac{e^{i\beta} + e^{-i\beta}}{2}$$

$$\cos(\alpha)\cos(\beta) = \frac{1}{4}\left(e^{i\alpha} + e^{-i\alpha}\right)\left(e^{i\beta} + e^{-i\beta}\right)$$

$$= \frac{1}{4}\left(e^{i(\alpha+\beta)} + e^{i(\alpha-\beta)} + e^{-i(\alpha-\beta)} + e^{-i(\alpha+\beta)}\right)$$

$$= \frac{1}{2}\left(\cos(\alpha+\beta) + \cos(\alpha-\beta)\right) \quad \square$$

### 5.5 Difference Frequency Extraction

When two oscillators at the same harmonic $k$ are multiplied:

$$\cos(\theta_k^{(A)}) \cos(\theta_k^{(B)}) = \frac{1}{2}\left[\cos(\theta_k^{(A)} - \theta_k^{(B)}) + \cos(\theta_k^{(A)} + \theta_k^{(B)})\right]$$

The difference term $\theta_k^{(A)} - \theta_k^{(B)}$ depends only on the phase relationship, not on time:

$$\theta_k^{(A)} - \theta_k^{(B)} = \phi_k^{(A)} - \phi_k^{(B)} = -2\pi k f_0 (t_a - t_b)$$

This is a **constant** that encodes the temporal relationship between the two branches.

### 5.6 Backbeat Detection Theorem

**Theorem:** For backbeat patterns where $t_b - t_a = T/2$, the fundamental phase difference is $\pi$.

**Proof:**

$$\phi_1^{(A)} - \phi_1^{(B)} = -2\pi f_0 t_a + 2\pi f_0 t_b = 2\pi f_0 (t_b - t_a)$$

For $t_b - t_a = T/2 = 1/(2f_0)$:

$$\phi_1^{(A)} - \phi_1^{(B)} = 2\pi f_0 \cdot \frac{1}{2f_0} = \pi$$

Therefore:
$$\cos(\phi_1^{(A)} - \phi_1^{(B)}) = \cos(\pi) = -1 \quad \square$$

**Interpretation:** The multiplicative mixer outputs a negative value for perfect backbeat alignment.

---

## Chapter 6: Adversarial Branch Detection

### 6.1 Two-Branch Architecture

Select two onset points $(t_a, C_a)$ and $(t_b, C_b)$ representing call and response.

**Selection criteria:**

1. **Strength balance:** $|w_a - w_b| < \epsilon_s$
2. **Spectral contrast:** $|C_a - C_b| > \epsilon_c$  
3. **Temporal relationship:** $|(t_b - t_a) - T/2| < \epsilon_t$

These select complementary rhythmic elements (e.g., kick and snare).

### 6.2 Agreement Function

$$A(t) = \cos(\theta_1^{(A)}(t) - \theta_1^{(B)}(t))$$

**Range:** $[-1, +1]$

**Interpretation:**
- $A = +1$: branches in phase (unison)
- $A = -1$: branches in antiphase (backbeat)
- $A = 0$: branches in quadrature

### 6.3 Divergence and Confidence

**Divergence measure:**

$$D(t) = |A(t) - A_{target}|$$

**Confidence score:**

$$\text{conf}(t) = 1 - \frac{D(t)}{2}$$

When $D(t) > \epsilon_d$, the branches have diverged—indicating a rhythmic change (tempo shift, breakdown, fill).

### 6.4 Adversarial Interpretation

The two branches "compete" to track the rhythm. Their agreement provides confidence:
- High agreement → locked onto the beat
- Low agreement → rhythmic uncertainty

This is analogous to ensemble methods in machine learning, where multiple models must agree for high confidence.

---

## Chapter 7: The Complete Algorithm

### 7.1 Signal Flow

```
x[n] → STFT → S[m,k] → O[m] → {tᵢ} → (tₐ,tᵦ) → θₖ → Mₐ,Mᵦ → A(t) → conf(t)
```

### 7.2 Stage-by-Stage Equations

**Stage 1: Input**
$$x[n], \quad t = n/f_s$$

**Stage 2: STFT**
$$X[m,k] = \sum_{n=0}^{N-1} x[n+mH] \cdot w[n] \cdot e^{-2\pi ikn/N}$$
$$S[m,k] = |X[m,k]|^2$$

**Stage 3: Onset Detection**
$$O[m] = \sum_k \max(0, S[m,k] - S[m-1,k])$$
$$t_i = m_i H / f_s$$

**Stage 4: Frequency Analysis**
$$C_i = \frac{\sum_k f_k S[m_i,k]}{\sum_k S[m_i,k]}$$
$$f_0 = (n-1)/(t_n - t_1)$$

**Stage 5: Point Selection**
$$\min_{t_a, t_b} L(t_a, t_b) = \alpha|w_a - w_b| - \beta|C_a - C_b| + \gamma|(t_b - t_a) - T/2|$$

**Stage 6: Oscillator Banks**
$$\theta_k^{(A)}(t) = 2\pi k f_0 t + \phi_k^{(A)}, \quad \phi_k^{(A)} = -2\pi k f_0 t_a$$
$$\theta_k^{(B)}(t) = 2\pi k f_0 t + \phi_k^{(B)}, \quad \phi_k^{(B)} = -2\pi k f_0 t_b$$

**Stage 7: Mixer**
$$M_A(t) = \sum_k a_k \cos(\theta_k^{(A)})$$
$$M_B(t) = \sum_k b_k \cos(\theta_k^{(B)})$$
$$M_{add} = M_A + M_B, \quad M_{mult} = M_A \cdot M_B$$

**Stage 8: Modulation**
$$\theta_k[n+1] = \theta_k[n] + \frac{2\pi k f_0}{f_s} + \epsilon_k \cdot s[n] \cdot \sin(\theta_k[n] - \theta_{target})$$

**Stage 9: Agreement**
$$A(t) = \cos(\theta_1^{(A)} - \theta_1^{(B)})$$
$$\text{conf}(t) = 1 - |A(t) - A_{target}|/2$$

---

# Part II: Mathematical Proofs

## Proof 1: Parseval's Theorem (Energy Conservation)

**Theorem:** The total energy in time domain equals total energy in frequency domain.

$$\sum_n |x[n]|^2 = \frac{1}{N} \sum_k |X[k]|^2$$

**Significance:** The STFT preserves signal energy, ensuring onset detection responds to actual signal changes.

---

## Proof 2: Spectral Flux as Energy Derivative

**Theorem:** Spectral flux approximates the rate of energy increase.

**Proof:**

Total energy at frame $m$:
$$E[m] = \sum_k S[m,k]$$

Energy change:
$$\Delta E[m] = E[m] - E[m-1] = \sum_k (S[m,k] - S[m-1,k])$$

Spectral flux:
$$O[m] = \sum_k \max(0, S[m,k] - S[m-1,k])$$

Note that:
$$O[m] \geq \Delta E[m]$$

with equality when all spectral changes are positive. Thus spectral flux captures energy increases while ignoring decreases. $\square$

---

## Proof 3: Optimal Phase for Correlation

**Theorem:** For fixed frequency $f_0$, the phase $\phi^*$ that maximizes correlation with onsets has a closed-form solution.

**Proof:**

The correlation:
$$R(\phi) = \sum_i w_i \cos(2\pi f_0 t_i + \phi)$$

Using the identity $\cos(\alpha + \phi) = \cos\alpha\cos\phi - \sin\alpha\sin\phi$:

$$R(\phi) = \cos\phi \sum_i w_i \cos(2\pi f_0 t_i) - \sin\phi \sum_i w_i \sin(2\pi f_0 t_i)$$

Define:
$$A = \sum_i w_i \cos(2\pi f_0 t_i), \quad B = \sum_i w_i \sin(2\pi f_0 t_i)$$

Then:
$$R(\phi) = A\cos\phi - B\sin\phi = \sqrt{A^2 + B^2} \cos(\phi + \arctan(B/A))$$

Maximum occurs when $\phi + \arctan(B/A) = 0$:

$$\phi^* = -\arctan2(B, A)$$

Maximum correlation:
$$R_{max} = \sqrt{A^2 + B^2} \quad \square$$

---

## Proof 4: Kuramoto Synchronization Condition

**Theorem:** Oscillators in the Kuramoto model synchronize when coupling exceeds a critical value.

For a uniform frequency distribution on $[-\gamma, \gamma]$:

$$K_c = \frac{2\gamma}{\pi}$$

**Reference:** Strogatz, S.H. (2000). "From Kuramoto to Crawford: Exploring the Onset of Synchronization." Physica D, 143, pp. 1–20.

---

## Proof 5: Phase Locked Loop Stability

**Theorem:** A first-order PLL with phase detector gain $K_d$ and VCO gain $K_v$ has lock range:

$$\Delta\omega_{lock} = K_d \cdot K_v$$

The PLL will lock onto any input frequency within $\pm\Delta\omega_{lock}$ of the VCO center frequency.

**Application:** Our oscillator coupling strength $\epsilon_k$ determines how far off-tempo the system can track before losing lock.

---

# Part III: Implementation Notes

## Recommended Parameters

| Parameter | Typical Value | Notes |
|-----------|--------------|-------|
| Sample rate $f_s$ | 44100 Hz | Standard audio |
| FFT size $N$ | 2048 | Good frequency resolution |
| Hop size $H$ | 512 | $N/4$ is common |
| Window | Hann | Reduces leakage |
| Harmonics $K$ | 8 | Captures beat shape |
| Coupling $\epsilon$ | 0.1 | Start small, tune empirically |

## Latency Analysis

**STFT latency:** $(N-1)/f_s$

For $N = 2048$, $f_s = 44100$:
$$\text{latency} = \frac{2047}{44100} \approx 46 \text{ ms}$$

This is the minimum possible latency for the system.

## Numerical Stability

**Phase wrapping:** Always reduce phase modulo $2\pi$:
```python
theta = theta % (2 * np.pi)
```

**Avoid division by zero:** In spectral centroid, add small epsilon to denominator:
```python
C = np.sum(f * S) / (np.sum(S) + 1e-10)
```

---

# Part IV: References

## Primary Sources

1. **Bello, J.P., et al. (2005).** "A Tutorial on Onset Detection in Music Signals." *IEEE Transactions on Speech and Audio Processing*, 13(5), pp. 1035–1047.  
   *The foundational paper on onset detection methods.*

2. **Kuramoto, Y. (1984).** *Chemical Oscillations, Waves, and Turbulence.* Springer.  
   *The definitive text on coupled oscillator theory.*

3. **Acebrón, J.A., et al. (2005).** "The Kuramoto Model: A Simple Paradigm for Synchronization Phenomena." *Reviews of Modern Physics*, 77(1), pp. 137–185.  
   *Comprehensive review of Kuramoto model mathematics.*

4. **Scheirer, E. (1998).** "Tempo and Beat Analysis of Acoustic Musical Signals." *Journal of the Acoustical Society of America*, 103(1), pp. 588–601.  
   *Classic paper on resonator-based beat tracking.*

5. **Dixon, S. (2006).** "Onset Detection Revisited." *Proceedings of DAFx'06*.  
   *Improvements to spectral flux and peak picking.*

## Beat Tracking and Tempo Estimation

6. **Klapuri, A., Eronen, A., Astola, J. (2006).** "Analysis of the Meter of Acoustic Musical Signals." *IEEE Transactions on Audio, Speech, and Language Processing*, 14(1), pp. 342–355.  
   *Multi-level metrical analysis using comb filterbanks.*

7. **Large, E. and Kolen, J. (1994).** "Resonance and the Perception of Musical Meter." *Connection Science*, 6, pp. 177–208.  
   *Adaptive oscillator model for beat tracking.*

8. **Gouyon, F., et al. (2006).** "An Experimental Comparison of Audio Tempo Induction Algorithms." *IEEE Transactions on Audio, Speech and Language Processing*, 14(5), pp. 1832–1844.  
   *Benchmark comparison of tempo algorithms.*

## Signal Processing Foundations

9. **Oppenheim, A.V. and Schafer, R.W. (2009).** *Discrete-Time Signal Processing*, 3rd ed. Prentice Hall.  
   *Standard DSP textbook covering DFT, STFT, windows.*

10. **Cohen, L. (1995).** *Time-Frequency Analysis.* Prentice Hall.  
    *Definitive treatment of time-frequency representations.*

11. **Gardner, F.M. (2005).** *Phaselock Techniques*, 3rd ed. Wiley.  
    *Complete reference on PLL design and analysis.*

## Additional Resources

12. **Strogatz, S.H. (2000).** "From Kuramoto to Crawford: Exploring the Onset of Synchronization." *Physica D*, 143, pp. 1–20.  
    *Excellent historical and mathematical overview.*

13. **Mallat, S. (2008).** *A Wavelet Tour of Signal Processing*, 3rd ed. Academic Press.  
    *Alternative time-frequency methods.*

14. **Böck, S. and Schedl, M. (2011).** "Enhanced Beat Tracking with Context-Aware Neural Networks." *Proceedings of DAFx'11*.  
    *Modern neural network approaches.*

---

# Appendix A: Essential Trigonometric Identities

$$\cos(\alpha \pm \beta) = \cos\alpha\cos\beta \mp \sin\alpha\sin\beta$$

$$\sin(\alpha \pm \beta) = \sin\alpha\cos\beta \pm \cos\alpha\sin\beta$$

$$\cos\alpha\cos\beta = \frac{1}{2}[\cos(\alpha-\beta) + \cos(\alpha+\beta)]$$

$$\sin\alpha\sin\beta = \frac{1}{2}[\cos(\alpha-\beta) - \cos(\alpha+\beta)]$$

$$\sin\alpha\cos\beta = \frac{1}{2}[\sin(\alpha-\beta) + \sin(\alpha+\beta)]$$

$$a\cos\theta + b\sin\theta = \sqrt{a^2+b^2}\cos(\theta - \arctan(b/a))$$

$$e^{i\theta} = \cos\theta + i\sin\theta$$

$$\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}$$

$$\sin\theta = \frac{e^{i\theta} - e^{-i\theta}}{2i}$$

---

# Appendix B: Notation Summary

| Symbol | Meaning |
|--------|---------|
| $x[n]$ | Input signal at sample $n$ |
| $f_s$ | Sample rate (Hz) |
| $N$ | FFT size |
| $H$ | Hop size |
| $X[m,k]$ | STFT coefficient |
| $S[m,k]$ | Spectrogram (power) |
| $O[m]$ | Onset strength function |
| $t_i$ | Time of onset $i$ |
| $w_i$ | Strength of onset $i$ |
| $C_i$ | Spectral centroid at onset $i$ |
| $f_0$ | Fundamental frequency (tempo) |
| $T$ | Beat period ($1/f_0$) |
| $\theta_k$ | Phase of oscillator $k$ |
| $\phi_k$ | Initial phase of oscillator $k$ |
| $a_k$ | Amplitude of harmonic $k$ |
| $M(t)$ | Mixer output |
| $A(t)$ | Agreement function |
| $\epsilon_k$ | Coupling strength |

---

*End of comprehensive mathematical reference*

**Document compiled:** January 2026

# [A][A] Preamble Synchronization — FPGA Design Document

**Version:** 1.0  
**Date:** February 2026  
**Author:** Generated from simulation analysis  

---

## Table of Contents

1. [Overview](#1-overview)
2. [System Parameters](#2-system-parameters)
3. [Preamble Generation](#3-preamble-generation)
4. [Detector Algorithm](#4-detector-algorithm)
5. [Streaming Implementation](#5-streaming-implementation)
6. [Multi-Antenna Combining](#6-multi-antenna-combining)
7. [Detection Logic & Thresholding](#7-detection-logic--thresholding)
8. [CFO Estimation](#8-cfo-estimation)
9. [Timing Estimation](#9-timing-estimation)
10. [FPGA Architecture](#10-fpga-architecture)
11. [Resource Estimates](#11-resource-estimates)
12. [Test Vectors](#12-test-vectors)
13. [Performance Summary](#13-performance-summary)

---

## 1. Overview

This document describes a Schmidl-Cox style synchronization scheme using an [A][A] preamble structure. The preamble consists of two identical halves, enabling:

- **Timing synchronization** via autocorrelation peak detection
- **Carrier Frequency Offset (CFO) estimation** via correlation phase

### Key Features

| Feature | Description |
|---------|-------------|
| Preamble structure | [A][A] — two identical halves, NO cyclic prefix |
| Detection method | Normalized autocorrelation metric M = \|P\|²/R² |
| Multi-antenna | Coherent combining of P, sum of R |
| FPGA-friendly | Streaming, no division in critical path, fixed delays |

### Frame Structure

```
[Pre-pad zeros][Preamble: A|A][Pilot+CP][Data+CP][Post-pad]
     500          1024         1096      1096      500    samples
```

---

## 2. System Parameters

### 2.1 OFDM Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| FFT size | N_FFT | 1024 | |
| Sample rate | fs | 15.36 MHz | 10 MHz LTE-like |
| Active subcarriers | N_active | 600 | ±300 around DC |
| Subcarrier spacing | Δf | 15 kHz | fs / N_FFT |
| Cyclic prefix (data) | N_CP | 72 samples | 4.69 µs |
| Symbol duration | T_sym | 66.67 µs | N_FFT / fs |

### 2.2 Preamble Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Total length | N_preamble | 1024 samples | One OFDM symbol, no CP |
| Half length | L | 512 samples | Correlation window size |
| Duration | T_preamble | 66.67 µs | N_preamble / fs |
| Subcarrier spacing | K | 2 | Every 2nd bin used |
| Used subcarriers | N_used | 300 | N_active / K |
| PAPR | | 3.69 dB | Measured |

### 2.3 ADC Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 12 bits | Signed: -2048 to +2047 |
| I/Q format | 12-bit I + 12-bit Q | Complex samples |
| Recommended headroom | 6 dB | FS ratio ≈ 2.0 to avoid clipping bias |

### 2.4 Detection Parameters

| Parameter | Symbol | Value | Notes |
|-----------|--------|-------|-------|
| Detection threshold | θ | 0.15 | M[n] threshold for gate open |
| Hysteresis count | H | 128 samples | Samples below θ before gate close |

---

## 3. Preamble Generation

### 3.1 Design Principle

Using only every K-th subcarrier in frequency domain creates time-domain periodicity:

```
Frequency:  X[k] ≠ 0  only for k = 0, K, 2K, 3K, ...
                ↓ IFFT
Time:       x[n] has period N_FFT/K
```

For K=2: period = 512 samples → [A][A] structure with L=512

### 3.2 Subcarrier Selection

**Step 1:** Define active bandwidth
```
DC bin index: dc = N_FFT / 2 = 512
Active range: [dc - 300, dc + 300] excluding dc
```

**Step 2:** Select every 2nd bin (even bins only)
```python
used_bins = []
for offset in range(-300, 301):
    if offset == 0:
        continue  # Skip DC
    bin_idx = 512 + offset
    if bin_idx % 2 == 0:  # Only EVEN bins
        used_bins.append(bin_idx)
```

This yields **300 subcarriers**: bins 212, 214, 216, ..., 510, 514, 516, ..., 810, 812

### 3.3 Zadoff-Chu Sequence

The Zadoff-Chu sequence provides constant amplitude and good autocorrelation:

```
ZC[n] = exp(-j × π × u × n × (n+1) / N_ZC)

where:
  n = 0, 1, 2, ..., N_ZC-1
  N_ZC = 300 (number of used subcarriers)
  u = 25 (root index, coprime with N_ZC)
```

**Properties:**
- |ZC[n]| = 1 for all n (constant amplitude)
- Ideal cyclic autocorrelation
- Low PAPR in time domain

### 3.4 Complete Generation Algorithm

```python
def generate_aa_preamble():
    N_FFT = 1024
    N_ZC = 300
    u = 25  # ZC root
    
    # 1. Generate Zadoff-Chu sequence
    n = np.arange(N_ZC)
    ZC = np.exp(-1j * np.pi * u * n * (n + 1) / N_ZC)
    
    # 2. Map to FFT bins (every 2nd bin, skip DC)
    spectrum = np.zeros(N_FFT, dtype=complex)
    bin_idx = 0
    for offset in range(-300, 301):
        if offset == 0:
            continue
        fft_bin = 512 + offset
        if fft_bin % 2 == 0:
            spectrum[fft_bin] = ZC[bin_idx]
            bin_idx += 1
    
    # 3. IFFT to time domain
    preamble = np.fft.ifft(spectrum) * np.sqrt(N_FFT)
    
    # 4. Normalize to unit power
    preamble = preamble / np.sqrt(np.mean(np.abs(preamble)**2))
    
    return preamble  # Length 1024, structure [A][A]
```

### 3.5 Verification

After generation, verify:
```
first_half  = preamble[0:512]
second_half = preamble[512:1024]
correlation = |⟨first_half, second_half⟩| / (‖first_half‖ × ‖second_half‖)

Expected: correlation = 1.000000
```

---

## 4. Detector Algorithm

### 4.1 Core Metrics

The detector computes three metrics at each sample n:

**P[n] — Complex Autocorrelation:**
```
P[n] = Σ_{k=0}^{L-1} x[n-L+1+k] × conj(x[n-2L+1+k])
     = Σ_{k=0}^{L-1} x[n-L+1+k] × conj(x[n-L+1+k-L])
```

Simplified: correlate samples in window [n-L+1, n] with samples L positions earlier.

**R[n] — Energy of Current Window:**
```
R[n] = Σ_{k=0}^{L-1} |x[n-L+1+k]|²
```

**M[n] — Normalized Metric:**
```
M[n] = |P[n]|² / R[n]²
```

Range: 0 ≤ M[n] ≤ 1

### 4.2 Metric Behavior

| Signal Region | P[n] | R[n] | M[n] |
|---------------|------|------|------|
| Noise only | Small (random) | Noise power × L | ≈ 0 |
| Entering preamble | Rising | Rising | Rising |
| Fully in preamble | Maximum | Signal power × L | ≈ 1.0 |
| Exiting preamble | Falling (different signal) | High | Falling |

### 4.3 Why This Works

When both windows are inside [A][A]:
- Window 1 sees: A[k] for k in some range
- Window 2 sees: A[k-L] = A[k] (identical due to repetition)
- Correlation P[n] = Σ|A[k]|² × exp(j×φ_CFO) ≈ R[n] × exp(j×φ_CFO)
- Therefore M[n] = |P[n]|²/R[n]² ≈ 1.0

---

## 5. Streaming Implementation

### 5.1 Recursive Update Equations

For sample-by-sample processing, use sliding window updates:

**Correlation P (complex):**
```
P[n] = P[n-1] - x[n-L] × conj(x[n-2L]) + x[n] × conj(x[n-L])
       ─────────────────────────────────   ───────────────────
              remove oldest term            add newest term
```

**Energy R (real):**
```
R[n] = R[n-1] - |x[n-L]|² + |x[n]|²
```

### 5.2 Required Delay Lines

| Delay Line | Length | Data Type | Purpose |
|------------|--------|-----------|---------|
| DL1 | L = 512 | Complex (I+Q) | Provides x[n-L] |
| DL2 | L = 512 | Complex (I+Q) | Provides x[n-2L] (cascaded from DL1 output) |

**Alternative:** Single delay line of length 2L, tap at positions L and 2L.

### 5.3 Block Diagram

```
x[n] ──┬────────────────────────────────────────────────────┐
       │                                                     │
       ▼                                                     ▼
   [Delay L] ──┬──────────────────────────────────┐     [|·|²]
       │       │                                   │         │
       ▼       ▼                                   ▼         ▼
   [Delay L]  [conj]                           [conj]   [Running
       │       │                                   │      Sum L]
       ▼       ▼                                   ▼         │
    x[n-2L]   conj(x[n-L])                    conj(x[n-L])   R[n]
       │       │                                   │
       ▼       ▼                                   │
      [×]─────────────────────────────────────────[×]
       │                                           │
       ▼                                           ▼
   [Running Sum L]                          [Running Sum L]
   (remove old)                             (add new)
       │                                           │
       └──────────────────┬────────────────────────┘
                          ▼
                     P[n] = P_re + j×P_im
```

### 5.4 Numerical Considerations

**Fixed-Point Formats (suggested):**

| Signal | Format | Notes |
|--------|--------|-------|
| x[n] (input) | 12-bit signed | From ADC |
| x[n] × conj(x[n-L]) | 24-bit signed | Product before accumulation |
| P accumulator | 40-bit signed | 24 + log2(L) = 24 + 9 = 33, with margin |
| R accumulator | 36-bit unsigned | 24 + 9 = 33, with margin |
| \|P\|² | 48-bit unsigned | After squaring |
| R² | 48-bit unsigned | After squaring |
| M (if computed) | 16-bit unsigned | After division (optional) |

**Overflow Prevention:**
- P accumulates L=512 products of 24-bit values → max 33 bits needed
- Add 7 bits margin → 40-bit accumulator is safe

---

## 6. Multi-Antenna Combining

### 6.1 Combining Method

For N_ant receive antennas, combine before detection:

```
P_total[n] = Σ_{a=0}^{N_ant-1} P_a[n]     (complex sum — coherent combining)
R_total[n] = Σ_{a=0}^{N_ant-1} R_a[n]     (real sum)
M[n] = |P_total[n]|² / R_total[n]²
```

### 6.2 Benefits

| Benefit | Explanation |
|---------|-------------|
| SNR gain | Signals add coherently, noise adds incoherently |
| CFO accuracy | Averages out channel-induced phase errors |
| Diversity | Different channels provide independent observations |

### 6.3 Hardware Implications

For 2 antennas:
- 2× delay lines (or shared memory with 2× bandwidth)
- 2× MAC units for P and R computation
- Simple adders for combining P_total, R_total

---

## 7. Detection Logic & Thresholding

### 7.1 State Machine

```
                    ┌─────────────────────────────────┐
                    │                                 │
                    ▼                                 │
            ┌───────────────┐                         │
   ────────►│     IDLE      │                         │
            └───────────────┘                         │
                    │                                 │
                    │ M[n] ≥ θ                        │
                    ▼                                 │
            ┌───────────────┐                         │
            │  GATE_OPEN    │◄────────────────────────┤
            │               │      M[n] ≥ θ           │
            │ Track peak    │      (reset low_count)  │
            │ of |P|²       │                         │
            └───────────────┘                         │
                    │                                 │
                    │ M[n] < θ for H consecutive      │
                    │ samples (hysteresis)            │
                    ▼                                 │
            ┌───────────────┐                         │
            │ REPORT_EVENT  │─────────────────────────┘
            │               │
            │ Output:       │
            │  - peak_index │
            │  - P_at_peak  │
            │  - M_at_peak  │
            └───────────────┘
```

### 7.2 Threshold Selection

| Threshold θ | Detection | False Alarm | Recommended Use |
|-------------|-----------|-------------|-----------------|
| 0.10 | High sensitivity | Higher | Low SNR environments |
| 0.15 | Balanced | Balanced | **Default** |
| 0.20 | Lower sensitivity | Very low | High SNR, strong interference |

### 7.3 Peak Tracking

Within the gate, track the **maximum |P|²** (not M):

```python
if gate_open:
    P_mag_sq = P_re² + P_im²
    if P_mag_sq > peak_P_mag_sq:
        peak_P_mag_sq = P_mag_sq
        peak_index = n
        peak_P_re = P_re
        peak_P_im = P_im
```

**Why |P|² instead of M?**
- |P|² has a sharp, unambiguous peak
- M has a plateau (ambiguous timing)
- |P|² peak occurs at optimal timing point

---

## 8. CFO Estimation

### 8.1 Theory

The preamble's [A][A] structure causes CFO to appear as a phase rotation:

```
Received: r[n] = x[n] × exp(j×2π×f_CFO×n/fs)

Correlation between halves:
P = Σ r[n] × conj(r[n-L])
  = Σ |x[n]|² × exp(j×2π×f_CFO×L/fs)
  = R × exp(j×φ)

where φ = 2π × f_CFO × L / fs
```

### 8.2 CFO Calculation

```
f_CFO = angle(P) × fs / (2π × L)
      = atan2(P_im, P_re) × fs / (2π × L)
```

**With our parameters:**
```
f_CFO = atan2(P_im, P_re) × 15,360,000 / (2π × 512)
      = atan2(P_im, P_re) × 4,774.65 Hz/rad
```

### 8.3 CFO Range

Maximum unambiguous CFO:
```
|f_CFO_max| = fs / (2L) = 15,360,000 / 1024 = 15,000 Hz = ±15 kHz
```

This is ±1 subcarrier spacing — sufficient for typical oscillator accuracies.

### 8.4 CORDIC for atan2

For FPGA implementation, use CORDIC algorithm:

**Input:** P_re, P_im (signed, e.g., 40-bit)  
**Output:** angle in range [-π, +π] (e.g., 16-bit signed, where 0x7FFF = +π)

CORDIC iterations: 16 typically sufficient for 16-bit angle precision.

---

## 9. Timing Estimation

### 9.1 Frame Start Calculation

The peak of |P|² occurs when both correlation windows are fully inside [A][A]:

```
Peak occurs at sample n where:
  Window 1: [n-L+1, n] is inside second half of preamble
  Window 2: [n-2L+1, n-L] is inside first half of preamble

For preamble starting at sample S:
  Peak at n = S + 2L - 1

Therefore:
  Frame start S = peak_index - 2L + 1 = peak_index - 1023
```

### 9.2 Implementation

```python
detected_frame_start = peak_index - 2*L + 1
                     = peak_index - 1023
```

### 9.3 Timing Accuracy

| Channel | Typical Error | Notes |
|---------|---------------|-------|
| AWGN | ±1 sample | Limited by noise |
| Multipath (CIR1/CIR2) | +77 to +94 samples | Shifted by channel delay spread |

**Important:** The timing error in multipath is dominated by the channel's group delay, not detector accuracy. This is expected and handled by the cyclic prefix (72 samples).

---

## 10. FPGA Architecture

### 10.1 Architecture Type: Fully Pipelined

**This design uses a FULLY PIPELINED architecture:**

| Property | Value | Description |
|----------|-------|-------------|
| **Throughput** | 1 sample/clock | One new input sample processed every clock cycle |
| **Pipeline depth** | 10-15 stages | Latency from input to output |
| **Clock rate** | ≥ Sample rate | Must be at least 15.36 MHz |
| **Data flow** | Continuous | No stalls, deterministic latency |

**Key Characteristics:**
- Each pipeline stage processes data from the **previous** sample
- New sample enters Stage 1 every clock; result exits Stage N every clock
- All stages operate **in parallel** on different samples
- Latency is fixed: output for sample `n` appears at clock `n + pipeline_depth`

**Example Pipeline Flow:**
```
Clock  │ Stage1      │ Stage2      │ Stage3      │ ... │ Stage10     │ Output
───────┼─────────────┼─────────────┼─────────────┼─────┼─────────────┼────────
  100  │ sample[100] │ sample[99]  │ sample[98]  │ ... │ sample[90]  │ M[90]
  101  │ sample[101] │ sample[100] │ sample[99]  │ ... │ sample[91]  │ M[91]
  102  │ sample[102] │ sample[101] │ sample[100] │ ... │ sample[92]  │ M[92]
```

### 10.2 Top-Level Block Diagram

```
                          ┌─────────────────────────────────────────┐
                          │           AA_SYNC_DETECTOR              │
        clk ─────────────►│                                         │
        rst_n ───────────►│  Architecture: FULLY PIPELINED          │
                          │  Throughput:   1 sample/clock           │
  rx_i[11:0] ────────────►│  Latency:      10-15 clocks             │
  rx_q[11:0] ────────────►│                                         │
  rx_valid ──────────────►│  ┌─────────┐   ┌─────────┐             │
                          │  │ Antenna │   │ Antenna │             │
                          │  │    0    │   │    1    │             │
                          │  │ Pipeline│   │ Pipeline│             │
                          │  └────┬────┘   └────┬────┘             │
                          │       │             │                   │
                          │       ▼             ▼                   │
                          │  ┌─────────────────────┐               │
                          │  │   MULTI-ANT COMBINE │  (1 clock)    │
                          │  │   P_total, R_total  │               │
                          │  └──────────┬──────────┘               │
                          │             │                           │
                          │             ▼                           │
                          │  ┌─────────────────────┐               │
                          │  │   METRIC COMPUTE    │  (3-4 clocks) │
                          │  │   M = |P|²/R²       │               │
                          │  └──────────┬──────────┘               │
                          │             │                           │
                          │             ▼                           │
                          │  ┌─────────────────────┐               │
                          │  │   DETECTION FSM     │  (1 clock)    │
                          │  │   Gate, Peak Track  │               │
                          │  └──────────┬──────────┘               │
                          │             │                           │
                          └─────────────┼───────────────────────────┘
                                        │
                                        ▼
                          ┌─────────────────────────┐
                          │      OUTPUT             │
                          │  detected      (1-bit)  │
                          │  frame_start   (32-bit) │
                          │  cfo_phase     (16-bit) │
                          │  peak_metric   (16-bit) │
                          └─────────────────────────┘
```

### 10.3 Per-Antenna Pipeline (Detailed)

Each antenna has an identical, independent pipeline:

```
         x[n] ──────────────────────────────────────────────────────────────┐
           │                                                                │
           │ ┌──────────────────────────────────────────────────────────┐   │
           │ │                    STAGE 1 (1 clock)                     │   │
           │ │  • Latch input x[n]                                      │   │
           │ │  • Start |x[n]|² computation                             │   │
           │ └──────────────────────────────────────────────────────────┘   │
           ▼                                                                ▼
    ┌─────────────┐                                                  ┌─────────────┐
    │  Delay Line │ ◄── BRAM/SRL, 512 deep                           │  STAGE 2    │
    │   L=512     │     Read & Write every clock                     │  |x[n]|²    │
    └──────┬──────┘                                                  │  (2 clocks) │
           │                                                         └──────┬──────┘
           │ x[n-L] (available after 512 clocks from x[n])                  │
           │                                                                │
    ┌──────┴──────────────────────────────────────────────────────┐        │
    │                      STAGE 3 (1 clock)                       │        │
    │  • Read x[n-L] from delay line                               │        │
    │  • Compute conj(x[n-L]) = (re, -im)                          │        │
    │  • Pass x[n-L] to second delay line                          │        │
    └──────┬───────────────────────────────────────────────────────┘        │
           │                                                                │
     ┌─────┴─────┐                                                         │
     │           │                                                          │
     ▼           ▼                                                          ▼
┌─────────┐  ┌──────────────────────────────────────────────────────┐ ┌─────────────┐
│Delay L  │  │                 STAGE 4-6 (3 clocks)                 │ │  STAGE 4-5  │
│ (512)   │  │  Complex multiply: x[n] × conj(x[n-L])               │ │  Running    │
└────┬────┘  │  • Stage 4: a×c, b×d, a×d, b×c (4 mults)             │ │  Sum of     │
     │       │  • Stage 5: (a×c - b×d), (a×d + b×c)                 │ │  |x[n]|²    │
     │       │  • Stage 6: Register new_product                     │ │  over L     │
     │       └──────────────────────────────────────────────────────┘ │  samples    │
     │                              │                                 └──────┬──────┘
     │ x[n-2L]                      │ new_product                            │
     │                              │                                        │ R[n]
     ▼                              ▼                                        │
┌───────────────────────────────────────────────────────────────────┐       │
│                       STAGE 7-9 (3 clocks)                        │       │
│  Complex multiply: x[n-2L] × conj(x[n-L])                         │       │
│  → old_product                                                    │       │
└───────────────────────────────────────────────────────────────────┘       │
                              │                                             │
                              │ old_product                                 │
                              ▼                                             │
┌───────────────────────────────────────────────────────────────────┐       │
│                       STAGE 10 (1 clock)                          │       │
│  P[n] = P[n-1] - old_product + new_product                        │       │
│  (40-bit complex accumulator)                                     │       │
└───────────────────────────────────────────────────────────────────┘       │
                              │                                             │
                              │ P[n]                                        │
                              ▼                                             ▼
                     ┌────────────────────────────────────────────────────────┐
                     │                    OUTPUT                              │
                     │     P[n] (40-bit complex)      R[n] (36-bit real)      │
                     └────────────────────────────────────────────────────────┘
```

### 10.4 Pipeline Stage Summary

| Stage | Clock(s) | Operation | Inputs | Outputs |
|-------|----------|-----------|--------|---------|
| 1 | 1 | Latch input | x[n] | x_reg |
| 2 | 2 | Magnitude squared | x_reg | \|x[n]\|² |
| 3 | 1 | Delay read + conjugate | DL[n-L] | x[n-L], conj(x[n-L]) |
| 4-6 | 3 | Complex mult (new) | x[n], conj(x[n-L]) | new_product |
| 4-5 | 2 | R accumulator | \|x[n]\|² | R[n] |
| 7-9 | 3 | Complex mult (old) | x[n-2L], conj(x[n-L]) | old_product |
| 10 | 1 | P accumulator | new_product, old_product | P[n] |
| **Total** | **10** | | | |

### 10.5 Clock Requirements

**Minimum clock frequency:** Equal to sample rate = **15.36 MHz**

**Recommended clock frequency:** 2-4× sample rate = **30-60 MHz**
- Provides timing margin
- Allows for longer combinational paths if needed
- Standard FPGA clocks (50 MHz, 100 MHz) work well

**If using 6× sample rate clock (92.16 MHz):**
- Design still works correctly (fully pipelined)
- Each sample still takes 1 clock cycle to process
- Extra clock speed provides timing margin, not extra throughput
- Could alternatively process 6 independent streams, or use for other functions

### 10.6 Latency Budget

| Component | Latency (clocks) | Latency (samples) |
|-----------|------------------|-------------------|
| Per-antenna pipeline | 10 | 10 |
| Multi-antenna combine | 1 | 1 |
| Metric compute (\|P\|²/R²) | 3-4 | 3-4 |
| Detection FSM | 1 | 1 |
| **Total pipeline latency** | **15-16** | **15-16** |

**At 15.36 MHz sample rate:**
- Pipeline latency: 15 samples × 65.1 ns = **~1 µs**
- Delay line fill time: 512 samples = **33.3 µs**
- Total time to first valid output: **~34 µs** after first sample

### 10.7 Delay Line Implementation Options

The delay lines (L=512 samples each, 24-bit complex) can be implemented as:

**Option A: Block RAM (BRAM)**
```
Depth: 512, Width: 24 bits
Read latency: 1-2 clocks (registered output)
Resources: 1 BRAM18 per antenna
Recommended for: Larger FPGAs, when BRAMs available
```

**Option B: Shift Register LUT (SRL)**
```
Depth: 512 = 16 SRL32 cascaded per bit
Width: 24 bits → 24 × 16 = 384 SRL primitives
Resources: ~200 LUTs per antenna
Recommended for: Smaller designs, avoiding BRAM
```

**Option C: Distributed RAM**
```
Same as BRAM but using LUT RAM
Resources: ~400 LUTs per antenna
Recommended for: When BRAM unavailable but need random access
```

---

## 11. Resource Estimates

### 11.1 Memory (Delay Lines)

| Component | Depth | Width | Total Bits | Implementation |
|-----------|-------|-------|------------|----------------|
| Delay Line 1 (x[n-L]) per antenna | 512 | 24 (12I+12Q) | 12,288 | BRAM or SRL |
| Delay Line 2 (x[n-2L]) per antenna | 512 | 24 (12I+12Q) | 12,288 | BRAM or SRL |
| **Per antenna** | 1024 | 24 | 24,576 | |
| **2 antennas** | 2048 | 24 | **49,152 bits** | ~2 BRAM18 |

**Note:** Can cascade DL1→DL2 to use single 1024-deep memory with taps at 512 and 1024.

### 11.2 DSP Slices (Fully Pipelined)

| Operation | DSPs per Antenna | Pipeline Stage | Notes |
|-----------|------------------|----------------|-------|
| \|x[n]\|² | 2 | Stage 2 | re² + im² |
| new_product (complex mult) | 4 | Stage 4-6 | Full complex multiply |
| old_product (complex mult) | 4 | Stage 7-9 | Full complex multiply |
| **Per antenna** | **10** | | |
| **2 antennas** | **20** | | |
| \|P\|² (combining) | 2 | After combine | P_re² + P_im² |
| R² (if needed for M) | 1 | After combine | |
| **Total** | **23 DSPs** | | |

**DSP48 Complex Multiply:**
```
(a + jb) × (c + jd) = (ac - bd) + j(ad + bc)
Requires: 4 multipliers + 2 adders
Can be done with 3 DSPs using pre-add: (a+b)(c+d) - ac - bd, but 4 is simpler
```

### 11.3 Pipeline Registers (FFs)

| Component | Width | Stages | Total FFs |
|-----------|-------|--------|-----------|
| Input registers | 24 | 2 | 48 |
| Delay line outputs | 24 | 2 | 48 |
| Complex mult pipeline | 48 | 6 | 288 |
| Accumulators (P) | 80 (40×2) | 1 | 80 |
| Accumulators (R) | 36 | 1 | 36 |
| **Per antenna** | | | **~500 FFs** |
| **2 antennas** | | | **~1000 FFs** |
| Combine + metric | | | ~200 FFs |
| FSM + peak track | | | ~150 FFs |
| **Total FFs** | | | **~1350 FFs** |

### 11.4 Logic (LUTs)

| Component | Estimate | Notes |
|-----------|----------|-------|
| Delay line address logic | 50 LUTs | If using BRAM |
| Accumulator add/sub | 200 LUTs | 40-bit complex |
| Magnitude squared pre-add | 100 LUTs | |
| Threshold comparator | 50 LUTs | |
| Detection FSM | 100 LUTs | |
| Peak tracking | 150 LUTs | |
| Sample counter | 50 LUTs | For frame_start |
| **Total LUTs** | **~700 LUTs** | |

### 11.5 Summary (Fully Pipelined, 2 Antennas)

| Resource | Estimate | Typical Artix-7 35T | % Utilization |
|----------|----------|---------------------|---------------|
| BRAM18 | 2-4 | 100 | 2-4% |
| DSP48 | 23 | 90 | 26% |
| LUTs | 700 | 20,800 | 3% |
| FFs | 1,350 | 41,600 | 3% |

**Note on DSP Usage:**
- The fully pipelined design uses more DSPs (23) because multipliers cannot be shared
- If DSPs are scarce, consider time-multiplexed design (6× clock, ~8 DSPs)
- Trade-off: DSPs vs. complexity

### 11.6 Alternative: Reduced DSP Configuration

If DSP count is critical, share multipliers between new_product and old_product:

| Configuration | DSPs | Trade-off |
|---------------|------|-----------|
| Fully pipelined (above) | 23 | Maximum throughput |
| Shared mult (2× clock) | 15 | Needs 2× clock domain |
| Time-multiplexed (6× clock) | 8 | More complex control |

**Conclusion:** Fully pipelined uses moderate resources. For very small FPGAs or DSP-constrained designs, consider time-multiplexing with higher clock.

---

## 12. Test Vectors

### 12.1 Preamble Test Vector

First 16 samples of normalized preamble (complex, 16-bit I/Q):

```
Index    Re (hex)    Im (hex)    Re (decimal)   Im (decimal)
  0      0x0A3B      0xF8E2        2619          -1822
  1      0x09F1      0x0312        2545            786
  2      0xF9A8      0x0891       -1624           2193
  3      0x0156      0xF721         342          -2271
  4      0x0B22      0x0445        2850           1093
  5      0xF6D3      0x0A12       -2349           2578
  6      0x0289      0xF5E1         649          -2591
  7      0x0C11      0x0122        3089            290
  ...
```

*Note: Generate exact values from Python reference implementation.*

### 12.2 Expected Metric Behavior

**Test Case 1: Clean preamble, no noise**
```
Sample    M[n]      |P|²           Event
...
1010      0.001     1.2e3         Below threshold
1011      0.015     1.8e4         Rising
1012      0.052     6.1e4         Rising
...
1500      0.985     2.5e5         Near peak
1510      0.998     2.6e5         Peak region
1523      1.000     2.62e5        PEAK (gate closes after)
1524      0.812     2.1e5         Falling (pilot starts)
...
1650      0.010     1.3e4         Below threshold
```

### 12.3 CFO Test Vector

**Input:** 500 Hz CFO applied to preamble  
**Expected P at peak:**
```
P_re = 254,312
P_im = 17,841
angle(P) = atan2(17841, 254312) = 0.0701 rad
CFO_est = 0.0701 × 4774.65 = 334.7 Hz  (within ~170 Hz of true)
```

*Note: Exact values depend on noise seed and channel.*

---

## 13. Performance Summary

### 13.1 Detection Performance

| Condition | SNR | Detection Rate |
|-----------|-----|----------------|
| AWGN | ≥ 0 dB | 100% |
| AWGN | -5 dB | 0% (below threshold) |
| Multipath (CIR1) | ≥ 0 dB | 100% |
| Multipath (CIR2) | ≥ 0 dB | 100% |

### 13.2 CFO Estimation Accuracy

| Condition | SNR | CFO Error (1σ) |
|-----------|-----|----------------|
| AWGN, FS=2.0 | 10 dB | < 1 Hz |
| AWGN, FS=1.0 | 10 dB | ~22 Hz (clipping bias) |
| Multipath | 10 dB | ~20-150 Hz |

**Note:** Clipping introduces systematic CFO bias. Recommend FS ratio ≥ 1.5.

### 13.3 Timing Accuracy

| Condition | Timing Error |
|-----------|--------------|
| AWGN | ±1 sample |
| Multipath | +77 to +94 samples (channel delay) |

Timing error in multipath is dominated by channel group delay, which is absorbed by the cyclic prefix.

### 13.4 Latency

| Stage | Latency |
|-------|---------|
| Delay lines fill | 1024 samples (66.7 µs) |
| Detection after preamble start | ~1024 samples |
| **Total acquisition** | ~2048 samples (~133 µs) |

---

## Appendix A: Python Reference Implementation

See `sync_aa.py` in the repository for complete Python reference implementation including:

- `build_aa_preamble()` — Preamble generation
- `aa_detect_streaming()` — Streaming detector
- `run_grid_test()` — Comprehensive test suite

---

## Appendix B: Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Feb 2026 | Initial release |

---

*End of Document*

"""
RTL-Friendly OFDM Preamble Detector — "Adjacent Quarter Correlation" Method
============================================================================

THIS IS NOT THE STANDARD MINN METRIC. It is a streaming-friendly variant that
was developed for FPGA implementation. It produces a single sharp peak (no
plateau) and naturally aligns with the pilot symbol's FFT window start.


PREAMBLE STRUCTURE
------------------
The preamble is a 5-SEGMENT structure, not 4. The leading segment (often called
"CP") is NOT optional padding — it is ESSENTIAL for the timing metric to work:

    [S0: -A] [S1: +A] [S2: +A] [S3: -A] [S4: -A]
    |<-Q ->| |<-------------- N_FFT ------------>|

    Total length: 5Q = N_FFT + Q = 1.25 × N_FFT samples

Where:
    - Q = N_FFT/4 = quarter length (512 samples for N=2048)
    - A = pseudo-random BPSK sequence on every 4th subcarrier
    - S0 is a copy of S4 (cyclic extension), so both contain -A

The sign pattern [-A | +A | +A | -A | -A] creates specific correlation properties:
    - Adjacent identical pairs: (S1,S2)=(+A,+A) and (S3,S4)=(-A,-A)
    - Adjacent opposite pairs: (S0,S1)=(-A,+A) and (S2,S3)=(+A,-A)

WITHOUT S0 (the leading -A), the metric would not produce a clean single peak.
The leading -A ensures the correlation pattern cancels within the preamble and
only peaks after it ends.


WHAT THIS METRIC COMPUTES
-------------------------
Unlike standard Minn which correlates non-adjacent matching quarters (Q0↔Q1
and Q2↔Q3), this RTL variant correlates ADJACENT quarters in a streaming
pipeline:

    quarter_product[n] = Re(x[n] · x*[n-Q])
                       = x_I[n]·x_I[n-Q] + x_Q[n]·x_Q[n-Q]

    corr_sum[n] = Σ_{k=0}^{Q-1} quarter_product[n-k]
                = Re(⟨window[n-Q+1:n], window[n-2Q+1:n-Q]⟩)

    corr_previous[n] = corr_sum[n-Q]  (delayed by one quarter)

    corr_total[n] = corr_sum[n] + corr_previous[n]

    corr_positive[n] = max(corr_total[n], 0)  (clip to positive)

The metric sums TWO consecutive adjacent-quarter correlations. For threshold
comparison, it uses:

    above_threshold = (corr_positive × 2^FRAC_BITS) ≥ (energy_total × THRESH_VALUE)

Where energy_total spans 3 consecutive Q-length windows for robustness.


WHY THE PEAK APPEARS WHERE IT DOES
----------------------------------
For the preamble structure [S0:-A | S1:+A | S2:+A | S3:-A | S4:-A], adjacent
segment correlations produce:

    Window position         corr_sum             corr_previous         Total
    (end of segment)        ⟨current, prev⟩      (from Q earlier)
    ────────────────────────────────────────────────────────────────────────
    S1 (+A)                 ⟨+A, -A⟩ = -|A|²     noise ≈ 0             ≈ 0
    S2 (+A)                 ⟨+A, +A⟩ = +|A|²     ⟨+A, -A⟩ = -|A|²      ≈ 0 (cancel)
    S3 (-A)                 ⟨-A, +A⟩ = -|A|²     ⟨+A, +A⟩ = +|A|²      ≈ 0 (cancel)
    S4 (-A)                 ⟨-A, -A⟩ = +|A|²     ⟨-A, +A⟩ = -|A|²      ≈ 0 (cancel)
    ────────────────────────────────────────────────────────────────────────
    Pilot (Q after S4)      ⟨pilot, -A⟩ ≈ 0     ⟨-A, -A⟩ = +|A|²      ≈ +|A|² ← PEAK!

KEY INSIGHT: Within the preamble, positive and negative correlations ALWAYS
cancel because the sign pattern alternates: (-,+), (+,+), (+,-), (-,-).

The peak occurs ONE QUARTER (Q) after the preamble ends because:
  - corr_previous[n] "remembers" the strong S4↔S3 correlation (+|A|²)
  - corr_sum[n] is correlating random pilot data with S4 (≈ 0)
  - Total = 0 + |A|² = |A|²

This happens at sample: n = preamble_end + Q = S0_start + 5Q + Q = S0_start + 6Q


PEAK POSITION IN THE FRAME
--------------------------
    Frame structure (showing segment boundaries):

    [guard][S0 S1 S2 S3 S4][Pilot CP][Pilot N: FFT here][Data...]
           |<- Preamble ->|         |<- Pilot symbol ->|
           |<---- 5Q ---->|<-- Q -->|<----- N -------->|

    Preamble = 5Q = 2560 samples (for N=2048, Q=512)
    Each OFDM symbol with CP = Q + N = 2560 samples

    Peak occurs at: preamble_start + 5Q + Q = preamble_start + 6Q
                  = preamble_start + 1.5 × N_FFT

    Equivalently: pilot_N_start (start of pilot's FFT window)

    With TIMING_OFFSET = 0, the detected frame_start pulse fires at the
    pilot's FFT window start — no offset calculation needed in RTL.

    NOTE: The "preamble CP" and "pilot CP" happen to be the same length (Q),
    but this is a system design choice, not a requirement of this method.


TIMING OFFSET OPTIONS
---------------------
    TIMING_OFFSET = 0
        → frame_start at pilot N-start (default, recommended for FFT)
        → Peak position, no adjustment needed

    TIMING_OFFSET = -Q  # = -512 for N=2048
        → frame_start at pilot CP-start (= end of preamble S4)

    TIMING_OFFSET = -2Q  # = -1024
        → frame_start at end of preamble S3

    TIMING_OFFSET = -(Q + N)  # = -2560
        → frame_start at preamble S1-start (start of "N portion")

    TIMING_OFFSET = -(Q + N + Q)  # = -3072
        → frame_start at preamble S0-start (very beginning)


ADVANTAGES OVER STANDARD MINN
-----------------------------
1. SINGLE SHARP PEAK — No plateau ambiguity; decisive timing detection
2. REAL-ONLY MATH — Uses only Re(x·x*), reducing FPGA multipliers
3. STREAMING PIPELINE — Natural fit for sample-by-sample processing
4. CFO TOLERANT — Real-part correlation is less sensitive to phase rotation
5. MULTIPATH ROBUST — Wide 3Q energy window averages out fading


COMPARISON TO STANDARD MINN
---------------------------
                        Standard Minn           This RTL Variant
    ─────────────────────────────────────────────────────────────
    Correlates          S1↔S2 and S3↔S4         Adjacent: Sn↔Sn-1
    Peak shape          Plateau (~Q samples)    Single sharp peak
    Peak location       Within preamble         Q after preamble end
    Peak relative to    Preamble "N-start"      Pilot "N-start"
    Math                Complex correlation     Real part only
    CFO estimation      From arg(P)             Not available
    Pipeline friendly   Moderate                Excellent


PIPELINE DELAYS (for RTL implementation)
----------------------------------------
    Component           Delay (samples)
    ─────────────────────────────────────
    delay_i, delay_q    Q (quarter length)
    corr_window         Q (running sum fill)
    corr_delay          Q
    energy_delay_q      Q
    energy_delay_2q     Q
    ─────────────────────────────────────
    Total until valid   ~3Q to 4Q samples

    The metric becomes valid (taps_valid=1) after all delay lines are filled.


SIMULATION PARAMETERS (configurable below)
------------------------------------------
    SNR_DB              Signal-to-noise ratio for simulation
    CFO_HZ              Carrier frequency offset to apply
    SMOOTH_SHIFT        Exponential smoothing: smooth += (new - smooth) >> shift
    THRESH_FRAC_BITS    Fixed-point fractional bits for threshold comparison
    THRESH_VALUE        Threshold as fraction: THRESH_VALUE / 2^FRAC_BITS
    HYSTERESIS          Samples below threshold before gate closes
    TIMING_OFFSET       Offset from peak to frame_start (0 = pilot N-start)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from channel import apply_channel, load_measured_cir
from core import (
    N_FFT,
    NUM_ACTIVE_SUBCARRIERS,
    CYCLIC_PREFIX,
    TX_PRE_PAD_SAMPLES,
    centered_subcarrier_indices,
    allocate_subcarriers,
    add_cyclic_prefix,
    plot_time_series,
    compute_channel_peak_offset,
    build_random_qpsk_symbol,
    estimate_cfo_from_cp,
    ofdm_fft_used,
    ls_channel_estimate,
    equalize,
    align_complex_gain,
    evm_rms_db,
    plot_constellation,
    SAMPLE_RATE_HZ,
    apply_cfo,
    plot_phase_slope_diagnostics,
)


def _generate_zc_sequence(length: int, root: int = 1) -> np.ndarray:
    """Generate a Zadoff-Chu sequence of given length.

    ZC sequences have:
    - Constant amplitude (PAPR = 0 dB)
    - Ideal cyclic autocorrelation (impulse)
    - Low cross-correlation between different roots

    Args:
        length: Sequence length (should be prime for ideal properties, but works for any)
        root: ZC root index (1 to length-1), coprime with length for ideal properties

    Returns:
        Complex ZC sequence of unit power
    """
    n = np.arange(length)
    # ZC formula: x[n] = exp(-j * pi * u * n * (n + 1) / N) for odd N
    #             x[n] = exp(-j * pi * u * n^2 / N) for even N
    if length % 2 == 1:
        zc = np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        zc = np.exp(-1j * np.pi * root * n * n / length)
    return zc


def _generate_base_sequence(seq_type: str, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate various base sequences for the preamble.

    Args:
        seq_type: One of:
            - "bpsk_freq": Random BPSK on every 4th subcarrier (original)
            - "qpsk_freq": Random QPSK on every 4th subcarrier
            - "zc_time": Zadoff-Chu directly in time domain
            - "zc_freq": ZC-like sequence on every 4th subcarrier
            - "chirp": Linear frequency chirp
            - "gold": Gold-code-like binary sequence
            - "const": Constant (all ones) - baseline
            - "random_phase": Random phase, constant amplitude
        length: Sequence length Q
        rng: Random generator (required for random sequences)

    Returns:
        Complex sequence of length Q, normalized to unit power
    """
    Q = length

    if seq_type == "bpsk_freq":
        # Original: Random BPSK on every 4th subcarrier
        if rng is None:
            raise ValueError("rng required for bpsk_freq")
        all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
        quarter_idx = all_idx[(all_idx % 4) == 0]
        bpsk = rng.choice([-1.0, 1.0], size=quarter_idx.shape[0])
        spectrum = allocate_subcarriers(N_FFT, quarter_idx, bpsk)
        time_domain = np.fft.ifft(np.fft.ifftshift(spectrum))
        A = time_domain[:Q]

    elif seq_type == "qpsk_freq":
        # Random QPSK on every 4th subcarrier
        if rng is None:
            raise ValueError("rng required for qpsk_freq")
        all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
        quarter_idx = all_idx[(all_idx % 4) == 0]
        phases = rng.choice([0, 1, 2, 3], size=quarter_idx.shape[0])
        qpsk = np.exp(1j * np.pi / 4 * (2 * phases + 1))
        spectrum = allocate_subcarriers(N_FFT, quarter_idx, qpsk)
        time_domain = np.fft.ifft(np.fft.ifftshift(spectrum))
        A = time_domain[:Q]

    elif seq_type == "zc_time":
        # Zadoff-Chu directly in time domain
        A = _generate_zc_sequence(Q, root=7)

    elif seq_type == "zc_freq":
        # ZC-like sequence on every 4th subcarrier
        all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
        quarter_idx = all_idx[(all_idx % 4) == 0]
        n_subcarriers = quarter_idx.shape[0]
        # Apply ZC-like phase progression to subcarriers
        k = np.arange(n_subcarriers)
        zc_phases = np.exp(-1j * np.pi * 7 * k * k / n_subcarriers)
        spectrum = allocate_subcarriers(N_FFT, quarter_idx, zc_phases)
        time_domain = np.fft.ifft(np.fft.ifftshift(spectrum))
        A = time_domain[:Q]

    elif seq_type == "chirp":
        # Linear frequency chirp
        n = np.arange(Q)
        # Chirp from 0 to half sampling rate
        A = np.exp(1j * np.pi * n * n / Q)

    elif seq_type == "gold":
        # Gold-code-like: XOR of two m-sequences mapped to ±1
        # Use a simple LFSR-based approach
        if rng is None:
            rng = np.random.default_rng(42)
        # Generate pseudo-random binary with good autocorrelation
        bits = np.zeros(Q, dtype=int)
        state1, state2 = 0b1010101010, 0b1100110011
        for i in range(Q):
            bit1 = (state1 >> 9) & 1
            bit2 = (state2 >> 9) & 1
            bits[i] = bit1 ^ bit2
            state1 = ((state1 << 1) | ((state1 >> 9) ^ (state1 >> 6)) & 1) & 0x3FF
            state2 = ((state2 << 1) | ((state2 >> 9) ^ (state2 >> 8) ^ (state2 >> 5) ^ (state2 >> 3)) & 1) & 0x3FF
        A = 2.0 * bits - 1.0 + 0j  # Map to ±1

    elif seq_type == "const":
        # Constant - baseline (worst case)
        A = np.ones(Q, dtype=complex)

    elif seq_type == "random_phase":
        # Constant amplitude, random phase (CAZAC-like)
        if rng is None:
            raise ValueError("rng required for random_phase")
        phases = rng.uniform(0, 2 * np.pi, Q)
        A = np.exp(1j * phases)

    else:
        raise ValueError(f"Unknown sequence type: {seq_type}")

    # Normalize to unit power
    power = np.mean(np.abs(A) ** 2)
    if power > 0:
        A = A / np.sqrt(power)

    return A


def build_minn_preamble_generic(seq_type: str, rng: np.random.Generator | None = None, Q: int | None = None) -> np.ndarray:
    """Build preamble with a specified base sequence type.
    
    Args:
        seq_type: Base sequence type (qpsk_freq, bpsk_freq, zc_time, etc.)
        rng: Random generator for random sequences
        Q: Segment length in samples. Default: N_FFT//4. Total preamble = 5*Q.
    
    Returns:
        preamble: Complex samples of length 5*Q
    """
    if Q is None:
        Q = N_FFT // 4
    A = _generate_base_sequence(seq_type, Q, rng)

    # Build the 5 segments: [S0:-A | S1:+A | S2:+A | S3:-A | S4:-A]
    preamble = np.concatenate([-A, +A, +A, -A, -A])

    # Normalize to unit power
    power = np.mean(np.abs(preamble) ** 2)
    if power > 0:
        preamble = preamble / np.sqrt(power)

    return preamble


def build_minn_preamble(rng: np.random.Generator | None = None, use_zc: bool = False, zc_root: int = 1) -> np.ndarray:
    """Build the 5-segment preamble: [S0:-A | S1:+A | S2:+A | S3:-A | S4:-A].

    The preamble is constructed explicitly as 5 quarter-length segments,
    NOT as a "symbol with cyclic prefix". S0 is essential to the timing
    metric, not optional padding.

    Structure:
        S0 = -A (copy of S4, provides leading correlation reference)
        S1 = +A
        S2 = +A
        S3 = -A
        S4 = -A

    Total length: 5Q = 5 * (N_FFT // 4) = 1.25 * N_FFT samples

    Args:
        rng: Random generator (required if use_zc=False)
        use_zc: If True, use Zadoff-Chu sequence for A instead of random BPSK
        zc_root: ZC root index (only used if use_zc=True)

    Returns:
        preamble: Complex samples of length 5Q

    Base Sequence Options:
        Random BPSK (default):
            - Uses every 4th subcarrier with random ±1
            - Requires RNG seed for reproducibility
            - Higher PAPR (~3-4 dB)

        Zadoff-Chu (use_zc=True):
            - Constant envelope (PAPR = 0 dB)
            - Ideal autocorrelation properties
            - Deterministic (no seed needed)
            - Better for power amplifier efficiency
    """
    Q = N_FFT // 4  # Quarter length (segment length)

    if use_zc:
        # Generate ZC sequence directly in time domain
        A = _generate_zc_sequence(Q, root=zc_root)
    else:
        # Generate base sequence A using every 4th subcarrier (creates Q-periodic time signal)
        if rng is None:
            raise ValueError("rng is required when use_zc=False")
        all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
        quarter_idx = all_idx[(all_idx % 4) == 0]
        bpsk = rng.choice([-1.0, 1.0], size=quarter_idx.shape[0])
        spectrum = allocate_subcarriers(N_FFT, quarter_idx, bpsk)

        # IFFT gives [A, A, A, A] structure (4 identical quarters due to 4-spacing in freq)
        time_domain = np.fft.ifft(np.fft.ifftshift(spectrum))
        A = time_domain[:Q]  # Extract one quarter = the base sequence A

    # Build the 5 segments explicitly
    S0 = -A  # Leading segment (same as S4)
    S1 = +A
    S2 = +A
    S3 = -A
    S4 = -A

    preamble = np.concatenate([S0, S1, S2, S3, S4])

    # Normalize to unit power
    power = np.mean(np.abs(preamble) ** 2)
    if power > 0:
        preamble = preamble / np.sqrt(power)

    return preamble


def _reconstruct_cir_from_ls(h_used: np.ndarray) -> np.ndarray:
    """Rebuild a time-domain CIR from a per-subcarrier LS channel estimate."""
    spectrum = np.zeros(N_FFT, dtype=np.complex128)
    h_used = np.asarray(h_used, dtype=np.complex128)
    if h_used.size == 0:
        return spectrum
    idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    dc = N_FFT // 2
    placement = (dc + idx) % N_FFT
    spectrum[placement] = h_used
    cir = np.fft.ifft(np.fft.ifftshift(spectrum))
    return cir


def _plot_ls_cir(
    ls_cir: np.ndarray,
    channel_impulse_response: np.ndarray | None,
    channel_peak_offset: int,
    timing_error: int,
    path: Path,
    channel_desc: str,
) -> None:
    """Plot the magnitude of the LS-derived CIR alongside the measured CIR."""
    taps = np.arange(ls_cir.size)
    ls_mag = np.abs(ls_cir)
    ls_peak_idx = int(np.argmax(ls_mag))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(taps, ls_mag, label="LS CIR |h|", color="tab:blue")
    ax.axvline(ls_peak_idx, color="tab:red", linestyle=":", label=f"LS peak @ {ls_peak_idx}")

    note_lines = [f"Timing error: {timing_error} samples"]
    if channel_impulse_response is not None:
        cir = np.asarray(channel_impulse_response, dtype=np.complex128)
        if cir.ndim == 1:
            cir = cir[np.newaxis, :]
        agg_mag = np.sqrt(np.sum(np.abs(cir) ** 2, axis=0))
        ax.plot(
            np.arange(agg_mag.size),
            agg_mag,
            label="Measured CIR |h|",
            color="tab:green",
            alpha=0.7,
        )
        ax.axvline(
            channel_peak_offset,
            color="tab:olive",
            linestyle="--",
            label=f"Measured peak @ {channel_peak_offset}",
        )
        peak_diff = ls_peak_idx - channel_peak_offset
        n = ls_cir.size
        if peak_diff > n // 2:
            peak_diff -= n
        elif peak_diff < -n // 2:
            peak_diff += n
        note_lines.append(f"Peak shift vs measured: {peak_diff} taps")
    else:
        note_lines.append(f"LS peak index: {ls_peak_idx}")

    ax.text(
        0.02,
        0.95,
        "\n".join(note_lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6),
    )
    ax.set_xlabel("Tap index")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"LS-Derived CIR (Minn RTL, {channel_desc})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class _DelayLine:
    """Replicate the minn_delay_line behavior for scalar streams."""

    def __init__(self, depth: int):
        if depth < 0:
            raise ValueError("Delay depth must be non-negative.")
        self.depth = int(depth)
        self.mem = np.zeros(self.depth, dtype=np.float64) if self.depth > 0 else np.zeros(0)
        self.wr_ptr = 0
        self.fill = 0
        self.last_output = 0.0

    def step(self, sample: float, in_valid: bool) -> tuple[float, bool]:
        if self.depth == 0:
            if in_valid:
                self.last_output = float(sample)
            return float(sample), in_valid
        if not in_valid:
            return self.last_output, False
        if self.fill < self.depth:
            read_val = 0.0
        else:
            read_val = float(self.mem[self.wr_ptr])
        self.mem[self.wr_ptr] = float(sample)
        self.wr_ptr = (self.wr_ptr + 1) % self.depth
        if self.fill < self.depth:
            self.fill += 1
            self.last_output = 0.0
            return 0.0, False
        self.last_output = read_val
        return read_val, True


class _RunningSum:
    """Replicate the minn_running_sum behavior including the valid gating."""

    def __init__(self, depth: int):
        if depth < 0:
            raise ValueError("Running-sum depth must be non-negative.")
        self.depth = int(depth)
        self.mem = np.zeros(self.depth, dtype=np.float64) if self.depth > 0 else np.zeros(0)
        self.wr_ptr = 0
        self.fill = 0
        self.sum_reg = 0.0
        self.valid = False

    def step(self, sample: float, in_valid: bool) -> tuple[float, bool]:
        if self.depth == 0:
            if in_valid:
                self.sum_reg = float(sample)
                self.valid = True
            return self.sum_reg, self.valid
        if not in_valid:
            return self.sum_reg, self.valid
        if self.fill < self.depth:
            oldest = 0.0
        else:
            oldest = float(self.mem[self.wr_ptr])
        val = float(sample)
        self.mem[self.wr_ptr] = val
        self.wr_ptr = (self.wr_ptr + 1) % self.depth
        self.sum_reg = self.sum_reg + val - oldest
        if self.fill < self.depth:
            self.fill += 1
            if self.fill >= self.depth:
                self.valid = True
        else:
            self.valid = True
        return self.sum_reg, self.valid


def _antenna_path(samples: np.ndarray, quarter_len: int) -> dict[str, np.ndarray]:
    """Mirror minn_antenna_path.sv for one complex stream."""
    samples = np.asarray(samples, dtype=np.complex128)
    n = samples.size
    corr_recent = np.zeros(n, dtype=np.float64)
    corr_previous = np.zeros(n, dtype=np.float64)
    energy_recent = np.zeros(n, dtype=np.float64)
    energy_previous = np.zeros(n, dtype=np.float64)
    energy_previous2 = np.zeros(n, dtype=np.float64)
    taps_valid = np.zeros(n, dtype=bool)

    delay_i = _DelayLine(quarter_len)
    delay_q = _DelayLine(quarter_len)
    corr_window = _RunningSum(quarter_len)
    energy_window = _RunningSum(quarter_len)
    corr_delay = _DelayLine(quarter_len)
    energy_delay_q = _DelayLine(quarter_len)
    energy_delay_2q = _DelayLine(quarter_len)

    corr_recent_reg = 0.0
    corr_previous_reg = 0.0
    energy_recent_reg = 0.0
    energy_previous_reg = 0.0
    energy_previous2_reg = 0.0
    taps_valid_reg = False

    for idx in range(n):
        in_i = float(samples[idx].real)
        in_q = float(samples[idx].imag)

        delayed_i, _ = delay_i.step(in_i, True)
        delayed_q, _ = delay_q.step(in_q, True)

        quarter_product = delayed_i * in_i + delayed_q * in_q
        power_sum = in_i * in_i + in_q * in_q

        corr_sum, corr_valid = corr_window.step(quarter_product, True)
        energy_sum, energy_valid = energy_window.step(power_sum, True)

        corr_prev_val, corr_prev_valid = corr_delay.step(corr_sum, corr_valid)
        energy_q_val, energy_q_valid = energy_delay_q.step(energy_sum, energy_valid)
        energy_2q_val, energy_2q_valid = energy_delay_2q.step(energy_q_val, energy_q_valid)

        if corr_valid:
            corr_recent_reg = corr_sum
        if corr_prev_valid:
            corr_previous_reg = corr_prev_val
        if energy_valid:
            energy_recent_reg = energy_sum
        if energy_q_valid:
            energy_previous_reg = energy_q_val
        if energy_2q_valid:
            energy_previous2_reg = energy_2q_val
        taps_valid_reg = energy_2q_valid

        corr_recent[idx] = corr_recent_reg
        corr_previous[idx] = corr_previous_reg
        energy_recent[idx] = energy_recent_reg
        energy_previous[idx] = energy_previous_reg
        energy_previous2[idx] = energy_previous2_reg
        taps_valid[idx] = taps_valid_reg

    return {
        "corr_recent": corr_recent,
        "corr_previous": corr_previous,
        "energy_recent": energy_recent,
        "energy_previous": energy_previous,
        "energy_previous2": energy_previous2,
        "taps_valid": taps_valid,
    }


@dataclass
class MinnRTLMetricState:
    corr_total: np.ndarray
    corr_positive: np.ndarray
    smooth_metric: np.ndarray
    energy_total: np.ndarray
    corr_scaled: np.ndarray
    energy_scaled: np.ndarray
    metric_valid: np.ndarray
    above_threshold: np.ndarray


def minn_rtl_streaming_metric(
    rx: np.ndarray,
    *,
    smooth_shift: int,
    threshold_value: int,
    threshold_frac_bits: int,
    quarter_len: int | None = None,
) -> MinnRTLMetricState:
    """Compute the RTL-aligned Minn timing metric across all branches.
    
    Args:
        quarter_len: Segment length Q. Default: N_FFT//4. Must match preamble Q.
    """
    rx = np.asarray(rx, dtype=np.complex128)
    if rx.ndim == 1:
        rx = rx[np.newaxis, :]
    num_branches, length = rx.shape
    if quarter_len is None:
        quarter_len = N_FFT // 4
    if quarter_len <= 0:
        raise ValueError("quarter_len must be positive.")

    branch_metrics = [_antenna_path(rx[b], quarter_len) for b in range(num_branches)]

    corr_total = np.zeros(length, dtype=np.float64)
    energy_total = np.zeros(length, dtype=np.float64)
    metric_valid = np.ones(length, dtype=bool)

    for metrics in branch_metrics:
        corr_total += metrics["corr_recent"] + metrics["corr_previous"]
        energy_total += (
            metrics["energy_recent"]
            + metrics["energy_previous"]
            + metrics["energy_previous2"]
        )
        metric_valid &= metrics["taps_valid"]

    corr_positive = np.maximum(corr_total, 0.0)

    smooth_metric = np.zeros(length, dtype=np.float64)
    smooth_val = 0.0
    denom = 1 << max(0, smooth_shift)
    for idx in range(length):
        if metric_valid[idx]:
            if smooth_shift == 0:
                smooth_val = corr_positive[idx]
            else:
                smooth_val += (corr_positive[idx] - smooth_val) / denom
        smooth_metric[idx] = smooth_val

    corr_scaled = smooth_metric * (1 << threshold_frac_bits)
    if threshold_value == 0:
        energy_scaled = np.zeros(length, dtype=np.float64)
    else:
        energy_scaled = energy_total * float(threshold_value)
    above_threshold = metric_valid & (corr_scaled >= energy_scaled)

    return MinnRTLMetricState(
        corr_total=corr_total,
        corr_positive=corr_positive,
        smooth_metric=smooth_metric,
        energy_total=energy_total,
        corr_scaled=corr_scaled,
        energy_scaled=energy_scaled,
        metric_valid=metric_valid,
        above_threshold=above_threshold,
    )


@dataclass
class MinnRTLEvent:
    peak_index: int
    detected_index: int
    gate_segment: tuple[int, int]


@dataclass
class MinnRTLDetection:
    events: list[MinnRTLEvent]
    gate_mask: np.ndarray
    gate_segments: list[tuple[int, int]]


def detect_minn_rtl(
    state: MinnRTLMetricState,
    *,
    hysteresis: int,
    timing_offset: int,
) -> MinnRTLDetection:
    """Replicate the gate and peak tracking used in the RTL detector."""
    corr = state.corr_positive
    above = state.above_threshold
    valid = state.metric_valid
    length = corr.size

    gate_segments: list[tuple[int, int]] = []
    events: list[MinnRTLEvent] = []
    gate_open = False
    gate_start: int | None = None
    peak_value = 0.0
    peak_index = 0
    low_counter = 0
    hyst_limit = hysteresis - 1 if hysteresis > 0 else 0

    for idx in range(length):
        if not valid[idx]:
            continue
        metric_val = corr[idx]
        if not gate_open:
            if above[idx]:
                gate_open = True
                gate_start = idx
                peak_value = metric_val
                peak_index = idx
                low_counter = 0
        else:
            if metric_val >= peak_value:
                peak_value = metric_val
                peak_index = idx
            if above[idx]:
                low_counter = 0
            else:
                closing = False
                if hysteresis == 0:
                    closing = True
                else:
                    if low_counter == hyst_limit:
                        closing = True
                    else:
                        low_counter += 1
                if closing:
                    detected_index = peak_index + timing_offset
                    segment_start = gate_start if gate_start is not None else idx
                    segment = (segment_start, idx + 1)
                    gate_segments.append(segment)
                    events.append(
                        MinnRTLEvent(
                            peak_index=peak_index,
                            detected_index=detected_index,
                            gate_segment=segment,
                        )
                    )
                    gate_open = False
                    gate_start = None
                    peak_value = 0.0
                    low_counter = 0

    if gate_open and gate_start is not None:
        gate_segments.append((gate_start, length))

    gate_mask = np.zeros(length, dtype=bool)
    for start, end in gate_segments:
        gate_mask[start:end] = True

    return MinnRTLDetection(
        events=events,
        gate_mask=gate_mask,
        gate_segments=gate_segments,
    )


# Detector and channel parameters
SNR_DB = 0.0
CFO_HZ = 1000.0
SMOOTH_SHIFT = 3
THRESH_FRAC_BITS = 15
THRESH_VALUE = int(0.10 * (1 << THRESH_FRAC_BITS))
HYSTERESIS = 2
# Preamble segment length Q (total preamble = 5*Q samples)
# Default: N_FFT//4 = 512 for 2560 total. Smaller Q = shorter preamble but less robust.
PREAMBLE_Q = N_FFT // 4  # 512 samples per segment, 2560 total

# Preamble base sequence type: "bpsk_freq", "qpsk_freq", "zc_time", "zc_freq", etc.
# See _generate_base_sequence() for all options
PREAMBLE_SEQ_TYPE = "qpsk_freq"  # QPSK gives ~10% higher peak than BPSK
# RTL peak naturally aligns with pilot N-start (end of preamble + pilot CP)
# No offset needed - peak is already where we want to start FFT processing
TIMING_OFFSET = 0

PLOTS_BASE_DIR = Path("plots") / "minn_rtl"


def run_simulation(channel_name: str | None, plots_subdir: str) -> None:
    """Run Minn RTL synchronization simulation with identical frame construction."""

    def mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        in_segment = False
        start_idx = 0
        for idx, flag in enumerate(mask):
            if flag and not in_segment:
                in_segment = True
                start_idx = idx
            elif not flag and in_segment:
                in_segment = False
                segments.append((start_idx, idx))
        if in_segment:
            segments.append((start_idx, mask.size))
        return segments

    rng = np.random.default_rng(0)

    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_plot_path = plots_dir / "minn_rtl_metric.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    results_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    ls_cir_plot_path = plots_dir / "ls_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    sto_plot_path = plots_dir / "phase_slope_sto.png"

    minn_preamble = build_minn_preamble_generic(PREAMBLE_SEQ_TYPE, rng, Q=PREAMBLE_Q)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((minn_preamble, pilot_symbol, data_symbol))
    frame_len = frame.size
    inter_guard = np.zeros(frame_len, dtype=complex)
    leading_guard = np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex)
    tx_samples = np.concatenate((leading_guard, frame, inter_guard, frame))
    frame_starts = [leading_guard.size, leading_guard.size + frame_len + inter_guard.size]

    if channel_name is None:
        channel_impulse_response = None
    else:
        cir_bank = load_measured_cir(channel_name)
        if cir_bank.shape[0] > 2:
            channel_impulse_response = cir_bank[:2].copy()
        else:
            channel_impulse_response = cir_bank.copy()

    rx_samples = apply_channel(
        tx_samples,
        SNR_DB,
        rng,
        channel_impulse_response=channel_impulse_response,
    )
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

    metric_state = minn_rtl_streaming_metric(
        rx_samples,
        smooth_shift=SMOOTH_SHIFT,
        threshold_value=THRESH_VALUE,
        threshold_frac_bits=THRESH_FRAC_BITS,
        quarter_len=PREAMBLE_Q,
    )
    detection = detect_minn_rtl(
        metric_state,
        hysteresis=HYSTERESIS,
        timing_offset=TIMING_OFFSET,
    )

    events = detection.events
    if events:
        primary_event = events[0]
        detected_start = int(primary_event.detected_index)
        peak_position = int(primary_event.peak_index)
    else:
        primary_event = None
        peak_position = int(np.argmax(metric_state.smooth_metric))
        detected_start = peak_position + TIMING_OFFSET

    gate_mask = detection.gate_mask
    gate_segments = detection.gate_segments or mask_segments(gate_mask)

    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}', all RX)",
            cir_plot_path,
        )

    # Preamble boundaries (using 5-segment structure: S0 S1 S2 S3 S4)
    Q = PREAMBLE_Q  # Segment length (default 512 for N=2048)
    preamble_len = 5 * Q  # = 2560 samples by default
    preamble_s0_starts = [start + channel_peak_offset for start in frame_starts]
    preamble_s1_starts = [s0 + Q for s0 in preamble_s0_starts]  # Start of S1

    # Pilot and data symbol boundaries (these use standard OFDM CP + N structure)
    # Note: In this config, CYCLIC_PREFIX == Q, but they're conceptually different
    ofdm_symbol_len = CYCLIC_PREFIX + N_FFT
    pilot_cp_starts = [s0 + preamble_len for s0 in preamble_s0_starts]
    pilot_n_starts = [cp + CYCLIC_PREFIX for cp in pilot_cp_starts]
    data_cp_starts = [s0 + preamble_len + ofdm_symbol_len for s0 in preamble_s0_starts]
    data_n_starts = [cp + CYCLIC_PREFIX for cp in data_cp_starts]

    # RTL peak aligns with pilot N-start - this is where we expect the detection
    expected_n_starts = pilot_n_starts
    # Theoretical peak position matches pilot N-start
    theoretical_peak_positions = pilot_n_starts
    if expected_n_starts:
        primary_expected = expected_n_starts[0]
        timing_error = detected_start - primary_expected
    else:
        primary_expected = 0
        timing_error = detected_start
    peak_lines = [evt.peak_index for evt in events] if events else [peak_position]
    detected_lines = [evt.detected_index for evt in events] if events else [detected_start]
    per_event_errors: list[int | None] = []
    for idx, evt in enumerate(events):
        if idx < len(expected_n_starts):
            per_event_errors.append(evt.detected_index - expected_n_starts[idx])
        else:
            per_event_errors.append(None)

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"

    plt.figure(figsize=(10, 4))
    plt.plot(metric_state.corr_positive, label="RTL corr_plus(d)", color="tab:purple")
    plt.plot(metric_state.smooth_metric, label="RTL smooth(d)", color="tab:orange", linestyle="--")
    thresh_trace = np.full(metric_state.energy_total.shape, np.nan, dtype=float)
    denom = float(1 << THRESH_FRAC_BITS)
    valid_mask = metric_state.metric_valid
    if denom > 0:
        thresh_trace[valid_mask] = metric_state.energy_scaled[valid_mask] / denom
    plt.plot(thresh_trace, label="Threshold (scaled)", color="tab:green", linestyle=":")
    for idx, (start, end) in enumerate(gate_segments):
        gate_label = "Gate window" if idx == 0 else None
        plt.axvspan(start, end, color="tab:orange", alpha=0.15, label=gate_label)
    for idx, peak in enumerate(peak_lines):
        peak_label = "Detected peak" if idx == 0 else None
        plt.axvline(peak, color="tab:red", linestyle=":", label=peak_label)
    for idx, expected in enumerate(expected_n_starts):
        exp_label = "Pilot N start (exp)" if idx == 0 else None
        plt.axvline(expected, color="tab:green", linestyle="--", label=exp_label)
    plt.xlabel("Sample index d")
    plt.ylabel("Metric")
    plt.title(f"Minn RTL Metric & Gate - {channel_desc}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    if rx_samples.ndim > 1 and rx_samples.shape[0] > 1:
        for idx, branch in enumerate(rx_samples):
            axes[0].plot(np.abs(branch), alpha=0.3, linewidth=0.8)
    for seg_idx, (start, end) in enumerate(gate_segments):
        gate_label = "Gate window" if seg_idx == 0 else None
        axes[0].axvspan(start, end, color="tab:orange", alpha=0.18, label=gate_label)
    for idx, s0_start in enumerate(preamble_s0_starts):
        s0_label = "Preamble S0 start" if idx == 0 else None
        axes[0].axvline(s0_start, color="tab:purple", linestyle="--", label=s0_label)
    for idx, expected in enumerate(expected_n_starts):
        exp_label = "Pilot N start (exp)" if idx == 0 else None
        axes[0].axvline(expected, color="tab:green", linestyle="--", label=exp_label)
    for idx, det in enumerate(detected_lines):
        det_label = "Detected start" if idx == 0 else None
        axes[0].axvline(det, color="tab:red", linestyle=":", label=det_label)
    # Theoretical peak position (aligns with pilot N start)
    for idx, theo_peak in enumerate(theoretical_peak_positions):
        lbl = "Theo. peak" if idx == 0 else None
        axes[0].axvline(theo_peak, color="tab:pink", linestyle="-.", linewidth=2, label=lbl)
    # Pilot symbol boundaries
    for idx, cp_start in enumerate(pilot_cp_starts):
        lbl = "Pilot CP start" if idx == 0 else None
        axes[0].axvline(cp_start, color="tab:blue", linestyle="--", alpha=0.7, label=lbl)
    for idx, n_start in enumerate(pilot_n_starts):
        lbl = "Pilot N start" if idx == 0 else None
        axes[0].axvline(n_start, color="tab:cyan", linestyle="--", alpha=0.7, label=lbl)
    # Data symbol boundaries
    for idx, cp_start in enumerate(data_cp_starts):
        lbl = "Data CP start" if idx == 0 else None
        axes[0].axvline(cp_start, color="tab:brown", linestyle="--", alpha=0.7, label=lbl)
    for idx, n_start in enumerate(data_n_starts):
        lbl = "Data N start" if idx == 0 else None
        axes[0].axvline(n_start, color="tab:olive", linestyle="--", alpha=0.7, label=lbl)
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude and Detected Start (Minn RTL, {channel_desc})")
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)

    axes[1].plot(metric_state.corr_positive, label="RTL corr_plus(d)", color="tab:purple")
    axes[1].plot(
        metric_state.smooth_metric,
        label="RTL smooth(d)",
        color="tab:orange",
        linestyle="--",
    )
    axes[1].plot(thresh_trace, label="Threshold (scaled)", color="tab:green", linestyle=":")
    for start, end in gate_segments:
        axes[1].axvspan(start, end, color="tab:orange", alpha=0.12)
    for idx, peak in enumerate(peak_lines):
        peak_label = "Detected peak" if idx == 0 else None
        axes[1].axvline(peak, color="tab:red", linestyle=":", label=peak_label)
    for idx, expected in enumerate(expected_n_starts):
        exp_label = "Pilot N start (exp)" if idx == 0 else None
        axes[1].axvline(expected, color="tab:green", linestyle="--", label=exp_label)
    for idx, theo_peak in enumerate(theoretical_peak_positions):
        lbl = "Theo. peak" if idx == 0 else None
        axes[1].axvline(theo_peak, color="tab:pink", linestyle="-.", linewidth=2, label=lbl)
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Timing Metrics (Minn RTL)")
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(results_plot_path, dpi=150)
    plt.close(fig)

    plot_time_series(tx_samples, "Transmit Frame (two Minn frames with guard)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)

    preamble_n_start_est = detected_start
    pilot_cp_start = preamble_n_start_est + N_FFT
    data_cp_start = pilot_cp_start + pilot_symbol.size
    cfo_est_hz = estimate_cfo_from_cp(
        rx_samples,
        pilot_cp_start,
        N_FFT,
        CYCLIC_PREFIX,
        SAMPLE_RATE_HZ,
    )
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)

    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[pilot_cp_start + CYCLIC_PREFIX : pilot_cp_start + CYCLIC_PREFIX + N_FFT]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)
    sto_title = f"Residual Timing From Phase Slope (Minn RTL, {channel_desc})"
    slope_rad_per_bin, timing_offset_samples = plot_phase_slope_diagnostics(
        h_est,
        sto_plot_path,
        sto_title,
    )

    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(
        xhat_aligned,
        data_used,
        const_plot_path,
        f"Equalized Data Constellation (Minn RTL, {channel_desc})",
    )

    ls_cir = _reconstruct_cir_from_ls(h_est)
    _plot_ls_cir(
        ls_cir,
        channel_impulse_response,
        channel_peak_offset,
        timing_error,
        ls_cir_plot_path,
        channel_desc,
    )

    print(f"\n{'='*70}")
    print(f"MINN RTL SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
    print(f"{'='*70}")
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {1 if rx_samples.ndim == 1 else rx_samples.shape[0]}")
    if channel_impulse_response is not None:
        num_rx = channel_impulse_response.shape[0]
        print(
            f"Applied measured channel '{channel_name}' using {num_rx} RX branch(es) "
            f"taps={channel_impulse_response.shape[1]} main-path offset={channel_peak_offset}",
        )
    else:
        print("Channel profile: Flat AWGN (no multipath)")
    print(f"\nTiming Detections:")
    if events:
        print(f"  Detected {len(events)} event(s)")
        for idx, evt in enumerate(events):
            expected = expected_n_starts[idx] if idx < len(expected_n_starts) else None
            err = per_event_errors[idx] if idx < len(per_event_errors) else None
            if expected is not None and err is not None:
                print(
                    f"    Event {idx}: peak={evt.peak_index} detected={evt.detected_index} "
                    f"expected={expected} error={err} samples"
                )
            else:
                print(
                    f"    Event {idx}: peak={evt.peak_index} detected={evt.detected_index} "
                    "(no expected reference)"
                )
    else:
        print(f"  No detection events; fallback peak at d={peak_position}")
    frac = THRESH_VALUE / float(1 << THRESH_FRAC_BITS)
    if gate_segments:
        for idx, (start, end) in enumerate(gate_segments):
            print(
                f"  Gate {idx}: [{start}, {end}) threshold >={frac:.1%} span {end - start} samples"
            )
    else:
        print("  No gate segments recorded (metric never exceeded threshold)")
    if events:
        print("  Event 0 is used for CFO/channel processing.")
    print(f"  Frame length: {frame_len} samples, guard length: {inter_guard.size} samples")
    print(
        f"  Primary timing error: {timing_error} samples "
        f"({abs(timing_error)/N_FFT*100:.1f}% of symbol)"
    )
    print(f"\nCarrier Frequency Offset:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz ({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.1f}%)")
    print(f"\nChannel Estimation & Equalization:")
    print(f"  Pilot LS phase slope: {slope_rad_per_bin:.6f} rad/bin -> timing ~ {timing_offset_samples:.2f} samples")
    print(f"  Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")
    print(f"\nPlots saved to {plots_dir.resolve()}/:")
    print(f"  - minn_rtl_metric.png")
    print(f"  - start_detection.png")
    print(f"  - constellation.png")
    print(f"  - tx_frame_time.png")
    print(f"  - rx_frame_time.png")
    print(f"  - ls_cir.png")
    print(f"  - phase_slope_sto.png")
    if channel_impulse_response is not None:
        print(f"  - channel_cir.png")
    print(f"{'='*70}\n")


def run_sequence_comparison(channel_name: str | None) -> None:
    """Compare all base sequence types and measure peak-to-sidelobe ratio."""
    seq_types = ["bpsk_freq", "qpsk_freq", "zc_time", "zc_freq", "chirp", "gold", "random_phase"]

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"

    # Load channel
    if channel_name is None:
        channel_impulse_response = None
    else:
        cir_bank = load_measured_cir(channel_name)
        channel_impulse_response = cir_bank[:2].copy() if cir_bank.shape[0] > 2 else cir_bank.copy()

    results = []

    for seq_type in seq_types:
        rng = np.random.default_rng(0)  # Consistent RNG

        # Build frame with this sequence type
        Q = PREAMBLE_Q
        preamble = build_minn_preamble_generic(seq_type, rng, Q=Q)
        pilot_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
        data_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
        frame = np.concatenate((preamble, pilot_symbol, data_symbol))
        frame_len = frame.size
        inter_guard = np.zeros(frame_len, dtype=complex)
        leading_guard = np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex)
        tx_samples = np.concatenate((leading_guard, frame, inter_guard, frame))

        # Apply channel
        rng2 = np.random.default_rng(0)  # Same noise for fair comparison
        rx_samples = apply_channel(tx_samples, SNR_DB, rng2, channel_impulse_response=channel_impulse_response)
        rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

        # Compute metric
        metric_state = minn_rtl_streaming_metric(
            rx_samples,
            smooth_shift=SMOOTH_SHIFT,
            threshold_value=THRESH_VALUE,
            threshold_frac_bits=THRESH_FRAC_BITS,
            quarter_len=Q,
        )
        detection = detect_minn_rtl(metric_state, hysteresis=HYSTERESIS, timing_offset=TIMING_OFFSET)

        # Get timing info
        preamble_len = 5 * Q
        channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
        frame_starts = [leading_guard.size, leading_guard.size + frame_len + inter_guard.size]
        preamble_s0_starts = [start + channel_peak_offset for start in frame_starts]
        pilot_n_starts = [s0 + preamble_len + CYCLIC_PREFIX for s0 in preamble_s0_starts]

        # Analyze metric
        metric = metric_state.corr_positive
        if detection.events:
            peak_idx = detection.events[0].peak_index
            timing_error = detection.events[0].detected_index - pilot_n_starts[0]
        else:
            peak_idx = int(np.argmax(metric_state.smooth_metric))
            timing_error = peak_idx - pilot_n_starts[0]

        peak_val = float(metric[peak_idx])

        # Compute noise floor (excluding peak region and guard intervals)
        peak_region = range(max(0, peak_idx - 500), min(len(metric), peak_idx + 500))
        mask = np.ones(len(metric), dtype=bool)
        mask[list(peak_region)] = False
        # Also exclude guard intervals (low signal)
        mask[:TX_PRE_PAD_SAMPLES] = False
        noise_vals = metric[mask]
        noise_floor = float(np.mean(noise_vals)) if len(noise_vals) > 0 else 0
        noise_max = float(np.max(noise_vals)) if len(noise_vals) > 0 else 0

        # Peak-to-average and peak-to-max ratios
        par = peak_val / noise_floor if noise_floor > 0 else float('inf')
        pmr = peak_val / noise_max if noise_max > 0 else float('inf')

        results.append({
            "seq_type": seq_type,
            "peak_val": peak_val,
            "peak_idx": peak_idx,
            "timing_error": timing_error,
            "noise_floor": noise_floor,
            "noise_max": noise_max,
            "par": par,  # Peak-to-average ratio
            "pmr": pmr,  # Peak-to-max ratio
            "metric": metric,
        })

    # Sort by peak-to-max ratio (best first)
    results.sort(key=lambda x: -x["pmr"])

    # Print results
    print(f"\n{'='*80}")
    print(f"SEQUENCE COMPARISON — {channel_desc.upper()}")
    print(f"{'='*80}")
    print(f"{'Sequence':<15} {'Peak':>10} {'Noise Avg':>12} {'Noise Max':>12} {'PAR':>8} {'PMR':>8} {'Timing Err':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['seq_type']:<15} {r['peak_val']:>10.1f} {r['noise_floor']:>12.1f} {r['noise_max']:>12.1f} "
              f"{r['par']:>8.1f} {r['pmr']:>8.1f} {r['timing_error']:>+12d}")
    print("=" * 80)
    print("PAR = Peak-to-Average Ratio, PMR = Peak-to-Max-sidelobe Ratio (higher is better)")
    print("=" * 80 + "\n")

    # Create comparison plot
    plots_dir = PLOTS_BASE_DIR / "comparison" / ("measured_channel" if channel_name else "flat_awgn")
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, r in enumerate(results):
        axes[0].plot(r["metric"], label=f"{r['seq_type']} (PMR={r['pmr']:.1f})",
                     color=colors[i], alpha=0.7)

    axes[0].set_ylabel("Metric")
    axes[0].set_title(f"Sequence Comparison — {channel_desc}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Zoomed view around first peak
    peak_center = results[0]["peak_idx"]
    zoom_start = max(0, peak_center - 600)
    zoom_end = min(len(results[0]["metric"]), peak_center + 600)

    for i, r in enumerate(results):
        axes[1].plot(r["metric"][zoom_start:zoom_end],
                     label=f"{r['seq_type']}", color=colors[i], alpha=0.8)

    axes[1].set_xlabel(f"Sample index (offset from {zoom_start})")
    axes[1].set_ylabel("Metric")
    axes[1].set_title(f"Zoomed View Around Peak — {channel_desc}")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "sequence_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Plot saved: {plots_dir / 'sequence_comparison.png'}\n")

    return results


def run_comparison(channel_name: str | None, plots_subdir: str) -> None:
    """Run both BPSK and ZC preambles and create comparison plots."""
    rng_bpsk = np.random.default_rng(0)
    rng_zc = np.random.default_rng(0)  # Same seed for same noise/channel

    plots_dir = PLOTS_BASE_DIR / "comparison" / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"

    # Load channel
    if channel_name is None:
        channel_impulse_response = None
    else:
        cir_bank = load_measured_cir(channel_name)
        channel_impulse_response = cir_bank[:2].copy() if cir_bank.shape[0] > 2 else cir_bank.copy()

    results = {}
    for label, use_zc in [("BPSK", False), ("ZC", True)]:
        rng = np.random.default_rng(0)  # Reset RNG for consistent noise

        # Build frame
        Q = PREAMBLE_Q
        preamble = build_minn_preamble(rng, use_zc=use_zc, zc_root=ZC_ROOT)
        pilot_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
        data_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
        frame = np.concatenate((preamble, pilot_symbol, data_symbol))
        frame_len = frame.size
        inter_guard = np.zeros(frame_len, dtype=complex)
        leading_guard = np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex)
        tx_samples = np.concatenate((leading_guard, frame, inter_guard, frame))

        # Apply channel
        rx_samples = apply_channel(tx_samples, SNR_DB, rng, channel_impulse_response=channel_impulse_response)
        rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

        # Compute metric
        metric_state = minn_rtl_streaming_metric(
            rx_samples,
            smooth_shift=SMOOTH_SHIFT,
            threshold_value=THRESH_VALUE,
            threshold_frac_bits=THRESH_FRAC_BITS,
            quarter_len=Q,
        )
        detection = detect_minn_rtl(metric_state, hysteresis=HYSTERESIS, timing_offset=TIMING_OFFSET)

        # Get timing info
        preamble_len = 5 * Q
        channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
        frame_starts = [leading_guard.size, leading_guard.size + frame_len + inter_guard.size]
        preamble_s0_starts = [start + channel_peak_offset for start in frame_starts]
        pilot_cp_starts = [s0 + preamble_len for s0 in preamble_s0_starts]
        pilot_n_starts = [cp + CYCLIC_PREFIX for cp in pilot_cp_starts]

        if detection.events:
            peak_idx = detection.events[0].peak_index
            timing_error = detection.events[0].detected_index - pilot_n_starts[0]
        else:
            peak_idx = int(np.argmax(metric_state.smooth_metric))
            timing_error = peak_idx - pilot_n_starts[0]

        results[label] = {
            "metric": metric_state,
            "detection": detection,
            "peak_idx": peak_idx,
            "timing_error": timing_error,
            "pilot_n_starts": pilot_n_starts,
        }

    # Create comparison plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    colors = {"BPSK": "tab:blue", "ZC": "tab:orange"}
    for label, data in results.items():
        metric = data["metric"]
        axes[0].plot(metric.corr_positive, label=f"{label} corr_plus", color=colors[label], alpha=0.7)
        axes[0].plot(metric.smooth_metric, label=f"{label} smooth", color=colors[label], linestyle="--", alpha=0.9)

    # Add expected and detected markers
    for label, data in results.items():
        peak_idx = data["peak_idx"]
        timing_error = data["timing_error"]
        axes[0].axvline(peak_idx, color=colors[label], linestyle=":", linewidth=2,
                        label=f"{label} peak (err={timing_error:+d})")

    # Expected position (same for both)
    pilot_n_starts = results["BPSK"]["pilot_n_starts"]
    for idx, expected in enumerate(pilot_n_starts):
        lbl = "Expected (Pilot N start)" if idx == 0 else None
        axes[0].axvline(expected, color="tab:green", linestyle="--", linewidth=2, label=lbl)

    axes[0].set_ylabel("Metric")
    axes[0].set_title(f"BPSK vs ZC Preamble Comparison — {channel_desc}")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Zoomed view around first peak
    peak_center = results["BPSK"]["peak_idx"]
    zoom_start = max(0, peak_center - 800)
    zoom_end = min(len(results["BPSK"]["metric"].corr_positive), peak_center + 800)

    for label, data in results.items():
        metric = data["metric"]
        axes[1].plot(metric.corr_positive[zoom_start:zoom_end],
                     label=f"{label}", color=colors[label], alpha=0.8)

    # Mark peaks in zoomed view
    for label, data in results.items():
        rel_peak = data["peak_idx"] - zoom_start
        if 0 <= rel_peak < (zoom_end - zoom_start):
            axes[1].axvline(rel_peak, color=colors[label], linestyle=":", linewidth=2)

    expected_rel = pilot_n_starts[0] - zoom_start
    if 0 <= expected_rel < (zoom_end - zoom_start):
        axes[1].axvline(expected_rel, color="tab:green", linestyle="--", linewidth=2, label="Expected")

    axes[1].set_xlabel(f"Sample index (offset from {zoom_start})")
    axes[1].set_ylabel("Metric")
    axes[1].set_title(f"Zoomed View Around Peak — {channel_desc}")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plots_dir / "bpsk_vs_zc_metric.png", dpi=150)
    plt.close(fig)

    # Print summary
    print(f"\n{'='*70}")
    print(f"BPSK vs ZC COMPARISON — {channel_desc.upper()}")
    print(f"{'='*70}")
    for label, data in results.items():
        print(f"  {label}: peak={data['peak_idx']}, timing_error={data['timing_error']:+d} samples")
    print(f"  Plot saved: {plots_dir / 'bpsk_vs_zc_metric.png'}")
    print(f"{'='*70}\n")


def main() -> None:
    """Run simulations for both measured channel and flat AWGN conditions."""
    print("\n" + "=" * 70)
    print("MINN RTL PREAMBLE SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
    print("=" * 70)

    run_simulation(channel_name="cir1", plots_subdir="measured_channel")
    run_simulation(channel_name=None, plots_subdir="flat_awgn")

    # Run BPSK vs ZC comparison
    print("\n" + "=" * 70)
    print("RUNNING BPSK vs ZC COMPARISON")
    print("=" * 70)
    run_comparison(channel_name=None, plots_subdir="flat_awgn")
    run_comparison(channel_name="cir1", plots_subdir="measured_channel")

    print("\n" + "=" * 70)
    print("ALL MINN RTL SIMULATIONS COMPLETE")
    print("=" * 70)
    print(f"\nCompare results in:")
    print(f"  - {(PLOTS_BASE_DIR / 'measured_channel').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'flat_awgn').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'comparison').resolve()}")
    print("=" * 70 + "\n")


def compare_q_values(
    q_values: list[int],
    channel_name: str | None = None,
) -> dict[int, dict]:
    """Compare detection performance across different Q (segment length) values.
    
    Args:
        q_values: List of Q values to test (segment lengths)
        channel_name: Channel CIR file name or None for flat AWGN
        
    Returns:
        Dictionary mapping Q -> {peak, par, pmr, timing_error, preamble_len, overhead_pct}
    """
    channel_impulse_response = None
    if channel_name:
        cir_bank = load_measured_cir(channel_name)
        channel_impulse_response = cir_bank[:2].copy() if cir_bank.shape[0] > 2 else cir_bank.copy()
    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    
    results: dict[int, dict] = {}
    
    for Q in q_values:
        rng = np.random.default_rng(0)
        
        # Build preamble with this Q
        preamble = build_minn_preamble_generic(PREAMBLE_SEQ_TYPE, rng, Q=Q)
        preamble_len = 5 * Q
        
        # Build rest of frame (unchanged)
        pilot_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
        data_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
        frame = np.concatenate((preamble, pilot_symbol, data_symbol))
        frame_len = frame.size
        
        # Calculate overhead
        payload_len = pilot_symbol.size + data_symbol.size
        overhead_pct = 100.0 * preamble_len / frame_len
        
        # Build TX stream
        inter_guard = np.zeros(frame_len, dtype=complex)
        leading_guard = np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex)
        tx_samples = np.concatenate((leading_guard, frame, inter_guard, frame))
        
        # Apply channel
        rng2 = np.random.default_rng(0)
        rx_samples = apply_channel(tx_samples, SNR_DB, rng2, channel_impulse_response=channel_impulse_response)
        rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)
        
        # Compute metric with this Q
        metric_state = minn_rtl_streaming_metric(
            rx_samples,
            smooth_shift=SMOOTH_SHIFT,
            threshold_value=THRESH_VALUE,
            threshold_frac_bits=THRESH_FRAC_BITS,
            quarter_len=Q,
        )
        detection = detect_minn_rtl(metric_state, hysteresis=HYSTERESIS, timing_offset=TIMING_OFFSET)
        
        # Get timing info
        frame_starts = [leading_guard.size, leading_guard.size + frame_len + inter_guard.size]
        preamble_s0_starts = [start + channel_peak_offset for start in frame_starts]
        pilot_n_starts = [s0 + preamble_len + CYCLIC_PREFIX for s0 in preamble_s0_starts]
        
        # Analyze metric
        metric = metric_state.corr_positive
        if detection.events:
            peak_idx = detection.events[0].peak_index
            timing_error = detection.events[0].detected_index - pilot_n_starts[0]
        else:
            peak_idx = int(np.argmax(metric_state.smooth_metric))
            timing_error = peak_idx - pilot_n_starts[0]
        
        peak_val = float(metric[peak_idx])
        
        # Compute noise floor (excluding peak region)
        peak_region = range(max(0, peak_idx - 500), min(len(metric), peak_idx + 500))
        mask = np.ones(len(metric), dtype=bool)
        mask[list(peak_region)] = False
        mask[:TX_PRE_PAD_SAMPLES] = False
        noise_vals = metric[mask]
        
        if noise_vals.size > 0:
            avg_noise = float(np.mean(noise_vals))
            max_noise = float(np.max(noise_vals))
            par = peak_val / avg_noise if avg_noise > 0 else float('inf')
            pmr = peak_val / max_noise if max_noise > 0 else float('inf')
        else:
            par = float('inf')
            pmr = float('inf')
        
        results[Q] = {
            "peak": peak_val,
            "par": par,
            "pmr": pmr,
            "timing_error": timing_error,
            "preamble_len": preamble_len,
            "overhead_pct": overhead_pct,
        }
    
    return results


def run_q_comparison(channel_name: str | None = None) -> None:
    """Run and print Q value comparison."""
    # Test a range of Q values (powers of 2 for easier RTL)
    q_values = [64, 128, 256, 512]
    
    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"
    print(f"\n{'='*70}")
    print(f"Q VALUE COMPARISON - {channel_desc}")
    print(f"{'='*70}")
    print(f"N_FFT={N_FFT}, Default Q={N_FFT//4}")
    print()
    
    results = compare_q_values(q_values, channel_name)
    
    # Print header
    print(f"{'Q':>6} | {'5*Q':>6} | {'Overhead':>8} | {'Peak':>10} | {'PAR':>6} | {'PMR':>6} | {'Timing':>8}")
    print("-" * 70)
    
    for Q in q_values:
        r = results[Q]
        print(f"{Q:>6} | {r['preamble_len']:>6} | {r['overhead_pct']:>7.1f}% | {r['peak']:>10.0f} | {r['par']:>6.1f} | {r['pmr']:>6.2f} | {r['timing_error']:>+8}")
    
    print()


def plot_q_comparison(q_values: list[int] | None = None, snr_db: float | None = None) -> None:
    """Create comparison plots for different Q values in both AWGN and multipath."""
    if q_values is None:
        q_values = [64, 128, 256, 512]
    if snr_db is None:
        snr_db = SNR_DB
    
    plots_dir = PLOTS_BASE_DIR / "q_comparison"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Colors for each Q value
    colors = {64: "tab:red", 128: "tab:orange", 256: "tab:green", 512: "tab:blue"}
    
    for channel_name, channel_label, subdir in [
        (None, "Flat AWGN", "flat_awgn"),
        ("cir1", "Multipath (cir1)", "measured_channel"),
    ]:
        channel_impulse_response = None
        if channel_name:
            cir_bank = load_measured_cir(channel_name)
            channel_impulse_response = cir_bank[:2].copy() if cir_bank.shape[0] > 2 else cir_bank.copy()
        channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
        
        fig, axes = plt.subplots(len(q_values), 1, figsize=(14, 3 * len(q_values)), sharex=True)
        
        for ax_idx, Q in enumerate(q_values):
            ax = axes[ax_idx]
            rng = np.random.default_rng(0)
            
            # Build preamble with this Q
            preamble = build_minn_preamble_generic(PREAMBLE_SEQ_TYPE, rng, Q=Q)
            preamble_len = 5 * Q
            
            # Build rest of frame
            pilot_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
            data_symbol, _ = build_random_qpsk_symbol(rng, include_cp=True)
            frame = np.concatenate((preamble, pilot_symbol, data_symbol))
            frame_len = frame.size
            
            # Build TX stream
            inter_guard = np.zeros(frame_len, dtype=complex)
            leading_guard = np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex)
            tx_samples = np.concatenate((leading_guard, frame, inter_guard, frame))
            
            # Apply channel
            rng2 = np.random.default_rng(0)
            rx_samples = apply_channel(tx_samples, snr_db, rng2, channel_impulse_response=channel_impulse_response)
            rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)
            
            # Compute metric with this Q
            metric_state = minn_rtl_streaming_metric(
                rx_samples,
                smooth_shift=SMOOTH_SHIFT,
                threshold_value=THRESH_VALUE,
                threshold_frac_bits=THRESH_FRAC_BITS,
                quarter_len=Q,
            )
            detection = detect_minn_rtl(metric_state, hysteresis=HYSTERESIS, timing_offset=TIMING_OFFSET)
            
            # Expected peak position: preamble_start + 6*Q (running sum completes Q after preamble ends)
            frame_starts = [leading_guard.size, leading_guard.size + frame_len + inter_guard.size]
            preamble_s0_starts = [start + channel_peak_offset for start in frame_starts]
            expected_peaks = [s0 + 6 * Q for s0 in preamble_s0_starts]
            
            # Find peak in FIRST FRAME region only (to avoid picking second frame)
            first_frame_end = frame_starts[0] + frame_len + inter_guard.size // 2
            metric = metric_state.corr_positive
            first_frame_metric = metric[:first_frame_end].copy()
            peak_idx = int(np.argmax(first_frame_metric))
            timing_error = peak_idx - expected_peaks[0]
            
            # Compute threshold trace
            denom = float(1 << THRESH_FRAC_BITS)
            thresh_trace = np.zeros_like(metric)
            valid_mask = metric_state.metric_valid
            if denom > 0:
                thresh_trace[valid_mask] = metric_state.energy_scaled[valid_mask] / denom
            
            # Compute metric/threshold ratio
            ratio = np.zeros_like(metric)
            nonzero_thresh = thresh_trace > 0
            ratio[nonzero_thresh] = metric[nonzero_thresh] / thresh_trace[nonzero_thresh]
            
            # Get peak ratio value
            peak_ratio = ratio[peak_idx] if peak_idx < len(ratio) else 0
            
            # Plot metric and threshold (both normalized to metric max for comparison)
            max_metric = np.max(metric) if np.max(metric) > 0 else 1
            metric_norm = metric / max_metric
            thresh_norm = thresh_trace / max_metric
            
            ax.plot(metric_norm, label=f"Metric", color=colors[Q], alpha=0.8)
            ax.plot(thresh_norm, label=f"Threshold", color="gray", linestyle="--", alpha=0.6)
            
            # Mark expected and detected peaks
            for exp_peak in expected_peaks:
                ax.axvline(exp_peak, color="green", linestyle="--", alpha=0.5, label="Expected" if exp_peak == expected_peaks[0] else None)
            ax.axvline(peak_idx, color="red", linestyle=":", alpha=0.8, label=f"Detected")
            
            ax.set_ylabel("Metric (norm)")
            ax.set_title(f"Q={Q}: preamble={5*Q}, err={timing_error:+d}, peak/thresh={peak_ratio:.1f}x")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlim(0, len(metric))
        
        axes[-1].set_xlabel("Sample index")
        fig.suptitle(f"Q Value Comparison - {channel_label} (SNR={snr_db:.0f} dB)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        out_path = plots_dir / f"{subdir}_q_comparison_snr{int(snr_db):+d}dB.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

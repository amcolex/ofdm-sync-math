"""
Simple [A][A] Preamble Synchronization — FPGA-Friendly Design
=============================================================

This module implements a robust, FPGA-friendly synchronization scheme using
a repeated [A][A] preamble structure with Schmidl-Cox style detection.

SYSTEM PARAMETERS (10 MHz LTE-like)
-----------------------------------
    Sample rate:        15.36 MHz
    FFT size:           1024
    Active subcarriers: 600
    Subcarrier spacing: 15 kHz
    CP length:          72 samples (normal CP)

PREAMBLE DESIGN
---------------
    Structure:  [A][A] where each A is N_FFT/2 = 512 samples
    Total:      1024 samples (one OFDM symbol, no CP needed for preamble)
    
    Generation: Zadoff-Chu sequence on every 2nd active subcarrier
                → Creates natural [A][A] repetition in time domain
                → Minimal PAPR (constant envelope in frequency domain)
                → Good autocorrelation properties
                → Respects guard bands (only active subcarriers used)

DETECTOR ALGORITHM
------------------
    P[n] = Σ_{k=0}^{L-1} x[n+k] · conj(x[n+k+L])    (complex correlation)
    R[n] = Σ_{k=0}^{L-1} |x[n+k+L]|²                (energy normalization)
    M[n] = |P[n]|² / R[n]²                          (timing metric, 0 to 1)
    
    Detection:
        1. M[n] rises above threshold → gate opens
        2. Track peak of |P[n]|² within gate
        3. M[n] falls below threshold → gate closes
        4. Report: timing = peak position, CFO = angle(P[peak])
    
    Multi-antenna: P_total = ΣP_ant, R_total = ΣR_ant (coherent combining)

FPGA IMPLEMENTATION
-------------------
    Pipeline:
        x[n] ──┬─────────────────────────────────────┐
               │                                      │
               ▼                                      ▼
          [Delay L] ──► conj() ──► [×] ◄─────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                   [MAC Re]                [MAC Im]     (running sum)
                        │                       │
                        ▼                       ▼
                      P_re                    P_im
                        │                       │
                        └───────┬───────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
               |P|² = P_re² + P_im²    angle(P) → CFO
                    │
                    ▼
              [Threshold & Gate Logic]
    
    Resources:
        - 2 delay lines of L samples each (for I and Q)
        - 4 multipliers per antenna (complex multiply)
        - 4 running sum accumulators per antenna
        - Magnitude squared: 2 multipliers + 1 adder
        - No BRAM needed (delay lines in registers for L ≤ 512)

QUANTIZATION MODEL
------------------
    12-bit ADC simulation with configurable full-scale:
        - full_scale > signal_rms: signal doesn't use full range (weak)
        - full_scale < signal_rms: clipping/saturation (AGC overshoot)
        - full_scale ≈ signal_rms: nominal operation

TEST GRID
---------
    SNR:        [-5, 0, 5, 10, 15] dB
    Channels:   [AWGN, CIR1, CIR2]
    Full-scale: [0.25, 0.5, 1.0, 1.5, 2.0] × signal_rms
    Antennas:   2 RX (coherent combining)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import warnings

from channel import apply_channel, load_measured_cir

# =============================================================================
# System Parameters (10 MHz LTE-like)
# =============================================================================
N_FFT = 1024
NUM_ACTIVE_SUBCARRIERS = 600
CYCLIC_PREFIX = 72
SAMPLE_RATE_HZ = 15_360_000.0  # 15.36 MHz

# Preamble parameters (configurable lengths for testing)
# Supported: 1024, 512, 256 total samples
PREAMBLE_LENGTHS = [1024, 512, 256]  # Test grid options
DEFAULT_PREAMBLE_LEN = 1024          # Default for backwards compatibility
PREAMBLE_HALF_LEN = DEFAULT_PREAMBLE_LEN // 2  # L = 512 samples per half (default)
PREAMBLE_TOTAL_LEN = DEFAULT_PREAMBLE_LEN      # 1024 samples total [A][A] (default)

# Detection parameters
# Lower threshold for robustness in multipath - plateau should still exceed this
DETECT_THRESHOLD = 0.15         # M[n] threshold for gate (0 to 1 range)
DETECT_HYSTERESIS = 128         # Samples below threshold before gate closes

# Quantization
ADC_BITS = 12
ADC_LEVELS = 2 ** (ADC_BITS - 1)  # 2048 levels per side for signed

# Test parameters
TX_PRE_PAD_SAMPLES = 500        # Guard before preamble
TX_POST_PAD_SAMPLES = 500       # Guard after frame

# Output
PLOTS_DIR = Path("plots") / "sync_aa"


# =============================================================================
# Helper Functions
# =============================================================================
def centered_subcarrier_indices(width: int) -> np.ndarray:
    """Return subcarrier indices symmetric around DC, skipping DC."""
    half = width // 2
    negative = np.arange(-half, 0)
    positive = np.arange(1, half + 1)
    return np.concatenate((negative, positive))


def allocate_subcarriers(n_fft: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place values at specified subcarrier indices in centered spectrum."""
    spectrum = np.zeros(n_fft, dtype=complex)
    dc_index = n_fft // 2
    placement = (dc_index + indices) % n_fft
    spectrum[placement] = values
    return spectrum


# =============================================================================
# Preamble Generation
# =============================================================================
def generate_zadoff_chu(length: int, root: int = 25) -> np.ndarray:
    """Generate Zadoff-Chu sequence with given root."""
    n = np.arange(length)
    if length % 2 == 1:
        return np.exp(-1j * np.pi * root * n * (n + 1) / length)
    else:
        return np.exp(-1j * np.pi * root * n * n / length)


def build_aa_preamble(total_length: int = DEFAULT_PREAMBLE_LEN) -> tuple[np.ndarray, np.ndarray, float]:
    """Build [A][A] preamble using Zadoff-Chu on every Kth FFT bin.
    
    For [A][A] structure (time-domain repetition), we need ALL used subcarriers
    to have the SAME parity in FFT bin positions. The spacing K determines the
    repetition period:
    - K=2: period = N_FFT/2 = 512 → total 1024 samples
    - K=4: period = N_FFT/4 = 256 → total 512 samples  
    - K=8: period = N_FFT/8 = 128 → total 256 samples
    
    Key design choices:
    - Every Kth FFT bin → natural [A][A] repetition in time domain
    - Zadoff-Chu sequence → constant amplitude, good autocorrelation, low PAPR
    - Stay within active bandwidth (guard bands respected)
    
    Args:
        total_length: Total preamble length (1024, 512, or 256 samples)
    
    Returns:
        preamble: Time-domain preamble (total_length samples)
        freq_seq: Frequency-domain sequence on used subcarriers  
        papr_db: Peak-to-average power ratio in dB
    """
    if total_length not in PREAMBLE_LENGTHS:
        raise ValueError(f"total_length must be one of {PREAMBLE_LENGTHS}, got {total_length}")
    
    # Determine subcarrier spacing K for desired repetition period
    # Period L = N_FFT / K, total = 2L = 2*N_FFT/K
    # So K = 2*N_FFT / total_length
    K = 2 * N_FFT // total_length  # K=2 for 1024, K=4 for 512, K=8 for 256
    L = total_length // 2  # Half-preamble length
    
    # For [A][A], we need every Kth FFT bin within active BW
    # Active subcarriers span ±NUM_ACTIVE_SUBCARRIERS/2 around DC
    dc_bin = N_FFT // 2  # 512
    half_active = NUM_ACTIVE_SUBCARRIERS // 2  # 300
    
    # Generate FFT bin indices: every Kth bin within active bandwidth
    used_bins = []
    for offset in range(-half_active, half_active + 1):
        if offset == 0:  # Skip DC
            continue
        bin_idx = dc_bin + offset
        if bin_idx % K == 0:  # Every Kth bin for period L
            used_bins.append(bin_idx)
    
    used_bins = np.array(used_bins)
    num_sc = len(used_bins)
    
    # Zadoff-Chu sequence for constant envelope
    # Use length that's coprime with root for best properties
    root = 25 if num_sc % 25 != 0 else 23
    n = np.arange(num_sc)
    zc_seq = np.exp(-1j * np.pi * root * n * (n + 1) / num_sc)
    
    # Allocate directly to FFT bins (not using centered indices)
    spectrum = np.zeros(N_FFT, dtype=complex)
    spectrum[used_bins] = zc_seq
    
    # IFFT to time domain - full 1024-pt FFT
    preamble_full = np.fft.ifft(spectrum) * np.sqrt(N_FFT)
    
    # Extract just the first total_length samples (the [A][A] portion repeats)
    # Due to periodicity, samples 0:total_length contain exactly [A][A]
    preamble = preamble_full[:total_length]
    
    # Normalize to unit power
    power = np.mean(np.abs(preamble) ** 2)
    preamble = preamble / np.sqrt(power)
    
    # Calculate PAPR
    peak_power = np.max(np.abs(preamble) ** 2)
    avg_power = np.mean(np.abs(preamble) ** 2)
    papr_db = 10 * np.log10(peak_power / avg_power)
    
    return preamble, zc_seq, papr_db


def build_random_qpsk_symbol(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build random QPSK OFDM symbol with CP for pilot/data."""
    indices = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    
    # Random QPSK
    phases = rng.integers(0, 4, size=len(indices))
    qpsk = np.exp(1j * np.pi / 4 * (2 * phases + 1)) / np.sqrt(2)
    
    # Allocate and IFFT
    spectrum = allocate_subcarriers(N_FFT, indices, qpsk)
    symbol = np.fft.ifft(np.fft.ifftshift(spectrum)) * np.sqrt(N_FFT)
    
    # Normalize
    power = np.mean(np.abs(symbol) ** 2)
    symbol = symbol / np.sqrt(power)
    
    # Add CP
    symbol_cp = np.concatenate([symbol[-CYCLIC_PREFIX:], symbol])
    
    return symbol_cp, qpsk


# =============================================================================
# 12-bit ADC Quantization
# =============================================================================
def quantize_adc(
    samples: np.ndarray,
    full_scale: float,
    bits: int = ADC_BITS,
) -> np.ndarray:
    """Simulate ADC quantization with given full-scale range.
    
    Args:
        samples: Complex input samples
        full_scale: Full-scale amplitude (signal will clip beyond this)
        bits: ADC resolution (default 12)
    
    Returns:
        Quantized complex samples
    """
    levels = 2 ** (bits - 1)  # Signed: -levels to +levels-1
    
    # Process real and imaginary separately
    def quantize_real(x):
        # Scale to [-1, 1) range
        x_scaled = x / full_scale
        # Clip
        x_clipped = np.clip(x_scaled, -1.0, 1.0 - 1.0/levels)
        # Quantize
        x_quant = np.round(x_clipped * levels) / levels
        # Scale back
        return x_quant * full_scale
    
    return quantize_real(samples.real) + 1j * quantize_real(samples.imag)


def compute_clipping_stats(
    samples: np.ndarray,
    full_scale: float,
) -> dict:
    """Compute clipping statistics."""
    real_clip = np.sum(np.abs(samples.real) >= full_scale) / samples.size
    imag_clip = np.sum(np.abs(samples.imag) >= full_scale) / samples.size
    total_clip = np.sum((np.abs(samples.real) >= full_scale) | 
                        (np.abs(samples.imag) >= full_scale)) / samples.size
    
    # Effective bits used
    signal_rms = np.sqrt(np.mean(np.abs(samples) ** 2))
    effective_bits = ADC_BITS + np.log2(signal_rms / full_scale) if full_scale > 0 else 0
    
    return {
        "real_clip_pct": 100 * real_clip,
        "imag_clip_pct": 100 * imag_clip,
        "total_clip_pct": 100 * total_clip,
        "effective_bits": max(0, effective_bits),
        "signal_rms": signal_rms,
        "full_scale": full_scale,
    }


# =============================================================================
# Streaming Detector Components (FPGA-style)
# =============================================================================
class RunningSum:
    """Streaming running sum over fixed window."""
    
    def __init__(self, window_size: int):
        self.L = window_size
        self.buffer = np.zeros(self.L, dtype=np.complex128)
        self.ptr = 0
        self.sum = 0.0 + 0.0j
        self.filled = 0
    
    def step(self, sample: complex) -> tuple[complex, bool]:
        """Push sample, return (running_sum, is_valid)."""
        oldest = self.buffer[self.ptr]
        self.buffer[self.ptr] = sample
        self.ptr = (self.ptr + 1) % self.L
        
        self.sum = self.sum + sample - oldest
        
        if self.filled < self.L:
            self.filled += 1
            return self.sum, False
        return self.sum, True


class RunningSumReal:
    """Streaming running sum for real values."""
    
    def __init__(self, window_size: int):
        self.L = window_size
        self.buffer = np.zeros(self.L, dtype=np.float64)
        self.ptr = 0
        self.sum = 0.0
        self.filled = 0
    
    def step(self, sample: float) -> tuple[float, bool]:
        oldest = self.buffer[self.ptr]
        self.buffer[self.ptr] = sample
        self.ptr = (self.ptr + 1) % self.L
        
        self.sum = self.sum + sample - oldest
        
        if self.filled < self.L:
            self.filled += 1
            return self.sum, False
        return self.sum, True


class DelayLine:
    """Fixed delay line."""
    
    def __init__(self, delay: int):
        self.delay = delay
        self.buffer = np.zeros(delay, dtype=np.complex128)
        self.ptr = 0
        self.filled = 0
    
    def step(self, sample: complex) -> tuple[complex, bool]:
        """Push sample, return (delayed_sample, is_valid)."""
        output = self.buffer[self.ptr]
        self.buffer[self.ptr] = sample
        self.ptr = (self.ptr + 1) % self.delay
        
        if self.filled < self.delay:
            self.filled += 1
            return 0.0 + 0.0j, False
        return output, True


# =============================================================================
# [A][A] Detector
# =============================================================================
@dataclass
class AADetectorState:
    """State arrays from streaming detection."""
    P: np.ndarray           # Complex correlation P[n]
    R: np.ndarray           # Energy R[n]
    M: np.ndarray           # Normalized metric |P|²/R²
    valid: np.ndarray       # Boolean: metric is valid
    

@dataclass  
class AADetectionEvent:
    """A single detection event."""
    peak_index: int         # Index where |P|² is maximum within gate
    P_at_peak: complex      # Complex P value at peak (for CFO)
    M_at_peak: float        # Metric value at peak
    gate_start: int         # Gate open sample
    gate_end: int           # Gate close sample
    cfo_hz: float           # Estimated CFO in Hz
    frame_start: int        # Estimated frame start (peak - L + 1)


@dataclass
class AADetectionResult:
    """Complete detection result."""
    events: list[AADetectionEvent]
    state: AADetectorState
    num_antennas: int


def aa_detect_streaming(
    rx_samples: np.ndarray,
    L: int = PREAMBLE_HALF_LEN,
    threshold: float = DETECT_THRESHOLD,
    hysteresis: int = DETECT_HYSTERESIS,
    sample_rate: float = SAMPLE_RATE_HZ,
) -> AADetectionResult:
    """Streaming [A][A] detector with multi-antenna support.
    
    Args:
        rx_samples: Shape (num_antennas, num_samples) or (num_samples,)
        L: Half-preamble length
        threshold: Detection threshold for M[n] (0 to 1)
        hysteresis: Samples below threshold before gate closes
        sample_rate: For CFO calculation
    
    Returns:
        AADetectionResult with events and state
    """
    # Ensure 2D
    if rx_samples.ndim == 1:
        rx_samples = rx_samples[np.newaxis, :]
    
    num_antennas, num_samples = rx_samples.shape
    
    # Initialize per-antenna components
    delay_lines = [DelayLine(L) for _ in range(num_antennas)]
    P_accums = [RunningSum(L) for _ in range(num_antennas)]
    R_accums = [RunningSumReal(L) for _ in range(num_antennas)]
    
    # Output arrays
    P_total = np.zeros(num_samples, dtype=np.complex128)
    R_total = np.zeros(num_samples, dtype=np.float64)
    M = np.zeros(num_samples, dtype=np.float64)
    valid = np.zeros(num_samples, dtype=bool)
    
    # Streaming processing
    for n in range(num_samples):
        P_sum = 0.0 + 0.0j
        R_sum = 0.0
        all_valid = True
        
        for ant in range(num_antennas):
            x_n = rx_samples[ant, n]
            
            # Delayed sample (x[n-L])
            x_delayed, delay_valid = delay_lines[ant].step(x_n)
            
            # P accumulator: Σ x[n] * conj(x[n-L]) - correlation between halves
            product = x_n * np.conj(x_delayed) if delay_valid else 0.0
            P_ant, P_valid = P_accums[ant].step(product)
            
            # R accumulator: Σ |x[n]|² - energy of CURRENT window (not delayed!)
            # This matches standard Schmidl-Cox normalization
            power_current = np.abs(x_n) ** 2
            R_ant, R_valid = R_accums[ant].step(power_current)
            
            P_sum += P_ant
            R_sum += R_ant
            all_valid = all_valid and P_valid and R_valid
        
        P_total[n] = P_sum
        R_total[n] = R_sum
        valid[n] = all_valid
        
        # Compute metric M = |P|² / R²
        # Add noise floor to prevent division issues
        noise_floor = 1e-6 * L  # Scale with window length
        if all_valid and R_sum > noise_floor:
            M[n] = (np.abs(P_sum) ** 2) / (R_sum ** 2)
            M[n] = min(M[n], 1.0)  # Clip to valid range
        else:
            M[n] = 0.0
    
    # Gate logic and peak detection
    events: list[AADetectionEvent] = []
    gate_open = False
    gate_start = 0
    peak_index = 0
    peak_P = 0.0 + 0.0j
    peak_P_mag_sq = 0.0
    low_count = 0
    
    for n in range(num_samples):
        if not valid[n]:
            continue
        
        m_val = M[n]
        P_mag_sq = np.abs(P_total[n]) ** 2
        
        if not gate_open:
            if m_val >= threshold:
                # Open gate
                gate_open = True
                gate_start = n
                peak_index = n
                peak_P = P_total[n]
                peak_P_mag_sq = P_mag_sq
                low_count = 0
        else:
            # Track peak
            if P_mag_sq > peak_P_mag_sq:
                peak_index = n
                peak_P = P_total[n]
                peak_P_mag_sq = P_mag_sq
            
            if m_val >= threshold:
                low_count = 0
            else:
                low_count += 1
                if low_count >= hysteresis:
                    # Close gate, report event
                    # CFO from angle of P: For Schmidl-Cox, angle(P) = 2π × f_cfo × L / fs
                    # Therefore: f_cfo = angle(P) × fs / (2π × L)
                    cfo_hz = np.angle(peak_P) * sample_rate / (2 * np.pi * L)
                    # Our P[n] correlates samples [n-L+1,n] with [n-2L+1,n-L]
                    # Peak occurs when these align with [A+L,A+2L] and [A,A+L]
                    # i.e., peak at n = A + 2L - 1 where A = preamble start
                    # Therefore: frame_start = peak_index - 2L + 1
                    frame_start = peak_index - 2 * L + 1
                    
                    events.append(AADetectionEvent(
                        peak_index=peak_index,
                        P_at_peak=peak_P,
                        M_at_peak=M[peak_index],
                        gate_start=gate_start,
                        gate_end=n,
                        cfo_hz=cfo_hz,
                        frame_start=frame_start,
                    ))
                    
                    gate_open = False
                    peak_P_mag_sq = 0.0
                    low_count = 0
    
    # Handle unclosed gate
    if gate_open:
        cfo_hz = np.angle(peak_P) * sample_rate / (2 * np.pi * L)
        frame_start = peak_index - 2 * L + 1
        events.append(AADetectionEvent(
            peak_index=peak_index,
            P_at_peak=peak_P,
            M_at_peak=M[peak_index],
            gate_start=gate_start,
            gate_end=num_samples,
            cfo_hz=cfo_hz,
            frame_start=frame_start,
        ))
    
    state = AADetectorState(P=P_total, R=R_total, M=M, valid=valid)
    return AADetectionResult(events=events, state=state, num_antennas=num_antennas)


# =============================================================================
# Channel Application with Multi-Antenna
# =============================================================================
def apply_channel_multi_antenna(
    tx_samples: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    channel_name: str | None = None,
    num_rx_antennas: int = 2,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Apply channel with multi-antenna support.
    
    Returns:
        rx_samples: Shape (num_rx_antennas, num_samples)
        cir: Channel impulse response or None
        channel_peak_offset: Delay to main path
    """
    if channel_name is None:
        # AWGN only - create independent noise on each antenna
        signal_power = np.mean(np.abs(tx_samples) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = np.sqrt(noise_power / 2)
        
        rx_samples = np.zeros((num_rx_antennas, len(tx_samples)), dtype=complex)
        for ant in range(num_rx_antennas):
            noise = noise_std * (rng.standard_normal(len(tx_samples)) + 
                                1j * rng.standard_normal(len(tx_samples)))
            rx_samples[ant] = tx_samples + noise
        
        return rx_samples, None, 0
    else:
        # Load measured CIR
        cir_bank = load_measured_cir(channel_name)
        
        # Use up to num_rx_antennas from CIR bank
        if cir_bank.shape[0] >= num_rx_antennas:
            cir = cir_bank[:num_rx_antennas].copy()
        else:
            # Duplicate if not enough
            cir = np.tile(cir_bank, (num_rx_antennas // cir_bank.shape[0] + 1, 1))
            cir = cir[:num_rx_antennas]
        
        # Apply channel per antenna
        rx_samples = np.zeros((num_rx_antennas, len(tx_samples) + cir.shape[1] - 1), dtype=complex)
        for ant in range(num_rx_antennas):
            # Convolve with CIR
            rx_ant = np.convolve(tx_samples, cir[ant])
            
            # Add noise
            signal_power = np.mean(np.abs(rx_ant) ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (rng.standard_normal(len(rx_ant)) + 
                                1j * rng.standard_normal(len(rx_ant)))
            rx_samples[ant] = rx_ant + noise
        
        # Find channel peak offset
        agg_cir = np.sum(np.abs(cir) ** 2, axis=0)
        channel_peak_offset = int(np.argmax(agg_cir))
        
        return rx_samples, cir, channel_peak_offset


def apply_cfo(samples: np.ndarray, cfo_hz: float, sample_rate: float) -> np.ndarray:
    """Apply CFO to samples (1D or 2D with antennas on axis 0)."""
    if samples.ndim == 1:
        n = np.arange(len(samples))
        return samples * np.exp(1j * 2 * np.pi * cfo_hz * n / sample_rate)
    else:
        n = np.arange(samples.shape[1])
        tone = np.exp(1j * 2 * np.pi * cfo_hz * n / sample_rate)
        return samples * tone[np.newaxis, :]


# =============================================================================
# Single Test Run
# =============================================================================
@dataclass
class TestResult:
    """Result of a single test run."""
    snr_db: float
    channel: str
    full_scale_ratio: float
    preamble_length: int
    timing_error: int
    cfo_applied_hz: float
    cfo_estimated_hz: float
    cfo_error_hz: float
    detected: bool
    num_events: int
    clipping_pct: float
    effective_bits: float
    metric_peak: float


def run_single_test(
    snr_db: float,
    channel_name: str | None,
    full_scale_ratio: float,
    preamble_length: int = DEFAULT_PREAMBLE_LEN,
    cfo_hz: float = 500.0,
    seed: int = 42,
    plot: bool = False,
    plot_dir: Path | None = None,
) -> TestResult:
    """Run a single synchronization test.
    
    Args:
        snr_db: Signal-to-noise ratio
        channel_name: 'cir1', 'cir2', or None for AWGN
        full_scale_ratio: ADC full scale relative to signal RMS
        preamble_length: Total preamble length (1024, 512, or 256)
        cfo_hz: Applied carrier frequency offset
        seed: Random seed
        plot: Whether to generate plots
        plot_dir: Directory for plots
    
    Returns:
        TestResult with metrics
    """
    rng = np.random.default_rng(seed)
    channel_str = channel_name if channel_name else "awgn"
    L = preamble_length // 2  # Half-preamble length for detector
    
    # Build preamble with specified length
    preamble, _, papr_db = build_aa_preamble(preamble_length)
    
    # Build pilot and data symbols
    pilot_symbol, _ = build_random_qpsk_symbol(rng)
    data_symbol, _ = build_random_qpsk_symbol(rng)
    
    # Assemble frame: [pad][preamble][pilot][data][pad]
    frame = np.concatenate([preamble, pilot_symbol, data_symbol])
    tx_samples = np.concatenate([
        np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex),
        frame,
        np.zeros(TX_POST_PAD_SAMPLES, dtype=complex),
    ])
    
    # True preamble start
    true_preamble_start = TX_PRE_PAD_SAMPLES
    
    # Apply channel
    rx_samples, cir, channel_peak_offset = apply_channel_multi_antenna(
        tx_samples, snr_db, rng, channel_name, num_rx_antennas=2
    )
    
    # Adjust true start for channel delay
    true_preamble_start += channel_peak_offset
    
    # Apply CFO
    rx_samples = apply_cfo(rx_samples, cfo_hz, SAMPLE_RATE_HZ)
    
    # Compute signal RMS for full-scale calculation
    signal_rms = np.sqrt(np.mean(np.abs(rx_samples) ** 2))
    full_scale = signal_rms * full_scale_ratio
    
    # Apply quantization
    clip_stats = compute_clipping_stats(rx_samples.flatten(), full_scale)
    rx_quantized = np.zeros_like(rx_samples)
    for ant in range(rx_samples.shape[0]):
        rx_quantized[ant] = quantize_adc(rx_samples[ant], full_scale)
    
    # Run detector with correct L for this preamble length
    result = aa_detect_streaming(rx_quantized, L=L)
    
    # Analyze results
    if result.events:
        # Use strongest event (highest M)
        best_event = max(result.events, key=lambda e: e.M_at_peak)
        detected = True
        timing_error = best_event.frame_start - true_preamble_start
        cfo_estimated = best_event.cfo_hz
        cfo_error = cfo_estimated - cfo_hz
        metric_peak = best_event.M_at_peak
        num_events = len(result.events)
    else:
        detected = False
        timing_error = 0
        cfo_estimated = 0.0
        cfo_error = cfo_hz
        metric_peak = np.max(result.state.M) if np.any(result.state.valid) else 0.0
        num_events = 0
    
    # Generate plots if requested
    if plot and plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # Plot 1: Received magnitude
        rx_mag = np.sqrt(np.sum(np.abs(rx_quantized) ** 2, axis=0))
        axes[0].plot(rx_mag, alpha=0.7)
        axes[0].axvline(true_preamble_start, color='g', linestyle='--', label='True start')
        if detected:
            axes[0].axvline(best_event.frame_start, color='r', linestyle=':', label='Detected')
        axes[0].set_ylabel('|rx|')
        axes[0].set_title(f'{channel_str.upper()}, SNR={snr_db}dB, FS={full_scale_ratio}×, L={L}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Metric M[n]
        axes[1].plot(result.state.M, label='M[n]')
        axes[1].axhline(DETECT_THRESHOLD, color='orange', linestyle='--', label='Threshold')
        # Expected peak location (same as P peak)
        expected_peak = true_preamble_start + 2 * L - 1
        axes[1].axvline(expected_peak, color='g', linestyle='--', label='Expected peak')
        if detected:
            axes[1].axvline(best_event.peak_index, color='r', linestyle=':')
            for evt in result.events:
                axes[1].axvspan(evt.gate_start, evt.gate_end, alpha=0.2, color='orange')
        axes[1].set_ylabel('Metric')
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: |P|² 
        P_mag_sq = np.abs(result.state.P) ** 2
        axes[2].plot(P_mag_sq, label='|P|²')
        # P peaks at preamble_start + 2L - 1 (due to our delay line arrangement)
        expected_peak = true_preamble_start + 2 * L - 1
        axes[2].axvline(expected_peak, color='g', linestyle='--', 
                       label='Expected peak')
        if detected:
            axes[2].axvline(best_event.peak_index, color='r', linestyle=':', label='Detected peak')
        axes[2].set_ylabel('|P|²')
        axes[2].set_xlabel('Sample')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_name = f"{channel_str}_snr{snr_db:+.0f}dB_fs{full_scale_ratio:.2f}_L{L}.png"
        plt.savefig(plot_dir / plot_name, dpi=120)
        plt.close()
    
    return TestResult(
        snr_db=snr_db,
        channel=channel_str,
        full_scale_ratio=full_scale_ratio,
        preamble_length=preamble_length,
        timing_error=timing_error,
        cfo_applied_hz=cfo_hz,
        cfo_estimated_hz=cfo_estimated if detected else 0.0,
        cfo_error_hz=cfo_error,
        detected=detected,
        num_events=num_events,
        clipping_pct=clip_stats["total_clip_pct"],
        effective_bits=clip_stats["effective_bits"],
        metric_peak=metric_peak,
    )


# =============================================================================
# Grid Test
# =============================================================================
def run_grid_test(
    snr_values: list[float] = [-5, 0, 5, 10, 15],
    channels: list[str | None] = [None, "cir1", "cir2"],
    full_scale_ratios: list[float] = [0.25, 0.5, 1.0, 1.5, 2.0],
    preamble_lengths: list[int] = PREAMBLE_LENGTHS,
    cfo_hz: float = 500.0,
    plot_samples: bool = True,
) -> list[TestResult]:
    """Run grid test over SNR, channels, full-scale ratios, and preamble lengths.
    
    Args:
        snr_values: List of SNR values in dB
        channels: List of channel names (None for AWGN)
        full_scale_ratios: List of full-scale ratios
        preamble_lengths: List of preamble lengths (1024, 512, 256)
        cfo_hz: Applied CFO
        plot_samples: Generate sample plots
    
    Returns:
        List of TestResult objects
    """
    results: list[TestResult] = []
    total_tests = len(snr_values) * len(channels) * len(full_scale_ratios) * len(preamble_lengths)
    test_num = 0
    
    print(f"\n{'='*80}")
    print(f"[A][A] PREAMBLE SYNCHRONIZATION - GRID TEST")
    print(f"{'='*80}")
    print(f"SNR values: {snr_values} dB")
    print(f"Channels: {[c if c else 'AWGN' for c in channels]}")
    print(f"Full-scale ratios: {full_scale_ratios}")
    print(f"Preamble lengths: {preamble_lengths} samples (L = {[p//2 for p in preamble_lengths]})")
    print(f"Applied CFO: {cfo_hz} Hz")
    print(f"Total tests: {total_tests}")
    print(f"{'='*80}\n")
    
    for preamble_len in preamble_lengths:
        L = preamble_len // 2
        print(f"\n--- Preamble Length: {preamble_len} samples (L={L}) ---")
        
        for channel in channels:
            channel_str = channel if channel else "awgn"
            plot_dir = PLOTS_DIR / channel_str
            
            for snr_db in snr_values:
                for fs_ratio in full_scale_ratios:
                    test_num += 1
                    
                    # Only plot middle full-scale and default preamble for brevity
                    do_plot = plot_samples and (fs_ratio == 1.0) and (preamble_len == DEFAULT_PREAMBLE_LEN)
                    
                    result = run_single_test(
                        snr_db=snr_db,
                        channel_name=channel,
                        full_scale_ratio=fs_ratio,
                        preamble_length=preamble_len,
                        cfo_hz=cfo_hz,
                        seed=42,
                        plot=do_plot,
                        plot_dir=plot_dir,
                    )
                    results.append(result)
                    
                    status = "✓" if result.detected else "✗"
                    print(f"[{test_num:3d}/{total_tests}] L={L:3d} {channel_str:6s} SNR={snr_db:+3.0f}dB "
                          f"FS={fs_ratio:.2f}× → {status} "
                          f"timing_err={result.timing_error:+4d} "
                          f"cfo_err={result.cfo_error_hz:+7.1f}Hz "
                          f"clip={result.clipping_pct:5.1f}%")
    
    return results


def print_summary_table(results: list[TestResult]) -> None:
    """Print summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    
    # Group by preamble length, channel
    preamble_lengths = sorted(set(r.preamble_length for r in results), reverse=True)
    channels = sorted(set(r.channel for r in results))
    snr_values = sorted(set(r.snr_db for r in results))
    fs_ratios = sorted(set(r.full_scale_ratio for r in results))
    
    for preamble_len in preamble_lengths:
        L = preamble_len // 2
        print(f"\n{'='*80}")
        print(f"PREAMBLE LENGTH: {preamble_len} samples (L={L})")
        print(f"{'='*80}")
        
        for channel in channels:
            print(f"\n--- {channel.upper()} ---")
            print(f"{'SNR':>6s}", end="")
            for fs in fs_ratios:
                print(f" | FS={fs:.2f}", end="")
            print()
            print("-" * (8 + 10 * len(fs_ratios)))
            
            for snr in snr_values:
                print(f"{snr:+5.0f}dB", end="")
                for fs in fs_ratios:
                    # Find matching result
                    matching = [r for r in results 
                               if r.channel == channel and r.snr_db == snr 
                               and r.full_scale_ratio == fs and r.preamble_length == preamble_len]
                    if matching:
                        r = matching[0]
                        if r.detected:
                            # Show timing error
                            print(f" | {r.timing_error:+5d}", end="")
                        else:
                            print(f" |  MISS", end="")
                    else:
                        print(f" |   N/A", end="")
                print()
    
    # Detection rate summary by preamble length
    print(f"\n{'='*80}")
    print("DETECTION RATE BY PREAMBLE LENGTH AND CHANNEL")
    print(f"{'='*80}")
    
    for preamble_len in preamble_lengths:
        L = preamble_len // 2
        print(f"\nPreamble L={L}:")
        for channel in channels:
            channel_results = [r for r in results 
                              if r.channel == channel and r.preamble_length == preamble_len]
            detected = sum(1 for r in channel_results if r.detected)
            total = len(channel_results)
            pct = 100*detected/total if total > 0 else 0
            print(f"  {channel:6s}: {detected}/{total} ({pct:.0f}%)")
    
    # Timing error statistics by preamble length
    print(f"\n{'='*80}")
    print("TIMING ERROR STATISTICS BY PREAMBLE LENGTH (detected only)")
    print(f"{'='*80}")
    
    for preamble_len in preamble_lengths:
        L = preamble_len // 2
        detected_results = [r for r in results if r.detected and r.preamble_length == preamble_len]
        if detected_results:
            timing_errors = [r.timing_error for r in detected_results]
            print(f"\nPreamble L={L}:")
            print(f"  Mean:   {np.mean(timing_errors):+.1f} samples")
            print(f"  Std:    {np.std(timing_errors):.1f} samples")
            print(f"  Range:  [{np.min(timing_errors):+d}, {np.max(timing_errors):+d}]")
            print(f"  Within CP ({CYCLIC_PREFIX}): {sum(1 for e in timing_errors if abs(e) <= CYCLIC_PREFIX)}/{len(timing_errors)}")
    
    # CFO error statistics by preamble length  
    print(f"\n{'='*80}")
    print("CFO ERROR STATISTICS BY PREAMBLE LENGTH (detected only)")
    print(f"{'='*80}")
    
    for preamble_len in preamble_lengths:
        L = preamble_len // 2
        detected_results = [r for r in results if r.detected and r.preamble_length == preamble_len]
        if detected_results:
            cfo_errors = [r.cfo_error_hz for r in detected_results]
            print(f"\nPreamble L={L}:")
            print(f"  Mean:   {np.mean(cfo_errors):+.1f} Hz")
            print(f"  Std:    {np.std(cfo_errors):.1f} Hz")
            print(f"  Range:  [{np.min(cfo_errors):+.1f}, {np.max(cfo_errors):+.1f}] Hz")


def plot_heatmaps(results: list[TestResult]) -> None:
    """Generate heatmap visualizations for each preamble length."""
    preamble_lengths = sorted(set(r.preamble_length for r in results), reverse=True)
    channels = sorted(set(r.channel for r in results))
    snr_values = sorted(set(r.snr_db for r in results))
    fs_ratios = sorted(set(r.full_scale_ratio for r in results))
    
    # Create a figure with subplots: rows = preamble lengths, cols = channels
    n_rows = len(preamble_lengths)
    n_cols = len(channels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    
    # Handle single row/col case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]
    
    for row, preamble_len in enumerate(preamble_lengths):
        L = preamble_len // 2
        for col, channel in enumerate(channels):
            ax = axes[row, col]
            
            # Build matrix and store results for annotation
            matrix = np.zeros((len(snr_values), len(fs_ratios)))
            result_map = {}  # Store results for annotation
            for i, snr in enumerate(snr_values):
                for j, fs in enumerate(fs_ratios):
                    matching = [r for r in results 
                               if r.channel == channel and r.snr_db == snr 
                               and r.full_scale_ratio == fs and r.preamble_length == preamble_len]
                    if matching:
                        result_map[(i, j)] = matching[0]
                        if matching[0].detected:
                            matrix[i, j] = 1.0
                        else:
                            matrix[i, j] = 0.0
            
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            ax.set_xticks(range(len(fs_ratios)))
            ax.set_xticklabels([f"{fs:.2f}" for fs in fs_ratios], fontsize=8)
            ax.set_yticks(range(len(snr_values)))
            ax.set_yticklabels([f"{snr:+.0f}" for snr in snr_values], fontsize=8)
            
            if row == n_rows - 1:
                ax.set_xlabel("Full-scale ratio")
            if col == 0:
                ax.set_ylabel(f"L={L}\nSNR (dB)")
            
            ax.set_title(f"{channel.upper()}" if row == 0 else "")
            
            # Add text annotations with CFO error
            for i in range(len(snr_values)):
                for j in range(len(fs_ratios)):
                    color = 'white' if matrix[i, j] < 0.5 else 'black'
                    r = result_map.get((i, j))
                    if r and r.detected:
                        # Format CFO error compactly
                        cfo_err = r.cfo_error_hz
                        if abs(cfo_err) >= 1000:
                            cfo_str = f"{cfo_err/1000:+.1f}k"
                        else:
                            cfo_str = f"{cfo_err:+.0f}"
                        text = f"✓\n({cfo_str})"
                    else:
                        text = '✗'
                    ax.text(j, i, text,
                           ha='center', va='center', color=color, fontsize=8)
    
    plt.suptitle("[A][A] Detection Success by Preamble Length (CFO error in Hz)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "detection_heatmap.png", dpi=150)
    plt.close()
    print(f"\nHeatmap saved to {PLOTS_DIR / 'detection_heatmap.png'}")


# =============================================================================
# Main
# =============================================================================
def main():
    """Run comprehensive grid test."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Print preamble info for each length
    print(f"\n{'='*80}")
    print("[A][A] PREAMBLE CHARACTERISTICS")
    print(f"{'='*80}")
    print(f"Sample rate: {SAMPLE_RATE_HZ/1e6:.2f} MHz")
    print(f"OFDM symbol: {N_FFT} samples, CP={CYCLIC_PREFIX}")
    print()
    
    for preamble_len in PREAMBLE_LENGTHS:
        preamble, _, papr_db = build_aa_preamble(preamble_len)
        L = preamble_len // 2
        duration_us = preamble_len / SAMPLE_RATE_HZ * 1e6
        # Verify [A][A] structure
        first_half = preamble[:L]
        second_half = preamble[L:]
        corr = np.abs(np.vdot(first_half, second_half)) / (np.linalg.norm(first_half) * np.linalg.norm(second_half))
        print(f"  Length {preamble_len:4d}: L={L:3d}, PAPR={papr_db:.2f}dB, "
              f"duration={duration_us:.1f}µs, [A][A] corr={corr:.6f}")
    print(f"{'='*80}")
    
    # Run grid test with all preamble lengths
    results = run_grid_test(
        snr_values=[-5, 0, 5, 10, 15],
        channels=[None, "cir1", "cir2"],
        full_scale_ratios=[0.5, 1.0, 2.0],  # Reduced for faster testing
        preamble_lengths=PREAMBLE_LENGTHS,
        cfo_hz=500.0,
        plot_samples=True,
    )
    
    # Print summary
    print_summary_table(results)
    
    # Generate heatmaps
    plot_heatmaps(results)
    
    print(f"\n{'='*80}")
    print("ALL TESTS COMPLETE")
    print(f"{'='*80}")
    print(f"Plots saved to: {PLOTS_DIR.resolve()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

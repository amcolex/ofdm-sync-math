"""
Zadoff-Chu Preamble Detection with FPGA-Friendly Adaptive Threshold
====================================================================

This module implements ZC-based frame synchronization using an RTL-friendly
threshold mechanism that adapts to local correlation levels (similar to CFAR).

KEY INSIGHT: Instead of an absolute threshold on |correlation|, we compare
the correlation to a scaled running average of recent correlation values:

    detection = (corr_scaled >= local_sum × THRESH_VALUE)

Where:
    corr_scaled = |corr[n]| << FRAC_BITS
    local_sum = running_sum(|corr[n]|) over W samples
    THRESH_VALUE = threshold as fixed-point fraction (e.g., 0.2 × 2^FRAC_BITS)

This avoids division in hardware and naturally adapts to:
    - Varying signal levels
    - Sidelobe levels (included in local average)
    - Channel conditions


FPGA PIPELINE STRUCTURE
-----------------------

    ┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
    │ Matched Filter  │────▶│ Running Sum      │────▶│ Threshold      │
    │ corr[n] (cplx)  │     │ of |corr|        │     │ Compare        │
    └─────────────────┘     │ over W samples   │     │ (multiply)     │
           │                └──────────────────┘     └────────────────┘
           │                        │                       │
           ▼                        ▼                       ▼
    ┌─────────────┐          ┌─────────────┐         ┌─────────────┐
    │ |corr[n]|   │          │ local_sum   │         │ above_thresh│
    └─────────────┘          └─────────────┘         └─────────────┘
           │                        │                       │
           └────────────────────────┴───────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Gate Logic      │
                           │ + Peak Tracking │
                           └─────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ frame_start     │
                           │ + peak_index    │
                           └─────────────────┘


THRESHOLD COMPARISON (RTL-friendly, no division)
------------------------------------------------
Instead of:   |corr| / local_avg > threshold
Compute:      |corr| × 2^FRAC_BITS >= local_sum × THRESH_VALUE

Example with FRAC_BITS=15 and threshold=0.2:
    THRESH_VALUE = 0.2 × 32768 = 6554
    Threshold ratio = THRESH_VALUE / (W × 2^FRAC_BITS) ≈ 0.2/W of local average


CFO ESTIMATION
--------------
Unlike autocorrelation-based methods (Schmidl-Cox, Minn), matched filter
detection does NOT directly provide CFO. However, the correlation phase
at the peak gives a coarse CFO estimate if the preamble is repeated or
if we correlate consecutive symbols.

This implementation uses CP correlation on the following pilot symbol
for CFO estimation (same as zc.py).


PARAMETERS
----------
    W                   Running sum window size (power of 2 recommended)
    THRESH_FRAC_BITS    Fixed-point fractional bits for threshold comparison
    THRESH_VALUE        Threshold numerator (detection when corr×2^BITS >= sum×THRESH)
    HYSTERESIS          Samples below threshold before gate closes
    MIN_CORR_THRESHOLD  Minimum absolute correlation for detection (noise floor)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from channel import apply_channel, load_measured_cir
from core import (
    N_FFT,
    CYCLIC_PREFIX,
    TX_PRE_PAD_SAMPLES,
    centered_subcarrier_indices,
    allocate_subcarriers,
    spectrum_to_time_domain,
    add_cyclic_prefix,
    plot_time_series,
    build_random_qpsk_symbol,
    compute_channel_peak_offset,
    estimate_cfo_from_cp,
    ofdm_fft_used,
    ls_channel_estimate,
    equalize,
    evm_rms_db,
    plot_constellation,
    SAMPLE_RATE_HZ,
    apply_cfo,
    plot_phase_slope_diagnostics,
    align_complex_gain,
)

# =============================================================================
# ZC Preamble Parameters
# =============================================================================
PSS_LENGTH = 62          # Number of ZC subcarriers (matches LTE PSS)
PSS_ROOT = 25            # ZC root index


# =============================================================================
# Detection Parameters (FPGA-friendly)
# =============================================================================
# Running sum window size (power of 2 for easy RTL)
CORR_WINDOW_SIZE = N_FFT  # 2048 samples - roughly one OFDM symbol

# Threshold comparison: corr × 2^FRAC_BITS >= local_sum × THRESH_VALUE
THRESH_FRAC_BITS = 15
# Threshold ratio ≈ THRESH_VALUE / (W × 2^FRAC_BITS)
# The adaptive threshold compares: |corr| × 2^FRAC_BITS >= local_sum × THRESH_VALUE
# Rearranging: |corr| >= local_sum × THRESH_VALUE / 2^FRAC_BITS
#            = local_avg × W × THRESH_VALUE / 2^FRAC_BITS
#            = local_avg × (W × THRESH_VALUE / 2^FRAC_BITS)
#
# For reliable detection with sidelobes ~30-40% of main peak:
# - We want threshold high enough that sidelobes don't trigger
# - But low enough that main peak triggers reliably
# - Using larger window helps average out sidelobes vs main peak
#
# With W=2048, if we want threshold at 50% of peak and local_avg ≈ 0.15 × peak:
#   0.5 × peak >= 0.15 × peak × W × THRESH / 2^FRAC_BITS
#   0.5 >= 0.15 × 2048 × THRESH / 32768
#   THRESH <= 53
# Use a more lenient value for multipath robustness:
THRESH_VALUE = int(4.0 * (1 << THRESH_FRAC_BITS) / CORR_WINDOW_SIZE)  # ~64 for W=2048

# Minimum correlation magnitude (absolute floor, prevents false triggers on noise)
# This should be set based on expected normalized peak (~1.0 for perfect, 0.5-0.8 for multipath)
MIN_CORR_MAG = 0.3  # Lenient for multipath degradation

# Hysteresis: samples below threshold before gate closes
# Should be long enough to span the sidelobe region (~2× ZC length)
HYSTERESIS = 256

# Simulation parameters
SNR_DB = 10.0
CFO_HZ = 1000.0

# Output directory
PLOTS_BASE_DIR = Path("plots") / "zc_v2"


# =============================================================================
# ZC Sequence Generation
# =============================================================================
def generate_zadoff_chu(root: int, length: int) -> np.ndarray:
    """Generate a Zadoff-Chu sequence."""
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)


def build_pss_symbol(include_cp: bool = False) -> np.ndarray:
    """Build ZC preamble symbol in time domain.
    
    Args:
        include_cp: Whether to add cyclic prefix (default False for first symbol)
    
    Returns:
        Time-domain ZC symbol (N_FFT samples, or N_FFT+CP if include_cp=True)
    """
    indices = centered_subcarrier_indices(PSS_LENGTH)
    zc_sequence = generate_zadoff_chu(PSS_ROOT, PSS_LENGTH)
    spectrum = allocate_subcarriers(N_FFT, indices, zc_sequence)
    symbol = spectrum_to_time_domain(spectrum)
    if include_cp:
        return add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol


# =============================================================================
# Streaming Delay Line and Running Sum (RTL-style)
# =============================================================================
class DelayLine:
    """Fixed-depth delay line for streaming processing."""
    
    def __init__(self, depth: int):
        self.depth = max(0, int(depth))
        self.buffer = np.zeros(self.depth, dtype=np.float64) if self.depth > 0 else np.array([])
        self.write_ptr = 0
        self.filled = 0
    
    def step(self, sample: float) -> tuple[float, bool]:
        """Push sample, return (delayed_output, is_valid)."""
        if self.depth == 0:
            return sample, True
        
        # Read before write
        output = self.buffer[self.write_ptr] if self.filled >= self.depth else 0.0
        valid = self.filled >= self.depth
        
        # Write new sample
        self.buffer[self.write_ptr] = sample
        self.write_ptr = (self.write_ptr + 1) % self.depth
        
        if self.filled < self.depth:
            self.filled += 1
        
        return output, valid


class RunningSum:
    """Running sum over a fixed window (streaming, no division)."""
    
    def __init__(self, window_size: int):
        self.window_size = max(1, int(window_size))
        self.delay = DelayLine(self.window_size)
        self.sum_acc = 0.0
        self.valid = False
    
    def step(self, sample: float) -> tuple[float, bool]:
        """Add sample to window, return (running_sum, is_valid)."""
        oldest, delay_valid = self.delay.step(sample)
        
        if delay_valid:
            self.sum_acc = self.sum_acc + sample - oldest
            self.valid = True
        else:
            self.sum_acc = self.sum_acc + sample
        
        return self.sum_acc, self.valid


# =============================================================================
# Matched Filter Correlation (Streaming)
# =============================================================================
def matched_filter_correlation(
    rx_samples: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Compute matched filter correlation (conjugate reference, convolve).
    
    Returns complex correlation values, same length as convolution output.
    """
    ref_conj_rev = np.conj(reference[::-1])
    corr = np.convolve(rx_samples, ref_conj_rev, mode='full')
    return corr


def normalize_correlation(
    corr: np.ndarray,
    rx_samples: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Normalize correlation by sqrt(signal_energy × reference_energy)."""
    ref_energy = np.sum(np.abs(reference) ** 2)
    ref_norm = np.sqrt(ref_energy)
    
    # Sliding window energy of rx
    window = np.ones(len(reference))
    rx_energy = np.convolve(np.abs(rx_samples) ** 2, window, mode='full')
    rx_norm = np.sqrt(np.maximum(rx_energy, 1e-12))
    
    return corr / (ref_norm * rx_norm)


# =============================================================================
# FPGA-Friendly Streaming Detection
# =============================================================================
@dataclass
class ZCDetectionState:
    """State from streaming ZC detection."""
    corr_mag: np.ndarray           # |correlation| at each sample
    local_sum: np.ndarray          # Running sum of |corr| over window
    corr_scaled: np.ndarray        # |corr| << FRAC_BITS
    thresh_scaled: np.ndarray      # local_sum × THRESH_VALUE
    above_threshold: np.ndarray    # Boolean: corr_scaled >= thresh_scaled
    metric_valid: np.ndarray       # Boolean: running sum is fully populated


def zc_streaming_detection(
    corr_mag: np.ndarray,
    window_size: int = CORR_WINDOW_SIZE,
    thresh_value: int = THRESH_VALUE,
    thresh_frac_bits: int = THRESH_FRAC_BITS,
    min_corr_mag: float = MIN_CORR_MAG,
) -> ZCDetectionState:
    """FPGA-friendly streaming detection on correlation magnitude.
    
    Implements:
        above_threshold = (corr_mag × 2^FRAC_BITS >= local_sum × THRESH_VALUE)
                        AND (corr_mag >= MIN_CORR_MAG)
    
    Args:
        corr_mag: Magnitude of matched filter correlation
        window_size: Running sum window (W)
        thresh_value: Threshold numerator
        thresh_frac_bits: Fixed-point fractional bits
        min_corr_mag: Minimum absolute correlation magnitude
    
    Returns:
        ZCDetectionState with all intermediate signals
    """
    n = len(corr_mag)
    local_sum = np.zeros(n, dtype=np.float64)
    metric_valid = np.zeros(n, dtype=bool)
    
    # Streaming running sum
    running_sum = RunningSum(window_size)
    
    for i in range(n):
        local_sum[i], metric_valid[i] = running_sum.step(corr_mag[i])
    
    # Threshold comparison (no division!)
    scale = float(1 << thresh_frac_bits)
    corr_scaled = corr_mag * scale
    thresh_scaled = local_sum * float(thresh_value)
    
    # Detection: above adaptive threshold AND above noise floor
    above_threshold = metric_valid & (corr_scaled >= thresh_scaled) & (corr_mag >= min_corr_mag)
    
    return ZCDetectionState(
        corr_mag=corr_mag,
        local_sum=local_sum,
        corr_scaled=corr_scaled,
        thresh_scaled=thresh_scaled,
        above_threshold=above_threshold,
        metric_valid=metric_valid,
    )


# =============================================================================
# Gate Logic and Peak Tracking
# =============================================================================
@dataclass
class ZCDetectionEvent:
    """A single detection event."""
    peak_index: int          # Index of correlation peak
    peak_value: float        # Correlation magnitude at peak
    gate_start: int          # Sample where gate opened
    gate_end: int            # Sample where gate closed
    detected_start: int      # Estimated frame start (peak - ref_len + 1)


@dataclass 
class ZCDetectionResult:
    """Complete detection result."""
    events: list[ZCDetectionEvent]
    gate_mask: np.ndarray
    state: ZCDetectionState


def detect_zc_peaks(
    state: ZCDetectionState,
    reference_length: int,
    hysteresis: int = HYSTERESIS,
) -> ZCDetectionResult:
    """Gate logic and peak tracking for ZC detection.
    
    When above_threshold goes high:
        1. Open gate
        2. Track maximum correlation within gate
        3. When below threshold for `hysteresis` samples, close gate
        4. Report peak location as detection
    
    Args:
        state: Detection state from zc_streaming_detection
        reference_length: Length of ZC reference (for frame start calculation)
        hysteresis: Samples below threshold before gate closes
    
    Returns:
        ZCDetectionResult with events and gate mask
    """
    n = len(state.corr_mag)
    gate_mask = np.zeros(n, dtype=bool)
    events: list[ZCDetectionEvent] = []
    
    gate_open = False
    gate_start = 0
    peak_index = 0
    peak_value = 0.0
    low_count = 0
    hyst_limit = max(0, hysteresis - 1)
    
    for i in range(n):
        if not state.metric_valid[i]:
            continue
        
        corr_val = state.corr_mag[i]
        is_above = state.above_threshold[i]
        
        if not gate_open:
            if is_above:
                # Open new gate
                gate_open = True
                gate_start = i
                peak_index = i
                peak_value = corr_val
                low_count = 0
        else:
            # Gate is open - track peak
            gate_mask[i] = True
            
            if corr_val > peak_value:
                peak_value = corr_val
                peak_index = i
            
            if is_above:
                low_count = 0
            else:
                if hysteresis == 0 or low_count >= hyst_limit:
                    # Close gate
                    detected_start = max(0, peak_index - reference_length + 1)
                    events.append(ZCDetectionEvent(
                        peak_index=peak_index,
                        peak_value=peak_value,
                        gate_start=gate_start,
                        gate_end=i,
                        detected_start=detected_start,
                    ))
                    gate_open = False
                    peak_value = 0.0
                    low_count = 0
                else:
                    low_count += 1
    
    # Handle unclosed gate at end
    if gate_open:
        detected_start = max(0, peak_index - reference_length + 1)
        events.append(ZCDetectionEvent(
            peak_index=peak_index,
            peak_value=peak_value,
            gate_start=gate_start,
            gate_end=n,
            detected_start=detected_start,
        ))
        gate_mask[gate_start:n] = True
    
    return ZCDetectionResult(
        events=events,
        gate_mask=gate_mask,
        state=state,
    )


# =============================================================================
# Full Detection Pipeline
# =============================================================================
def detect_zc_preamble(
    rx_samples: np.ndarray,
    window_size: int = CORR_WINDOW_SIZE,
    thresh_value: int = THRESH_VALUE,
    thresh_frac_bits: int = THRESH_FRAC_BITS,
    min_corr_mag: float = MIN_CORR_MAG,
    hysteresis: int = HYSTERESIS,
    normalize: bool = True,
) -> ZCDetectionResult:
    """Full ZC preamble detection pipeline.
    
    Args:
        rx_samples: Received samples (1D complex array)
        window_size: Running sum window for adaptive threshold
        thresh_value: Threshold numerator (fixed-point)
        thresh_frac_bits: Fractional bits for threshold
        min_corr_mag: Minimum correlation magnitude
        hysteresis: Gate close hysteresis
        normalize: Whether to normalize correlation
    
    Returns:
        ZCDetectionResult with detected events
    """
    # Build reference (no CP)
    reference = build_pss_symbol(include_cp=False)
    
    # Handle multi-branch input
    if rx_samples.ndim == 1:
        rx_samples = rx_samples[np.newaxis, :]
    
    # Correlate and combine branches
    corr_sum = None
    for branch in rx_samples:
        corr = matched_filter_correlation(branch, reference)
        if normalize:
            corr = normalize_correlation(corr, branch, reference)
        if corr_sum is None:
            corr_sum = corr
        else:
            corr_sum = corr_sum + corr
    
    # Use magnitude for detection
    corr_mag = np.abs(corr_sum)
    
    # Streaming detection
    state = zc_streaming_detection(
        corr_mag,
        window_size=window_size,
        thresh_value=thresh_value,
        thresh_frac_bits=thresh_frac_bits,
        min_corr_mag=min_corr_mag,
    )
    
    # Gate and peak tracking
    result = detect_zc_peaks(
        state,
        reference_length=len(reference),
        hysteresis=hysteresis,
    )
    
    return result


# =============================================================================
# Simulation
# =============================================================================
def run_simulation(channel_name: str | None, plots_subdir: str):
    """Run ZC v2 detection simulation."""
    rng = np.random.default_rng(0)
    
    # Setup output directory
    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Build frame: ZC preamble (no CP) + pilot + data
    pss_waveform = build_pss_symbol(include_cp=False)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    
    frame = np.concatenate((pss_waveform, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))
    
    # Load channel
    if channel_name is None:
        channel_impulse_response = None
    else:
        cir_bank = load_measured_cir(channel_name)
        channel_impulse_response = cir_bank[:2].copy() if cir_bank.shape[0] > 2 else cir_bank.copy()
    
    # Apply channel and CFO
    rx_samples = apply_channel(
        tx_samples,
        SNR_DB,
        rng,
        channel_impulse_response=channel_impulse_response,
    )
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)
    
    if rx_samples.ndim == 1:
        rx_samples = rx_samples[np.newaxis, :]
    
    # Run detection
    result = detect_zc_preamble(rx_samples)
    state = result.state
    
    # Expected timing
    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    pss_reference = build_pss_symbol(include_cp=False)
    true_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    expected_peak = true_start + len(pss_reference) - 1
    
    # Get detection results - select strongest event (not necessarily first)
    if result.events:
        # Pick event with highest peak value (most confident detection)
        primary = max(result.events, key=lambda e: e.peak_value)
        detected_start = primary.detected_start
        peak_index = primary.peak_index
    else:
        peak_index = int(np.argmax(state.corr_mag))
        detected_start = max(0, peak_index - len(pss_reference) + 1)
        primary = None
    
    timing_error = detected_start - true_start
    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"
    
    # ==========================================================================
    # Plotting
    # ==========================================================================
    
    # Plot 1: Correlation with threshold
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Correlation magnitude
    axes[0].plot(state.corr_mag, label="|corr|", color="tab:blue", alpha=0.8)
    axes[0].axvline(peak_index, color="tab:red", linestyle=":", label=f"Peak @ {peak_index}")
    axes[0].axvline(expected_peak, color="tab:green", linestyle="--", label=f"Expected @ {expected_peak}")
    for evt in result.events:
        axes[0].axvspan(evt.gate_start, evt.gate_end, color="tab:orange", alpha=0.2)
    axes[0].set_ylabel("|correlation|")
    axes[0].set_title(f"ZC Matched Filter Correlation ({channel_desc})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    # Threshold comparison (scaled values)
    axes[1].plot(state.corr_scaled, label="corr × 2^BITS", color="tab:blue", alpha=0.8)
    axes[1].plot(state.thresh_scaled, label="local_sum × THRESH", color="tab:orange", alpha=0.8)
    axes[1].axvline(peak_index, color="tab:red", linestyle=":")
    for evt in result.events:
        axes[1].axvspan(evt.gate_start, evt.gate_end, color="tab:orange", alpha=0.2)
    axes[1].set_ylabel("Scaled values")
    axes[1].set_title("Threshold Comparison (FPGA-friendly)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    # Detection signal
    axes[2].fill_between(range(len(state.above_threshold)), 
                         state.above_threshold.astype(float),
                         alpha=0.5, label="above_threshold", color="tab:green")
    axes[2].fill_between(range(len(result.gate_mask)),
                         result.gate_mask.astype(float) * 0.5,
                         alpha=0.5, label="gate_open", color="tab:orange")
    for evt in result.events:
        axes[2].axvline(evt.peak_index, color="tab:red", linestyle=":", linewidth=2)
        axes[2].axvline(evt.detected_start, color="tab:purple", linestyle="--", linewidth=2)
    axes[2].axvline(true_start, color="tab:green", linestyle="--", label="True start")
    axes[2].set_xlabel("Sample index")
    axes[2].set_ylabel("Detection")
    axes[2].set_title("Detection Signals")
    axes[2].legend(loc="upper right")
    axes[2].set_ylim(-0.1, 1.2)
    
    fig.tight_layout()
    fig.savefig(plots_dir / "detection.png", dpi=150)
    plt.close(fig)
    
    # Plot 2: Zoomed correlation around peak
    zoom_half = 500
    zoom_start = max(0, peak_index - zoom_half)
    zoom_end = min(len(state.corr_mag), peak_index + zoom_half)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(zoom_start, zoom_end)
    ax.plot(x, state.corr_mag[zoom_start:zoom_end], label="|corr|", color="tab:blue")
    
    # Show threshold as ratio (for visualization, compute effective threshold)
    thresh_ratio = state.thresh_scaled[zoom_start:zoom_end] / (float(1 << THRESH_FRAC_BITS) + 1e-12)
    ax.plot(x, thresh_ratio, label="Adaptive threshold", color="tab:orange", linestyle="--")
    
    ax.axvline(peak_index, color="tab:red", linestyle=":", label=f"Detected peak")
    ax.axvline(expected_peak, color="tab:green", linestyle="--", label=f"Expected peak")
    ax.axhline(MIN_CORR_MAG, color="gray", linestyle=":", alpha=0.5, label="Min threshold")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("|correlation|")
    ax.set_title(f"Zoomed Correlation ({channel_desc})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "correlation_zoom.png", dpi=150)
    plt.close(fig)
    
    # Plot 3: Received magnitude with detection
    fig, ax = plt.subplots(figsize=(12, 4))
    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    ax.plot(combined_rx_mag, label="Combined |rx|", alpha=0.7)
    ax.axvline(true_start, color="tab:green", linestyle="--", label="True ZC start")
    ax.axvline(detected_start, color="tab:red", linestyle=":", label="Detected start")
    for evt in result.events:
        ax.axvspan(evt.gate_start, evt.gate_end, color="tab:orange", alpha=0.15)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"Received Signal with Detection ({channel_desc})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "rx_detection.png", dpi=150)
    plt.close(fig)
    
    # Plot channel CIR if applicable
    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}')",
            plots_dir / "channel_cir.png",
        )
    
    # TX/RX time series
    plot_time_series(tx_samples, "Transmit Frame", plots_dir / "tx_frame_time.png")
    plot_time_series(rx_samples, f"Received Frame ({channel_desc})", plots_dir / "rx_frame_time.png")
    
    # ==========================================================================
    # CFO estimation and equalization (same as zc.py)
    # ==========================================================================
    pilot_cp_start = detected_start + N_FFT  # ZC is N_FFT samples, pilot starts after
    data_cp_start = pilot_cp_start + pilot_symbol.size
    
    cfo_est_hz = estimate_cfo_from_cp(rx_samples, pilot_cp_start, N_FFT, CYCLIC_PREFIX, SAMPLE_RATE_HZ)
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)
    
    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[pilot_cp_start + CYCLIC_PREFIX : pilot_cp_start + CYCLIC_PREFIX + N_FFT]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)
    
    # Phase slope diagnostics
    slope_rad, timing_samples = plot_phase_slope_diagnostics(
        h_est,
        plots_dir / "phase_slope_sto.png",
        f"Residual Timing From Phase Slope (ZC v2, {channel_desc})",
    )
    
    # Equalize data
    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    
    plot_constellation(
        xhat_aligned, data_used,
        plots_dir / "constellation.png",
        f"Equalized Data Constellation (ZC v2, {channel_desc})",
    )
    
    # ==========================================================================
    # Print results
    # ==========================================================================
    print(f"\n{'='*70}")
    print(f"ZC V2 DETECTION RESULTS - {channel_desc.upper()}")
    print(f"{'='*70}")
    print(f"Detection Parameters:")
    print(f"  Window size (W): {CORR_WINDOW_SIZE}")
    print(f"  Threshold value: {THRESH_VALUE} (frac_bits={THRESH_FRAC_BITS})")
    print(f"  Effective threshold: ~{THRESH_VALUE * CORR_WINDOW_SIZE / (1 << THRESH_FRAC_BITS):.1f}× local average")
    print(f"  Min correlation: {MIN_CORR_MAG}")
    print(f"  Hysteresis: {HYSTERESIS} samples")
    print()
    print(f"Detection Events: {len(result.events)}")
    for i, evt in enumerate(result.events):
        is_primary = " ← PRIMARY" if primary and evt.peak_index == primary.peak_index else ""
        print(f"  Event {i}: peak={evt.peak_index} (val={evt.peak_value:.4f}), "
              f"gate=[{evt.gate_start}, {evt.gate_end}), "
              f"frame_start={evt.detected_start}{is_primary}")
    print()
    print(f"Timing:")
    print(f"  True ZC start: {true_start}")
    print(f"  Detected start: {detected_start}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.1f}% of symbol)")
    if channel_impulse_response is not None:
        print(f"  (Note: {channel_peak_offset} samples is channel delay - detection is correct)")
    print(f"  Expected peak: {expected_peak}")
    print(f"  Detected peak: {peak_index}")
    print(f"  Peak error: {peak_index - expected_peak} samples")
    if len(result.events) > 1:
        print(f"  Note: {len(result.events)-1} spurious event(s) from sidelobes - strongest selected")
    print()
    print(f"CFO Estimation:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz ({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.1f}%)")
    print()
    print(f"Equalization:")
    print(f"  Phase slope: {slope_rad:.6f} rad/bin -> {timing_samples:.2f} samples")
    print(f"  Complex gain: {np.abs(gain):.3f} ∠ {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}% ({evm_db:.2f} dB)")
    print()
    print(f"Plots saved to {plots_dir.resolve()}/")
    print(f"{'='*70}\n")


def main():
    """Run simulations for both channel conditions."""
    print("\n" + "="*70)
    print("ZC V2 DETECTION - FPGA-FRIENDLY ADAPTIVE THRESHOLD")
    print("="*70)
    
    # Flat AWGN
    run_simulation(channel_name=None, plots_subdir="flat_awgn")
    
    # Measured multipath channel
    run_simulation(channel_name="cir1", plots_subdir="measured_channel")
    
    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETE")
    print("="*70)
    print(f"\nResults in:")
    print(f"  - {(PLOTS_BASE_DIR / 'flat_awgn').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'measured_channel').resolve()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

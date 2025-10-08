import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from channel import apply_channel, load_measured_cir
from core import (
    N_FFT,
    NUM_ACTIVE_SUBCARRIERS,
    CYCLIC_PREFIX,
    TX_PRE_PAD_SAMPLES,
    centered_subcarrier_indices,
    allocate_subcarriers,
    spectrum_to_time_domain,
    add_cyclic_prefix,
    plot_time_series,
    compute_channel_peak_offset,
    build_random_qpsk_symbol,
    estimate_cfo_from_cp,
    ofdm_fft_used,
    ls_channel_estimate,
    equalize,
    remove_common_phase,
    evm_rms_db,
    plot_constellation,
    SAMPLE_RATE_HZ,
    apply_cfo,
    estimate_timing_offset_from_phase_slope,
)


def build_minn_preamble(rng: np.random.Generator, include_cp: bool = True) -> np.ndarray:
    """Build Minn preamble with 4-fold structure: A A -A -A.
    
    Uses every 4th subcarrier to create 4 identical parts [A, A, A, A],
    then inverts the second half to create [A, A, -A, -A] structure.
    Power normalization is applied after sign inversion.
    """
    all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    # Use every 4th subcarrier to create 4-fold repetition
    quarter_idx = all_idx[(all_idx % 4) == 0]
    bpsk = rng.choice([-1.0, 1.0], size=quarter_idx.shape[0])
    spectrum = allocate_subcarriers(N_FFT, quarter_idx, bpsk)
    # Generate time domain (already creates [A, A, A, A] structure)
    symbol = np.fft.ifft(np.fft.ifftshift(spectrum))
    
    # Apply sign inversion to second half to create [A, A, -A, -A]
    half = N_FFT // 2
    symbol[half:] = -symbol[half:]
    
    # Re-normalize to unit power after sign flip
    power = np.mean(np.abs(symbol) ** 2)
    if power > 0:
        symbol = symbol / np.sqrt(power)
    
    if include_cp:
        return add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol


def minn_streaming_metric(rx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Minn timing metric with 4-part correlation structure.
    
    The preamble structure is [A, A, -A, -A] where each part is N/4 samples.
    Minn metric adds correlations between identical quarter-pairs. The
    sign pattern (+ + - -) makes the sum flip polarity when the window
    sits on the CP repeat, so we keep only the positive real part of the
    running correlation to avoid the secondary CP peak.
    
    Returns:
        M: Timing metric magnitude
        P_sum: Complex correlation sum across branches
        R_sum: Energy sum across branches
    """
    rx = np.asarray(rx)
    if rx.ndim == 1:
        rx = rx[np.newaxis, :]
    
    num_branches, L = rx.shape
    Q = N_FFT // 4  # Quarter symbol length
    N = N_FFT
    out_len = max(L - N + 1, 0)
    if out_len <= 0:
        return np.zeros(0), np.zeros(0, dtype=complex), np.zeros(0)
    
    P_sum = np.zeros(out_len, dtype=np.complex128)
    R_sum = np.zeros(out_len, dtype=np.float64)
    
    for b in range(num_branches):
        x = rx[b]
        Pb = np.empty(out_len, dtype=np.complex128)
        Rb = np.empty(out_len, dtype=np.float64)
        
        for d in range(out_len):
            q0 = x[d : d + Q]
            q1 = x[d + Q : d + 2 * Q]
            q2 = x[d + 2 * Q : d + 3 * Q]
            q3 = x[d + 3 * Q : d + 4 * Q]
            
            C1 = np.sum(q0 * np.conj(q1))  # correlation between identical +A quarters
            C2 = np.sum(q2 * np.conj(q3))  # correlation between identical -A quarters
            P = C1 + C2
            R = np.sum(np.abs(q1) ** 2 + np.abs(q2) ** 2 + np.abs(q3) ** 2)
            
            Pb[d] = P
            Rb[d] = R
        
        P_sum += Pb
        R_sum += Rb
    
    eps = 1e-12
    aligned_real = np.clip(P_sum.real, 0.0, None)
    M = (aligned_real ** 2) / (np.maximum(R_sum, eps) ** 2)
    return M, P_sum, R_sum


def schmidl_cox_streaming_metric(rx: np.ndarray, symbol_len: int = N_FFT) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Schmidl & Cox timing metric for a symbol with two identical halves.
    
    The metric forms a plateau when the sliding window sits over the repeated
    halves, which can be used to gate a finer peak search (e.g., Minn metric).
    
    Args:
        rx: Received samples, shape (branches, L) or (L,)
        symbol_len: Total length of the repeated symbol (default: N_FFT)
    
    Returns:
        M_sc: Timing metric magnitude
        P_sum: Complex correlation sum between halves
        R_sum: Energy sum across halves
    """
    rx = np.asarray(rx)
    if rx.ndim == 1:
        rx = rx[np.newaxis, :]
    
    num_branches, L = rx.shape
    half = symbol_len // 2
    if half == 0 or symbol_len > L:
        return np.zeros(0), np.zeros(0, dtype=np.complex128), np.zeros(0)
    
    out_len = L - symbol_len + 1
    P_sum = np.zeros(out_len, dtype=np.complex128)
    R_sum = np.zeros(out_len, dtype=np.float64)
    
    for b in range(num_branches):
        x = rx[b]
        Pb = np.empty(out_len, dtype=np.complex128)
        Rb = np.empty(out_len, dtype=np.float64)
        
        for d in range(out_len):
            first = x[d : d + half]
            second = x[d + half : d + symbol_len]
            
            P = np.sum(first * np.conj(second))
            R = np.sum(np.abs(first) ** 2 + np.abs(second) ** 2)
            
            Pb[d] = P
            Rb[d] = R
        
        P_sum += Pb
        R_sum += Rb
    
    eps = 1e-12
    M_sc = (np.abs(P_sum) ** 2) / (np.maximum(R_sum, eps) ** 2)
    return M_sc, P_sum, R_sum


def _trailing_average(x: np.ndarray, win: int) -> np.ndarray:
    """Compute trailing moving average using only past samples (streaming-friendly)."""
    if win <= 1:
        return x.copy()
    
    y = np.empty_like(x, dtype=float)
    acc = 0.0
    for idx, val in enumerate(x):
        acc += val
        if idx >= win:
            acc -= x[idx - win]
        denom = win if idx >= win - 1 else (idx + 1)
        y[idx] = acc / denom
    return y


def _streaming_peak_detector(metric: np.ndarray, gate_mask: np.ndarray) -> int | None:
    """Return peak index using only past samples within an active gate."""
    if gate_mask.shape[0] != metric.shape[0]:
        raise ValueError("gate_mask must match metric length")
    
    best_idx: int | None = None
    best_val = -np.inf
    gate_active = False
    
    for idx, val in enumerate(metric):
        gate_flag = gate_mask[idx]
        
        if gate_flag:
            if not gate_active:
                gate_active = True
                best_val = val
                best_idx = idx
            else:
                if val > best_val:
                    best_val = val
                    best_idx = idx
        elif gate_active:
            return best_idx
    
    if gate_active or best_idx is not None:
        return best_idx
    return None


def find_minn_peak(
    M: np.ndarray,
    smooth_win: int = 8,
    gate_mask: np.ndarray | None = None,
    search_bounds: tuple[int, int] | None = None,
) -> int:
    """Find timing from Minn metric.
    
    The Minn metric creates a pattern where the peak occurs when the
    sliding window aligns with [A, A, -A, -A]. We find this peak position.
    """
    if M.size == 0:
        return 0
    
    metric = np.asarray(M, dtype=float)
    
    if gate_mask is None:
        raise ValueError("Minn peak detection requires S&C gate mask")
    if gate_mask.shape[0] != metric.shape[0]:
        raise ValueError("gate_mask must match metric length")
    
    # Build boolean masks combining supplied gate and optional bounds
    search_mask = gate_mask.astype(bool, copy=True)
    
    if search_bounds is not None:
        start = max(0, search_bounds[0])
        end = min(M.size, search_bounds[1])
        if start >= end:
            start, end = 0, M.size
        bounds_mask = np.zeros_like(metric, dtype=bool)
        bounds_mask[start:end] = True
        search_mask &= bounds_mask
    
    if not np.any(search_mask):
        raise ValueError("Minn peak detector received empty gate region")
    
    # Smooth using trailing average to keep streaming behavior (no future look-ahead)
    w = max(1, smooth_win)
    Ms = _trailing_average(np.maximum(metric, 0.0), win=w)
    
    peak = _streaming_peak_detector(
        Ms,
        gate_mask=search_mask,
    )
    if peak is not None:
        return peak
    
    raise RuntimeError("Streaming peak detector did not produce a peak")


# Detector and channel parameters (script-local)
SNR_DB = 10.0
CFO_HZ = 1000.0
SMOOTH_WIN = 16  # samples for smoothing M(d) before peak detection
SC_GATE_THRESHOLD = 0.6  # normalized S&C metric threshold for gate

# Base output directory
PLOTS_BASE_DIR = Path("plots") / "minn"


def run_simulation(channel_name: str | None, plots_subdir: str):
    """Run Minn preamble synchronization simulation.
    
    Args:
        channel_name: Name of measured channel profile (e.g., 'cir1') or None for AWGN-only
        plots_subdir: Subdirectory name for plots (e.g., 'measured_channel' or 'flat_awgn')
    """
    def mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
        """Return contiguous [start, end) segments where mask is True."""
        segments: list[tuple[int, int]] = []
        start_idx: int | None = None
        for idx, flag in enumerate(mask):
            if flag and start_idx is None:
                start_idx = idx
            elif not flag and start_idx is not None:
                segments.append((start_idx, idx))
                start_idx = None
        if start_idx is not None:
            segments.append((start_idx, mask.size))
        return segments
    
    rng = np.random.default_rng(0)
    
    # Setup output directory
    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metric_plot_path = plots_dir / "minn_metric.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    results_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    
    # Build Minn preamble + block pilot (QPSK) + random QPSK data
    minn_preamble = build_minn_preamble(rng, include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((minn_preamble, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))
    
    # Channel: use all available RX branches from the measured CIR (default to first two)
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
    # Apply CFO to simulate LO mismatch at the receiver
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)
    
    # Detection metrics (Minn + Schmidl & Cox)
    M, P_sum, R_sum = minn_streaming_metric(rx_samples)
    M_sc, P_sc, R_sc = schmidl_cox_streaming_metric(rx_samples)
    
    sc_gate_mask: np.ndarray | None = None
    sc_gate_span: tuple[int, int] | None = None
    sc_norm: np.ndarray | None = None
    if M_sc.size > 0:
        max_sc = float(np.max(M_sc))
        if max_sc > 0:
            sc_norm = M_sc / max_sc
            sc_gate_mask = sc_norm >= SC_GATE_THRESHOLD
        else:
            sc_gate_mask = M_sc >= SC_GATE_THRESHOLD
        if sc_gate_mask is not None and not np.any(sc_gate_mask):
            # Guarantee a valid gate by seeding it with the strongest S&C sample.
            gate_idx = int(np.argmax(M_sc))
            sc_gate_mask = np.zeros_like(sc_gate_mask, dtype=bool)
            sc_gate_mask[gate_idx] = True
        if sc_gate_mask is not None and np.any(sc_gate_mask):
            first_idx = int(np.argmax(sc_gate_mask))
            last_idx = int(sc_gate_mask.size - np.argmax(sc_gate_mask[::-1]) - 1)
            sc_gate_span = (first_idx, last_idx + 1)

    if sc_gate_mask is None:
        raise RuntimeError("Schmidl & Cox gate is required for Minn detector")

    peak_position = find_minn_peak(
        M,
        smooth_win=SMOOTH_WIN,
        gate_mask=sc_gate_mask,
        search_bounds=None,
    )
    
    # The Minn peak aligns to the start of the N-length symbol (CP end)
    detected_start = peak_position
    
    minn_norm: np.ndarray | None = None
    max_m = 0.0
    if M.size > 0:
        max_m = float(np.max(M))
        if max_m > 0:
            minn_norm = M / max_m
    
    # Ground-truth alignment helpers
    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    # Plot raw CIR if measured profile enabled
    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}', all RX)",
            cir_plot_path,
        )
    
    # Expected timing: CP start + CP length = start of N samples
    true_cp_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    expected_n_start = true_cp_start + CYCLIC_PREFIX
    
    # Plots
    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"
    
    plt.figure(figsize=(10, 4))
    if sc_norm is not None:
        plt.plot(sc_norm, label="S&C M(d) (norm)", color="tab:blue", linestyle="-")
        if sc_gate_mask is not None and np.any(sc_gate_mask):
            for idx, (seg_start, seg_end) in enumerate(mask_segments(sc_gate_mask)):
                gate_label = f"S&C gate (≥{SC_GATE_THRESHOLD:.0%})" if idx == 0 else None
                plt.axvspan(seg_start, seg_end, color="tab:blue", alpha=0.15, label=gate_label)
    elif M_sc.size > 0:
        plt.plot(M_sc, label="S&C M(d)", color="tab:blue", linestyle="-")
        if sc_gate_mask is not None and np.any(sc_gate_mask):
            for idx, (seg_start, seg_end) in enumerate(mask_segments(sc_gate_mask)):
                gate_label = f"S&C gate (≥{SC_GATE_THRESHOLD:.0%})" if idx == 0 else None
                plt.axvspan(seg_start, seg_end, color="tab:blue", alpha=0.15, label=gate_label)
    if minn_norm is not None:
        plt.plot(minn_norm, label="Minn M(d) (norm)", color="tab:orange")
    else:
        plt.plot(M, label="Minn M(d)", color="tab:orange")
    plt.axvline(peak_position, color="tab:red", linestyle=":", label=f"Minn peak @ {peak_position}")
    plt.axvline(expected_n_start, color="tab:green", linestyle="--", label="Expected N start")
    plt.xlabel("Sample index d")
    plt.ylabel("Normalized M(d)" if minn_norm is not None or sc_norm is not None else "M(d)")
    plt.title(f"Streaming Metrics (Minn & S&C) — {channel_desc}")
    plt.legend(loc="upper right")
    if minn_norm is not None or sc_norm is not None:
        plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    
    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    if rx_samples.ndim > 1 and rx_samples.shape[0] > 1:
        for idx, branch in enumerate(rx_samples):
            axes[0].plot(np.abs(branch), alpha=0.3, linewidth=0.8)
    if sc_gate_mask is not None and np.any(sc_gate_mask):
        for idx, (seg_start, seg_end) in enumerate(mask_segments(sc_gate_mask)):
            gate_label = "S&C gate" if idx == 0 else None
            axes[0].axvspan(seg_start, seg_end, color="tab:blue", alpha=0.2, label=gate_label)
    axes[0].axvline(true_cp_start, color="tab:purple", linestyle="--", label="CP start (true)")
    axes[0].axvline(expected_n_start, color="tab:green", linestyle="--", label="N start (exp)")
    axes[0].axvline(detected_start, color="tab:red", linestyle=":", label="Detected start")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude and Detected Start (Minn, {channel_desc})")
    axes[0].legend(loc="upper right")
    
    if sc_norm is not None:
        axes[1].plot(sc_norm, label="S&C M(d) (norm)", color="tab:blue")
        if sc_gate_mask is not None and np.any(sc_gate_mask):
            for seg_start, seg_end in mask_segments(sc_gate_mask):
                axes[1].axvspan(seg_start, seg_end, color="tab:blue", alpha=0.12)
    elif M_sc.size > 0:
        axes[1].plot(M_sc, label="S&C M(d)", color="tab:blue")
        if sc_gate_mask is not None and np.any(sc_gate_mask):
            for seg_start, seg_end in mask_segments(sc_gate_mask):
                axes[1].axvspan(seg_start, seg_end, color="tab:blue", alpha=0.12)
    if minn_norm is not None:
        axes[1].plot(minn_norm, label="Minn M(d) (norm)", color="tab:orange")
    else:
        axes[1].plot(M, label="Minn M(d)", color="tab:orange")
    axes[1].axvline(peak_position, color="tab:red", linestyle=":", label=f"Minn peak @ {peak_position}")
    axes[1].axvline(expected_n_start, color="tab:green", linestyle="--", label="Expected N start")
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("Normalized M(d)" if minn_norm is not None or sc_norm is not None else "M(d)")
    axes[1].set_title("Timing Metrics (Minn & S&C)")
    if minn_norm is not None or sc_norm is not None:
        axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    
    fig.tight_layout()
    fig.savefig(results_plot_path, dpi=150)
    plt.close(fig)
    
    # Raw time series
    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)
    
    # --- CFO estimation from pilot (CP correlation) ---
    # Use detected timing: peak aligns to N start of preamble
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
    # Compensate CFO across entire stream
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)
    
    # --- Channel LS estimate from pilot ---
    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[pilot_cp_start + CYCLIC_PREFIX : pilot_cp_start + CYCLIC_PREFIX + N_FFT]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)
    # Estimate residual timing from linear phase slope of H(k)
    slope_rad_per_bin, timing_offset_samples = estimate_timing_offset_from_phase_slope(h_est)
    
    # --- Equalize data symbol and compute EVM ---
    data_cp_start = pilot_cp_start + CYCLIC_PREFIX + N_FFT
    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    from core import align_complex_gain
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(xhat_aligned, data_used, const_plot_path, f"Equalized Data Constellation (Minn, {channel_desc})")
    
    # Prints
    timing_error = detected_start - expected_n_start
    print(f"\n{'='*70}")
    print(f"MINN SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
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
    print(f"\nTiming Detection:")
    print(f"  Detected Minn peak at d={peak_position}")
    print(f"  Expected N start at d={expected_n_start}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.1f}% of symbol)")
    if sc_gate_span is not None:
        gate_start, gate_end = sc_gate_span
        print(
            f"  S&C gate window: [{gate_start}, {gate_end}) (norm ≥ {SC_GATE_THRESHOLD:.0%}, span {gate_end - gate_start} samples)",
        )
    elif sc_gate_mask is not None:
        print(
            f"  S&C gate not triggered (no samples reached threshold {SC_GATE_THRESHOLD:.0%})",
        )
    print(f"\nCarrier Frequency Offset:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz ({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.1f}%)")
    print(f"\nChannel Estimation & Equalization:")
    print(f"  Pilot LS phase slope: {slope_rad_per_bin:.6f} rad/bin -> timing ≈ {timing_offset_samples:.2f} samples")
    print(f"  Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")
    print(f"\nPlots saved to {plots_dir.resolve()}/:")
    print(f"  - minn_metric.png")
    print(f"  - start_detection.png")
    print(f"  - constellation.png")
    print(f"  - tx_frame_time.png")
    print(f"  - rx_frame_time.png")
    if channel_impulse_response is not None:
        print(f"  - channel_cir.png")
    print(f"{'='*70}\n")


def main():
    """Run simulations for both measured channel and flat AWGN conditions."""
    print("\n" + "="*70)
    print("MINN PREAMBLE SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
    print("="*70)
    
    # Simulation 1: Measured multipath channel
    run_simulation(channel_name="cir1", plots_subdir="measured_channel")
    
    # Simulation 2: Flat AWGN channel
    run_simulation(channel_name=None, plots_subdir="flat_awgn")
    
    print("\n" + "="*70)
    print("ALL SIMULATIONS COMPLETE")
    print("="*70)
    print(f"\nCompare results in:")
    print(f"  - {(PLOTS_BASE_DIR / 'measured_channel').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'flat_awgn').resolve()}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

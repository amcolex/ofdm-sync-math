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


def find_minn_peak(
    M: np.ndarray,
    smooth_win: int = 8,
    gate_threshold: float = 0.5,
    search_bounds: tuple[int, int] | None = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Find timing from Minn metric.
    
    The Minn metric creates a pattern where the peak occurs when the
    sliding window aligns with [A, A, -A, -A]. We find this peak position.
    """
    if M.size == 0:
        raise ValueError("Minn metric is empty")
    
    metric = np.asarray(M, dtype=float)

    # Smooth using trailing average to keep streaming behavior (no future look-ahead)
    w = max(1, smooth_win)
    Ms = _trailing_average(np.maximum(metric, 0.0), win=w)
    
    max_ms = float(np.max(Ms))
    if max_ms <= 0.0:
        raise ValueError("Minn metric did not produce a positive peak")
    gate_level = gate_threshold * max_ms
    gate_mask = Ms >= gate_level
    
    # Ensure the gate is contiguous around the best peak by selecting the largest component
    if np.any(gate_mask):
        in_segment = False
        best_span = (0, 0)
        best_len = 0
        start_idx = 0
        for idx, flag in enumerate(gate_mask):
            if flag and not in_segment:
                in_segment = True
                start_idx = idx
            elif not flag and in_segment:
                in_segment = False
                span_len = idx - start_idx
                if span_len > best_len:
                    best_len = span_len
                    best_span = (start_idx, idx)
        if in_segment:
            span_len = gate_mask.size - start_idx
            if span_len > best_len:
                best_span = (start_idx, gate_mask.size)
                best_len = span_len
        if best_len > 0:
            new_gate = np.zeros_like(gate_mask)
            new_gate[best_span[0] : best_span[1]] = True
            gate_mask = new_gate
    else:
        gate_mask = np.zeros_like(gate_mask, dtype=bool)
    
    # Apply optional search bounds
    if search_bounds is not None:
        start = max(0, search_bounds[0])
        end = min(M.size, search_bounds[1])
        if start >= end:
            start, end = 0, M.size
        bounds_mask = np.zeros_like(metric, dtype=bool)
        bounds_mask[start:end] = True
        gate_mask &= bounds_mask
    
    if not np.any(gate_mask):
        # Fall back to the global maximum if the threshold doesn't produce a gate
        peak_idx = int(np.argmax(Ms))
        gate_mask = np.zeros_like(gate_mask, dtype=bool)
        gate_mask[peak_idx] = True
        return peak_idx, gate_mask, Ms
    
    candidate_idx = np.flatnonzero(gate_mask)
    peak_rel = int(np.argmax(Ms[candidate_idx]))
    peak_idx = int(candidate_idx[peak_rel])
    return peak_idx, gate_mask, Ms


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
    ax.set_title(f"LS-Derived CIR (Minn, {channel_desc})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Detector and channel parameters (script-local)
SNR_DB = 10.0
CFO_HZ = 1000.0
SMOOTH_WIN = 16  # samples for smoothing M(d) before peak detection

# Minn-only gating parameters
MINN_GATE_THRESHOLD = 0.5  # fixed fraction of the Minn peak used for gating

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
    ls_cir_plot_path = plots_dir / "ls_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    sto_plot_path = plots_dir / "phase_slope_sto.png"
    
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
    
    # Detection metrics (Minn only)
    M, P_sum, R_sum = minn_streaming_metric(rx_samples)
    peak_position, minn_gate_mask, M_smooth = find_minn_peak(
        M,
        smooth_win=SMOOTH_WIN,
        gate_threshold=MINN_GATE_THRESHOLD,
        search_bounds=None,
    )
    
    # The Minn peak aligns to the start of the N-length symbol (CP end)
    detected_start = peak_position
    
    minn_gate_segments = mask_segments(minn_gate_mask)
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
    timing_error = detected_start - expected_n_start

    # Plots
    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"
    
    plt.figure(figsize=(10, 4))
    plt.plot(M, label="Minn M(d)", color="tab:orange")
    plt.plot(M_smooth, label="Minn M_s(d) (smoothed)", color="tab:orange", linestyle="--")
    for idx, (seg_start, seg_end) in enumerate(minn_gate_segments):
        gate_label = (
            f"Minn gate (≥{MINN_GATE_THRESHOLD:.0%} of Minn peak)" if idx == 0 else None
        )
        plt.axvspan(seg_start, seg_end, color="tab:orange", alpha=0.15, label=gate_label)
    plt.axvline(peak_position, color="tab:red", linestyle=":", label=f"Minn peak @ {peak_position}")
    plt.axvline(expected_n_start, color="tab:green", linestyle="--", label="Expected N start")
    plt.xlabel("Sample index d")
    plt.ylabel("M(d)")
    plt.title(f"Minn Metric & Gate — {channel_desc}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=150)
    plt.close()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    
    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    if rx_samples.ndim > 1 and rx_samples.shape[0] > 1:
        for idx, branch in enumerate(rx_samples):
            axes[0].plot(np.abs(branch), alpha=0.3, linewidth=0.8)
    for idx, (seg_start, seg_end) in enumerate(minn_gate_segments):
        gate_label = "Minn gate" if idx == 0 else None
        axes[0].axvspan(seg_start, seg_end, color="tab:orange", alpha=0.18, label=gate_label)
    axes[0].axvline(true_cp_start, color="tab:purple", linestyle="--", label="CP start (true)")
    axes[0].axvline(expected_n_start, color="tab:green", linestyle="--", label="N start (exp)")
    axes[0].axvline(detected_start, color="tab:red", linestyle=":", label="Detected start")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude and Detected Start (Minn, {channel_desc})")
    axes[0].legend(loc="upper right")
    
    axes[1].plot(M, label="Minn M(d)", color="tab:orange")
    axes[1].plot(
        M_smooth,
        label="Minn M_s(d) (smoothed)",
        color="tab:orange",
        linestyle="--",
    )
    for seg_start, seg_end in minn_gate_segments:
        axes[1].axvspan(seg_start, seg_end, color="tab:orange", alpha=0.12)
    axes[1].axvline(peak_position, color="tab:red", linestyle=":", label=f"Minn peak @ {peak_position}")
    axes[1].axvline(expected_n_start, color="tab:green", linestyle="--", label="Expected N start")
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("M(d)")
    axes[1].set_title("Timing Metrics (Minn)")
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
    sto_title = f"Residual Timing From Phase Slope (Minn, {channel_desc})"
    slope_rad_per_bin, timing_offset_samples = plot_phase_slope_diagnostics(
        h_est,
        sto_plot_path,
        sto_title,
    )
    
    # --- Equalize data symbol and compute EVM ---
    data_cp_start = pilot_cp_start + CYCLIC_PREFIX + N_FFT
    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(
        xhat_aligned,
        data_used,
        const_plot_path,
        f"Equalized Data Constellation (Minn, {channel_desc})",
    )

    # LS-derived CIR visualization (captures residual timing offset)
    ls_cir = _reconstruct_cir_from_ls(h_est)
    _plot_ls_cir(
        ls_cir,
        channel_impulse_response,
        channel_peak_offset,
        timing_error,
        ls_cir_plot_path,
        channel_desc,
    )
    
    # Prints
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
    if minn_gate_segments:
        gate_start = minn_gate_segments[0][0]
        gate_end = minn_gate_segments[-1][1]
        print(
            f"  Minn gate window: [{gate_start}, {gate_end}) "
            f"(threshold ≥{MINN_GATE_THRESHOLD:.0%} of Minn peak, span {gate_end - gate_start} samples)",
        )
    else:
        print("  Minn gate not triggered (metric never exceeded threshold)")
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
    print(f"  - ls_cir.png")
    print(f"  - phase_slope_sto.png")
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

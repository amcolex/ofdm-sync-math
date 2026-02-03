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
    plot_phase_slope_diagnostics,
)


def build_sc_preamble(rng: np.random.Generator, include_cp: bool = True) -> np.ndarray:
    all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    even_idx = all_idx[(all_idx % 2) == 0]
    bpsk = rng.choice([-1.0, 1.0], size=even_idx.shape[0])
    spectrum = allocate_subcarriers(N_FFT, even_idx, bpsk)
    symbol = spectrum_to_time_domain(spectrum)
    if include_cp:
        return add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol


def sc_streaming_metric(rx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rx = np.asarray(rx)
    if rx.ndim == 1:
        rx = rx[np.newaxis, :]

    num_branches, L = rx.shape
    half = N_FFT // 2
    N = N_FFT
    out_len = max(L - N + 1, 0)
    if out_len <= 0:
        return np.zeros(0), np.zeros(0, dtype=complex), np.zeros(0)

    P_sum = np.zeros(out_len, dtype=np.complex128)
    R_sum = np.zeros(out_len, dtype=np.float64)

    for b in range(num_branches):
        x = rx[b]
        P = np.sum(x[0:half] * np.conj(x[half:N]))
        R = np.sum(np.abs(x[half:N]) ** 2)
        Pb = np.empty(out_len, dtype=np.complex128)
        Rb = np.empty(out_len, dtype=np.float64)
        Pb[0] = P
        Rb[0] = R
        for d in range(1, out_len):
            old_a = x[d - 1]
            old_b = x[d - 1 + half]
            new_b = x[d - 1 + N]
            P = P - old_a * np.conj(old_b) + old_b * np.conj(new_b)
            R = R - (np.abs(old_b) ** 2) + (np.abs(new_b) ** 2)
            Pb[d] = P
            Rb[d] = R
        P_sum += Pb
        R_sum += Rb

    eps = 1e-12
    M = (np.abs(P_sum) ** 2) / (np.maximum(R_sum, eps) ** 2)
    return M, P_sum, R_sum


def find_plateau_end_from_metric(
    M: np.ndarray, cp_len: int, lookahead: int | None = None, smooth_win: int = 8
) -> int:
    """Estimate the end of the first S&C plateau (≈ CP end / N start).

    Strategy:
    1) Smooth M(d) and pick the earliest contiguous run above a relative
       threshold of the global maximum. The plateau end is the right edge of
       that earliest sufficiently long run (length ≥ cp_len/2).
    2) If that fails (e.g., no long-enough run), fall back to a slope-based
       method that looks for the largest drop over a small lookahead window
       around the strongest plateau and returns the drop center.
    """
    if M.size == 0:
        return 0

    # --- Common prep ---
    L = (cp_len // 4) if lookahead is None else int(max(1, lookahead))
    w = max(1, smooth_win)
    Ms = np.convolve(M, np.ones(w, dtype=float) / w, mode="same")

    # --- Very early boundary: first small drop after the local maximum ---
    # Anchor at the strongest plateau maximum, then move forward and pick the
    # first index that falls below a high fraction of that max (default 95%).
    # This tends to align with the CP end rather than the later gradual tail.
    center = int(np.argmax(Ms))
    post_hi = min(Ms.size, center + cp_len)
    if post_hi > center + 1:
        frac = 0.95
        thr_local = frac * float(Ms[center])
        post = Ms[center:post_hi]
        below = np.flatnonzero(post <= thr_local)
        if below.size > 0:
            return int(center + below[0])

    # --- Primary: earliest-long-run above threshold ---
    min_run = max(8, cp_len // 2)
    peak = float(np.max(Ms))
    if peak > 0:
        # Use a conservative relative threshold to be robust to CFO/noise.
        thr = 0.6 * peak
        hi_mask = Ms >= thr
        # Find contiguous runs in hi_mask
        # Convert to indices of True and segment them.
        idx = np.flatnonzero(hi_mask)
        if idx.size > 0:
            # Split where gaps > 1 sample
            splits = np.where(np.diff(idx) > 1)[0] + 1
            segments = np.split(idx, splits)
            # Pick earliest segment with sufficient length
            for seg in segments:
                if seg.size >= min_run:
                    return int(seg[-1])  # right edge of earliest valid plateau

    # --- Fallback: slope-based drop around strongest plateau ---
    center = int(np.argmax(Ms))
    lo = max(0, center - cp_len)
    hi = min(Ms.size - L - 1, center + cp_len)
    window = Ms[lo:hi]
    ahead = Ms[lo + L : hi + L]
    drop = window - ahead
    if drop.size == 0:
        return center
    d_rel = int(np.argmax(drop))
    plateau_end = lo + d_rel + (L // 2)
    return plateau_end


# Detector and channel parameters (script-local)
SNR_DB = 10.0
CFO_HZ = 1000.0
SC_DELTA = 16  # step back from plateau end to sit inside CP (8–16 suggested)
SMOOTH_WIN = 16  # samples for smoothing M(d) before slope detection

# Base output directory
PLOTS_BASE_DIR = Path("plots") / "sc"


def run_simulation(channel_name: str | None, plots_subdir: str):
    """Run Schmidl & Cox preamble synchronization simulation.
    
    Args:
        channel_name: Name of measured channel profile (e.g., 'cir1') or None for AWGN-only
        plots_subdir: Subdirectory name for plots (e.g., 'measured_channel' or 'flat_awgn')
    """
    rng = np.random.default_rng(0)
    
    # Setup output directory
    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    metric_plot_path = plots_dir / "sc_metric.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    results_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    sto_plot_path = plots_dir / "phase_slope_sto.png"

    # Build S&C preamble + block pilot (QPSK) + random QPSK data
    sc_preamble = build_sc_preamble(rng, include_cp=False)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((sc_preamble, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))

    # Channel: use only ch1 (index 1) for SISO when a profile is set
    if channel_name is None:
        channel_impulse_response = None
    else:
        cir_bank = load_measured_cir(channel_name)
        channel_impulse_response = cir_bank[1:2]  # shape (1, taps)

    rx_samples = apply_channel(
        tx_samples,
        SNR_DB,
        rng,
        channel_impulse_response=channel_impulse_response,
    )
    # Apply CFO to simulate LO mismatch at the receiver
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

    # Detection metric (S&C)
    M, P_sum, R_sum = sc_streaming_metric(rx_samples)
    plateau_end = find_plateau_end_from_metric(
        M,
        CYCLIC_PREFIX,
        lookahead=CYCLIC_PREFIX // 4,
        smooth_win=SMOOTH_WIN,
    )
    coarse_start = max(plateau_end - SC_DELTA, 0)

    # Ground-truth alignment helpers
    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    # Plot raw CIR if measured profile enabled
    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}', ch1)",
            cir_plot_path,
        )

    # The S&C plateau "left" edge aligns at CP end (useful start).
    # Show this as the expected left edge for clarity.
    true_cp_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    expected_left_edge = true_cp_start + CYCLIC_PREFIX

    # Plots
    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"
    
    plt.figure(figsize=(10, 4))
    plt.plot(M, label="S&C M(d)")
    plt.axvline(plateau_end, color="tab:red", linestyle=":", label="Plateau end")
    plt.axvline(expected_left_edge, color="tab:green", linestyle="--", label="Plateau start (exp)")
    plt.xlabel("Sample index d")
    plt.ylabel("M(d)")
    plt.title(f"Schmidl & Cox Streaming Metric ({channel_desc})")
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
    axes[0].axvline(true_cp_start, color="tab:purple", linestyle="--", label="CP start (true)")
    axes[0].axvline(expected_left_edge, color="tab:green", linestyle="--", label="Plateau start (exp)")
    axes[0].axvline(plateau_end, color="tab:red", linestyle=":", label="Plateau end (det)")
    axes[0].axvline(coarse_start, color="tab:orange", linestyle=":", label=f"Coarse start = end-{SC_DELTA}")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude and Detected Start (S&C, {channel_desc})")
    axes[0].legend(loc="upper right")

    axes[1].plot(M, label="S&C M(d)")
    axes[1].axvline(plateau_end, color="tab:red", linestyle=":", label="Plateau end (det)")
    axes[1].axvline(expected_left_edge, color="tab:green", linestyle="--", label="Plateau start (exp)")
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("M(d)")
    axes[1].set_title("Plateau-Based Timing (End minus delta)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_plot_path, dpi=150)
    plt.close(fig)

    # Raw time series
    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)

    # --- CFO estimation from pilot (CP correlation) ---
    # Use ONLY coarse timing: plateau_end ≈ preamble N-start. No CP refinement.
    preamble_n_start_est = plateau_end
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
    sto_title = f"Residual Timing From Phase Slope (S&C, {channel_desc})"
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
    from core import align_complex_gain
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(xhat_aligned, data_used, const_plot_path, f"Equalized Data Constellation (S&C, {channel_desc})")

    # Prints
    timing_error = coarse_start - true_cp_start
    print(f"\n{'='*70}")
    print(f"SCHMIDL & COX SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
    print(f"{'='*70}")
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {1 if rx_samples.ndim == 1 else rx_samples.shape[0]}")
    if channel_impulse_response is not None:
        print(
            f"Applied measured channel '{channel_name}' (ch1 only) taps={channel_impulse_response.shape[1]} main-path offset={channel_peak_offset}",
        )
    else:
        print("Channel profile: Flat AWGN (no multipath)")
    print(f"\nTiming Detection:")
    print(f"  Detected plateau end at d={plateau_end}")
    print(f"  Coarse start (end - {SC_DELTA}) at d={coarse_start}")
    print(f"  Expected plateau start at d={expected_left_edge}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.1f}% of symbol)")
    print(f"\nCarrier Frequency Offset:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz ({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.1f}%)")
    print(f"\nChannel Estimation & Equalization:")
    print(f"  Pilot LS phase slope: {slope_rad_per_bin:.6f} rad/bin -> timing ≈ {timing_offset_samples:.2f} samples")
    print(f"  Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")
    print(f"\nPlots saved to {plots_dir.resolve()}/:")
    print(f"  - sc_metric.png")
    print(f"  - start_detection.png")
    print(f"  - constellation.png")
    print(f"  - tx_frame_time.png")
    print(f"  - rx_frame_time.png")
    print(f"  - phase_slope_sto.png")
    if channel_impulse_response is not None:
        print(f"  - channel_cir.png")
    print(f"{'='*70}\n")


def main():
    """Run simulations for both measured channel and flat AWGN conditions."""
    print("\n" + "="*70)
    print("SCHMIDL & COX SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
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

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
    add_cyclic_prefix,
    plot_time_series,
    compute_channel_peak_offset,
    build_random_qpsk_symbol,
    estimate_cfo_from_cp,
    ofdm_fft_used,
    ls_channel_estimate,
    equalize,
    evm_rms_db,
    plot_constellation,
    SAMPLE_RATE_HZ,
    apply_cfo,
    estimate_timing_offset_from_phase_slope,
    align_complex_gain,
)


PARK_PREAMBLE_CP = CYCLIC_PREFIX // 2


def build_park_preamble(rng: np.random.Generator, include_cp: bool = True) -> np.ndarray:
    """Build Park-style preamble with contiguous active tones around DC."""
    if N_FFT % 4 != 0:
        raise ValueError("N_FFT must be divisible by 4 for Park preamble")

    quarter = N_FFT // 4
    bits = rng.integers(0, 4, size=quarter)
    qpsk_vals = np.exp(1j * (np.pi / 2.0) * bits)
    A = qpsk_vals
    B = A[::-1]
    x_ideal = np.concatenate([A, B, np.conj(A), np.conj(B)])

    X = np.fft.fftshift(np.fft.fft(x_ideal, N_FFT))
    mask = np.zeros(N_FFT, dtype=float)
    idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    dc = N_FFT // 2
    mask[(dc + idx) % N_FFT] = 1.0
    X_masked = X * mask
    x_masked = np.fft.ifft(np.fft.ifftshift(X_masked), N_FFT)

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.abs(x) ** 2)))

    denom = _rms(x_masked)
    if denom > 0:
        x_masked *= _rms(x_ideal) / denom

    if include_cp:
        return add_cyclic_prefix(x_masked, PARK_PREAMBLE_CP)
    return x_masked


def park_streaming_metric(rx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Park timing metric across one or more receive branches.

    Returns:
        ds: Sample indices where the metric is evaluated (centers of the window)
        M: Normalized Park timing metric
        P_sum: Complex correlation sum across branches
        E_sum: Energy sum across branches
    """
    x = np.asarray(rx)
    if x.ndim == 1:
        x = x[np.newaxis, :]

    num_branches, L = x.shape
    half = N_FFT // 2
    if half == 0 or L < (2 * half + 1):
        return (
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=np.complex128),
            np.zeros(0, dtype=float),
        )

    valid_lo = half
    valid_hi = L - half - 1
    if valid_hi < valid_lo:
        return (
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=np.complex128),
            np.zeros(0, dtype=float),
        )

    ds = np.arange(valid_lo, valid_hi + 1)
    P_sum = np.zeros(ds.size, dtype=np.complex128)
    E_sum = np.zeros(ds.size, dtype=np.float64)
    offsets = np.arange(half)

    for b in range(num_branches):
        xb = x[b]
        for idx, d in enumerate(ds):
            seg_fwd = xb[d : d + half]
            seg_bwd = xb[d - offsets]
            P = np.sum(seg_bwd * seg_fwd)
            E = np.sum(np.abs(seg_fwd) ** 2)
            P_sum[idx] += P
            E_sum[idx] += E

    eps = 1e-12
    M = (np.abs(P_sum) ** 2) / (np.maximum(E_sum, eps) ** 2)
    return ds, M, P_sum, E_sum


# Detector/channel parameters
SNR_DB = 10.0
CFO_HZ = 1000.0
PLOTS_BASE_DIR = Path("plots") / "park"


def run_simulation(channel_name: str | None, plots_subdir: str) -> None:
    """Run Park preamble synchronization in requested channel setting."""
    rng = np.random.default_rng(0)

    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_plot_path = plots_dir / "park_metric.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    detection_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    const_plot_path = plots_dir / "constellation.png"

    park_preamble = build_park_preamble(rng, include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((park_preamble, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))

    if channel_name is None:
        channel_impulse_response = None
    else:
        channel_impulse_response = load_measured_cir(channel_name)

    rx_samples = apply_channel(
        tx_samples,
        SNR_DB,
        rng,
        channel_impulse_response=channel_impulse_response,
    )
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

    ds, M, P_sum, E_sum = park_streaming_metric(rx_samples)
    if ds.size == 0:
        raise RuntimeError("Park metric window is empty; increase frame length or adjust parameters.")

    peak_rel = int(np.argmax(M))
    det_center = int(ds[peak_rel])
    det_symbol_start = max(det_center - (N_FFT // 2), 0)
    det_cp_start = max(det_symbol_start - PARK_PREAMBLE_CP, 0)

    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    true_cp_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    true_symbol_start = true_cp_start + PARK_PREAMBLE_CP
    true_center = true_symbol_start + (N_FFT // 2)

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"

    plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(ds, np.abs(P_sum), label="|P(d)|")
    ax1.axvline(true_center, color="tab:green", linestyle="--", label="True center")
    ax1.axvline(det_center, color="tab:red", linestyle=":", label="Detected center")
    ax1.set_ylabel("|P(d)|")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(ds, E_sum, label="E(d)", color="tab:orange")
    ax2.set_ylabel("E(d)")
    ax2.grid(alpha=0.3)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(ds, M, label="Park metric M(d)", color="tab:blue")
    ax3.axvline(true_center, color="tab:green", linestyle="--", label="True center")
    ax3.axvline(det_center, color="tab:red", linestyle=":", label="Detected center")
    ax3.set_xlabel("Sample index d")
    ax3.set_ylabel("M(d)")
    ax3.grid(alpha=0.3)
    ax3.legend(loc="upper right")

    plt.suptitle(f"Park Correlation Components — {channel_desc}")
    plt.tight_layout()
    plt.savefig(metric_plot_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    if rx_samples.ndim > 1 and rx_samples.shape[0] > 1:
        for idx, branch in enumerate(rx_samples):
            axes[0].plot(np.abs(branch), alpha=0.35, linewidth=0.8, label=f"|rx_ch{idx}|" if idx < 4 else None)
    axes[0].axvline(true_cp_start, color="tab:purple", linestyle="--", label="CP start (true)")
    axes[0].axvline(true_symbol_start, color="tab:green", linestyle="--", label="Symbol start (true)")
    axes[0].axvline(det_symbol_start, color="tab:red", linestyle=":", label="Symbol start (det)")
    axes[0].axvspan(
        det_cp_start,
        det_cp_start + PARK_PREAMBLE_CP,
        color="tab:red",
        alpha=0.1,
        label="Detected CP span",
    )
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Frame & Detection (Park, {channel_desc})")
    axes[0].legend(loc="upper right")

    axes[1].plot(ds, M, color="tab:blue", label="Park metric")
    axes[1].axvline(true_center, color="tab:green", linestyle="--", label="True center")
    axes[1].axvline(det_center, color="tab:red", linestyle=":", label="Detected center")
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("M(d)")
    axes[1].set_title("Timing Metric Around Detection")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(detection_plot_path, dpi=150)
    plt.close(fig)

    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)

    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}')",
            cir_plot_path,
        )

    pilot_cp_start_est = det_symbol_start + N_FFT
    max_start = rx_samples.shape[-1] - (N_FFT + CYCLIC_PREFIX)
    if max_start < 0:
        raise RuntimeError("Received frame shorter than one OFDM symbol with CP.")
    pilot_cp_start_est = int(np.clip(pilot_cp_start_est, 0, max_start))
    cfo_est_hz = estimate_cfo_from_cp(
        rx_samples,
        pilot_cp_start_est,
        N_FFT,
        CYCLIC_PREFIX,
        SAMPLE_RATE_HZ,
    )
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)

    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[
        pilot_cp_start_est + CYCLIC_PREFIX : pilot_cp_start_est + CYCLIC_PREFIX + N_FFT
    ]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)
    slope_rad_per_bin, timing_offset_samples = estimate_timing_offset_from_phase_slope(h_est)

    data_cp_start = pilot_cp_start_est + CYCLIC_PREFIX + N_FFT
    max_data_start = rx_eff.shape[-1] - (N_FFT + CYCLIC_PREFIX)
    data_cp_start = int(np.clip(data_cp_start, 0, max_data_start))
    data_td = rx_eff[
        data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT
    ]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(
        xhat_aligned,
        data_used,
        const_plot_path,
        f"Equalized Data Constellation (Park, {channel_desc})",
    )

    timing_error = det_symbol_start - true_symbol_start
    print(f"\n{'='*70}")
    print(f"PARK SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
    print(f"{'='*70}")
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {1 if rx_samples.ndim == 1 else rx_samples.shape[0]}")
    if channel_impulse_response is None:
        print("Channel profile: Flat AWGN (no multipath)")
    else:
        num_rx = channel_impulse_response.shape[0]
        taps = channel_impulse_response.shape[1]
        print(
            f"Applied measured channel '{channel_name}' with {num_rx} RX branch(es), "
            f"{taps} taps, main-path offset {channel_peak_offset}",
        )
    print("\nTiming Detection:")
    print(f"  Detected center index: {det_center}")
    print(f"  Detected symbol start: {det_symbol_start}")
    print(f"  True symbol start:     {true_symbol_start}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.2f}% of symbol)")
    print("\nCarrier Frequency Offset:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz "
          f"({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.2f}%)")
    print("\nChannel Estimation & Equalization:")
    print(f"  Pilot LS phase slope: {slope_rad_per_bin:.6f} rad/bin "
          f"-> timing ≈ {timing_offset_samples:.2f} samples")
    print(f"  Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")
    print(f"\nPlots saved to {plots_dir.resolve()}/:")
    print("  - park_metric.png")
    print("  - start_detection.png")
    print("  - constellation.png")
    print("  - tx_frame_time.png")
    print("  - rx_frame_time.png")
    if channel_impulse_response is not None:
        print("  - channel_cir.png")
    print(f"{'='*70}\n")


def main() -> None:
    """Run Park synchronization simulations for multipath and AWGN channels."""
    print("\n" + "=" * 70)
    print("PARK PREAMBLE SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
    print("=" * 70)

    run_simulation(channel_name="cir1", plots_subdir="measured_channel")
    run_simulation(channel_name=None, plots_subdir="flat_awgn")

    print("\n" + "=" * 70)
    print("ALL PARK SIMULATIONS COMPLETE")
    print("=" * 70)
    print(f"\nCompare results in:")
    print(f"  - {(PLOTS_BASE_DIR / 'measured_channel').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'flat_awgn').resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

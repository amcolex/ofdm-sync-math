import numpy as np
import matplotlib.pyplot as plt
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
)


PSS_LENGTH = 62
PSS_ROOT = 25

# Simulation defaults
SNR_DB = 10.0
CFO_HZ = 0.0  # Assume upstream NCO already corrected CFO

PLOTS_BASE_DIR = Path("plots") / "zc_freq"


def generate_zadoff_chu(root: int, length: int) -> np.ndarray:
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * (n + 1) / length)


def build_pss_symbol(include_cp: bool = True) -> np.ndarray:
    indices = centered_subcarrier_indices(PSS_LENGTH)
    zc_sequence = generate_zadoff_chu(PSS_ROOT, PSS_LENGTH)
    spectrum = allocate_subcarriers(N_FFT, indices, zc_sequence)
    symbol = spectrum_to_time_domain(spectrum)
    if include_cp:
        return add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol


def make_pss_frequency_template() -> tuple[np.ndarray, np.ndarray, float]:
    """Return (centered_bin_indices, template_bins, template_energy)."""
    bin_indices = centered_subcarrier_indices(PSS_LENGTH)
    template_bins = generate_zadoff_chu(PSS_ROOT, PSS_LENGTH)
    energy = float(np.sum(np.abs(template_bins) ** 2))
    return bin_indices, template_bins, energy


def compute_frequency_metric(
    rx_samples: np.ndarray,
    bin_indices: np.ndarray,
    template_bins: np.ndarray,
    template_energy: float,
) -> np.ndarray:
    """Evaluate the LTE-style frequency-domain metric across all offsets."""
    if rx_samples.ndim == 1:
        rx_samples = rx_samples[np.newaxis, :]

    cp = CYCLIC_PREFIX
    usable = N_FFT + cp
    total_len = rx_samples.shape[1]
    num_offsets = total_len - usable + 1
    if num_offsets <= 0:
        msg = "Received stream is shorter than a single OFDM symbol."
        raise ValueError(msg)

    metric = np.zeros(num_offsets, dtype=float)
    dc = N_FFT // 2
    positions = (dc + bin_indices) % N_FFT
    eps = 1e-12

    for offset in range(num_offsets):
        corr_sum = 0.0 + 0.0j
        branch_energy_sum = 0.0
        start = offset + cp
        end = start + N_FFT
        for branch in rx_samples:
            symbol_td = branch[start:end]
            symbol_fd = np.fft.fftshift(np.fft.fft(symbol_td, n=N_FFT))
            bins = symbol_fd[positions]
            corr_sum += np.vdot(template_bins, bins)
            branch_energy_sum += np.sum(np.abs(bins) ** 2)
        denom = max(template_energy * branch_energy_sum, eps)
        metric[offset] = (np.abs(corr_sum) ** 2) / denom

    return metric


def run_simulation(channel_name: str | None, plots_subdir: str) -> None:
    rng = np.random.default_rng(0)

    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    corr_plot_path = plots_dir / "correlation.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    results_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    sto_plot_path = plots_dir / "phase_slope_sto.png"

    pss_symbol = build_pss_symbol(include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((pss_symbol, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))

    if channel_name is None:
        channel_impulse_response = None
    else:
        cir_bank = load_measured_cir(channel_name)
        channel_impulse_response = cir_bank.copy()

    rx_samples = apply_channel(
        tx_samples,
        SNR_DB,
        rng,
        channel_impulse_response=channel_impulse_response,
    )

    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)
    if rx_samples.ndim == 1:
        rx_samples = rx_samples[np.newaxis, :]

    bin_positions, template_bins, template_energy = make_pss_frequency_template()
    metric = compute_frequency_metric(
        rx_samples,
        bin_positions,
        template_bins,
        template_energy,
    )

    peak_index = int(np.argmax(metric))
    detected_cp_start = peak_index
    detected_symbol_start = peak_index + CYCLIC_PREFIX

    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    true_cp_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    timing_error = detected_cp_start - true_cp_start

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"

    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}', all RX)",
            cir_plot_path,
        )

    plt.figure(figsize=(10, 4))
    plt.plot(metric)
    plt.axvline(peak_index, color="tab:red", linestyle="--", label=f"Peak @ {peak_index}")
    plt.title(f"Frequency-domain PSS Metric ({channel_desc})")
    plt.xlabel("Candidate CP start index")
    plt.ylabel("Normalized metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    axes[0].axvline(true_cp_start, color="tab:green", linestyle="--", label="Expected CP start")
    axes[0].axvline(detected_cp_start, color="tab:red", linestyle=":", label="Detected CP start")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude with Start Detection (ZC FD, {channel_desc})")
    axes[0].legend(loc="upper right")

    axes[1].plot(metric, label="Normalized metric")
    axes[1].axvline(peak_index, color="tab:red", linestyle=":", label="Peak index")
    axes[1].axvline(true_cp_start, color="tab:green", linestyle="--", label="Expected CP start")
    axes[1].set_xlabel("Candidate CP start index")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Frequency-domain Detector Output")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_plot_path, dpi=150)
    plt.close(fig)

    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)

    pilot_cp_start = detected_symbol_start + N_FFT
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

    sto_title = f"Residual Timing From Phase Slope (ZC FD, {channel_desc})"
    slope_rad_per_bin, timing_offset_samples = plot_phase_slope_diagnostics(
        h_est,
        sto_plot_path,
        sto_title,
    )

    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    from core import align_complex_gain

    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(
        xhat_aligned,
        data_used,
        const_plot_path,
        f"Equalized Data Constellation (ZC FD, {channel_desc})",
    )

    print(f"\n{'='*70}")
    print(f"FREQUENCY-DOMAIN ZC SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
    print(f"{'='*70}")
    print(f"Frame length (without pad): {frame.size} samples")
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {rx_samples.shape[0]}")
    if channel_impulse_response is not None:
        print(
            f"Applied measured channel '{channel_name}' with taps={channel_impulse_response.shape[1]} "
            f"main-path offset={channel_peak_offset}",
        )
    else:
        print("Channel profile: Flat AWGN (no multipath)")
    print(f"\nTiming Detection:")
    print(f"  Detected CP start sample: {detected_cp_start}")
    print(f"  Expected CP start sample: {true_cp_start}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.2f}% of symbol)")
    print(f"\nCarrier Frequency Offset:")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"\nChannel Estimation & Equalization:")
    print(f"  Pilot LS phase slope: {slope_rad_per_bin:.6f} rad/bin -> timing ≈ {timing_offset_samples:.2f} samples")
    print(f"  Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")
    print(f"\nPlots saved to {plots_dir.resolve()}/:")
    print(f"  - correlation.png")
    print(f"  - start_detection.png")
    print(f"  - constellation.png")
    print(f"  - tx_frame_time.png")
    print(f"  - rx_frame_time.png")
    print(f"  - phase_slope_sto.png")
    if channel_impulse_response is not None:
        print(f"  - channel_cir.png")
    print(f"{'='*70}\n")


def main() -> None:
    print("\n" + "=" * 70)
    print("FREQUENCY-DOMAIN ZC SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
    print("=" * 70)

    run_simulation(channel_name="cir1", plots_subdir="measured_channel")
    run_simulation(channel_name=None, plots_subdir="flat_awgn")

    print("\n" + "=" * 70)
    print("ALL SIMULATIONS COMPLETE")
    print("=" * 70)
    print(f"\nCompare results in:")
    print(f"  - {(PLOTS_BASE_DIR / 'measured_channel').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'flat_awgn').resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

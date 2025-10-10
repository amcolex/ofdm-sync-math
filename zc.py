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
    remove_common_phase,
    evm_rms_db,
    plot_constellation,
    SAMPLE_RATE_HZ,
    apply_cfo,
    plot_phase_slope_diagnostics,
)

# ZC-specific parameters
PSS_LENGTH = 62
PSS_ROOT = 25


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


# Script-local parameters
SNR_DB = 10.0
CFO_HZ = 1000.0

# Base output directory
PLOTS_BASE_DIR = Path("plots") / "zc"


def run_simulation(channel_name: str | None, plots_subdir: str):
    """Run Zadoff-Chu preamble synchronization simulation.
    
    Args:
        channel_name: Name of measured channel profile (e.g., 'cir1') or None for AWGN-only
        plots_subdir: Subdirectory name for plots (e.g., 'measured_channel' or 'flat_awgn')
    """
    rng = np.random.default_rng(0)
    
    # Setup output directory
    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    corr_plot_path = plots_dir / "correlation.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    results_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    sto_plot_path = plots_dir / "phase_slope_sto.png"

    pss_waveform = build_pss_symbol(include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((pss_waveform, pilot_symbol, data_symbol))
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
    if rx_samples.ndim == 1:
        rx_samples = rx_samples[np.newaxis, :]

    num_branches = rx_samples.shape[0]
    pss_reference = build_pss_symbol(include_cp=False)
    pss_conj = np.conj(pss_reference[::-1])
    pss_energy = np.sum(np.abs(pss_reference) ** 2)
    reference_norm = np.sqrt(pss_energy)
    corr_eps = 1e-12
    window = np.ones(pss_reference.size, dtype=float)

    numerator_sum: np.ndarray | None = None
    power_sum: np.ndarray | None = None
    for branch in rx_samples:
        numerator = np.convolve(branch, pss_conj)
        branch_power = np.convolve(np.abs(branch) ** 2, window)
        if numerator_sum is None:
            numerator_sum = numerator
            power_sum = branch_power
        else:
            numerator_sum += numerator
            power_sum += branch_power
    assert numerator_sum is not None and power_sum is not None  # branches > 0
    denom = reference_norm * np.sqrt(np.maximum(power_sum, 0.0) + corr_eps)
    combined_corr = numerator_sum / denom

    correlation_mag = np.abs(combined_corr)
    peak_index = int(np.argmax(correlation_mag))
    detected_start = max(peak_index - pss_reference.size + 1, 0)

    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    pss_reference_lead = max(pss_waveform.size - pss_reference.size, 0)

    # Plot the raw channel impulse response if measured profile enabled
    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}', all RX)",
            cir_plot_path,
        )

    true_start = TX_PRE_PAD_SAMPLES + channel_peak_offset + pss_reference_lead
    expected_peak_index = true_start + pss_reference.size - 1
    timing_error = detected_start - true_start

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"
    
    plt.figure(figsize=(10, 4))
    plt.plot(correlation_mag)
    plt.axvline(peak_index, color="tab:red", linestyle="--", label=f"Peak @ {peak_index}")
    plt.title(f"Cross-correlation with ZC PSS Reference ({channel_desc})")
    plt.xlabel("Sample index")
    plt.ylabel("|normalized corr|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    if num_branches > 1:
        for idx, branch in enumerate(rx_samples):
            axes[0].plot(np.abs(branch), alpha=0.3, linewidth=0.8)
    axes[0].axvline(true_start, color="tab:green", linestyle="--", label="Expected ZC start")
    axes[0].axvline(detected_start, color="tab:red", linestyle=":", label="Detected ZC start")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude with Start Detection (ZC, {channel_desc})")
    axes[0].legend(loc="upper right")

    axes[1].plot(correlation_mag, label="Combined normalized |corr|")
    axes[1].axvline(peak_index, color="tab:red", linestyle=":", label="Peak index")
    axes[1].axvline(expected_peak_index, color="tab:green", linestyle="--", label="Expected peak")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("PSS Correlation Alignment")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_plot_path, dpi=150)
    plt.close(fig)

    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)

    # --- CFO estimation from pilot (CP correlation) ---
    # detected_start aligns to start of the N-length part of the preamble
    # Do not refine with correlation search; rely on ZC timing only
    pilot_cp_start = detected_start + N_FFT
    data_cp_start = pilot_cp_start + pilot_symbol.size
    cfo_est_hz = estimate_cfo_from_cp(rx_samples, pilot_cp_start, N_FFT, CYCLIC_PREFIX, SAMPLE_RATE_HZ)
    # Compensate CFO across entire stream
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)

    # --- Channel LS estimate from pilot ---
    # Combine branches by simple average after CFO correction
    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[pilot_cp_start + CYCLIC_PREFIX : pilot_cp_start + CYCLIC_PREFIX + N_FFT]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)
    # Estimate residual timing from linear phase slope of H(k)
    sto_title = f"Residual Timing From Phase Slope (ZC, {channel_desc})"
    slope_rad_per_bin, timing_offset_samples = plot_phase_slope_diagnostics(
        h_est,
        sto_plot_path,
        sto_title,
    )

    # --- Equalize data symbol and compute EVM ---
    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    # Align residual common phase and amplitude via complex LS gain
    from core import align_complex_gain
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(xhat_aligned, data_used, const_plot_path, f"Equalized Data Constellation (ZC, {channel_desc})")

    peak_error = peak_index - expected_peak_index
    print(f"\n{'='*70}")
    print(f"ZADOFF-CHU SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
    print(f"{'='*70}")
    print(f"Frame length (without pad): {frame.size} samples")
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {num_branches}")
    if channel_impulse_response is not None:
        print(
            f"Applied measured channel '{channel_name}' using {channel_impulse_response.shape[0]} RX branch(es) "
            f"taps={channel_impulse_response.shape[1]} main-path offset={channel_peak_offset}",
        )
    else:
        print("Channel profile: Flat AWGN (no multipath)")
    print(f"\nTiming Detection:")
    print(f"  Matched filter peak index: {peak_index}")
    print(f"  Expected peak index: {expected_peak_index}")
    print(f"  Detected ZC start sample: {detected_start}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.1f}% of symbol)")
    print(f"  Peak index error: {peak_error} samples")
    print(f"\nCarrier Frequency Offset:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz ({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.1f}%)")
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


def main():
    """Run simulations for both measured channel and flat AWGN conditions."""
    print("\n" + "="*70)
    print("ZADOFF-CHU SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
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

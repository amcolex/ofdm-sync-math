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

PLOTS_DIR = Path("plots") / "zc"
CORR_PLOT_PATH = PLOTS_DIR / "correlation.png"
TX_PLOT_PATH = PLOTS_DIR / "tx_frame_time.png"
RX_PLOT_PATH = PLOTS_DIR / "rx_frame_time.png"
RESULTS_PLOT_PATH = PLOTS_DIR / "start_detection.png"
CIR_PLOT_PATH = PLOTS_DIR / "channel_cir.png"
CONST_PLOT_PATH = PLOTS_DIR / "constellation.png"

CHANNEL_PROFILE = "cir1" # Set to None to bypass measured CIR
CHANNEL_RX_INDICES: tuple[int, ...] | None = (0, 1)


def run_simulation():
    rng = np.random.default_rng(0)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pss_waveform = build_pss_symbol(include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((pss_waveform, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))

    channel_impulse_response = None
    used_rx_indices: tuple[int, ...] = (0,)
    if CHANNEL_PROFILE:
        cir_bank = load_measured_cir(CHANNEL_PROFILE)
        available_channels = cir_bank.shape[0]
        if CHANNEL_RX_INDICES is None or len(CHANNEL_RX_INDICES) == 0:
            selected_indices = tuple(range(available_channels))
        else:
            selected_indices = tuple(CHANNEL_RX_INDICES)
        invalid = [idx for idx in selected_indices if idx < 0 or idx >= available_channels]
        if invalid:
            raise ValueError(
                f"Channel indices {invalid} out of range for profile '{CHANNEL_PROFILE}' (0..{available_channels - 1})",
            )
        used_rx_indices = selected_indices
        channel_impulse_response = cir_bank[np.array(selected_indices)]

    rx_samples = apply_channel(
        tx_samples,
        SNR_DB,
        rng,
        channel_impulse_response=channel_impulse_response,
    )
    # Apply CFO to simulate LO mismatch at the receiver
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

    num_branches = rx_samples.shape[0]
    pss_reference = build_pss_symbol(include_cp=False)
    pss_conj = np.conj(pss_reference[::-1])
    pss_energy = np.sum(np.abs(pss_reference) ** 2)
    reference_norm = np.sqrt(pss_energy)
    corr_eps = 1e-12
    window = np.ones(pss_reference.size, dtype=float)

    normalized_corr = []
    for branch in rx_samples:
        numerator = np.convolve(branch, pss_conj)
        branch_power = np.convolve(np.abs(branch) ** 2, window)
        denom = reference_norm * np.sqrt(branch_power + corr_eps)
        normalized = numerator / denom
        normalized_corr.append(normalized)
    normalized_corr = np.stack(normalized_corr)

    correlation_mag = np.sqrt(np.sum(np.abs(normalized_corr) ** 2, axis=0) / num_branches)
    peak_index = int(np.argmax(correlation_mag))
    detected_start = max(peak_index - pss_reference.size + 1, 0)

    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    pss_reference_lead = max(pss_waveform.size - pss_reference.size, 0)

    if channel_impulse_response is not None:
        # Plot the raw channel impulse response per branch
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{CHANNEL_PROFILE}', branches {used_rx_indices})",
            CIR_PLOT_PATH,
        )

    true_start = TX_PRE_PAD_SAMPLES + channel_peak_offset + pss_reference_lead
    expected_peak_index = true_start + pss_reference.size - 1
    timing_error = detected_start - true_start

    plt.figure(figsize=(10, 4))
    plt.plot(correlation_mag)
    plt.axvline(peak_index, color="tab:red", linestyle="--", label=f"Peak @ {peak_index}")
    plt.title("Cross-correlation with ZC PSS Reference")
    plt.xlabel("Sample index")
    plt.ylabel("|normalized corr|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CORR_PLOT_PATH, dpi=150)
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
    axes[0].set_title("Received Magnitude with Start Detection")
    axes[0].legend(loc="upper right")

    axes[1].plot(correlation_mag, label="Combined normalized |corr|")
    axes[1].axvline(peak_index, color="tab:red", linestyle=":", label="Peak index")
    axes[1].axvline(expected_peak_index, color="tab:green", linestyle="--", label="Expected peak")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("PSS Correlation Alignment")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(RESULTS_PLOT_PATH, dpi=150)
    plt.close(fig)

    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", TX_PLOT_PATH)
    plot_time_series(rx_samples, "Received Frame After Channel", RX_PLOT_PATH)

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

    # --- Equalize data symbol and compute EVM ---
    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    # Align residual common phase and amplitude via complex LS gain
    from core import align_complex_gain
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(xhat_aligned, data_used, CONST_PLOT_PATH, "Equalized Data Constellation (ZC)")

    print(f"Frame length (without pad): {frame.size} samples")
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {num_branches}")
    if channel_impulse_response is not None:
        print(
            f"Applied channel '{CHANNEL_PROFILE}' (branches {used_rx_indices}) "
            f"taps={channel_impulse_response.shape[1]} main-path offset={channel_peak_offset}",
        )
    else:
        print("Channel profile disabled (AWGN only)")
    peak_error = peak_index - expected_peak_index
    print(f"Matched filter peak index: {peak_index}")
    print(f"Expected peak index: {expected_peak_index}")
    print(f"Detected ZC start sample: {detected_start}")
    print(f"Timing error (detected - expected): {timing_error} samples")
    print(f"Peak index error (detected - expected): {peak_error} samples")
    print(f"Saved correlation plot to {CORR_PLOT_PATH.resolve()}")
    print(f"Saved transmit time series plot to {TX_PLOT_PATH.resolve()}")
    print(f"Saved receive time series plot to {RX_PLOT_PATH.resolve()}")
    print(f"Saved start detection plot to {RESULTS_PLOT_PATH.resolve()}")
    if channel_impulse_response is not None:
        print(f"Saved channel CIR plot to {CIR_PLOT_PATH.resolve()}")
    print(f"Applied CFO: {CFO_HZ} Hz at Fs={SAMPLE_RATE_HZ} Hz")
    print(f"Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")


def main():
    run_simulation()


if __name__ == "__main__":
    main()

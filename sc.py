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


def find_plateau_end_from_metric(M: np.ndarray, smooth_win: int, n_fft: int) -> int:
    if M.size == 0:
        return 0
    w = max(1, smooth_win)
    kernel = np.ones(w, dtype=float) / w
    Ms = np.convolve(M, kernel, mode="same")
    dM = np.diff(Ms)
    center = int(np.argmax(Ms))
    search_radius = n_fft // 4
    lo = max(1, center - search_radius)
    hi = min(dM.size - 1, center + search_radius)
    if hi <= lo:
        drop_idx = int(np.argmin(dM))
    else:
        local = dM[lo:hi]
        drop_idx = int(lo + np.argmin(local))
    plateau_end = drop_idx + 1
    return plateau_end


# Detector and channel parameters (script-local)
SNR_DB = 35.0
CFO_HZ = 1000.0
SC_DELTA = 12  # step back from plateau end to sit inside CP (8–16 suggested)
SMOOTH_WIN = 32  # samples for smoothing M(d) before slope detection

# Output paths
PLOTS_DIR = Path("plots") / "sc"
METRIC_PLOT_PATH = PLOTS_DIR / "sc_metric.png"
TX_PLOT_PATH = PLOTS_DIR / "tx_frame_time.png"
RX_PLOT_PATH = PLOTS_DIR / "rx_frame_time.png"
RESULTS_PLOT_PATH = PLOTS_DIR / "start_detection.png"
CIR_PLOT_PATH = PLOTS_DIR / "channel_cir.png"
CONST_PLOT_PATH = PLOTS_DIR / "constellation.png"

# Channel profile (set to 'cir1' or 'cir2' to enable measured CIR)
CHANNEL_PROFILE = None
CHANNEL_RX_INDICES: tuple[int, ...] | None = (0, 1)


def run_simulation():
    rng = np.random.default_rng(0)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build S&C preamble + block pilot (QPSK) + random QPSK data
    sc_preamble = build_sc_preamble(rng, include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((sc_preamble, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))

    # Optional channel
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

    # Detection metric (S&C)
    M, P_sum, R_sum = sc_streaming_metric(rx_samples)
    plateau_end = find_plateau_end_from_metric(M, SMOOTH_WIN, N_FFT)
    coarse_start = max(plateau_end - SC_DELTA, 0)

    # Ground-truth alignment helpers
    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    if channel_impulse_response is not None:
        # Plot raw CIR per branch
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{CHANNEL_PROFILE}', branches {used_rx_indices})",
            CIR_PLOT_PATH,
        )

    # The S&C plateau ideally spans the CP; its "end" aligns near CP end
    true_cp_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    expected_plateau_end = true_cp_start + CYCLIC_PREFIX

    # Plots
    plt.figure(figsize=(10, 4))
    plt.plot(M, label="S&C M(d)")
    plt.axvline(plateau_end, color="tab:red", linestyle=":", label="Plateau end")
    plt.axvline(expected_plateau_end, color="tab:green", linestyle="--", label="Expected end")
    plt.xlabel("Sample index d")
    plt.ylabel("M(d)")
    plt.title("Schmidl & Cox Streaming Metric")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(METRIC_PLOT_PATH, dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    combined_rx_mag = np.sqrt(np.sum(np.abs(rx_samples) ** 2, axis=0))
    axes[0].plot(combined_rx_mag, label="Combined |rx|")
    if rx_samples.ndim > 1 and rx_samples.shape[0] > 1:
        for idx, branch in enumerate(rx_samples):
            axes[0].plot(np.abs(branch), alpha=0.3, linewidth=0.8)
    axes[0].axvline(true_cp_start, color="tab:purple", linestyle="--", label="CP start (true)")
    axes[0].axvline(expected_plateau_end, color="tab:green", linestyle="--", label="Plateau end (exp)")
    axes[0].axvline(plateau_end, color="tab:red", linestyle=":", label="Plateau end (det)")
    axes[0].axvline(coarse_start, color="tab:orange", linestyle=":", label=f"Coarse start = end-{SC_DELTA}")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Received Magnitude and Detected Start (S&C)")
    axes[0].legend(loc="upper right")

    axes[1].plot(M, label="S&C M(d)")
    axes[1].axvline(plateau_end, color="tab:red", linestyle=":", label="Plateau end (det)")
    axes[1].axvline(expected_plateau_end, color="tab:green", linestyle="--", label="Plateau end (exp)")
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("M(d)")
    axes[1].set_title("Plateau-Based Timing (End minus delta)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(RESULTS_PLOT_PATH, dpi=150)
    plt.close(fig)

    # Raw time series
    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", TX_PLOT_PATH)
    plot_time_series(rx_samples, "Received Frame After Channel", RX_PLOT_PATH)

    # --- CFO estimation from pilot (CP correlation) ---
    # plateau_end approximates the end of CP and start of N for the preamble
    preamble_n_start_est = plateau_end
    pilot_cp_nom = preamble_n_start_est + N_FFT
    from core import find_cp_start_via_corr
    pilot_cp_start = find_cp_start_via_corr(
        rx_samples, pilot_cp_nom, N_FFT, CYCLIC_PREFIX, search_half=2 * CYCLIC_PREFIX
    )
    data_cp_start = pilot_cp_start + pilot_symbol.size
    cfo_est_hz = estimate_cfo_from_cp(rx_samples, pilot_cp_start, N_FFT, CYCLIC_PREFIX, SAMPLE_RATE_HZ)
    # Compensate CFO across entire stream
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)

    # --- Channel LS estimate from pilot ---
    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[pilot_cp_start + CYCLIC_PREFIX : pilot_cp_start + CYCLIC_PREFIX + N_FFT]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)

    # --- Equalize data symbol and compute EVM ---
    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    from core import align_complex_gain
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(xhat_aligned, data_used, CONST_PLOT_PATH, "Equalized Data Constellation (S&C)")

    # Prints
    print(f"Transmit sequence length: {tx_samples.size} samples")
    print(f"Receive branches: {1 if rx_samples.ndim == 1 else rx_samples.shape[0]}")
    if channel_impulse_response is not None:
        print(
            f"Applied channel '{CHANNEL_PROFILE}' (branches {used_rx_indices}) taps={channel_impulse_response.shape[1]} main-path offset={channel_peak_offset}",
        )
    else:
        print("Channel profile disabled (AWGN only)")
    print(f"Detected plateau end at d={plateau_end}")
    print(f"Coarse start (end - {SC_DELTA}) at d={coarse_start}")
    print(f"Expected plateau end at d={expected_plateau_end}")
    print(f"Saved S&C metric plot to {METRIC_PLOT_PATH.resolve()}")
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

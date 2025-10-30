import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
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
    """Build Minn preamble with 4-fold structure: A A -A -A."""
    all_idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    quarter_idx = all_idx[(all_idx % 4) == 0]
    bpsk = rng.choice([-1.0, 1.0], size=quarter_idx.shape[0])
    spectrum = allocate_subcarriers(N_FFT, quarter_idx, bpsk)
    symbol = np.fft.ifft(np.fft.ifftshift(spectrum))

    half = N_FFT // 2
    symbol[half:] = -symbol[half:]

    power = np.mean(np.abs(symbol) ** 2)
    if power > 0:
        symbol = symbol / np.sqrt(power)

    if include_cp:
        return add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol


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
    ax.set_title(f"LS-Derived CIR (Minn RTL, {channel_desc})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class _DelayLine:
    """Replicate the minn_delay_line behavior for scalar streams."""

    def __init__(self, depth: int):
        if depth < 0:
            raise ValueError("Delay depth must be non-negative.")
        self.depth = int(depth)
        self.mem = np.zeros(self.depth, dtype=np.float64) if self.depth > 0 else np.zeros(0)
        self.wr_ptr = 0
        self.fill = 0
        self.last_output = 0.0

    def step(self, sample: float, in_valid: bool) -> tuple[float, bool]:
        if self.depth == 0:
            if in_valid:
                self.last_output = float(sample)
            return float(sample), in_valid
        if not in_valid:
            return self.last_output, False
        if self.fill < self.depth:
            read_val = 0.0
        else:
            read_val = float(self.mem[self.wr_ptr])
        self.mem[self.wr_ptr] = float(sample)
        self.wr_ptr = (self.wr_ptr + 1) % self.depth
        if self.fill < self.depth:
            self.fill += 1
            self.last_output = 0.0
            return 0.0, False
        self.last_output = read_val
        return read_val, True


class _RunningSum:
    """Replicate the minn_running_sum behavior including the valid gating."""

    def __init__(self, depth: int):
        if depth < 0:
            raise ValueError("Running-sum depth must be non-negative.")
        self.depth = int(depth)
        self.mem = np.zeros(self.depth, dtype=np.float64) if self.depth > 0 else np.zeros(0)
        self.wr_ptr = 0
        self.fill = 0
        self.sum_reg = 0.0
        self.valid = False

    def step(self, sample: float, in_valid: bool) -> tuple[float, bool]:
        if self.depth == 0:
            if in_valid:
                self.sum_reg = float(sample)
                self.valid = True
            return self.sum_reg, self.valid
        if not in_valid:
            return self.sum_reg, self.valid
        if self.fill < self.depth:
            oldest = 0.0
        else:
            oldest = float(self.mem[self.wr_ptr])
        val = float(sample)
        self.mem[self.wr_ptr] = val
        self.wr_ptr = (self.wr_ptr + 1) % self.depth
        self.sum_reg = self.sum_reg + val - oldest
        if self.fill < self.depth:
            self.fill += 1
            if self.fill >= self.depth:
                self.valid = True
        else:
            self.valid = True
        return self.sum_reg, self.valid


def _antenna_path(samples: np.ndarray, quarter_len: int) -> dict[str, np.ndarray]:
    """Mirror minn_antenna_path.sv for one complex stream."""
    samples = np.asarray(samples, dtype=np.complex128)
    n = samples.size
    corr_recent = np.zeros(n, dtype=np.float64)
    corr_previous = np.zeros(n, dtype=np.float64)
    energy_recent = np.zeros(n, dtype=np.float64)
    energy_previous = np.zeros(n, dtype=np.float64)
    energy_previous2 = np.zeros(n, dtype=np.float64)
    taps_valid = np.zeros(n, dtype=bool)

    delay_i = _DelayLine(quarter_len)
    delay_q = _DelayLine(quarter_len)
    corr_window = _RunningSum(quarter_len)
    energy_window = _RunningSum(quarter_len)
    corr_delay = _DelayLine(quarter_len)
    energy_delay_q = _DelayLine(quarter_len)
    energy_delay_2q = _DelayLine(quarter_len)

    corr_recent_reg = 0.0
    corr_previous_reg = 0.0
    energy_recent_reg = 0.0
    energy_previous_reg = 0.0
    energy_previous2_reg = 0.0
    taps_valid_reg = False

    for idx in range(n):
        in_i = float(samples[idx].real)
        in_q = float(samples[idx].imag)

        delayed_i, _ = delay_i.step(in_i, True)
        delayed_q, _ = delay_q.step(in_q, True)

        quarter_product = delayed_i * in_i + delayed_q * in_q
        power_sum = in_i * in_i + in_q * in_q

        corr_sum, corr_valid = corr_window.step(quarter_product, True)
        energy_sum, energy_valid = energy_window.step(power_sum, True)

        corr_prev_val, corr_prev_valid = corr_delay.step(corr_sum, corr_valid)
        energy_q_val, energy_q_valid = energy_delay_q.step(energy_sum, energy_valid)
        energy_2q_val, energy_2q_valid = energy_delay_2q.step(energy_q_val, energy_q_valid)

        if corr_valid:
            corr_recent_reg = corr_sum
        if corr_prev_valid:
            corr_previous_reg = corr_prev_val
        if energy_valid:
            energy_recent_reg = energy_sum
        if energy_q_valid:
            energy_previous_reg = energy_q_val
        if energy_2q_valid:
            energy_previous2_reg = energy_2q_val
        taps_valid_reg = energy_2q_valid

        corr_recent[idx] = corr_recent_reg
        corr_previous[idx] = corr_previous_reg
        energy_recent[idx] = energy_recent_reg
        energy_previous[idx] = energy_previous_reg
        energy_previous2[idx] = energy_previous2_reg
        taps_valid[idx] = taps_valid_reg

    return {
        "corr_recent": corr_recent,
        "corr_previous": corr_previous,
        "energy_recent": energy_recent,
        "energy_previous": energy_previous,
        "energy_previous2": energy_previous2,
        "taps_valid": taps_valid,
    }


@dataclass
class MinnRTLMetricState:
    corr_total: np.ndarray
    corr_positive: np.ndarray
    smooth_metric: np.ndarray
    energy_total: np.ndarray
    corr_scaled: np.ndarray
    energy_scaled: np.ndarray
    metric_valid: np.ndarray
    above_threshold: np.ndarray


def minn_rtl_streaming_metric(
    rx: np.ndarray,
    *,
    smooth_shift: int,
    threshold_value: int,
    threshold_frac_bits: int,
) -> MinnRTLMetricState:
    """Compute the RTL-aligned Minn timing metric across all branches."""
    rx = np.asarray(rx, dtype=np.complex128)
    if rx.ndim == 1:
        rx = rx[np.newaxis, :]
    num_branches, length = rx.shape
    quarter_len = N_FFT // 4
    if quarter_len <= 0:
        raise ValueError("N_FFT must be divisible by 4.")

    branch_metrics = [_antenna_path(rx[b], quarter_len) for b in range(num_branches)]

    corr_total = np.zeros(length, dtype=np.float64)
    energy_total = np.zeros(length, dtype=np.float64)
    metric_valid = np.ones(length, dtype=bool)

    for metrics in branch_metrics:
        corr_total += metrics["corr_recent"] + metrics["corr_previous"]
        energy_total += (
            metrics["energy_recent"]
            + metrics["energy_previous"]
            + metrics["energy_previous2"]
        )
        metric_valid &= metrics["taps_valid"]

    corr_positive = np.maximum(corr_total, 0.0)

    smooth_metric = np.zeros(length, dtype=np.float64)
    smooth_val = 0.0
    denom = 1 << max(0, smooth_shift)
    for idx in range(length):
        if metric_valid[idx]:
            if smooth_shift == 0:
                smooth_val = corr_positive[idx]
            else:
                smooth_val += (corr_positive[idx] - smooth_val) / denom
        smooth_metric[idx] = smooth_val

    corr_scaled = smooth_metric * (1 << threshold_frac_bits)
    if threshold_value == 0:
        energy_scaled = np.zeros(length, dtype=np.float64)
    else:
        energy_scaled = energy_total * float(threshold_value)
    above_threshold = metric_valid & (corr_scaled >= energy_scaled)

    return MinnRTLMetricState(
        corr_total=corr_total,
        corr_positive=corr_positive,
        smooth_metric=smooth_metric,
        energy_total=energy_total,
        corr_scaled=corr_scaled,
        energy_scaled=energy_scaled,
        metric_valid=metric_valid,
        above_threshold=above_threshold,
    )


@dataclass
class MinnRTLDetection:
    detected_index: int | None
    peak_index: int | None
    gate_mask: np.ndarray
    gate_segments: list[tuple[int, int]]


def detect_minn_rtl(
    state: MinnRTLMetricState,
    *,
    hysteresis: int,
    timing_offset: int,
) -> MinnRTLDetection:
    """Replicate the gate and peak tracking used in the RTL detector."""
    corr = state.corr_positive
    above = state.above_threshold
    valid = state.metric_valid
    length = corr.size

    gate_segments: list[tuple[int, int]] = []
    gate_open = False
    gate_start: int | None = None
    peak_value = 0.0
    peak_index = 0
    low_counter = 0
    detected_index: int | None = None
    detected_peak: int | None = None
    hyst_limit = hysteresis - 1 if hysteresis > 0 else 0

    for idx in range(length):
        if not valid[idx]:
            continue
        metric_val = corr[idx]
        if not gate_open:
            if above[idx]:
                gate_open = True
                gate_start = idx
                peak_value = metric_val
                peak_index = idx
                low_counter = 0
        else:
            if metric_val >= peak_value:
                peak_value = metric_val
                peak_index = idx
            if above[idx]:
                low_counter = 0
            else:
                if hysteresis == 0:
                    if detected_index is None:
                        detected_index = peak_index + timing_offset
                        detected_peak = peak_index
                    gate_open = False
                    if gate_start is not None:
                        gate_segments.append((gate_start, idx + 1))
                        gate_start = None
                    peak_value = 0.0
                    low_counter = 0
                else:
                    if low_counter == hyst_limit:
                        if detected_index is None:
                            detected_index = peak_index + timing_offset
                            detected_peak = peak_index
                        gate_open = False
                        if gate_start is not None:
                            gate_segments.append((gate_start, idx + 1))
                            gate_start = None
                        peak_value = 0.0
                        low_counter = 0
                    else:
                        low_counter += 1

    if gate_open and gate_start is not None:
        gate_segments.append((gate_start, length))

    gate_mask = np.zeros(length, dtype=bool)
    for start, end in gate_segments:
        gate_mask[start:end] = True

    return MinnRTLDetection(
        detected_index=detected_index,
        peak_index=detected_peak,
        gate_mask=gate_mask,
        gate_segments=gate_segments,
    )


# Detector and channel parameters
SNR_DB = 30.0
CFO_HZ = 1000.0
SMOOTH_SHIFT = 3
THRESH_FRAC_BITS = 15
THRESH_VALUE = int(0.10 * (1 << THRESH_FRAC_BITS))
HYSTERESIS = 2
TIMING_OFFSET = 0

PLOTS_BASE_DIR = Path("plots") / "minn_rtl"


def run_simulation(channel_name: str | None, plots_subdir: str) -> None:
    """Run Minn RTL synchronization simulation with identical frame construction."""

    def mask_segments(mask: np.ndarray) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        in_segment = False
        start_idx = 0
        for idx, flag in enumerate(mask):
            if flag and not in_segment:
                in_segment = True
                start_idx = idx
            elif not flag and in_segment:
                in_segment = False
                segments.append((start_idx, idx))
        if in_segment:
            segments.append((start_idx, mask.size))
        return segments

    rng = np.random.default_rng(0)

    plots_dir = PLOTS_BASE_DIR / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    metric_plot_path = plots_dir / "minn_rtl_metric.png"
    tx_plot_path = plots_dir / "tx_frame_time.png"
    rx_plot_path = plots_dir / "rx_frame_time.png"
    results_plot_path = plots_dir / "start_detection.png"
    cir_plot_path = plots_dir / "channel_cir.png"
    ls_cir_plot_path = plots_dir / "ls_cir.png"
    const_plot_path = plots_dir / "constellation.png"
    sto_plot_path = plots_dir / "phase_slope_sto.png"

    minn_preamble = build_minn_preamble(rng, include_cp=True)
    pilot_symbol, pilot_used = build_random_qpsk_symbol(rng, include_cp=True)
    data_symbol, data_used = build_random_qpsk_symbol(rng, include_cp=True)
    frame = np.concatenate((minn_preamble, pilot_symbol, data_symbol))
    tx_samples = np.concatenate((np.zeros(TX_PRE_PAD_SAMPLES, dtype=complex), frame))

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
    rx_samples = apply_cfo(rx_samples, CFO_HZ, SAMPLE_RATE_HZ)

    metric_state = minn_rtl_streaming_metric(
        rx_samples,
        smooth_shift=SMOOTH_SHIFT,
        threshold_value=THRESH_VALUE,
        threshold_frac_bits=THRESH_FRAC_BITS,
    )
    detection = detect_minn_rtl(
        metric_state,
        hysteresis=HYSTERESIS,
        timing_offset=TIMING_OFFSET,
    )

    if detection.detected_index is not None:
        detected_start = int(detection.detected_index)
        peak_position = int(detection.peak_index if detection.peak_index is not None else detected_start)
    else:
        peak_position = int(np.argmax(metric_state.smooth_metric))
        detected_start = peak_position + TIMING_OFFSET

    gate_mask = detection.gate_mask
    gate_segments = detection.gate_segments or mask_segments(gate_mask)

    channel_peak_offset = compute_channel_peak_offset(channel_impulse_response)
    if channel_impulse_response is not None:
        plot_time_series(
            channel_impulse_response,
            f"Measured Channel CIR ('{channel_name}', all RX)",
            cir_plot_path,
        )

    true_cp_start = TX_PRE_PAD_SAMPLES + channel_peak_offset
    expected_n_start = true_cp_start + CYCLIC_PREFIX
    timing_error = detected_start - expected_n_start

    channel_desc = f"Measured CIR '{channel_name}'" if channel_name else "Flat AWGN"

    plt.figure(figsize=(10, 4))
    plt.plot(metric_state.corr_positive, label="RTL corr_plus(d)", color="tab:purple")
    plt.plot(metric_state.smooth_metric, label="RTL smooth(d)", color="tab:orange", linestyle="--")
    thresh_trace = np.full(metric_state.energy_total.shape, np.nan, dtype=float)
    denom = float(1 << THRESH_FRAC_BITS)
    valid_mask = metric_state.metric_valid
    if denom > 0:
        thresh_trace[valid_mask] = metric_state.energy_scaled[valid_mask] / denom
    plt.plot(thresh_trace, label="Threshold (scaled)", color="tab:green", linestyle=":")
    for idx, (start, end) in enumerate(gate_segments):
        gate_label = "Gate window" if idx == 0 else None
        plt.axvspan(start, end, color="tab:orange", alpha=0.15, label=gate_label)
    plt.axvline(peak_position, color="tab:red", linestyle=":", label=f"Peak @ {peak_position}")
    plt.axvline(expected_n_start, color="tab:green", linestyle="--", label="Expected N start")
    plt.xlabel("Sample index d")
    plt.ylabel("Metric")
    plt.title(f"Minn RTL Metric & Gate - {channel_desc}")
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
    for seg_idx, (start, end) in enumerate(gate_segments):
        gate_label = "Gate window" if seg_idx == 0 else None
        axes[0].axvspan(start, end, color="tab:orange", alpha=0.18, label=gate_label)
    axes[0].axvline(true_cp_start, color="tab:purple", linestyle="--", label="CP start (true)")
    axes[0].axvline(expected_n_start, color="tab:green", linestyle="--", label="N start (exp)")
    axes[0].axvline(detected_start, color="tab:red", linestyle=":", label="Detected start")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title(f"Received Magnitude and Detected Start (Minn RTL, {channel_desc})")
    axes[0].legend(loc="upper right")

    axes[1].plot(metric_state.corr_positive, label="RTL corr_plus(d)", color="tab:purple")
    axes[1].plot(
        metric_state.smooth_metric,
        label="RTL smooth(d)",
        color="tab:orange",
        linestyle="--",
    )
    axes[1].plot(thresh_trace, label="Threshold (scaled)", color="tab:green", linestyle=":")
    for start, end in gate_segments:
        axes[1].axvspan(start, end, color="tab:orange", alpha=0.12)
    axes[1].axvline(peak_position, color="tab:red", linestyle=":", label=f"Peak @ {peak_position}")
    axes[1].axvline(expected_n_start, color="tab:green", linestyle="--", label="Expected N start")
    axes[1].set_xlabel("Sample index d")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Timing Metrics (Minn RTL)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(results_plot_path, dpi=150)
    plt.close(fig)

    plot_time_series(tx_samples, "Transmit Frame (with Leading Zeros)", tx_plot_path)
    plot_time_series(rx_samples, f"Received Frame After Channel ({channel_desc})", rx_plot_path)

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
    rx_cfo_corr = apply_cfo(rx_samples, -cfo_est_hz, SAMPLE_RATE_HZ)

    rx_eff = rx_cfo_corr if rx_cfo_corr.ndim == 1 else np.mean(rx_cfo_corr, axis=0)
    pilot_td = rx_eff[pilot_cp_start + CYCLIC_PREFIX : pilot_cp_start + CYCLIC_PREFIX + N_FFT]
    y_pilot_used = ofdm_fft_used(pilot_td)
    h_est = ls_channel_estimate(y_pilot_used, pilot_used)
    sto_title = f"Residual Timing From Phase Slope (Minn RTL, {channel_desc})"
    slope_rad_per_bin, timing_offset_samples = plot_phase_slope_diagnostics(
        h_est,
        sto_plot_path,
        sto_title,
    )

    data_td = rx_eff[data_cp_start + CYCLIC_PREFIX : data_cp_start + CYCLIC_PREFIX + N_FFT]
    y_data_used = ofdm_fft_used(data_td)
    xhat = equalize(y_data_used, h_est)
    xhat_aligned, gain = align_complex_gain(xhat, data_used)
    evm_rms, evm_db = evm_rms_db(xhat_aligned, data_used)
    plot_constellation(
        xhat_aligned,
        data_used,
        const_plot_path,
        f"Equalized Data Constellation (Minn RTL, {channel_desc})",
    )

    ls_cir = _reconstruct_cir_from_ls(h_est)
    _plot_ls_cir(
        ls_cir,
        channel_impulse_response,
        channel_peak_offset,
        timing_error,
        ls_cir_plot_path,
        channel_desc,
    )

    print(f"\n{'='*70}")
    print(f"MINN RTL SYNCHRONIZATION RESULTS - {channel_desc.upper()}")
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
    print(f"  Detected peak at d={peak_position}")
    print(f"  Detected start after timing offset: d={detected_start}")
    print(f"  Expected N start at d={expected_n_start}")
    print(f"  Timing error: {timing_error} samples ({abs(timing_error)/N_FFT*100:.1f}% of symbol)")
    if gate_segments:
        gate_start, gate_end = gate_segments[0][0], gate_segments[-1][1]
        frac = THRESH_VALUE / float(1 << THRESH_FRAC_BITS)
        print(
            f"  RTL gate window: [{gate_start}, {gate_end}) "
            f"(threshold >={frac:.1%} of energy proxy, span {gate_end - gate_start} samples)",
        )
    else:
        print("  RTL gate not triggered (metric never exceeded threshold)")
    print(f"\nCarrier Frequency Offset:")
    print(f"  Applied CFO: {CFO_HZ} Hz")
    print(f"  Estimated CFO from CP: {cfo_est_hz:.2f} Hz")
    print(f"  CFO error: {abs(cfo_est_hz - CFO_HZ):.2f} Hz ({abs(cfo_est_hz - CFO_HZ)/CFO_HZ*100:.1f}%)")
    print(f"\nChannel Estimation & Equalization:")
    print(f"  Pilot LS phase slope: {slope_rad_per_bin:.6f} rad/bin -> timing ~ {timing_offset_samples:.2f} samples")
    print(f"  Post-EQ complex gain (mag, angle): {np.abs(gain):.3f}, {np.angle(gain):.3f} rad")
    print(f"  EVM RMS: {100*evm_rms:.2f}%  ({evm_db:.2f} dB)")
    print(f"\nPlots saved to {plots_dir.resolve()}/:")
    print(f"  - minn_rtl_metric.png")
    print(f"  - start_detection.png")
    print(f"  - constellation.png")
    print(f"  - tx_frame_time.png")
    print(f"  - rx_frame_time.png")
    print(f"  - ls_cir.png")
    print(f"  - phase_slope_sto.png")
    if channel_impulse_response is not None:
        print(f"  - channel_cir.png")
    print(f"{'='*70}\n")


def main() -> None:
    """Run simulations for both measured channel and flat AWGN conditions."""
    print("\n" + "=" * 70)
    print("MINN RTL PREAMBLE SYNCHRONIZATION - DUAL CONDITION ANALYSIS")
    print("=" * 70)

    run_simulation(channel_name="cir1", plots_subdir="measured_channel")
    run_simulation(channel_name=None, plots_subdir="flat_awgn")

    print("\n" + "=" * 70)
    print("ALL MINN RTL SIMULATIONS COMPLETE")
    print("=" * 70)
    print(f"\nCompare results in:")
    print(f"  - {(PLOTS_BASE_DIR / 'measured_channel').resolve()}")
    print(f"  - {(PLOTS_BASE_DIR / 'flat_awgn').resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

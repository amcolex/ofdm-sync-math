import os
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb_test.simulator import run

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ofdm import OFDMParameters, generate_preamble, generate_qpsk_symbol

VERILATOR = shutil.which("verilator")

INPUT_WIDTH = 12
NFFT = 2048
CP_LEN = 512
THRESH_FRAC_BITS = 15
THRESH_VALUE = int(0.1 * (1 << THRESH_FRAC_BITS))
SMOOTH_SHIFT = 3
HYSTERESIS = 2
TIMING_OFFSET = 0
OUTPUT_MARGIN = CP_LEN
LEADING_GUARD_LEN = 256
TRAILING_GUARD_LEN = NFFT + OUTPUT_MARGIN
AWGN_SNR_DB = 10.0


def _pack_axis_samples(ch0_i: int, ch0_q: int, ch1_i: int, ch1_q: int) -> int:
    mask = (1 << INPUT_WIDTH) - 1
    word = (ch0_i & mask)
    word |= (ch0_q & mask) << INPUT_WIDTH
    word |= (ch1_i & mask) << (2 * INPUT_WIDTH)
    word |= (ch1_q & mask) << (3 * INPUT_WIDTH)
    return word


def running_sum(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.zeros_like(values, dtype=np.float64)
    buffer = np.zeros(window, dtype=np.float64)
    accumulator = 0.0
    index = 0
    for idx, sample in enumerate(values):
        oldest = buffer[index]
        accumulator += sample - oldest
        buffer[index] = sample
        index = (index + 1) % window
        out[idx] = accumulator
    return out


def antenna_metrics(i_samples: np.ndarray, q_samples: np.ndarray, quarter_len: int) -> Tuple[np.ndarray, ...]:
    n = i_samples.size
    delayed_i = np.zeros(n, dtype=np.float64)
    delayed_q = np.zeros(n, dtype=np.float64)
    if quarter_len > 0:
        delayed_i[quarter_len:] = i_samples[:-quarter_len]
        delayed_q[quarter_len:] = q_samples[:-quarter_len]
    products = delayed_i * i_samples + delayed_q * q_samples
    power = i_samples * i_samples + q_samples * q_samples
    corr_recent = running_sum(products, quarter_len)
    energy_recent = running_sum(power, quarter_len)

    corr_prev = np.zeros(n, dtype=np.float64)
    energy_prev = np.zeros(n, dtype=np.float64)
    energy_prev2 = np.zeros(n, dtype=np.float64)
    if quarter_len > 0:
        corr_prev[quarter_len:] = corr_recent[:-quarter_len]
        energy_prev[quarter_len:] = energy_recent[:-quarter_len]
    if 2 * quarter_len <= n:
        energy_prev2[2 * quarter_len :] = energy_recent[:-2 * quarter_len]
    return corr_recent, corr_prev, energy_recent, energy_prev, energy_prev2


def minn_reference(
    ch0_i: np.ndarray,
    ch0_q: np.ndarray,
    ch1_i: np.ndarray,
    ch1_q: np.ndarray,
    *,
    nfft: int,
    threshold_value: int,
    threshold_frac_bits: int,
    smooth_shift: int,
    hysteresis: int,
    timing_offset: int,
) -> int:
    quarter_len = nfft // 4
    corr0, corr0_prev, e0, e0_prev, e0_prev2 = antenna_metrics(ch0_i, ch0_q, quarter_len)
    corr1, corr1_prev, e1, e1_prev, e1_prev2 = antenna_metrics(ch1_i, ch1_q, quarter_len)

    corr_total = (corr0 + corr0_prev) + (corr1 + corr1_prev)
    energy_total = (e0 + e0_prev + e0_prev2) + (e1 + e1_prev + e1_prev2)
    corr_positive = np.maximum(corr_total, 0.0)

    valid_start = max(0, 3 * quarter_len - 1)
    smooth_metric = 0.0
    gate_open = False
    peak_value = 0.0
    peak_index = 0
    low_counter = 0
    timing_adjust = timing_offset

    for idx in range(ch0_i.size):
        if idx < valid_start:
            continue
        metric = corr_positive[idx]
        if smooth_shift > 0:
            smooth_metric += (metric - smooth_metric) / (2 ** smooth_shift)
        else:
            smooth_metric = metric

        corr_scaled = smooth_metric * (2 ** threshold_frac_bits)
        energy_scaled = energy_total[idx] * threshold_value
        above_thresh = corr_scaled >= energy_scaled

        if not gate_open:
            if above_thresh:
                gate_open = True
                peak_value = smooth_metric
                peak_index = idx
                low_counter = 0
        else:
            if smooth_metric >= peak_value:
                peak_value = smooth_metric
                peak_index = idx
            if above_thresh:
                low_counter = 0
            else:
                if hysteresis == 0 or low_counter >= hysteresis:
                    return peak_index + timing_adjust
                low_counter += 1
    raise RuntimeError("Reference detector did not trigger")


def quantize_samples(samples: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray, float]:
    min_val = -(1 << (width - 1))
    max_val = (1 << (width - 1)) - 1
    max_mag = np.max(np.abs(samples))
    if max_mag == 0:
        scale = 1.0
    else:
        scale = (max_val - 1) / max_mag
    scaled = samples * scale
    real = np.clip(np.round(scaled.real), min_val, max_val).astype(np.int32)
    imag = np.clip(np.round(scaled.imag), min_val, max_val).astype(np.int32)
    return real, imag, scale


def add_awgn(samples: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    base = np.asarray(samples, dtype=np.complex128)
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(np.abs(base) ** 2)
    if signal_power <= 0.0:
        return base.copy()
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise_sigma = np.sqrt(noise_power / 2.0)
    noise = rng.normal(0.0, noise_sigma, base.shape) + 1j * rng.normal(0.0, noise_sigma, base.shape)
    return base + noise


@cocotb.test()
async def minn_detector_flags_expected_sample(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tdata.value = 0
    dut.s_axis_tlast.value = 0
    dut.m_axis_tready.value = 1

    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    params = OFDMParameters(n_fft=NFFT, cp_len=CP_LEN)
    preamble, _ = generate_preamble(params=params)
    data_symbol, _ = generate_qpsk_symbol(params=params)
    preamble_len = preamble.size
    data_len = data_symbol.size
    leading_guard = np.zeros(LEADING_GUARD_LEN, dtype=np.complex128)
    trailing_guard = np.zeros(TRAILING_GUARD_LEN, dtype=np.complex128)
    base_signal = np.concatenate((preamble, data_symbol))
    full_signal = np.concatenate((leading_guard, base_signal, trailing_guard))

    rng = np.random.default_rng(0)
    noisy_ch0 = add_awgn(full_signal, AWGN_SNR_DB, rng=rng)
    noisy_ch1 = add_awgn(full_signal, AWGN_SNR_DB, rng=rng)

    ch0_i, ch0_q, _ = quantize_samples(noisy_ch0, INPUT_WIDTH)
    ch1_i, ch1_q, _ = quantize_samples(noisy_ch1, INPUT_WIDTH)

    quarter_len = NFFT // 4
    (
        ch0_corr_recent,
        ch0_corr_prev,
        ch0_energy_recent,
        ch0_energy_prev,
        ch0_energy_prev2,
    ) = antenna_metrics(ch0_i.astype(np.float64), ch0_q.astype(np.float64), quarter_len)
    (
        ch1_corr_recent,
        ch1_corr_prev,
        ch1_energy_recent,
        ch1_energy_prev,
        ch1_energy_prev2,
    ) = antenna_metrics(ch1_i.astype(np.float64), ch1_q.astype(np.float64), quarter_len)

    corr_total = (ch0_corr_recent + ch0_corr_prev) + (ch1_corr_recent + ch1_corr_prev)
    energy_total = (ch0_energy_recent + ch0_energy_prev + ch0_energy_prev2) + (
        ch1_energy_recent + ch1_energy_prev + ch1_energy_prev2
    )
    corr_positive = np.maximum(corr_total, 0.0)
    valid_start = max(0, (3 * quarter_len) - 1)
    smooth_metric_trace = np.zeros_like(corr_positive)
    smooth_value = 0.0
    for idx, metric in enumerate(corr_positive):
        if idx >= valid_start:
            if SMOOTH_SHIFT == 0:
                smooth_value = metric
            else:
                smooth_value += (metric - smooth_value) / (2 ** SMOOTH_SHIFT)
        smooth_metric_trace[idx] = smooth_value

    metric_scaled_reference = smooth_metric_trace * float(1 << THRESH_FRAC_BITS)
    energy_scaled_trace = energy_total * float(THRESH_VALUE)

    expected_flag_index = minn_reference(
        ch0_i.astype(np.float64),
        ch0_q.astype(np.float64),
        ch1_i.astype(np.float64),
        ch1_q.astype(np.float64),
        nfft=NFFT,
        threshold_value=THRESH_VALUE,
        threshold_frac_bits=THRESH_FRAC_BITS,
        smooth_shift=SMOOTH_SHIFT,
        hysteresis=HYSTERESIS,
        timing_offset=TIMING_OFFSET,
    )

    total_samples: Iterable[Tuple[int, int, int, int]] = list(zip(ch0_i, ch0_q, ch1_i, ch1_q))

    metric_dbg_available = hasattr(dut, "metric_dbg")

    out_index = 0
    flagged_indices = []
    output_indices = []
    frame_start_trace = []
    input_ch0_i_trace = []
    input_ch0_q_trace = []
    metric_dbg_trace = []

    for ch0_i_val, ch0_q_val, ch1_i_val, ch1_q_val in total_samples:
        ch0_i_int = int(ch0_i_val)
        ch0_q_int = int(ch0_q_val)
        ch1_i_int = int(ch1_i_val)
        ch1_q_int = int(ch1_q_val)
        input_ch0_i_trace.append(ch0_i_int)
        input_ch0_q_trace.append(ch0_q_int)

        dut.s_axis_tdata.value = _pack_axis_samples(ch0_i_int, ch0_q_int, ch1_i_int, ch1_q_int)
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tlast.value = 0

        while True:
            await RisingEdge(dut.clk)
            if dut.m_axis_tvalid.value.integer:
                frame_start_int = dut.frame_start.value.integer
                output_indices.append(out_index)
                frame_start_trace.append(frame_start_int)
                if frame_start_int:
                    flagged_indices.append(out_index)
                out_index += 1
            if dut.s_axis_tvalid.value.integer and dut.s_axis_tready.value.integer:
                break

        dut.s_axis_tvalid.value = 0
        dut.s_axis_tlast.value = 0
        dut.s_axis_tdata.value = 0

        if metric_dbg_available:
            metric_dbg_trace.append(int(dut.metric_dbg.value))
        else:
            metric_dbg_trace.append(0)

    for _ in range(5):
        await RisingEdge(dut.clk)
        if dut.m_axis_tvalid.value.integer:
            frame_start_int = dut.frame_start.value.integer
            output_indices.append(out_index)
            frame_start_trace.append(frame_start_int)
            if frame_start_int:
                flagged_indices.append(out_index)
            out_index += 1

    observed_index = flagged_indices[0] if flagged_indices else None

    plot_dir = Path(__file__).resolve().parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "minn_preamble_detector.png"

    # Persist diagnostic plot with separate signal and metric subplots if available.
    if metric_dbg_available:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax_signal, ax_metric = axes
    else:
        fig, ax_signal = plt.subplots(figsize=(10, 4))
        ax_metric = None

    # Incoming stimulus captured while driving the DUT.
    driven_indices = np.arange(len(input_ch0_i_trace))
    ch0_i_samples = np.asarray(input_ch0_i_trace, dtype=np.int32)
    ch0_q_samples = np.asarray(input_ch0_q_trace, dtype=np.int32)
    ax_signal.plot(driven_indices, ch0_i_samples, label="Ch0 I")
    ax_signal.plot(driven_indices, ch0_q_samples, label="Ch0 Q")
    ax_signal.set_title("Quantized Input Signal w/ Detector Flag")
    ax_signal.set_ylabel("Amplitude (LSBs)")
    ax_signal.set_xlabel("Sample")
    total_len = len(input_ch0_i_trace)
    preamble_start = LEADING_GUARD_LEN
    preamble_end = preamble_start + preamble_len
    data_start = preamble_end
    data_end = data_start + data_len
    if LEADING_GUARD_LEN > 0:
        ax_signal.axvspan(
            0,
            LEADING_GUARD_LEN,
            color="0.9",
            alpha=0.25,
            label="Leading guard",
        )
    if TRAILING_GUARD_LEN > 0 and total_len >= TRAILING_GUARD_LEN:
        ax_signal.axvspan(
            total_len - TRAILING_GUARD_LEN,
            total_len,
            color="0.85",
            alpha=0.25,
            label="Trailing guard",
        )
    ax_signal.axvspan(
        preamble_start,
        preamble_end,
        color="tab:blue",
        alpha=0.12,
        label="Preamble window",
    )
    ax_signal.axvspan(
        data_start,
        data_end,
        color="tab:green",
        alpha=0.08,
        label="Data symbol",
    )
    ax_flag = ax_signal.twinx()
    if output_indices:
        ax_flag.step(
            output_indices,
            frame_start_trace,
            where="post",
            label="frame_start",
            color="tab:red",
        )
    ax_flag.set_ylabel("frame_start")
    ax_flag.set_ylim(-0.1, 1.1)
    ax_flag.axvline(
        expected_flag_index,
        color="tab:green",
        linestyle="--",
        label="Expected trigger",
    )
    if observed_index is not None:
        ax_flag.axvline(
            observed_index,
            color="tab:purple",
            linestyle=":",
            label="Observed trigger",
        )
    if ax_metric is not None:
        metric_dbg_samples = np.asarray(metric_dbg_trace, dtype=np.int64)
        scale_factor = float(1 << THRESH_FRAC_BITS)
        if metric_dbg_samples.size:
            metric_dbg_scaled = metric_dbg_samples.astype(np.float64) * scale_factor
            ax_metric.plot(
                driven_indices,
                metric_dbg_scaled,
                color="tab:orange",
                linewidth=1.0,
                label=f"metric_dbg × 2^{THRESH_FRAC_BITS}",
            )
        ax_metric.plot(
            driven_indices,
            metric_scaled_reference,
            color="tab:blue",
            linestyle="--",
            linewidth=1.0,
            label="Smooth metric × 2^{THRESH_FRAC_BITS}",
        )
        ax_metric.plot(
            driven_indices,
            energy_scaled_trace,
            color="tab:green",
            linestyle=":",
            linewidth=1.0,
            label="Energy × THRESH_VALUE",
        )
        ax_metric.set_ylabel("Scaled metric (fixed-point units)")
        ax_metric.set_xlabel("Sample")
        ax_metric.set_ylim(bottom=0.0)
    else:
        ax_signal.set_xlabel("Sample")

    handles, labels = ax_signal.get_legend_handles_labels()
    flag_handles, flag_labels = ax_flag.get_legend_handles_labels()
    if flag_handles:
        handles += flag_handles
        labels += flag_labels
    if handles:
        ax_signal.legend(handles, labels, loc="upper right")
    if ax_metric is not None:
        metric_handles, metric_labels = ax_metric.get_legend_handles_labels()
        if metric_handles:
            ax_metric.legend(metric_handles, metric_labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    dut._log.info(f"Saved Minn preamble detector plot to {plot_path}")

    assert flagged_indices, "Detector did not assert frame_start"
    observed_index = flagged_indices[0]
    assert abs(observed_index - expected_flag_index) <= 16, (
        f"Expected first frame_start near index {expected_flag_index}, "
        f"observed {flagged_indices}"
    )


@pytest.mark.skipif(VERILATOR is None, reason="Verilator executable not found")
def test_minn_preamble_detector():
    rtl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rtl"))
    sources = [
        os.path.join(rtl_dir, "minn_delay_line.sv"),
        os.path.join(rtl_dir, "minn_running_sum.sv"),
        os.path.join(rtl_dir, "minn_antenna_path.sv"),
        os.path.join(rtl_dir, "minn_preamble_detector.sv"),
    ]
    build_dir = os.path.join("tests", "sim_build", "minn_preamble_detector")

    run(
        verilog_sources=sources,
        toplevel="minn_preamble_detector",
        toplevel_lang="verilog",
        module=os.path.splitext(os.path.basename(__file__))[0],
        parameters={
            "INPUT_WIDTH": INPUT_WIDTH,
            "NFFT": NFFT,
            "CP_LEN": CP_LEN,
            "OUTPUT_MARGIN": OUTPUT_MARGIN,
            "THRESH_VALUE": THRESH_VALUE,
            "THRESH_FRAC_BITS": THRESH_FRAC_BITS,
            "SMOOTH_SHIFT": SMOOTH_SHIFT,
            "HYSTERESIS": HYSTERESIS,
            "TIMING_OFFSET": TIMING_OFFSET,
        },
        sim_build=build_dir,
        simulator="verilator",
        extra_env={
            "COCOTB_RESULTS_FILE": os.path.join(build_dir, "results.xml"),
        },
        defines={
            "MINN_METRIC_DEBUG": 1,
        },
    )

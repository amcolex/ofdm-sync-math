import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Centralized system defaults
N_FFT = 2048
NUM_ACTIVE_SUBCARRIERS = 1200
CYCLIC_PREFIX = 512
TX_PRE_PAD_SAMPLES = 1337
SAMPLE_RATE_HZ = 30_720_000.0  # default 30.72 MHz (adjust as needed)


def centered_subcarrier_indices(width: int) -> np.ndarray:
    """Return subcarrier indices symmetric around DC while skipping 0."""
    half = width // 2
    negative = np.arange(-half, 0)
    positive = np.arange(1, half + 1)
    return np.concatenate((negative, positive))


def allocate_subcarriers(n_fft: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place the provided subcarrier values into a centered spectrum array."""
    if indices.shape[0] != values.shape[0]:
        msg = "Subcarrier index and value arrays must have the same length."
        raise ValueError(msg)

    spectrum = np.zeros(n_fft, dtype=complex)
    dc_index = n_fft // 2
    placement = (dc_index + indices) % n_fft
    spectrum[placement] = values
    return spectrum


def spectrum_to_time_domain(spectrum: np.ndarray) -> np.ndarray:
    """Convert a centered spectrum to a unit-power time-domain waveform."""
    time_domain = np.fft.ifft(np.fft.ifftshift(spectrum))
    power = np.mean(np.abs(time_domain) ** 2)
    if power == 0:
        return time_domain
    return time_domain / np.sqrt(power)


def add_cyclic_prefix(symbol: np.ndarray, cp_length: int) -> np.ndarray:
    """Prepend the cyclic prefix to the OFDM symbol."""
    if cp_length <= 0:
        return symbol
    return np.concatenate((symbol[-cp_length:], symbol))


def build_random_bpsk_symbol(rng: np.random.Generator, include_cp: bool = True) -> np.ndarray:
    indices = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    bits = rng.choice([-1.0, 1.0], size=indices.shape[0])
    spectrum = allocate_subcarriers(N_FFT, indices, bits)
    symbol = spectrum_to_time_domain(spectrum)
    if include_cp:
        return add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol


def plot_time_series(samples: np.ndarray, title: str, path: Path) -> None:
    """Save real/imag/magnitude views of the time-domain waveform."""
    samples = np.asarray(samples)

    if samples.ndim == 1:
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(samples.real)
        axes[0].set_ylabel("Re")

        axes[1].plot(samples.imag)
        axes[1].set_ylabel("Im")

        axes[2].plot(np.abs(samples))
        axes[2].set_ylabel("|x|")
        axes[2].set_xlabel("Sample index")

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    if samples.ndim != 2:  # defensive
        raise ValueError("Expected 1D or 2D array for plotting")

    num_channels = samples.shape[0]
    fig, axes = plt.subplots(
        num_channels,
        3,
        figsize=(10, 2.5 * num_channels),
        sharex=True,
    )
    if num_channels == 1:
        axes = axes[np.newaxis, :]

    for idx in range(num_channels):
        ch = samples[idx]
        axes[idx, 0].plot(ch.real)
        axes[idx, 0].set_ylabel(f"Re ch{idx}")

        axes[idx, 1].plot(ch.imag)
        axes[idx, 1].set_ylabel(f"Im ch{idx}")

        axes[idx, 2].plot(np.abs(ch))
        axes[idx, 2].set_ylabel(f"|ch{idx}|")
        axes[idx, 2].set_xlabel("Sample index")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compute_channel_peak_offset(channel_impulse_response: np.ndarray | None) -> int:
    """Return the strongest-path index from a possibly multi-branch CIR array."""
    if channel_impulse_response is None:
        return 0
    aggregate_cir = np.sum(np.abs(channel_impulse_response) ** 2, axis=0)
    if np.any(aggregate_cir):
        return int(np.argmax(aggregate_cir))
    return 0


def apply_cfo(samples: np.ndarray, cfo_hz: float, fs_hz: float) -> np.ndarray:
    """Apply a carrier frequency offset to 1D or 2D samples.

    - If 2D, treats axis 0 as branches and applies the same tone to all.
    """
    x = np.asarray(samples)
    if x.ndim == 1:
        n = np.arange(x.size, dtype=float)
        tone = np.exp(1j * 2 * np.pi * cfo_hz * n / fs_hz)
        return x * tone
    if x.ndim != 2:
        raise ValueError("samples must be 1D or 2D")
    L = x.shape[1]
    n = np.arange(L, dtype=float)
    tone = np.exp(1j * 2 * np.pi * cfo_hz * n / fs_hz)
    return x * tone[np.newaxis, :]


# -----------------------------
# Extra OFDM helpers for pilots
# -----------------------------

def _qpsk_values(rng: np.random.Generator, size: int) -> np.ndarray:
    m = rng.integers(0, 4, size=size)
    re = (m & 1) * 2 - 1  # 0->-1, 1->+1 for LSB
    im = ((m >> 1) & 1) * 2 - 1
    vals = (re + 1j * im) / np.sqrt(2.0)
    return vals.astype(np.complex128)


def build_random_qpsk_symbol(
    rng: np.random.Generator, include_cp: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_domain_symbol, used_subcarrier_values) for a full-band QPSK OFDM symbol.

    - Active subcarriers are the centered set of width NUM_ACTIVE_SUBCARRIERS (DC skipped).
    - QPSK constellation with unit average power on used tones.
    - Time-domain symbol is unit-power normalized.
    """
    indices = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    qpsk_vals = _qpsk_values(rng, indices.shape[0])
    spectrum = allocate_subcarriers(N_FFT, indices, qpsk_vals)
    symbol = spectrum_to_time_domain(spectrum)
    if include_cp:
        symbol = add_cyclic_prefix(symbol, CYCLIC_PREFIX)
    return symbol, qpsk_vals


def ofdm_fft_used(symbol_time_no_cp: np.ndarray) -> np.ndarray:
    """FFT an OFDM symbol (no CP) and return only used, centered subcarriers."""
    spectrum_full = np.fft.fftshift(np.fft.fft(symbol_time_no_cp, n=N_FFT))
    dc = N_FFT // 2
    idx = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)
    return spectrum_full[(dc + idx) % N_FFT]


def estimate_cfo_from_cp(
    rx: np.ndarray, start: int, n_fft: int, cp_len: int, fs_hz: float
) -> float:
    """Estimate CFO (Hz) from CP correlation on a symbol whose CP starts at `start`.

    Uses P = sum r[start+n] r*[start+n+N] over CP length.
    Supports 1D (single branch) or 2D (branches x time) arrays by summing across branches.
    """
    x = np.asarray(rx)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    a = x[:, start : start + cp_len]
    b = x[:, start + n_fft : start + n_fft + cp_len]
    P = np.sum(a * np.conj(b))
    angle = np.angle(P)
    # angle(P) ≈ -2π * f_cfo * N / Fs  => f_cfo ≈ -angle * Fs / (2π N)
    cfo_hz = -angle * fs_hz / (2 * np.pi * n_fft)
    return float(cfo_hz)


def estimate_cfo_from_cp_robust(
    rx: np.ndarray,
    cp_start_est: int,
    n_fft: int,
    cp_len: int,
    fs_hz: float,
    span: int | None = None,
    win_len: int | None = None,
) -> float:
    """Robust CFO estimate by aggregating CP correlations around an estimated start.

    - Sums P(d) over d in [cp_start_est - span, cp_start_est + span] using a
      shorter window length `win_len` to tolerate misalignment.
    - Returns CFO in Hz with the same sign convention as `estimate_cfo_from_cp`.
    """
    x = np.asarray(rx)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    L = x.shape[1]
    span = cp_len // 2 if span is None else int(max(0, span))
    win = cp_len // 2 if win_len is None else int(max(1, win_len))
    d_lo = max(0, cp_start_est - span)
    d_hi = min(L - (n_fft + win), cp_start_est + span)
    if d_hi <= d_lo:
        return estimate_cfo_from_cp(x, cp_start_est, n_fft, min(cp_len, win), fs_hz)
    P_acc = 0.0 + 0.0j
    for d in range(d_lo, d_hi):
        a = x[:, d : d + win]
        b = x[:, d + n_fft : d + n_fft + win]
        P_acc += np.sum(a * np.conj(b))
    angle = np.angle(P_acc)
    cfo_hz = -angle * fs_hz / (2 * np.pi * n_fft)
    return float(cfo_hz)


def estimate_cfo_from_cp_peak(
    rx: np.ndarray,
    cp_start_est: int,
    n_fft: int,
    cp_len: int,
    fs_hz: float,
    span: int | None = None,
) -> float:
    """Pick the CP offset with maximum |P(d)| near the estimated CP start and use its phase.

    This does not change frame timing — it only chooses the best CP correlation
    offset for CFO estimation within a small local window.
    """
    x = np.asarray(rx)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    L = x.shape[1]
    span = cp_len // 2 if span is None else int(max(0, span))
    d_lo = max(0, cp_start_est - span)
    d_hi = min(L - (n_fft + cp_len), cp_start_est + span)
    if d_hi <= d_lo:
        return estimate_cfo_from_cp(x, cp_start_est, n_fft, cp_len, fs_hz)
    best_P = 0.0 + 0.0j
    best_mag = -1.0
    for d in range(d_lo, d_hi):
        a = x[:, d : d + cp_len]
        b = x[:, d + n_fft : d + n_fft + cp_len]
        P = np.sum(a * np.conj(b))
        mag = float(np.abs(P))
        if mag > best_mag:
            best_mag = mag
            best_P = P
    angle = np.angle(best_P)
    cfo_hz = -angle * fs_hz / (2 * np.pi * n_fft)
    return float(cfo_hz)


def estimate_cfo_from_cp_peak_with_index(
    rx: np.ndarray,
    cp_start_est: int,
    n_fft: int,
    cp_len: int,
    fs_hz: float,
    span: int | None = None,
) -> tuple[float, int]:
    """Like estimate_cfo_from_cp_peak, but also return the best CP offset index used."""
    x = np.asarray(rx)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    L = x.shape[1]
    span = cp_len // 2 if span is None else int(max(0, span))
    d_lo = max(0, cp_start_est - span)
    d_hi = min(L - (n_fft + cp_len), cp_start_est + span)
    if d_hi <= d_lo:
        return estimate_cfo_from_cp(x, cp_start_est, n_fft, cp_len, fs_hz), cp_start_est
    best_P = 0.0 + 0.0j
    best_mag = -1.0
    best_d = d_lo
    for d in range(d_lo, d_hi):
        a = x[:, d : d + cp_len]
        b = x[:, d + n_fft : d + n_fft + cp_len]
        P = np.sum(a * np.conj(b))
        mag = float(np.abs(P))
        if mag > best_mag:
            best_mag = mag
            best_P = P
            best_d = d
    angle = np.angle(best_P)
    cfo_hz = -angle * fs_hz / (2 * np.pi * n_fft)
    return float(cfo_hz), int(best_d)


def find_cp_start_via_corr(
    rx: np.ndarray,
    est_start: int,
    n_fft: int,
    cp_len: int,
    search_half: int = 1024,
) -> int:
    """Refine CP start using the magnitude of CP correlation |P(d)|.

    Searches over d in [est_start - search_half, est_start + search_half] and
    returns the d that maximizes |sum r[d+n] r*[d+n+n_fft]|.
    """
    x = np.asarray(rx)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    L = x.shape[1]
    lo = max(0, est_start - search_half)
    hi = min(L - (n_fft + cp_len), est_start + search_half)
    if hi <= lo:
        return est_start
    best_d = lo
    best_val = -1.0
    for d in range(lo, hi):
        a = x[:, d : d + cp_len]
        b = x[:, d + n_fft : d + n_fft + cp_len]
        P = np.sum(a * np.conj(b))
        val = float(np.abs(P))
        if val > best_val:
            best_val = val
            best_d = d
    return best_d


def ls_channel_estimate(y_used: np.ndarray, x_used: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Least-squares per-subcarrier channel estimate H = Y/X with small regularization."""
    return y_used / (x_used + eps)


def equalize(y_used: np.ndarray, h_est: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return y_used / (h_est + eps)


def remove_common_phase(x: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """De-rotate by common phase error. If `ref` given, align x to ref; else use mean angle of x."""
    if ref is None:
        cpe = np.angle(np.mean(x))
    else:
        cpe = np.angle(np.vdot(ref, x) / (np.vdot(ref, ref) + 1e-12))
    return x * np.exp(-1j * cpe), float(cpe)


def align_complex_gain(x: np.ndarray, ref: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, complex]:
    """Scale x by complex gain g to best fit ref in LS sense: minimize ||g x - ref||^2."""
    num = np.vdot(x, ref)
    den = np.vdot(x, x) + eps
    g = num / den
    return x * g, g


def evm_rms_db(x: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Return (evm_rms, evm_db) where EVM is normalized to reference RMS magnitude."""
    err = x - ref
    evm_rms = np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(ref) ** 2))
    evm_db = 20 * np.log10(evm_rms + 1e-12)
    return float(evm_rms), float(evm_db)


def plot_constellation(x: np.ndarray, ref: np.ndarray | None, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x.real, x.imag, s=6, alpha=0.6, label="Equalized")
    if ref is not None:
        ax.scatter(ref.real, ref.imag, s=36, alpha=0.8, marker="x", label="Ideal")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------
# Timing offset from phase slope across subcarriers
# -----------------------------------------------------
def estimate_timing_offset_from_phase_slope(
    h_used: np.ndarray,
) -> tuple[float, float]:
    """Estimate residual timing from linear phase slope of LS channel.

    A time shift by Δ samples yields a linear phase across subcarriers:
    angle ~ -2π k Δ / N_FFT. We fit a line to unwrapped phase vs. used
    centered subcarrier index k (excluding DC). Returns:
      - slope_rad_per_bin: fitted slope in radians per subcarrier index
      - timing_offset_samples: Δ ≈ -slope * N_FFT / (2π)

    Assumes `h_used` corresponds to subcarriers returned by
    `centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS)` in the same order
    as `ofdm_fft_used`.
    """
    if h_used.size == 0:
        return 0.0, 0.0
    k = centered_subcarrier_indices(NUM_ACTIVE_SUBCARRIERS).astype(np.float64)
    phi = np.unwrap(np.angle(h_used.astype(np.complex128)))
    k_mean = float(np.mean(k))
    phi_mean = float(np.mean(phi))
    k_zero = k - k_mean
    phi_zero = phi - phi_mean
    denom = float(np.sum(k_zero * k_zero)) + 1e-12
    slope = float(np.sum(k_zero * phi_zero) / denom)  # rad per subcarrier
    delta_samples = -slope * N_FFT / (2.0 * np.pi)
    return slope, float(delta_samples)

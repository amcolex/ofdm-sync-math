import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Centralized system defaults
N_FFT = 2048
NUM_ACTIVE_SUBCARRIERS = 1200
CYCLIC_PREFIX = 512
SNR_DB = 35.0
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

from functools import lru_cache
from pathlib import Path

import numpy as np

_BASE_DIR = Path(__file__).resolve().parent

_CHANNEL_DIR = _BASE_DIR / "channel_models"
_CHANNEL_MAP = {
    "cir1": _CHANNEL_DIR / "cir1.csv",
    "cir2": _CHANNEL_DIR / "cir2.csv",
}


@lru_cache(maxsize=None)
def load_measured_cir(name: str) -> np.ndarray:
    """Load all receive-channel CIRs for the requested profile."""
    try:
        path = _CHANNEL_MAP[name]
    except KeyError as exc:  # pragma: no cover - defensive, easy to hit in future
        raise ValueError(f"Unknown channel profile '{name}'") from exc

    if not path.exists():  # pragma: no cover - file-system guard
        raise FileNotFoundError(f"CIR data '{path}' not found")

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    num_channels = (data.shape[1] - 1) // 2
    cir_list: list[np.ndarray] = []
    for chan in range(num_channels):
        real_idx = 1 + 2 * chan
        imag_idx = real_idx + 1
        real = data[:, real_idx]
        imag = data[:, imag_idx]
        mask = np.isfinite(real) & np.isfinite(imag)
        cir = real[mask] + 1j * imag[mask]
        cir_list.append(cir.astype(np.complex128))

    if not cir_list:
        raise ValueError(f"Profile '{name}' contains no CIR taps")

    max_len = max(cir.shape[0] for cir in cir_list)
    padded = np.zeros((len(cir_list), max_len), dtype=np.complex128)
    for idx, cir in enumerate(cir_list):
        padded[idx, : cir.shape[0]] = cir
    return padded


def _compute_awgn_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Return complex AWGN matching the requested SNR."""
    signal = np.asarray(signal)
    snr_linear = 10 ** (snr_db / 10)

    if signal.ndim == 1:
        signal_power = np.mean(np.abs(signal) ** 2)
        if signal_power == 0:
            return np.zeros_like(signal)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (
            rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
        )
        return noise

    if signal.ndim != 2:  # pragma: no cover - defensive branch
        raise ValueError("Signal must be 1D or 2D array")

    signal_power = np.mean(np.abs(signal) ** 2, axis=1, keepdims=True)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)
    noise = noise_std * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )
    noise[signal_power.squeeze(axis=1) == 0] = 0
    return noise


def apply_channel(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    channel_impulse_response: np.ndarray | None = None,
) -> np.ndarray:
    """Apply an optional CIR followed by complex AWGN to the input signal."""
    signal = np.asarray(signal)

    if channel_impulse_response is None:
        faded = signal[np.newaxis, :]
    else:
        cir = np.asarray(channel_impulse_response)
        if cir.ndim == 1:
            cir = cir[np.newaxis, :]
        faded = np.stack([np.convolve(signal, taps, mode="full") for taps in cir])

    noise = _compute_awgn_noise(faded, snr_db, rng)
    return faded + noise

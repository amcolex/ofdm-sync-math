from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Default system parameters keep the module easy to use out of the box.
N_FFT = 2048
NUM_ACTIVE_SUBCARRIERS = 1200
CYCLIC_PREFIX = 512
SAMPLE_RATE_HZ = 30_720_000.0  # 30.72 MHz default sampling rate


@dataclass(frozen=True)
class OFDMParameters:
    """Container for core OFDM dimensions."""

    n_fft: int = N_FFT
    num_active: int = NUM_ACTIVE_SUBCARRIERS
    cp_len: int = CYCLIC_PREFIX

    def __post_init__(self) -> None:
        if self.n_fft % 2:
            raise ValueError("FFT size must be even.")
        if self.num_active % 2:
            raise ValueError("Active subcarrier count must be even to skip DC.")
        if self.num_active >= self.n_fft:
            raise ValueError("Active subcarrier count must be smaller than FFT size.")
        if self.cp_len < 0:
            raise ValueError("Cyclic prefix length must be non-negative.")


def centered_subcarrier_indices(num_active: int, *, spacing: int = 1) -> np.ndarray:
    """Return symmetric subcarrier indices around DC while skipping the 0th bin."""
    if num_active % 2:
        raise ValueError("num_active must be even so the DC bin can be skipped cleanly.")
    if spacing <= 0:
        raise ValueError("spacing must be a positive integer.")
    half = num_active // 2
    negative = np.arange(-half, 0, dtype=int)
    positive = np.arange(1, half + 1, dtype=int)
    indices = np.concatenate((negative, positive))
    if spacing != 1:
        indices = indices * spacing
    return indices


def allocate_subcarriers(n_fft: int, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Place active subcarrier values into an FFT-sized spectrum."""
    if indices.shape[0] != values.shape[0]:
        raise ValueError("indices and values must be the same length.")
    spectrum = np.zeros(n_fft, dtype=np.complex128)
    dc_index = n_fft // 2
    placement = (dc_index + indices) % n_fft
    spectrum[placement] = values
    return spectrum


def spectrum_to_time_domain(
    spectrum: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Convert a centered spectrum into the time domain."""
    time_domain = np.fft.ifft(np.fft.ifftshift(spectrum))
    if normalize:
        power = np.mean(np.abs(time_domain) ** 2)
        if power > 0:
            time_domain = time_domain / np.sqrt(power)
    return time_domain.astype(np.complex128, copy=False)


def add_cyclic_prefix(symbol: np.ndarray, cp_len: int) -> np.ndarray:
    """Prepend the last cp_len samples as the cyclic prefix."""
    if cp_len <= 0:
        return np.asarray(symbol, dtype=np.complex128)
    symbol = np.asarray(symbol, dtype=np.complex128)
    return np.concatenate((symbol[-cp_len:], symbol))


def remove_cyclic_prefix(symbol: np.ndarray, cp_len: int) -> np.ndarray:
    """Drop the cyclic prefix from a symbol."""
    if cp_len <= 0:
        return np.asarray(symbol, dtype=np.complex128)
    return np.asarray(symbol, dtype=np.complex128)[cp_len:]


def generate_ofdm_symbol(
    subcarrier_values: np.ndarray,
    *,
    params: OFDMParameters | None = None,
    include_cp: bool = True,
    normalize: bool = True,
    spacing: int = 1,
) -> np.ndarray:
    """Build a single OFDM symbol from active subcarrier values."""
    params = OFDMParameters() if params is None else params
    values = np.asarray(subcarrier_values, dtype=np.complex128)
    indices = centered_subcarrier_indices(values.size, spacing=spacing)
    if values.size and np.max(np.abs(indices)) >= params.n_fft // 2:
        raise ValueError("Active subcarriers exceed the available FFT bins.")
    spectrum = allocate_subcarriers(params.n_fft, indices, values)
    symbol = spectrum_to_time_domain(spectrum, normalize=normalize)
    if include_cp:
        symbol = add_cyclic_prefix(symbol, params.cp_len)
    return symbol


def random_qpsk_values(
    num_active: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return unit-power QPSK symbols for the requested number of subcarriers."""
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.integers(0, 4, size=num_active)
    real = np.where(raw & 1, 1.0, -1.0)
    imag = np.where(raw & 2, 1.0, -1.0)
    return (real + 1j * imag) / np.sqrt(2.0)


def generate_qpsk_symbol(
    *,
    params: OFDMParameters | None = None,
    rng: np.random.Generator | None = None,
    include_cp: bool = True,
    normalize: bool = True,
    spacing: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random QPSK OFDM symbol and return (time_domain, subcarrier_values)."""
    params = OFDMParameters() if params is None else params
    if params.num_active % spacing:
        raise ValueError("params.num_active must be divisible by the spacing value.")
    values = random_qpsk_values(params.num_active // spacing, rng=rng)
    symbol = generate_ofdm_symbol(
        values,
        params=params,
        include_cp=include_cp,
        normalize=normalize,
        spacing=spacing,
    )
    return symbol, values


def generate_preamble(
    *,
    params: OFDMParameters | None = None,
    include_cp: bool = True,
    normalize: bool = True,
    subcarrier_values: np.ndarray | None = None,
    subcarrier_value: complex | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a preamble symbol shaped as [A A -A -A] in the time domain.

    The implementation activates every fourth subcarrier from the standard
    allocation, then flips the sign of the second half of the time-domain symbol
    to create the requested pattern. Returns (time_domain_symbol,
    active_subcarrier_values) where the time-domain vector already includes the
    cyclic prefix if `include_cp` is True.
    """
    params = OFDMParameters() if params is None else params
    if params.n_fft % 4:
        raise ValueError("Preamble generation requires an FFT length divisible by 4.")
    all_indices = centered_subcarrier_indices(params.num_active)
    quarter_indices = all_indices[(all_indices % 4) == 0]
    if quarter_indices.size == 0:
        raise ValueError("Not enough active subcarriers to build a quarter-band preamble.")

    if subcarrier_values is None:
        if subcarrier_value is not None:
            values = np.full(quarter_indices.size, subcarrier_value, dtype=np.complex128)
            # Enforce Hermitian symmetry to keep the time-domain preamble real when possible.
            pos_mask = quarter_indices > 0
            values[quarter_indices < 0] = np.conj(values[pos_mask][::-1])
        else:
            rng = np.random.default_rng(0) if rng is None else rng
            pos_mask = quarter_indices > 0
            pos_values = rng.choice([-1.0, 1.0], size=pos_mask.sum()).astype(np.complex128)
            values = np.zeros(quarter_indices.size, dtype=np.complex128)
            values[pos_mask] = pos_values
            values[~pos_mask] = np.conj(pos_values[::-1])
    else:
        values = np.asarray(subcarrier_values, dtype=np.complex128)
        if values.shape[0] != quarter_indices.size:
            raise ValueError(
                f"Expected {quarter_indices.size} subcarrier values, got {values.shape[0]} instead."
            )

    spectrum = allocate_subcarriers(params.n_fft, quarter_indices, values)
    base_symbol = spectrum_to_time_domain(spectrum, normalize=normalize)

    # Flip the sign on the second half of the time-domain symbol.
    preamble = base_symbol.copy()
    preamble[params.n_fft // 2 :] *= -1.0  # results in [A A -A -A]

    if include_cp:
        preamble = add_cyclic_prefix(preamble, params.cp_len)
    return preamble, values


def generate_frame(
    data_subcarriers: Sequence[np.ndarray],
    *,
    params: OFDMParameters | None = None,
    include_cp: bool = True,
    normalize: bool = True,
    add_preamble: bool = True,
    preamble: np.ndarray | None = None,
    spacing: int = 1,
) -> np.ndarray:
    """Assemble a frame from optional preamble plus a list of data symbols."""
    params = OFDMParameters() if params is None else params
    if params.num_active % spacing:
        raise ValueError("params.num_active must be divisible by the chosen spacing.")

    symbols: list[np.ndarray] = []
    if add_preamble:
        if preamble is None:
            preamble, _ = generate_preamble(
                params=params,
                include_cp=include_cp,
                normalize=normalize,
            )
        symbols.append(np.asarray(preamble, dtype=np.complex128))

    for block in data_subcarriers:
        block = np.asarray(block, dtype=np.complex128)
        expected = params.num_active // spacing
        if block.size != expected:
            raise ValueError(f"Each data block must contain {expected} subcarriers.")
        symbol = generate_ofdm_symbol(
            block,
            params=params,
            include_cp=include_cp,
            normalize=normalize,
            spacing=spacing,
        )
        symbols.append(symbol)

    if not symbols:
        return np.array([], dtype=np.complex128)
    return np.concatenate(symbols)


def _plot_frame(frame: np.ndarray, *, title: str | None = None, output: str | None = None) -> None:
    """Render the provided complex baseband frame as I/Q and magnitude."""
    import matplotlib.pyplot as plt

    indices = np.arange(frame.size)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(indices, frame.real, label="In-phase (I)")
    axes[0].plot(indices, frame.imag, label="Quadrature (Q)")
    axes[0].set_ylabel("Amplitude (arb)")
    axes[0].set_title(title or "OFDM Frame Time Series")
    axes[0].legend(loc="upper right")

    axes[1].plot(indices, np.abs(frame), label="Magnitude", color="tab:purple")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("|IQ|")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    if output:
        plt.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def _build_demo_frame(
    *,
    params: OFDMParameters,
    include_cp: bool,
    normalize: bool,
) -> np.ndarray:
    preamble, _ = generate_preamble(params=params, include_cp=include_cp, normalize=normalize)
    data_symbol, _ = generate_qpsk_symbol(
        params=params, include_cp=include_cp, normalize=normalize
    )
    return np.concatenate((preamble, data_symbol))


def _main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Plot a generated OFDM frame.")
    parser.add_argument(
        "--nfft",
        type=int,
        default=N_FFT,
        help="FFT length used for frame generation (default: %(default)s)",
    )
    parser.add_argument(
        "--cp",
        type=int,
        default=CYCLIC_PREFIX,
        help="Cyclic prefix length (default: %(default)s)",
    )
    parser.add_argument(
        "--no-cp",
        dest="include_cp",
        action="store_false",
        help="Exclude the cyclic prefix from generated symbols.",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable power normalization for generated symbols.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write PNG output instead of showing an interactive window.",
    )
    args = parser.parse_args()

    params = OFDMParameters(n_fft=args.nfft, cp_len=args.cp)
    frame = _build_demo_frame(params=params, include_cp=args.include_cp, normalize=args.normalize)

    title = (
        f"OFDM Frame (nfft={params.n_fft}, cp_len={params.cp_len}, normalize={args.normalize})"
    )
    output_path = Path(args.output).expanduser() if args.output else None
    _plot_frame(frame, title=title, output=str(output_path) if output_path else None)


if __name__ == "__main__":
    _main()

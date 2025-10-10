
import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 512             # Number of subcarriers (FFT size)
CP = 128            # Cyclic prefix length (1/4 of N, same ratio as the earlier example)
NUM_SYMBOLS = 2     # Transmit two OFDM symbols back-to-back
EARLY_SAMPLES = 16  # Start FFT window 16 samples early (still inside CP)
LATE_SAMPLES = 16   # Start FFT window 16 samples late (into CP of the second symbol)
SNR_dB = 30         # Additive white Gaussian noise level at the receiver
np.random.seed(7)   # Reproducibility

# QPSK mapping: ±1/√2 ± j±1/√2
bits_i = np.random.randint(0, 2, (NUM_SYMBOLS, N))
bits_q = np.random.randint(0, 2, (NUM_SYMBOLS, N))
qpsk_symbols = (2*bits_i - 1) + 1j*(2*bits_q - 1)
qpsk_symbols = qpsk_symbols / np.sqrt(2)

# IFFT to create the OFDM symbols (numpy's ifft includes the 1/N factor)
time_domain_symbols = np.fft.ifft(qpsk_symbols, n=N, axis=1)

# Add cyclic prefixes and serialize the stream
cp = time_domain_symbols[:, -CP:]
tx_symbols = np.concatenate([cp, time_domain_symbols], axis=1)
tx = tx_symbols.reshape(NUM_SYMBOLS * (N + CP))

# Add complex AWGN to observe robustness of STO estimation
signal_power = np.mean(np.abs(tx)**2)
snr_linear = 10**(SNR_dB / 10)
noise_variance = signal_power / snr_linear
noise = np.sqrt(noise_variance / 2) * (np.random.randn(tx.size) + 1j * np.random.randn(tx.size))
rx_stream = tx + noise

# --- Receiver windows ---
symbol_starts = np.arange(NUM_SYMBOLS) * (N + CP)
fft_starts = symbol_starts + CP

# Perfect alignment for symbol 0 and symbol 1
rx_sym0 = rx_stream[fft_starts[0]:fft_starts[0]+N]
rx_sym1 = rx_stream[fft_starts[1]:fft_starts[1]+N]
S_sym0 = np.fft.fft(rx_sym0, n=N)
S_sym1 = np.fft.fft(rx_sym1, n=N)

# Early FFT window for symbol 0 (still within CP, so only phase rotation)
early_start = fft_starts[0] - EARLY_SAMPLES
rx_early = rx_stream[early_start:early_start+N]
S_early = np.fft.fft(rx_early, n=N)

# Late FFT window for symbol 0 (runs into CP of symbol 1)
late_start = fft_starts[0] + LATE_SAMPLES
rx_late = rx_stream[late_start:late_start+N]
S_late = np.fft.fft(rx_late, n=N)

# Plot constellation diagrams
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

axes[0].scatter(S_sym0.real, S_sym0.imag, s=12)
axes[0].axhline(0, linewidth=0.5)
axes[0].axvline(0, linewidth=0.5)
axes[0].set_aspect('equal', 'box')
axes[0].set_title(f'Symbol 0 — Perfect Alignment (N={N}, CP={CP})')
axes[0].set_xlabel('In-phase')
axes[0].set_ylabel('Quadrature')
axes[0].grid(True)

axes[1].scatter(S_early.real, S_early.imag, s=12)
axes[1].axhline(0, linewidth=0.5)
axes[1].axvline(0, linewidth=0.5)
axes[1].set_aspect('equal', 'box')
axes[1].set_title(f'Symbol 0 — FFT Starts {EARLY_SAMPLES} Samples Early')
axes[1].set_xlabel('In-phase')
axes[1].grid(True)

axes[2].scatter(S_late.real, S_late.imag, s=12)
axes[2].axhline(0, linewidth=0.5)
axes[2].axvline(0, linewidth=0.5)
axes[2].set_aspect('equal', 'box')
axes[2].set_title(f'Symbol 0 — FFT Starts {LATE_SAMPLES} Samples Late')
axes[2].set_xlabel('In-phase')
axes[2].grid(True)

axes[3].scatter(S_sym1.real, S_sym1.imag, s=12)
axes[3].axhline(0, linewidth=0.5)
axes[3].axvline(0, linewidth=0.5)
axes[3].set_aspect('equal', 'box')
axes[3].set_title('Symbol 1 — Perfect Alignment')
axes[3].set_xlabel('In-phase')
axes[3].grid(True)

plt.tight_layout()
plt.show()

# --- Phase slope analysis to estimate sample timing offset (STO) ---
k = np.arange(N)
ratio_early = S_early / S_sym0
ratio_late = S_late / S_sym0
phase_early = np.unwrap(np.angle(ratio_early))
phase_late = np.unwrap(np.angle(ratio_late))

slope_early, intercept_early = np.polyfit(k, phase_early, 1)
slope_late, intercept_late = np.polyfit(k, phase_late, 1)
sto_est_early = -slope_early * N / (2 * np.pi)
sto_est_late = -slope_late * N / (2 * np.pi)

fig2, ax_phase = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax_phase[0].plot(k, phase_early, '.', markersize=4, label='Measured phase')
ax_phase[0].plot(k, slope_early * k + intercept_early, 'r-', label='Linear fit')
ax_phase[0].set_ylabel('Phase [rad]')
ax_phase[0].set_title(f'Phase Slope — FFT {EARLY_SAMPLES} Samples Early (STO ≈ {sto_est_early:.2f})')
ax_phase[0].grid(True)
ax_phase[0].legend()

ax_phase[1].plot(k, phase_late, '.', markersize=4, label='Measured phase')
ax_phase[1].plot(k, slope_late * k + intercept_late, 'r-', label='Linear fit')
ax_phase[1].set_xlabel('Subcarrier index k')
ax_phase[1].set_ylabel('Phase [rad]')
ax_phase[1].set_title(f'Phase Slope — FFT {LATE_SAMPLES} Samples Late (STO ≈ {sto_est_late:.2f})')
ax_phase[1].grid(True)
ax_phase[1].legend()

plt.tight_layout()
plt.show()

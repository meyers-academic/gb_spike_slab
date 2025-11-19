"""
Quick Start: Minimal Working Example

This script shows the absolute minimum code needed to:
1. Generate noise
2. Create waveforms
3. Inject signals
"""

import jax
import jax.numpy as jnp
import numpy as np
from gb_spike_slab import NoiseGenerator, WaveformGenerator, SignalInjector

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Setup
T_OBS = 365 * 24 * 3600  # 1 year
DELTA_T = 5.0
N_SAMPLES = 128

print("Quick Start: Gravitational Wave Signal Injection")
print("=" * 60)

# Step 1: Generate noise
print("\n1. Generating noise...")
noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=42)
noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
freqs = noise_gen.get_frequency_grid()
print(f"   Created noise with {len(freqs)} frequency bins")

# Step 2: Define source parameters
print("\n2. Defining sources...")
# Format: [f0, fdot, amplitude, ecliptic_lat, ecliptic_lon, polarization, inclination, initial_phase]
params = jnp.array([
    [0.00135962, 8.94581279e-19, 1.07345e-22, 0.312414, -2.75291, 3.5621656, 0.523599, 3.0581565],
    [0.00136062, 8.5e-19, 0.8e-22, 0.35, -2.34, 3.5, 0.6, 2.8],
])
print(f"   Number of sources: {params.shape[0]}")
print(f"   Frequencies: {params[:, 0]}")

# Step 3: Generate waveforms
print("\n3. Generating waveforms...")
wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params)
wf_freqs = wf_gen.get_waveform_frequencies(params)
print(f"   Waveform shape: {A_wf.shape}")

# Step 4: Inject signals
print("\n4. Injecting signals into noise...")
injector = SignalInjector(noise_freqs=freqs, t_obs=T_OBS)
data_A, data_E, data_T = injector.inject_signals(
    noise_A, noise_E, noise_T,
    A_wf, E_wf, T_wf,
    wf_freqs
)
print(f"   Created injected data with shape: {data_A.shape}")

# Step 5: Compute SNRs
print("\n5. Computing SNRs...")
from gb_spike_slab.utils import compute_snr
psd_A, psd_E, psd_T = noise_gen.get_psds()

# Get individual waveforms on full grid
_, interpolated_wfs = injector.inject_signals(
    noise_A, noise_E, noise_T,
    A_wf, E_wf, T_wf,
    wf_freqs,
    return_individual=True
)

for i in range(params.shape[0]):
    signal_A = interpolated_wfs['A'][i]
    signal_E = interpolated_wfs['E'][i]
    snr_A = compute_snr(signal_A, psd_A, T_OBS)
    snr_E = compute_snr(signal_E, psd_E, T_OBS)
    snr_combined = np.sqrt(snr_A**2 + snr_E**2)
    print(f"   Source {i+1}: SNR(A)={snr_A:.1f}, SNR(E)={snr_E:.1f}, Combined={snr_combined:.1f}")

print("\n" + "=" * 60)
print("Done! Data is ready for analysis.")
print("\nNext steps:")
print("  - Use 'data_A', 'data_E', 'data_T' for parameter estimation")
print("  - See example_usage.py for visualization")
print("  - See README.md for complete documentation")

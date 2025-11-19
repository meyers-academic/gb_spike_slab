"""
Example: End-to-End Gravitational Wave Signal Injection

This script demonstrates:
1. Generating fake noise for LISA TDI channels
2. Creating vectorized waveforms for multiple sources
3. Interpolating waveforms to match noise frequencies
4. Injecting signals into noise
5. Visualizing results
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gb_spike_slab import NoiseGenerator, WaveformGenerator, SignalInjector, FrequencyGrid

# Configure JAX for 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)


def main():
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Observation parameters
    T_OBS = 30 * 24 * 3600  # 1 month in seconds
    DELTA_T = 5.0  # Sampling interval in seconds
    N_SAMPLES = 128  # Number of frequency samples per waveform
    SEED = 42  # Random seed for reproducibility
    
    print("=" * 70)
    print("Gravitational Wave Signal Injection Example")
    print("=" * 70)
    print(f"Observation time: {T_OBS / (365*24*3600):.1f} years")
    print(f"Sampling interval: {DELTA_T} seconds")
    print(f"Frequency resolution: {1/T_OBS:.2e} Hz")
    print()
    
    # =========================================================================
    # Step 1: Generate Fake Noise
    # =========================================================================
    
    print("Step 1: Generating fake noise...")
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=SEED)
    
    # Generate noise for all channels
    noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
    freqs = noise_gen.get_frequency_grid()
    psd_A, psd_E, psd_T = noise_gen.get_psds()
    
    print(f"  - Generated noise with {len(freqs)} frequency bins")
    print(f"  - Frequency range: {freqs[1]:.2e} to {freqs[-1]:.2e} Hz")
    print()
    
    # =========================================================================
    # Step 2: Create Multiple Waveforms (Vectorized)
    # =========================================================================
    
    print("Step 2: Creating waveforms for multiple sources...")
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    
    # Define the three sources (same as example_usage.py)
    source_1 = jnp.array([
        0.0013596,       # f0 (Hz) - within 1 microhertz of others
        8.94581279e-19,  # fdot (Hz/s)
        1.07345e-22,     # amplitude
        0.312414,        # ecliptic latitude (rad)
        -2.75291,        # ecliptic longitude (rad)
        3.5621656,       # polarization (rad)
        0.523599,        # inclination (rad)
        3.0581565,       # initial phase (rad)
    ])
    
    source_2 = jnp.array([
        0.00136,         # f0 (center frequency)
        8.5e-19,         # fdot
        0.8e-22,         # amplitude (weaker)
        0.35,            # ecliptic latitude
        -2.34,           # ecliptic longitude
        3.5,             # polarization
        0.6,             # inclination
        2.8,             # initial phase
    ])
    
    source_3 = jnp.array([
        0.0013604,       # f0 (Hz) - within 1 microhertz of others
        9.2e-19,         # fdot
        1.2e-22,         # amplitude
        -0.4,            # ecliptic latitude
        1.5,             # ecliptic longitude
        2.1,             # polarization
        1.0,             # inclination
        0.5,             # initial phase
    ])
    
    params_true = jnp.vstack([source_1, source_2, source_3])
    n_sources = params_true.shape[0]
    
    print(f"  - Number of sources: {n_sources}")
    print(f"  - True frequencies: {params_true[:, 0]}")
    print()
    
    # Generate waveforms for all sources (vectorized)
    A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params_true, tdi_combination="AET")
    print(f"  - Generated waveforms with shape: {A_wf.shape}")
    
    # Get frequency grids for each waveform
    wf_freqs = wf_gen.get_waveform_frequencies(params_true)
    print(f"  - Waveform frequency grids shape: {wf_freqs.shape}")
    print()
    
    # =========================================================================
    # Step 2: Define Frequency Band and Crop Noise
    # =========================================================================
    
    print("Step 2: Defining frequency band and cropping noise...")
    
    # Center frequency (middle of the three sources)
    f_center = 0.00136  # Hz
    
    # Search range: ±1e-5 Hz around center
    search_width = 1e-5  # Hz (10 microhertz)
    f_min = f_center - search_width
    f_max = f_center + search_width
    
    print(f"  - Center frequency: {f_center:.8f} Hz")
    print(f"  - Search range: {f_min:.8f} to {f_max:.8f} Hz")
    print(f"  - Search width: ±{search_width:.2e} Hz")
    
    # Create injector with full frequency grid (needed for cropping)
    injector = SignalInjector(noise_freqs=freqs, t_obs=T_OBS)
    
    # Crop noise data to frequency band
    noise_A_cropped, noise_E_cropped, noise_T_cropped, freqs_cropped = injector.crop_data(
        noise_A, noise_E, noise_T, f_min, f_max
    )
    
    # Crop PSDs
    idx_low = np.argmin(np.abs(freqs - f_min))
    idx_high = np.argmin(np.abs(freqs - f_max))
    psd_A_cropped = psd_A[idx_low:idx_high]
    psd_E_cropped = psd_E[idx_low:idx_high]
    
    print(f"  - Cropped to {len(freqs_cropped)} frequency bins")
    print()
    
    # =========================================================================
    # Step 3: Interpolate Waveforms to Cropped Frequency Grid
    # =========================================================================
    
    print("Step 3: Interpolating waveforms to cropped frequency grid...")
    
    # Interpolate waveforms directly to cropped frequency grid
    interpolated_wfs = {}
    interpolated_wfs['A'] = wf_gen.interpolate_waveform(A_wf, wf_freqs, freqs_cropped)
    interpolated_wfs['E'] = wf_gen.interpolate_waveform(E_wf, wf_freqs, freqs_cropped)
    interpolated_wfs['T'] = wf_gen.interpolate_waveform(T_wf, wf_freqs, freqs_cropped)
    
    # Sum all sources
    total_signal_A = jnp.sum(interpolated_wfs['A'], axis=0)
    total_signal_E = jnp.sum(interpolated_wfs['E'], axis=0)
    total_signal_T = jnp.sum(interpolated_wfs['T'], axis=0)
    
    # Add to cropped noise
    data_A = np.array(noise_A_cropped) + np.array(total_signal_A)
    data_E = np.array(noise_E_cropped) + np.array(total_signal_E)
    data_T = np.array(noise_T_cropped) + np.array(total_signal_T)
    
    print(f"  - Interpolated waveforms to {len(freqs_cropped)} frequency bins")
    print(f"  - Created injected data (noise + signal)")
    print()
    
    # =========================================================================
    # Step 4: Visualize Results
    # =========================================================================
    
    print("Step 4: Visualizing results...")
    
    # Plot 1: Cropped frequency range showing noise and injected signals
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # A channel
    axes[0].loglog(freqs_cropped, np.abs(noise_A_cropped)**2 / T_OBS, 
                   c='C0', alpha=0.3, label='Noise realization')
    axes[0].loglog(freqs_cropped, np.abs(data_A)**2 / T_OBS, 
                   c='C0', linewidth=1.5, label='Data (noise + signal)')
    axes[0].loglog(freqs_cropped, psd_A_cropped, 'k--', linewidth=2, label='Expected PSD')
    axes[0].set_xlim(f_min, f_max)
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density (strain²/Hz)')
    axes[0].set_title('A Channel: Cropped Frequency Range')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # E channel
    axes[1].loglog(freqs_cropped, np.abs(noise_E_cropped)**2 / T_OBS, 
                   c='C1', alpha=0.3, label='Noise realization')
    axes[1].loglog(freqs_cropped, np.abs(data_E)**2 / T_OBS, 
                   c='C1', linewidth=1.5, label='Data (noise + signal)')
    axes[1].loglog(freqs_cropped, psd_E_cropped, 'k--', linewidth=2, label='Expected PSD')
    axes[1].set_xlim(f_min, f_max)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (strain²/Hz)')
    axes[1].set_title('E Channel: Cropped Frequency Range')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./overview.png', dpi=150, bbox_inches='tight')
    print("  - Saved: overview.png")
    
    # Plot 2: Zoom in around the signal frequencies
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for i, f0 in enumerate(params_true[:, 0]):
        # Just plot the cropped data and zoom with xlim
        # The data is already cropped to a narrow range
        axes[i].plot(freqs_cropped, np.abs(data_A), 
                    linewidth=2, label=f'Source {i+1} injected data', alpha=0.7)
        axes[i].axvline(f0, color='r', linestyle='--', alpha=0.5, linewidth=2,
                       label=f'True frequency: {f0:.6f} Hz')
        axes[i].set_xlim(f0 - 1e-5, f0 + 1e-5)  # Zoom to a narrow range around each source
        axes[i].set_xlabel('Frequency (Hz)')
        axes[i].set_ylabel('Amplitude')
        axes[i].set_title(f'Source {i+1}: Zoomed View (A Channel)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./zoomed_signals.png', dpi=150, bbox_inches='tight')
    print("  - Saved: zoomed_signals.png")
    
    # Plot 3: Individual interpolated waveforms
    fig, axes = plt.subplots(n_sources, 2, figsize=(14, 4*n_sources))
    
    for i in range(n_sources):
        # Real part
        axes[i, 0].plot(freqs_cropped, interpolated_wfs['A'][i].real, label='A channel')
        axes[i, 0].plot(freqs_cropped, interpolated_wfs['E'][i].real, label='E channel', alpha=0.7)
        axes[i, 0].axvline(params_true[i, 0], color='r', linestyle='--', alpha=0.5, label=f'f0 = {params_true[i, 0]:.6f} Hz')
        axes[i, 0].set_xlabel('Frequency (Hz)')
        axes[i, 0].set_ylabel('Real Part')
        axes[i, 0].set_title(f'Source {i+1}: Waveform Real Part')
        axes[i, 0].set_xlim(params_true[i, 0] - 5e-4, params_true[i, 0] + 5e-4)
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Imaginary part
        axes[i, 1].plot(freqs_cropped, interpolated_wfs['A'][i].imag, label='A channel')
        axes[i, 1].plot(freqs_cropped, interpolated_wfs['E'][i].imag, label='E channel', alpha=0.7)
        axes[i, 1].axvline(params_true[i, 0], color='r', linestyle='--', alpha=0.5, label=f'f0 = {params_true[i, 0]:.6f} Hz')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Imaginary Part')
        axes[i, 1].set_title(f'Source {i+1}: Waveform Imaginary Part')
        axes[i, 1].set_xlim(params_true[i, 0] - 5e-4, params_true[i, 0] + 5e-4)
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./interpolated_waveforms.png', dpi=150, bbox_inches='tight')
    print("  - Saved: interpolated_waveforms.png")
    
    plt.close('all')
    
    # =========================================================================
    # Step 5: Compute SNRs
    # =========================================================================
    
    print()
    print("Step 5: Computing signal-to-noise ratios...")
    
    from gb_spike_slab.utils import compute_snr
    
    for i in range(n_sources):
        # Compute SNR for each source using cropped data
        signal_A = interpolated_wfs['A'][i]
        snr_A = compute_snr(signal_A, psd_A_cropped, T_OBS)
        
        signal_E = interpolated_wfs['E'][i]
        snr_E = compute_snr(signal_E, psd_E_cropped, T_OBS)
        
        # Combined SNR
        snr_combined = np.sqrt(snr_A**2 + snr_E**2)
        
        print(f"  Source {i+1}:")
        print(f"    - SNR (A channel): {snr_A:.2f}")
        print(f"    - SNR (E channel): {snr_E:.2f}")
        print(f"    - Combined SNR: {snr_combined:.2f}")
    
    print()
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

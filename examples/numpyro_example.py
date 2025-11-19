"""
NumPyro Inference Example: Recovering Multiple Galactic Binary Sources

This example:
1. Uses the same three sources from example_usage.py
2. Injects them into noise
3. Crops to a frequency band encompassing all three sources
4. Runs Bayesian inference to recover the parameters
"""

import numpy as np
import jax.numpy as jnp
import jax
from gb_spike_slab import NoiseGenerator, WaveformGenerator, SignalInjector
from gb_spike_slab.inference import multi_source_model
import numpyro
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Set random seeds for reproducibility
np.random.seed(42)
numpyro.set_platform("cpu")  # Change to "gpu" if available
numpyro.set_host_device_count(1)


def main():
    print("=" * 70)
    print("NumPyro Inference Example: Multiple Source Recovery")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Step 1: Generate Data (Same as example_usage.py)
    # =========================================================================
    
    print("Step 1: Generating data with injected signals...")
    
    T_OBS = 30 * 24 * 3600  # 1 month
    DELTA_T = 5.0
    N_SAMPLES = 128
    SEED = 42
    
    # Generate noise
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=SEED)
    noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
    freqs_full = noise_gen.get_frequency_grid()
    psd_A_full, psd_E_full, psd_T_full = noise_gen.get_psds()
    
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
        0.8e-22,         # amplitude
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
    
    # Generate waveforms
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params_true)
    wf_freqs = wf_gen.get_waveform_frequencies(params_true)
    
    # Inject signals
    injector = SignalInjector(noise_freqs=freqs_full, t_obs=T_OBS)
    data_A_full, data_E_full, data_T_full = injector.inject_signals(
        noise_A, noise_E, noise_T,
        A_wf, E_wf, T_wf,
        wf_freqs,
        waveform_generator=wf_gen
    )
    
    print("  - Signals injected into noise")
    print()
    
    # =========================================================================
    # Step 2: Crop to Frequency Band
    # =========================================================================
    
    print("Step 2: Cropping to frequency band...")
    
    # Center frequency (middle of the three sources)
    f_center = 0.00136  # Hz
    
    # Wider search range: ±1e-5 Hz around center
    search_width = 1e-5  # Hz (10 microhertz)
    f_min = f_center - search_width
    f_max = f_center + search_width
    
    print(f"  - Center frequency: {f_center:.8f} Hz")
    print(f"  - Search range: {f_min:.8f} to {f_max:.8f} Hz")
    print(f"  - Search width: ±{search_width:.2e} Hz")
    
    # Crop data
    data_A, data_E, data_T, freqs = injector.crop_data(
        data_A_full, data_E_full, data_T_full, f_min, f_max
    )
    
    # Crop PSDs
    idx_low = np.argmin(np.abs(freqs_full - f_min))
    idx_high = np.argmin(np.abs(freqs_full - f_max))
    psd_A = psd_A_full[idx_low:idx_high]
    psd_E = psd_E_full[idx_low:idx_high]
    
    print(f"  - Cropped to {len(freqs)} frequency bins")
    print()
    
    # =========================================================================
    # Step 3: Run Inference
    # =========================================================================
    
    print("Step 3: Running Bayesian inference...")
    print(f"  - Number of sources to infer: {n_sources}")
    print(f"  - Using NUTS sampler")
    print()
    
    # Convert to JAX arrays
    data_A_jax = jnp.asarray(data_A)
    data_E_jax = jnp.asarray(data_E)
    freqs_jax = jnp.asarray(freqs)
    psd_A_jax = jnp.asarray(psd_A)
    psd_E_jax = jnp.asarray(psd_E)
    
    # Create initialization values near the true parameters
    # Transform true values to the sampled parameter space
    scaled_fdot_true = params_true[:, 1] * 1e18  # Scale fdot
    log_amplitude_true = jnp.log10(params_true[:, 2])  # Log of amplitude
    
    # Initialize with values near the true parameters (with small perturbations)
    init_params = {
        "f0": params_true[:, 0] + jnp.array([-0.2e-7, 0.0, 0.2e-7]),  # Small perturbations
        "scaled_fdot": scaled_fdot_true + jnp.array([-1.0, 0.0, 1.0]),  # Small perturbations
        "log_amplitude": log_amplitude_true + jnp.array([0.1, 0.0, -0.1]),  # Small perturbations
        "ecliptic_lat": params_true[:, 3],
        "ecliptic_lon": params_true[:, 4],
        "polarization": params_true[:, 5],
        "inclination": params_true[:, 6],
        "initial_phase": params_true[:, 7],
    }
    
    # Set up MCMC
    nuts_kernel = NUTS(multi_source_model, max_tree_depth=8)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=3000,
        num_samples=3000,
        num_chains=1,
        progress_bar=True
    )
    
    # Run MCMC with initialization
    mcmc.run(
        jax.random.PRNGKey(42),
        data_A_jax,
        data_E_jax,
        freqs_jax,
        psd_A_jax,
        psd_E_jax,
        T_OBS,
        wf_gen,
        n_sources,
        f_min,
        f_max,
        init_params=init_params
    )
    mcmc.print_summary()
    
    # Get samples
    samples = mcmc.get_samples()
    
    # Transform back to physical parameters for analysis
    fdot_samples = samples["scaled_fdot"] / 1e18  # Convert back from scaled
    amplitude_samples = 10**(samples["log_amplitude"])  # Convert back from log
    
    print()
    print("  - MCMC completed")
    print()
    
    # =========================================================================
    # Step 4: Analyze Results
    # =========================================================================
    
    print("Step 4: Analyzing results...")
    print()
    
    # Extract parameter estimates (medians)
    f0_samples = samples["f0"]  # Shape: (n_samples, n_sources)
    
    print("True vs Recovered Parameters:")
    print("-" * 70)
    print(f"{'Source':<8} {'Parameter':<15} {'True':<15} {'Recovered':<15} {'Error':<15}")
    print("-" * 70)
    
    for i in range(n_sources):
        f0_true = float(params_true[i, 0])
        f0_rec = float(np.median(f0_samples[:, i]))
        f0_err = f0_rec - f0_true
        
        fdot_true = float(params_true[i, 1])
        fdot_rec = float(np.median(fdot_samples[:, i]))
        fdot_err = fdot_rec - fdot_true
        
        amp_true = float(params_true[i, 2])
        amp_rec = float(np.median(amplitude_samples[:, i]))
        amp_err = amp_rec - amp_true
        
        print(f"{i+1:<8} {'f0 (Hz)':<15} {f0_true:<15.8e} {f0_rec:<15.8e} {f0_err:<15.8e}")
        print(f"{'':<8} {'fdot (Hz/s)':<15} {fdot_true:<15.8e} {fdot_rec:<15.8e} {fdot_err:<15.8e}")
        print(f"{'':<8} {'amplitude':<15} {amp_true:<15.8e} {amp_rec:<15.8e} {amp_err:<15.8e}")
        print()
    
    # =========================================================================
    # Step 5: Plot Results
    # =========================================================================
    
    print("Step 5: Creating plots...")
    
    # Plot 1: Corner plot for frequencies
    fig, axes = plt.subplots(n_sources, 1, figsize=(10, 3*n_sources))
    
    for i in range(n_sources):
        axes[i].hist(f0_samples[:, i], bins=50, alpha=0.7, density=True, label='Posterior')
        axes[i].axvline(params_true[i, 0], color='r', linestyle='--', linewidth=2, label='True')
        axes[i].set_xlabel(f'Source {i+1}: f0 (Hz)')
        axes[i].set_ylabel('Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpyro_f0_posteriors.png', dpi=150, bbox_inches='tight')
    print("  - Saved: numpyro_f0_posteriors.png")
    
    # Plot 2: Data and recovered templates
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Get median parameters (transform back from sampled space)
    params_median = jnp.stack([
        jnp.median(f0_samples, axis=0),
        jnp.median(fdot_samples, axis=0),
        jnp.median(amplitude_samples, axis=0),
        jnp.median(samples["ecliptic_lat"], axis=0),
        jnp.median(samples["ecliptic_lon"], axis=0),
        jnp.median(samples["polarization"], axis=0),
        jnp.median(samples["inclination"], axis=0),
        jnp.median(samples["initial_phase"], axis=0),
    ], axis=1)
    
    # Generate recovered waveforms
    A_wf_rec, E_wf_rec, T_wf_rec = wf_gen.generate_waveforms(params_median)
    wf_freqs_rec = wf_gen.get_waveform_frequencies(params_median)
    
    # Interpolate
    template_A_rec = wf_gen.interpolate_waveform(A_wf_rec, wf_freqs_rec, freqs_jax)
    template_A_rec_total = jnp.sum(template_A_rec, axis=0)
    
    template_E_rec = wf_gen.interpolate_waveform(E_wf_rec, wf_freqs_rec, freqs_jax)
    template_E_rec_total = jnp.sum(template_E_rec, axis=0)
    
    # A channel
    axes[0].plot(freqs, np.abs(data_A)**2 / T_OBS, 'C0', alpha=0.5, label='Data')
    axes[0].plot(freqs, np.abs(template_A_rec_total)**2 / T_OBS, 'C1', linewidth=2, label='Recovered')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power Spectral Density')
    axes[0].set_title('A Channel: Data vs Recovered Template')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # E channel
    axes[1].plot(freqs, np.abs(data_E)**2 / T_OBS, 'C0', alpha=0.5, label='Data')
    axes[1].plot(freqs, np.abs(template_E_rec_total)**2 / T_OBS, 'C1', linewidth=2, label='Recovered')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density')
    axes[1].set_title('E Channel: Data vs Recovered Template')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('numpyro_recovery.png', dpi=150, bbox_inches='tight')
    print("  - Saved: numpyro_recovery.png")
    
    plt.close('all')
    
    print()
    print("=" * 70)
    print("Inference example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

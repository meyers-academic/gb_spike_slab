"""
NumPyro Spike-and-Slab Inference Example: Variable Number of Sources

This example:
1. Uses the same three sources from example_usage.py
2. Injects them into noise
3. Crops to a frequency band encompassing all three sources
4. Runs Bayesian inference with spike-and-slab prior to recover parameters
5. Allows for up to 5 potential sources (only 3 are actually present)
"""

import numpy as np
import jax.numpy as jnp
import jax
from gb_spike_slab import NoiseGenerator, WaveformGenerator, SignalInjector
from gb_spike_slab.inference import spike_slab_model
import numpyro
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs
import matplotlib.pyplot as plt
import pandas as pd

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Set random seeds for reproducibility
np.random.seed(42)
numpyro.set_platform("cpu")  # Change to "gpu" if available
numpyro.set_host_device_count(1)


def main():
    print("=" * 70)
    print("NumPyro Spike-and-Slab Inference Example")
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
    
    # Generate 10 sources with frequencies spread across the search range
    np.random.seed(42)  # For reproducibility
    n_sources_true = 10
    
    # Center frequency and search range
    f_center = 0.00136  # Hz
    search_width = 1e-5  # Hz (10 microhertz)
    f_min = f_center - search_width
    f_max = f_center + search_width
    
    # Generate frequencies evenly spaced across the range
    f0_values = np.linspace(f_min + 0.1*search_width, f_max - 0.1*search_width, n_sources_true)
    
    # Generate random parameters for each source
    sources_list = []
    for i in range(n_sources_true):
        source = jnp.array([
            f0_values[i],                    # f0 (Hz)
            8.5e-19 + np.random.uniform(-1e-19, 1e-19),  # fdot (Hz/s)
            1e-22 * (1 + np.random.uniform(-0.2, 0.2)),  # amplitude
            np.random.uniform(-np.pi/2, np.pi/2),  # ecliptic latitude
            np.random.uniform(-np.pi, np.pi),  # ecliptic longitude
            np.random.uniform(0, 2*np.pi),  # polarization
            np.random.uniform(0, np.pi),  # inclination
            np.random.uniform(0, 2*np.pi),  # initial phase
        ])
        sources_list.append(source)
    
    params_true = jnp.vstack(sources_list)
    
    print(f"  - Number of true sources: {n_sources_true}")
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
    # Step 3: Run Inference with Spike-and-Slab
    # =========================================================================
    
    print("Step 3: Running Bayesian inference with spike-and-slab prior...")
    
    n_max_sources = 20  # Allow up to 20 potential sources
    print(f"  - Maximum number of potential sources: {n_max_sources}")
    print(f"  - True number of sources: {n_sources_true}")
    print(f"  - Using NUTS + DiscreteHMCGibbs sampler")
    print()
    
    # Convert to JAX arrays
    data_A_jax = jnp.asarray(data_A)
    data_E_jax = jnp.asarray(data_E)
    freqs_jax = jnp.asarray(freqs)
    psd_A_jax = jnp.asarray(psd_A)
    psd_E_jax = jnp.asarray(psd_E)
    
    # Create initialization values
    # Initialize with true values for first n_sources_true, random for remaining
    n_extra = n_max_sources - n_sources_true
    
    # Transform true values to the sampled parameter space
    scaled_fdot_true = params_true[:, 1] * 1e18
    log_amplitude_true = jnp.log10(params_true[:, 2])
    
    # Calculate spacing for extra sources (use a fraction of search width)
    f_spacing = search_width / (n_extra + 1) if n_extra > 0 else search_width
    
    # Create z array: first n_sources_true active, rest inactive
    z_init = jnp.concatenate([
        jnp.ones(n_sources_true),
        jnp.zeros(n_extra)
    ])
    
    # Initialize frequencies: true values with small perturbations, then evenly spaced
    # Create perturbations for true sources (small random offsets)
    f0_perturbations = jnp.linspace(-0.2e-6, 0.2e-6, n_sources_true)
    f0_init = jnp.concatenate([
        params_true[:, 0] + f0_perturbations,
        jnp.linspace(f_center - f_spacing, f_center + f_spacing, n_extra)
    ])
    
    # Initialize scaled_fdot
    scaled_fdot_init = jnp.concatenate([
        scaled_fdot_true + jnp.linspace(-1.0, 1.0, n_sources_true),
        jnp.zeros(n_extra)
    ])
    
    # Initialize log_amplitude
    log_amplitude_init = jnp.concatenate([
        log_amplitude_true + jnp.linspace(0.1, -0.1, n_sources_true),
        jnp.full(n_extra, -21.0)
    ])
    
    # Initialize other parameters
    ecliptic_lat_init = jnp.concatenate([
        params_true[:, 3],
        jnp.zeros(n_extra)
    ])
    
    ecliptic_lon_init = jnp.concatenate([
        params_true[:, 4],
        jnp.zeros(n_extra)
    ])
    
    polarization_init = jnp.concatenate([
        params_true[:, 5],
        jnp.full(n_extra, np.pi)
    ])
    
    inclination_init = jnp.concatenate([
        params_true[:, 6],
        jnp.full(n_extra, np.pi/2)
    ])
    
    initial_phase_init = jnp.concatenate([
        params_true[:, 7],
        jnp.full(n_extra, np.pi)
    ])
    
    init_params = {
        "z": z_init,
        "f0": f0_init,
        "scaled_fdot": scaled_fdot_init,
        "log_amplitude": log_amplitude_init,
        "ecliptic_lat": ecliptic_lat_init,
        "ecliptic_lon": ecliptic_lon_init,
        "polarization": polarization_init,
        "inclination": inclination_init,
        "initial_phase": initial_phase_init,
        "inclusion_prob": 0.5,
    }
    
    # Set up MCMC with mixed sampling
    # Use NUTS for continuous variables and DiscreteHMCGibbs for discrete
    nuts_kernel = NUTS(spike_slab_model, max_tree_depth=10)
    discrete_kernel = DiscreteHMCGibbs(nuts_kernel)
    
    mcmc = MCMC(
        discrete_kernel,
        num_warmup=3000,
        num_samples=3000,
        num_chains=1,
        progress_bar=True
    )
    
    # Run MCMC with initialization
    mcmc.run(
        jax.random.key(42),
        data_A_jax,
        data_E_jax,
        freqs_jax,
        psd_A_jax,
        psd_E_jax,
        T_OBS,
        wf_gen,
        n_max_sources,
        f_min,
        f_max,
        init_params=init_params
    )
    
    # Get samples
    samples = mcmc.get_samples()
    
    # Convert to DataFrame and save
    # Convert 2D arrays to separate columns for each source
    samples_flat = {}
    for key, value in samples.items():
        arr = np.array(value)
        if arr.ndim == 2:
            # Flatten 2D arrays: create columns for each source index
            for i in range(arr.shape[1]):
                samples_flat[f'{key}_{i}'] = arr[:, i]
        else:
            # Keep 1D arrays as-is
            samples_flat[key] = arr
    
    df = pd.DataFrame(samples_flat)
    df.to_feather('spike_slab_samples.feather')
    print("  - Saved: spike_slab_samples.feather")
    print()
    
    # Transform back to physical parameters for analysis
    fdot_samples = samples["fdot"]  # Already transformed in model
    amplitude_samples = samples["amplitude"]  # Already transformed in model
    z_samples = samples["z"]  # Indicator variables
    
    print()
    print("  - MCMC completed")
    print()
    
    # =========================================================================
    # Step 4: Analyze Results
    # =========================================================================
    
    print("Step 4: Analyzing results...")
    print()
    
    # Extract parameter estimates
    f0_samples = samples["f0"]  # Shape: (n_samples, n_max_sources)
    n_active_samples = samples["n_active_sources"]  # Shape: (n_samples,)
    
    # Compute inclusion probabilities
    inclusion_probs = jnp.mean(z_samples, axis=0)
    
    print("Source Inclusion Probabilities:")
    print("-" * 70)
    for i in range(n_max_sources):
        print(f"  Source {i+1}: {float(inclusion_probs[i]):.3f}")
    print()
    
    print(f"Number of active sources (posterior mean): {float(jnp.mean(n_active_samples)):.2f}")
    print(f"True number of sources: {n_sources_true}")
    print()
    
    # For active sources, show parameter recovery
    print("True vs Recovered Parameters (for likely active sources):")
    print("-" * 70)
    print(f"{'Source':<8} {'Incl. Prob':<12} {'Parameter':<15} {'True':<15} {'Recovered':<15} {'Error':<15}")
    print("-" * 70)
    
    # Match recovered sources to true sources (simple: by frequency proximity)
    for i in range(min(n_sources_true, n_max_sources)):
        if inclusion_probs[i] > 0.5:  # Only show if likely active
            f0_true = float(params_true[i, 0])
            f0_rec = float(np.median(f0_samples[:, i]))
            f0_err = f0_rec - f0_true
            
            fdot_true = float(params_true[i, 1])
            fdot_rec = float(np.median(fdot_samples[:, i]))
            fdot_err = fdot_rec - fdot_true
            
            amp_true = float(params_true[i, 2])
            amp_rec = float(np.median(amplitude_samples[:, i]))
            amp_err = amp_rec - amp_true
            
            print(f"{i+1:<8} {float(inclusion_probs[i]):<12.3f} {'f0 (Hz)':<15} {f0_true:<15.8e} {f0_rec:<15.8e} {f0_err:<15.8e}")
            print(f"{'':<8} {'':<12} {'fdot (Hz/s)':<15} {fdot_true:<15.8e} {fdot_rec:<15.8e} {fdot_err:<15.8e}")
            print(f"{'':<8} {'':<12} {'amplitude':<15} {amp_true:<15.8e} {amp_rec:<15.8e} {amp_err:<15.8e}")
            print()
    
    # =========================================================================
    # Step 5: Plot Results
    # =========================================================================
    
    print("Step 5: Creating plots...")
    
    # Plot 1: Inclusion probabilities
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(range(1, n_max_sources + 1), inclusion_probs, alpha=0.7, color='C0')
    ax.axhline(0.5, color='r', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_xlabel('Source Index')
    ax.set_ylabel('Inclusion Probability')
    ax.set_title('Spike-and-Slab: Source Inclusion Probabilities')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spike_slab_inclusion_probs.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_inclusion_probs.png")
    
    # Plot 2: Number of active sources
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(n_active_samples, bins=range(n_max_sources + 2), alpha=0.7, density=True, align='left')
    ax.axvline(n_sources_true, color='r', linestyle='--', linewidth=2, label=f'True: {n_sources_true}')
    ax.set_xlabel('Number of Active Sources')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution: Number of Active Sources')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spike_slab_n_sources.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_n_sources.png")
    
    # Plot 3: Frequencies for active sources
    fig, axes = plt.subplots(n_max_sources, 1, figsize=(10, 3*n_max_sources))
    
    for i in range(n_max_sources):
        # Only plot if source is sometimes active
        if inclusion_probs[i] > 0.1:
            # Get samples where this source is active
            active_mask = z_samples[:, i] == 1
            if jnp.any(active_mask):
                f0_active = f0_samples[active_mask, i]
                axes[i].hist(f0_active, bins=50, alpha=0.7, density=True, label='Posterior (active)')
            
            # Show true frequency if this corresponds to a true source
            if i < n_sources_true:
                axes[i].axvline(params_true[i, 0], color='r', linestyle='--', linewidth=2, label='True')
            
            axes[i].set_xlabel(f'Source {i+1}: f0 (Hz) (Incl. Prob: {float(inclusion_probs[i]):.2f})')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, f'Source {i+1}: Not Active\n(Incl. Prob: {float(inclusion_probs[i]):.2f})',
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_xlabel(f'Source {i+1}: f0 (Hz)')
    
    plt.tight_layout()
    plt.savefig('spike_slab_f0_posteriors.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_f0_posteriors.png")
    
    # Plot 4: Frequency vs MCMC steps (trace plot) - all sources on one plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    n_steps = f0_samples.shape[0]
    steps = np.arange(n_steps)
    
    # Plot each source with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_max_sources))
    
    # First, plot horizontal lines for true injected frequencies (low alpha)
    for j in range(n_sources_true):
        ax.axhline(params_true[j, 0], color='r', linestyle='--', 
                  linewidth=2, alpha=0.2, 
                  label='True injected frequencies' if j == 0 else '')
    
    # Then plot recovered sources
    for i in range(n_max_sources):
        # Get frequencies and indicator variables for this source
        f0_source = np.array(f0_samples[:, i])
        z_source = np.array(z_samples[:, i])
        
        # Plot active samples (z=1) with high alpha
        active_mask = z_source == 1
        if np.any(active_mask):
            ax.plot(steps[active_mask], f0_source[active_mask], 
                   '-', color=colors[i], alpha=0.7, linewidth=1.0, 
                   label=f'Recovered source {i+1} (active)')
        
        # Plot inactive samples with very low alpha (essentially invisible)
        inactive_mask = z_source == 0
        if np.any(inactive_mask):
            ax.plot(steps[inactive_mask], f0_source[inactive_mask], 
                   '-', color=colors[i], alpha=0.01, linewidth=0.5)
    
    ax.set_xlabel('MCMC Step')
    ax.set_ylabel('Frequency f0 (Hz)')
    ax.set_title('Frequency Trace: All Sources')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spike_slab_f0_trace.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_f0_trace.png")
    
    plt.close('all')
    
    print()
    print("=" * 70)
    print("Spike-and-slab inference example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import json
import pandas as pd
from numpyro.infer import init_to_value, MCMC, NUTS, DiscreteHMCGibbs
from model import numpyro_model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tophat_populations.confusion_noise import get_threshold_and_conf_noise_for_powerlaw
import matplotlib.pyplot as plt

def main():
    print("=" * 70)
    print("Try this")
    print("=" * 70)
    print()
    
    # =========================================================================
    # Step 1: Import Data
    # =========================================================================
    
    print("Step 1: Importing data (pre-cropped)...")

    DATADIR = '/Users/aria/PROJECTS/slotflow-geebee-comparison/air_dataset/'
    OUTFOLDER = './data_tests/'

    data_sf = np.load(DATADIR + 'sample_0001.npz')
    metadata_sf = json.load(open(DATADIR + 'metadata.json'))

    data_A = data_sf['Ar_zoom']
    data_E = data_sf['Er_zoom']
    data_strain = [data_A, data_E]

    freqs = data_sf['freqs_zoom']
    f_min = freqs[0]
    f_max = freqs[-1]
    search_width = f_max - f_min
    f_center = 0.5 * (f_min + f_max)

    print(f"  - Data path: {DATADIR + 'sample_0001.npz'}")
    print(f"  - Frequency range: [{freqs[0]:.2e}, {freqs[-1]:.2e}] Hz")
    print(f"  - Center frequency: {f_center:.2e} Hz")
    print(f"  - Search width: {search_width:.2e} Hz")

    psd_A = data_sf['psd_A']
    psd_E = data_sf['psd_E']
    data_psd = [psd_A, psd_E]

    n_sources_true = metadata_sf['n_samples']
    params_true = data_sf['params']
    T_OBS = metadata_sf['T_obs']

    # convert params to GBObject format for waveform generation -- maybe not necessary
    # cat = pd.DataFrame(
    #     params_true,
    #     columns=['Frequency', 'FrequencyDerivative', 'Amplitude',
    #              'Inclination', 'RightAscension', 'Declination',
    #              'Polarization', 'InitialPhase'],
    # )
    # gbo_injections = GBObject.from_pandas_dataframe(cat, t_init=0.0)
    

    # =========================================================================
    # Step 2: Run Inference with Spike-and-Slab + Population Model
    # =========================================================================

    max_num_resolvable = 10
    n_extra = max_num_resolvable - n_sources_true
 
    # Transform true values to the sampled parameter space
    scaled_fdot_true = params_true[:, 1] * 1e18
    log_amplitude_true = jnp.log10(params_true[:, 2])

    # Initialize z: first n_sources_true active, rest inactive
    z_init = jnp.concatenate([
        jnp.ones(n_sources_true),
        jnp.zeros(n_extra)
    ])

    # Initialize frequencies: true values with small perturbations, then evenly spaced
    # Create perturbations for true sources (small random offsets)
    f0_perturbations = jnp.linspace(-0.2e-7, 0.2e-7, n_sources_true)
    f_spacing = search_width / (n_extra + 1) if n_extra > 0 else search_width
    f0_init = jnp.concatenate([
        params_true[:, 0] + f0_perturbations,
        jnp.linspace(f_center - f_spacing, f_center + f_spacing, n_extra)
    ])

    # Initialize scaled_fdot
    scaled_fdot_init = jnp.concatenate([
        scaled_fdot_true,
        jnp.zeros(n_extra)
    ])

    # Initialize log_amplitude
    log_amplitude_init = jnp.concatenate([
        log_amplitude_true + jnp.linspace(0.1, -0.1, n_sources_true),
        jnp.full(n_extra, -21.0) # this value is probably relevant too
    ])
    
    # Initialize other parameters (param_order: f,fdot,A,inc,ra,dec,pol,phi)
    ra_init = jnp.concatenate([
        params_true[:, 4],
        jnp.full(n_extra, np.pi)  # centre of Uniform(0, 2π) — boundary (0) maps to -inf in unconstrained space
    ])

    dec_init = jnp.concatenate([
        params_true[:, 5],
        jnp.zeros(n_extra)
    ])
    
    polarization_init = jnp.concatenate([
        params_true[:, 6],
        jnp.full(n_extra, np.pi)
    ])
    
    inclination_init = jnp.concatenate([
        params_true[:, 3],
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
        "ra": ra_init,
        "dec": dec_init,
        "polarization": polarization_init,
        "inclination": inclination_init,
        "initial_phase": initial_phase_init,
        "inclusion_prob": 0.5,
    }

    kernel = DiscreteHMCGibbs(
        NUTS(numpyro_model, init_strategy=init_to_value(values=init_params), max_tree_depth=6, target_accept_prob=0.9),
        modified=True)
    mcmc = MCMC(kernel, 
                    num_warmup=3000, 
                    num_samples=10000, 
                    num_chains=1, 
                    chain_method=jax.vmap,
                    progress_bar=True)

    mcmc.run(jax.random.key(0), data_strain, data_psd, max_num_resolvable=max_num_resolvable, frequencies=freqs, T_obs=T_OBS, f_min=f_min, f_max=f_max, population_flag=False, total_N_sources=n_sources_true)
    
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
    df.to_feather(f'{OUTFOLDER}/spike_slab_samples.feather')
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
    # Step 3: Analyze Results
    # =========================================================================
    
    print("Step 3: Analyzing results...")
    print()
    
    # Extract parameter estimates
    f0_samples = samples["f0"]  # Shape: (n_samples, n_max_sources)
    n_active_samples = samples["n_active_sources"]  # Shape: (n_samples,)
    
    # Compute inclusion probabilities
    inclusion_probs = jnp.mean(z_samples, axis=0)
    
    print("Source Inclusion Probabilities:")
    print("-" * 70)
    for i in range(max_num_resolvable):
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
    for i in range(min(n_sources_true, max_num_resolvable)):
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
    # Step 4: Plot Results
    # =========================================================================
    
    print("Step 4: Creating plots...")
    
    # Plot 1: Inclusion probabilities
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.bar(range(1, max_num_resolvable + 1), inclusion_probs, alpha=0.7, color='C0')
    ax.axhline(0.5, color='r', linestyle='--', linewidth=2, label='50% threshold')
    ax.set_xlabel('Source Index')
    ax.set_ylabel('Inclusion Probability')
    ax.set_title('Spike-and-Slab: Source Inclusion Probabilities')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTFOLDER}/spike_slab_inclusion_probs.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_inclusion_probs.png")
    
    # Plot 2: Number of active sources
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(n_active_samples, bins=range(max_num_resolvable + 2), alpha=0.7, density=True, align='left')
    ax.axvline(n_sources_true, color='r', linestyle='--', linewidth=2, label=f'True: {n_sources_true}')
    ax.set_xlabel('Number of Active Sources')
    ax.set_ylabel('Density')
    ax.set_title('Posterior Distribution: Number of Active Sources')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTFOLDER}/spike_slab_n_sources.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_n_sources.png")
    
    # Plot 3: Frequencies for active sources
    fig, axes = plt.subplots(max_num_resolvable, 1, figsize=(10, 3*max_num_resolvable))
    
    for i in range(max_num_resolvable):
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
    plt.savefig(f'{OUTFOLDER}/spike_slab_f0_posteriors.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_f0_posteriors.png")
    
    # Plot 4: Frequency vs MCMC steps (trace plot) - all sources on one plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    n_steps = f0_samples.shape[0]
    steps = np.arange(n_steps)
    
    # Plot each source with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, max_num_resolvable))
    
    # First, plot horizontal lines for true injected frequencies (low alpha)
    for j in range(n_sources_true):
        ax.axhline(params_true[j, 0], color='r', linestyle='--', 
                  linewidth=2, alpha=0.2, 
                  label='True injected frequencies' if j == 0 else '')
    
    # Then plot recovered sources
    for i in range(max_num_resolvable):
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
    plt.savefig(f'{OUTFOLDER}/spike_slab_f0_trace.png', dpi=150, bbox_inches='tight')
    print("  - Saved: spike_slab_f0_trace.png")
    
    plt.close('all')
    
    print()
    print("=" * 70)
    print("Spike-and-slab inference example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

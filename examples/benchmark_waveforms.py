"""
Benchmarking script for waveform evaluation performance.

Tests how long it takes to evaluate N waveforms (128 samples each) for:
- 1 year of observation time
- 1 microhertz frequency band around 1 mHz
"""

import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt
from gb_spike_slab import NoiseGenerator, WaveformGenerator, SignalInjector
from gb_spike_slab.utils import compute_likelihood

# Configure JAX
jax.config.update("jax_enable_x64", True)

# Warm up JIT
print("Warming up JIT compilation...")
wf_gen_warmup = WaveformGenerator(t_obs=365*24*3600, n_samples=128)
params_warmup = jnp.array([[0.001, 9e-19, 1e-22, 0.3, -2.7, 3.5, 0.5, 3.0]])
A_warmup, E_warmup, T_warmup = wf_gen_warmup.generate_waveforms(params_warmup)
wf_freqs_warmup = wf_gen_warmup.get_waveform_frequencies(params_warmup)
freqs_dummy = jnp.linspace(0.001 - 1e-6, 0.001 + 1e-6, 100)
_ = wf_gen_warmup.interpolate_waveform(A_warmup, wf_freqs_warmup, freqs_dummy)
print("  - JIT warmup complete")
print()


def evaluate_waveforms(wf_gen, params, freqs_target):
    """
    Evaluate waveforms: generate and interpolate.
    
    Returns the interpolated waveforms for A and E channels.
    """
    # Generate waveforms
    A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params)
    wf_freqs = wf_gen.get_waveform_frequencies(params)
    
    # Interpolate to target frequency grid
    A_interp = wf_gen.interpolate_waveform(A_wf, wf_freqs, freqs_target)
    E_interp = wf_gen.interpolate_waveform(E_wf, wf_freqs, freqs_target)
    
    return A_interp, E_interp

# JIT compile the function
evaluate_waveforms_jit = jax.jit(evaluate_waveforms, static_argnames=['wf_gen'])


def benchmark_evaluation(N_values, n_trials=10):
    """
    Benchmark waveform evaluation for different numbers of sources.
    
    Parameters
    ----------
    N_values : list of int
        Number of waveforms to test
    n_trials : int
        Number of trials per N value for averaging
        
    Returns
    -------
    times : dict
        Dictionary with 'mean' and 'std' arrays of evaluation times
    """
    T_OBS = 365 * 24 * 3600  # 1 year
    DELTA_T = 5.0
    N_SAMPLES = 128
    
    # Set up frequency band: 1 microhertz around 1 mHz
    f_center = 0.001  # 1 mHz
    f_band = 1e-6  # 1 microhertz
    f_min = f_center - f_band / 2
    f_max = f_center + f_band / 2
    
    # Generate noise to get frequency grid
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=42)
    freqs_full = noise_gen.get_frequency_grid()
    
    # Crop to frequency band
    injector = SignalInjector(noise_freqs=freqs_full, t_obs=T_OBS)
    _, _, _, freqs_cropped = injector.crop_data(
        np.zeros_like(freqs_full), np.zeros_like(freqs_full), 
        np.zeros_like(freqs_full), f_min, f_max
    )
    freqs_target = jnp.asarray(freqs_cropped)
    
    print(f"Frequency band: {f_min:.9f} to {f_max:.9f} Hz")
    print(f"Number of frequency bins: {len(freqs_target)}")
    print()
    
    # Initialize waveform generator
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    
    times_mean = []
    times_std = []
    
    for N in N_values:
        print(f"Benchmarking N={N} waveforms...")
        
        # Generate random parameters for N sources
        np.random.seed(42)
        params = jnp.array([
            [
                f_center + np.random.uniform(-f_band/4, f_band/4),  # f0
                9e-19 + np.random.uniform(-1e-19, 1e-19),  # fdot
                1e-22 * (1 + np.random.uniform(-0.2, 0.2)),  # amplitude
                np.random.uniform(-np.pi/2, np.pi/2),  # ecliptic_lat
                np.random.uniform(-np.pi, np.pi),  # ecliptic_lon
                np.random.uniform(0, 2*np.pi),  # polarization
                np.random.uniform(0, np.pi),  # inclination
                np.random.uniform(0, 2*np.pi),  # initial_phase
            ]
            for _ in range(N)
        ])
        
        # JIT compile for this N
        print(f"  - Compiling for N={N}...")
        _ = evaluate_waveforms_jit(wf_gen, params, freqs_target)
        
        # Benchmark: run 100 evaluations per trial and divide by 100
        n_evaluations_per_trial = 100
        times_trial = []
        for trial in range(n_trials):
            start = time.perf_counter()
            for _ in range(n_evaluations_per_trial):
                A_interp, E_interp = evaluate_waveforms_jit(wf_gen, params, freqs_target)
                # Block until computation is done
                _ = jnp.sum(A_interp) + jnp.sum(E_interp)
            end = time.perf_counter()
            # Time per evaluation
            time_per_eval = (end - start) / n_evaluations_per_trial
            times_trial.append(time_per_eval)
        
        mean_time = np.mean(times_trial)
        std_time = np.std(times_trial)
        times_mean.append(mean_time)
        times_std.append(std_time)
        
        print(f"  - Mean time per evaluation: {mean_time*1000:.2f} ms ± {std_time*1000:.2f} ms")
        print()
    
    return {
        'mean': np.array(times_mean),
        'std': np.array(times_std),
        'N_values': np.array(N_values)
    }


def main():
    print("=" * 70)
    print("Waveform Evaluation Benchmarking")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Observation time: 1 year")
    print("  - Frequency band: 1 microhertz around 1 mHz")
    print("  - Waveform samples: 128 per source")
    print()
    
    # Test different numbers of waveforms
    N_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    results = benchmark_evaluation(N_values, n_trials=10)
    
    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.errorbar(results['N_values'], results['mean'] * 1000, 
                yerr=results['std'] * 1000, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    ax.set_xlabel('Number of Waveforms (N)', fontsize=12)
    ax.set_ylabel('Evaluation Time (ms)', fontsize=12)
    ax.set_title('Waveform Evaluation Time vs Number of Sources', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    # Add linear fit line to check scaling
    if len(results['N_values']) > 1:
        log_N = np.log(results['N_values'])
        log_time = np.log(results['mean'])
        coeffs = np.polyfit(log_N, log_time, 1)
        fit_line = np.exp(coeffs[1]) * results['N_values']**coeffs[0]
        ax.plot(results['N_values'], fit_line * 1000, 'r--', alpha=0.5, 
               label=f'Fit: t ∝ N^{coeffs[0]:.2f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('waveform_benchmark.png', dpi=150, bbox_inches='tight')
    print("Saved: waveform_benchmark.png")
    
    # Print summary
    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    for i, N in enumerate(results['N_values']):
        print(f"N={N:3d}: {results['mean'][i]*1000:6.2f} ± {results['std'][i]*1000:4.2f} ms")
    print()


if __name__ == "__main__":
    main()

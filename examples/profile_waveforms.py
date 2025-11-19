"""
Profiling script for waveform evaluation with N=4 sources.

Uses JAX profiling to identify bottlenecks in waveform generation and interpolation.
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import profiler
import os
from claude_gb_toolkit import NoiseGenerator, WaveformGenerator, SignalInjector
from claude_gb_toolkit.utils import compute_likelihood

# Configure JAX
jax.config.update("jax_enable_x64", True)
# JIT is enabled for production-like profiling


def evaluate_waveforms_full(wf_gen, params, freqs_target, psd_A, psd_E, data_A, data_E, t_obs):
    """
    Full evaluation: generate, interpolate, and compute likelihood.
    """
    # Generate waveforms
    A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params)
    wf_freqs = wf_gen.get_waveform_frequencies(params)
    
    # Interpolate to target frequency grid
    A_interp = wf_gen.interpolate_waveform(A_wf, wf_freqs, freqs_target)
    E_interp = wf_gen.interpolate_waveform(E_wf, wf_freqs, freqs_target)
    
    # Sum over sources
    template_A = jnp.sum(A_interp, axis=0)
    template_E = jnp.sum(E_interp, axis=0)
    
    # Compute likelihood
    log_like_A = compute_likelihood(data_A, template_A, psd_A, t_obs)
    log_like_E = compute_likelihood(data_E, template_E, psd_E, t_obs)
    log_like = log_like_A + log_like_E
    
    return log_like


def main():
    print("=" * 70)
    print("Waveform Evaluation Profiling (N=4)")
    print("=" * 70)
    print()
    
    T_OBS = 365 * 24 * 3600  # 1 year
    DELTA_T = 5.0
    N_SAMPLES = 128
    N = 4  # Number of waveforms
    
    # Set up frequency band: 1 microhertz around 1 mHz
    f_center = 0.001  # 1 mHz
    f_band = 1e-6  # 1 microhertz
    f_min = f_center - f_band / 2
    f_max = f_center + f_band / 2
    
    print("Configuration:")
    print(f"  - Observation time: {T_OBS / (365*24*3600):.1f} years")
    print(f"  - Frequency band: {f_min:.9f} to {f_max:.9f} Hz")
    print(f"  - Number of waveforms: {N}")
    print(f"  - Samples per waveform: {N_SAMPLES}")
    print()
    
    # Generate noise to get frequency grid
    print("Generating noise...")
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=42)
    noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
    freqs_full = noise_gen.get_frequency_grid()
    psd_A_full, psd_E_full, psd_T_full = noise_gen.get_psds()
    
    # Crop to frequency band
    injector = SignalInjector(noise_freqs=freqs_full, t_obs=T_OBS)
    noise_A_cropped, noise_E_cropped, _, freqs_cropped = injector.crop_data(
        noise_A, noise_E, noise_T, f_min, f_max
    )
    idx_low = np.argmin(np.abs(freqs_full - f_min))
    idx_high = np.argmin(np.abs(freqs_full - f_max))
    psd_A = psd_A_full[idx_low:idx_high]
    psd_E = psd_E_full[idx_low:idx_high]  # Fixed: was idx_high:idx_high
    
    freqs_target = jnp.asarray(freqs_cropped)
    data_A = jnp.asarray(noise_A_cropped)
    data_E = jnp.asarray(noise_E_cropped)
    psd_A_jax = jnp.asarray(psd_A)
    psd_E_jax = jnp.asarray(psd_E)
    
    print(f"  - Cropped to {len(freqs_target)} frequency bins")
    print()
    
    # Initialize waveform generator
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    
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
    
    print(f"Generated parameters for {N} sources")
    print()
    
    # JIT compile for production-like profiling
    print("Compiling JIT...")
    eval_fn = jax.jit(evaluate_waveforms_full, static_argnames=['wf_gen', 't_obs'])
    _ = eval_fn(wf_gen, params, freqs_target, 
                psd_A_jax, psd_E_jax, data_A, data_E, T_OBS)
    print("  - Compilation complete")
    print()
    
    # Create output directory
    logdir = "./jax_profiler_output"
    os.makedirs(logdir, exist_ok=True)
    
    # Profile with JAX profiler using start_trace/stop_trace
    profiler.start_trace(logdir)
    
    try:
        # Warm up
        print("  - Warming up...")
        for _ in range(3):
            _ = eval_fn(wf_gen, params, freqs_target, 
                       psd_A_jax, psd_E_jax, data_A, data_E, T_OBS)
            _.block_until_ready()
        
        # Profile with many iterations to accumulate measurable time
        print("  - Profiling (this may take a minute)...")
        import time
        start_time = time.time()
        for i in range(1000):  # Run many times to accumulate time
            log_like = eval_fn(wf_gen, params, freqs_target, 
                              psd_A_jax, psd_E_jax, data_A, data_E, T_OBS)
            log_like.block_until_ready()  # Ensure computation completes
            if (i + 1) % 200 == 0:
                print(f"  - Completed {i+1} evaluations...")
        elapsed = time.time() - start_time
        print(f"  - Total wall-clock time: {elapsed:.2f} seconds")
        print(f"  - Average per evaluation: {elapsed/1000*1000:.3f} ms")
    finally:
        profiler.stop_trace()
    
    print("Profiling complete!")
    print()
    print(f"Profiling output saved to: {logdir}/")
    print()
    print("Note: This profile was run WITH JIT compilation.")
    print("      This shows the actual performance of compiled code.")
    print()
    print("To view the profile:")
    print("  1. Install tensorboard: pip install tensorboard")
    print(f"  2. Run: tensorboard --logdir={logdir}")
    print("  3. Open http://localhost:6006 in your browser")
    print()
    print("=" * 70)
    print("Profiling script completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()

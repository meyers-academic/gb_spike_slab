"""
Test Suite for Gravitational Wave Data Analysis Package

Run this to verify all components are working correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
from claude_gb_toolkit import NoiseGenerator, WaveformGenerator, SignalInjector, FrequencyGrid
from claude_gb_toolkit.utils import compute_snr, validate_parameters

jax.config.update("jax_enable_x64", True)


def test_noise_generation():
    """Test noise generation module."""
    print("Testing NoiseGenerator...")
    
    T_OBS = 365 * 24 * 3600
    DELTA_T = 5.0
    
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=42)
    
    # Test noise generation
    noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
    assert noise_A.shape == noise_E.shape == noise_T.shape
    assert len(noise_A) == len(noise_gen.get_frequency_grid())
    
    # Test reproducibility
    noise_A2, _, _ = noise_gen.generate_all_channels(seed=42)
    assert np.allclose(noise_A, noise_A2), "Noise generation not reproducible"
    
    # Test PSDs
    psd_A, psd_E, psd_T = noise_gen.get_psds()
    assert len(psd_A) == len(noise_A)
    assert np.all(psd_A > 0), "PSD should be positive"
    
    print("  ✓ NoiseGenerator passed all tests")


def test_waveform_generation():
    """Test waveform generation module."""
    print("Testing WaveformGenerator...")
    
    T_OBS = 365 * 24 * 3600
    N_SAMPLES = 128
    
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    
    # Test single source
    params_single = jnp.array([[0.00136, 8.9e-19, 1e-22, 0.3, -2.7, 3.5, 0.5, 3.0]])
    A, E, T = wf_gen.generate_waveforms(params_single)
    assert A.shape == (1, N_SAMPLES)
    assert E.shape == (1, N_SAMPLES)
    assert T.shape == (1, N_SAMPLES)
    
    # Test multiple sources (vectorization)
    params_multi = jnp.array([
        [0.00136, 8.9e-19, 1e-22, 0.3, -2.7, 3.5, 0.5, 3.0],
        [0.00137, 9.1e-19, 0.9e-22, 0.4, -2.5, 3.6, 0.6, 2.9],
        [0.00138, 8.7e-19, 1.1e-22, 0.2, -2.8, 3.4, 0.4, 3.1],
    ])
    A, E, T = wf_gen.generate_waveforms(params_multi)
    assert A.shape == (3, N_SAMPLES)
    
    # Test frequency grid
    wf_freqs = wf_gen.get_waveform_frequencies(params_multi)
    assert wf_freqs.shape == (3, N_SAMPLES)
    
    # Test parameter validation
    assert validate_parameters(params_multi), "Valid parameters should pass"
    
    invalid_params = jnp.array([[0.00136, 8.9e-19, -1e-22, 0.3, -2.7, 3.5, 0.5, 3.0]])  # negative amplitude
    assert not validate_parameters(invalid_params), "Invalid parameters should fail"
    
    print("  ✓ WaveformGenerator passed all tests")


def test_signal_injection():
    """Test signal injection module."""
    print("Testing SignalInjector...")
    
    T_OBS = 365 * 24 * 3600
    DELTA_T = 5.0
    N_SAMPLES = 128
    
    # Generate noise
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=42)
    noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
    freqs = noise_gen.get_frequency_grid()
    
    # Generate waveforms
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    params = jnp.array([
        [0.00136, 8.9e-19, 1e-22, 0.3, -2.7, 3.5, 0.5, 3.0],
        [0.00137, 9.1e-19, 0.9e-22, 0.4, -2.5, 3.6, 0.6, 2.9],
    ])
    A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params)
    wf_freqs = wf_gen.get_waveform_frequencies(params)
    
    # Test interpolation injection
    injector = SignalInjector(noise_freqs=freqs, t_obs=T_OBS)
    data_A, data_E, data_T = injector.inject_signals(
        noise_A, noise_E, noise_T,
        A_wf, E_wf, T_wf,
        wf_freqs
    )
    
    assert data_A.shape == noise_A.shape
    assert not np.allclose(data_A, noise_A), "Data should differ from noise after injection"
    
    # Test with return_individual
    (data_A2, _, _), interpolated_wfs = injector.inject_signals(
        noise_A, noise_E, noise_T,
        A_wf, E_wf, T_wf,
        wf_freqs,
        return_individual=True
    )
    
    assert np.allclose(data_A, data_A2), "Should get same result with return_individual"
    assert 'A' in interpolated_wfs
    assert interpolated_wfs['A'].shape == (2, len(freqs))
    
    # Test cropping
    f_min, f_max = 0.001, 0.002
    cropped_A, cropped_E, cropped_T, cropped_freqs = injector.crop_data(
        data_A, data_E, data_T, f_min, f_max
    )
    assert len(cropped_A) < len(data_A)
    assert cropped_freqs[0] >= f_min
    assert cropped_freqs[-1] <= f_max
    
    print("  ✓ SignalInjector passed all tests")


def test_utilities():
    """Test utility functions."""
    print("Testing utilities...")
    
    T_OBS = 365 * 24 * 3600
    
    # Create fake signal and PSD
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    psd = np.ones(1000) * 1e-40
    
    # Test SNR computation
    snr = compute_snr(signal, psd, T_OBS)
    assert snr > 0, "SNR should be positive"
    assert np.isfinite(snr), "SNR should be finite"
    
    # Test FrequencyGrid
    grid = FrequencyGrid(t_obs=T_OBS, delta_t=5.0)
    assert len(grid) > 0
    assert grid.df == 1.0 / T_OBS
    
    idx = grid.get_bin_index(0.001)
    assert 0 <= idx < len(grid)
    
    idx_low, idx_high = grid.get_frequency_range(0.001, 0.002)
    assert idx_low < idx_high
    
    print("  ✓ Utilities passed all tests")


def test_end_to_end():
    """Test complete end-to-end workflow."""
    print("Testing end-to-end workflow...")
    
    T_OBS = 365 * 24 * 3600
    DELTA_T = 5.0
    N_SAMPLES = 128
    
    # Generate noise
    noise_gen = NoiseGenerator(t_obs=T_OBS, delta_t=DELTA_T, seed=42)
    noise_A, noise_E, noise_T = noise_gen.generate_all_channels()
    freqs = noise_gen.get_frequency_grid()
    psd_A, psd_E, psd_T = noise_gen.get_psds()
    
    # Generate waveforms for 10 sources
    wf_gen = WaveformGenerator(t_obs=T_OBS, n_samples=N_SAMPLES)
    n_sources = 10
    params = jnp.array([
        [0.00136 + i*1e-5, 8.9e-19, 1e-22, 0.3, -2.7, 3.5, 0.5, 3.0]
        for i in range(n_sources)
    ])
    
    A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params)
    wf_freqs = wf_gen.get_waveform_frequencies(params)
    
    # Inject signals
    injector = SignalInjector(noise_freqs=freqs, t_obs=T_OBS)
    (data_A, data_E, data_T), interpolated_wfs = injector.inject_signals(
        noise_A, noise_E, noise_T,
        A_wf, E_wf, T_wf,
        wf_freqs,
        return_individual=True
    )
    
    # Verify shapes
    assert data_A.shape == noise_A.shape
    assert interpolated_wfs['A'].shape == (n_sources, len(freqs))
    
    # Compute SNRs
    snrs = []
    for i in range(n_sources):
        signal_A = interpolated_wfs['A'][i]
        signal_E = interpolated_wfs['E'][i]
        snr_A = compute_snr(signal_A, psd_A, T_OBS)
        snr_E = compute_snr(signal_E, psd_E, T_OBS)
        snr_combined = np.sqrt(snr_A**2 + snr_E**2)
        snrs.append(snr_combined)
        assert snr_combined > 0, f"SNR for source {i} should be positive"
    
    print(f"  ✓ Successfully processed {n_sources} sources")
    print(f"  ✓ SNR range: {min(snrs):.1f} to {max(snrs):.1f}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Running Test Suite for GW Data Analysis Package")
    print("=" * 70)
    print()
    
    try:
        test_noise_generation()
        test_waveform_generation()
        test_signal_injection()
        test_utilities()
        test_end_to_end()
        
        print()
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()

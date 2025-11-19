# Galactic Binary Spike and Slab Method

A Python package for detecting and characterizing galactic binary gravitational wave sources using a spike-and-slab Bayesian inference approach. Designed for LISA (Laser Interferometer Space Antenna) data analysis.

## Installation

```bash
# Install the package in development mode
pip install -e .

# Install with development dependencies (includes pytest)
pip install -e .[dev]
```

## Core Modules

The package is organized into several modules, each handling a specific aspect of gravitational wave data analysis:

### 1. Noise Generation (`gb_spike_slab.noise`)

Generates realistic LISA TDI (Time-Delay Interferometry) noise for the A, E, and T channels.

**Usage:**
```python
from gb_spike_slab import NoiseGenerator
import jax
jax.config.update("jax_enable_x64", True)

# Create noise generator
noise_gen = NoiseGenerator(t_obs=365*24*3600, delta_t=5.0, seed=42)

# Generate noise for all channels
noise_A, noise_E, noise_T = noise_gen.generate_all_channels()

# Get frequency grid and PSDs
freqs = noise_gen.get_frequency_grid()
psd_A, psd_E, psd_T = noise_gen.get_psds()
```

**Key Methods:**
- `generate_all_channels()`: Generate noise for A, E, T channels
- `get_frequency_grid()`: Get the frequency grid
- `get_psds()`: Get power spectral densities for each channel

**Run Example:**
```bash
python examples/quick_start.py  # See basic noise generation
```

### 2. Waveform Generation (`gb_spike_slab.waveforms`)

Generates gravitational wave waveforms for galactic binary sources in the frequency domain.

**Usage:**
```python
from gb_spike_slab import WaveformGenerator
import jax.numpy as jnp

# Create waveform generator
wf_gen = WaveformGenerator(t_obs=365*24*3600, n_samples=128)

# Define source parameters: [f0, fdot, amplitude, ecliptic_lat, 
#                            ecliptic_lon, polarization, inclination, initial_phase]
params = jnp.array([
    [0.00136, 8.9e-19, 1e-22, 0.3, -2.7, 3.5, 0.5, 3.0],
    [0.00137, 9.1e-19, 0.9e-22, 0.4, -2.5, 3.6, 0.6, 2.9],
])

# Generate waveforms (vectorized for multiple sources)
A_wf, E_wf, T_wf = wf_gen.generate_waveforms(params)
wf_freqs = wf_gen.get_waveform_frequencies(params)

# Interpolate to target frequency grid
target_freqs = jnp.linspace(0.001, 0.002, 1000)
A_interp = wf_gen.interpolate_waveform(A_wf, wf_freqs, target_freqs)
```

**Key Methods:**
- `generate_waveforms(params)`: Generate waveforms for one or more sources
- `get_waveform_frequencies(params)`: Get frequency grids for waveforms
- `interpolate_waveform(waveform, wf_freqs, target_freqs)`: Interpolate to target frequencies

**Run Example:**
```bash
python examples/benchmark_waveforms.py  # Benchmark waveform generation
```

### 3. Signal Injection (`gb_spike_slab.injection`)

Injects gravitational wave signals into noise and manages frequency-domain data.

**Usage:**
```python
from gb_spike_slab import SignalInjector

# Create injector
injector = SignalInjector(noise_freqs=freqs, t_obs=365*24*3600)

# Inject signals
data_A, data_E, data_T = injector.inject_signals(
    noise_A, noise_E, noise_T,
    A_wf, E_wf, T_wf,
    wf_freqs
)

# Get individual interpolated waveforms
(data_A, data_E, data_T), interpolated_wfs = injector.inject_signals(
    noise_A, noise_E, noise_T,
    A_wf, E_wf, T_wf,
    wf_freqs,
    return_individual=True
)

# Crop to frequency band
f_min, f_max = 0.001, 0.002
cropped_A, cropped_E, cropped_T, cropped_freqs = injector.crop_data(
    data_A, data_E, data_T, f_min, f_max
)
```

**Key Methods:**
- `inject_signals()`: Inject waveforms into noise
- `crop_data()`: Crop data to a specific frequency range

**Run Example:**
```bash
python examples/example_usage.py  # Full end-to-end example with visualization
```

### 4. Utilities (`gb_spike_slab.utils`)

Provides utility functions for frequency grid management, SNR computation, and likelihood calculations.

**Usage:**
```python
from gb_spike_slab import FrequencyGrid
from gb_spike_slab.utils import compute_snr, compute_likelihood

# Frequency grid
grid = FrequencyGrid(t_obs=365*24*3600, delta_t=5.0)
idx = grid.get_bin_index(0.001)  # Get bin index for frequency
idx_low, idx_high = grid.get_frequency_range(0.001, 0.002)

# Compute SNR
snr = compute_snr(signal, psd, t_obs)

# Compute log-likelihood
log_like = compute_likelihood(data, template, psd, t_obs)
```

### 5. Bayesian Inference (`gb_spike_slab.inference`)

Provides NumPyro models for Bayesian parameter estimation.

**Available Models:**
- `single_source_model`: Infer parameters for a single source
- `multi_source_model`: Infer parameters for a fixed number of sources
- `spike_slab_model`: Infer parameters with variable number of sources (spike-and-slab prior)

**Run Examples:**
```bash
python examples/numpyro_example.py           # Multi-source inference
python examples/numpyro_spike_slab_example.py  # Spike-and-slab inference
```

## Spike-and-Slab Example: `numpyro_spike_slab_example.py`

The spike-and-slab method allows Bayesian inference to determine how many galactic binary sources are present in the data, rather than requiring the number to be specified in advance. Each potential source has a binary indicator variable: when the indicator is 0 (spike), the source is not present; when it's 1 (slab), the source is present and its parameters are sampled. This approach enables automatic model selection, where sources with low inclusion probability are effectively excluded, while providing full posterior distributions for all parameters. The example generates synthetic data with multiple sources, runs MCMC inference using NumPyro with mixed sampling (NUTS for continuous parameters and DiscreteHMCGibbs for binary indicators), and produces plots showing inclusion probabilities, the number of active sources, and parameter posteriors. Run with `python examples/numpyro_spike_slab_example.py`.

## Other Examples

### Quick Start (`quick_start.py`)
Minimal example showing noise generation, waveform creation, and signal injection.

```bash
python examples/quick_start.py
```

### Full Example (`example_usage.py`)
Complete workflow with visualization of multiple sources, frequency cropping, and SNR computation.

```bash
python examples/example_usage.py
```

### Multi-Source Inference (`numpyro_example.py`)
Bayesian inference for a fixed number of sources (3 sources).

```bash
python examples/numpyro_example.py
```

### Benchmarking (`benchmark_waveforms.py`)
Performance benchmarking for waveform generation and interpolation.

```bash
python examples/benchmark_waveforms.py
```

### Profiling (`profile_waveforms.py`)
JAX profiling to identify performance bottlenecks.

```bash
python examples/profile_waveforms.py
tensorboard --logdir=./jax_profiler_output
```

## Testing

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run with output capture disabled
pytest -s
```

## Dependencies

- `numpy >= 1.20`
- `jax >= 0.4.0`
- `jaxlib >= 0.4.0`
- `matplotlib >= 3.0`
- `interpax`
- `jaxgb`
- `numpyro` (for Bayesian inference examples)

## Citation

If you use this package in your research, please cite appropriately.

## License

See LICENSE file for details.

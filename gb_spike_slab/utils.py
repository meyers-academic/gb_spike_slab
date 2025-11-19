"""
Utility functions and classes for gravitational wave data analysis.
"""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional


class FrequencyGrid:
    """
    Manage frequency grids for gravitational wave data analysis.
    
    Parameters
    ----------
    t_obs : float
        Observation time in seconds
    delta_t : float
        Time sampling interval in seconds
    """
    
    def __init__(self, t_obs: float, delta_t: float):
        self.t_obs = t_obs
        self.delta_t = delta_t
        self.df = 1.0 / t_obs  # Frequency resolution
        self.f_nyquist = 1.0 / (2.0 * delta_t)  # Nyquist frequency
        
        # Create frequency grid
        self.freqs = np.arange(0, self.f_nyquist + self.df, self.df)
        
    def get_bin_index(self, frequency: float) -> int:
        """
        Get the frequency bin index closest to a given frequency.
        
        Parameters
        ----------
        frequency : float
            Target frequency in Hz
            
        Returns
        -------
        index : int
            Closest frequency bin index
        """
        return int(np.argmin(np.abs(self.freqs - frequency)))
    
    def get_frequency_range(self, f_min: float, f_max: float) -> Tuple[int, int]:
        """
        Get the bin indices for a frequency range.
        
        Parameters
        ----------
        f_min, f_max : float
            Minimum and maximum frequencies in Hz
            
        Returns
        -------
        idx_low, idx_high : tuple of int
            Starting and ending bin indices
        """
        idx_low = self.get_bin_index(f_min)
        idx_high = self.get_bin_index(f_max)
        return idx_low, idx_high
    
    def crop_to_range(self, f_min: float, f_max: float) -> np.ndarray:
        """
        Get a cropped frequency array.
        
        Parameters
        ----------
        f_min, f_max : float
            Minimum and maximum frequencies in Hz
            
        Returns
        -------
        cropped_freqs : np.ndarray
            Frequency array within the specified range
        """
        idx_low, idx_high = self.get_frequency_range(f_min, f_max)
        return self.freqs[idx_low:idx_high]
    
    @property
    def n_bins(self) -> int:
        """Total number of frequency bins."""
        return len(self.freqs)
    
    def __len__(self) -> int:
        """Total number of frequency bins."""
        return self.n_bins
    
    def __getitem__(self, key):
        """Access frequency values by index."""
        return self.freqs[key]


def compute_snr(
    signal: jnp.ndarray,
    psd: jnp.ndarray,
    t_obs: float
) -> jnp.ndarray:
    """
    Compute the signal-to-noise ratio (SNR) for a signal.
    
    This function is JAX-jittable. Zero PSD values are handled automatically.
    
    Parameters
    ----------
    signal : jnp.ndarray
        Complex frequency-domain signal
    psd : jnp.ndarray
        One-sided power spectral density
    t_obs : float
        Observation time in seconds
        
    Returns
    -------
    snr : jnp.ndarray
        Signal-to-noise ratio (scalar)
    """
    # Compute signal power
    signal_power = signal * jnp.conj(signal)
    
    # Handle zero PSD values: replace zeros with a small value to avoid division by zero
    # This is safe because if PSD is zero, there's no noise contribution anyway
    psd_safe = jnp.where(psd > 0, psd, jnp.inf)
    
    # Inner product: <h|h> = 4 * sum( |h(f)|^2 / S(f) * df )
    # For discrete FFT: df = 1/T, so we get 4/T * sum(...)
    integrand = signal_power / psd_safe
    inner_product = 4.0 * jnp.real(jnp.sum(integrand)) / t_obs
    
    # Ensure non-negative and take square root
    snr = jnp.sqrt(jnp.maximum(inner_product, 0.0))
    
    return snr


def compute_likelihood(
    data: jnp.ndarray,
    template: jnp.ndarray,
    psd: jnp.ndarray,
    t_obs: float
) -> jnp.ndarray:
    """
    Compute the log-likelihood for a template.
    
    This function is JAX-jittable. Zero PSD values are handled automatically.
    
    Parameters
    ----------
    data : jnp.ndarray
        Complex frequency-domain data
    template : jnp.ndarray
        Complex frequency-domain template waveform
    psd : jnp.ndarray
        One-sided power spectral density
    t_obs : float
        Observation time in seconds
        
    Returns
    -------
    log_likelihood : jnp.ndarray
        Log-likelihood value (scalar)
    """
    residual = data - template
    residual_power = residual * jnp.conj(residual)
    
    # Handle zero PSD values: replace zeros with infinity so those terms contribute nothing
    psd_safe = jnp.where(psd > 0, psd, jnp.inf)
    
    # Log-likelihood = -0.5 * <r|r> where <r|r> = 4 * sum(|r(f)|^2 / S(f)) / T
    integrand = residual_power / psd_safe
    inner_product = 4.0 * jnp.real(jnp.sum(integrand)) / t_obs
    log_likelihood = -0.5 * inner_product
    
    return log_likelihood


def combine_channels(
    data_A: np.ndarray,
    data_E: np.ndarray,
    psd_A: np.ndarray,
    psd_E: np.ndarray,
    t_obs: float,
    weights: Optional[Tuple[float, float]] = None
) -> Tuple[float, np.ndarray]:
    """
    Combine multiple TDI channels using optimal weighting.
    
    Parameters
    ----------
    data_A, data_E : np.ndarray
        Data for each channel
    psd_A, psd_E : np.ndarray
        Power spectral densities for each channel
    t_obs : float
        Observation time in seconds
    weights : Optional[Tuple[float, float]]
        Custom weights for each channel. If None, uses optimal weighting.
        
    Returns
    -------
    combined_snr : float
        Combined SNR across channels
    optimal_weights : np.ndarray
        Weights used for combination
    """
    if weights is None:
        # Optimal weights are inversely proportional to noise variance
        weight_A = 1.0 / psd_A
        weight_E = 1.0 / psd_E
    else:
        weight_A = weights[0] * np.ones_like(psd_A)
        weight_E = weights[1] * np.ones_like(psd_E)
    
    # Normalize weights
    total_weight = weight_A + weight_E
    weight_A /= total_weight
    weight_E /= total_weight
    
    # Compute combined SNR
    snr_A_sq = 4.0 * np.real(np.sum(data_A * np.conj(data_A) / psd_A)) / t_obs
    snr_E_sq = 4.0 * np.real(np.sum(data_E * np.conj(data_E) / psd_E)) / t_obs
    
    combined_snr = np.sqrt(snr_A_sq + snr_E_sq)
    
    return combined_snr, np.array([weight_A, weight_E])


def safe_log10(x: np.ndarray, floor: float = 1e-50) -> np.ndarray:
    """
    Compute log10 with a floor to avoid infinities.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    floor : float, optional
        Minimum value to use (default: 1e-50)
        
    Returns
    -------
    log10_x : np.ndarray
        Log10 of input with floor applied
    """
    return np.log10(np.maximum(x, floor))


def validate_parameters(params: jnp.ndarray) -> bool:
    """
    Validate parameter array for physical constraints.
    
    Parameters
    ----------
    params : jnp.ndarray
        Parameter array of shape (n_sources, 8)
        
    Returns
    -------
    valid : bool
        True if all parameters are physically valid
    """
    params = jnp.atleast_2d(params)
    
    # Check frequency is positive
    if jnp.any(params[:, 0] <= 0):
        return False
    
    # Check amplitude is positive
    if jnp.any(params[:, 2] <= 0):
        return False
    
    # Check angles are in valid ranges
    # Ecliptic latitude: [-π/2, π/2]
    if jnp.any(jnp.abs(params[:, 3]) > np.pi/2):
        return False
    
    # Ecliptic longitude: [-π, π]
    if jnp.any(jnp.abs(params[:, 4]) > np.pi):
        return False
    
    # Polarization: [0, 2π]
    if jnp.any((params[:, 5] < 0) | (params[:, 5] > 2*np.pi)):
        return False
    
    # Inclination: [0, π]
    if jnp.any((params[:, 6] < 0) | (params[:, 6] > np.pi)):
        return False
    
    # Initial phase: [0, 2π]
    if jnp.any((params[:, 7] < 0) | (params[:, 7] > 2*np.pi)):
        return False
    
    return True

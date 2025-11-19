"""
Signal injection module for adding waveforms to noise data.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, Callable


class SignalInjector:
    """
    Inject gravitational wave signals into noise data.
    
    This class handles:
    1. Adding injected signals to noise realizations
    2. Managing multiple simultaneous injections
    
    Note: Waveform interpolation is handled by WaveformGenerator.interpolate_waveform()
    for better JAX-jittability and performance.
    
    Parameters
    ----------
    noise_freqs : np.ndarray
        Frequency grid of the noise data
    t_obs : float
        Observation time in seconds
    """
    
    def __init__(self, noise_freqs: np.ndarray, t_obs: float):
        self.noise_freqs = noise_freqs
        self.t_obs = t_obs
        self.n_freqs = len(noise_freqs)
    
    def inject_signals(
        self,
        noise_A: np.ndarray,
        noise_E: np.ndarray,
        noise_T: np.ndarray,
        waveforms_A: jnp.ndarray,
        waveforms_E: jnp.ndarray,
        waveforms_T: jnp.ndarray,
        wf_freqs: jnp.ndarray,
        waveform_generator: Optional[object] = None,
        return_individual: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject signals into noise data after interpolation.
        
        Parameters
        ----------
        noise_A, noise_E, noise_T : np.ndarray
            Noise realizations for each TDI channel, shape (n_freqs,)
        waveforms_A, waveforms_E, waveforms_T : jnp.ndarray
            Waveforms for each channel, shape (n_sources, n_samples)
        wf_freqs : jnp.ndarray
            Frequency grid for waveforms, shape (n_sources, n_samples)
        waveform_generator : WaveformGenerator, optional
            WaveformGenerator instance to use for interpolation. If None, creates
            a temporary one (less efficient for repeated calls).
        return_individual : bool, optional
            If True, also return individual interpolated waveforms (default: False)
            
        Returns
        -------
        injected_A, injected_E, injected_T : tuple of np.ndarray
            Data with injected signals, shape (n_freqs,)
        interpolated_wfs : optional, only if return_individual=True
            Dictionary with keys 'A', 'E', 'T' containing interpolated waveforms
            for each channel, shape (n_sources, n_freqs)
        """
        # Ensure waveforms are 2D
        waveforms_A = jnp.atleast_2d(waveforms_A)
        waveforms_E = jnp.atleast_2d(waveforms_E)
        waveforms_T = jnp.atleast_2d(waveforms_T)
        wf_freqs = jnp.atleast_2d(wf_freqs)
        
        n_sources = waveforms_A.shape[0]
        
        # Get or create waveform generator for interpolation
        if waveform_generator is None:
            # Create a temporary one - not ideal but works
            from claude_gb_toolkit import WaveformGenerator
            # We need n_samples, estimate from waveform shape
            n_samples = waveforms_A.shape[1]
            waveform_generator = WaveformGenerator(t_obs=self.t_obs, n_samples=n_samples)
        
        # Interpolate all waveforms to noise frequency grid using vectorized method
        interp_A = waveform_generator.interpolate_waveform(
            waveforms_A, wf_freqs, self.noise_freqs
        )
        interp_E = waveform_generator.interpolate_waveform(
            waveforms_E, wf_freqs, self.noise_freqs
        )
        interp_T = waveform_generator.interpolate_waveform(
            waveforms_T, wf_freqs, self.noise_freqs
        )
        
        # Sum all sources
        total_signal_A = jnp.sum(interp_A, axis=0)
        total_signal_E = jnp.sum(interp_E, axis=0)
        total_signal_T = jnp.sum(interp_T, axis=0)
        
        # Add to noise
        injected_A = np.array(noise_A) + np.array(total_signal_A)
        injected_E = np.array(noise_E) + np.array(total_signal_E)
        injected_T = np.array(noise_T) + np.array(total_signal_T)
        
        if return_individual:
            interpolated_wfs = {
                'A': np.array(interp_A),
                'E': np.array(interp_E),
                'T': np.array(interp_T)
            }
            return (injected_A, injected_E, injected_T), interpolated_wfs
        
        return injected_A, injected_E, injected_T
    
    def inject_signals_direct(
        self,
        noise_A: np.ndarray,
        noise_E: np.ndarray,
        noise_T: np.ndarray,
        waveforms_A: jnp.ndarray,
        waveforms_E: jnp.ndarray,
        waveforms_T: jnp.ndarray,
        kmin: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Inject signals directly without interpolation (for aligned frequency grids).
        
        This method is faster but requires that the waveform frequency grid
        is a subset of the noise frequency grid starting at index kmin.
        
        Parameters
        ----------
        noise_A, noise_E, noise_T : np.ndarray
            Noise realizations for each TDI channel, shape (n_freqs,)
        waveforms_A, waveforms_E, waveforms_T : jnp.ndarray
            Waveforms for each channel, shape (n_sources, n_samples)
        kmin : np.ndarray
            Starting frequency bin index for each source, shape (n_sources,)
            
        Returns
        -------
        injected_A, injected_E, injected_T : tuple of np.ndarray
            Data with injected signals, shape (n_freqs,)
        """
        # Ensure waveforms are 2D
        waveforms_A = jnp.atleast_2d(waveforms_A)
        waveforms_E = jnp.atleast_2d(waveforms_E)
        waveforms_T = jnp.atleast_2d(waveforms_T)
        kmin = np.atleast_1d(kmin)
        
        n_sources = waveforms_A.shape[0]
        n_samples = waveforms_A.shape[1]
        
        # Copy noise
        injected_A = np.copy(noise_A)
        injected_E = np.copy(noise_E)
        injected_T = np.copy(noise_T)
        
        # Add each source directly at its frequency bin
        for i in range(n_sources):
            k = int(kmin[i])
            injected_A[k:k+n_samples] += np.array(waveforms_A[i])
            injected_E[k:k+n_samples] += np.array(waveforms_E[i])
            injected_T[k:k+n_samples] += np.array(waveforms_T[i])
            
        return injected_A, injected_E, injected_T
    
    def crop_data(
        self,
        data_A: np.ndarray,
        data_E: np.ndarray,
        data_T: np.ndarray,
        f_min: float,
        f_max: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop data to a specific frequency range.
        
        Parameters
        ----------
        data_A, data_E, data_T : np.ndarray
            Data for each TDI channel
        f_min, f_max : float
            Minimum and maximum frequencies (Hz)
            
        Returns
        -------
        cropped_A, cropped_E, cropped_T : tuple of np.ndarray
            Cropped data
        cropped_freqs : np.ndarray
            Cropped frequency grid
        """
        idx_low = np.argmin(np.abs(self.noise_freqs - f_min))
        idx_high = np.argmin(np.abs(self.noise_freqs - f_max))
        
        cropped_A = data_A[idx_low:idx_high]
        cropped_E = data_E[idx_low:idx_high]
        cropped_T = data_T[idx_low:idx_high]
        cropped_freqs = self.noise_freqs[idx_low:idx_high]
        
        return cropped_A, cropped_E, cropped_T, cropped_freqs

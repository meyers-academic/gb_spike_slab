"""
Waveform generation module for LISA galactic binaries.
"""

import jax.numpy as jnp
import jax
from jaxgb import jaxgb
from lisaorbits import EqualArmlengthOrbits
from typing import Tuple, Optional
import numpy as np
from interpax import interp1d


class WaveformGenerator:
    """
    Generate gravitational waveforms for galactic binaries using JAX.
    
    This class handles vectorized waveform generation for multiple sources
    simultaneously, computing the response in LISA TDI channels.
    
    Parameters
    ----------
    t_obs : float
        Observation time in seconds
    n_samples : int
        Number of frequency samples per waveform
    use_x64 : bool, optional
        Whether to use 64-bit precision (default: True)
    """
    
    def __init__(self, t_obs: float, n_samples: int, use_x64: bool = True):
        self.t_obs = t_obs
        self.n_samples = n_samples
        
        if use_x64:
            jax.config.update("jax_enable_x64", True)
        
        # Initialize LISA orbits and JaxGB
        self.orbits = EqualArmlengthOrbits()
        self.jgb = jaxgb.JaxGB(self.orbits, t_obs=t_obs, n=n_samples)
    
    def generate_waveforms(
        self, 
        params: jnp.ndarray,
        tdi_combination: str = "AET"
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate waveforms for multiple sources.
        
        Parameters
        ----------
        params : jnp.ndarray
            Parameter array of shape (n_sources, 8), where each row contains:
            [f0, fdot, amplitude, ecliptic_lat, ecliptic_lon, polarization, inclination, initial_phase]
            
            - f0: Initial frequency (Hz)
            - fdot: Frequency derivative (Hz/s)
            - amplitude: Strain amplitude
            - ecliptic_lat: Ecliptic latitude (radians)
            - ecliptic_lon: Ecliptic longitude (radians)
            - polarization: Polarization angle (radians)
            - inclination: Inclination angle (radians)
            - initial_phase: Initial phase (radians)
            
        tdi_combination : str, optional
            TDI channel combination to compute (default: "AET")
            
        Returns
        -------
        A, E, T : tuple of jnp.ndarray
            Complex waveforms for each TDI channel, shape (n_sources, n_samples)
        """
        # Ensure params is 2D
        params = jnp.atleast_2d(params)
        n_sources = params.shape[0]
        
        if params.shape[1] != 8:
            raise ValueError(f"Expected 8 parameters per source, got {params.shape[1]}")
        
        # Generate TDI waveforms
        A, E, T = self.jgb.get_tdi(params, tdi_combination=tdi_combination)
        
        return A, E, T
    
    def get_waveform_frequencies(self, params: jnp.ndarray, t_ref: float = 0.0) -> jnp.ndarray:
        """
        Get the frequency grid for each waveform.
        
        Parameters
        ----------
        params : jnp.ndarray
            Parameter array of shape (n_sources, 8)
        t_ref : float, optional
            Reference time for computing kmin (default: 0.0)
            
        Returns
        -------
        freqs : jnp.ndarray
            Frequency grid for each source, shape (n_sources, n_samples)
        """
        params = jnp.atleast_2d(params)
        
        # Extract f0 and fdot
        f0 = params[:, 0]
        fdot = params[:, 1]
        
        # Compute minimum frequency bin index for each source
        kmin = self.jgb.get_kmin(f0, fdot, t_ref)
        kmin = jnp.atleast_1d(kmin)
        
        # Get frequency grid for each source
        # Reshape kmin to (n_sources, 1) for get_frequency_grid
        if kmin.ndim == 1:
            kmin = kmin.reshape(-1, 1)
        elif kmin.ndim == 0:
            kmin = kmin.reshape(-1, 1)
            
        freqs = self.jgb.get_frequency_grid(kmin)
        
        # Squeeze the extra dimension if get_frequency_grid returns (n_sources, 1, n_samples)
        if freqs.ndim == 3 and freqs.shape[1] == 1:
            freqs = jnp.squeeze(freqs, axis=1)
        
        return freqs
    
    def interpolate_waveform(
        self,
        waveform: jnp.ndarray,
        wf_freqs: jnp.ndarray,
        target_freqs: jnp.ndarray,
        extrap: bool = True
    ) -> jnp.ndarray:
        """
        Interpolate waveforms from their native frequency grid to a target grid.
        
        This method is fully vectorized and JAX-jittable, using vmap to handle
        multiple waveforms efficiently.
        
        Parameters
        ----------
        waveform : jnp.ndarray
            Complex waveform values, shape (n_sources, n_samples) or (n_samples,)
        wf_freqs : jnp.ndarray
            Frequency grid of the waveform, shape (n_sources, n_samples) or (n_samples,)
        target_freqs : jnp.ndarray
            Target frequency grid, shape (n_target_freqs,)
        extrap : bool, optional
            Whether to extrapolate outside the waveform frequency range (default: True).
            Note: Values outside the waveform frequency range are always set to zero
            to maintain JAX-jittability.
            
        Returns
        -------
        interpolated_wf : jnp.ndarray
            Waveform interpolated to target frequencies, with zeros outside the
            original waveform frequency range. Shape: (n_sources, n_target_freqs) or (n_target_freqs,)
        """
        # Convert to JAX arrays
        target_freqs = jnp.asarray(target_freqs)
        waveform = jnp.asarray(waveform)
        wf_freqs = jnp.asarray(wf_freqs)
        
        # Handle single waveform
        if waveform.ndim == 1:
            # Interpolate
            interp_result = interp1d(target_freqs, wf_freqs, waveform, extrap=extrap)
            
            # Zero out values outside the waveform frequency range
            f_min = jnp.min(wf_freqs)
            f_max = jnp.max(wf_freqs)
            mask = (target_freqs >= f_min) & (target_freqs <= f_max)
            return jnp.where(mask, interp_result, 0.0)
        
        # Handle multiple waveforms using vmap for full vectorization
        # Define a function that interpolates a single waveform
        def _interp_single(wf, wf_f):
            # Interpolate
            interp_result = interp1d(target_freqs, wf_f, wf, extrap=extrap)
            
            # Zero out values outside the waveform frequency range
            f_min = jnp.min(wf_f)
            f_max = jnp.max(wf_f)
            mask = (target_freqs >= f_min) & (target_freqs <= f_max)
            return jnp.where(mask, interp_result, 0.0)
        
        # Vectorize over the first dimension (sources)
        return jax.vmap(_interp_single)(waveform, wf_freqs)
    
    def get_single_kmin(self, f0: float, fdot: float, t_ref: float = 0.0) -> int:
        """
        Get the minimum frequency bin index for a single source.
        
        Parameters
        ----------
        f0 : float
            Initial frequency (Hz)
        fdot : float
            Frequency derivative (Hz/s)
        t_ref : float, optional
            Reference time (default: 0.0)
            
        Returns
        -------
        kmin : int
            Minimum frequency bin index
        """
        return self.jgb.get_kmin(f0, fdot, t_ref)
    
    @property
    def frequency_resolution(self) -> float:
        """Get the frequency resolution (df = 1/T_obs)."""
        return 1.0 / self.t_obs
    
    @staticmethod
    def create_parameter_array(
        f0: np.ndarray,
        fdot: np.ndarray,
        amplitude: np.ndarray,
        ecliptic_lat: np.ndarray,
        ecliptic_lon: np.ndarray,
        polarization: np.ndarray,
        inclination: np.ndarray,
        initial_phase: np.ndarray
    ) -> jnp.ndarray:
        """
        Create a properly formatted parameter array from individual parameter arrays.
        
        All input arrays should have the same length (n_sources).
        
        Returns
        -------
        params : jnp.ndarray
            Parameter array of shape (n_sources, 8)
        """
        params = jnp.stack([
            jnp.asarray(f0),
            jnp.asarray(fdot),
            jnp.asarray(amplitude),
            jnp.asarray(ecliptic_lat),
            jnp.asarray(ecliptic_lon),
            jnp.asarray(polarization),
            jnp.asarray(inclination),
            jnp.asarray(initial_phase)
        ], axis=1)
        
        return params

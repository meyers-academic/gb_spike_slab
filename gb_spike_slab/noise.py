"""
Noise generation module for LISA gravitational wave data analysis.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Tuple
import ldc.lisa.noise.noise as lisanoise


class NoiseGenerator:
    """
    Generate frequency-domain noise for LISA TDI channels.
    
    Parameters
    ----------
    t_obs : float
        Observation time in seconds
    delta_t : float
        Time sampling interval in seconds
    seed : Optional[int]
        Random seed for reproducibility
    """
    
    def __init__(self, t_obs: float, delta_t: float, seed: Optional[int] = None):
        self.t_obs = t_obs
        self.delta_t = delta_t
        self.seed = seed
        
        # Compute frequency grid
        self.flow = 1.0 / t_obs
        self.freqs = np.arange(0, 1/delta_t, self.flow)
        
        # Initialize noise PSDs
        self._initialize_psds()
        
    def _initialize_psds(self):
        """Initialize analytical noise PSDs for A, E, T channels."""
        noise = lisanoise.AnalyticNoise(self.freqs[1:])
        
        # Add zero at DC for each channel
        self.psd_A = np.concatenate([[0], noise.psd(tdi2=True, option='A')])
        self.psd_E = np.concatenate([[0], noise.psd(tdi2=True, option='E')])
        self.psd_T = np.concatenate([[0], noise.psd(tdi2=True, option='T')])
        
    @staticmethod
    def generate_fd_noise(psd: np.ndarray, t_obs: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate one realization of complex Gaussian noise in the frequency domain
        consistent with a one-sided PSD.

        Parameters
        ----------
        psd : np.ndarray
            One-sided PSD values (strain^2 / Hz) defined for f >= 0.
            Must have length N//2 + 1, matching np.fft.rfftfreq(N, d=1/fs).
        t_obs : float
            Duration of the corresponding time series (seconds).
            The frequency resolution is df = 1/T.
        seed : Optional[int]
            Random seed for reproducibility.

        Returns
        -------
        noise_fd : np.ndarray (complex)
            Complex Fourier coefficients for positive frequencies only (rFFT format).
            E[|noise_fd[k]|^2] = (T/2) * psd[k].
        """
        rng = np.random.default_rng(seed)
        psd = np.asarray(psd)
        M = len(psd)   # M = N//2 + 1 for the implied time-series length N = 2*(M-1)

        # Standard deviation for complex bins (real & imag parts)
        sigma_complex = np.sqrt(t_obs * psd / 4.0)

        # Draw random real and imaginary parts
        re = rng.normal(scale=sigma_complex)
        im = rng.normal(scale=sigma_complex)
        noise_fd = re + 1j * im

        # DC bin (f=0) should be purely real, variance = T*S1/2
        noise_fd[0] = rng.normal(scale=np.sqrt(t_obs * psd[0] / 2.0))

        # Nyquist bin (if present) should also be real
        if M > 1:
            noise_fd[-1] = rng.normal(scale=np.sqrt(t_obs * psd[-1] / 2.0))

        return noise_fd
    
    def generate_all_channels(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate noise realizations for all TDI channels (A, E, T).
        
        Parameters
        ----------
        seed : Optional[int]
            Random seed for reproducibility. If None, uses the instance seed.
            
        Returns
        -------
        noise_A, noise_E, noise_T : tuple of np.ndarray
            Complex noise realizations for each TDI channel
        """
        use_seed = seed if seed is not None else self.seed
        
        # Generate different noise for each channel by incrementing seed
        if use_seed is not None:
            noise_A = self.generate_fd_noise(self.psd_A, self.t_obs, use_seed)
            noise_E = self.generate_fd_noise(self.psd_E, self.t_obs, use_seed + 1)
            noise_T = self.generate_fd_noise(self.psd_T, self.t_obs, use_seed + 2)
        else:
            noise_A = self.generate_fd_noise(self.psd_A, self.t_obs, None)
            noise_E = self.generate_fd_noise(self.psd_E, self.t_obs, None)
            noise_T = self.generate_fd_noise(self.psd_T, self.t_obs, None)
            
        return noise_A, noise_E, noise_T
    
    def get_frequency_grid(self) -> np.ndarray:
        """Return the frequency grid used for noise generation."""
        return self.freqs
    
    def get_psds(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the power spectral densities for A, E, T channels."""
        return self.psd_A, self.psd_E, self.psd_T

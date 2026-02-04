"""
Frequency-domain chirp templates via time-domain generation + FFT.

Handles all regimes correctly:
- fdot -> 0: recovers sinc (monochromatic)
- large fdot: matches SPA
- transition: exact (within windowing effects)
"""

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


@partial(jit, static_argnames=('N', 'window_type', 'ilow', 'ihigh'))
def chirp_template_fd_fft(
    A: float,
    f0: float,
    fdot: float,
    phi0: float,
    T_obs: float,
    N: int,
    window_type: str = 'tukey',
    ilow: int = 0,
    ihigh: int = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Frequency-domain chirp template via time-domain FFT.
    
    Parameters
    ----------
    A : float
        Amplitude
    f0 : float
        Reference frequency at t=0 [Hz]
    fdot : float
        Frequency derivative [Hz/s]
    phi0 : float
        Phase at t=0 [rad]
    T_obs : float
        Observation time [s]
    N : int
        Number of time samples
    window_type : str
        Window type: 'tukey', 'hann', or 'rect'
        
    Returns
    -------
    freqs : jnp.ndarray
        Frequency array [Hz]
    h_fd : jnp.ndarray (complex)
        Frequency-domain template
    """
    if ihigh is None:
        ihigh = N
    dt = T_obs / N
    
    # Time array centered at t=0
    t = (jnp.arange(N) - N // 2) * dt
    
    # Time-domain phase and signal
    phase = 2 * jnp.pi * f0 * t + jnp.pi * fdot * t**2 + phi0
    h_td = A * jnp.cos(phase)
    
    # Window
    if window_type == 'tukey':
        window = tukey_window(N, alpha=0.1)
    elif window_type == 'hann':
        window = jnp.hanning(N)
    else:  # rect
        window = jnp.ones(N)
    
    h_td_windowed = h_td * window
    
    # FFT with proper normalization
    # ifftshift to handle centered time array, then rfft
    h_fd = jnp.fft.rfft(jnp.fft.ifftshift(h_td_windowed)) * dt
    freqs = jnp.fft.rfftfreq(N, d=dt)
    
    return freqs[ilow:ihigh], h_fd[ilow:ihigh]


def tukey_window(N: int, alpha: float = 0.1) -> jnp.ndarray:
    """
    Tukey (tapered cosine) window.
    
    alpha = 0: rectangular
    alpha = 1: Hann
    """
    n = jnp.arange(N)
    
    # Three regions: taper up, flat, taper down
    width = alpha * (N - 1) / 2
    
    # Left taper
    left = 0.5 * (1 + jnp.cos(jnp.pi * (n / width - 1)))
    # Right taper  
    right = 0.5 * (1 + jnp.cos(jnp.pi * (n / width - 2 / alpha + 1)))
    
    window = jnp.where(n < width, left,
             jnp.where(n > N - 1 - width, right, 1.0))
    
    return window


@partial(jit, static_argnames=('N', 'window_type', 'ilow', 'ihigh'))
def chirp_template_fd_fft_batch(
    params: jnp.ndarray,
    T_obs: float,
    N: int,
    window_type: str = 'tukey',
    ilow: int = 0,
    ihigh: int = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batch version for multiple sources.
    
    Parameters
    ----------
    params : array, shape (N_sources, 4)
        Each row: [A, f0, fdot, phi0]
    T_obs : float
        Observation time [s]
    N : int
        Number of time samples
    window_type : str
        Window type
        
    Returns
    -------
    freqs : jnp.ndarray, shape (N_freq,)
        Frequency array
    h_fd : jnp.ndarray, shape (N_sources, N_freq)
        Templates for each source
    """
    if ihigh is None:
        ihigh = N
    dt = T_obs / N
    t = (jnp.arange(N) - N // 2) * dt  # (N,)
    
    # Extract params: (N_sources, 1) for broadcasting
    # A = params[:, 0, None]
    # f0 = params[:, 1, None]
    # fdot = params[:, 2, None]
    # phi0 = params[:, 3, None]
    
    A = params['A'][:, None]
    f0 = params['f0'][:, None]
    fdot = params['fdot'][:, None]
    phi0 = params['phi0'][:, None]
    # Time-domain signals: (N_sources, N)
    phase = 2 * jnp.pi * f0 * t + jnp.pi * fdot * t**2 + phi0
    h_td = A * jnp.cos(phase)
    
    # Window
    if window_type == 'tukey':
        window = tukey_window(N, alpha=0.1)
    elif window_type == 'hann':
        window = jnp.hanning(N)
    else:
        window = jnp.ones(N)
    
    h_td_windowed = h_td * window  # broadcasts over sources
    
    # FFT each source
    # vmap over the batch dimension
    def single_fft(h):
        return jnp.fft.rfft(jnp.fft.ifftshift(h)) * dt
    
    h_fd = vmap(single_fft)(h_td_windowed)
    freqs = jnp.fft.rfftfreq(N, d=dt)
    
    return freqs[ilow:ihigh], h_fd[:, ilow:ihigh]


import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import jit, vmap

def tophat_fd_waveform(A, f_center, phi_center, fdot, frequencies):
    """
    Simplified tophat waveform
    that just uses SPA. Not the correct FD waveform if fdot * T <= delta_f (or really even when they're close)
    because in that case it's a sinusoid (and you can see this does not reduce properly...)

    For "proper" FD waveform for linear chirps of arbitrary \dot{f}, see `waveform.py`
    """
    delta_f = frequencies[2] - frequencies[1]
    Tobs = 1 / delta_f
    w = fdot * Tobs
    flow = f_center - w/2.
    fhigh = f_center + w/2.
    
    Amp_per_bin = A / jnp.sqrt(w * delta_f)
    
    phases = phi_center + (jnp.pi * (frequencies - f_center)**2) / (delta_f * w)
    # juuuust hack it off. This is probably not right, but `waveform.py` includes
    # the time domain to frequency domain with an fft that does this correctly
    # especially in cases of small fdots. 
    return jnp.where((frequencies >= flow) * (frequencies <= fhigh), Amp_per_bin * jnp.exp(1j * phases), jnp.zeros(frequencies.size))

def get_snrs_theoretical(A, psd_level, frequencies):
    delta_f = frequencies[2] - frequencies[1]
    Tobs = 1 / delta_f
    return jnp.sqrt(4 * Tobs * A**2 / psd_level)
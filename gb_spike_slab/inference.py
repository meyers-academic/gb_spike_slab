"""
Inference module for gravitational wave parameter estimation using NumPyro.

This module provides Bayesian inference capabilities for recovering galactic binary
parameters from LISA data using NumPyro and NUTS sampling.
"""

import jax.numpy as jnp
import jax
import numpy as np
from typing import Tuple, Optional, Dict
from numpyro import sample, plate
from numpyro.distributions import (
    Uniform, Normal, LogNormal, VonMises, Bernoulli
)
import numpyro
from numpyro.contrib.control_flow import scan


def log_likelihood_multi_channel(
    data_A: jnp.ndarray,
    data_E: jnp.ndarray,
    template_A: jnp.ndarray,
    template_E: jnp.ndarray,
    psd_A: jnp.ndarray,
    psd_E: jnp.ndarray,
    t_obs: float
) -> jnp.ndarray:
    """
    Compute log-likelihood for multiple TDI channels.
    
    This is JAX-jittable and works with NumPyro.
    
    Parameters
    ----------
    data_A, data_E : jnp.ndarray
        Complex frequency-domain data for A and E channels
    template_A, template_E : jnp.ndarray
        Complex frequency-domain template waveforms
    psd_A, psd_E : jnp.ndarray
        Power spectral densities for A and E channels
    t_obs : float
        Observation time in seconds
        
    Returns
    -------
    log_likelihood : jnp.ndarray
        Combined log-likelihood across channels
    """
    from .utils import compute_likelihood
    
    # Compute likelihood for each channel
    log_like_A = compute_likelihood(data_A, template_A, psd_A, t_obs)
    log_like_E = compute_likelihood(data_E, template_E, psd_E, t_obs)
    
    # Combined likelihood (assuming independent channels)
    return log_like_A + log_like_E


def single_source_model(
    data_A: jnp.ndarray,
    data_E: jnp.ndarray,
    freqs: jnp.ndarray,
    psd_A: jnp.ndarray,
    psd_E: jnp.ndarray,
    t_obs: float,
    waveform_generator,
    f_min: float,
    f_max: float,
    f0_prior_center: Optional[float] = None,
    f0_prior_width: Optional[float] = None
):
    """
    NumPyro model for single source parameter estimation.
    
    Parameters
    ----------
    data_A, data_E : jnp.ndarray
        Complex frequency-domain data
    freqs : jnp.ndarray
        Frequency grid
    psd_A, psd_E : jnp.ndarray
        Power spectral densities
    t_obs : float
        Observation time
    waveform_generator : WaveformGenerator
        Waveform generator instance
    f_min, f_max : float
        Frequency range for inference
    f0_prior_center : Optional[float]
        Center of f0 prior (default: midpoint of f_min, f_max)
    f0_prior_width : Optional[float]
        Width of f0 prior (default: (f_max - f_min) / 2)
    """
    # Set up f0 prior
    if f0_prior_center is None:
        f0_prior_center = (f_min + f_max) / 2.0
    if f0_prior_width is None:
        f0_prior_width = (f_max - f_min) / 2.0
    
    # Sample parameters
    f0 = sample("f0", Uniform(f0_prior_center - f0_prior_width, 
                              f0_prior_center + f0_prior_width))
    
    # fdot: sample scaled version (1e18 * fdot) with uniform prior
    # This makes fdot ~ order unity for better sampling
    scaled_fdot = sample("scaled_fdot", Uniform(-100.0, 100.0))
    fdot = scaled_fdot / 1e18
    
    # Amplitude: sample in log space
    log_amplitude = sample("log_amplitude", Uniform(-50.0, -20.0))  # Roughly 1e-22 to 1e-9
    amplitude = jnp.exp(log_amplitude)
    
    # Ecliptic latitude: [-π/2, π/2]
    ecliptic_lat = sample("ecliptic_lat", Uniform(-np.pi/2, np.pi/2))
    
    # Ecliptic longitude: [-π, π]
    ecliptic_lon = sample("ecliptic_lon", Uniform(-np.pi, np.pi))
    
    # Polarization: [0, 2π]
    polarization = sample("polarization", Uniform(0, 2*np.pi))
    
    # Inclination: [0, π]
    inclination = sample("inclination", Uniform(0, np.pi))
    
    # Initial phase: [0, 2π]
    initial_phase = sample("initial_phase", Uniform(0, 2*np.pi))
    
    # Stack parameters
    params = jnp.array([f0, fdot, amplitude, ecliptic_lat, ecliptic_lon, 
                        polarization, inclination, initial_phase])
    params = jnp.atleast_2d(params)
    
    # Generate waveform
    A_wf, E_wf, T_wf = waveform_generator.generate_waveforms(params)
    wf_freqs = waveform_generator.get_waveform_frequencies(params)
    
    # Interpolate to data frequency grid
    template_A = waveform_generator.interpolate_waveform(
        A_wf, wf_freqs, freqs
    )[0]  # [0] because single source returns (1, n_freqs)
    
    template_E = waveform_generator.interpolate_waveform(
        E_wf, wf_freqs, freqs
    )[0]
    
    # Compute log-likelihood
    log_like = log_likelihood_multi_channel(
        data_A, data_E, template_A, template_E, psd_A, psd_E, t_obs
    )
    
    # Observe the data
    numpyro.factor("obs", log_like)


def multi_source_model(
    data_A: jnp.ndarray,
    data_E: jnp.ndarray,
    freqs: jnp.ndarray,
    psd_A: jnp.ndarray,
    psd_E: jnp.ndarray,
    t_obs: float,
    waveform_generator,
    n_sources: int,
    f_min: float,
    f_max: float
):
    """
    NumPyro model for multiple source parameter estimation.
    
    Parameters
    ----------
    data_A, data_E : jnp.ndarray
        Complex frequency-domain data
    freqs : jnp.ndarray
        Frequency grid
    psd_A, psd_E : jnp.ndarray
        Power spectral densities
    t_obs : float
        Observation time
    waveform_generator : WaveformGenerator
        Waveform generator instance
    n_sources : int
        Number of sources to infer
    f_min, f_max : float
        Frequency range for inference
    """
    f0_prior_center = (f_min + f_max) / 2.0
    f0_prior_width = (f_max - f_min) / 2.0
    
    # Sample parameters for each source
    with plate("sources", n_sources):
        f0 = sample("f0", Uniform(f0_prior_center - f0_prior_width,
                                  f0_prior_center + f0_prior_width))
        
        # fdot: sample scaled version (1e18 * fdot) with uniform prior
        scaled_fdot = sample("scaled_fdot", Uniform(-100.0, 100.0))
        fdot = scaled_fdot / 1e18
        numpyro.deterministic("fdot", fdot)
        
        # Amplitude: sample in log space
        log_amplitude = sample("log_amplitude", Uniform(-23.0, -19.0))
        amplitude = 10**log_amplitude

        numpyro.deterministic("amplitude", amplitude)

        ecliptic_lat = sample("ecliptic_lat", Uniform(-np.pi/2, np.pi/2))
        ecliptic_lon = sample("ecliptic_lon", Uniform(-np.pi, np.pi))
        polarization = sample("polarization", Uniform(0, 2*np.pi))
        inclination = sample("inclination", Uniform(0, np.pi))
        initial_phase = sample("initial_phase", Uniform(0, 2*np.pi))
    
    # Stack parameters: shape (n_sources, 8)
    params = jnp.stack([
        f0, fdot, amplitude, ecliptic_lat, ecliptic_lon,
        polarization, inclination, initial_phase
    ], axis=1)
    
    # Generate waveforms for all sources
    A_wf, E_wf, T_wf = waveform_generator.generate_waveforms(params)
    wf_freqs = waveform_generator.get_waveform_frequencies(params)
    
    # Interpolate to data frequency grid
    template_A = waveform_generator.interpolate_waveform(
        A_wf, wf_freqs, freqs
    )  # Shape: (n_sources, n_freqs)
    
    template_E = waveform_generator.interpolate_waveform(
        E_wf, wf_freqs, freqs
    )
    
    # Sum templates over sources
    template_A_total = jnp.sum(template_A, axis=0)
    template_E_total = jnp.sum(template_E, axis=0)
    
    # Compute log-likelihood
    log_like = log_likelihood_multi_channel(
        data_A, data_E, template_A_total, template_E_total, 
        psd_A, psd_E, t_obs
    )
    
    # Observe the data
    numpyro.factor("obs", log_like)


def spike_slab_model(
    data_A: jnp.ndarray,
    data_E: jnp.ndarray,
    freqs: jnp.ndarray,
    psd_A: jnp.ndarray,
    psd_E: jnp.ndarray,
    t_obs: float,
    waveform_generator,
    n_max_sources: int,
    f_min: float,
    f_max: float,
):
    """
    NumPyro model with spike-and-slab prior for variable number of sources.
    
    Each potential source has a binary indicator variable. When the indicator
    is 0 (spike), the source is not present. When it's 1 (slab), the source
    parameters are sampled normally.
    
    Parameters
    ----------
    data_A, data_E : jnp.ndarray
        Complex frequency-domain data
    freqs : jnp.ndarray
        Frequency grid
    psd_A, psd_E : jnp.ndarray
        Power spectral densities
    t_obs : float
        Observation time
    waveform_generator : WaveformGenerator
        Waveform generator instance
    n_max_sources : int
        Maximum number of potential sources
    f_min, f_max : float
        Frequency range for inference
    """
    f0_prior_center = (f_min + f_max) / 2.0
    f0_prior_width = (f_max - f_min) / 2.0
    # Prior probability of including each source
    inclusion_prob = sample("inclusion_prob", Uniform(0.0, 1.0))
    
    # Sample indicator variables (spike-and-slab)
    with plate("sources", n_max_sources):
        # Binary indicator: 1 if source is present, 0 if not
        z = sample("z", Bernoulli(probs=inclusion_prob))
        
        # Sample parameters for all sources (even if not present)
        # When z=0, these won't contribute to the likelihood
        f0 = sample("f0", Uniform(f0_prior_center - f0_prior_width,
                                 f0_prior_center + f0_prior_width))
        
        # fdot: sample scaled version (1e18 * fdot) with uniform prior
        scaled_fdot = sample("scaled_fdot", Uniform(-100.0, 100.0))
        fdot = scaled_fdot / 1e18
        numpyro.deterministic("fdot", fdot)
        
        # Amplitude: sample in log space
        log_amplitude = sample("log_amplitude", Uniform(-23.0, -19.0))
        amplitude = 10**log_amplitude
        numpyro.deterministic("amplitude", amplitude)
        
        ecliptic_lat = sample("ecliptic_lat", Uniform(-np.pi/2, np.pi/2))
        ecliptic_lon = sample("ecliptic_lon", Uniform(-np.pi, np.pi))
        polarization = sample("polarization", Uniform(0, 2*np.pi))
        inclination = sample("inclination", Uniform(0, np.pi))
        initial_phase = sample("initial_phase", Uniform(0, 2*np.pi))
    
    # Stack parameters: shape (n_max_sources, 8)
    params = jnp.stack([
        f0, fdot, amplitude, ecliptic_lat, ecliptic_lon,
        polarization, inclination, initial_phase
    ], axis=1)
    
    # Generate waveforms for all potential sources
    A_wf, E_wf, T_wf = waveform_generator.generate_waveforms(params)
    wf_freqs = waveform_generator.get_waveform_frequencies(params)
    
    # Interpolate to data frequency grid
    template_A = waveform_generator.interpolate_waveform(
        A_wf, wf_freqs, freqs
    )  # Shape: (n_max_sources, n_freqs)
    
    template_E = waveform_generator.interpolate_waveform(
        E_wf, wf_freqs, freqs
    )
    
    # Apply spike-and-slab: multiply by indicators
    # Reshape z to (n_max_sources, 1) for broadcasting
    z_reshaped = z[:, jnp.newaxis]
    template_A_active = template_A * z_reshaped
    template_E_active = template_E * z_reshaped
    
    # Sum templates over active sources
    template_A_total = jnp.sum(template_A_active, axis=0)
    template_E_total = jnp.sum(template_E_active, axis=0)
    
    # Compute log-likelihood
    log_like = log_likelihood_multi_channel(
        data_A, data_E, template_A_total, template_E_total, 
        psd_A, psd_E, t_obs
    )
    
    # Observe the data
    numpyro.factor("obs", log_like)
    
    # Track number of active sources
    n_active = jnp.sum(z)
    numpyro.deterministic("n_active_sources", n_active)

"""
Band dataclass and utility functions for multi-band source management.

Each frequency band owns a set of template slots with source parameters
and spike-and-slab indicators. Sources are assigned to bands by their
center frequency.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional


@dataclass
class Band:
    """
    A single frequency band with its source parameters.

    Parameters
    ----------
    k : int
        Band index (0-based).
    f_low : float
        Nominal lower frequency boundary [Hz].
    f_high : float
        Nominal upper frequency boundary [Hz].
    w : float
        Template width in frequency [Hz].
    source_freqs : jnp.ndarray (N_templates,)
        Center frequencies of template slots.
    source_amps : jnp.ndarray (N_templates,)
        Amplitudes.
    source_phases : jnp.ndarray (N_templates,)
        Phases.
    z_indicators : jnp.ndarray (N_templates,)
        Spike-and-slab indicators (0 or 1).
    S_conf : float
        Current confusion noise PSD in this band.
    """
    k: int
    f_low: float
    f_high: float
    w: float
    source_freqs: jnp.ndarray
    source_amps: jnp.ndarray
    source_phases: jnp.ndarray
    z_indicators: jnp.ndarray
    S_conf: float = 0.0

    @property
    def f_low_ext(self):
        """Extended lower boundary (includes buffer)."""
        return self.f_low - self.w / 2

    @property
    def f_high_ext(self):
        """Extended upper boundary (includes buffer)."""
        return self.f_high + self.w / 2

    @property
    def f_center(self):
        """Band center frequency."""
        return 0.5 * (self.f_low + self.f_high)

    @property
    def delta_f(self):
        """Band width."""
        return self.f_high - self.f_low

    @property
    def N_res(self):
        """Number of active (resolved) sources."""
        return jnp.sum(self.z_indicators)

    @property
    def N_templates(self):
        """Total number of template slots."""
        return len(self.z_indicators)

    def left_buffer_mask(self):
        """Mask for sources whose templates extend into the left neighbour."""
        return self.source_freqs < self.f_low + self.w

    def right_buffer_mask(self):
        """Mask for sources whose templates extend into the right neighbour."""
        return self.source_freqs > self.f_high - self.w

    def migrants_left_mask(self):
        """Sources that have moved left of the nominal band."""
        return (self.z_indicators > 0.5) & (self.source_freqs < self.f_low)

    def migrants_right_mask(self):
        """Sources that have moved right of the nominal band."""
        return (self.z_indicators > 0.5) & (self.source_freqs >= self.f_high)


def create_bands(band_edges, w, n_templates_per_band):
    """
    Create a list of empty Band objects from band edges.

    Parameters
    ----------
    band_edges : array (N_bands + 1,)
        Frequency band edges [Hz].
    w : float
        Template width [Hz].
    n_templates_per_band : int
        Number of template slots per band.

    Returns
    -------
    bands : list of Band
    """
    band_edges = jnp.asarray(band_edges)
    n_bands = len(band_edges) - 1
    bands = []
    for k in range(n_bands):
        bands.append(Band(
            k=k,
            f_low=float(band_edges[k]),
            f_high=float(band_edges[k + 1]),
            w=w,
            source_freqs=jnp.zeros(n_templates_per_band),
            source_amps=jnp.zeros(n_templates_per_band),
            source_phases=jnp.zeros(n_templates_per_band),
            z_indicators=jnp.zeros(n_templates_per_band),
        ))
    return bands


def collect_band_summary(bands):
    """
    Collect S_conf, N_res, and f_center from all bands.

    Returns
    -------
    S_conf_all : array (N_bands,)
    N_res_all : array (N_bands,)
    f_centers : array (N_bands,)
    """
    S_conf_all = jnp.array([b.S_conf for b in bands])
    N_res_all = jnp.array([b.N_res for b in bands])
    f_centers = jnp.array([b.f_center for b in bands])
    return S_conf_all, N_res_all, f_centers

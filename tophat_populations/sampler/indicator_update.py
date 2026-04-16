"""
Spike-and-slab z indicator update via Gibbs sampling.

Scans over template slots in a band, conditionally sampling each z_k
given all others, using the matched-filter sufficient statistics (dd, hd, hh)
and a Poisson prior on the total resolved count.

All functions are pure JAX and fully vmappable across bands.
"""

import jax
import jax.numpy as jnp

from tophat_populations.matched_filter import filter_coefficients
from tophat_populations.waveform_simplified import tophat_fd_waveform


def update_indicators(rng_key, z, dd, hd, hh, lambda_res_k,
                      amp_log_odds=None, temperature=1.0):
    """
    Gibbs-sample spike-and-slab indicators given matched-filter statistics.

    Computes the log-odds of z_k=1 vs z_k=0 for ALL k simultaneously using
    a single matrix-vector multiply, then draws all z_k in parallel.

    This is a synchronous (parallel) Gibbs update: each z_k is sampled from
    p(z_k=1 | z_{-k}) using the same current z for all k, rather than the
    sequential scan that conditions on already-updated neighbours.  The
    stationary distribution is identical; mixing is slightly different but
    convergence is the same.  The parallel form is essential for GPU
    performance: it replaces a serial scan of N_templates steps with a
    single batched matmul + elementwise ops.

    Derivation of the vectorised log-odds:
        ll_diff[k] = ll(z_on) - ll(z_off)
                   = hd[k] - (hh @ z)[k] + hh[k,k] * (z[k] - 0.5)
    where z_on = z.at[k].set(1), z_off = z.at[k].set(0).

    Prior difference (Poisson):
        prior_diff[k] = log λ - log(n - z[k] + 1)
    where n = sum(z) and n - z[k] + 1 is the active count when z_k = 1.

    Amplitude prior difference (truncation at A_th):
        amp_log_odds[k] = log_sigmoid((A_k - A_th) / margin)
    This is the log-odds contribution from the soft truncation prior:
    when z_k=1, sources with A < A_th are strongly penalised; when z_k=0
    this term is zero.  Without this, the z update ignores the amplitude
    prior and happily turns on sub-threshold sources.

    Parameters
    ----------
    rng_key : jax PRNGKey
    z : array (N_templates,)
        Current indicator values (0 or 1).
    dd : scalar
        <d|d> inner product.
    hd : array (N_templates,)
        <h_k|d> inner products.
    hh : array (N_templates, N_templates)
        <h_k|h_j> Gram matrix.
    lambda_res_k : scalar
        Poisson rate for the expected number of resolved sources.
    amp_log_odds : array (N_templates,) or None
        Per-template log-odds from the amplitude truncation prior.
        Typically ``log_sigmoid((amps - A_th) / margin)``.
        If None, no amplitude prior is applied (backward compatible).

    Returns
    -------
    z_new : array (N_templates,)
        Updated indicator values.
    """
    hh_diag = jnp.diagonal(hh)              # (N,)
    hz = hh @ z                              # (N,)  — single matmul

    # Log-likelihood difference: z_k=1 vs z_k=0 (all k in parallel)
    ll_diff = hd - hz + hh_diag * (z - 0.5)  # (N,)

    # Poisson prior difference: log P(count_on | λ) - log P(count_off | λ)
    #   count_on[k] = n - z[k] + 1  (sum when z[k] forced to 1)
    n = jnp.sum(z)
    n_when_on = n - z + 1.0                  # (N,)
    prior_diff = jnp.log(lambda_res_k + 1e-30) - jnp.log(n_when_on)

    # Amplitude truncation prior: penalise turning on sources with A < A_th
    log_odds = ll_diff / temperature + prior_diff
    if amp_log_odds is not None:
        log_odds = log_odds + amp_log_odds

    p_on = jax.nn.sigmoid(log_odds)  # (N,)
    return jax.random.bernoulli(rng_key, p_on).astype(jnp.float32)


def compute_filter_coefficients(amps, freqs, phases, fdot,
                                data_fd, psd_total, freq_grid):
    """
    Compute matched-filter sufficient statistics for one band.

    Generates waveforms for all template slots and computes the
    inner products (dd, hd, hh). Pure JAX, vmappable across bands.

    Parameters
    ----------
    amps : array (N_templates,)
        Source amplitudes.
    freqs : array (N_templates,)
        Source center frequencies.
    phases : array (N_templates,)
        Source phases.
    fdot : scalar
        Frequency derivative (template width parameter).
    data_fd : complex array (N_freq,)
        Frequency-domain strain data for this band.
    psd_total : scalar or array (N_freq,)
        Total noise PSD (S_instr + S_conf).
    freq_grid : array (N_freq,)
        Frequency grid for this band.

    Returns
    -------
    dd : scalar
        <d|d>
    hd : array (N_templates,)
        <h_k|d>
    hh : array (N_templates, N_templates)
        <h_k|h_j>
    """
    waveforms = jax.vmap(
        lambda a, fc, phi: tophat_fd_waveform(a, fc, phi, fdot, freq_grid)
    )(amps, freqs, phases)
    return filter_coefficients(waveforms, data_fd, psd_total, freq_grid)

"""
Continuous source parameter update via BlackJAX NUTS.

Updates (A, f, phi) for all template slots in a band, conditioned on
fixed z indicators. Uses the matched-filter log-likelihood and
power-law / uniform priors.

Parameters are internally reparameterized to unconstrained space via
sigmoid transforms, making the log-density smooth and differentiable
everywhere — essential for NUTS's leapfrog integrator.

The prior applies to ALL template slots (not just z=1), which:
  - Is the correct Bayesian formulation (prior is independent of z)
  - Provides gradients in the z=0 dimensions, preventing NUTS divergences
  - Keeps inactive templates at valid parameter values for when z flips to 1

All functions are pure JAX and vmappable across bands.
"""

import jax
import jax.numpy as jnp
import blackjax

from tophat_populations.matched_filter import log_likelihood


# ── Sigmoid reparameterization ──────────────────────────────────────────────
#
# Constrained params (A, f, phi) live on bounded intervals.
# NUTS needs unconstrained params u ∈ R^n with smooth gradients.
#
# Transform:  x = lo + (hi - lo) * sigmoid(u)
# Inverse:    u = logit((x - lo) / (hi - lo))
# Jacobian:   log|dx/du| = log(hi-lo) + log_sigmoid(u) + log_sigmoid(-u)
#
# Each template has 3 params, so theta has shape (3*N,).  The unconstrained
# vector u has the same shape.

_EPS = 1e-7   # clamp for numerical safety in logit


def to_unconstrained(amps, freqs, phases, A_min, A_max, f_low, f_high):
    """Map constrained (A, f, phi) → unconstrained u via logit."""
    def _logit(x, lo, hi):
        t = jnp.clip((x - lo) / (hi - lo), _EPS, 1.0 - _EPS)
        return jnp.log(t / (1.0 - t))

    u_A   = _logit(amps,   A_min, A_max)
    u_f   = _logit(freqs,  f_low, f_high)
    u_phi = _logit(phases, 0.0,   2.0 * jnp.pi)
    return jnp.concatenate([u_A, u_f, u_phi])


def to_constrained(u, N, A_min, A_max, f_low, f_high):
    """Map unconstrained u → constrained (amps, freqs, phases)."""
    u_A   = u[:N]
    u_f   = u[N:2*N]
    u_phi = u[2*N:]

    amps   = A_min + (A_max - A_min) * jax.nn.sigmoid(u_A)
    freqs  = f_low + (f_high - f_low) * jax.nn.sigmoid(u_f)
    phases = 2.0 * jnp.pi * jax.nn.sigmoid(u_phi)
    return amps, freqs, phases


def _log_jacobian(u, N, A_min, A_max, f_low, f_high):
    """Log |det J| for the unconstrained → constrained transform."""
    u_A   = u[:N]
    u_f   = u[N:2*N]
    u_phi = u[2*N:]

    log_jac = (
        jnp.sum(jnp.log(A_max - A_min)
                + jax.nn.log_sigmoid(u_A) + jax.nn.log_sigmoid(-u_A))
        + jnp.sum(jnp.log(f_high - f_low)
                  + jax.nn.log_sigmoid(u_f) + jax.nn.log_sigmoid(-u_f))
        + jnp.sum(jnp.log(2.0 * jnp.pi)
                  + jax.nn.log_sigmoid(u_phi) + jax.nn.log_sigmoid(-u_phi))
    )
    return log_jac


# ── Log-densities ──────────────────────────────────────────────────────────

def make_logdensity_continuous(z, fdot, data_fd, psd_total, freq_grid,
                               A_min, A_max, f_low, f_high, alpha, A_th,
                               temperature=1.0):
    """
    Build log-density for continuous source parameters given fixed z.

    The parameter vector theta = concat(amps, freqs, phases) has
    shape (3 * N_templates,).  Operates in **constrained** space.

    The prior applies to ALL template slots:
      - Active (z=1): truncated power law on [A_th, A_max]
      - Inactive (z=0): power law on [A_min, A_max] (keeps params valid)

    Parameters
    ----------
    A_th : scalar
        Detection threshold amplitude.  Active sources with A < A_th
        are penalised by the truncated prior.
    """
    N = z.shape[0]

    def logdensity(theta):
        amps   = theta[:N]
        freqs  = theta[N:2*N]
        phases = theta[2*N:]

        # Power-law prior on ALL sources
        log_prior_A = -alpha * jnp.log(jnp.clip(amps, _EPS))

        # Active sources: soft truncation at A_th (smooth sigmoid barrier)
        # Penalises A < A_th for z=1 templates, transparent for z=0
        margin = 0.05 * A_th  # ~5% of threshold for smooth transition
        log_truncation = z * jax.nn.log_sigmoid((amps - A_th) / jnp.clip(margin, _EPS))

        log_prior = jnp.sum(log_prior_A + log_truncation)

        # Matched-filter likelihood (z masks inactive sources internally)
        ll = log_likelihood(amps, freqs, phases,
                           fdot * jnp.ones(N), z,
                           data_fd, psd_total, freq_grid)
        return ll / temperature + log_prior

    return logdensity


def make_logdensity_unconstrained(z, fdot, data_fd, psd_total, freq_grid,
                                   A_min, A_max, f_low, f_high, alpha, A_th,
                                   temperature=1.0):
    """
    Build log-density in **unconstrained** space for NUTS.

    Internally transforms u → (A, f, phi), evaluates the likelihood + prior,
    and adds the log-Jacobian correction from the sigmoid reparameterization.

    Active (z=1) sources have a truncated power law prior with lower
    cutoff at A_th (detection threshold).
    """
    N = z.shape[0]

    def logdensity(u):
        amps, freqs, phases = to_constrained(u, N, A_min, A_max, f_low, f_high)

        # Power-law prior on ALL sources (gradient flows to all params)
        log_prior_A = -alpha * jnp.log(jnp.clip(amps, _EPS))

        # Active sources: soft truncation at A_th
        margin = 0.05 * A_th
        log_truncation = z * jax.nn.log_sigmoid((amps - A_th) / jnp.clip(margin, _EPS))

        log_prior = jnp.sum(log_prior_A + log_truncation)

        # For the likelihood, stop gradients for inactive (z=0) sources.
        # The likelihood is independent of z=0 params (z masks them out),
        # but JAX would still autodiff through their waveforms.  This
        # blocks that wasted computation while keeping the prior gradient.
        active = z > 0.5
        amps_ll = jnp.where(active, amps, jax.lax.stop_gradient(amps))
        freqs_ll = jnp.where(active, freqs, jax.lax.stop_gradient(freqs))
        phases_ll = jnp.where(active, phases, jax.lax.stop_gradient(phases))

        # Matched-filter likelihood
        ll = log_likelihood(amps_ll, freqs_ll, phases_ll,
                           fdot * jnp.ones(N), z,
                           data_fd, psd_total, freq_grid)

        # Jacobian of the sigmoid transform
        log_jac = _log_jacobian(u, N, A_min, A_max, f_low, f_high)

        return ll / temperature + log_prior + log_jac

    return logdensity


# ── NUTS step ───────────────────────────────────────────────────────────────

def nuts_step(rng_key, u, logdensity_fn, step_size, inverse_mass_matrix,
              max_num_doublings=3):
    """
    One NUTS step using BlackJAX. Pure JAX, vmappable across bands.

    Parameters
    ----------
    rng_key : PRNGKey
    u : array (3 * N_templates,)
        Current position in unconstrained space.
    logdensity_fn : callable
        u -> scalar log-probability (unconstrained space).
    step_size : scalar
        Leapfrog step size.
    inverse_mass_matrix : array (3 * N_templates,)
        Diagonal inverse mass matrix.
    max_num_doublings : int
        NUTS tree depth (2**d - 1 max leapfrog steps).

    Returns
    -------
    u_new : array (3 * N_templates,)
        Updated position in unconstrained space.
    info : NUTSInfo
        Contains num_integration_steps, is_divergent, energy, etc.
    """
    kernel = blackjax.nuts.build_kernel()
    state = blackjax.nuts.init(u, logdensity_fn)
    new_state, info = kernel(
        rng_key, state, logdensity_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        max_num_doublings=max_num_doublings,
    )
    return new_state.position, info


def build_source_nuts_params(N_templates, step_size=0.1):
    """
    Build default (unadapted) NUTS parameters for source updates.

    These are fallback values used before adaptation runs.  After
    ``adapt_source_nuts_params`` runs, per-band adapted values replace these.
    """
    inverse_mass_matrix = jnp.ones(3 * N_templates)
    return step_size, inverse_mass_matrix


def adapt_source_nuts_params(rng_key, bands, data_fd, freq_grid, S_instr,
                             fdot, A_min, A_max, alpha, rho_th,
                             num_warmup=200, max_num_doublings=3):
    """
    Run BlackJAX window adaptation per band to get adapted (step_size,
    inverse_mass_matrix) for each band.

    This is sequential over bands (adaptation can't be vmapped), but only
    needs to run once at startup (and optionally re-run periodically).
    The adapted parameters are then stacked and vmapped for all subsequent
    NUTS steps.

    Parameters
    ----------
    rng_key : PRNGKey
    bands : list of Band
        Current band state (used for initial position and logdensity).
    data_fd, freq_grid, S_instr, fdot, A_min, A_max, alpha, rho_th :
        Same as ``update_parity_bands``.
    num_warmup : int
        Number of window adaptation steps per band.
    max_num_doublings : int
        Max NUTS tree depth during adaptation.

    Returns
    -------
    step_sizes : array (N_bands,)
        Adapted step size per band.
    inv_mass_matrices : array (N_bands, 3*N_templates)
        Adapted diagonal inverse mass matrix per band.
    """
    from tophat_populations.multiband.residuals import _freq_slice_indices

    N_bands = len(bands)

    step_sizes = []
    inv_mass_matrices = []

    keys = jax.random.split(rng_key, N_bands)

    for b_idx, band in enumerate(bands):
        psd_total = S_instr + band.S_conf

        # Build the unconstrained logdensity for this band
        # (simplified: no neighbor subtraction, uses raw data slice)
        i_lo, i_hi = _freq_slice_indices(freq_grid, band.f_low_ext, band.f_high_ext)
        data_slice = data_fd[i_lo:i_hi]
        freq_slice = freq_grid[i_lo:i_hi]
        delta_f = freq_slice[1] - freq_slice[0]
        A_th = (rho_th / 2.0) * jnp.sqrt(psd_total * delta_f)

        logdensity_fn = make_logdensity_unconstrained(
            band.z_indicators, fdot, data_slice, psd_total, freq_slice,
            A_min, A_max, band.f_low, band.f_high, alpha, A_th,
        )

        # Initial position in unconstrained space
        u0 = to_unconstrained(
            band.source_amps, band.source_freqs, band.source_phases,
            A_min, A_max, band.f_low, band.f_high,
        )

        # Run window adaptation
        warmup_algo = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_fn,
            progress_bar=False,
            max_num_doublings=max_num_doublings,
        )
        (_, kernel_params), _ = warmup_algo.run(
            keys[b_idx], u0, num_steps=num_warmup,
        )

        step_sizes.append(kernel_params['step_size'])
        inv_mass_matrices.append(kernel_params['inverse_mass_matrix'])

    return jnp.array(step_sizes), jnp.stack(inv_mass_matrices)


# ── Legacy API (kept for backwards compatibility) ───────────────────────────

def rmh_step(rng_key, theta, logdensity_fn, sigma):
    """One random-walk Metropolis-Hastings step using BlackJAX."""
    sampler = blackjax.normal_random_walk(logdensity_fn, sigma)
    state = sampler.init(theta)
    new_state, info = sampler.step(rng_key, state)
    return new_state.position, info


def build_proposal_sigma(N_templates, sig_A=0.01, sig_f=1e-6, sig_phi=0.1):
    """Build diagonal proposal covariance matrix for RMH (legacy)."""
    sigmas = jnp.concatenate([
        sig_A * jnp.ones(N_templates),
        sig_f * jnp.ones(N_templates),
        sig_phi * jnp.ones(N_templates),
    ])
    return jnp.diag(sigmas)

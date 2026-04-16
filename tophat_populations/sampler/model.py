"""
NumPyro generative model for multi-band confusion noise + resolved counts.

This model is used in the hyperparameter update step of the Gibbs sampler.
It samples population hyperparameters Λ = (N_tot, α, β) and per-band
latent log S_conf values, then observes N_res per band via the analytic
Poisson factor.

The MDN parameters are frozen (not sampled) — they serve as a pre-trained
prior on log S_conf conditioned on Λ.
"""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from .mdn_prior import MDNPrior
from tophat_populations.noise_emulator.network import compute_lambda_res


def hierarchical_model(
    N_res_obs,
    f_centers,
    mdn_params,
    X_norm_stats,
    S_instr,
    T_obs,
    rho_th,
    A_min,
    A_max,
    delta_f_band,
    f_min,
    f_max,
    lambda_bounds,
    n_components=5,
):
    """
    NumPyro generative model for the population hyperparameter update.

    Generative story:
      1. Draw Λ = (N_tot, α, β) from priors
      2. For each band k:
         a. Draw log S_conf_k ~ MDNPrior(Λ, f_k)
         b. Compute λ_res_k analytically from S_conf_k and Λ
         c. Observe N_res_k ~ Poisson(λ_res_k)

    Parameters
    ----------
    N_res_obs : array (N_bands,)
        Observed resolved source counts per band (from spike-and-slab z_k).
    f_centers : array (N_bands,)
        Band center frequencies.
    mdn_params : dict
        Frozen MDN parameters.
    X_norm_stats : tuple (X_mean, X_std)
        Input normalisation statistics.
    S_instr : array (N_bands,) or float
        Instrumental noise PSD per band.
    T_obs : float
        Observation time [s].
    rho_th : float
        SNR detection threshold.
    A_min, A_max : float
        Amplitude bounds.
    delta_f_band : float
        Band width [Hz].
    f_min, f_max : float
        Full frequency range [Hz].
    lambda_bounds : dict
        Prior ranges for hyperparameters.
    n_components : int
        Number of MDN mixture components.
    """
    N_bands = len(f_centers)
    S_instr = jnp.broadcast_to(jnp.asarray(S_instr, dtype=float), (N_bands,))

    N_lo, N_hi = lambda_bounds['N_tot']
    a_lo, a_hi = lambda_bounds['alpha']
    b_lo, b_hi = lambda_bounds['beta']

    # --- Population hyperparameters ---
    # N_tot prior: log-uniform => sample log10(N_tot) uniformly
    log10_N_tot = numpyro.sample(
        "log10_N_tot",
        dist.Uniform(jnp.log10(N_lo), jnp.log10(N_hi))
    )
    N_tot = numpyro.deterministic("N_tot", 10**log10_N_tot)

    alpha = numpyro.sample("alpha", dist.Uniform(a_lo, a_hi))
    beta = numpyro.sample("beta", dist.Uniform(b_lo, b_hi))

    # Lambda vector for MDN input: (log10(N_tot), alpha, beta)
    Lambda = jnp.array([log10_N_tot, alpha, beta])

    # --- Per-band latent variables and observations ---
    for k in range(N_bands):
        # Sample log S_conf from the MDN's learned prior
        log_S_conf_k = numpyro.sample(
            f"log_S_conf_{k}",
            MDNPrior(mdn_params, Lambda, f_centers[k], X_norm_stats,
                     n_components=n_components)
        )

        # Deterministic: compute Poisson rate from S_conf and Λ
        lambda_res_k = compute_lambda_res(
            log_S_conf_k, N_tot, alpha, beta, f_centers[k],
            S_instr[k], T_obs, rho_th, A_min, A_max,
            delta_f_band, f_min, f_max
        )
        numpyro.deterministic(f"lambda_res_{k}", lambda_res_k)

        # Observe: N_res from spike-and-slab indicator counts
        numpyro.sample(
            f"N_res_{k}",
            dist.Poisson(jnp.clip(lambda_res_k, 1e-10)),
            obs=N_res_obs[k]
        )

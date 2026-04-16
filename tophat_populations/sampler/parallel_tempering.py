"""
Parallel tempering (replica exchange MCMC) for the multi-band Gibbs sampler.

Runs K chains at geometrically spaced temperatures T_1=1 (cold) through
T_K=T_max.  After each Gibbs iteration, adjacent chains propose Metropolis
swaps.  Only the T=1 chain contributes posterior samples; hot chains
explore more freely and inject diversity via swaps.

Temperature scales the likelihood only:
    log_density = ll / T + log_prior + log_jacobian

The hyper update (S_conf, lambda_res) runs only on the cold chain.
Results are broadcast to all chains — they share the same noise model.

Band updates are GPU-parallel: all K chains × N_parity bands are stacked
into a single vmap call (one GPU kernel launch) via update_parity_bands_pt.
Migration and hyper updates remain sequential (rare, cheap, O(K*N_bands)).
"""

import copy

import jax
import jax.numpy as jnp

from tophat_populations.multiband.migration import migrate_sources
from tophat_populations.sampler.band_update import update_parity_bands_pt
from tophat_populations.sampler.hyper_update import (
    HyperUpdater, warmup_and_sample, hyper_update_step,
)


# ── Temperature schedule ────────────────────────────────────────────────────

def geometric_temperature_schedule(n_chains, T_max):
    """
    Build a geometric temperature ladder T[0]=1.0, ..., T[K-1]=T_max.

    Parameters
    ----------
    n_chains : int
        Number of chains (K).
    T_max : float
        Maximum temperature for the hottest chain.

    Returns
    -------
    temperatures : jnp.ndarray, shape (K,)
    """
    if n_chains == 1:
        return jnp.array([1.0])
    exponents = jnp.linspace(0.0, 1.0, n_chains)
    return T_max ** exponents


# ── Replica swap ────────────────────────────────────────────────────────────

def replica_swap_accept(rng_key, ll_i, ll_j, T_i, T_j):
    """
    Metropolis acceptance for swapping chains i and j.

    Accept probability:
        alpha = min(1, exp((beta_i - beta_j) * (ll_j - ll_i)))

    where beta = 1/T (inverse temperature).

    Parameters
    ----------
    rng_key : PRNGKey
    ll_i, ll_j : float
        Untempered log-likelihoods of chains i and j.
    T_i, T_j : float
        Temperatures of chains i and j.

    Returns
    -------
    accept : bool
    """
    beta_i = 1.0 / T_i
    beta_j = 1.0 / T_j
    log_alpha = (beta_i - beta_j) * (ll_j - ll_i)
    u = jax.random.uniform(rng_key)
    return jnp.log(u) < log_alpha


def swap_band_states(bands_i, bands_j):
    """
    Swap the source state between two chains' band lists.

    Creates new Band objects with swapped arrays (z, amps, freqs, phases,
    S_conf).  Band geometry (k, f_low, f_high, w) is preserved.

    Parameters
    ----------
    bands_i, bands_j : list of Band
        Band lists for chains i and j. Must have the same length.

    Returns
    -------
    new_bands_i, new_bands_j : list of Band
        Swapped band lists.
    """
    from tophat_populations.multiband.bands import Band

    new_i, new_j = [], []
    for bi, bj in zip(bands_i, bands_j):
        # i gets j's source state
        new_i.append(Band(
            k=bi.k, f_low=bi.f_low, f_high=bi.f_high, w=bi.w,
            source_freqs=bj.source_freqs, source_amps=bj.source_amps,
            source_phases=bj.source_phases, z_indicators=bj.z_indicators,
            S_conf=bj.S_conf,
        ))
        # j gets i's source state
        new_j.append(Band(
            k=bj.k, f_low=bj.f_low, f_high=bj.f_high, w=bj.w,
            source_freqs=bi.source_freqs, source_amps=bi.source_amps,
            source_phases=bi.source_phases, z_indicators=bi.z_indicators,
            S_conf=bi.S_conf,
        ))
    return new_i, new_j


# ── PT Gibbs iteration ─────────────────────────────────────────────────────

def pt_gibbs_iteration(
    rng_key,
    chain_bands,
    chain_lambda_res,
    temperatures,
    data_fd,
    freq_grid,
    S_instr,
    fdot,
    A_min,
    A_max,
    alpha,
    rho_th,
    proposal_sigmas,
    hyper_model_fn,
    hyper_model_kwargs_fn,
    extract_hyper_fn,
    hyper_updater=None,
    hyper_num_warmup=200,
    hyper_num_samples=1,
    current_hyper_values=None,
    iteration=0,
    propose_swap=True,
):
    """
    One parallel-tempered Gibbs iteration across K chains.

    Band updates are GPU-parallel: all K chains' same-parity bands are
    stacked into a single vmap call via update_parity_bands_pt.  Migration
    is a cheap Python loop per chain.  Hyper update runs only on the cold
    chain and results are broadcast.

    Steps:
      1. Even band update (all K chains in one vmap)
      2. Migrate even→odd (Python loop over K chains)
      3. Odd band update (all K chains in one vmap)
      4. Migrate odd→even (Python loop over K chains)
      5. Hyper update (cold chain only, broadcast to hot chains)
      6. Replica exchange swaps (adjacent pairs)

    Parameters
    ----------
    rng_key : PRNGKey
    chain_bands : list of K lists of Band
        chain_bands[c] is the band list for chain c.
    chain_lambda_res : list of K dicts
        chain_lambda_res[c] is the lambda_res_per_band dict for chain c.
    temperatures : array (K,)
        Temperature schedule.
    iteration : int
        Current iteration number (determines even/odd swap parity).
    [remaining params: see gibbs_iteration]

    Returns
    -------
    chain_bands : list of K lists of Band
        Updated band lists for all chains.
    chain_lambda_res : list of K dicts
        Updated lambda_res dicts for all chains.
    cold_hyper_samples : dict
        Hyper samples from the cold chain.
    diagnostics : dict
        Per-chain log-likelihoods, swap acceptance, hyper_updater.
    """
    K = len(chain_bands)
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

    # ── 1. Even band update (all K chains, one GPU kernel) ──────────────
    chain_even = [[b for b in chain_bands[c] if b.k % 2 == 0] for c in range(K)]
    chain_odd = [[b for b in chain_bands[c] if b.k % 2 == 1] for c in range(K)]

    updated_even, acc_even, ll_even = update_parity_bands_pt(
        k1, chain_even, chain_odd,
        data_fd, freq_grid, S_instr, fdot,
        chain_lambda_res, A_min, A_max, alpha, rho_th,
        proposal_sigmas, temperatures,
    )

    # Write even results back into chain_bands
    for c in range(K):
        for band in updated_even[c]:
            chain_bands[c][band.k] = band

    # ── 2. Migrate even→odd (Python loop, cheap) ────────────────────────
    n_mig_even = []
    for c in range(K):
        even_c = [chain_bands[c][b.k] for b in chain_even[c]]
        n_mig_even.append(migrate_sources(even_c, chain_bands[c]))

    # ── 3. Odd band update (all K chains, one GPU kernel) ───────────────
    # Refresh band lists after migration
    chain_odd_fresh = [[chain_bands[c][b.k] for b in chain_odd[c]] for c in range(K)]
    chain_even_fresh = [[chain_bands[c][b.k] for b in chain_even[c]] for c in range(K)]

    updated_odd, acc_odd, ll_odd = update_parity_bands_pt(
        k3, chain_odd_fresh, chain_even_fresh,
        data_fd, freq_grid, S_instr, fdot,
        chain_lambda_res, A_min, A_max, alpha, rho_th,
        proposal_sigmas, temperatures,
    )

    for c in range(K):
        for band in updated_odd[c]:
            chain_bands[c][band.k] = band

    # ── 4. Migrate odd→even (Python loop, cheap) ────────────────────────
    n_mig_odd = []
    for c in range(K):
        odd_c = [chain_bands[c][b.k] for b in chain_odd[c]]
        n_mig_odd.append(migrate_sources(odd_c, chain_bands[c]))

    # ── 5. Hyper update (cold chain only) ───────────────────────────────
    cold_hyper_samples = {}
    model_kwargs = hyper_model_kwargs_fn(chain_bands[0])

    if hyper_updater is None:
        cold_hyper_samples, hyper_updater = warmup_and_sample(
            k5,
            model_fn=hyper_model_fn,
            model_kwargs=model_kwargs,
            init_values=current_hyper_values,
            num_warmup=hyper_num_warmup,
            num_samples=hyper_num_samples,
        )
    else:
        cold_hyper_samples, hyper_updater = hyper_update_step(
            k5, hyper_updater, model_kwargs,
            num_samples=hyper_num_samples,
        )

    extract_hyper_fn(cold_hyper_samples, chain_bands[0], chain_lambda_res[0])

    # Broadcast cold chain's S_conf and lambda_res to hot chains
    cold_bands = chain_bands[0]
    cold_lambda = chain_lambda_res[0]
    for c in range(1, K):
        for b_idx in range(len(cold_bands)):
            chain_bands[c][b_idx].S_conf = cold_bands[b_idx].S_conf
        chain_lambda_res[c] = copy.copy(cold_lambda)

    # ── 6. Compute per-chain total LL and (optionally) replica exchange ──
    chain_ll = []
    for c in range(K):
        ll_total = jnp.sum(ll_even[c]) + jnp.sum(ll_odd[c])
        chain_ll.append(ll_total)

    swap_accepted = []

    if propose_swap:
        swap_keys = jax.random.split(k4, K)
        # Even iterations: swap (0,1), (2,3), ...
        # Odd iterations:  swap (1,2), (3,4), ...
        start = iteration % 2
        for pair_idx in range(start, K - 1, 2):
            i, j = pair_idx, pair_idx + 1
            accept = replica_swap_accept(
                swap_keys[pair_idx],
                chain_ll[i], chain_ll[j],
                float(temperatures[i]), float(temperatures[j]),
            )
            swap_accepted.append((i, j, bool(accept)))

            if accept:
                chain_bands[i], chain_bands[j] = swap_band_states(
                    chain_bands[i], chain_bands[j])
                chain_ll[i], chain_ll[j] = chain_ll[j], chain_ll[i]

    diagnostics = {
        'chain_ll': chain_ll,
        'swap_accepted': swap_accepted,
        'hyper_updater': hyper_updater,
        'n_migrations_even': n_mig_even,
        'n_migrations_odd': n_mig_odd,
    }

    return chain_bands, chain_lambda_res, cold_hyper_samples, diagnostics

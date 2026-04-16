"""
Top-level Gibbs sampler combining multi-band source updates with
MDN-based hyperparameter inference.

Each Gibbs iteration:
  1. Even band update (parallel via vmap): z indicators + source params
  2. Migrate even → odd
  3. Odd band update (parallel via vmap): z indicators + source params
  4. Migrate odd → even
  5. Hyperparameter update: Λ + {log S_conf_k} via BlackJAX NUTS
"""

import copy

import jax
import jax.numpy as jnp
from tqdm import tqdm

from tophat_populations.multiband.bands import collect_band_summary
from tophat_populations.multiband.migration import migrate_sources
from tophat_populations.sampler.band_update import update_parity_bands
from tophat_populations.sampler.hyper_update import (
    HyperUpdater, warmup_and_sample, hyper_update_step,
)


def default_extract_hyper(hyper_samples, all_bands, lambda_res_per_band):
    """
    Default extractor for ``hierarchical_model`` output.

    Reads ``log_S_conf_{k}`` and ``lambda_res_{k}`` keys from the NUTS
    samples and pushes them into the band objects and lambda_res dict.

    Parameters
    ----------
    hyper_samples : dict
        Posterior samples from ``hyper_update_step``.  Each value has
        shape ``(num_samples,)``; we take the last sample.
    all_bands : list of Band
        Mutated in place: ``S_conf`` updated per band.
    lambda_res_per_band : dict
        Mutated in place: Poisson rate updated per band.
    """
    n_bands = len(all_bands)
    for k in range(n_bands):
        log_s_key = f"log_S_conf_{k}"
        if log_s_key in hyper_samples:
            log_s = hyper_samples[log_s_key][-1]
            all_bands[k].S_conf = jnp.exp(log_s)

        lr_key = f"lambda_res_{k}"
        if lr_key in hyper_samples:
            lambda_res_per_band[k] = hyper_samples[lr_key][-1]


def gibbs_iteration(
    rng_key,
    all_bands,
    data_fd,
    freq_grid,
    S_instr,
    fdot,
    A_min,
    A_max,
    alpha,
    rho_th,
    lambda_res_per_band,
    proposal_sigmas,
    hyper_model_fn,
    hyper_model_kwargs_fn,
    extract_hyper_fn=default_extract_hyper,
    hyper_updater=None,
    hyper_num_warmup=200,
    hyper_num_samples=1,
    current_hyper_values=None,
    profile=False,
    temperature=1.0,
    skip_hyper=False,
):
    """
    Run one full Gibbs iteration.

    Parameters
    ----------
    rng_key : jax PRNGKey
    all_bands : list of Band
        All frequency bands.
    data_fd : complex array (N_freq_bins,)
        Full frequency-domain strain data.
    freq_grid : array (N_freq_bins,)
        Frequency grid.
    S_instr : float
        Instrumental noise PSD.
    fdot : float
        Frequency derivative (template width parameter).
    A_min, A_max : float
        Amplitude bounds.
    alpha : float
        Power-law index for amplitude prior.
    rho_th : float
        SNR detection threshold (used for A_th truncation in source prior).
    lambda_res_per_band : dict
        Mutable dict mapping band index -> Poisson rate for resolved count.
        Updated in place via ``extract_hyper_fn``.
    step_sizes : array (N_bands,)
        Per-band adapted NUTS leapfrog step sizes.
    inv_mass_matrices : array (N_bands, 3*N_templates)
        Per-band adapted diagonal inverse mass matrices for NUTS.
    hyper_model_fn : callable
        NumPyro model for the hyperparameter update (e.g. ``hierarchical_model``).
    hyper_model_kwargs_fn : callable
        ``hyper_model_kwargs_fn(all_bands) -> dict`` builds kwargs for
        ``hyper_model_fn`` from the current band state.
    extract_hyper_fn : callable
        ``extract_hyper_fn(hyper_samples, all_bands, lambda_res_per_band)``
        pushes NUTS samples back into band objects. Defaults to
        ``default_extract_hyper`` (reads ``log_S_conf_{k}`` / ``lambda_res_{k}``).
    hyper_updater : HyperUpdater or None
        Reusable NUTS sampler. If None, one is created (with warmup).
    hyper_num_warmup : int
        Window adaptation steps (only used when creating a new updater).
    hyper_num_samples : int
        NUTS samples per iteration (typically 1 for Gibbs).
    current_hyper_values : dict or None
        Current hyperparameter values for warm-starting (only used when
        creating a new updater).

    Returns
    -------
    all_bands : list of Band
        Updated bands.
    hyper_samples : dict
        Hyperparameter posterior samples from this iteration.
    diagnostics : dict
        Migration counts, acceptance rates, timing, and hyper_updater.
    """
    k1, k2, k3, k4, k5 = jax.random.split(rng_key, 5)

    if profile:
        import time as _time
        _t0 = _time.perf_counter()

    even_bands = [b for b in all_bands if b.k % 2 == 0]
    odd_bands = [b for b in all_bands if b.k % 2 == 1]

    # ── 1. Even band update (parallel via single vmap) ────────────────────
    updated_even, accepted_even, ll_even = update_parity_bands(
        k1, even_bands, odd_bands, data_fd, freq_grid,
        S_instr, fdot, lambda_res_per_band,
        A_min, A_max, alpha, rho_th, proposal_sigmas,
        temperature=temperature,
    )
    for i, band in enumerate(updated_even):
        all_bands[even_bands[i].k] = band

    if profile:
        jax.block_until_ready(accepted_even)
        _t1 = _time.perf_counter()

    # ── 2. Migrate even → odd ────────────────────────────────────────────
    even_bands_updated = [all_bands[b.k] for b in even_bands]
    n_mig_even = migrate_sources(even_bands_updated, all_bands)

    # ── 3. Odd band update (parallel via single vmap) ─────────────────────
    odd_bands = [all_bands[b.k] for b in odd_bands]   # refresh after migration
    even_bands_updated = [all_bands[b.k] for b in even_bands]  # refresh
    updated_odd, accepted_odd, ll_odd = update_parity_bands(
        k3, odd_bands, even_bands_updated, data_fd, freq_grid,
        S_instr, fdot, lambda_res_per_band,
        A_min, A_max, alpha, rho_th, proposal_sigmas,
        temperature=temperature,
    )
    for i, band in enumerate(updated_odd):
        all_bands[odd_bands[i].k] = band

    if profile:
        jax.block_until_ready(accepted_odd)
        _t2 = _time.perf_counter()

    # ── 4. Migrate odd → even ────────────────────────────────────────────
    odd_bands_updated = [all_bands[b.k] for b in odd_bands]
    n_mig_odd = migrate_sources(odd_bands_updated, all_bands)

    # ── 5. Hyperparameter update ─────────────────────────────────────────
    hyper_samples = {}
    if not skip_hyper:
        model_kwargs = hyper_model_kwargs_fn(all_bands)

        if hyper_updater is None:
            # First iteration: trace model + warmup
            hyper_samples, hyper_updater = warmup_and_sample(
                k5,
                model_fn=hyper_model_fn,
                model_kwargs=model_kwargs,
                init_values=current_hyper_values,
                num_warmup=hyper_num_warmup,
                num_samples=hyper_num_samples,
            )
        else:
            # Fast path: JIT-compiled NUTS step, no warmup
            hyper_samples, hyper_updater = hyper_update_step(
                k5, hyper_updater, model_kwargs,
                num_samples=hyper_num_samples,
            )

        if profile:
            jax.block_until_ready(hyper_samples)
            _t3 = _time.perf_counter()

        # Push hyper samples back into bands and lambda_res_per_band
        extract_hyper_fn(hyper_samples, all_bands, lambda_res_per_band)

    # Total untempered log-likelihood across all bands
    log_likelihood = jnp.sum(ll_even) + jnp.sum(ll_odd)

    diagnostics = {
        'n_migrations_even': n_mig_even,
        'n_migrations_odd': n_mig_odd,
        'rmh_accepted_even': accepted_even,
        'rmh_accepted_odd': accepted_odd,
        'hyper_updater': hyper_updater,
        'log_likelihood': log_likelihood,
    }

    if profile:
        if not skip_hyper:
            diagnostics['timing'] = {
                'band_updates_even': _t1 - _t0,
                'band_updates_odd': _t2 - _t1,
                'hyper_update': _t3 - _t2,
            }
        else:
            diagnostics['timing'] = {
                'band_updates_even': _t1 - _t0,
                'band_updates_odd': _t2 - _t1,
            }

    return all_bands, hyper_samples, diagnostics


def run_gibbs(
    rng_key,
    all_bands,
    data_fd,
    freq_grid,
    S_instr,
    fdot,
    A_min,
    A_max,
    alpha,
    rho_th,
    lambda_res_per_band,
    proposal_sigmas,
    hyper_model_fn,
    hyper_model_kwargs_fn,
    extract_hyper_fn=default_extract_hyper,
    n_iterations=1000,
    n_burnin=200,
    hyper_num_warmup=200,
    hyper_rewarmup_interval=100,
    show_progress=True,
    init_hyper_values=None,
):
    """
    Run the full Gibbs sampler.

    The NUTS hyperparameter kernel is traced and adapted once on the first
    iteration, then reused for all subsequent iterations.

    Parameters
    ----------
    [see gibbs_iteration for source-update parameters]
    hyper_model_fn : callable
        NumPyro model for the hyperparameter update.
    hyper_model_kwargs_fn : callable
        ``hyper_model_kwargs_fn(all_bands) -> dict`` builds kwargs for
        ``hyper_model_fn`` from the current band state.
    extract_hyper_fn : callable
        Pushes NUTS samples back into bands. See ``default_extract_hyper``.
    n_iterations : int
        Total number of Gibbs iterations.
    n_burnin : int
        Number of burn-in iterations (not saved).
    hyper_num_warmup : int
        Window adaptation steps (first iteration only).
    show_progress : bool
        Show progress bar.

    Returns
    -------
    chain : dict
        Posterior chain. Each value is array of shape (n_samples,).
    all_bands : list of Band
        Final state of all bands.
    diagnostics : list of dict
        Per-iteration diagnostics.
    """
    chain = {}
    n_active_trace = []   # (n_iterations, n_bands) — active z indicators
    lambda_res_trace = [] # (n_iterations, n_bands) — Poisson rate from hyper model
    all_diagnostics = []
    hyper_updater = None

    iterator = range(n_iterations)
    if show_progress:
        iterator = tqdm(iterator, desc="Gibbs sampler")

    for i in iterator:
        rng_key, subkey = jax.random.split(rng_key)

        # Periodically re-warmup the hyper NUTS to adapt to the
        # changing posterior geometry (N_res evolves as sources improve)
        if hyper_updater is not None and i > 0 and i % hyper_rewarmup_interval == 0:
            hyper_updater = None  # forces re-warmup next iteration

        all_bands, hyper_samples, diag = gibbs_iteration(
            subkey, all_bands, data_fd, freq_grid,
            S_instr, fdot, A_min, A_max, alpha, rho_th,
            lambda_res_per_band, proposal_sigmas,
            hyper_model_fn=hyper_model_fn,
            hyper_model_kwargs_fn=hyper_model_kwargs_fn,
            extract_hyper_fn=extract_hyper_fn,
            hyper_updater=hyper_updater,
            hyper_num_warmup=hyper_num_warmup,
            hyper_num_samples=1,
            current_hyper_values=init_hyper_values if hyper_updater is None else None,
        )

        # Cache updater for next iteration
        hyper_updater = diag['hyper_updater']

        # Record active z indicators and lambda_res per band every iteration
        n_active_trace.append([b.N_res for b in all_bands])
        n_bands = len(all_bands)
        lambda_res_trace.append([
            hyper_samples[f'lambda_res_{k}'][-1]
            if f'lambda_res_{k}' in hyper_samples
            else lambda_res_per_band.get(k, 0.0)
            for k in range(n_bands)
        ])

        all_diagnostics.append(diag)

        # Record post-burnin samples
        if i >= n_burnin:
            for key, val in hyper_samples.items():
                if key not in chain:
                    chain[key] = []
                chain[key].append(val[-1])

    # Convert lists to arrays
    chain = {k: jnp.array(v) for k, v in chain.items()}
    traces = {
        'n_active': jnp.array(n_active_trace),      # (n_iterations, n_bands)
        'lambda_res': jnp.array(lambda_res_trace),   # (n_iterations, n_bands)
    }

    return chain, all_bands, all_diagnostics, traces


def run_gibbs_pt(
    rng_key,
    all_bands,
    data_fd,
    freq_grid,
    S_instr,
    fdot,
    A_min,
    A_max,
    alpha,
    rho_th,
    lambda_res_per_band,
    proposal_sigmas,
    hyper_model_fn,
    hyper_model_kwargs_fn,
    extract_hyper_fn=default_extract_hyper,
    n_iterations=1000,
    n_burnin=200,
    hyper_num_warmup=200,
    hyper_rewarmup_interval=100,
    show_progress=True,
    init_hyper_values=None,
    n_chains=4,
    T_max=10.0,
    swap_interval=10,
):
    """
    Run the Gibbs sampler with parallel tempering.

    Runs K chains at geometrically spaced temperatures.  Only the cold
    chain (T=1) contributes posterior samples.  Adjacent chains propose
    Metropolis replica-exchange swaps every ``swap_interval`` iterations,
    giving each chain time to decorrelate at its temperature before
    proposing an exchange.

    Parameters
    ----------
    [see run_gibbs for shared parameters]
    n_chains : int
        Number of temperature chains.
    T_max : float
        Maximum temperature for the hottest chain.
    swap_interval : int
        Number of Gibbs iterations between replica-exchange swap proposals.
        Each chain runs this many iterations at its own temperature before
        swaps are attempted, ensuring decorrelated samples for the swap
        acceptance criterion.

    Returns
    -------
    chain : dict
        Posterior chain from the cold chain.
    all_bands : list of Band
        Final state of cold chain bands.
    all_diagnostics : list of dict
        Per-iteration diagnostics (includes swap info).
    traces : dict
        Traces of n_active, lambda_res, chain_ll, swap_rates.
    """
    from tophat_populations.sampler.parallel_tempering import (
        geometric_temperature_schedule,
        pt_gibbs_iteration,
    )

    temperatures = geometric_temperature_schedule(n_chains, T_max)
    print(f"PT temperature schedule: {temperatures}")

    # Initialize K chain copies
    chain_bands = [all_bands]
    chain_lambda_res = [lambda_res_per_band]
    for _ in range(1, n_chains):
        chain_bands.append(copy.deepcopy(all_bands))
        chain_lambda_res.append(copy.copy(lambda_res_per_band))

    chain = {}
    n_active_trace = []
    lambda_res_trace = []
    chain_ll_trace = []
    swap_rate_trace = []
    all_diagnostics = []
    hyper_updater = None

    iterator = range(n_iterations)
    if show_progress:
        iterator = tqdm(iterator, desc="PT Gibbs sampler")

    for i in iterator:
        rng_key, subkey = jax.random.split(rng_key)

        # Periodically re-warmup hyper NUTS (adapted step sizes go stale)
        if hyper_updater is not None and i > 0 and i % hyper_rewarmup_interval == 0:
            hyper_updater = None

        do_swap = (i > 0) and (i % swap_interval == 0)

        chain_bands, chain_lambda_res, hyper_samples, diag = pt_gibbs_iteration(
            subkey,
            chain_bands, chain_lambda_res, temperatures,
            data_fd, freq_grid, S_instr, fdot,
            A_min, A_max, alpha, rho_th,
            proposal_sigmas,
            hyper_model_fn=hyper_model_fn,
            hyper_model_kwargs_fn=hyper_model_kwargs_fn,
            extract_hyper_fn=extract_hyper_fn,
            hyper_updater=hyper_updater,
            hyper_num_warmup=hyper_num_warmup,
            hyper_num_samples=1,
            current_hyper_values=init_hyper_values if hyper_updater is None else None,
            iteration=i,
            propose_swap=do_swap,
        )

        hyper_updater = diag['hyper_updater']

        # Record cold chain state
        cold_bands = chain_bands[0]
        n_bands = len(cold_bands)
        n_active_trace.append([b.N_res for b in cold_bands])
        lambda_res_trace.append([
            hyper_samples[f'lambda_res_{k}'][-1]
            if f'lambda_res_{k}' in hyper_samples
            else chain_lambda_res[0].get(k, 0.0)
            for k in range(n_bands)
        ])

        # Per-chain log-likelihoods
        chain_ll_trace.append(diag['chain_ll'])

        # Swap acceptance rates (NaN for non-swap iterations)
        swaps = diag['swap_accepted']
        if swaps:
            swap_rate_trace.append(
                sum(1 for _, _, acc in swaps if acc) / len(swaps))
        else:
            swap_rate_trace.append(float('nan'))

        all_diagnostics.append(diag)

        # Record post-burnin cold chain samples
        if i >= n_burnin:
            for key, val in hyper_samples.items():
                if key not in chain:
                    chain[key] = []
                chain[key].append(val[-1])

    chain = {k: jnp.array(v) for k, v in chain.items()}
    traces = {
        'n_active': jnp.array(n_active_trace),
        'lambda_res': jnp.array(lambda_res_trace),
        'chain_ll': jnp.array(chain_ll_trace),
        'swap_rate': jnp.array(swap_rate_trace),
    }

    return chain, chain_bands[0], all_diagnostics, traces

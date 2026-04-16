"""
End-to-end Gibbs sampler for multi-band galactic binary separation.

Loads a trained MDN emulator, generates synthetic strain data with
injected sources, initialises bands, and runs the full Gibbs sampler.

Usage
-----
    conda activate gb_spike_slab
    python tophat_populations/run_gibbs_sampler.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tophat_populations.waveform_simplified import tophat_fd_waveform
from tophat_populations.multiband.bands import create_bands, collect_band_summary
from tophat_populations.sampler.source_update import build_proposal_sigma
from tophat_populations.sampler.model import hierarchical_model
from tophat_populations.sampler.gibbs import run_gibbs, run_gibbs_pt, gibbs_iteration, default_extract_hyper
from tophat_populations.noise_emulator.iterative_sub import (
    sample_amplitudes, sample_frequencies,
)
from tophat_populations.noise_emulator.network import gated_mdn_forward

# ── Paths ────────────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__),
                         "noise_emulator", "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "mdn_model.pkl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "gibbs_output")


# ── Waveform function used throughout ────────────────────────────────────────

def waveform_fn(amp, freq, phase, freq_grid):
    """Wrapper matching the signature expected by compute_residuals_complex."""
    # fdot is baked in via the frequency grid's delta_f
    delta_f = freq_grid[1] - freq_grid[0]
    T_obs = 1.0 / delta_f
    fdot_local = FDOT  # uses module-level constant
    return tophat_fd_waveform(amp, freq, phase, fdot_local, freq_grid)


# ── Data generation ──────────────────────────────────────────────────────────

def generate_synthetic_data(
    freq_grid, N_sources_true, alpha_true, beta_true, A_min, A_max,
    fdot, S_instr, seed=42
):
    """
    Generate synthetic frequency-domain strain with injected sources.

    Parameters
    ----------
    beta_true : float
        Frequency distribution taper: p(f) ~ (1 - beta*f). beta=0 is uniform.

    Returns
    -------
    data_fd : complex array (N_freq,)
    true_amps : array (N_sources,)
    true_freqs : array (N_sources,)
    true_phases : array (N_sources,)
    """
    rng = np.random.default_rng(seed)

    f_min, f_max = float(freq_grid[0]), float(freq_grid[-1])

    # Draw source parameters using the same distributions as the training data
    true_amps = sample_amplitudes(rng, N_sources_true, alpha_true, A_min, A_max)
    true_freqs = sample_frequencies(rng, N_sources_true, beta_true, f_min, f_max)
    true_phases = rng.uniform(0, 2 * np.pi, N_sources_true)

    # Build signal
    true_amps_j = jnp.array(true_amps)
    true_freqs_j = jnp.array(true_freqs)
    true_phases_j = jnp.array(true_phases)

    vmap_waveform = jax.vmap(
        lambda a, fc, phi: tophat_fd_waveform(a, fc, phi, fdot, freq_grid)
    )
    signal = jnp.sum(vmap_waveform(true_amps_j, true_freqs_j, true_phases_j), axis=0)

    # Add Gaussian noise
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    delta_f = freq_grid[1] - freq_grid[0]
    noise_sigma = jnp.sqrt(S_instr / (2 * delta_f))
    noise = (jax.random.normal(k1, freq_grid.shape) * noise_sigma
             + 1j * jax.random.normal(k2, freq_grid.shape) * noise_sigma)

    data_fd = signal + noise
    return data_fd, true_amps, true_freqs, true_phases


# ── Band initialisation ─────────────────────────────────────────────────────

def initialise_bands(band_edges, w, n_templates_per_band,
                     true_amps, true_freqs, true_phases,
                     snr_threshold, S_instr, T_obs,
                     mdn_params, X_norm_stats, n_components,
                     log10_N_tot, alpha, beta):
    """
    Create bands and seed them with the brightest injected sources.

    Uses the MDN to predict S_conf from the true population hyperparameters
    (log10_N_tot, alpha, beta) at each band's center frequency.  This gives
    each band the correct noise floor from the start.
    """
    from tophat_populations.noise_emulator.network import gated_mdn_forward

    bands = create_bands(band_edges, w, n_templates_per_band)
    Lambda = jnp.array([log10_N_tot, alpha, beta])

    for band in bands:
        # Predict S_conf from MDN at this band's center frequency
        f_k = band.f_center
        x_raw = jnp.concatenate([Lambda, jnp.atleast_1d(f_k)])
        X_mean, X_std = X_norm_stats
        x_norm = (x_raw - X_mean) / X_std
        log_pi, mu, _, _ = gated_mdn_forward(
            mdn_params, x_norm[None, :], n_components)
        # MDN mixture mean: E[log_S_conf] = sum(pi_j * mu_j)
        pi = jnp.exp(log_pi[0])
        log_S_conf_mean = float(jnp.sum(pi * mu[0]))
        S_conf_band = float(jnp.exp(log_S_conf_mean))
        band.S_conf = S_conf_band

        # Detection threshold from total noise
        delta_f = 1.0 / T_obs
        S_total = S_instr + S_conf_band
        A_th = (snr_threshold / 2.0) * np.sqrt(S_total * delta_f)

        # All injected sources in this band
        in_band = ((true_freqs >= band.f_low) & (true_freqs < band.f_high))
        band_amps = true_amps[in_band]
        band_freqs = true_freqs[in_band]
        band_phases = true_phases[in_band]
        n_total = len(band_amps)

        # Place the brightest resolved sources (above A_th, up to N_templates)
        resolved = band_amps >= A_th
        resolved_idx = np.where(resolved)[0]
        n_resolved = len(resolved_idx)
        if n_resolved > 0:
            amp_order = np.argsort(band_amps[resolved_idx])[::-1]
            n_slot = min(n_resolved, band.N_templates)
            idx = resolved_idx[amp_order[:n_slot]]

            band.source_amps = band.source_amps.at[:n_slot].set(
                jnp.array(band_amps[idx]))
            band.source_freqs = band.source_freqs.at[:n_slot].set(
                jnp.array(band_freqs[idx]))
            band.source_phases = band.source_phases.at[:n_slot].set(
                jnp.array(band_phases[idx]))
            band.z_indicators = band.z_indicators.at[:n_slot].set(1.0)

        print(f"  Band {band.k}: {n_total} sources, {n_resolved} resolved, "
              f"{min(n_resolved, band.N_templates)} placed, "
              f"S_conf={S_conf_band:.4f}, A_th={A_th:.4e}")

    # Remaining template slots: scatter uniformly in band
    rng = np.random.default_rng(123)
    for band in bands:
        n_active = int(jnp.sum(band.z_indicators))
        n_empty = band.N_templates - n_active
        if n_empty > 0:
            rand_freqs = rng.uniform(band.f_low, band.f_high, n_empty)
            rand_amps = rng.uniform(0.1 * A_MIN, A_MIN, n_empty)
            rand_phases = rng.uniform(0, 2 * np.pi, n_empty)
            band.source_freqs = band.source_freqs.at[n_active:].set(
                jnp.array(rand_freqs))
            band.source_amps = band.source_amps.at[n_active:].set(
                jnp.array(rand_amps))
            band.source_phases = band.source_phases.at[n_active:].set(
                jnp.array(rand_phases))
    return bands


# ── Hyper-model kwargs builder ───────────────────────────────────────────────

def make_hyper_model_kwargs_fn(mdn_params, X_norm_stats, S_instr, T_obs,
                                rho_th, A_min, A_max, delta_f_band,
                                f_min, f_max, lambda_bounds, n_components):
    """
    Return a closure that builds model_kwargs from the current band state.
    This is the glue between the Gibbs loop and the hierarchical model.
    """
    def hyper_model_kwargs_fn(all_bands):
        _, N_res_all, f_centers = collect_band_summary(all_bands)
        return dict(
            N_res_obs=N_res_all,
            f_centers=f_centers,
            mdn_params=mdn_params,
            X_norm_stats=X_norm_stats,
            S_instr=S_instr,
            T_obs=T_obs,
            rho_th=rho_th,
            A_min=A_min,
            A_max=A_max,
            delta_f_band=delta_f_band,
            f_min=f_min,
            f_max=f_max,
            lambda_bounds=lambda_bounds,
            n_components=n_components,
        )
    return hyper_model_kwargs_fn


# ── Diagnostic plots ─────────────────────────────────────────────────────────

def plot_sconf_prior(chain, mdn_params, X_norm_stats, n_components,
                     f_min, f_max, all_bands, output_dir,
                     n_freq=200, n_samples=500, ci=90):
    """
    Plot the posterior predictive S_conf(f).

    For each posterior Lambda sample, evaluates the MDN mean of log S_conf
    at a fine frequency grid.  Shows median + credible interval.
    Overlays the direct chain samples of log_S_conf_k at band centres.
    """
    os.makedirs(output_dir, exist_ok=True)
    X_mean, X_std = X_norm_stats

    log10_N = np.array(chain.get('log10_N_tot', []))
    alphas  = np.array(chain.get('alpha', []))
    betas   = np.array(chain.get('beta', []))
    if len(log10_N) == 0:
        return

    # Subsample posterior for speed
    idx = np.random.choice(len(log10_N), size=min(n_samples, len(log10_N)),
                           replace=False)
    log10_N = log10_N[idx]
    alphas  = alphas[idx]
    betas   = betas[idx]

    f_grid = np.linspace(f_min, f_max, n_freq)

    # Build input matrix: (n_samples, n_freq, 4), then flatten to (n_samples*n_freq, 4)
    Lambda_rep = np.stack([
        np.repeat(log10_N[:, None], n_freq, axis=1),
        np.repeat(alphas[:, None],  n_freq, axis=1),
        np.repeat(betas[:, None],   n_freq, axis=1),
        np.tile(f_grid[None, :],    (len(log10_N), 1)),
    ], axis=-1)  # (n_samples, n_freq, 4)

    flat = Lambda_rep.reshape(-1, 4)
    flat_j = (jnp.array(flat) - X_mean) / X_std

    log_pi, mu, _, _ = gated_mdn_forward(mdn_params, flat_j, n_components)
    pi = jnp.exp(log_pi)
    log_S_mean = jnp.sum(pi * mu, axis=-1)  # MDN mixture mean of log S_conf

    log_S_grid = np.array(log_S_mean).reshape(len(log10_N), n_freq)
    S_grid = np.exp(log_S_grid)

    lo = (100 - ci) / 2
    hi = 100 - lo
    med  = np.median(S_grid, axis=0)
    p_lo = np.percentile(S_grid, lo, axis=0)
    p_hi = np.percentile(S_grid, hi, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(f_grid, p_lo, p_hi, alpha=0.3, label=f'{ci}% CI')
    ax.plot(f_grid, med, lw=1.5, label='Median')

    # Overlay direct chain samples at band centres
    f_centers = np.array([b.f_center for b in all_bands])
    for k, fc in enumerate(f_centers):
        key = f'log_S_conf_{k}'
        if key in chain:
            s_vals = np.exp(np.array(chain[key])[idx[idx < len(chain[key])]])
            if len(s_vals):
                ax.scatter([fc] * len(s_vals), s_vals, s=2, alpha=0.15,
                           color='C2', zorder=3)

    ax.set_yscale('log')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$S_{\rm conf}(f)$')
    ax.set_title(r'Posterior predictive confusion noise $S_{\rm conf}(f)$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sconf_vs_freq.png'), dpi=150)
    plt.close(fig)
    print(f"Saved sconf_vs_freq.png")


def plot_diagnostics(chain, all_bands, true_params, output_dir, traces=None):
    """Save trace plots and source recovery summary."""
    os.makedirs(output_dir, exist_ok=True)

    # Trace plots for hyperparameters
    hyper_keys = ['log10_N_tot', 'alpha', 'beta']
    true_vals = {
        'log10_N_tot': np.log10(true_params['N_sources']),
        'alpha': true_params['alpha'],
        'beta': true_params.get('beta', None),
    }
    true_vals = {k: v for k, v in true_vals.items() if v is not None}

    fig, axes = plt.subplots(len(hyper_keys), 1, figsize=(10, 3 * len(hyper_keys)),
                             sharex=True)
    for ax, key in zip(axes, hyper_keys):
        if key in chain:
            ax.plot(np.array(chain[key]), alpha=0.7)
            ax.set_ylabel(key)
            if key in true_vals:
                ax.axhline(true_vals[key], color='r', ls='--', label='true')
                ax.legend()
    axes[-1].set_xlabel('Post-burnin iteration')
    fig.suptitle('Hyperparameter traces')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hyper_traces.png'), dpi=150)
    plt.close(fig)

    # Posterior histograms for hyperparameters
    fig, axes = plt.subplots(1, len(hyper_keys), figsize=(4 * len(hyper_keys), 4))
    if len(hyper_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, hyper_keys):
        if key in chain and len(chain[key]) > 0:
            samples = np.array(chain[key])
            ax.hist(samples, bins=30, density=True, alpha=0.7, edgecolor='black',
                    linewidth=0.5)
            ax.axvline(np.mean(samples), color='C0', ls='-', lw=1.5,
                       label=f'mean={np.mean(samples):.3f}')
            if key in true_vals:
                ax.axvline(true_vals[key], color='r', ls='--', lw=1.5,
                           label=f'true={true_vals[key]:.3f}')
            ax.set_xlabel(key)
            ax.legend(fontsize=8)
    fig.suptitle('Hyperparameter posteriors')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'hyper_posteriors.png'), dpi=150)
    plt.close(fig)

    # Source counts per band (final state)
    fig, ax = plt.subplots(figsize=(8, 4))
    band_indices = [b.k for b in all_bands]
    n_res = [b.N_res for b in all_bands]
    ax.bar(band_indices, n_res)
    ax.set_xlabel('Band index')
    ax.set_ylabel('N resolved (final)')
    ax.set_title('Resolved source counts per band')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'source_counts.png'), dpi=150)
    plt.close(fig)

    # N_res and lambda_res traces per band
    if traces is not None:
        n_active_arr = np.array(traces['n_active'])     # (n_iterations, n_bands)
        lambda_res_arr = np.array(traces['lambda_res'])  # (n_iterations, n_bands)
        n_bands = n_active_arr.shape[1]

        # Per-band: overlay active z count and lambda_res (Poisson mean)
        fig, axes = plt.subplots(n_bands, 1, figsize=(10, 2.5 * n_bands),
                                 sharex=True)
        if n_bands == 1:
            axes = [axes]
        for k, ax in enumerate(axes):
            ax.plot(n_active_arr[:, k], alpha=0.6, label='N active (z)')
            ax.plot(lambda_res_arr[:, k], alpha=0.8, ls='--',
                    label=r'$\lambda_{\rm res}$ (Poisson mean)')
            ax.set_ylabel(f'Band {k}')
            if k == 0:
                ax.legend(fontsize=8)
        axes[-1].set_xlabel('Iteration')
        fig.suptitle(r'Active indicators vs $\lambda_{\rm res}$ per band')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'nres_per_band_trace.png'), dpi=150)
        plt.close(fig)

        # Total across all bands
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.sum(n_active_arr, axis=1), alpha=0.7, label='Total N active (z)')
        ax.plot(np.sum(lambda_res_arr, axis=1), alpha=0.8, ls='--',
                label=r'Total $\lambda_{\rm res}$')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Count')
        ax.set_title(r'Total active indicators vs $\lambda_{\rm res}$')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'nres_total_trace.png'), dpi=150)
        plt.close(fig)

    # S_conf traces per band
    n_bands = len(all_bands)
    s_conf_keys = [f'log_S_conf_{k}' for k in range(n_bands)]
    available = [k for k in s_conf_keys if k in chain]
    if available:
        fig, ax = plt.subplots(figsize=(10, 4))
        for key in available:
            ax.plot(np.array(chain[key]), alpha=0.5, label=key)
        ax.set_ylabel('log S_conf')
        ax.set_xlabel('Post-burnin iteration')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'sconf_traces.png'), dpi=150)
        plt.close(fig)

    print(f"Diagnostic plots saved to {output_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global FDOT, A_MIN  # used by waveform_fn and initialise_bands

    # ── 1. Load trained MDN ──────────────────────────────────────────────
    print("Loading trained MDN from", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)

    mdn_params = jax.tree.map(jnp.asarray, model_data["params"])
    X_mean = jnp.asarray(model_data["X_mean"])
    X_std = jnp.asarray(model_data["X_std"])
    X_norm_stats = (X_mean, X_std)
    config = model_data["config"]

    # Physical constants from MDN config
    T_OBS = config.get("T_obs", 3.15e7)
    A_MIN = config.get("A_min", 1e-3)
    A_MAX = config.get("A_max", 10.0)
    S_INSTR = config.get("S_instr", 1.1234)
    RHO_TH = config.get("rho_th", 5.0)
    LAMBDA_BOUNDS = config["lambda_bounds"]
    N_COMPONENTS = config["n_components"]

    # Band / frequency range — override here to use a subset of the
    # MDN's trained range.  Set to None to use the full trained config.
    # BAND_EDGES_OVERRIDE = np.array([1e-4, 2e-4, 3e-4, 4e-4])  # 3 bands
    BAND_EDGES_OVERRIDE = None  # uncomment to use full trained config

    if BAND_EDGES_OVERRIDE is not None:
        BAND_EDGES = BAND_EDGES_OVERRIDE
        F_MIN = float(BAND_EDGES[0])
        F_MAX = float(BAND_EDGES[-1])
    else:
        BAND_EDGES = np.array(config["band_edges"])
        F_MIN = config["f_min"]
        F_MAX = config["f_max"]

    DELTA_F_BAND = float(BAND_EDGES[1] - BAND_EDGES[0])
    N_BANDS = len(BAND_EDGES) - 1

    DELTA_F = 1.0 / T_OBS
    W = 0.1 * DELTA_F_BAND  # template width = 10% of band width
    FDOT = W / T_OBS

    # ── 2. Set up frequency grid ─────────────────────────────────────────
    N_FREQ = int((F_MAX - F_MIN) / DELTA_F) + 1
    freq_grid = jnp.linspace(F_MIN, F_MAX, N_FREQ)
    print(f"Frequency grid: {N_FREQ} bins, [{F_MIN:.1e}, {F_MAX:.1e}] Hz")

    # ── 3. Injection parameters ──────────────────────────────────────────
    N_SOURCES_TRUE = 10000
    ALPHA_TRUE = 4.0
    BETA_TRUE = 100   # uniform in frequency
    SEED = 42

    print(f"Injecting {N_SOURCES_TRUE} sources "
          f"(alpha={ALPHA_TRUE}, beta={BETA_TRUE})")
    data_fd, true_amps, true_freqs, true_phases = generate_synthetic_data(
        freq_grid, N_SOURCES_TRUE, ALPHA_TRUE, BETA_TRUE,
        A_MIN, A_MAX, FDOT, S_INSTR, seed=SEED,
    )

    # ── 4. Initialise bands ──────────────────────────────────────────────
    N_TEMPLATES = 50  # template slots per band

    bands = initialise_bands(
        BAND_EDGES, W, N_TEMPLATES,
        true_amps, true_freqs, true_phases,
        RHO_TH, S_INSTR, T_OBS,
        mdn_params, X_norm_stats, N_COMPONENTS,
        jnp.log10(float(N_SOURCES_TRUE)), float(ALPHA_TRUE), float(BETA_TRUE),
    )
    print(f"Created {N_BANDS} bands, {N_TEMPLATES} templates each")
    for b in bands:
        print(f"  Band {b.k}: [{b.f_low:.2e}, {b.f_high:.2e}] Hz, "
              f"{b.N_res} active sources, S_conf={b.S_conf:.2f}")

    # ── 5. Build sampler inputs ──────────────────────────────────────────
    rng_key = jax.random.PRNGKey(0)

    # Per-band RMH proposal sigmas (diagonal, in unconstrained space)
    proposal_sigmas = {
        b.k: build_proposal_sigma(N_TEMPLATES) for b in bands
    }
    # Initialize lambda_res to match the actual number of active sources
    # per band (from the initialization).  Starting at 2.0 creates a
    # self-fulfilling prophecy: the Poisson prior penalizes all but ~2
    # sources, they get turned off, and the hyper model never corrects.
    lambda_res_per_band = {b.k: float(b.N_res) for b in bands}

    # Start hyperparameters at their true values so the sampler
    # doesn't waste burn-in recovering from a bad initial point.
    init_hyper_values = {
        'log10_N_tot': jnp.log10(float(N_SOURCES_TRUE)),
        'alpha': float(ALPHA_TRUE),
        'beta': float(BETA_TRUE),
    }
    for b in bands:
        init_hyper_values[f'log_S_conf_{b.k}'] = jnp.log(jnp.asarray(b.S_conf))

    hyper_model_kwargs_fn = make_hyper_model_kwargs_fn(
        mdn_params, X_norm_stats, S_INSTR, T_OBS, RHO_TH,
        A_MIN, A_MAX, DELTA_F_BAND, F_MIN, F_MAX,
        LAMBDA_BOUNDS, N_COMPONENTS,
    )

    # ── 6. Run Gibbs sampler ─────────────────────────────────────────────
    N_ITERATIONS = 2000
    N_BURNIN = 0
    HYPER_NUM_WARMUP = 100  # window adaptation steps (first iteration only)

    # Parallel tempering config
    N_CHAINS = 1 # number of temperature chains (1 = no PT)
    T_MAX = 1    # maximum temperature for hottest chain

    if N_CHAINS > 1:
        print(f"\nRunning PT Gibbs sampler: {N_ITERATIONS} iterations "
              f"({N_BURNIN} burn-in), {N_CHAINS} chains, T_max={T_MAX}")

        chain, final_bands, diagnostics, traces = run_gibbs_pt(
            rng_key,
            bands,
            data_fd,
            freq_grid,
            S_INSTR,
            FDOT,
            A_MIN,
            A_MAX,
            ALPHA_TRUE,
            RHO_TH,
            lambda_res_per_band,
            proposal_sigmas,
            hyper_model_fn=hierarchical_model,
            hyper_model_kwargs_fn=hyper_model_kwargs_fn,
            extract_hyper_fn=default_extract_hyper,
            n_iterations=N_ITERATIONS,
            n_burnin=N_BURNIN,
            hyper_num_warmup=HYPER_NUM_WARMUP,
            show_progress=True,
            n_chains=N_CHAINS,
            T_max=T_MAX,
            init_hyper_values=init_hyper_values,
        )
    else:
        print(f"\nRunning Gibbs sampler: {N_ITERATIONS} iterations "
              f"({N_BURNIN} burn-in)")

        chain, final_bands, diagnostics, traces = run_gibbs(
            rng_key,
            bands,
            data_fd,
            freq_grid,
            S_INSTR,
            FDOT,
            A_MIN,
            A_MAX,
            ALPHA_TRUE,
            RHO_TH,
            lambda_res_per_band,
            proposal_sigmas,
            hyper_model_fn=hierarchical_model,
            hyper_model_kwargs_fn=hyper_model_kwargs_fn,
            extract_hyper_fn=default_extract_hyper,
            n_iterations=N_ITERATIONS,
            n_burnin=N_BURNIN,
            hyper_num_warmup=HYPER_NUM_WARMUP,
            show_progress=True,
            init_hyper_values=init_hyper_values,
        )

    # ── 7. Print summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if 'log10_N_tot' in chain and len(chain['log10_N_tot']) > 0:
        log10N = np.array(chain['log10_N_tot'])
        print(f"log10(N_tot): {np.mean(log10N):.3f} +/- {np.std(log10N):.3f} "
              f"(true: {np.log10(N_SOURCES_TRUE):.3f})")

    if 'alpha' in chain and len(chain['alpha']) > 0:
        a = np.array(chain['alpha'])
        print(f"alpha:        {np.mean(a):.3f} +/- {np.std(a):.3f} "
              f"(true: {ALPHA_TRUE:.3f})")

    if 'beta' in chain and len(chain['beta']) > 0:
        b = np.array(chain['beta'])
        print(f"beta:         {np.mean(b):.3f} +/- {np.std(b):.3f} "
              f"(true: {BETA_TRUE:.3f})")

    total_resolved = sum(band.N_res for band in final_bands)
    print(f"\nTotal resolved sources: {total_resolved}")
    for band in final_bands:
        print(f"  Band {band.k}: N_res={band.N_res}, S_conf={band.S_conf:.4e}")

    # PT swap diagnostics
    if 'swap_rate' in traces:
        swap_rates = np.array(traces['swap_rate'])
        n_swap_rounds = int(np.sum(~np.isnan(swap_rates)))
        if n_swap_rounds > 0:
            mean_rate = float(np.nanmean(swap_rates))
            print(f"\nPT swap acceptance rate: {mean_rate:.3f} "
                  f"({n_swap_rounds} swap rounds out of {len(swap_rates)} iterations)")
        else:
            print("\nNo PT swaps were proposed.")

    # ── 8. Save results ──────────────────────────────────────────────────
    true_params = {'N_sources': N_SOURCES_TRUE, 'alpha': ALPHA_TRUE,
                    'beta': BETA_TRUE, 'S_instr': S_INSTR, 'rho_th': RHO_TH}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_results(OUTPUT_DIR, chain, traces, true_params, final_bands,
                 MODEL_PATH, F_MIN, F_MAX, N_COMPONENTS,
                 true_amps=true_amps, true_freqs=true_freqs,
                 true_phases=true_phases, data_fd=data_fd,
                 freq_grid=freq_grid)

    # ── 9. Diagnostic plots ──────────────────────────────────────────────
    replot(OUTPUT_DIR)

    print("\nDone.")


def save_results(output_dir, chain, traces, true_params, final_bands,
                 model_path, f_min, f_max, n_components,
                 true_amps=None, true_freqs=None, true_phases=None,
                 data_fd=None, freq_grid=None):
    """
    Save sampler outputs to output_dir using open formats.

    Files written
    -------------
    chain.feather       — posterior chain (one column per parameter)
    traces.feather      — per-iteration traces (n_active_k, lambda_res_k per band)
    metadata.json       — true_params, physical constants, model_path
    final_bands.npz     — band arrays (source_freqs/amps/phases, z_indicators, S_conf)
    """
    # Chain → feather
    chain_df = pd.DataFrame({k: np.array(v) for k, v in chain.items()})
    chain_df.to_feather(os.path.join(output_dir, 'chain.feather'))

    # Traces → feather (flatten band dimension into columns)
    n_active = np.array(traces['n_active'])    # (n_iter, n_bands)
    lam_res  = np.array(traces['lambda_res'])  # (n_iter, n_bands)
    n_bands  = n_active.shape[1]
    traces_df = pd.DataFrame({
        **{f'n_active_{k}': n_active[:, k] for k in range(n_bands)},
        **{f'lambda_res_{k}': lam_res[:, k] for k in range(n_bands)},
    })
    traces_df.to_feather(os.path.join(output_dir, 'traces.feather'))

    # Metadata → JSON
    meta = dict(
        true_params=true_params,
        f_min=f_min, f_max=f_max,
        n_components=n_components,
        model_path=model_path,
        n_bands=n_bands,
    )
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Final band arrays → npz
    np.savez(
        os.path.join(output_dir, 'final_bands.npz'),
        k=np.array([b.k for b in final_bands]),
        f_low=np.array([b.f_low for b in final_bands]),
        f_high=np.array([b.f_high for b in final_bands]),
        w=np.array([b.w for b in final_bands]),
        S_conf=np.array([float(b.S_conf) for b in final_bands]),
        source_freqs=np.stack([np.array(b.source_freqs) for b in final_bands]),
        source_amps=np.stack([np.array(b.source_amps) for b in final_bands]),
        source_phases=np.stack([np.array(b.source_phases) for b in final_bands]),
        z_indicators=np.stack([np.array(b.z_indicators) for b in final_bands]),
    )

    # Injections + strain data → npz (for overlay plots)
    if true_amps is not None:
        np.savez(
            os.path.join(output_dir, 'injections.npz'),
            true_amps=np.array(true_amps),
            true_freqs=np.array(true_freqs),
            true_phases=np.array(true_phases),
            data_fd=np.array(data_fd),
            freq_grid=np.array(freq_grid),
        )

    print(f"Saved chain.feather, traces.feather, metadata.json, final_bands.npz → {output_dir}/")


def load_results(output_dir):
    """Load results saved by save_results. Returns (chain, traces, meta, final_bands)."""
    from tophat_populations.multiband.bands import Band

    chain_df  = pd.read_feather(os.path.join(output_dir, 'chain.feather'))
    traces_df = pd.read_feather(os.path.join(output_dir, 'traces.feather'))

    with open(os.path.join(output_dir, 'metadata.json')) as f:
        meta = json.load(f)

    chain  = {col: chain_df[col].values for col in chain_df.columns}
    n_bands = meta['n_bands']
    traces = {
        'n_active':   np.stack([traces_df[f'n_active_{k}'].values
                                 for k in range(n_bands)], axis=1),
        'lambda_res': np.stack([traces_df[f'lambda_res_{k}'].values
                                 for k in range(n_bands)], axis=1),
    }

    d = np.load(os.path.join(output_dir, 'final_bands.npz'))
    final_bands = [
        Band(k=int(d['k'][i]), f_low=float(d['f_low'][i]),
             f_high=float(d['f_high'][i]), w=float(d['w'][i]),
             source_freqs=jnp.array(d['source_freqs'][i]),
             source_amps=jnp.array(d['source_amps'][i]),
             source_phases=jnp.array(d['source_phases'][i]),
             z_indicators=jnp.array(d['z_indicators'][i]),
             S_conf=float(d['S_conf'][i]))
        for i in range(n_bands)
    ]

    return chain, traces, meta, final_bands


def profile_one_iteration():
    """
    Run two Gibbs iterations with profiling and print a timing breakdown.

    Does the full setup (load MDN, generate data, init bands) then runs
    one warmup iteration and one JIT-compiled iteration, printing wall
    times for each sub-step.

    Usage
    -----
        python -c "from tophat_populations.run_gibbs_sampler import profile_one_iteration; profile_one_iteration()"
    """
    import time
    global FDOT, A_MIN

    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    print("Loading MDN...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)

    mdn_params   = jax.tree.map(jnp.asarray, model_data['params'])
    X_mean       = jnp.asarray(model_data['X_mean'])
    X_std        = jnp.asarray(model_data['X_std'])
    X_norm_stats = (X_mean, X_std)
    config       = model_data['config']

    F_MIN        = config['f_min']
    F_MAX        = config['f_max']
    T_OBS        = config.get('T_obs', 3.15e7)
    A_MIN        = config.get('A_min', 1e-3)
    A_MAX        = config.get('A_max', 10.0)
    S_INSTR      = config.get('S_instr', 1.1234)
    RHO_TH       = config.get('rho_th', 5.0)
    BAND_EDGES   = np.array(config['band_edges'])
    DELTA_F_BAND = config.get('delta_f_band', float(BAND_EDGES[1] - BAND_EDGES[0]))
    LAMBDA_BOUNDS = config['lambda_bounds']
    N_COMPONENTS = config['n_components']

    DELTA_F = 1.0 / T_OBS
    W       = 0.1 * DELTA_F_BAND
    FDOT    = W / T_OBS

    N_FREQ   = int((F_MAX - F_MIN) / DELTA_F) + 1
    freq_grid = jnp.linspace(F_MIN, F_MAX, N_FREQ)

    N_SOURCES_TRUE = 10000
    ALPHA_TRUE     = 4.0
    BETA_TRUE      = 100
    data_fd, true_amps, true_freqs, true_phases = generate_synthetic_data(
        freq_grid, N_SOURCES_TRUE, ALPHA_TRUE, BETA_TRUE,
        A_MIN, A_MAX, FDOT, S_INSTR, seed=42,
    )

    N_TEMPLATES  = 15
    bands = initialise_bands(
        BAND_EDGES, W, N_TEMPLATES,
        true_amps, true_freqs, true_phases,
        RHO_TH, S_INSTR, T_OBS,
        mdn_params, X_norm_stats, N_COMPONENTS,
        jnp.log10(float(N_SOURCES_TRUE)), float(ALPHA_TRUE), float(BETA_TRUE),
    )

    lambda_res_per_band = {b.k: float(b.N_res) for b in bands}

    hyper_model_kwargs_fn = make_hyper_model_kwargs_fn(
        mdn_params, X_norm_stats, S_INSTR, T_OBS, RHO_TH,
        A_MIN, A_MAX, DELTA_F_BAND, F_MIN, F_MAX,
        LAMBDA_BOUNDS, N_COMPONENTS,
    )

    rng = jax.random.PRNGKey(0)

    # Per-band RMH proposal sigmas
    proposal_sigmas = {
        b.k: build_proposal_sigma(N_TEMPLATES) for b in bands
    }

    # ── Iteration 1: warmup (expected to be slow) ──────────────────────
    print("\nIteration 1 (warmup)...")
    t0 = time.perf_counter()
    bands_tmp = list(bands)
    lam_tmp   = dict(lambda_res_per_band)
    bands_tmp, hyper_samples, diag = gibbs_iteration(
        rng, bands_tmp, data_fd, freq_grid,
        S_INSTR, FDOT, A_MIN, A_MAX, ALPHA_TRUE, RHO_TH,
        lam_tmp, proposal_sigmas,
        hyper_model_fn=hierarchical_model,
        hyper_model_kwargs_fn=hyper_model_kwargs_fn,
        profile=True,
    )
    jax.block_until_ready(hyper_samples)
    print(f"  Wall time: {time.perf_counter() - t0:.3f}s")
    print(f"  Breakdown: {diag.get('timing')}")

    # ── Iteration 2: post-warmup JIT path ──────────────────────────────
    print("\nIteration 2 (JIT path)...")
    rng, sub = jax.random.split(rng)
    t0 = time.perf_counter()
    bands_tmp, hyper_samples, diag = gibbs_iteration(
        sub, bands_tmp, data_fd, freq_grid,
        S_INSTR, FDOT, A_MIN, A_MAX, ALPHA_TRUE, RHO_TH,
        lam_tmp, proposal_sigmas,
        hyper_model_fn=hierarchical_model,
        hyper_model_kwargs_fn=hyper_model_kwargs_fn,
        hyper_updater=diag['hyper_updater'],
        profile=True,
    )
    jax.block_until_ready(hyper_samples)
    print(f"  Wall time: {time.perf_counter() - t0:.3f}s")
    print(f"  Breakdown: {diag.get('timing')}")

    # ── Iteration 3: fully warmed up (should be stable) ────────────────
    print("\nIteration 3 (stable)...")
    rng, sub = jax.random.split(rng)
    t0 = time.perf_counter()
    bands_tmp, hyper_samples, diag = gibbs_iteration(
        sub, bands_tmp, data_fd, freq_grid,
        S_INSTR, FDOT, A_MIN, A_MAX, ALPHA_TRUE, RHO_TH,
        lam_tmp, proposal_sigmas,
        hyper_model_fn=hierarchical_model,
        hyper_model_kwargs_fn=hyper_model_kwargs_fn,
        hyper_updater=diag['hyper_updater'],
        profile=True,
    )
    jax.block_until_ready(hyper_samples)
    print(f"  Wall time: {time.perf_counter() - t0:.3f}s")
    print(f"  Breakdown: {diag.get('timing')}")


def plot_band_overlay(band_k, output_dir=None, match_tol_w=0.5):
    """
    Overlay plot for one band: data amplitude spectrum, noise floor,
    injected source amplitudes, and recovered (z=1) source amplitudes.

    All quantities plotted as per-bin strain amplitude |h_i|, so that:
      - data bins:         |data_fd[i]|
      - noise floor:       sqrt(S_tot / (2 * delta_f))   [ASD in per-bin units]
      - source amplitude:  A / sqrt(w * delta_f)          [= Amp_per_bin from waveform]

    A source is detectable when its Amp_per_bin exceeds the noise floor.

    Matching: injected ↔ recovered within match_tol_w template widths in frequency.

    Parameters
    ----------
    band_k : int
        Band index to plot.
    output_dir : str or None
        Directory with saved results (default: gibbs_output/).
    match_tol_w : float
        Matching tolerance in units of template width w.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'gibbs_output')

    # Load band state and chain
    chain, _, meta, final_bands = load_results(output_dir)

    band = next((b for b in final_bands if b.k == band_k), None)
    if band is None:
        raise ValueError(f"Band {band_k} not found. Available: "
                         f"{[b.k for b in final_bands]}")

    # Load injections + data
    inj_path = os.path.join(output_dir, 'injections.npz')
    if not os.path.exists(inj_path):
        raise FileNotFoundError(
            f"{inj_path} not found — re-run the sampler to save injections.")
    d = np.load(inj_path)
    true_amps   = d['true_amps']
    true_freqs  = d['true_freqs']
    data_fd     = d['data_fd']
    freq_grid   = d['freq_grid']

    delta_f = float(freq_grid[1] - freq_grid[0])
    w       = float(band.w)

    # Frequency slice for this band (extended range)
    f_lo, f_hi = band.f_low_ext, band.f_high_ext
    mask = (freq_grid >= f_lo) & (freq_grid <= f_hi)
    freq_ext  = freq_grid[mask]
    data_ext  = np.abs(data_fd[mask])           # per-bin amplitude

    # Noise floor in same units
    S_conf  = float(band.S_conf)
    S_instr = meta['true_params'].get('S_instr',
              # fall back: read from model config via metadata
              1.1234)
    # Try loading S_instr from model config if saved
    try:
        with open(meta['model_path'], 'rb') as f_pkl:
            _md = pickle.load(f_pkl)
        S_instr = _md['config'].get('S_instr', S_instr)
    except Exception:
        pass
    S_tot       = S_instr + S_conf
    noise_floor = np.sqrt(S_tot / (2 * delta_f))   # per-bin ASD (recovered)

    # Per-bin amplitude of a source with amplitude A:  A / sqrt(w * delta_f)
    def amp_per_bin(A):
        return A / np.sqrt(w * delta_f)

    rho_th  = float(meta['true_params'].get('rho_th', 5.0))

    # ── Injected sources in this band ─────────────────────────────────────
    in_band      = np.where((true_freqs >= band.f_low) & (true_freqs < band.f_high))[0]
    all_inj_f    = true_freqs[in_band]
    all_inj_A    = true_amps[in_band]
    n_inj_total  = len(in_band)

    # True confusion noise from MDN prediction at band center
    from tophat_populations.noise_emulator.network import gated_mdn_forward
    true_params = meta['true_params']
    log10_N_tot = np.log10(true_params['N_sources'])
    alpha = true_params['alpha']
    beta = true_params.get('beta', 100.0)
    Lambda = jnp.array([log10_N_tot, alpha, beta])
    with open(meta['model_path'], 'rb') as f_pkl:
        model_data = pickle.load(f_pkl)
    mdn_params = jax.tree.map(jnp.asarray, model_data['params'])
    X_norm_stats = (jnp.asarray(model_data['X_mean']),
                    jnp.asarray(model_data['X_std']))
    n_components = meta['n_components']
    x_raw = jnp.concatenate([Lambda, jnp.atleast_1d(jnp.array(band.f_center))])
    X_mean, X_std = X_norm_stats
    x_norm = (x_raw - X_mean) / X_std
    log_pi, mu, _, _ = gated_mdn_forward(
        mdn_params, x_norm[None, :], n_components)
    pi = jnp.exp(log_pi[0])
    log_S_conf_mean = float(jnp.sum(pi * mu[0]))
    true_S_conf = float(jnp.exp(log_S_conf_mean))

    true_S_tot = S_instr + true_S_conf
    A_th = (rho_th / 2.0) * np.sqrt(true_S_tot * delta_f)
    det_mask = all_inj_A > A_th
    det_indices  = set(np.where(det_mask)[0])
    n_det        = int(det_mask.sum())
    true_noise_floor = np.sqrt(true_S_tot / (2 * delta_f))

    # All detectable injections for plotting
    det_idx      = np.where(det_mask)[0]
    inj_f_plot   = all_inj_f[det_idx]
    inj_A_plot   = all_inj_A[det_idx]

    # ── All template slots ──────────────────────────────────────────────
    z      = np.array(band.z_indicators)
    all_tmpl_f = np.array(band.source_freqs)
    all_tmpl_A = np.array(band.source_amps)
    rec_f  = all_tmpl_f[z > 0.5]
    rec_A  = all_tmpl_A[z > 0.5]
    off_f  = all_tmpl_f[z < 0.5]
    off_A  = all_tmpl_A[z < 0.5]

    # Per-source SNR for active templates (using true S_tot for consistency
    # with the true A_th detection threshold)
    T_obs  = 1.0 / delta_f
    rec_snr = 2 * rec_A * np.sqrt(T_obs / true_S_tot)

    # ── Match recovered → ALL injected ───────────────────────────────────
    tol = match_tol_w * w
    matched_inj  = set()   # indices into all_inj_f
    matched_rec  = set()   # indices into rec_f
    for ri, rf in enumerate(rec_f):
        dists = np.abs(all_inj_f - rf)
        if len(dists) and dists.min() < tol:
            ii = int(dists.argmin())
            if ii not in matched_inj:
                matched_inj.add(ii)
                matched_rec.add(ri)

    n_tp   = len(matched_rec)
    n_fp   = len(rec_f) - n_tp
    purity = n_tp / max(len(rec_f), 1)

    # Completeness / FN over DETECTABLE sources only
    n_tp_det     = len(matched_inj & det_indices)
    n_fn_det     = n_det - n_tp_det
    completeness = n_tp_det / max(n_det, 1)

    # Which of the plotted detectable injections were recovered?
    matched_plot = {i for i, gi in enumerate(det_idx) if gi in matched_inj}

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.semilogy(freq_ext, data_ext, lw=0.6, alpha=0.5, color='C0',
                label='Data |h(f)|')
    ax.axvline(band.f_low,  color='grey', ls=':', lw=0.8)
    ax.axvline(band.f_high, color='grey', ls=':', lw=0.8)

    # Two categories for detectable injected markers:
    #   green ▼  recovered (TP)
    #   orange ▼ missed (FN)
    tp_f, tp_A, fn_f, fn_A = [], [], [], []
    for i, (f, A) in enumerate(zip(inj_f_plot, amp_per_bin(inj_A_plot))):
        if i in matched_plot:
            tp_f.append(f); tp_A.append(A)
        else:
            fn_f.append(f); fn_A.append(A)

    if tp_f:
        ax.scatter(tp_f,  tp_A,  marker='v', s=80, color='C2',    zorder=5,
                   label=f'Injected detectable, recovered ({len(tp_f)})')
    if fn_f:
        ax.scatter(fn_f,  fn_A,  marker='v', s=80, color='orange', zorder=5,
                   label=f'Injected detectable, missed ({len(fn_f)})')

    # Recovered sources (z=1): green ▲ = TP, red ▲ = FP
    rec_tp_f, rec_tp_A, rec_fp_f, rec_fp_A = [], [], [], []
    for i, (f, A) in enumerate(zip(rec_f, amp_per_bin(rec_A))):
        if i in matched_rec:
            rec_tp_f.append(f); rec_tp_A.append(A)
        else:
            rec_fp_f.append(f); rec_fp_A.append(A)

    if rec_tp_f:
        ax.scatter(rec_tp_f, rec_tp_A, marker='^', s=80, color='C1', zorder=6,
                   label='Recovered (true positive)')
    if rec_fp_f:
        ax.scatter(rec_fp_f, rec_fp_A, marker='^', s=80, color='red', zorder=6,
                   label='Recovered (false positive)')

    # Inactive templates (z=0) — small red dots to show where they ended up
    if len(off_f) > 0:
        ax.scatter(off_f, amp_per_bin(off_A), marker='x', s=20, color='red',
                   alpha=0.3, zorder=3, label=f'Inactive templates (z=0, n={len(off_f)})')

    # True detection threshold (from true S_conf)
    ax.axhline(amp_per_bin(A_th), color='purple', ls='-.', lw=1.2,
               label=f'True A_th  ρ={rho_th:.0f}  (A_th={A_th:.2e})')

    # Recovered noise floor + A_th: 90% CI from chain samples of log_S_conf
    sconf_key = f'log_S_conf_{band_k}'
    if sconf_key in chain and len(chain[sconf_key]) > 0:
        sconf_samples = np.exp(np.array(chain[sconf_key]))
        n_draw = min(100, len(sconf_samples))
        idx_draw = np.random.choice(len(sconf_samples), size=n_draw, replace=False)
        sconf_draw = sconf_samples[idx_draw]

        # Noise floor CI
        nf_draw = np.sqrt((S_instr + sconf_draw) / (2 * delta_f))
        nf_lo, nf_med, nf_hi = np.percentile(nf_draw, [5, 50, 95])
        ax.axhspan(nf_lo, nf_hi, color='k', alpha=0.15,
                   label=f'Recovered noise floor 90% CI')
        ax.axhline(nf_med, color='k', ls='--', lw=1.5,
                   label=f'Recovered median  [S_conf={np.median(sconf_draw):.2e}]')

        # A_th CI from recovered S_conf samples
        ath_draw = (rho_th / 2.0) * np.sqrt((S_instr + sconf_draw) * delta_f)
        ath_lo, ath_med, ath_hi = np.percentile(amp_per_bin(ath_draw), [5, 50, 95])
        ax.axhspan(ath_lo, ath_hi, color='purple', alpha=0.1,
                   label=f'Recovered A_th 90% CI')
        ax.axhline(ath_med, color='purple', ls='--', lw=1.2,
                   label=f'Recovered A_th median')
    else:
        ax.axhline(noise_floor, color='k', ls='--', lw=1.5,
                   label=f'Recovered noise floor  [S_conf={S_conf:.2e}]')
    ax.axhline(true_noise_floor, color='C3', ls='--', lw=1.5,
               label=f'True noise floor  [true S_conf={true_S_conf:.2e}]')

    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Per-bin amplitude  |h|')
    ax.set_xlim(f_lo, f_hi)
    ax.legend(fontsize=7)
    ax.set_title(
        f'Band {band_k}  [{band.f_low:.3e}, {band.f_high:.3e}] Hz\n'
        f'{n_inj_total} total injected  |  {n_det} detectable (ρ>{rho_th:.0f})  |  '
        f'{len(rec_f)} recovered  |  '
        f'purity {purity:.2f}  completeness {completeness:.2f}'
    )

    fig.tight_layout()
    out_path = os.path.join(output_dir, f'band_{band_k}_overlay.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    n_rec = len(rec_f)
    n_tp_subth = n_tp - n_tp_det   # TPs that matched a sub-threshold injection
    print(f"Saved {out_path}")
    print(f"  {n_inj_total} injected total, {n_det} detectable (ρ>{rho_th:.0f}), "
          f"{n_rec} recovered (z=1)")
    print(f"  Recoveries: {n_tp} matched an injection ({n_tp_det} detectable, "
          f"{n_tp_subth} sub-threshold), {n_fp} unmatched")
    print(f"  Detectable sources: {n_tp_det} recovered, {n_fn_det} missed")
    print(f"  Purity {purity:.2f} (fraction of recoveries matching any injection)")
    print(f"  Completeness {completeness:.2f} (fraction of detectable sources recovered)")
    if len(rec_snr) > 0:
        sort_idx = np.argsort(rec_snr)[::-1]
        print(f"  Recovered source SNRs (loudest first):")
        for j in sort_idx[:10]:
            tag = "TP" if j in matched_rec else "FP"
            print(f"    [{tag}] f={rec_f[j]:.4e} Hz  A={rec_A[j]:.3e}  ρ={rec_snr[j]:.1f}")
        if len(sort_idx) > 10:
            print(f"    ... and {len(sort_idx) - 10} more")


def replot(output_dir=None):
    """
    Re-generate all diagnostic plots from a saved output directory.

    Usage
    -----
        python -c "from tophat_populations.run_gibbs_sampler import replot; replot()"
    or pass an explicit directory path.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'gibbs_output')

    chain, traces, meta, final_bands = load_results(output_dir)

    with open(meta['model_path'], 'rb') as f:
        model_data = pickle.load(f)
    mdn_params   = jax.tree.map(jnp.asarray, model_data['params'])
    X_norm_stats = (jnp.asarray(model_data['X_mean']),
                    jnp.asarray(model_data['X_std']))

    plot_diagnostics(chain, final_bands, meta['true_params'], output_dir,
                     traces=traces)
    plot_sconf_prior(chain, mdn_params, X_norm_stats, meta['n_components'],
                     meta['f_min'], meta['f_max'], final_bands, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-band', type=int, default=None,
                        help='Plot overlay for this band index and exit')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Results directory (default: gibbs_output/)')
    args = parser.parse_args()

    if args.plot_band is not None:
        plot_band_overlay(args.plot_band, output_dir=args.output_dir)
    else:
        main()

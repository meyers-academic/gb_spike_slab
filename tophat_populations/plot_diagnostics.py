"""
Re-generate diagnostic plots from a saved Gibbs sampler output directory.

Usage
-----
    python tophat_populations/plot_diagnostics.py path/to/gibbs_output
    python tophat_populations/plot_diagnostics.py  # uses default gibbs_output/
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import pandas as pd

from tophat_populations.multiband.bands import Band
from tophat_populations.noise_emulator.network import gated_mdn_forward


# ── Load results ──────────────────────────────────────────────────────────────

def load_results(output_dir):
    """Load chain, traces, metadata, and final bands from saved output."""
    chain_df = pd.read_feather(os.path.join(output_dir, 'chain.feather'))
    traces_df = pd.read_feather(os.path.join(output_dir, 'traces.feather'))

    with open(os.path.join(output_dir, 'metadata.json')) as f:
        meta = json.load(f)

    chain = {col: chain_df[col].values for col in chain_df.columns}
    n_bands = meta['n_bands']
    traces = {
        'n_active': np.stack([traces_df[f'n_active_{k}'].values
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


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_hyper_traces(chain, true_params, output_dir):
    """Trace plots for hyperparameters."""
    hyper_keys = ['log10_N_tot', 'alpha', 'beta']
    true_vals = {
        'log10_N_tot': np.log10(true_params['N_sources']),
        'alpha': true_params['alpha'],
        'beta': true_params.get('beta', None),
    }
    true_vals = {k: v for k, v in true_vals.items() if v is not None}

    fig, axes = plt.subplots(len(hyper_keys), 1,
                             figsize=(10, 3 * len(hyper_keys)), sharex=True)
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


def plot_hyper_posteriors(chain, true_params, output_dir):
    """Posterior histograms for hyperparameters."""
    hyper_keys = ['log10_N_tot', 'alpha', 'beta']
    true_vals = {
        'log10_N_tot': np.log10(true_params['N_sources']),
        'alpha': true_params['alpha'],
        'beta': true_params.get('beta', None),
    }
    true_vals = {k: v for k, v in true_vals.items() if v is not None}

    fig, axes = plt.subplots(1, len(hyper_keys),
                             figsize=(4 * len(hyper_keys), 4))
    if len(hyper_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, hyper_keys):
        if key in chain and len(chain[key]) > 0:
            samples = np.array(chain[key])
            ax.hist(samples, bins=30, density=True, alpha=0.7,
                    edgecolor='black', linewidth=0.5)
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


def plot_source_counts(all_bands, output_dir):
    """Bar chart of resolved source counts per band."""
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


def plot_nres_traces(traces, output_dir):
    """N_active and lambda_res traces per band and total."""
    n_active_arr = np.array(traces['n_active'])
    lambda_res_arr = np.array(traces['lambda_res'])
    n_bands = n_active_arr.shape[1]

    # Per-band
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

    # Total
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.sum(n_active_arr, axis=1), alpha=0.7,
            label='Total N active (z)')
    ax.plot(np.sum(lambda_res_arr, axis=1), alpha=0.8, ls='--',
            label=r'Total $\lambda_{\rm res}$')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Count')
    ax.set_title(r'Total active indicators vs $\lambda_{\rm res}$')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'nres_total_trace.png'), dpi=150)
    plt.close(fig)


def plot_sconf_traces(chain, all_bands, output_dir):
    """log S_conf traces per band."""
    n_bands = len(all_bands)
    s_conf_keys = [f'log_S_conf_{k}' for k in range(n_bands)]
    available = [k for k in s_conf_keys if k in chain]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for key in available:
        ax.plot(np.array(chain[key]), alpha=0.5, label=key)
    ax.set_ylabel('log S_conf')
    ax.set_xlabel('Post-burnin iteration')
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'sconf_traces.png'), dpi=150)
    plt.close(fig)


def plot_sconf_prior(chain, mdn_params, X_norm_stats, n_components,
                     f_min, f_max, all_bands, output_dir,
                     n_freq=200, n_samples=500, ci=90):
    """Posterior predictive S_conf(f) from the MDN."""
    X_mean, X_std = X_norm_stats

    log10_N = np.array(chain.get('log10_N_tot', []))
    alphas = np.array(chain.get('alpha', []))
    betas = np.array(chain.get('beta', []))
    if len(log10_N) == 0:
        return

    idx = np.random.choice(len(log10_N),
                           size=min(n_samples, len(log10_N)), replace=False)
    log10_N = log10_N[idx]
    alphas = alphas[idx]
    betas = betas[idx]

    f_grid = np.linspace(f_min, f_max, n_freq)

    Lambda_rep = np.stack([
        np.repeat(log10_N[:, None], n_freq, axis=1),
        np.repeat(alphas[:, None], n_freq, axis=1),
        np.repeat(betas[:, None], n_freq, axis=1),
        np.tile(f_grid[None, :], (len(log10_N), 1)),
    ], axis=-1)

    flat = Lambda_rep.reshape(-1, 4)
    flat_j = (jnp.array(flat) - X_mean) / X_std

    log_pi, mu, _, _ = gated_mdn_forward(mdn_params, flat_j, n_components)
    pi = jnp.exp(log_pi)
    log_S_mean = jnp.sum(pi * mu, axis=-1)

    log_S_grid = np.array(log_S_mean).reshape(len(log10_N), n_freq)
    S_grid = np.exp(log_S_grid)

    lo = (100 - ci) / 2
    hi = 100 - lo
    med = np.median(S_grid, axis=0)
    p_lo = np.percentile(S_grid, lo, axis=0)
    p_hi = np.percentile(S_grid, hi, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(f_grid, p_lo, p_hi, alpha=0.3, label=f'{ci}% CI')
    ax.plot(f_grid, med, lw=1.5, label='Median')

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


# ── Main entry point ─────────────────────────────────────────────────────────

def replot(output_dir):
    """Load saved results and regenerate all diagnostic plots."""
    print(f"Loading results from {output_dir}/")
    chain, traces, meta, final_bands = load_results(output_dir)
    true_params = meta['true_params']

    os.makedirs(output_dir, exist_ok=True)

    plot_hyper_traces(chain, true_params, output_dir)
    plot_hyper_posteriors(chain, true_params, output_dir)
    plot_source_counts(final_bands, output_dir)
    plot_nres_traces(traces, output_dir)
    plot_sconf_traces(chain, final_bands, output_dir)

    # S_conf vs freq needs MDN params
    with open(meta['model_path'], 'rb') as f:
        model_data = pickle.load(f)
    mdn_params = jax.tree.map(jnp.asarray, model_data['params'])
    X_norm_stats = (jnp.asarray(model_data['X_mean']),
                    jnp.asarray(model_data['X_std']))
    plot_sconf_prior(chain, mdn_params, X_norm_stats, meta['n_components'],
                     meta['f_min'], meta['f_max'], final_bands, output_dir)

    print(f"All plots saved to {output_dir}/")


if __name__ == '__main__':
    default_dir = os.path.join(os.path.dirname(__file__), 'gibbs_output')
    output_dir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    replot(output_dir)

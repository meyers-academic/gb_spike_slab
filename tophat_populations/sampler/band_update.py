"""
Parallel band update orchestration.

Integrates residual computation, z-indicator Gibbs, and BlackJAX NUTS
into a single vmapped step across all same-parity bands.

The key GPU-performance insight: by using jax.lax.dynamic_slice to extract
each band's frequency slice *inside* the vmapped function, we reduce the
entire even (or odd) parity update to a single JAX kernel launch, eliminating
the O(N_bands) Python loop that caused excessive kernel dispatch overhead.
"""

import jax
import jax.numpy as jnp

from tophat_populations.multiband.bands import Band
from tophat_populations.multiband.residuals import _freq_slice_indices
from tophat_populations.waveform_simplified import tophat_fd_waveform
from tophat_populations.sampler.indicator_update import (
    update_indicators,
    compute_filter_coefficients,
)
from tophat_populations.sampler.source_update import (
    make_logdensity_unconstrained,
    rmh_step,
    to_unconstrained,
    to_constrained,
)


def update_parity_bands(rng_key, bands, fixed_bands, data_fd, freq_grid,
                        S_instr, fdot, lambda_res_per_band,
                        A_min, A_max, alpha, rho_th,
                        proposal_sigmas,
                        temperature=1.0):
    """
    Update all bands of one parity in a single vmap call.

    Residual computation (neighbor subtraction) is integrated into the
    per-band step and executed inside the vmap, so the entire parity
    update dispatches exactly one JAX kernel to the GPU.

    Parameters
    ----------
    rng_key : PRNGKey
    bands : list of Band
        Same-parity bands to update.
    fixed_bands : list of Band
        Opposite-parity bands (held fixed). Used to compute residuals.
    data_fd : complex array (N_freq,)
        Full frequency-domain strain data.
    freq_grid : array (N_freq,)
        Full frequency grid.
    S_instr : scalar
        Instrument noise PSD.
    fdot : scalar
        Frequency derivative (template width parameter).
    lambda_res_per_band : dict
        Mapping band index -> Poisson rate for resolved count.
    A_min, A_max : scalar
        Amplitude bounds.
    alpha : scalar
        Power-law index.
    step_sizes : array (N_bands,)
        Per-band adapted NUTS leapfrog step sizes (indexed by band.k).
    inv_mass_matrices : array (N_bands, 3*N_templates)
        Per-band adapted diagonal inverse mass matrices (indexed by band.k).

    Returns
    -------
    updated_bands : list of Band
    nuts_info : per-band NUTS diagnostics (vmapped)
    """
    N_parity = len(bands)
    N_templates = len(bands[0].z_indicators)
    fixed_lookup = {b.k: b for b in fixed_bands}
    zeros_t = jnp.zeros(N_templates)

    # ── Precompute slice indices (Python only, no JAX sync) ────────────────
    i_los, i_his = [], []
    for band in bands:
        i_lo, i_hi = _freq_slice_indices(freq_grid, band.f_low_ext, band.f_high_ext)
        i_los.append(i_lo)
        i_his.append(i_hi)

    slice_len = max(i_hi - i_lo for i_lo, i_hi in zip(i_los, i_his))
    n_freq = len(freq_grid)
    # Ensure dynamic_slice never reads past the end of freq_grid
    i_los_safe = [min(i_lo, n_freq - slice_len) for i_lo in i_los]

    # ── Stack per-band arrays ──────────────────────────────────────────────
    keys = jax.random.split(rng_key, N_parity)
    z_batch = jnp.stack([b.z_indicators for b in bands])
    amps_batch = jnp.stack([b.source_amps for b in bands])
    freqs_batch = jnp.stack([b.source_freqs for b in bands])
    phases_batch = jnp.stack([b.source_phases for b in bands])
    S_conf_batch = jnp.array([b.S_conf for b in bands])
    f_low_batch = jnp.array([b.f_low for b in bands])
    f_high_batch = jnp.array([b.f_high for b in bands])
    lambda_batch = jnp.array([lambda_res_per_band[b.k] for b in bands])
    temp_batch = jnp.broadcast_to(jnp.asarray(temperature), (N_parity,))
    i_lo_batch = jnp.array(i_los_safe, dtype=jnp.int32)

    # Per-band RMH proposal sigmas: index by band.k
    sigma_batch = jnp.stack([proposal_sigmas[b.k] for b in bands])

    # ── Precompute neighbor arrays per band (Python loop, O(N_parity)) ────
    def _neighbor_arrays(band_k, use_right_buffer):
        """Return (amps, freqs, phases, weights) for fixed neighbour of band_k."""
        if band_k in fixed_lookup:
            nb = fixed_lookup[band_k]
            mask = (nb.right_buffer_mask() if use_right_buffer
                    else nb.left_buffer_mask())
            return nb.source_amps, nb.source_freqs, nb.source_phases, nb.z_indicators * mask
        return zeros_t, zeros_t, zeros_t, zeros_t

    left_a, left_f, left_p, left_w = zip(
        *[_neighbor_arrays(b.k - 1, use_right_buffer=True) for b in bands])
    right_a, right_f, right_p, right_w = zip(
        *[_neighbor_arrays(b.k + 1, use_right_buffer=False) for b in bands])

    left_amps_b   = jnp.stack(left_a)
    left_freqs_b  = jnp.stack(left_f)
    left_phases_b = jnp.stack(left_p)
    left_weights_b = jnp.stack(left_w)
    right_amps_b   = jnp.stack(right_a)
    right_freqs_b  = jnp.stack(right_f)
    right_phases_b = jnp.stack(right_p)
    right_weights_b = jnp.stack(right_w)

    # ── Define per-band step as a closure over shared quantities ──────────
    # Closing over: data_fd, freq_grid, slice_len, S_instr, fdot,
    #               A_min, A_max, alpha
    # Per-band (vmap axis 0): everything else including step_size, inv_mass
    def _step(rng_key, z, amps, freqs, phases, S_conf,
              i_lo, lambda_res_k, f_low, f_high, temp,
              sigma,
              l_amps, l_freqs, l_phases, l_weights,
              r_amps, r_freqs, r_phases, r_weights):
        k1, k2 = jax.random.split(rng_key)
        psd_total = S_instr + S_conf

        # Extract this band's slice from full arrays (GPU-friendly: no sync)
        data_slice = jax.lax.dynamic_slice(data_fd, (i_lo,), (slice_len,))
        freq_slice = jax.lax.dynamic_slice(freq_grid, (i_lo,), (slice_len,))

        # Subtract fixed-parity neighbour contributions (weighted vmap, no Python loop)
        all_left_h = jax.vmap(
            lambda a, fc, phi: tophat_fd_waveform(a, fc, phi, fdot, freq_slice)
        )(l_amps, l_freqs, l_phases)   # (N_templates, slice_len)
        all_right_h = jax.vmap(
            lambda a, fc, phi: tophat_fd_waveform(a, fc, phi, fdot, freq_slice)
        )(r_amps, r_freqs, r_phases)

        data_residual = (data_slice
                         - jnp.dot(l_weights, all_left_h)
                         - jnp.dot(r_weights, all_right_h))

        # Matched-filter sufficient statistics
        dd, hd, hh = compute_filter_coefficients(
            amps, freqs, phases, fdot, data_residual, psd_total, freq_slice)

        # Detection threshold (needed by both z update and source prior)
        N = z.shape[0]
        delta_f = freq_slice[1] - freq_slice[0]
        A_th = (rho_th / 2.0) * jnp.sqrt(psd_total * delta_f)

        # Amplitude truncation prior: log-odds for z_k=1 vs z_k=0
        # Sources with A < A_th get large negative log-odds → strongly penalised
        _EPS_margin = 1e-7
        margin = 0.05 * A_th
        amp_log_odds = jax.nn.log_sigmoid(
            (amps - A_th) / jnp.clip(margin, _EPS_margin))

        # Gibbs z-indicator update
        z_new = update_indicators(k1, z, dd, hd, hh, lambda_res_k,
                                  amp_log_odds=amp_log_odds, temperature=temp)
        logdensity_u = make_logdensity_unconstrained(
            z_new, fdot, data_residual, psd_total, freq_slice,
            A_min, A_max, f_low, f_high, alpha, A_th,
            temperature=temp)

        # Transform current params to unconstrained space
        u = to_unconstrained(amps, freqs, phases, A_min, A_max, f_low, f_high)

        # RMH step in unconstrained space
        u_new, info = rmh_step(k2, u, logdensity_u, sigma)

        # Transform back to constrained space
        amps_new, freqs_new, phases_new = to_constrained(
            u_new, N, A_min, A_max, f_low, f_high)

        # Untempered log-likelihood (reuse dd, hd, hh — zero extra cost)
        ll = -0.5 * (dd - 2.0 * jnp.dot(z_new, hd) + jnp.dot(z_new, hh @ z_new))

        return z_new, amps_new, freqs_new, phases_new, info.is_accepted, ll

    # ── Single vmap over all same-parity bands (1 kernel launch) ──────────
    batched_step = jax.vmap(_step)

    z_new, amps_new, freqs_new, phases_new, is_accepted, ll_per_band = batched_step(
        keys, z_batch, amps_batch, freqs_batch, phases_batch, S_conf_batch,
        i_lo_batch, lambda_batch, f_low_batch, f_high_batch, temp_batch,
        sigma_batch,
        left_amps_b, left_freqs_b, left_phases_b, left_weights_b,
        right_amps_b, right_freqs_b, right_phases_b, right_weights_b,
    )

    # ── Unpack into Band objects ───────────────────────────────────────────
    updated = []
    for i, band in enumerate(bands):
        updated.append(Band(
            k=band.k,
            f_low=band.f_low,
            f_high=band.f_high,
            w=band.w,
            source_freqs=freqs_new[i],
            source_amps=amps_new[i],
            source_phases=phases_new[i],
            z_indicators=z_new[i],
            S_conf=band.S_conf,
        ))

    return updated, is_accepted, ll_per_band


def update_parity_bands_pt(rng_key, all_chain_bands, all_chain_fixed_bands,
                           data_fd, freq_grid, S_instr, fdot,
                           all_chain_lambda_res, A_min, A_max, alpha, rho_th,
                           proposal_sigmas, temperatures):
    """
    Update all same-parity bands across K temperature chains in one vmap.

    Stacks K chains × N_parity bands into a single batch of size K*N_parity
    and dispatches exactly one JAX kernel to the GPU, giving near-linear
    speedup over the sequential per-chain loop.

    Parameters
    ----------
    rng_key : PRNGKey
    all_chain_bands : list of K lists of Band
        all_chain_bands[c] = same-parity bands for chain c.
    all_chain_fixed_bands : list of K lists of Band
        all_chain_fixed_bands[c] = opposite-parity bands for chain c.
    all_chain_lambda_res : list of K dicts
        Per-chain lambda_res_per_band.
    temperatures : array (K,)
        Temperature for each chain.
    [remaining params: see update_parity_bands]

    Returns
    -------
    all_updated : list of K lists of Band
        Updated bands per chain.
    all_divergent : list of K arrays
        Per-band divergence flags per chain.
    all_ll : list of K arrays
        Per-band untempered log-likelihoods per chain.
    """
    K = len(all_chain_bands)
    N_parity = len(all_chain_bands[0])
    N_templates = len(all_chain_bands[0][0].z_indicators)
    zeros_t = jnp.zeros(N_templates)

    # All chains share the same band geometry, so slice indices are identical
    bands_ref = all_chain_bands[0]
    i_los, i_his = [], []
    for band in bands_ref:
        i_lo, i_hi = _freq_slice_indices(freq_grid, band.f_low_ext, band.f_high_ext)
        i_los.append(i_lo)
        i_his.append(i_hi)

    slice_len = max(i_hi - i_lo for i_lo, i_hi in zip(i_los, i_his))
    n_freq = len(freq_grid)
    i_los_safe = [min(i_lo, n_freq - slice_len) for i_lo in i_los]

    # ── Stack K*N_parity arrays ─────────────────────────────────────────
    total = K * N_parity
    keys = jax.random.split(rng_key, total)

    all_z, all_amps, all_freqs, all_phases = [], [], [], []
    all_S_conf, all_f_low, all_f_high, all_lambda, all_temp = [], [], [], [], []
    all_i_lo = []
    all_sigma = []
    all_l_a, all_l_f, all_l_p, all_l_w = [], [], [], []
    all_r_a, all_r_f, all_r_p, all_r_w = [], [], [], []

    for c in range(K):
        bands_c = all_chain_bands[c]
        fixed_c = all_chain_fixed_bands[c]
        lambda_c = all_chain_lambda_res[c]
        fixed_lookup_c = {b.k: b for b in fixed_c}
        T_c = float(temperatures[c])

        for b_idx, band in enumerate(bands_c):
            all_z.append(band.z_indicators)
            all_amps.append(band.source_amps)
            all_freqs.append(band.source_freqs)
            all_phases.append(band.source_phases)
            all_S_conf.append(band.S_conf)
            all_f_low.append(band.f_low)
            all_f_high.append(band.f_high)
            all_lambda.append(lambda_c[band.k])
            all_temp.append(T_c)
            all_i_lo.append(i_los_safe[b_idx])
            all_sigma.append(proposal_sigmas[band.k])

            # Left neighbor
            nk = band.k - 1
            if nk in fixed_lookup_c:
                nb = fixed_lookup_c[nk]
                mask = nb.right_buffer_mask()
                all_l_a.append(nb.source_amps)
                all_l_f.append(nb.source_freqs)
                all_l_p.append(nb.source_phases)
                all_l_w.append(nb.z_indicators * mask)
            else:
                all_l_a.append(zeros_t)
                all_l_f.append(zeros_t)
                all_l_p.append(zeros_t)
                all_l_w.append(zeros_t)

            # Right neighbor
            nk = band.k + 1
            if nk in fixed_lookup_c:
                nb = fixed_lookup_c[nk]
                mask = nb.left_buffer_mask()
                all_r_a.append(nb.source_amps)
                all_r_f.append(nb.source_freqs)
                all_r_p.append(nb.source_phases)
                all_r_w.append(nb.z_indicators * mask)
            else:
                all_r_a.append(zeros_t)
                all_r_f.append(zeros_t)
                all_r_p.append(zeros_t)
                all_r_w.append(zeros_t)

    # Stack everything
    z_batch = jnp.stack(all_z)
    amps_batch = jnp.stack(all_amps)
    freqs_batch = jnp.stack(all_freqs)
    phases_batch = jnp.stack(all_phases)
    S_conf_batch = jnp.array(all_S_conf)
    f_low_batch = jnp.array(all_f_low)
    f_high_batch = jnp.array(all_f_high)
    lambda_batch = jnp.array(all_lambda)
    temp_batch = jnp.array(all_temp)
    i_lo_batch = jnp.array(all_i_lo, dtype=jnp.int32)
    sigma_batch = jnp.stack(all_sigma)

    left_amps_b = jnp.stack(all_l_a)
    left_freqs_b = jnp.stack(all_l_f)
    left_phases_b = jnp.stack(all_l_p)
    left_weights_b = jnp.stack(all_l_w)
    right_amps_b = jnp.stack(all_r_a)
    right_freqs_b = jnp.stack(all_r_f)
    right_phases_b = jnp.stack(all_r_p)
    right_weights_b = jnp.stack(all_r_w)

    # ── Reuse the same _step closure ────────────────────────────────────
    def _step(rng_key, z, amps, freqs, phases, S_conf,
              i_lo, lambda_res_k, f_low, f_high, temp,
              sigma,
              l_amps, l_freqs, l_phases, l_weights,
              r_amps, r_freqs, r_phases, r_weights):
        k1, k2 = jax.random.split(rng_key)
        psd_total = S_instr + S_conf

        data_slice = jax.lax.dynamic_slice(data_fd, (i_lo,), (slice_len,))
        freq_slice = jax.lax.dynamic_slice(freq_grid, (i_lo,), (slice_len,))

        all_left_h = jax.vmap(
            lambda a, fc, phi: tophat_fd_waveform(a, fc, phi, fdot, freq_slice)
        )(l_amps, l_freqs, l_phases)
        all_right_h = jax.vmap(
            lambda a, fc, phi: tophat_fd_waveform(a, fc, phi, fdot, freq_slice)
        )(r_amps, r_freqs, r_phases)

        data_residual = (data_slice
                         - jnp.dot(l_weights, all_left_h)
                         - jnp.dot(r_weights, all_right_h))

        dd, hd, hh = compute_filter_coefficients(
            amps, freqs, phases, fdot, data_residual, psd_total, freq_slice)

        N = z.shape[0]
        delta_f = freq_slice[1] - freq_slice[0]
        A_th = (rho_th / 2.0) * jnp.sqrt(psd_total * delta_f)

        _EPS_margin = 1e-7
        margin = 0.05 * A_th
        amp_log_odds = jax.nn.log_sigmoid(
            (amps - A_th) / jnp.clip(margin, _EPS_margin))

        z_new = update_indicators(k1, z, dd, hd, hh, lambda_res_k,
                                  amp_log_odds=amp_log_odds, temperature=temp)

        logdensity_u = make_logdensity_unconstrained(
            z_new, fdot, data_residual, psd_total, freq_slice,
            A_min, A_max, f_low, f_high, alpha, A_th,
            temperature=temp)

        u = to_unconstrained(amps, freqs, phases, A_min, A_max, f_low, f_high)
        u_new, info = rmh_step(k2, u, logdensity_u, sigma)
        amps_new, freqs_new, phases_new = to_constrained(
            u_new, N, A_min, A_max, f_low, f_high)

        ll = -0.5 * (dd - 2.0 * jnp.dot(z_new, hd) + jnp.dot(z_new, hh @ z_new))
        return z_new, amps_new, freqs_new, phases_new, info.is_accepted, ll

    # ── Single vmap over K*N_parity (one GPU kernel) ────────────────────
    batched_step = jax.vmap(_step)

    z_new, amps_new, freqs_new, phases_new, is_accepted, ll_flat = batched_step(
        keys, z_batch, amps_batch, freqs_batch, phases_batch, S_conf_batch,
        i_lo_batch, lambda_batch, f_low_batch, f_high_batch, temp_batch,
        sigma_batch,
        left_amps_b, left_freqs_b, left_phases_b, left_weights_b,
        right_amps_b, right_freqs_b, right_phases_b, right_weights_b,
    )

    # ── Reshape (K*N_parity, ...) → per-chain lists of Band ────────────
    all_updated = []
    all_accepted = []
    all_ll = []
    for c in range(K):
        start = c * N_parity
        end = start + N_parity
        bands_c = all_chain_bands[c]
        updated_c = []
        for b_idx, band in enumerate(bands_c):
            flat_idx = start + b_idx
            updated_c.append(Band(
                k=band.k,
                f_low=band.f_low,
                f_high=band.f_high,
                w=band.w,
                source_freqs=freqs_new[flat_idx],
                source_amps=amps_new[flat_idx],
                source_phases=phases_new[flat_idx],
                z_indicators=z_new[flat_idx],
                S_conf=band.S_conf,
            ))
        all_updated.append(updated_c)
        all_accepted.append(is_accepted[start:end])
        all_ll.append(ll_flat[start:end])

    return all_updated, all_accepted, all_ll

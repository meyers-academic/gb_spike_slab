"""
Residual computation for multi-band updates.

When updating one parity of bands (e.g. even), the opposite parity (odd)
is held fixed. Sources from the fixed bands that bleed into the buffer
zones of the updating bands must be subtracted to form the residual.
"""

import jax
import jax.numpy as jnp


def compute_residuals(data_psd, freq_grid, bands_to_update, fixed_bands,
                      template_power_fn):
    """
    Compute residual PSDs for each band to be updated.

    Subtracts contributions from fixed-parity neighbour sources that
    bleed into the buffer zones of the bands being updated.

    Parameters
    ----------
    data_psd : array (N_freq_bins,)
        Full-band periodogram / PSD.
    freq_grid : array (N_freq_bins,)
        Frequency grid corresponding to data_psd.
    bands_to_update : list of Band
        Bands being updated (e.g. even bands).
    fixed_bands : list of Band
        Bands held fixed (e.g. odd bands). Indexed by band number.
    template_power_fn : callable
        Function(amp, freq, phase, freq_grid) -> array of PSD contribution.
        Computes the power contribution of a single source template at
        the given frequency grid points.

    Returns
    -------
    residuals : list of array
        Residual PSD for each band in bands_to_update, covering the
        extended frequency range [f_low_ext, f_high_ext].
    freq_slices : list of array
        Corresponding frequency grids for each residual.
    """
    residuals = []
    freq_slices = []

    # Build a lookup from band index to fixed band
    fixed_lookup = {b.k: b for b in fixed_bands}

    for band in bands_to_update:
        # Extract data in extended range
        ext_mask = (freq_grid >= band.f_low_ext) & (freq_grid <= band.f_high_ext)
        residual = data_psd[ext_mask].copy()
        freq_ext = freq_grid[ext_mask]

        # Subtract left neighbour's contributions in left buffer
        left_k = band.k - 1
        if left_k in fixed_lookup:
            left_band = fixed_lookup[left_k]
            buffer_mask = left_band.right_buffer_mask()
            active_mask = buffer_mask & (left_band.z_indicators > 0.5)
            if jnp.any(active_mask):
                idxs = jnp.where(active_mask)[0]
                for j in idxs:
                    residual = residual - template_power_fn(
                        left_band.source_amps[j],
                        left_band.source_freqs[j],
                        left_band.source_phases[j],
                        freq_ext,
                    )

        # Subtract right neighbour's contributions in right buffer
        right_k = band.k + 1
        if right_k in fixed_lookup:
            right_band = fixed_lookup[right_k]
            buffer_mask = right_band.left_buffer_mask()
            active_mask = buffer_mask & (right_band.z_indicators > 0.5)
            if jnp.any(active_mask):
                idxs = jnp.where(active_mask)[0]
                for j in idxs:
                    residual = residual - template_power_fn(
                        right_band.source_amps[j],
                        right_band.source_freqs[j],
                        right_band.source_phases[j],
                        freq_ext,
                    )

        residuals.append(residual)
        freq_slices.append(freq_ext)

    return residuals, freq_slices


def _freq_slice_indices(freq_grid, f_lo, f_hi):
    """
    Return Python integer (i_lo, i_hi) such that freq_grid[i_lo:i_hi]
    covers [f_lo, f_hi].  Assumes uniform spacing.  No JAX sync needed.
    """
    df = float(freq_grid[1] - freq_grid[0])
    f0 = float(freq_grid[0])
    i_lo = max(0, int((f_lo - f0) / df))
    i_hi = min(len(freq_grid), int((f_hi - f0) / df) + 2)
    return i_lo, i_hi


def _subtract_neighbor(residual, freq_ext, neighbor_band, buffer_mask, waveform_fn):
    """
    Subtract all neighbor contributions in one batched vmap call.

    weights = z_indicators * buffer_mask  (both are 0/1 arrays)
    Sum over templates: residual -= sum_j weights_j * h_j(f)

    No jnp.any / jnp.where — no GPU sync.
    """
    weights = neighbor_band.z_indicators * buffer_mask  # (N_templates,)
    # vmap waveform over all templates, then weighted sum
    all_h = jax.vmap(
        lambda a, f, phi: waveform_fn(a, f, phi, freq_ext)
    )(neighbor_band.source_amps, neighbor_band.source_freqs, neighbor_band.source_phases)
    # all_h: (N_templates, N_freq_ext)
    return residual - jnp.dot(weights, all_h)


def compute_residuals_complex(data_fd, freq_grid, bands_to_update, fixed_bands,
                              waveform_fn):
    """
    Compute complex residual strain for each band to be updated.

    Same logic as before but GPU-friendly:
    - Uses integer slice indexing (no boolean mask → no shape-dependent sync)
    - Uses weighted vmap sum to subtract neighbors (no jnp.any/jnp.where)

    Parameters
    ----------
    data_fd : complex array (N_freq_bins,)
        Full-band frequency-domain strain data.
    freq_grid : array (N_freq_bins,)
        Frequency grid corresponding to data_fd.
    bands_to_update : list of Band
        Bands being updated (e.g. even bands).
    fixed_bands : list of Band
        Bands held fixed (e.g. odd bands).
    waveform_fn : callable
        Function(amp, freq, phase, freq_grid) -> complex array.

    Returns
    -------
    residuals : list of complex array
    freq_slices : list of array
    """
    fixed_lookup = {b.k: b for b in fixed_bands}
    residuals = []
    freq_slices = []

    for band in bands_to_update:
        # Integer slice — no data-dependent shape, no GPU sync
        i_lo, i_hi = _freq_slice_indices(freq_grid, band.f_low_ext, band.f_high_ext)
        residual = data_fd[i_lo:i_hi]
        freq_ext = freq_grid[i_lo:i_hi]

        # Left neighbour: sources in its right buffer bleed into this band
        left_k = band.k - 1
        if left_k in fixed_lookup:
            left_band = fixed_lookup[left_k]
            buffer_mask = left_band.right_buffer_mask()  # JAX bool array, no sync
            residual = _subtract_neighbor(
                residual, freq_ext, left_band, buffer_mask, waveform_fn)

        # Right neighbour: sources in its left buffer bleed into this band
        right_k = band.k + 1
        if right_k in fixed_lookup:
            right_band = fixed_lookup[right_k]
            buffer_mask = right_band.left_buffer_mask()
            residual = _subtract_neighbor(
                residual, freq_ext, right_band, buffer_mask, waveform_fn)

        residuals.append(residual)
        freq_slices.append(freq_ext)

    return residuals, freq_slices

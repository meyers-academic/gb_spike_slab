"""
Iterative subtraction algorithm for training data generation.

For each population draw, iteratively classifies sources as resolved or
unresolved until self-consistency is reached, then records S_conf and N_res
per frequency band.
"""

import numpy as np


def sample_amplitudes(rng, N, alpha, A_min, A_max):
    """
    Draw N amplitudes from p(A) ~ A^{-alpha} on [A_min, A_max].

    Uses inverse CDF sampling.
    """
    u = rng.uniform(size=N)
    if np.abs(alpha - 1.0) < 1e-10:
        # Special case alpha = 1: log-uniform
        return A_min * np.exp(u * np.log(A_max / A_min))
    else:
        lo = A_min ** (1 - alpha)
        hi = A_max ** (1 - alpha)
        return (lo + u * (hi - lo)) ** (1.0 / (1 - alpha))


def sample_frequencies(rng, N, beta, f_min, f_max):
    """
    Draw N frequencies from p(f) ~ (1 - beta*f) on [f_min, f_max].

    Uses inverse CDF via a precomputed table + interpolation.
    For beta=0 this reduces to a uniform distribution.
    """
    if np.abs(beta) < 1e-12:
        return rng.uniform(f_min, f_max, size=N)

    # Build CDF table
    n_table = 10000
    f_table = np.linspace(f_min, f_max, n_table)
    pdf_unnorm = 1.0 - beta * f_table
    if np.any(pdf_unnorm < 0):
        raise ValueError(
            f"beta={beta} too large: p(f) goes negative before f_max={f_max}"
        )
    cdf = np.cumsum(pdf_unnorm)
    cdf = cdf / cdf[-1]

    # Inverse CDF via interpolation
    u = rng.uniform(size=N)
    return np.interp(u, cdf, f_table)


def assign_bands(frequencies, band_edges):
    """
    Assign each source to a frequency band.

    Parameters
    ----------
    frequencies : array (N,)
        Source frequencies.
    band_edges : array (N_bands + 1,)
        Edges of frequency bands.

    Returns
    -------
    band_idx : array (N,)
        Band index for each source (-1 if outside all bands).
    """
    # np.digitize returns 1-based indices; subtract 1 to get 0-based
    idx = np.digitize(frequencies, band_edges) - 1
    n_bands = len(band_edges) - 1
    # Mark sources outside the band range
    idx[(idx < 0) | (idx >= n_bands)] = -1
    return idx


def iterative_subtraction(
    N_tot,
    alpha,
    beta,
    f_min,
    f_max,
    band_edges,
    S_instr,
    T_obs,
    A_min,
    A_max,
    rho_th=5.0,
    max_iter=20,
    rng=None,
):
    """
    Run iterative subtraction on one population realisation.

    Parameters
    ----------
    N_tot : int
        Total number of sources.
    alpha : float
        Amplitude power-law index.
    beta : float
        Frequency distribution taper parameter.
    f_min, f_max : float
        Frequency range [Hz].
    band_edges : array (N_bands + 1,)
        Band edges [Hz].
    S_instr : array (N_bands,) or float
        Instrumental noise PSD in each band (or scalar for flat).
    T_obs : float
        Observation time [s].
    A_min, A_max : float
        Amplitude bounds.
    rho_th : float
        SNR detection threshold.
    max_iter : int
        Maximum number of iterations.
    rng : numpy Generator or None
        Random number generator.

    Returns
    -------
    S_conf : array (N_bands,)
        Confusion noise PSD in each band.
    N_res : array (N_bands,)
        Number of resolved sources per band.
    converged : bool
        Whether the iteration converged.
    n_iter : int
        Number of iterations used.
    """
    if rng is None:
        rng = np.random.default_rng()

    band_edges = np.asarray(band_edges)
    n_bands = len(band_edges) - 1
    delta_f_bands = np.diff(band_edges)
    band_centers = 0.5 * (band_edges[:-1] + band_edges[1:])

    S_instr = np.broadcast_to(np.asarray(S_instr, dtype=float), (n_bands,)).copy()

    # Draw sources
    amplitudes = sample_amplitudes(rng, N_tot, alpha, A_min, A_max)
    frequencies = sample_frequencies(rng, N_tot, beta, f_min, f_max)

    # Assign to bands
    band_idx = assign_bands(frequencies, band_edges)
    in_range = band_idx >= 0  # mask for sources inside the band range

    # Initialise: all sources are unresolved
    resolved = np.zeros(N_tot, dtype=bool)

    for iteration in range(max_iter):
        # Compute S_conf from unresolved sources in each band
        S_conf = np.zeros(n_bands)
        for b in range(n_bands):
            mask = in_range & (band_idx == b) & (~resolved)
            if np.any(mask):
                S_conf[b] = 2.0 * np.sum(amplitudes[mask] ** 2) / delta_f_bands[b]

        # Total noise
        S_tot = S_instr + S_conf

        # Compute SNR for all sources using the band they belong to
        snr = np.zeros(N_tot)
        for b in range(n_bands):
            mask = in_range & (band_idx == b)
            if np.any(mask):
                snr[mask] = 2.0 * amplitudes[mask] * np.sqrt(T_obs / S_tot[b])

        # Update resolved set
        new_resolved = snr > rho_th
        new_resolved[~in_range] = False

        if np.array_equal(new_resolved, resolved):
            # Converged
            # Recompute final S_conf
            S_conf_final = np.zeros(n_bands)
            N_res_final = np.zeros(n_bands, dtype=int)
            for b in range(n_bands):
                mask_unres = in_range & (band_idx == b) & (~resolved)
                mask_res = in_range & (band_idx == b) & resolved
                if np.any(mask_unres):
                    S_conf_final[b] = (
                        2.0 * np.sum(amplitudes[mask_unres] ** 2) / delta_f_bands[b]
                    )
                N_res_final[b] = int(np.sum(mask_res))

            return S_conf_final, N_res_final, True, iteration + 1

        resolved = new_resolved

    # Did not converge -- return last state
    S_conf_final = np.zeros(n_bands)
    N_res_final = np.zeros(n_bands, dtype=int)
    for b in range(n_bands):
        mask_unres = in_range & (band_idx == b) & (~resolved)
        mask_res = in_range & (band_idx == b) & resolved
        if np.any(mask_unres):
            S_conf_final[b] = (
                2.0 * np.sum(amplitudes[mask_unres] ** 2) / delta_f_bands[b]
            )
        N_res_final[b] = int(np.sum(mask_res))

    return S_conf_final, N_res_final, False, max_iter

"""
Generate training data for the MDN by running iterative subtraction
over many population realisations.
"""

import numpy as np
from tqdm import tqdm

from .iterative_sub import iterative_subtraction


def generate_training_data(
    n_realisations,
    band_edges,
    S_instr,
    T_obs,
    A_min,
    A_max,
    f_min,
    f_max,
    lambda_bounds,
    rho_th=5.0,
    seed=None,
    show_progress=True,
):
    """
    Generate (X, Y) training pairs via iterative subtraction (single band).

    For each realisation, hyperparameters Lambda = (N_tot, alpha, beta) are
    drawn from the prior, then iterative subtraction produces S_conf and N_res
    for the single band.  Each realisation contributes one row.

    The MDN training target Y is 1D (log S_conf only).  N_res is returned
    separately for validation but is NOT a training target — it is computed
    analytically from S_conf and Λ at inference time.

    Parameters
    ----------
    n_realisations : int
        Number of population draws.
    band_edges : array (2,)
        Frequency band edges [Hz] for the single band.
    S_instr : float
        Instrumental noise PSD.
    T_obs : float
        Observation time [s].
    A_min, A_max : float
        Amplitude bounds for source draws.
    f_min, f_max : float
        Frequency range [Hz].
    lambda_bounds : dict
        Prior ranges, e.g.
        {'N_tot': (1000, 100000), 'alpha': (1.5, 4.0), 'beta': (0.0, 0.5)}
    rho_th : float
        SNR detection threshold.
    seed : int or None
        Random seed.
    show_progress : bool
        Show progress bar.

    Returns
    -------
    X : array (n_realisations, 3)
        Inputs: (log10(N_tot), alpha, beta).
    Y : array (n_realisations,)
        Targets: log S_conf (1D). Placeholder value for resolved rows.
    N_res : array (n_realisations,)
        Resolved source counts (for validation, not training).
    resolved_flag : array (n_realisations,)
        1.0 if all sources are resolved (S_conf == 0), 0.0 otherwise.
    realisation_ids : array (n_realisations,)
        Realisation index for each row (for train/val splitting by realisation).
    """
    rng = np.random.default_rng(seed)
    band_edges = np.asarray(band_edges)

    X_list = []
    Y_list = []
    N_res_list = []
    resolved_flag_list = []
    ids_list = []

    N_lo, N_hi = lambda_bounds['N_tot']
    a_lo, a_hi = lambda_bounds['alpha']
    b_lo, b_hi = lambda_bounds['beta']

    iterator = range(n_realisations)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating training data")

    n_converged = 0
    for i in iterator:
        # Draw hyperparameters from prior
        log_N_lo, log_N_hi = np.log10(N_lo), np.log10(N_hi)
        N_tot = int(10 ** rng.uniform(log_N_lo, log_N_hi))
        alpha = rng.uniform(a_lo, a_hi)
        beta = rng.uniform(b_lo, b_hi)

        S_conf, N_res, converged, n_iter = iterative_subtraction(
            N_tot=N_tot,
            alpha=alpha,
            beta=beta,
            f_min=f_min,
            f_max=f_max,
            band_edges=band_edges,
            S_instr=S_instr,
            T_obs=T_obs,
            A_min=A_min,
            A_max=A_max,
            rho_th=rho_th,
            rng=rng,
        )
        n_converged += int(converged)

        # One row per realisation (single band)
        fully_resolved = S_conf[0] == 0.0
        log_S = np.log(S_conf[0]) if not fully_resolved else 0.0
        X_list.append([np.log10(N_tot), alpha, beta])
        Y_list.append(log_S)
        N_res_list.append(N_res[0])
        resolved_flag_list.append(1.0 if fully_resolved else 0.0)
        ids_list.append(i)

    if show_progress:
        print(f"Convergence rate: {n_converged}/{n_realisations}")

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)
    N_res_out = np.array(N_res_list, dtype=np.float64)
    resolved_flag = np.array(resolved_flag_list, dtype=np.float64)
    realisation_ids = np.array(ids_list, dtype=int)
    return X, Y, N_res_out, resolved_flag, realisation_ids


def train_val_split(X, Y, N_res, resolved_flag, realisation_ids,
                    val_fraction=0.2, seed=None):
    """
    Split data by realisation (not by row) to avoid leakage.

    Returns
    -------
    X_train, Y_train, N_res_train, rf_train,
    X_val, Y_val, N_res_val, rf_val
    """
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(realisation_ids)
    rng.shuffle(unique_ids)

    n_val = max(1, int(len(unique_ids) * val_fraction))
    val_ids = set(unique_ids[:n_val])

    val_mask = np.array([rid in val_ids for rid in realisation_ids])
    train_mask = ~val_mask

    return (X[train_mask], Y[train_mask], N_res[train_mask],
            resolved_flag[train_mask],
            X[val_mask], Y[val_mask], N_res[val_mask],
            resolved_flag[val_mask])
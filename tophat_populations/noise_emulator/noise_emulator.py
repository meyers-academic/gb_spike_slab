"""
Generate training data and train the MDN confusion noise emulator (single band).

Provides ``train_and_save(config, output_dir)`` which can be called from a
notebook or from the command line via ``python -m noise_emulator.noise_emulator``.
"""

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from .data_gen import generate_training_data, train_val_split
from .training import normalise_inputs, train_gated_mdn
from .network import (
    gated_mdn_predict_mean, gated_mdn_predict_variance,
    gated_mdn_predict_resolved_prob, compute_lambda_res,
)

# Output directory for saved artefacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "trained_models")

# ── Default physical config ──────────────────────────────────────────────

DEFAULT_CONFIG = {
    "f_min": 1e-4,
    "f_max": 1e-3,
    "T_obs": 3.15e7,
    "A_min": 1e-3,
    "A_max": 10,
    "S_instr": 1.1234,
    "rho_th": 9.0,
    "lambda_bounds": {
        "N_tot": (1000, 30_000),
        "alpha": (2.5, 5.0),
        "beta": (0.0, 800.0),
    },
    # MDN / training hyperparameters
    "n_components": 5,
    "n_hidden": 64,
    "n_steps": 15_000,
    "lr": 1e-3,
    "n_realisations": 10_000,
    "val_fraction": 0.2,
    "seed": 42,
}

# Keys that define the physical setup — if any of these differ the model
# must be retrained.
PHYSICS_KEYS = [
    "f_min", "f_max", "T_obs", "A_min", "A_max",
    "S_instr", "rho_th", "lambda_bounds",
]


def make_config(**overrides):
    """Return a config dict starting from DEFAULT_CONFIG with overrides applied."""
    cfg = {**DEFAULT_CONFIG, **overrides}
    # Derived quantities
    cfg["band_edges"] = [cfg["f_min"], cfg["f_max"]]
    cfg["delta_f_band"] = cfg["f_max"] - cfg["f_min"]
    cfg["f_center"] = 0.5 * (cfg["f_min"] + cfg["f_max"])
    return cfg


def configs_match(saved_cfg, desired_cfg):
    """Check whether the physics-relevant keys agree between two configs."""
    for key in PHYSICS_KEYS:
        saved_val = saved_cfg.get(key)
        desired_val = desired_cfg.get(key)
        if isinstance(saved_val, float) and isinstance(desired_val, float):
            if not np.isclose(saved_val, desired_val, rtol=1e-6):
                return False
        elif saved_val != desired_val:
            return False
    return True


def train_and_save(config=None, output_dir=None):
    """
    Generate training data, train the gated MDN, and save the model.

    Parameters
    ----------
    config : dict or None
        Physical + training config.  Uses DEFAULT_CONFIG if None.
    output_dir : str or None
        Where to save artefacts.  Defaults to ``trained_models/`` next to
        this file.

    Returns
    -------
    params : dict   — trained JAX parameter pytree
    norm_stats : tuple (X_mean, X_std)
    config : dict   — the config that was used
    """
    if config is None:
        config = make_config()
    if output_dir is None:
        output_dir = OUTPUT_DIR

    f_min = config["f_min"]
    f_max = config["f_max"]
    band_edges = np.array(config["band_edges"])
    delta_f_band = config["delta_f_band"]
    f_center = config["f_center"]
    T_obs = config["T_obs"]
    A_min = config["A_min"]
    A_max = config["A_max"]
    S_instr = config["S_instr"]
    rho_th = config["rho_th"]
    lambda_bounds = config["lambda_bounds"]

    n_components = config["n_components"]
    n_hidden = config["n_hidden"]
    n_steps = config["n_steps"]
    lr = config["lr"]
    n_realisations = config["n_realisations"]
    val_fraction = config["val_fraction"]
    seed = config["seed"]

    print("=" * 60)
    print("MDN Confusion Noise Emulator — Generate & Train (single band)")
    print("=" * 60)
    print()
    print(f"Frequency range : [{f_min:.0e}, {f_max:.0e}] Hz (single band)")
    print(f"Band center     : {f_center:.2e} Hz")
    print(f"T_obs           : {T_obs:.2e} s")
    print(f"A range         : [{A_min:.0e}, {A_max:.0e}]")
    print(f"S_instr         : {S_instr:.0e} Hz^-1")
    print(f"rho_th          : {rho_th}")
    print(f"Lambda bounds   : {lambda_bounds}")
    print(f"Realisations    : {n_realisations}")
    print()

    # ── 1. Generate training data ─────────────────────────────────────────
    print("Step 1: Generating training data via iterative subtraction...")
    X, Y, N_res, resolved_flag, ids = generate_training_data(
        n_realisations=n_realisations,
        band_edges=band_edges,
        S_instr=S_instr,
        T_obs=T_obs,
        A_min=A_min,
        A_max=A_max,
        f_min=f_min,
        f_max=f_max,
        lambda_bounds=lambda_bounds,
        rho_th=rho_th,
        seed=seed,
    )
    n_resolved = int(np.sum(resolved_flag))
    print(f"  Total training rows: {X.shape[0]}  ({n_realisations} realisations)")
    print(f"  Fully resolved: {n_resolved} ({n_resolved / len(resolved_flag):.1%})")
    print()

    # ── 2. Train/val split (by realisation) ───────────────────────────────
    (X_tr, Y_tr, Nres_tr, rf_tr,
     X_val, Y_val, Nres_val, rf_val) = train_val_split(
        X, Y, N_res, resolved_flag, ids,
        val_fraction=val_fraction, seed=seed
    )
    print(f"  Train: {X_tr.shape[0]} rows,  Val: {X_val.shape[0]} rows")

    # ── 3. Normalise inputs ───────────────────────────────────────────────
    X_tr_jax = jnp.array(X_tr)
    Y_tr_jax = jnp.array(Y_tr)
    rf_tr_jax = jnp.array(rf_tr)
    X_val_jax = jnp.array(X_val)
    Y_val_jax = jnp.array(Y_val)
    rf_val_jax = jnp.array(rf_val)
    Nres_val_jax = jnp.array(Nres_val)

    X_tr_norm, norm_stats = normalise_inputs(X_tr_jax)
    X_mean, X_std = norm_stats
    X_val_norm = (X_val_jax - X_mean) / X_std

    # ── 4. Train gated MDN ───────────────────────────────────────────────
    print()
    print(f"Step 2: Training gated MDN (K={n_components}, hidden={n_hidden}, steps={n_steps})...")
    key = jax.random.PRNGKey(0)
    params, history = train_gated_mdn(
        key,
        X_tr_norm,
        Y_tr_jax,
        rf_tr_jax,
        X_val=X_val_norm,
        Y_val=Y_val_jax,
        resolved_flag_val=rf_val_jax,
        n_components=n_components,
        n_hidden=n_hidden,
        n_steps=n_steps,
        lr=lr,
        print_every=500,
    )

    # ── 5. Quick diagnostic ───────────────────────────────────────────────
    print()
    print("Step 3: Quick validation diagnostics...")

    # Gate accuracy
    pred_resolved_prob = gated_mdn_predict_resolved_prob(params, X_val_norm, n_components=n_components)
    pred_resolved = pred_resolved_prob > 0.5
    gate_accuracy = float(jnp.mean(pred_resolved == (rf_val_jax > 0.5)))
    n_val_resolved = int(jnp.sum(rf_val_jax > 0.5))
    print(f"  Gate accuracy: {gate_accuracy:.2%}  "
          f"({n_val_resolved} / {len(rf_val_jax)} val rows fully resolved)")

    # MDN diagnostics on non-resolved validation rows only
    not_resolved_mask = rf_val_jax < 0.5
    if jnp.any(not_resolved_mask):
        X_val_nr = X_val_norm[not_resolved_mask]
        Y_val_nr = Y_val_jax[not_resolved_mask]

        pred_mean = gated_mdn_predict_mean(params, X_val_nr, n_components=n_components)
        pred_var = gated_mdn_predict_variance(params, X_val_nr, n_components=n_components)

        residuals = Y_val_nr - pred_mean
        rmse_log_s = float(jnp.sqrt(jnp.mean(residuals ** 2)))
        print(f"  RMSE(log S_conf) = {rmse_log_s:.3f}  (non-resolved rows only)")

        pred_std = jnp.sqrt(pred_var)
        within_1sig = jnp.abs(residuals) < pred_std
        frac_1sig_s = float(jnp.mean(within_1sig))
        print(f"  Fraction within 1-sigma (log S_conf): {frac_1sig_s:.2%} (expect ~68%)")

        # Analytic Poisson check
        X_val_nr_raw = X_val_jax[not_resolved_mask]
        Nres_val_nr = Nres_val_jax[not_resolved_mask]
        pred_lambda = jax.vmap(
            lambda log_s, x: compute_lambda_res(
                log_s, 10**x[0], x[1], x[2], f_center,
                S_instr, T_obs, rho_th, A_min, A_max,
                delta_f_band, f_min, f_max
            )
        )(pred_mean, X_val_nr_raw)
        rmse_nres = float(jnp.sqrt(jnp.mean((Nres_val_nr - pred_lambda) ** 2)))
        print(f"  RMSE(N_res vs analytic lambda_res) = {rmse_nres:.3f}")

    # ── 6. Save everything ─────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        os.path.join(output_dir, "training_data.npz"),
        X=X, Y=Y, N_res=N_res, resolved_flag=resolved_flag,
        realisation_ids=ids,
        X_train=X_tr, Y_train=Y_tr, N_res_train=Nres_tr,
        resolved_flag_train=rf_tr,
        X_val=X_val, Y_val=Y_val, N_res_val=Nres_val,
        resolved_flag_val=rf_val,
        band_edges=band_edges,
    )

    model_data = {
        "params": jax.tree.map(np.asarray, params),
        "X_mean": np.asarray(X_mean),
        "X_std": np.asarray(X_std),
        "history": history,
        "config": config,
    }
    model_path = os.path.join(output_dir, "mdn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print()
    print(f"Saved training data to {output_dir}/training_data.npz")
    print(f"Saved MDN model to     {model_path}")
    print("Done.")
    return params, norm_stats, config


def main():
    """Train with default config (CLI entry point)."""
    train_and_save(make_config(), OUTPUT_DIR)


if __name__ == "__main__":
    main()

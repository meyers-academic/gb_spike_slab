"""
Generate training data and train the MDN confusion noise emulator.

Physical setup
--------------
- 10 frequency bands linearly spaced between f_min=1e-4 Hz and f_max=1e-3 Hz
- Amplitude power-law index prior centred on alpha = 4  (range [2.5, 5.0])
- Frequency taper beta chosen so that p(f) stays positive across the band:
      p(f) ~ (1 - beta*f)  on [1e-4, 1e-3]
  For p(f) > 0 at f_max = 1e-3 we need beta < 1/f_max = 1000.
  A moderate taper: beta in [0, 800] gives 20% suppression at high-f end.
- N_tot = 10000 total sources across all bands (range [1000, 30000] for training)
- Instrument noise:  flat PSD at 1.1234 Hz^{-1}  (toy level)
- T_obs = 1 year ~ 3.15e7 s
- Amplitude bounds: A_min = 1e-3, A_max = 10
- SNR threshold rho_th = 5

Usage
-----
    conda activate gb_spike_slab
    python tophat_populations/noise_emulator/run_generate_and_train.py
"""

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp

from tophat_populations.noise_emulator.data_gen import generate_training_data, train_val_split
from tophat_populations.noise_emulator.training import normalise_inputs, train_gated_mdn
from tophat_populations.noise_emulator.network import (
    gated_mdn_predict_mean, gated_mdn_predict_variance,
    gated_mdn_predict_resolved_prob, compute_lambda_res,
)

# Output directory for saved artefacts
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "trained_models")


# ── Physical parameters ───────────────────────────────────────────────────

F_MIN = 1e-4          # Hz
F_MAX = 1e-3          # Hz
N_BANDS = 10
T_OBS = 3.15e7        # ~ 1 year in seconds
A_MIN = 1e-3
A_MAX = 10
S_INSTR = 1.1234       # flat instrument PSD [Hz^{-1}]
RHO_TH = 9.0

# Band edges: linearly spaced
BAND_EDGES = np.linspace(F_MIN, F_MAX, N_BANDS + 1)
DELTA_F_BAND = BAND_EDGES[1] - BAND_EDGES[0]

# ── Prior ranges for training ─────────────────────────────────────────────

LAMBDA_BOUNDS = {
    "N_tot": (1000, 30_000),
    "alpha": (2.5, 5.0),
    "beta": (0.0, 800.0),
}

# ── Data generation settings ──────────────────────────────────────────────

N_REALISATIONS = 10000
VAL_FRACTION = 0.2
SEED = 42

# ── MDN hyperparameters ───────────────────────────────────────────────────

N_COMPONENTS = 5
N_HIDDEN = 64
N_STEPS = 15_000
LR = 1e-3


def main():
    print("=" * 60)
    print("MDN Confusion Noise Emulator — Generate & Train (1D)")
    print("=" * 60)
    print()
    print(f"Frequency range : [{F_MIN:.0e}, {F_MAX:.0e}] Hz")
    print(f"Bands           : {N_BANDS} (linearly spaced)")
    print(f"T_obs           : {T_OBS:.2e} s")
    print(f"A range         : [{A_MIN:.0e}, {A_MAX:.0e}]")
    print(f"S_instr         : {S_INSTR:.0e} Hz^-1")
    print(f"rho_th          : {RHO_TH}")
    print(f"Lambda bounds   : {LAMBDA_BOUNDS}")
    print(f"Realisations    : {N_REALISATIONS}")
    print()

    # ── 1. Generate training data ─────────────────────────────────────────
    print("Step 1: Generating training data via iterative subtraction...")
    X, Y, N_res, resolved_flag, ids = generate_training_data(
        n_realisations=N_REALISATIONS,
        band_edges=BAND_EDGES,
        S_instr=S_INSTR,
        T_obs=T_OBS,
        A_min=A_MIN,
        A_max=A_MAX,
        f_min=F_MIN,
        f_max=F_MAX,
        lambda_bounds=LAMBDA_BOUNDS,
        rho_th=RHO_TH,
        seed=SEED,
    )
    n_resolved = int(np.sum(resolved_flag))
    print(f"  Total training rows: {X.shape[0]}  ({N_REALISATIONS} realisations x {N_BANDS} bands)")
    print(f"  Fully resolved bands: {n_resolved} ({n_resolved / len(resolved_flag):.1%})")
    print()

    # ── 2. Train/val split (by realisation) ───────────────────────────────
    (X_tr, Y_tr, Nres_tr, rf_tr,
     X_val, Y_val, Nres_val, rf_val) = train_val_split(
        X, Y, N_res, resolved_flag, ids,
        val_fraction=VAL_FRACTION, seed=SEED
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
    print(f"Step 2: Training gated MDN (K={N_COMPONENTS}, hidden={N_HIDDEN}, steps={N_STEPS})...")
    key = jax.random.PRNGKey(0)
    params, history = train_gated_mdn(
        key,
        X_tr_norm,
        Y_tr_jax,
        rf_tr_jax,
        X_val=X_val_norm,
        Y_val=Y_val_jax,
        resolved_flag_val=rf_val_jax,
        n_components=N_COMPONENTS,
        n_hidden=N_HIDDEN,
        n_steps=N_STEPS,
        lr=LR,
        print_every=500,
    )

    # ── 5. Quick diagnostic ───────────────────────────────────────────────
    print()
    print("Step 3: Quick validation diagnostics...")

    # Gate accuracy
    pred_resolved_prob = gated_mdn_predict_resolved_prob(params, X_val_norm, n_components=N_COMPONENTS)
    pred_resolved = pred_resolved_prob > 0.5
    gate_accuracy = float(jnp.mean(pred_resolved == (rf_val_jax > 0.5)))
    n_val_resolved = int(jnp.sum(rf_val_jax > 0.5))
    print(f"  Gate accuracy: {gate_accuracy:.2%}  "
          f"({n_val_resolved} / {len(rf_val_jax)} val bands fully resolved)")

    # MDN diagnostics on non-resolved validation rows only
    not_resolved_mask = rf_val_jax < 0.5
    if jnp.any(not_resolved_mask):
        X_val_nr = X_val_norm[not_resolved_mask]
        Y_val_nr = Y_val_jax[not_resolved_mask]

        pred_mean = gated_mdn_predict_mean(params, X_val_nr, n_components=N_COMPONENTS)
        pred_var = gated_mdn_predict_variance(params, X_val_nr, n_components=N_COMPONENTS)

        residuals = Y_val_nr - pred_mean
        rmse_log_s = float(jnp.sqrt(jnp.mean(residuals ** 2)))
        print(f"  RMSE(log S_conf) = {rmse_log_s:.3f}  (non-resolved bands only)")

        pred_std = jnp.sqrt(pred_var)
        within_1sig = jnp.abs(residuals) < pred_std
        frac_1sig_s = float(jnp.mean(within_1sig))
        print(f"  Fraction within 1-sigma (log S_conf): {frac_1sig_s:.2%} (expect ~68%)")

        # Analytic Poisson check: compare predicted lambda_res with observed N_res
        X_val_nr_raw = X_val_jax[not_resolved_mask]
        Nres_val_nr = Nres_val_jax[not_resolved_mask]
        # Compute lambda_res from predicted log S_conf
        pred_lambda = jax.vmap(
            lambda log_s, x: compute_lambda_res(
                log_s, 10**x[0], x[1], x[2], x[3],
                S_INSTR, T_OBS, RHO_TH, A_MIN, A_MAX,
                DELTA_F_BAND, F_MIN, F_MAX
            )
        )(pred_mean, X_val_nr_raw)
        rmse_nres = float(jnp.sqrt(jnp.mean((Nres_val_nr - pred_lambda) ** 2)))
        print(f"  RMSE(N_res vs analytic lambda_res) = {rmse_nres:.3f}")

    # ── 6. Save everything ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Training data (numpy arrays)
    np.savez(
        os.path.join(OUTPUT_DIR, "training_data.npz"),
        X=X, Y=Y, N_res=N_res, resolved_flag=resolved_flag,
        realisation_ids=ids,
        X_train=X_tr, Y_train=Y_tr, N_res_train=Nres_tr,
        resolved_flag_train=rf_tr,
        X_val=X_val, Y_val=Y_val, N_res_val=Nres_val,
        resolved_flag_val=rf_val,
        band_edges=BAND_EDGES,
    )

    # MDN params + normalisation stats + history (pickle for JAX pytree)
    model_data = {
        "params": jax.tree.map(np.asarray, params),
        "X_mean": np.asarray(X_mean),
        "X_std": np.asarray(X_std),
        "history": history,
        "config": {
            "n_components": N_COMPONENTS,
            "n_hidden": N_HIDDEN,
            "n_steps": N_STEPS,
            "lr": LR,
            "band_edges": BAND_EDGES.tolist(),
            "delta_f_band": float(DELTA_F_BAND),
            "f_min": F_MIN,
            "f_max": F_MAX,
            "T_obs": T_OBS,
            "A_min": A_MIN,
            "A_max": A_MAX,
            "S_instr": S_INSTR,
            "rho_th": RHO_TH,
            "lambda_bounds": LAMBDA_BOUNDS,
        },
    }
    model_path = os.path.join(OUTPUT_DIR, "mdn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print()
    print(f"Saved training data to {OUTPUT_DIR}/training_data.npz")
    print(f"Saved MDN model to     {model_path}")
    print("Done.")
    return params, norm_stats, history


if __name__ == "__main__":
    main()

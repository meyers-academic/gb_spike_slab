# Top-Hat Population Model: Architecture & How to Run

## Overview

A hierarchical Bayesian pipeline for separating galactic binaries (GBs) from
LISA-like data. It combines:

1. **A trained MDN emulator** that learns `p(log S_conf | N_tot, alpha, beta, f)`
2. **A multi-band source separator** using spike-and-slab indicators per template
3. **A Gibbs sampler** that alternates between source-level and population-level updates

---

## System Diagram

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                         DATA GENERATION                              │
 │   run_generate_and_train.py                                          │
 │                                                                      │
 │   For many draws of (N_tot, alpha, beta):                            │
 │     - Sample sources from power-law amplitude + tapered frequency    │
 │     - Run iterative subtraction to get S_conf per band               │
 │     - Record: X = [log10(N_tot), alpha, beta, f_k]                   │
 │               Y = log(S_conf)                                        │
 │               N_res = count of resolved sources                      │
 │                                                                      │
 │   Then train gated MDN on (X -> Y) with resolved_flag gate           │
 │                                                                      │
 │   Outputs:  trained_models/mdn_model.pkl                             │
 │             trained_models/training_data.npz                         │
 └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                     TRAINED MDN EMULATOR                             │
 │   noise_emulator/network.py                                          │
 │                                                                      │
 │   Input:  x = [log10(N_tot), alpha, beta, f_k]  (normalised)        │
 │                                                                      │
 │   Output: Gated Gaussian mixture for log S_conf                      │
 │           - K component weights (log_pi)                             │
 │           - K means (mu)                                             │
 │           - K std devs (sigma)                                       │
 │           - 1 gate logit  (p_resolved = sigmoid(gate_logit))         │
 │                                                                      │
 │   Also:   compute_lambda_res()  — analytic Poisson rate              │
 │           Given S_conf + (N_tot, alpha, beta, f_k),                  │
 │           derives expected N_res WITHOUT learning it.                 │
 └──────────────────────────────────────────────────────────────────────┘
                                    │
                     used as a prior in ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                       GIBBS SAMPLER                                  │
 │   sampler/gibbs.py  ::  run_gibbs()                                  │
 │                                                                      │
 │   Repeats gibbs_iteration() for n_iterations:                        │
 │                                                                      │
 │   ┌────────────────────────────────────────────────────────────────┐ │
 │   │  STEP 1: UPDATE EVEN BANDS              ✓ IMPLEMENTED (vmap)│ │
 │   │                                                                │ │
 │   │   Band 0        Band 2        Band 4        ...               │ │
 │   │   ┌──────┐     ┌──────┐     ┌──────┐                         │ │
 │   │   │ A,f,φ│     │ A,f,φ│     │ A,f,φ│  ← update per source   │ │
 │   │   │  z_i │     │  z_i │     │  z_i │  ← flip spike/slab     │ │
 │   │   └──────┘     └──────┘     └──────┘                         │ │
 │   │                                                                │ │
 │   │   Odd bands FROZEN — their buffer-zone sources subtracted     │ │
 │   │   as fixed background via compute_residuals().                 │ │
 │   │                                                                │ │
 │   │   This step requires: p(d_k | {A,f,φ,z})                     │ │
 │   │   i.e. the matched-filter / Whittle likelihood from           │ │
 │   │   matched_filter.py :: log_likelihood()                       │ │
 │   └────────────────────────────────────────────────────────────────┘ │
 │                              │                                       │
 │                              ▼                                       │
 │   ┌────────────────────────────────────────────────────────────────┐ │
 │   │  STEP 2: MIGRATE EVEN → ODD                       [exists]   │ │
 │   │                                                                │ │
 │   │   Sources whose f drifted across a band edge get              │ │
 │   │   transferred to the destination band.                        │ │
 │   │   multiband/migration.py :: migrate_sources()                 │ │
 │   └────────────────────────────────────────────────────────────────┘ │
 │                              │                                       │
 │                              ▼                                       │
 │   ┌────────────────────────────────────────────────────────────────┐ │
 │   │  STEP 3: UPDATE ODD BANDS               ✓ IMPLEMENTED (vmap)│ │
 │   │                                                                │ │
 │   │   Band 1        Band 3        Band 5        ...               │ │
 │   │   ┌──────┐     ┌──────┐     ┌──────┐                         │ │
 │   │   │ A,f,φ│     │ A,f,φ│     │ A,f,φ│                         │ │
 │   │   │  z_i │     │  z_i │     │  z_i │                         │ │
 │   │   └──────┘     └──────┘     └──────┘                         │ │
 │   │                                                                │ │
 │   │   Even bands FROZEN as background this time.                  │ │
 │   └────────────────────────────────────────────────────────────────┘ │
 │                              │                                       │
 │                              ▼                                       │
 │   ┌────────────────────────────────────────────────────────────────┐ │
 │   │  STEP 4: MIGRATE ODD → EVEN                       [exists]   │ │
 │   └────────────────────────────────────────────────────────────────┘ │
 │                              │                                       │
 │                              ▼                                       │
 │   ┌────────────────────────────────────────────────────────────────┐ │
 │   │  STEP 5: HYPERPARAMETER UPDATE                     [exists]   │ │
 │   │   sampler/hyper_update.py  →  sampler/model.py                │ │
 │   │                                                                │ │
 │   │   Inputs:  N_res_k = Σ z_i  per band (from steps 1+3)        │ │
 │   │            f_centers per band                                  │ │
 │   │                                                                │ │
 │   │   Runs NUTS HMC on the hierarchical NumPyro model:            │ │
 │   │                                                                │ │
 │   │     Λ = (N_tot, α, β)  ~  priors                              │ │
 │   │              │                                                  │ │
 │   │              ▼                                                  │ │
 │   │     log S_conf_k  ~  MDNPrior(Λ, f_k)    ← trained network   │ │
 │   │              │                                                  │ │
 │   │              ▼                                                  │ │
 │   │     λ_res_k = compute_lambda_res(S_conf_k, Λ, physics)       │ │
 │   │              │                                                  │ │
 │   │              ▼                                                  │ │
 │   │     N_res_k  ~  Poisson(λ_res_k)          ← OBSERVED          │ │
 │   │                                                                │ │
 │   │   Outputs: updated Λ, log S_conf per band                     │ │
 │   └────────────────────────────────────────────────────────────────┘ │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## Package Layout

```
tophat_populations/
├── confusion_noise.py               # Analytic S_conf solver (Newton iteration)
├── confusion_background_simulator.py # Full simulation: draw sources, compute PSD
├── priors.py                        # Power-law and frequency sampling utilities
├── waveform.py                      # Full chirp template (FFT-based)
├── waveform_simplified.py           # Top-hat frequency-domain waveform
├── matched_filter.py                # ⟨h|d⟩ inner products + log-likelihood
│
├── noise_emulator/                  # MDN confusion noise emulator
│   ├── network.py                   #   Gated MDN: init, forward, log_prob, predict, compute_lambda_res
│   ├── data_gen.py                  #   Training data generation (iterative subtraction)
│   ├── training.py                  #   Training loop (optax Adam)
│   ├── iterative_sub.py            #   Iterative subtraction algorithm
│   ├── run_generate_and_train.py   #   End-to-end: generate data → train → save
│   ├── explore_mdn.ipynb           #   Validation notebook (scatter, residuals, calibration)
│   └── trained_models/             #   Saved artefacts (mdn_model.pkl, training_data.npz)
│
├── multiband/                       # Multi-band source management
│   ├── bands.py                     #   Band dataclass, create_bands(), collect_band_summary()
│   ├── residuals.py                 #   Subtract fixed-parity neighbours from buffer zones
│   └── migration.py                 #   Detect + execute source migrations across band edges
│
├── sampler/                         # Gibbs sampler + NumPyro model
│   ├── mdn_prior.py                #   MDNPrior(dist.Distribution) — wraps MDN as NumPyro prior
│   ├── model.py                    #   hierarchical_model() — NumPyro generative model
│   ├── indicator_update.py         #   update_indicators() — scan-based z Gibbs (vmappable)
│   ├── source_update.py            #   BlackJAX RMH for (A, f, phi) per band (vmappable)
│   ├── band_update.py              #   update_parity_bands() — vmap orchestration over bands
│   ├── hyper_update.py             #   hyper_update_step() — NUTS HMC on hierarchical model
│   └── gibbs.py                    #   gibbs_iteration() + run_gibbs() — full Gibbs loop
│
└── final_tophat_design_doc/        # Design documents (read-only reference)
    ├── combined_design.md
    ├── multiband_design.md
    └── mdn_emulator_design.md
```

---

## How to Run

### Prerequisites

```bash
conda activate discovery   # or your env with jax, numpyro, optax
```

### 1. Train the MDN emulator

This generates training data (iterative subtraction over 10,000 realisations of
the population parameters) and trains the gated MDN:

```bash
python tophat_populations/noise_emulator/run_generate_and_train.py
```

Outputs saved to `tophat_populations/noise_emulator/trained_models/`:
- `mdn_model.pkl` — trained MDN params, normalisation stats, config
- `training_data.npz` — full training/validation arrays

Takes ~10-30 min depending on hardware (10k realisations x 10 bands, then
40k training steps).

### 2. Validate the emulator

Open and run the notebook:

```
tophat_populations/noise_emulator/explore_mdn.ipynb
```

This produces:
- Scatter plots of predicted vs true log S_conf (colored by N_tot)
- Analytic N_res comparison
- Residual distributions
- Calibration plots (are 1-sigma intervals actually 68%?)

### 3. Run tests

```bash
python -m pytest tests/ -v
```

45 tests covering:
- `test_mdn.py` (33): MDN init/forward/loss, compute_lambda_res, data generation, training loop
- `test_source_update.py` (12): filter coefficients, z-indicator Gibbs, BlackJAX RMH, band step, complex residuals

### 4. Run the Gibbs sampler

All inference components are implemented. The Gibbs sampler lives in
`sampler/gibbs.py :: run_gibbs()`. It requires a trained MDN, frequency-domain
strain data, and a `hyper_model_fn` + `hyper_model_kwargs_fn` to define the
hierarchical model (see "Pluggable hyperparameter model" below).

---

## What's Implemented vs What's Missing

| Component | Status | Location |
|---|---|---|
| MDN emulator (1D gated) | **Done** | `noise_emulator/network.py` |
| Training data generation | **Done** | `noise_emulator/data_gen.py` |
| Training loop | **Done** | `noise_emulator/training.py` |
| Run script (generate + train) | **Done** | `noise_emulator/run_generate_and_train.py` |
| Validation notebook | **Done** | `noise_emulator/explore_mdn.ipynb` |
| `compute_lambda_res` (analytic Poisson) | **Done** | `noise_emulator/network.py` |
| `MDNPrior` NumPyro distribution | **Done** | `sampler/mdn_prior.py` |
| Hierarchical NumPyro model | **Done** | `sampler/model.py` |
| NUTS hyper-update step (model-agnostic) | **Done** | `sampler/hyper_update.py` |
| Gibbs loop | **Done** | `sampler/gibbs.py` |
| Band dataclass + management | **Done** | `multiband/bands.py` |
| Residual computation | **Done** | `multiband/residuals.py` |
| Source migration | **Done** | `multiband/migration.py` |
| Matched filter likelihood | **Done** | `matched_filter.py` |
| z-indicator Gibbs (vmappable) | **Done** | `sampler/indicator_update.py` |
| BlackJAX RMH for (A, f, phi) | **Done** | `sampler/source_update.py` |
| Parallel band update (vmap) | **Done** | `sampler/band_update.py` |
| Complex residual computation | **Done** | `multiband/residuals.py` |
| End-to-end run script | **Done** | `run_gibbs_sampler.py` |

### Source update architecture

The per-band source update uses BlackJAX + custom Gibbs, fully vmapped:

```
update_parity_bands(bands)           sampler/band_update.py
  ├── compute_filter_coefficients()  sampler/indicator_update.py  (vmapped)
  ├── update_indicators()            sampler/indicator_update.py  (vmapped, jax.lax.scan)
  └── rmh_step()                     sampler/source_update.py     (vmapped, BlackJAX)
```

The BlackJAX RMH kernel can be swapped for NUTS later without changing the
logdensity function or calling code.

### Pluggable hyperparameter model

`hyper_update_step()` is model-agnostic — it takes any NumPyro model function
and a dict of kwargs. The Gibbs loop is wired via three callables:

- `hyper_model_fn` — any NumPyro model (e.g. `hierarchical_model`)
- `hyper_model_kwargs_fn(all_bands) -> dict` — builds model kwargs from current band state
- `extract_hyper_fn(samples, all_bands, lambda_res_per_band)` — pushes NUTS
  samples back into bands (defaults to `default_extract_hyper`)

To plug in a new hierarchical model, write a new NumPyro model, a kwargs
builder, and (if sample site names differ) a new extractor. The Gibbs loop
itself does not change.

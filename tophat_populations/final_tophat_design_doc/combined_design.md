# Hierarchical Multi-Band Inference — Combined Design Document

## Overview

This document describes the full inference pipeline for jointly recovering individual resolved sources, the confusion noise background, and population hyperparameters from a multi-band gravitational-wave dataset of galactic white-dwarf binaries.

The system combines two subsystems documented separately:

- **Multi-band Gibbs sampler** (`multiband_design.md`): even/odd parallel updates of per-band source parameters and spike-and-slab indicators
- **MDN confusion noise emulator** (`mdn_emulator_design.md`): a pre-trained mixture density network that provides the prior on `log S_conf` per band, conditioned on population hyperparameters

This document describes how they fit together: the full generative model, the complete Gibbs cycle, the data flow between subsystems, initialisation, and the implementation plan.

---

## 1. The Full Generative Model

### 1.1 Hierarchy

The model has three levels:

```
Level 3 (population):    Λ = (N_tot, α, β)
                              │
                              ▼
Level 2 (per-band):      log S_conf,k  and  λ_res,k    for k = 1, ..., N_bands
                              │                  │
                              ▼                  ▼
Level 1 (per-source):    {A_j, f_j, φ_j, z_j}          for sources j in band k
                              │
                              ▼
Data:                     d(f)  — the frequency-domain data (periodogram)
```

### 1.2 Generative story

```
1. Draw population hyperparameters:
     N_tot ~ Prior(...)
     α     ~ Uniform(1.5, 4.0)
     β     ~ Uniform(0.0, 0.5)

2. For each frequency band k = 1, ..., N_bands:

   a. Draw confusion noise from the MDN prior:
        log S_conf,k ~ MDNPrior(Λ, f_k)
        (the MDN has absorbed the self-consistency condition)

   b. Compute the resolve threshold (deterministic):
        S_tot,k = S_instr,k + exp(log S_conf,k)
        A_th,k  = (ρ_th / 2) * sqrt(S_tot,k / T_obs)

   c. Compute the Poisson rate for resolved sources (deterministic):
        p_f,k   = (1 - β f_k) / Z · Δf_band
        p_res,k = (A_th,k^{1-α} - A_max^{1-α}) / (A_min^{1-α} - A_max^{1-α})
        λ_res,k = N_tot · p_f,k · p_res,k

   d. Draw resolved source count:
        N_res,k ~ Poisson(λ_res,k)

   e. For each template slot j in band k:
        z_j ~ Bernoulli(N_res,k / N_templates,k)   [spike-and-slab indicator]
        if z_j = 1:
            A_j   ~ p(A | A > A_th,k)              [truncated power law]
            f_j   ~ Uniform(f_k^low, f_k^high)
            φ_j   ~ Uniform(0, 2π)

3. The data in each frequency bin is:
     d(f) = Σ_{active sources} h_j(f) + n_instr(f) + n_conf(f)

   where the Whittle likelihood treats the periodogram as:
     |d(f)|^2 / S_tot(f)  ~  Exponential(1)
```

### 1.3 Plate diagram

See `plate_diagram.pdf` / `plate_diagram.tex` for the graphical model of levels 2–3 (the MDN prior on `S_conf`, the analytic Poisson for `N_res`, and the hyperparameters). Level 1 (individual source parameters) is internal to each band's Gibbs update.

---

## 2. The Complete Gibbs Cycle

Each iteration of the sampler updates all three levels. The multi-band even/odd scheme handles levels 1–2, and the MDN handles the level 2–3 interface.

```
┌──────────────────────────────────────────────────────────────────┐
│  LEVEL 1–2: EVEN BAND UPDATE  (parallel, k = 2, 4, 6, ...)     │
│                                                                  │
│  For each even band (in parallel):                               │
│    1. Compute residual = data - fixed odd-neighbour templates    │
│    2. Update source parameters (A_j, f_j, φ_j) via Whittle      │
│       likelihood on the residual in B_k^ext                     │
│    3. Update z_j indicators (spike-and-slab)                     │
│    4. Compute local S_conf,k and N_res,k = Σ z_j                │
├────────────────────── sync ──────────────────────────────────────┤
│  MIGRATE EVEN → ODD                                              │
│  Transfer sources that crossed band boundaries                   │
├──────────────────────────────────────────────────────────────────┤
│  LEVEL 1–2: ODD BAND UPDATE  (parallel, k = 1, 3, 5, ...)      │
│                                                                  │
│  Same as above, roles swapped                                    │
├────────────────────── sync ──────────────────────────────────────┤
│  MIGRATE ODD → EVEN                                              │
│  Transfer sources that crossed band boundaries                   │
├──────────────────────────────────────────────────────────────────┤
│  LEVEL 2–3: HYPERPARAMETER UPDATE  (global)                      │
│                                                                  │
│  Inputs from all bands:                                          │
│    S_conf,k  — confusion PSD per band (from step 4 above)       │
│    N_res,k   — resolved count per band (from step 4 above)      │
│    f_k       — band center frequencies                           │
│                                                                  │
│  Sample Λ = (N_tot, α, β) and {log S_conf,k}:                   │
│    log S_conf,k ~ MDNPrior(Λ, f_k)           [1D Gaussian mix]  │
│    N_res,k | S_conf,k, Λ ~ Poisson(λ_res,k)  [analytic]         │
│    Λ ~ Prior(...)                              [hyperpriors]      │
│                                                                  │
│  Use HMC/NUTS on the joint (Λ, {log S_conf,k}) posterior.       │
│  The MDN params are frozen — not sampled.                        │
├────────────────────── sync ──────────────────────────────────────┤
│  → next iteration                                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Between Subsystems

### 3.1 Multi-band → MDN (upward flow)

After both parity updates complete, each band reports two numbers:

```python
S_conf_k = band_k.confusion_psd()       # from unresolved sources
N_res_k  = jnp.sum(band_k.z_indicators) # count of active templates
```

These are collected into arrays `S_conf_all`, `N_res_all` of shape `(N_bands,)` and passed to the hyperparameter update.

### 3.2 MDN → Multi-band (downward flow)

The hyperparameter update produces:

```python
Lambda = (N_tot, alpha, beta)       # updated population hyperparameters
log_S_conf = (N_bands,) array       # updated latent S_conf per band
```

These feed back into the per-band updates in two ways:

1. **S_conf sets the noise floor for the Whittle likelihood.** Each band's total noise PSD is `S_tot,k = S_instr,k + exp(log_S_conf,k)`. This determines the SNR of individual sources and thus which templates are active.

2. **Λ sets the expected source count per band** via the Poisson rate `λ_res,k`. This informs the spike-and-slab prior on the `z_j` indicators — bands with higher expected `λ_res` have a higher prior probability of template activation.

### 3.3 The consistency loop

There is a subtle feedback loop:

```
Λ → MDN prior on S_conf
  → S_conf sets A_th
    → A_th determines which sources are resolvable
      → resolved sources are subtracted from data
        → residual determines S_conf (measured)
          → measured S_conf constrains Λ
```

This loop is the self-consistency condition. In the Gibbs sampler, it is broken into conditional updates that cycle through the chain. The MDN has learned the equilibrium distribution of `S_conf | Λ` from iterative subtraction simulations, so the sampler converges to the joint posterior over all levels without ever explicitly solving the self-consistency equation.

---

## 4. Initialisation

### 4.1 Strategy

Cold-start initialisation can be slow to converge because the levels are coupled. A warm-start strategy:

```
1. INITIAL S_conf ESTIMATE:
   Compute a rough periodogram of the data.
   Smooth it to get an initial S_tot estimate.
   Subtract S_instr to get initial S_conf,k per band.

2. INITIAL SOURCE DETECTION:
   Using the initial S_tot as the noise floor, compute SNR
   for a grid of template frequencies.
   Activate (z_j = 1) all templates exceeding ρ_th.
   Set initial A_j, f_j from the matched-filter estimates.

3. INITIAL Λ:
   From the initial S_conf and N_res per band, use the MDN
   in reverse: find the Λ that maximises the MDN likelihood.
   (Or just use a reasonable prior mean.)

4. BURN-IN:
   Run the full Gibbs cycle for N_burn iterations to
   equilibrate all three levels. Monitor S_conf and Λ traces
   for convergence.
```

### 4.2 NumPyro init strategy

For the hyperparameter block (which uses NumPyro HMC internally), use `init_to_value` with the warm-start Λ and S_conf estimates:

```python
init_values = {
    "N_tot": N_tot_init,
    "alpha": alpha_init,
    "beta": beta_init,
    **{f"log_S_conf_{k}": log_S_conf_init[k] for k in range(N_bands)}
}
init_strategy = numpyro.infer.init_to_value(values=init_values)
```

Note: `init_to_value` takes **constrained** values (the actual parameter values), not unconstrained. This is consistent with NumPyro's API (see project notes on the `init_params` vs `init_to_value` inconsistency).

---

## 5. Implementation Plan

### 5.1 Phasing

The system should be built and tested incrementally:

```
Phase 1: Single-band baseline
  - Existing gb_spike_slab code
  - Verify source recovery and S_conf estimation
  - This is the foundation; do not proceed until it works

Phase 2: MDN emulator (standalone)
  - Generate training data via iterative subtraction
  - Train gated MDN on log S_conf
  - Validate: check calibration, gate accuracy, Poisson branch
  - Test: MDNPrior distribution in NumPyro, sample + log_prob
  - Deliverable: trained MDN params + MDNPrior class

Phase 3: Single-band + MDN
  - Replace the analytic self-consistency solver with MDNPrior
  - Add Poisson(N_res) observation
  - Add Λ as sampled hyperparameters
  - Verify: Λ posterior recovers truth on synthetic single-band data
  - This tests the level 2–3 interface without multi-band complexity

Phase 4: Multi-band (even/odd) without MDN
  - Implement band partitioning, buffer zones, residual passing
  - Implement source migration protocol
  - Use a FIXED S_conf per band (no hyperparameter update)
  - Verify: sources near band boundaries are recovered correctly
  - Verify: even/odd parallelism gives same results as sequential
  - This tests the level 1–2 multi-band mechanics in isolation

Phase 5: Full system
  - Combine Phase 3 (MDN hyperparameter update) with Phase 4
    (multi-band even/odd)
  - Verify: Λ posterior recovers truth on synthetic multi-band data
  - Verify: S_conf,k and N_res,k across bands are consistent
  - Performance: profile and optimise the parallel updates
```

### 5.2 What to defer

Some extensions should be deferred until the core system works:

- **Chirped templates**: the current model uses top-hat (non-chirping) waveforms. Extending to ḟ ≠ 0 changes the template overlap structure but not the even/odd scheme.
- **Non-Gaussian likelihood**: the compound Poisson / characteristic function approach for the transition regime (μ ~ 1 sources per bin). For now, use the Whittle likelihood everywhere.
- **Multi-channel (A, E, T)**: LISA TDI channels. The Gibbs structure is the same; the likelihood becomes a sum over channels.
- **Anisotropy / sky position**: adds parameters per source but doesn't change the band structure.

---

## 6. File Structure

```
gb_spike_slab/
├── mdn/
│   ├── __init__.py
│   ├── network.py            # Gated MDN: forward pass, init, MDNPrior distribution
│   ├── training.py           # Training loop, loss, validation
│   ├── iterative_sub.py      # Iterative subtraction for training data
│   ├── data_gen.py           # Build training set from population draws
│   └── poisson.py            # compute_lambda_res, p_res analytics
│
├── multiband/
│   ├── __init__.py
│   ├── bands.py              # Band dataclass, buffer logic, source ownership
│   ├── residuals.py          # Compute residuals with buffer subtraction
│   ├── migration.py          # Source migration detection and transfer
│   └── parallel.py           # Even/odd dispatch, vmap over bands
│
├── sampler/
│   ├── __init__.py
│   ├── gibbs.py              # Top-level Gibbs loop (this document's Section 2)
│   ├── source_update.py      # Per-band source parameter update (Whittle)
│   ├── indicator_update.py   # Per-band z_j spike-and-slab update
│   ├── hyper_update.py       # Λ + {log S_conf,k} update via NumPyro HMC
│   └── model.py              # NumPyro generative model (MDN prior + Poisson)
│
├── scripts/
│   ├── generate_training_data.py
│   ├── train_mdn.py
│   ├── run_single_band.py    # Phase 1/3 testing
│   ├── run_multiband.py      # Phase 4/5 testing
│   └── diagnostics.py        # R-hat, ESS, trace plots, S_conf vs f plots
│
├── tests/
│   ├── test_mdn.py           # MDN unit tests (see mdn_emulator_design.md §7)
│   ├── test_multiband.py     # Multi-band unit tests (see multiband_design.md §9)
│   ├── test_integration.py   # End-to-end synthetic data recovery
│   └── test_migration.py     # Migration-specific tests
│
└── docs/
    ├── combined_design.md        # This document
    ├── mdn_emulator_design.md    # MDN subsystem design
    ├── multiband_design.md       # Multi-band subsystem design
    └── plate_diagram.tex/.pdf    # Graphical model
```

---

## 7. Key Interfaces

### 7.1 MDN → Sampler

```python
# In sampler/model.py:
from mdn.network import MDNPrior

# Used inside the NumPyro model:
log_S_conf_k = numpyro.sample(
    f"log_S_conf_{k}",
    MDNPrior(mdn_params, Lambda, f_centers[k], X_norm_stats)
)
```

### 7.2 Multi-band → Sampler

```python
# In sampler/gibbs.py:
from multiband.parallel import update_even_bands, update_odd_bands
from multiband.migration import migrate_sources
from multiband.residuals import compute_residuals

# One Gibbs iteration:
even_residuals = compute_residuals(data, even_bands, odd_bands)
even_bands = update_even_bands(even_bands, even_residuals, rng_key)
migrate_sources(even_bands, odd_bands)

odd_residuals = compute_residuals(data, odd_bands, even_bands)
odd_bands = update_odd_bands(odd_bands, odd_residuals, rng_key)
migrate_sources(odd_bands, even_bands)
```

### 7.3 Sampler → MDN (hyperparameter update)

```python
# In sampler/hyper_update.py:
from sampler.model import hierarchical_model

S_conf_all = collect_S_conf(all_bands)
N_res_all  = collect_N_res(all_bands)

# Run NumPyro HMC on the hierarchical model
kernel = NUTS(hierarchical_model)
mcmc = MCMC(kernel, num_warmup=0, num_samples=1)  # single step
mcmc.run(rng_key, S_conf_obs=S_conf_all, N_res_obs=N_res_all, ...)
Lambda_new = mcmc.get_samples()
```

Note: the hyperparameter update runs a **single HMC step** (or a few leapfrog steps) per Gibbs iteration, not a full MCMC chain. It is one block of the Gibbs sampler. The outer Gibbs loop provides the iteration.

---

## 8. Diagnostics and Validation

### 8.1 Per-iteration monitoring

At each Gibbs iteration, log:
- `Λ = (N_tot, α, β)` — trace plots for convergence
- `S_conf(f_k)` for all bands — trace + spectrum plot
- `N_res(f_k)` for all bands — compare to Poisson(λ_res) expectation
- Total number of active sources `Σ_k N_res,k`
- Number of source migrations (should be ~0 most iterations)
- Gate probabilities `g(Λ, f_k)` per band (fraction expected fully resolved)

### 8.2 Convergence diagnostics

- **R-hat**: compute for Λ components across multiple chains
- **ESS**: effective sample size for Λ, S_conf, and individual loud source parameters
- **Geweke diagnostic**: compare first 10% and last 50% of chain

### 8.3 Synthetic data validation (ground truth known)

The ultimate test: generate synthetic data from a known `Λ_true`, run the full pipeline, verify:
- Λ posterior contains Λ_true at appropriate credible level
- Recovered S_conf(f) matches the true confusion noise spectrum
- Individual bright sources are recovered with correct parameters
- N_res per band is consistent with the known population

---

## 9. Summary of Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| MDN target | 1D (log S_conf only) | N_res is analytic given S_conf; avoids negative-count issue |
| MDN architecture | Gated MDN (gate + GMM) | Handles S_conf = 0 edge case cleanly |
| Self-consistency | Encoded in MDN training data | No iterative solver at inference time |
| N_res likelihood | Analytic Poisson | Correct integer distribution, no network needed |
| N_tot interpretation | Fixed physical number | Not a Poisson rate; scatter from bin allocation |
| Band parallelisation | Even/odd (red-black) | Adjacent bands coupled by template overlap width w |
| Band independence condition | Δf_band > w | Even bands separated by full odd band |
| Source migration | Transfer at sync point | Migrant keeps params, lands in destination before residuals |
| Slot allocation | Use inactive z=0 slots | No wasted memory; migration is rare |
| Hyperparameter sampler | Single HMC step per Gibbs iteration | One block of the outer Gibbs loop |
| Implementation | JAX / NumPyro | GPU-native, vmap for parallelism, NumPyro for HMC block |

# MDN Confusion Noise Emulator — Design Document

## Overview

We are building a **Mixture Density Network (MDN)** that learns the stochastic mapping from population hyperparameters to confusion noise PSD across frequency bands. The MDN handles only the 1D distribution of `log S_conf`; the resolved source count `N_res` is computed analytically via a Poisson factor conditioned on `S_conf`. This replaces an analytic self-consistency solver (Newton iteration for `S_conf ↔ A_th`) with a pre-trained emulator that can be evaluated in microseconds during MCMC sampling.

The key decomposition is:

```
p(log S_conf, N_res | Λ, f_k) = p_MDN(log S_conf | Λ, f_k) · Poisson(N_res; λ_res(S_conf, Λ, f_k))
```

where the MDN learns the first factor from iterative subtraction simulations, and the second factor is closed-form given `S_conf` and `Λ`. The self-consistency condition is baked into the MDN's learned distribution — it absorbed the output of thousands of iterative subtraction runs during training.

The full pipeline has three stages:
1. **Training data generation** via iterative subtraction on simulated populations
2. **MDN training** with a standard gradient descent loop (Adam + optax)
3. **Integration into the Gibbs sampler** as a NumPyro generative model

Everything is in **JAX/NumPyro**. No PyTorch, no TensorFlow.

---

## 1. Source Model (Toy Top-Hat)

### Population model

The population is described by hyperparameters `Λ = (N_tot, α, β)`:

- **Total number of sources:** `N_tot`
- **Amplitude distribution:** `p(A) ∝ A^{-α}` on `[A_min, A_max]`
- **Frequency distribution:** `p(f) ∝ (1 - β·f)` on `[f_min, f_max]` (linear taper with frequency)
- **Phases:** uniform on `[0, 2π)`

### Sources of stochasticity

`N_tot` is a **fixed physical number** — there is some definite count of DWD binaries in the galaxy. It is not drawn from a Poisson rate process (unlike e.g. LIGO merger rates). We place a prior on `N_tot` reflecting our ignorance, but it is not itself a random variable with a Poisson generating process.

The stochasticity in `S_conf` at fixed `Λ` comes from two sources:

1. **Multinomial allocation to frequency bands.** Given `N_tot` and `p(f)`, the number of sources landing in band `k` is approximately Poisson with mean `μ_k = N_tot · p(f_k) · Δf_band`. This is the standard Poisson limit of the multinomial for many bins.
2. **Amplitude draws.** Which sources in a band cross the SNR threshold depends on the random amplitude draws from `p(A) ∝ A^{-α}`.

The MDN learns the distribution of `log S_conf` that results from both effects combined. The resolved count `N_res` is then derived analytically: given `S_conf`, the resolve threshold `A_th` is deterministic, the resolve probability `p_res = P(A > A_th)` is a closed-form integral of the power law, and `N_res ~ Poisson(μ_k · p_res)`. See Section 5.

The gate (Section 3.2) handles the discrete event where all sources in a band are resolved (`S_conf = 0`).

### Waveform model

Each source is a **top-hat in the frequency domain**: constant amplitude `A_k` over a bandwidth `w` centered at frequency `f_k`. The frequency-domain signal occupies bins `[f_k - w/2, f_k + w/2]`.

### SNR calculation

For a source with time-domain strain amplitude `A_k` at frequency `f_k`, against a total noise PSD `S_tot(f_k)`:

```
ρ_k = 2 * A_k * sqrt(T_obs / S_tot(f_k))
```

where `T_obs` is the observation time. The template bandwidth `w` does **not** enter here — the matched filter coherently sums over all frequency bins the signal occupies, and `A` already encodes the total signal power.

### Confusion noise PSD

The confusion noise in a frequency band centered at `f_k` with bandwidth `Δf_band` is the sum of power from all unresolved sources in that band:

```
S_conf(f_k) = (2 / Δf_band) * Σ_{unresolved sources in band k} A_j^2
```

The factor of 2 comes from the one-sided PSD convention (all power at positive frequencies), consistent with the factor of 2 in the SNR formula. The `1/Δf_band` is the discrete approximation to `dN/df` within the band. The template bandwidth `w` does **not** appear here — total signal power is set by `A` regardless of how the Fourier power is distributed across bins.

---

## 2. Iterative Subtraction (Training Data Generation)

This generates `(input, output)` pairs for the MDN. For each training sample:

### Algorithm

```
Input:  Λ = (N_tot, α, β), drawn from a prior
Output: S_conf(f_k), N_res(f_k) at each band center f_k

1. Draw N_tot source frequencies from p(f) ∝ (1 - β·f)
   - Use inverse CDF sampling (see Section 2.1 below)

2. Draw N_tot amplitudes from p(A) ∝ A^{-α} on [A_min, A_max]
   - Inverse CDF: A = (A_min^{1-α} + u·(A_max^{1-α} - A_min^{1-α}))^{1/(1-α)}
   - Special case α=1: A = A_min * exp(u * log(A_max/A_min))

3. Initialise: resolved_set = {} (empty)

4. Iterate until convergence (typically 3-6 iterations):
   a. Compute S_conf(f_k) from currently UNRESOLVED sources only
   b. Set S_tot(f_k) = S_instr(f_k) + S_conf(f_k)
   c. Compute SNR for ALL sources: ρ_k = 2 * A_k * sqrt(T_obs / S_tot(f_k))
   d. Update resolved_set = {sources with ρ_k > ρ_th}
   e. If resolved_set hasn't changed → converged, stop

5. Record S_conf(f_k) and N_res(f_k) = count of resolved sources per band
```

### 2.1 Inverse CDF for frequency distribution

For `p(f) ∝ (1 - β·f)` on `[f_min, f_max]`:

```
Normalisation: Z = (f_max - f_min) - β·(f_max^2 - f_min^2)/2
CDF: F(f) = [(f - f_min) - β·(f^2 - f_min^2)/2] / Z
```

Invert numerically (e.g. with `jnp.searchsorted` on a precomputed table, or Newton's method on the quadratic).

### 2.2 Training set structure

From a single realisation with `N_bands` frequency bands, we get `N_bands` training rows:

```
X_n = (N_tot, α, β, f_k)         — shape (4,)
Y_n = log S_conf(f_k)             — scalar (1D target)
is_zero_n = (S_conf(f_k) == 0)    — boolean scalar
```

The MDN target is **1D** — only `log S_conf`. The resolved source count `N_res` is not a training target; it is computed analytically from `S_conf` and `Λ` via the Poisson factor at inference time (see Section 5).

We work with `log S_conf` because it spans orders of magnitude. For fully-resolved bands where `S_conf = 0`, the `log S_conf` value is unused (the gated likelihood handles this case analytically — see Section 3.2). Set it to a sentinel like `0.0`; it won't affect training because the gate branch doesn't use it.

Generate `N_train` realisations (each with different Λ drawn from the prior), yielding `N_train × N_bands` total training rows.

### 2.3 Prior ranges for training

Cover the prior volume generously. Example:

```python
Lambda_bounds = {
    'N_tot': (1_000, 100_000),    # or whatever your expected range is
    'alpha': (1.5, 4.0),
    'beta':  (0.0, 0.5),          # units depend on f normalisation
}
```

Sample `N_tot` uniformly (or log-uniformly for wide ranges), `α` and `β` uniformly.

### 2.4 Practical notes

- The iterative subtraction loop uses a **Python for-loop** (not jittable) because the number of iterations is unknown and the resolved set changes. This is fine — it only runs during training data generation, not during MCMC.
- Typical convergence: 3-6 iterations.
- `N_train = 5000-10000` realisations is a reasonable starting point. Scale up if validation loss plateaus.

---

## 3. Gated MDN Architecture

### 3.1 The problem with S_conf = 0

When all sources in a band are resolved, `S_conf = 0` and `log S_conf → -∞`. A Gaussian mixture cannot represent a point mass at `-∞`. In practice the MDN tries to push one component's mean to very negative values and shrink its variance, distorting the fit for nearby non-zero cases too (because the hidden layers are shared).

### 3.2 Solution: zero-inflated gated model

We split the density into two branches:

```
p(log S_conf | x) = (1 - g(x)) · p_MDN(log S_conf | x)    [not fully resolved]
                  +     g(x)   · δ(S_conf = 0)              [fully resolved]
```

where `g(x) = sigmoid(gate_logit)` is the probability that band `(Λ, f_k)` is fully resolved. The MDN head only needs to model the 1D density of `log S_conf` conditional on there being *some* unresolved sources. The gate handles the discrete question separately.

### 3.3 Network structure

```
Input x = (N_tot, α, β, f_k)      — 4 dimensions
            │
    ┌───────┴───────┐
    │  shared trunk │
    │  hidden 1     │   n_hidden neurons, tanh
    │  hidden 2     │   n_hidden neurons, tanh
    └───────┬───────┘
            │
    ┌───────┴───────────┐
    │                    │
  gate head           MDN head
  (1 logit →          (K mixture params →
   sigmoid)            log_π, μ, σ for 1D target)
```

The **gate head** produces `p(fully resolved | x)`. The **MDN head** produces 1D Gaussian mixture parameters for `log S_conf`.

### 3.4 Output layer partitioning (MDN head)

The MDN target is now **1D** (`d = 1`, just `log S_conf`). Raw output `r ∈ R^{3K}` split into:

```
r = [logits (K values) | raw_means (K values) | raw_log_sigmas (K values)]

log π_j = log_softmax(logits)          — ensures Σ π_j = 1, π_j > 0
μ_j     = raw_means[j]                 — unconstrained
σ_j     = exp(raw_log_sigmas[j])       — ensures σ > 0, clip at 1e-6
```

Total output dimension: `3K` (e.g. 15 for K=5).

### 3.5 Parameter initialisation

Use He initialisation for weights: `w ~ Normal(0, sqrt(2/fan_in))`, biases = 0.

```python
def init_layer(key, d_in, d_out):
    k1, k2 = jax.random.split(key)
    scale = jnp.sqrt(2.0 / d_in)
    w = scale * jax.random.normal(k1, (d_in, d_out))
    b = jnp.zeros(d_out)
    return w, b
```

Store params as a dict:
```python
params = {
    'layer1': (w1, b1),     # shapes: (4, n_hidden), (n_hidden,)
    'layer2': (w2, b2),     # shapes: (n_hidden, n_hidden), (n_hidden,)
    'mdn_out': (w3, b3),    # shapes: (n_hidden, 3*K), (3*K,)
    'gate_w': w_gate,        # shape: (n_hidden, 1)
    'gate_b': b_gate,        # shape: (1,)
}
```

### 3.6 Forward pass

```python
def gated_mdn_forward(params, x, n_components=5):
    """
    x: (..., 4)
    Returns:
        log_g:     (...,)   log p(fully resolved)
        log_1mg:   (...,)   log p(not fully resolved)
        log_pi:    (..., K) mixture log-weights
        mu:        (..., K) mixture means (for 1D log S_conf)
        sigma:     (..., K) mixture stdevs
    """
    K = n_components
    w1, b1 = params['layer1']
    w2, b2 = params['layer2']
    w3, b3 = params['mdn_out']

    # Shared trunk
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)

    # Gate head
    gate_logit = (h @ params['gate_w'] + params['gate_b'])[..., 0]
    log_g   = jax.nn.log_sigmoid(gate_logit)
    log_1mg = log_g - gate_logit  # = -softplus(gate_logit)

    # MDN head (1D target: 3K outputs)
    r = h @ w3 + b3
    logits   = r[..., :K]
    mu       = r[..., K:2*K]
    log_sig  = r[..., 2*K:3*K]

    log_pi = jax.nn.log_softmax(logits, axis=-1)
    sigma  = jnp.exp(log_sig)
    sigma  = jnp.clip(sigma, 1e-6)

    return log_g, log_1mg, log_pi, mu, sigma
```

### 3.7 Analytic Poisson factor for N_res

Given `S_conf`, the resolve threshold and Poisson rate are deterministic:

```python
def compute_lambda_res(S_conf, Lambda, f_k, S_instr_k,
                        T_obs, rho_th, A_min, A_max,
                        delta_f_band, f_min, f_max):
    """
    Compute the Poisson rate λ_res for resolved sources in band k.
    All inputs are scalars.
    """
    N_tot, alpha, beta = Lambda

    # Amplitude threshold from S_conf
    S_tot = S_instr_k + S_conf
    A_th = (rho_th / 2.0) * jnp.sqrt(S_tot / T_obs)

    # Resolve probability: fraction of p(A) above threshold
    # p(A) ∝ A^{-α} on [A_min, A_max]
    A_th_clipped = jnp.clip(A_th, A_min, A_max)
    p_res = jnp.where(
        A_th < A_min,
        1.0,  # everything resolved
        (A_th_clipped**(1-alpha) - A_max**(1-alpha)) /
        (A_min**(1-alpha) - A_max**(1-alpha))
    )

    # Expected number of sources in this band
    Z = (f_max - f_min) - beta * (f_max**2 - f_min**2) / 2
    p_f = (1 - beta * f_k) / Z
    mu_k = N_tot * p_f * delta_f_band

    return mu_k * p_res
```

---

## 4. Training

### 4.1 Loss function

The gated log-likelihood for the **1D** target `log S_conf`:

```python
def gated_mdn_log_prob(params, x, y, is_zero, n_components=5):
    """
    x:        (N, 4) — normalised inputs
    y:        (N,)   — log S_conf values (unused where is_zero=True)
    is_zero:  (N,)   — boolean, True if S_conf = 0 for this sample
    Returns:  (N,) log-probabilities
    """
    log_g, log_1mg, log_pi, mu, sigma = gated_mdn_forward(params, x, n_components)
    # log_pi: (N, K), mu: (N, K), sigma: (N, K)

    # --- MDN branch (1D Gaussian mixture) ---
    y_exp = y[:, None]  # (N, 1)
    log_gauss = -0.5 * (
        ((y_exp - mu) / sigma)**2 + 2*jnp.log(sigma) + jnp.log(2*jnp.pi)
    )  # (N, K)
    log_mdn = jax.nn.logsumexp(log_pi + log_gauss, axis=-1)  # (N,)

    # --- Combine via gate ---
    log_prob = jnp.where(
        is_zero,
        log_g,                  # fully resolved: just gate probability
        log_1mg + log_mdn       # continuous branch: gate × GMM
    )
    return log_prob


def gated_mdn_loss(params, x, y, is_zero):
    return -jnp.mean(gated_mdn_log_prob(params, x, y, is_zero))
```

Note: the fully-resolved branch contributes only the gate term `log g(x)` during training. There is no `N_res` target to fit here — the Poisson factor for `N_res` is computed analytically at inference time, not learned. Both branches flow gradients through the shared trunk.

### 4.2 Training loop

```python
import optax

def train_mdn(key, X_train, Y_train, is_zero_train, n_steps=5000, lr=1e-3):
    """
    X_train:        (N, 4) normalised inputs
    Y_train:        (N,)   log S_conf values
    is_zero_train:  (N,)   boolean, True where S_conf = 0
    """
    params = init_mdn_params(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y, is_zero):
        loss, grads = jax.value_and_grad(gated_mdn_loss)(params, x, y, is_zero)
        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    for i in range(n_steps):
        params, opt_state, loss = step(
            params, opt_state, X_train, Y_train, is_zero_train)
        if i % 500 == 0:
            print(f"Step {i}: loss = {loss:.4f}")

    return params
```

### 4.3 Input normalisation

**Important:** normalise the inputs before training. Each feature should be roughly zero-mean, unit-variance. Compute mean/std from the training set and store them alongside the MDN params:

```python
X_mean = X_train.mean(axis=0)
X_std  = X_train.std(axis=0)
X_train_norm = (X_train - X_mean) / X_std
```

Apply the same transform at inference time.

### 4.4 Validation

Hold out ~20% of realisations (not rows — entire realisations, to avoid data leakage from correlated bands). Monitor validation loss. If it plateaus while training loss decreases → overfitting → add dropout or reduce `n_hidden`.

### 4.5 Hyperparameters to tune

| Parameter | Starting value | Notes |
|---|---|---|
| `n_hidden` | 64 | Try 32, 128 |
| `K` (mixture components) | 5 | Try 3, 8, 10 |
| `lr` | 1e-3 | Reduce if training unstable |
| `n_steps` | 5000 | Increase if loss still dropping |
| `N_train` | 5000 realisations | Scale up if val loss high |

---

## 5. Integration into the Gibbs Sampler

### 5.1 Custom NumPyro distribution

The MDN is wrapped as a NumPyro distribution so that `log S_conf` becomes a proper latent variable in the generative model:

```python
import numpyro.distributions as dist
from numpyro.distributions import constraints

class MDNPrior(dist.Distribution):
    """
    1D Gaussian mixture prior on log S_conf, conditioned on (Λ, f_k).
    Wraps a pre-trained gated MDN.
    """
    support = constraints.real

    def __init__(self, mdn_params, Lambda, f_k, X_norm_stats):
        self.mdn_params = mdn_params
        # Build and normalise input
        x_raw = jnp.array([*Lambda, f_k])
        X_mean, X_std = X_norm_stats
        self.x = (x_raw - X_mean) / X_std

        # Forward pass (once per construction)
        log_g, log_1mg, log_pi, mu, sigma = gated_mdn_forward(
            mdn_params, self.x[None, :])
        self.log_g   = log_g[0]
        self.log_1mg = log_1mg[0]
        self.log_pi  = log_pi[0]     # (K,)
        self.mu      = mu[0]         # (K,)
        self.sigma   = sigma[0]      # (K,)

        batch_shape = ()
        event_shape = ()
        super().__init__(batch_shape, event_shape)

    def log_prob(self, value):
        # 1D Gaussian mixture log-prob
        log_gauss = -0.5 * (
            ((value - self.mu) / self.sigma)**2
            + 2 * jnp.log(self.sigma)
            + jnp.log(2 * jnp.pi)
        )  # (K,)
        log_mdn = jax.nn.logsumexp(self.log_pi + log_gauss)

        # Use the continuous branch
        # (for the fully-resolved edge case, the gate probability
        #  will push this sample toward S_conf → 0 naturally)
        return self.log_1mg + log_mdn

    def sample(self, key, sample_shape=()):
        k1, k2 = jax.random.split(key)
        # Choose a mixture component
        j = jax.random.categorical(k1, self.log_pi)
        # Draw from that Gaussian
        return self.mu[j] + self.sigma[j] * jax.random.normal(k2)
```

### 5.2 The generative model

The NumPyro model reads exactly like the physics:

```python
def model(data_residual_psd, z_counts_per_band, f_centers, 
          mdn_params, X_norm_stats, S_instr, T_obs, rho_th,
          A_min, A_max, delta_f_band, f_min, f_max):
    """
    Generative model for multi-band confusion noise + resolved counts.
    
    Parameters
    ----------
    data_residual_psd : (N_bands,) current residual PSD in each band
                        (data minus resolved templates)
    z_counts_per_band : (N_bands,) number of active z_k=1 indicators per band
    f_centers :         (N_bands,) band center frequencies
    mdn_params :        trained MDN weights (frozen, not sampled)
    X_norm_stats :      (X_mean, X_std) from training normalisation
    S_instr :           (N_bands,) instrumental noise PSD
    """
    N_bands = len(f_centers)

    # --- Population hyperparameters ---
    N_tot = numpyro.sample("N_tot", dist.Uniform(1000, 100000))
    alpha = numpyro.sample("alpha", dist.Uniform(1.5, 4.0))
    beta  = numpyro.sample("beta",  dist.Uniform(0.0, 0.5))
    Lambda = jnp.array([N_tot, alpha, beta])

    # --- Frequency distribution normalisation ---
    Z_freq = (f_max - f_min) - beta * (f_max**2 - f_min**2) / 2

    for k in range(N_bands):
        # 1. Sample log S_conf from the MDN's learned prior
        log_S_conf_k = numpyro.sample(
            f"log_S_conf_{k}",
            MDNPrior(mdn_params, Lambda, f_centers[k], X_norm_stats)
        )
        S_conf_k = jnp.exp(log_S_conf_k)

        # 2. Deterministic: resolve threshold from S_conf
        S_tot_k = S_instr[k] + S_conf_k
        A_th_k = (rho_th / 2.0) * jnp.sqrt(S_tot_k / T_obs)

        # 3. Deterministic: resolve probability from amplitude distribution
        A_th_clipped = jnp.clip(A_th_k, A_min, A_max)
        p_res_k = jnp.where(
            A_th_k < A_min,
            1.0,
            (A_th_clipped**(1 - alpha) - A_max**(1 - alpha)) /
            (A_min**(1 - alpha) - A_max**(1 - alpha))
        )

        # 4. Deterministic: expected sources in this band
        p_f_k = (1 - beta * f_centers[k]) / Z_freq
        mu_k = N_tot * p_f_k * delta_f_band
        lambda_res_k = mu_k * p_res_k

        # 5. Observe: N_res = sum of z_k indicators in this band
        numpyro.sample(
            f"N_res_{k}",
            dist.Poisson(jnp.clip(lambda_res_k, 1e-10)),
            obs=z_counts_per_band[k]
        )
```

### 5.3 How it reads as physics

The generative story is:

1. **Draw population hyperparameters** `Λ = (N_tot, α, β)` from priors
2. **At each frequency band**, the self-consistent confusion noise follows the MDN's learned distribution: `log S_conf(f_k) ~ MDNPrior(Λ, f_k)`
3. **Given `S_conf`**, the amplitude threshold `A_th` and resolve probability `p_res` are deterministic arithmetic — no iteration, no solver
4. **The number of resolved sources** follows: `N_res ~ Poisson(μ_k · p_res)`, observed as `Σ z_k` in that band

The self-consistency equation is never solved at inference time. It is encoded in the MDN's learned distribution, which absorbed the results of thousands of iterative subtraction runs during training. The MDN is a lookup table with uncertainty, not a solver.

### 5.4 What `log S_conf` is at each Gibbs step

`log S_conf_k` is a **latent variable** that gets sampled at each MCMC step. It is *not* directly observed. The data constrains it through the Poisson factor: if the proposed `S_conf` implies a resolve rate inconsistent with the observed number of active indicators `Σ z_k`, the sample gets rejected.

HMC explores the joint posterior over `Λ` and `{log S_conf_k}` together. The MDN prior shapes the marginal on `S_conf` and the Poisson observation links it to the data.

### 5.5 Where it fits in the Gibbs cycle

```
For each Gibbs iteration:
  1. Update individual source parameters (amplitudes, frequencies)
     — using the Whittle likelihood within each band
  2. Update z_k indicators (spike-and-slab, resolved/unresolved)
     — using DiscreteHMCGibbs with modified=True
  3. Update Λ = (N_tot, α, β) and {log S_conf_k}
     — HMC jointly over the continuous latents
     — the MDN prior on S_conf and Poisson(N_res) provide the likelihood
     — this is a standard NumPyro HMC step, no custom code needed
```

### 5.6 On the fully-resolved edge case

When the gate probability `g(x)` is high (all sources expected to be resolved), the MDN prior will push `log S_conf` toward very negative values. In the limit, `S_conf → 0`, `A_th → A_th_min` (set by instrumental noise alone), and `p_res → 1`, so `λ_res → μ_k`. The Poisson factor then constrains `N_res` against the total expected count in the band.

If you want a hard cutoff, you can add logic to set `S_conf = 0` when `g(x) > 0.99` and switch to the analytic Poisson-only branch. But in practice the continuous formulation should handle it smoothly.

---

## 6. File Structure

Suggested layout:

```
gb_spike_slab/
├── mdn/
│   ├── __init__.py
│   ├── network.py          # Gated MDN: forward pass, init_params, MDNPrior distribution
│   ├── training.py         # Training loop, loss function, validation, checkpointing
│   ├── iterative_sub.py    # Iterative subtraction for training data gen
│   ├── data_gen.py         # Build training set from population draws
│   └── poisson.py          # compute_lambda_res, p_res from (S_conf, Λ, f_k)
├── sampler/
│   ├── gibbs.py            # Main Gibbs loop
│   └── model.py            # NumPyro generative model (Section 5.2)
└── scripts/
    ├── generate_training_data.py
    ├── train_mdn.py
    └── run_sampler.py
```

---

## 7. Testing & Validation Checklist

- [ ] **Unit test: iterative subtraction converges** — run on a known population, verify resolved count matches expectations
- [ ] **Unit test: gated MDN log-prob is finite** — check no NaN/Inf for random inputs, including `is_zero=True` cases
- [ ] **Unit test: mixture weights sum to 1** — verify `exp(log_pi).sum(axis=-1) ≈ 1`
- [ ] **Unit test: gate outputs valid probabilities** — verify `0 ≤ sigmoid(gate_logit) ≤ 1`
- [ ] **Unit test: Poisson rate is non-negative** — verify `lambda_res ≥ 0` for all valid inputs, including edge cases `A_th < A_min` and `A_th > A_max`
- [ ] **Unit test: MDNPrior distribution** — verify `log_prob` and `sample` are consistent (sample from prior, evaluate log_prob, check distribution)
- [ ] **Validation: gate calibration** — compare predicted `p(fully resolved)` against empirical fraction in test set, binned by `(Λ, f)`
- [ ] **Validation: MDN captures the mean** — compare MDN predictive mean `Σ π_j μ_j` against sample mean of `log S_conf` from test realisations (non-zero cases only)
- [ ] **Validation: MDN captures the variance** — compare MDN predictive variance against sample variance from test realisations
- [ ] **Validation: 1D calibration** — for test samples, check that ~68% of `log S_conf` values fall within the MDN's ±1σ predictive interval
- [ ] **Validation: Poisson factor** — for test realisations, verify that `N_res` empirical distribution is consistent with `Poisson(λ_res(S_conf, Λ, f_k))` computed from the true `S_conf`
- [ ] **Validation: end-to-end** — sample `log S_conf` from MDN, compute `λ_res`, draw `N_res ~ Poisson(λ_res)`, compare joint `(S_conf, N_res)` distribution against iterative subtraction ground truth
- [ ] **Integration test:** run the Gibbs sampler on synthetic data with known Λ_true, verify posterior concentrates around truth
- [ ] **Convergence diagnostic:** check Λ and `log S_conf_k` chains with R-hat, ESS

---

## 8. Potential Extensions

- **Joint multi-band output:** instead of treating bands independently, have the MDN output `log S_conf` for all bands simultaneously to capture inter-band correlations (e.g. a realisation with extra sources at low f has fewer available at nearby frequencies). This would make `log S_conf` an `N_bands`-dimensional target, but the Poisson factors for `N_res` remain per-band and analytic.
- **Normalising flow:** if the 1D GMM doesn't capture the conditional density of `log S_conf` well enough, replace with a conditional normalising flow (more expressive, same closed-form density). May be overkill for a 1D target.
- **Input features:** could add derived quantities like `A_min/A_max`, or the analytic prediction for `S_conf` (from the simple `N_tot * <A^2> / Δf` formula) as an additional input to help the network.
- **Gate diagnostics:** monitor the gate's predicted `p(fully resolved)` vs the empirical fraction in validation data as a function of `(Λ, f)`. If the gate is miscalibrated, consider temperature scaling or adding a gate-specific hidden layer.
- **Vectorised model:** the `for k in range(N_bands)` loop in the NumPyro model can be replaced with `numpyro.plate` and vectorised MDN evaluation if performance is an issue. Requires batching the `MDNPrior` construction.

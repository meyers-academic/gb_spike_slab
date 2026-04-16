"""
Mixture Density Network for confusion noise emulation.

Maps population hyperparameters (log10(N_tot), alpha, beta, f_k) to a conditional
density over log S_conf using a 1D Gaussian mixture model with a gate for the
fully-resolved case (S_conf = 0).

The MDN only models p(log S_conf | Λ, f_k). The resolved source count N_res is
computed analytically via compute_lambda_res(), not learned by the MDN.
"""

import jax
import jax.numpy as jnp


def init_layer(key, d_in, d_out):
    """He initialisation for a single dense layer."""
    k1, k2 = jax.random.split(key)
    scale = jnp.sqrt(2.0 / d_in)
    w = scale * jax.random.normal(k1, (d_in, d_out))
    b = jnp.zeros(d_out)
    return w, b


def _hidden_forward(params, x):
    """Shared hidden layers forward pass. Returns hidden activations."""
    w1, b1 = params['layer1']
    w2, b2 = params['layer2']
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return h


# ── Gated MDN: gate + 1D Gaussian mixture for log S_conf ──────────────────


def init_gated_mdn_params(key, n_hidden=64, n_components=5):
    """
    Initialise parameters for the gated MDN.

    Output layer has K * 3 neurons (1D target):
      - K logits (mixture weights)
      - K means  (Gaussian mu for log S_conf)
      - K log-sigmas (Gaussian sigma for log S_conf)

    Plus a gate layer (1 logit for P(fully resolved)).
    """
    K = n_components
    d_out = K * 3  # logits + mu + log_sigma

    k1, k2, k3, k4 = jax.random.split(key, 4)
    params = {
        'layer1': init_layer(k1, 4, n_hidden),
        'layer2': init_layer(k2, n_hidden, n_hidden),
        'output': init_layer(k3, n_hidden, d_out),
        'gate': init_layer(k4, n_hidden, 1),
    }
    return params


def gated_mdn_forward(params, x, n_components=5):
    """
    Forward pass of the gated MDN.

    Parameters
    ----------
    params : dict
        Network parameters from init_gated_mdn_params.
    x : array (..., 4)
        Input features (log10(N_tot), alpha, beta, f_center).

    Returns
    -------
    log_pi : array (..., K)
        Log mixture weights.
    mu : array (..., K)
        Gaussian means for log S_conf.
    sigma : array (..., K)
        Gaussian std devs for log S_conf.
    gate_logit : array (...,)
        Raw logit for P(fully resolved).
    """
    K = n_components
    h = _hidden_forward(params, x)

    # MDN head: K logits + K mu + K log_sigma = 3K
    w3, b3 = params['output']
    r = h @ w3 + b3

    logits = r[..., :K]
    mu = r[..., K:2*K]
    log_sig = r[..., 2*K:3*K]

    log_pi = jax.nn.log_softmax(logits, axis=-1)
    sigma = jnp.clip(jnp.exp(log_sig), 1e-6)

    # Gate head
    wg, bg = params['gate']
    gate_logit = (h @ wg + bg)[..., 0]

    return log_pi, mu, sigma, gate_logit


def gated_mdn_log_prob(params, x, y, resolved_flag, n_components=5):
    """
    Per-sample log-probability under the gated MDN.

    For non-resolved rows:
        log(1 - p_gate) + logsumexp_j [log pi_j + log N(log_S; mu_j, sigma_j^2)]
    For resolved rows:
        log(p_gate)

    Parameters
    ----------
    params : dict
    x : array (N, 4)
    y : array (N,)
        Targets: log S_conf values. Values for resolved rows are ignored.
    resolved_flag : array (N,)
        1.0 if fully resolved, 0.0 otherwise.
    n_components : int

    Returns
    -------
    log_p : array (N,)
    """
    log_pi, mu, sigma, gate_logit = gated_mdn_forward(
        params, x, n_components
    )

    # Gate log-probabilities (numerically stable)
    log_p_gate = -jnp.logaddexp(0.0, -gate_logit)      # log sigmoid(logit)
    log_p_not_gate = -jnp.logaddexp(0.0, gate_logit)    # log sigmoid(-logit)

    # 1D Gaussian log-prob for log S_conf: (N, K)
    y_exp = y[:, None]  # (N, 1)
    log_gauss = -0.5 * (
        ((y_exp - mu) / sigma) ** 2
        + 2 * jnp.log(sigma)
        + jnp.log(2 * jnp.pi)
    )  # (N, K)

    # Mixture log-prob
    mdn_lp = jax.nn.logsumexp(log_pi + log_gauss, axis=-1)  # (N,)

    # Combine gate
    log_p = jnp.where(
        resolved_flag > 0.5,
        log_p_gate,
        log_p_not_gate + mdn_lp,
    )
    return log_p


def gated_mdn_loss(params, x, y, resolved_flag, n_components=5):
    """Negative mean log-likelihood for the gated MDN."""
    return -jnp.mean(gated_mdn_log_prob(params, x, y, resolved_flag, n_components))


def gated_mdn_predict_mean(params, x, n_components=5):
    """
    Predictive mean of log S_conf (conditional on not being fully resolved).

    Returns
    -------
    mean : array (...,)
        E[log S_conf] = sum_j pi_j mu_j
    """
    log_pi, mu, sigma, _ = gated_mdn_forward(params, x, n_components)
    pi = jnp.exp(log_pi)  # (..., K)
    return jnp.sum(pi * mu, axis=-1)  # (...,)


def gated_mdn_predict_variance(params, x, n_components=5):
    """
    Predictive variance of log S_conf (conditional on not being fully resolved).

    Uses law of total variance:
      Var = sum_j pi_j (sigma_j^2 + mu_j^2) - (sum_j pi_j mu_j)^2

    Returns
    -------
    var : array (...,)
    """
    log_pi, mu, sigma, _ = gated_mdn_forward(params, x, n_components)
    pi = jnp.exp(log_pi)

    mean = jnp.sum(pi * mu, axis=-1)
    var = (
        jnp.sum(pi * (sigma ** 2 + mu ** 2), axis=-1)
        - mean ** 2
    )
    return var


def gated_mdn_predict_resolved_prob(params, x, n_components=5):
    """Return P(fully resolved | x)."""
    _, _, _, gate_logit = gated_mdn_forward(params, x, n_components)
    return jax.nn.sigmoid(gate_logit)


# ── Analytic Poisson rate for N_res ────────────────────────────────────────


def compute_lambda_res(log_S_conf, N_tot, alpha, beta, f_k,
                       S_instr_k, T_obs, rho_th, A_min, A_max,
                       delta_f_band, f_min, f_max):
    """
    Compute the Poisson rate lambda_res for resolved sources in band k.

    Given S_conf, the amplitude threshold A_th is deterministic, the resolve
    probability p_res is a closed-form integral of the power law, and
    lambda_res = N_tot * p_f * delta_f_band * p_res.

    All inputs are scalars (or broadcastable).

    Parameters
    ----------
    log_S_conf : float
        Natural log of confusion noise PSD.
    N_tot : float
        Total number of sources.
    alpha : float
        Amplitude power-law index.
    beta : float
        Frequency taper parameter.
    f_k : float
        Band center frequency.
    S_instr_k : float
        Instrumental noise PSD in this band.
    T_obs : float
        Observation time [s].
    rho_th : float
        SNR detection threshold.
    A_min, A_max : float
        Amplitude bounds.
    delta_f_band : float
        Band width [Hz].
    f_min, f_max : float
        Full frequency range [Hz].

    Returns
    -------
    lambda_res : float
        Poisson rate for resolved sources.
    """
    S_conf = jnp.exp(log_S_conf)
    S_tot = S_instr_k + S_conf
    A_th = (rho_th / 2.0) * jnp.sqrt(S_tot / T_obs)

    # Resolve probability: fraction of p(A) above threshold
    A_th_clipped = jnp.clip(A_th, A_min, A_max)
    p_res = jnp.where(
        A_th < A_min,
        1.0,  # everything resolved
        (A_th_clipped**(1 - alpha) - A_max**(1 - alpha)) /
        (A_min**(1 - alpha) - A_max**(1 - alpha))
    )

    # Expected number of sources in this band
    Z_freq = (f_max - f_min) - beta * (f_max**2 - f_min**2) / 2
    p_f = (1 - beta * f_k) / Z_freq
    mu_k = N_tot * p_f * delta_f_band

    return mu_k * p_res

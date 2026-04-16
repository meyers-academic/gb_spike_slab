"""
MDNPrior: a NumPyro distribution wrapping the pre-trained gated MDN.

Provides p(log S_conf | Λ, f_k) as a 1D Gaussian mixture, suitable for
use as a prior in the hierarchical model.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from tophat_populations.noise_emulator.network import gated_mdn_forward


class MDNPrior(dist.Distribution):
    """
    1D Gaussian mixture prior on log S_conf, conditioned on (Λ, f_k).
    Wraps a pre-trained gated MDN.

    The gate probability is not used directly in log_prob (we sample
    log S_conf only for the continuous branch). The gate is handled
    externally by the hierarchical model's structure.

    Parameters
    ----------
    mdn_params : dict
        Frozen MDN parameters (not sampled).
    Lambda : array (3,)
        Population hyperparameters (log10(N_tot), alpha, beta).
    f_k : float
        Band center frequency.
    X_norm_stats : tuple (X_mean, X_std)
        Normalisation statistics from training.
    n_components : int
        Number of mixture components.
    """
    support = constraints.real
    arg_constraints = {}

    def __init__(self, mdn_params, Lambda, f_k, X_norm_stats,
                 n_components=5, validate_args=None):
        self.mdn_params = mdn_params
        self.n_components = n_components

        # Build and normalise input
        x_raw = jnp.concatenate([Lambda, jnp.atleast_1d(f_k)])
        X_mean, X_std = X_norm_stats
        self.x = (x_raw - X_mean) / X_std

        # Forward pass (once per construction)
        log_pi, mu, sigma, gate_logit = gated_mdn_forward(
            mdn_params, self.x[None, :], n_components)
        self.log_pi = log_pi[0]     # (K,)
        self.mu = mu[0]             # (K,)
        self.sigma = sigma[0]       # (K,)
        self.gate_logit = gate_logit[0]  # scalar

        batch_shape = ()
        event_shape = ()
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        """
        Log-probability of log S_conf under the MDN's continuous branch.

        Uses: log(1 - p_gate) + logsumexp_j [log pi_j + log N(value; mu_j, sigma_j)]
        """
        # Gate: log(1 - p_gate)
        log_1mg = -jnp.logaddexp(0.0, self.gate_logit)

        # 1D Gaussian mixture log-prob
        log_gauss = -0.5 * (
            ((value - self.mu) / self.sigma) ** 2
            + 2 * jnp.log(self.sigma)
            + jnp.log(2 * jnp.pi)
        )  # (K,)
        log_mdn = jax.nn.logsumexp(self.log_pi + log_gauss)

        return log_1mg + log_mdn

    def sample(self, key, sample_shape=()):
        """Sample from the 1D Gaussian mixture (continuous branch)."""
        k1, k2 = jax.random.split(key)
        # Choose a mixture component
        j = jax.random.categorical(k1, self.log_pi)
        # Draw from that Gaussian
        return self.mu[j] + self.sigma[j] * jax.random.normal(k2, sample_shape)

    @property
    def resolved_prob(self):
        """P(fully resolved | x)."""
        return jax.nn.sigmoid(self.gate_logit)

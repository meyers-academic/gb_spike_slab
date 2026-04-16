"""
Unit tests for the MDN confusion noise emulator (1D gated architecture).

Run with: pytest tests/test_mdn.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tophat_populations.noise_emulator.network import (
    init_gated_mdn_params,
    gated_mdn_forward,
    gated_mdn_log_prob,
    gated_mdn_loss,
    gated_mdn_predict_mean,
    gated_mdn_predict_variance,
    gated_mdn_predict_resolved_prob,
    compute_lambda_res,
)
from tophat_populations.noise_emulator.iterative_sub import (
    sample_amplitudes,
    sample_frequencies,
    assign_bands,
    iterative_subtraction,
)
from tophat_populations.noise_emulator.training import normalise_inputs, train_gated_mdn
from tophat_populations.noise_emulator.data_gen import (
    generate_training_data,
    train_val_split,
)


# ── Gated MDN network tests ──────────────────────────────────────────────


class TestGatedMDNInit:
    def test_param_shapes(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=3)

        w1, b1 = params["layer1"]
        assert w1.shape == (4, 32)
        assert b1.shape == (32,)

        w2, b2 = params["layer2"]
        assert w2.shape == (32, 32)
        assert b2.shape == (32,)

        # output: K * 3 = 3 * 3 = 9 (1D target: logits + mu + log_sigma)
        w3, b3 = params["output"]
        assert w3.shape == (32, 9)
        assert b3.shape == (9,)

        # gate: 1 output
        wg, bg = params["gate"]
        assert wg.shape == (32, 1)
        assert bg.shape == (1,)

    def test_default_shapes(self):
        key = jax.random.PRNGKey(1)
        params = init_gated_mdn_params(key)
        w3, b3 = params["output"]
        # K=5, 1D target => 5*3 = 15
        assert w3.shape == (64, 15)


class TestGatedMDNForward:
    @pytest.fixture
    def params_and_x(self):
        key = jax.random.PRNGKey(42)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        return params, x

    def test_output_shapes(self, params_and_x):
        params, x = params_and_x
        log_pi, mu, sigma, gate_logit = gated_mdn_forward(params, x, n_components=5)
        assert log_pi.shape == (10, 5)
        assert mu.shape == (10, 5)       # 1D: no extra dimension
        assert sigma.shape == (10, 5)
        assert gate_logit.shape == (10,)

    def test_mixture_weights_sum_to_one(self, params_and_x):
        params, x = params_and_x
        log_pi, _, _, _ = gated_mdn_forward(params, x, n_components=5)
        pi = jnp.exp(log_pi)
        np.testing.assert_allclose(pi.sum(axis=-1), 1.0, atol=1e-5)

    def test_sigma_positive(self, params_and_x):
        params, x = params_and_x
        _, _, sigma, _ = gated_mdn_forward(params, x, n_components=5)
        assert jnp.all(sigma > 0)

    def test_log_pi_finite(self, params_and_x):
        params, x = params_and_x
        log_pi, _, _, _ = gated_mdn_forward(params, x, n_components=5)
        assert jnp.all(jnp.isfinite(log_pi))

    def test_single_sample(self):
        key = jax.random.PRNGKey(7)
        params = init_gated_mdn_params(key, n_hidden=16, n_components=3)
        x = jnp.ones((1, 4))
        log_pi, mu, sigma, gate_logit = gated_mdn_forward(params, x, n_components=3)
        assert log_pi.shape == (1, 3)
        assert mu.shape == (1, 3)


class TestGatedMDNLogProb:
    def test_log_prob_finite(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (20, 4))
        y = jax.random.normal(jax.random.PRNGKey(2), (20,))  # 1D target
        rf = jnp.zeros(20)  # all non-resolved

        lp = gated_mdn_log_prob(params, x, y, rf, n_components=5)
        assert lp.shape == (20,)
        assert jnp.all(jnp.isfinite(lp))

    def test_resolved_branch(self):
        """Resolved rows should only use gate probability."""
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (5, 4))
        y = jnp.zeros(5)  # placeholder
        rf = jnp.ones(5)  # all resolved

        lp = gated_mdn_log_prob(params, x, y, rf, n_components=5)
        assert jnp.all(jnp.isfinite(lp))
        # Should be log(sigmoid(gate_logit)) which is <= 0
        assert jnp.all(lp <= 0)

    def test_mixed_resolved_non_resolved(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        y = jax.random.normal(jax.random.PRNGKey(2), (10,))
        rf = jnp.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0], dtype=float)

        lp = gated_mdn_log_prob(params, x, y, rf, n_components=5)
        assert lp.shape == (10,)
        assert jnp.all(jnp.isfinite(lp))


class TestGatedMDNLoss:
    def test_loss_finite(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key)
        x = jax.random.normal(jax.random.PRNGKey(1), (20, 4))
        y = jax.random.normal(jax.random.PRNGKey(2), (20,))
        rf = jnp.zeros(20)
        loss = gated_mdn_loss(params, x, y, rf)
        assert jnp.isfinite(loss)

    def test_loss_differentiable(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=16, n_components=3)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        y = jax.random.normal(jax.random.PRNGKey(2), (10,))
        rf = jnp.zeros(10)

        grads = jax.grad(gated_mdn_loss)(params, x, y, rf, 3)
        for layer_name, (gw, gb) in grads.items():
            assert jnp.all(jnp.isfinite(gw)), f"NaN grad in {layer_name} weights"
            assert jnp.all(jnp.isfinite(gb)), f"NaN grad in {layer_name} biases"


class TestGatedMDNPredictions:
    def test_predict_mean_shape(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        mean = gated_mdn_predict_mean(params, x)
        assert mean.shape == (10,)  # 1D: scalar per sample
        assert jnp.all(jnp.isfinite(mean))

    def test_predict_variance_positive(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        var = gated_mdn_predict_variance(params, x)
        assert var.shape == (10,)  # 1D: scalar per sample
        assert jnp.all(var >= 0)

    def test_resolved_prob_range(self):
        key = jax.random.PRNGKey(0)
        params = init_gated_mdn_params(key, n_hidden=32, n_components=5)
        x = jax.random.normal(jax.random.PRNGKey(1), (10, 4))
        p = gated_mdn_predict_resolved_prob(params, x)
        assert p.shape == (10,)
        assert jnp.all(p >= 0)
        assert jnp.all(p <= 1)


# ── compute_lambda_res tests ─────────────────────────────────────────────


class TestComputeLambdaRes:
    def test_non_negative(self):
        lam = compute_lambda_res(
            log_S_conf=0.0, N_tot=10000.0, alpha=3.0, beta=0.0,
            f_k=5e-4, S_instr_k=1.0, T_obs=3.15e7, rho_th=5.0,
            A_min=1e-3, A_max=10.0, delta_f_band=9e-5,
            f_min=1e-4, f_max=1e-3,
        )
        assert lam >= 0

    def test_more_sources_more_resolved(self):
        """Higher N_tot should give higher lambda_res."""
        kwargs = dict(
            log_S_conf=0.0, alpha=3.0, beta=0.0,
            f_k=5e-4, S_instr_k=1.0, T_obs=3.15e7, rho_th=5.0,
            A_min=1e-3, A_max=10.0, delta_f_band=9e-5,
            f_min=1e-4, f_max=1e-3,
        )
        lam_low = compute_lambda_res(N_tot=1000.0, **kwargs)
        lam_high = compute_lambda_res(N_tot=30000.0, **kwargs)
        assert lam_high > lam_low

    def test_vmappable(self):
        """Should work with vmap."""
        log_s = jnp.array([0.0, 1.0, 2.0])
        lam = jax.vmap(
            lambda ls: compute_lambda_res(
                ls, 10000.0, 3.0, 0.0, 5e-4,
                1.0, 3.15e7, 5.0, 1e-3, 10.0,
                9e-5, 1e-4, 1e-3,
            )
        )(log_s)
        assert lam.shape == (3,)
        assert jnp.all(jnp.isfinite(lam))


# ── Iterative subtraction tests ───────────────────────────────────────────


class TestSampleAmplitudes:
    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        amps = sample_amplitudes(rng, 10000, alpha=3.0, A_min=0.1, A_max=10.0)
        assert np.all(amps >= 0.1)
        assert np.all(amps <= 10.0)

    def test_power_law_shape(self):
        rng = np.random.default_rng(42)
        amps = sample_amplitudes(rng, 100000, alpha=3.0, A_min=0.1, A_max=10.0)
        median = np.median(amps)
        assert median < 1.0

    def test_alpha_one(self):
        rng = np.random.default_rng(42)
        amps = sample_amplitudes(rng, 10000, alpha=1.0, A_min=0.1, A_max=10.0)
        assert np.all(amps >= 0.1)
        assert np.all(amps <= 10.0)
        log_median = np.median(np.log(amps))
        log_midpoint = 0.5 * (np.log(0.1) + np.log(10.0))
        np.testing.assert_allclose(log_median, log_midpoint, atol=0.1)


class TestSampleFrequencies:
    def test_within_bounds(self):
        rng = np.random.default_rng(42)
        freqs = sample_frequencies(rng, 10000, beta=0.0, f_min=1e-4, f_max=1e-3)
        assert np.all(freqs >= 1e-4)
        assert np.all(freqs <= 1e-3)

    def test_uniform_when_beta_zero(self):
        rng = np.random.default_rng(42)
        freqs = sample_frequencies(rng, 100000, beta=0.0, f_min=0.0, f_max=1.0)
        mean = np.mean(freqs)
        np.testing.assert_allclose(mean, 0.5, atol=0.01)

    def test_taper_with_positive_beta(self):
        rng = np.random.default_rng(42)
        freqs = sample_frequencies(rng, 100000, beta=500.0, f_min=1e-4, f_max=1e-3)
        mid = 0.5 * (1e-4 + 1e-3)
        frac_below = np.mean(freqs < mid)
        assert frac_below > 0.5

    def test_negative_pdf_raises(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="goes negative"):
            sample_frequencies(rng, 100, beta=2000.0, f_min=1e-4, f_max=1e-3)


class TestAssignBands:
    def test_basic(self):
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        freqs = np.array([0.5, 1.5, 2.5, -0.1, 3.1])
        idx = assign_bands(freqs, edges)
        np.testing.assert_array_equal(idx, [0, 1, 2, -1, -1])


class TestIterativeSubtraction:
    def test_converges(self):
        band_edges = np.linspace(1e-4, 1e-3, 11)
        S_conf, N_res, converged, n_iter = iterative_subtraction(
            N_tot=5000, alpha=3.0, beta=0.0,
            f_min=1e-4, f_max=1e-3, band_edges=band_edges,
            S_instr=1e-40, T_obs=3.15e7,
            A_min=1e-22, A_max=1e-19, rho_th=5.0,
            rng=np.random.default_rng(42),
        )
        assert converged
        assert n_iter <= 20
        assert S_conf.shape == (10,)
        assert N_res.shape == (10,)
        assert np.all(S_conf >= 0)
        assert np.all(N_res >= 0)

    def test_s_conf_positive(self):
        band_edges = np.linspace(1e-4, 1e-3, 6)
        S_conf, N_res, _, _ = iterative_subtraction(
            N_tot=10000, alpha=3.5, beta=0.0,
            f_min=1e-4, f_max=1e-3, band_edges=band_edges,
            S_instr=1e-40, T_obs=3.15e7,
            A_min=1e-22, A_max=1e-19, rho_th=5.0,
            rng=np.random.default_rng(123),
        )
        assert np.sum(S_conf > 0) >= 3

    def test_more_sources_more_confusion(self):
        band_edges = np.linspace(1e-4, 1e-3, 6)
        kwargs = dict(
            alpha=3.0, beta=0.0, f_min=1e-4, f_max=1e-3,
            band_edges=band_edges, S_instr=1e-40, T_obs=3.15e7,
            A_min=1e-22, A_max=1e-19, rho_th=5.0,
        )
        S1, _, _, _ = iterative_subtraction(N_tot=3000, rng=np.random.default_rng(0), **kwargs)
        S2, _, _, _ = iterative_subtraction(N_tot=30000, rng=np.random.default_rng(0), **kwargs)
        assert np.sum(S2) > np.sum(S1)


# ── Training / data gen tests ─────────────────────────────────────────────


class TestNormalise:
    def test_zero_mean_unit_var(self):
        X = np.random.randn(100, 4) * 5 + 3
        X_norm, (mean, std) = normalise_inputs(jnp.array(X))
        np.testing.assert_allclose(jnp.mean(X_norm, axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(jnp.std(X_norm, axis=0), 1.0, atol=1e-5)


class TestTraining:
    def test_loss_decreases(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(jax.random.PRNGKey(1), (50, 4))
        Y = jax.random.normal(jax.random.PRNGKey(2), (50,))  # 1D target
        rf = jnp.zeros(50)

        params, history = train_gated_mdn(
            key, X, Y, rf,
            n_components=3, n_hidden=16, n_steps=200, lr=1e-3, print_every=0
        )
        losses = history["train_loss"]
        assert np.mean(losses[-10:]) < np.mean(losses[:10])


class TestDataGen:
    def test_shapes(self):
        band_edges = np.linspace(1e-4, 1e-3, 4)  # 3 bands
        X, Y, N_res, rf, ids = generate_training_data(
            n_realisations=5,
            band_edges=band_edges,
            S_instr=1e-40,
            T_obs=3.15e7,
            A_min=1e-22,
            A_max=1e-19,
            f_min=1e-4,
            f_max=1e-3,
            lambda_bounds={"N_tot": (1000, 10000), "alpha": (2.0, 4.0), "beta": (0.0, 0.3)},
            seed=42,
            show_progress=False,
        )
        assert X.shape == (15, 4)   # 5 realisations * 3 bands
        assert Y.shape == (15,)     # 1D target
        assert N_res.shape == (15,)
        assert rf.shape == (15,)
        assert ids.shape == (15,)
        assert set(np.unique(rf)).issubset({0.0, 1.0})

    def test_train_val_split_no_leakage(self):
        ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        X = np.random.randn(12, 4)
        Y = np.random.randn(12)    # 1D
        N_res = np.random.randint(0, 10, 12).astype(float)
        rf = np.zeros(12)

        (X_tr, Y_tr, Nres_tr, rf_tr,
         X_val, Y_val, Nres_val, rf_val) = train_val_split(
            X, Y, N_res, rf, ids, val_fraction=0.5, seed=0
        )
        assert len(X_tr) + len(X_val) == 12
        tr_ids = set()
        val_ids = set()
        for i in range(12):
            row = X[i]
            if any(np.allclose(row, x) for x in X_tr):
                tr_ids.add(ids[i])
            if any(np.allclose(row, x) for x in X_val):
                val_ids.add(ids[i])
        assert len(tr_ids & val_ids) == 0

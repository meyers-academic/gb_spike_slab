"""
Hyperparameter update step for the Gibbs sampler.

Model-agnostic: takes any NumPyro model function and its kwargs,
extracts the log-density, and runs BlackJAX NUTS.

Key optimisations:
  - Window adaptation (warmup) runs only once.
  - The NUTS step is JIT-compiled with changing data (N_res_obs) passed
    as a dynamic JAX argument, avoiding costly retracing (~2ms/step vs
    ~1200ms without JIT).
  - Static model kwargs (MDN params, physical constants) are captured
    by the JIT closure once at initialisation.
"""

import jax
import jax.numpy as jnp
import blackjax
import numpyro
from numpyro.infer.util import initialize_model, potential_energy


class HyperUpdater:
    """
    Persistent NUTS sampler for the hyperparameter block.

    Traces the NumPyro model once at construction, runs window adaptation
    once via ``warmup()``, then provides cheap JIT-compiled ``step()`` calls.

    Parameters
    ----------
    rng_key : jax PRNGKey
        Used for ``initialize_model``.
    model_fn : callable
        NumPyro model function.
    model_kwargs : dict
        Full keyword arguments. Split internally into static (captured
        by JIT closure) and dynamic (passed as args each step).
    init_values : dict or None
        Constrained-space initial values for warm-starting.
    dynamic_kwarg_keys : tuple of str
        Keys in ``model_kwargs`` whose values change between iterations.
        These are passed as explicit JAX arguments to the JIT-compiled
        NUTS step. All other keys are treated as static (frozen at init).
        Defaults to ``("N_res_obs",)``.
    """

    def __init__(self, rng_key, model_fn, model_kwargs, init_values=None,
                 dynamic_kwarg_keys=("N_res_obs",)):
        if init_values is not None:
            init_strategy = numpyro.infer.init_to_value(values=init_values)
        else:
            init_strategy = numpyro.infer.init_to_median()

        # Trace the model once to get initial position + transforms
        info = initialize_model(
            rng_key, model_fn,
            model_kwargs=model_kwargs,
            init_strategy=init_strategy,
        )
        self._init_position = dict(info.param_info.z)
        self._postprocess_fn = info.postprocess_fn

        # Split kwargs into static (JIT-captured) and dynamic (passed each step)
        self._dynamic_keys = tuple(dynamic_kwarg_keys)
        self._static_kwargs = {
            k: v for k, v in model_kwargs.items()
            if k not in self._dynamic_keys
        }
        self._model_fn = model_fn

        # Build JIT-compiled potential that takes dynamic kwargs as args
        static_kw = self._static_kwargs
        mfn = model_fn
        dkeys = self._dynamic_keys

        def _potential(params, *dynamic_args):
            kwargs = dict(static_kw)
            for key, val in zip(dkeys, dynamic_args):
                kwargs[key] = val
            return potential_energy(mfn, (), kwargs, params)

        self._potential = _potential

        # Build JIT-compiled logdensity
        def _logdensity(params, *dynamic_args):
            return -_potential(params, *dynamic_args)

        self._logdensity = _logdensity

        # These are set by warmup()
        self._kernel_params = None
        self._state = None
        self._jitted_nuts_step = None

    def warmup(self, rng_key, model_kwargs, num_warmup=200,
               max_num_doublings=8):
        """
        Run BlackJAX window adaptation.

        Call this once (typically on the first Gibbs iteration).

        Parameters
        ----------
        max_num_doublings : int
            Maximum NUTS tree depth (2**d - 1 leapfrog steps max).
            Default 5 → at most 31 steps. BlackJAX default is 10 (1023 steps).
        """
        # Extract dynamic args for warmup
        dynamic_args = tuple(model_kwargs[k] for k in self._dynamic_keys)

        # Warmup needs a logdensity_fn(params) -> scalar
        logdensity_warmup = lambda params: self._logdensity(params, *dynamic_args)

        warmup_algo = blackjax.window_adaptation(
            blackjax.nuts,
            logdensity_warmup,
            progress_bar=False,
            max_num_doublings=max_num_doublings,
        )
        (self._state, self._kernel_params), _ = warmup_algo.run(
            rng_key, self._init_position, num_steps=num_warmup,
        )

        # Build JIT-compiled NUTS step.
        # nuts.init is included INSIDE the jitted function so the gradient
        # recomputation (needed when N_res_obs changes) is fused into a single
        # XLA execution — no Python-level GPU sync between init and step.
        nuts_kernel = blackjax.nuts.build_kernel()
        kernel_params = self._kernel_params
        logdensity = self._logdensity

        @jax.jit
        def _jitted_step(rng_key, position, *dynamic_args):
            ld = lambda params: logdensity(params, *dynamic_args)
            state = blackjax.nuts.init(position, ld)
            return nuts_kernel(rng_key, state, ld, **kernel_params)

        self._jitted_nuts_step = _jitted_step

        # JIT-compile the postprocess step too
        self._jitted_postprocess = jax.jit(self._postprocess_fn)

    @property
    def is_warmed_up(self):
        return self._kernel_params is not None

    def step(self, rng_key, model_kwargs, num_samples=1):
        """
        Take NUTS step(s) with pre-adapted kernel parameters (JIT-compiled).

        Parameters
        ----------
        rng_key : jax PRNGKey
        model_kwargs : dict
            Current model kwargs (must include dynamic keys like N_res_obs).
        num_samples : int
            Number of NUTS steps (typically 1).

        Returns
        -------
        samples : dict
            Constrained-space posterior samples.
        """
        dynamic_args = tuple(model_kwargs[k] for k in self._dynamic_keys)

        keys = jax.random.split(rng_key, num_samples)
        positions = []
        # Pass position only — nuts.init is fused inside _jitted_step
        position = self._state.position
        for i in range(num_samples):
            self._state, _info = self._jitted_nuts_step(
                keys[i], position, *dynamic_args,
            )
            position = self._state.position
            positions.append(position)

        # Convert unconstrained -> constrained samples
        samples = jax.vmap(self._jitted_postprocess)(
            jax.tree.map(lambda *xs: jnp.stack(xs), *positions)
        )

        return samples


# ── Functional API ───────────────────────────────────────────────────────

def warmup_and_sample(
    rng_key,
    model_fn,
    model_kwargs,
    init_values=None,
    num_warmup=200,
    num_samples=1,
    dynamic_kwarg_keys=("N_res_obs",),
):
    """
    Run full warmup + sampling from scratch.

    Returns a ``HyperUpdater`` instance for reuse, plus the first
    batch of samples.

    Returns
    -------
    samples : dict
        Constrained-space posterior samples.
    updater : HyperUpdater
        Reusable updater for subsequent iterations.
    """
    k1, k2, k3 = jax.random.split(rng_key, 3)

    updater = HyperUpdater(
        k1, model_fn, model_kwargs,
        init_values=init_values,
        dynamic_kwarg_keys=dynamic_kwarg_keys,
    )
    updater.warmup(k2, model_kwargs=model_kwargs, num_warmup=num_warmup)
    samples = updater.step(k3, model_kwargs, num_samples=num_samples)

    return samples, updater


def hyper_update_step(rng_key, updater, model_kwargs, num_samples=1):
    """
    Take NUTS step(s) using an existing ``HyperUpdater``.

    Returns
    -------
    samples : dict
    updater : HyperUpdater (same object, mutated)
    """
    samples = updater.step(rng_key, model_kwargs, num_samples=num_samples)
    return samples, updater

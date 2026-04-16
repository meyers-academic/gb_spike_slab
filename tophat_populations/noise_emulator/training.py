"""
Training loop for the MDN confusion noise emulator.

Uses Adam optimiser via optax with optional mini-batching and
train/validation split.
"""

import jax
import jax.numpy as jnp
import optax

from .network import init_gated_mdn_params, gated_mdn_loss


def normalise_inputs(X_train):
    """
    Compute normalisation statistics and return normalised data.

    Returns
    -------
    X_norm : array
        Zero-mean, unit-variance features.
    stats : tuple (X_mean, X_std)
        Statistics to apply at inference time.
    """
    X_mean = jnp.mean(X_train, axis=0)
    X_std = jnp.std(X_train, axis=0)
    X_std = jnp.where(X_std < 1e-12, 1.0, X_std)  # guard against constant features
    X_norm = (X_train - X_mean) / X_std
    return X_norm, (X_mean, X_std)


def train_gated_mdn(
    key,
    X_train,
    Y_train,
    resolved_flag_train,
    X_val=None,
    Y_val=None,
    resolved_flag_val=None,
    n_components=5,
    n_hidden=64,
    n_steps=5000,
    lr=1e-3,
    batch_size=None,
    print_every=500,
):
    """
    Train the gated MDN (gate head + 1D Gaussian mixture head).

    Parameters
    ----------
    X_train : array (N, 4)
        Training inputs (already normalised).
    Y_train : array (N,)
        Training targets: log S_conf (1D).
    resolved_flag_train : array (N,)
        1.0 if fully resolved, 0.0 otherwise.
    X_val, Y_val, resolved_flag_val : optional arrays
        Validation data.
    n_components, n_hidden, n_steps, lr, batch_size, print_every :
        Hyperparameters.

    Returns
    -------
    params : dict
        Trained gated MDN parameters.
    history : dict
        Training and validation loss histories.
    """
    k1, k2 = jax.random.split(key)
    params = init_gated_mdn_params(k1, n_hidden=n_hidden, n_components=n_components)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    N = X_train.shape[0]
    if batch_size is None:
        batch_size = N

    @jax.jit
    def step(params, opt_state, x_batch, y_batch, rf_batch):
        loss, grads = jax.value_and_grad(gated_mdn_loss)(
            params, x_batch, y_batch, rf_batch, n_components
        )
        updates, new_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    train_losses = []
    val_losses = []

    for i in range(n_steps):
        if batch_size < N:
            k2, subkey = jax.random.split(k2)
            idx = jax.random.choice(subkey, N, shape=(batch_size,), replace=False)
            x_batch = X_train[idx]
            y_batch = Y_train[idx]
            rf_batch = resolved_flag_train[idx]
        else:
            x_batch = X_train
            y_batch = Y_train
            rf_batch = resolved_flag_train

        params, opt_state, loss = step(params, opt_state, x_batch, y_batch, rf_batch)
        train_losses.append(float(loss))

        if X_val is not None and (i % print_every == 0 or i == n_steps - 1):
            val_loss = float(gated_mdn_loss(
                params, X_val, Y_val, resolved_flag_val, n_components
            ))
            val_losses.append(val_loss)

        if print_every > 0 and (i % print_every == 0 or i == n_steps - 1):
            msg = f"Step {i:5d}: train loss = {loss:.4f}"
            if X_val is not None:
                msg += f"  val loss = {val_losses[-1]:.4f}"
            print(msg)

    history = {"train_loss": train_losses, "val_loss": val_losses}
    return params, history

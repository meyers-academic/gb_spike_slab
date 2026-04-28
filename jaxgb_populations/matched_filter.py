import jax
import jax.numpy as jnp
import numpy as np


def filter_coefficients(templates, data, psd, frequencies):
    """Compute matched-filter inner products for a set of frequency-domain waveforms.

    Uses the standard gravitational-wave inner product:

    .. math::

        \\langle a \\mid b \\rangle = 4\\,\\Delta f\\,
        \\mathrm{Re}\\!\\left[\\sum_k \\frac{a^*[k]\\,b[k]}{S_n[k]}\\right]

    assuming a one-sided PSD :math:`S_n` defined on uniformly-spaced positive
    frequencies.

    **Connection to the GW likelihood.**
    A common signal model is a superposition of templates with amplitudes
    :math:`\\alpha_k`:

    .. math::

        h = \\sum_k \\alpha_k\\, h_k

    The Gaussian log-likelihood is proportional to the noise-weighted residual:

    .. math::

        \\ln\\mathcal{L} \\propto
        -\\tfrac{1}{2}\\langle d - h \\mid d - h \\rangle

    Expanding, and denoting :math:`\\boldsymbol{\\alpha}` as the vector of
    amplitudes:

    .. math::

        \\langle d - h \\mid d - h \\rangle
        = \\underbrace{\\langle d \\mid d \\rangle}_{\\texttt{dd}}
        - 2\\,\\boldsymbol{\\alpha}^{\\!\\top}
          \\underbrace{\\langle h_k \\mid d \\rangle}_{\\texttt{hd}}
        + \\boldsymbol{\\alpha}^{\\!\\top}
          \\underbrace{\\langle h_k \\mid h_{k'} \\rangle}_{\\texttt{hh}}
          \\boldsymbol{\\alpha}

    This is a quadratic form in :math:`\\boldsymbol{\\alpha}`, so the
    maximum-likelihood amplitudes are

    .. math::

        \\boldsymbol{\\hat{\\alpha}} = \\texttt{hh}^{-1}\\,\\texttt{hd}

    and the matched-filter SNR is obtained by evaluating the likelihood at
    :math:`\\boldsymbol{\\hat{\\alpha}}`.

    Parameters
    ----------
    templates : array_like, shape (n_templates, n_freq)
        Frequency-domain template waveforms :math:`h_k`.
    data : array_like, shape (n_freq,)
        Frequency-domain strain data :math:`d`.
    psd : array_like, shape (n_freq,) or scalar
        One-sided power spectral density :math:`S_n(f)`.  A scalar is
        broadcast to a flat PSD across all frequencies.
    frequencies : array_like, shape (n_freq,)
        Uniformly-spaced positive frequencies (Hz).  Used to determine
        :math:`\\Delta f`.

    Returns
    -------
    dd : float
        :math:`\\langle d \\mid d \\rangle`
    hd : jnp.ndarray, shape (n_templates,)
        :math:`\\langle h_k \\mid d \\rangle` for each template :math:`k`.
    hh : jnp.ndarray, shape (n_templates, n_templates)
        :math:`\\langle h_k \\mid h_{k'} \\rangle` for all pairs of templates.
        Symmetric matrix.
    """
    if np.size(psd) == 1:
        psd = psd * jnp.ones_like(frequencies)

    df = frequencies[1] - frequencies[0]

    def inner_product(a, b):
        return 4 * df * jnp.real(jnp.dot(jnp.conj(a) / psd, b))

    dd = inner_product(data, data)

    hd = jax.vmap(lambda h: inner_product(h, data))(templates)

    inner_row = jax.vmap(inner_product, in_axes=(None, 0))
    hh = jax.vmap(inner_row, in_axes=(0, None))(templates, templates)

    return dd, hd, hh


def log_likelihood_multi_channel(
    data: list[jnp.ndarray],
    templates: list[jnp.ndarray],
    psds: list[jnp.ndarray],
    frequencies: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute log-likelihood for multiple channels.
    
    This is JAX-jittable and works with NumPyro.
    
    Parameters
    ----------
    data : list[jnp.ndarray]
        List of complex frequency-domain data arrays, one per channel
    templates : list[jnp.ndarray]
        List of complex frequency-domain template waveforms, one per channel
    psds : list[jnp.ndarray]
        List of power spectral densities, one per channel
    frequencies : jnp.ndarray
        Uniformly-spaced positive frequencies (Hz)
    z : jnp.ndarray
        Binary indicator variables for each template
    t_obs : float
        Observation time in seconds
        
    Returns
    -------
    log_likelihood : jnp.ndarray
        Combined log-likelihood across all channels
    """
    log_likes = [
        log_likelihood(d, t, p, frequencies, z)
        for d, t, p in zip(data, templates, psds)
    ]
    return jnp.sum(jnp.array(log_likes))


def log_likelihood(
    data: jnp.ndarray,
    templates: jnp.ndarray,
    psd: jnp.ndarray,
    frequencies: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the log-likelihood for a template.
    
    This function is JAX-jittable. Zero PSD values are handled automatically.
    
    Parameters
    ----------
    data : jnp.ndarray
        Complex frequency-domain data
    templates : list[jnp.ndarray]
        List of complex frequency-domain template waveforms
    psd : jnp.ndarray
        One-sided power spectral density
    t_obs : float
        Observation time in seconds
        
    Returns
    -------
    log_likelihood : jnp.ndarray
        Log-likelihood value (scalar)
    """
    # changing the likelihood - just keeping this as a reminder, should delete.
    # residual = data - template
    # residual_power = residual * jnp.conj(residual)
    
    # # Handle zero PSD values: replace zeros with infinity so those terms contribute nothing
    # psd_safe = jnp.where(psd > 0, psd, jnp.inf)
    
    # # Log-likelihood = -0.5 * <r|r> where <r|r> = 4 * sum(|r(f)|^2 / S(f)) / T
    # integrand = residual_power / psd_safe
    # inner_product = 4.0 * jnp.real(jnp.sum(integrand)) / t_obs
    # log_likelihood = -0.5 * inner_product
    
    dd, hd, hh = filter_coefficients(templates, data, psd, frequencies)
    n_bins = frequencies.size

    if jnp.ndim(psd) == 0:
        log_det = n_bins * jnp.log(psd)
    else:
        log_det = jnp.sum(jnp.log(psd))

    log_likelihood = -0.5 * (dd - 2 * jnp.dot(z, hd) + jnp.dot(z, hh @ z)) - log_det

    return log_likelihood

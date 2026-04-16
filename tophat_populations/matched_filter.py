import jax
import jax.numpy as jnp
import numpy as np
from tophat_populations.waveform_simplified import tophat_fd_waveform


def filter_coefficients(waveforms, data, psd, frequencies):
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
    waveforms : array_like, shape (n_waveforms, n_freq)
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
    hd : jnp.ndarray, shape (n_waveforms,)
        :math:`\\langle h_k \\mid d \\rangle` for each template :math:`k`.
    hh : jnp.ndarray, shape (n_waveforms, n_waveforms)
        :math:`\\langle h_k \\mid h_{k'} \\rangle` for all pairs of templates.
        Symmetric matrix.
    """
    if np.size(psd) == 1:
        psd = psd * jnp.ones_like(frequencies)

    df = frequencies[1] - frequencies[0]

    def inner_product(a, b):
        return 4 * df * jnp.real(jnp.dot(jnp.conj(a) / psd, b))

    dd = inner_product(data, data)

    hd = jax.vmap(lambda h: inner_product(h, data))(waveforms)

    inner_row = jax.vmap(inner_product, in_axes=(None, 0))
    hh = jax.vmap(inner_row, in_axes=(0, None))(waveforms, waveforms)

    return dd, hd, hh


def log_likelihood(A, f_center, phi_0, fdot, z, data, psd, frequencies):
    """Gaussian log-likelihood for a spike-and-slab superposition of tophat binaries.

    The signal model is

    .. math::

        h = \\sum_k z_k\\, h_k(A_k, f_{\\mathrm{c},k}, \\phi_{0,k}, \\dot{f}_k)

    where :math:`z_k \\in \\{0, 1\\}` is the indicator that source :math:`k`
    contributes to the data.  Substituting into the Gaussian likelihood and
    expanding with :func:`filter_coefficients` gives

    .. math::

        \\ln\\mathcal{L} = -\\tfrac{1}{2}\\Bigl(
            \\texttt{dd}
            - 2\\,\\mathbf{z}^\\top \\texttt{hd}
            + \\mathbf{z}^\\top \\texttt{hh}\\,\\mathbf{z}
        \\Bigr)

    Parameters
    ----------
    A : array_like, shape (n_source,)
        Strain amplitudes.
    f_center : array_like, shape (n_source,)
        Central frequencies (Hz).
    phi_0 : array_like, shape (n_source,)
        Phases at the central frequency (rad).
    fdot : array_like, shape (n_source,)
        Frequency derivatives (Hz/s).
    z : array_like, shape (n_source,)
        Indicator variables (0 or 1) selecting active sources.
    data : array_like, shape (n_freq,)
        Frequency-domain strain data.
    psd : array_like, shape (n_freq,) or scalar
        One-sided power spectral density.
    frequencies : array_like, shape (n_freq,)
        Uniformly-spaced positive frequencies (Hz).

    Returns
    -------
    float
        :math:`\\ln\\mathcal{L}`
    """
    waveforms = jax.vmap(
        lambda a, fc, phi, fd: tophat_fd_waveform(a, fc, phi, fd, frequencies)
    )(A, f_center, phi_0, fdot)

    dd, hd, hh = filter_coefficients(waveforms, data, psd, frequencies)
    n_bins = frequencies.size

    if jnp.ndim(psd) == 0:
        log_det = n_bins * jnp.log(psd)
    else:
        log_det = jnp.sum(jnp.log(psd))

    return -0.5 * (dd - 2 * jnp.dot(z, hd) + jnp.dot(z, hh @ z)) - log_det

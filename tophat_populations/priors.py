import numpy as np
import scipy.stats as ss
import jax.numpy as jnp

def draw_from_fdot_gaussian(mean, scale, N):
    """
    Docstring for draw_from_fdot_gaussian
    
    :param mean: mean of dot distribution
    :param scale: scale of fdot distribution
    :param N: number of dots to draw
    """
    return np.random.randn(N) * scale + mean

def sample_power_law_bounded(alpha, A_l, A_h, N):
    """Power law with both lower and upper cutoffs."""
    u = np.random.rand(N)
    # CDF normalization factor for truncation
    x_l, x_h = 1.0, (A_h / A_l) ** (1 - alpha)
    # Inverse CDF with truncation
    return A_l * (x_l - u * (x_l - x_h)) ** (1.0 / (1.0 - alpha))

def power_law_pdf_bounded(A, alpha, A_l, A_h):
    """
    Normalized PDF for power law with lower and upper cutoffs.
    
    Parameters
    ----------
    A : array
        Amplitude values
    alpha : float
        Power law index (must be > 1)
    A_l : float
        Lower amplitude cutoff
    A_h : float
        Upper amplitude cutoff
    
    Returns
    -------
    pdf : array
        Probability density (zero outside [A_l, A_h])
    """
    # Normalization: 1 / ∫_{A_l}^{A_h} A^{-alpha} dA
    norm = (alpha - 1.0) / (A_l ** (1.0 - alpha) - A_h ** (1.0 - alpha))
    
    return jnp.where(
        (A >= A_l) & (A <= A_h),
        norm * A ** (-alpha),
        0.0
    )
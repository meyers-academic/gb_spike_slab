import numpy as np
from jaxopt import Bisection, ScipyRootFinding
import jax.numpy as jnp
import jax

def _bisect_F(beta, gamma, alpha, lower=1.0, upper=10000.0, n_iter=60):
    """Pure-JAX bisection for x^2 - beta - gamma*(x^(3-alpha) - 1) = 0."""
    def F(x):
        return x**2 - beta - gamma * (x**(3 - alpha) - 1)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        lo = jnp.where(F(lo) * F(mid) <= 0, lo, mid)
        hi = jnp.where(F(lo) * F(mid) <= 0, mid, hi)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lower, upper))
    return 0.5 * (lo + hi)

def _newton_F(beta, gamma, alpha, x0=10.0, n_iter=12):
    def F(x):
        return x**2 - beta - gamma * (x**(3 - alpha) - 1)
    def dF(x):
        return 2*x - gamma * (3 - alpha) * x**(2 - alpha)
    def body(_, x):
        return x - F(x) / dF(x)
    return jax.lax.fori_loop(0, n_iter, body, x0)

def get_threshold_and_conf_noise_for_powerlaw(Tobs, delta_f_band, Sw, Ntot, alpha, rho_th, Amin,
                                              w_fdot, phase_variance=True, N_variance=False):
    Nbins = delta_f_band * Tobs # number of bins
    sigma_0_sq = Sw / Tobs # time domain wn variance
    mu = Ntot / Nbins # expected sources per bin

    beta = rho_th**2 * sigma_0_sq / Amin**2 / 4

    gamma = rho_th**2 * mu * (alpha - 1) / (3 - alpha) / 2.

    def F(x):
        return x**2 - beta - gamma * (x**(3 - alpha) - 1)
    # if False: # alpha == 4:
    #     # Cubic: x^3 - (beta - gamma)*x - gamma = 0
    #     coeffs = jnp.array([1, 0, -(beta - gamma), -gamma])
    #     roots = jnp.roots(coeffs)
    #     # Take the real, positive root
    #     xth = jnp.real(roots[jnp.isreal(roots) & (roots > 0)][0])
    # else:
        # there's almost certainly a faster way to do this...
    xth = _newton_F(beta, gamma, alpha)
        # xth = bisec.run().params
    Ath = xth * Amin
 
    Sconf = (4 * Tobs * Amin**2 / rho_th**2) * (xth**2 - beta)
    avgA2 = (alpha - 1) / (3 - alpha) * Amin**(alpha - 1) * (Ath**(3-alpha) - Amin**(3-alpha))
    avgA4 = (alpha - 1) / (5 - alpha) * Amin**(alpha - 1) * (Ath**(5-alpha) - Amin**(5-alpha))
    # amplitude variance
    # lam = Ntot * w_fdot / delta_f_band   # actual overlap rate per bin
    lam = mu

    var_Sconf  = 4 * Ntot * (avgA4 - avgA2**2) / (delta_f_band**2)#  * w_fdot * Tobs)
    var_Sconf += 4 * Ntot * mu * avgA2**2 / delta_f_band**2

    # var_Sconf  = 4 * Ntot * (avgA4 - avgA2**2) / delta_f_band**2
    # var_Sconf += 4 * Ntot**2 * w_fdot * avgA2**2 / delta_f_band**3

    return Ath, Sconf, var_Sconf
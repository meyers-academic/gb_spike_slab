import numpy as np
from jaxopt import Bisection, ScipyRootFinding

def get_threshold_and_conf_noise_for_powerlaw(Tobs, delta_f_band, Sw, Ntot, alpha, rho_th, Amin,
                                              w_fdot):
    Nbins = delta_f_band * Tobs # number of bins
    sigma_0_sq = Sw / Tobs # time domain wn variance
    mu = Ntot / Nbins # expected sources per bin

    beta = rho_th**2 * sigma_0_sq / Amin**2 / 4

    gamma = rho_th**2 * mu * (alpha - 1) / (3 - alpha) / 2.

    def F(x):
        return x**2 - beta - gamma * (x**(3 - alpha) - 1)
    if alpha == 4:
        # Cubic: x^3 - (beta - gamma)*x - gamma = 0
        coeffs = [1, 0, -(beta - gamma), -gamma]
        roots = np.roots(coeffs)
        # Take the real, positive root
        xth = np.real(roots[np.isreal(roots) & (roots > 0)][0])
    else:
        # there's almost certainly a faster way to do this...
        bisec = Bisection(optimality_fun=F, lower=1, upper=10000)
        xth = bisec.run().params
    Ath = xth * Amin
 
    Sconf = (4 * Tobs * Amin**2 / rho_th**2) * (xth**2 - beta)
    avgA2 = (alpha - 1) / (3 - alpha) * Amin**(alpha - 1) * (Ath**(3-alpha) - Amin**(3-alpha))
    avgA4 = (alpha - 1) / (5 - alpha) * Amin**(alpha - 1) * (Ath**(5-alpha) - Amin**(5-alpha))
    var_Sconf = 4 * Ntot * (avgA4) / delta_f_band**2
    return Ath, Sconf, var_Sconf
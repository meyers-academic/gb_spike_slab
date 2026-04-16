"""Example usage of confusion background simulation classes."""

import numpy as np
import matplotlib.pyplot as plt
from confusion_background_simulator import (
    SimulationParameters, 
    ConfusionBackgroundSimulator,
    compute_complex_correlation
)
from waveform_simplified import tophat_fd_waveform
import priors

from scipy.optimize import curve_fit
# You'll need to import your waveform function and priors module
# from your_module import tophat_fd_waveform, priors


def example_basic_simulation(tophat_fd_waveform, priors):
    """Basic simulation with power-law amplitude distribution."""
    
    params = SimulationParameters(
        Tobs=86400 * 30,  # 30 days
        flow=1e-4,
        fhigh=1e-3,
        N_sources=1000,
        alpha=4,
        A_min=1e-3,
        psd_level=2.111,
        rho_th=5.0,
        w=0.5e-4,
    )
    
    print(f"mu (sources per bin): {params.mu:.2f}")
    print(f"Nbins: {params.Nbins}")
    print(f"fdot: {params.fdot:.2e}")
    
    sim = ConfusionBackgroundSimulator(params, tophat_fd_waveform, priors)
    results = sim.run(n_realizations=5000, seed=42)
    
    results.print_summary()
    
    return results


def example_fixed_amplitude(tophat_fd_waveform, priors):
    """Simulation with fixed amplitude (for testing phase variance)."""
    
    params = SimulationParameters(
        Tobs=86400 * 30,
        flow=1e-4,
        fhigh=1e-3,
        N_sources=100,
        alpha=4,
        A_min=1e-3,
        psd_level=2.111,
        w=1e-4,
        fixed_amplitude=1e-3,  # Fixed amplitude
    )
    
    sim = ConfusionBackgroundSimulator(params, tophat_fd_waveform, priors)
    results = sim.run(n_realizations=5000, seed=42)
    
    print("Fixed amplitude test:")
    results.print_summary()
    
    # Compare to analytic phase variance
    analytic_phase_var = 4 * params.N_sources * params.fixed_amplitude**4 / params.B**2
    print(f"Analytic phase var: {analytic_phase_var:.3e}")
    print(f"Ratio: {analytic_phase_var / results.measured_variance:.2f}")
    
    return results


def example_poisson_N(tophat_fd_waveform, priors):
    """Simulation with Poisson-distributed number of sources."""
    
    params = SimulationParameters(
        Tobs=86400 * 30,
        flow=1e-4,
        fhigh=1e-3,
        N_sources=100,
        alpha=4,
        A_min=1e-3,
        psd_level=2.111,
        w=1e-4,
        poisson_N=True,  # Poisson number of sources
    )
    
    sim = ConfusionBackgroundSimulator(params, tophat_fd_waveform, priors)
    results = sim.run(n_realizations=5000, seed=42)
    
    print("Poisson N test:")
    results.print_summary()
    
    return results


def example_correlation_analysis(results):
    """Analyze correlations between frequency bins."""
    
    signals = np.array([r for r in results.power_spectra])  # Already power, need complex
    # Note: For correlation analysis, you need to store the complex signals
    
    max_lag = int(3 * results.params.w * results.params.Tobs)
    
    # Plot correlation function
    # xi_complex = compute_complex_correlation(signals, max_lag)
    # ... plotting code


def example_parameter_scan(tophat_fd_waveform, priors):
    """Scan over N_sources to check variance scaling."""
    
    # N_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16_000, 32_000]
    N_values = [200, 500, 1000, 2000, 4000, 8000, 16_000, 32_000]
    ws = [0.5e-4, 1e-4, 1.5e-4, 2e-4] #, 3e-4, 5e-4]
    ratios_total = []
    ratios2_total = []
    for w in ws:
        results_list = []
        ratios = []
        ratios2 = []
        for N in N_values:
            params = SimulationParameters(
                Tobs=86400 * 30,
                flow=1e-4,
                fhigh=1e-3,
                N_sources=N,
                alpha=4,
                A_min=1e-3,
                psd_level=2.111,
                w=w,
                # fixed_amplitude=1e-3,
            )
            
            sim = ConfusionBackgroundSimulator(params, tophat_fd_waveform, priors)
            if N < 100:
                results = sim.run(n_realizations=100, seed=42, show_progress=False)
            else:
                results = sim.run(n_realizations=100, seed=42, show_progress=False)
            B = sim.params.fhigh - sim.params.flow 
            # analytic = 4 * N * (1e-3)**4 / params.B**2
            analytic = results.var_Sconf_tf * B**2 / (B - w)**2
            ratio = analytic / results.measured_variance
            print(f"N={N:4d}, w={w}: measured={results.measured_variance:.3e}, "
                f"analytic={analytic:.3e}, ratio={ratio:.2f}")
            ratios.append(ratio)
            results_list.append(results)
        ratios_total.append(ratios)
        ratios2_total.append(ratios2)

        plt.plot(np.log10(N_values), np.log10(ratios), '-o', label=f'w/B={np.round(w/B, 4)}')
    plt.xlabel("$\\log_{10} N_{tot}$")
    plt.ylabel("$\\sigma_{an} / \\sigma_{sim}$")
    plt.legend()
    plt.savefig('plots/ratio_test.png')
    plt.close()

    ratios_total = np.array(ratios_total)
    print('ratios shape, ', ratios_total.shape)
    print('ws shape', len(ws))
    for ii, N in enumerate(N_values):
        plt.plot(ws, ratios_total[:, ii], '-o', label=f'N={N}')
    plt.xlabel("$w$")
    plt.ylabel("$\\sigma_{an} / \\sigma_{sim}$")
    plt.legend()
    plt.savefig('plots/ratios_vs_ws.png')
    plt.close()

    mus = np.array(N_values)[None, :] * np.array(ws)[:, None] / B
    log_ratio = np.log10(ratios_total)
    log_mu = np.log10(mus)
    p = np.polyfit(log_mu.flatten(), log_ratio.flatten(), 1)
    print('fit results', p)


#     def linear(x, a, b):
#         return a + b * x
# 
#     popt, _ = curve_fit(linear, log_mu, log_ratio)
#     print(f"a = {popt[0]:.3f}, b = {popt[1]:.3f}")
    N_grid = np.broadcast_to(np.array(N_values)[None, :], mus.shape)

    plt.figure()
    sc = plt.scatter(log_mu.flatten(), log_ratio.flatten(), c=np.log10(N_grid.flatten()), cmap='viridis')

    # plt.scatter(np.log10(mus.flatten()), np.log10(ratios_total.flatten()), label=f'N={N}')

    plt.xlabel("$\mu$")
    plt.ylabel("$\\sigma_{an} / \\sigma_{sim}$")
    plt.legend()
    plt.savefig('plots/ratios_vs_mus.png')
    plt.close()


    return results_list



if __name__ == "__main__":
    # Import your modules here
    # from your_code import tophat_fd_waveform, priors
    
    # results = example_basic_simulation(tophat_fd_waveform, priors)
    # results = example_fixed_amplitude(tophat_fd_waveform, priors)
    # results = example_poisson_N(tophat_fd_waveform, priors)
    results = example_parameter_scan(tophat_fd_waveform, priors)
    
    print("Import your waveform function and priors module to run examples.")
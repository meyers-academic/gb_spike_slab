import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tophat_populations.confusion_noise import get_threshold_and_conf_noise_for_powerlaw as get_threshold_and_conf_noise
from jaxgb_populations.matched_filter import log_likelihood_multi_channel

def numpyro_model(data, psd, waveform_generator=None, max_num_resolvable=20, A_lower=None,
                  frequencies=None, T_obs=30*86400, rho_th=10, f_min=None, f_max=None, population_flag=True, n_samples=256, total_N_sources=None):

    if waveform_generator is None:
        try:
            from gb_spike_slab import WaveformGenerator
            waveform_generator = WaveformGenerator(t_obs=T_obs, n_samples=n_samples)
        except ImportError:
            raise ImportError("WaveformGenerator not found. Please ensure gb_spike_slab is installed and accessible or pass a custom generator.")


    # sample psd
    #log10_psd = jnp.log10(psd)# if we want to sample the psd, do it in log: numpyro.sample('log10_psd', dist.Uniform(-1, 2))

    # Self-consistency: get threshold and confusion noise
    alpha = numpyro.deterministic("alpha", 4.)

    if population_flag:
        # sample total number of sources    
        log_10_N_sources = numpyro.sample("log_10_N_sources", dist.Uniform(1, 4))
        print(f"Sampling log_10_N_sources: {log_10_N_sources}")

        Ath, Sconf_mean, var_Sconf_ff = get_threshold_and_conf_noise(
        T_obs, f_max - f_min, psd, 10**log_10_N_sources, alpha, rho_th, A_lower, 0, )

        sconf = numpyro.sample("sconf", dist.TruncatedNormal(Sconf_mean, jnp.sqrt(2 * var_Sconf_ff), low=0.0))

    else:
        # there is no confusion noise, include all sources so the threshold amplitude is 0.
        Ath = 0.0
        Sconf_mean = 0.0
        var_Sconf_ff = 0.0
        sconf = jnp.zeros_like(frequencies)

    # keep track of these in case we want them
    numpyro.deterministic("Ath", Ath)
    numpyro.deterministic("Sconf_mean", Sconf_mean)
    numpyro.deterministic("Sconf_var_an", 2 * var_Sconf_ff)


    # Confusion noise, incorporating its variance
    # Truncated normal seems right, given simulations.
    # sconf_sigma = numpyro.sample('sconf_sigma', dist.Uniform(0.1, jnp.sqrt(var_Sconf_ff) * 5))
    # var_Sconf_ff may still not be right yet. But it's working here...so...yay

    # Resolvable source parameters — amplitude truncated at Ath
    if population_flag:
        # Use Ath as the lower bound on the amplitudes
        # I think this should work for implementing our hard SNR cut. 
        # Instead of using a powerlaw, inverse CDF sample.
        
        # draw from a value in the CDF
        u_amp = numpyro.sample("u_amp", dist.Uniform(0, 1).expand([max_num_resolvable]))
        # transform from CDF to PDF of the prior
        Ath_1ma = Ath**(1 - alpha)
        hi_1ma = jnp.array(100.0)**(1 - alpha)
        amplitude = numpyro.deterministic("amplitude",
            (Ath_1ma + u_amp * (hi_1ma - Ath_1ma))**(1 / (1 - alpha)))

    else:
        # if we're not doing the population inference, just draw from the original prior
        # TODO find a way to pass priors around... 
        log_amplitude = numpyro.sample("log_amplitude", dist.Uniform(-23.0, -19.0).expand([max_num_resolvable]))
        amplitude = 10**log_amplitude
        numpyro.deterministic("amplitude", amplitude)

    # wf params
    f0_prior_center = (f_min + f_max) / 2.0
    f0_prior_width = (f_max - f_min) / 2.0
    f0 = numpyro.sample("f0", dist.Uniform(f0_prior_center - f0_prior_width,
                                         f0_prior_center + f0_prior_width).expand([max_num_resolvable]))

    # fdot: sample scaled version (1e18 * fdot) with uniform prior
    scaled_fdot = numpyro.sample("scaled_fdot", dist.Uniform(-1000.0, 1000.0).expand([max_num_resolvable]))
    fdot = scaled_fdot / 1e18
    numpyro.deterministic("fdot", fdot)

    dec = numpyro.sample("dec", dist.Uniform(-np.pi/2, np.pi/2).expand([max_num_resolvable]))
    ra = numpyro.sample("ra", dist.Uniform(0, 2*np.pi).expand([max_num_resolvable]))
    # actually just fix these for now.
    dec = jnp.full(len(dec), np.pi/4.)
    ra =  jnp.full(len(dec), np.pi/4.)

    polarization = numpyro.sample("polarization", dist.Uniform(0, 2*np.pi).expand([max_num_resolvable]))
    inclination = numpyro.sample("inclination", dist.Uniform(0, np.pi).expand([max_num_resolvable]))
    initial_phase = numpyro.sample("initial_phase", dist.Uniform(0, 2*np.pi).expand([max_num_resolvable]))

    params = jnp.stack([
        f0, fdot, amplitude, ra, dec,
        polarization, inclination, initial_phase
    ], axis=1)

    # Indicators start at 50% 
    z = numpyro.sample("z", dist.Bernoulli(0.5 * jnp.ones(max_num_resolvable)))

    # Generate waveform templates for all potential sources
    A_wf, E_wf, _ = waveform_generator.generate_waveforms(params)
    wf_freqs = waveform_generator.get_waveform_frequencies(params)
    
    # Interpolate to data frequency grid
    template_A = waveform_generator.interpolate_waveform(
        A_wf, wf_freqs, frequencies
    )  # Shape: (n_max_sources, n_freqs)
    
    template_E = waveform_generator.interpolate_waveform(
        E_wf, wf_freqs, frequencies
    ) 

    # Poisson prior on number of active sources
    if population_flag:
        N_res = 10**log_10_N_sources * (A_lower / Ath)**(alpha - 1)
    else:
        N_res = total_N_sources
    
    numpyro.deterministic("Nres", N_res)
    numpyro.deterministic("n_active_sources", jnp.sum(z))
    numpyro.factor("n_sources_prior", dist.Poisson(N_res).log_prob(jnp.sum(z)))

    templates = [template_A, template_E]
    variance = psd + [sconf for _ in range(len(psd))]  # Add confusion noise to the variance for both channels

    # Likelihood
    ll = numpyro.deterministic("likelihood", log_likelihood_multi_channel(
        data, templates, variance, frequencies, z))
    numpyro.factor('ll', ll)
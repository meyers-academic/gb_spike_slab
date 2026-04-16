import numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from tqdm import tqdm
from jaxopt import Bisection


@dataclass
class SimulationParameters:
    """Parameters for confusion background simulation."""
    
    # Observation parameters
    Tobs: float  # Observation time [s]
    flow: float  # Low frequency [Hz]
    fhigh: float  # High frequency [Hz]
    
    # Source population parameters
    N_sources: int  # Number of sources
    alpha: float  # Power law index
    A_min: float  # Minimum amplitude

    
    # Noise parameters
    psd_level: float  # One-sided PSD level of instrument noise
    
    # Detection parameters
    rho_th: float = 5.0  # SNR threshold
    
    # Waveform parameters
    w: float = 1e-4  # Width in frequency space
    A_max: float = 10.0  # Maximum amplitude 
    # Simulation options
    fixed_amplitude: Optional[float] = None  # If set, use fixed amplitude for all sources
    poisson_N: bool = False  # If True, draw N from Poisson distribution
    
    @property
    def df(self) -> float:
        """Frequency resolution [Hz]."""
        return 1 / self.Tobs
    
    @property
    def B(self) -> float:
        """Bandwidth [Hz]."""
        return self.fhigh - self.flow
    
    @property
    def Nbins(self) -> int:
        """Number of frequency bins."""
        return int(self.B * self.Tobs)
    
    @property
    def mu(self) -> float:
        """Expected sources per bin."""
        return self.N_sources * self.w / self.B
    
    @property
    def fdot(self) -> float:
        """Frequency derivative for linear chirp."""
        return self.w / self.Tobs
    
    @property
    def frequencies(self) -> np.ndarray:
        """Frequency array."""
        return np.arange(self.flow, self.fhigh + self.df, self.df)
    
    def get_threshold_and_confusion(
        self, 
        phase_variance: bool = True, 
        N_variance: bool = False
    ) -> Tuple[float, float, float]:
        """
        Compute detection threshold and confusion noise.
        
        Returns:
            Ath: Amplitude threshold
            Sconf: Confusion noise PSD
            var_Sconf: Variance of confusion noise
        """
        Nbins = self.B * self.Tobs
        sigma_0_sq = self.psd_level / self.Tobs
        mu = self.N_sources / Nbins
        
        beta = self.rho_th**2 * sigma_0_sq / self.A_min**2 / 4
        gamma = self.rho_th**2 * mu * (self.alpha - 1) / (3 - self.alpha) / 2.
        
        def F(x):
            return x**2 - beta - gamma * (x**(3 - self.alpha) - 1)
        
        if self.alpha == 4:
            coeffs = [1, 0, -(beta - gamma), -gamma]
            roots = np.roots(coeffs)
            xth = np.real(roots[np.isreal(roots) & (roots > 0)][0])
        else:
            bisec = Bisection(optimality_fun=F, lower=1, upper=10000)
            xth = bisec.run().params
        
        Ath = xth * self.A_min
        Sconf = (4 * self.Tobs * self.A_min**2 / self.rho_th**2) * (xth**2 - beta)
        
        avgA2 = (self.alpha - 1) / (3 - self.alpha) * self.A_min**(self.alpha - 1) * (Ath**(3-self.alpha) - self.A_min**(3-self.alpha))
        avgA4 = (self.alpha - 1) / (5 - self.alpha) * self.A_min**(self.alpha - 1) * (Ath**(5-self.alpha) - self.A_min**(5-self.alpha))
        
        # Amplitude variance
        var_Sconf = 4 * self.N_sources * (avgA4 - avgA2**2) / self.B**2
        
        # Phase variance
        if phase_variance:
            var_Sconf += avgA2**2 * 4 * self.N_sources / self.B**2
        
        # Number variance
        if N_variance:
            var_Sconf += avgA2**2 * 4 * self.N_sources / self.B**2
        
        return Ath, Sconf, var_Sconf


@dataclass
class SimulationResult:
    """Results from a single simulation realization."""
    
    signals: np.ndarray  # Complex frequency-domain signals
    amplitudes: np.ndarray  # Source amplitudes
    fcenters: np.ndarray  # Source center frequencies
    phases: np.ndarray  # Source phases
    snrs: np.ndarray  # Source SNRs


@dataclass 
class SimulationResults:
    """Aggregated results from multiple simulation realizations."""
    
    params: SimulationParameters
    Ath: float
    Sconf: float
    var_Sconf_ff: float  # amplitude only
    var_Sconf_tf: float  # amplitude + phase
    var_Sconf_tt: float  # amplitude + phase + N
    
    # Per-realization data
    power_spectra: np.ndarray = field(default_factory=lambda: np.array([]))
    correlations: List[np.ndarray] = field(default_factory=list)
    all_snrs: List[np.ndarray] = field(default_factory=list)
    
    @property
    def n_realizations(self) -> int:
        return len(self.power_spectra)
    
    @property
    def mean_power_spectrum(self) -> np.ndarray:
        return np.mean(self.power_spectra, axis=0)
    
    @property
    def band_averaged_power(self) -> np.ndarray:
        """Band-averaged power for each realization."""
        return np.array([np.mean(ps) for ps in self.power_spectra])
    
    @property
    def measured_variance(self) -> float:
        """Measured variance of band-averaged power."""
        return np.var(self.band_averaged_power)
    
    @property
    def measured_mean(self) -> float:
        """Measured mean of band-averaged power."""
        return np.mean(self.band_averaged_power)
    
    def variance_ratios(self) -> dict:
        """Compare measured variance to analytic predictions."""
        mv = self.measured_variance
        return {
            'ff (amp only)': self.var_Sconf_ff / mv,
            'tf (amp+phase)': self.var_Sconf_tf / mv,
            'tt (amp+phase+N)': self.var_Sconf_tt / mv,
        }
    
    def print_summary(self):
        """Print summary statistics."""
        print(f"N realizations: {self.n_realizations}")
        print(f"Measured mean: {self.measured_mean:.3e}")
        print(f"Analytic Sconf: {self.Sconf:.3e}")
        print(f"Measured variance: {self.measured_variance:.3e}")
        print(f"Variance ratios (analytic/measured):")
        for name, ratio in self.variance_ratios().items():
            print(f"  {name}: {ratio:.2f}")

    @property
    def mean_correlation(self) -> np.ndarray:
        """Average correlation function over realizations."""
        return np.mean(self.correlations, axis=0)
    
    def compute_power_correlation(self, max_lag: int = None) -> np.ndarray:
        """Correlation of power (|h_k|^2) between bins."""
        if max_lag is None:
            max_lag = int(self.params.w / self.params.df)
        
        # Use power spectra, not complex signals
        powers = self.power_spectra  # shape (n_realizations, n_bins)
        
        rho = np.zeros(max_lag)
        for lag in range(max_lag):
            if lag == 0:
                rho[0] = np.mean(powers * powers)
            else:
                rho[lag] = np.mean(powers[:, :-lag] * powers[:, lag:])
        
        rho /= rho[0]
        return rho

    # @property
    # def correlation_correction_factor(self) -> float:
    #     """Correction factor from bin-to-bin correlations."""
    #     rho = self.mean_correlation
    #     max_lag = int(self.params.w / self.params.df)
    #     return 1 + 2 * np.sum(rho[1:max_lag])
    @property
    def correlation_correction_factor(self) -> float:
        """Correction factor from bin-to-bin correlations."""
        max_lag = int(self.params.w / self.params.df)
        rho = self.compute_power_correlation(max_lag)
        return 1 + 2 * np.sum(rho[1:])

    @property
    def N_eff(self) -> float:
        """Effective number of independent bins."""
        K = self.params.Nbins
        return K / self.correlation_correction_factor

    @property
    def var_Sconf_corrected(self) -> float:
        """Variance corrected for bin correlations."""
        # Original formula has 1/B^2 assuming N_eff = B/w
        # Correct by ratio of assumed to actual N_eff
        N_eff_assumed = self.params.B / self.params.w
        return self.var_Sconf_tf * (N_eff_assumed / self.N_eff)


class ConfusionBackgroundSimulator:
    """Simulator for gravitational wave confusion backgrounds."""
    
    def __init__(self, params: SimulationParameters, waveform_func, priors_module):
        """
        Initialize simulator.
        
        Args:
            params: Simulation parameters
            waveform_func: Function to generate frequency-domain waveform
            priors_module: Module with sample_power_law_bounded function
        """
        self.params = params
        self.waveform_func = waveform_func
        self.priors = priors_module
        self.frequencies = jnp.array(params.frequencies)
        # Vectorized waveform function
        self.vmap_waveform = jit(vmap(
            lambda A, fc, phi: waveform_func(A, fc, phi, params.fdot, self.frequencies)
        ))
        
        # Precompute thresholds
        self.Ath, self.Sconf, _ = params.get_threshold_and_confusion()
    
    def generate_sources(self, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random source parameters."""
        if rng is None:
            rng = np.random.default_rng()
        
        # Number of sources
        if self.params.poisson_N:
            N = rng.poisson(self.params.N_sources)
        else:
            N = self.params.N_sources
        
        # Center frequencies
        fcenters = rng.uniform(self.params.flow, self.params.fhigh, N)
        
        # Amplitudes
        if self.params.fixed_amplitude is not None:
            amplitudes = np.ones(N) * self.params.fixed_amplitude
        else:
            amplitudes = self.priors.sample_power_law_bounded(
                self.params.alpha, self.params.A_min, self.params.A_max, N
            )
        
        # Phases
        phases = rng.uniform(0, 2 * np.pi, N)
        
        return amplitudes, fcenters, phases
    
    def compute_SB_self_consistent(self, amplitudes: np.ndarray, tol: float = 1e-8, max_iter: int = 100) -> Tuple[float, float]:
        """
        Find self-consistent threshold and confusion noise from simulated amplitudes.
        
        Args:
            amplitudes: Array of source amplitudes
            tol: Relative tolerance for convergence
            max_iter: Maximum iterations
            
        Returns:
            A_th: Self-consistent amplitude threshold
            S_B: Confusion noise from sub-threshold sources
        """
        S_n = self.params.psd_level
        Tobs = self.params.Tobs
        B = self.params.B
        rho_th = self.params.rho_th
        
        # Initial guess: threshold based on instrument noise only
        A_th = (rho_th / 2) * np.sqrt(S_n / Tobs)
        
        for _ in range(max_iter):
            # S_B from sources below current threshold
            S_B = 2 * np.sum(amplitudes[amplitudes < A_th]**2) / B
            
            # Updated threshold from total noise
            A_th_new = (rho_th / 2) * np.sqrt((S_n + S_B) / Tobs)
            
            if np.abs(A_th_new - A_th) < tol * A_th:
                return A_th_new, S_B
            
            A_th = A_th_new
        
        return A_th, S_B
    
    def compute_snrs(self, amplitudes: np.ndarray) -> np.ndarray:
        """Compute SNRs for given amplitudes."""
        return 2 * amplitudes / np.sqrt(self.Sconf + self.params.psd_level) * np.sqrt(self.params.Tobs)
    
    def generate_signals(
        self, 
        amplitudes: np.ndarray, 
        fcenters: np.ndarray, 
        phases: np.ndarray
    ) -> np.ndarray:
        """Generate frequency-domain signals from sub-threshold sources."""
        # Filter to sub-threshold sources
        mask = amplitudes < self.Ath
        
        if not np.any(mask):
            return jnp.zeros(len(self.frequencies), dtype=complex)
        
        signals = jnp.sum(
            self.vmap_waveform(amplitudes[mask], fcenters[mask], phases[mask]), 
            axis=0
        )
        return signals
    
    def compute_correlation(self, signals: np.ndarray) -> np.ndarray:
        """Compute autocorrelation of signals."""
        xi = np.fft.ifft(np.abs(np.fft.fft(signals))**2).real
        xi = xi / xi[0]
        return xi
    
    def run_single(self, rng: Optional[np.random.Generator] = None) -> SimulationResult:
        """Run a single simulation realization."""
        amplitudes, fcenters, phases = self.generate_sources(rng)
        snrs = self.compute_snrs(amplitudes)
        signals = self.generate_signals(amplitudes, fcenters, phases)
        
        return SimulationResult(
            signals=signals,
            amplitudes=amplitudes,
            fcenters=fcenters,
            phases=phases,
            snrs=snrs
        )


    
    def run(self, n_realizations: int, seed: Optional[int] = None, show_progress: bool = True) -> SimulationResults:
        """
        Run multiple simulation realizations.
        
        Args:
            n_realizations: Number of realizations to run
            seed: Random seed
            show_progress: Show progress bar
            
        Returns:
            SimulationResults object
        """
        rng = np.random.default_rng(seed)
        
        # Get variance predictions
        _, _, var_ff = self.params.get_threshold_and_confusion(phase_variance=False, N_variance=False)
        _, _, var_tf = self.params.get_threshold_and_confusion(phase_variance=True, N_variance=False)
        _, _, var_tt = self.params.get_threshold_and_confusion(phase_variance=True, N_variance=True)
        
        # Storage
        power_spectra = []
        correlations = []
        all_snrs = []
        
        # Edge trimming indices
        edge_bins = int(self.params.w / 2 * self.params.Tobs)
        
        iterator = range(n_realizations)
        if show_progress:
            iterator = tqdm(iterator)
        
        for _ in iterator:
            result = self.run_single(rng)
            
            # Trim edges and compute power spectrum
            trimmed_signals = result.signals[edge_bins:-edge_bins]
            power = 2 * np.abs(trimmed_signals)**2 / self.params.Tobs
            power_spectra.append(power)
            
            # Correlation
            correlations.append(self.compute_correlation(result.signals))
            
            # SNRs
            all_snrs.append(result.snrs)
        
        return SimulationResults(
            params=self.params,
            Ath=self.Ath,
            Sconf=self.Sconf,
            var_Sconf_ff=var_ff,
            var_Sconf_tf=var_tf,
            var_Sconf_tt=var_tt,
            power_spectra=np.array(power_spectra),
            correlations=correlations,
            all_snrs=all_snrs
        )


def compute_complex_correlation(signals: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute complex correlation function between frequency bins.
    
    Args:
        signals: Array of shape (n_realizations, n_bins)
        max_lag: Maximum lag to compute
        
    Returns:
        Complex correlation vs lag
    """
    xi_complex = np.zeros(max_lag, dtype=complex)
    for lag in range(max_lag):
        xi_complex[lag] = np.mean(
            signals[:, :-max_lag] * np.conj(signals[:, lag:lag-max_lag or None])
        )
    xi_complex /= xi_complex[0]
    return xi_complex


"""
synthetic_lc.py
===============
Simulate synthetic stellar cluster lightcurves for studying how populations of
oscillators — combined with observational window functions and flux noise — shape
the Lomb-Scargle periodogram.

Classes
-------
Oscillator
    Single sinusoidal oscillator (one star).
OscillatorCluster
    Superposition of N oscillators, defined directly or from distributions.
WindowFunction
    Specifies which timestamps from a dense baseline survive as observations.
NoiseModel
    Adds flux noise and assigns per-point error bars.
SyntheticLC
    Combines the above to produce (t, x, err_x) ready for compute_lsp().

Typical usage
-------------
>>> from synthetic_lc import (Oscillator, OscillatorCluster,
...                            WindowFunction, NoiseModel, SyntheticLC)
>>> from invariant_LSP import compute_lsp
>>> import scipy.stats as stats
>>>
>>> # Single oscillator, 3-sector TESS window
>>> cluster = OscillatorCluster([Oscillator(frequency=1/5.0, amplitude=0.01)])
>>> window  = WindowFunction.from_theoretical(n_sectors=3)
>>> noise   = NoiseModel.gaussian(sigma=0.003)
>>> slc     = SyntheticLC(cluster, window, noise, seed=0)
>>> t, x, err_x = slc.generate()
>>> result  = compute_lsp(t, x, err_x)
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from invariant_LSP import compute_lsp, _effective_baseline


# ---------------------------------------------------------------------------
# LSP constants — must match invariant_LSP.py defaults
# ---------------------------------------------------------------------------
LSP_P_MIN = 0.1    # days
LSP_P_MAX = 10.0   # days
LSP_ALPHA = 5
LSP_F_LIM = (1/LSP_P_MAX, 1/LSP_P_MIN)   # (0.1, 10) 1/day


# ---------------------------------------------------------------------------
# Oscillator
# ---------------------------------------------------------------------------

class Oscillator:
    """Single sinusoidal oscillator.

    Flux contribution:  A * cos(2π f t + φ)

    Parameters
    ----------
    frequency : float
        Oscillation frequency in 1/day.  Use ``1/period`` to specify by period.
    amplitude : float
        Peak fractional flux amplitude (unitless; 0.01 = 1% peak variation).
    phase : float
        Initial phase in radians.  Default 0.

    # FUTURE LOCATION FOR Multi-harmonic sinusoid Sources
    """

    def __init__(self, frequency: float, amplitude: float, phase: float = 0.0):
        self.frequency = float(frequency)
        self.amplitude = float(amplitude)
        self.phase     = float(phase)

    def flux(self, t: np.ndarray) -> np.ndarray:
        """Return the flux contribution at times *t* (days)."""
        return self.amplitude * np.cos(2.0 * np.pi * self.frequency * t + self.phase)

    def __repr__(self):
        return (f"Oscillator(f={self.frequency:.4g} /day, "
                f"P={1/self.frequency:.3g} day, "
                f"A={self.amplitude:.3g}, φ={self.phase:.3g} rad)")


# ---------------------------------------------------------------------------
# OscillatorCluster
# ---------------------------------------------------------------------------

class OscillatorCluster:
    """Superposition of N sinusoidal oscillators, one per star in a cluster.

    The combined normalized flux is

        F(t) = 1 + Σ_i  A_i * cos(2π f_i t + φ_i)

    which is centred on 1, matching the convention used by ``normalize_flux``
    in ``invariant_summary_stats.py``.

    Parameters
    ----------
    oscillators : list[Oscillator]
        Individual oscillators to superpose.
    seed : int or None
        Random seed used to assign phases when N > 1.  Each oscillator's phase
        is replaced with a draw from Uniform[0, 2π).  For N=1 the single
        oscillator's phase is kept as-is.
    """

    def __init__(self, oscillators: list, seed=None):
        osc_list = list(oscillators)
        if len(osc_list) > 1:
            rng = np.random.default_rng(seed)
            phases = rng.uniform(0.0, 2.0 * np.pi, size=len(osc_list))
            osc_list = [Oscillator(o.frequency, o.amplitude, phi)
                        for o, phi in zip(osc_list, phases)]
        self._oscillators = osc_list

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_distributions(cls, N: int, freq_dist, amp_dist, seed=None):
        """Build a cluster by sampling frequencies and amplitudes from distributions.

        Parameters
        ----------
        N : int
            Number of oscillators (stars).
        freq_dist : scipy.stats distribution
            Distribution for frequencies (1/day).  Must have an ``rvs(N)`` method.
            Example: ``scipy.stats.uniform(loc=1/10, scale=1/0.5 - 1/10)``
        amp_dist : scipy.stats distribution
            Distribution for amplitudes (fractional flux).  Must have ``rvs(N)``.
            Example: ``scipy.stats.lognorm(s=0.5, scale=0.005)``
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        OscillatorCluster
        """
        rng = np.random.default_rng(seed)
        # Convert rng seed to an int for scipy.stats.  We derive it from rng so
        # all random draws share the same top-level seed.
        scipy_seed = int(rng.integers(0, 2**31))

        freqs  = freq_dist.rvs(N, random_state=scipy_seed)
        amps   = amp_dist.rvs(N, random_state=scipy_seed + 1)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=N)

        oscillators = [
            Oscillator(frequency=f, amplitude=a, phase=phi)
            for f, a, phi in zip(freqs, amps, phases)
        ]
        return cls(oscillators)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def flux(self, t: np.ndarray) -> np.ndarray:
        """Return the combined normalized flux at times *t* (days).

        The result is 1 + Σ individual oscillator contributions, centred on 1.
        """
        t = np.asarray(t, dtype=float)
        total = np.ones_like(t)
        for osc in self._oscillators:
            total += osc.flux(t)
        return total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def oscillators(self):
        return list(self._oscillators)

    @property
    def frequencies(self) -> np.ndarray:
        return np.array([o.frequency for o in self._oscillators])

    @property
    def amplitudes(self) -> np.ndarray:
        return np.array([o.amplitude for o in self._oscillators])

    @property
    def phases(self) -> np.ndarray:
        return np.array([o.phase for o in self._oscillators])

    def __len__(self):
        return len(self._oscillators)

    def __repr__(self):
        return f"OscillatorCluster(N={len(self)})"


# ---------------------------------------------------------------------------
# WindowFunction
# ---------------------------------------------------------------------------

class WindowFunction:
    """Defines which timestamps in a dense baseline grid are 'observed'.

    Two construction modes:

    * **Theoretical** — built from ``n_sectors`` × ``sector_length`` blocks
      separated by ``gap_durations``.  ``apply(t_full)`` returns the subset of
      the dense time grid that falls inside any sector interval.

    * **Real data** — wraps an existing time array (e.g., from
      ``build_lc_for_row``).  ``apply()`` ignores ``t_full`` and returns the
      stored array directly.
    """

    def __init__(self, *, intervals=None, real_t=None):
        """Private constructor — use the class-methods below."""
        self._intervals = intervals   # list of (t_start, t_end) or None
        self._real_t    = real_t      # ndarray or None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_theoretical(cls, n_sectors: int, sector_length: float = 27.0,
                         gap_durations=None, t0: float = 0.0):
        """Build a TESS-like window from sector lengths and inter-sector gaps.

        Parameters
        ----------
        n_sectors : int
            Number of observed sectors.
        sector_length : float
            Duration of each sector in days.  Default 27.0 (one TESS sector).
        gap_durations : list[float] or None
            Inter-sector gap lengths in days, length ``n_sectors - 1``.
            If ``None``, defaults to 13.5 days between every sector (typical TESS
            inter-sector gap).
        t0 : float
            Start time of the first sector.  Default 0.

        Returns
        -------
        WindowFunction
        """
        if n_sectors < 1:
            raise ValueError("n_sectors must be >= 1")
        if gap_durations is None:
            gap_durations = [13.5] * (n_sectors - 1)
        if len(gap_durations) != n_sectors - 1:
            raise ValueError(
                f"gap_durations must have length n_sectors-1={n_sectors-1}, "
                f"got {len(gap_durations)}"
            )

        intervals = []
        cursor = float(t0)
        for i in range(n_sectors):
            t_start = cursor
            t_end   = cursor + sector_length
            intervals.append((t_start, t_end))
            cursor  = t_end
            if i < n_sectors - 1:
                cursor += gap_durations[i]

        return cls(intervals=intervals)

    @classmethod
    def from_real_data(cls, t: np.ndarray):
        """Wrap an existing time array as a window function.

        Parameters
        ----------
        t : array-like
            Observation times (days) from a real lightcurve.

        Returns
        -------
        WindowFunction
        """
        return cls(real_t=np.asarray(t, dtype=float))

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def apply(self, t_full: np.ndarray) -> np.ndarray:
        """Return the observed subset of *t_full*.

        For theoretical windows: returns points of ``t_full`` that fall inside
        any sector interval ``[t_start, t_end]``.

        For real-data windows: returns the stored time array directly (ignoring
        ``t_full``).

        Parameters
        ----------
        t_full : ndarray
            Dense time grid built by ``SyntheticLC``.

        Returns
        -------
        ndarray
            Subset of observation times.
        """
        if self._real_t is not None:
            return self._real_t.copy()

        t_full = np.asarray(t_full, dtype=float)
        mask = np.zeros(len(t_full), dtype=bool)
        for t_start, t_end in self._intervals:
            mask |= (t_full >= t_start) & (t_full <= t_end)
        return t_full[mask]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_span(self) -> float:
        """Total time span from first observation to last (days)."""
        if self._real_t is not None:
            return float(np.max(self._real_t) - np.min(self._real_t))
        t_start_first = self._intervals[0][0]
        t_end_last    = self._intervals[-1][1]
        return float(t_end_last - t_start_first)

    @property
    def intervals(self):
        """List of (t_start, t_end) sector intervals, or None for real-data mode."""
        return self._intervals

    def __repr__(self):
        if self._real_t is not None:
            return f"WindowFunction(real_data, N={len(self._real_t)}, span={self.total_span:.1f} d)"
        n = len(self._intervals)
        return f"WindowFunction(theoretical, n_sectors={n}, span={self.total_span:.1f} d)"


# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

class NoiseModel:
    """Adds flux noise to a lightcurve and provides per-point error bars.

    Two modes:

    * **Gaussian** — constant noise level σ.  ``err_x = σ * ones``;
      noise drawn as N(0, σ) at every point.

    * **Real errors** — bootstrap-samples from a provided ``err_x`` array.
      For each observed timestamp an error value is drawn from the real
      distribution; noise is then drawn as N(0, err_i).
    """

    def __init__(self, *, sigma=None, real_err=None):
        """Private constructor — use class-methods below."""
        self._sigma    = sigma
        self._real_err = real_err

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def gaussian(cls, sigma: float):
        """Constant Gaussian noise model.

        Parameters
        ----------
        sigma : float
            Noise standard deviation in normalized flux units.

        Returns
        -------
        NoiseModel
        """
        return cls(sigma=float(sigma))

    @classmethod
    def from_real_errors(cls, err_x: np.ndarray):
        """Bootstrap from a real flux-error array.

        Parameters
        ----------
        err_x : array-like
            Per-point flux uncertainties from a real lightcurve.

        Returns
        -------
        NoiseModel
        """
        return cls(real_err=np.asarray(err_x, dtype=float))

    # ------------------------------------------------------------------
    # Core method
    # ------------------------------------------------------------------

    def apply(self, x: np.ndarray, rng=None) -> tuple:
        """Add noise to *x* and return (x_noisy, err_x).

        Parameters
        ----------
        x : ndarray
            Clean flux values (typically from ``OscillatorCluster.flux``).
        rng : numpy.random.Generator or None
            Random generator for reproducibility.  If ``None`` a new default
            generator is used (non-reproducible).

        Returns
        -------
        x_noisy : ndarray
        err_x   : ndarray
        """
        if rng is None:
            rng = np.random.default_rng()

        x = np.asarray(x, dtype=float)
        N = len(x)

        if self._sigma is not None:
            err_x   = np.full(N, self._sigma)
            noise   = rng.normal(0.0, self._sigma, size=N)
        else:
            # Bootstrap-sample per-point errors from real error distribution
            idx     = rng.integers(0, len(self._real_err), size=N)
            err_x   = self._real_err[idx]
            noise   = rng.normal(0.0, err_x)

        return x + noise, err_x

    def __repr__(self):
        if self._sigma is not None:
            return f"NoiseModel(gaussian, σ={self._sigma:.3g})"
        return f"NoiseModel(real_errors, N_pool={len(self._real_err)})"


# ---------------------------------------------------------------------------
# SyntheticLC
# ---------------------------------------------------------------------------

class SyntheticLC:
    """Combine an OscillatorCluster, WindowFunction, and NoiseModel to produce
    a synthetic lightcurve ready for ``compute_lsp()``.

    Design
    ------
    1. A **dense** time grid ``t_full`` is built over ``[0, length]`` days at
       ``cadence_min`` spacing — this represents the "ground truth" LC.
    2. The window function selects the **observed** subset of timestamps.
    3. The cluster signal is evaluated at the observed timestamps.
    4. Noise is added via the noise model.

    The output ``(t, x, err_x)`` from ``generate()`` has the same format as
    ``build_lc_for_row()`` in ``invariant_LSP.py`` and drops directly into
    ``compute_lsp()``.

    Parameters
    ----------
    cluster : OscillatorCluster
    window : WindowFunction
    noise : NoiseModel
    length : float or None
        Duration of the dense baseline in days.  If ``None``, defaults to
        ``window.total_span``.
    cadence_min : float
        Cadence of the dense baseline in minutes.  Default 30.0.
    seed : int or None
        Master random seed.
    """

    def __init__(self, cluster: OscillatorCluster, window: WindowFunction,
                 noise: NoiseModel, length: float = None,
                 cadence_min: float = 30.0, seed=None):
        self.cluster     = cluster
        self.window      = window
        self.noise       = noise
        self.cadence_min = float(cadence_min)
        self.seed        = seed
        self._length     = float(length) if length is not None else window.total_span

        # Build the dense time grid once at construction time
        step = self.cadence_min / 1440.0          # convert minutes → days
        self._t_full = np.arange(0.0, self._length + step * 0.5, step)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(self) -> tuple:
        """Generate the windowed, noise-added lightcurve.

        Returns
        -------
        t : ndarray
            Observation times (days).
        x : ndarray
            Normalized flux with noise added.
        err_x : ndarray
            Per-point flux uncertainties.
        """
        rng   = np.random.default_rng(self.seed)
        t_obs = self.window.apply(self._t_full)
        x_obs = self.cluster.flux(t_obs)
        x_noisy, err_x = self.noise.apply(x_obs, rng=rng)
        return t_obs, x_noisy, err_x

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    @property
    def t_full(self) -> np.ndarray:
        """Dense time grid before windowing (days)."""
        return self._t_full.copy()

    def x_full(self) -> np.ndarray:
        """True (noiseless, unwindowed) cluster flux over the full baseline."""
        return self.cluster.flux(self._t_full)

    def __repr__(self):
        return (f"SyntheticLC(cluster={self.cluster!r}, "
                f"window={self.window!r}, "
                f"noise={self.noise!r}, "
                f"length={self._length:.1f} d, "
                f"cadence={self.cadence_min:.1f} min)")


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def run_sim(cluster, window, noise, length=None, cadence_min=30.0, seed=0,
            P_min=LSP_P_MIN, P_max=LSP_P_MAX, alpha=LSP_ALPHA, T_effective=True):
    """Generate a synthetic LC and compute its LSP.

    Defaults for P_min, P_max, alpha match invariant_LSP.add_invariant_LSP_stats
    so results are directly comparable to real-data periodograms.

    Parameters
    ----------
    T_effective : bool
        If True (default), df = 1/(alpha * T_eff) where T_eff is the sum of
        contiguous segment durations, *excluding* inter-sector gaps.
        If False, df uses the full wall-clock span t_max - t_min (gaps included),
        giving finer frequency bins but more aliasing structure from the gaps.

    Returns (slc, t, x, err_x, lsp_result).
    """
    slc = SyntheticLC(cluster, window, noise,
                      length=length, cadence_min=cadence_min, seed=seed)
    t, x, err_x = slc.generate()
    result = compute_lsp(t, x, err_x, P_min=P_min, P_max=P_max, alpha=alpha,
                         T_effective=T_effective)
    return slc, t, x, err_x, result


def plot_lc_and_lsp(slc, t, x, result, title='', ax_lc=None, ax_lsp=None,
                    color='C0', lc_alpha=0.6, lsp_alpha=0.9, show_true_lc=True,
                    yscale='log', xscale='linear', x_lim=LSP_F_LIM):
    """Plot the LC (left) and LSP (right) for one simulation."""
    standalone = (ax_lc is None)
    if standalone:
        fig, (ax_lc, ax_lsp) = plt.subplots(1, 2, figsize=(12, 3.5))

    if show_true_lc:
        ax_lc.plot(slc.t_full, slc.x_full(), lw=0.8, alpha=0.35, color=color, label='true signal')
    ax_lc.plot(t, x, '.', ms=1.5, alpha=lc_alpha, color=color, label='observed')
    ax_lc.set_xlabel('Time (days)')
    ax_lc.set_ylabel('Normalized flux')
    if title:
        ax_lc.set_title(title)

    freqs = result['freqs']
    power = result['power']
    fap   = result['fap_threshold']
    if x_lim is not None:
        m = (freqs >= x_lim[0]) & (freqs <= x_lim[1])
        freqs, power = freqs[m], power[m]
    ax_lsp.plot(freqs, power, lw=0.9, alpha=lsp_alpha, color=color)
    if np.isfinite(fap):
        ax_lsp.axhline(fap, color=color, lw=1.0, ls=':', alpha=0.7, label='1% FAP')
    ax_lsp.set_xlabel('Frequency (1/day)')
    ax_lsp.set_ylabel('LSP power')
    ax_lsp.set_yscale(yscale)
    ax_lsp.set_xscale(xscale)
    if x_lim is not None:
        ax_lsp.set_xlim(x_lim)

    if standalone:
        plt.tight_layout()
        plt.show()


def compare_periodograms(sims_dict, title='', yscale='log', xscale='linear',
                         x_lim=LSP_F_LIM, figsize=(10, 4), colors=None):
    """Overlay multiple periodograms on one axes.

    Parameters
    ----------
    sims_dict : dict
        Keys are labels (str), values are lsp_result dicts from run_sim().
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, ax = plt.subplots(figsize=figsize)
    for (label, result), color in zip(sims_dict.items(), colors):
        freqs = result['freqs']
        power = result['power']
        if x_lim is not None:
            m = (freqs >= x_lim[0]) & (freqs <= x_lim[1])
            freqs, power = freqs[m], power[m]
        ax.plot(freqs, power, lw=0.9, label=label, color=color)
        fap = result['fap_threshold']
        if np.isfinite(fap):
            ax.axhline(fap, color=color, lw=0.8, ls=':', alpha=0.6)

    ax.set_xlabel('Frequency (1/day)')
    ax.set_ylabel('LSP power')
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.legend(fontsize=8, ncol=2)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# run_configs — single-dict interface
# ---------------------------------------------------------------------------
#
# BASE_CONFIG keys
# ----------------
# Oscillator
#   N                : int or list[int]   -- number of stars
#   freq_distribution: str or list[str]   -- 'uniform' or 'lognormal'
#   freq             : float or list      -- center freq (1/day); width=0 -> delta
#   freq_width       : float or list      -- uniform width or lognormal sigma
#   amp_distribution : str or list[str]   -- 'uniform' or 'lognormal'
#   amp              : float or list      -- center amplitude; width=0 -> delta
#   amp_width        : float or list      -- uniform width or lognormal sigma
#
# LC / window
#   FullLC_Baseline  : float (scalar)     -- dense baseline length in days
#   sector_length    : float or list      -- sector duration in days (default 27)
#   cadence          : float or list      -- cadence in minutes (default 30)
#   n_sectors        : list of (int, list) tuples  -- (n_sectors, gap_durations)
#                      iterated as a unit; e.g. [(1,[]), (2,[265])]
#
# Noise
#   sigma            : float or list      -- Gaussian noise sigma
#
# LSP
#   T_effective      : bool or list[bool] -- True: df=1/(alpha*T_eff) gaps excluded
#                                            False: df=1/(alpha*T_full) gaps incl.
#   P_min, P_max     : float or list      -- period range in days
#   alpha            : int or list        -- oversampling factor (default 5)
#
# Any param with >1 list element is varied.
# multipanel=False: all cartesian-product combinations on one plot.
# multipanel=True : one row per varied param (OFAT); default = first value of each
#                   varied param; each row shows default + variations of one param.
# plot_LC=True    : add a left-hand LC column alongside each periodogram panel.
# LC_xlim=(0,27)  : x-axis limits for LC panels; None shows the full baseline.


def _to_list(v):
    return v if isinstance(v, list) else [v]


def _build_freq_dist(dist_type, freq, freq_width):
    if freq_width == 0:
        return stats.uniform(loc=freq, scale=0)
    if dist_type == 'uniform':
        return stats.uniform(loc=freq, scale=freq_width)
    if dist_type == 'lognormal':
        P0, s = 1.0 / freq, freq_width
        class _Dist:
            def rvs(self_, N, random_state=None):
                P = stats.lognorm(s=s, scale=P0).rvs(N, random_state=random_state)
                return 1.0 / np.clip(P, 0.1, 10.0)
        return _Dist()
    raise ValueError(f'Unknown freq_distribution: {dist_type!r}')


def _build_amp_dist(dist_type, amp, amp_width):
    if amp_width == 0:
        return stats.uniform(loc=amp, scale=0)
    if dist_type == 'uniform':
        return stats.uniform(loc=amp, scale=amp_width)
    if dist_type == 'lognormal':
        return stats.lognorm(s=amp_width, scale=amp)
    raise ValueError(f'Unknown amp_distribution: {dist_type!r}')


def _fmt_val(k, v):
    if k == 'n_sectors':
        ns, gaps = v
        gap_str = f'{gaps[0]:.0f}d' if gaps else '0d'
        return f'ns={ns},gap={gap_str}'
    if k == 'T_effective':
        return 'T=eff' if v else 'T=full'
    if isinstance(v, float):
        return f'{k}={v:.3g}'
    return f'{k}={v}'


def _run_one(run_cfg, baseline, seed, sim_seed):
    """Run one simulation. Returns (lsp_result, t, x, meta)."""
    ns_val, gaps = run_cfg.get('n_sectors', (1, []))
    cluster = OscillatorCluster.from_distributions(
        N         = run_cfg.get('N', 1),
        freq_dist = _build_freq_dist(
                        run_cfg.get('freq_distribution', 'uniform'),
                        run_cfg.get('freq',       1.0),
                        run_cfg.get('freq_width', 0),
                    ),
        amp_dist  = _build_amp_dist(
                        run_cfg.get('amp_distribution', 'uniform'),
                        run_cfg.get('amp',       1.0),
                        run_cfg.get('amp_width', 0),
                    ),
        seed=seed,
    )
    window = WindowFunction.from_theoretical(
        n_sectors     = ns_val,
        sector_length = run_cfg.get('sector_length', 27.0),
        gap_durations = gaps if gaps else None,
    )
    sigma = run_cfg.get('sigma', 0)
    noise = NoiseModel.gaussian(sigma if sigma > 0 else 1e-12)
    _, t, x, err_x, result = run_sim(
        cluster, window, noise,
        length      = baseline,
        cadence_min = run_cfg.get('cadence',     30.0),
        seed        = sim_seed,
        P_min       = run_cfg.get('P_min',  LSP_P_MIN),
        P_max       = run_cfg.get('P_max',  LSP_P_MAX),
        alpha       = run_cfg.get('alpha',  LSP_ALPHA),
        T_effective = run_cfg.get('T_effective', True),
    )
    T_full = float(t.max() - t.min())
    T_eff  = float(_effective_baseline(t))
    df     = float(result['freqs'][1] - result['freqs'][0]) if len(result['freqs']) > 1 else float('nan')
    meta   = dict(T_full=T_full, T_eff=T_eff, df=df)
    return result, t, x, meta


def _meta_suffix(meta):
    return f"  [T_eff={meta['T_eff']:.1f}d, T_full={meta['T_full']:.1f}d, df={meta['df']:.2e}/d]"


def _draw_pg_on_ax(ax, pg_dict, yscale, xscale, x_lim, colors):
    for (label, result), color in zip(pg_dict.items(), colors):
        freqs = result['freqs']
        power = result['power']
        if x_lim is not None:
            m = (freqs >= x_lim[0]) & (freqs <= x_lim[1])
            freqs, power = freqs[m], power[m]
        ax.plot(freqs, power, lw=0.9, label=label, color=color)
        fap = result['fap_threshold']
        if np.isfinite(fap):
            ax.axhline(fap, color=color, lw=0.8, ls=':', alpha=0.6)
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_xlabel('Frequency (1/day)')
    ax.set_ylabel('LSP power')
    ax.legend(fontsize=8, ncol=1)


def _draw_lc_on_ax(ax, lc_dict, LC_xlim, colors):
    for (label, (t, x)), color in zip(lc_dict.items(), colors):
        ax.plot(t, x, '.', ms=1.2, alpha=0.5, color=color, label=label)
    if LC_xlim is not None:
        ax.set_xlim(LC_xlim)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Normalized flux')
    ax.legend(fontsize=8, ncol=1)


def run_configs(base_config, title='', yscale='log', xscale='linear',
                x_lim=LSP_F_LIM, figsize=None, multipanel=False,
                plot_LC=False, LC_xlim=(0, 27), seed=42, sim_seed=0):
    """Run simulations for all combinations of list parameters in base_config.

    Every legend label automatically shows T_eff, T_full, and df for that curve.

    Parameters
    ----------
    multipanel : bool
        False — all cartesian-product combinations on one axes.
        True  — one row per varied param (OFAT); default = first value of each
                varied param; each row shows default + variations of that param.
    plot_LC : bool
        If True, add a left-hand LC column alongside each periodogram panel.
    LC_xlim : tuple or None
        x-axis limits for LC panels; None shows full baseline.
    """
    baseline = base_config.get('FullLC_Baseline', 200)
    if isinstance(baseline, list):
        baseline = baseline[0]

    params = {k: _to_list(v)
              for k, v in base_config.items() if k != 'FullLC_Baseline'}

    varied_keys = [k for k, v in params.items() if len(v) > 1]
    varied_vals = [params[k] for k in varied_keys]
    fixed       = {k: v[0] for k, v in params.items() if len(v) == 1}
    default_cfg = {**fixed, **{k: v[0] for k, v in zip(varied_keys, varied_vals)}}

    n_cols      = 2 if plot_LC else 1
    prop_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # ---- single-panel: full cartesian product --------------------------
    if not multipanel:
        combos = list(itertools.product(*varied_vals)) if varied_vals else [()]
        pg_dict = {}
        lc_dict = {}
        for combo in combos:
            run = dict(fixed)
            label_parts = []
            for k, v in zip(varied_keys, combo):
                run[k] = v
                label_parts.append(_fmt_val(k, v))
            result, t, x, meta = _run_one(run, baseline, seed, sim_seed)
            label = (', '.join(label_parts) if label_parts else 'default') + _meta_suffix(meta)
            pg_dict[label] = result
            lc_dict[label] = (t, x)

        colors = prop_colors[:len(pg_dict)]
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=figsize or (14 if plot_LC else 10, 4),
                                 squeeze=False)
        col = 0
        if plot_LC:
            _draw_lc_on_ax(axes[0, col], lc_dict, LC_xlim, colors)
            col += 1
        _draw_pg_on_ax(axes[0, col], pg_dict, yscale, xscale, x_lim, colors)
        if title:
            fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.show()
        return pg_dict

    # ---- multi-panel: OFAT (one factor at a time) ----------------------
    n_rows = max(len(varied_keys), 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=figsize or (14 if plot_LC else 10, 4 * n_rows),
                             squeeze=False)

    default_result, default_t, default_x, default_meta = _run_one(default_cfg, baseline, seed, sim_seed)
    default_label = ('default (' + ', '.join(
        _fmt_val(k, params[k][0]) for k in varied_keys) + ')'
        + _meta_suffix(default_meta))

    all_results = {}
    for row_i, (key, vals) in enumerate(zip(varied_keys, varied_vals)):
        pg_row = {default_label: default_result}
        lc_row = {default_label: (default_t, default_x)}
        for v in vals[1:]:
            cfg = {**default_cfg, key: v}
            result, t, x, meta = _run_one(cfg, baseline, seed, sim_seed)
            lbl = _fmt_val(key, v) + _meta_suffix(meta)
            pg_row[lbl] = result
            lc_row[lbl] = (t, x)

        colors = prop_colors[:len(pg_row)]
        col = 0
        if plot_LC:
            _draw_lc_on_ax(axes[row_i, col], lc_row, LC_xlim, colors)
            axes[row_i, col].set_title(f'LC  |  varying: {key}', fontsize=10)
            col += 1
        _draw_pg_on_ax(axes[row_i, col], pg_row, yscale, xscale, x_lim, colors)
        axes[row_i, col].set_title(f'LSP  |  varying: {key}', fontsize=10)
        all_results[key] = pg_row

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    return all_results

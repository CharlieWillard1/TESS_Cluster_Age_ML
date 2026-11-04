import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
from tinygp import GaussianProcess
from tinygp.kernels import quasisep
import time
import os
import pickle

from .gp_fit_result import GPFitResult

jax.config.update("jax_enable_x64", True)

_RN_Q = 1.0 / np.sqrt(2.0)  # fixed Q for red noise SHO (critical damping, Matérn-3/2 limit)

# ---------------------------------------------------------------------------
# Bounds and priors on the periodic SHO quality factor Q (docs/gp_fitting.tex S2, S4)
# ---------------------------------------------------------------------------
# The SHO kernel for Q > 1/2 is an envelope exp(-omega_0 tau / 2Q) times a cosine
# (Foreman-Mackey et al. 2017, eq. 23), so its e-folding decay time is
#
#     tau_d = 2 Q / omega_0 = Q * P / pi ,
#
# which celerite2 exposes directly as the (rho, tau) parameterisation of SHOTerm.
# Q / pi is the number of coherent cycles before the amplitude e-folds.
#
# UPPER bound -- a fixed decay-time cap, not a fixed Q cap.  Once tau_d exceeds the
# length of the light curve the likelihood is flat in Q: a component coherent for 27 d
# and one coherent for ever are indistinguishable in 27 d of data, and the optimiser
# drifts to whatever box edge it was given.  Nicholson & Aigrain (2022) show the
# recovered evolution timescale of a quasi-periodic GP tracks the truth only up to about
# half the baseline and scatters freely beyond it.  So the cap is on tau_d, and in Q it
# is period-dependent:
#
#     Q_max(P) = pi * TAU_CAP_DAYS / P .
#
# The equivalent frequency-domain statement is that the SHO line (FWHM ~ f0/Q) must
# not be narrower than the Rayleigh resolution 1/T, which gives Q < T/P -- the same
# scaling, a factor pi tighter.  TAU_CAP_DAYS is pinned to ONE TESS SECTOR for every
# row, including multi-sector rows (fit_multi_sho_gp uses min(TAU_CAP_DAYS, baseline)):
# a stitching offset between sectors can only be absorbed as an 80-day coherent signal
# if the box allows an 80-day decay time, and the Q feature should mean the same thing
# whatever the sector count.  Before 2026-11 the cap was a flat Q <= 500, which at
# P = 0.1 d cut below the baseline and at P = 5 d allowed decay times of 800 d; on the
# 2026-09-21 real table 38% of components sat on it and 43% had tau_d longer than their
# own light curve.
TAU_CAP_DAYS = 27.0

# LOWER bound -- physical.  An SHO's PSD has a peak at non-zero frequency only for
# Q > 1/sqrt(2); below that it is a continuum, not an oscillator.  Hard bound, = _RN_Q.
Q_MIN = _RN_Q

# SOFT floor -- where spot modulation lives.  Angus et al. (2018) argue the evolution
# timescale of a rotating spotted star should exceed one period (l > P), and the spot
# simulations of Aigrain et al. (2015) / Nicholson & Aigrain (2022) use lifetimes of
# 1-10 P.  In the SHO that is tau_d > P, i.e. Q > pi.  _q_neg_log_prior pushes
# quadratically in log Q below this threshold; it is the thing that stops a signal
# component from broadening into the red-noise continuum.  Was 2.0 (tau_d > 0.64 P).
Q_PRIOR_THRESHOLD = float(np.pi)

# Red-noise term models (see docs/fixing_RN_fitting.tex):
#   'sho'      : one fixed-Q SHO, theta = [log_sigma, log_period].  A broken power law
#                (flat, then f^-4); its effective slope carries no information on alpha.
#   'powerlaw' : fixed log-spaced basis of OU (quasisep.Exp) terms whose weights are tied
#                to alpha, theta = [ln S0, alpha].  S(f) ~ S0 * (f/f0)^-alpha in band.
RN_MODELS = ("sho", "powerlaw")
RN_PIVOT_FREQ = 1.0              # 1/day; S0 is the one-sided RN PSD at this frequency
RN_ALPHA_BOUNDS = (0.05, 1.95)   # the Lorentzian-sum construction only converges for 0<alpha<2
RN_ALPHA_WN_PRIOR_WEIGHT = 1.0   # strength of the alpha -> 0 prior in the white-noise limit
RN_ALPHA_WN_PRIOR_RATIO = 0.1    # RN/WN power ratio at which that prior is half-strength
RN_ALPHA_WN_PRIOR_WIDTH = 0.5    # decades over which it switches on
RN_WN_FLAG_RATIO = 0.05          # RN PSD / white-noise PSD at RN_PIVOT_FREQ below which the
                                 # power-law RN term is flagged near white noise (alpha undefined)


# ============================================================
# 1. Data prep
# ============================================================

def clean_lc(t, flux, flux_err):
    t = np.asarray(t, dtype=float)
    y = np.asarray(flux, dtype=float)
    yerr = np.asarray(flux_err, dtype=float)

    m = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    t, y, yerr = t[m], y[m], yerr[m]

    order = np.argsort(t)
    t, y, yerr = t[order], y[order], yerr[order]

    y = y - np.nanmedian(y)
    x = t - np.min(t)

    return t, x, y, yerr


# ============================================================
# 2. Multi-SHO GP model
# ============================================================

def sho_q_cap(period, tau_cap, Q_max=None):
    """Largest Q allowed at this period: the decay time Q P / pi may not exceed ``tau_cap``.

    ``Q_max`` is an optional absolute ceiling on top (None = none).  Works on floats,
    numpy and jax arrays.
    """
    cap = jnp.pi * tau_cap / period
    if Q_max is not None:
        cap = jnp.minimum(cap, Q_max)
    return cap


def _q_log_range(period, Q_min, tau_cap, Q_max=None):
    """(log Q_min, log Q_max(P)) with the top clamped just above the floor so the range
    never inverts (it would for P > pi tau_cap / Q_min, i.e. 120 d at the defaults)."""
    log_lo = jnp.log(Q_min)
    log_hi = jnp.log(jnp.maximum(sho_q_cap(period, tau_cap, Q_max), Q_min * (1.0 + 1e-6)))
    return log_lo, log_hi


def q_from_coherence(s, period, Q_min, tau_cap, Q_max=None):
    """Q from the coherence fraction ``s`` in [0, 1].

    L-BFGS-B only takes box bounds that are fixed before the fit starts, but both limits
    on Q move with the period (floor Q_min is fixed, ceiling pi tau_cap / P is not; in
    decay time it is the other way round).  So theta carries ``s``, where Q sits between
    its floor and its ceiling in log space::

        log Q = log Q_min + s * (log Q_max(P) - log Q_min) .

    s = 0 is the continuum limit, s = 1 is "coherent for the whole cap", and both are
    exact hard walls because the box on s is [0, 1].  The derived Q is what the kernel,
    the soft prior and the reported features all use; nothing downstream sees s.
    """
    log_lo, log_hi = _q_log_range(period, Q_min, tau_cap, Q_max)
    return jnp.exp(log_lo + s * (log_hi - log_lo))


def coherence_from_q(Q, period, Q_min, tau_cap, Q_max=None):
    """Inverse of q_from_coherence (numpy; used to seed s from a Q guess)."""
    log_lo, log_hi = _q_log_range(np.asarray(period, float), Q_min, tau_cap, Q_max)
    log_lo, log_hi = float(log_lo), float(log_hi)
    return (np.log(Q) - log_lo) / (log_hi - log_lo)


def unpack_theta(theta, n_components, tau_cap=None, Q_min=Q_MIN, Q_max=None):
    """
    theta structure:
    [log_sigma_1, log_period_1, q_1,
     log_sigma_2, log_period_2, q_2,
     ...
     log_jitter]

    The third slot of each component is the coherence fraction s in [0, 1] when
    ``tau_cap`` is given (see q_from_coherence), and log Q when it is None.  The None
    form is the pre-2026-11 layout; it is kept so that pickled fits made with a flat
    Q box still unpack correctly (they carry no ``tau_cap`` key).

    Each component dict holds sigma, period, omega, Q, and also tau (= Q P / pi, the
    e-folding decay time of the kernel in days), Q_cap (the ceiling at that period, NaN
    in the legacy layout) and coherence (s, NaN in the legacy layout).
    """
    theta = jnp.asarray(theta)
    comps = []

    for i in range(n_components):
        log_sigma = theta[3*i + 0]
        log_period = theta[3*i + 1]
        q_raw = theta[3*i + 2]

        sigma = jnp.exp(log_sigma)
        period = jnp.exp(log_period)
        if tau_cap is None:
            Q = jnp.exp(q_raw)
            q_cap = jnp.nan
            s = jnp.nan
        else:
            Q = q_from_coherence(q_raw, period, Q_min, tau_cap, Q_max)
            q_cap = sho_q_cap(period, tau_cap, Q_max)
            s = q_raw
        omega = 2.0 * jnp.pi / period

        comps.append({
            "sigma": sigma,
            "period": period,
            "omega": omega,
            "Q": Q,
            "tau": Q * period / jnp.pi,
            "Q_cap": q_cap,
            "coherence": s,
        })

    jitter = jnp.exp(theta[-1])
    return comps, jitter


def _unpack_rn_comp(theta, n_components):
    """Unpack the fixed-Q red noise SHO from theta[3*n : 3*n+2]."""
    sigma  = jnp.exp(theta[3 * n_components])
    period = jnp.exp(theta[3 * n_components + 1])
    return {"sigma": sigma, "period": period, "omega": 2.0 * jnp.pi / period, "Q": _RN_Q}


def _unpack_rn_powerlaw(theta, n_components):
    """Unpack the power-law red-noise parameters from theta[3*n : 3*n+2]."""
    log_S0 = theta[3 * n_components]
    alpha  = theta[3 * n_components + 1]
    return {"S0": jnp.exp(log_S0), "log_S0": log_S0, "alpha": alpha}


def rn_basis_frequencies(band, pad_decades=1.2, per_decade=2.0):
    """Fixed log-spaced break frequencies (1/day) for the power-law red-noise basis.

    The basis extends ``pad_decades`` beyond each edge of ``band``.  Accuracy is set by
    truncation at the basis edges, not by density: at 1.2 dec of padding, 2 terms per
    decade is as good as 6, while removing the padding raises the in-band deviation from
    a pure power law several-fold.  For band = (0.1, 12) this gives 9 terms.
    """
    lo = np.log10(band[0]) - pad_decades
    hi = np.log10(band[1]) + pad_decades
    n = max(3, int(round(per_decade * (hi - lo))))
    return np.logspace(lo, hi, n)


_EXPOSURE_ALPHA_GRID = np.round(np.arange(0.0, 2.0001, 0.025), 3)
_exposure_table_cache = {}


def exposure_weight_table(fj, exposure_days):
    """Log-weight corrections u_j(alpha) making the OU basis match an exposure-averaged power law.

    Every point in the light curve is an average over ``exposure_days`` (30 min after
    ``resample_lc``), so the data see ``S(f) sinc^2(f E)``, not ``S(f)``.  A GP whose kernel
    ignores that filter fits alpha steep (about +0.3 on the synthetic set).  Here the
    basis weights ``c_j = f_j^-alpha * exp(u_j)`` are fitted, once per (basis, exposure), so
    the summed OU spectrum matches the FILTERED power law for each alpha on a 0.025 grid:

    * log residual on 0.033 d^-1 .. Nyquist (1/2E), weight 1;
    * above Nyquist, the mean of the sinc^2 envelope, weight 0.1 (keeps basis power that
      would alias back into the band from growing);
    * ridge 0.1 on u, warm-started along alpha, so u varies smoothly (median step 0.03).

    Accuracy for E = 30 min, basis 0.1-12 d^-1: 0.02-0.05 dex over 0.1-12 d^-1 at every
    alpha, worst at Nyquist (0.06-0.13 dex for alpha <= 1.5, 0.27 at 1.9).  The kernel stays
    a sum of OU terms, so fits cost the same.

    Returns (alpha_grid, U) with U of shape (len(alpha_grid), len(fj)).
    """
    from scipy.optimize import least_squares
    fj = np.asarray(fj, dtype=float)
    key = (tuple(np.round(fj, 12)), round(float(exposure_days), 12))
    if key in _exposure_table_cache:
        return _exposure_table_cache[key]
    E = float(exposure_days)
    nyq = 0.5 / E
    f_in = np.logspace(np.log10(fj.min() * 5.0), np.log10(nyq), 300)
    f_hi = np.logspace(np.log10(nyq), np.log10(10.0 * nyq), 60)[1:]
    L_in = 1.0 / (1.0 + (f_in[None, :] / fj[:, None]) ** 2)
    L_hi = 1.0 / (1.0 + (f_hi[None, :] / fj[:, None]) ** 2)
    U, u = [], np.zeros(len(fj))
    for a in _EXPOSURE_ALPHA_GRID:
        T_in = f_in ** (-a) * np.sinc(f_in * E) ** 2
        T_hi = f_hi ** (-a) * 0.5 / (np.pi * f_hi * E) ** 2
        w0 = fj ** (-a)

        def resid(u, T_in=T_in, T_hi=T_hi, w0=w0):
            w = w0 * np.exp(u)
            return np.concatenate([np.log(w @ L_in / T_in),
                                   0.1 * np.log(w @ L_hi / T_hi),
                                   0.1 * u])
        u = least_squares(resid, u, method="lm").x
        U.append(u)
    out = (_EXPOSURE_ALPHA_GRID, np.array(U))
    _exposure_table_cache[key] = out
    return out


def _powerlaw_rn_sigmas(fj, S0, alpha, f0=RN_PIVOT_FREQ, exposure_days=None):
    """Per-term OU amplitudes sigma_j for the power-law basis (JAX-differentiable).

    With ``exposure_days`` set, the weights come from ``exposure_weight_table`` instead:
    the summed PSD then approximates ``S0 (f/f0)^-alpha sinc^2(f E)``, i.e. S0 and alpha
    describe the INTRINSIC spectrum and the kernel carries the exposure filter.

    Each ``quasisep.Exp(scale=1/(2 pi f_j), sigma_j)`` has one-sided PSD
    ``c_j / (1 + (f/f_j)^2)`` with ``c_j = 2 sigma_j^2 / (pi f_j)``.  Choosing
    ``c_j = K f_j^-alpha`` makes the sum ~ f^-alpha (continuum limit, 0 < alpha < 2), and K
    is fixed so that the summed PSD equals S0 at f0.  Hence
    ``sigma_j^2 = (pi/2) K f_j^(1-alpha)``.
    """
    if exposure_days is not None:
        grid, U = exposure_weight_table(fj, exposure_days)
        grid_j, U_j = jnp.asarray(grid), jnp.asarray(U)
        u = jnp.stack([jnp.interp(alpha, grid_j, U_j[:, j]) for j in range(U.shape[1])])
        fj = jnp.asarray(fj)
        c = S0 * f0 ** alpha * fj ** (-alpha) * jnp.exp(u)
        return jnp.sqrt(0.5 * jnp.pi * c * fj)
    fj = jnp.asarray(fj)
    shape = fj ** (-alpha) / (1.0 + (f0 / fj) ** 2)
    K = S0 / jnp.sum(shape)
    return jnp.sqrt(0.5 * jnp.pi * K * fj ** (1.0 - alpha))


def powerlaw_rn_psd(freq, fj, S0, alpha, f0=RN_PIVOT_FREQ, exposure_days=None):
    """One-sided analytic PSD of the power-law red-noise basis (same convention as ``sho_psd``).

    This is the kernel's spectrum; with ``exposure_days`` it includes the exposure filter.
    """
    freq = np.asarray(freq, dtype=float)
    fj = np.asarray(fj, dtype=float)
    sig2 = np.asarray(_powerlaw_rn_sigmas(fj, float(S0), float(alpha), f0, exposure_days)) ** 2
    c = 2.0 * sig2 / (np.pi * fj)
    return np.sum(c[:, None] / (1.0 + (freq[None, :] / fj[:, None]) ** 2), axis=0)


def rn_psd_from_fit(freq, fit):
    """Analytic PSD of a fit's red-noise term, whichever model it uses (zeros if none)."""
    freq = np.asarray(freq, dtype=float)
    if not fit.get("has_red_noise", False):
        return np.zeros_like(freq)
    theta = jnp.asarray(fit["theta"])
    n = fit["n_components"]
    if fit.get("rn_model", "sho") == "powerlaw":
        rn = _unpack_rn_powerlaw(theta, n)
        return powerlaw_rn_psd(freq, fit["rn_basis_freqs"], float(rn["S0"]), float(rn["alpha"]),
                               exposure_days=fit.get("rn_exposure_days"))
    rn = _unpack_rn_comp(theta, n)
    return sho_psd(freq, float(rn["sigma"]), float(rn["omega"]), float(rn["Q"]))


def rn_white_noise_features(fit, x, yerr, flag_ratio=RN_WN_FLAG_RATIO):
    """How far the power-law red-noise term rises above the white noise.

    ``rn_wn_ratio`` is the RN PSD at RN_PIVOT_FREQ over the one-sided white-noise PSD,
    ``2 * median(yerr^2 + jitter^2) * dt`` with dt the median time step.

    When the data hold no detectable red noise, the jitter already describes the white
    noise, so the fit drives the RN amplitude to zero rather than alpha to 0 -- and alpha
    then floats, often to its upper bound.  On 60 real single-sector rows, every fit with
    ratio < 0.05 had an identical likelihood with alpha free or fixed at 1.  Such fits get
    ``rn_near_white_noise = True``: alpha is undefined there and should not be read as steep.

    Returns {} for fits without a power-law RN term.
    """
    if not fit.get("has_red_noise", False) or fit.get("rn_model") != "powerlaw":
        return {}
    theta = jnp.asarray(fit["theta"])
    rn = _unpack_rn_powerlaw(theta, fit["n_components"])
    jitter = float(jnp.exp(theta[-1]))
    dt = float(np.median(np.diff(np.asarray(x, dtype=float))))
    wn_psd = 2.0 * float(np.median(np.asarray(yerr, dtype=float) ** 2 + jitter ** 2)) * dt
    ratio = float(rn["S0"]) / wn_psd if wn_psd > 0 else np.inf
    return {"rn_wn_ratio": ratio, "rn_near_white_noise": bool(ratio < flag_ratio)}


def sho_psd(freq, sigma, omega0, Q):
    """One-sided analytic PSD of a ``tinygp.kernels.quasisep.SHO`` component.

    Single source of truth for the SHO power spectrum.  Lives next to
    ``build_multi_sho_gp`` so the kernel and its spectrum cannot drift apart::

                            4 * sigma^2 * omega0^3
        S(f) = -------------------------------------------------------
                Q * ( (w^2 - omega0^2)^2 + w^2 * omega0^2 / Q^2 )

    with ``w = 2*pi*freq``.  Asymptotically ``S ~ f^-4``, matching the kernel.

    Normalisation is fixed by the kernel itself: ``sigma`` is the process standard
    deviation (``k(0) == sigma^2``), so the one-sided PSD integrates back to the
    variance, ``integral_0^inf S(f) df == sigma^2``.  The prefactor was obtained by
    solving that condition numerically; ``A * Q / period == 2/pi`` is universal in Q.

    Historical note: this previously carried ``(omega0^2 + w^2)`` in the numerator,
    which falls as ``f^-2`` and disagreed with the fitted kernel by up to ~2.2 dex
    inside the 0.1-12 /day band.  The fits themselves were never affected -- the
    likelihood uses the real kernel -- only the reported spectra.

    Parameters
    ----------
    freq : array-like
        Frequencies in 1/day.
    sigma : float
        Component amplitude; the standard deviation of that component.
    omega0 : float
        Angular frequency in rad/day (``2*pi/period``).
    Q : float
        Quality factor.

    Returns
    -------
    ndarray
        PSD in units of ``sigma^2 * day``.
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    numer = 4.0 * sigma ** 2 * omega0 ** 3
    denom = Q * ((omega ** 2 - omega0 ** 2) ** 2 + omega ** 2 * omega0 ** 2 / Q ** 2)
    return numer / denom


def build_multi_sho_gp(theta, x, yerr, n_components, has_red_noise=False,
                       rn_model="sho", rn_basis_freqs=None, rn_exposure_days=None,
                       tau_cap=None, Q_min=Q_MIN, Q_max=None):
    comps, jitter = unpack_theta(theta, n_components, tau_cap=tau_cap, Q_min=Q_min, Q_max=Q_max)

    kernel = None

    for comp in comps:
        k = quasisep.SHO(
            omega=comp["omega"],
            quality=comp["Q"],
            sigma=comp["sigma"],
        )

        kernel = k if kernel is None else kernel + k

    if has_red_noise:
        if rn_model == "powerlaw":
            rn = _unpack_rn_powerlaw(theta, n_components)
            sig = _powerlaw_rn_sigmas(rn_basis_freqs, rn["S0"], rn["alpha"],
                                      exposure_days=rn_exposure_days)
            for j, f_j in enumerate(np.asarray(rn_basis_freqs)):
                k = quasisep.Exp(scale=1.0 / (2.0 * np.pi * float(f_j)), sigma=sig[j])
                kernel = k if kernel is None else kernel + k
        else:
            rn = _unpack_rn_comp(theta, n_components)
            k = quasisep.SHO(omega=rn["omega"], quality=_RN_Q, sigma=rn["sigma"])
            kernel = k if kernel is None else kernel + k

    diag = yerr**2 + jitter**2

    return GaussianProcess(kernel, x, diag=diag, mean=0.0)


def neg_log_likelihood(theta, x, y, yerr, n_components, has_red_noise=False,
                       rn_model="sho", rn_basis_freqs=None, rn_exposure_days=None,
                       tau_cap=None, Q_min=Q_MIN, Q_max=None):
    gp = build_multi_sho_gp(theta, x, yerr, n_components, has_red_noise,
                            rn_model, rn_basis_freqs, rn_exposure_days,
                            tau_cap=tau_cap, Q_min=Q_min, Q_max=Q_max)
    return -gp.log_probability(y)


def _period_neg_log_prior(theta, period_seeds_j, n_components, sigma_dex):
    """Non-negative penalty: negative log of a log-normal prior on each periodic SHO period.

    Returned value is added to the NLL to form the negative log-posterior.
    Covers only the n_components periodic SHOs; RN period is excluded.
    """
    sigma_ln    = sigma_dex * jnp.log(10.0)  # dex → natural log
    log_seeds   = jnp.log(period_seeds_j)
    log_periods = theta[:3 * n_components].reshape(n_components, 3)[:, 1]
    return 0.5 * jnp.sum(((log_periods - log_seeds) / sigma_ln) ** 2)


def _q_neg_log_prior(theta, n_components, q_prior_weight, q_threshold=Q_PRIOR_THRESHOLD,
                     tau_cap=None, Q_min=Q_MIN, Q_max=None):
    """Half-quadratic penalty in log-Q space that activates below ``q_threshold``.

    No penalty above the threshold (uniform prior).  Below it the penalty grows
    quadratically in log Q, so with the default threshold of pi (decay time of one
    period -- see Q_PRIOR_THRESHOLD) Q = 2 is a small nudge, Q = 1 a moderate push and
    Q < 1 a strong pull back up.  Q is the DERIVED quality factor (unpack_theta), so this
    works identically in the coherence-fraction and legacy log-Q layouts.
    """
    comps, _ = unpack_theta(theta, n_components, tau_cap=tau_cap, Q_min=Q_min, Q_max=Q_max)
    log_q_threshold = jnp.log(q_threshold)
    total = jnp.zeros(())
    for c in comps:
        deficit = jnp.maximum(0.0, log_q_threshold - jnp.log(c["Q"]))
        total = total + deficit ** 2
    return q_prior_weight * total


def _rn_alpha_wn_neg_log_prior(theta, n_components, med_yerr_sq, dt, weight, ratio0, width):
    """Pull the red-noise slope to 0 only where the red noise is lost in the white noise.

    A red-noise term with alpha = 0 is a flat spectrum, which is white noise -- and the
    kernel already has a jitter term for that.  The two are degenerate, so when the data
    hold no red noise the fit drives the RN AMPLITUDE to zero instead of alpha, leaving
    alpha unconstrained: on the real table 33% of rows end that way, with a median alpha of
    1.76 and half of them piled at the upper bound (docs/fixing_RN_fitting.tex S10).

    This penalty is ``weight * alpha^2 * w(ratio)``, where ``ratio = S0 / WN_PSD`` measures how
    far the red noise rises above the white noise at the pivot frequency and

        w = sigmoid(-log10(ratio / ratio0) / width)

    goes to 1 when ratio << ratio0 and to 0 when ratio >> ratio0.  So a row with real red
    noise is fitted exactly as before, and a row without it reports the physical limit
    (flat) instead of an arbitrary slope.  It adds no information: those rows still have no
    measurable slope, and ``rn_wn_ratio`` / ``rn_near_white_noise`` still say so.
    """
    rn = _unpack_rn_powerlaw(theta, n_components)
    jitter_sq = jnp.exp(2.0 * theta[-1])
    wn_psd = 2.0 * (med_yerr_sq + jitter_sq) * dt
    ratio = rn["S0"] / wn_psd
    w = jax.nn.sigmoid(-jnp.log10(ratio / ratio0) / width)
    return weight * rn["alpha"] ** 2 * w


# ============================================================
# 3. Fit GP
# ============================================================

def make_initial_theta(period_guesses, y, yerr, init_Q=5.0,
                       has_red_noise=False, rn_max_period=10.0,
                       rn_model="sho", rn_band=None, rn_alpha_init=1.0,
                       tau_cap=None, Q_min=Q_MIN, Q_max=None):
    n_components = len(period_guesses)

    y_std = np.nanstd(y)
    med_err = np.nanmedian(yerr)

    pieces = []

    # Conservative amplitude initialization for low-SNR peaks.
    # Total GP RMS is roughly amp_frac * y_std, split across components.
    amp_frac = 0.3 #was originaly 1 which was high?
    init_sigma = amp_frac * y_std / np.sqrt(max(n_components, 1))
    init_sigma = max(init_sigma, 1e-8)

    for p in period_guesses:
        if tau_cap is None:
            q_seed = np.log(init_Q)
        else:
            # Seed the coherence fraction at the Q guess, clipped strictly inside [0, 1]
            # so the first L-BFGS-B step is not degenerate (init_Q may exceed the cap at
            # long periods: at P = 10 d the cap is 8.5 for a 27 d tau_cap).
            q_seed = float(np.clip(coherence_from_q(init_Q, p, Q_min, tau_cap, Q_max),
                                   0.05, 0.95))
        pieces.extend([
            np.log(init_sigma),
            np.log(p),
            q_seed,
        ])

    if has_red_noise and rn_model == "powerlaw":
        # Seed alpha = rn_alpha_init with in-band RN variance (amp_frac * y_std)^2.  For
        # S = S0 (f/f0)^-a the band integral is S0 f0^a * int f^-a df, which fixes S0.
        f_lo, f_hi = rn_band
        a0 = float(rn_alpha_init)
        band_var = max((amp_frac * y_std) ** 2, 1e-16)
        if abs(a0 - 1.0) < 1e-6:
            shape = np.log(f_hi / f_lo)
        else:
            shape = (f_hi ** (1.0 - a0) - f_lo ** (1.0 - a0)) / (1.0 - a0)
        S0 = band_var / (RN_PIVOT_FREQ ** a0 * shape)
        pieces.extend([np.log(S0), a0])
    elif has_red_noise:
        rn_sigma = max(init_sigma, 1e-8)
        # Seed at 0.5 * rn_max_period (interior of bound; avoids degenerate first L-BFGS-B step)
        rn_period = 0.5 * rn_max_period
        pieces.extend([np.log(rn_sigma), np.log(rn_period)])

    pieces.append(np.log(max(med_err, 1e-8)))  # jitter always last

    return np.array(pieces, dtype=float)


def make_bounds(n_components, min_period, max_period, Q_min=Q_MIN, Q_max=None,
                has_red_noise=False, rn_max_period=None, rn_model="sho", tau_cap=None):
    """Box bounds for L-BFGS-B.

    With ``tau_cap`` set the third slot of each component is the coherence fraction s and
    its box is [0, 1]; the period-dependent Q limits are then exact by construction (see
    q_from_coherence).  With ``tau_cap=None`` (legacy layout) the slot is log Q and the
    box is [Q_min, Q_max], Q_max defaulting to 100 as before.
    """
    bounds = []

    if tau_cap is None:
        q_bounds = (np.log(Q_min), np.log(100.0 if Q_max is None else Q_max))
    else:
        q_bounds = (0.0, 1.0)

    for _ in range(n_components):
        bounds.extend([
            (np.log(1e-10), np.log(1.0)),              # sigma
            (np.log(min_period), np.log(max_period)),  # period
            q_bounds,                                  # coherence fraction s (or log Q, legacy)
        ])

    if has_red_noise and rn_model == "powerlaw":
        bounds.extend([
            (np.log(1e-20), np.log(1.0)),              # ln S0  (fractional flux^2 * day)
            RN_ALPHA_BOUNDS,                           # alpha  (linear, not log)
        ])
    elif has_red_noise:
        # RN period range extends to the full time baseline so it can absorb
        # slow trends that would otherwise leak into the periodic SHOs.
        rn_per_max = rn_max_period if rn_max_period is not None else max_period
        bounds.extend([
            (np.log(1e-10), np.log(1.0)),              # sigma_rn
            (np.log(min_period), np.log(rn_per_max)),  # period_rn
            # Q is fixed at _RN_Q — not in theta
        ])

    bounds.append((np.log(1e-10), np.log(1.0)))  # jitter always last

    return bounds


def fit_multi_sho_gp(
    x,
    y,
    yerr,
    period_guesses,
    min_period,
    max_period,
    Q_min=Q_MIN,
    Q_max=None,
    init_Q=10.0,
    tau_cap_days=TAU_CAP_DAYS,
    q_prior_threshold=Q_PRIOR_THRESHOLD,
    has_red_noise=False,
    use_lognormal_period_prior=False,
    period_prior_sigma_dex=0.3,
    two_phase_fitting=False,
    use_q_prior=False,
    q_prior_weight=1.0,
    rn_period_cap=10.0,
    rn_model="powerlaw",
    rn_pad_decades=1.2,
    rn_per_decade=2.0,
    rn_alpha_init=1.0,
    rn_alpha_release_last=False,
    rn_alpha_fixed=False,
    rn_exposure_days=None,
    rn_alpha_wn_prior_weight=RN_ALPHA_WN_PRIOR_WEIGHT,
    rn_alpha_wn_prior_ratio=RN_ALPHA_WN_PRIOR_RATIO,
    rn_alpha_wn_prior_width=RN_ALPHA_WN_PRIOR_WIDTH,
):
    """Fit one m-component SHO (+ red noise + jitter) GP by staged L-BFGS-B.

    Q bounds (see the TAU_CAP_DAYS / Q_MIN / Q_PRIOR_THRESHOLD notes at the top of the
    module):

    tau_cap_days : float or None
        Hard ceiling on each component's e-folding decay time tau_d = Q P / pi, so the
        ceiling in Q is pi * tau_cap / P.  The cap actually applied is
        ``min(tau_cap_days, baseline)`` and is stored in the returned fit as ``tau_cap``;
        with the default 27 d it is one TESS sector for every row, however many sectors
        the row stitches together.  None restores the pre-2026-11 flat box in log Q with
        ``Q_max`` as its top (default 100 then).
    Q_min : float
        Hard floor on Q (default 1/sqrt(2), where the SHO PSD stops having a peak).
    Q_max : float or None
        Optional absolute ceiling on Q applied in addition to the decay-time cap.
    q_prior_threshold : float
        Where the soft Q prior (``use_q_prior``) starts pushing, default pi = one period
        of decay time.
    """
    if rn_model not in RN_MODELS:
        raise ValueError(f"rn_model must be one of {RN_MODELS}, got {rn_model!r}")
    n_components = len(period_guesses)
    _baseline = float(np.max(x))
    rn_max_period = min(_baseline, rn_period_cap) if rn_period_cap is not None else _baseline

    # Decay-time cap for the periodic components: one sector, or the row if shorter.
    tau_cap = None if tau_cap_days is None else float(min(tau_cap_days, _baseline))
    if tau_cap is not None and np.pi * tau_cap / max_period <= Q_min:
        import warnings
        warnings.warn(
            f"tau_cap={tau_cap:.1f} d gives Q_max(P={max_period:.2f} d) = "
            f"{np.pi * tau_cap / max_period:.2f} <= Q_min={Q_min:.3f}; the Q range is "
            "pinned at Q_min for the longest periods.", RuntimeWarning, stacklevel=2)
    q_kw = dict(tau_cap=tau_cap, Q_min=Q_min, Q_max=Q_max)

    # The power-law basis spans the fitted period range, padded (see rn_basis_frequencies).
    rn_band = (1.0 / max_period, 1.0 / min_period)
    rn_basis_freqs = (rn_basis_frequencies(rn_band, rn_pad_decades, rn_per_decade)
                      if (has_red_noise and rn_model == "powerlaw") else None)
    if rn_basis_freqs is None:
        rn_exposure_days = None

    theta0 = make_initial_theta(
        period_guesses, y, yerr, init_Q=init_Q,
        has_red_noise=has_red_noise, rn_max_period=rn_max_period,
        rn_model=rn_model, rn_band=rn_band,
        # keep the seed strictly inside the bounds so the first L-BFGS-B step is not degenerate
        rn_alpha_init=float(np.clip(rn_alpha_init, RN_ALPHA_BOUNDS[0] + 0.01,
                                    RN_ALPHA_BOUNDS[1] - 0.01)),
        **q_kw,
    )
    bounds = make_bounds(
        n_components,
        min_period=min_period,
        max_period=max_period,
        Q_min=Q_min,
        Q_max=Q_max,
        has_red_noise=has_red_noise,
        rn_max_period=rn_max_period,
        rn_model=rn_model,
        tau_cap=tau_cap,
    )

    # Convert data to JAX arrays ONCE, not inside every likelihood call
    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)
    yerr_j = jnp.asarray(yerr)

    # JIT-compiled objective.  Which penalties apply is decided at Python level (not inside
    # JIT), so each compiled function has a fixed structure.
    _use_period_prior = use_lognormal_period_prior and n_components > 0
    period_seeds_j = jnp.asarray(period_guesses) if _use_period_prior else None
    _use_rn_alpha_wn_prior = (rn_alpha_wn_prior_weight > 0 and has_red_noise
                              and rn_model == "powerlaw" and not rn_alpha_fixed)
    if _use_rn_alpha_wn_prior:
        _med_yerr_sq = float(np.median(np.asarray(yerr, dtype=float) ** 2))
        _dt = float(np.median(np.diff(np.asarray(x, dtype=float))))

    penalties = []
    if _use_period_prior:
        penalties.append(lambda th: _period_neg_log_prior(th, period_seeds_j, n_components,
                                                          period_prior_sigma_dex))
    if use_q_prior:
        penalties.append(lambda th: _q_neg_log_prior(th, n_components, q_prior_weight,
                                                     q_prior_threshold, **q_kw))
    if _use_rn_alpha_wn_prior:
        penalties.append(lambda th: _rn_alpha_wn_neg_log_prior(
            th, n_components, _med_yerr_sq, _dt, rn_alpha_wn_prior_weight,
            rn_alpha_wn_prior_ratio, rn_alpha_wn_prior_width))

    @jax.jit
    def objective(theta):
        total = neg_log_likelihood(theta, x_j, y_j, yerr_j, n_components, has_red_noise,
                                   rn_model, rn_basis_freqs, rn_exposure_days, **q_kw)
        for pen in penalties:
            total = total + pen(theta)
        return total

    # JIT-compiled value + gradient
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    # scipy-compatible wrapper
    def scipy_objective(theta_np):
        val, grad = value_and_grad(jnp.asarray(theta_np))
        return float(val), np.asarray(grad, dtype=float)

    opt_options = {"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6, "maxls": 20}

    # Stage list.  Each stage warm-starts from the previous result and is kept only if
    # it does not worsen the objective (the previous solution is always feasible for the
    # next stage, so a worse result means the optimiser escaped the basin).
    #   two_phase_fitting     : stage "periods frozen" before the full fit.
    #   rn_alpha_release_last : alpha is held at its seed in every stage, then released
    #                           in a final stage.  The result is then never worse than the
    #                           fixed-alpha fit.  Without it, on real rows the free-alpha
    #                           fit ended below the alpha=1 fit in 4/30 random and 13/30
    #                           steep rows (docs/fixing_RN_fitting.tex).
    alpha_idx = 3 * n_components + 1
    powerlaw_rn = has_red_noise and rn_model == "powerlaw"
    hold_alpha = (rn_alpha_release_last or rn_alpha_fixed) and powerlaw_rn

    def _stage_bounds(freeze_periods, freeze_alpha):
        b = list(bounds)
        if freeze_periods:
            for i, p_seed in enumerate(period_guesses):
                log_p = np.log(p_seed)
                b[3 * i + 1] = (log_p, log_p)
        if freeze_alpha:
            b[alpha_idx] = (theta0[alpha_idx], theta0[alpha_idx])
        return b

    stages = []
    if two_phase_fitting and n_components > 0:
        # Pass 1: freeze each periodic SHO period to its LSP seed so that σ, Q,
        # the red-noise term, and jitter can find their level while the period is
        # pinned.  This "switches on" the component so the likelihood has a sharp
        # minimum at the correct period by the time we release it in pass 2.
        stages.append(("pass-1", _stage_bounds(True, hold_alpha)))
    stages.append(("pass-2" if stages else "fit", _stage_bounds(False, hold_alpha)))
    if hold_alpha and not rn_alpha_fixed:
        stages.append(("alpha-release", _stage_bounds(False, False)))

    result, x_start, val_prev = None, theta0, None
    for label, stage_bounds in stages:
        res = minimize(
            scipy_objective, x_start,
            method="L-BFGS-B", jac=True,
            bounds=stage_bounds, options=opt_options,
        )
        val, _ = scipy_objective(res.x)
        if result is not None and val > val_prev:
            import warnings
            warnings.warn(
                f"staged fitting: {label} objective ({val:.4f}) is worse than the "
                f"previous stage ({val_prev:.4f}). Keeping the previous result.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        result, x_start, val_prev = res, res.x, val

    theta_best = result.x

    gp = build_multi_sho_gp(
        jnp.asarray(theta_best),
        x_j,
        yerr_j,
        n_components,
        has_red_noise,
        rn_model,
        rn_basis_freqs,
        rn_exposure_days,
        **q_kw,
    )

    # BIC uses the raw log-likelihood (prior excluded by convention — we rank
    # models with an MLE-flavored criterion, not a true posterior comparison).
    loglike = float(gp.log_probability(y_j))

    return {
        "theta": theta_best,
        "gp": gp,
        "result": result,
        "loglike": loglike,
        "n_components": n_components,
        "has_red_noise": has_red_noise,
        "rn_model": rn_model if has_red_noise else None,
        "rn_basis_freqs": rn_basis_freqs,
        "rn_band": rn_band,
        "rn_exposure_days": rn_exposure_days,
        # Q layout of theta: tau_cap None = legacy log-Q slot; else coherence fraction.
        # Everything that unpacks theta later (PSD, plots, unpickling) reads these.
        "tau_cap": tau_cap,
        "Q_min": Q_min,
        "Q_max": Q_max,
    }


def q_layout_kwargs(fit):
    """The unpack_theta / build_multi_sho_gp keywords recorded in a fit dict.

    Fits pickled before 2026-11 carry none of the keys and unpack as the legacy log-Q
    layout (``tau_cap=None``).
    """
    return dict(tau_cap=fit.get("tau_cap"), Q_min=fit.get("Q_min", Q_MIN),
                Q_max=fit.get("Q_max"))


# ============================================================
# 4. Prediction and residuals
# ============================================================

def predict_gp_at_training_times(fit, x, y):
    cond_gp = fit["gp"].condition(jnp.asarray(y), jnp.asarray(x)).gp
    mu = np.asarray(cond_gp.mean)
    std = np.asarray(jnp.sqrt(cond_gp.variance))
    return mu, std


def compute_gp_residuals(fit, x, y):
    mu, std = predict_gp_at_training_times(fit, x, y)
    resid = y - mu
    return resid, mu, std


# ============================================================
# 5. LSP (single peak or full spectrum)
# ============================================================

def residual_lsp_peak(
    t,
    residual,
    *,
    flux_err=None,
    min_period,
    max_period,
    samples_per_peak=10,
):
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period

    if flux_err is None:
        ls = LombScargle(t, residual)
    else:
        ls = LombScargle(t, residual, flux_err)

    freq, power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=samples_per_peak,
    )

    idx = np.nanargmax(power)

    best_freq = freq[idx]
    best_period = 1.0 / best_freq
    best_power = power[idx]

    try:
        fap = ls.false_alarm_probability(best_power)
    except Exception:
        fap = np.nan

    return {
        "freq": freq,
        "power": power,
        "best_freq": float(best_freq),
        "best_period": float(best_period),
        "best_power": float(best_power),
        "fap": float(fap),
    }


# ============================================================
# 6. White-noise level estimation
# ============================================================

def estimate_lsp_white_noise_level(
    t,
    flux_err,
    min_period,
    max_period,
    samples_per_peak=10,
    n_boot=200,
    percentile=99.0,
    random_state=123,
):
    """
    Estimates a white-noise LSP peak threshold via bootstrap.

    Generates pure Gaussian noise realizations using the measured flux errors,
    computes the maximum LSP power for each, and returns a high percentile of
    those maxima.  Peaks above this threshold are treated as significant.
    """
    rng = np.random.default_rng(random_state)

    t = np.asarray(t)
    flux_err = np.asarray(flux_err)

    max_powers = []

    for _ in range(n_boot):
        noise = rng.normal(0.0, flux_err)

        lsp = residual_lsp_peak(
            t,
            noise,
            flux_err=flux_err,
            min_period=min_period,
            max_period=max_period,
            samples_per_peak=samples_per_peak,
        )

        max_powers.append(lsp["best_power"])

    max_powers = np.asarray(max_powers)

    return {
        "white_noise_power_threshold": float(np.nanpercentile(max_powers, percentile)),
        "white_noise_max_powers": max_powers,
        "percentile": percentile,
    }


# ============================================================
# 7. Model comparison
# ============================================================

def compute_bic(loglike, n_params, n_data):
    return n_params * np.log(n_data) - 2.0 * loglike


def summarize_components(theta, n_components, has_red_noise=False,
                         rn_model="sho", rn_basis_freqs=None, rn_band=None,
                         rn_exposure_days=None, tau_cap=None, Q_min=Q_MIN, Q_max=None):
    comps, jitter = unpack_theta(jnp.asarray(theta), n_components,
                                 tau_cap=tau_cap, Q_min=Q_min, Q_max=Q_max)

    out = {}

    for i, c in enumerate(comps):
        out[f"sho_{i+1}_sigma"] = float(c["sigma"])
        out[f"sho_{i+1}_period_days"] = float(c["period"])
        out[f"sho_{i+1}_omega"] = float(c["omega"])
        out[f"sho_{i+1}_Q"] = float(c["Q"])
        out[f"sho_{i+1}_tau_days"] = float(c["tau"])         # e-folding decay time Q P / pi
        out[f"sho_{i+1}_Q_cap"] = float(c["Q_cap"])          # ceiling at this period (NaN: legacy)
        out[f"sho_{i+1}_coherence"] = float(c["coherence"])  # s in [0, 1]; 1 = at the cap

    if tau_cap is not None:
        out["sho_tau_cap_days"] = float(tau_cap)

    if has_red_noise and rn_model == "powerlaw":
        rn = _unpack_rn_powerlaw(jnp.asarray(theta), n_components)
        out["rn_model"]    = "powerlaw"
        out["rn_alpha"]    = float(rn["alpha"])
        out["rn_log10_S0"] = float(rn["log_S0"]) / np.log(10.0)   # PSD at RN_PIVOT_FREQ
        # In-band RMS of the RN term: sqrt(integral of its PSD over the fitted band).
        f = np.linspace(rn_band[0], rn_band[1], 4000)
        S = powerlaw_rn_psd(f, rn_basis_freqs, float(rn["S0"]), float(rn["alpha"]),
                            exposure_days=rn_exposure_days)
        out["rn_rms_band"] = float(np.sqrt(np.trapezoid(S, f)))
        # Least-squares log-log slope of the RN PSD over the band.  The basis compresses
        # the in-band slope toward 1 at the alpha extremes (alpha=0.25 -> 0.33), so THIS is
        # the number comparable to a power-law slope measured from data.
        fl = np.logspace(np.log10(rn_band[0]), np.log10(rn_band[1]), 200)
        Sl = powerlaw_rn_psd(fl, rn_basis_freqs, float(rn["S0"]), float(rn["alpha"]),
                             exposure_days=rn_exposure_days)
        if rn_exposure_days is not None:
            # report the INTRINSIC slope: divide the kernel's filter back out
            Sl = Sl / np.sinc(fl * rn_exposure_days) ** 2
            out["rn_exposure_days"] = float(rn_exposure_days)
        out["rn_alpha_band"] = float(-np.polyfit(np.log10(fl), np.log10(Sl), 1)[0])
    elif has_red_noise:
        rn = _unpack_rn_comp(jnp.asarray(theta), n_components)
        out["rn_model"]      = "sho"
        out["rn_sigma"]      = float(rn["sigma"])
        out["rn_period_days"] = float(rn["period"])
        out["rn_Q"]          = float(rn["Q"])  # always _RN_Q ≈ 0.707

    out["sho_jitter"] = float(jitter)

    return out


# ============================================================
# 8. Upfront multi-peak detection with harmonic masking
# ============================================================

def _select_peak(freq, power, mask, white_noise_threshold, continuum=None):
    """One step of the greedy peak walk: the index of the next peak, or None.

    Acceptance is always ``power > threshold`` at the chosen bin.  Ordering depends on the
    threshold form (see ``find_all_significant_peaks``).
    """
    if continuum is not None:
        # Unbiased ordering: largest excess over the CONTINUUM, not over the scaled
        # threshold.
        #
        # The ordering must be restricted to bins that already pass the threshold.
        # When ordering and acceptance use the SAME quantity (the other branches),
        # argmax failing the test proves no bin passes, so the loop can terminate on
        # it.  That proof does not hold here: the bin maximising P - C may fail
        # P > threshold while a different bin passes.  Selecting among eligible bins
        # only makes termination depend on the order-independent question "is any bin
        # significant?", which is what keeps this mode consistent with 'red_noise'.
        thr = (np.asarray(white_noise_threshold)
               if not np.isscalar(white_noise_threshold)
               else np.full(len(power), float(white_noise_threshold)))
        eligible = mask & (power > thr)
        if not eligible.any():
            return None
        masked_excess = np.where(eligible, power - continuum, -np.inf)
        idx = int(np.argmax(masked_excess))
        t_idx = float(thr[idx])
    elif np.isscalar(white_noise_threshold):
        # Scalar: highest absolute power is the right candidate
        masked_power = np.where(mask, power, -np.inf)
        idx = int(np.argmax(masked_power))
        t_idx = float(white_noise_threshold)
    else:
        # Per-frequency array: find the bin with the largest excess above
        # its local threshold so low-frequency peaks aren't missed when a
        # higher-power but sub-threshold high-frequency bin dominates argmax
        masked_excess = np.where(mask, power - white_noise_threshold, -np.inf)
        idx = int(np.argmax(masked_excess))
        t_idx = float(white_noise_threshold[idx])

    if not mask[idx] or power[idx] <= t_idx:
        return None
    return idx


def _refine_peak_frequency(t, y, yerr, f0, df, n=41):
    """Best frequency within +-df of the grid peak f0 on a finer local grid."""
    from astropy.timeseries import LombScargle
    fine = np.linspace(max(f0 - df, 1e-6), f0 + df, n)
    pw = LombScargle(t, y, yerr, normalization='psd').power(fine)
    return float(fine[int(np.argmax(pw))])


def prewhiten_sinusoid(t, y, yerr, f):
    """Subtract the least-squares sinusoid (plus offset) at frequency f from y."""
    from astropy.timeseries import LombScargle
    ls = LombScargle(t, y, yerr, normalization='psd')
    return y - ls.model(t, f)


def multisine_model(t, y, yerr, freqs):
    """Weighted least-squares fit of an offset plus a sinusoid at EACH of ``freqs``, jointly.

    On a gapped window, sinusoids a few resolution elements apart are not orthogonal, so
    fitting and subtracting them one at a time also removes part of each neighbour.
    Fitting all of them together (as Period04 does after every new frequency) assigns the
    shared power correctly.
    """
    t = np.asarray(t, float); y = np.asarray(y, float); w = 1.0 / np.asarray(yerr, float)
    cols = [np.ones_like(t)]
    for f in freqs:
        ph = 2.0 * np.pi * float(f) * t
        cols += [np.sin(ph), np.cos(ph)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return A @ coef


def find_all_significant_peaks(
    freq,
    power,
    white_noise_threshold,
    harmonic_tolerance=0.10,
    harmonic_masking=True,
    continuum=None,
    prewhiten=None,
    refine_frequency=True,
    prewhiten_mode="sequential",
):
    """
    Iteratively find all LSP peaks above `white_noise_threshold`.

    After each peak at best_freq, a window of ±harmonic_tolerance (fractional)
    around best_freq itself is always masked.  When harmonic_masking=True,
    windows around best_freq × h for h in [2, 3, 4, 0.5, 1/3] are also masked
    (i.e. harmonics and sub-harmonics are suppressed before searching for the
    next peak).  Set harmonic_masking=False to allow harmonic peaks to appear
    in the output.

    Parameters
    ----------
    continuum : ndarray or None
        If given, peaks are ORDERED by ``power - continuum`` instead of by
        ``power - white_noise_threshold``.  Acceptance is unchanged: a peak is kept
        only while ``power > white_noise_threshold``.

        This separates the two jobs the statistic was doing.  Ordering by
        ``power - gamma*C`` over-subtracts the continuum by gamma (~12.7), which
        penalises low frequencies far more than high ones and biases selection toward
        short periods.  Ordering by ``power - C`` is unbiased: for a signal added
        independently to red noise, E[P - C] = S is the same at every frequency, so a
        sinusoid of fixed physical amplitude competes equally regardless of period.
    prewhiten : (t, y, yerr) or None
        When given, the search PREWHITENS: after each accepted peak the least-squares
        sinusoid at that frequency is subtracted from the light curve and ``power`` is
        recomputed from the residual (astropy ``LombScargle``, ``normalization='psd'``,
        same grid) before the next peak is looked for.  The threshold and continuum are
        NOT recomputed.

        Why: the periodogram of a finite, gapped window puts a comb of sidelobes around
        every real peak (spacing 1/T_sector and 1/T_orbit for TESS, i.e. 0.04-0.07 /day
        whatever the period).  Below f ~ 1.5 /day the fractional mask is narrower than one
        sidelobe spacing, so each sidelobe of a loud signal is accepted as its own peak:
        on the synthetic set 1.45 seeds per injection, raw purity 0.67, with 71% of the
        spurious seeds at sidelobe offsets (docs/gp_fitting.tex S4).  Sidelobes are a
        property of the sampling, not the data, so subtracting the sinusoid removes them
        with the peak.  Unlike an absolute-frequency mask this costs no resolution: a
        second real signal one orbital frequency from a loud one is still found.

        The fractional mask is kept because a quasi-periodic (low-Q) signal is not a
        sinusoid and leaves residual power at its own frequency after prewhitening.
        Expect prewhitening to be most effective on coherent signals; on broad peaks the
        mask does the work, as before.
    prewhiten_mode : {'sequential', 'joint'}
        'sequential' (default): subtract only the newest sinusoid from the running residual.
        'joint': after each new peak, refit ALL accepted frequencies together
        (``multisine_model``, as Period04 does) and recompute the periodogram of data minus
        that model.  Joint fitting is the textbook remedy for non-orthogonal sinusoids on a
        gapped window, but on the synthetic set it made no difference (529 vs 531 of 611
        detectable injections seeded, 439 vs 445 seeds), so the validated default is
        'sequential'.  The injections prewhitening does not seed are almost all unresolved
        from a louder neighbour or detectable only through its leakage
        (figs/sidelobe_removal/seeding_prewhitening.png, panel d).
    refine_frequency : bool
        With ``prewhiten``, refine each accepted frequency on a 41-point grid within one
        grid step before fitting the sinusoid (a sinusoid subtracted at a frequency off by
        up to half a grid step leaves a beat residual).  The reported ``freq``/``period``
        of the peak is the refined value.

    Returns
    -------
    peaks : list of dicts [{freq, period, power, power_raw}, ...]
        Significant peaks in descending order of the ordering statistic.  ``power`` is
        the periodogram value at acceptance (the residual periodogram when prewhitening),
        ``power_raw`` the value in the input periodogram at the same bin.
    masked_windows : list of (f_lo, f_hi) tuples
        Frequency intervals that were masked out during the search.
        Useful for visualising which regions were suppressed.
    """
    freq = np.asarray(freq)
    power = np.asarray(power)
    power_raw = power
    if continuum is not None:
        continuum = np.asarray(continuum)

    mask = np.ones(len(freq), dtype=bool)  # True = still searchable
    peaks = []
    masked_windows = []

    harmonics = [1.0, 2.0, 3.0, 4.0, 0.5, 1.0 / 3.0] if harmonic_masking else [1.0]

    if prewhiten is not None:
        from astropy.timeseries import LombScargle
        t_pw, y_pw, yerr_pw = (np.asarray(a, dtype=float) for a in prewhiten)
        y_res = y_pw.copy()
        df_grid = float(np.median(np.diff(freq))) if len(freq) > 1 else 0.0

    while True:
        idx = _select_peak(freq, power, mask, white_noise_threshold, continuum)
        if idx is None:
            break

        best_freq = float(freq[idx])
        if prewhiten is not None and refine_frequency and df_grid > 0:
            best_freq = _refine_peak_frequency(t_pw, y_res, yerr_pw, best_freq, df_grid)
        peaks.append({
            "freq": best_freq,
            "period": float(1.0 / best_freq),
            "power": float(power[idx]),
            "power_raw": float(power_raw[idx]),
        })

        for h in harmonics:
            target = best_freq * h
            f_lo = target * (1.0 - harmonic_tolerance)
            f_hi = target * (1.0 + harmonic_tolerance)
            masked_windows.append((float(f_lo), float(f_hi)))
            mask[np.abs(freq / target - 1.0) <= harmonic_tolerance] = False

        if prewhiten is not None:
            if prewhiten_mode == "joint":
                y_res = y_pw - multisine_model(t_pw, y_pw, yerr_pw, [q["freq"] for q in peaks])
            elif prewhiten_mode == "sequential":
                y_res = prewhiten_sinusoid(t_pw, y_res, yerr_pw, best_freq)
            else:
                raise ValueError(f"prewhiten_mode must be 'joint' or 'sequential', got {prewhiten_mode!r}")
            power = LombScargle(t_pw, y_res, yerr_pw, normalization='psd').power(freq)

    return peaks, masked_windows


# ============================================================
# 8b. Per-component evidence: leave-one-out log-likelihood drop
# ============================================================

def component_loglike_drops(fit, x, y, yerr, sequential=True):
    """Per-component evidence: log-likelihood lost when a periodic component is removed.

    Two variants, both without refitting (one GP likelihood call per evaluation, every
    other parameter -- the other components, the red-noise term, jitter -- held as fitted;
    a component is "removed" by setting its sigma to e^-50):

    * ``dloglike`` (single pass, leave-one-out): ``loglike(full) - loglike(without i)``
      for each i against the full model.
    * ``dloglike_seq`` (backward elimination): repeatedly remove the component with the
      smallest current drop, re-evaluating the remaining components' drops after each
      removal, and record for each component the drop it had when it was removed.
      Thresholding ``dloglike_seq >= T`` reproduces backward elimination stopped at T.

    Why two: when two seeds share one signal (a window sidelobe, an unresolved
    neighbour), each one's single-pass drop is small because the OTHER still covers the
    signal, so a single-pass threshold can remove both and lose the signal.  Backward
    elimination removes one, after which the partner's drop jumps to the evidence of the
    signal itself, and keeps it.

    This is the evidence the amplitude cut in ``gp_table_io.filter_gp_components`` was
    standing in for.  A duplicate seed is fitted with ~zero amplitude and has ~zero drop
    either way.  A faint but coherent real signal can have a tiny sigma yet a drop of many
    units, because the GP gains likelihood from every cycle it predicts -- which an
    amplitude threshold cannot see.

    Not refitting makes each drop an upper bound on the refitted evidence (the others
    could re-absorb some of the removed variance); the bound is tight when the removed
    component overlaps nothing.

    Returns ``(dloglike, dloglike_seq)``, each a list aligned with the components (empty
    lists for a red-noise-only fit).  ``dloglike_seq`` is None when ``sequential=False``.
    """
    n = int(fit["n_components"])
    if n == 0:
        return [], ([] if sequential else None)
    theta = np.array(fit["theta"], dtype=float)
    x_j, y_j, yerr_j = jnp.asarray(x), jnp.asarray(y), jnp.asarray(yerr)
    kw = dict(has_red_noise=fit.get("has_red_noise", False),
              rn_model=fit.get("rn_model") or "sho",
              rn_basis_freqs=fit.get("rn_basis_freqs"),
              rn_exposure_days=fit.get("rn_exposure_days"),
              **q_layout_kwargs(fit))

    def _ll(th):
        return float(build_multi_sho_gp(jnp.asarray(th), x_j, yerr_j, n, **kw).log_probability(y_j))

    def _without(th, idx):
        out = th.copy()
        for i in idx:
            out[3 * i] = -50.0      # sigma_i = e^-50: the component contributes nothing
        return out

    ll_full = _ll(theta)
    drops = [ll_full - _ll(_without(theta, [i])) for i in range(n)]
    if not sequential:
        return drops, None

    seq = [np.nan] * n
    removed = []
    current = list(range(n))
    ll_cur = ll_full
    cur_drops = dict(zip(current, drops))      # first step reuses the single-pass values
    while current:
        i_min = min(current, key=lambda i: cur_drops[i])
        seq[i_min] = cur_drops[i_min]
        removed.append(i_min)
        current.remove(i_min)
        if not current:
            break
        ll_cur = _ll(_without(theta, removed))
        cur_drops = {i: ll_cur - _ll(_without(theta, removed + [i])) for i in current}
    return drops, seq


# ============================================================
# 9. Checkpoint I/O
# ============================================================

def _save_checkpoint(checkpoint_dir, m, t, x, y, yerr, fits,
                     white_noise_info, initial_lsp, all_peaks, n):
    """Save a partial GPFitResult (best-BIC of completed fits) to checkpoint_dir."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    bics = [f["bic"] for f in fits]
    best_idx = int(np.argmin(bics))
    best_fit = fits[best_idx]

    final_resid, final_mu, final_std = compute_gp_residuals(best_fit, x, y)
    final_features = summarize_components(best_fit["theta"], best_fit["n_components"],
                                          has_red_noise=best_fit.get("has_red_noise", False),
                                          rn_model=best_fit.get("rn_model") or "sho",
                                          rn_basis_freqs=best_fit.get("rn_basis_freqs"),
                                          rn_band=best_fit.get("rn_band"),
                                          rn_exposure_days=best_fit.get("rn_exposure_days"),
                                          **q_layout_kwargs(best_fit))
    final_features.update(rn_white_noise_features(best_fit, x, yerr))
    final_features.update({
        "n_gp_components": best_fit["n_components"],
        "gp_log_likelihood": best_fit["loglike"],
        "gp_bic": best_fit["bic"],
        "gp_residual_std": float(np.nanstd(final_resid)),
        "gp_residual_rms": float(np.sqrt(np.nanmean(final_resid**2))),
        "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
        "initial_lsp_period_days": initial_lsp["best_period"],
        "initial_lsp_power": initial_lsp["best_power"],
        "initial_lsp_n_peaks": n,
        "initial_lsp_peak_periods": [p["period"] for p in all_peaks[:n]],
        "lsp_n_peaks_found": len(all_peaks),
        "components_capped": bool(len(all_peaks) > n),
    })

    result = GPFitResult(
        t=t, x=x, y=y, yerr=yerr,
        gp_mean=final_mu, gp_std=final_std, residual=final_resid,
        features=final_features,
        final_fit=best_fit,
        all_accepted_fits=fits,
        white_noise_info=white_noise_info,
        all_fits=fits,
        peaks=all_peaks,
    )

    path = os.path.join(checkpoint_dir, f"checkpoint_m{m:02d}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(result, fh)


def save_fits(path, subset_table, results):
    """Save (subset_table, list[GPFitResult]) to a single pickle file."""
    with open(path, "wb") as fh:
        pickle.dump({"subset_table": subset_table, "results": results}, fh)


def load_fits(path):
    """Load a saved (subset_table, list[GPFitResult]) pair."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data["subset_table"], data["results"]


def load_gp_checkpoint(path):
    """Load a single inner checkpoint saved by iterative_sho_gp_fit."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ============================================================
# 10. Main exhaustive-search GP fit
# ============================================================

def iterative_sho_gp_fit(
    t,
    norm_flux,
    flux_err,
    lsp_freq,
    lsp_power,
    threshold_mode='white_noise',
    wn_threshold=None,
    fap_threshold=None,
    red_noise_params=None,
    excess_frac=1.0,
    max_components=6,
    min_period=1/24,
    max_period=10.0,
    Q_min=Q_MIN,
    Q_max=None,
    init_Q=10.0,
    tau_cap_days=TAU_CAP_DAYS,
    q_prior_threshold=Q_PRIOR_THRESHOLD,
    harmonic_tolerance=0.10,
    harmonic_masking=True,
    prewhiten_seeds=False,
    prewhiten_mode="sequential",
    fit_full_only=False,
    bic_improvement_threshold=10.0,
    debug_fitalln=False,
    verbose=0,
    checkpoint_dir=None,
    fit_red_noise_component=False,
    use_lognormal_period_prior=False,
    period_prior_sigma_dex=0.3,
    two_phase_fitting=False,
    use_q_prior=False,
    q_prior_weight=1.0,
    rn_period_cap=10.0,
    rn_model="powerlaw",
    rn_pad_decades=1.2,
    rn_per_decade=2.0,
    rn_alpha_init=1.0,
    rn_alpha_release_last=False,
    rn_alpha_fixed=False,
    rn_exposure_min=30.0,
    rn_wn_flag_ratio=RN_WN_FLAG_RATIO,
    rn_alpha_wn_prior_weight=RN_ALPHA_WN_PRIOR_WEIGHT,
    rn_alpha_wn_prior_ratio=RN_ALPHA_WN_PRIOR_RATIO,
    rn_alpha_wn_prior_width=RN_ALPHA_WN_PRIOR_WIDTH,
):
    """
    Exhaustive-search multi-component SHO GP fit.

    Phase 1: identify all significant peaks in the pre-computed LSP above a
    threshold determined by ``threshold_mode``.  When harmonic_masking=True
    (default), harmonics and sub-harmonics of each found peak are suppressed
    before searching for the next one.

    Phase 2: for m = 1 … n, fit an m-component GP seeded by the top-m peaks.
    BIC early-stopping is active by default; set debug_fitalln=True to fit all
    n models regardless.

    Parameters
    ----------
    lsp_freq : array-like
        Pre-computed LSP frequency grid (1/day) from the table's LSP_freq column.
    lsp_power : array-like
        Pre-computed LSP power from the table's LSP_power column.
    threshold_mode : {'white_noise', 'fap', 'red_noise', 'red_noise_subtracted'}
        Which significance threshold to use for peak detection.
        'white_noise' : scalar 99th-pct bootstrap max (LSP_WN_threshold).
        'fap'         : scalar 1% FAP analytic threshold (LSP_FAP_power / Baluev).
        'red_noise'   : per-frequency Vaughan 1% significance (LSP_red_noise_params).
    wn_threshold : float, optional
        Required when threshold_mode='white_noise'.
    fap_threshold : float, optional
        Required when threshold_mode='fap'.
    red_noise_params : tuple, optional
        (log10_N, alpha, P_N_empirical, P_N_theoretical) from LSP_red_noise_params.
        Required for 'red_noise' and 'red_noise_subtracted'.
    excess_frac : float, default 1.0
        Only used by threshold_mode='red_noise_subtracted'.  Peaks are accepted while
        P - C > k*C, with the threshold expressed as a FRACTION of the significance
        gate so the caller never needs gamma (which depends on len(freq) and so is not
        knowable outside this function)::

            k = excess_frac * (gamma - 1)

          1.0 : the 1% significance gate.  Identical to the 'red_noise' acceptance
                (P/C > gamma), so the two modes find the SAME peaks and differ only in
                the order they are seeded.  Default.
          0.0 : pure subtraction -- every bin above the continuum qualifies.
          0.5 : halfway between, in k.
          >1  : stricter than the 1% gate.

        NOTE: the parameter is linear in the threshold but *exponential* in how many
        bins pass, since the null acceptance rate is exp(-(1+k)).  For a typical
        N_f = 3200 bins the expected number of pure-noise bins accepted is::

            excess_frac  0.00   0.25    0.50    0.75    1.00
            null bins    1178     64     3.4    0.19    0.01

        So 0.5 is already close to the significance end, not the midpoint in practice,
        and 0.0 is qualitatively different: the search then never self-terminates and
        runs to max_components on every row, including pure noise.
        Required when threshold_mode='red_noise'.
    Q_min, tau_cap_days, Q_max, q_prior_threshold : float
        Bounds on each periodic SHO's quality factor (module notes on TAU_CAP_DAYS,
        Q_MIN, Q_PRIOR_THRESHOLD).  Q_min = 1/sqrt(2) is the hard floor.  tau_cap_days
        (default 27, one TESS sector) is a hard ceiling on the decay time Q P / pi, i.e.
        Q <= pi * min(tau_cap_days, baseline) / P -- period-dependent, the same for every
        row whatever its sector count.  Q_max is an optional extra absolute ceiling.
        q_prior_threshold (default pi, one period of decay) is where ``use_q_prior``
        starts to push.  tau_cap_days=None reverts to the pre-2026-11 flat box
        [Q_min, Q_max] in log Q.
    prewhiten_seeds : bool, default False
        Prewhiten during the peak search: after each accepted peak subtract the
        least-squares sinusoid at that frequency and recompute the periodogram of the
        residual before looking for the next one (``find_all_significant_peaks``).
        Removes window-function sidelobes, which otherwise seed ~45% extra components
        (docs/gp_fitting.tex S4).  Threshold and continuum stay those of the original
        periodogram.  ``prewhiten_mode`` ('sequential' default, or 'joint') is passed through.
    fit_red_noise_component : bool
        Add a red-noise term to the kernel (2 extra free parameters either way).
    rn_model : {'powerlaw', 'sho'}
        'powerlaw' (default): fixed log-spaced basis of OU terms with weights tied to a
        fitted slope, S(f) ~ S0 (f/1 d^-1)^-alpha, alpha in RN_ALPHA_BOUNDS.  Reports
        rn_alpha, rn_log10_S0, rn_rms_band.
        'sho': the legacy fixed-Q=1/sqrt(2) SHO (rn_sigma, rn_period_days, rn_Q).  Its
        effective slope is uncorrelated with the true alpha -- kept for reproducibility.
    rn_pad_decades, rn_per_decade : float
        Basis layout for 'powerlaw' (see rn_basis_frequencies).  Defaults give 9 terms
        for a 0.1-12 /day band.
    rn_alpha_init : float or 'measured'
        Starting alpha for the power-law RN term.  'measured' uses the periodogram
        continuum slope from red_noise_params (clipped inside RN_ALPHA_BOUNDS).
    rn_exposure_min : float or None, default 30.0
        Exposure (averaging) length of each data point in minutes.  30 is right for the
        pipeline light curves, which ``resample_lc`` puts on 30-min bins whatever the native
        cadence.  The power-law basis weights then include the exposure filter
        (``exposure_weight_table``), so rn_alpha estimates the INTRINSIC slope; without it
        the fit reads about 0.3 steep because the kernel assumes instantaneous samples
        (docs/fixing_RN_fitting.tex, S9).  Pass None for point-sampled data, or the actual
        exposure if the light curves were not resampled.  Ignored unless
        rn_model='powerlaw'.
    rn_alpha_fixed : bool, default False
        Hold alpha at rn_alpha_init for the whole fit (never released).  With
        rn_alpha_init='measured' the GP uses the periodogram continuum slope; alpha then
        does not count as a free parameter in the BIC.
    rn_alpha_release_last : bool, default False
        Hold alpha at rn_alpha_init through the usual fit, then release it in one extra
        L-BFGS-B stage.  Guarantees the result is no worse than the fixed-alpha fit, but
        alpha then mostly stays at its seed (within 0.02 in 40% of synthetic rows) and the
        synthetic comparison got worse (docs/fixing_RN_fitting.tex, S7.2).  Off by default.
    rn_alpha_wn_prior_weight, rn_alpha_wn_prior_ratio, rn_alpha_wn_prior_width : float
        Prior pulling the RN slope to 0 where the red noise is buried in the white noise
        (see _rn_alpha_wn_neg_log_prior).  weight=0 disables it.  Defaults: 1.0, ratio 0.1
        (half strength), width 0.5 decades.
    rn_wn_flag_ratio : float
        Threshold for the rn_near_white_noise flag (see rn_white_noise_features).
    verbose : int
        0 = silent (default). 1 = peak list, per-fit BIC/timing, early-stop reason.
        2 = verbose=1 plus initial LSP plot and component breakdown after each fit.
        3 = verbose=2 plus per-fit component breakdown plots during fitting.
    """

    t0_total = time.perf_counter()
    t_clean, x, y, yerr = clean_lc(t, norm_flux, flux_err)

    powerlaw_alpha_fixed = bool(rn_alpha_fixed and rn_model == "powerlaw")
    if isinstance(rn_alpha_init, str):
        if rn_alpha_init != "measured":
            raise ValueError(f"rn_alpha_init must be a number or 'measured', got {rn_alpha_init!r}")
        if red_noise_params is None or not np.isfinite(red_noise_params[1]):
            raise ValueError("rn_alpha_init='measured' needs red_noise_params with a finite alpha")
        rn_alpha_init = float(red_noise_params[1])

    freq_arr  = np.asarray(lsp_freq)
    power_arr = np.asarray(lsp_power)

    # Resolve threshold based on mode
    peak_continuum = None       # only 'red_noise_subtracted' orders by P - C
    if threshold_mode == 'white_noise':
        peak_threshold = float(wn_threshold)
    elif threshold_mode == 'fap':
        peak_threshold = float(fap_threshold)
    elif threshold_mode == 'red_noise':
        log10_N, alpha, P_N_empirical, _ = red_noise_params
        P_model = 10**log10_N * freq_arr**(-alpha)
        gamma   = -np.log(1 - 0.99**(1 / len(freq_arr)))
        peak_threshold = P_model * gamma   # ndarray, per-frequency
    elif threshold_mode == 'red_noise_subtracted':
        log10_N, alpha, _, _ = red_noise_params
        P_model = 10**log10_N * freq_arr**(-alpha)
        gamma   = -np.log(1 - 0.99**(1 / len(freq_arr)))
        try:
            frac = float(excess_frac)
        except (TypeError, ValueError):
            raise ValueError(f"excess_frac must be a number, got {excess_frac!r}")
        if not np.isfinite(frac) or frac < 0:
            raise ValueError(f"excess_frac must be finite and >= 0, got {excess_frac!r}")
        # frac = 1 -> k = gamma-1 -> acceptance identical to 'red_noise'
        # frac = 0 -> k = 0       -> every bin above the continuum
        k = frac * (gamma - 1.0)
        # Accept while P - C > k*C, i.e. P > (1+k)*C.  Order by P - C (see
        # find_all_significant_peaks), which is the unbiased locator.
        peak_threshold = (1.0 + k) * P_model
        peak_continuum = P_model
    else:
        raise ValueError(
            "threshold_mode must be 'white_noise', 'fap', 'red_noise', or "
            f"'red_noise_subtracted', got {threshold_mode!r}"
        )

    scalar_repr = float(np.mean(peak_threshold)) if not np.isscalar(peak_threshold) else peak_threshold
    white_noise_info = {
        "white_noise_power_threshold": scalar_repr,
        "threshold_mode": threshold_mode,
    }

    idx_best = int(np.nanargmax(power_arr))
    initial_lsp = {
        "freq":        freq_arr,
        "power":       power_arr,
        "best_freq":   float(freq_arr[idx_best]),
        "best_period": float(1.0 / freq_arr[idx_best]),
        "best_power":  float(power_arr[idx_best]),
        "fap":         np.nan,
    }

    all_peaks, masked_windows = find_all_significant_peaks(
        initial_lsp["freq"],
        initial_lsp["power"],
        peak_threshold,
        harmonic_tolerance=harmonic_tolerance,
        harmonic_masking=harmonic_masking,
        continuum=peak_continuum,
        prewhiten=(x, y, yerr) if prewhiten_seeds else None,
        prewhiten_mode=prewhiten_mode,
    )

    n = min(len(all_peaks), max_components)

    if verbose >= 4:
        print(f"\n  Peak-finding diagnostic  ({len(all_peaks)} peaks found, {n} used):")
        print(f"  {'#':>3}  {'freq (1/d)':>12}  {'period (d)':>10}  "
              f"{'power':>12}  {'threshold':>12}  {'excess':>8}  used")
        for i, p in enumerate(all_peaks):
            f  = p["freq"]
            pw = p["power"]
            if np.isscalar(peak_threshold):
                t = float(peak_threshold)
            else:
                t = float(peak_threshold[int(np.argmin(np.abs(freq_arr - f)))])
            used_str = "yes" if i < n else "no (>max_components)"
            print(f"  {i+1:>3}  {f:>12.5f}  {1/f:>10.4f}  "
                  f"{pw:>12.6f}  {t:>12.6f}  {pw-t:>8.6f}  {used_str}")
        print()

    if n == 0:
        # No significant peaks.  With a red-noise term in the kernel there is nothing to
        # seed and no need to invent a component: fit red noise + jitter alone (m=0).
        # The old fallback (one SHO at P=1 d) produced a broad Q~0.5 component that ABSORBED
        # the continuum -- on the real table those rows read a median gp_rn_alpha of 0.05
        # against 0.96 elsewhere, i.e. the red-noise parameters were contaminated by a
        # component that was never a detection.  Without a red-noise term the kernel would be
        # empty, so the old behaviour is kept in that case.
        _fb_components = [] if fit_red_noise_component else [float(np.clip(1.0, min_period, max_period))]
        if verbose >= 1:
            print("  No significant LSP peaks found — fitting "
                  + ("red noise + jitter only (no SHO component)." if fit_red_noise_component
                     else "fallback 1-component SHO (P_seed=1 d, Q_seed=1)."))
        if verbose >= 2:
            from .visualize_gp_duringfit import plot_initial_lsp
            plot_initial_lsp(
                initial_lsp["freq"],
                initial_lsp["power"],
                [],
                masked_windows,
                white_noise_info,
                peak_threshold=peak_threshold,
            )
        t0_fb = time.perf_counter()
        fb_fit = fit_multi_sho_gp(
            x, y, yerr,
            period_guesses=_fb_components,
            min_period=min_period,
            max_period=max_period,
            Q_min=Q_min,
            Q_max=Q_max,
            init_Q=1.0,
            tau_cap_days=tau_cap_days,
            q_prior_threshold=q_prior_threshold,
            has_red_noise=fit_red_noise_component,
            use_lognormal_period_prior=use_lognormal_period_prior,
            period_prior_sigma_dex=period_prior_sigma_dex,
            use_q_prior=use_q_prior,
            q_prior_weight=q_prior_weight,
            rn_period_cap=rn_period_cap,
            rn_model=rn_model,
            rn_pad_decades=rn_pad_decades,
            rn_per_decade=rn_per_decade,
            rn_alpha_init=rn_alpha_init,
            rn_alpha_release_last=rn_alpha_release_last,
            rn_alpha_fixed=rn_alpha_fixed,
            rn_exposure_days=None if rn_exposure_min is None else rn_exposure_min / 1440.0,
            rn_alpha_wn_prior_weight=rn_alpha_wn_prior_weight,
            rn_alpha_wn_prior_ratio=rn_alpha_wn_prior_ratio,
            rn_alpha_wn_prior_width=rn_alpha_wn_prior_width,
        )
        fb_n_params = (3 * len(_fb_components) + 1
                       + ((1 if powerlaw_alpha_fixed else 2) if fit_red_noise_component else 0))
        fb_fit["bic"] = compute_bic(fb_fit["loglike"], n_params=fb_n_params, n_data=len(y))

        if verbose >= 2:
            if _fb_components:
                fb_comps, _ = unpack_theta(fb_fit["theta"], 1, **q_layout_kwargs(fb_fit))
                print(f"  Fallback fit: period={float(fb_comps[0]['period']):.3f} d, "
                      f"BIC={fb_fit['bic']:.1f}  ({time.perf_counter() - t0_fb:.1f} s)")
            else:
                print(f"  Red-noise-only fit: BIC={fb_fit['bic']:.1f}  "
                      f"({time.perf_counter() - t0_fb:.1f} s)")

        fb_resid, fb_mu, fb_std = compute_gp_residuals(fb_fit, x, y)
        fb_features = summarize_components(fb_fit["theta"], len(_fb_components),
                                           has_red_noise=fit_red_noise_component,
                                           rn_model=fb_fit.get("rn_model") or "sho",
                                           rn_basis_freqs=fb_fit.get("rn_basis_freqs"),
                                           rn_band=fb_fit.get("rn_band"),
                                           rn_exposure_days=fb_fit.get("rn_exposure_days"),
                                           **q_layout_kwargs(fb_fit))
        fb_features.update(rn_white_noise_features(fb_fit, x, yerr, rn_wn_flag_ratio))
        fb_features.update({
            "n_gp_components":          len(_fb_components),
            "gp_log_likelihood":        fb_fit["loglike"],
            "gp_bic":                   fb_fit["bic"],
            "gp_residual_std":          float(np.nanstd(fb_resid)),
            "gp_residual_rms":          float(np.sqrt(np.nanmean(fb_resid**2))),
            "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
            "initial_lsp_period_days":  initial_lsp["best_period"],
            "initial_lsp_power":        initial_lsp["best_power"],
            "initial_lsp_n_peaks":      0,
            "initial_lsp_peak_periods": [],
            "lsp_n_peaks_found":        0,
            "components_capped":        False,
        })
        return GPFitResult(
            t=t_clean, x=x, y=y, yerr=yerr,
            gp_mean=fb_mu, gp_std=fb_std, residual=fb_resid,
            features=fb_features,
            final_fit=fb_fit,
            all_accepted_fits=[fb_fit],
            white_noise_info=white_noise_info,
            all_fits=[fb_fit],
            peaks=[],
        )

    if verbose >= 2:
        print(
            f"Found {len(all_peaks)} significant peak(s); "
            f"fitting up to {n} component(s)."
        )
        for i, p in enumerate(all_peaks[:n]):
            print(f"  Peak {i+1}: period={p['period']:.3f} d, power={p['power']:.4f}")

    if verbose >= 2:
        from .visualize_gp_duringfit import plot_initial_lsp
        plot_initial_lsp(
            initial_lsp["freq"],
            initial_lsp["power"],
            all_peaks[:n],
            masked_windows,
            white_noise_info,
            peak_threshold=peak_threshold,
        )

    fits = []
    prev_bic = None

    m_range = [n] if fit_full_only else range(1, n + 1)

    for m in m_range:
        period_guesses = [all_peaks[i]["period"] for i in range(m)]

        if verbose >= 2:
            print(f"Fitting {m}-component model...")

        t0 = time.perf_counter()
        fit = fit_multi_sho_gp(
            x,
            y,
            yerr,
            period_guesses,
            min_period=min_period,
            max_period=max_period,
            Q_min=Q_min,
            Q_max=Q_max,
            init_Q=init_Q,
            tau_cap_days=tau_cap_days,
            q_prior_threshold=q_prior_threshold,
            has_red_noise=fit_red_noise_component,
            use_lognormal_period_prior=use_lognormal_period_prior,
            period_prior_sigma_dex=period_prior_sigma_dex,
            two_phase_fitting=two_phase_fitting,
            use_q_prior=use_q_prior,
            q_prior_weight=q_prior_weight,
            rn_period_cap=rn_period_cap,
            rn_model=rn_model,
            rn_pad_decades=rn_pad_decades,
            rn_per_decade=rn_per_decade,
            rn_alpha_init=rn_alpha_init,
            rn_alpha_release_last=rn_alpha_release_last,
            rn_alpha_fixed=rn_alpha_fixed,
            rn_exposure_days=None if rn_exposure_min is None else rn_exposure_min / 1440.0,
            rn_alpha_wn_prior_weight=rn_alpha_wn_prior_weight,
            rn_alpha_wn_prior_ratio=rn_alpha_wn_prior_ratio,
            rn_alpha_wn_prior_width=rn_alpha_wn_prior_width,
        )
        if verbose >= 2:
            print(f"  Fit {m} component(s): {time.perf_counter() - t0:.1f} s")

        n_params = 3 * m + 1 + ((1 if powerlaw_alpha_fixed else 2) if fit_red_noise_component else 0)
        fit["bic"] = compute_bic(fit["loglike"], n_params=n_params, n_data=len(y))

        fits.append(fit)

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir, m, t_clean, x, y, yerr,
                fits, white_noise_info, initial_lsp, all_peaks, n,
            )

        if verbose >= 2:
            comps_m, _ = unpack_theta(fit["theta"], m, **q_layout_kwargs(fit))
            period_list = [f"{float(c['period']):.3f}" for c in comps_m]
            print(
                f"  {m} component(s): periods={period_list} d, "
                f"BIC={fit['bic']:.1f}"
            )

        if verbose >= 3:
            from .visualize_gp_duringfit import plot_fit_summary
            plot_fit_summary(x, y, yerr, fits, m, fit["bic"])

        # BIC early stopping (bypassed in debug mode)
        if not debug_fitalln and prev_bic is not None:
            delta_bic = prev_bic - fit["bic"]
            if delta_bic < bic_improvement_threshold:
                if verbose >= 2:
                    print(
                        f"  BIC improvement {delta_bic:.1f} < "
                        f"threshold ({bic_improvement_threshold:.1f}). Stopping."
                    )
                break

        prev_bic = fit["bic"]
        jax.clear_caches()

    # Select best-BIC model
    bics = [f["bic"] for f in fits]
    best_idx = int(np.argmin(bics))
    best_fit = fits[best_idx]

    if debug_fitalln:
        print("\nBIC summary (all models):")
        for j, f in enumerate(fits):
            m_j = f["n_components"]
            bic_j = f["bic"]
            is_best = (j == best_idx)
            if j == 0:
                delta_str = ""
            else:
                delta = fits[j - 1]["bic"] - bic_j
                symbol = "+" if delta >= bic_improvement_threshold else "~"
                delta_str = f"  DBIC={delta:+.1f} {symbol}"
            best_str = "  <- best" if is_best else ""
            print(f"  m={m_j}: BIC={bic_j:.1f}{delta_str}{best_str}")
        print()

    if verbose >= 2:
        print(
            f"Best model: {best_fit['n_components']} component(s) "
            f"(BIC={best_fit['bic']:.1f})"
        )
        print(f"Total: {time.perf_counter() - t0_total:.1f} s")

    final_resid, final_mu, final_std = compute_gp_residuals(best_fit, x, y)

    final_features = summarize_components(best_fit["theta"], best_fit["n_components"],
                                          has_red_noise=fit_red_noise_component,
                                          rn_model=best_fit.get("rn_model") or "sho",
                                          rn_basis_freqs=best_fit.get("rn_basis_freqs"),
                                          rn_band=best_fit.get("rn_band"),
                                          rn_exposure_days=best_fit.get("rn_exposure_days"),
                                          **q_layout_kwargs(best_fit))
    final_features.update(rn_white_noise_features(best_fit, x, yerr, rn_wn_flag_ratio))
    _dll, _dll_seq = component_loglike_drops(best_fit, x, y, yerr)
    for i, (d, ds) in enumerate(zip(_dll, _dll_seq)):
        final_features[f"sho_{i+1}_dloglike"] = float(d)
        final_features[f"sho_{i+1}_dloglike_seq"] = float(ds)
    final_features.update({
        "n_gp_components": best_fit["n_components"],
        "gp_log_likelihood": best_fit["loglike"],
        "gp_bic": best_fit["bic"],
        "gp_residual_std": float(np.nanstd(final_resid)),
        "gp_residual_rms": float(np.sqrt(np.nanmean(final_resid**2))),
        "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
        "initial_lsp_period_days": initial_lsp["best_period"],
        "initial_lsp_power": initial_lsp["best_power"],
        "initial_lsp_n_peaks": n,
        "initial_lsp_peak_periods": [p["period"] for p in all_peaks[:n]],
        # More significant peaks than the component budget means the fit ran out of
        # components, so unmodelled oscillator power leaks into the red-noise term: on the
        # synthetic set those rows carry the worst rn_alpha errors (docs/fixing_RN_fitting.tex S9).
        "lsp_n_peaks_found": len(all_peaks),
        "components_capped": bool(len(all_peaks) > n),
    })

    return GPFitResult(
        t=t_clean,
        x=x,
        y=y,
        yerr=yerr,
        gp_mean=final_mu,
        gp_std=final_std,
        residual=final_resid,
        features=final_features,
        final_fit=best_fit,
        all_accepted_fits=fits,
        white_noise_info=white_noise_info,
        all_fits=fits,
        peaks=all_peaks,
    )

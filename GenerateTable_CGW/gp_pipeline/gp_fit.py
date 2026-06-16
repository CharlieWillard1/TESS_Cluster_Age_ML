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

def unpack_theta(theta, n_components):
    """
    theta structure:
    [log_sigma_1, log_period_1, log_Q_1,
     log_sigma_2, log_period_2, log_Q_2,
     ...
     log_jitter]
    """
    theta = jnp.asarray(theta)
    comps = []

    for i in range(n_components):
        log_sigma = theta[3*i + 0]
        log_period = theta[3*i + 1]
        log_Q = theta[3*i + 2]

        sigma = jnp.exp(log_sigma)
        period = jnp.exp(log_period)
        Q = jnp.exp(log_Q)
        omega = 2.0 * jnp.pi / period

        comps.append({
            "sigma": sigma,
            "period": period,
            "omega": omega,
            "Q": Q,
        })

    jitter = jnp.exp(theta[-1])
    return comps, jitter


def _unpack_rn_comp(theta, n_components):
    """Unpack the fixed-Q red noise SHO from theta[3*n : 3*n+2]."""
    sigma  = jnp.exp(theta[3 * n_components])
    period = jnp.exp(theta[3 * n_components + 1])
    return {"sigma": sigma, "period": period, "omega": 2.0 * jnp.pi / period, "Q": _RN_Q}


def build_multi_sho_gp(theta, x, yerr, n_components, has_red_noise=False):
    comps, jitter = unpack_theta(theta, n_components)

    kernel = None

    for comp in comps:
        k = quasisep.SHO(
            omega=comp["omega"],
            quality=comp["Q"],
            sigma=comp["sigma"],
        )

        kernel = k if kernel is None else kernel + k

    if has_red_noise:
        rn = _unpack_rn_comp(theta, n_components)
        kernel = kernel + quasisep.SHO(omega=rn["omega"], quality=_RN_Q, sigma=rn["sigma"])

    diag = yerr**2 + jitter**2

    return GaussianProcess(kernel, x, diag=diag, mean=0.0)


def neg_log_likelihood(theta, x, y, yerr, n_components, has_red_noise=False):
    gp = build_multi_sho_gp(theta, x, yerr, n_components, has_red_noise)
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


def _q_neg_log_prior(theta, n_components, q_prior_weight):
    """Half-quadratic penalty in log-Q space that activates below Q=2.

    No penalty for Q >= 2 (uniform prior). Below Q=2 the penalty grows
    quadratically in log space so that Q=1.5 is a small nudge, Q=1 is
    a moderate push, and Q<1 is a strong pull back toward Q=2.
    """
    log_q_threshold = jnp.log(2.0)
    total = jnp.zeros(())
    for i in range(n_components):
        log_Q = theta[3 * i + 2]
        deficit = jnp.maximum(0.0, log_q_threshold - log_Q)
        total = total + deficit ** 2
    return q_prior_weight * total


# ============================================================
# 3. Fit GP
# ============================================================

def make_initial_theta(period_guesses, y, yerr, init_Q=5.0,
                       has_red_noise=False, rn_max_period=10.0):
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
        pieces.extend([
            np.log(init_sigma),
            np.log(p),
            np.log(init_Q),
        ])

    if has_red_noise:
        rn_sigma = max(init_sigma, 1e-8)
        # Seed at 0.5 * rn_max_period (interior of bound; avoids degenerate first L-BFGS-B step)
        rn_period = 0.5 * rn_max_period
        pieces.extend([np.log(rn_sigma), np.log(rn_period)])

    pieces.append(np.log(max(med_err, 1e-8)))  # jitter always last

    return np.array(pieces, dtype=float)


def make_bounds(n_components, min_period, max_period, Q_min=0.71, Q_max=100.0,
                has_red_noise=False, rn_max_period=None):
    bounds = []

    for _ in range(n_components):
        bounds.extend([
            (np.log(1e-10), np.log(1.0)),              # sigma
            (np.log(min_period), np.log(max_period)),  # period
            (np.log(Q_min), np.log(Q_max)),            # Q
        ])

    if has_red_noise:
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
    Q_min=0.71,
    Q_max=100.0,
    init_Q=10.0,
    has_red_noise=False,
    use_lognormal_period_prior=False,
    period_prior_sigma_dex=0.3,
    two_phase_fitting=False,
    use_q_prior=False,
    q_prior_weight=1.0,
    rn_period_cap=10.0,
):
    n_components = len(period_guesses)
    _baseline = float(np.max(x))
    rn_max_period = min(_baseline, rn_period_cap) if rn_period_cap is not None else _baseline

    theta0 = make_initial_theta(
        period_guesses, y, yerr, init_Q=init_Q,
        has_red_noise=has_red_noise, rn_max_period=rn_max_period,
    )
    bounds = make_bounds(
        n_components,
        min_period=min_period,
        max_period=max_period,
        Q_min=Q_min,
        Q_max=Q_max,
        has_red_noise=has_red_noise,
        rn_max_period=rn_max_period,
    )

    # Convert data to JAX arrays ONCE, not inside every likelihood call
    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)
    yerr_j = jnp.asarray(yerr)

    # JIT-compiled objective. Branches are resolved at Python level (not inside
    # JIT) so each compiled function has a fixed structure.
    _use_period_prior = use_lognormal_period_prior and n_components > 0
    if _use_period_prior:
        period_seeds_j = jnp.asarray(period_guesses)

    if _use_period_prior and use_q_prior:
        @jax.jit
        def objective(theta):
            return (neg_log_likelihood(theta, x_j, y_j, yerr_j, n_components, has_red_noise)
                    + _period_neg_log_prior(theta, period_seeds_j, n_components, period_prior_sigma_dex)
                    + _q_neg_log_prior(theta, n_components, q_prior_weight))
    elif _use_period_prior:
        @jax.jit
        def objective(theta):
            return (neg_log_likelihood(theta, x_j, y_j, yerr_j, n_components, has_red_noise)
                    + _period_neg_log_prior(theta, period_seeds_j, n_components, period_prior_sigma_dex))
    elif use_q_prior:
        @jax.jit
        def objective(theta):
            return (neg_log_likelihood(theta, x_j, y_j, yerr_j, n_components, has_red_noise)
                    + _q_neg_log_prior(theta, n_components, q_prior_weight))
    else:
        @jax.jit
        def objective(theta):
            return neg_log_likelihood(theta, x_j, y_j, yerr_j, n_components, has_red_noise)

    # JIT-compiled value + gradient
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    # scipy-compatible wrapper
    def scipy_objective(theta_np):
        val, grad = value_and_grad(jnp.asarray(theta_np))
        return float(val), np.asarray(grad, dtype=float)

    opt_options = {"maxiter": 1000, "ftol": 1e-8, "gtol": 1e-6, "maxls": 20}

    if two_phase_fitting and n_components > 0:
        # Pass 1: freeze each periodic SHO period to its LSP seed so that σ, Q,
        # the red-noise term, and jitter can find their level while the period is
        # pinned.  This "switches on" the component so the likelihood has a sharp
        # minimum at the correct period by the time we release it in pass 2.
        frozen_bounds = list(bounds)
        for i, p_seed in enumerate(period_guesses):
            log_p = np.log(p_seed)
            frozen_bounds[3 * i + 1] = (log_p, log_p)

        result1 = minimize(
            scipy_objective, theta0,
            method="L-BFGS-B", jac=True,
            bounds=frozen_bounds, options=opt_options,
        )

        # Pass 2: full period bounds, warm-started from pass-1 solution.
        result2 = minimize(
            scipy_objective, result1.x,
            method="L-BFGS-B", jac=True,
            bounds=bounds, options=opt_options,
        )

        # Fallback: pass-1's solution is always feasible for pass 2, so pass 2
        # should never produce a worse objective.  If it does (optimizer escaped
        # the basin or hit an iteration limit mid-flight), revert to pass 1.
        val1, _ = scipy_objective(result1.x)
        val2, _ = scipy_objective(result2.x)
        if val2 > val1:
            import warnings
            warnings.warn(
                f"two_phase_fitting: pass-2 objective ({val2:.4f}) is worse than "
                f"pass-1 ({val1:.4f}). Reverting to pass-1 result.",
                RuntimeWarning,
                stacklevel=2,
            )
            result = result1
        else:
            result = result2
    else:
        result = minimize(
            scipy_objective, theta0,
            method="L-BFGS-B", jac=True,
            bounds=bounds, options=opt_options,
        )

    theta_best = result.x

    gp = build_multi_sho_gp(
        jnp.asarray(theta_best),
        x_j,
        yerr_j,
        n_components,
        has_red_noise,
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
    }


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


def summarize_components(theta, n_components, has_red_noise=False):
    comps, jitter = unpack_theta(jnp.asarray(theta), n_components)

    out = {}

    for i, c in enumerate(comps):
        out[f"sho_{i+1}_sigma"] = float(c["sigma"])
        out[f"sho_{i+1}_period_days"] = float(c["period"])
        out[f"sho_{i+1}_omega"] = float(c["omega"])
        out[f"sho_{i+1}_Q"] = float(c["Q"])

    if has_red_noise:
        rn = _unpack_rn_comp(jnp.asarray(theta), n_components)
        out["rn_sigma"]      = float(rn["sigma"])
        out["rn_period_days"] = float(rn["period"])
        out["rn_Q"]          = float(rn["Q"])  # always _RN_Q ≈ 0.707

    out["sho_jitter"] = float(jitter)

    return out


# ============================================================
# 8. Upfront multi-peak detection with harmonic masking
# ============================================================

def find_all_significant_peaks(
    freq,
    power,
    white_noise_threshold,
    harmonic_tolerance=0.10,
    harmonic_masking=True,
):
    """
    Iteratively find all LSP peaks above `white_noise_threshold`.

    After each peak at best_freq, a window of ±harmonic_tolerance (fractional)
    around best_freq itself is always masked.  When harmonic_masking=True,
    windows around best_freq × h for h in [2, 3, 4, 0.5, 1/3] are also masked
    (i.e. harmonics and sub-harmonics are suppressed before searching for the
    next peak).  Set harmonic_masking=False to allow harmonic peaks to appear
    in the output.

    Returns
    -------
    peaks : list of dicts [{freq, period, power}, ...]
        Significant peaks in descending power order.
    masked_windows : list of (f_lo, f_hi) tuples
        Frequency intervals that were masked out during the search.
        Useful for visualising which regions were suppressed.
    """
    freq = np.asarray(freq)
    power = np.asarray(power)

    mask = np.ones(len(freq), dtype=bool)  # True = still searchable
    peaks = []
    masked_windows = []

    harmonics = [1.0, 2.0, 3.0, 4.0, 0.5, 1.0 / 3.0] if harmonic_masking else [1.0]

    while True:
        if np.isscalar(white_noise_threshold):
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
            break

        best_freq = freq[idx]
        peaks.append({
            "freq": float(best_freq),
            "period": float(1.0 / best_freq),
            "power": float(power[idx]),
        })

        for h in harmonics:
            target = best_freq * h
            f_lo = target * (1.0 - harmonic_tolerance)
            f_hi = target * (1.0 + harmonic_tolerance)
            masked_windows.append((float(f_lo), float(f_hi)))
            mask[np.abs(freq / target - 1.0) <= harmonic_tolerance] = False

    return peaks, masked_windows


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
                                          has_red_noise=best_fit.get("has_red_noise", False))
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
    max_components=6,
    min_period=1/24,
    max_period=10.0,
    Q_min=0.71,
    Q_max=100.0,
    init_Q=10.0,
    harmonic_tolerance=0.10,
    harmonic_masking=True,
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
    threshold_mode : {'white_noise', 'fap', 'red_noise'}
        Which significance threshold to use for peak detection.
        'white_noise' : scalar 99th-pct bootstrap max (LSP_WN_threshold).
        'fap'         : scalar 1% FAP analytic threshold (LSP_FAP / Baluev).
        'red_noise'   : per-frequency Vaughan 1% significance (LSP_red_noise_params).
    wn_threshold : float, optional
        Required when threshold_mode='white_noise'.
    fap_threshold : float, optional
        Required when threshold_mode='fap'.
    red_noise_params : tuple, optional
        (log10_N, alpha, P_N_empirical, P_N_theoretical) from LSP_red_noise_params.
        Required when threshold_mode='red_noise'.
    verbose : int
        0 = silent (default). 1 = peak list, per-fit BIC/timing, early-stop reason.
        2 = verbose=1 plus initial LSP plot and component breakdown after each fit.
        3 = verbose=2 plus per-fit component breakdown plots during fitting.
    """

    t0_total = time.perf_counter()
    t_clean, x, y, yerr = clean_lc(t, norm_flux, flux_err)

    freq_arr  = np.asarray(lsp_freq)
    power_arr = np.asarray(lsp_power)

    # Resolve threshold based on mode
    if threshold_mode == 'white_noise':
        peak_threshold = float(wn_threshold)
    elif threshold_mode == 'fap':
        peak_threshold = float(fap_threshold)
    elif threshold_mode == 'red_noise':
        log10_N, alpha, P_N_empirical, _ = red_noise_params
        P_model = 10**log10_N * freq_arr**(-alpha)
        gamma   = -np.log(1 - 0.99**(1 / len(freq_arr)))
        peak_threshold = P_model * gamma   # ndarray, per-frequency
    else:
        raise ValueError(
            f"threshold_mode must be 'white_noise', 'fap', or 'red_noise', got {threshold_mode!r}"
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
        if verbose >= 1:
            print("  No significant LSP peaks found — fallback: 1-component SHO "
                  "(P_seed=1 d, Q_seed=1).")
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
        fallback_period = float(np.clip(1.0, min_period, max_period))
        t0_fb = time.perf_counter()
        fb_fit = fit_multi_sho_gp(
            x, y, yerr,
            period_guesses=[fallback_period],
            min_period=min_period,
            max_period=max_period,
            Q_min=Q_min,
            Q_max=Q_max,
            init_Q=1.0,
            has_red_noise=fit_red_noise_component,
            use_lognormal_period_prior=use_lognormal_period_prior,
            period_prior_sigma_dex=period_prior_sigma_dex,
            use_q_prior=use_q_prior,
            q_prior_weight=q_prior_weight,
            rn_period_cap=rn_period_cap,
        )
        fb_n_params = 4 + (2 if fit_red_noise_component else 0)
        fb_fit["bic"] = compute_bic(fb_fit["loglike"], n_params=fb_n_params, n_data=len(y))

        if verbose >= 2:
            fb_comps, _ = unpack_theta(fb_fit["theta"], 1)
            print(f"  Fallback fit: period={float(fb_comps[0]['period']):.3f} d, "
                  f"BIC={fb_fit['bic']:.1f}  ({time.perf_counter() - t0_fb:.1f} s)")

        fb_resid, fb_mu, fb_std = compute_gp_residuals(fb_fit, x, y)
        fb_features = summarize_components(fb_fit["theta"], 1, has_red_noise=fit_red_noise_component)
        fb_features.update({
            "n_gp_components":          1,
            "gp_log_likelihood":        fb_fit["loglike"],
            "gp_bic":                   fb_fit["bic"],
            "gp_residual_std":          float(np.nanstd(fb_resid)),
            "gp_residual_rms":          float(np.sqrt(np.nanmean(fb_resid**2))),
            "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
            "initial_lsp_period_days":  initial_lsp["best_period"],
            "initial_lsp_power":        initial_lsp["best_power"],
            "initial_lsp_n_peaks":      0,
            "initial_lsp_peak_periods": [],
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
            has_red_noise=fit_red_noise_component,
            use_lognormal_period_prior=use_lognormal_period_prior,
            period_prior_sigma_dex=period_prior_sigma_dex,
            two_phase_fitting=two_phase_fitting,
            use_q_prior=use_q_prior,
            q_prior_weight=q_prior_weight,
            rn_period_cap=rn_period_cap,
        )
        if verbose >= 2:
            print(f"  Fit {m} component(s): {time.perf_counter() - t0:.1f} s")

        n_params = 3 * m + 1 + (2 if fit_red_noise_component else 0)
        fit["bic"] = compute_bic(fit["loglike"], n_params=n_params, n_data=len(y))

        fits.append(fit)

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir, m, t_clean, x, y, yerr,
                fits, white_noise_info, initial_lsp, all_peaks, n,
            )

        if verbose >= 2:
            comps_m, _ = unpack_theta(fit["theta"], m)
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
                                          has_red_noise=fit_red_noise_component)
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

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

from gp_fit_result import GPFitResult

jax.config.update("jax_enable_x64", True)


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


def build_multi_sho_gp(theta, x, yerr, n_components):
    comps, jitter = unpack_theta(theta, n_components)

    kernel = None

    for comp in comps:
        k = quasisep.SHO(
            omega=comp["omega"],
            quality=comp["Q"],
            sigma=comp["sigma"],
        )

        kernel = k if kernel is None else kernel + k

    diag = yerr**2 + jitter**2

    return GaussianProcess(kernel, x, diag=diag, mean=0.0)


def neg_log_likelihood(theta, x, y, yerr, n_components):
    gp = build_multi_sho_gp(theta, x, yerr, n_components)
    return -gp.log_probability(y)


# ============================================================
# 3. Fit GP
# ============================================================

def make_initial_theta(period_guesses, y, yerr, init_Q=2.0):
    n_components = len(period_guesses)

    y_std = np.nanstd(y)
    med_err = np.nanmedian(yerr)

    pieces = []

    # split variance roughly across components
    init_sigma = max(y_std / np.sqrt(max(n_components, 1)), 1e-8)

    for p in period_guesses:
        pieces.extend([
            np.log(init_sigma),
            np.log(p),
            np.log(init_Q),
        ])

    pieces.append(np.log(max(med_err, 1e-8)))  # jitter

    return np.array(pieces, dtype=float)


def make_bounds(n_components, min_period=0.05, max_period=50.0, Q_min=0.51, Q_max=100.0):
    bounds = []

    for _ in range(n_components):
        bounds.extend([
            (np.log(1e-10), np.log(1.0)),              # sigma
            (np.log(min_period), np.log(max_period)),  # period
            (np.log(Q_min), np.log(Q_max)),            # Q
        ])

    bounds.append((np.log(1e-10), np.log(1.0)))  # jitter

    return bounds


# def fit_multi_sho_gp(
#     x,
#     y,
#     yerr,
#     period_guesses,
#     min_period=0.05,
#     max_period=50.0,
#     Q_min=0.51,
#     Q_max=100.0,
#     init_Q=10.0,
# ):
#     n_components = len(period_guesses)

#     theta0 = make_initial_theta(period_guesses, y, yerr, init_Q=init_Q)
#     bounds = make_bounds(
#         n_components,
#         min_period=min_period,
#         max_period=max_period,
#         Q_min=Q_min,
#         Q_max=Q_max,
#     )

#     result = minimize(
#         lambda p: np.asarray(
#             neg_log_likelihood(
#                 jnp.asarray(p),
#                 jnp.asarray(x),
#                 jnp.asarray(y),
#                 jnp.asarray(yerr),
#                 n_components,
#             )
#         ),
#         theta0,
#         method="L-BFGS-B",
#         bounds=bounds,
#     )

#     theta_best = result.x

#     gp = build_multi_sho_gp(
#         jnp.asarray(theta_best),
#         jnp.asarray(x),
#         jnp.asarray(yerr),
#         n_components,
#     )

#     loglike = float(gp.log_probability(jnp.asarray(y)))

#     return {
#         "theta": theta_best,
#         "gp": gp,
#         "result": result,
#         "loglike": loglike,
#         "n_components": n_components,
#     }


def fit_multi_sho_gp(
    x,
    y,
    yerr,
    period_guesses,
    min_period=0.05,
    max_period=50.0,
    Q_min=0.51,
    Q_max=100.0,
    init_Q=10.0,
):
    n_components = len(period_guesses)

    theta0 = make_initial_theta(period_guesses, y, yerr, init_Q=init_Q)
    bounds = make_bounds(
        n_components,
        min_period=min_period,
        max_period=max_period,
        Q_min=Q_min,
        Q_max=Q_max,
    )

    # Convert data to JAX arrays ONCE, not inside every likelihood call
    x_j = jnp.asarray(x)
    y_j = jnp.asarray(y)
    yerr_j = jnp.asarray(yerr)

    # JIT-compiled objective
    @jax.jit
    def objective(theta):
        return neg_log_likelihood(theta, x_j, y_j, yerr_j, n_components)

    # JIT-compiled value + gradient
    value_and_grad = jax.jit(jax.value_and_grad(objective))

    # scipy-compatible wrapper
    def scipy_objective(theta_np):
        val, grad = value_and_grad(jnp.asarray(theta_np))
        return float(val), np.asarray(grad, dtype=float)

    result = minimize(
        scipy_objective,
        theta0,
        method="L-BFGS-B",
        jac=True,      # THIS IS THE IMPORTANT CHANGE
        bounds=bounds,
        options={
            "maxiter": 1000,
            "ftol": 1e-8,
            "gtol": 1e-6,
            "maxls": 20,
        },
    )

    theta_best = result.x

    gp = build_multi_sho_gp(
        jnp.asarray(theta_best),
        x_j,
        yerr_j,
        n_components,
    )

    loglike = float(gp.log_probability(y_j))

    return {
        "theta": theta_best,
        "gp": gp,
        "result": result,
        "loglike": loglike,
        "n_components": n_components,
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
    flux_err=None,
    min_period=0.05,
    max_period=50.0,
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
    min_period=0.05,
    max_period=50.0,
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


def summarize_components(theta, n_components):
    comps, jitter = unpack_theta(jnp.asarray(theta), n_components)

    out = {}

    for i, c in enumerate(comps):
        out[f"sho_{i+1}_sigma"] = float(c["sigma"])
        out[f"sho_{i+1}_period_days"] = float(c["period"])
        out[f"sho_{i+1}_omega"] = float(c["omega"])
        out[f"sho_{i+1}_Q"] = float(c["Q"])

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
        masked_power = np.where(mask, power, -np.inf)
        idx = int(np.argmax(masked_power))

        if not mask[idx] or power[idx] <= white_noise_threshold:
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

def _save_checkpoint(checkpoint_dir, m, t, x, y, yerr, fits, residual_lsps,
                     white_noise_info, initial_lsp, all_peaks, n):
    """Save a partial GPFitResult (best-BIC of completed fits) to checkpoint_dir."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    bics = [f["bic"] for f in fits]
    best_idx = int(np.argmin(bics))
    best_fit = fits[best_idx]

    final_resid, final_mu, final_std = compute_gp_residuals(best_fit, x, y)
    final_features = summarize_components(best_fit["theta"], best_fit["n_components"])
    final_features.update({
        "n_gp_components": best_fit["n_components"],
        "gp_log_likelihood": best_fit["loglike"],
        "gp_bic": best_fit["bic"],
        "gp_residual_std": float(np.nanstd(final_resid)),
        "gp_residual_rms": float(np.sqrt(np.nanmean(final_resid**2))),
        "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
        "initial_lsp_period_days": initial_lsp["best_period"],
        "initial_lsp_power": initial_lsp["best_power"],
        "initial_lsp_fap": initial_lsp["fap"],
        "initial_lsp_n_peaks": n,
        "initial_lsp_peak_periods": [p["period"] for p in all_peaks[:n]],
    })

    result = GPFitResult(
        t=t, x=x, y=y, yerr=yerr,
        gp_mean=final_mu, gp_std=final_std, residual=final_resid,
        features=final_features,
        final_fit=best_fit,
        all_accepted_fits=fits,
        residual_lsps=residual_lsps,
        white_noise_info=white_noise_info,
        all_fits=fits,
        all_residual_lsps=residual_lsps,
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
    """Load a saved (subset_table, list[GPFitResult]) pair.

    Returns
    -------
    subset_table : astropy Table or pandas DataFrame (whatever was saved)
    results : list of GPFitResult
    """
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data["subset_table"], data["results"]


def load_gp_checkpoint(path):
    """Load a single inner checkpoint saved by iterative_sho_gp_fit.

    Returns
    -------
    GPFitResult — best-BIC model from the fits completed at the time of the checkpoint.
                  Equivalent to the function's return value if fitting had stopped then.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ============================================================
# 10. Main exhaustive-search GP fit
# ============================================================

def iterative_sho_gp_fit(
    t,
    norm_flux,
    flux_err,
    max_components=6,
    min_period=0.05,
    max_period=50.0,
    samples_per_peak=10,
    Q_min=0.51,
    Q_max=100.0,
    init_Q=10.0,
    harmonic_tolerance=0.10,
    harmonic_masking=True,
    fit_full_only=False,
    bic_improvement_threshold=10.0,
    debug_fitalln=False,
    n_white_noise_boot=200,
    white_noise_percentile=99.0,
    random_state=123,
    verbose=0,
    checkpoint_dir=None,
):
    """
    Exhaustive-search multi-component SHO GP fit.

    Phase 1: compute LSP once on the original LC; identify all n significant
    peaks (above white-noise bootstrap threshold).  When harmonic_masking=True
    (default), harmonics and sub-harmonics of each found peak are suppressed
    before searching for the next one.  Set harmonic_masking=False to allow all
    peaks above the threshold regardless of harmonic relationships.

    Phase 2: for m = 1 … n, fit an m-component GP seeded by the top-m peaks.
    BIC early-stopping is active by default; set debug_fitalln=True to fit all
    n models regardless.

    The returned GPFitResult reports the best-BIC model as the primary result,
    but stores all attempted fits in all_fits for inspection.
    """

    t0_total = time.perf_counter()
    t_clean, x, y, yerr = clean_lc(t, norm_flux, flux_err)

    t0 = time.perf_counter()
    white_noise_info = estimate_lsp_white_noise_level(
        t_clean,
        yerr,
        min_period=min_period,
        max_period=max_period,
        samples_per_peak=samples_per_peak,
        n_boot=n_white_noise_boot,
        percentile=white_noise_percentile,
        random_state=random_state,
    )
    if verbose >= 1:
        print(f"White-noise bootstrap: {time.perf_counter() - t0:.1f} s")

    # Compute LSP once on original data
    t0 = time.perf_counter()
    initial_lsp = residual_lsp_peak(
        t_clean,
        y,
        flux_err=yerr,
        min_period=min_period,
        max_period=max_period,
        samples_per_peak=samples_per_peak,
    )
    if verbose >= 1:
        print(f"Initial LSP: {time.perf_counter() - t0:.1f} s")

    all_peaks, masked_windows = find_all_significant_peaks(
        initial_lsp["freq"],
        initial_lsp["power"],
        white_noise_info["white_noise_power_threshold"],
        harmonic_tolerance=harmonic_tolerance,
        harmonic_masking=harmonic_masking,
    )

    n = min(len(all_peaks), max_components)

    if n == 0:
        if verbose >= 1:
            print("No significant LSP peaks found above white-noise threshold.")
        return None

    if verbose >= 1:
        print(
            f"Found {len(all_peaks)} significant peak(s); "
            f"fitting up to {n} component(s)."
        )
        for i, p in enumerate(all_peaks[:n]):
            print(f"  Peak {i+1}: period={p['period']:.3f} d, power={p['power']:.4f}")

    if verbose >= 2:
        from visualize_gp_duringfit import plot_initial_lsp
        plot_initial_lsp(
            initial_lsp["freq"],
            initial_lsp["power"],
            all_peaks[:n],
            masked_windows,
            white_noise_info,
        )

    fits = []
    residual_lsps = []
    prev_bic = None

    m_range = [n] if fit_full_only else range(1, n + 1)

    for m in m_range:
        period_guesses = [all_peaks[i]["period"] for i in range(m)]

        if verbose >= 1:
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
        )
        if verbose >= 1:
            print(f"  Fit {m} component(s): {time.perf_counter() - t0:.1f} s")

        n_params = 3 * m + 1
        fit["bic"] = compute_bic(fit["loglike"], n_params=n_params, n_data=len(y))

        resid, mu, std = compute_gp_residuals(fit, x, y)

        resid_lsp = residual_lsp_peak(
            t_clean,
            resid,
            flux_err=yerr,
            min_period=min_period,
            max_period=max_period,
            samples_per_peak=samples_per_peak,
        )

        fits.append(fit)
        residual_lsps.append(resid_lsp)

        if checkpoint_dir is not None:
            _save_checkpoint(
                checkpoint_dir, m, t_clean, x, y, yerr,
                fits, residual_lsps, white_noise_info,
                initial_lsp, all_peaks, n,
            )

        if verbose >= 1:
            comps_m, _ = unpack_theta(fit["theta"], m)
            period_list = [f"{float(c['period']):.3f}" for c in comps_m]
            print(
                f"  {m} component(s): periods={period_list} d, "
                f"BIC={fit['bic']:.1f}"
            )

        if verbose >= 2:
            from visualize_gp_duringfit import plot_fit_summary
            plot_fit_summary(
                x, y, yerr, fits, m,
                resid, resid_lsp, white_noise_info, fit["bic"],
            )

        # BIC early stopping (bypassed in debug mode)
        if not debug_fitalln and prev_bic is not None:
            delta_bic = prev_bic - fit["bic"]
            if delta_bic < bic_improvement_threshold:
                if verbose >= 1:
                    print(
                        f"  BIC improvement {delta_bic:.1f} < "
                        f"threshold ({bic_improvement_threshold:.1f}). Stopping."
                    )
                break

        prev_bic = fit["bic"]

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
                symbol = "✓" if delta >= bic_improvement_threshold else "~"
                delta_str = f"  ΔBIC={delta:+.1f} {symbol}"
            best_str = "  ← best" if is_best else ""
            print(f"  m={m_j}: BIC={bic_j:.1f}{delta_str}{best_str}")
        print()

    if verbose >= 1:
        print(
            f"Best model: {best_fit['n_components']} component(s) "
            f"(BIC={best_fit['bic']:.1f})"
        )
        print(f"Total: {time.perf_counter() - t0_total:.1f} s")

    final_resid, final_mu, final_std = compute_gp_residuals(best_fit, x, y)

    final_features = summarize_components(best_fit["theta"], best_fit["n_components"])
    final_features.update({
        "n_gp_components": best_fit["n_components"],
        "gp_log_likelihood": best_fit["loglike"],
        "gp_bic": best_fit["bic"],
        "gp_residual_std": float(np.nanstd(final_resid)),
        "gp_residual_rms": float(np.sqrt(np.nanmean(final_resid**2))),
        "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
        "initial_lsp_period_days": initial_lsp["best_period"],
        "initial_lsp_power": initial_lsp["best_power"],
        "initial_lsp_fap": initial_lsp["fap"],
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
        residual_lsps=residual_lsps,
        white_noise_info=white_noise_info,
        all_fits=fits,
        all_residual_lsps=residual_lsps,
        peaks=all_peaks,
    )

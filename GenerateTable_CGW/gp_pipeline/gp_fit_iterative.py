import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from astropy.timeseries import LombScargle
from tinygp import GaussianProcess
from tinygp.kernels import quasisep

from .gp_fit_result import GPFitResult

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


def make_bounds(n_components, min_period, max_period):
    bounds = []

    for _ in range(n_components):
        bounds.extend([
            (np.log(1e-10), np.log(1.0)),       # sigma
            (np.log(min_period), np.log(max_period)),  # period
            (np.log(0.51), np.log(100.0)),      # Q
        ])

    bounds.append((np.log(1e-10), np.log(1.0)))  # jitter

    return bounds


def fit_multi_sho_gp(
    x,
    y,
    yerr,
    period_guesses,
    min_period,
    max_period,
    init_Q=2.0,
):
    n_components = len(period_guesses)

    theta0 = make_initial_theta(period_guesses, y, yerr, init_Q=init_Q)
    bounds = make_bounds(
        n_components,
        min_period=min_period,
        max_period=max_period,
    )

    result = minimize(
        lambda p: np.asarray(
            neg_log_likelihood(
                jnp.asarray(p),
                jnp.asarray(x),
                jnp.asarray(y),
                jnp.asarray(yerr),
                n_components,
            )
        ),
        theta0,
        method="L-BFGS-B",
        bounds=bounds,
    )

    theta_best = result.x

    gp = build_multi_sho_gp(
        jnp.asarray(theta_best),
        jnp.asarray(x),
        jnp.asarray(yerr),
        n_components,
    )

    loglike = float(gp.log_probability(jnp.asarray(y)))

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
# 5. LSP peak after subtracting GP model
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
# 6. White-noise level / stopping rule
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
    Estimates a white-noise LSP peak threshold.

    It generates pure Gaussian noise using the measured flux errors,
    computes the maximum LSP power for each realization, and returns
    a high percentile of those maxima.

    If residual max power is below this threshold, treat residual LSP
    as white-noise dominated.
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


def should_add_component(
    residual_lsp,
    white_noise_info,
    fap_threshold=0.01,
):
    power_ok = (
        residual_lsp["best_power"]
        > white_noise_info["white_noise_power_threshold"]
    )

    fap = residual_lsp["fap"]
    fap_ok = np.isfinite(fap) and (fap < fap_threshold)

    return bool(power_ok and fap_ok)


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
# 8. Main iterative Approach B
# ============================================================

def iterative_sho_gp_fit(
    t,
    norm_flux,
    flux_err,
    max_components=3,
    min_period=1/24,
    max_period=10.0,
    samples_per_peak=10,
    fap_threshold=0.01,
    bic_improvement_threshold=10.0,
    n_white_noise_boot=200,
    white_noise_percentile=99.0,
    random_state=123,
    verbose=0,
):
    """
    Approach B:
    1. Start from strongest LSP peak in the original LC.
    2. Fit 1-component SHO GP.
    3. Subtract GP posterior mean.
    4. Run LSP on residuals.
    5. If residual peak is above white-noise threshold, add a component.
    6. Jointly refit all components.
    7. Accept the new model only if BIC improves sufficiently.
    8. Repeat.
    """

    t_clean, x, y, yerr = clean_lc(t, norm_flux, flux_err)

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

    # Initial peak from original data
    initial_lsp = residual_lsp_peak(
        t_clean,
        y,
        flux_err=yerr,
        min_period=min_period,
        max_period=max_period,
        samples_per_peak=samples_per_peak,
    )

    period_guesses = [initial_lsp["best_period"]]

    fits = []
    residual_lsps = []

    if verbose >= 1:
        print("Fitting component 1...")
    if verbose >= 2:
        from visualize_gp_duringfit import plot_before_fit
        plot_before_fit(x, y, yerr, initial_lsp, white_noise_info, component_idx=1)

    # Fit M = 1
    current_fit = fit_multi_sho_gp(
        x,
        y,
        yerr,
        period_guesses,
        min_period=min_period,
        max_period=max_period,
    )

    n_params = 3 * current_fit["n_components"] + 1
    current_fit["bic"] = compute_bic(
        current_fit["loglike"],
        n_params=n_params,
        n_data=len(y),
    )

    fits.append(current_fit)

    if verbose >= 1:
        comps_1, _ = unpack_theta(current_fit["theta"], 1)
        print(
            f"  Component 1 accepted: period={float(comps_1[0]['period']):.2f} days, "
            f"Q={float(comps_1[0]['Q']):.2f}, BIC={current_fit['bic']:.1f}"
        )
    if verbose >= 2:
        from visualize_gp_duringfit import plot_after_fit
        plot_after_fit(x, y, yerr, fits, component_idx=1)

    # Iterative additions
    for m in range(2, max_components + 1):
        if verbose >= 1:
            print(f"Checking for component {m}...")

        resid, mu, std = compute_gp_residuals(current_fit, x, y)

        res_lsp = residual_lsp_peak(
            t_clean,
            resid,
            flux_err=yerr,
            min_period=min_period,
            max_period=max_period,
            samples_per_peak=samples_per_peak,
        )

        residual_lsps.append(res_lsp)

        if verbose >= 2:
            from visualize_gp_duringfit import plot_before_fit
            plot_before_fit(x, resid, yerr, res_lsp, white_noise_info, component_idx=m)

        add_more = should_add_component(
            res_lsp,
            white_noise_info,
            fap_threshold=fap_threshold,
        )

        if not add_more:
            if verbose >= 1:
                print(
                    f"  No significant residual peak (FAP={res_lsp['fap']:.4f}, "
                    f"power={res_lsp['best_power']:.4f} < "
                    f"threshold={white_noise_info['white_noise_power_threshold']:.4f}). Stopping."
                )
            break

        if verbose >= 1:
            print(
                f"  Residual peak significant at period={res_lsp['best_period']:.2f} days "
                f"(FAP={res_lsp['fap']:.4f}). Fitting..."
            )

        candidate_periods = period_guesses + [res_lsp["best_period"]]

        candidate_fit = fit_multi_sho_gp(
            x,
            y,
            yerr,
            candidate_periods,
            min_period=min_period,
            max_period=max_period,
        )

        n_params = 3 * candidate_fit["n_components"] + 1
        candidate_fit["bic"] = compute_bic(
            candidate_fit["loglike"],
            n_params=n_params,
            n_data=len(y),
        )

        delta_bic = current_fit["bic"] - candidate_fit["bic"]
        candidate_fit["delta_bic_vs_previous"] = float(delta_bic)

        if delta_bic < bic_improvement_threshold:
            if verbose >= 1:
                print(
                    f"  Component {m} rejected: BIC improvement {delta_bic:.1f} < "
                    f"threshold ({bic_improvement_threshold:.1f}). Stopping."
                )
            break

        current_fit = candidate_fit
        period_guesses = candidate_periods
        fits.append(current_fit)

        if verbose >= 1:
            comps_m, _ = unpack_theta(current_fit["theta"], current_fit["n_components"])
            print(
                f"  Component {m} accepted: period={float(comps_m[-1]['period']):.2f} days, "
                f"BIC improvement={delta_bic:.1f}"
            )
        if verbose >= 2:
            from visualize_gp_duringfit import plot_after_fit
            plot_after_fit(x, y, yerr, fits, component_idx=m)

    if verbose >= 1:
        final_n = current_fit["n_components"]
        final_comps, _ = unpack_theta(current_fit["theta"], final_n)
        period_list = [f"{float(c['period']):.2f}" for c in final_comps]
        print(f"Final model: {final_n} component(s). Periods: {period_list} days")

    final_resid, final_mu, final_std = compute_gp_residuals(current_fit, x, y)

    final_features = summarize_components(
        current_fit["theta"],
        current_fit["n_components"],
    )

    final_features.update({
        "n_gp_components": current_fit["n_components"],
        "gp_log_likelihood": current_fit["loglike"],
        "gp_bic": current_fit["bic"],
        "gp_residual_std": float(np.nanstd(final_resid)),
        "gp_residual_rms": float(np.sqrt(np.nanmean(final_resid**2))),
        "white_noise_power_threshold": white_noise_info["white_noise_power_threshold"],
        "initial_lsp_period_days": initial_lsp["best_period"],
        "initial_lsp_power": initial_lsp["best_power"],
        "initial_lsp_fap": initial_lsp["fap"],
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
        final_fit=current_fit,
        all_accepted_fits=fits,
        residual_lsps=residual_lsps,
        white_noise_info=white_noise_info,
    )
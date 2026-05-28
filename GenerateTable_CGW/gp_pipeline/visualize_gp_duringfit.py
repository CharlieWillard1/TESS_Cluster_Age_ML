import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from .gp_fit import build_multi_sho_gp, predict_gp_at_training_times


def _get_full_gp_mean(fit, x, y):
    mu, _ = predict_gp_at_training_times(fit, x, y)
    return mu


def _get_component_contribution(theta, component_idx, x, y, yerr):
    """
    Isolate component j's posterior mean from a joint M-component fit.

    Uses sequential conditioning: build (M-1)-component GP for the other
    components, subtract their posterior mean from y, then condition
    component j on the remainder.
    """
    theta = np.asarray(theta)
    n_components = (len(theta) - 1) // 3
    j = component_idx - 1  # 0-indexed
    log_jitter = theta[-1]

    # Sub-theta for component j only (+ shared jitter)
    theta_j = np.array([theta[3*j], theta[3*j + 1], theta[3*j + 2], log_jitter])

    # Sub-theta for all other components (+ shared jitter)
    other_pieces = []
    for k in range(n_components):
        if k == j:
            continue
        other_pieces.extend([theta[3*k], theta[3*k + 1], theta[3*k + 2]])
    other_pieces.append(log_jitter)
    theta_others = np.array(other_pieces)
    n_others = n_components - 1

    if n_others == 0:
        y_j = np.asarray(y)
    else:
        gp_others = build_multi_sho_gp(
            jnp.asarray(theta_others), jnp.asarray(x), jnp.asarray(yerr), n_others
        )
        mu_others = np.asarray(
            gp_others.condition(jnp.asarray(y), jnp.asarray(x)).gp.mean
        )
        y_j = np.asarray(y) - mu_others

    gp_j = build_multi_sho_gp(
        jnp.asarray(theta_j), jnp.asarray(x), jnp.asarray(yerr), 1
    )
    mu_j = np.asarray(gp_j.condition(jnp.asarray(y_j), jnp.asarray(x)).gp.mean)
    return mu_j


def plot_initial_lsp(freq, power, peaks, masked_windows, white_noise_info,
                     peak_threshold=None):
    """
    Show the initial LSP (frequency domain) before any fitting, with:
      - noise threshold as a dashed line (horizontal for scalar, plotted curve for array)
      - found significant peaks marked with vertical lines
      - masked harmonic regions shaded in translucent red

    Parameters
    ----------
    freq : array
        Frequency array from the LSP.
    power : array
        LSP power array.
    peaks : list of dicts
        [{freq, period, power}, ...] — the significant peaks that will be seeded.
    masked_windows : list of (f_lo, f_hi) tuples
        Frequency intervals that were suppressed during peak search.
    white_noise_info : dict
        Must contain 'white_noise_power_threshold' and optionally 'threshold_mode'.
    peak_threshold : float or array-like or None
        The actual threshold used for peak detection.  If an ndarray (red_noise mode),
        plotted as a curve over freq; if scalar or None, plotted as axhline.
    """
    freq  = np.asarray(freq)
    power = np.asarray(power)

    threshold_mode = white_noise_info.get("threshold_mode", "white_noise")
    _mode_labels = {
        'white_noise': 'WN threshold (99th pct)',
        'fap':         '1% FAP threshold',
        'red_noise':   'Red-noise 1% threshold',
    }
    thresh_label = _mode_labels.get(threshold_mode, 'Threshold')

    thresh_is_array = (peak_threshold is not None and not np.isscalar(peak_threshold))
    if thresh_is_array:
        thresh_arr = np.asarray(peak_threshold)
    else:
        thresh_scalar = float(peak_threshold) if peak_threshold is not None \
            else white_noise_info["white_noise_power_threshold"]

    colors_peaks = plt.cm.plasma(np.linspace(0.15, 0.85, max(len(peaks), 1)))

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(18, 4))
    fig.suptitle("Initial LSP — significant peaks and masked regions", fontsize=12)

    for ax, log_y in ((ax_lin, False), (ax_log, True)):
        ax.plot(freq, power, color='steelblue', lw=0.9, alpha=0.9)

        if thresh_is_array:
            ax.plot(freq, thresh_arr, color='tomato', lw=1.5, ls='--',
                    label=thresh_label)
        else:
            ax.axhline(thresh_scalar, color='tomato', lw=1.5, ls='--',
                       label=f"{thresh_label} = {thresh_scalar:.4f}")

        added_label = False
        for f_lo, f_hi in masked_windows:
            label = "Masked regions" if not added_label else None
            ax.axvspan(f_lo, f_hi, color='tomato', alpha=0.12, lw=0, label=label)
            added_label = True

        for i, (peak, c) in enumerate(zip(peaks, colors_peaks)):
            ax.axvline(
                peak["freq"], color=c, lw=1.3, ls=':',
                label=f"Peak {i+1}: f={peak['freq']:.3f} /d  ({peak['period']:.3f} d)  power={peak['power']:.4f}",
            )

        ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel("Frequency (1/day)")
        ax.set_ylabel("LSP power")
        ax.set_title("log x" if not log_y else "log-log")
        ax.legend(fontsize=8, loc='lower left')

    fig.tight_layout()
    plt.show()


def plot_before_fit(x, y, yerr, lsp_dict, white_noise_info, component_idx, max_days=30):
    """
    2-panel diagnostic plot before fitting component_idx.

    Left:  LC (first max_days days) with error bars.
    Right: LSP power vs period with white-noise threshold line.

    For component 1 pass original (x, y); for component m > 1 pass (x, residual).
    """
    fig, (ax_lc, ax_lsp) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Before Component {component_idx} Fit", fontsize=13)

    mask = x <= max_days
    ax_lc.errorbar(
        x[mask], np.asarray(y)[mask], yerr=np.asarray(yerr)[mask],
        fmt='.', color='steelblue', alpha=0.5, ms=3, elinewidth=0.5
    )
    ax_lc.set_xlabel("Time (days)")
    ax_lc.set_ylabel("Flux" if component_idx == 1 else "Residual flux")
    ax_lc.set_title("Light curve (first 30 days)")

    freq    = np.asarray(lsp_dict["freq"])
    power   = np.asarray(lsp_dict["power"])
    period  = 1.0 / freq
    threshold = white_noise_info["white_noise_power_threshold"]

    sort_idx = np.argsort(period)
    ax_lsp.plot(period[sort_idx], power[sort_idx], color='steelblue', lw=0.8, alpha=0.8)
    ax_lsp.axhline(
        threshold, color='tomato', lw=1.5, ls='--',
        label=f"WN threshold = {threshold:.4f}"
    )
    ax_lsp.axvline(
        lsp_dict["best_period"], color='orange', lw=1.2, ls=':',
        label=f"Peak = {lsp_dict['best_period']:.3f} d  (FAP={lsp_dict['fap']:.4f})"
    )
    ax_lsp.set_xscale("log")
    ax_lsp.set_xlabel("Period (days)")
    ax_lsp.set_ylabel("LSP power")
    ax_lsp.set_title("Lomb-Scargle periodogram")
    ax_lsp.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_after_fit(x, y, yerr, all_accepted_fits, component_idx, max_days=30):
    """
    Component-wise breakdown after fitting component_idx components.

    Row 1:      raw data + full GP mean (all components).
    Row j > 1:  (y - GP mean of j-1 components) + component j's contribution.
    """
    n_rows = component_idx
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), squeeze=False)
    axes = axes[:, 0]

    mask = x <= max_days
    final_fit = all_accepted_fits[-1]
    theta_final = np.asarray(final_fit["theta"])
    n_comp_final = final_fit["n_components"]

    # Row 1: raw data + full GP mean
    mu_full = _get_full_gp_mean(final_fit, x, y)
    ax = axes[0]
    ax.errorbar(
        x[mask], np.asarray(y)[mask], yerr=np.asarray(yerr)[mask],
        fmt='.', color='steelblue', alpha=0.4, ms=3, elinewidth=0.5, label='Data'
    )
    ax.plot(
        x[mask], mu_full[mask],
        color='tomato', lw=1.5, label=f'GP mean ({n_comp_final} components)'
    )
    ax.set_ylabel("Flux")
    ax.set_title(f"Full GP fit ({n_comp_final} components)")
    ax.legend(fontsize=8)

    # Rows j > 1: residual after j-1 components + component j
    for j in range(2, n_rows + 1):
        prev_fit = all_accepted_fits[j - 2]
        mu_prev = _get_full_gp_mean(prev_fit, x, y)
        y_resid = np.asarray(y) - mu_prev

        mu_j = _get_component_contribution(theta_final, j, x, y, yerr)

        ax = axes[j - 1]
        ax.errorbar(
            x[mask], y_resid[mask], yerr=np.asarray(yerr)[mask],
            fmt='.', color='steelblue', alpha=0.4, ms=3, elinewidth=0.5,
            label=f'Residual after {j-1} comp.'
        )
        ax.plot(x[mask], mu_j[mask], color='tomato', lw=1.5, label=f'Component {j}')
        ax.set_ylabel("Flux")
        ax.set_title(f"Component {j} isolated")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"After Component {component_idx} Fit", fontsize=13)
    fig.tight_layout()
    plt.show()


def plot_fit_summary(x, y, yerr, fits, component_idx, bic, max_days=30):
    """
    Component breakdown after fitting component_idx components.

    Row 1: raw data + full GP mean.
    Row j > 1: residual after j-1 components + component j's contribution.
    BIC is shown in the figure title.
    """
    n_rows = component_idx
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3.5 * n_rows), squeeze=False)
    axes = axes[:, 0]

    mask = x <= max_days
    final_fit = fits[-1]
    theta_final = np.asarray(final_fit["theta"])
    mu_full = _get_full_gp_mean(final_fit, x, y)

    axes[0].errorbar(
        x[mask], np.asarray(y)[mask], yerr=np.asarray(yerr)[mask],
        fmt='.', color='steelblue', alpha=0.4, ms=3, elinewidth=0.5, label='Data'
    )
    axes[0].plot(
        x[mask], mu_full[mask],
        color='tomato', lw=1.5, label=f'GP mean ({component_idx} components)'
    )
    axes[0].set_ylabel("Flux")
    axes[0].set_title(f"Full GP fit ({component_idx} components) | BIC = {bic:.1f}", fontsize=11)
    axes[0].legend(fontsize=8)

    for j in range(2, n_rows + 1):
        prev_fit = fits[j - 2]
        mu_prev = _get_full_gp_mean(prev_fit, x, y)
        y_resid_j = np.asarray(y) - mu_prev
        mu_j = _get_component_contribution(theta_final, j, x, y, yerr)

        axes[j - 1].errorbar(
            x[mask], y_resid_j[mask], yerr=np.asarray(yerr)[mask],
            fmt='.', color='steelblue', alpha=0.4, ms=3, elinewidth=0.5,
            label=f'Residual after {j-1} comp.'
        )
        axes[j - 1].plot(x[mask], mu_j[mask], color='tomato', lw=1.5, label=f'Component {j}')
        axes[j - 1].set_ylabel("Flux")
        axes[j - 1].set_title(f"Component {j} isolated", fontsize=10)
        axes[j - 1].legend(fontsize=8)

    axes[-1].set_xlabel("Time (days)")
    fig.suptitle(f"GP Fit Summary — {component_idx} component(s)", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()

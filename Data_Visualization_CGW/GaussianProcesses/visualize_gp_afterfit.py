import numpy as np
import matplotlib.pyplot as plt
import jax


# ============================================================
# Helpers
# ============================================================

def _to_list(results, table_rows):
    """Normalize single-item or list inputs to lists."""
    if not isinstance(results, list):
        results = [results]
    if table_rows is not None:
        if hasattr(table_rows, "iloc"):          # pandas DataFrame
            table_rows = [table_rows.iloc[i] for i in range(len(table_rows))]
        elif not isinstance(table_rows, list):   # single row
            table_rows = [table_rows]
    return results, table_rows


def _gp_mean_at_x(fit, x_pred, y):
    """Posterior mean of fit's GP at x_pred given observations y at training points."""
    import jax.numpy as jnp
    cond_gp = fit["gp"].condition(jnp.asarray(y), jnp.asarray(x_pred)).gp
    return np.asarray(cond_gp.mean)


def _label(table_row, idx):
    """Build a plot label from a master-table row, falling back to index."""
    if table_row is None:
        return f"Star {idx + 1}"
    name = table_row["name"]
    if isinstance(name, bytes):
        name = name.decode()
    sectors = table_row["sectors"]
    return f"{name} | sectors={list(sectors)}"


# ============================================================
# Public plotting functions
# ============================================================

def plot_gp_fit(results, table_rows=None, plot_all_ncomponent_fits=False):
    """
    Plot GP fit vs light curve data for one or more GPFitResult objects.

    Layout
    ------
    First n rows : individual LC (data + GP mean + ±1σ band), one per result.
    Last row     : all GP means overlaid on a single axes, no LC scatter.

    Parameters
    ----------
    results : GPFitResult or list of GPFitResult
    table_rows : table row or list of table rows, optional
        Master-table rows used for axis titles/labels.
        Each row must support dict-style access with keys 'name' and 'sectors'.
    plot_all_ncomponent_fits : bool
        If True, overlay the lower-m fits from result.all_fits as dashed same-color lines.
    """
    results, table_rows = _to_list(results, table_rows)
    n = len(results)
    n_rows = n + 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.5 * n_rows), squeeze=False)
    axes = axes[:, 0]

    for i, res in enumerate(results):
        ax = axes[i]
        label = _label(table_rows[i] if table_rows else None, i)

        ax.errorbar(
            res.t, res.y, yerr=res.yerr,
            fmt='.', color='black', alpha=0.3, ms=3, elinewidth=0.5,
            label='Data',
        )

        if plot_all_ncomponent_fits:
            for fit in res.all_fits:
                m = fit["n_components"]
                if m == res.n_components:
                    continue
                mean = _gp_mean_at_x(fit, res.x, res.y)
                ax.plot(res.t, mean, color='red', lw=1.0, ls='--', alpha=0.5,
                        label=f"GP mean (m={m})")

        best_label = f"GP mean (m={res.n_components}, best)" if plot_all_ncomponent_fits else "GP mean"
        ax.plot(res.t, res.gp_mean, color='red', lw=1.5, label=best_label)
        ax.fill_between(
            res.t,
            res.gp_mean - res.gp_std,
            res.gp_mean + res.gp_std,
            alpha=0.3, color='blue', label='GP ±1σ',
        )
        ax.set_xlim(res.t.min(), res.t.min() + 27)
        ax.set_ylabel("Relative flux")
        ax.set_title(label)
        ax.legend(fontsize=8)

    # Final row: all GP means overlaid
    ax = axes[-1]
    for i, res in enumerate(results):
        label = _label(table_rows[i] if table_rows else None, i)
        line, = ax.plot(res.t, res.gp_mean, lw=1.2, alpha=0.85, label=label)
        ax.set_xlim(res.t.min(), res.t.min() + 27)

        if plot_all_ncomponent_fits:
            c = line.get_color()
            for fit in res.all_fits:
                m = fit["n_components"]
                if m == res.n_components:
                    continue
                mean = _gp_mean_at_x(fit, res.x, res.y)
                ax.plot(res.t, mean, color=c, lw=0.9, ls='--', alpha=0.5,
                        label=f"{label} m={m}")

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Relative flux")
    ax.set_title("All GP means")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def _cluster_color_map(table_rows):
    """
    One tab10 color per unique cluster name.  Returns None when there is only
    one cluster (caller falls back to matplotlib's default color cycle).
    """
    if not table_rows:
        return None
    names = []
    for row in table_rows:
        name = row["name"] if row is not None else None
        if isinstance(name, bytes):
            name = name.decode()
        names.append(name)
    unique = list(dict.fromkeys(n for n in names if n is not None))
    if len(unique) <= 1:
        return None
    palette = {n: plt.cm.tab10(i % 10) for i, n in enumerate(unique)}
    return [palette.get(n) for n in names]


def _analytic_psd(fit, freq_min, freq_max, n_freq):
    """Analytic SHO kernel PSD for a single fit dict."""
    from GP_Fit import unpack_theta
    import jax.numpy as jnp
    freq  = np.linspace(freq_min, freq_max, n_freq)
    omega = 2.0 * np.pi * freq
    comps, _ = unpack_theta(jnp.asarray(fit["theta"]), fit["n_components"])
    power = np.zeros(n_freq)
    for comp in comps:
        sigma  = float(comp["sigma"])
        omega0 = float(comp["omega"])
        Q      = float(comp["Q"])
        numer  = sigma**2 * (omega0 / Q) * (omega0**2 + omega**2)
        denom  = (omega**2 - omega0**2)**2 + omega**2 * omega0**2 / Q**2
        power += numer / denom
    return freq, power


def _realized_psd(fit, res, n_uniform, subtract_mean, window):
    """FFT power spectrum of the posterior mean for a specific fit dict."""
    import jax.numpy as jnp
    x_uniform = np.linspace(0.0, float(res.x.max()), n_uniform)
    cond_gp = fit["gp"].condition(jnp.asarray(res.y), jnp.asarray(x_uniform)).gp
    gp_u = np.array(cond_gp.mean)
    if subtract_mean:
        gp_u -= np.nanmean(gp_u)
    dt = x_uniform[1] - x_uniform[0]
    fft_input = gp_u * np.hanning(n_uniform) if window else gp_u
    pwr  = np.abs(np.fft.rfft(fft_input)) ** 2
    freq = np.fft.rfftfreq(n_uniform, d=dt)
    return freq[1:], pwr[1:]   # skip DC


def _fits_to_plot(res, plot_components, base_label):
    """
    Return list of (fit_dict, linestyle, legend_label) for the kspace panels.

    plot_components options
    -----------------------
    'best' : BIC-optimal model only, solid line.
    'max'  : highest-m model only, solid line.
    'main' : best (solid) + max (dashed), same color.  Single entry if best==max.
    'all'  : best (solid) + every other m (dashed), same color, labeled by m.
    """
    best_m  = res.n_components
    max_fit = max(res.all_fits, key=lambda f: f["n_components"])
    max_m   = max_fit["n_components"]

    # Each tuple: (fit_dict, linestyle, legend_label, color_override)
    # color_override=None → caller uses the per-result cluster/cycle color.

    if plot_components == 'best':
        return [(res.final_fit, '-', base_label, None)]

    if plot_components == 'max':
        return [(max_fit, '-', base_label, None)]

    if plot_components == 'main':
        if best_m == max_m:
            return [(res.final_fit, '-', base_label, None)]
        return [
            (res.final_fit, '-',  f"{base_label}  m={best_m} (best)", None),
            (max_fit,       '--', f"{base_label}  m={max_m} (max)",   None),
        ]

    if plot_components == 'all':
        fits_sorted = sorted(res.all_fits, key=lambda f: f["n_components"])
        n = len(fits_sorted)
        out = []
        for idx, fit in enumerate(fits_sorted):
            m     = fit["n_components"]
            ls    = '-' if m == best_m else '--'
            lbl   = f"{base_label}  m={m}" if m == best_m else f"m={m}"
            color = plt.cm.tab10(idx % 10)
            out.append((fit, ls, lbl, color))
        return out

    raise ValueError(
        f"plot_components must be 'best', 'max', 'main', or 'all', got {plot_components!r}"
    )


def plot_kspace(results, table_rows=None, log_y=False,
                n_uniform=4096, subtract_mean=True, window=True,
                plot_components='main',
                plot_realized_gp=False,
                period_lim=(1/24, 10.0)):
    """
    Single-column frequency-content figure for one or more GPFitResult objects.

    Rows
    ----
    0            : LSP of the real data (white-noise threshold as dashed line)
    1 (optional) : kspace_realized — FFT of GP posterior mean (plot_realized_gp=True)
    1 or 2       : kspace_true — analytic prior PSD of the GP kernel

    Each panel shows frequency on the bottom x-axis (log scale) and period
    on the top x-axis (derived via 1/f).

    When table_rows spans more than one unique cluster name, each cluster gets
    a distinct tab10 color; within a cluster, linestyle distinguishes models.

    Parameters
    ----------
    results : GPFitResult or list of GPFitResult
    table_rows : table row or list of table rows, optional
    log_y : bool
        Log scale on all y-axes (default False).
    n_uniform : int
        Grid points for kspace_realized.
    subtract_mean, window : bool
        Passed through to kspace_realized.
    plot_components : {'best', 'max', 'main', 'all'}
        Which component models to overlay on the kspace panels.
        'best' — BIC-optimal model, solid (default).
        'max'  — highest-m model, solid.
        'main' — best (solid) + max (dashed), same color.
        'all'  — best (solid) + every attempted m (dashed), same color.
        The LSP panel always shows one line per result.
    plot_realized_gp : bool
        Include the realized (FFT of GP posterior mean) row (default False).
    period_lim : (float, float)
        Period range in days.
    """
    from GP_Fit import residual_lsp_peak

    results, table_rows = _to_list(results, table_rows)

    period_min, period_max = period_lim
    freq_lim    = (1.0 / period_max, 1.0 / period_min)
    n_freq_true = max(2000, int(2000 * np.log10(freq_lim[1] / freq_lim[0])))

    color_map = _cluster_color_map(table_rows)  # None → matplotlib default cycle

    row_keys = ["lsp", "realized", "true"] if plot_realized_gp else ["lsp", "true"]
    titles = {
        "lsp":      "Observed-data Lomb–Scargle periodogram",
        "realized": "FFT power of GP posterior mean on uniform grid",
        "true":     "Analytic PSD of fitted GP kernel",
    }
    fft_ylabel = (
        r"Hann-windowed FFT power $|\mathrm{FFT}(w\!\cdot\!\mu_{\rm GP})|^2$ [$y^2$]"
        if window else
        r"Raw FFT power $|\mathrm{FFT}(\mu_{\rm GP})|^2$ [$y^2$]"
    )
    ylabels = {
        "lsp":      "Lomb–Scargle power [dimensionless]",
        "realized": fft_ylabel,
        "true":     r"GP kernel PSD [$y^2\,\mathrm{day}$]",
    }

    n_rows = len(row_keys)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False)
    axes = axes[:, 0]
    fig.suptitle(r"GP Model Frequency Content  ($y$ = relative flux)", fontsize=13)

    for ax in axes:
        sec = ax.secondary_xaxis('top', functions=(lambda f: 1.0 / f, lambda p: 1.0 / p))
        sec.set_xlabel("Period [days]")

    for i, res in enumerate(results):
        base_label   = _label(table_rows[i] if table_rows else None, i)
        preset_color = color_map[i] if color_map is not None else None

        # ── LSP (one line per result) ─────────────────────────────────────────
        ax_lsp    = axes[row_keys.index("lsp")]
        lsp       = residual_lsp_peak(res.t, res.y, flux_err=res.yerr,
                                      min_period=period_min, max_period=period_max)
        threshold = res.white_noise_info["white_noise_power_threshold"]
        kw = dict(lw=0.8, alpha=0.85, label=base_label)
        if preset_color is not None:
            kw["color"] = preset_color
        line, = ax_lsp.plot(np.asarray(lsp["freq"]), np.asarray(lsp["power"]), **kw)
        c = line.get_color()
        ax_lsp.axhline(threshold, color=c, lw=1.0, ls='--', alpha=0.5)

        # ── Kspace panels — iterate over fits selected by plot_components ─────
        fits_spec = _fits_to_plot(res, plot_components, base_label)

        if plot_realized_gp:
            ax_kr = axes[row_keys.index("realized")]
            for fit, ls, lbl, col in fits_spec:
                freq_r, pwr_r = _realized_psd(fit, res, n_uniform, subtract_mean, window)
                ax_kr.plot(freq_r, pwr_r, color=col if col is not None else c,
                           ls=ls, lw=0.8, alpha=0.85, label=lbl)

        ax_kt = axes[row_keys.index("true")]
        for fit, ls, lbl, col in fits_spec:
            freq_t, pwr_t = _analytic_psd(fit, freq_lim[0], freq_lim[1], n_freq_true)
            ax_kt.plot(freq_t, pwr_t, color=col if col is not None else c,
                       ls=ls, lw=0.8, alpha=0.85, label=lbl)

    # ── Axis formatting ───────────────────────────────────────────────────────
    for row_idx, key in enumerate(row_keys):
        ax = axes[row_idx]
        ax.set_xlim(*freq_lim)
        ax.set_xscale("log")
        ax.set_xlabel("Frequency [1/day]")
        ax.set_ylabel(ylabels[key])
        ax.set_title(titles[key])
        ax.legend(fontsize=8)
        if log_y:
            ax.set_yscale("log")

    fig.tight_layout()
    plt.show()
    plt.close(fig)
    jax.clear_caches()


def plot_summary_stats(table, cluster_name=None, stats=None, seed=None):
    """
    Scatter-plot GP summary statistics vs number of sectors for one or more clusters.

    Layout: N_stats rows × N_clusters columns, shared x-axis (n_sectors).
    LSP stats in red, GP kspace stats in blue, offset slightly in x.
    If n_rows_per_cluster ≤ 10, each point is annotated with its sector list
    (annotation shown only on the top stat row to avoid clutter).

    Parameters
    ----------
    table : astropy Table or pandas DataFrame
        Must already have lsp_* and gp_kspace_* columns from add_gp_summary_stats.
    cluster_name : str or None
        If given, plot only this cluster (1-column figure).
        If None, pick up to 4 random clusters from the table.
    stats : list of str or None
        Stat suffixes to plot (e.g. ['peak_freq', 'band_power_0']).
        If None, all detected lsp_/gp_kspace_ columns are plotted.
    seed : int or None
        Random seed for cluster selection when cluster_name is None.
    """
    is_pandas = hasattr(table, "iloc")
    all_cols = list(table.columns) if is_pandas else list(table.colnames)

    # ── Detect paired stat suffixes ──────────────────────────────────────────
    suffixes = [
        c[4:] for c in all_cols
        if c.startswith("lsp_") and f"gp_kspace_{c[4:]}" in all_cols
    ]
    if stats is not None:
        suffixes = [s for s in stats if s in suffixes]
    if not suffixes:
        raise ValueError("No paired lsp_/gp_kspace_ stat columns found in table.")

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _decode(n):
        return n.decode() if isinstance(n, bytes) else str(n)

    if is_pandas:
        unique_names = [_decode(n) for n in table["name"].unique()]
    else:
        unique_names = list(dict.fromkeys(_decode(n) for n in table["name"]))

    # ── Determine clusters to plot ───────────────────────────────────────────
    if cluster_name is not None:
        cluster_names = [_decode(cluster_name)]
    else:
        rng = np.random.default_rng(seed)
        n_select = min(4, len(unique_names))
        cluster_names = list(rng.choice(unique_names, size=n_select, replace=False))

    n_stat_rows = len(suffixes)
    n_cols = len(cluster_names)

    # Collect all n_sectors values across clusters for consistent x-ticks
    all_ns = set()
    cluster_subsets = {}
    for cname in cluster_names:
        if is_pandas:
            sub = table[table["name"].apply(_decode) == cname]
            ns = sub["n_sectors"].values
        else:
            name_arr = np.array([_decode(n) for n in table["name"]])
            sub = table[name_arr == cname]
            ns = np.array(sub["n_sectors"])
        cluster_subsets[cname] = (sub, ns)
        all_ns.update(ns.tolist())
    all_ns = sorted(all_ns)

    # ── Decide log scale per stat row (across ALL clusters) ──────────────────
    use_log = [False] * n_stat_rows
    for row_i, suffix in enumerate(suffixes):
        all_vals = []
        for cname in cluster_names:
            sub, _ = cluster_subsets[cname]
            if is_pandas:
                all_vals.extend(sub[f"lsp_{suffix}"].values.tolist())
                all_vals.extend(sub[f"gp_kspace_{suffix}"].values.tolist())
            else:
                all_vals.extend(np.array(sub[f"lsp_{suffix}"]).tolist())
                all_vals.extend(np.array(sub[f"gp_kspace_{suffix}"]).tolist())
        finite_pos = [v for v in all_vals if np.isfinite(v) and v > 0]
        if len(finite_pos) >= 2:
            lo, hi = min(finite_pos), max(finite_pos)
            if hi > lo and np.log10(hi / lo) > 1.0:
                use_log[row_i] = True

    # ── Build figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        n_stat_rows, n_cols,
        figsize=(4.5 * n_cols, 2.5 * n_stat_rows),
        sharex=True, sharey="row",
        squeeze=False,
    )

    offset = 0.12

    for col_j, cname in enumerate(cluster_names):
        sub, ns = cluster_subsets[cname]
        annotate = len(ns) <= 10

        if is_pandas:
            sectors_list = [list(v) for v in sub["sectors"]]
        else:
            sectors_list = [list(row["sectors"]) for row in sub]

        for row_i, suffix in enumerate(suffixes):
            ax = axes[row_i, col_j]

            if is_pandas:
                lsp_vals = sub[f"lsp_{suffix}"].values
                gp_vals  = sub[f"gp_kspace_{suffix}"].values
            else:
                lsp_vals = np.array(sub[f"lsp_{suffix}"])
                gp_vals  = np.array(sub[f"gp_kspace_{suffix}"])

            x_lsp = ns - offset
            x_gp  = ns + offset

            ax.scatter(x_lsp, lsp_vals, color="red",  alpha=0.8, s=35,
                       label="LSP" if (row_i == 0 and col_j == 0) else None)
            ax.scatter(x_gp,  gp_vals,  color="blue", alpha=0.8, s=35,
                       label="GP kspace" if (row_i == 0 and col_j == 0) else None)

            if use_log[row_i]:
                ax.set_yscale("log")

            # Sector annotations — top stat row only to avoid clutter
            if annotate and row_i == 0:
                for xi, yi, secs in zip(x_lsp, lsp_vals, sectors_list):
                    ax.annotate(
                        str(secs), (xi, yi),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=5, ha="center", color="darkred",
                    )

            if col_j == 0:
                ax.set_ylabel(suffix, fontsize=8)
            if row_i == 0:
                ax.set_title(cname, fontsize=9)
            if row_i == n_stat_rows - 1:
                ax.set_xlabel("N sectors")

    # Shared x-ticks at integer n_sectors values
    axes[0, 0].set_xticks(all_ns)

    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_kspace_compare_allfits(result, freq_min=0.1, freq_max=24.0, log_y=False):
    """
    Overlay analytic PSD for every attempted m-component model in result.all_fits.

    Each curve is labelled with its component count, BIC, and fitted periods.
    The best-BIC model is drawn with a thicker line and marked in the legend.

    Parameters
    ----------
    result : GPFitResult
    freq_min, freq_max : float
        Frequency range in 1/day.
    log_y : bool
        Use log scale on power axes (default False).
    """
    entries = result.kspace_compare_allfits(freq_min=freq_min, freq_max=freq_max)
    best_bic = result.bic
    best_n = result.n_components

    fig, (ax_freq, ax_per) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Analytic GP PSD — all attempted models", fontsize=13)

    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(entries) - 1, 1)) for i in range(len(entries))]

    for entry, color in zip(entries, colors):
        m    = entry["n_components"]
        bic  = entry["bic"]
        freq = entry["freq"]
        pwr  = entry["power"]
        per  = 1.0 / freq
        period_str = ", ".join(f"{p:.2f}" for p in entry["periods"])

        is_best = (m == best_n and abs(bic - best_bic) < 1e-3)
        lw      = 2.5 if is_best else 1.2
        alpha   = 1.0 if is_best else 0.7
        label   = f"{m} comp | BIC={bic:.1f} | P=[{period_str}] d"
        if is_best:
            label += " ★"

        sort_p = np.argsort(per)

        ax_freq.plot(freq, pwr, color=color, lw=lw, alpha=alpha, label=label)
        ax_per.plot(per[sort_p], pwr[sort_p], color=color, lw=lw, alpha=alpha, label=label)

    for ax, xlabel, xscale in [
        (ax_freq, "Frequency [1/day]", "linear"),
        (ax_per,  "Period [days]",     "log"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"GP kernel PSD [$y^2\,\mathrm{day}$]")
        ax.set_xscale(xscale)
        ax.set_xlim(freq_min, freq_max)
        ax.legend(fontsize=7)
        if log_y:
            ax.set_yscale("log")

    fig.tight_layout()
    plt.show()
    plt.close(fig)

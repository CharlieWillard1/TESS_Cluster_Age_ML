import numpy as np
import matplotlib.pyplot as plt


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


def plot_kspace(results, table_rows=None, log_y=False,
                n_uniform=4096, subtract_mean=True, window=True,
                plot_all_ncomponent_fits=False,
                period_lim=(0.1, 50.0)):
    """
    3-row × 2-col frequency content figure for one or more GPFitResult objects.

    All results are overlaid within each panel.

    Rows
    ----
    0 : LSP of the real data (with white-noise threshold as a dashed line)
    1 : kspace_realized — FFT of the GP posterior mean (data-dependent)
    2 : kspace_true     — analytic prior PSD of the GP kernel

    Columns
    -------
    0 : Frequency domain  (linear x, xlim [0.1, 10] day⁻¹)
    1 : Period domain     (log x,    xlim [0.1, 10] days)

    Parameters
    ----------
    results : GPFitResult or list of GPFitResult
    table_rows : table row or list of table rows, optional
    log_y : bool
        If True, use log scale on all y-axes (default False = linear).
    n_uniform : int
        Grid points for kspace_realized (passed through).
    subtract_mean, window : bool
        Passed through to kspace_realized.
    plot_all_ncomponent_fits : bool
        If True, overlay lower-m fits from result.all_fits as dashed same-color lines
        on rows 1 (realized kspace) and 2 (true kspace). Row 0 is unaffected.
    """
    from GP_Fit import residual_lsp_peak  # deferred to avoid circular import

    results, table_rows = _to_list(results, table_rows)

    period_min, period_max = period_lim
    freq_lim = (1.0 / period_max, 1.0 / period_min)
    # Scale analytic PSD samples with log-frequency span so resolution stays constant per decade
    n_freq_true = max(2000, int(2000 * np.log10(freq_lim[1] / freq_lim[0])))

    if plot_all_ncomponent_fits:
        max_m = max(len(res.all_fits) for res in results)
        m_colors = {m: plt.cm.tab10(m % 10) for m in range(1, max_m + 1)}

    row_titles = [
        "Observed-data Lomb–Scargle periodogram",
        "FFT power of GP posterior mean on uniform grid",
        "Analytic PSD of fitted GP kernel",
    ]
    fft_ylabel = (
        r"Hann-windowed FFT power $|\mathrm{FFT}(w\!\cdot\!\mu_{\rm GP})|^2$ [$y^2$]"
        if window else
        r"Raw FFT power $|\mathrm{FFT}(\mu_{\rm GP})|^2$ [$y^2$]"
    )
    y_labels = [
        "Lomb–Scargle power [dimensionless]",
        fft_ylabel,
        r"GP kernel PSD [$y^2\,\mathrm{day}$]",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), squeeze=False)
    fig.suptitle(r"GP Model Frequency Content  ($y$ = relative flux)", fontsize=13)

    for i, res in enumerate(results):
        label = _label(table_rows[i] if table_rows else None, i)

        # ── Row 0: LSP of real data ──────────────────────────────────────────
        lsp = residual_lsp_peak(res.t, res.y, flux_err=res.yerr)
        lsp_freq  = np.asarray(lsp["freq"])
        lsp_power = np.asarray(lsp["power"])
        lsp_period = 1.0 / lsp_freq
        threshold  = res.white_noise_info["white_noise_power_threshold"]

        line, = axes[0, 0].plot(lsp_freq, lsp_power, lw=0.8, alpha=0.85, label=label)
        c = line.get_color()
        axes[0, 0].axhline(threshold, color=c, lw=1.0, ls='--', alpha=0.5)

        sort_p = np.argsort(lsp_period)
        axes[0, 1].plot(lsp_period[sort_p], lsp_power[sort_p],
                        lw=0.8, alpha=0.85, color=c, label=label)
        axes[0, 1].axhline(threshold, color=c, lw=1.0, ls='--', alpha=0.5)

        # ── Row 1: realized kspace ───────────────────────────────────────────
        kr = res.kspace_realized(n_uniform=n_uniform,
                                 subtract_mean=subtract_mean, window=window)
        kr_freq  = kr["freq"][1:]          # skip DC
        kr_power = kr["power"][1:]
        kr_period = 1.0 / kr_freq

        axes[1, 0].plot(kr_freq, kr_power, color=c, lw=0.8, alpha=0.85, label=label)
        sort_p = np.argsort(kr_period)
        axes[1, 1].plot(kr_period[sort_p], kr_power[sort_p],
                        color=c, lw=0.8, alpha=0.85, label=label)

        if plot_all_ncomponent_fits:
            import jax.numpy as jnp
            x_uniform = np.linspace(0.0, float(res.x.max()), n_uniform)
            for fit in res.all_fits:
                m = fit["n_components"]
                if m == res.n_components:
                    continue
                cond_gp = fit["gp"].condition(jnp.asarray(res.y),
                                              jnp.asarray(x_uniform)).gp
                gp_u = np.array(cond_gp.mean)  # np.array() ensures a writable copy
                if subtract_mean:
                    gp_u -= np.nanmean(gp_u)
                dt = x_uniform[1] - x_uniform[0]
                fft_input = gp_u * np.hanning(n_uniform) if window else gp_u
                pwr = np.abs(np.fft.rfft(fft_input)) ** 2
                freq = np.fft.rfftfreq(n_uniform, d=dt)
                freq, pwr = freq[1:], pwr[1:]   # skip DC
                per = 1.0 / freq
                sort_p2 = np.argsort(per)
                mc = m_colors[m]
                axes[1, 0].plot(freq, pwr, color=mc, lw=0.8, ls='--', alpha=0.7,
                                label=f"m={m}")
                axes[1, 1].plot(per[sort_p2], pwr[sort_p2], color=mc, lw=0.8,
                                ls='--', alpha=0.7, label=f"m={m}")

        # ── Row 2: analytic (true) kspace ────────────────────────────────────
        kt = res.kspace_true(freq_min=freq_lim[0], freq_max=freq_lim[1], n_freq=n_freq_true)
        kt_period = 1.0 / kt["freq"]

        axes[2, 0].plot(kt["freq"], kt["power"], color=c, lw=0.8, alpha=0.85, label=label)
        sort_p = np.argsort(kt_period)
        axes[2, 1].plot(kt_period[sort_p], kt["power"][sort_p],
                        color=c, lw=0.8, alpha=0.85, label=label)

        if plot_all_ncomponent_fits:
            entries = res.kspace_compare_allfits(freq_min=freq_lim[0], freq_max=freq_lim[1], n_freq=n_freq_true)
            for entry in entries:
                m = entry["n_components"]
                if m == res.n_components:
                    continue
                mc = m_colors[m]
                freq_e = entry["freq"]
                pwr_e  = entry["power"]
                per_e  = 1.0 / freq_e
                sort_p2 = np.argsort(per_e)
                axes[2, 0].plot(freq_e, pwr_e, color=mc, lw=0.8, ls='--', alpha=0.7,
                                label=f"m={m}")
                axes[2, 1].plot(per_e[sort_p2], pwr_e[sort_p2], color=mc, lw=0.8,
                                ls='--', alpha=0.7, label=f"m={m}")

    # ── Axis formatting ──────────────────────────────────────────────────────
    for row in range(3):
        axes[row, 0].set_xlabel("Frequency [1/day]")
        axes[row, 1].set_xlabel("Period [days]")
        for col in range(2):
            ax = axes[row, col]
            ax.set_xlim(*(freq_lim if col == 0 else period_lim))
            ax.set_xscale("log")
            ax.set_ylabel(y_labels[row])
            ax.set_title(f"{row_titles[row]} — {'period' if col else 'frequency'} domain")
            ax.legend(fontsize=8)
            if log_y:
                ax.set_yscale("log")

    fig.tight_layout()
    plt.show()


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


def plot_kspace_compare_allfits(result, freq_min=0.1, freq_max=10.0, log_y=False):
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

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

def plot_gp_fit(results, table_rows=None):
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
        ax.plot(res.t, res.gp_mean, color='red', lw=1.5, label='GP mean')
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
        ax.plot(res.t, res.gp_mean, lw=1.2, alpha=0.85, label=label)
        ax.set_xlim(res.t.min(), res.t.min() + 27)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Relative flux")
    ax.set_title("All GP means")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_kspace(results, table_rows=None, log_y=False,
                n_uniform=4096, subtract_mean=True, window=True):
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
    """
    from GP_Fit import residual_lsp_peak  # deferred to avoid circular import

    results, table_rows = _to_list(results, table_rows)

    row_titles = ["LSP of data", "GP realization kspace", "True GP kspace"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), squeeze=False)
    fig.suptitle("GP Model Frequency Content", fontsize=13)

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

        axes[1, 0].plot(kr_freq, kr_power, lw=0.8, alpha=0.85, label=label)
        sort_p = np.argsort(kr_period)
        axes[1, 1].plot(kr_period[sort_p], kr_power[sort_p],
                        lw=0.8, alpha=0.85, label=label)

        # ── Row 2: analytic (true) kspace ────────────────────────────────────
        kt = res.kspace_true()
        kt_period = 1.0 / kt["freq"]

        axes[2, 0].plot(kt["freq"], kt["power"], lw=0.8, alpha=0.85, label=label)
        sort_p = np.argsort(kt_period)
        axes[2, 1].plot(kt_period[sort_p], kt["power"][sort_p],
                        lw=0.8, alpha=0.85, label=label)

    # ── Axis formatting ──────────────────────────────────────────────────────
    for row in range(3):
        axes[row, 0].set_xlabel("Frequency [1/day]")
        axes[row, 1].set_xlabel("Period [days]")
        for col in range(2):
            ax = axes[row, col]
            ax.set_xlim(0.1, 10)
            ax.set_xscale("log" if col == 1 else "linear")
            ax.set_ylabel("Power")
            ax.set_title(f"{row_titles[row]} — {'period' if col else 'frequency'} domain")
            ax.legend(fontsize=8)
            if log_y:
                ax.set_yscale("log")

    fig.tight_layout()
    plt.show()

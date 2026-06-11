import numpy as np


# ============================================================
# Internal helpers
# ============================================================

def _add_stats_columns(table, stats_list, prefix):
    """Add {prefix}_{key} columns to an astropy Table or pandas DataFrame.

    Scalar values → one column.  List/array values → one column per element
    named {prefix}_{key}_{i}, so the table stays RF-flat.
    """
    if not stats_list:
        return table

    keys = list(stats_list[0].keys())
    is_pandas = hasattr(table, "iloc")

    for key in keys:
        first = stats_list[0][key]
        col_name = f"{prefix}_{key}"
        if isinstance(first, (list, np.ndarray)):
            values = [np.array(s[key], dtype=np.float64) for s in stats_list]
        else:
            values = np.array([s[key] for s in stats_list])
        if is_pandas:
            table = table.copy()
        table[col_name] = values

    return table


# ============================================================
# Public: spectral stats workhorse
# ============================================================

def compute_spectral_stats(freq, power, freq_lim, n_bins=10, eps=1e-20):
    """
    Compute RF-ready spectral features from an integrated PSD.

    Integrates power over linear frequency (np.trapezoid) in each of n_bins
    log-spaced frequency bins, then log-transforms each result.

    Parameters
    ----------
    freq : array-like
        Frequencies in 1/day, monotonically increasing, all positive.
    power : array-like
        Power spectral density at each freq.
    freq_lim : (float, float)
        (f_min, f_max) in 1/day.
    n_bins : int
        Number of log-spaced frequency bins.
    eps : float
        Floor added before log10 to avoid log(0).

    Returns
    -------
    dict with keys:
        bin_edges          : list of n_bins+1 log-spaced freq edges (1/day)
        bin_centers        : list of n_bins geometric-mean bin centres (1/day)
        log_band_powers    : list of n_bins log10(P_i + eps)
                             P_i = int S(f) df over bin i; index 0 = lowest freq
        log_low_high_ratio : scalar log10(P_low / P_high), split at middle bin
        log_band_ratios    : list of n_bins-1 log10(P_i / P_{i+1})
    """
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)

    f_min, f_max = freq_lim
    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean

    nan = float("nan")

    mask = (freq >= f_min) & (freq <= f_max) & (freq > 0)
    freq_r = freq[mask]
    power_r = power[mask]

    if len(freq_r) < 2:
        print(f"  [NaN] freq_r has {len(freq_r)} point(s) in freq_lim — "
              f"all stats set to NaN  (f_min={f_min:.4f}, f_max={f_max:.4f})")
        return {
            "bin_edges":          list(edges),
            "bin_centers":        list(centers),
            "log_band_powers":    [nan] * n_bins,
            "log_low_high_ratio": nan,
            "log_band_ratios":    [nan] * (n_bins - 1),
        }

    # Per-bin integrated power: P_i = int S(f) df over [f_lo, f_hi]
    bin_powers = np.full(n_bins, nan)
    for j in range(n_bins):
        f_lo, f_hi = edges[j], edges[j + 1]
        band_mask = (freq_r >= f_lo) & (freq_r <= f_hi)
        n_pts = int(band_mask.sum())
        if n_pts >= 2:
            bin_powers[j] = float(np.trapezoid(power_r[band_mask], freq_r[band_mask]))
        else:
            print(f"  [NaN] bin {j}: {n_pts} point(s) in "
                  f"[{f_lo:.4f}, {f_hi:.4f}] 1/day — set to NaN")

    log_band_powers = [
        float(np.log10(max(p, 0.0) + eps)) if np.isfinite(p) else nan
        for p in bin_powers
    ]

    # Low / high ratio
    mid = n_bins // 2
    p_low = np.nansum(bin_powers[:mid])
    p_high = np.nansum(bin_powers[mid:])
    if p_low > 0 and p_high > 0:
        log_low_high_ratio = float(np.log10(p_low / p_high))
    else:
        print(f"  [NaN] log_low_high_ratio: p_low={p_low:.3g}, p_high={p_high:.3g}")
        log_low_high_ratio = nan

    # Adjacent-bin ratios: log10(P_i / P_{i+1}), i = 0 ... n_bins-2
    log_band_ratios = []
    for i in range(n_bins - 1):
        p_i, p_j = bin_powers[i], bin_powers[i + 1]
        if np.isfinite(p_i) and np.isfinite(p_j) and p_i > 0 and p_j > 0:
            log_band_ratios.append(float(np.log10(p_i / p_j)))
        else:
            log_band_ratios.append(nan)

    return {
        "bin_edges":          list(edges),
        "bin_centers":        list(centers),
        "log_band_powers":    log_band_powers,
        "log_low_high_ratio": log_low_high_ratio,
        "log_band_ratios":    log_band_ratios,
    }


# ============================================================
# Public: main entry point
# ============================================================

def add_gp_summary_stats(
    subset_table,
    results,
    freq_lim=(1 / 10.0, 24.0),
    n_bins=10,
    include_lsp=False,
    eps=1e-20,
):
    """
    Add GP PSD spectral band features to subset_table.

    Computes log-integrated-power features from the analytic GP kernel PSD
    (kspace_true) for each result. LSP features are optionally included.

    Parameters
    ----------
    subset_table : astropy Table or pandas DataFrame
    results : list of GPFitResult (same order/length as subset_table rows)
    freq_lim : (float, float)
        (f_min, f_max) in 1/day. Default: (0.1, 24) -> 10 d down to 1 hr.
    n_bins : int
        Number of log-spaced frequency bins.
    include_lsp : bool
        If True, also compute the same features for the data LSP.
    eps : float
        Floor for log10 to avoid log(0).

    Returns
    -------
    subset_table with new columns added (prefix 'gp_kspace', optionally 'lsp').
    """
    f_min, f_max = freq_lim

    gp_stats_list = []
    lsp_stats_list = [] if include_lsp else None

    for res in results:
        kt = res.kspace_true(freq_min=f_min, freq_max=f_max, n_freq=2000)
        gp_stats_list.append(
            compute_spectral_stats(kt["freq"], kt["power"],
                                   freq_lim=freq_lim, n_bins=n_bins, eps=eps)
        )

        if include_lsp:
            from .gp_fit import residual_lsp_peak
            lsp = residual_lsp_peak(
                res.t, res.y,
                flux_err=res.yerr,
                min_period=1.0 / f_max,
                max_period=1.0 / f_min,
            )
            lsp_stats_list.append(
                compute_spectral_stats(lsp["freq"], lsp["power"],
                                       freq_lim=freq_lim, n_bins=n_bins, eps=eps)
            )

    edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    print("GP PSD spectral bands (log_band_powers[0] = lowest freq / longest period):")
    for j in range(n_bins):
        print(f"  [{j}]: {edges[j]:.4f} - {edges[j+1]:.4f} 1/day  "
              f"({1/edges[j+1]:.2f} - {1/edges[j]:.2f} d)")

    subset_table = _add_stats_columns(subset_table, gp_stats_list, prefix="gp_kspace")
    if include_lsp and lsp_stats_list:
        subset_table = _add_stats_columns(subset_table, lsp_stats_list, prefix="lsp")

    return subset_table

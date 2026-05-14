import numpy as np


# ============================================================
# Internal helpers
# ============================================================

def _resolve_bins(bins, period_lim):
    """
    Return (freq_edges ascending, period_edges ascending) from a bins spec.

    bins=None  → 3 equal log-spaced bins within period_lim
    bins=int   → that many equal log-spaced bins within period_lim
    bins=array → explicit period-space edges in days (sorted ascending);
                 period_lim is ignored — the bin edges define the range.
    """
    period_min, period_max = period_lim
    if bins is None:
        period_edges = np.logspace(np.log10(period_min), np.log10(period_max), 4)
    elif isinstance(bins, int):
        period_edges = np.logspace(np.log10(period_min), np.log10(period_max), bins + 1)
    else:
        period_edges = np.sort(np.asarray(bins, dtype=float))

    # Convert to frequency space (invert + reverse so freq is ascending)
    freq_edges = np.sort(1.0 / period_edges)
    return freq_edges, period_edges


def _add_stats_columns(table, stats_list, prefix):
    """Add {prefix}_{key} columns to an astropy Table or pandas DataFrame."""
    if not stats_list:
        return table

    keys = list(stats_list[0].keys())
    is_pandas = hasattr(table, "iloc")

    for key in keys:
        col_name = f"{prefix}_{key}"
        values = np.array([s[key] for s in stats_list])
        if is_pandas:
            table = table.copy()
            table[col_name] = values
        else:
            table[col_name] = values

    return table


# ============================================================
# Public: spectral stats workhorse
# ============================================================

def compute_spectral_stats(freq, power, bins=None, period_lim=(0.5, 10.0)):
    """
    Compute spectral summary statistics from a frequency-power spectrum.

    All integrals use d(log10 f) so that raw-amplitude and shape statistics
    are consistent with each other and directly comparable between spectra
    with different frequency-axis units (e.g. LSP vs GP PSD).

    Parameters
    ----------
    freq : array-like
        Frequencies in 1/day, monotonically increasing, all positive.
    power : array-like
        Power values corresponding to freq.
    bins : None, int, or array-like
        Frequency band specification (in period space, days):
        - None  : 3 equal log-spaced bins within period_lim
        - int   : that many equal log-spaced bins within period_lim
        - array : explicit period-space bin edges, e.g. [0.5, 1, 2, 5, 10];
                  the analysis range is derived from the edges — period_lim ignored.
        band_power_0 is always the lowest-frequency (longest-period) bin.
    period_lim : (float, float)
        Period range in days when bins is None or int; ignored when bins is an array.

    Returns
    -------
    dict with scalar values:

    Raw amplitude (depend on the absolute power scale):
        peak_freq         : frequency of highest-power peak (1/day)
        peak_period       : corresponding period (days)
        peak_amplitude    : power at that peak
        total_power       : ∫ P d(log10 f) over period_lim
        band_power_0…_N   : ∫ P d(log10 f) per band, band_power_0 = longest periods

    Normalised shape (dimensionless, comparable across LSP and GP PSD):
        band_frac_0…_N         : band_power_j / total_power
        spectral_centroid_logfreq : ∫ log10(f)·P d(log10 f) / total_power
        spectral_width_logfreq    : sqrt(∫ (log10 f − centroid)²·P d(log10 f) / total_power)
        power_entropy          : −∫ p·log(p) d(log10 f), p = P/total_power
        peak_to_continuum      : peak_amplitude / median(power in range)
    """
    freq = np.asarray(freq, dtype=float)
    power = np.asarray(power, dtype=float)

    freq_edges, _ = _resolve_bins(bins, period_lim)
    freq_min = freq_edges[0]
    freq_max = freq_edges[-1]
    n_bins = len(freq_edges) - 1

    mask = (freq >= freq_min) & (freq <= freq_max) & (freq > 0)
    freq_r = freq[mask]
    power_r = power[mask]

    nan = float("nan")
    stats = {}

    if len(freq_r) < 2:
        print(f"  [NaN] freq_r has {len(freq_r)} point(s) in period_lim — "
              f"all stats set to NaN  (freq_min={freq_min:.4f}, freq_max={freq_max:.4f})")
        stats["peak_freq"] = nan
        stats["peak_period"] = nan
        stats["peak_amplitude"] = nan
        stats["total_power"] = nan
        for j in range(n_bins):
            stats[f"band_power_{j}"] = nan
            stats[f"band_frac_{j}"] = nan
        stats["spectral_centroid_logfreq"] = nan
        stats["spectral_width_logfreq"] = nan
        stats["power_entropy"] = nan
        stats["peak_to_continuum"] = nan
        return stats

    nan_power = np.sum(np.isnan(power_r))
    if nan_power:
        print(f"  [NaN] input power array has {nan_power} NaN value(s) "
              f"within period_lim — downstream stats may be NaN")

    log_f = np.log10(freq_r)

    # ── Peak ────────────────────────────────────────────────────────────────
    peak_idx = int(np.argmax(power_r))
    stats["peak_freq"] = float(freq_r[peak_idx])
    stats["peak_period"] = float(1.0 / freq_r[peak_idx])
    stats["peak_amplitude"] = float(power_r[peak_idx])

    # ── Total power ∫ P d(log10 f) — raw amplitude measure ──────────────────
    total_power = float(np.trapezoid(power_r, log_f))
    stats["total_power"] = total_power
    if np.isnan(total_power):
        print(f"  [NaN] total_power is NaN — check for NaN/inf in power array")
    elif total_power <= 0:
        print(f"  [NaN] total_power={total_power:.4g} <= 0 — "
              f"band_frac and shape stats will be NaN")

    # ── Band powers: ∫ P d(log10 f) per band; band_frac = band / total ──────
    for j in range(n_bins):
        f_lo, f_hi = freq_edges[j], freq_edges[j + 1]
        band_mask = (freq_r >= f_lo) & (freq_r <= f_hi)
        n_pts = band_mask.sum()
        if n_pts >= 2:
            bp = float(np.trapezoid(power_r[band_mask], log_f[band_mask]))
            stats[f"band_power_{j}"] = bp
            stats[f"band_frac_{j}"] = bp / total_power if total_power > 0 else nan
            if np.isnan(bp):
                print(f"  [NaN] band_power_{j} is NaN despite {n_pts} points "
                      f"(f=[{f_lo:.4f}, {f_hi:.4f}]) — check for NaN in power")
        else:
            print(f"  [NaN] band_power_{j} / band_frac_{j}: only {n_pts} point(s) "
                  f"in freq band [{f_lo:.4f}, {f_hi:.4f}] 1/day "
                  f"(periods [{1/f_hi:.2f}, {1/f_lo:.2f}] d)")
            stats[f"band_power_{j}"] = nan
            stats[f"band_frac_{j}"] = nan

    # ── Spectral moments: weighted integrals over d(log10 f) ────────────────
    if total_power > 0:
        centroid = float(np.trapezoid(log_f * power_r, log_f) / total_power)
        width = float(np.sqrt(
            np.trapezoid((log_f - centroid) ** 2 * power_r, log_f) / total_power
        ))
        if np.isnan(centroid):
            print(f"  [NaN] spectral_centroid_logfreq is NaN")
        if np.isnan(width):
            print(f"  [NaN] spectral_width_logfreq is NaN "
                  f"(variance integrand may be negative due to NaN power values)")
    else:
        print(f"  [NaN] spectral_centroid_logfreq / spectral_width_logfreq: "
              f"total_power={total_power:.4g} <= 0")
        centroid = nan
        width = nan
    stats["spectral_centroid_logfreq"] = centroid
    stats["spectral_width_logfreq"] = width

    # ── Spectral entropy: −∫ p·log(p) d(log10 f), p = P/total_power ─────────
    # p is a density (integrates to 1 over d(log f)), so this is differential entropy.
    if total_power > 0:
        p_density = power_r / total_power
        integrand = np.where(p_density > 0, -p_density * np.log(p_density), 0.0)
        stats["power_entropy"] = float(np.trapezoid(integrand, log_f))
        if np.isnan(stats["power_entropy"]):
            print(f"  [NaN] power_entropy is NaN — check for NaN in power array")
    else:
        print(f"  [NaN] power_entropy: total_power={total_power:.4g} <= 0")
        stats["power_entropy"] = nan

    # ── Peak-to-continuum ────────────────────────────────────────────────────
    continuum = float(np.median(power_r))
    if continuum <= 0:
        print(f"  [NaN] peak_to_continuum: median power={continuum:.4g} <= 0")
    stats["peak_to_continuum"] = (
        stats["peak_amplitude"] / continuum if continuum > 0 else nan
    )

    return stats


# ============================================================
# Public: main entry point
# ============================================================

def add_gp_summary_stats(subset_table, results, bins=None, period_lim=(0.5, 10.0)):
    """
    Add spectral summary statistics to subset_table for each GP fit result.

    Computes stats on two spectra per row:
      1. Data LSP (recomputed from raw light curve)
      2. Analytic GP kernel PSD (kspace_true)

    New columns are added with 'lsp_' and 'gp_kspace_' prefixes respectively.

    Parameters
    ----------
    subset_table : astropy Table or pandas DataFrame
    results : list of GPFitResult (same order/length as subset_table rows)
    bins : None, int, or array-like
        Band specification passed to compute_spectral_stats.
        None → 3 default bins. int → that many log-spaced bins.
        array → period-space bin edges in days; period_lim is ignored.
    period_lim : (float, float)
        Period range in days when bins is None or int; ignored when bins is an array.

    Returns
    -------
    subset_table with new columns added.
    """
    from GP_Fit import residual_lsp_peak

    freq_edges, period_edges = _resolve_bins(bins, period_lim)
    freq_min = freq_edges[0]
    freq_max = freq_edges[-1]
    period_min = period_edges[0]
    period_max = period_edges[-1]
    n_bins = len(freq_edges) - 1

    lsp_stats_list = []
    gp_stats_list = []

    for res in results:
        # ── LSP: recompute from raw data (full spectrum not stored in GPFitResult)
        lsp = residual_lsp_peak(
            res.t, res.y,
            flux_err=res.yerr,
            min_period=period_min,
            max_period=period_max,
        )
        lsp_stats_list.append(
            compute_spectral_stats(lsp["freq"], lsp["power"], bins=bins, period_lim=period_lim)
        )

        # ── Analytic GP kspace (kspace_true) ──────────────────────────────────
        kt = res.kspace_true(freq_min=freq_min, freq_max=freq_max, n_freq=2000)
        gp_stats_list.append(
            compute_spectral_stats(kt["freq"], kt["power"], bins=bins, period_lim=period_lim)
        )

    # Print bin layout once so user knows what band_power_0/1/2 correspond to
    print(f"Spectral bands (band_power_0 = longest periods, "
          f"period range {period_min:.2f}–{period_max:.2f} d):")
    for j in range(n_bins):
        p_hi = 1.0 / freq_edges[j]
        p_lo = 1.0 / freq_edges[j + 1]
        print(f"  band_power_{j}: {p_lo:.2f} – {p_hi:.2f} d  "
              f"({freq_edges[j]:.3f} – {freq_edges[j+1]:.3f} 1/day)")

    subset_table = _add_stats_columns(subset_table, lsp_stats_list, prefix="lsp")
    subset_table = _add_stats_columns(subset_table, gp_stats_list, prefix="gp_kspace")

    return subset_table

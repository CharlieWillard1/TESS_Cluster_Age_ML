"""Peak statistics derived from a stored Lomb-Scargle periodogram.

Everything here is computed post-hoc from the ``LSP_freq`` / ``LSP_power`` arrays and the
red-noise fit already in the table, so it costs one cheap pass and never requires
re-running ``expand_table``.

Background
----------
``compute_lsp`` computes the peak power, frequency, and period, but
``sector_expansion`` stores none of them -- the rotation period, the most directly
relevant observable for gyrochronology, was being discarded at table construction.  This
module restores it and adds the red-noise-normalised significance.

Locate with the difference, test with the ratio
------------------------------------------------
For a periodic signal added independently to red noise, x = s + n, the periodogram
expectation at the signal frequency is E[P] = C(f) + S, so

    E[P - C] = S        independent of frequency  -> unbiased LOCATOR
    E[P / C] = 1 + S/C  rescaled by 1/C(f)        -> biased toward short periods

but only the ratio is homoscedastic under the null (P/C ~ Exp(1) at every frequency),
which is what a single significance threshold requires.  The two therefore answer
different questions and both are kept:

``LSP_peak_period``
    Period at the maximum of the raw power.  Biased toward long periods, since the peak
    inherits an additive C(f_rot) boost that is largest at low frequency.

``LSP_peak_period_diff``
    Period at the maximum of P - C.  **The unbiased locator.**  Injection-recovery over
    a log- and a linear-uniform period distribution gives recovery flat to within a
    factor 1.5-1.7 across 0.15-8 d, against 1.8 (raw) and 5.5-7.2 (ratio).

``LSP_peak_snr_rn``
    max(P / (C*gamma)) -- the global detection statistic.  Significant when > 1.

``LSP_peak_snr_at_diff``
    P / (C*gamma) evaluated at the located peak.  Distinct from the global maximum:
    argmax(P-C) and argmax(P/C) coincide only ~52% of the time, and 142/1250 rows are
    globally significant while their located peak is not.  Gate the located period on
    THIS, not on LSP_peak_snr_rn.
"""

import numpy as np
import pandas as pd


# Suspect periods (days) for the harmonic check.
_TESS_SUSPECTS = {
    13.7:   "momentum dump / orbit",
    6.85:   "orbit / 2",
    4.567:  "orbit / 3",
    3.425:  "orbit / 4",
    2.74:   "orbit / 5",
    2.0:    "2 d alias",
    1.0:    "Earth / scattered light",
    0.5:    "0.5 d alias",
}


# ---------------------------------------------------------------------------
# Red-noise threshold
# ---------------------------------------------------------------------------

def red_noise_threshold(freq, log10_N, alpha, fap=0.01):
    """Per-frequency Vaughan (2005) significance threshold.

    Matches ``gp_pipeline/gp_fit.py`` exactly so that peak significance computed here is
    on the same footing as the GP pipeline's peak detection::

        P_model = 10**log10_N * f**(-alpha)
        gamma   = -log(1 - (1-fap)**(1/n_freq))
        thresh  = P_model * gamma

    The white-noise floor is deliberately not added.  It is numerically irrelevant over
    this frequency range (argmax location unchanged, ratios shift ~2%), and omitting it
    keeps this consistent with the existing GP peak detection.
    """
    freq = np.asarray(freq, dtype=float)
    P_model = 10.0 ** log10_N * freq ** (-alpha)
    gamma = -np.log(1.0 - (1.0 - fap) ** (1.0 / len(freq)))
    return P_model * gamma


# ---------------------------------------------------------------------------
# Boundary diagnostics
# ---------------------------------------------------------------------------

def _boundary_report(table, period_col, idx_col, k_bins=(1, 3, 10),
                     fracs=(0.95, 0.90, 0.80), P_max=None):
    """Print the fraction of rows whose peak sits against the long-period edge.

    Two framings are reported because neither alone is adequate.  The LSP grid is linear
    in FREQUENCY, so period resolution collapses at the long-period end: a
    ``P > 0.99 * P_max`` cut spans exactly one bin at every n_sectors.  Conversely a
    fixed period cut spans differing bin counts per baseline (P > 9 d covers 4 bins for
    1-sector rows but 9 for 3-sector), so it is a stricter test on short baselines.

      grid-aware : peak within the first k frequency bins -- comparable in
                   resolution-element terms
      fixed      : peak period above a fraction of P_max -- comparable in physical terms
    """
    n = len(table)
    if n == 0:
        return
    if P_max is None:
        P_max = float(table[period_col].max())

    print(f"  peak period at the boundary  [{period_col}]  (P_max = {P_max:.3g} d)")
    print("    grid-aware (peak within first k frequency bins):")
    for k in k_bins:
        m = table[idx_col] < k
        by_ns = "  ".join(
            f"n{int(ns)}={np.mean(table.loc[table['n_sectors'] == ns, idx_col] < k):.3f}"
            for ns in sorted(table['n_sectors'].unique())
        ) if 'n_sectors' in table.columns else ""
        print(f"      k={k:>2} : {m.mean():.4f}  ({int(m.sum())}/{n})   {by_ns}")

    print("    fixed period (spans differing bin counts per n_sectors):")
    for fr in fracs:
        m = table[period_col] > fr * P_max
        print(f"      > {fr:.2f}*P_max : {m.mean():.4f}  ({int(m.sum())}/{n})")


# ---------------------------------------------------------------------------
# Table-level entry point
# ---------------------------------------------------------------------------

def add_lsp_metrics(table, fap=0.01, k_bins=(1, 3, 10), fracs=(0.95, 0.90, 0.80),
                    verbose=True):
    """Add LSP peak statistics to the table in-place.

    Parameters
    ----------
    table : pd.DataFrame
        Must have ``LSP_freq``, ``LSP_power``, ``rn_log10_N``, ``rn_alpha``.
    fap : float
        False-alarm probability for the red-noise threshold.  Default 0.01.
    k_bins, fracs : tuple
        Thresholds for the boundary quality check.
    verbose : bool
        Print the boundary report.

    New columns
    -----------
    LSP_peak_power      float64  max(LSP_power)
    LSP_peak_freq       float64  frequency at that maximum (1/day)
    LSP_peak_period     float64  1 / LSP_peak_freq (days) -- rotation-period candidate
    LSP_peak_snr_rn      float64  max(LSP_power / red_noise_threshold); >1 = significant
    LSP_peak_period_diff float64  period at argmax(P - C) -- the unbiased locator
    LSP_peak_excess_diff float64  (P - C) at that frequency; the excess power S
    LSP_peak_snr_at_diff float64  P/(C*gamma) at that frequency; >1 = located peak is
                                  itself significant
    """
    n_rows = len(table)
    print(f"[add_lsp_metrics] processing {n_rows} rows  |  fap={fap}")

    idx = table.index
    cols = {c: pd.Series(np.nan, index=idx, dtype=float) for c in
            ('LSP_peak_power', 'LSP_peak_freq', 'LSP_peak_period',
             'LSP_peak_snr_rn',
             'LSP_peak_period_diff', 'LSP_peak_excess_diff', 'LSP_peak_snr_at_diff')}
    # Peak bin indices, kept locally for the boundary report (not stored).
    i_raw = pd.Series(np.nan, index=idx, dtype=float)
    i_rn  = pd.Series(np.nan, index=idx, dtype=float)

    n_ok = n_failed = 0
    for label, row in table.iterrows():
        try:
            f = np.asarray(row['LSP_freq'], dtype=float)
            p = np.asarray(row['LSP_power'], dtype=float)
            if len(f) < 2:
                n_failed += 1
                continue

            j = int(np.nanargmax(p))
            cols['LSP_peak_power'][label]  = float(p[j])
            cols['LSP_peak_freq'][label]   = float(f[j])
            cols['LSP_peak_period'][label] = 1.0 / float(f[j])
            i_raw[label] = j

            log10_N = float(row['rn_log10_N']); alpha = float(row['rn_alpha'])
            thresh = red_noise_threshold(f, log10_N, alpha, fap=fap)
            ratio = p / thresh
            # Global detection: is anything significant anywhere?
            cols['LSP_peak_snr_rn'][label] = float(np.nanmax(ratio))

            # Unbiased location: argmax of the continuum-SUBTRACTED spectrum.
            C = 10.0 ** log10_N * f ** (-alpha)
            jd = int(np.nanargmax(p - C))
            cols['LSP_peak_period_diff'][label] = 1.0 / float(f[jd])
            cols['LSP_peak_excess_diff'][label] = float(p[jd] - C[jd])
            # Significance OF THE LOCATED PEAK -- not the global max.
            cols['LSP_peak_snr_at_diff'][label] = float(ratio[jd])
            i_rn[label] = jd
            n_ok += 1

        except Exception as e:
            name = row['name'].decode() if isinstance(row.get('name'), bytes) else row.get('name')
            print(f"  WARNING [{name}] row {label}: {e}")
            n_failed += 1

    for c, v in cols.items():
        table[c] = v
    print(f"[add_lsp_metrics] done  |  ok={n_ok}  failed={n_failed}")

    if verbose and n_ok:
        tmp = table.assign(_i_raw=i_raw, _i_rn=i_rn)
        P_max = float(np.nanmax(table['LSP_peak_period']))
        print()
        _boundary_report(tmp, 'LSP_peak_period', '_i_raw', k_bins, fracs, P_max)
        print()
        _boundary_report(tmp, 'LSP_peak_period_diff', '_i_rn', k_bins, fracs, P_max)

    return table


# ---------------------------------------------------------------------------
# Harmonic / systematics check
# ---------------------------------------------------------------------------

def check_lsp_harmonics(table, tol=0.02, period_cols=('LSP_peak_period',
                                                     'LSP_peak_period_diff')):
    """Test whether peak periods pile up at known TESS systematics.

    For each suspect period P_s, counts rows with |P/P_s - 1| < tol and compares against
    a local background estimated from the surrounding log-period region (the same
    fractional width, offset to either side).  ``excess_ratio`` well above 1 means a real
    spike rather than the ambient period distribution.

    Parameters
    ----------
    table : pd.DataFrame
        Must already have been through ``add_lsp_metrics``.
    tol : float
        Fractional half-width of the test window.  Default 0.02 (+/- 2%).
    period_cols : tuple of str
        Which peak-period columns to test.

    Returns
    -------
    pd.DataFrame
        Columns: column, period, label, n_within, n_background, excess_ratio.
    """
    rows = []
    for col in period_cols:
        if col not in table.columns:
            continue
        P = pd.to_numeric(table[col], errors='coerce').dropna().values
        if len(P) == 0:
            continue
        for Ps, label in _TESS_SUSPECTS.items():
            n_in = int(np.sum(np.abs(P / Ps - 1.0) < tol))
            # Background: same fractional width, displaced +/- 3*tol either side.
            lo = np.sum((np.abs(P / (Ps * (1 - 3 * tol)) - 1.0) < tol))
            hi = np.sum((np.abs(P / (Ps * (1 + 3 * tol)) - 1.0) < tol))
            n_bg = float(lo + hi) / 2.0
            rows.append({
                'column':       col,
                'period':       Ps,
                'label':        label,
                'n_within':     n_in,
                'n_background': n_bg,
                'excess_ratio': (n_in / n_bg) if n_bg > 0 else np.nan,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        print(f"[check_lsp_harmonics] tol=+/-{tol:.1%}  "
              f"({len(table)} rows)  excess_ratio >> 1 indicates a real pileup")
        print(out.to_string(index=False))
    return out

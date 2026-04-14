import numpy as np
import pandas as pd
from astropy.io import fits as astropy_fits
from astropy.timeseries import LombScargle

from invariant_summary_stats import get_lc_path, resample_lc, normalize_flux


# ---------------------------------------------------------------------------
# Frequency grid
# ---------------------------------------------------------------------------

def _effective_baseline(t, gap_threshold_days=1.0):
    """
    Compute the effective time baseline of a lightcurve, excluding inter-sector
    gaps.

    Splits the time array into contiguous segments wherever consecutive points
    are separated by more than ``gap_threshold_days``, then sums each segment's
    duration.  For TESS data a threshold of 1 day cleanly separates the ~30-min
    within-sector cadence from the multi-day gaps between sectors.

    Parameters
    ----------
    t : array-like
        Observation times (days), must be sorted.
    gap_threshold_days : float
        Time difference (days) above which a break is declared. Default 1.0.

    Returns
    -------
    float
        Sum of (t_end - t_start) over all contiguous segments.
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return 0.0
    dt = np.diff(t)
    break_pts = np.where(dt > gap_threshold_days)[0]   # last idx before each gap
    starts = np.concatenate([[0],          break_pts + 1])
    ends   = np.concatenate([break_pts,    [len(t) - 1]])
    return float(np.sum(t[ends] - t[starts]))


def ls_bins(P_min, P_max, t, alpha=5, T_effective=True):
    """
    Return Lomb-Scargle frequency bins.

    Parameters
    ----------
    P_min : float
        Smallest period (days).
    P_max : float
        Largest period (days).
    t : array-like
        Observation times (days).
    alpha : float
        Oversampling factor (~5–10).
    T_effective : bool
        If True (default), use the effective baseline — the sum of contiguous
        segment durations, excluding inter-sector gaps — to set df.  If False,
        use the simple wall-clock span ``max(t) - min(t)``, which inflates df
        for multi-sector LCs with large gaps between sectors.

    Returns
    -------
    freqs : ndarray
        Frequency grid (1/day).
    """
    f_min = 1.0 / P_max
    f_max = 1.0 / P_min

    if T_effective:
        T = _effective_baseline(t)
    else:
        T = float(np.max(t) - np.min(t))

    df = 1.0 / (alpha * T)
    Nf = int((f_max - f_min) / df)

    freqs = f_min + df * np.arange(Nf)
    return freqs


# ---------------------------------------------------------------------------
# LC builder — stitches all sectors for one master-table row
# ---------------------------------------------------------------------------

def build_lc_for_row(row, master_table, cadence_bin_min=30.0):
    """
    Build a stitched, cadence-resampled, normalized lightcurve for one row of
    the master table.

    Mirrors the per-row logic inside ``add_variability_metrics``:
      1. Locate the cluster's FITS file via ``get_lc_path``.
      2. For each sector index listed in ``row['sectors']``:
         - Load HDU ``int(s) + 1`` → raw time, flux, flux_err.
         - Mask non-finite values.
         - ``resample_lc`` — block-average raw flux to ``cadence_bin_min``.
         - ``normalize_flux`` — relative flux centred on 1 (per-sector median),
           removing inter-sector flux level offsets.
      3. Concatenate all sectors and sort by time.

    Parameters
    ----------
    row : pd.Series
        One row from the master table. Must have columns ``name``, ``sectors``.
    master_table : pd.DataFrame
        Full master table (needed by ``get_lc_path`` for the LOC column).
    cadence_bin_min : float
        Target resampling cadence in minutes.

    Returns
    -------
    t, x, err_x : np.ndarray
        Time (days), normalized relative flux, normalized flux uncertainty.

    Raises
    ------
    FileNotFoundError
        If the FITS file for the cluster cannot be located.
    ValueError
        If no valid data survive masking/resampling/normalization in any sector.
    """
    path = get_lc_path(master_table, row['name'])
    sectors = row['sectors']

    all_t, all_x, all_err_x = [], [], []

    with astropy_fits.open(path) as hdul:
        for s in sectors:
            hdu_idx = int(s) + 1
            data = hdul[hdu_idx].data

            time = np.asarray(data['time'],     dtype=float)
            flux = np.asarray(data['flux'],     dtype=float)
            ferr = np.asarray(data['flux_err'], dtype=float)

            mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
            time, flux, ferr = time[mask], flux[mask], ferr[mask]

            if len(time) < 2:
                continue

            time, flux, ferr = resample_lc(time, flux, ferr, cadence_bin_min)

            try:
                t_s, x_s, err_x_s = normalize_flux(time, flux, ferr)
            except ValueError:
                continue

            all_t.append(t_s)
            all_x.append(x_s)
            all_err_x.append(err_x_s)

    if not all_t:
        raise ValueError("No valid data survived masking/resampling/normalization "
                         "in any sector.")

    t     = np.concatenate(all_t)
    x     = np.concatenate(all_x)
    err_x = np.concatenate(all_err_x)

    order = np.argsort(t)
    return t[order], x[order], err_x[order]


# ---------------------------------------------------------------------------
# Band-power helper
# ---------------------------------------------------------------------------

def _sum_band_power(freqs, power, P_lo, P_hi):
    """
    Sum LSP power over a period band [P_lo, P_hi] days.

    Parameters
    ----------
    freqs : ndarray
        Frequency grid (1/day).
    power : ndarray
        LSP power at each frequency.
    P_lo, P_hi : float
        Lower and upper period bounds (days). ``P_lo < P_hi``.

    Returns
    -------
    float
        Sum of ``power`` where ``P_lo <= 1/freq <= P_hi``.
        Returns 0.0 if no frequencies fall in the band.
    """
    with np.errstate(divide='ignore'):
        periods = 1.0 / freqs
    mask = (periods >= P_lo) & (periods <= P_hi)
    return float(power[mask].sum())


# ---------------------------------------------------------------------------
# Core LSP computation
# ---------------------------------------------------------------------------

def compute_lsp(t, x, err_x, P_min=0.1, P_max=10.0, alpha=5, fap_level=0.01,
                T_effective=True):
    """
    Compute the Lomb-Scargle periodogram for a stitched lightcurve.

    Parameters
    ----------
    t, x, err_x : np.ndarray
        Time (days), normalized flux, flux uncertainty.
    P_min : float
        Minimum period (days).
    P_max : float
        Maximum period (days). Capped internally to T/2 to avoid aliasing on
        short baselines (T is the effective or wall-clock baseline depending on
        ``T_effective``).
    alpha : float
        Frequency grid oversampling factor passed to ``ls_bins``.
    fap_level : float
        False-alarm probability level for the FAP threshold (default 0.01 = 1%).
    T_effective : bool
        Passed to ``ls_bins``.  If True (default), df is set using the effective
        baseline (sum of contiguous segment lengths, gaps excluded) rather than
        the wall-clock span.

    Returns
    -------
    dict with keys:
        freqs                 : ndarray  — frequency grid (1/day)
        power                 : ndarray  — LSP power (standard normalization)
        max_power             : float    — peak power
        freq_at_max_power     : float    — frequency of peak (1/day)
        period_at_max_power   : float    — period of peak (days)
        fap_threshold         : float    — power level at ``fap_level`` FAP (Baluev)
        SumPow_10_7           : float    — sum of power in 7–10 day period band
        SumPow_7_4            : float    — sum of power in 4–7 day period band
        SumPow_4_1            : float    — sum of power in 1–4 day period band
        SumPow_1_p5           : float    — sum of power in 0.5–1 day period band
    """
    T = _effective_baseline(t) if T_effective else float(np.max(t) - np.min(t))
    P_max_eff = min(P_max, T / 2.0)

    freqs = ls_bins(P_min, P_max_eff, t, alpha, T_effective=T_effective)

    ls = LombScargle(t, x, err_x, normalization='standard')
    power = ls.power(freqs)

    fap_threshold = float(ls.false_alarm_level(fap_level, method='baluev'))

    peak_idx = int(np.argmax(power))
    freq_peak = float(freqs[peak_idx])

    return {
        'freqs':               freqs,
        'power':               power,
        'max_power':           float(power[peak_idx]),
        'freq_at_max_power':   freq_peak,
        'period_at_max_power': 1.0 / freq_peak,
        'fap_threshold':       fap_threshold,
        'SumPow_10_7': _sum_band_power(freqs, power, 7.0,  10.0),
        'SumPow_7_4':  _sum_band_power(freqs, power, 4.0,   7.0),
        'SumPow_4_1':  _sum_band_power(freqs, power, 1.0,   4.0),
        'SumPow_1_p5': _sum_band_power(freqs, power, 0.5,   1.0),
    }


# ---------------------------------------------------------------------------
# Master-table wrapper
# ---------------------------------------------------------------------------

def add_invariant_LSP_stats(table, P_min=0.1, P_max=10.0, alpha=5,
                             cadence_bin_min=30.0, fap_level=0.01,
                             T_effective=True):
    """
    Add invariant Lomb-Scargle periodogram statistics to the master table
    in-place.

    For each row the function:
      1. Stitches the multi-sector lightcurve via ``build_lc_for_row``.
      2. Computes the LSP on a frequency grid from ``ls_bins`` with
         ``P_min``–``P_max`` days (``P_max`` is capped to T/2 per row).
      3. Stores the full frequency grid and power array, plus scalar summaries.

    Rows whose FITS file is missing or that fail for any other reason are
    left as NaN / None.

    Parameters
    ----------
    table : pd.DataFrame
        Master table (modified in-place).
    P_min : float
        Minimum period (days). Default 0.1.
    P_max : float
        Maximum period (days). Default 10.0.
    alpha : float
        Frequency grid oversampling factor. Default 5.
    cadence_bin_min : float
        Target resampling cadence in minutes. Default 30.
    fap_level : float
        FAP probability for the significance threshold. Default 0.01 (1%).
    T_effective : bool
        If True (default), the frequency spacing df is computed from the
        effective baseline (sum of contiguous segment durations, gaps excluded)
        rather than the wall-clock span max(t)-min(t).  This prevents df from
        being artificially fine for multi-sector LCs whose gaps inflate the
        apparent span.

    Returns
    -------
    table : pd.DataFrame
        Same object, with new columns added.

    New columns
    -----------
    LSP_freq                object (ndarray)   frequency grid per row (1/day)
    LSP_power               object (ndarray)   LSP power per row
    LSP_max_power           float64            peak power
    LSP_freq_at_max_power   float64            frequency of peak (1/day)
    LSP_period_at_max_power float64            period of peak (days)
    LSP_FAP                 float64            power threshold at ``fap_level`` FAP
    LSP_SumPow_10_7         float64            sum of power in 7–10 day band
    LSP_SumPow_7_4          float64            sum of power in 4–7 day band
    LSP_SumPow_4_1          float64            sum of power in 1–4 day band
    LSP_SumPow_1_p5         float64            sum of power in 0.5–1 day band
    """
    n_rows = len(table)
    n_clusters = table['name'].nunique()
    print(f"[add_invariant_LSP_stats] processing {n_rows} rows across "
          f"{n_clusters} clusters  |  P=[{P_min}, {P_max}] days  "
          f"alpha={alpha}  cadence={cadence_bin_min} min  FAP={fap_level}  "
          f"T_effective={T_effective}")

    # Pre-allocate output Series keyed by the table's own index labels so that
    # label-based assignment (idx from iterrows) works correctly on subsets
    # where the DataFrame index does not start at 0.
    idx = table.index
    lsp_freq  = pd.Series([None] * n_rows, index=idx, dtype=object)
    lsp_power = pd.Series([None] * n_rows, index=idx, dtype=object)
    max_power           = pd.Series(np.nan, index=idx, dtype=float)
    freq_at_max_power   = pd.Series(np.nan, index=idx, dtype=float)
    period_at_max_power = pd.Series(np.nan, index=idx, dtype=float)
    lsp_fap             = pd.Series(np.nan, index=idx, dtype=float)
    sum_10_7 = pd.Series(np.nan, index=idx, dtype=float)
    sum_7_4  = pd.Series(np.nan, index=idx, dtype=float)
    sum_4_1  = pd.Series(np.nan, index=idx, dtype=float)
    sum_1_p5 = pd.Series(np.nan, index=idx, dtype=float)

    n_ok, n_missing, n_failed = 0, 0, 0

    for name, group in table.groupby('name'):
        name_str = name.decode() if isinstance(name, bytes) else name

        # Check that the FITS file exists once per cluster
        try:
            get_lc_path(table, name)
        except FileNotFoundError:
            print(f"  WARNING [add_invariant_LSP_stats] {name_str}: "
                  f"LC file not found — skipping cluster")
            n_missing += len(group)
            continue

        for label, row in group.iterrows():
            try:
                t, x, err_x = build_lc_for_row(row, table, cadence_bin_min)
                result = compute_lsp(t, x, err_x,
                                     P_min=P_min, P_max=P_max,
                                     alpha=alpha, fap_level=fap_level,
                                     T_effective=T_effective)

                lsp_freq[label]  = result['freqs']
                lsp_power[label] = result['power']
                max_power[label]           = result['max_power']
                freq_at_max_power[label]   = result['freq_at_max_power']
                period_at_max_power[label] = result['period_at_max_power']
                lsp_fap[label]             = result['fap_threshold']
                sum_10_7[label] = result['SumPow_10_7']
                sum_7_4[label]  = result['SumPow_7_4']
                sum_4_1[label]  = result['SumPow_4_1']
                sum_1_p5[label] = result['SumPow_1_p5']

                n_ok += 1

            except (FileNotFoundError, ValueError) as e:
                print(f"  WARNING [{name_str}] row {label} "
                      f"sectors={list(row['sectors'])}: {e}")
                n_failed += 1
            except Exception as e:
                print(f"  WARNING [{name_str}] row {label}: unexpected error — {e}")
                n_failed += 1

    print(f"[add_invariant_LSP_stats] done  |  "
          f"ok={n_ok}  missing={n_missing}  failed={n_failed}")

    table['LSP_freq']                = lsp_freq
    table['LSP_power']               = lsp_power
    table['LSP_max_power']           = max_power
    table['LSP_freq_at_max_power']   = freq_at_max_power
    table['LSP_period_at_max_power'] = period_at_max_power
    table['LSP_FAP']                 = lsp_fap
    table['LSP_SumPow_10_7']         = sum_10_7
    table['LSP_SumPow_7_4']          = sum_7_4
    table['LSP_SumPow_4_1']          = sum_4_1
    table['LSP_SumPow_1_p5']         = sum_1_p5

    return table

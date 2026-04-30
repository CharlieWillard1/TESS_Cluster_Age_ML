import os
import glob as glob_module
import numpy as np
import pandas as pd
from astropy.io import fits as astropy_fits
from astropy.timeseries import LombScargle

LC_BASE = '/astro/users/cgwill/TESS_Cluster_Age_ML/light_curves'


# ---------------------------------------------------------------------------
# File-system utilities
# ---------------------------------------------------------------------------

def _slug(name):
    """Canonical key for matching: lowercase, strip spaces, hyphens, and brackets."""
    s = name.decode() if isinstance(name, bytes) else name
    return s.lower().replace(' ', '').replace('-', '').replace('[', '').replace(']', '')


def get_lc_path(master_table, name):
    """
    Given a cluster name (str or bytes), return the path to its LC .fits file.
    Uses the LOC column from master_table to determine the subdirectory.
    Matches by stripping spaces/hyphens from both the name and filename slug,
    to handle complex multi-word catalog names (e.g. 'OGLE CL SMC 133' -> ogle-cl-smc-133).
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    row = master_table[master_table['name'] == name].iloc[0]
    loc = row['LOC'].decode() if isinstance(row['LOC'], bytes) else row['LOC']

    target = _slug(name_str)
    pattern = f'{LC_BASE}/{loc}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits'
    for fpath in glob_module.glob(pattern):
        basename = os.path.basename(fpath)
        slug = basename.replace('hlsp_elk_tess_ffi_', '').replace('_tess_v1_llc.fits', '')
        if _slug(slug) == target:
            return fpath

    raise FileNotFoundError(f"No LC file found for '{name_str}' in {LC_BASE}/{loc}/")


# ---------------------------------------------------------------------------
# LC preprocessing
# ---------------------------------------------------------------------------

def resample_lc(t, f, err_f, cadence_bin_min=30.0):
    """
    Downsample a raw lightcurve to a common cadence by block-averaging
    groups of N consecutive points.

    Resampling is performed on raw flux before normalization. Groups that
    contain an internal time gap (diff > 1.5 × native cadence) are discarded
    entirely so that no averaging occurs across gaps.

    TESS cadences are 3.33, 10, and 30 min — integer multiples of 3 — so the
    decimation factor N is always 1, 3, or 9. An assertion enforces this.

    Parameters
    ----------
    t, f, err_f : np.ndarray
        Time (days), raw flux, and raw flux uncertainty. Must be finite and
        time-sorted before calling.
    cadence_bin_min : float
        Target cadence in minutes. Default 30.

    Returns
    -------
    t_out, f_out, err_f_out : np.ndarray
        Resampled arrays. Returned unchanged if N <= 1.
    """
    if len(t) < 2:
        return t.copy(), f.copy(), err_f.copy()

    native_cadence_min = np.nanmedian(np.diff(t)) * 1440.0
    N = int(round(cadence_bin_min / native_cadence_min))

    if N <= 1:
        return t.copy(), f.copy(), err_f.copy()

    assert N == 1 or N % 3 == 0, (
        f"Unexpected decimation factor N={N} for native cadence {native_cadence_min:.2f} min "
        f"and target {cadence_bin_min:.1f} min. Expected N in {{1, 3, 9}} (TESS cadences are "
        f"3.33, 10, 30 min)."
    )

    dt = np.diff(t)
    gap_threshold = 1.5 * (native_cadence_min / 1440.0)

    t_out, f_out, err_out = [], [], []

    for i in range(0, len(t) - N + 1, N):
        if np.any(dt[i:i + N - 1] > gap_threshold):
            continue
        t_out.append(np.mean(t[i:i + N]))
        f_out.append(np.mean(f[i:i + N]))
        err_out.append(np.sqrt(np.sum(err_f[i:i + N] ** 2)) / N)

    if len(t_out) < 2:
        return t.copy(), f.copy(), err_f.copy()

    return np.array(t_out), np.array(f_out), np.array(err_out)


def normalize_flux(t, f, err_f):
    """
    Normalize a lightcurve to relative flux centered on 1.

    Drops all points where t, f, or err_f is non-finite, then computes a
    single global median and transforms:

        x_i     = f_i / median(f)
        err_x_i = err_f_i / median(f)

    Parameters
    ----------
    t, f, err_f : array-like
        Time, flux, and flux uncertainty arrays of equal length.

    Returns
    -------
    t_clean, x, err_x : np.ndarray
        Filtered and normalized arrays.

    Raises
    ------
    ValueError
        If fewer than 2 finite points remain after filtering.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    err_f = np.asarray(err_f, dtype=float)

    mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(err_f)
    t, f, err_f = t[mask], f[mask], err_f[mask]

    if len(f) < 2:
        raise ValueError("Fewer than 2 finite points after masking NaNs/infs.")

    med = np.median(f)
    if med == 0 or not np.isfinite(med):
        raise ValueError(f"Median flux is {med}; cannot normalize.")

    return t, f / med, err_f / med


# ---------------------------------------------------------------------------
# LSP utilities
# ---------------------------------------------------------------------------

def _effective_baseline(t, gap_threshold_days=1.0):
    """
    Compute the effective time baseline of a lightcurve, excluding inter-sector
    gaps.

    Splits the time array into contiguous segments wherever consecutive points
    are separated by more than ``gap_threshold_days``, then sums each segment's
    duration. For TESS data a threshold of 1 day cleanly separates the ~30-min
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
    break_pts = np.where(dt > gap_threshold_days)[0]
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
        use the simple wall-clock span ``max(t) - min(t)``.

    Returns
    -------
    freqs : ndarray
        Frequency grid (1/day).
    """
    f_min = 1.0 / P_max
    f_max = 1.0 / P_min

    T = _effective_baseline(t) if T_effective else float(np.max(t) - np.min(t))
    df = 1.0 / (alpha * T)
    Nf = int((f_max - f_min) / df)

    return f_min + df * np.arange(Nf)


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


def _compute_noise_floor(t, err_x, freqs, n_realizations=100, seed=None):
    """
    Estimate the frequency-dependent LSP noise floor via Monte Carlo Gaussian
    noise realizations.

    For each realization a pure-noise lightcurve is constructed by drawing
    ``y_noise ~ Normal(0, err_x)`` at the original timestamps, then the
    weighted LSP is computed on the same frequency grid used for the real data.
    The noise floor is the median power across all realizations at each frequency.

    Parameters
    ----------
    t : ndarray
        Observation times (days).
    err_x : ndarray
        Per-point flux uncertainties (same shape as ``t``).
    freqs : ndarray
        Frequency grid (1/day) — must match the grid used for the real LSP.
    n_realizations : int
        Number of Gaussian noise draws. Default 100.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (len(freqs),)
        Median LSP power at each frequency across all realizations.
    """
    rng = np.random.default_rng(seed)
    powers = np.empty((n_realizations, len(freqs)))
    for i in range(n_realizations):
        y_noise = rng.normal(0.0, err_x)
        ls_noise = LombScargle(t, y_noise, err_x, normalization='standard')
        powers[i] = ls_noise.power(freqs)
    return np.median(powers, axis=0)


def compute_lsp(t, x, err_x, P_min=0.1, P_max=10.0, alpha=5, fap_level=0.01,
                T_effective=True):
    """
    Compute the Lomb-Scargle periodogram for a single stitched lightcurve.

    Parameters
    ----------
    t, x, err_x : np.ndarray
        Time (days), normalized flux, flux uncertainty.
    P_min : float
        Minimum period (days).
    P_max : float
        Maximum period (days). Capped internally to T/2 to avoid aliasing on
        short baselines.
    alpha : float
        Frequency grid oversampling factor passed to ``ls_bins``.
    fap_level : float
        False-alarm probability level for the FAP threshold (default 0.01 = 1%).
    T_effective : bool
        If True (default), df is set using the effective baseline (sum of
        contiguous segment lengths, gaps excluded) rather than the wall-clock span.

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
# Function 1 — add lightcurves to master table
# ---------------------------------------------------------------------------

def add_lightcurves(table, cadence_bin_min=30.0):
    """
    Load, stitch, resample, and normalize lightcurves for every row of the
    master table, storing the results as new columns.

    For each row, the function opens the cluster's FITS file once and processes
    each sector listed in ``row['sectors']``:
      1. Load raw time, flux, flux_err from the sector HDU.
      2. Mask non-finite values.
      3. ``resample_lc`` — block-average raw flux to ``cadence_bin_min``.
      4. ``normalize_flux`` — relative flux centred on 1 (per-sector median),
         removing inter-sector flux level offsets.
    All sectors are concatenated and sorted by time.

    Parameters
    ----------
    table : pd.DataFrame
        Master table. Must have columns ``name``, ``sectors``, ``LOC``.
    cadence_bin_min : float
        Target resampling cadence in minutes. Default 30.

    Returns
    -------
    table : pd.DataFrame
        Same object, with new columns added in-place.

    New columns
    -----------
    LC_t     object (ndarray)   time (days), sorted
    LC_x     object (ndarray)   normalized relative flux
    LC_err_x object (ndarray)   normalized flux uncertainty
    """
    n_rows = len(table)
    n_clusters = table['name'].nunique()
    print(f"[add_lightcurves] processing {n_rows} rows across "
          f"{n_clusters} clusters  |  cadence={cadence_bin_min} min")

    idx = table.index
    lc_t   = pd.Series([None] * n_rows, index=idx, dtype=object)
    lc_x   = pd.Series([None] * n_rows, index=idx, dtype=object)
    lc_err = pd.Series([None] * n_rows, index=idx, dtype=object)

    n_ok, n_missing, n_failed = 0, 0, 0

    for name, group in table.groupby('name'):
        name_str = name.decode() if isinstance(name, bytes) else name
        try:
            path = get_lc_path(table, name)
        except FileNotFoundError:
            print(f"  WARNING [add_lightcurves] {name_str}: LC file not found — skipping")
            n_missing += len(group)
            continue

        try:
            with astropy_fits.open(path) as hdul:
                for label, row in group.iterrows():
                    try:
                        sectors = row['sectors']
                        all_t, all_x, all_err_x = [], [], []

                        for s in sectors:
                            hdu_idx = int(s) + 1
                            data = hdul[hdu_idx].data

                            time = np.asarray(data['time'],     dtype=float)
                            flux = np.asarray(data['flux'],     dtype=float)
                            ferr = np.asarray(data['flux_err'], dtype=float)

                            mask = (np.isfinite(time) &
                                    np.isfinite(flux) &
                                    np.isfinite(ferr))
                            time, flux, ferr = time[mask], flux[mask], ferr[mask]
                            if len(time) < 2:
                                print(f"  WARNING [{name_str}] row {label} sector {s}: "
                                      f"too few finite points — skipping sector")
                                continue

                            time, flux, ferr = resample_lc(time, flux, ferr, cadence_bin_min)
                            try:
                                t_s, x_s, err_x_s = normalize_flux(time, flux, ferr)
                            except ValueError as e:
                                print(f"  WARNING [{name_str}] row {label} sector {s}: "
                                      f"normalize_flux failed — {e}")
                                continue

                            all_t.append(t_s)
                            all_x.append(x_s)
                            all_err_x.append(err_x_s)

                        if not all_t:
                            print(f"  WARNING [{name_str}] row {label}: "
                                  f"no valid data in any sector — skipping")
                            n_failed += 1
                            continue

                        t_arr   = np.concatenate(all_t)
                        x_arr   = np.concatenate(all_x)
                        err_arr = np.concatenate(all_err_x)

                        order = np.argsort(t_arr)
                        lc_t[label]   = t_arr[order]
                        lc_x[label]   = x_arr[order]
                        lc_err[label] = err_arr[order]
                        n_ok += 1

                    except Exception as e:
                        print(f"  WARNING [{name_str}] row {label}: unexpected error — {e}")
                        n_failed += 1

        except Exception as e:
            print(f"  WARNING [{name_str}]: failed to open FITS — {e}")
            n_failed += len(group)

    print(f"[add_lightcurves] done  |  ok={n_ok}  missing={n_missing}  failed={n_failed}")

    table['LC_t']     = lc_t
    table['LC_x']     = lc_x
    table['LC_err_x'] = lc_err
    return table


# ---------------------------------------------------------------------------
# Function 2 — compute and store LSPs from stored lightcurves
# ---------------------------------------------------------------------------

def add_lsps(table, P_min=0.1, P_max=10.0, alpha=5, fap_level=0.01,
             T_effective=True, compute_noise_floor=False, n_noise_realizations=100):
    """
    Compute the Lomb-Scargle periodogram for every row using the LC arrays
    stored by ``add_lightcurves``.

    Parameters
    ----------
    table : pd.DataFrame
        Must have columns ``LC_t``, ``LC_x``, ``LC_err_x`` (added by
        ``add_lightcurves``).
    P_min : float
        Minimum period (days). Default 0.1.
    P_max : float
        Maximum period (days). Capped internally to T/2 per row. Default 10.0.
    alpha : float
        Frequency grid oversampling factor. Default 5.
    fap_level : float
        FAP probability for the significance threshold. Default 0.01 (1%).
    T_effective : bool
        If True (default), df is computed from the effective baseline (sum of
        contiguous segment durations, gaps excluded) rather than the wall-clock span.
    compute_noise_floor : bool
        If True, estimate the frequency-dependent noise floor via Monte Carlo
        Gaussian noise realizations and store it in ``LSP_noise_floor``.
        Default False.
    n_noise_realizations : int
        Number of Gaussian noise draws when ``compute_noise_floor=True``.
        Default 100.

    Returns
    -------
    table : pd.DataFrame
        Same object, with new columns added in-place.

    New columns
    -----------
    LSP_freq        object (ndarray)   frequency grid per row (1/day)
    LSP_power       object (ndarray)   LSP power per row (standard normalization)
    LSP_FAP         float64            power threshold at ``fap_level`` FAP (Baluev)
    LSP_noise_floor object (ndarray)   median noise-only LSP (only when compute_noise_floor=True)
    """
    n_rows = len(table)
    noise_info = f"  n_noise={n_noise_realizations}" if compute_noise_floor else ""
    print(f"[add_lsps] processing {n_rows} rows  |  P=[{P_min}, {P_max}] days  "
          f"alpha={alpha}  FAP={fap_level}  T_effective={T_effective}{noise_info}")

    idx = table.index
    lsp_freq  = pd.Series([None] * n_rows, index=idx, dtype=object)
    lsp_power = pd.Series([None] * n_rows, index=idx, dtype=object)
    lsp_fap   = pd.Series(np.nan, index=idx, dtype=float)
    if compute_noise_floor:
        lsp_noise_floor = pd.Series([None] * n_rows, index=idx, dtype=object)

    n_ok, n_skipped, n_failed = 0, 0, 0

    for label, row in table.iterrows():
        t     = row.get('LC_t')
        x     = row.get('LC_x')
        err_x = row.get('LC_err_x')

        if t is None or x is None or err_x is None:
            n_skipped += 1
            continue

        try:
            result = compute_lsp(t, x, err_x,
                                 P_min=P_min, P_max=P_max,
                                 alpha=alpha, fap_level=fap_level,
                                 T_effective=T_effective)
            lsp_freq[label]  = result['freqs']
            lsp_power[label] = result['power']
            lsp_fap[label]   = result['fap_threshold']
            if compute_noise_floor:
                lsp_noise_floor[label] = _compute_noise_floor(
                    t, err_x, result['freqs'],
                    n_realizations=n_noise_realizations,
                )
            n_ok += 1

        except Exception as e:
            name_str = row['name'].decode() if isinstance(row['name'], bytes) else row['name']
            print(f"  WARNING [{name_str}] row {label}: {e}")
            n_failed += 1

    print(f"[add_lsps] done  |  ok={n_ok}  skipped={n_skipped}  failed={n_failed}")

    table['LSP_freq']  = lsp_freq
    table['LSP_power'] = lsp_power
    table['LSP_FAP']   = lsp_fap
    if compute_noise_floor:
        table['LSP_noise_floor'] = lsp_noise_floor
    return table

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


def get_lc_path(name, origin, lc_dir=LC_BASE):
    """
    Given a cluster name and origin (MW/SMC/LMC), return the path to its LC .fits file.

    Uses slug matching (lowercase, no spaces/hyphens/brackets) to handle complex
    multi-word catalog names (e.g. 'OGLE CL SMC 133' -> ogle-cl-smc-133).

    Parameters
    ----------
    name : str or bytes
        Cluster name.
    origin : str
        One of 'MW', 'SMC', 'LMC' — used as the subdirectory under lc_dir.
    lc_dir : str
        Base directory containing per-origin subdirectories.

    Returns
    -------
    str
        Absolute path to the cluster's LC FITS file.

    Raises
    ------
    FileNotFoundError
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    loc = origin.decode() if isinstance(origin, bytes) else origin

    target = _slug(name_str)
    pattern = f'{lc_dir}/{loc}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits'
    for fpath in glob_module.glob(pattern):
        basename = os.path.basename(fpath)
        slug = basename.replace('hlsp_elk_tess_ffi_', '').replace('_tess_v1_llc.fits', '')
        if _slug(slug) == target:
            return fpath

    raise FileNotFoundError(f"No LC file found for '{name_str}' in {lc_dir}/{loc}/")


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
        ls_noise = LombScargle(t, y_noise, err_x, normalization='psd')
        powers[i] = ls_noise.power(freqs)
    return np.median(powers, axis=0)


def compute_lsp(t, x, err_x, P_min=0.1, P_max=10.0, alpha=5, fap_level=0.01,
                T_effective=True, n_bootstraps=200):
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
    n_bootstraps : int
        Number of bootstrap draws for the FAP threshold. Default 200.

    Returns
    -------
    dict with keys:
        freqs                 : ndarray  — frequency grid (1/day)
        power                 : ndarray  — LSP power (psd normalization)
        max_power             : float    — peak power
        freq_at_max_power     : float    — frequency of peak (1/day)
        period_at_max_power   : float    — period of peak (days)
        fap_threshold         : float    — power level at ``fap_level`` FAP (bootstrap)
        SumPow_10_7           : float    — sum of power in 7–10 day period band
        SumPow_7_4            : float    — sum of power in 4–7 day period band
        SumPow_4_1            : float    — sum of power in 1–4 day period band
        SumPow_1_p5           : float    — sum of power in 0.5–1 day period band
    """
    T = _effective_baseline(t) if T_effective else float(np.max(t) - np.min(t))
    P_max_eff = min(P_max, T / 2.0)

    freqs = ls_bins(P_min, P_max_eff, t, alpha, T_effective=T_effective)

    ls = LombScargle(t, x, err_x, normalization='psd')
    power = ls.power(freqs)

    fap_threshold = float(ls.false_alarm_level(
        fap_level, method='bootstrap',
        minimum_frequency=float(freqs[0]),
        maximum_frequency=float(freqs[-1]),
        samples_per_peak=alpha,
        method_kwds={"n_bootstraps": n_bootstraps},
    ))

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
# Combined loader for a set of sector HDUs (called by sector_expansion)
# ---------------------------------------------------------------------------

def _compute_wn_thresholds(t, err, freqs, n_boot=200, percentile=99.0, seed=0):
    """
    Estimate white-noise LSP power thresholds via bootstrap — both scalar and
    per-frequency — in a single pass.

    Generates n_boot pure-Gaussian noise realizations using the measured flux
    errors, computes the LSP power on the given frequency grid for each, and
    returns two thresholds at the requested percentile:
      - scalar: percentile of the per-realization *maximum* power (global WN floor)
      - per-freq: percentile at each individual frequency bin

    Parameters
    ----------
    t : ndarray
        Observation times (days).
    err : ndarray
        Per-point flux uncertainties.
    freqs : ndarray
        Frequency grid (1/day) — must match the grid used for the real LSP.
    n_boot : int
        Number of bootstrap iterations. Default 200.
    percentile : float
        Percentile of the power distributions to return. Default 99.0.
    seed : int
        Random seed for reproducibility. Default 0.

    Returns
    -------
    wn_scalar : float
        Percentile of max-power distribution (one value per bootstrap draw).
    wn_mean : ndarray, shape (len(freqs),)
        Per-frequency mean power across bootstrap draws. For astropy's psd
        normalization this is approximately 1.0 at every frequency (the expected
        white-noise floor is chi-squared(2)/2 = Exp(1), mean = 1).
    """
    rng = np.random.default_rng(seed)
    all_powers = np.empty((n_boot, len(freqs)))
    for i in range(n_boot):
        noise = rng.normal(0.0, err)
        ls = LombScargle(t, noise, err, normalization='psd')
        all_powers[i] = ls.power(freqs)
    wn_scalar = float(np.percentile(np.nanmax(all_powers, axis=1), percentile))
    wn_mean   = np.mean(all_powers, axis=0)
    return wn_scalar, wn_mean


def _fit_red_noise_vaughan(freqs, power, P_N):
    """
    Fit a power-law continuum to the LSP following Vaughan (2005), using psd
    normalization.

    Fits log10(power) vs log10(freq) directly across all valid frequencies — no
    floor subtraction, no SNR masking. Applies the log-space bias correction
    E[log10(Exp(1))] = -0.25068, which is valid for psd normalization because
    I/P_model ~ Exp(1) under H0 (Vaughan 2005, §3).

    Parameters
    ----------
    freqs : ndarray
        Frequency grid (1/day).
    power : ndarray
        LSP power at each frequency (psd normalization).
    P_N : float
        White-noise floor in psd units (use mean of wn_mean from
        _compute_wn_thresholds; ≈ 1.0 for astropy's psd normalization).

    Returns
    -------
    (log10_N, alpha, P_N) : tuple of float
        Power-law fit parameters and noise floor. Full model: P̂(f) = 10**log10_N * f**(-alpha) + P_N.

    Notes
    -----
    Significance threshold (analysis layer, Vaughan 2005 Eq. 16):
        gamma     = -2 * np.log(1 - (1 - fap)**(1/n_freq))
        threshold = P_model * gamma / 2
    where P_model = 10**log10_N * freq**(-alpha) + P_N.
    """
    mask  = np.isfinite(np.log10(freqs)) & np.isfinite(np.log10(power)) & (power > 0)
    log_f = np.log10(freqs[mask])
    log_I = np.log10(power[mask])
    b, a  = np.polyfit(log_f, log_I, 1)         # slope b = -alpha
    log10_N = a + 0.25068                        # bias correction: E[log10(Exp(1))] = -0.25068
    alpha   = -b
    return (float(log10_N), float(alpha), float(P_N))


def load_and_process_sectors(hdul, sector_indices, cadence_bin_min=30.0,
                              P_min=0.1, P_max=10.0, alpha=5,
                              n_wn_boot=200, wn_percentile=99.0):
    """
    Load, resample, normalize, concatenate, compute LSP, and compute the
    white-noise bootstrap threshold for a list of sector HDUs.

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        Open FITS HDU list for the cluster.
    sector_indices : list of int
        Zero-based sector indices. HDU index used is ``sector_index + 1``.
    cadence_bin_min : float
        Target resampling cadence in minutes.
    P_min, P_max : float
        Period range in days for LSP computation.
    n_wn_boot : int
        Number of bootstrap iterations for the WN threshold. Default 200.
    wn_percentile : float
        Percentile of the bootstrap max-power distribution used as the
        significance threshold. Default 99.0.

    Returns
    -------
    t : ndarray
        Sorted time array (days).
    flux : ndarray
        Normalized relative flux centered on 1.
    flux_err : ndarray
        Normalized flux uncertainty.
    lsp_dict : dict
        Output of ``compute_lsp``.
    wn_threshold : float
        White-noise bootstrap power threshold at ``wn_percentile`` (global max).
    red_noise_params : tuple (log10_N, alpha, P_N_empirical, P_N_theoretical)
        Vaughan (2005) power-law continuum fit. Model: P̂(f) = 10**log10_N * f**(-alpha).

    Raises
    ------
    ValueError
        If no valid data remains from any sector.
    """
    all_t, all_flux, all_err = [], [], []

    for s in sector_indices:
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
            t_s, x_s, err_s = normalize_flux(time, flux, ferr)
        except ValueError:
            continue

        all_t.append(t_s)
        all_flux.append(x_s)
        all_err.append(err_s)

    if not all_t:
        raise ValueError("No valid data from any sector in combination")

    t_arr   = np.concatenate(all_t)
    flux_arr = np.concatenate(all_flux)
    err_arr  = np.concatenate(all_err)

    order = np.argsort(t_arr)
    t_arr, flux_arr, err_arr = t_arr[order], flux_arr[order], err_arr[order]

    lsp_dict = compute_lsp(t_arr, flux_arr, err_arr, P_min=P_min, P_max=P_max, alpha=alpha)

    wn_threshold, wn_mean = _compute_wn_thresholds(
        t_arr, err_arr, lsp_dict['freqs'],
        n_boot=n_wn_boot, percentile=wn_percentile,
    )

    P_N_empirical   = float(np.mean(wn_mean))
    P_N_theoretical = 1.0   # expected white-noise floor for astropy psd normalization
    ratio = P_N_empirical / P_N_theoretical
    if not (0.9 <= ratio <= 1.1):
        import warnings
        warnings.warn(
            f"WN noise floor divergence: empirical P_N={P_N_empirical:.4g} vs "
            f"theoretical 1.0 (ratio={ratio:.2f}). "
            "May indicate non-Gaussian noise or poorly scaled flux errors.",
            RuntimeWarning, stacklevel=2,
        )

    log10_N, alpha, _ = _fit_red_noise_vaughan(
        lsp_dict['freqs'], lsp_dict['power'], P_N_empirical,
    )
    red_noise_params = (log10_N, alpha, P_N_empirical, P_N_theoretical)

    return t_arr, flux_arr, err_arr, lsp_dict, wn_threshold, red_noise_params


# ---------------------------------------------------------------------------
# Standalone table-level functions (optional alternative to sector_expansion)
# ---------------------------------------------------------------------------

def add_lightcurves(table, cadence_bin_min=30.0, lc_dir=LC_BASE):
    """
    Load, stitch, resample, and normalize lightcurves for every row of a
    table, storing the results as new columns.

    Expects columns: ``name``, ``sectors``, ``origin``.
    Adds columns: ``LC_t``, ``LC_flux``, ``LC_flux_err``.

    Parameters
    ----------
    table : pd.DataFrame
    cadence_bin_min : float
        Target resampling cadence in minutes. Default 30.
    lc_dir : str
        Base directory for LC FITS files. Default LC_BASE.

    Returns
    -------
    table : pd.DataFrame
        Same object, modified in-place.
    """
    n_rows = len(table)
    n_clusters = table['name'].nunique()
    print(f"[add_lightcurves] processing {n_rows} rows across "
          f"{n_clusters} clusters  |  cadence={cadence_bin_min} min")

    idx = table.index
    lc_t   = pd.Series([None] * n_rows, index=idx, dtype=object)
    lc_f   = pd.Series([None] * n_rows, index=idx, dtype=object)
    lc_err = pd.Series([None] * n_rows, index=idx, dtype=object)

    n_ok, n_missing, n_failed = 0, 0, 0

    for name, group in table.groupby('name'):
        name_str = name.decode() if isinstance(name, bytes) else name
        origin = group.iloc[0]['origin']
        try:
            path = get_lc_path(name, origin, lc_dir)
        except FileNotFoundError:
            print(f"  WARNING [add_lightcurves] {name_str}: LC file not found — skipping")
            n_missing += len(group)
            continue

        try:
            with astropy_fits.open(path) as hdul:
                for label, row in group.iterrows():
                    try:
                        t, flux, flux_err, _ = load_and_process_sectors(
                            hdul, row['sectors'], cadence_bin_min
                        )
                        lc_t[label]   = t
                        lc_f[label]   = flux
                        lc_err[label] = flux_err
                        n_ok += 1
                    except Exception as e:
                        print(f"  WARNING [{name_str}] row {label}: {e}")
                        n_failed += 1
        except Exception as e:
            print(f"  WARNING [{name_str}]: failed to open FITS — {e}")
            n_failed += len(group)

    print(f"[add_lightcurves] done  |  ok={n_ok}  missing={n_missing}  failed={n_failed}")

    table['LC_t']        = lc_t
    table['LC_flux']     = lc_f
    table['LC_flux_err'] = lc_err
    return table


def add_lsps(table, P_min=0.1, P_max=10.0, alpha=5, fap_level=0.01,
             T_effective=True, compute_noise_floor=False, n_noise_realizations=100):
    """
    Compute LSP for every row using the LC arrays stored by ``add_lightcurves``.

    Expects columns: ``LC_t``, ``LC_flux``, ``LC_flux_err``.
    Adds columns: ``LSP_freq``, ``LSP_power``, ``LSP_FAP``.

    Parameters
    ----------
    table : pd.DataFrame
    P_min, P_max : float
        Period range in days.
    alpha : float
        Frequency grid oversampling factor. Default 5.
    fap_level : float
        FAP probability for the significance threshold. Default 0.01.
    T_effective : bool
        Use gap-aware effective baseline for df. Default True.
    compute_noise_floor : bool
        If True, add ``LSP_noise_floor`` column. Default False.
    n_noise_realizations : int
        Monte Carlo draws when ``compute_noise_floor=True``. Default 100.

    Returns
    -------
    table : pd.DataFrame
        Same object, modified in-place.
    """
    n_rows = len(table)
    print(f"[add_lsps] processing {n_rows} rows  |  P=[{P_min}, {P_max}] days  "
          f"alpha={alpha}  FAP={fap_level}  T_effective={T_effective}")

    idx = table.index
    lsp_freq  = pd.Series([None] * n_rows, index=idx, dtype=object)
    lsp_power = pd.Series([None] * n_rows, index=idx, dtype=object)
    lsp_fap   = pd.Series(np.nan, index=idx, dtype=float)
    if compute_noise_floor:
        lsp_noise_floor = pd.Series([None] * n_rows, index=idx, dtype=object)

    n_ok, n_skipped, n_failed = 0, 0, 0

    for label, row in table.iterrows():
        t     = row.get('LC_t')
        x     = row.get('LC_flux')
        err_x = row.get('LC_flux_err')

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

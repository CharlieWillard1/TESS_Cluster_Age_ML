import os
import glob as glob_module
import numpy as np
from astropy.io import fits as astropy_fits

LC_BASE = '/astro/users/cgwill/TESS_Cluster_Age_ML/light_curves'


def _slug(name):
    """Canonical key for matching: lowercase, strip all spaces and hyphens."""
    s = name.decode() if isinstance(name, bytes) else name
    return s.lower().replace(' ', '').replace('-', '')


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
# Step 1: Flux normalization
# ---------------------------------------------------------------------------

def normalize_flux(t, f, err_f):
    """
    Normalize a lightcurve to zero-median relative flux.

    Drops all points where t, f, or err_f is non-finite, then computes a
    single global median and transforms:

        x_i     = f_i / median(f) - 1
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

    n_in = len(t)
    mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(err_f)
    t, f, err_f = t[mask], f[mask], err_f[mask]
    n_clean = len(t)
    print(f"  [normalize_flux] input points: {n_in}  |  after NaN/inf mask: {n_clean}  "
          f"({n_in - n_clean} dropped)")

    if n_clean < 2:
        raise ValueError("Fewer than 2 finite points after masking NaNs/infs.")

    med = np.median(f)
    print(f"  [normalize_flux] median flux: {med:.6g}  |  "
          f"baseline: {t.min():.4f} – {t.max():.4f} days  "
          f"({t.max() - t.min():.2f} day span)")

    if med == 0 or not np.isfinite(med):
        raise ValueError(f"Median flux is {med}; cannot normalize.")

    x = f / med - 1.0 #REMOVE THE MINUS ONE
    err_x = err_f / med
    print(f"  [normalize_flux] normalized flux range: [{x.min():.4f}, {x.max():.4f}]  |  "
          f"median err_x: {np.median(err_x):.4e}")

    return t, x, err_x


# ---------------------------------------------------------------------------
# Step 2: Cadence rebinning
# ---------------------------------------------------------------------------

def rebin_lc(t, x, err_x, cadence_bin_min=30.0):
    """
    Rebin a normalized lightcurve onto a fixed cadence grid using
    inverse-variance weighted means.

    Bins with no data are omitted entirely; the function never interpolates
    across gaps. If the native cadence is already coarser than
    ``cadence_bin_min``, the inputs are returned unchanged.

    Parameters
    ----------
    t, x, err_x : np.ndarray
        Time (days), normalized flux, and normalized flux uncertainty.
    cadence_bin_min : float
        Target bin width in minutes. Default 30.

    Returns
    -------
    x_binned, err_x_binned : np.ndarray
        Rebinned flux and uncertainty arrays (length = number of non-empty bins).
    """
    cadence_bin_days = cadence_bin_min / 1440.0

    if len(t) < 2:
        print(f"  [rebin_lc] fewer than 2 points — returning input unchanged")
        return x.copy(), err_x.copy()

    native_cadence_min = np.nanmedian(np.diff(t)) * 1440.0
    print(f"  [rebin_lc] native cadence: {native_cadence_min:.2f} min  |  "
          f"target cadence: {cadence_bin_min:.1f} min")

    if native_cadence_min >= cadence_bin_min:
        print(f"  [rebin_lc] native cadence >= target — returning input unchanged "
              f"({len(x)} points)")
        return x.copy(), err_x.copy()

    bins = np.arange(t.min(), t.max() + cadence_bin_days, cadence_bin_days)
    bin_idx = np.digitize(t, bins)

    x_binned, sigma_binned = [], []
    for b in range(1, len(bins) + 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        w = 1.0 / err_x[mask] ** 2
        w_sum = w.sum()
        x_binned.append((w * x[mask]).sum() / w_sum)
        sigma_binned.append(1.0 / np.sqrt(w_sum))

    if len(x_binned) < 2:
        print(f"  [rebin_lc] binning produced < 2 bins — returning input unchanged")
        return x.copy(), err_x.copy()

    n_bins = len(x_binned)
    n_empty = len(bins) - n_bins
    print(f"  [rebin_lc] {len(t)} points → {n_bins} bins  "
          f"({n_empty} empty bins omitted, coverage {100*n_bins/len(bins):.1f}%)")

    return np.array(x_binned), np.array(sigma_binned)


# ---------------------------------------------------------------------------
# Step 3: Noise-corrected statistics
# ---------------------------------------------------------------------------

def _compute_stats(x_binned, err_x_binned, sigma_clip_thresh=None):
    """
    Compute noise-corrected variability statistics on a rebinned lightcurve.

    Parameters
    ----------
    x_binned, err_x_binned : np.ndarray
        Rebinned normalized flux and uncertainty.
    sigma_clip_thresh : float or None
        If given, iteratively sigma-clip outliers at this threshold (up to
        5 iterations) before computing statistics.

    Returns
    -------
    dict with keys:
        intrinsic_std   : noise-corrected standard deviation
        intrinsic_rms   : noise-corrected RMS
        intrinsic_mad   : noise-corrected MAD (Gaussian-equivalent sigma)
        n_bins          : number of bins used after clipping
        sigma_phot      : sqrt(mean photon noise variance)
        sigma_obs       : observed standard deviation (before noise correction)
        rms_obs         : observed RMS (before noise correction)
        sigma_mad_obs   : observed MAD scaled to Gaussian sigma (1.4826 * MAD)
    """
    x = x_binned.copy()
    sx = err_x_binned.copy()

    if sigma_clip_thresh is not None:
        n_before = len(x)
        for i in range(5):
            mu = x.mean()
            std = x.std()
            if std == 0:
                break
            keep = np.abs(x - mu) <= sigma_clip_thresh * std
            if keep.sum() == len(x):
                break
            x, sx = x[keep], sx[keep]
        print(f"  [_compute_stats] sigma-clip ({sigma_clip_thresh}σ): "
              f"{n_before} → {len(x)} bins  ({n_before - len(x)} clipped)")
    else:
        print(f"  [_compute_stats] no sigma-clipping applied")

    N = len(x)
    sigma_phot2 = np.mean(sx ** 2)
    sigma_phot  = np.sqrt(sigma_phot2)

    sigma_obs2 = np.mean((x - x.mean()) ** 2)
    sigma_obs  = np.sqrt(sigma_obs2)
    intrinsic_std = np.sqrt(max(0.0, sigma_obs2 - sigma_phot2))

    rms_obs2 = np.mean(x ** 2)
    rms_obs  = np.sqrt(rms_obs2)
    intrinsic_rms = np.sqrt(max(0.0, rms_obs2 - sigma_phot2))

    mad_obs = np.median(np.abs(x - np.median(x)))
    sigma_mad_obs = 1.4826 * mad_obs
    intrinsic_mad = np.sqrt(max(0.0, sigma_mad_obs ** 2 - sigma_phot2))

    print(f"  [_compute_stats] n_bins={N}  |  "
          f"sigma_phot={sigma_phot:.4e}  |  "
          f"sigma_obs={sigma_obs:.4e}  |  "
          f"rms_obs={rms_obs:.4e}  |  "
          f"sigma_mad_obs={sigma_mad_obs:.4e}")
    print(f"  [_compute_stats] intrinsic_std={intrinsic_std:.4e}  |  "
          f"intrinsic_rms={intrinsic_rms:.4e}  |  "
          f"intrinsic_mad={intrinsic_mad:.4e}")

    return {
        'intrinsic_std':  intrinsic_std,
        'intrinsic_rms':  intrinsic_rms,
        'intrinsic_mad':  intrinsic_mad,
        'n_bins':         N,
        'sigma_phot':     sigma_phot,
        'sigma_obs':      sigma_obs,
        'rms_obs':        rms_obs,
        'sigma_mad_obs':  sigma_mad_obs,
    }


# ---------------------------------------------------------------------------
# Step 4: Full pipeline wrapper
# ---------------------------------------------------------------------------

def compute_variability_metrics(t, f, err_f,
                                cadence_bin_min=30.0,
                                sigma_clip_thresh=None):
    """
    Compute full-baseline, cadence-standardized, noise-corrected variability
    metrics for a TESS lightcurve.

    Pipeline
    --------
    1. normalize_flux  — zero-mean relative flux via global median
    2. rebin_lc        — inverse-variance weighted rebinning to common cadence
    3. _compute_stats  — noise-corrected STD, RMS, MAD

    Corrections applied
    -------------------
    - Flux normalization: relative flux x_i = f_i/median(f) - 1
    - Cadence: all lightcurves rebinned to ``cadence_bin_min`` minutes before
      computing statistics, so metrics are comparable across different native
      cadences.
    - Photometric noise: mean measurement variance subtracted in quadrature.

    NOT corrected
    -------------
    - Total baseline length: longer baselines (more stitched sectors) may
      naturally show higher variability if the source has long-timescale power.
      No duration normalization is applied.

    Parameters
    ----------
    t, f, err_f : array-like
        Time (days), raw flux, and raw flux uncertainty.
    cadence_bin_min : float
        Target binning cadence in minutes. Default 30.
    sigma_clip_thresh : float or None
        Sigma-clipping threshold applied after rebinning. None = no clipping.

    Returns
    -------
    dict
        All keys from ``_compute_stats`` plus ``cadence_bin_min``.
        Returns None if normalization fails (e.g. all-NaN input).
    """
    print(f"[compute_variability_metrics] cadence_bin_min={cadence_bin_min} min  |  "
          f"sigma_clip_thresh={sigma_clip_thresh}  |  "
          f"n_input={len(np.asarray(t))}")
    try:
        t_norm, x, err_x = normalize_flux(t, f, err_f)
    except ValueError as e:
        print(f"  [compute_variability_metrics] normalize_flux failed: {e}")
        return None

    x_binned, err_x_binned = rebin_lc(t_norm, x, err_x, cadence_bin_min)

    result = _compute_stats(x_binned, err_x_binned, sigma_clip_thresh)
    result['cadence_bin_min'] = cadence_bin_min
    return result


# ---------------------------------------------------------------------------
# Master-table wrapper
# ---------------------------------------------------------------------------

def add_variability_metrics(master_table, cadence_bin_min=30.0,
                             sigma_clip_thresh=None):
    """
    Add full-baseline variability metric columns to master_table in-place.

    For each row, loads all sectors listed in the 'sectors' column, concatenates
    them into a single lightcurve without per-sector normalization, then calls
    ``compute_variability_metrics``.

    Adds columns: ``intrinsic_std``, ``intrinsic_rms``, ``intrinsic_mad``,
    ``n_bins_used``. Rows whose lightcurve file cannot be found or that fail
    for any other reason are left as NaN.

    Note: no per-sector flux offset correction is applied before stitching.
    Inter-sector flux offsets will inflate variability estimates for
    multi-sector lightcurves.

    Parameters
    ----------
    master_table : pd.DataFrame
    cadence_bin_min : float
    sigma_clip_thresh : float or None

    Returns
    -------
    master_table : pd.DataFrame (modified in-place)
    """
    n_clusters = master_table['name'].nunique()
    n_rows = len(master_table)
    print(f"[add_variability_metrics] processing {n_rows} rows across "
          f"{n_clusters} clusters  |  cadence_bin_min={cadence_bin_min} min")

    n = len(master_table)
    intrinsic_std_vals = np.full(n, np.nan)
    intrinsic_rms_vals = np.full(n, np.nan)
    intrinsic_mad_vals = np.full(n, np.nan)
    n_bins_vals        = np.full(n, np.nan)

    n_ok, n_missing, n_failed = 0, 0, 0

    for name, group in master_table.groupby('name'):
        name_str = name.decode() if isinstance(name, bytes) else name
        try:
            path = get_lc_path(master_table, name)
        except FileNotFoundError:
            print(f"  [add_variability_metrics] {name_str}: LC file not found — skipping")
            n_missing += len(group)
            continue

        print(f"  [add_variability_metrics] {name_str}: {len(group)} row(s)  |  {path}")

        try:
            with astropy_fits.open(path) as hdul:
                for idx, row in group.iterrows():
                    try:
                        sectors = row['sectors']
                        all_t, all_x, all_err_x = [], [], []

                        for s in sectors:
                            hdu_idx = int(s) + 1
                            data = hdul[hdu_idx].data

                            time = np.asarray(data['time'],     dtype=float)
                            flux = np.asarray(data['flux'],     dtype=float)
                            ferr = np.asarray(data['flux_err'], dtype=float)

                            try:
                                t_s, x_s, err_x_s = normalize_flux(time, flux, ferr)
                            except ValueError as e:
                                print(f"    row {idx} sector {s}: normalize_flux failed — {e}")
                                continue
                            all_t.append(t_s)
                            all_x.append(x_s)
                            all_err_x.append(err_x_s)

                        if not all_t:
                            print(f"    row {idx}: no valid data in any sector — skipping")
                            n_failed += 1
                            continue

                        t = np.concatenate(all_t)
                        x = np.concatenate(all_x)
                        err_x = np.concatenate(all_err_x)

                        sort_idx = np.argsort(t)
                        t, x, err_x = t[sort_idx], x[sort_idx], err_x[sort_idx]

                        print(f"    row {idx}: sectors={list(sectors)}  |  "
                              f"total points={len(t)}  |  "
                              f"baseline={t.min():.3f}–{t.max():.3f} days")

                        x_binned, err_x_binned = rebin_lc(t, x, err_x, cadence_bin_min)
                        result = _compute_stats(x_binned, err_x_binned, sigma_clip_thresh)
                        result['cadence_bin_min'] = cadence_bin_min

                        if result is None:
                            print(f"    row {idx}: _compute_stats returned None — skipping")
                            n_failed += 1
                            continue

                        intrinsic_std_vals[idx] = result['intrinsic_std']
                        intrinsic_rms_vals[idx] = result['intrinsic_rms']
                        intrinsic_mad_vals[idx] = result['intrinsic_mad']
                        n_bins_vals[idx]        = result['n_bins']
                        n_ok += 1

                    except Exception as e:
                        print(f"    row {idx}: unexpected error — {e}")
                        n_failed += 1

        except Exception as e:
            print(f"  [add_variability_metrics] {name_str}: failed to open FITS — {e}")
            n_failed += len(group)

    print(f"[add_variability_metrics] done  |  "
          f"ok={n_ok}  missing={n_missing}  failed={n_failed}")

    master_table['intrinsic_std'] = intrinsic_std_vals
    master_table['intrinsic_rms'] = intrinsic_rms_vals
    master_table['intrinsic_mad'] = intrinsic_mad_vals
    master_table['n_bins_used']   = n_bins_vals
    return master_table


def add_std_norm(master_table, cadence_bin_min=30.0):
    """Deprecated alias for add_variability_metrics."""
    return add_variability_metrics(master_table, cadence_bin_min=cadence_bin_min)

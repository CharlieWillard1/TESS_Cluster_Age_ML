import numpy as np
import pandas as pd

from extract_LC_LSP import resample_lc, normalize_flux


# ---------------------------------------------------------------------------
# Noise-corrected statistics
# ---------------------------------------------------------------------------

def _compute_stats(x_binned, err_x_binned, sigma_clip_thresh=None):
    """
    Compute noise-corrected variability statistics on a resampled lightcurve.

    Parameters
    ----------
    x_binned, err_x_binned : np.ndarray
        Resampled normalized flux and uncertainty.
    sigma_clip_thresh : float or None
        If given, iteratively sigma-clip outliers at this threshold (up to
        5 iterations) before computing statistics.

    Returns
    -------
    dict with keys:
        intrinsic_std   : noise-corrected standard deviation
        intrinsic_rms   : noise-corrected RMS
        intrinsic_mad   : noise-corrected MAD (Gaussian-equivalent sigma)
        n_bins          : number of points used after clipping
        sigma_phot      : sqrt(mean photon noise variance)
        sigma_obs       : observed standard deviation (before noise correction)
        rms_obs         : observed RMS around 1 (before noise correction)
        sigma_mad_obs   : observed MAD scaled to Gaussian sigma (1.4826 * MAD)
    """
    x = x_binned.copy()
    sx = err_x_binned.copy()

    if sigma_clip_thresh is not None:
        for _ in range(5):
            mu = x.mean()
            std = x.std()
            if std == 0:
                break
            keep = np.abs(x - mu) <= sigma_clip_thresh * std
            if keep.sum() == len(x):
                break
            x, sx = x[keep], sx[keep]

    N = len(x)
    sigma_phot2 = np.mean(sx ** 2)
    sigma_phot  = np.sqrt(sigma_phot2)

    sigma_obs2 = np.mean((x - x.mean()) ** 2)
    sigma_obs  = np.sqrt(sigma_obs2)
    intrinsic_std = np.sqrt(max(0.0, sigma_obs2 - sigma_phot2))

    rms_obs2 = np.mean((x - 1.0) ** 2)
    rms_obs  = np.sqrt(rms_obs2)
    intrinsic_rms = np.sqrt(max(0.0, rms_obs2 - sigma_phot2))

    mad_obs = np.median(np.abs(x - np.median(x)))
    sigma_mad_obs = 1.4826 * mad_obs
    intrinsic_mad = np.sqrt(max(0.0, sigma_mad_obs ** 2 - sigma_phot2))

    for stat_name, val in [('sigma_phot', sigma_phot), ('sigma_obs', sigma_obs),
                            ('rms_obs', rms_obs), ('sigma_mad_obs', sigma_mad_obs),
                            ('intrinsic_std', intrinsic_std), ('intrinsic_rms', intrinsic_rms),
                            ('intrinsic_mad', intrinsic_mad)]:
        if not np.isfinite(val):
            print(f"  WARNING [_compute_stats] {stat_name}=NaN/inf  |  "
                  f"N={N}  sigma_phot2={sigma_phot2:.4e}  sigma_obs2={sigma_obs2:.4e}  "
                  f"rms_obs2={rms_obs2:.4e}  sigma_mad_obs2={sigma_mad_obs**2:.4e}  "
                  f"x: min={x.min():.4e} max={x.max():.4e} mean={x.mean():.4e}  "
                  f"sx: min={sx.min():.4e} max={sx.max():.4e}")

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


def compute_variability_metrics(t, f, err_f, cadence_bin_min=30.0, sigma_clip_thresh=None):
    """
    Compute full-baseline, cadence-standardized, noise-corrected variability
    metrics for a single TESS lightcurve (standalone use).

    Pipeline: resample_lc → normalize_flux → _compute_stats.

    Parameters
    ----------
    t, f, err_f : array-like
        Time (days), raw flux, and raw flux uncertainty.
    cadence_bin_min : float
        Target resampling cadence in minutes. Default 30.
    sigma_clip_thresh : float or None
        Sigma-clipping threshold applied after resampling. None = no clipping.

    Returns
    -------
    dict
        All keys from ``_compute_stats`` plus ``cadence_bin_min``.
        Returns None if normalization fails.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    err_f = np.asarray(err_f, dtype=float)

    t, f, err_f = resample_lc(t, f, err_f, cadence_bin_min)

    try:
        t_norm, x, err_x = normalize_flux(t, f, err_f)
    except ValueError as e:
        print(f"  WARNING [compute_variability_metrics] normalize_flux failed: {e}")
        return None

    result = _compute_stats(x, err_x, sigma_clip_thresh)
    result['cadence_bin_min'] = cadence_bin_min
    return result


# ---------------------------------------------------------------------------
# Function 3 — add variability metrics to master table
# ---------------------------------------------------------------------------

def add_variability_metrics(table, sigma_clip_thresh=None):
    """
    Add noise-corrected variability statistics to the master table in-place,
    reading from the ``LC_x`` and ``LC_err_x`` columns stored by
    ``add_lightcurves``.

    Parameters
    ----------
    table : pd.DataFrame
        Must have columns ``LC_x``, ``LC_err_x`` (added by ``add_lightcurves``).
    sigma_clip_thresh : float or None
        Sigma-clipping threshold passed to ``_compute_stats``. None = no clipping.

    Returns
    -------
    table : pd.DataFrame
        Same object, with new columns added in-place.

    New columns
    -----------
    intrinsic_std  float64   noise-corrected standard deviation
    intrinsic_rms  float64   noise-corrected RMS around 1
    intrinsic_mad  float64   noise-corrected MAD (Gaussian-equivalent sigma)
    n_bins_used    float64   number of points used after sigma-clipping
    """
    n_rows = len(table)
    print(f"[add_variability_metrics] processing {n_rows} rows")

    idx = table.index
    intrinsic_std_vals = pd.Series(np.nan, index=idx, dtype=float)
    intrinsic_rms_vals = pd.Series(np.nan, index=idx, dtype=float)
    intrinsic_mad_vals = pd.Series(np.nan, index=idx, dtype=float)
    n_bins_vals        = pd.Series(np.nan, index=idx, dtype=float)

    n_ok, n_skipped, n_failed = 0, 0, 0

    for label, row in table.iterrows():
        x     = row.get('LC_x')
        err_x = row.get('LC_err_x')

        if x is None or err_x is None:
            n_skipped += 1
            continue

        try:
            result = _compute_stats(x, err_x, sigma_clip_thresh)
            intrinsic_std_vals[label] = result['intrinsic_std']
            intrinsic_rms_vals[label] = result['intrinsic_rms']
            intrinsic_mad_vals[label] = result['intrinsic_mad']
            n_bins_vals[label]        = result['n_bins']
            n_ok += 1

        except Exception as e:
            name_str = row['name'].decode() if isinstance(row['name'], bytes) else row['name']
            print(f"  WARNING [{name_str}] row {label}: {e}")
            n_failed += 1

    print(f"[add_variability_metrics] done  |  ok={n_ok}  skipped={n_skipped}  failed={n_failed}")

    table['intrinsic_std'] = intrinsic_std_vals
    table['intrinsic_rms'] = intrinsic_rms_vals
    table['intrinsic_mad'] = intrinsic_mad_vals
    table['n_bins_used']   = n_bins_vals
    return table

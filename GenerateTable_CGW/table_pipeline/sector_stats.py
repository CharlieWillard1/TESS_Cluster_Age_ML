"""Per-sector variability statistics with explicit aggregation.

Motivation
----------
After per-sector median normalisation the pooled moments of a stitched light curve
mix linearly,

    sigma^2 ~= sum_k w_k sigma_k^2 ,        mu3 ~= sum_k w_k mu3_k ,

with w_k = n_k / N, but their *ratios* do not.  Substituting gives

    g1_pooled      = sum_k w_k g1_k sigma_k^3   / (sum_k w_k sigma_k^2)^(3/2)
    gamma_p_pooled = sum_k w_k gamma_p_k sigma_k / (sum_k w_k sigma_k^2)^(1/2)

so the effective weight on sector k is w_k * sigma_k^3 for the moment skewness and
w_k * sigma_k for the Pearson skew: the loudest sector dominates cubically for g1.
A skewness measured on the stitched array is therefore not a better estimate of the
single-sector value, it is a *different quantity* that drifts systematically with
n_sectors.  The amplitudes have no such problem (they are monotone functions of a
clean linear mixture), which is why only the two shape statistics are computed here
and the amplitudes stay pooled in ``summary_stats.add_variability_metrics``.

Sector boundaries
-----------------
``load_and_process_sectors`` concatenates the per-sector arrays and sorts by time,
discarding the boundaries.  They cannot be recovered from time gaps: TESS sectors are
consecutive ~27 d windows separated only by ~1 d downlink gaps, indistinguishable from
mid-sector gaps (empirically a 5 d threshold splits only 372/574 multi-sector rows
correctly).  Instead the boundaries are reconstructed exactly from the FITS time
ranges, since the ``sectors`` column holds HDU indices and sector windows are disjoint.
"""

import numpy as np
import pandas as pd
from astropy.io import fits as astropy_fits

from .lc_lsp import get_lc_path, LC_BASE


# ---------------------------------------------------------------------------
# Sector segmentation
# ---------------------------------------------------------------------------

def _sector_time_ranges(name, origin, sectors, lc_dir=LC_BASE, _cache=None):
    """Return [(t_min, t_max), ...] for each HDU index in ``sectors``.

    ``_cache`` is an optional dict keyed by (name, origin); passing one across many
    rows of the same cluster avoids re-opening the same FITS file for every sector
    combination (the table holds ~1250 rows over ~348 clusters).
    """
    key = (name, origin)
    if _cache is not None and key in _cache:
        ranges_by_hdu = _cache[key]
    else:
        ranges_by_hdu = {}
        with astropy_fits.open(get_lc_path(name, origin, lc_dir)) as hdul:
            for i in range(len(hdul) - 1):
                data = hdul[i + 1].data
                if data is None:
                    continue
                t = np.asarray(data['time'], dtype=float)
                t = t[np.isfinite(t)]
                if len(t) < 2:
                    continue
                ranges_by_hdu[i] = (float(t.min()), float(t.max()))
        if _cache is not None:
            _cache[key] = ranges_by_hdu

    return [ranges_by_hdu[int(s)] for s in sectors]


def sector_segments(row, lc_dir=LC_BASE, _cache=None):
    """Boolean masks into ``row['LC_t']``, one per entry of ``row['sectors']``.

    Raises
    ------
    ValueError
        If the masks do not partition the light curve exactly -- i.e. any sample is
        claimed by zero or by more than one sector.  This is a hard error rather than
        a warning: a silent mis-partition would corrupt every downstream statistic.
    """
    t = np.asarray(row['LC_t'], dtype=float)
    ranges = _sector_time_ranges(row['name'], row['origin'], row['sectors'],
                                 lc_dir=lc_dir, _cache=_cache)

    masks = [(t >= lo) & (t <= hi) for (lo, hi) in ranges]
    hits = np.sum(masks, axis=0)

    if not np.all(hits == 1):
        raise ValueError(
            f"sector partition failed for {row['name']} sectors={list(row['sectors'])}: "
            f"{int((hits == 0).sum())} sample(s) in no sector, "
            f"{int((hits > 1).sum())} in more than one"
        )
    return masks


# ---------------------------------------------------------------------------
# Per-sector statistics
# ---------------------------------------------------------------------------

def per_sector_stats(row, lc_dir=LC_BASE, _cache=None):
    """Compute the two shape statistics separately on each sector of a row.

    Returns a list of dicts (one per sector), each with keys ``n``, ``sigma_obs``,
    ``sigma_phot``, ``excess_var``, ``mu3``, ``delta``, ``gamma_p``, ``g1_int``,
    and ``snr``.  ``g1_int`` is NaN where ``excess_var <= 0``; the SNR gate itself is
    applied at aggregation time so the threshold can be varied without recomputing.
    """
    x_all   = np.asarray(row['LC_flux'], dtype=float)
    err_all = np.asarray(row['LC_flux_err'], dtype=float)

    out = []
    for mask in sector_segments(row, lc_dir=lc_dir, _cache=_cache):
        x, sx = x_all[mask], err_all[mask]
        n = len(x)
        if n < 2:
            continue

        d = x - x.mean()
        sigma_obs2  = float(np.mean(d ** 2))
        sigma_phot2 = float(np.mean(sx ** 2))
        excess_var  = sigma_obs2 - sigma_phot2          # signed, unfloored
        mu3         = float(np.mean(d ** 3))
        delta       = float(x.mean() - np.median(x))

        sigma_obs = np.sqrt(sigma_obs2)
        out.append({
            'n':          n,
            'sigma_obs':  sigma_obs,
            'sigma_phot': np.sqrt(sigma_phot2),
            'excess_var': excess_var,
            'mu3':        mu3,
            'delta':      delta,
            'gamma_p':    delta / sigma_obs if sigma_obs > 0 else np.nan,
            # mu3 needs no noise correction (symmetric noise contributes zero third
            # moment); only the sigma^3 normalisation is contaminated, and excess_var
            # is exactly the noise-corrected variance.
            'g1_int':     mu3 / excess_var ** 1.5 if excess_var > 0 else np.nan,
            'snr':        excess_var / sigma_phot2 if sigma_phot2 > 0 else np.nan,
        })
    return out


def _weighted_mean(values, weights):
    """n-weighted mean over finite entries; NaN if nothing is usable.

    Weighting by n_k is the inverse-variance optimum for both shape statistics, whose
    null standard errors are sqrt((pi/2 - 1)/N) and sqrt(6/N) -- both ~ 1/sqrt(N_k).
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))


# ---------------------------------------------------------------------------
# Table-level entry point
# ---------------------------------------------------------------------------

def add_sector_stats(table, lc_dir=LC_BASE, snr_min=1.0):
    """Add per-sector-computed shape statistics to the table in-place.

    Parameters
    ----------
    table : pd.DataFrame
        Must have ``name``, ``origin``, ``sectors``, ``LC_t``, ``LC_flux``,
        ``LC_flux_err``.
    lc_dir : str
        Base directory of the per-origin LC FITS files.
    snr_min : float
        Minimum per-sector ``excess_var / sigma_phot^2`` for a sector to contribute to
        ``g1_int_sec``.  Sectors failing the gate still contribute to ``gamma_p_sec``.
        Without a gate g1_int is unusable -- the excess_var^1.5 denominator diverges as
        the signal approaches the noise floor.

    New columns
    -----------
    gamma_p_sec        float64   n-weighted mean of per-sector Pearson skew
    g1_int_sec         float64   n-weighted mean of per-sector noise-corrected moment
                                 skew over gate-passing sectors; **0.0** where no sector
                                 passes, so the column is always finite and safe to feed
                                 to a model directly
    g1_valid           int64     1 if at least one sector passed the SNR gate, else 0
    n_sectors_valid    int64     how many sectors passed the gate
    gamma_p_persector  object    per-sector gamma_p values (may contain NaN; diagnostic)
    g1_int_persector   object    per-sector g1_int values, pre-gate (may contain NaN)

    Notes
    -----
    ``g1_int_sec`` is imputed with 0.0 rather than NaN so that downstream models do not
    have to handle missing values.  Zero is not a neutral value for a skewness -- it
    asserts "perfectly symmetric" where the truth is "not measurable" -- so ``g1_valid``
    MUST be carried alongside it as a feature.  With the flag present this is the
    standard missing-indicator imputation and the model can learn to discount
    ``g1_int_sec`` wherever ``g1_valid == 0``.  Zero is also close to the sample median
    of the measurable rows, making it the least informative constant available.
    """
    n_rows = len(table)
    print(f"[add_sector_stats] processing {n_rows} rows  |  snr_min={snr_min}")

    idx = table.index
    gamma_p_sec_vals = pd.Series(np.nan, index=idx, dtype=float)
    # Default 0.0, not NaN: rows that fail the gate keep a finite, model-safe value.
    g1_int_sec_vals  = pd.Series(0.0,    index=idx, dtype=float)
    g1_valid_vals    = pd.Series(0,      index=idx, dtype='int64')
    n_valid_vals     = pd.Series(0,      index=idx, dtype='int64')
    gamma_p_ps       = pd.Series([None] * n_rows, index=idx, dtype=object)
    g1_int_ps        = pd.Series([None] * n_rows, index=idx, dtype=object)

    cache = {}
    n_ok = n_failed = 0

    for label, row in table.iterrows():
        try:
            stats = per_sector_stats(row, lc_dir=lc_dir, _cache=cache)
            if not stats:
                # g1_int_sec stays 0.0 and g1_valid stays 0 (series defaults)
                n_failed += 1
                continue

            n_k       = [s['n'] for s in stats]
            gamma_p_k = [s['gamma_p'] for s in stats]
            g1_k      = [s['g1_int'] for s in stats]

            gamma_p_sec_vals[label] = _weighted_mean(gamma_p_k, n_k)

            # Gate applies to g1_int only.
            passed = [s['snr'] > snr_min and s['excess_var'] > 0 for s in stats]
            g1_gated = [g if p else np.nan for g, p in zip(g1_k, passed)]
            n_valid = int(np.sum([p and np.isfinite(g)
                                  for p, g in zip(passed, g1_k)]))
            n_valid_vals[label] = n_valid
            g1_valid_vals[label] = int(n_valid > 0)
            # _weighted_mean returns NaN when nothing is usable; impute 0.0 there so the
            # column is always finite.  g1_valid records which rows were imputed.
            g1_agg = _weighted_mean(g1_gated, n_k)
            g1_int_sec_vals[label] = 0.0 if not np.isfinite(g1_agg) else g1_agg

            gamma_p_ps[label] = np.asarray(gamma_p_k, dtype=float)
            g1_int_ps[label]  = np.asarray(g1_k, dtype=float)
            n_ok += 1

        except Exception as e:
            name_str = row['name'].decode() if isinstance(row['name'], bytes) else row['name']
            print(f"  WARNING [{name_str}] row {label}: {e}")
            n_failed += 1

    print(f"[add_sector_stats] done  |  ok={n_ok}  failed={n_failed}  "
          f"g1_valid={int(g1_valid_vals.sum())}/{n_rows}")

    table['gamma_p_sec']       = gamma_p_sec_vals
    table['g1_int_sec']        = g1_int_sec_vals
    table['g1_valid']          = g1_valid_vals
    table['n_sectors_valid']   = n_valid_vals
    table['gamma_p_persector'] = gamma_p_ps
    table['g1_int_persector']  = g1_int_ps
    return table

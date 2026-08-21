import numpy as np
import pandas as pd
from astropy.io import fits as astropy_fits
from concurrent.futures import ProcessPoolExecutor, as_completed

from .lc_lsp import get_lc_path, load_and_process_sectors, LC_BASE
from .random_sector_combos import make_distinct_sector_combinations

# Known TESS cadences in minutes (3.33 min = ~200-second FFI, 10 min, 30 min)
_TESS_CADENCES = [3.33, 10.0, 30.0]


def _round_cadence(median_dt_min):
    """Snap a measured median cadence to the nearest known TESS cadence."""
    return min(_TESS_CADENCES, key=lambda c: abs(c - median_dt_min))


def _get_sector_cadences(hdul):
    """
    Return a list of (hdu_0based_idx, cadence_min) for every data HDU in hdul.
    HDUs with fewer than 2 finite time points are skipped.
    """
    result = []
    n_hdus = len(hdul) - 1  # HDU[0] is primary
    for i in range(n_hdus):
        hdu = hdul[i + 1]
        if hdu.data is None:
            continue
        t = np.asarray(hdu.data['time'], dtype=float)
        t_finite = t[np.isfinite(t)]
        if len(t_finite) < 2:
            continue
        dt_min = float(np.nanmedian(np.diff(t_finite))) * 1440.0  # days → minutes
        cadence_min = _round_cadence(dt_min)
        result.append((i, cadence_min))
    return result


def _process_one_cluster_row(name, age, origin, lc_dir, f_max, seed, shuffle,
                              include_full, cadence_bin_min, P_min, P_max, alpha,
                              n_wn_boot, wn_percentile, nsigma):
    """
    Process a single cluster row: open FITS, generate sector combos, compute
    LC + LSP + noise thresholds for each combo.

    Returns a dict with keys: rows, n_ok, n_missing, n_failed_combo, msg.
    """
    rows = []
    n_ok = n_missing = n_failed_combo = 0

    try:
        fits_path = get_lc_path(name, origin, lc_dir)
    except FileNotFoundError:
        return {
            "rows": rows, "n_ok": 0, "n_missing": 1, "n_failed_combo": 0,
            "msg": f"WARNING [{name}]: LC file not found — skipping",
        }

    try:
        with astropy_fits.open(fits_path) as hdul:
            sector_cadences = _get_sector_cadences(hdul)
            if not sector_cadences:
                return {
                    "rows": rows, "n_ok": 0, "n_missing": 1, "n_failed_combo": 0,
                    "msg": f"[{name}]  ({origin})  → no valid sectors",
                }

            cadence_groups = {}
            for (hdu_idx, cad) in sector_cadences:
                cadence_groups.setdefault(cad, []).append(hdu_idx)

            all_combos_with_cad = []
            for cad, group_indices in cadence_groups.items():
                _, local_combos = make_distinct_sector_combinations(
                    len(group_indices), f_max=f_max, seed=seed,
                    shuffle=shuffle, include_full=include_full,
                )
                for local_combo in local_combos:
                    global_combo = [group_indices[i] for i in local_combo]
                    all_combos_with_cad.append((global_combo, cad))

            for (combo, cad) in all_combos_with_cad:
                try:
                    t, flux, flux_err, lsp_dict, wn_threshold, red_noise_params = \
                        load_and_process_sectors(
                            hdul, combo,
                            cadence_bin_min=cadence_bin_min,
                            P_min=P_min, P_max=P_max, alpha=alpha,
                            n_wn_boot=n_wn_boot, wn_percentile=wn_percentile,
                            nsigma=nsigma,
                        )
                    rows.append({
                        'name':                   name,
                        'age':                    age,
                        'origin':                 origin,
                        'sectors':                combo,
                        'n_sectors':              len(combo),
                        'cadence':                cad,
                        'LC_t':                   t,
                        'LC_flux':                flux,
                        'LC_flux_err':            flux_err,
                        'LSP_freq':               lsp_dict['freqs'],
                        'LSP_power':              lsp_dict['power'],
                        'LSP_FAP_power':                lsp_dict['fap_threshold'],
                        'LSP_WN_threshold':       wn_threshold,
                        'rn_log10_N':             red_noise_params[0],
                        'rn_alpha':               red_noise_params[1],
                    })
                    n_ok += 1
                except Exception as e:
                    print(f"    WARNING [{name}] combo={combo}: {e}")
                    n_failed_combo += 1

    except Exception as e:
        return {
            "rows": rows, "n_ok": 0, "n_missing": 1, "n_failed_combo": 0,
            "msg": f"WARNING [{name}]: failed to open FITS — {e}",
        }

    msg = (
        f"[{name}]  ({origin})"
        f"  → {len(sector_cadences)} sectors, "
        f"{len(cadence_groups)} cadence group(s), "
        f"{len(all_combos_with_cad)} combos"
    )
    return {"rows": rows, "n_ok": n_ok, "n_missing": n_missing,
            "n_failed_combo": n_failed_combo, "msg": msg}


def expand_table(base_table, lc_dir=LC_BASE,
                 f_max=0.5, seed=0, shuffle=True, include_full=True,
                 cadence_bin_min=30.0, P_min=0.1, P_max=10.0, alpha=5,
                 n_wn_boot=200, wn_percentile=99.0, n_workers=1,
                 nsigma=None):
    """
    Expand a base cluster table into one row per distinct sector combination,
    computing the LC and LSP for each row in a single FITS-file pass.

    For each cluster:
      1. Locate its FITS file via ``get_lc_path(name, origin, lc_dir)``.
      2. Discover all available data HDUs and their cadences.
      3. Group sectors by cadence — no combo will mix different cadences.
      4. Call ``make_distinct_sector_combinations`` independently per cadence group.
      5. For each combo: load, resample, normalize, and concatenate the sector LCs,
         then compute the Lomb-Scargle periodogram and noise thresholds.

    Parameters
    ----------
    base_table : pd.DataFrame
        Output of ``build_base_table``; must have columns ``name``, ``age``, ``origin``.
    lc_dir : str
        Base directory for LC FITS files. Default LC_BASE.
    f_max : float
        Maximum fractional overlap between same-size sector combos. Default 0.5.
    seed : int
        Random seed for ``make_distinct_sector_combinations``. Default 0.
    shuffle : bool
        Shuffle combo candidates before greedy selection. Default True.
    include_full : bool
        Include the full n-sector combination. Default True.
    cadence_bin_min : float
        Target resampling cadence in minutes. Default 30.
    P_min, P_max : float
        Period range in days for LSP. Defaults 0.1 and 10.0.
    alpha : float
        LSP frequency grid oversampling factor. Default 5.
    n_wn_boot : int
        Number of white-noise bootstrap iterations. Default 200.
    wn_percentile : float
        Percentile for WN thresholds. Default 99.0.
    n_workers : int
        Number of parallel worker processes. 1 = sequential (default).
    nsigma : float or None
        If given, apply MAD-based sigma clipping to each sector LC individually
        after resampling and before stitching.  Typical value: 5.0.
        None (default) disables clipping.

    Returns
    -------
    pd.DataFrame
        Columns: name, age, origin, sectors, n_sectors, cadence,
        LC_t, LC_flux, LC_flux_err, LSP_freq, LSP_power, LSP_FAP_power,
        LSP_WN_threshold, rn_log10_N, rn_alpha.

        ``rn_P_empirical`` / ``rn_P_theoretical`` are no longer emitted -- they were
        constant (1.0 exactly, and 1.000 +/- 0.003) and unused downstream.
    """
    n_clusters = len(base_table)
    print(f"[expand_table] {n_clusters} clusters  |  n_workers={n_workers}")

    common_kwargs = dict(
        lc_dir=lc_dir, f_max=f_max, seed=seed, shuffle=shuffle,
        include_full=include_full, cadence_bin_min=cadence_bin_min,
        P_min=P_min, P_max=P_max, alpha=alpha,
        n_wn_boot=n_wn_boot, wn_percentile=wn_percentile,
        nsigma=nsigma,
    )

    rows = []
    n_ok = n_missing = n_failed_combo = 0

    if n_workers == 1:
        for cluster_idx, cr in enumerate(base_table.itertuples(index=False)):
            result = _process_one_cluster_row(cr.name, cr.age, cr.origin, **common_kwargs)
            rows.extend(result["rows"])
            n_ok          += result["n_ok"]
            n_missing     += result["n_missing"]
            n_failed_combo += result["n_failed_combo"]
            print(f"  [{cluster_idx+1}/{n_clusters}] {result['msg']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_one_cluster_row, cr.name, cr.age, cr.origin,
                                **common_kwargs): cr
                for cr in base_table.itertuples(index=False)
            }
            n_done = 0
            for future in as_completed(futures):
                result = future.result()
                rows.extend(result["rows"])
                n_ok          += result["n_ok"]
                n_missing     += result["n_missing"]
                n_failed_combo += result["n_failed_combo"]
                n_done += 1
                print(f"  [{n_done}/{n_clusters}] {result['msg']}", flush=True)

    print(f"\n[expand_table] done  |  rows={n_ok}  "
          f"missing_clusters={n_missing}  failed_combos={n_failed_combo}")

    if not rows:
        return pd.DataFrame(columns=[
            'name', 'age', 'origin', 'sectors', 'n_sectors', 'cadence',
            'LC_t', 'LC_flux', 'LC_flux_err', 'LSP_freq', 'LSP_power', 'LSP_FAP_power',
            'LSP_WN_threshold', 'rn_log10_N', 'rn_alpha',
        ])

    return pd.DataFrame(rows)

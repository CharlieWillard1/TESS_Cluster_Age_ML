import pickle
import numpy as np
import pandas as pd

from .gp_fit import iterative_sho_gp_fit
from .gp_summary_stats import compute_spectral_stats, add_gp_summary_stats


def add_gp_fits(
    table,
    save_path=None,
    freq_lim=(0.1, 24.0),
    n_bins=10,
    max_components=6,
    threshold_mode='white_noise',
    verbose=0,
    **gp_kwargs,
):
    """
    Fit a multi-component SHO GP to every row in `table` and populate GP columns.

    Uses the pre-computed LSP from 'LSP_freq'/'LSP_power' columns to skip the
    internal LSP computation in iterative_sho_gp_fit, saving significant time.

    Parameters
    ----------
    table : pd.DataFrame
        Must contain 'LC_t', 'LC_flux', 'LC_flux_err', 'LSP_freq', 'LSP_power'.
    save_path : str or None
        If given, pickle the results dict to this path after fitting.
    freq_lim : (float, float)
        Frequency range in 1/day. Defines both the GP period search bounds
        (min_period = 1/freq_lim[1], max_period = 1/freq_lim[0]) and the
        log-spaced bins for spectral statistics.
    n_bins : int
        Number of log-spaced frequency bins for spectral stats. Band powers are
        computed as int S(f) df within each bin (variance per band, not per log-freq).
    max_components : int
        Maximum number of SHO components to try.
    threshold_mode : {'white_noise', 'fap', 'red_noise'}
        Threshold used for LSP peak detection.
        'white_noise' : scalar 99th-pct bootstrap max (LSP_WN_threshold). Default.
        'fap'         : scalar 1% FAP analytic threshold (LSP_FAP / Baluev).
        'red_noise'   : per-frequency Vaughan 1% significance (LSP_red_noise_params).
    verbose : int
        0 = one summary line per cluster (default).
        1 = per-fit peak/BIC/timing info.
        2 = per-fit plots (initial LSP + component breakdown after each m).
        3 = component breakdown plot after every m-component fit.
        Also forwarded to iterative_sho_gp_fit.
    **gp_kwargs
        Extra keyword arguments forwarded to iterative_sho_gp_fit.

    New columns added to table (in-place)
    --------------------------------------
    gp_result              object     GPFitResult or None
    GP_freq                object     ndarray, 1/day, from kspace_true
    GP_PSD                 object     ndarray, analytic PSD from kspace_true
    GP_PSD_binned          object     (log_band_powers, bin_centers) tuple
    GP_PSD_bin_ratios      object     (log_band_ratios, bin_midpoints) tuple
    GP_PSD_high_low_ratio  float64    scalar log10 ratio
    gp_n_components              float64    best-model SHO count (NaN on failure)
    gp_bic                       float64    best-model BIC
    gp_log_likelihood            float64    best-model log-likelihood
    gp_jitter                    float64    fitted white-noise jitter term
    gp_periods                   object     list of fitted periods in days (length m)
    gp_sho_sigmas                object     list of fitted sigma per component (length m)
    gp_sho_omegas                object     list of fitted omega per component (length m)
    gp_sho_Qs                    object     list of fitted Q per component (length m)
    gp_initial_lsp_peak_periods  object     LSP seed periods used to initialise fit
    gp_kspace_*                  float64    flat spectral columns for plot_summary_stats
    """
    min_period = 1.0 / freq_lim[1]
    max_period = 1.0 / freq_lim[0]

    n_rows = len(table)
    idx = table.index

    # Scalar columns
    gp_n_comp_vals    = pd.Series(np.nan, index=idx, dtype=float)
    gp_bic_vals       = pd.Series(np.nan, index=idx, dtype=float)
    gp_loglike_vals   = pd.Series(np.nan, index=idx, dtype=float)
    gp_hl_ratio_vals  = pd.Series(np.nan, index=idx, dtype=float)
    gp_jitter_vals    = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_sigma_vals  = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_period_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_Q_vals      = pd.Series(np.nan, index=idx, dtype=float)

    # Object columns
    gp_result_vals    = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_freq_vals      = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_psd_vals       = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_psd_binned     = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_psd_ratios     = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_periods_vals   = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_sho_sigmas_vals = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_sho_omegas_vals = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_sho_Qs_vals     = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_init_lsp_vals   = pd.Series([None] * n_rows, index=idx, dtype=object)

    # For add_gp_summary_stats we need aligned lists
    results_for_stats = []
    results_index     = []

    n_ok, n_null, n_failed = 0, 0, 0

    clusters = list(table.groupby(['name', 'origin'], sort=False))
    n_clusters = len(clusters)
    print(f"[add_gp_fits] {n_clusters} clusters, {n_rows} rows")

    for cluster_idx, ((name, origin), cluster_rows) in enumerate(clusters):
        name_str = name.decode() if isinstance(name, bytes) else name
        origin_str = origin.decode() if isinstance(origin, bytes) else origin

        cluster_bics = []
        cluster_ok = 0
        cluster_null = 0
        cluster_failed = 0

        for label, row in cluster_rows.iterrows():
            t     = row.get("LC_t")
            flux  = row.get("LC_flux")
            err   = row.get("LC_flux_err")
            lfreq = row.get("LSP_freq")
            lpow  = row.get("LSP_power")

            if flux is None or err is None or lfreq is None or lpow is None:
                cluster_null += 1
                continue

            if threshold_mode == 'white_noise':
                thresh_kwargs = dict(wn_threshold=float(row.get("LSP_WN_threshold")))
            elif threshold_mode == 'fap':
                thresh_kwargs = dict(fap_threshold=float(row.get("LSP_FAP")))
            elif threshold_mode == 'red_noise':
                thresh_kwargs = dict(red_noise_params=(
                    row["rn_log10_N"], row["rn_alpha"],
                    row["rn_P_empirical"], row["rn_P_theoretical"],
                ))
            else:
                raise ValueError(f"Unknown threshold_mode: {threshold_mode!r}")

            try:
                result = iterative_sho_gp_fit(
                    t, flux, err,
                    lsp_freq=np.asarray(lfreq),
                    lsp_power=np.asarray(lpow),
                    threshold_mode=threshold_mode,
                    **thresh_kwargs,
                    max_components=max_components,
                    min_period=min_period,
                    max_period=max_period,
                    verbose=verbose,
                    **gp_kwargs,
                )

                if result is None:
                    cluster_null += 1
                    continue

                # Analytic PSD
                kt = result.kspace_true(freq_min=freq_lim[0], freq_max=freq_lim[1])
                gp_freq = kt["freq"]
                gp_psd  = kt["power"]

                # Spectral stats
                stats = compute_spectral_stats(gp_freq, gp_psd, freq_lim, n_bins)

                log_band_powers = stats["log_band_powers"]
                bin_centers     = stats["bin_centers"]
                log_band_ratios = stats["log_band_ratios"]
                bin_midpoints = [
                    np.sqrt(bin_centers[i] * bin_centers[i + 1])
                    for i in range(len(bin_centers) - 1)
                ]

                m = result.n_components

                gp_result_vals[label]    = result
                gp_freq_vals[label]      = gp_freq
                gp_psd_vals[label]       = gp_psd
                gp_psd_binned[label]     = (log_band_powers, bin_centers)
                gp_psd_ratios[label]     = (log_band_ratios, bin_midpoints)
                gp_hl_ratio_vals[label]  = stats["log_low_high_ratio"]
                gp_n_comp_vals[label]    = m
                gp_bic_vals[label]       = result.bic
                gp_loglike_vals[label]   = result.log_likelihood
                gp_periods_vals[label]   = result.periods
                gp_jitter_vals[label]    = result.features["sho_jitter"]
                gp_sho_sigmas_vals[label] = [result.features[f"sho_{i}_sigma"] for i in range(1, m + 1)]
                gp_sho_omegas_vals[label] = [result.features[f"sho_{i}_omega"] for i in range(1, m + 1)]
                gp_sho_Qs_vals[label]     = [result.features[f"sho_{i}_Q"]     for i in range(1, m + 1)]
                gp_init_lsp_vals[label]   = result.features["initial_lsp_peak_periods"]
                gp_rn_sigma_vals[label]   = result.features.get("rn_sigma",       np.nan)
                gp_rn_period_vals[label]  = result.features.get("rn_period_days", np.nan)
                gp_rn_Q_vals[label]       = result.features.get("rn_Q",           np.nan)

                results_for_stats.append(result)
                results_index.append(label)

                cluster_bics.append(result.bic)
                cluster_ok += 1

            except Exception as e:
                sectors = list(row.get("sectors", []))
                print(f"  WARNING [{name_str} {sectors}] row {label}: {e}")
                cluster_failed += 1

        n_ok     += cluster_ok
        n_null   += cluster_null
        n_failed += cluster_failed

        max_bic_str = f"max_BIC={max(cluster_bics):.1f}" if cluster_bics else "no fits"
        n_combos = len(cluster_rows)
        print(
            f"  [{cluster_idx+1}/{n_clusters}] {name_str}  ({origin_str})"
            f"  → {n_combos} combo(s)  {max_bic_str}"
            + (f"  [{cluster_failed} failed]" if cluster_failed else ""),
            flush=True,
        )

    print(f"[add_gp_fits] done  |  ok={n_ok}  null={n_null}  failed={n_failed}")

    # Populate scalar / object columns
    table["gp_result"]                   = gp_result_vals
    table["GP_freq"]                     = gp_freq_vals
    table["GP_PSD"]                      = gp_psd_vals
    table["GP_PSD_binned"]               = gp_psd_binned
    table["GP_PSD_bin_ratios"]           = gp_psd_ratios
    table["GP_PSD_high_low_ratio"]       = gp_hl_ratio_vals
    table["gp_n_components"]             = gp_n_comp_vals
    table["gp_bic"]                      = gp_bic_vals
    table["gp_log_likelihood"]           = gp_loglike_vals
    table["gp_jitter"]                   = gp_jitter_vals
    table["gp_periods"]                  = gp_periods_vals
    table["gp_sho_sigmas"]               = gp_sho_sigmas_vals
    table["gp_sho_omegas"]               = gp_sho_omegas_vals
    table["gp_sho_Qs"]                   = gp_sho_Qs_vals
    table["gp_initial_lsp_peak_periods"] = gp_init_lsp_vals
    table["gp_rn_sigma"]                 = gp_rn_sigma_vals
    table["gp_rn_period"]                = gp_rn_period_vals
    table["gp_rn_Q"]                     = gp_rn_Q_vals

    # Flat gp_kspace_* columns for plot_summary_stats
    if results_for_stats:
        subset = table.loc[results_index].copy()
        subset = add_gp_summary_stats(
            subset, results_for_stats,
            freq_lim=freq_lim, n_bins=n_bins, include_lsp=False,
        )
        # Merge flat columns back into full table
        gp_kspace_cols = [c for c in subset.columns if c.startswith("gp_kspace_")]
        for col in gp_kspace_cols:
            table[col] = subset[col]

    # Save GPFitResult objects keyed by (name, tuple(sectors))
    if save_path is not None:
        results_dict = {}
        for label, row in table.iterrows():
            res = row["gp_result"]
            if res is None:
                continue
            name    = row["name"].decode() if isinstance(row["name"], bytes) else row["name"]
            sectors = tuple(row.get("sectors", []))
            results_dict[(name, sectors)] = res
        with open(save_path, "wb") as fh:
            pickle.dump(results_dict, fh)
        print(f"[add_gp_fits] saved {len(results_dict)} GPFitResult objects to {save_path}")

    return table


def load_gp_results(table_subset, save_path):
    """
    Load GPFitResult objects from a pickle file and align them with table rows.

    Parameters
    ----------
    table_subset : pd.DataFrame
        Subset of the table; must have 'name' and 'sectors' columns.
    save_path : str
        Path to the pickle file written by add_gp_fits (save_path=...).

    Returns
    -------
    list of GPFitResult or None
        One entry per row in table_subset. None for rows not found in the pickle.
    """
    with open(save_path, "rb") as fh:
        results_dict = pickle.load(fh)

    out = []
    for _, row in table_subset.iterrows():
        name    = row["name"].decode() if isinstance(row["name"], bytes) else row["name"]
        sectors = tuple(row.get("sectors", []))
        out.append(results_dict.get((name, sectors), None))
    return out

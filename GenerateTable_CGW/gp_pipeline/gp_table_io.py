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
    threshold_mode : {'white_noise', 'fap', 'red_noise', 'red_noise_subtracted'}
        Threshold used for LSP peak detection.
        'white_noise' : scalar 99th-pct bootstrap max (LSP_WN_threshold). Default.
        'fap'         : scalar 1% FAP analytic threshold (LSP_FAP_power / Baluev).
        'red_noise'   : per-frequency Vaughan 1% significance (LSP_red_noise_params).
                        Orders candidate peaks by P - gamma*C, which is biased toward
                        short periods.
        'red_noise_subtracted' : same acceptance test, but orders peaks by P - C, which
                        is unbiased in period. Pass excess_frac through **gp_kwargs to
                        relax the acceptance: it is a fraction of the significance gate,
                        so 1.0 (default) reproduces 'red_noise' exactly and 0.0 accepts
                        every bin above the continuum. No need to know gamma.
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
    gp_sho_taus                  object     list of decay times Q P / pi in days (length m);
                                            capped at gp_sho_tau_cap, so a value at the cap
                                            means "coherent for the whole window", not a
                                            measurement
    gp_sho_coherence             object     list of coherence fractions s in [0, 1] (length m):
                                            where Q sits between its floor 1/sqrt(2) and its
                                            period-dependent ceiling pi * tau_cap / P, in log.
                                            1 = pinned at the cap (gp_fit.q_from_coherence)
    gp_sho_tau_cap               float64    decay-time cap applied to this row in days
                                            (min(tau_cap_days, baseline); NaN = legacy flat Q box)
    gp_initial_lsp_peak_periods  object     LSP seed periods used to initialise fit
    gp_rn_sigma, gp_rn_period, gp_rn_Q
                                 float64    red-noise SHO parameters (rn_model='sho' only)
    gp_rn_alpha                  float64    fitted red-noise slope (rn_model='powerlaw' only)
    gp_rn_alpha_band             float64    in-band log-log slope of the fitted RN PSD ('powerlaw');
                                            compare THIS to measured slopes
    gp_rn_log10_S0               float64    log10 one-sided RN PSD at 1 /day ('powerlaw')
    gp_rn_rms_band               float64    sqrt of RN PSD integrated over the band ('powerlaw')
    gp_rn_basis_freqs            object     fixed OU basis break frequencies ('powerlaw')
    gp_rn_exposure_days          float64    exposure filter built into the RN basis (NaN = none)
    gp_n_peaks_found             float64    significant LSP peaks found, before max_components
    gp_components_capped         object     True when more peaks were found than max_components:
                                            leftover oscillator power biases gp_rn_alpha steep
    gp_rn_wn_ratio               float64    RN PSD / white-noise PSD at 1 /day ('powerlaw')
    gp_rn_near_white_noise       object     True when that ratio < rn_wn_flag_ratio: no detectable
                                            red noise, so gp_rn_alpha is undefined ('powerlaw')
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
    gp_rn_alpha_vals  = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_alpha_band_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_log10_S0_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_rms_band_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_wn_ratio_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_rn_near_wn_vals  = pd.Series([None] * n_rows, index=idx, dtype=object)

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
    gp_sho_taus_vals   = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_sho_coh_vals    = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_sho_tau_cap_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_init_lsp_vals   = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_rn_basis_vals   = pd.Series([None] * n_rows, index=idx, dtype=object)
    gp_rn_exposure_vals = pd.Series(np.nan, index=idx, dtype=float)
    gp_n_peaks_vals     = pd.Series(np.nan, index=idx, dtype=float)
    gp_capped_vals      = pd.Series([None] * n_rows, index=idx, dtype=object)

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
                thresh_kwargs = dict(fap_threshold=float(row.get("LSP_FAP_power")))
            elif threshold_mode in ('red_noise', 'red_noise_subtracted'):
                # Trailing two slots are placeholders: gp_fit unpacks P_N_empirical and
                # then never references it, and P_N_theoretical was always 1.0.
                thresh_kwargs = dict(red_noise_params=(
                    float(row["rn_log10_N"]), float(row["rn_alpha"]), 1.0, 1.0,
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
                gp_sho_taus_vals[label]   = [result.features.get(f"sho_{i}_tau_days", np.nan) for i in range(1, m + 1)]
                gp_sho_coh_vals[label]    = [result.features.get(f"sho_{i}_coherence", np.nan) for i in range(1, m + 1)]
                gp_sho_tau_cap_vals[label] = result.features.get("sho_tau_cap_days", np.nan)
                gp_init_lsp_vals[label]   = result.features["initial_lsp_peak_periods"]
                gp_rn_sigma_vals[label]   = result.features.get("rn_sigma",       np.nan)
                gp_rn_period_vals[label]  = result.features.get("rn_period_days", np.nan)
                gp_rn_Q_vals[label]       = result.features.get("rn_Q",           np.nan)
                gp_rn_alpha_vals[label]   = result.features.get("rn_alpha",       np.nan)
                gp_rn_alpha_band_vals[label] = result.features.get("rn_alpha_band", np.nan)
                gp_rn_log10_S0_vals[label] = result.features.get("rn_log10_S0",   np.nan)
                gp_rn_rms_band_vals[label] = result.features.get("rn_rms_band",   np.nan)
                gp_rn_wn_ratio_vals[label] = result.features.get("rn_wn_ratio",   np.nan)
                gp_rn_near_wn_vals[label]  = result.features.get("rn_near_white_noise")
                gp_rn_basis_vals[label]   = result.final_fit.get("rn_basis_freqs")
                _e = result.final_fit.get("rn_exposure_days")
                gp_rn_exposure_vals[label] = np.nan if _e is None else float(_e)
                gp_n_peaks_vals[label]     = result.features.get("lsp_n_peaks_found", np.nan)
                gp_capped_vals[label]      = result.features.get("components_capped")

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
    table["gp_sho_taus"]                 = gp_sho_taus_vals
    table["gp_sho_coherence"]            = gp_sho_coh_vals
    table["gp_sho_tau_cap"]              = gp_sho_tau_cap_vals
    table["gp_initial_lsp_peak_periods"] = gp_init_lsp_vals
    table["gp_rn_sigma"]                 = gp_rn_sigma_vals
    table["gp_rn_period"]                = gp_rn_period_vals
    table["gp_rn_Q"]                     = gp_rn_Q_vals
    table["gp_rn_alpha"]                 = gp_rn_alpha_vals
    table["gp_rn_alpha_band"]            = gp_rn_alpha_band_vals
    table["gp_rn_log10_S0"]              = gp_rn_log10_S0_vals
    table["gp_rn_rms_band"]              = gp_rn_rms_band_vals
    table["gp_rn_wn_ratio"]              = gp_rn_wn_ratio_vals
    table["gp_rn_near_white_noise"]      = gp_rn_near_wn_vals
    table["gp_rn_basis_freqs"]           = gp_rn_basis_vals
    table["gp_rn_exposure_days"]         = gp_rn_exposure_vals
    table["gp_n_peaks_found"]            = gp_n_peaks_vals
    table["gp_components_capped"]        = gp_capped_vals

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


def recompute_gp_psd_columns(table, freq_lim=(0.1, 12.0), n_bins=10, verbose=True):
    """Rebuild the GP PSD columns from stored parameters -- no refitting required.

    ``GP_freq``, ``GP_PSD`` and the ``gp_kspace_*`` family are a pure function of the
    fitted component parameters, all of which are already stored as table columns
    (``gp_sho_sigmas``, ``gp_sho_omegas``, ``gp_sho_Qs`` and ``gp_rn_sigma`` /
    ``gp_rn_period`` / ``gp_rn_Q``).  So they can be regenerated directly from a saved
    table without re-running ``add_gp_fits`` and without ``gp_result``, which the table
    writers drop anyway.

    The reason to need this: the analytic SHO PSD was corrected on 2026-08-21 (it fell as
    f^-2 rather than the kernel's f^-4).  Tables written before then carry stale spectra
    even though their fitted parameters are correct.

    Parameters
    ----------
    table : pd.DataFrame
        Must have ``gp_sho_sigmas``, ``gp_sho_omegas``, ``gp_sho_Qs``.  The red-noise
        term is included when ``gp_rn_alpha`` / ``gp_rn_log10_S0`` / ``gp_rn_basis_freqs``
        (power-law model) or ``gp_rn_sigma`` / ``gp_rn_period`` (SHO model) are present
        and finite.
    freq_lim : (float, float)
        Frequency range in 1/day.  Use the same values as the original ``add_gp_fits``
        call, or the band definitions will not line up.
    n_bins : int
        Number of log-spaced bins for the spectral statistics.

    Notes
    -----
    Reproduces ``add_gp_fits`` exactly, including its two different grids: ``GP_freq`` /
    ``GP_PSD`` are stored on ``kspace_true``'s default 1000-point grid, while the
    ``gp_kspace_*`` statistics are computed by ``add_gp_summary_stats`` on a 2000-point
    grid.  The mismatch is pre-existing; it is mirrored here rather than tidied so that
    recomputed columns are bit-comparable with freshly fitted ones.

    Returns
    -------
    table : pd.DataFrame
        Same object, columns replaced in-place.
    """
    from .gp_fit import sho_psd, powerlaw_rn_psd

    f_min, f_max = freq_lim
    freq = np.linspace(f_min, f_max, 1000)        # matches kspace_true default
    freq_stats = np.linspace(f_min, f_max, 2000)  # matches add_gp_summary_stats
    idx = table.index

    gp_freq_vals = pd.Series([None] * len(table), index=idx, dtype=object)
    gp_psd_vals = pd.Series([None] * len(table), index=idx, dtype=object)
    stats_list, stats_index = [], []
    n_ok = n_skip = 0

    for label, row in table.iterrows():
        sig = row.get('gp_sho_sigmas')
        om = row.get('gp_sho_omegas')
        Qs = row.get('gp_sho_Qs')
        if sig is None or om is None or Qs is None:
            n_skip += 1
            continue
        sig = np.atleast_1d(np.asarray(sig, dtype=float))
        om = np.atleast_1d(np.asarray(om, dtype=float))
        Qs = np.atleast_1d(np.asarray(Qs, dtype=float))
        if len(sig) == 0 or not (len(sig) == len(om) == len(Qs)):
            n_skip += 1
            continue

        def _total_psd(grid):
            out = np.zeros_like(grid)
            for s_, o_, q_ in zip(sig, om, Qs):
                out += sho_psd(grid, s_, o_, q_)
            rn_a, rn_S0 = row.get('gp_rn_alpha'), row.get('gp_rn_log10_S0')
            rn_fj = row.get('gp_rn_basis_freqs')
            if (rn_a is not None and rn_S0 is not None and rn_fj is not None
                    and np.isfinite(rn_a) and np.isfinite(rn_S0)):
                rn_e = row.get('gp_rn_exposure_days')
                out += powerlaw_rn_psd(grid, np.asarray(rn_fj, dtype=float),
                                       10.0 ** float(rn_S0), float(rn_a),
                                       exposure_days=(float(rn_e) if rn_e is not None
                                                      and np.isfinite(rn_e) else None))
                return out
            rn_s, rn_p = row.get('gp_rn_sigma'), row.get('gp_rn_period')
            rn_q = row.get('gp_rn_Q')
            if (rn_s is not None and rn_p is not None
                    and np.isfinite(rn_s) and np.isfinite(rn_p) and rn_p > 0):
                q = (float(rn_q) if (rn_q is not None and np.isfinite(rn_q))
                     else 1.0 / np.sqrt(2.0))
                out += sho_psd(grid, float(rn_s), 2.0 * np.pi / float(rn_p), q)
            return out

        power = _total_psd(freq)
        gp_freq_vals[label] = freq.copy()
        gp_psd_vals[label] = power
        stats_list.append(compute_spectral_stats(freq_stats, _total_psd(freq_stats),
                                                 freq_lim, n_bins))
        stats_index.append(label)
        n_ok += 1

    table['GP_freq'] = gp_freq_vals
    table['GP_PSD'] = gp_psd_vals

    # Mirror add_gp_fits: flat gp_kspace_* columns, plus the paired tuple columns.
    if stats_list:
        keys = list(stats_list[0].keys())
        for k in keys:
            col = f'gp_kspace_{k}'
            first = stats_list[0][k]
            vals = pd.Series([None] * len(table), index=idx, dtype=object) \
                if isinstance(first, (list, np.ndarray)) \
                else pd.Series(np.nan, index=idx, dtype=float)
            for lab, st in zip(stats_index, stats_list):
                vals[lab] = (np.asarray(st[k], dtype=float)
                             if isinstance(first, (list, np.ndarray)) else st[k])
            table[col] = vals

        binned = pd.Series([None] * len(table), index=idx, dtype=object)
        ratios = pd.Series([None] * len(table), index=idx, dtype=object)
        hl = pd.Series(np.nan, index=idx, dtype=float)
        for lab, st in zip(stats_index, stats_list):
            bc = st['bin_centers']
            binned[lab] = (st['log_band_powers'], bc)
            ratios[lab] = (st['log_band_ratios'],
                           [np.sqrt(bc[i] * bc[i + 1]) for i in range(len(bc) - 1)])
            hl[lab] = st['log_low_high_ratio']
        table['GP_PSD_binned'] = binned
        table['GP_PSD_bin_ratios'] = ratios
        table['GP_PSD_high_low_ratio'] = hl

    if verbose:
        print(f"[recompute_gp_psd_columns] ok={n_ok}  skipped={n_skip}  "
              f"freq_lim={freq_lim}  n_bins={n_bins}")
    return table


def filter_gp_components(table, sigma_rel_min=0.10, Q_min=None, n_cycles_min=1.6,
                         suffix='_filtered', verbose=True):
    """Prune low-amplitude / incoherent GP components, post hoc.

    The coherence cut is expressed in CYCLES, ``tau_d / P = Q / pi``, not in Q.  Since
    2026-11 the fit caps Q at pi * tau_cap / P (decay time at most one sector), so at
    P = 10 d the ceiling is Q = 8.5 and a flat ``Q >= 5`` would sit almost on the cap while
    at P = 0.1 d it would be 1% of it.  A cycles cut means the same thing at every period.
    The default ``n_cycles_min=1.6`` is exactly the old ``Q >= 5`` (5 / pi = 1.59).  Pass
    ``Q_min`` instead to cut on Q directly (then ``n_cycles_min`` is ignored).

    The fit is deliberately generous (``max_components=16``): a peak finder run without
    harmonic masking reports sidelobes and harmonics of a strong signal as separate peaks, so
    several seeds can belong to the same physical signal.  The GP handles that gracefully --
    on the synthetic set a duplicate component is fitted at a median 4% of the amplitude of
    the component that holds the real signal, and at Q ~ 40 against Q ~ 500 for a real match
    -- so duplicates waste budget rather than corrupting the fit.  That makes pruning a
    post-processing choice instead of something baked into the fit.

    Keeps components with ``sigma >= sigma_rel_min * max(sigma)`` in their own row AND
    ``Q >= Q_min``.  Measured on the synthetic set (611 detectable injections, max_comp=16):

        cut                       components   purity   completeness
        none                          1689      0.665       0.974
        sigma >= 5%                    754      0.862       0.926
        sigma >= 10%                   652      0.862       0.894
        Q >= 100                       667      0.895       0.841
        sigma >= 10% and Q >= 20       549      0.947       0.863
        sigma >= 10% and Q >= 100      454      0.954       0.802

    That table is measured on PURE SINUSOID injections, which are driven to the Q cap, and it
    over-sells the Q cut: real components sit in Q = 2-20 far more often (24% of real
    components vs 17% synthetic), because spot modulation is quasi-periodic.  On the real
    table Q_min=20 discarded clearly real signals -- e.g. NGC2136, an obvious 4.4 d modulation
    fitted at Q ~ 14.  Hence the default Q_min=5: it removes the broad continuum-like
    components while keeping finite-coherence signals.  Raise it only if purity matters more
    than completeness, and check what it cuts.

    Parameters
    ----------
    table : pd.DataFrame
        Needs ``gp_periods``, ``gp_sho_sigmas``, ``gp_sho_Qs`` (and ``gp_sho_omegas`` if present).
    sigma_rel_min : float
        Amplitude floor, as a fraction of the row's strongest component.
    n_cycles_min : float
        Coherence floor in cycles, Q / pi >= n_cycles_min.  Default 1.6 (= Q >= 5).
    Q_min : float or None
        Quality-factor floor; overrides ``n_cycles_min`` when given.
    suffix : str
        Added to the new column names.

    New columns
    -----------
    gp_periods{suffix}, gp_sho_sigmas{suffix}, gp_sho_Qs{suffix}, gp_sho_omegas{suffix}
    gp_n_components{suffix}   number of components kept
    gp_component_kept         list of bools aligned with ``gp_periods`` -- which components survived,
                              so a plot can draw kept and cut components differently
    """
    q_floor = float(Q_min) if Q_min is not None else float(n_cycles_min) * np.pi
    cut_label = f"Q >= {Q_min}" if Q_min is not None else f"cycles Q/pi >= {n_cycles_min} (Q >= {q_floor:.2f})"
    cols = ['gp_periods', 'gp_sho_sigmas', 'gp_sho_Qs', 'gp_sho_omegas', 'gp_sho_taus', 'gp_sho_coherence']
    present = [c for c in cols if c in table.columns]
    idx = table.index
    out = {c + suffix: pd.Series([None] * len(table), index=idx, dtype=object) for c in present}
    n_kept = pd.Series(np.nan, index=idx, dtype=float)
    kept_mask = pd.Series([None] * len(table), index=idx, dtype=object)

    for label, row in table.iterrows():
        sig = row.get('gp_sho_sigmas')
        Q = row.get('gp_sho_Qs')
        if sig is None or Q is None:
            continue
        sig = np.atleast_1d(np.asarray(sig, dtype=float))
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        if len(sig) != len(Q):
            continue
        if len(sig) == 0:
            # red-noise-only fit (no significant peaks): zero components, not "unknown"
            for c in present:
                out[c + suffix][label] = []
            n_kept[label] = 0
            kept_mask[label] = []
            continue
        smax = np.nanmax(sig)
        keep = np.isfinite(sig) & (sig >= sigma_rel_min * smax) & (Q >= q_floor) if smax > 0 else np.zeros(len(sig), bool)
        for c in present:
            v = np.atleast_1d(np.asarray(row.get(c), dtype=float))
            out[c + suffix][label] = list(v[keep]) if len(v) == len(keep) else None
        n_kept[label] = int(keep.sum())
        kept_mask[label] = [bool(k) for k in keep]      # aligned with gp_periods, for plotting

    for c, v in out.items():
        table[c] = v
    table['gp_n_components' + suffix] = n_kept
    table['gp_component_kept'] = kept_mask
    if verbose:
        tot = pd.to_numeric(table.get('gp_n_components'), errors='coerce').sum()
        print(f"[filter_gp_components] sigma_rel >= {sigma_rel_min}, {cut_label}: "
              f"kept {int(n_kept.sum())} of {int(tot)} components "
              f"({n_kept.sum() / max(tot, 1):.0%}) across {int(n_kept.notna().sum())} rows")
    return table

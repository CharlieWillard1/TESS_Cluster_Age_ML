"""Ground-truth statistics for the synthetic set.

Computed on the **noise-free, full-baseline** signal held in memory during generation;
the clean light curve is never written to disk, only the resulting numbers.

All statistics reuse the pipeline's own functions so the definitions cannot drift.  The
only exception is ``g1`` (the noise-corrected moment skewness), which the pipeline does
not compute -- it is three lines here, matching ``docs/statistics_formulas.tex``.

Error conventions
-----------------
The statistics disagree about what "no noise" means, so two conventions are used and
both are recorded:

* **Moment statistics** (``excess_var``, ``sigma_mad``, ``mean_median_offset``,
  ``gamma_p``, ``g1``) are computed with ``err -> 0``.  Then ``sigma_phot -> 0`` and
  ``excess_var`` reduces exactly to ``Var(signal)``, which is the quantity wanted.
  ``snr`` is *not* reported as truth: it would be ``excess_var / 0``.
* **``stetson_j``** is error-normalised and diverges as ``err -> 0``.  It is computed
  with the **real** per-point uncertainties, so truth means "J if these same error bars
  were attached to a noise-free signal".
* **``vn_ratio``** needs no uncertainties at all.

Known offsets vs the pipeline
-----------------------------
Truth spans the full ~1143 d baseline and is globally median-normalised; the pipeline
sees 27-d sectors normalised per sector.  For ``alpha = 1`` red-noise variance grows as
``ln(f_max * T)``, so truth carries roughly 1.6x the variance a single sector can reach.
Sinusoid variance is unaffected because all injected periods lie inside 0.1-9 d.  Expect
a consistent offset in ``excess_var``, not agreement -- see README.
"""

import numpy as np

from table_pipeline.summary_stats import (_compute_stats, von_neumann_ratio_gap_aware,
                                          J_stetson_gap_aware)

_TINY_ERR = 1e-12       # err -> 0 without dividing by zero downstream


def moment_skewness(x):
    """g1 = mu3 / sigma^3 on the raw series (no noise correction needed here).

    On noise-free data the noise-corrected and uncorrected forms coincide, since
    symmetric noise contributes zero third moment and there is no noise variance to
    remove from the denominator.
    """
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    s = np.sqrt(np.mean(d ** 2))
    return float(np.mean(d ** 3) / s ** 3) if s > 0 else np.nan


def truth_statistics(t, x, err_real=None):
    """Ground-truth summary statistics for one noise-free light curve.

    Parameters
    ----------
    t, x : ndarray
        Times (days) and noise-free fractional flux (mean ~1).
    err_real : ndarray or None
        Real per-point uncertainties in the same fractional units, used only for
        ``stetson_j``.  If None that entry is NaN.
    """
    x = np.asarray(x, dtype=float)
    tiny = np.full(len(x), _TINY_ERR)
    s = _compute_stats(x, tiny)

    out = {
        'true_excess_var':          s['excess_var'],       # == Var(signal)
        'true_sigma_mad':           s['sigma_mad_obs'],
        'true_mean_median_offset':  s['mean_median_offset'],
        'true_gamma_p':             s['gamma_p'],
        'true_g1':                  moment_skewness(x),
        'true_intrinsic_std':       s['intrinsic_std'],
        'true_vn_ratio':            von_neumann_ratio_gap_aware(t, x),
        'true_rms':                 float(np.sqrt(np.mean((x - x.mean()) ** 2))),
        'true_n_points':            int(len(x)),
        'true_baseline_days':       float(np.ptp(t)) if len(t) else np.nan,
    }
    out['true_stetson_j'] = (J_stetson_gap_aware(t, x, np.asarray(err_real, float))
                             if err_real is not None else np.nan)
    return out


def injected_signal_summary(params):
    """Quantities derived from the generating parameters alone.

    The summed sinusoid variance is analytic: independent sinusoids of amplitude A_i
    contribute A_i^2 / 2 each, so this is the part of ``true_excess_var`` the pipeline
    should be able to recover in full (all periods lie inside the search band).
    """
    A = np.asarray(params.get('amplitudes', []), dtype=float)
    P = np.asarray(params.get('periods', []), dtype=float)
    return {
        'n_osc':              int(params.get('n_osc', 0)),
        'inj_var_sinusoids':  float(np.sum(A ** 2) / 2.0) if len(A) else 0.0,
        'inj_amp_max':        float(np.max(A)) if len(A) else 0.0,
        'inj_period_of_max':  float(P[int(np.argmax(A))]) if len(A) else np.nan,
        'inj_amp_sum':        float(np.sum(A)) if len(A) else 0.0,
    }

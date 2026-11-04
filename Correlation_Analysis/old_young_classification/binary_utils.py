"""Shared pieces for the young/old classification notebooks.

The target is a single cut at 10^8 yr: ``old = log10(age) >= 8.0``.  Everything else --
the table, the feature transforms, the nuisance definitions, the cluster-grouped splits --
is imported unchanged from ``../analysis_utils.py`` so the classification results can be
compared directly with the regression ones.

Three things that make binary framing different from the regression, and which the
notebooks lean on:

1. **The classes are not balanced.**  129 young vs 219 old clusters, so a model that always
   answers "old" already scores 0.629 accuracy.  Raw accuracy is therefore not interpretable
   against 50%; ROC AUC and balanced accuracy are (both have a 0.5 null), and they are what
   the notebooks report.
2. **Galaxy predicts the label on its own.**  MW is 73% old, SMC 70%, LMC 47%.  Any claim
   about light curves has to beat a nuisance-only classifier, not the 0.5 null.
3. **AUC is a rank statistic.**  Per-feature AUC is a monotone transform of the Mann-Whitney
   U, so it measures exactly what a Spearman correlation measured in the regression version,
   on a scale where 0.5 means "no information" -- which is the comparison being asked for.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_PARENT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import analysis_utils as A            # noqa: E402,F401  (re-exported for the notebooks)

AGE_SPLIT = 8.0                       # log10(yr); old = >= 10^8 yr
LABELS = {0: 'young (<10⁸ yr)', 1: 'old (≥10⁸ yr)'}
COLOURS = {0: '#2a78d6', 1: '#eb6834'}
ORIGIN_COLOURS = {'MW': '#2a78d6', 'LMC': '#eb6834', 'SMC': '#1baf7a'}


def make_label(age, split=AGE_SPLIT):
    """1 = old, 0 = young.  NaN ages propagate as NaN so they can be dropped explicitly."""
    a = pd.to_numeric(pd.Series(age), errors='coerce')
    return np.where(a.isna(), np.nan, (a >= split).astype(float))


def class_baselines(y):
    """The three numbers every score in these notebooks must be read against."""
    y = np.asarray(y, float)
    p = np.nanmean(y)
    return dict(n=int(np.isfinite(y).sum()), n_old=int(np.nansum(y)),
                n_young=int(np.nansum(1 - y)), frac_old=float(p),
                majority_accuracy=float(max(p, 1 - p)),
                balanced_accuracy_null=0.5, auc_null=0.5)


def auc(x, y):
    """ROC AUC of a single feature against the binary label, via the rank identity.

    Equivalent to Mann-Whitney U / (n0*n1), so it is a rank statistic: invariant to any
    monotone transform of ``x``, exactly like the Spearman correlation it replaces.
    NaNs in ``x`` are dropped pairwise.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n1, n0 = float((y == 1).sum()), float((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(x, y, n_boot=2000, seed=0, groups=None):
    """Bootstrap CI for a feature AUC, resampling CLUSTERS when ``groups`` is given."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    obs = auc(x, y)
    if groups is None:
        idx_pool = [np.arange(len(x))]
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        by_g = {g: np.where(groups == g)[0] for g in uniq}
    out = []
    for _ in range(n_boot):
        if groups is None:
            i = rng.integers(0, len(x), len(x))
        else:
            pick = rng.choice(uniq, len(uniq), replace=True)
            i = np.concatenate([by_g[g] for g in pick])
        a = auc(x[i], y[i])
        if np.isfinite(a):
            out.append(a)
    if not out:
        return obs, np.nan, np.nan
    return obs, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def auc_permutation_p(x, y, strata=None, n_perm=2000, seed=0):
    """Two-sided p for AUC != 0.5, shuffling labels WITHIN ``strata`` (e.g. galaxy).

    Shuffling within galaxy removes the galaxy-driven part of the association, so the null
    answers "does this feature separate young from old beyond what knowing the galaxy gives",
    which is the question the nuisance controls ask elsewhere.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)
    obs = auc(x, y)
    if not np.isfinite(obs):
        return obs, np.nan
    strata = np.zeros(len(y)) if strata is None else np.asarray(strata)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for s in np.unique(strata):
            i = np.where(strata == s)[0]
            yp[i] = rng.permutation(y[i])
        a = auc(x, yp)
        if np.isfinite(a):
            null.append(a)
    null = np.asarray(null)
    p = (np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (len(null) + 1)
    return obs, float(p)


def residualise_on_nuisances(df, feature, nuisances):
    """Feature with the nuisance-predictable part removed (OLS on the design matrix).

    The partial analogue of ``A.partial_spearman`` for a binary target: the label is left
    alone and the FEATURE is cleaned, so the resulting AUC still has a 0.5 null and stays
    directly comparable with the raw AUC.
    """
    X = A.design_matrix(df, nuisances, add_const=True)
    v = pd.to_numeric(df[feature], errors='coerce')
    m = v.notna() & X.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index)
    if m.sum() > len(nuisances) + 5:
        Xm = X[m].to_numpy(float)
        beta, *_ = np.linalg.lstsq(Xm, v[m].to_numpy(float), rcond=None)
        out[m] = v[m].to_numpy(float) - Xm @ beta
    return out


def feature_auc_table(df, features, y, nuisances=None, strata=None, groups=None,
                      n_boot=1000, n_perm=1000, seed=0):
    """Raw AUC, nuisance-residualised AUC, CI, permutation p and BH-FDR, per feature.

    ``|AUC - 0.5|`` is reported as ``strength`` so features can be ranked regardless of
    which class they point at.
    """
    rows = []
    for f in features:
        a, lo, hi = auc_ci(df[f], y, n_boot=n_boot, seed=seed, groups=groups)
        _, p = auc_permutation_p(df[f], y, strata=strata, n_perm=n_perm, seed=seed)
        rec = dict(feature=f, auc=a, lo=lo, hi=hi, p=p, strength=abs(a - 0.5) if np.isfinite(a) else np.nan)
        if nuisances:
            rec['auc_partial'] = auc(residualise_on_nuisances(df, f, nuisances), y)
            rec['strength_partial'] = (abs(rec['auc_partial'] - 0.5)
                                       if np.isfinite(rec['auc_partial']) else np.nan)
            rec['lost_to_nuisances'] = rec['strength'] - rec['strength_partial']
        rows.append(rec)
    out = pd.DataFrame(rows)
    out['q'] = A.bh_fdr(out.p.fillna(1.0).to_numpy())
    return out.sort_values('strength', ascending=False).reset_index(drop=True)


def holdout_and_folds(groups, y, frac=0.2, n_splits=5, seed=0):
    """Cluster-grouped holdout + CV folds, stratified on the BINARY label (n_bins=2)."""
    hold = A.holdout_by_cluster(groups, y=y, frac=frac, seed=seed, n_bins=2, unit='clusters')
    fold = np.full(len(np.asarray(groups)), -1)
    pool = ~hold
    fold[pool] = A.group_folds(np.asarray(groups)[pool], y=np.asarray(y)[pool],
                               n_splits=n_splits, seed=seed, n_bins=2)
    return hold, fold

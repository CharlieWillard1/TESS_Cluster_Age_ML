"""Summary figures for the correlation analysis: the null result, with its uncertainties.

Recomputes the headline numbers of notebooks 01/02 (regression) and
old_young_classification/02 (young/old at 10^8 yr) with the SAME table, features, nuisances,
splits and forest settings, but keeps the per-fold scores, bootstrap draws and permutation
nulls so the scatter can be drawn rather than quoted.

    python make_summary.py            # compute (cached to summary_cache.pkl) + plot
    python make_summary.py --plot     # re-plot from the cache only

Run with the tess_gp env.  The permutation nulls dominate the runtime (~10-20 min).
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, PARENT)
sys.path.insert(0, os.path.join(PARENT, 'old_young_classification'))
import analysis_utils as A            # noqa: E402
import binary_utils as B              # noqa: E402

GT = '/astro/users/cgwill/TESS_Cluster_Age_ML/GenerateTable_CGW/data'
TABLE = f'{GT}/cluster_table_Sep23_all_combos_le3sectors.pkl'
CACHE = os.path.join(HERE, 'summary_cache.pkl')
SEED = 0
N_PERM = int(os.environ.get("N_PERM", 100))   # within-galaxy label shuffles per null
N_JOBS = 12           # permutations run in parallel, one core per forest


# ---------------------------------------------------------------------------------------
# Data, exactly as in the notebooks
# ---------------------------------------------------------------------------------------

def load():
    rows = A.load_rows(TABLE, verbose=False)
    meta = A.cluster_metadata()
    clusters = A.cluster_table(rows, meta=meta)
    clusters = clusters[clusters.age.notna()].reset_index(drop=True)
    features = [c for c, _, _ in A.feature_columns(rows)]
    nuisances = [n for n, _, _ in A.NUISANCE_SPEC if n in clusters.columns]
    reliability = A.feature_reliability(rows)
    clusters_f, families, fam_cols = A.build_family_table(clusters, features, threshold=0.8,
                                                          reliability=reliability)
    return clusters_f, features, nuisances, families, fam_cols


# ---------------------------------------------------------------------------------------
# Linear: marginal vs partial rank correlation per family, with cluster-bootstrap CIs
# ---------------------------------------------------------------------------------------

def linear(clusters, nuisances, families, fam_cols):
    out = []
    for col in fam_cols:
        rep = col.replace('FAM_', '')
        m_rho, _, (mlo, mhi) = A.spearman_ci(clusters[col], clusters.age, n_boot=1000, seed=SEED)
        p_rho, p_p, (plo, phi), n = A.partial_spearman_ci(clusters, col, 'age', nuisances,
                                                          n_boot=1000, seed=SEED)
        _, p_perm = A.stratified_permutation_p(clusters, col, 'age', 'origin', n_perm=2000,
                                               seed=SEED)
        members = families[families.family == rep].feature.tolist()
        out.append(dict(family=rep, members=members, n=n, rho=m_rho, rho_lo=mlo, rho_hi=mhi,
                        rho_partial=p_rho, partial_lo=plo, partial_hi=phi, p_partial=p_p,
                        p_perm_within_galaxy=p_perm))
    tab = pd.DataFrame(out)
    tab['p_partial_fdr'] = A.bh_fdr(tab.p_partial.fillna(1))
    tab['p_perm_fdr'] = A.bh_fdr(tab.p_perm_within_galaxy.fillna(1))
    return tab.sort_values('rho_partial', key=abs, ascending=False).reset_index(drop=True)


def partial_residuals(clusters, col, nuisances):
    """The two residualised rank vectors that partial_spearman correlates."""
    sub = clusters[[col, 'age', 'origin'] + [n for n in nuisances if n != 'origin']].dropna()
    X = A.design_matrix(sub, nuisances)
    rx = A._residualise(sub[col].rank(), X)
    ry = A._residualise(sub.age.rank(), X)
    m = np.isfinite(rx) & np.isfinite(ry)
    return dict(x=rx[m], y=ry[m], origin=sub.origin.values[m], raw_x=sub[col].rank().values[m],
                raw_y=sub.age.values[m])


# ---------------------------------------------------------------------------------------
# Random forests: regression (log age) and classification (old = >= 10^8 yr)
# ---------------------------------------------------------------------------------------

def _forest(kind, seed, n_jobs):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    if kind == 'reg':
        rf = RandomForestRegressor(n_estimators=600, min_samples_leaf=3, max_features='sqrt',
                                   random_state=seed, n_jobs=n_jobs)
    else:
        rf = RandomForestClassifier(n_estimators=600, min_samples_leaf=3, max_features='sqrt',
                                    class_weight='balanced', random_state=seed, n_jobs=n_jobs)
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('rf', rf)])


def reg_cv(X, y, n_repeats, seed, n_jobs=8):
    """Per-fold out-of-fold R2, RepeatedKFold as in notebook 02."""
    from sklearn.metrics import r2_score
    from sklearn.model_selection import RepeatedKFold
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    r2 = []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        m = _forest('reg', seed + k, n_jobs).fit(X.iloc[tr], y[tr])
        r2.append(r2_score(y[te], m.predict(X.iloc[te])))
    return np.array(r2)


def clf_oof(X, y, fold, seed, n_jobs=8):
    """Out-of-fold probabilities on the fixed cluster-grouped folds of the classification nb."""
    p = np.full(len(y), np.nan)
    for k in range(fold.max() + 1):
        tr, te = (fold != k) & (fold >= 0), fold == k
        m = _forest('clf', seed + k, n_jobs).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def shuffle_within(y, origin, rng):
    y = y.copy()
    for o in np.unique(origin):
        i = np.where(origin == o)[0]
        y[i] = rng.permutation(y[i])
    return y


def _reg_delta(Xn, Xf, y, seed):
    return (reg_cv(pd.concat([Xn, Xf], axis=1), y, 2, seed, n_jobs=1).mean()
            - reg_cv(Xn, y, 2, seed, n_jobs=1).mean())


def _clf_delta(Xn, Xfull, y, fold, seed):
    from sklearn.metrics import roc_auc_score
    m = fold >= 0
    pf = clf_oof(Xfull, y, fold, seed, n_jobs=1)
    pn = clf_oof(Xn, y, fold, seed, n_jobs=1)
    return roc_auc_score(y[m], pf[m]) - roc_auc_score(y[m], pn[m])


def forests(clusters, features, nuisances):
    from joblib import Parallel, delayed
    from sklearn.metrics import r2_score, roc_auc_score

    X_nuis = A.design_matrix(clusters, nuisances, add_const=False)
    X_feat = clusters[features].apply(pd.to_numeric, errors='coerce')
    X_full = pd.concat([X_nuis, X_feat], axis=1)
    sets = {'nuisance-only': X_nuis, 'features-only': X_feat, 'full': X_full}
    origin = clusters.origin.values
    res = {}

    # --- regression, as notebook 02 -----------------------------------------------------
    y = clusters.age.values
    hold = A.holdout_by_cluster(clusters.name.values, y=y, frac=0.2, seed=SEED)
    pool = ~hold
    reg = {'folds': {}, 'holdout': {}}
    for lab, X in sets.items():
        reg['folds'][lab] = reg_cv(X[pool], y[pool], 5, SEED)
        m = _forest('reg', SEED, 8).fit(X[pool], y[pool])
        reg['holdout'][lab] = r2_score(y[hold], m.predict(X[hold]))
        print(f'  reg {lab:>14}: CV R2 {reg["folds"][lab].mean():+.3f} '
              f'+/- {reg["folds"][lab].std():.3f} | holdout {reg["holdout"][lab]:+.3f}', flush=True)
    Xn_p, Xf_p, y_p, o_p = X_nuis[pool], X_feat[pool], y[pool], origin[pool]
    reg['obs_delta'] = _reg_delta(Xn_p, Xf_p, y_p, SEED)
    rng = np.random.default_rng(SEED)
    perms = [shuffle_within(y_p, o_p, rng) for _ in range(N_PERM)]
    reg['null_delta'] = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(_reg_delta)(Xn_p, Xf_p, yp, SEED + 100 + i) for i, yp in enumerate(perms)))
    reg['n_pool'], reg['n_hold'] = int(pool.sum()), int(hold.sum())
    print(f'  reg dR2 obs {reg["obs_delta"]:+.3f} | null {reg["null_delta"].mean():+.3f} '
          f'+/- {reg["null_delta"].std():.3f}', flush=True)
    res['reg'] = reg

    # --- classification, as old_young_classification/02 --------------------------------
    yb = B.make_label(clusters.age).astype(int)
    hold_b, fold = B.holdout_and_folds(clusters.name.values, yb, frac=0.2, n_splits=5, seed=SEED)
    m = fold >= 0
    clf = {'oof_auc': {}, 'boot_auc': {}, 'holdout': {}}
    oof = {}
    for lab, X in sets.items():
        oof[lab] = clf_oof(X, yb, fold, SEED)
        clf['oof_auc'][lab] = roc_auc_score(yb[m], oof[lab][m])
        mdl = _forest('clf', SEED, 8).fit(X[m], yb[m])
        clf['holdout'][lab] = roc_auc_score(yb[hold_b], mdl.predict_proba(X[hold_b])[:, 1])
    # paired cluster bootstrap of the OOF AUCs (one row per cluster)
    idx = np.where(m)[0]
    rng = np.random.default_rng(SEED)
    boots = {lab: [] for lab in sets}
    for _ in range(2000):
        i = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(yb[i])) < 2:
            continue
        for lab in sets:
            boots[lab].append(roc_auc_score(yb[i], oof[lab][i]))
    clf['boot_auc'] = {k: np.array(v) for k, v in boots.items()}
    clf['obs_delta'] = clf['oof_auc']['full'] - clf['oof_auc']['nuisance-only']
    Xn_np, Xfull_np = X_nuis.to_numpy(float), X_full.to_numpy(float)
    perms = [np.where(m, shuffle_within(np.where(m, yb, 0), np.where(m, origin, '_hold'), rng), yb)
             for _ in range(N_PERM)]
    clf['null_delta'] = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(_clf_delta)(Xn_np, Xfull_np, yp, fold, SEED + 100 + i)
        for i, yp in enumerate(perms)))
    clf['n_pool'], clf['n_hold'] = int(m.sum()), int(hold_b.sum())
    clf['n_young'], clf['n_old'] = int((yb == 0).sum()), int((yb == 1).sum())
    print(f'  clf dAUC obs {clf["obs_delta"]:+.3f} | null {clf["null_delta"].mean():+.3f} '
          f'+/- {clf["null_delta"].std():.3f}', flush=True)
    res['clf'] = clf
    return res


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------

INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8f8e88', '#e6e5e0', '#fcfcfb'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GAL = {'MW': BLUE, 'LMC': ORANGE, 'SMC': AQUA}
PRETTY = {
    'sigma_mad': 'Variability amplitude (σ_MAD, excess var, …)',
    'stetson_j_gap_aware': 'Noise redness (Stetson J, von Neumann, α)',
    'gp_n_components_filtered': 'GP: number of oscillators',
    'LSP_peak_snr_rn': 'Periodogram peak S/N',
    'gp_rn_rms_band': 'GP red-noise amplitude',
    'gp_jitter': 'GP white-noise jitter',
    'gp_rn_wn_ratio': 'GP red/white noise ratio',
    'g1_int_sec': 'Skewness (g1)',
    'gp_period_strongest': 'GP strongest period',
    'mean_median_offset': 'Mean − median offset',
    'gp_Q_strongest': 'GP strongest Q',
    'LSP_peak_period_diff': 'Periodogram peak period',
}


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': MUTED,
        'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'legend.frameon': False, 'font.size': 10, 'axes.titlesize': 11,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
    return plt


def fig_linear(lin, resid, path):
    plt = _style()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

    t = lin.iloc[::-1].reset_index(drop=True)
    yy = np.arange(len(t))
    ax.axvspan(-0.1, 0.1, color=GRID, alpha=.6, lw=0, zorder=0)
    ax.axvline(0, color=MUTED, lw=1)
    ax.errorbar(t.rho, yy + 0.18, xerr=[t.rho - t.rho_lo, t.rho_hi - t.rho], fmt='o', ms=6,
                mfc=SURF, mec=MUTED, ecolor=MUTED, elinewidth=1.6, capsize=0,
                label='raw (no controls)')
    ax.errorbar(t.rho_partial, yy - 0.18, xerr=[t.rho_partial - t.partial_lo,
                                                 t.partial_hi - t.rho_partial],
                fmt='o', ms=7, color=BLUE, mec=SURF, mew=1.2, elinewidth=2, capsize=0,
                label='partial (galaxy + observing setup removed)')
    for k, r in t.iterrows():
        if r.p_partial_fdr < 0.05:
            ax.text(0.33, k - 0.18, f'ρ²={r.rho_partial**2:.0%}', va='center', ha='left',
                    fontsize=8.5, color=INK2)
    ax.set_yticks(yy, [PRETTY.get(f, f) for f in t.family])
    ax.set_xlim(-0.5, 0.42)
    ax.set_xlabel('Spearman ρ with log age   (bars: 95% cluster-bootstrap CI)')
    ax.set_title('a · Every family is weak; controls shrink the strongest')
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=8.5)
    ax.text(0.33, len(t) - 0.45, 'ρ² (partial)', fontsize=8.5, color=INK2, fontweight='bold')
    ax.text(0, len(t) - 0.35, '|ρ|<0.1', ha='center', fontsize=8, color=MUTED)

    # strongest relation: residual ranks, coloured by galaxy
    top = lin.iloc[0]
    for o in ['MW', 'LMC', 'SMC']:
        s = resid['origin'] == o
        ax2.scatter(resid['x'][s], resid['y'][s], s=16, color=GAL[o], alpha=.65, lw=0, label=o)
    b = np.polyfit(resid['x'], resid['y'], 1)
    xs = np.linspace(resid['x'].min(), resid['x'].max(), 50)
    ax2.plot(xs, np.polyval(b, xs), color=INK, lw=2)
    ax2.set_xlabel(f'{PRETTY[top.family].split(" (")[0]} — rank, nuisances removed')
    ax2.set_ylabel('log age — rank, nuisances removed')
    ax2.set_title('b · Strongest surviving relation')
    ax2.legend(loc='upper right', fontsize=8.5, markerscale=1.4)
    ax2.text(0.02, 0.03,
             f'ρ = {top.rho_partial:.2f}  [{top.partial_lo:.2f}, {top.partial_hi:.2f}]\n'
             f'ρ² ≈ {top.rho_partial**2:.0%} of age-rank variance; '
             f'{1 - top.rho_partial**2:.0%} is scatter', transform=ax2.transAxes, fontsize=9,
             color=INK, va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', fc=SURF, ec=GRID))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_forest(res, path):
    plt = _style()
    reg, clf = res['reg'], res['clf']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    labs = ['nuisance-only', 'features-only', 'full']
    cols = {'nuisance-only': MUTED, 'features-only': BLUE, 'full': INK}
    pretty = {'nuisance-only': 'nuisances\nonly', 'features-only': 'light curve\nonly',
              'full': 'nuisances +\nlight curve'}
    rng = np.random.default_rng(1)

    # (a) regression R2 per fold
    ax = axes[0, 0]
    for k, lab in enumerate(labs):
        f = reg['folds'][lab]
        ax.scatter(k + rng.uniform(-.13, .13, len(f)), f, s=18, color=cols[lab], alpha=.45, lw=0)
        ax.errorbar(k + .28, f.mean(), yerr=f.std(), fmt='o', color=cols[lab], ms=8, elinewidth=2.2)
        ax.scatter(k + .42, reg['holdout'][lab], marker='D', s=46, color=SURF, edgecolor=cols[lab],
                   lw=1.8, zorder=3)
        ax.text(k + .28, f.mean() + f.std() + .03, f'{f.mean():+.2f}\n±{f.std():.2f}', ha='center',
                fontsize=8.5, color=INK2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(3), [pretty[l] for l in labs])
    ax.set_xlim(-.5, 2.7)
    ax.set_ylabel('out-of-fold R² (log age)')
    ax.set_title('a · Regression: fold-to-fold scatter swamps the gain')
    ax.scatter([], [], s=18, color=MUTED, label='one CV fold (5×5)')
    ax.errorbar([], [], yerr=[], fmt='o', color=MUTED, label='mean ± 1σ over folds')
    ax.scatter([], [], marker='D', s=40, color=SURF, edgecolor=MUTED, lw=1.8,
               label=f'held-out clusters (n={reg["n_hold"]})')
    ax.legend(loc='lower right', fontsize=8)

    def null_panel(ax, null, obs, xlabel, title):
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        ax.hist(null, bins=18, color=MUTED, alpha=.75, lw=0,
                label=f'age shuffled within galaxy ({len(null)}×)')
        ax.axvline(obs, color=ORANGE, lw=2.5, label=f'observed {obs:+.3f}')
        ax.axvline(0, color=INK2, lw=1, ls=':')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('permutations')
        ax.set_title(title)
        ax.text(0.98, 0.95, f'p = {p:.2f}\nnull {null.mean():+.3f} ± {null.std():.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc=SURF, ec=GRID))
        ax.legend(loc='upper left', fontsize=8)
        return p

    reg['p'] = null_panel(axes[0, 1], reg['null_delta'], reg['obs_delta'],
                          'ΔR² = full − nuisance-only (CV)',
                          'b · Regression: light-curve gain vs chance')

    # (c) classification AUC with bootstrap CI
    ax = axes[1, 0]
    for k, lab in enumerate(labs):
        bt = clf['boot_auc'][lab]
        parts = ax.violinplot(bt, positions=[k], widths=.55, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(cols[lab]); pc.set_alpha(.25); pc.set_edgecolor('none')
        lo, hi = np.percentile(bt, [2.5, 97.5])
        ax.errorbar(k, clf['oof_auc'][lab], yerr=[[clf['oof_auc'][lab] - lo], [hi - clf['oof_auc'][lab]]],
                    fmt='o', color=cols[lab], ms=8, elinewidth=2.2)
        ax.scatter(k + .3, clf['holdout'][lab], marker='D', s=46, color=SURF, edgecolor=cols[lab],
                   lw=1.8, zorder=3)
    ax.axhline(0.5, color=MUTED, lw=1, ls=':')
    ax.text(-.7, 0.505, 'guessing (AUC 0.5)', fontsize=8, color=MUTED, ha='left', va='bottom')
    ci = {l: np.percentile(clf['boot_auc'][l], [2.5, 97.5]) for l in labs}
    ax.set_xticks(range(3), [f'{pretty[l]}\n{clf["oof_auc"][l]:.2f} [{ci[l][0]:.2f}, {ci[l][1]:.2f}]'
                             for l in labs])
    ax.set_xlim(-.75, 2.5)
    ax.set_ylim(0.40, 0.86)
    ax.set_ylabel('ROC AUC, young vs old (10⁸ yr)')
    ax.set_title('c · Young/old split: same story')
    ax.errorbar([], [], yerr=[], fmt='o', color=MUTED, label='out-of-fold AUC, 95% bootstrap CI')
    ax.scatter([], [], marker='D', s=40, color=SURF, edgecolor=MUTED, lw=1.8,
               label=f'held-out clusters (n={clf["n_hold"]})')
    ax.legend(loc='lower right', fontsize=8)

    clf['p'] = null_panel(axes[1, 1], clf['null_delta'], clf['obs_delta'],
                          'ΔAUC = full − nuisance-only (CV)',
                          'd · Young/old: light-curve gain vs chance')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    if '--plot' not in sys.argv or not os.path.exists(CACHE):
        clusters, features, nuisances, families, fam_cols = load()
        print(f'{len(clusters)} clusters, {len(features)} features, {len(fam_cols)} families',
              flush=True)
        lin = linear(clusters, nuisances, families, fam_cols)
        resid = partial_residuals(clusters, 'FAM_' + lin.family.iloc[0], nuisances)
        print(lin.drop(columns='members').round(3).to_string(), flush=True)
        res = forests(clusters, features, nuisances)
        with open(CACHE, 'wb') as fh:
            pickle.dump(dict(lin=lin, resid=resid, res=res), fh)
    with open(CACHE, 'rb') as fh:
        c = pickle.load(fh)
    fig_linear(c['lin'], c['resid'], os.path.join(HERE, 'fig1_linear_correlations.png'))
    fig_forest(c['res'], os.path.join(HERE, 'fig2_random_forest_null.png'))
    c['lin'].drop(columns='members').to_csv(os.path.join(HERE, 'summary_linear.csv'), index=False)
    print('wrote figures to', HERE)


if __name__ == '__main__':
    main()

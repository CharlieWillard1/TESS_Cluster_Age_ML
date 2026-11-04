"""Shared data preparation and statistics for the age-correlation notebooks.

The two notebooks (01 linear, 02 random forest) must see exactly the same table, the same
feature transforms and the same nuisance definitions, or their conclusions cannot be
compared.  That is what lives here.

Three things drive every design choice in this module:

1. **Rows are not independent.**  The pipeline emits one row per sector combination, and
   combinations of the same cluster share sectors -- up to 409 rows for one LMC cluster in
   the full <=3-sector table.  Treating rows as independent would inflate every significance
   by roughly sqrt(n_rows per cluster).  So the unit of analysis is the CLUSTER
   (``cluster_table``), and anything row-level resamples or splits by cluster.

2. **Galaxy is confounded with age.**  LMC clusters in this sample are young (mean log age
   7.87, none above 8.8) while MW spans 6.4-10.1.  Distance, crowding and metallicity all
   differ by galaxy too, so any feature tracking brightness or noise will correlate with age
   through origin alone.  Hence the within-origin and partial analyses.

3. **Brightness and cadence drive the variability statistics directly.**  A fainter cluster
   has a higher photometric floor, which changes excess variance, the periodogram continuum
   and the GP amplitudes.  These are nuisances, not discoveries.

Flags (``gp_rn_near_white_noise``, ``gp_components_capped``) are deliberately NOT nuisance
regressors: they are outcomes of the fit and correlate with age themselves (the capped
fraction has rho = -0.18 against age), so conditioning on them opens a collider path.  Use
them to describe the sample, or as sensitivity checks, not as controls.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------
# Feature and nuisance definitions
# ---------------------------------------------------------------------------------------

# (column, transform, family).  'log10' is applied to strictly positive quantities so that
# a multiplicative spread becomes additive; Spearman is rank-based and unaffected, but the
# linear models and the forest both behave better on the transformed scale.
FEATURE_SPEC = [
    # amplitude of variability
    ('excess_var',            'log10', 'amplitude'),
    ('intrinsic_std',         'log10', 'amplitude'),
    ('sigma_mad',             'log10', 'amplitude'),
    ('intrinsic_rms',         'log10', 'amplitude'),
    # shape / distribution of the light curve
    ('vn_ratio_gap_aware',    'none',  'shape'),
    ('stetson_j_gap_aware',   'signed_log10', 'shape'),
    ('gamma_p_sec',           'none',  'shape'),
    ('g1_int_sec',            'none',  'shape'),
    ('mean_median_offset',    'none',  'shape'),
    # periodogram
    ('LSP_peak_period_diff',  'log10', 'period'),
    ('LSP_peak_snr_at_diff',  'log10', 'periodogram'),
    ('LSP_peak_snr_rn',       'log10', 'periodogram'),
    ('rn_alpha',              'none',  'red noise (LSP)'),
    ('rn_log10_N',            'none',  'red noise (LSP)'),
    # GP: red noise
    ('gp_rn_alpha_band',      'none',  'red noise (GP)'),
    ('gp_rn_log10_S0',        'none',  'red noise (GP)'),
    ('gp_rn_rms_band',        'log10', 'red noise (GP)'),
    ('gp_rn_wn_ratio',        'log10', 'red noise (GP)'),
    # GP: oscillators
    ('gp_n_components_filtered', 'none', 'oscillators'),
    ('gp_sigma_max',          'log10', 'oscillators'),
    ('gp_period_strongest',   'log10', 'oscillators'),
    ('gp_Q_strongest',        'log10', 'oscillators'),
    ('gp_jitter',             'log10', 'noise'),
]

# Nuisances: observational or galaxy-level properties that plausibly drive the features and
# are also associated with age through sample selection.
NUISANCE_SPEC = [
    ('origin',            'categorical', 'galaxy (distance, metallicity, crowding)'),
    ('log_cadence',       'numeric',     'observing cadence, 3.33/10/30 min'),
    ('n_sectors',         'numeric',     'how many sectors were stitched'),
    ('log_n_points',      'numeric',     'light-curve length after resampling'),
    ('log_med_err_frac',  'numeric',     'photometric noise floor (fractional)'),
    ('log_med_flux',      'numeric',     'brightness, a distance/crowding proxy'),
    ('cluster_radius',    'numeric',     'aperture size from the ELK header'),
    ('dec',               'numeric',     'declination'),
    ('gal_b',             'numeric',     'galactic latitude, a crowding proxy'),
]
# NOTE: log_med_flux is the median of the *normalised* light curve, so it is identically 0
# (log10 1).  It is kept above only so older notebooks reproduce; it carries no information.
# Use log_F below for brightness.

# Brightness / noise reparameterisation (see brightness_nuisances.tex).  log_med_err_frac is
# ~ F^-1/2 (photon noise), so it mixes a measurement nuisance (the noise floor) with
# brightness, which in the LMC is itself an age correlator (M/L).  These split it into a
# brightness term that can be switched on and off and a noise term that is (nearly)
# independent of brightness.  Added by add_brightness_nuisances(), not by load_rows(), so
# notebooks that do not ask for them are unchanged.
BRIGHTNESS_SPEC = [
    ('log_F',          'numeric', 'aperture brightness: log10 median raw ELK flux [e/s]'),
    ('log_noise_q',    'numeric', 'log10(err_frac * sqrt(F)) = log10(sigma / sqrt(F)): noise per photon-noise unit'),
    ('log_noise_perp', 'numeric', 'log_med_err_frac residual on log_F (per galaxy): brightness-orthogonal noise'),
]


def _decode(x):
    return x.decode() if isinstance(x, bytes) else x


# ---------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------

def load_rows(path, verbose=True):
    """Load a pipeline table and add the per-row quantities the analysis needs.

    Derived here rather than in the notebooks so both use identical definitions:
    brightness and noise proxies from the stored light curve, and scalar summaries of the
    GP component lists (strongest component by amplitude, after the quality filter).
    """
    df = pd.read_pickle(path)
    for c in ('name', 'origin'):
        if c in df:
            df[c] = df[c].map(_decode)

    # ---- brightness / noise proxies, straight from the light curve ----------------------
    if 'LC_flux' in df:
        med_flux, med_err, npts, span = [], [], [], []
        for f, e, t in zip(df['LC_flux'], df['LC_flux_err'], df['LC_t']):
            f = np.asarray(f, float); e = np.asarray(e, float); t = np.asarray(t, float)
            mf = np.nanmedian(f)
            med_flux.append(mf)
            med_err.append(np.nanmedian(e) / mf if mf > 0 else np.nan)
            npts.append(len(t)); span.append(np.ptp(t) if len(t) > 1 else np.nan)
        df['med_flux'] = med_flux
        df['med_err_frac'] = med_err
        df['n_points'] = npts
        df['baseline_days'] = span

    # ---- scalar summaries of the GP component lists ------------------------------------
    per, sig, qq, ncomp = [], [], [], []
    pcol = 'gp_periods_filtered' if 'gp_periods_filtered' in df else 'gp_periods'
    scol = 'gp_sho_sigmas_filtered' if 'gp_sho_sigmas_filtered' in df else 'gp_sho_sigmas'
    qcol = 'gp_sho_Qs_filtered' if 'gp_sho_Qs_filtered' in df else 'gp_sho_Qs'
    for P, S, Q in zip(df.get(pcol, []), df.get(scol, []), df.get(qcol, [])):
        P = np.atleast_1d(np.asarray(P if P is not None else [], float))
        S = np.atleast_1d(np.asarray(S if S is not None else [], float))
        Q = np.atleast_1d(np.asarray(Q if Q is not None else [], float))
        if len(S) and len(S) == len(P) == len(Q):
            k = int(np.nanargmax(S))                 # "the" signal = the strongest component
            per.append(P[k]); sig.append(S[k]); qq.append(Q[k]); ncomp.append(len(S))
        else:
            per.append(np.nan); sig.append(np.nan); qq.append(np.nan); ncomp.append(0)
    df['gp_period_strongest'] = per
    df['gp_sigma_max'] = sig
    df['gp_Q_strongest'] = qq
    if 'gp_n_components_filtered' not in df:
        df['gp_n_components_filtered'] = ncomp

    # ---- nuisance columns ---------------------------------------------------------------
    df['log_cadence'] = np.log10(pd.to_numeric(df['cadence'], errors='coerce'))
    if 'n_points' in df:
        df['log_n_points'] = np.log10(df['n_points'].clip(lower=1))
        df['log_med_err_frac'] = np.log10(df['med_err_frac'].clip(lower=1e-12))
        df['log_med_flux'] = np.log10(df['med_flux'].clip(lower=1e-12))
    df['age'] = pd.to_numeric(df['age'], errors='coerce')

    if verbose:
        print(f'{len(df)} rows, {df.name.nunique()} clusters from {os.path.basename(path)}')
        print('rows per cluster: median %.0f, max %d' % (df.groupby('name').size().median(),
                                                         df.groupby('name').size().max()))
    return df


def cluster_metadata(lc_dir='/astro/users/cgwill/TESS_Cluster_Age_ML/light_curves'):
    """Sky position and aperture radius per cluster, from the ELK primary headers.

    RA/DEC are not in the pipeline table, but every light-curve file carries LOCATION and
    RADIUS.  Galactic latitude is added as a crowding proxy: an MW cluster in the plane sits
    in a far denser field than one out of it, which affects the photometry.
    """
    import glob
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    rows = []
    for p in glob.glob(f'{lc_dir}/*/*.fits'):
        with fits.open(p) as h:
            hdr = h[0].header
            loc = hdr.get('LOCATION', '')
            try:
                ra, dec = [float(v) for v in str(loc).split(',')]
            except Exception:
                ra = dec = np.nan
            rows.append(dict(name=_decode(hdr.get('NAME')), ra=ra, dec=dec,
                             cluster_radius=hdr.get('RADIUS', np.nan)))
    meta = pd.DataFrame(rows).dropna(subset=['name']).drop_duplicates('name')
    ok = meta[['ra', 'dec']].notna().all(axis=1)
    gal_l = np.full(len(meta), np.nan); gal_b = np.full(len(meta), np.nan)
    if ok.any():
        c = SkyCoord(ra=meta.loc[ok, 'ra'].values * u.deg,
                     dec=meta.loc[ok, 'dec'].values * u.deg).galactic
        gal_l[ok.values] = c.l.deg; gal_b[ok.values] = c.b.deg
    meta['gal_l'] = gal_l
    meta['gal_b'] = gal_b
    return meta


def sector_raw_flux(lc_dir='/astro/users/cgwill/TESS_Cluster_Age_ML/light_curves',
                    cache=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sector_raw_flux.csv'),
                    refresh=False):
    """Median raw (un-normalised) ELK flux of every sector, one row per (cluster, sector).

    `sector` is the 0-based data-HDU index, the same convention as the pipeline's `sectors`
    column (HDU = sector + 1).  Points are kept with the same finiteness mask the pipeline
    uses before resampling.  Reading every FITS file takes a few minutes, so the result is
    cached as CSV next to this module.
    """
    if os.path.exists(cache) and not refresh:
        return pd.read_csv(cache)
    import glob
    from astropy.io import fits

    rows = []
    for p in sorted(glob.glob(f'{lc_dir}/*/*.fits')):
        with fits.open(p) as h:
            name = _decode(h[0].header.get('NAME'))
            for s, ext in enumerate(h[1:]):
                d = ext.data
                if d is None or len(d) == 0:
                    continue
                t = np.asarray(d['time'], float); f = np.asarray(d['flux'], float)
                e = np.asarray(d['flux_err'], float)
                ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
                rows.append(dict(name=name, sector=s,
                                 med_flux_raw=np.median(f[ok]) if ok.any() else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def add_brightness_nuisances(rows, perp_slopes=None, verbose=True, **kw):
    """Add log_F, log_noise_q and log_noise_perp (BRIGHTNESS_SPEC) to a row table.

    log_F          log10 of the median over the combination's sectors of each sector's median
                   raw flux (the per-combination version of crowd's F_meas).
    log_noise_q    log_med_err_frac + 0.5 log_F.  err_frac is the median normalised error of
                   the light curve actually analysed (after 30-min resampling), so this is
                   sigma_raw / sqrt(F): ~constant for pure photon noise, larger where sky or
                   other extra noise matters.  Exactly a relabelling: {log_F, log_noise_q}
                   spans the same space as {log_F, log_med_err_frac}.
    log_noise_perp log_med_err_frac minus its OLS fit on log_F, fitted per galaxy on the
                   per-cluster medians (so a cluster with many combinations does not dominate)
                   and applied to every row.  Uncorrelated with log_F by construction; also a
                   relabelling.  Pass perp_slopes={'LMC': (slope, intercept), ...} to fix the
                   coefficients instead of fitting them.

    The fitted coefficients are stored in rows.attrs['noise_perp_fit'].
    """
    raw = sector_raw_flux(**kw).set_index(['name', 'sector']).med_flux_raw
    lookup = raw.to_dict()
    F = [np.nanmedian([lookup.get((n, int(s)), np.nan) for s in secs]) if len(secs) else np.nan
         for n, secs in zip(rows['name'], rows['sectors'])]
    rows = rows.copy()
    rows['log_F'] = np.log10(pd.Series(F, index=rows.index).where(lambda x: x > 0))
    rows['log_noise_q'] = rows['log_med_err_frac'] + 0.5 * rows['log_F']

    fit = {}
    rows['log_noise_perp'] = np.nan
    for g, d in rows.groupby('origin'):
        if perp_slopes and g in perp_slopes:
            b, a = perp_slopes[g]
        else:
            pc = d.groupby('name')[['log_F', 'log_med_err_frac']].median().dropna()
            b, a = np.polyfit(pc.log_F, pc.log_med_err_frac, 1)
        fit[g] = (float(b), float(a))
        rows.loc[d.index, 'log_noise_perp'] = d.log_med_err_frac - (a + b * d.log_F)
    rows.attrs['noise_perp_fit'] = fit
    if verbose:
        print('log_noise_perp fit, d log err_frac / d log F (photon noise = -0.5): '
              + ', '.join(f'{g} {b:+.3f}' for g, (b, _) in fit.items()))
    return rows


def feature_columns(df):
    """The FEATURE_SPEC entries actually present, with their transformed column names."""
    out = []
    for col, tf, fam in FEATURE_SPEC:
        if col in df.columns:
            out.append((col, tf, fam))
    return out


def apply_transform(s, how):
    s = pd.to_numeric(s, errors='coerce')
    if how == 'log10':
        return np.log10(s.where(s > 0))
    if how == 'signed_log10':
        return np.sign(s) * np.log10(1 + s.abs())
    return s


def cluster_table(rows, meta=None, min_rows=1):
    """Collapse rows to one entry per cluster: the unit of analysis.

    Features are aggregated with the MEDIAN over that cluster's sector combinations, which is
    robust to the one-or-two combinations that fit badly, and keeps a cluster with 409 rows
    from outvoting one with a single row.  Flags are averaged (a fraction), and kept only to
    describe the sample.
    """
    feats = feature_columns(rows)
    work = rows.copy()
    for col, tf, _ in feats:
        work['T_' + col] = apply_transform(work[col], tf)

    num_nuis = [n for n, kind, _ in NUISANCE_SPEC + BRIGHTNESS_SPEC if kind == 'numeric' and n in work.columns]
    agg = {('T_' + c): 'median' for c, _, _ in feats}
    agg.update({n: 'median' for n in num_nuis})
    agg['age'] = 'first'
    agg['origin'] = 'first'

    g = work.groupby('name')
    out = g.agg(agg)
    out['n_rows'] = g.size()
    for flag in ('gp_rn_near_white_noise', 'gp_components_capped'):
        if flag in work:
            out['frac_' + flag] = g[flag].apply(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).mean())
    out = out.reset_index()
    if meta is not None:
        out = out.merge(meta, on='name', how='left')
    out = out[out.n_rows >= min_rows]
    out = out.rename(columns={('T_' + c): c for c, _, _ in feats})   # keep the plain names
    return out


# ---------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------

def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (independent or positively dependent tests)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        adj[i] = prev
    return adj


def design_matrix(df, nuisances, add_const=True):
    """Numeric design matrix for the nuisances: dummies for categoricals, columns as-is otherwise."""
    parts = []
    for n in nuisances:
        kind = dict((a, b) for a, b, _ in NUISANCE_SPEC).get(n, 'numeric')
        if kind == 'categorical':
            d = pd.get_dummies(df[n].astype(str), prefix=n, drop_first=True).astype(float)
            parts.append(d)
        else:
            parts.append(pd.to_numeric(df[n], errors='coerce').rename(n))
    X = pd.concat(parts, axis=1)
    if add_const:
        X.insert(0, 'const', 1.0)
    return X


def _residualise(y, X):
    """Least-squares residuals of y on X, ignoring rows with any missing value."""
    y = np.asarray(y, float)
    Xv = np.asarray(X, float)
    ok = np.isfinite(y) & np.isfinite(Xv).all(axis=1)
    r = np.full_like(y, np.nan)
    if ok.sum() > Xv.shape[1] + 2:
        beta, *_ = np.linalg.lstsq(Xv[ok], y[ok], rcond=None)
        r[ok] = y[ok] - Xv[ok] @ beta
    return r


def partial_spearman(df, feature, target, nuisances):
    """Spearman correlation of feature and target after removing the nuisances.

    Ranks are taken first, then both rank vectors are residualised on the nuisance design and
    correlated.  This is the rank analogue of a partial correlation: it answers "within
    clusters that share observing conditions and galaxy, does this feature still track age?"
    """
    from scipy import stats
    sub = df[[feature, target] + list(nuisances)].copy()
    ok = sub[[feature, target]].notna().all(axis=1)
    sub = sub[ok]
    if len(sub) < 20:
        return np.nan, np.nan, len(sub)
    X = design_matrix(sub, nuisances)
    ry = _residualise(sub[target].rank(), X)
    rx = _residualise(sub[feature].rank(), X)
    m = np.isfinite(ry) & np.isfinite(rx)
    if m.sum() < 20:
        return np.nan, np.nan, int(m.sum())
    r = np.corrcoef(rx[m], ry[m])[0, 1]
    # Degrees of freedom must pay for the nuisances: k columns were fitted away before
    # correlating, so the usual n-2 would overstate the evidence.  With ~10 nuisance columns
    # and a few hundred clusters the difference is small but real, and it is free to get right.
    k = X.shape[1] - 1                       # exclude the intercept
    dof = int(m.sum()) - 2 - k
    if dof <= 0 or not np.isfinite(r) or abs(r) >= 1:
        return r, np.nan, int(m.sum())
    t = r * np.sqrt(dof / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, int(m.sum())


def spearman_ci(x, y, n_boot=2000, seed=0, groups=None):
    """Spearman rho with a bootstrap confidence interval.

    ``groups`` resamples whole groups (clusters) rather than rows, which is the honest thing
    to do whenever several rows describe the same object.
    """
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if groups is not None:
        groups = np.asarray(groups)[ok]
    if len(x) < 10:
        return np.nan, np.nan, (np.nan, np.nan)
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    if groups is None:
        idx_pool = np.arange(len(x))
        for _ in range(n_boot):
            i = rng.choice(idx_pool, len(idx_pool), replace=True)
            if len(np.unique(y[i])) > 2:
                boots.append(spearmanr(x[i], y[i])[0])
    else:
        uniq = np.unique(groups)
        index_of = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, len(uniq), replace=True)
            i = np.concatenate([index_of[g] for g in pick])
            if len(np.unique(y[i])) > 2:
                boots.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return rho, p, (lo, hi)


def stratified_permutation_p(df, feature, target, strata, n_perm=2000, seed=0, statistic=None):
    """P-value from shuffling the target WITHIN strata (default: origin).

    Shuffling within galaxy destroys any feature-age link while preserving the galaxy-age
    association, so it tests the thing we actually care about rather than rediscovering that
    the LMC is young.
    """
    from scipy.stats import spearmanr
    sub = df[[feature, target, strata]].dropna()
    if len(sub) < 20:
        return np.nan, np.nan
    stat = statistic or (lambda a, b: spearmanr(a, b)[0])
    obs = stat(sub[feature].values, sub[target].values)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    vals = sub[target].values.copy()
    idx_by_stratum = [np.where(sub[strata].values == s)[0] for s in sub[strata].unique()]
    for k in range(n_perm):
        shuffled = vals.copy()
        for idx in idx_by_stratum:
            shuffled[idx] = rng.permutation(vals[idx])
        null[k] = stat(sub[feature].values, shuffled)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


def partial_spearman_ci(df, feature, target, nuisances, n_boot=500, seed=0):
    """Bootstrap CI for the partial rank correlation, resampling clusters."""
    rho, p, n = partial_spearman(df, feature, target, nuisances)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    boots = []
    for _ in range(n_boot):
        take = rng.choice(idx, len(idx), replace=True)
        r, _, _ = partial_spearman(df.iloc[take], feature, target, nuisances)
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return rho, p, (lo, hi), n


def confounder_attribution(df, feature, target, nuisances):
    """Which single nuisance explains a marginal correlation?

    Controls for one nuisance at a time and records what is left.  The nuisance whose removal
    shrinks |rho| the most is the one carrying the association -- the answer to "why did this
    feature stop correlating with age once I controlled for things".

    Returns a Series indexed by nuisance (plus 'marginal' and 'all'), holding the partial rho.
    """
    from scipy.stats import spearmanr
    sub = df[[feature, target] + list(nuisances)].dropna(subset=[feature, target])
    out = {'marginal': spearmanr(sub[feature], sub[target])[0]}
    for n in nuisances:
        r, _, _ = partial_spearman(sub, feature, target, [n])
        out[n] = r
    r_all, _, _ = partial_spearman(sub, feature, target, list(nuisances))
    out['all'] = r_all
    return pd.Series(out)


def attribution_table(df, features, target, nuisances):
    """confounder_attribution for many features, plus a verdict for each.

    verdict:
      'survives'          -- still correlated after removing every nuisance
      'explained by X'    -- controlling X alone removes most of the marginal correlation
      'weak'              -- no appreciable marginal correlation to begin with
    """
    rows = []
    for f in features:
        a = confounder_attribution(df, f, target, nuisances)
        marg, allp = a['marginal'], a['all']
        singles = a.drop(['marginal', 'all'])
        killer = singles.abs().idxmin() if len(singles) else None
        shrink = 1 - abs(allp) / abs(marg) if abs(marg) > 1e-9 else np.nan
        single_shrink = 1 - abs(singles[killer]) / abs(marg) if killer and abs(marg) > 1e-9 else 0.0
        if abs(allp) > abs(marg) + 0.05 and abs(allp) >= 0.12:
            # suppression: a nuisance was MASKING a real correlation, so controlling it reveals
            # one.  Worth flagging separately -- it is easy to mistake for noise in the raw data.
            booster = singles.abs().idxmax() if len(singles) else None
            verdict = 'revealed by control'
        elif abs(marg) < 0.12:
            verdict = 'weak'
        elif abs(allp) >= 0.7 * abs(marg):
            verdict = 'survives'
        elif single_shrink < 0.25:
            verdict = 'weakened by nuisances jointly'      # no single one accounts for it
        elif abs(allp) < 0.12:
            verdict = f'explained by {killer}'
        else:
            verdict = f'weakened by {killer}'
        rows.append(dict(feature=f, rho_marginal=marg, rho_partial=allp,
                         dominant_nuisance=killer,
                         rho_controlling_dominant=(singles[killer] if killer else np.nan),
                         shrinkage=shrink, verdict=verdict))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# Feature families: handling redundancy
# ---------------------------------------------------------------------------------------

def feature_reliability(rows, features=None, min_rows=3):
    """How repeatable is each feature across a cluster's own sector combinations?

    Returns an intraclass-correlation-like ratio, between-cluster variance over total.  1.0
    means the sector combinations of a cluster agree perfectly; 0 means the feature is noise.
    Used to choose a family's representative WITHOUT looking at age.
    """
    feats = features or [c for c, _, _ in feature_columns(rows)]
    work = rows.copy()
    for col, tf, _ in feature_columns(rows):
        work['T_' + col] = apply_transform(work[col], tf)
    out = {}
    for f in feats:
        g = work[['name', 'T_' + f]].dropna()
        g = g[g.groupby('name')['T_' + f].transform('size') >= min_rows]
        if g['name'].nunique() < 10:
            out[f] = np.nan
            continue
        mu = g.groupby('name')['T_' + f].mean()
        within = g.groupby('name')['T_' + f].var().mean()
        between = mu.var()
        out[f] = between / (between + within) if (between + within) > 0 else np.nan
    return pd.Series(out, name='reliability')


def feature_families(df, features, threshold=0.8, reliability=None):
    """Group features that measure the same thing, by rank-correlation clustering.

    Several columns describe one quantity -- excess_var, intrinsic_std, intrinsic_rms and
    sigma_mad are four views of the same scatter -- so testing them separately triple-counts
    one result and drags every other feature through a harsher FDR correction than it
    deserves.  Average-linkage clustering on 1 - |Spearman| cuts at ``threshold``.

    **Age is never used here.**  Families come from feature-feature structure only, so the
    grouping cannot be tuned, even accidentally, to the answer.

    ``reliability`` (see ``feature_reliability``) picks each family's representative; without
    it the member with the fewest missing values wins.

    Returns a DataFrame: feature, family, representative, n_members, mean_abs_rho.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    R = df[features].corr(method='spearman')
    D = (1 - R.abs()).to_numpy(copy=True)
    np.fill_diagonal(D, 0.0)
    D = np.nan_to_num(D, nan=1.0)
    D = 0.5 * (D + D.T)                                  # enforce symmetry for squareform
    labels = fcluster(linkage(squareform(D, checks=False), 'average'),
                      t=1 - threshold, criterion='distance')

    rows_out = []
    for lab in np.unique(labels):
        members = [f for f, l in zip(features, labels) if l == lab]
        if reliability is not None and reliability.reindex(members).notna().any():
            rep = reliability.reindex(members).idxmax()
        else:
            rep = df[members].notna().sum().idxmax()
        sub = R.loc[members, members].to_numpy()
        iu = np.triu_indices(len(members), 1)
        mean_rho = float(np.abs(sub[iu]).mean()) if len(members) > 1 else 1.0
        for f in members:
            rows_out.append(dict(feature=f, family=rep, representative=rep,
                                 n_members=len(members), mean_abs_rho=mean_rho))
    return pd.DataFrame(rows_out)


def family_index(df, members, anchor=None):
    """One composite per family: the mean of z-scored ranks, signs aligned.

    Ranks first (so the composite inherits Spearman's robustness), then align each member to
    an anchor using their mutual correlation -- again without reference to age -- so that
    members which are negatively related do not cancel.
    """
    if len(members) == 1:
        return pd.to_numeric(df[members[0]], errors='coerce').rank()
    R = df[members].corr(method='spearman')
    if anchor is None:
        anchor = R.abs().mean().idxmax()                 # the most central member
    z = df[members].rank()
    z = (z - z.mean()) / z.std(ddof=0)
    signs = np.sign(R[anchor].reindex(members).fillna(1.0))
    signs[signs == 0] = 1.0
    return (z * signs.values).mean(axis=1)


def build_family_table(df, features, threshold=0.8, reliability=None):
    """``feature_families`` plus an index column per family added to a copy of ``df``.

    Returns (df_with_indices, families, index_columns).
    """
    fams = feature_families(df, features, threshold=threshold, reliability=reliability)
    out = df.copy()
    cols = []
    for rep, grp in fams.groupby('family'):
        col = f'FAM_{rep}'
        out[col] = family_index(df, list(grp.feature))
        cols.append(col)
    return out, fams, cols


# ---------------------------------------------------------------------------------------
# Splitting: always by cluster
# ---------------------------------------------------------------------------------------

def group_folds(groups, y=None, n_splits=5, seed=0, n_bins=5):
    """Assign whole clusters to folds, balanced in ROW COUNT and stratified in the target.

    Two things a plain round-robin over clusters gets wrong on this table:

    * **row balance** -- one cluster carries 409 rows and another carries 1, so assigning
      clusters in turn leaves wildly unequal folds and a noisy score;
    * **target balance** -- with 348 clusters spread over 3.7 dex of age, a random split can
      hand one fold most of the old clusters.

    Clusters are therefore binned by target, and within each bin the largest cluster goes to
    whichever fold currently holds the fewest rows (greedy bin packing).  A cluster never
    spans folds, which is the point: every row of a cluster shares one age, so a split that
    separates them lets the model recognise the cluster and recite the answer.

    Returns an integer fold index per ROW.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)

    if y is not None:
        y = np.asarray(y, dtype=float)
        gy = pd.Series(y).groupby(pd.Series(groups)).first().reindex(uniq).to_numpy()
        ranks = pd.Series(gy).rank(method='first', na_option='bottom').to_numpy()
        strata = np.floor((ranks - 1) / len(uniq) * n_bins).astype(int)
    else:
        strata = np.zeros(len(uniq), dtype=int)

    load = np.zeros(n_splits)
    fold_of = {}
    for b in range(strata.max() + 1):
        idx = np.where(strata == b)[0]
        idx = idx[rng.permutation(len(idx))]
        idx = idx[np.argsort(-counts[idx], kind='stable')]      # biggest clusters placed first
        for i in idx:
            f = int(np.argmin(load))
            fold_of[uniq[i]] = f
            load[f] += counts[i]
    return np.array([fold_of[g] for g in groups])


def holdout_by_cluster(groups, y=None, frac=0.2, seed=0, n_bins=5, unit='clusters'):
    """A final test set of whole clusters, stratified in the target.

    Cross-validation measures a fixed model honestly, but it cannot protect against the
    choices an analyst makes after looking at results.  Setting clusters aside once, and
    touching them only at the end, is what makes the final number defensible.

    ``unit`` decides what ``frac`` is a fraction OF, and this matters more than it looks:

    * ``'clusters'`` (default) -- hold out that fraction of CLUSTERS.  The split is decided on
      the cluster list alone, so a row-level table and its cluster-level collapse hold out the
      SAME clusters and their scores are comparable.
    * ``'rows'`` -- hold out that fraction of ROWS.  Because cluster sizes span 1 to 409 rows,
      this lands on a different set of clusters entirely: on the <=3-sector table the two
      options share only 15 of ~75 held-out clusters.  Use it only when the test set needs a
      given number of rows rather than a given number of clusters.

    Returns a boolean mask (True = held out) per ROW.
    """
    groups = np.asarray(groups)
    n_splits = max(2, int(round(1 / frac)))
    if unit == 'rows':
        return group_folds(groups, y=y, n_splits=n_splits, seed=seed, n_bins=n_bins) == 0
    if unit != 'clusters':
        raise ValueError("unit must be 'clusters' or 'rows'")

    uniq = np.unique(groups)
    y_clu = None
    if y is not None:
        y = np.asarray(y, float)
        y_clu = np.array([np.nanmedian(y[groups == g]) for g in uniq])
    fold = group_folds(uniq, y=y_clu, n_splits=n_splits, seed=seed, n_bins=n_bins)
    held = set(uniq[fold == 0])
    return np.array([g in held for g in groups])


# ---------------------------------------------------------------------------------------
# GP spectral bins
# ---------------------------------------------------------------------------------------

def band_power_columns(rows, col='gp_kspace_log_band_powers', prefix='gpb'):
    """Expand the stored GP band-power vector into one column per frequency bin.

    ``add_gp_summary_stats`` evaluates the fitted GP PSD on a fixed log-spaced frequency grid
    and integrates it into 10 bins over 0.1-12 /day.  The edges are identical on every row, so
    the vector is directly comparable across clusters -- unlike the component lists, which vary
    in length and have no canonical ordering.

    Two representations are returned:

    * ``<prefix>_b0 .. b9``  -- log10 power in each bin (amplitude AND shape);
    * ``<prefix>_s0 .. s9``  -- the same minus each row's own mean (SHAPE only).

    The shape columns sum to zero by construction, so one is redundant; the forest is
    indifferent to that, and keeping all ten keeps the bin labels interpretable.
    Also returns ``<prefix>_level``, the row mean, so amplitude can be tested on its own.

    Returns (df_with_columns, power_cols, shape_cols, level_col, bin_centers).
    """
    if col not in rows.columns:
        raise KeyError(f'{col} not in table -- was the GP summary-stats step run?')
    P = np.vstack([np.asarray(v, float) if np.size(v) else np.full(10, np.nan)
                   for v in rows[col]])
    n_bins = P.shape[1]

    centers = None
    for key in ('gp_kspace_bin_centers', 'gp_kspace_bin_edges'):
        if key in rows.columns:
            for v in rows[key]:
                if np.size(v):
                    e = np.asarray(v, float)
                    centers = e if len(e) == n_bins else np.sqrt(e[:-1] * e[1:])
                    break
        if centers is not None:
            break

    out = rows.copy()
    pcols = [f'{prefix}_b{i}' for i in range(n_bins)]
    scols = [f'{prefix}_s{i}' for i in range(n_bins)]
    level = f'{prefix}_level'
    out[level] = np.nanmean(P, axis=1)
    for i in range(n_bins):
        out[pcols[i]] = P[:, i]
        out[scols[i]] = P[:, i] - out[level].values
    return out, pcols, scols, level, centers


def aggregate_to_clusters(rows, columns, meta=None, extra_first=('age', 'origin')):
    """Median over each cluster's sector combinations, for an arbitrary column list.

    ``cluster_table`` handles the FEATURE_SPEC columns; this is the same collapse for columns
    added afterwards (the spectral bins), so the two stay comparable.
    """
    agg = {c: 'median' for c in columns}
    num_nuis = [n for n, kind, _ in NUISANCE_SPEC
                if kind == 'numeric' and n in rows.columns and n not in columns]
    agg.update({n: 'median' for n in num_nuis})
    agg.update({c: 'first' for c in extra_first if c in rows.columns})
    g = rows.groupby('name')
    out = g.agg(agg)
    out['n_rows'] = g.size()
    out = out.reset_index()
    if meta is not None:
        out = out.merge(meta, on='name', how='left')
    return out


# ---------------------------------------------------------------------------------------
# Nuisance regression (residualized features) and the Freedman-Lane null
# ---------------------------------------------------------------------------------------

class NuisanceResidualizer:
    """Features minus their OLS prediction from the nuisances, fit on training data only.

    scikit-learn compatible (fit / transform / get_params), so it sits inside a Pipeline and is
    refit in every CV fold: the test clusters never help define "expected given the
    nuisances".  Input columns are ``[nuisances | features]``; output is the residualized
    features.  Missing values are median-imputed with training medians first.

    Removes LINEAR nuisance dependence only; curvature in a feature-nuisance relation survives.
    """

    def __init__(self, n_nuis):
        self.n_nuis = n_nuis

    def get_params(self, deep=True):
        return {'n_nuis': self.n_nuis}

    def set_params(self, **p):
        self.n_nuis = p.get('n_nuis', self.n_nuis)
        return self

    def _design(self, X):
        X = np.asarray(X, float)
        Nu, F = X[:, :self.n_nuis], X[:, self.n_nuis:]
        Nu = np.where(np.isnan(Nu), self.nmed_, Nu)
        F = np.where(np.isnan(F), self.fmed_, F)
        return np.column_stack([np.ones(len(Nu)), Nu]), F

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.nmed_ = np.nan_to_num(np.nanmedian(X[:, :self.n_nuis], axis=0))
        self.fmed_ = np.nan_to_num(np.nanmedian(X[:, self.n_nuis:], axis=0))
        D, F = self._design(X)
        self.beta_, *_ = np.linalg.lstsq(D, F, rcond=None)
        return self

    def transform(self, X):
        D, F = self._design(X)
        return F - D @ self.beta_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def freedman_lane_sampler(X_nuis, y, seed=0):
    """Return a function that draws Freedman-Lane permuted targets.

    y = fitted(nuisances) + residual; each draw permutes only the residual, so every linear
    age-nuisance relation is kept and only the part of age the nuisances cannot explain is
    scrambled.  The right null for partial / residualized / nested tests.
    """
    X = np.asarray(X_nuis, float)
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    D = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(D, np.asarray(y, float), rcond=None)
    fit = D @ b
    res = np.asarray(y, float) - fit
    rng = np.random.default_rng(seed)
    return lambda: fit + rng.permutation(res)

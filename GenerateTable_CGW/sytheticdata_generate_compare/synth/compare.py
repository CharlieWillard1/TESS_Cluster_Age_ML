"""Compare pipeline output on the synthetic set against the known truth.

Join model
----------
Truth is **one row per cluster**; the pipeline emits **many rows per cluster** (one per
sector combination).  The join is therefore one-to-many on ``name``; facet or aggregate
by ``n_sectors`` when interpreting.
"""

import numpy as np
import pandas as pd

from table_pipeline.lsp_stats import red_noise_threshold


# Pipeline column -> truth column, for statistics that have a truth counterpart.
STAT_PAIRS = [
    ('excess_var',         'true_excess_var'),
    ('sigma_mad',          'true_sigma_mad'),
    ('mean_median_offset', 'true_mean_median_offset'),
    ('gamma_p',            'true_gamma_p'),
    ('vn_ratio_gap_aware', 'true_vn_ratio'),
    ('stetson_j_gap_aware','true_stetson_j'),
    ('intrinsic_std',      'true_intrinsic_std'),
]


# Truth columns whose names collide with pipeline columns holding the RECOVERED value.
# Without renaming, the merge silently suffixes both and downstream lookups fail.
_INJ_RENAME = {
    'rn_alpha': 'inj_rn_alpha',      # injected slope vs the pipeline's fitted slope
}


def join(pipeline_table, truth_table):
    """One-to-many join of truth onto the pipeline rows.

    Truth is one row per cluster, the pipeline emits one row per sector combination, so
    this is deliberately many-to-one. Injected quantities that share a name with a
    recovered pipeline column are prefixed ``inj_`` first -- otherwise pandas suffixes
    both to ``_x``/``_y`` and every later lookup breaks.
    """
    drop = ('age', 'origin', 'path')
    keep = [c for c in truth_table.columns if c not in drop]
    t = truth_table[keep].rename(columns=_INJ_RENAME)
    clash = (set(t.columns) & set(pipeline_table.columns)) - {'name'}
    if clash:
        raise ValueError(f"unhandled truth/pipeline column collision: {sorted(clash)} "
                         f"-- add them to _INJ_RENAME")
    return pipeline_table.merge(t, on='name', how='left', validate='many_to_one')


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def stat_comparison(df, pairs=STAT_PAIRS):
    """Median ratio and scatter of pipeline vs truth for each statistic.

    ``ratio_median`` far from 1 is a systematic offset.  For ``excess_var`` a offset is
    *expected*, not a failure: truth spans the full multi-year baseline while the
    pipeline sees per-sector-normalised 27 d chunks, and red-noise variance grows as
    ln(f_max*T).  See README.
    """
    rows = []
    for pcol, tcol in pairs:
        if pcol not in df.columns or tcol not in df.columns:
            continue
        a = pd.to_numeric(df[pcol], errors='coerce')
        b = pd.to_numeric(df[tcol], errors='coerce')
        m = np.isfinite(a) & np.isfinite(b) & (b != 0)
        if m.sum() < 5:
            continue
        r = (a[m] / b[m]).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(dict(statistic=pcol, n=int(m.sum()),
                         ratio_median=float(r.median()),
                         ratio_p16=float(r.quantile(.16)),
                         ratio_p84=float(r.quantile(.84)),
                         spearman=float(a[m].corr(b[m], method='spearman'))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Period recovery
# ---------------------------------------------------------------------------

def ratio_peak_period(row):
    """argmax(P/C) recomputed locally.

    The ratio-based locator was removed from the pipeline (biased toward short periods),
    but it is wanted here for the three-way comparison.  Recomputing it from the stored
    ``LSP_freq``/``LSP_power`` keeps the comparison complete without touching the
    pipeline.
    """
    f = np.asarray(row['LSP_freq'], float)
    p = np.asarray(row['LSP_power'], float)
    C = 10.0 ** float(row['rn_log10_N']) * f ** (-float(row['rn_alpha']))
    return 1.0 / f[int(np.nanargmax(p / C))]


def add_ratio_period(df):
    df = df.copy()
    df['LSP_peak_period_ratio'] = df.apply(ratio_peak_period, axis=1)
    return df


def period_recovery(df, finders=('LSP_peak_period', 'LSP_peak_period_diff',
                                 'LSP_peak_period_ratio'),
                    tol=0.05, bins=(0.1, 0.4, 1.0, 2.5, 9.0), match='any'):
    """Recovery fraction by injected period, per finder.

    ``match='any'``  -- recovered if the reported period matches ANY injected period.
    ``match='loudest'`` -- must match the highest-amplitude injected period.

    A flat column across the period bins means the finder is unbiased: the recovered
    distribution then reproduces the injected one whatever its shape.
    """
    out = []
    sub = df[df['n_osc'] > 0]
    for name in finders:
        if name not in sub.columns:
            continue
        for lo, hi in zip(bins[:-1], bins[1:]):
            hits = tot = 0
            for _, r in sub.iterrows():
                P = np.atleast_1d(np.asarray(r['periods'], float))
                A = np.atleast_1d(np.asarray(r['amplitudes'], float))
                if len(P) == 0:
                    continue
                targets = (P if match == 'any' else P[[int(np.argmax(A))]])
                targets = targets[(targets >= lo) & (targets < hi)]
                if len(targets) == 0:
                    continue
                tot += 1
                rec = float(r[name])
                if np.isfinite(rec) and np.any(np.abs(rec / targets - 1.0) < tol):
                    hits += 1
            if tot:
                out.append(dict(finder=name, p_lo=lo, p_hi=hi, n=tot,
                                recovery=hits / tot))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# GP recovery
# ---------------------------------------------------------------------------

def match_gp_periods(fitted, injected, tol=0.10):
    """Greedy nearest-match in fractional period distance.

    Returns (matched pairs, n_spurious, n_missed).  Each injected period is used at most
    once, so duplicate GP components matching the same injection count as spurious.
    """
    fitted = list(np.atleast_1d(np.asarray(fitted, float)))
    injected = list(np.atleast_1d(np.asarray(injected, float)))
    pairs, used = [], set()
    for pf in fitted:
        if not np.isfinite(pf) or not injected:
            continue
        d = [abs(pf / pi - 1.0) if j not in used else np.inf
             for j, pi in enumerate(injected)]
        j = int(np.argmin(d))
        if d[j] < tol:
            used.add(j)
            pairs.append((pf, injected[j], d[j]))
    return pairs, len(fitted) - len(pairs), len(injected) - len(used)


def gp_recovery(df, tol=0.10):
    """Per-row GP recovery summary against the injected oscillators."""
    rows = []
    for _, r in df.iterrows():
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        fit = r.get('gp_periods')
        fit = np.atleast_1d(np.asarray(fit, float)) if fit is not None else np.array([])
        pairs, spur, miss = match_gp_periods(fit, inj, tol=tol)
        rows.append(dict(
            name=r['name'], n_sectors=r.get('n_sectors', np.nan),
            n_osc=int(len(inj)), n_gp=int(len(fit)),
            n_matched=len(pairs), n_spurious=spur, n_missed=miss,
            frac_recovered=(len(pairs) / len(inj)) if len(inj) else np.nan,
            median_frac_err=float(np.median([p[2] for p in pairs])) if pairs else np.nan,
            inj_amp_max=float(r.get('inj_amp_max', np.nan)),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-oscillator diagnostics
# ---------------------------------------------------------------------------
#
# Row-level GP recovery ("did this row find its injections") conflates two very
# different things: injections that were never detectable, and injections the GP
# genuinely lost.  The amplitude distribution deliberately spans well below the noise,
# so a low raw recovery fraction is expected and says nothing about the fitter.
#
# These functions work at the level of the individual injected oscillator, and classify
# each one by how far its LSP peak rises above the fitted red-noise continuum at its own
# frequency.  That is the quantity that decides whether recovery was ever possible.

def per_oscillator_table(df, tol=0.10, window=0.02):
    """Explode the joined table to one row per (pipeline row x injected oscillator).

    For each injected oscillator, measure the observed LSP power near its frequency and
    compare it to the fitted continuum ``C(f) = 10**rn_log10_N * f**(-rn_alpha)``.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``join`` (and ideally ``add_ratio_period``).
    tol : float
        Fractional tolerance for calling a GP period a match to an injected one.
    window : float
        Fractional half-width around the injected frequency in which to take the local
        maximum of the periodogram.  The peak can sit a bin or two off the exact
        frequency, so reading a single bin would understate it.

    Returns
    -------
    pd.DataFrame with columns
        name, n_sectors, n_osc, P, A       -- identity and injected parameters
        obs_power, C, gamma, ratio         -- measured height above the continuum
        signif                             -- obs_power > gamma * C, i.e. detectable
        recovered                          -- a GP period matched within ``tol``
        n_gp                               -- components fitted for that row
    """
    rows = []
    for _, r in df.iterrows():
        P = np.atleast_1d(np.asarray(r.get('periods', []), float))
        A = np.atleast_1d(np.asarray(r.get('amplitudes', []), float))
        if len(P) == 0:
            continue
        f = np.asarray(r['LSP_freq'], float)
        p = np.asarray(r['LSP_power'], float)
        C = 10.0 ** float(r['rn_log10_N']) * f ** (-float(r['rn_alpha']))
        gam = -np.log(1 - 0.99 ** (1 / len(f)))
        gp = r.get('gp_periods')
        gp = np.atleast_1d(np.asarray(gp, float)) if gp is not None else np.array([])

        for Pi, Ai in zip(P, A):
            fi = 1.0 / Pi
            if fi < f[0] or fi > f[-1]:          # injected outside the search band
                obs = Ci = np.nan
            else:
                w = np.abs(f / fi - 1.0) < window
                obs = float(np.nanmax(p[w])) if w.any() else np.nan
                Ci = float(np.interp(fi, f, C))
            ok = np.isfinite(obs) and np.isfinite(Ci) and Ci > 0
            rows.append(dict(
                name=r['name'], n_sectors=r.get('n_sectors', np.nan), n_osc=int(len(P)),
                P=Pi, A=Ai, obs_power=obs, C=Ci, gamma=gam,
                ratio=(obs / Ci) if ok else np.nan,
                signif=bool(obs > gam * Ci) if ok else False,
                recovered=bool(len(gp)) and bool(np.any(np.abs(gp / Pi - 1.0) < tol)),
                n_gp=int(len(gp))))
    return pd.DataFrame(rows)


def recovery_summary(osc):
    """Headline numbers: recovery split on whether the signal was ever detectable."""
    n_bad = int((~osc['recovered']).sum())
    missed = osc[~osc['recovered']]
    return dict(
        n_oscillators=len(osc),
        recovery_overall=float(osc['recovered'].mean()),
        n_detectable=int(osc['signif'].sum()),
        recovery_detectable=float(osc.loc[osc['signif'], 'recovered'].mean()),
        recovery_undetectable=float(osc.loc[~osc['signif'], 'recovered'].mean()),
        frac_of_misses_undetectable=float((~missed['signif']).mean()) if n_bad else np.nan,
        n_detectable_but_missed=int((osc['signif'] & ~osc['recovered']).sum()),
    )


def plot_recovery_vs_floor(osc, edges=(0.3, 1, 3, 10, 30, 100, 300, 3000)):
    """Recovery as a function of height above the red-noise continuum.

    The headline diagnostic.  If the curve rises through ~1 at the detection threshold
    and the misses sit at low ratio, the fitter is behaving: what it lost was never
    recoverable.
    """
    import matplotlib.pyplot as plt
    edges = np.asarray(edges, float)
    gam = float(np.nanmedian(osc['gamma']))
    ctr = np.sqrt(edges[:-1] * edges[1:])
    rec, n = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (osc['ratio'] >= lo) & (osc['ratio'] < hi)
        rec.append(osc.loc[m, 'recovered'].mean())
        n.append(int(m.sum()))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].plot(ctr, rec, 'o-', color='steelblue')
    for x, y, c in zip(ctr, rec, n):
        axs[0].annotate(f'{c}', (x, y), textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=7, color='gray')
    axs[0].axvline(gam, color='darkred', ls='--',
                   label=f'1% detection threshold (γ={gam:.1f})')
    axs[0].set(xscale='log', ylim=(-.03, 1.03),
               xlabel='LSP peak power / red-noise continuum at that frequency',
               ylabel='fraction recovered by GP',
               title='Recovery is set by height above the red-noise floor')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

    s = recovery_summary(osc)
    vals = [s['recovery_detectable'], s['recovery_undetectable']]
    cnt = [s['n_detectable'], len(osc) - s['n_detectable']]
    bars = axs[1].bar(['above threshold\n(detectable)', 'below threshold\n(not detectable)'],
                      vals, color=['seagreen', 'lightcoral'], width=.6)
    for b, v, c in zip(bars, vals, cnt):
        axs[1].text(b.get_x() + b.get_width() / 2, v + .02, f'{v:.1%}\nn={c}',
                    ha='center', fontsize=10)
    axs[1].set(ylim=(0, 1.15), ylabel='fraction recovered',
               title=f"{s['frac_of_misses_undetectable']:.1%} of misses were never detectable")
    plt.tight_layout()
    return fig


def plot_period_amplitude(osc):
    """Where failures live in (period, amplitude), split by detectability."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (sel, ttl) in zip(axs, [(osc['signif'], 'detectable (above threshold)'),
                                    (~osc['signif'], 'not detectable (below threshold)')]):
        s = osc[sel]
        for m, c, l in [(s['recovered'], 'seagreen', 'recovered'),
                        (~s['recovered'], 'crimson', 'missed')]:
            ax.scatter(s.loc[m, 'P'], s.loc[m, 'A'], s=13, alpha=.55, c=c,
                       label=f'{l} ({int(m.sum())})')
        ax.set(yscale='log', xlabel='injected period (d)',
               ylabel='injected amplitude (fractional)', title=ttl)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    plt.tight_layout()
    return fig


def plot_period_bias(osc, bins=((0.1, 0.5), (0.5, 1), (1, 2.5), (2.5, 5), (5, 9))):
    """Recovery vs period *conditional on detectability*.

    Flat means the GP has no residual period preference once the signal is findable;
    any decline in the unconditional numbers is then purely a detectability effect.
    """
    import matplotlib.pyplot as plt
    s = osc[osc['signif']]
    x = [np.sqrt(a * b) for a, b in bins]
    y = [s[(s['P'] >= a) & (s['P'] < b)]['recovered'].mean() for a, b in bins]
    n = [int(((s['P'] >= a) & (s['P'] < b)).sum()) for a, b in bins]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x, y, 'o-', color='seagreen')
    for xi, yi, ci in zip(x, y, n):
        ax.annotate(f'{ci}', (xi, yi), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=7, color='gray')
    ax.set(xscale='log', ylim=(0, 1.05), xlabel='injected period (d)',
           ylabel='recovery | detectable',
           title='Flat = no residual period bias in the GP')
    ax.grid(alpha=.3); plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# GP component purity
# ---------------------------------------------------------------------------

def gp_component_purity(df, tol=0.10):
    """One row per fitted GP component, flagged by whether it matches an injection.

    Components that match nothing injected are spurious.  Pairing this with the fitted
    Q gives a usable purity filter, since a coherent sinusoid drives Q high while a
    component absorbing broadband power does not.
    """
    rows = []
    for _, r in df.iterrows():
        Q = r.get('gp_sho_Qs')
        P = r.get('gp_periods')
        Q = np.atleast_1d(np.asarray(Q, float)) if Q is not None else np.array([])
        P = np.atleast_1d(np.asarray(P, float)) if P is not None else np.array([])
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        for q, p in zip(Q, P):
            rows.append(dict(name=r['name'], Q=q, period=p,
                             matches_injected=bool(len(inj)) and
                             bool(np.any(np.abs(inj / p - 1.0) < tol))))
    return pd.DataFrame(rows)


def purity_curve(qs, cuts=(0, 2, 5, 10, 20, 50, 100, 200, 499)):
    """Purity and completeness of the GP component list as a function of a Q cut."""
    tot = qs['matches_injected'].sum()
    out = []
    for c in cuts:
        k = qs['Q'] >= c
        out.append(dict(Q_cut=c, frac_kept=float(k.mean()),
                        purity=float(qs.loc[k, 'matches_injected'].mean()) if k.any() else np.nan,
                        completeness=float(qs.loc[k, 'matches_injected'].sum() / tot) if tot else np.nan))
    return pd.DataFrame(out)


def plot_q_purity(qs, cuts=(0, 2, 5, 10, 20, 50, 100, 200, 499)):
    """Q distribution for real vs spurious components, and the purity/completeness curve."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    hi = float(np.nanmax(qs['Q'])) if len(qs) else 500.0
    axs[0].hist(np.clip(qs.loc[qs['matches_injected'], 'Q'], 0, hi), bins=40, alpha=.65,
                color='seagreen', label='matches an injection')
    axs[0].hist(np.clip(qs.loc[~qs['matches_injected'], 'Q'], 0, hi), bins=40, alpha=.65,
                color='crimson', label='spurious')
    axs[0].set(yscale='log', xlabel='fitted Q', ylabel='GP components',
               title='Q separates real components from spurious')
    axs[0].legend(fontsize=8)

    pc = purity_curve(qs, cuts)
    axs[1].plot(pc['Q_cut'], pc['purity'], 'o-', color='seagreen', label='purity')
    axs[1].plot(pc['Q_cut'], pc['completeness'], 's-', color='steelblue', label='completeness')
    axs[1].set(xscale='symlog', ylim=(0, 1.05), xlabel='Q cut',
               ylabel='fraction', title='Filtering GP components on Q')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Root cause: continuum contamination
# ---------------------------------------------------------------------------

def alpha_bias(df):
    """Fitted minus injected red-noise slope, per pipeline row.

    ``_fit_red_noise_vaughan`` fits every frequency bin with no peak masking, so injected
    oscillators bias the continuum steep.  A steeper continuum lifts the detection
    threshold at low frequency, which suppresses seeding of genuine peaks -- the dominant
    cause of the residual failures.
    """
    out = df[['name', 'n_osc', 'rn_alpha', 'inj_rn_alpha']].copy()
    out['d_alpha'] = out['rn_alpha'] - out['inj_rn_alpha']
    return out


def plot_alpha_bias(df, bins=((0, 1), (1, 4), (4, 7), (7, 11))):
    """Continuum slope bias as a function of how crowded the spectrum is."""
    import matplotlib.pyplot as plt
    a = alpha_bias(df)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].scatter(a['n_osc'], a['d_alpha'], s=12, alpha=.4, color='steelblue')
    med = [a[(a['n_osc'] >= lo) & (a['n_osc'] < hi)]['d_alpha'].median() for lo, hi in bins]
    ctr = [(lo + hi - 1) / 2 for lo, hi in bins]
    axs[0].plot(ctr, med, 'o-', color='darkred', label='median')
    axs[0].axhline(0, color='k', lw=1, ls=':')
    axs[0].set(xlabel='number of injected oscillators', ylabel='fitted α − injected α',
               title='Injected signal biases the continuum steep')
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

    axs[1].scatter(a['inj_rn_alpha'], a['rn_alpha'], s=12, alpha=.4,
                   c=a['n_osc'], cmap='viridis')
    lim = [np.nanmin(a[['inj_rn_alpha', 'rn_alpha']].values),
           np.nanmax(a[['inj_rn_alpha', 'rn_alpha']].values)]
    axs[1].plot(lim, lim, 'k--', lw=1, label='1:1')
    sc = axs[1].collections[0]
    plt.colorbar(sc, ax=axs[1], label='n_osc')
    axs[1].set(xlabel='injected α', ylabel='fitted α', title='Recovery of the red-noise slope')
    axs[1].legend(fontsize=8)
    plt.tight_layout()
    return fig


def gp_rn_effective_alpha(period, Q=1.0 / np.sqrt(2.0), f_lo=0.1, f_hi=12.0, n=400):
    """Effective power-law slope of the GP's red-noise SHO component.

    The GP does not fit a power law.  Its red-noise term is a Simple Harmonic Oscillator
    kernel at **fixed** Q = 1/sqrt(2) (the standard granulation kernel), whose PSD is flat
    below the break frequency f0 = 1/period and falls as f^-4 above it (verified against
    the kernel's own autocovariance).  So there is no fitted alpha to compare with the
    Vaughan fit directly.

    What *is* comparable is the slope you would measure if you fitted a power law to that
    SHO over the same band the Vaughan fit uses.  This does exactly that, mirroring
    ``_fit_red_noise_vaughan``: unweighted least squares of log10 S against log10 f.

    Because the SHO is a *broken* power law, the effective slope depends on where the
    break sits relative to the band -- a break below f_lo gives ~4, a break above f_hi
    gives ~0, and anything between is intermediate.  That is a real limitation of the
    comparison, not a defect of the fit.
    """
    period = float(period)
    if not np.isfinite(period) or period <= 0:
        return np.nan
    from gp_pipeline.gp_fit import sho_psd   # single source of truth for the SHO PSD
    f = np.logspace(np.log10(f_lo), np.log10(f_hi), n)
    omega0 = 2.0 * np.pi / period
    S = sho_psd(f, 1.0, omega0, Q)          # slope is scale-free, so sigma is arbitrary
    good = np.isfinite(S) & (S > 0)
    if good.sum() < 3:
        return np.nan
    b, _ = np.polyfit(np.log10(f[good]), np.log10(S[good]), 1)
    return float(-b)


def gp_rn_bias(df, f_lo=0.1, f_hi=12.0):
    """GP red-noise component vs the injected red noise, per pipeline row."""
    out = df[['name', 'n_osc', 'gp_rn_sigma', 'gp_rn_period', 'gp_rn_Q',
              'rn_rms_realized', 'inj_rn_alpha', 'rn_alpha']].copy()
    out['gp_rn_alpha_eff'] = [gp_rn_effective_alpha(p, q if np.isfinite(q) else 1/np.sqrt(2),
                                                    f_lo, f_hi)
                              for p, q in zip(out['gp_rn_period'], out['gp_rn_Q'])]
    out['d_alpha_gp'] = out['gp_rn_alpha_eff'] - out['inj_rn_alpha']
    out['d_alpha_vaughan'] = out['rn_alpha'] - out['inj_rn_alpha']
    out['amp_ratio'] = out['gp_rn_sigma'] / out['rn_rms_realized']
    return out


def plot_gp_rn_bias(df, bins=((0, 1), (1, 4), (4, 7), (7, 11)), f_lo=0.1, f_hi=12.0):
    """Companion to ``plot_alpha_bias``, using the GP's red-noise term instead.

    Left  : GP red-noise amplitude vs the injected red-noise RMS -- directly comparable,
            both are fractional-flux amplitudes.
    Right : effective slope of the GP's SHO red-noise term vs injected alpha, with the
            standalone Vaughan fit overplotted for reference.
    """
    import matplotlib.pyplot as plt
    a = gp_rn_bias(df, f_lo, f_hi)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))

    m = np.isfinite(a['gp_rn_sigma']) & np.isfinite(a['rn_rms_realized'])
    sc = axs[0].scatter(a.loc[m, 'rn_rms_realized'], a.loc[m, 'gp_rn_sigma'],
                        s=14, alpha=.5, c=a.loc[m, 'n_osc'], cmap='viridis')
    lim = [min(a.loc[m, 'rn_rms_realized'].min(), a.loc[m, 'gp_rn_sigma'].min()),
           max(a.loc[m, 'rn_rms_realized'].max(), a.loc[m, 'gp_rn_sigma'].max())]
    axs[0].plot(lim, lim, 'k--', lw=1, label='1:1')
    plt.colorbar(sc, ax=axs[0], label='n_osc')
    axs[0].set(xscale='log', yscale='log',
               xlabel='injected red-noise RMS (rn_rms_realized)',
               ylabel='GP red-noise amplitude (gp_rn_sigma)',
               title=f"GP red-noise amplitude  (median ratio "
                     f"{a.loc[m,'amp_ratio'].median():.2f})")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=.3)

    ctr = [(lo + hi - 1) / 2 for lo, hi in bins]
    for col, c, lbl in [('d_alpha_gp', 'darkorange', 'GP SHO effective slope'),
                        ('d_alpha_vaughan', 'darkred', 'standalone Vaughan fit')]:
        med = [a[(a['n_osc'] >= lo) & (a['n_osc'] < hi)][col].median() for lo, hi in bins]
        axs[1].plot(ctr, med, 'o-', color=c, label=lbl)
    axs[1].axhline(0, color='k', lw=1, ls=':')
    axs[1].set(xlabel='number of injected oscillators',
               ylabel='fitted α − injected α',
               title='Slope bias: GP red-noise term vs the Vaughan fit')
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3)
    plt.tight_layout()
    return fig

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

def per_oscillator_table(df, tol=0.10, window=0.02, continuum=None):
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
    continuum : pd.DataFrame or None
        Optional ``rn_log10_N`` / ``rn_alpha`` columns (same index as ``df``) to judge
        detectability with instead of the row's own.  Use one fixed continuum when
        comparing pipeline configurations whose continuum fits differ, or "detectable"
        changes meaning between them.

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
    for label, r in df.iterrows():
        cr = continuum.loc[label] if continuum is not None else r
        P = np.atleast_1d(np.asarray(r.get('periods', []), float))
        A = np.atleast_1d(np.asarray(r.get('amplitudes', []), float))
        if len(P) == 0:
            continue
        f = np.asarray(r['LSP_freq'], float)
        p = np.asarray(r['LSP_power'], float)
        C = 10.0 ** float(cr['rn_log10_N']) * f ** (-float(cr['rn_alpha']))
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

    Also carries ``cycles`` (= Q / pi, the decay time in periods), ``tau`` and ``coherence``
    (NaN for tables fitted before the decay-time cap, which have no such columns) and the
    row's ``tau_cap``, so the same table serves the cycles- and coherence-based views.
    """
    rows = []
    for _, r in df.iterrows():
        Q = r.get('gp_sho_Qs')
        P = r.get('gp_periods')
        Q = np.atleast_1d(np.asarray(Q, float)) if Q is not None else np.array([])
        P = np.atleast_1d(np.asarray(P, float)) if P is not None else np.array([])
        n = len(Q)
        def _opt(col):
            v = r.get(col)
            v = np.atleast_1d(np.asarray(v, float)) if v is not None else np.array([])
            return v if len(v) == n else np.full(n, np.nan)
        tau, coh = _opt('gp_sho_taus'), _opt('gp_sho_coherence')
        tau_cap = float(r.get('gp_sho_tau_cap', np.nan)) if r.get('gp_sho_tau_cap') is not None else np.nan
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        for q, p, t, s in zip(Q, P, tau, coh):
            rows.append(dict(name=r['name'], n_sectors=r.get('n_sectors', np.nan),
                             Q=q, period=p, cycles=q / np.pi, tau=t, coherence=s,
                             tau_cap=tau_cap,
                             matches_injected=bool(len(inj)) and
                             bool(np.any(np.abs(inj / p - 1.0) < tol))))
    return pd.DataFrame(rows)


def purity_curve(qs, cuts=(0, 2, 5, 10, 20, 50, 100, 200, 499), by='Q'):
    """Purity and completeness of the GP component list as a function of a cut.

    ``by`` is the column cut on: ``'Q'`` (default), ``'cycles'`` (= Q / pi, the decay time in
    periods -- the cut that means the same thing at every period under the decay-time cap)
    or ``'coherence'`` (s in [0, 1], the position between the Q floor and the
    period-dependent ceiling).
    """
    tot = qs['matches_injected'].sum()
    out = []
    for c in cuts:
        k = qs[by] >= c
        out.append(dict(cut=c, frac_kept=float(k.mean()),
                        purity=float(qs.loc[k, 'matches_injected'].mean()) if k.any() else np.nan,
                        completeness=float(qs.loc[k, 'matches_injected'].sum() / tot) if tot else np.nan))
    out = pd.DataFrame(out)
    return out.rename(columns={'cut': f'{by}_cut'})


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


def plot_coherence_purity(qs, cycle_cuts=(0, 0.5, 1, 1.6, 3, 6, 10, 30, 100, 300)):
    """The decay-time view of component purity (tables fitted with the tau_d cap).

    (a) coherence fraction s for real vs spurious components -- s = 1 is "pinned at the
        cap", s = 0 is the Q floor;
    (b) decay time in periods (cycles = Q / pi) for the same two populations;
    (c) purity / completeness against a cycles cut, which is what
        ``filter_gp_components(n_cycles_min=...)`` applies.
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    real, spur = qs[qs['matches_injected']], qs[~qs['matches_injected']]

    ax = axs[0]
    ax.hist(real['coherence'].dropna(), bins=np.linspace(0, 1, 41), alpha=.65, color='seagreen',
            label=f'matches an injection ({len(real)})')
    ax.hist(spur['coherence'].dropna(), bins=np.linspace(0, 1, 41), alpha=.65, color='crimson',
            label=f'spurious ({len(spur)})')
    ax.set(yscale='log', xlabel='coherence fraction s  (0 = Q floor, 1 = decay-time cap)',
           ylabel='GP components', title='(a) Where components sit between the Q bounds')
    ax.legend(fontsize=8)

    ax = axs[1]
    lo = max(np.nanmin(qs['cycles']), 0.1)
    bins = np.logspace(np.log10(lo), np.log10(np.nanmax(qs['cycles']) * 1.05), 40)
    ax.hist(real['cycles'], bins=bins, alpha=.65, color='seagreen', label='matches an injection')
    ax.hist(spur['cycles'], bins=bins, alpha=.65, color='crimson', label='spurious')
    ax.axvline(1.0, color='#5f5e5a', ls=':', lw=1)
    ax.text(1.0, ax.get_ylim()[1] * .8, ' soft prior\n (1 period)', fontsize=8, color='#5f5e5a')
    ax.set(xscale='log', yscale='log', xlabel='decay time in periods  (Q / π)',
           ylabel='GP components', title='(b) Decay time in periods')
    ax.legend(fontsize=8)

    pc = purity_curve(qs, cycle_cuts, by='cycles')
    ax = axs[2]
    ax.plot(pc['cycles_cut'], pc['purity'], 'o-', color='seagreen', label='purity')
    ax.plot(pc['cycles_cut'], pc['completeness'], 's-', color='steelblue', label='completeness')
    ax.axvline(1.6, color='#5f5e5a', ls='--', lw=1)
    ax.text(1.65, .05, 'default cut 1.6\n(= old Q ≥ 5)', fontsize=8, color='#5f5e5a')
    ax.set(xscale='symlog', ylim=(0, 1.05), xlabel='cycles cut  (Q / π ≥)',
           ylabel='fraction', title='(c) Filtering GP components on cycles')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
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
    """GP red-noise component vs the injected red noise, per pipeline row.

    Works for both GP red-noise models.  With ``rn_model='powerlaw'`` the slope is
    fitted directly and ``gp_rn_alpha_band`` (the in-band slope of the fitted PSD) is used
    as ``gp_rn_alpha_eff``; with the legacy SHO it is derived from the break period.  The
    amplitude column is ``gp_rn_rms_band`` (power law) or ``gp_rn_sigma`` (SHO).
    """
    cols = ['name', 'n_osc', 'rn_rms_realized', 'inj_rn_alpha', 'rn_alpha']
    out = df[cols].copy()
    for c in ('gp_rn_sigma', 'gp_rn_period', 'gp_rn_Q', 'gp_rn_alpha_band', 'gp_rn_rms_band'):
        out[c] = pd.to_numeric(df[c], errors='coerce') if c in df.columns else np.nan
    powerlaw = np.isfinite(out['gp_rn_alpha_band'])
    sho_eff = [gp_rn_effective_alpha(p, q if np.isfinite(q) else 1/np.sqrt(2), f_lo, f_hi)
               if not pl else np.nan
               for p, q, pl in zip(out['gp_rn_period'], out['gp_rn_Q'], powerlaw)]
    out['gp_rn_alpha_eff'] = np.where(powerlaw, out['gp_rn_alpha_band'], sho_eff)
    out['gp_rn_sigma'] = np.where(powerlaw, out['gp_rn_rms_band'], out['gp_rn_sigma'])
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
    for col, c, lbl in [('d_alpha_gp', 'darkorange', 'GP red-noise slope'),
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


# ---------------------------------------------------------------------------
# Red-noise fix: compare pipeline configurations (docs/fixing_RN_fitting.tex)
# ---------------------------------------------------------------------------

_CROWD_BINS = ((0, 1), (1, 4), (4, 7), (7, 11))


def _crowding_label(n):
    for lo, hi in _CROWD_BINS:
        if lo <= n < hi:
            return f'{lo}-{hi - 1}'
    return 'other'


def slope_recovery_table(configs):
    """Median (fitted - injected) alpha by crowding, for every slope estimate available.

    Parameters
    ----------
    configs : dict
        label -> joined table (output of ``join``).  All must hold the same rows.

    Returns
    -------
    pd.DataFrame
        One row per (config, estimator): median bias per crowding bin, median |bias|,
        and Pearson correlation with the injected alpha.  Estimators: ``vaughan`` (the
        table's ``rn_alpha``) and ``gp`` (``gp_rn_bias``'s slope, whichever RN model).
    """
    rows = []
    for label, df in configs.items():
        a = gp_rn_bias(df)
        crowd = a['n_osc'].map(_crowding_label)
        powerlaw = 'gp_rn_alpha' in df.columns and np.isfinite(
            pd.to_numeric(df['gp_rn_alpha'], errors='coerce')).any()
        for est, col in (('vaughan', 'rn_alpha'), ('gp', 'gp_rn_alpha_eff')):
            fit = pd.to_numeric(a[col], errors='coerce')
            d = fit - a['inj_rn_alpha']
            m = np.isfinite(d)
            row = dict(config=label, estimator=est)
            for lo, hi in _CROWD_BINS:
                k = f'{lo}-{hi - 1}'
                row[f'bias n_osc {k}'] = float(d[m & (crowd == k)].median())
            row['median |bias|'] = float(d[m].abs().median())
            row['corr w/ truth'] = float(np.corrcoef(fit[m], a.loc[m, 'inj_rn_alpha'])[0, 1])
            if est == 'gp' and powerlaw:
                p = pd.to_numeric(df['gp_rn_alpha'], errors='coerce')
                row['frac at alpha bound'] = float(((p <= 0.051) | (p >= 1.949)).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def gp_recovery_table(configs, reference):
    """Oscillator recovery and component purity per configuration.

    Detectability is judged against ONE continuum for every config (``reference``'s
    ``rn_log10_N``/``rn_alpha``), so "detectable" means the same thing in each row.
    Pass the peak-masked table: its continuum is the closer one to the truth.
    """
    rows = []
    ref = reference[['rn_log10_N', 'rn_alpha']]
    for label, df in configs.items():
        osc = per_oscillator_table(df, continuum=ref)
        s = recovery_summary(osc)
        qs = gp_component_purity(df)
        rows.append(dict(
            config=label,
            recovery_overall=s['recovery_overall'],
            recovery_detectable=s['recovery_detectable'],
            recovery_undetectable=s['recovery_undetectable'],
            n_detectable=s['n_detectable'],
            n_detectable_but_missed=s['n_detectable_but_missed'],
            n_gp_components=len(qs),
            component_purity=float(qs['matches_injected'].mean()),
            mean_n_components=float(pd.to_numeric(df['gp_n_components'], errors='coerce').mean()),
        ))
    return pd.DataFrame(rows)


def bic_comparison(configs, baseline):
    """Per-row BIC difference of each config against ``baseline`` (negative = better).

    Rows share their light curves across configs, so the log-likelihoods are directly
    comparable; red-noise terms have 2 parameters in both models.  Restricting to rows
    with the same number of SHO components isolates the red-noise term itself.
    """
    b = pd.to_numeric(configs[baseline]['gp_bic'], errors='coerce')
    mb = pd.to_numeric(configs[baseline]['gp_n_components'], errors='coerce')
    rows = []
    for label, df in configs.items():
        if label == baseline:
            continue
        d = pd.to_numeric(df['gp_bic'], errors='coerce') - b
        same_m = pd.to_numeric(df['gp_n_components'], errors='coerce') == mb
        rows.append(dict(config=label, vs=baseline,
                         median_dBIC=float(d.median()),
                         frac_better=float((d < 0).mean()),
                         n_same_n_components=int(same_m.sum()),
                         median_dBIC_same_n=float(d[same_m].median()),
                         frac_better_same_n=float((d[same_m] < 0).mean())))
    return pd.DataFrame(rows)


def plot_slope_recovery(configs):
    """Fitted vs injected alpha: top row the Vaughan continuum, bottom row the GP term."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(configs), figsize=(4.6 * len(configs), 8.6), squeeze=False)
    sc = None
    for j, (label, df) in enumerate(configs.items()):
        a = gp_rn_bias(df)
        for i, (what, col) in enumerate((('Vaughan continuum', 'rn_alpha'),
                                         ('GP red-noise term', 'gp_rn_alpha_eff'))):
            ax = axs[i, j]
            x = a['inj_rn_alpha']
            y = pd.to_numeric(a[col], errors='coerce')
            m = np.isfinite(y)
            sc = ax.scatter(x[m], y[m], s=10, alpha=.5, c=a.loc[m, 'n_osc'],
                            cmap='viridis', vmin=0, vmax=10)
            ax.plot([0, 2.4], [0, 2.4], 'k--', lw=1)
            r = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 else np.nan
            ax.set(xlim=(-0.1, 2.5), ylim=(-0.1, 4.1), xlabel='injected α', ylabel='fitted α',
                   title=f'{label}: {what}\nmedian bias {np.median(y[m] - x[m]):+.2f},  '
                         f'r = {r:.2f}')
            ax.grid(alpha=.3)
    fig.subplots_adjust(hspace=0.4)
    fig.colorbar(sc, ax=axs, label='n_osc', shrink=0.6)
    return fig


def row_white_noise(r):
    """Median photometric error and fitted jitter in quadrature (= gp_table_io.row_white_noise)."""
    err = r.get('LC_flux_err')
    err = float(np.nanmedian(np.asarray(err, float))) if err is not None else np.nan
    jit = r.get('gp_jitter', np.nan)
    jit = float(jit) if jit is not None and np.isfinite(float(jit)) else 0.0
    return float(np.sqrt(err ** 2 + jit ** 2))


def component_filter_scan(df, cuts=((0, 0), (0.05, 0), (0.10, 0), (0.20, 0), (0, 20), (0, 100),
                                    (0.10, 20), (0.10, 100), (0.20, 100)), tol=0.10, window=0.02,
                          by='Q', amp='rel'):
    """Purity and completeness of the GP component list under amplitude / coherence cuts.

    Companion to ``gp_table_io.filter_gp_components``: it measures what a given cut costs.
    Purity is the fraction of surviving components matching an injected period; completeness
    is the fraction of DETECTABLE injections (LSP peak above the row's significance threshold)
    still matched by a surviving component.  ``cuts`` are (amplitude_min, coherence_min)
    pairs.  The first element is sigma / max(sigma in row) when ``amp='rel'`` or
    sigma / white noise when ``amp='snr'`` (the pipeline default since 2026-11); the second
    is a Q cut when ``by='Q'`` or a cycles (Q / pi) cut when ``by='cycles'``.  Output columns
    are named ``sigma_rel_cut`` / ``snr_cut`` and ``Q_cut`` / ``cycles_cut`` accordingly.
    """
    if by not in ('Q', 'cycles'):
        raise ValueError(f"by must be 'Q' or 'cycles', got {by!r}")
    if amp not in ('rel', 'snr'):
        raise ValueError(f"amp must be 'rel' or 'snr', got {amp!r}")
    q_scale = np.pi if by == 'cycles' else 1.0
    amp_col = 'sigma_rel' if amp == 'rel' else 'snr'
    comps, det_map, n_det = [], {}, 0
    for i, r in df.iterrows():
        P = np.atleast_1d(np.asarray(r.get('gp_periods', []), float))
        S = np.atleast_1d(np.asarray(r.get('gp_sho_sigmas', []), float))
        Q = np.atleast_1d(np.asarray(r.get('gp_sho_Qs', []), float))
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        f = np.asarray(r['LSP_freq'], float); p = np.asarray(r['LSP_power'], float)
        cont = 10.0 ** float(r['rn_log10_N']) * f ** (-float(r['rn_alpha']))
        gam = -np.log(1 - 0.99 ** (1 / len(f)))
        det = []
        for Pi in inj:
            fi = 1.0 / Pi
            if fi < f[0] or fi > f[-1]:
                continue
            w = np.abs(f / fi - 1) < window
            if w.any() and np.nanmax(p[w]) > gam * np.interp(fi, f, cont):
                det.append(Pi)
        det_map[i] = det; n_det += len(det)
        if len(P) == 0 or not (len(P) == len(S) == len(Q)):
            continue
        smax = np.nanmax(S) if len(S) else np.nan
        wn = row_white_noise(r)
        for per, sig, q in zip(P, S, Q):
            comps.append(dict(row=i, period=per, sigma_rel=sig / smax if smax > 0 else np.nan,
                              snr=sig / wn if np.isfinite(wn) and wn > 0 else np.nan, Q=q,
                              real=bool(len(inj)) and bool(np.any(np.abs(inj / per - 1) < tol))))
    comps = pd.DataFrame(comps)
    rows = []
    for smin, qmin in cuts:
        keep = comps[(comps[amp_col] >= smin) & (comps['Q'] >= qmin * q_scale)]
        by_row = {k: g['period'].to_numpy() for k, g in keep.groupby('row')}
        found = sum(1 for i, det in det_map.items() for Pi in det
                    if len(by_row.get(i, [])) and np.any(np.abs(by_row[i] / Pi - 1) < tol))
        rows.append({f'{amp_col}_cut': smin, f'{by}_cut': qmin, 'components': len(keep),
                     'purity': keep['real'].mean() if len(keep) else np.nan,
                     'completeness': found / n_det if n_det else np.nan, 'detectable': n_det})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Where the false positives come from
# ---------------------------------------------------------------------------
#
# ``gp_component_purity`` says HOW MANY fitted components match nothing injected; it
# does not say WHY, and the answer decides whether a spurious component matters.  A
# component sitting on top of a real signal but failing the 10% period test is the same
# signal counted twice; a component whose Q is on the floor is absorbing broadband power
# rather than claiming a periodicity.  Those are bookkeeping, not false detections.  A
# high-Q, high-amplitude component at a frequency unrelated to anything injected is the
# only kind that would corrupt a measurement.
#
# Two things make this classification honest rather than decorative:
#
# 1. **Frequency, not period.**  The 10% period tolerance used elsewhere is a frequency
#    window of 0.5/day at P = 0.2 d but 0.01/day at P = 10 d, so it means something
#    different in every part of the band.  Two frequencies closer than the resolution
#    1/T_baseline are not separable at all, so "within ``k_res``/T of an injection" is the
#    statement that a component and an injection are the same signal.
# 2. **A null.**  With ~9 injections per row spread over two decades, a component placed at
#    random lands near *something* surprisingly often -- an earlier version of this code
#    called 25% of all components "near-miss sidelobes" when the chance rate was 25.1%.
#    ``false_positive_null`` re-runs the identical ladder on randomly placed components, so
#    every class is reported as an excess over chance instead of a raw count.

FP_ORDER = ['matched', 'unresolved duplicate', 'harmonic of injection',
            'broadband (Q at floor)', 'low amplitude', 'unrelated frequency',
            'no injection in row']

# Colour-blind-safe. Green = real, red = the only class that is a genuine false detection.
FP_COLOURS = {'matched': '#1baf7a', 'unresolved duplicate': '#2a78d6',
              'harmonic of injection': '#7e57c2', 'broadband (Q at floor)': '#b0a24a',
              'low amplitude': '#c9c7bf', 'unrelated frequency': '#d1352b',
              'no injection in row': '#8f8e88'}


def _baseline_days(r):
    t = np.asarray(r.get('LC_t', []), float)
    return float(np.ptp(t)) if t.size > 1 else np.nan


def _classify_component(per, amp_low, q, inj, T, tol, k_res, harmonics, harmonic_tol,
                        q_floor, q_floor_tol):
    """The priority ladder; first match wins, so every component gets exactly one label.

    ``amp_low`` is the caller's verdict on the amplitude (below the filter's floor), so the
    same ladder serves the relative and the white-noise-referenced cut.
    """
    if not len(inj):
        return 'no injection in row', np.nan, np.nan, np.nan
    frac = np.abs(per / inj - 1.0)
    j = int(np.argmin(frac))
    nearest, offset, ratio = float(inj[j]), float(frac[j]), float(per / inj[j])
    df_min = float(np.min(np.abs(1.0 / per - 1.0 / inj)))

    if offset < tol:
        kind = 'matched'
    elif np.isfinite(T) and T > 0 and df_min < k_res / T:
        kind = 'unresolved duplicate'
    elif any(abs(ratio / h - 1.0) < harmonic_tol or abs(ratio * h - 1.0) < harmonic_tol
             for h in harmonics):
        kind = 'harmonic of injection'
    elif np.isfinite(q) and q <= q_floor * (1.0 + q_floor_tol):
        kind = 'broadband (Q at floor)'
    elif amp_low:
        kind = 'low amplitude'
    else:
        kind = 'unrelated frequency'
    return kind, nearest, offset, df_min


def _amp_low(sig, srel, snr, snr_min, sigma_rel_min):
    """True when the component fails the amplitude part of the filter."""
    low = False
    if snr_min is not None:
        low |= not (np.isfinite(snr) and snr >= snr_min)
    if sigma_rel_min is not None:
        low |= not (np.isfinite(srel) and srel >= sigma_rel_min)
    return bool(low)


def false_positive_table(df, tol=0.10, k_res=2.0, harmonics=(2, 3, 4), harmonic_tol=0.05,
                         q_floor=1.0 / np.sqrt(2.0), q_floor_tol=0.02,
                         snr_min=0.2, sigma_rel_min=None, Q_min=None, n_cycles_min=1.6):
    """One row per fitted GP component, classified by what it most likely is.

    ``survives_filter`` mirrors ``filter_gp_components``: amplitude sigma / white noise
    >= ``snr_min`` (default 0.2; ``sigma_rel_min`` adds the old relative floor when given)
    and coherence ``Q / pi >= n_cycles_min`` (default 1.6, the same as the old ``Q >= 5``;
    pass ``Q_min`` to cut on Q directly instead).  The ``low amplitude`` class uses the same
    amplitude verdict.

    Classes, in priority order:

    ``matched``                 within ``tol`` in fractional period -- not a false positive.
    ``unresolved duplicate``    a different period but the SAME frequency to within
                                ``k_res``/T_baseline: the peak finder seeded one signal twice
                                (sidelobe or a neighbouring periodogram bin) and the GP fitted
                                both.  Real structure, double-counted.
    ``harmonic of injection``   period ratio within ``harmonic_tol`` of an integer or its
                                reciprocal.  Also real structure, wrong bookkeeping.
    ``broadband (Q at floor)``  Q pinned at the hard floor: absorbing continuum power.
    ``low amplitude``           fails the amplitude floor (sigma / white noise < ``snr_min``).
    ``unrelated frequency``     none of the above -- **the only genuine false detection.**
    ``no injection in row``     the row had no oscillators at all, so every component is
                                spurious by construction; these rows measure the false-alarm
                                rate directly.

    ``survives_filter`` reproduces ``gp_table_io.filter_gp_components``, so the same table
    answers what the DELIVERED component list contains, not just what the fit emitted.
    """
    q_keep = float(Q_min) if Q_min is not None else float(n_cycles_min) * np.pi
    rows = []
    for _, r in df.iterrows():
        P = np.atleast_1d(np.asarray(r.get('gp_periods', []), float))
        S = np.atleast_1d(np.asarray(r.get('gp_sho_sigmas', []), float))
        Q = np.atleast_1d(np.asarray(r.get('gp_sho_Qs', []), float))
        if len(P) == 0 or not (len(P) == len(S) == len(Q)):
            continue
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        inj = inj[np.isfinite(inj)]
        T = _baseline_days(r)
        smax = np.nanmax(S) if len(S) and np.isfinite(np.nanmax(S)) else np.nan
        wn = row_white_noise(r)

        for per, sig, q in zip(P, S, Q):
            srel = sig / smax if (np.isfinite(smax) and smax > 0) else np.nan
            snr = sig / wn if (np.isfinite(wn) and wn > 0) else np.nan
            amp_low = _amp_low(sig, srel, snr, snr_min, sigma_rel_min)
            kind, nearest, offset, df_min = _classify_component(
                per, amp_low, q, inj, T, tol, k_res, harmonics, harmonic_tol,
                q_floor, q_floor_tol)
            rows.append(dict(name=r['name'], n_sectors=r.get('n_sectors', np.nan),
                             baseline_days=T, period=per, sigma=sig, sigma_rel=srel, snr=snr, Q=q,
                             cycles=q / np.pi,
                             n_inj=len(inj), nearest_inj=nearest, offset=offset,
                             delta_f=df_min, res_elements=df_min * T if np.isfinite(T) else np.nan,
                             kind=kind, matched=(kind == 'matched'),
                             survives_filter=bool(not amp_low and np.isfinite(q) and q >= q_keep)))
    out = pd.DataFrame(rows)
    if len(out):
        out['kind'] = pd.Categorical(out['kind'], categories=FP_ORDER, ordered=True)
    return out


def false_positive_null(df, n_draws=20, seed=0, p_range=None, **kw):
    """The same ladder applied to randomly placed components -- the chance baseline.

    Each row keeps its own component count, amplitudes and Qs (so the amplitude- and
    Q-based classes are calibrated too); only the PERIODS are redrawn log-uniformly over
    the observed range.  Returns per-class counts scaled to one draw.
    """
    rng = np.random.default_rng(seed)
    tol = kw.get('tol', 0.10)
    if p_range is None:
        allp = np.concatenate([np.atleast_1d(np.asarray(r.get('gp_periods', []), float))
                               for _, r in df.iterrows()] or [np.array([1.0])])
        allp = allp[np.isfinite(allp) & (allp > 0)]
        p_range = (float(np.min(allp)), float(np.max(allp)))
    lo, hi = np.log10(p_range[0]), np.log10(p_range[1])

    counts = {k: 0 for k in FP_ORDER}
    for _, r in df.iterrows():
        S = np.atleast_1d(np.asarray(r.get('gp_sho_sigmas', []), float))
        Q = np.atleast_1d(np.asarray(r.get('gp_sho_Qs', []), float))
        n = len(S)
        if n == 0 or len(Q) != n:
            continue
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        inj = inj[np.isfinite(inj)]
        T = _baseline_days(r)
        smax = np.nanmax(S) if np.isfinite(np.nanmax(S)) else np.nan
        wn = row_white_noise(r)
        for _ in range(n_draws):
            for per, sig, q in zip(10 ** rng.uniform(lo, hi, n), S, Q):
                srel = sig / smax if (np.isfinite(smax) and smax > 0) else np.nan
                snr = sig / wn if (np.isfinite(wn) and wn > 0) else np.nan
                amp_low = _amp_low(sig, srel, snr, kw.get('snr_min', 0.2), kw.get('sigma_rel_min'))
                kind, *_ = _classify_component(
                    per, amp_low, q, inj, T, tol, kw.get('k_res', 2.0),
                    kw.get('harmonics', (2, 3, 4)), kw.get('harmonic_tol', 0.05),
                    kw.get('q_floor', 1.0 / np.sqrt(2.0)), kw.get('q_floor_tol', 0.02))
                counts[kind] += 1
    return pd.Series({k: v / n_draws for k, v in counts.items()}, name='expected_by_chance')


def false_positive_summary(fp, null=None):
    """Counts per class, before and after the standard filter, with the chance baseline."""
    tot, kept = len(fp), int(fp['survives_filter'].sum())
    g = fp.groupby('kind', observed=False)
    out = g.agg(n=('period', 'size'), n_after_filter=('survives_filter', 'sum'),
                median_sigma_rel=('sigma_rel', 'median'), median_Q=('Q', 'median')).reset_index()
    out['frac'] = out.n / max(tot, 1)
    out['frac_after_filter'] = out.n_after_filter / max(kept, 1)
    out['filter_removes'] = 1 - out.n_after_filter / out.n.replace(0, np.nan)
    if null is not None:
        out['expected_by_chance'] = out.kind.map(null).astype(float)
        out['excess'] = out.n - out.expected_by_chance
    return out


def plot_false_positives(fp, null=None, snr_min=0.2, sigma_rel_min=None, Q_min=None,
                         n_cycles_min=1.6, q_floor=1.0 / np.sqrt(2.0), k_res=2.0):
    """Four panels: what the spurious components are, and which of them survive the filter.

    The cut arguments only draw the cut that ``false_positive_table`` applied; pass the same
    values to both.  Panel (b) is drawn in sigma / white noise when ``snr_min`` is set, else
    in sigma / max(sigma).
    """
    import matplotlib.pyplot as plt

    if Q_min is None:
        Q_min = float(n_cycles_min) * np.pi
        cut_label = f'Q/π ≥ {n_cycles_min:g} (Q ≥ {Q_min:.1f})'
    else:
        cut_label = f'Q ≥ {Q_min:g}'
    if snr_min is not None:
        amp_col, amp_min, amp_label, amp_axis = 'snr', snr_min, f'σ/WN ≥ {snr_min:g}', 'σ / white noise (per-point error ⊕ jitter)'
    else:
        amp_col, amp_min, amp_label, amp_axis = 'sigma_rel', sigma_rel_min, f'σ_rel ≥ {sigma_rel_min:g}', 'σ / max σ in row'
    summ = false_positive_summary(fp, null)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10.5))

    # (a) frequency offset from the nearest injection, against the chance expectation
    ax = axs[0, 0]
    sub = fp[np.isfinite(fp.res_elements)]
    bins = np.logspace(-2.2, 2.7, 34)
    ax.hist([sub.loc[sub.kind == k, 'res_elements'] for k in FP_ORDER if (sub.kind == k).any()],
            bins=bins, stacked=True,
            color=[FP_COLOURS[k] for k in FP_ORDER if (sub.kind == k).any()],
            label=[k for k in FP_ORDER if (sub.kind == k).any()])
    ax.axvline(k_res, color='#5f5e5a', ls='--', lw=1.2)
    ax.text(k_res * 1.15, ax.get_ylim()[1] * .92, f'{k_res:g} resolution\nelements', fontsize=8,
            color='#5f5e5a')
    ax.set(xscale='log', xlabel='|Δf| to nearest injection, in resolution elements (1/T)',
           ylabel='GP components')
    ax.set_title('(a) Components pile up ON the injected frequencies', loc='left',
                 fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper right')

    # (b) the plane the filter actually cuts in
    ax = axs[0, 1]
    yv = fp[amp_col]
    ylo = max(float(np.nanpercentile(yv[yv > 0], 0.5)) * 0.5, 1e-5) if (yv > 0).any() else 1e-3
    yhi = float(np.nanmax(yv)) * 1.5 if np.isfinite(np.nanmax(yv)) else 1.5
    for kind in FP_ORDER:
        s = fp[fp.kind == kind]
        if len(s):
            ax.scatter(s.Q.clip(upper=1e3), s[amp_col].clip(lower=ylo), s=15, alpha=.55, lw=0,
                       color=FP_COLOURS[kind])
    ax.axvline(Q_min, color='#5f5e5a', lw=1.2, ls='--')
    ax.axhline(amp_min, color='#5f5e5a', lw=1.2, ls='--')
    ax.axvline(q_floor, color='#b0a24a', lw=1, ls=':')
    ax.fill_between([Q_min, 1e3], amp_min, yhi, color='#1baf7a', alpha=.07, zorder=0)
    ax.text(Q_min * 1.35, yhi * 0.6, 'kept by filter', fontsize=9, color='#1baf7a')
    ax.set(xscale='log', yscale='log', xlabel='fitted Q (clipped at 1000)',
           ylabel=amp_axis, ylim=(ylo, yhi))
    ax.set_title(f'(b) The filter cut: {amp_label} and {cut_label}',
                 loc='left', fontweight='bold')

    # (c) counts per class, with the chance baseline and the effect of the filter
    ax = axs[1, 0]
    s = summ[summ.n > 0]
    ypos = np.arange(len(s))
    ax.barh(ypos, s.n, color=[FP_COLOURS[k] for k in s.kind], alpha=.45, label='fitted')
    ax.barh(ypos, s.n_after_filter, color=[FP_COLOURS[k] for k in s.kind],
            label='survives filter')
    if null is not None:
        ax.scatter(s.expected_by_chance, ypos, marker='|', s=260, color='black', lw=1.8,
                   zorder=5, label='expected by chance')
    for i, r in enumerate(s.itertuples()):
        ax.text(r.n * 1.02, i, f'  {int(r.n)} → {int(r.n_after_filter)}', va='center',
                fontsize=8.5, color='#5f5e5a')
    ax.set_yticks(ypos, list(s.kind))
    ax.invert_yaxis()
    ax.set(xlabel='GP components', xlim=(0, s.n.max() * 1.3))
    ax.set_title('(c) What they are, and what the filter removes', loc='left',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (d) what is left in the delivered component list
    ax = axs[1, 1]
    kept = fp[fp.survives_filter]
    pb = np.logspace(np.log10(max(np.nanmin(fp.period), 1e-2)), np.log10(np.nanmax(fp.period)), 28)
    ax.hist([kept.loc[kept.matched, 'period'], kept.loc[~kept.matched, 'period']],
            bins=pb, stacked=True, color=['#1baf7a', '#d1352b'],
            label=[f'matched ({int(kept.matched.sum())})',
                   f'spurious ({int((~kept.matched).sum())})'])
    bad = kept[kept.kind == 'unrelated frequency']
    ax.set(xscale='log', xlabel='component period (d)', ylabel='components after filter')
    pur = kept.matched.mean() if len(kept) else np.nan
    ax.set_title(f'(d) Delivered list: purity {pur:.1%}, '
                 f'{len(bad)} genuinely unrelated', loc='left', fontweight='bold')
    ax.legend(fontsize=8)

    fig.suptitle('Where the synthetic false positives come from', fontsize=14,
                 fontweight='bold', x=0.005, ha='left')
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    return fig

"""How the GP pipeline performs on the synthetic-population light curves.

Run after ``gp_eval_run`` (env ``tess_gp``, from ``GenerateTable_CGW/sytheticdata_generate_compare``):

    python -m sythn_real_variability.gp_eval_figs

Writes ``results/gp_eval.png`` and ``results/gp_eval_*.csv``.

What counts as a detectable injection
-------------------------------------
A cluster's signal is thousands of sinusoids, one or two per variable star, each of
fractional amplitude a_j = (star's share of the cluster flux) x (catalogue amplitude); almost
all are far below the noise.  Scoring against every one would be meaningless, and reading
the periodogram near each injected frequency (as for the 0-10-sinusoid set) would credit
noise and neighbours' leakage.  Instead each sinusoid's periodogram power is PREDICTED from
the row's own sampling and errors,

    P_pred(f_j) = (a_j sinc(E/P_j))^2 / 4 * sum_i 1/sigma_i^2      (astropy 'psd' normalisation),

with E the exposure / resampling bin, and the injection is **detectable** when P_pred exceeds
(gamma - 1) C(f_j), C the row's fitted red-noise continuum and gamma the Vaughan 1% factor:
its expected periodogram power, signal plus continuum, then clears the significance threshold
gamma C that the peak finder uses.  A detectable injection is **separable** when no LOUDER
detectable injection lies within one resolution element 1 / T_eff (effective baseline, gaps
excluded).

purity        delivered components within 10% in period of a detectable injection / delivered
completeness  separable detectable injections matched by a delivered component (10% in period)

Delivered = the RECOMMENDED filter for this set: evidence ``gp_sho_dloglike_seq >= 10`` and
cycles Q/pi >= 1.6, no amplitude cut.  The current default (sigma / white noise >= 0.2,
cycles >= 1.6) is scored alongside; panel (b) compares both cuts on this set and on the
clean-sinusoid injection set.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.abspath(os.path.join(HERE, '..', '..'))
if GT not in sys.path:
    sys.path.insert(0, GT)

from sythn_real_variability.gp_eval_run import GP_OUT                      # noqa: E402
from sythn_real_variability.config import MAIN_REPO                        # noqa: E402
from table_pipeline.lc_lsp import _effective_baseline                      # noqa: E402
from gp_pipeline.gp_table_io import row_white_noise                        # noqa: E402

TRUTH = os.path.join(MAIN_REPO, 'GenerateTable_CGW', 'sytheticdata_generate_compare',
                     'sythn_real_variability', 'data', 'ground_truth_populations.pkl')
RESULTS = os.path.join(HERE, 'results')
TOL = 0.10
SNR_MIN, N_CYCLES_MIN = 0.2, 1.6         # current default delivered-list filter
DLL_MIN = 10.0                           # recommended: evidence cut, ~ the BIC cost 1.5 ln N of a component
SNR_SCAN = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
DLL_SCAN = (0, 1, 3, 5, 7, 10, 15, 20, 30, 50)
Q_KEEP = N_CYCLES_MIN * np.pi
BIN_DAYS = 30.0 / 1440.0                        # pipeline resampling bin

# Validated categorical pair (dataviz validate_palette.js): blue / orange; neutral for "not the pipeline".
BLUE, ORANGE, NEUTRAL = '#2a78d6', '#eb6834', '#b5b4ad'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                     'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'legend.frameon': False,
                     'figure.dpi': 110, 'font.size': 10})


def _arr(v):
    return np.atleast_1d(np.asarray(v, float)) if v is not None else np.array([])


def load_sinusoids(path):
    """All injected sinusoid terms of one cluster: amplitude (fraction of cluster flux), period."""
    s = Table.read(path).to_pandas()
    F = s['F_es'].to_numpy(float)
    f = F / np.nansum(F)
    a, P = [], []
    for k in ('1', '2'):
        A = s['A' + k].to_numpy(float); Pk = s['P' + k].to_numpy(float)
        m = np.isfinite(A) & np.isfinite(Pk) & s['is_var'].to_numpy(bool)
        a.append(f[m] * A[m]); P.append(Pk[m])
    return np.concatenate(a), np.concatenate(P)


def evaluate_row(r, a, P):
    """Per-row detectability of every injection, and per-component classification."""
    t = _arr(r['LC_t']); err = _arr(r['LC_flux_err'])
    f = _arr(r['LSP_freq'])
    gam = -np.log(1 - 0.99 ** (1 / len(f)))
    T = _effective_baseline(t)
    w = np.nansum(1.0 / err[np.isfinite(err) & (err > 0)] ** 2)
    E = max(BIN_DAYS, float(r.get('cadence', 30.0)) / 1440.0)
    fi = 1.0 / P
    inb = (fi >= f[0]) & (fi <= f[-1])
    a_eff = np.abs(a * np.sinc(E / P))
    p_pred = a_eff ** 2 / 4.0 * w
    C = 10 ** float(r['rn_log10_N']) * np.where(inb, fi, 1.0) ** (-float(r['rn_alpha']))
    snr = p_pred / ((gam - 1.0) * C)                       # >= 1: expected power above threshold
    det = inb & (snr >= 1.0)
    di = np.flatnonzero(det)
    di = di[np.argsort(-a_eff[di])]                        # loudest first
    sep = np.ones(len(di), bool)
    for k in range(1, len(di)):
        sep[k] = np.all(np.abs(fi[di[:k]] - fi[di[k]]) * T >= 1.0)
    return dict(T=T, gam=gam, fi=fi, a_eff=a_eff, snr=snr, inb=inb, det_idx=di, sep=sep)


def analyse(df, truth):
    """Filter-independent tables: detectable injections, ALL fitted components, rows."""
    inj_rows, comp_rows, row_rows = [], [], []
    cache = {}
    for i, r in df.iterrows():
        name = r['name']
        if name not in cache:
            cache = {name: load_sinusoids(truth.loc[name, 'star_params_path'])}
        a, P = cache[name]
        ev = evaluate_row(r, a, P)
        di, sep = ev['det_idx'], ev['sep']
        Pdet = P[di]
        Pg, S, Q = _arr(r.get('gp_periods')), _arr(r.get('gp_sho_sigmas')), _arr(r.get('gp_sho_Qs'))
        D = _arr(r.get('gp_sho_dloglike_seq'))
        if len(D) != len(Pg):
            D = np.full(len(Pg), np.nan)
        wn = row_white_noise(r)
        snr_c = S / wn if np.isfinite(wn) and wn > 0 else np.full(len(S), np.nan)
        seeds = _arr(r.get('gp_initial_lsp_peak_periods'))

        for j, s_ok in zip(di, sep):
            inj_rows.append(dict(row=i, name=name, age=r['age'], n_sectors=int(r['n_sectors']),
                                 P=P[j], a=a[j], a_eff=ev['a_eff'][j], snr=ev['snr'][j],
                                 separable=bool(s_ok),
                                 seeded=bool(len(seeds) and np.any(np.abs(seeds / P[j] - 1) < TOL))))

        for per, sg, q, sn, d in zip(Pg, S, Q, snr_c, D):
            if len(Pdet) and np.any(np.abs(Pdet / per - 1) < TOL):
                kind = 'matched'
            else:
                fc = 1.0 / per
                win = ev['inb'] & (np.abs(ev['fi'] - fc) * ev['T'] < 1.0)
                blend = float(np.sum(ev['snr'][win])) if win.any() else 0.0
                ratio = per / Pdet if len(Pdet) else np.array([])
                harm = bool(len(ratio)) and any(np.any(np.abs(ratio / h - 1) < 0.05) or
                                                np.any(np.abs(ratio * h - 1) < 0.05) for h in (2, 3, 4))
                if not len(Pdet):
                    kind = 'no detectable injection in row'
                elif harm:
                    kind = 'harmonic of a detectable injection'
                elif blend >= 1.0:
                    kind = 'blend of sub-threshold injections'
                elif np.min(np.abs(1 / Pdet - fc)) * ev['T'] < 2.0:
                    kind = 'within 2 resolution elements of one'
                else:
                    kind = 'no injected signal there'
            comp_rows.append(dict(row=i, name=name, age=r['age'], n_sectors=int(r['n_sectors']),
                                  period=per, sigma=sg, snr=sn, Q=q, cycles=q / np.pi, dll_seq=d, kind=kind))

        row_rows.append(dict(row=i, name=name, age=r['age'], n_sectors=int(r['n_sectors']),
                             cadence=r.get('cadence'), T_eff=ev['T'], n_inj=len(a),
                             n_det=len(di), n_sep=int(sep.sum()), n_comp=int(len(Pg)),
                             gp_rn_rms=float(r.get('gp_rn_rms_band', np.nan)),
                             true_rms=float(truth.loc[name, 'clean_rms_30min']),
                             true_rn_rms=float(truth.loc[name, 'rn_rms_realized']),
                             fluxfrac_missing=float(truth.loc[name, 'fluxfrac_missing']),
                             white_noise=row_white_noise(r)))
    return pd.DataFrame(inj_rows), pd.DataFrame(comp_rows), pd.DataFrame(row_rows)


def apply_filter(inj, comp, keep, why_removed='filter'):
    """Recovered flag and first-applicable miss reason for every detectable injection."""
    comp = comp.assign(kept=np.asarray(keep, bool))
    kept = {k: g.period.to_numpy() for k, g in comp[comp.kept].groupby('row')}
    allc = {k: g for k, g in comp.groupby('row')}
    rec, why = [], []
    for _, q in inj.iterrows():
        Pk = kept.get(q.row, np.array([]))
        hit = bool(len(Pk) and np.any(np.abs(Pk / q.P - 1) < TOL))
        rec.append(hit)
        if hit:
            why.append('recovered')
        elif not q.separable:
            why.append('unresolved')
        elif not q.seeded:
            why.append('not seeded')
        else:
            g = allc.get(q.row)
            near = g is not None and np.any(np.abs(g.period.to_numpy() / q.P - 1) < TOL)
            why.append(why_removed if near else 'fit moved')
    return inj.assign(recovered=rec, reason=why), comp


def scores(inj, comp, rows=None):
    """inj/comp after apply_filter; purity over kept components, completeness over separable."""
    if rows is not None:
        inj = inj[inj.row.isin(rows)]; comp = comp[comp.row.isin(rows)]
    k = comp[comp.kept]
    s = inj[inj.separable]
    return dict(delivered=len(k), purity=float((k.kind == 'matched').mean()) if len(k) else np.nan,
                n_sep=len(s), completeness=float(s.recovered.mean()) if len(s) else np.nan,
                n_det=len(inj), completeness_all=float(inj.recovered.mean()) if len(inj) else np.nan)


MISS = [('unresolved', 'unresolved from a louder\ndetectable injection', False),
        ('not seeded', 'not seeded by the peak finder', True),
        ('fit moved', 'seeded, but the fit moved the\ncomponent > 10% in period', True),
        ('evidence cut', f'fitted, removed by the filter\n(ΔlogL ≥ {DLL_MIN:g}, cycles ≥ {N_CYCLES_MIN:g})', True)]
FP = ['harmonic of a detectable injection', 'within 2 resolution elements of one',
      'blend of sub-threshold injections', 'no injected signal there', 'no detectable injection in row']
FP_REAL = {'harmonic of a detectable injection', 'within 2 resolution elements of one',
           'blend of sub-threshold injections'}


def hbars(ax, labels, counts, colors, fmt='{:d}'):
    y = np.arange(len(labels))[::-1]
    for yi, n, c in zip(y, counts, colors):
        ax.barh(yi, n, height=.62, color=c)
        ax.text(n + max(max(counts), 1) * .015, yi, fmt.format(int(n)), va='center', fontsize=9.5, color=INK)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlim(0, max(max(counts), 1) * 1.25)
    ax.grid(axis='y', visible=False)
    return y


def clean_set_scans():
    """Purity / completeness scans on the clean-sinusoid injection set (prewhitened fit), if available."""
    try:
        sc_dir = os.path.abspath(os.path.join(HERE, '..'))
        if sc_dir not in sys.path:
            sys.path.insert(0, sc_dir)
        import sidelobe_removal_figs as S
        main = os.path.join(MAIN_REPO, 'GenerateTable_CGW')
        df = S.C.join(pd.read_pickle(os.path.join(main, 'data', 'cluster_table_Nov04_3sectorcombos_synthetic_expavg_gp_taucap_prewhite.pkl')),
                      pd.read_pickle(os.path.join(main, 'sytheticdata_generate_compare', 'data', 'ground_truth.pkl')))
        cp = S.components(df)
        iso = pd.read_csv(os.path.join(main, 'sytheticdata_generate_compare', 'figs', 'sidelobe_removal',
                                       'isolated_detectability.csv'))
    except Exception as exc:                                            # noqa: BLE001
        print('clean-sinusoid set unavailable, skipping its curves:', exc)
        return None
    out = {}
    for lab, col, cuts in (('amplitude', 'snr', SNR_SCAN), ('evidence', 'dll_seq', DLL_SCAN)):
        pts = []
        for c in cuts:
            k = (cp.coherent & (cp[col] >= c)).to_numpy()
            pts.append((c, float(cp.matched[k].mean()), S.recovered_on(cp, k, iso)))
        out[lab] = pd.DataFrame(pts, columns=['cut', 'purity', 'completeness'])
    return out


def pop_scans(inj, comp):
    out = {}
    coh = comp.cycles >= N_CYCLES_MIN
    for lab, col, cuts in (('amplitude', 'snr', SNR_SCAN), ('evidence', 'dll_seq', DLL_SCAN)):
        pts = []
        for c in cuts:
            i2, c2 = apply_filter(inj, comp, coh & (comp[col] >= c))
            s = scores(i2, c2)
            pts.append((c, s['purity'], s['completeness']))
        out[lab] = pd.DataFrame(pts, columns=['cut', 'purity', 'completeness'])
    return out


def figure(inj, comp, rows, scans_pop, scans_sin, base_scores, out, stamp):
    """inj / comp already filtered with the recommended filter (apply_filter)."""
    fig, axs = plt.subplots(2, 3, figsize=(21, 12.8), gridspec_kw=dict(wspace=.62, hspace=.58))
    overall = scores(inj, comp)

    # (a) purity & completeness, recommended filter ---------------------------------------
    ax = axs[0, 0]
    groups = [('all rows', None),
              ('1-sector rows', rows.row[rows.n_sectors == 1]),
              ('3-sector rows', rows.row[rows.n_sectors == 3]),
              ('log age < 8', rows.row[rows.age < 8]),
              ('log age 8 – 9', rows.row[(rows.age >= 8) & (rows.age < 9)]),
              ('log age ≥ 9', rows.row[rows.age >= 9])]
    sc = [(g, scores(inj, comp, r)) for g, r in groups]
    y = np.arange(len(sc))[::-1]
    h = .36
    for yi, (g, s) in zip(y, sc):
        for dy, key, c in ((h / 2 + .02, 'purity', BLUE), (-h / 2 - .02, 'completeness', ORANGE)):
            v = s[key]
            if np.isfinite(v):
                ax.barh(yi + dy, v, height=h, color=c)
                ax.text(v + .01, yi + dy, f'{v:.2f}', va='center', fontsize=8.5)
            else:
                ax.text(.01, yi + dy, 'no separable injections', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, [f'{g}\n{s["delivered"]} delivered, {s["n_sep"]} separable inj.' for g, s in sc], fontsize=8.3)
    ax.set(xlim=(0, 1.15), xlabel='fraction')
    ax.grid(axis='y', visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label='purity'),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, label='completeness')],
              loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_title(f'(a) Purity {overall["purity"]:.2f}   Completeness {overall["completeness"]:.2f}',
                 loc='left', fontweight='bold', fontsize=12.5, pad=26)
    ax.text(0, -.13, f'Filter: ΔlogL ≥ {DLL_MIN:g} and cycles ≥ {N_CYCLES_MIN:g}, no amplitude cut.\n'
                     f'Current default (σ/WN ≥ {SNR_MIN:g}, cycles ≥ {N_CYCLES_MIN:g}): purity '
                     f'{base_scores["purity"]:.2f}, completeness {base_scores["completeness"]:.2f}.',
            transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')

    # (b) amplitude vs evidence filter on both injection sets ---------------------------------
    ax = axs[0, 1]
    sets = [('population set', scans_pop, '-')] + ([('clean-sinusoid set', scans_sin, ':')] if scans_sin else [])
    for setlab, sc_, ls in sets:
        for lab, c, mk in (('amplitude', INK2, 'o'), ('evidence', BLUE, 's')):
            d = sc_[lab]
            ax.plot(d.completeness, d.purity, ls=ls, marker=mk, ms=5, lw=1.6, color=c,
                    label=f'{setlab}, {"σ/WN" if lab == "amplitude" else "ΔlogL"} cut')
            pick = d[d.cut == (SNR_MIN if lab == 'amplitude' else DLL_MIN)]
            for _, q in pick.iterrows():
                ax.plot(q.completeness, q.purity, marker=mk, ms=11, mfc='none', mec=c, mew=1.6)
                ax.annotate(f'{"σ/WN" if lab == "amplitude" else "ΔlogL"} ≥ {q.cut:g}', (q.completeness, q.purity),
                            xytext=(6, -12 if lab == 'amplitude' else 6), textcoords='offset points',
                            fontsize=8, color=c)
    ax.set(xlabel='completeness (separable detectable injections)', ylabel='purity', xlim=(.3, 1.0), ylim=(.55, 1.01))
    ax.legend(fontsize=7.8, loc='lower left')
    ax.set_title('(b) Evidence beats amplitude on faint signals', loc='left', fontweight='bold', fontsize=12.5, pad=26)

    # (c) completeness against predicted significance -------------------------------------------
    ax = axs[0, 2]
    s = inj[inj.separable]
    edges = np.array([1, 1.5, 2, 3, 5, 10, 30, 100, 1e5])
    xc = np.sqrt(edges[:-1] * edges[1:]); xc[-1] = 300
    for ns, c, mk in ((1, BLUE, 'o'), (3, ORANGE, 's')):
        q = s[s.n_sectors == ns]
        g = q.groupby(pd.cut(q.snr, edges), observed=False).recovered.agg(['mean', 'size'])
        ok = (g['size'] >= 3).to_numpy()
        ax.plot(xc[ok], g['mean'].to_numpy()[ok], marker=mk, ms=6, lw=2, color=c, label=f'{ns}-sector rows ({len(q)})')
    ax.set(xscale='log', ylim=(0, 1.05), xlabel='predicted power / significance excess  (1 = at threshold)',
           ylabel='completeness')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('(c) Completeness by how far above threshold', loc='left', fontweight='bold', fontsize=12.5, pad=26)

    # (d) misses ---------------------------------------------------------------------------------
    ax = axs[1, 0]
    mc = inj[~inj.recovered].reason.value_counts()
    counts = [int(mc.get(k, 0)) for k, _, _ in MISS]
    hbars(ax, [t for _, t, _ in MISS], counts, [ORANGE if p else NEUTRAL for _, _, p in MISS])
    ax.set_xlabel('missed detectable injections')
    n_pipe = sum(c for c, (_, _, p) in zip(counts, MISS) if p)
    ax.set_title(f'(d) Why {int((~inj.recovered).sum())} detectable injections are missed', loc='left',
                 fontweight='bold', fontsize=12.5, pad=26)
    ax.text(0, -.13, f'Grey: not separable (outside the completeness denominator).\n'
                     f'Orange: {n_pipe} pipeline misses of {overall["n_sep"]} separable injections.',
            transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')

    # (e) false positives ---------------------------------------------------------------------------
    ax = axs[1, 1]
    k = comp[comp.kept]
    fc = k[k.kind != 'matched'].kind.value_counts()
    counts = [int(fc.get(x, 0)) for x in FP]
    hbars(ax, [x.replace(' of a', '\nof a').replace(' of one', '\nof one').replace(' injections', '\ninjections')
               .replace(' in row', '\nin row').replace(' there', '\nthere') for x in FP],
          counts, [NEUTRAL if x in FP_REAL else BLUE for x in FP])
    ax.set_xlabel('delivered components matching no detectable injection')
    n_noise = sum(c for c, x in zip(counts, FP) if x not in FP_REAL)
    nk = k[k.kind.isin([x for x in FP if x not in FP_REAL])]
    ax.set_title(f'(e) Why {int((k.kind != "matched").sum())} of {len(k)} delivered components are unmatched',
                 loc='left', fontweight='bold', fontsize=12.5, pad=26)
    ax.text(0, -.13, f'Grey: real injected structure the match criterion does not credit.\n'
                     f'Blue: {n_noise} with no injected signal behind them; median P {np.median(nk.period):.1f} d, '
                     f'{int((nk.period >= 9.5).sum())} at the 10 d bound.\nThey sit where the injected red noise is strongest.',
            transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')

    # (f) age trend --------------------------------------------------------------------------------
    ax = axs[1, 2]
    kept_var = k.groupby('row').sigma.apply(lambda v: float(np.sum(np.asarray(v) ** 2)))
    rr = rows.assign(gp_var=rows.row.map(kept_var).fillna(0.0))
    g = rr.groupby('age').agg(true=('true_rms', lambda v: np.sqrt(np.mean(np.square(v)))),
                              gp=('gp_var', lambda v: np.sqrt(2 * np.mean(v))),
                              rn=('true_rn_rms', 'median'), wn=('white_noise', 'median'),
                              gprn=('gp_rn_rms', 'median'))
    ax.plot(g.index, g.true, color=INK, lw=2.2, label='injected population signal (rms)')
    ax.plot(g.index, g.gp, color=BLUE, lw=2.2, marker='o', ms=5, label='GP delivered periodic signal (rms)')
    ax.plot(g.index, g.rn, color=ORANGE, lw=1.6, ls=(0, (4, 3)), label='injected red noise (rms)')
    ax.plot(g.index, g.wn, color=INK2, lw=1.4, ls=':', label='per-point white noise')
    ax.set(yscale='log', xlabel='log age', ylabel='fraction of flux (rms over rows, per age)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('(f) Age trend: signal, recovered, and noise', loc='left', fontweight='bold', fontsize=12.5, pad=26)
    from scipy.stats import spearmanr
    ax.text(0, -.13, f'Spearman with age over rows: injected signal {spearmanr(rr.age, rr.true_rms)[0]:+.2f}, '
                     f'GP periodic {spearmanr(rr.age, rr.gp_var)[0]:+.2f},\n'
                     f'injected red noise {spearmanr(rr.age, rr.true_rn_rms)[0]:+.2f}, white noise '
                     f'{spearmanr(rr.age, rr.white_noise)[0]:+.2f}: brightness ties the noise to age.',
            transform=ax.transAxes, fontsize=8.5, color=INK2, va='top')

    fig.suptitle('GP pipeline on the synthetic-population LMC clusters (prewhitened seeds → capped GP → '
                 f'ΔlogL ≥ {DLL_MIN:g}, cycles ≥ {N_CYCLES_MIN:g})',
                 fontsize=14.5, fontweight='bold', x=.01, ha='left', y=.995)
    fig.text(.01, -.005, stamp, fontsize=7.5, color=INK2)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def main():
    df = pd.read_pickle(GP_OUT)
    truth = pd.read_pickle(TRUTH).set_index('name')
    inj0, comp0, rows = analyse(df, truth)
    os.makedirs(RESULTS, exist_ok=True)

    coh = comp0.cycles >= N_CYCLES_MIN
    inj_b, comp_b = apply_filter(inj0, comp0, coh & (comp0.snr >= SNR_MIN), 'amplitude cut')
    inj, comp = apply_filter(inj0, comp0, coh & (comp0.dll_seq >= DLL_MIN), 'evidence cut')
    base = scores(inj_b, comp_b)

    inj.to_csv(os.path.join(RESULTS, 'gp_eval_injections.csv'), index=False)
    comp.to_csv(os.path.join(RESULTS, 'gp_eval_components.csv'), index=False)
    rows.to_csv(os.path.join(RESULTS, 'gp_eval_rows.csv'), index=False)

    groups = [('all rows', None), ('1-sector', rows.row[rows.n_sectors == 1]),
              ('3-sector', rows.row[rows.n_sectors == 3])] + \
             [(f'log age {a:g}', rows.row[rows.age == a]) for a in sorted(rows.age.unique())]
    summ = pd.DataFrame([dict(filter=lab, group=g, rows=len(rows) if r is None else len(r), **scores(i_, c_, r))
                         for lab, i_, c_ in ((f'dll_seq>={DLL_MIN:g}+cycles', inj, comp),
                                             (f'snr>={SNR_MIN:g}+cycles', inj_b, comp_b))
                         for g, r in groups])
    summ.to_csv(os.path.join(RESULTS, 'gp_eval_summary.csv'), index=False)
    print(summ[summ.group.isin(['all rows', '1-sector', '3-sector'])].round(3).to_string())
    print('\nmissed by reason:'); print(inj[~inj.recovered].reason.value_counts().to_string())
    k = comp[comp.kept]
    print('\nunmatched delivered components:'); print(k[k.kind != 'matched'].kind.value_counts().to_string())

    sp, ss = pop_scans(inj0, comp0), clean_set_scans()
    scan = pd.concat([d.assign(set='population', filter=lab) for lab, d in sp.items()] +
                     ([d.assign(set='clean sinusoids', filter=lab) for lab, d in ss.items()] if ss else []))
    scan.to_csv(os.path.join(RESULTS, 'gp_eval_filter_scan.csv'), index=False)

    stamp = (f'{os.path.basename(GP_OUT)} | truth {os.path.basename(TRUTH)} | detectable = predicted power '
             f'above the Vaughan 1% threshold | match 10% in period | generated {datetime.now():%Y-%m-%d %H:%M}')
    figure(inj, comp, rows, sp, ss, base, os.path.join(RESULTS, 'gp_eval.png'), stamp)


if __name__ == '__main__':
    main()

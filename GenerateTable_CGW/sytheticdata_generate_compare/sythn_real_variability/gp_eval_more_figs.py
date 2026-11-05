"""Further evaluation figures for the GP pipeline on the synthetic-population clusters.

Reads the tables written by ``gp_eval_figs`` (results/gp_eval_*.csv) and, for the worked
examples, the fitted table and the truth.  Run from ``GenerateTable_CGW/sytheticdata_generate_compare``:

    python -m sythn_real_variability.gp_eval_more_figs

Writes
    results/gp_eval_cuts.png      what each cut measures, and where matched / false components fall
    results/gp_eval_recovery.png  period and amplitude accuracy; completeness by period, age, sectors
    results/gp_eval_examples.png  four periodograms: clean recovery, a miss, a false positive, a young cluster

The three cuts on a fitted SHO component
----------------------------------------
cycles   Q / pi = decay time / period: how many oscillations the component persists before its
         amplitude falls by e.  >= 1.6 removes broadband components that are really continuum.
sigma/WN SHO amplitude sigma over the row's per-point white noise (photometric error and fitted
         jitter in quadrature).  The pipeline default keeps >= 0.2.
dlogL    ``gp_sho_dloglike_seq``: the log-likelihood the fit LOSES when the component is removed
         (sigma -> 0, everything else held fixed), by backward elimination -- remove the weakest
         component, re-evaluate the rest, repeat; each component keeps the drop it had when it
         was removed.  dlogL = 10 is a likelihood ratio e^10 ~ 2 x 10^4 in favour of keeping it,
         and about what BIC charges a 3-parameter component (1.5 ln N ~ 10.7 for these rows).
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

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.abspath(os.path.join(HERE, '..', '..'))
if GT not in sys.path:
    sys.path.insert(0, GT)

from sythn_real_variability import gp_eval_figs as E                       # noqa: E402
from sythn_real_variability.gp_eval_run import GP_OUT                       # noqa: E402

R = E.RESULTS
BLUE, ORANGE, NEUTRAL, INK, INK2 = E.BLUE, E.ORANGE, E.NEUTRAL, E.INK, E.INK2
NOISE = {'no detectable injection in row', 'no injected signal there'}


def load():
    comp = pd.read_csv(os.path.join(R, 'gp_eval_components.csv'))
    inj = pd.read_csv(os.path.join(R, 'gp_eval_injections.csv'))
    rows = pd.read_csv(os.path.join(R, 'gp_eval_rows.csv'))
    comp['cls'] = np.where(comp.kind == 'matched', 'matched',
                           np.where(comp.kind.isin(NOISE), 'no signal behind it', 'other unmatched'))
    return comp, inj, rows


CLS = [('matched', BLUE), ('no signal behind it', ORANGE), ('other unmatched', NEUTRAL)]


def pair_recovered(inj, comp):
    """For every recovered injection, the delivered component nearest in fractional period."""
    k = comp[comp.kept]
    by_row = {r: g for r, g in k.groupby('row')}
    out = []
    for _, q in inj[inj.recovered].iterrows():
        g = by_row[q.row]
        j = (np.abs(g.period / q.P - 1)).idxmin()
        c = g.loc[j]
        out.append(dict(row=q.row, age=q.age, n_sectors=q.n_sectors, P=q.P, a_eff=q.a_eff, snr_pred=q.snr,
                        P_fit=c.period, sigma=c.sigma, dll=c.dll_seq, cycles=c.cycles))
    p = pd.DataFrame(out)
    p['frac_err'] = p.P_fit / p.P - 1
    return p


def step_hist(ax, data, bins, color, label):
    ax.hist(data, bins=bins, histtype='step', lw=2, color=color, label=f'{label} ({len(data)})')


def fig_cuts(comp, inj, n_med, out, stamp):
    fig, axs = plt.subplots(2, 3, figsize=(20, 11.5), gridspec_kw=dict(wspace=.38, hspace=.5))
    t_bic = 1.5 * np.log(n_med)

    # (a) evidence ---------------------------------------------------------------------
    ax = axs[0, 0]
    bins = np.logspace(-2, 5, 50)
    for lab, c in CLS:
        step_hist(ax, comp.dll_seq[comp.cls == lab].clip(1e-2), bins, c, lab)
    ax.axvline(E.DLL_MIN, color=INK, ls='--', lw=1.3)
    ax.text(E.DLL_MIN * 1.15, ax.get_ylim()[1] * .6, f'cut ΔlogL ≥ {E.DLL_MIN:g}\n(BIC cost {t_bic:.1f})', fontsize=8.5)
    ax.set(xscale='log', xlabel='ΔlogL: log-likelihood lost if the component is removed', ylabel='fitted components')
    ax.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(1, 1.0))
    ax.set_title('(a) Evidence: does the fit need this component?', loc='left', fontweight='bold')

    # (b) amplitude ----------------------------------------------------------------------
    ax = axs[0, 1]
    bins = np.logspace(-3, 1.5, 50)
    for lab, c in CLS:
        step_hist(ax, comp.snr[comp.cls == lab].clip(1e-3), bins, c, lab)
    ax.axvline(E.SNR_MIN, color=INK, ls='--', lw=1.3)
    ax.text(E.SNR_MIN * 1.15, ax.get_ylim()[1] * .5, f'old cut\nσ/WN ≥ {E.SNR_MIN:g}', fontsize=8.5)
    med = comp.groupby('cls').snr.median()
    hi = comp[(comp.snr >= E.SNR_MIN) & (comp.cycles >= E.N_CYCLES_MIN)].groupby('cls').snr.median()
    ax.text(.03, .97, f'median σ/WN, all fitted: matched {med["matched"]:.2f}, no signal {med["no signal behind it"]:.2f}\n'
                      f'above the cut: matched {hi["matched"]:.2f}, no signal {hi["no signal behind it"]:.2f}',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK2,
            bbox=dict(facecolor='#fcfcfb', edgecolor='none', pad=2))
    ax.set(xscale='log', xlabel='σ / white noise per point', ylabel='fitted components')
    ax.legend(fontsize=8.5, loc='upper right', bbox_to_anchor=(1, .84))
    ax.set_title('(b) Amplitude: the two populations overlap', loc='left', fontweight='bold')

    # (c) coherence -------------------------------------------------------------------------
    ax = axs[0, 2]
    bins = np.logspace(-.5, 2.6, 45)
    for lab, c in CLS:
        step_hist(ax, comp.cycles[comp.cls == lab], bins, c, lab)
    ax.axvline(E.N_CYCLES_MIN, color=INK, ls='--', lw=1.3)
    ax.text(E.N_CYCLES_MIN * 1.1, ax.get_ylim()[1] * .92, f'cut cycles ≥ {E.N_CYCLES_MIN:g}', fontsize=8.5)
    ax.set(xscale='log', xlabel='cycles = Q / π  (decay time in periods)', ylabel='fitted components')
    ax.legend(fontsize=8.5, loc='upper right')
    ax.set_title('(c) Coherence: is it an oscillation at all?', loc='left', fontweight='bold')

    # (d) amplitude vs evidence ---------------------------------------------------------------
    ax = axs[1, 0]
    for lab, c in CLS[::-1]:
        m = comp.cls == lab
        ax.scatter(comp.snr[m].clip(1e-3), comp.dll_seq[m].clip(1e-2), s=14, alpha=.75, lw=0, color=c, label=lab)
    ax.axvline(E.SNR_MIN, color=INK2, ls=':', lw=1.2)
    ax.axhline(E.DLL_MIN, color=INK, ls='--', lw=1.3)
    ax.set(xscale='log', yscale='log', xlabel='σ / white noise', ylabel='ΔlogL')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('(d) The two axes disagree', loc='left', fontweight='bold')
    q = comp[comp.cycles >= E.N_CYCLES_MIN]
    a = (q.cls == 'matched') & (q.snr < E.SNR_MIN) & (q.dll_seq >= E.DLL_MIN)
    b = (q.cls == 'no signal behind it') & (q.snr >= E.SNR_MIN) & (q.dll_seq < E.DLL_MIN)
    ax.text(.03, .97, f'{int(a.sum())} matched with σ/WN < {E.SNR_MIN:g} but ΔlogL ≥ {E.DLL_MIN:g}\n'
                      f'{int(b.sum())} false with σ/WN ≥ {E.SNR_MIN:g} but ΔlogL < {E.DLL_MIN:g}',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK2)

    # (e) what each cut removes on its own ---------------------------------------------------------
    ax = axs[1, 1]
    cuts = [(f'cycles ≥ {E.N_CYCLES_MIN:g}', comp.cycles >= E.N_CYCLES_MIN),
            (f'σ/WN ≥ {E.SNR_MIN:g}', comp.snr >= E.SNR_MIN),
            (f'ΔlogL ≥ {E.DLL_MIN:g}', comp.dll_seq >= E.DLL_MIN)]
    y = np.arange(len(cuts))[::-1]
    h = .36
    for yi, (lab, keep) in zip(y, cuts):
        rm = ~keep
        n_m = int((rm & (comp.cls == 'matched')).sum())
        n_f = int((rm & (comp.cls == 'no signal behind it')).sum())
        ax.barh(yi + h / 2 + .02, n_m, height=h, color=BLUE)
        ax.barh(yi - h / 2 - .02, n_f, height=h, color=ORANGE)
        ax.text(n_m + 3, yi + h / 2 + .02, f'{n_m} of {int((comp.cls == "matched").sum())}', va='center', fontsize=8.5)
        ax.text(n_f + 3, yi - h / 2 - .02, f'{n_f} of {int((comp.cls == "no signal behind it").sum())}', va='center', fontsize=8.5)
    ax.set_yticks(y, [c[0] for c in cuts])
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
    ax.set(xlabel='components removed by this cut alone')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE, label='matched (lost)'),
                       plt.Rectangle((0, 0), 1, 1, color=ORANGE, label='no signal behind it (removed)')],
              fontsize=8.5, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2)
    ax.set_title('(e) What each cut removes on its own', loc='left', fontweight='bold', pad=24)

    # (f) evidence scales with signal strength ---------------------------------------------------------
    ax = axs[1, 2]
    p = pair_recovered(inj, comp[comp.kept | True].assign(kept=True))
    for ns, c, mk in ((1, BLUE, 'o'), (3, ORANGE, 's')):
        m = p.n_sectors == ns
        ax.scatter(p.snr_pred[m], p.dll[m].clip(1e-2), s=16, alpha=.7, lw=0, color=c, marker=mk, label=f'{ns}-sector rows')
    x = np.logspace(0, 4, 50)
    ax.axhline(E.DLL_MIN, color=INK, ls='--', lw=1.3)
    ax.set(xscale='log', yscale='log', xlabel='predicted significance of the injection (1 = at threshold)',
           ylabel='ΔlogL of its fitted component')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('(f) Evidence tracks how detectable the injection is', loc='left', fontweight='bold')

    fig.suptitle('The three cuts on a fitted component: coherence, amplitude, evidence (all fitted components, '
                 'synthetic-population set)', fontsize=14, fontweight='bold', x=.01, ha='left', y=.995)
    fig.text(.01, -.01, stamp, fontsize=7.5, color=INK2)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def fig_recovery(comp, inj, rows, out, stamp):
    fig, axs = plt.subplots(2, 3, figsize=(20, 11.5), gridspec_kw=dict(wspace=.38, hspace=.5))
    p = pair_recovered(inj, comp)

    # (a) period accuracy -----------------------------------------------------------------------
    ax = axs[0, 0]
    for ns, c, mk in ((1, BLUE, 'o'), (3, ORANGE, 's')):
        m = p.n_sectors == ns
        ax.scatter(p.P[m], p.P_fit[m], s=16, alpha=.75, lw=0, color=c, marker=mk, label=f'{ns}-sector rows ({int(m.sum())})')
    lim = (0.07, 13)
    ax.plot(lim, lim, color=INK2, lw=1)
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='injected period (d)', ylabel='GP period (d)')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('(a) Recovered periods', loc='left', fontweight='bold')

    # (b) fractional period error ------------------------------------------------------------------
    ax = axs[0, 1]
    bins = np.linspace(-.1, .1, 41)
    for ns, c in ((1, BLUE), (3, ORANGE)):
        e = p.frac_err[p.n_sectors == ns]
        ax.hist(e, bins=bins, histtype='step', lw=2, color=c,
                label=f'{ns}-sector: median |err| {np.median(np.abs(e)) * 100:.2f}%')
    ax.set(xlabel='(GP period − injected) / injected', ylabel='recovered injections')
    ax.legend(fontsize=8.5, loc='upper left')
    ax.set_title('(b) Period error', loc='left', fontweight='bold')

    # (c) amplitude accuracy --------------------------------------------------------------------------
    ax = axs[0, 2]
    amp = p.sigma * np.sqrt(2)
    for ns, c, mk in ((1, BLUE, 'o'), (3, ORANGE, 's')):
        m = p.n_sectors == ns
        ax.scatter(p.a_eff[m], amp[m], s=16, alpha=.75, lw=0, color=c, marker=mk, label=f'{ns}-sector rows')
    lim = (1e-5, 3e-2)
    ax.plot(lim, lim, color=INK2, lw=1)
    r = np.log10(amp / p.a_eff)
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=lim, xlabel='injected amplitude (fraction of flux)',
           ylabel='GP amplitude √2 σ')
    ax.text(.03, .97, f'median log ratio {np.median(r):+.2f} dex,\nscatter (IQR/1.35) {(np.subtract(*np.percentile(r, [75, 25]))) / 1.35:.2f} dex',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK2)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('(c) Recovered amplitudes', loc='left', fontweight='bold')

    # (d) completeness by injected period --------------------------------------------------------------
    ax = axs[1, 0]
    s = inj[inj.separable]
    edges = np.array([.08, .15, .3, .6, 1.2, 2.5, 5, 7.5, 10.5])
    xc = np.sqrt(edges[:-1] * edges[1:])
    for ns, c, mk in ((1, BLUE, 'o'), (3, ORANGE, 's')):
        q = s[s.n_sectors == ns]
        g = q.groupby(pd.cut(q.P, edges), observed=False).recovered.agg(['mean', 'size'])
        ok = (g['size'] >= 3).to_numpy()
        ax.plot(xc[ok], g['mean'].to_numpy()[ok], marker=mk, ms=6, lw=2, color=c, label=f'{ns}-sector rows')
        for x_, n_, v_ in zip(xc[ok], g['size'].to_numpy()[ok], g['mean'].to_numpy()[ok]):
            ax.text(x_, v_ + (.04 if ns == 1 else -.07), str(n_), ha='center', fontsize=7, color=c)
    ax.set(xscale='log', ylim=(0, 1.08), xlabel='injected period (d)', ylabel='completeness (separable injections)')
    ax.legend(fontsize=8.5, loc='lower left')
    ax.set_title('(d) Completeness by period (labels = injections)', loc='left', fontweight='bold')

    # (e) purity & completeness by age --------------------------------------------------------------------
    ax = axs[1, 1]
    k = comp[comp.kept]
    ages = sorted(rows.age.unique())
    pur = [float((k[k.age == a].kind == 'matched').mean()) if (k.age == a).any() else np.nan for a in ages]
    com = [float(s[s.age == a].recovered.mean()) if (s.age == a).any() else np.nan for a in ages]
    nd = [int((k.age == a).sum()) for a in ages]
    ax.plot(ages, pur, marker='o', lw=2, color=BLUE, label='purity')
    ax.plot(ages, com, marker='s', lw=2, color=ORANGE, label='completeness')
    for a, n in zip(ages, nd):
        ax.text(a, 1.06, str(n), ha='center', fontsize=7, color=INK2)
    ax.text(ages[0], 1.13, 'delivered components per age:', fontsize=7.5, color=INK2)
    ax.set(ylim=(-.02, 1.18), xlabel='log age', ylabel='fraction')
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('(e) By age', loc='left', fontweight='bold', pad=22)

    # (f) delivered components vs detectable injections per row ------------------------------------------
    ax = axs[1, 2]
    nk = k.groupby('row').size()
    rr = rows.assign(n_kept=rows.row.map(nk).fillna(0).astype(int))
    g = rr.groupby(['n_sectors', 'n_det']).n_kept.agg(['mean', 'size']).reset_index()
    for ns, c, mk in ((1, BLUE, 'o'), (3, ORANGE, 's')):
        q = g[(g.n_sectors == ns) & (g['size'] >= 3)]
        ax.plot(q.n_det, q['mean'], marker=mk, ms=6, lw=2, color=c, label=f'{ns}-sector rows')
    mx = int(g.n_det.max())
    ax.plot([0, mx], [0, mx], color=INK2, lw=1, label='one per detectable injection')
    z = rr[rr.n_det == 0]
    ax.set(xlabel='detectable injections in the row', ylabel='delivered components (mean over rows)')
    ax.text(.03, .97, f'{len(z)} rows with nothing detectable deliver\n{int(z.n_kept.sum())} components '
                      f'({(z.n_kept > 0).mean():.0%} of them deliver at least one)',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK2)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.set_title('(f) Components delivered per row', loc='left', fontweight='bold')

    fig.suptitle(f'Recovery quality with the recommended filter (ΔlogL ≥ {E.DLL_MIN:g}, cycles ≥ {E.N_CYCLES_MIN:g})',
                 fontsize=14, fontweight='bold', x=.01, ha='left', y=.995)
    fig.text(.01, -.01, stamp, fontsize=7.5, color=INK2)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def fig_examples(comp, inj, rows, out, stamp):
    df = pd.read_pickle(GP_OUT)
    truth = pd.read_pickle(E.TRUTH).set_index('name')
    k = comp[comp.kept]
    stats = rows.set_index('row')
    n_fp = k[k.cls == 'no signal behind it'].groupby('row').size()
    n_miss = inj[inj.reason == 'not seeded'].groupby('row').size()
    n_rec = inj[inj.recovered].groupby('row').size()
    good = stats[(stats.n_sectors == 3)].assign(rec=lambda d: d.index.map(n_rec).fillna(0),
                                                fp=lambda d: d.index.map(n_fp).fillna(0),
                                                cut=lambda d: d.index.map(k[k.cls != 'matched'].groupby('row').size()).fillna(0))
    good = good[(good.rec == good.n_sep) & (good.cut == 0)].sort_values('rec', ascending=False)
    picks = [('clean recovery', good.index[0]),
             ('a detectable injection the seeder missed', n_miss.sort_values(ascending=False).index[0]),
             ('a false positive: no signal behind it',
              k[k.cls == 'no signal behind it'].sort_values('dll_seq', ascending=False).row.iloc[0]),
             ('a young cluster: nothing to find', stats[(stats.age == 6.75) & (stats.n_sectors == 3)].index[0])]

    fig, axs = plt.subplots(2, 2, figsize=(19, 11), gridspec_kw=dict(wspace=.2, hspace=.42))
    for ax, (title, i) in zip(axs.flat, picks):
        r = df.loc[i]
        a, P = E.load_sinusoids(truth.loc[r['name'], 'star_params_path'])
        ev = E.evaluate_row(r, a, P)
        f = np.asarray(r['LSP_freq'], float); pw = np.asarray(r['LSP_power'], float)
        C = 10 ** float(r['rn_log10_N']) * f ** (-float(r['rn_alpha']))
        ax.plot(f, pw, color=INK2, lw=.6, alpha=.8, label='periodogram')
        ax.plot(f, ev['gam'] * C, color=INK, lw=1.2, ls='--', label='significance threshold')
        # injections: predicted power, detectable ones highlighted
        sub = ev['inb'] & (ev['snr'] > .05)
        pp = ev['snr'] * (ev['gam'] - 1) * np.interp(np.clip(ev['fi'], f[0], f[-1]), f, C)
        ax.scatter(ev['fi'][sub], pp[sub], s=10, color=NEUTRAL, lw=0, label='injection, predicted power')
        di = ev['det_idx']
        ax.scatter(ev['fi'][di], pp[di], s=40, color=BLUE, marker='D', lw=0, label='detectable injection')
        cr = comp[comp.row == i]
        top = np.nanmax(pw) * 3
        for _, c in cr.iterrows():
            mk = 'v'
            col = BLUE if c.cls == 'matched' else (ORANGE if c.cls == 'no signal behind it' else NEUTRAL)
            ax.plot(1 / c.period, top, marker=mk, ms=9, lw=0, color=col if c.kept else 'none',
                    mec=col, mew=1.5)
        ax.set(xscale='log', yscale='log', xlim=(f[0], f[-1]), ylim=(max(np.nanpercentile(pw, 1), 1e-2), top * 4),
               xlabel='frequency (1/d)', ylabel='periodogram power')
        ir = inj[inj.row == i]
        ax.set_title(f'{title}\n{r["name"]}, log age {r["age"]:g}, {int(r["n_sectors"])} sector(s): '
                     f'{len(di)} detectable, {int(ir.recovered.sum())} recovered, '
                     f'{int(cr.kept.sum())} delivered', loc='left', fontweight='bold', fontsize=11)
    h = [plt.Line2D([], [], color=INK2, lw=1, label='periodogram'),
         plt.Line2D([], [], color=INK, lw=1.2, ls='--', label='significance threshold'),
         plt.Line2D([], [], marker='o', ls='', color=NEUTRAL, label='injection (predicted power)'),
         plt.Line2D([], [], marker='D', ls='', color=BLUE, label='detectable injection'),
         plt.Line2D([], [], marker='v', ls='', color=BLUE, label='fitted component: matched'),
         plt.Line2D([], [], marker='v', ls='', color=ORANGE, label='fitted: no signal behind it'),
         plt.Line2D([], [], marker='v', ls='', color='none', mec=INK2, mew=1.5, label='hollow = removed by filter')]
    fig.legend(handles=h, loc='lower center', ncol=7, fontsize=9, bbox_to_anchor=(.5, -.02))
    fig.suptitle('Worked examples: what the pipeline sees in one row', fontsize=14, fontweight='bold', x=.01, ha='left', y=1.0)
    fig.text(.01, -.05, stamp, fontsize=7.5, color=INK2)
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def main():
    comp, inj, rows = load()
    df = pd.read_pickle(GP_OUT)
    n_med = float(np.median(df.LC_t.apply(len)))
    stamp = (f'{os.path.basename(GP_OUT)} | tables from gp_eval_figs | generated {datetime.now():%Y-%m-%d %H:%M}')
    fig_cuts(comp, inj, n_med, os.path.join(R, 'gp_eval_cuts.png'), stamp)
    fig_recovery(comp, inj, rows, os.path.join(R, 'gp_eval_recovery.png'), stamp)
    fig_examples(comp, inj, rows, os.path.join(R, 'gp_eval_examples.png'), stamp)


if __name__ == '__main__':
    main()

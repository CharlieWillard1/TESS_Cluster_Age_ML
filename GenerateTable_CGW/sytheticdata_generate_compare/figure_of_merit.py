"""Figure of merit for the injection-recovery pipeline: purity, completeness, and why.

One figure, three panels, for the DELIVERED component list -- the fit and every cut the
pipeline applies:

    seeding     prewhitened peak finder (``prewhiten_seeds=True``), Vaughan 1% threshold
    fit         multi-SHO GP with the one-sector decay-time cap
    filter      filter_gp_components defaults: sigma / white noise >= 0.2, cycles Q/pi >= 1.6

(a) Purity and completeness, overall and by sector count.
(b) Every missed detectable injection, classified by the first reason that applies.
(c) Every delivered false positive, classified by what it most likely is.

Definitions
-----------
purity        delivered components matching an injected period within 10%, / all delivered.
completeness  injections matched by a delivered component, / injections that the data can
              deliver: detectable in the periodogram (peak above the row's Vaughan 1%
              threshold), detectable ON THEIR OWN once the other injections are removed, and
              resolved -- at least one resolution element 1/T from every louder injection
              (sidelobe_removal_figs.isolated_detectability).  The looser denominator,
              everything that clears the threshold in the raw periodogram, is reported too;
              the misses it adds are the first two classes of panel (b).

Run:  python figure_of_merit.py [--table PATH] [--figdir DIR]
Writes figs/figure_of_merit.png, figure_of_merit_summary.csv, figure_of_merit_missed.csv,
figure_of_merit_false_positives.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import sidelobe_removal_figs as S                                # noqa: E402
from make_figs import TRUTH_PATH, SNR_MIN, N_CYCLES_MIN          # noqa: E402

TABLE = S.PREWHITE_TABLE
FIGDIR = os.path.join(HERE, 'figs')
ISO_CACHE = os.path.join(HERE, 'figs', 'sidelobe_removal', 'isolated_detectability.csv')
TOL = S.TOL

# Validated categorical pair (dataviz validate_palette.js, light surface): purity / completeness.
BLUE, ORANGE = '#2a78d6', '#eb6834'
NEUTRAL = '#b5b4ad'                    # "not separable from the data": not a pipeline outcome
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'

plt.rcParams.update({'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                     'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'legend.frameon': False,
                     'figure.dpi': 110, 'font.size': 10})

MISS_ORDER = [
    ('unresolved', 'unresolved from a louder injection\n(< 1 resolution element)', False),
    ('leakage', "detectable only through a louder\nneighbour's sidelobe leakage", False),
    ('not seeded', 'not seeded by the peak finder', True),
    ('fit moved', 'seeded, but the fit moved the\ncomponent > 10% in period', True),
    ('amplitude cut', f'fitted, removed by σ/WN ≥ {SNR_MIN:g}', True),
    ('coherence cut', f'fitted, removed by cycles ≥ {N_CYCLES_MIN:g}', True),
]
FP_LABELS = {'unresolved duplicate': 'duplicate of an injection\n(within 2 resolution elements)',
             'harmonic of injection': 'harmonic of an injection',
             'broadband (Q at floor)': 'broadband (Q at its floor)',
             'low amplitude': 'low amplitude',
             'unrelated frequency': 'unrelated frequency',
             'no injection in row': 'row has no injections'}


def final_keep(cp):
    return cp.coherent & np.isfinite(cp.snr) & (cp.snr >= SNR_MIN)


def classify_misses(df, cp, keep, iso):
    """One row per detectable injection the delivered list misses, with the first reason."""
    kept = {k: g['period'].to_numpy() for k, g in cp[keep].groupby('row')}
    out = []
    for _, q in iso.iterrows():
        P_kept = kept.get(q.row, np.array([]))
        if len(P_kept) and np.any(np.abs(P_kept / q.P - 1) < TOL):
            continue
        r = df.loc[q.row]
        seeds = S._arr(r, 'gp_initial_lsp_peak_periods')
        raw = cp[(cp.row == q.row) & (np.abs(cp.period / q.P - 1) < TOL)]
        if not q.resolved:
            why = 'unresolved'
        elif not q.iso_det:
            why = 'leakage'
        elif not (len(seeds) and np.any(np.abs(seeds / q.P - 1) < TOL)):
            why = 'not seeded'
        elif raw.empty:
            why = 'fit moved'
        elif (raw.snr >= SNR_MIN).any():
            why = 'coherence cut'
        else:
            why = 'amplitude cut'
        best = raw.loc[raw.snr.idxmax()] if len(raw) else None
        out.append(dict(row=q.row, name=r['name'], n_sectors=int(r['n_sectors']), P=q.P, a_wn=q.a_wn,
                        sep=q.sep, iso_ratio=q.iso_ratio, reason=why,
                        comp_period=np.nan if best is None else best.period,
                        comp_snr=np.nan if best is None else best.snr,
                        comp_cycles=np.nan if best is None else best.cycles))
    return pd.DataFrame(out)


def rates(cp, keep, iso, rows=None):
    """Purity, completeness on separable injections, completeness on all detectable."""
    if rows is not None:
        cp_m = cp.row.isin(rows).to_numpy()
        iso = iso[iso.row.isin(rows)]
    else:
        cp_m = np.ones(len(cp), bool)
    k = keep & cp_m
    kept = {r: g['period'].to_numpy() for r, g in cp[k].groupby('row')}
    hit = np.array([bool(len(kept.get(q.row, [])) and np.any(np.abs(kept[q.row] / q.P - 1) < TOL))
                    for _, q in iso.iterrows()])
    sep = iso.real.to_numpy()
    return dict(components=int(k.sum()), purity=float(cp.matched[k].mean()) if k.any() else np.nan,
                n_sep=int(sep.sum()), completeness=float(hit[sep].mean()) if sep.any() else np.nan,
                n_det=len(iso), completeness_all=float(hit.mean()) if len(hit) else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=TABLE)
    ap.add_argument('--figdir', default=FIGDIR)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    df = S.load(args.table)
    cp = S.components(df)
    det = S.detectable(df)
    iso = S.isolated_detectability(df, det, cache=ISO_CACHE)
    keep = final_keep(cp).to_numpy()

    # injection funnel
    n_inj = n_band = 0
    for _, r in df.iterrows():
        inj = S._arr(r, 'periods'); f = np.asarray(r['LSP_freq'], float)
        n_inj += len(inj); n_band += int(((1 / inj >= f[0]) & (1 / inj <= f[-1])).sum())

    # ---- numbers --------------------------------------------------------------------
    overall = rates(cp, keep, iso)
    groups = [('all rows', None)] + [(f'{ns} sector{"s" if ns > 1 else ""}',
                                      df.index[df.n_sectors == ns]) for ns in (1, 2, 3)]
    summ = pd.DataFrame([dict(group=g, n_rows=len(df) if rows is None else len(rows),
                              **rates(cp, keep, iso, rows)) for g, rows in groups])
    summ.to_csv(os.path.join(args.figdir, 'figure_of_merit_summary.csv'), index=False)

    miss = classify_misses(df, cp, keep, iso)
    miss.to_csv(os.path.join(args.figdir, 'figure_of_merit_missed.csv'), index=False)

    fp = cp[keep & ~cp.matched.to_numpy()].copy()
    fp['sep_res'] = fp.delta_f * fp.row.map(lambda i: S.C._baseline_days(df.loc[i]))
    fp[['row', 'name', 'n_sectors', 'period', 'snr', 'cycles', 'dll_seq', 'nearest_inj',
        'delta_f', 'sep_res', 'kind']].to_csv(
        os.path.join(args.figdir, 'figure_of_merit_false_positives.csv'), index=False)

    print(summ.round(3).to_string())
    print(f'injections {n_inj}, in band {n_band}, detectable {len(iso)}, separable {int(iso.real.sum())}')
    print('missed by reason:'); print(miss.reason.value_counts().to_string())
    print('false positives:'); print(fp[['name', 'period', 'snr', 'cycles', 'nearest_inj', 'sep_res', 'kind']].round(3).to_string())

    # ---- figure ---------------------------------------------------------------------
    fig = plt.figure(figsize=(21, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=.95)

    # (a) purity & completeness -------------------------------------------------------
    ax = fig.add_subplot(gs[0])
    lab = summ.group.tolist()
    y = np.arange(len(lab))[::-1]
    h = .36
    ax.barh(y + h / 2 + .02, summ.purity, height=h, color=BLUE, label='purity')
    ax.barh(y - h / 2 - .02, summ.completeness, height=h, color=ORANGE, label='completeness')
    for yi, (_, s) in zip(y, summ.iterrows()):
        ax.text(s.purity + .01, yi + h / 2 + .02, f'{s.purity:.3f}', va='center', fontsize=9, color=INK)
        ax.text(s.completeness + .01, yi - h / 2 - .02, f'{s.completeness:.3f}', va='center', fontsize=9, color=INK)
    ax.set_yticks(y, [f'{g}\n{int(s.n_rows)} rows, {int(s.components)} comps,\n{int(s.n_sep)} separable inj.'
                      for g, (_, s) in zip(lab, summ.iterrows())], fontsize=8.5)
    ax.set(xlim=(0, 1.13), xlabel='fraction')
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_title(f'(a) Purity {overall["purity"]:.3f}   Completeness {overall["completeness"]:.3f}',
                 loc='left', fontweight='bold', fontsize=12.5, pad=26)
    ax.text(0, -.12, f'Completeness counts the {overall["n_sep"]} injections the data can\n'
                     f'deliver: above threshold, detectable on their own, and\n'
                     f'resolved. On all {overall["n_det"]} above threshold in the raw\n'
                     f'periodogram it is {overall["completeness_all"]:.3f}.\n'
                     f'Funnel: {n_inj} injected, {n_band} in band,\n'
                     f'{overall["n_det"]} above threshold, {overall["n_sep"]} separable.',
            transform=ax.transAxes, fontsize=8.3, color=INK2, va='top')

    # (b) missed injections -----------------------------------------------------------
    ax = fig.add_subplot(gs[1])
    counts = miss.reason.value_counts()
    y = np.arange(len(MISS_ORDER))[::-1]
    for yi, (key, text, pipeline) in zip(y, MISS_ORDER):
        n = int(counts.get(key, 0))
        ax.barh(yi, n, height=.62, color=ORANGE if pipeline else NEUTRAL)
        ax.text(n + .6, yi, str(n), va='center', fontsize=9.5, color=INK)
    ax.set_yticks(y, [t for _, t, _ in MISS_ORDER], fontsize=8.5)
    ax.axhline(y[1] - .5, color=INK2, lw=.8, ls=(0, (3, 3)))
    xmax = max(counts.max() if len(counts) else 1, 1) * 1.9
    ax.set(xlim=(0, xmax), xlabel='missed detectable injections')
    ax.grid(axis='y', visible=False)
    n_pipe = int(sum(counts.get(k, 0) for k, _, p in MISS_ORDER if p))
    n_sep = int(sum(counts.get(k, 0) for k, _, p in MISS_ORDER if not p))
    ax.text(xmax * .98, (y[0] + y[1]) / 2, 'not separable\nfrom the data:\noutside the\ncompleteness\ndenominator',
            ha='right', va='center', fontsize=8, color=INK2)
    ax.text(xmax * .98, (y[2] + y[3]) / 2, f'pipeline misses:\n{n_pipe} of the {overall["n_sep"]}\nseparable injections',
            ha='right', va='center', fontsize=8, color=INK2)
    ax.set_title(f'(b) Why {len(miss)} detectable injections are missed', loc='left',
                 fontweight='bold', fontsize=12.5, pad=26)
    ax.text(0, -.12, f'First reason that applies, top to bottom.\n'
                     f'{n_sep} cannot be separated by any method that does\n'
                     f'not already know the true frequencies. {n_pipe} are losses\n'
                     f'the pipeline could in principle recover.',
            transform=ax.transAxes, fontsize=8.3, color=INK2, va='top')

    # (c) false positives -------------------------------------------------------------
    ax = fig.add_subplot(gs[2])
    kinds = [k for k in S.C.FP_ORDER if k != 'matched']
    fc = fp.kind.astype(str).value_counts()
    y = np.arange(len(kinds))[::-1]
    for yi, k in zip(y, kinds):
        n = int(fc.get(k, 0))
        ax.barh(yi, n, height=.62, color=BLUE)
        ax.text(n + .05, yi, str(n), va='center', fontsize=9.5, color=INK)
    ax.set_yticks(y, [FP_LABELS[k] for k in kinds], fontsize=8.5)
    xmax = max(int(fc.max()) if len(fc) else 1, 1) * 1.6 + 1
    ax.set(xlim=(0, xmax), xlabel='delivered components matching no injection')
    ax.set_xticks(range(0, int(xmax) + 1, max(1, int(xmax) // 5)))
    ax.grid(axis='y', visible=False)
    ax.set_title(f'(c) Why {len(fp)} of {overall["components"]} delivered components are false',
                 loc='left', fontweight='bold', fontsize=12.5, pad=26)
    lines = [f'{r["name"]} ({int(r.n_sectors)} sect.): P = {r.period:.3f} d, σ/WN {r.snr:.2f},\n'
             f'    {r.cycles:.0f} cycles; nearest injection {r.nearest_inj:.3f} d, {r.sep_res:.1f}/T away'
             if np.isfinite(r.nearest_inj) else
             f'{r["name"]} ({int(r.n_sectors)} sect.): P = {r.period:.3f} d, σ/WN {r.snr:.2f}, {r.cycles:.0f} cycles'
             for _, r in fp.sort_values('kind').iterrows()]
    ax.text(0, -.12, 'Each one:\n' + '\n'.join(lines[:8]) + ('\n…' if len(lines) > 8 else ''),
            transform=ax.transAxes, fontsize=8.0, color=INK2, va='top')

    fig.suptitle('Injection recovery, end to end: prewhitened seeds → GP fit (decay-time cap) → '
                 f'σ/WN ≥ {SNR_MIN:g} and cycles ≥ {N_CYCLES_MIN:g}',
                 fontsize=14, fontweight='bold', x=.01, ha='left', y=1.02)
    fig.text(.01, -.2, f'{os.path.basename(args.table)} | truth {os.path.basename(TRUTH_PATH)} | '
                        f'match tolerance {TOL:.0%} in period | generated {datetime.now():%Y-%m-%d %H:%M}',
             fontsize=7.5, color=INK2)
    out = os.path.join(args.figdir, 'figure_of_merit.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


if __name__ == '__main__':
    main()

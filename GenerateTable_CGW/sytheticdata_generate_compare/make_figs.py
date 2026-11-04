"""Regenerate every synthetic-set figure from the CURRENT pipeline table.

Why this file exists
--------------------
The figures in ``figs/`` used to be saved by hand out of a notebook cell, so nothing tied
an image to the table it came from, and the cell that produced the six-panel evaluation
figure was later edited away -- leaving a PNG that could not be reproduced.  Everything
here is driven by the constants at the top and writes its provenance into the figure
itself, so a stale image is visible at a glance.

Run:  python make_figs.py [--table PATH] [--no-refilter]
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gp_pipeline.gp_table_io import filter_gp_components          # noqa: E402
from sytheticdata_generate_compare.synth import compare as C      # noqa: E402

PIPELINE_TABLE = os.path.join(ROOT, 'data',
                              'cluster_table_Sep17_3sectorcombos_synthetic_expavg_gp_current.pkl')
TRUTH_PATH = os.path.join(HERE, 'data', 'ground_truth.pkl')
FIGDIR = os.path.join(HERE, 'figs')

SIGMA_REL_MIN, Q_MIN = 0.10, 5.0
WN_FLAG_RATIO = 0.05

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 110})

CLEAN, CAPPED, NORN = '#2a78d6', '#eb6834', '#8f8e88'


def _flag(v):
    return pd.to_numeric(pd.Series(v), errors='coerce').fillna(0).astype(bool).to_numpy()


def row_classes(df):
    """clean / components capped / no red noise detected -- the three row populations."""
    capped = _flag(df.get('gp_components_capped', False))
    norn = _flag(df.get('gp_rn_near_white_noise', False))
    clean = ~capped & ~norn
    return [('clean', clean, CLEAN, 'o'),
            ('components capped', capped & ~norn, CAPPED, 's'),
            ('no red noise detected', norn, NORN, 'v')]


def _stats_box(ax, x, y, label='clean rows', log=False):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return
    if log:
        r = y[m] / x[m]
        txt = (f'{label} (n={m.sum()}):\nmedian ratio {np.median(r):.2f}\n'
               f'p16-p84 {np.percentile(r, 16):.2f}-{np.percentile(r, 84):.2f}\n'
               f'Spearman {pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"):.2f}')
    else:
        d = y[m] - x[m]
        txt = (f'{label} (n={m.sum()}):\nmedian bias {np.median(d):+.2f}\n'
               f'median |error| {np.median(np.abs(d)):.2f}\n'
               f'r = {np.corrcoef(x[m], y[m])[0, 1]:.2f}')
    ax.text(.03, .97, txt, transform=ax.transAxes, va='top', ha='left', fontsize=8.5,
            color='#3d3d39')


def figure_evaluation(df, osc, comps_total, comps_spurious, scan, stamp):
    """The six-panel headline: red-noise recovery (1-5) and oscillator recovery (6)."""
    fig, axs = plt.subplots(2, 3, figsize=(19.5, 11.5))
    classes = row_classes(df)

    inj_a = pd.to_numeric(df['inj_rn_alpha'], errors='coerce').to_numpy()
    clean_mask = classes[0][1]

    # 1 / 2 -- slope from the periodogram and from the GP kernel
    for ax, col, title in [(axs[0, 0], 'rn_alpha', '1 · Red-noise slope — periodogram (masked Vaughan)'),
                           (axs[0, 1], 'gp_rn_alpha_band', '2 · Red-noise slope — GP kernel')]:
        fit = pd.to_numeric(df[col], errors='coerce').to_numpy()
        for lab, m, c, mk in classes:
            ax.scatter(inj_a[m], fit[m], s=26, alpha=.75, lw=0, color=c, marker=mk)
        ax.plot([0, 2.5], [0, 2.5], '--', color='#5f5e5a', lw=1)
        ax.set(xlabel='injected α', ylabel='fitted α', xlim=(-.1, 2.5), ylim=(-.15, 2.6))
        ax.set_title(title, loc='left', fontweight='bold')
        _stats_box(ax, inj_a[clean_mask], fit[clean_mask])
    axs[0, 1].annotate('prior sets α→0 where\nno red noise is detected',
                       xy=(1.55, 0.06), xytext=(1.55, 0.55), fontsize=8, style='italic',
                       color='#5f5e5a', ha='center',
                       arrowprops=dict(arrowstyle='->', color='#5f5e5a', lw=.9))

    # 3 -- amplitude
    ax = axs[0, 2]
    inj_r = pd.to_numeric(df['rn_rms_realized'], errors='coerce').to_numpy()
    fit_r = pd.to_numeric(df['gp_rn_rms_band'], errors='coerce').to_numpy()
    for lab, m, c, mk in classes:
        ax.scatter(inj_r[m], fit_r[m], s=26, alpha=.75, lw=0, color=c, marker=mk)
    lim = [np.nanmin(inj_r) * .6, np.nanmax(inj_r) * 1.6]
    ax.plot(lim, lim, '--', color='#5f5e5a', lw=1)
    # flagged rows have gp_rn_rms_band ~ 1e-9; letting them set the range squashes the rest
    ylo = np.nanpercentile(fit_r[clean_mask], 0.5) * .5
    ax.set(xscale='log', yscale='log', xlim=lim, ylim=(ylo, np.nanmax(fit_r) * 1.6),
           xlabel='injected red-noise RMS (full baseline)', ylabel='GP red-noise RMS (in band)')
    ax.set_title('3 · Red-noise amplitude — GP kernel', loc='left', fontweight='bold')
    _stats_box(ax, inj_r[clean_mask], fit_r[clean_mask], log=True)
    ax.text(.97, .03, 'below 1:1 by construction:\nin-band vs full-baseline power',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, style='italic',
            color='#5f5e5a')

    # 4 -- was there red noise to measure at all
    ax = axs[1, 0]
    snr = inj_r / pd.to_numeric(df['err_frac_med'], errors='coerce').to_numpy()
    wn = pd.to_numeric(df['gp_rn_wn_ratio'], errors='coerce').to_numpy()
    for lab, m, c, mk in classes:
        ax.scatter(snr[m], wn[m], s=26, alpha=.75, lw=0, color=c, marker=mk)
    ax.axhline(WN_FLAG_RATIO, color=CAPPED, ls='--', lw=1.2)
    good = np.isfinite(snr) & np.isfinite(wn)
    ax.set(xscale='log', yscale='log', xlabel='injected red-noise RMS / photometric error',
           ylabel='fitted red-noise / white-noise power at 1 d⁻¹')
    # x in axes coords, y in data coords, so the label tracks the line without ever
    # touching the data limits.  Reading get_xlim() before set(xscale='log') would anchor
    # this at x=0, which the log axis then expands around -- a ~10^5 px wide canvas.
    from matplotlib.transforms import blended_transform_factory
    ax.text(.02, WN_FLAG_RATIO * 1.35, f'flag threshold  RN/WN = {WN_FLAG_RATIO}',
            transform=blended_transform_factory(ax.transAxes, ax.transData),
            fontsize=8, color=CAPPED, va='bottom')
    ax.set_title('4 · Is there red noise to measure?', loc='left', fontweight='bold')
    ax.text(.03, .97, f'Spearman {pd.Series(snr[good]).corr(pd.Series(wn[good]), method="spearman"):.2f}\n'
                      f'flagged: {int(classes[2][1].sum())} of {len(df)} rows',
            transform=ax.transAxes, va='top', fontsize=8.5, color='#3d3d39')

    # 5 -- alpha error against how much red noise there was
    ax = axs[1, 1]
    err = np.abs(pd.to_numeric(df['gp_rn_alpha_band'], errors='coerce').to_numpy() - inj_a)
    for lab, m, c, mk in classes:
        ax.scatter(wn[m], err[m], s=26, alpha=.75, lw=0, color=c, marker=mk)
    ok = np.isfinite(wn) & np.isfinite(err) & (wn > 0)
    if ok.sum() > 20:
        edges = np.logspace(np.log10(np.nanmin(wn[ok])), np.log10(np.nanmax(wn[ok])), 9)
        ctr, med = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = ok & (wn >= lo) & (wn < hi)
            if b.sum() >= 4:
                ctr.append(np.sqrt(lo * hi)); med.append(np.median(err[b]))
        ax.plot(ctr, med, 'D-', color='black', ms=6, lw=1.6, label='median |error|')
        ax.legend(fontsize=8, loc='lower left')
    ax.axvline(WN_FLAG_RATIO, color=CAPPED, ls='--', lw=1.2)
    ax.set(xscale='log', yscale='log', xlabel='fitted red-noise / white-noise power at 1 d⁻¹',
           ylabel='|fitted − injected α|')
    ax.set_title('5 · α is only measurable above the white noise', loc='left', fontweight='bold')

    # 6 -- oscillator recovery
    ax = axs[1, 2]
    det = osc[osc.signif]
    n_rec_det = int(det.recovered.sum())
    n_miss_det = int((~det.recovered).sum())
    n_rec_below = int(osc[~osc.signif].recovered.sum())
    bars = [('recovered\n(detectable)', n_rec_det), ('missed\n(detectable)', n_miss_det),
            ('recovered\n(below\nthreshold)', n_rec_below), ('spurious\ncomponents', comps_spurious)]
    ax.bar([b[0] for b in bars], [b[1] for b in bars], color=CLEAN, width=.62)
    for i, (_, v) in enumerate(bars):
        ax.text(i, v, f'{v}', ha='center', va='bottom', fontsize=10)
    ax.set(ylabel='injected oscillators / fitted components')
    ax.set_title('6 · Oscillator recovery', loc='left', fontweight='bold')
    after = scan[(scan.sigma_rel_cut == SIGMA_REL_MIN) & (scan.Q_cut == Q_MIN)]
    extra = ''
    if len(after):
        a = after.iloc[0]
        extra = (f'after filter (σ≥{SIGMA_REL_MIN:.0%}, Q≥{Q_MIN:g}): '
                 f'purity {a.purity:.1%}, completeness {a.completeness:.1%}\n')
    ax.text(.03, .88,
            f'detectable recovery {n_rec_det / max(len(det), 1):.1%}  ({n_rec_det}/{len(det)})\n'
            f'component purity {1 - comps_spurious / max(comps_total, 1):.1%}  ({comps_total} components)\n'
            f'{extra}{len(osc)} injected oscillators, {len(df)} rows',
            transform=ax.transAxes, va='top', fontsize=8.5, color='#3d3d39')

    handles = [plt.Line2D([], [], ls='', marker=mk, color=c, label=f'{lab}  (n={int(m.sum())})')
               for lab, m, c, mk in classes]
    handles.append(plt.Line2D([], [], ls='--', color='#5f5e5a', label='1:1'))
    fig.legend(handles=handles, loc='upper right', ncol=4, fontsize=10,
               bbox_to_anchor=(.995, 1.005))
    fig.suptitle('Pipeline evaluation on the synthetic set — exposure-averaged sampling, '
                 'masked continuum, power-law GP red noise', fontsize=13, x=.006, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color='#8f8e88')
    plt.tight_layout(rect=(0, .01, 1, .965))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=PIPELINE_TABLE)
    ap.add_argument('--no-refilter', action='store_true',
                    help='skip filter_gp_components (use the columns already in the table)')
    args = ap.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)
    pipe = pd.read_pickle(args.table)
    truth = pd.read_pickle(TRUTH_PATH)
    print(f'pipeline table : {os.path.basename(args.table)}  ({len(pipe)} rows)')

    if not args.no_refilter or 'gp_component_kept' not in pipe.columns:
        pipe = filter_gp_components(pipe, sigma_rel_min=SIGMA_REL_MIN, Q_min=Q_MIN)
        pipe.to_pickle(args.table)
        print(f'applied filter_gp_components(sigma_rel_min={SIGMA_REL_MIN}, Q_min={Q_MIN}) '
              f'and re-saved the table')

    df = C.join(pipe, truth)
    df = C.add_ratio_period(df)
    stamp = (f'{os.path.basename(args.table)} | truth {os.path.basename(TRUTH_PATH)} | '
             f'filter σ_rel≥{SIGMA_REL_MIN:g}, Q≥{Q_MIN:g} | generated {datetime.now():%Y-%m-%d %H:%M}')

    osc = C.per_oscillator_table(df)
    qs = C.gp_component_purity(df)
    scan = C.component_filter_scan(
        df, cuts=((0, 0), (0.05, 0), (SIGMA_REL_MIN, 0), (0, Q_MIN),
                  (SIGMA_REL_MIN, Q_MIN), (SIGMA_REL_MIN, 20), (SIGMA_REL_MIN, 100)))
    fp = C.false_positive_table(df, sigma_rel_min=SIGMA_REL_MIN, Q_min=Q_MIN)
    fp_null = C.false_positive_null(df, n_draws=20, sigma_rel_min=SIGMA_REL_MIN)

    fig = figure_evaluation(df, osc, len(qs), int((~qs.matches_injected).sum()), scan, stamp)
    fig.savefig(os.path.join(FIGDIR, 'synthetic_evaluation.png'), bbox_inches='tight')
    plt.close(fig)
    print('wrote figs/synthetic_evaluation.png')

    fig = C.plot_false_positives(fp, null=fp_null, sigma_rel_min=SIGMA_REL_MIN, Q_min=Q_MIN)
    fig.text(.006, .002, stamp, fontsize=7.5, color='#8f8e88')
    fig.savefig(os.path.join(FIGDIR, 'false_positives.png'), bbox_inches='tight')
    plt.close(fig)
    print('wrote figs/false_positives.png')

    fig = C.plot_q_purity(qs)
    fig.savefig(os.path.join(FIGDIR, 'q_purity.png'), bbox_inches='tight')
    plt.close(fig)
    fig = C.plot_recovery_vs_floor(osc)
    fig.savefig(os.path.join(FIGDIR, 'diag_recovery_vs_floor.png'), bbox_inches='tight')
    plt.close(fig)
    fig = C.plot_period_amplitude(osc)
    fig.savefig(os.path.join(FIGDIR, 'diag_period_amplitude.png'), bbox_inches='tight')
    plt.close(fig)
    fig = C.plot_period_bias(osc)
    fig.savefig(os.path.join(FIGDIR, 'diag_period_bias_and_Q.png'), bbox_inches='tight')
    plt.close(fig)
    print('wrote figs/q_purity.png, diag_recovery_vs_floor.png, diag_period_amplitude.png, '
          'diag_period_bias_and_Q.png')

    summ = C.false_positive_summary(fp, fp_null)
    summ.to_csv(os.path.join(FIGDIR, 'false_positive_summary.csv'), index=False)
    scan.to_csv(os.path.join(FIGDIR, 'component_filter_scan.csv'), index=False)
    print('\n' + summ.to_string(index=False))
    return fp, summ, scan


if __name__ == '__main__':
    main()

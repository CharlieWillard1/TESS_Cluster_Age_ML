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
                              'cluster_table_Nov03_3sectorcombos_synthetic_expavg_gp_taucap.pkl')
# The same base table fitted before the decay-time cap (flat Q box [1/sqrt2, 500], soft
# floor at Q=2).  Only used for the before/after figure; pass --old-table '' to skip it.
OLD_TABLE = os.path.join(ROOT, 'data',
                         'cluster_table_Sep17_3sectorcombos_synthetic_expavg_gp_current.pkl')
TRUTH_PATH = os.path.join(HERE, 'data', 'ground_truth.pkl')
FIGDIR = os.path.join(HERE, 'figs')

# The delivered-list filter (gp_table_io.filter_gp_components): amplitude >= 10% of the
# row's strongest component and a decay time of >= 1.6 periods (Q / pi >= 1.6, i.e. the
# old Q >= 5).  The cut is in cycles because the fit now caps Q at pi * 27 d / P, so a
# flat Q cut would mean something different at every period (docs/gp_fitting.tex S6).
SIGMA_REL_MIN, N_CYCLES_MIN = 0.10, 1.6
Q_MIN = N_CYCLES_MIN * np.pi
TAU_CAP_DAYS = 27.0
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
    after = scan[(scan.sigma_rel_cut == SIGMA_REL_MIN) & (scan.cycles_cut == N_CYCLES_MIN)]
    extra = ''
    if len(after):
        a = after.iloc[0]
        extra = (f'after filter (σ≥{SIGMA_REL_MIN:.0%}, Q/π≥{N_CYCLES_MIN:g}): '
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
                 'masked continuum, power-law GP red noise, SHO decay time capped at one sector',
                 fontsize=13, x=.006, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color='#8f8e88')
    plt.tight_layout(rect=(0, .01, 1, .965))
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=PIPELINE_TABLE)
    ap.add_argument('--old-table', default=OLD_TABLE,
                    help="pre-cap fit of the same base table for the before/after figure; '' to skip")
    ap.add_argument('--figdir', default=FIGDIR)
    ap.add_argument('--no-refilter', action='store_true',
                    help='skip filter_gp_components (use the columns already in the table)')
    args = ap.parse_args()
    figdir = args.figdir

    os.makedirs(figdir, exist_ok=True)
    pipe = pd.read_pickle(args.table)
    truth = pd.read_pickle(TRUTH_PATH)
    print(f'pipeline table : {os.path.basename(args.table)}  ({len(pipe)} rows)')

    if not args.no_refilter or 'gp_component_kept' not in pipe.columns:
        pipe = filter_gp_components(pipe, sigma_rel_min=SIGMA_REL_MIN, n_cycles_min=N_CYCLES_MIN)
        pipe.to_pickle(args.table)
        print(f'applied filter_gp_components(sigma_rel_min={SIGMA_REL_MIN}, '
              f'n_cycles_min={N_CYCLES_MIN}) and re-saved the table')

    df = C.join(pipe, truth)
    df = C.add_ratio_period(df)
    stamp = (f'{os.path.basename(args.table)} | truth {os.path.basename(TRUTH_PATH)} | '
             f'filter σ_rel≥{SIGMA_REL_MIN:g}, Q/π≥{N_CYCLES_MIN:g} | '
             f'generated {datetime.now():%Y-%m-%d %H:%M}')

    osc, qs, scan = analyse(df)
    fp = C.false_positive_table(df, sigma_rel_min=SIGMA_REL_MIN, n_cycles_min=N_CYCLES_MIN)
    fp_null = C.false_positive_null(df, n_draws=20, sigma_rel_min=SIGMA_REL_MIN)

    fig = figure_evaluation(df, osc, len(qs), int((~qs.matches_injected).sum()), scan, stamp)
    fig.savefig(os.path.join(figdir, 'synthetic_evaluation.png'), bbox_inches='tight')
    plt.close(fig)
    print('wrote synthetic_evaluation.png')

    fig = C.plot_false_positives(fp, null=fp_null, sigma_rel_min=SIGMA_REL_MIN,
                                 n_cycles_min=N_CYCLES_MIN)
    fig.text(.006, .002, stamp, fontsize=7.5, color='#8f8e88')
    fig.savefig(os.path.join(figdir, 'false_positives.png'), bbox_inches='tight')
    plt.close(fig)
    print('wrote false_positives.png')

    fig = C.plot_q_purity(qs)
    fig.savefig(os.path.join(figdir, 'q_purity.png'), bbox_inches='tight')
    plt.close(fig)
    if qs['coherence'].notna().any():
        fig = C.plot_coherence_purity(qs)
        fig.text(.006, .002, stamp, fontsize=7.5, color='#8f8e88')
        fig.savefig(os.path.join(figdir, 'coherence_purity.png'), bbox_inches='tight')
        plt.close(fig)
        print('wrote coherence_purity.png')
    fig = C.plot_recovery_vs_floor(osc)
    fig.savefig(os.path.join(figdir, 'diag_recovery_vs_floor.png'), bbox_inches='tight')
    plt.close(fig)
    fig = C.plot_period_amplitude(osc)
    fig.savefig(os.path.join(figdir, 'diag_period_amplitude.png'), bbox_inches='tight')
    plt.close(fig)
    fig = C.plot_period_bias(osc)
    fig.savefig(os.path.join(figdir, 'diag_period_bias_and_Q.png'), bbox_inches='tight')
    plt.close(fig)
    print('wrote q_purity.png, diag_recovery_vs_floor.png, diag_period_amplitude.png, '
          'diag_period_bias_and_Q.png')

    summ = C.false_positive_summary(fp, fp_null)
    summ.to_csv(os.path.join(figdir, 'false_positive_summary.csv'), index=False)
    scan.to_csv(os.path.join(figdir, 'component_filter_scan.csv'), index=False)
    print('\n' + summ.to_string(index=False))

    heads = [headline(df, osc, qs, scan, fp, 'tau_d cap (27 d), soft floor Q=pi')]
    if args.old_table:
        old = pd.read_pickle(args.old_table)
        if 'gp_component_kept' not in old.columns:
            old = filter_gp_components(old, sigma_rel_min=SIGMA_REL_MIN, n_cycles_min=N_CYCLES_MIN)
        old_df = C.add_ratio_period(C.join(old, truth))
        old_osc, old_qs, old_scan = analyse(old_df)
        old_fp = C.false_positive_table(old_df, sigma_rel_min=SIGMA_REL_MIN, n_cycles_min=N_CYCLES_MIN)
        heads.insert(0, headline(old_df, old_osc, old_qs, old_scan, old_fp,
                                 'flat Q box [1/sqrt2, 500], soft floor Q=2'))
        fig = figure_q_bounds_change(old_df, df, old_osc, osc, old_qs, qs, stamp)
        fig.savefig(os.path.join(figdir, 'q_bounds_before_after.png'), bbox_inches='tight')
        plt.close(fig)
        print('wrote q_bounds_before_after.png')
    heads = pd.DataFrame(heads).set_index('config').T
    heads.to_csv(os.path.join(figdir, 'headline_before_after.csv'))
    print('\n' + heads.to_string())
    return fp, summ, scan


def analyse(df):
    """The three per-table products every figure is built from."""
    osc = C.per_oscillator_table(df)
    qs = C.gp_component_purity(df)
    scan = C.component_filter_scan(
        df, by='cycles',
        cuts=((0, 0), (0.05, 0), (SIGMA_REL_MIN, 0), (0, N_CYCLES_MIN),
              (SIGMA_REL_MIN, N_CYCLES_MIN), (SIGMA_REL_MIN, 20 / np.pi), (SIGMA_REL_MIN, 100 / np.pi)))
    return osc, qs, scan


def headline(df, osc, qs, scan, fp, label):
    """The numbers quoted in docs/gp_fitting.tex S6 and S9, for one table."""
    classes = row_classes(df)
    clean = classes[0][1]
    inj_a = pd.to_numeric(df['inj_rn_alpha'], errors='coerce').to_numpy()
    has_rn = ~classes[2][1]
    out = {'config': label, 'rows': len(df), 'rows clean (neither flag)': int(clean.sum()),
           'rows components capped': int(classes[1][1].sum()),
           'rows no red noise detected': int(classes[2][1].sum()),
           'rows fitted with no components': int((pd.to_numeric(df['gp_n_components'], errors='coerce') == 0).sum())}
    for col, name in [('gp_rn_alpha_band', 'alpha GP'), ('rn_alpha', 'alpha periodogram')]:
        fit = pd.to_numeric(df[col], errors='coerce').to_numpy()
        d = (fit - inj_a)[clean]; d = d[np.isfinite(d)]
        out[f'{name}: bias (clean)'] = round(float(np.median(d)), 3)
        out[f'{name}: median |err| (clean)'] = round(float(np.median(np.abs(d))), 3)
        m = clean & np.isfinite(fit) & np.isfinite(inj_a)
        out[f'{name}: r (clean)'] = round(float(np.corrcoef(inj_a[m], fit[m])[0, 1]), 3)
    d = (pd.to_numeric(df['gp_rn_alpha_band'], errors='coerce').to_numpy() - inj_a)[has_rn]
    out['alpha GP: median |err| (rows with red noise)'] = round(float(np.nanmedian(np.abs(d))), 3)
    inj_r = pd.to_numeric(df['rn_rms_realized'], errors='coerce').to_numpy()
    fit_r = pd.to_numeric(df['gp_rn_rms_band'], errors='coerce').to_numpy()
    m = clean & np.isfinite(inj_r) & np.isfinite(fit_r)
    out['rn amplitude Spearman (clean)'] = round(float(pd.Series(inj_r[m]).corr(pd.Series(fit_r[m]), method='spearman')), 3)
    det = osc[osc.signif]
    out['detectable oscillators'] = int(len(det))
    out['detectable recovered'] = int(det.recovered.sum())
    out['detectable recovery'] = round(float(det.recovered.mean()), 4)
    out['components fitted'] = int(len(qs))
    out['purity raw'] = round(float(qs.matches_injected.mean()), 4)
    a = scan[(scan.sigma_rel_cut == SIGMA_REL_MIN) & (scan.cycles_cut == N_CYCLES_MIN)].iloc[0]
    out['components after filter'] = int(a.components)
    out['purity filtered'] = round(float(a.purity), 4)
    out['completeness filtered'] = round(float(a.completeness), 4)
    out['components with Q >= 475'] = int((qs.Q >= 475).sum())
    out['components at cap (s > 0.99)'] = int((qs.coherence > 0.99).sum()) if qs.coherence.notna().any() else np.nan
    out['components Q in 2-20'] = round(float(((qs.Q >= 2) & (qs.Q < 20)).mean()), 4)
    out['median Q matched'] = round(float(qs.loc[qs.matches_injected, 'Q'].median()), 1)
    out['median Q spurious'] = round(float(qs.loc[~qs.matches_injected, 'Q'].median()), 1)
    out['median cycles matched'] = round(float(qs.loc[qs.matches_injected, 'cycles'].median()), 2)
    out['median cycles spurious'] = round(float(qs.loc[~qs.matches_injected, 'cycles'].median()), 2)
    kept = fp[fp.survives_filter]
    out['delivered: genuinely unrelated'] = int((kept.kind == 'unrelated frequency').sum())
    return out


def figure_q_bounds_change(old_df, new_df, old_osc, new_osc, old_qs, new_qs, stamp):
    """Before/after the decay-time cap: where components sit, and what the cut costs."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 11))
    P = np.logspace(np.log10(0.08), 1, 200)

    # (a) Q against period: the flat box and the period-dependent cap
    ax = axs[0, 0]
    ax.scatter(old_qs.period, old_qs.Q, s=9, color='#b8b6ae', alpha=.5, lw=0, label='before (flat Q box)')
    ax.scatter(new_qs.loc[~new_qs.matches_injected, 'period'], new_qs.loc[~new_qs.matches_injected, 'Q'],
               s=13, color='crimson', alpha=.7, lw=0, label='after: spurious')
    ax.scatter(new_qs.loc[new_qs.matches_injected, 'period'], new_qs.loc[new_qs.matches_injected, 'Q'],
               s=13, color='seagreen', alpha=.7, lw=0, label='after: matches injection')
    ax.plot(P, np.pi * TAU_CAP_DAYS / P, '--', color='black', lw=1.3, label=f'cap π·{TAU_CAP_DAYS:g} d / P')
    ax.axhline(500, color='#5f5e5a', ls=':', lw=1.1, label='old cap Q = 500')
    ax.axhline(np.pi, color='#5f5e5a', ls='-.', lw=.9, label='soft floor Q = π (1 period)')
    ax.axhline(1 / np.sqrt(2), color='#b0a24a', ls=':', lw=1, label='hard floor 1/√2')
    ax.set(xscale='log', yscale='log', xlabel='component period (d)', ylabel='fitted Q',
           xlim=(0.08, 10.5), ylim=(0.5, 2000))
    ax.set_title('(a) Q against period: flat box → decay-time cap', loc='left', fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower left')

    # (b) decay time against period
    ax = axs[0, 1]
    old_tau = old_qs.Q * old_qs.period / np.pi
    ax.scatter(old_qs.period, old_tau, s=9, color='#b8b6ae', alpha=.5, lw=0, label='before')
    ax.scatter(new_qs.loc[~new_qs.matches_injected, 'period'], new_qs.loc[~new_qs.matches_injected, 'tau'],
               s=13, color='crimson', alpha=.7, lw=0, label='after: spurious')
    ax.scatter(new_qs.loc[new_qs.matches_injected, 'period'], new_qs.loc[new_qs.matches_injected, 'tau'],
               s=13, color='seagreen', alpha=.7, lw=0, label='after: matches injection')
    ax.axhline(TAU_CAP_DAYS, color='black', ls='--', lw=1.3, label=f'cap {TAU_CAP_DAYS:g} d (one sector)')
    ax.plot(P, P, '-.', color='#5f5e5a', lw=.9, label='τ_d = P (soft floor)')
    ax.set(xscale='log', yscale='log', xlabel='component period (d)', ylabel='decay time τ_d = Q P / π (d)',
           xlim=(0.08, 10.5))
    ax.set_title('(b) Decay time: the old box allowed τ_d ≫ the light curve', loc='left', fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right')

    # (c) purity / completeness against a cycles cut, before and after
    ax = axs[1, 0]
    cuts = (0, 0.5, 1, 1.6, 3, 6, 10, 30, 100, 300)
    for qs_, lab, ls in [(old_qs, 'before', ':'), (new_qs, 'after', '-')]:
        pc = C.purity_curve(qs_, cuts, by='cycles')
        ax.plot(pc.cycles_cut, pc.purity, 'o' + ls, color='seagreen', label=f'purity ({lab})')
        ax.plot(pc.cycles_cut, pc.completeness, 's' + ls, color='steelblue', label=f'completeness ({lab})')
    ax.axvline(N_CYCLES_MIN, color='#5f5e5a', ls='--', lw=1)
    ax.set(xscale='symlog', ylim=(0, 1.05), xlabel='cycles cut  (Q / π ≥)', ylabel='fraction')
    ax.set_title('(c) Component filter: what a cycles cut costs, before vs after', loc='left',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=2)

    # (d) detectable-oscillator recovery by injected period, before and after
    ax = axs[1, 1]
    edges = np.array([0.1, 0.25, 0.5, 1, 2, 4, 9])
    ctr = np.sqrt(edges[:-1] * edges[1:])
    for osc_, lab, c, mk in [(old_osc, 'before', '#8f8e88', 'o'), (new_osc, 'after', CLEAN, 's')]:
        det = osc_[osc_.signif]
        frac, n = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = det[(det.P >= lo) & (det.P < hi)]
            frac.append(b.recovered.mean() if len(b) else np.nan); n.append(len(b))
        ax.plot(ctr, frac, marker=mk, color=c, lw=1.5, label=f'{lab}  ({int(det.recovered.sum())}/{len(det)} overall)')
    for x, k in zip(ctr, n):
        ax.text(x, .5, f'n={k}', ha='center', fontsize=7.5, color='#8f8e88')
    ax.set(xscale='log', ylim=(0.45, 1.02), xlabel='injected period (d)',
           ylabel='fraction of detectable injections recovered')
    ax.set_title('(d) Recovery of detectable oscillators by period', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('SHO Q bounds: flat Q ∈ [1/√2, 500] (before) → decay time ≤ one sector (after)',
                 fontsize=14, fontweight='bold', x=0.005, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color='#8f8e88')
    plt.tight_layout(rect=(0, .01, 1, .985))
    return fig


if __name__ == '__main__':
    main()

"""Sidelobe removal: prewhitened seeding and the evidence (log-likelihood drop) filter.

Two changes to the GP pipeline, validated on the synthetic set against the capped fit of
2026-11-04 (docs/gp_fitting.tex S4, S6; figs/purity_diagnostics.png):

1. **Prewhitening in the peak finder** (``gp_fit.find_all_significant_peaks(prewhiten=...)``,
   ``iterative_sho_gp_fit(prewhiten_seeds=True)``).  After each accepted peak the
   least-squares sinusoid at that frequency is subtracted and the periodogram recomputed
   from the residual, so the window-function sidelobes of a loud signal are never seeded.

2. **Per-component evidence** (``gp_fit.component_loglike_drops``): the log-likelihood the
   fit loses when a component is removed, by backward elimination, stored as
   ``gp_sho_dloglike_seq``.  ``filter_gp_components(dloglike_min=...)`` cuts on it instead
   of (or as well as) the amplitude floor ``sigma / WN >= 0.2``.

Both tables were fitted by the same script with identical kwargs, differing only in
``prewhiten_seeds``.  The "plain" table reproduces the 2026-11-04 capped fit and adds the
evidence columns.

Run:  python sidelobe_removal_figs.py [--plain PATH] [--prewhite PATH] [--figdir DIR]
Writes figs/sidelobe_removal/{seeding_prewhitening.png, evidence_filter.png, *.csv}
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
from astropy.timeseries import LombScargle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sytheticdata_generate_compare.synth import compare as C   # noqa: E402
from gp_pipeline.gp_fit import clean_lc, prewhiten_sinusoid    # noqa: E402
from make_figs import TRUTH_PATH, SNR_MIN, N_CYCLES_MIN         # noqa: E402

PLAIN_TABLE = os.path.join(ROOT, 'data', 'cluster_table_Nov04_3sectorcombos_synthetic_expavg_gp_taucap_plain.pkl')
PREWHITE_TABLE = os.path.join(ROOT, 'data', 'cluster_table_Nov04_3sectorcombos_synthetic_expavg_gp_taucap_prewhite.pkl')
OLD_TABLE = os.path.join(ROOT, 'data', 'cluster_table_Nov03_3sectorcombos_synthetic_expavg_gp_taucap.pkl')
FIGDIR = os.path.join(HERE, 'figs', 'sidelobe_removal')

T_SECTOR, T_ORBIT = 27.4, 13.7
TOL = 0.10
Q_KEEP = N_CYCLES_MIN * np.pi
DLL_SCAN = (0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100)
SNR_SCAN = (0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 110})
GREEN, RED, GREY, BLUE, ORANGE, PURPLE = '#1baf7a', '#d1352b', '#8f8e88', '#2a78d6', '#eb6834', '#7e57c2'
INK = '#3d3d39'


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

def _arr(r, col):
    v = r.get(col)
    return np.atleast_1d(np.asarray(v, float)) if v is not None else np.array([])


def load(path):
    return C.join(pd.read_pickle(path), pd.read_pickle(TRUTH_PATH))


def components(df):
    """One row per fitted component: amplitude, coherence, evidence, truth match, class."""
    rows = []
    for i, r in df.iterrows():
        P, S, Q = _arr(r, 'gp_periods'), _arr(r, 'gp_sho_sigmas'), _arr(r, 'gp_sho_Qs')
        if len(P) == 0 or not (len(P) == len(S) == len(Q)):
            continue
        D, DS = _arr(r, 'gp_sho_dloglike'), _arr(r, 'gp_sho_dloglike_seq')
        if len(D) != len(P):
            D = np.full(len(P), np.nan)
        if len(DS) != len(P):
            DS = np.full(len(P), np.nan)
        inj = _arr(r, 'periods')
        inj = inj[np.isfinite(inj)]
        T = C._baseline_days(r)
        wn = C.row_white_noise(r)
        for per, sig, q, d, ds in zip(P, S, Q, D, DS):
            snr = sig / wn if np.isfinite(wn) and wn > 0 else np.nan
            amp_low = not (np.isfinite(snr) and snr >= SNR_MIN)
            kind, nearest, offset, df_min = C._classify_component(
                per, amp_low, q, inj, T, TOL, 2.0, (2, 3, 4), 0.05, 1 / np.sqrt(2), 0.02)
            rows.append(dict(row=i, name=r['name'], n_sectors=int(r.get('n_sectors', 0)),
                             period=per, sigma=sig, snr=snr, Q=q, cycles=q / np.pi,
                             dll=d, dll_seq=ds, kind=kind, matched=(kind == 'matched'),
                             nearest_inj=nearest, delta_f=df_min, n_pts=len(_arr(r, 'LC_t')),
                             coherent=bool(np.isfinite(q) and q >= Q_KEEP)))
    out = pd.DataFrame(rows)
    out['kind'] = pd.Categorical(out['kind'], categories=C.FP_ORDER, ordered=True)
    return out


def detectable(df, window=0.02):
    """Per row: the injections whose LSP peak clears the Vaughan threshold, with A / WN."""
    det = {}
    for i, r in df.iterrows():
        inj, amp = _arr(r, 'periods'), _arr(r, 'amplitudes')
        f, p = np.asarray(r['LSP_freq'], float), np.asarray(r['LSP_power'], float)
        cont = 10.0 ** float(r['rn_log10_N']) * f ** (-float(r['rn_alpha']))
        gam = -np.log(1 - 0.99 ** (1 / len(f)))
        wn = C.row_white_noise(r)
        lst = []
        for Pi, Ai in zip(inj, amp):
            fi = 1.0 / Pi
            if fi < f[0] or fi > f[-1]:
                continue
            w = np.abs(f / fi - 1) < window
            if w.any() and np.nanmax(p[w]) > gam * np.interp(fi, f, cont):
                lst.append((Pi, Ai / wn if np.isfinite(wn) and wn > 0 else np.nan))
        det[i] = lst
    return det


def isolated_detectability(df, det, cache=None):
    """Is each detectable injection detectable ON ITS OWN, and is it resolved?

    ``detectable`` reads the periodogram near the injected frequency, so it also counts
    power leaked there by a louder neighbour's window sidelobes.  Here every injected
    frequency is fitted jointly (weighted least squares, offset + sin + cos per frequency),
    the OTHER injections' fitted components are subtracted, and the residual periodogram
    near f_i is tested against the same Vaughan threshold.  Fitting all of them together
    keeps the tested injection's own power with it as far as least squares can separate
    the pair; fitting only the others would let a close neighbour absorb it.

    ``sep`` is the frequency separation to the nearest LOUDER injection in resolution
    elements (x T_baseline).  Below ~1 the pair is not separable from the data alone --
    only an oracle that knows both frequencies can split them -- so "resolved" means
    sep >= 1.

    Returns one row per detectable injection: row, P, a_wn, sep, iso_ratio (peak power /
    threshold after removal), iso_det, resolved, real = iso_det & resolved.
    """
    if cache and os.path.exists(cache):
        return pd.read_csv(cache)
    rows = []
    for i, r in df.iterrows():
        if not det[i]:
            continue
        t, x, y, yerr = clean_lc(r['LC_t'], r['LC_flux'], r['LC_flux_err'])
        f = np.asarray(r['LSP_freq'], float)
        cont = 10 ** float(r['rn_log10_N']) * f ** (-float(r['rn_alpha']))
        gam = -np.log(1 - 0.99 ** (1 / len(f)))
        inj, amp = _arr(r, 'periods'), _arr(r, 'amplitudes')
        T = C._baseline_days(r)
        wts = 1.0 / yerr
        cols = [np.ones_like(x)]
        for fk in 1.0 / inj:
            cols += [np.sin(2 * np.pi * fk * x), np.cos(2 * np.pi * fk * x)]
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A * wts[:, None], y * wts, rcond=None)
        for Pi, a in det[i]:
            k = int(np.argmin(np.abs(inj - Pi)))
            keep = np.zeros(A.shape[1], bool); keep[0] = True; keep[1 + 2 * k: 3 + 2 * k] = True
            y_iso = y - A[:, ~keep] @ coef[~keep]
            fi = 1.0 / Pi
            w = np.abs(f / fi - 1) < 0.02
            p_iso = LombScargle(x, y_iso, yerr, normalization='psd').power(f[w])
            louder = amp > amp[k]
            sep = (np.min(np.abs(1 / inj[louder] - fi)) * T) if louder.any() else np.inf
            rows.append(dict(row=i, P=Pi, a_wn=a, sep=sep,
                             iso_ratio=float(p_iso.max() / (gam * np.interp(fi, f, cont)))))
    R = pd.DataFrame(rows)
    R['iso_det'] = R.iso_ratio > 1
    R['resolved'] = R.sep >= 1.0
    R['real'] = R.iso_det & R.resolved
    if cache:
        R.to_csv(cache, index=False)
    return R


def recovered_on(comps, keep, iso):
    """Completeness restricted to injections that are resolved and detectable on their own."""
    rec = _recovered(comps, keep, {i: [(P, a) for P, a in zip(g.P, g.a_wn)]
                                   for i, g in iso[iso.real].groupby('row')})
    return float(rec.recovered.mean()) if len(rec) else np.nan


def seeded(df, iso):
    """Per detectable injection: is there a seed within TOL of it."""
    out = []
    for _, q in iso.iterrows():
        s_ = _arr(df.loc[q.row], 'gp_initial_lsp_peak_periods')
        out.append(bool(len(s_) and np.any(np.abs(s_ / q.P - 1) < TOL)))
    return np.array(out)


def seeds_table(df):
    """Per row: in-band injections, seeds, matched seeds, components capped."""
    rows = []
    for i, r in df.iterrows():
        inj = _arr(r, 'periods'); inj = inj[np.isfinite(inj)]
        f = np.asarray(r['LSP_freq'], float)
        in_band = inj[(1 / inj >= f[0]) & (1 / inj <= f[-1])]
        seeds = _arr(r, 'gp_initial_lsp_peak_periods')
        n_match = int(sum(1 for p in seeds if len(inj) and np.any(np.abs(inj / p - 1) < TOL)))
        rows.append(dict(row=i, name=r['name'], n_sectors=int(r.get('n_sectors', 0)),
                         n_inj_band=len(in_band), n_seeds=len(seeds), n_seeds_matched=n_match,
                         n_found=float(r.get('gp_n_peaks_found', np.nan)),
                         capped=bool(r.get('gp_components_capped', False))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# filter evaluation
# ---------------------------------------------------------------------------

def _recovered(comps, keep, det):
    """Per detectable injection: (row, P, A/WN, recovered) under the kept components."""
    by_row = {k: g['period'].to_numpy() for k, g in comps[keep].groupby('row')}
    out = []
    for i, lst in det.items():
        P = by_row.get(i, np.array([]))
        for Pi, a in lst:
            out.append((i, Pi, a, bool(len(P) and np.any(np.abs(P / Pi - 1) < TOL))))
    return pd.DataFrame(out, columns=['row', 'P', 'a_wn', 'recovered'])


def evaluate(comps, keep, det):
    rec = _recovered(comps, keep, det)
    return dict(components=int(keep.sum()),
                purity=float(comps.matched[keep].mean()) if keep.any() else np.nan,
                completeness=float(rec.recovered.mean()) if len(rec) else np.nan,
                detectable=len(rec))


def scan(comps, det, label, column, thresholds, coherence=True):
    rows = []
    for th in thresholds:
        keep = np.isfinite(comps[column]) & (comps[column] >= th)
        if coherence:
            keep &= comps.coherent
        rows.append(dict(filter=label, cut=th, **evaluate(comps, keep, det)))
    return pd.DataFrame(rows)


def pick_threshold(sc, target_purity):
    """Smallest evidence threshold whose purity reaches ``target_purity``."""
    ok = sc[sc.purity >= target_purity]
    return float(ok.cut.iloc[0]) if len(ok) else float(sc.cut.iloc[-1])


# ---------------------------------------------------------------------------
# figure 1: seeding
# ---------------------------------------------------------------------------

def fig_seeding(dfs, sts, cps, dets, iso, figdir, stamp):
    fig, axs = plt.subplots(2, 3, figsize=(20, 11.5))
    st_p, st_w = sts['plain'], sts['prewhite']
    cp_p, cp_w = cps['plain'], cps['prewhite']

    # (a) seeds vs injections --------------------------------------------------------
    ax = axs[0, 0]
    jit = np.random.default_rng(0).uniform(-.18, .18, len(st_p))
    ax.scatter(st_p.n_inj_band + jit, st_p.n_seeds + jit, s=22, alpha=.55, lw=0, color=GREY,
               label='fractional mask only (before)')
    ax.scatter(st_w.n_inj_band + jit, st_w.n_seeds + jit, s=22, alpha=.75, lw=0, color=BLUE,
               label='prewhitened (after)')
    ax.plot([0, 10.5], [0, 10.5], '--', color=INK, lw=1, label='1 seed per injection')
    ax.axhline(16, color=RED, ls=':', lw=1)
    ax.text(0.2, 16.3, 'max_components = 16', fontsize=8, color=RED)
    txt = []
    for lab, st in (('before', st_p), ('after', st_w)):
        txt.append(f'{lab}: {st.n_seeds.sum()} seeds, {st.n_seeds.sum() / st.n_inj_band.sum():.2f} per injection, '
                   f'purity {st.n_seeds_matched.sum() / st.n_seeds.sum():.0%}, {int(st.capped.sum())} rows capped')
    ax.text(.03, .86, '\n'.join(txt), transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
    ax.set(xlabel='injected oscillators in band', ylabel='LSP seeds (= fitted components)',
           xlim=(-.5, 10.8), ylim=(-.5, 17.5))
    ax.set_title('(a) Prewhitening removes the extra seeds', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (b) |df| of spurious components -------------------------------------------------
    ax = axs[0, 1]
    bins = np.logspace(-3, 1, 50)
    for lab, cp, c, fill in (('before', cp_p, GREY, True), ('after', cp_w, BLUE, True)):
        d = cp.delta_f[~cp.matched & np.isfinite(cp.delta_f)]
        ax.hist(d, bins=bins, color=c, alpha=.6 if fill else 1, histtype='stepfilled' if fill else 'step',
                lw=1.4, label=f'spurious components, {lab} ({len(d)})')
    for x, lab, ls in [(1 / T_SECTOR, '1/T_sector', '--'), (1 / T_ORBIT, '1/T_orbit = 2/T_sector', '-.')]:
        ax.axvline(x, color='black', ls=ls, lw=1.1)
        ax.text(x * 1.05, ax.get_ylim()[1] * .9, lab, rotation=90, fontsize=7.5, va='top', color=INK)
    for lab, cp, y in (('before', cp_p, .72), ('after', cp_w, .64)):
        d = cp.delta_f[~cp.matched & np.isfinite(cp.delta_f)]
        fr = float(((d > 0.5 / T_SECTOR) & (d < 2.5 / T_ORBIT)).mean()) if len(d) else np.nan
        ax.text(.97, y, f'{lab}: {fr:.0%} of spurious at sidelobe offsets', transform=ax.transAxes,
                ha='right', fontsize=8.5, color=INK)
    ax.set(xscale='log', xlabel='|Δf| to nearest injection (1/day)', ylabel='spurious components')
    ax.set_title('(b) The sidelobe comb disappears', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (c) purity by crowding -----------------------------------------------------------
    ax = axs[0, 2]
    for k, (lab, st, c) in enumerate((('before', st_p, GREY), ('after', st_w, BLUE))):
        g = st.groupby('n_inj_band').agg(seeds=('n_seeds', 'sum'), matched=('n_seeds_matched', 'sum'))
        pur = g.matched / g.seeds
        ax.bar(g.index + (k - .5) * .4, pur, width=.4, color=c, label=lab)
        for n, v in pur.items():
            if np.isfinite(v):
                ax.text(n + (k - .5) * .4, v + .015, f'{v:.0%}', ha='center', fontsize=7)
    ax.set(xlabel='injected oscillators in band', ylabel='seed purity', ylim=(0, 1.12))
    ax.set_title('(c) Purity by crowding', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (d) what the "lost" injections are ---------------------------------------------------
    ax = axs[1, 0]
    sp, sw = seeded(dfs['plain'], iso), seeded(dfs['prewhite'], iso)
    lost = sp & ~sw
    cats = [('unresolved from a louder\ninjection (sep < 1/T)', lost & ~iso.resolved.to_numpy(), GREY),
            ('resolved, but detectable only\nthrough neighbour leakage', lost & iso.resolved.to_numpy() & ~iso.iso_det.to_numpy(), ORANGE),
            ('resolved and detectable\non its own: a real loss', lost & iso.real.to_numpy(), RED)]
    for k, (lab, m, c) in enumerate(cats):
        ax.barh(k, int(m.sum()), color=c)
        ax.text(int(m.sum()) + .8, k, str(int(m.sum())), va='center', fontsize=9)
    ax.set_yticks(range(len(cats)), [c[0] for c in cats], fontsize=8.5)
    ax.invert_yaxis()
    ax.set(xlabel='"detectable" injections seeded before but not after')
    real = iso.real.to_numpy()
    ax.text(.97, .05, f'{int(lost.sum())} "lost" in all.\nOn the {int(real.sum())} injections that are resolved\n'
                      f'and detectable on their own, seeds reach\n{sp[real].mean():.1%} before, {sw[real].mean():.1%} after.',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5, color=INK)
    ax.set_title('(d) Most "lost" injections were never separable', loc='left', fontweight='bold')

    # (e) headline before/after, two completeness definitions ---------------------------
    ax = axs[1, 1]
    labels, vals = [], []
    for lab, cp, det in (('before', cp_p, dets['plain']), ('after', cp_w, dets['prewhite'])):
        for flab, keep in (('raw', np.ones(len(cp), bool)),
                           (f'σ/WN≥{SNR_MIN:g}', cp.coherent & np.isfinite(cp.snr) & (cp.snr >= SNR_MIN))):
            e = evaluate(cp, keep, det)
            e['completeness_real'] = recovered_on(cp, keep, iso)
            labels.append(f'{lab}\n{flab}'); vals.append(e)
    x = np.arange(len(labels))
    ax.bar(x - .27, [v['purity'] for v in vals], width=.27, color=GREEN, label='purity')
    ax.bar(x, [v['completeness'] for v in vals], width=.27, color=ORANGE, alpha=.55,
           label='completeness, all "detectable" (611)')
    ax.bar(x + .27, [v['completeness_real'] for v in vals], width=.27, color=ORANGE,
           label=f'completeness, resolved & detectable alone ({int(real.sum())})')
    for xi, v in zip(x, vals):
        for dx, key in ((-.27, 'purity'), (0, 'completeness'), (.27, 'completeness_real')):
            ax.text(xi + dx, v[key] + .01, f'{v[key]:.2f}', ha='center', fontsize=7.5)
        ax.text(xi, .03, f'{v["components"]} comps', ha='center', fontsize=7.5, color='white')
    ax.set_xticks(x, labels)
    ax.set(ylim=(0, 1.32))
    ax.set_title('(e) Component list, same amplitude filter', loc='left', fontweight='bold')
    ax.legend(fontsize=7.5, loc='upper left', ncol=2)

    # (f) example periodogram ------------------------------------------------------------
    ax = axs[1, 2]
    gain = (st_p.n_seeds - st_w.n_seeds)
    i = int(st_p.row[gain.idxmax()])
    r_p, r_w = dfs['plain'].loc[i], dfs['prewhite'].loc[i]
    f, p = np.asarray(r_p['LSP_freq'], float), np.asarray(r_p['LSP_power'], float)
    cont = 10.0 ** float(r_p['rn_log10_N']) * f ** (-float(r_p['rn_alpha']))
    gam = -np.log(1 - 0.99 ** (1 / len(f)))
    t, x_, y, yerr = clean_lc(r_p['LC_t'], r_p['LC_flux'], r_p['LC_flux_err'])
    y_res = y.copy()
    for per in _arr(r_w, 'gp_initial_lsp_peak_periods'):
        y_res = prewhiten_sinusoid(x_, y_res, yerr, 1.0 / per)
    p_res = LombScargle(x_, y_res, yerr, normalization='psd').power(f)
    ax.plot(f, p, color=GREY, lw=.7, label='LSP (before)')
    ax.plot(f, p_res, color=BLUE, lw=.7, label='residual LSP after all prewhitened seeds')
    ax.plot(f, gam * cont, color='black', lw=1, ls='--', label='threshold')
    inj = _arr(r_p, 'periods')
    for Pi in inj:
        ax.axvline(1 / Pi, color=GREEN, lw=.9, alpha=.6)
    top = np.nanmax(p) * 2
    for per in _arr(r_p, 'gp_initial_lsp_peak_periods'):
        m = len(inj) and np.any(np.abs(inj / per - 1) < TOL)
        ax.plot(1 / per, top, marker='v', ms=6, lw=0, color=GREEN if m else RED, markeredgecolor='black', markeredgewidth=.4)
    for per in _arr(r_w, 'gp_initial_lsp_peak_periods'):
        m = len(inj) and np.any(np.abs(inj / per - 1) < TOL)
        ax.plot(1 / per, top * 2.5, marker='v', ms=6, lw=0, color=GREEN if m else RED, markeredgecolor=BLUE, markeredgewidth=1)
    ax.text(f[0] * 1.05, top * 1.15, 'seeds before', fontsize=7.5, color=INK)
    ax.text(f[0] * 1.05, top * 2.9, 'seeds after', fontsize=7.5, color=BLUE)
    ax.set(xscale='log', yscale='log', xlabel='frequency (1/day)', ylabel='LSP power',
           ylim=(np.nanpercentile(p, 5), top * 6))
    ax.set_title(f'(f) {r_p["name"]}, {r_p["n_sectors"]} sector(s): {len(inj)} injected, '
                 f'{len(_arr(r_p, "gp_initial_lsp_peak_periods"))} → {len(_arr(r_w, "gp_initial_lsp_peak_periods"))} seeds',
                 loc='left', fontweight='bold')
    ax.legend(handles=[plt.Line2D([], [], color=GREY, lw=1, label='LSP (before)'),
                       plt.Line2D([], [], color=BLUE, lw=1, label='residual LSP (after)'),
                       plt.Line2D([], [], color=GREEN, lw=1, label='injected frequency'),
                       plt.Line2D([], [], marker='v', ls='', color=GREEN, label='seed, matched'),
                       plt.Line2D([], [], marker='v', ls='', color=RED, label='seed, spurious')],
              fontsize=7.5, loc='lower left')

    fig.suptitle('Prewhitening in the peak finder: subtract each accepted sinusoid before looking for the next peak',
                 fontsize=14, fontweight='bold', x=.005, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color=GREY)
    plt.tight_layout(rect=(0, .01, 1, .97))
    out = os.path.join(figdir, 'seeding_prewhitening.png')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


# ---------------------------------------------------------------------------
# figure 2: evidence filter
# ---------------------------------------------------------------------------

def fig_evidence(cps, dets, scans, t_star, t_bic, figdir, stamp):
    cp, det = cps['prewhite'], dets['prewhite']
    cp_p, det_p = cps['plain'], dets['plain']
    fig, axs = plt.subplots(2, 3, figsize=(20, 11.5))
    floor = 1e-2

    # (a) evidence distribution ---------------------------------------------------------
    ax = axs[0, 0]
    bins = np.logspace(-2, 4, 60)
    ax.hist(cp.dll_seq[cp.matched].clip(floor), bins=bins, color=GREEN, alpha=.65, label='matched')
    ax.hist(cp.dll_seq[~cp.matched].clip(floor), bins=bins, color=RED, alpha=.65, label='spurious')
    ax.axvline(t_star, color='black', ls='--', lw=1.2)
    ax.text(t_star * 1.2, ax.get_ylim()[1] * .9, f'ΔlogL ≥ {t_star:g}', fontsize=8)
    ax.axvline(t_bic, color=PURPLE, ls=':', lw=1.2)
    ax.text(t_bic * 1.2, ax.get_ylim()[1] * .6, f'BIC-neutral\n1.5 ln N = {t_bic:.0f}', fontsize=7.5, color=PURPLE)
    med_m, med_s = np.nanmedian(cp.dll_seq[cp.matched]), np.nanmedian(cp.dll_seq[~cp.matched])
    ax.text(.03, .97, f'median ΔlogL: matched {med_m:.0f}, spurious {med_s:.2f}\n'
                      f'(values ≤ {floor:g} shown at {floor:g})',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
    ax.set(xscale='log', yscale='log', xlabel='ΔlogL on removal (backward elimination)', ylabel='components')
    ax.set_title('(a) Evidence separates matched from spurious', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (b) amplitude vs evidence ------------------------------------------------------------
    ax = axs[0, 1]
    for m, c, lab, z in ((~cp.matched, RED, 'spurious', 1), (cp.matched, GREEN, 'matched', 2)):
        ax.scatter(cp.snr[m].clip(1e-4), cp.dll_seq[m].clip(floor), s=10, alpha=.5, lw=0, color=c, label=lab, zorder=z)
    ax.axvline(SNR_MIN, color='black', ls='--', lw=1.2)
    ax.axhline(t_star, color='black', ls='--', lw=1.2)
    faint_real = cp.matched & (cp.snr < SNR_MIN) & (cp.dll_seq >= t_star)
    loud_fake = ~cp.matched & (cp.snr >= SNR_MIN) & (cp.dll_seq < t_star)
    ax.text(.03, .97, f'{int(faint_real.sum())} matched components have σ/WN < {SNR_MIN:g}\n'
                      f'but ΔlogL ≥ {t_star:g}: faint signals the amplitude cut\ndiscards and the evidence cut keeps\n'
                      f'{int(loud_fake.sum())} spurious pass the amplitude cut but not evidence',
            transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
    ax.set(xscale='log', yscale='log', xlabel='σ / white noise', ylabel='ΔlogL on removal')
    ax.set_title('(b) Amplitude and evidence are different axes', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (c) purity-completeness curves ------------------------------------------------------
    ax = axs[0, 2]
    style = {'σ/WN + cycles≥1.6': (GREY, 'o'), 'ΔlogL_seq + cycles≥1.6': (BLUE, 's'),
             'ΔlogL_seq alone': (PURPLE, '^'), 'ΔlogL single-pass + cycles≥1.6': (ORANGE, 'd')}
    for lab, (c, mk) in style.items():
        s = scans[('prewhite', lab)]
        ax.plot(s.completeness, s.purity, marker=mk, ms=5, lw=1.2, color=c, label=lab)
        for _, q in s.iterrows():
            if lab.startswith('ΔlogL_seq +') or lab.startswith('σ/WN'):
                ax.annotate(f'{q.cut:g}', (q.completeness, q.purity), fontsize=6.5, color=c,
                            xytext=(3, 3), textcoords='offset points')
    ax.set(xlabel='completeness (detectable injections)', ylabel='purity', ylim=(0.6, 1.01))
    ax.set_title('(c) Purity vs completeness along each cut (labels = threshold)', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')

    # (d) completeness vs injected amplitude --------------------------------------------
    ax = axs[1, 0]
    edges = np.array([0.03, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 50])
    rules = (('unfiltered', np.ones(len(cp), bool), GREY),
             (f'σ/WN ≥ {SNR_MIN:g} + cycles', cp.coherent & (cp.snr >= SNR_MIN), 'black'),
             (f'ΔlogL ≥ {t_star:g} + cycles', cp.coherent & (cp.dll_seq >= t_star), BLUE),
             (f'ΔlogL ≥ {t_star:g} alone', cp.dll_seq >= t_star, PURPLE))
    for lab, keep, c in rules:
        rec = _recovered(cp, keep, det)
        rec['bin'] = pd.cut(rec.a_wn, edges)
        g = rec.groupby('bin', observed=False).recovered.agg(['mean', 'size'])
        xc = np.sqrt(edges[:-1] * edges[1:])
        ax.plot(xc, g['mean'], marker='o', ms=4, lw=1.3, color=c, label=lab)
    rec = _recovered(cp, np.ones(len(cp), bool), det)
    rec['bin'] = pd.cut(rec.a_wn, edges)
    for xi, n in zip(np.sqrt(edges[:-1] * edges[1:]), rec.groupby('bin', observed=False).size()):
        ax.text(xi, .02, f'n={n}', ha='center', fontsize=7, color=INK)
    ax.axvline(SNR_MIN * np.sqrt(2), color='black', ls=':', lw=1)
    ax.text(SNR_MIN * np.sqrt(2) * 1.05, .5, 'σ/WN = 0.2 for a sinusoid\n(A = √2 σ)', fontsize=7.5, rotation=90, va='center')
    ax.set(xscale='log', xlabel='injected amplitude A / white noise per point', ylabel='completeness', ylim=(0, 1.05))
    ax.set_title('(d) Where the amplitude cut loses signals', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (e) evidence filter on the old seeds too --------------------------------------------
    ax = axs[1, 1]
    for seeder, c in (('plain', GREY), ('prewhite', BLUE)):
        for lab, mk, ls in (('σ/WN + cycles≥1.6', 'o', ':'), ('ΔlogL_seq + cycles≥1.6', 's', '-')):
            s = scans[(seeder, lab)]
            ax.plot(s.completeness, s.purity, marker=mk, ms=4, lw=1.2, ls=ls, color=c,
                    label=f'{"before" if seeder == "plain" else "after"} prewhitening, {lab}')
    ax.set(xlabel='completeness (detectable injections)', ylabel='purity', ylim=(0.6, 1.01))
    ax.set_title('(e) Both fixes, both seeders', loc='left', fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower left')

    # (f) what survives, by class ----------------------------------------------------------
    ax = axs[1, 2]
    keep_e = cp.coherent & (cp.dll_seq >= t_star)
    keep_a = cp.coherent & (cp.snr >= SNR_MIN)
    kinds = [k for k in C.FP_ORDER if (cp.kind == k).any()]
    x = np.arange(len(kinds))
    n_all = [int((cp.kind == k).sum()) for k in kinds]
    n_a = [int(((cp.kind == k) & keep_a).sum()) for k in kinds]
    n_e = [int(((cp.kind == k) & keep_e).sum()) for k in kinds]
    ax.bar(x - .27, n_all, width=.27, color=GREY, label='fitted')
    ax.bar(x, n_a, width=.27, color='black', label=f'kept by σ/WN ≥ {SNR_MIN:g} + cycles')
    ax.bar(x + .27, n_e, width=.27, color=BLUE, label=f'kept by ΔlogL ≥ {t_star:g} + cycles')
    for xi, a, b, c_ in zip(x, n_all, n_a, n_e):
        ax.text(xi - .27, a * 1.05, str(a), ha='center', fontsize=7)
        ax.text(xi, max(b, .7) * 1.05, str(b), ha='center', fontsize=7)
        ax.text(xi + .27, max(c_, .7) * 1.05, str(c_), ha='center', fontsize=7)
    ax.set_xticks(x, [k.replace(' (', '\n(') for k in kinds], fontsize=8)
    ax.set(yscale='log', ylabel='components')
    ax.set_title('(f) What each filter keeps, by class (prewhitened fit)', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Evidence filter: cut on the log-likelihood a component contributes, not on its amplitude',
                 fontsize=14, fontweight='bold', x=.005, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color=GREY)
    plt.tight_layout(rect=(0, .01, 1, .97))
    out = os.path.join(figdir, 'evidence_filter.png')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


# ---------------------------------------------------------------------------

def fig_threshold(tags, sts, cps, dets, iso, t_star, figdir, stamp):
    """Seeder acceptance threshold with prewhitening: seeds, purity, completeness."""
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    cols = {tags[0]: GREY}
    pal = [BLUE, ORANGE, PURPLE, GREEN]
    for k, t in enumerate(tags[1:]):
        cols[t] = pal[k % len(pal)]

    # (a) seeds per injection, rows capped -------------------------------------------------
    ax = axs[0]
    x = np.arange(len(tags))
    spi = [sts[t].n_seeds.sum() / sts[t].n_inj_band.sum() for t in tags]
    pur = [sts[t].n_seeds_matched.sum() / sts[t].n_seeds.sum() for t in tags]
    ax.bar(x - .2, spi, width=.4, color=[cols[t] for t in tags], label='seeds per in-band injection')
    ax.bar(x + .2, pur, width=.4, color=[cols[t] for t in tags], alpha=.45, label='seed purity')
    for xi, a, b, t in zip(x, spi, pur, tags):
        ax.text(xi - .2, a + .02, f'{a:.2f}', ha='center', fontsize=8)
        ax.text(xi + .2, b + .02, f'{b:.0%}', ha='center', fontsize=8)
        ax.text(xi, .04, f'{int(sts[t].capped.sum())} rows\ncapped', ha='center', fontsize=7.5, color='white')
    ax.set_xticks(x, tags, fontsize=8)
    ax.axhline(1, color=INK, ls='--', lw=1)
    ax.set_title('(a) Seeds and seed purity by seeder', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (b) purity / completeness: raw, amplitude filter, evidence filter -----------------------
    ax = axs[1]
    for t in tags:
        cp, det = cps[t], dets[t]
        pts = [('raw', np.ones(len(cp), bool), 'x', (6, -10)),
               (f'σ/WN≥{SNR_MIN:g}', cp.coherent & (cp.snr >= SNR_MIN), 'o', (6, 4)),
               (f'ΔlogL≥{t_star:g}', cp.coherent & (cp.dll_seq >= t_star), 's', (-60, -12))]
        for lab, keep, mk, off in pts:
            e = evaluate(cp, keep, det)
            xr = recovered_on(cp, keep, iso)
            ax.plot(xr, e['purity'], marker=mk, ms=8, lw=0, color=cols[t],
                    markeredgecolor='black' if mk != 'x' else cols[t], markeredgewidth=.5)
            ax.annotate(lab, (xr, e['purity']), fontsize=7, color=cols[t],
                        xytext=off, textcoords='offset points')
    ax.set(xlabel=f'completeness, resolved & detectable alone ({int(iso.real.sum())})', ylabel='purity')
    for t in tags:
        ax.plot([], [], 's', color=cols[t], label=t)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('(b) raw (x), amplitude filter (o), evidence filter (s); cycles ≥ 1.6 on filters',
                 loc='left', fontweight='bold', fontsize=10.5)

    # (c) completeness vs injected amplitude, evidence-filtered ---------------------------------
    ax = axs[2]
    edges = np.array([0.03, 0.1, 0.2, 0.3, 0.5, 1, 2, 5, 50])
    xc = np.sqrt(edges[:-1] * edges[1:])
    for t in tags:
        cp, det = cps[t], dets[t]
        rec = _recovered(cp, cp.coherent & (cp.dll_seq >= t_star), det)
        rec['bin'] = pd.cut(rec.a_wn, edges)
        g = rec.groupby('bin', observed=False).recovered.mean()
        ax.plot(xc, g, marker='o', ms=4, lw=1.3, color=cols[t], label=f'{t}, ΔlogL≥{t_star:g}+cycles')
    ax.set(xscale='log', xlabel='injected amplitude A / white noise per point', ylabel='completeness', ylim=(0, 1.05))
    ax.set_title('(c) Completeness by injected amplitude, evidence-filtered', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Lowering the seeder threshold after prewhitening adds noise seeds that the evidence cut does not reject',
                 fontsize=14, fontweight='bold', x=.005, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color=GREY)
    plt.tight_layout(rect=(0, .02, 1, .95))
    out = os.path.join(figdir, 'seeder_threshold.png')
    fig.savefig(out, bbox_inches='tight'); plt.close(fig)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plain', default=PLAIN_TABLE)
    ap.add_argument('--prewhite', default=PREWHITE_TABLE,
                    help='the prewhitened fit used as "after" in the before/after figures')
    ap.add_argument('--extra', nargs='*', default=[], metavar='TAG=PATH',
                    help='further fits (e.g. other seeder thresholds) for seeder_threshold.png')
    ap.add_argument('--old', default=OLD_TABLE)
    ap.add_argument('--figdir', default=FIGDIR)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    stamp = (f'plain {os.path.basename(args.plain)} | prewhite {os.path.basename(args.prewhite)} | '
             f'truth {os.path.basename(TRUTH_PATH)} | generated {datetime.now():%Y-%m-%d %H:%M}')

    paths = {'plain': args.plain, 'prewhite': args.prewhite}
    for kv in args.extra:
        tag, path = kv.split('=', 1)
        paths[tag] = path
    dfs = {k: load(v) for k, v in paths.items()}

    # sanity: the plain refit must reproduce the 2026-11-04 capped fit's seeds
    if args.old and os.path.exists(args.old):
        old = pd.read_pickle(args.old)
        same = sum(np.array_equal(np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float)))
                   for a, b in zip(old['gp_initial_lsp_peak_periods'], pd.read_pickle(args.plain)['gp_initial_lsp_peak_periods']))
        print(f'plain refit reproduces the old seeds on {same} of {len(old)} rows')

    sts = {k: seeds_table(v) for k, v in dfs.items()}
    cps = {k: components(v) for k, v in dfs.items()}
    dets = {k: detectable(v) for k, v in dfs.items()}

    # scans
    scans = {}
    for k in dfs:
        cp, det = cps[k], dets[k]
        scans[(k, 'σ/WN + cycles≥1.6')] = scan(cp, det, 'snr+cycles', 'snr', SNR_SCAN)
        scans[(k, 'ΔlogL_seq + cycles≥1.6')] = scan(cp, det, 'dll_seq+cycles', 'dll_seq', DLL_SCAN)
        scans[(k, 'ΔlogL_seq alone')] = scan(cp, det, 'dll_seq', 'dll_seq', DLL_SCAN, coherence=False)
        scans[(k, 'ΔlogL single-pass + cycles≥1.6')] = scan(cp, det, 'dll+cycles', 'dll', DLL_SCAN)
    allscan = pd.concat([s.assign(seeder=k) for (k, _), s in scans.items()], ignore_index=True)
    allscan.to_csv(os.path.join(args.figdir, 'filter_scan.csv'), index=False)

    # thresholds: match the amplitude filter's purity on the prewhitened fit; BIC-neutral
    cp = cps['prewhite']
    ref = evaluate(cp, cp.coherent & (cp.snr >= SNR_MIN), dets['prewhite'])
    t_star = pick_threshold(scans[('prewhite', 'ΔlogL_seq + cycles≥1.6')], ref['purity'])
    t_bic = float(1.5 * np.log(np.nanmedian(cp.n_pts)))
    print(f'amplitude filter on prewhitened fit: purity {ref["purity"]:.3f}, completeness {ref["completeness"]:.3f}; '
          f'evidence threshold matching that purity: {t_star:g}; BIC-neutral 1.5 ln N = {t_bic:.1f}')

    # headline table
    rows = []
    for k in dfs:
        cp, det, st = cps[k], dets[k], sts[k]
        base = dict(seeder=k, seeds=int(st.n_seeds.sum()), seeds_per_injection=st.n_seeds.sum() / st.n_inj_band.sum(),
                    seed_purity=st.n_seeds_matched.sum() / st.n_seeds.sum(), rows_capped=int(st.capped.sum()))
        for lab, keep in (('none', np.ones(len(cp), bool)),
                          (f'snr>={SNR_MIN:g}+cycles>={N_CYCLES_MIN:g}', cp.coherent & (cp.snr >= SNR_MIN)),
                          (f'dll_seq>={t_star:g}+cycles>={N_CYCLES_MIN:g}', cp.coherent & (cp.dll_seq >= t_star)),
                          (f'dll_seq>={t_star:g}', cp.dll_seq >= t_star),
                          (f'dll_seq>={t_bic:.0f}+cycles>={N_CYCLES_MIN:g}', cp.coherent & (cp.dll_seq >= round(t_bic))),
                          (f'snr>={SNR_MIN:g}+dll_seq>={t_star:g}+cycles', cp.coherent & (cp.snr >= SNR_MIN) & (cp.dll_seq >= t_star))):
            rows.append(dict(**base, filter=lab, **evaluate(cp, keep, det)))
    head = pd.DataFrame(rows)
    head.to_csv(os.path.join(args.figdir, 'headline.csv'), index=False)
    print(head.round(3).to_string())

    # faint-signal accounting on the prewhitened fit
    rec_a = _recovered(cp, cp.coherent & (cp.snr >= SNR_MIN), dets['prewhite'])
    rec_e = _recovered(cp, cp.coherent & (cp.dll_seq >= t_star), dets['prewhite'])
    faint = rec_a.a_wn < SNR_MIN * np.sqrt(2)
    print(f'detectable injections with A/WN < {SNR_MIN * np.sqrt(2):.2f}: {int(faint.sum())}; '
          f'recovered by amplitude filter {int(rec_a.recovered[faint].sum())}, by evidence filter {int(rec_e.recovered[faint].sum())}')

    iso = isolated_detectability(dfs['plain'], dets['plain'],
                                 cache=os.path.join(args.figdir, 'isolated_detectability.csv'))
    sp, sw = seeded(dfs['plain'], iso), seeded(dfs['prewhite'], iso)
    real = iso.real.to_numpy()
    print(f'detectable {len(iso)}; resolved {int(iso.resolved.sum())}; detectable alone {int(iso.iso_det.sum())}; '
          f'both {int(real.sum())}')
    print(f'seeded, resolved & detectable alone: before {sp[real].mean():.3f}, after {sw[real].mean():.3f}; '
          f'lost {int((sp & ~sw).sum())} of which real {int((sp & ~sw & real).sum())}')
    for k in dfs:
        cp = cps[k]
        for lab, keep in (('raw', np.ones(len(cp), bool)),
                          ('snr', cp.coherent & (cp.snr >= SNR_MIN)),
                          ('dll', cp.coherent & (cp.dll_seq >= t_star))):
            print(f'  {k:18s} {lab:4s} completeness on resolved & detectable alone: {recovered_on(cp, keep, iso):.3f}')

    fig_seeding(dfs, sts, cps, dets, iso, args.figdir, stamp)
    fig_evidence(cps, dets, scans, t_star, t_bic, args.figdir, stamp)
    if args.extra:
        fig_threshold(list(paths), sts, cps, dets, iso, t_star, args.figdir, stamp)


if __name__ == '__main__':
    main()

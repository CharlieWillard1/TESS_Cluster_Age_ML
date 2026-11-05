"""Why is raw GP component purity only ~68%?  One figure, eight panels.

The GP fits the full seed list from the peak finder (``fit_full_only=True``), so every
fitted component is one LSP peak and raw purity is a property of the SEEDS, not of the
fit.  These panels ask where the extra seeds come from: how many seeds a row gets per
injection, how far the spurious ones sit from the nearest injection in absolute
frequency (window-function sidelobes have a fixed spacing in 1/day, set by the sector
length and the mid-sector gap, not by the period), whether crowding or multi-sector
windows make it worse, and what the seeds look like on a periodogram.

Run:  python diagnose_purity.py [--table PATH] [--figdir DIR]
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

from sytheticdata_generate_compare.synth import compare as C   # noqa: E402
from make_figs import PIPELINE_TABLE, TRUTH_PATH, FIGDIR, SNR_MIN, N_CYCLES_MIN  # noqa: E402

T_SECTOR, T_ORBIT = 27.4, 13.7          # TESS sector length and orbit (mid-sector gap) in days
TOL = 0.10                              # fractional period tolerance for "matched"
MASK_FRAC = 0.05                        # the peak finder's +-5% frequency mask (harmonic_tolerance)

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white',
                     'axes.grid': True, 'grid.color': '#e6e5e0', 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'figure.dpi': 110})
GREEN, RED, GREY, BLUE = '#1baf7a', '#d1352b', '#8f8e88', '#2a78d6'


def row_table(df):
    """One row per pipeline row: injections, seeds, components, matches."""
    rows = []
    for i, r in df.iterrows():
        inj = np.atleast_1d(np.asarray(r.get('periods', []), float))
        inj = inj[np.isfinite(inj)]
        seeds = np.atleast_1d(np.asarray(r.get('gp_initial_lsp_peak_periods', []), float))
        comps = np.atleast_1d(np.asarray(r.get('gp_periods', []), float))
        n_found = float(r.get('gp_n_peaks_found', np.nan))
        f = np.asarray(r['LSP_freq'], float)
        in_band = inj[(1 / inj >= f[0]) & (1 / inj <= f[-1])]

        def n_match(arr):
            return int(sum(1 for p in arr if len(inj) and np.any(np.abs(inj / p - 1) < TOL)))
        rows.append(dict(row=i, name=r['name'], n_sectors=int(r.get('n_sectors', 0)),
                         n_inj=len(inj), n_inj_band=len(in_band),
                         n_seeds=len(seeds), n_seeds_matched=n_match(seeds),
                         n_comps=len(comps), n_comps_matched=n_match(comps),
                         n_peaks_found=n_found,
                         capped=bool(r.get('gp_components_capped', False))))
    return pd.DataFrame(rows)


def comp_table(df):
    """One row per fitted component with the false-positive class and absolute offsets."""
    fp = C.false_positive_table(df, snr_min=SNR_MIN, n_cycles_min=N_CYCLES_MIN)
    # absolute frequency offset to the nearest injection, in 1/day and in sector-length units
    fp['df_sector'] = fp.delta_f * T_SECTOR
    fp['ratio'] = fp.period / fp.nearest_inj
    # would the peak finder's fractional mask around the nearest injection have covered it?
    fp['inside_frac_mask'] = np.abs(1 / fp.period / (1 / fp.nearest_inj) - 1) < MASK_FRAC
    return fp


def example_rows(df, fp, n=2):
    """Pick a crowded single-sector row and a multi-sector row with many spurious seeds."""
    spur = fp[~fp.matched].groupby(['name', 'n_sectors']).size().rename('n_spur').reset_index()
    picks = []
    for ns in (1, 3, 2):
        s = spur[spur.n_sectors == ns].sort_values('n_spur', ascending=False)
        if len(s):
            picks.append((s.iloc[0]['name'], ns))
        if len(picks) == n:
            break
    out = []
    for name, ns in picks:
        m = df[(df.name == name) & (df.n_sectors == ns)]
        if len(m):
            # the row of that cluster/sector-count with the most spurious components
            def n_spur(r):
                inj = np.atleast_1d(np.asarray(r['periods'], float))
                comps = np.atleast_1d(np.asarray(r['gp_periods'], float))
                return sum(1 for per in comps if not (len(inj) and np.any(np.abs(inj / per - 1) < TOL)))
            out.append(m.loc[m.apply(n_spur, axis=1).idxmax()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default=PIPELINE_TABLE)
    ap.add_argument('--figdir', default=FIGDIR)
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)

    pipe = pd.read_pickle(args.table)
    truth = pd.read_pickle(TRUTH_PATH)
    df = C.join(pipe, truth)
    rt = row_table(df)
    fp = comp_table(df)
    spur = fp[~fp.matched]
    stamp = (f'{os.path.basename(args.table)} | truth {os.path.basename(TRUTH_PATH)} | '
             f'generated {datetime.now():%Y-%m-%d %H:%M}')

    fig, axs = plt.subplots(2, 4, figsize=(24, 11))

    # (a) seeds vs injections per row ------------------------------------------------
    ax = axs[0, 0]
    jit = np.random.default_rng(0).uniform(-.18, .18, len(rt))
    for ns, c, mk in [(1, BLUE, 'o'), (2, '#eb6834', 's'), (3, '#7e57c2', '^')]:
        m = rt.n_sectors == ns
        ax.scatter(rt.n_inj_band[m] + jit[m], rt.n_seeds[m] + jit[m], s=22, alpha=.7, lw=0,
                   color=c, marker=mk, label=f'{ns} sector{"s" if ns > 1 else ""}  (n={int(m.sum())})')
    ax.plot([0, 10.5], [0, 10.5], '--', color='#5f5e5a', lw=1, label='1 seed per injection')
    ax.axhline(16, color=RED, ls=':', lw=1)
    ax.text(0.2, 16.3, 'max_components = 16', fontsize=8, color=RED)
    ax.set(xlabel='injected oscillators in band', ylabel='LSP seeds (= fitted components)',
           xlim=(-.5, 10.8), ylim=(-.5, 17.5))
    seed_pur = rt.n_seeds_matched.sum() / max(rt.n_seeds.sum(), 1)
    comp_pur = rt.n_comps_matched.sum() / max(rt.n_comps.sum(), 1)
    ax.text(.03, .97, f'seed purity {seed_pur:.1%}\ncomponent purity {comp_pur:.1%}\n'
                      f'seeds per in-band injection: {rt.n_seeds.sum() / max(rt.n_inj_band.sum(), 1):.2f}',
            transform=ax.transAxes, va='top', fontsize=8.5, color='#3d3d39')
    ax.set_title('(a) The fit takes every seed: purity is set by the peak finder', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')

    # (b) absolute frequency offset of spurious components --------------------------
    ax = axs[0, 1]
    d = spur.delta_f[np.isfinite(spur.delta_f)]
    bins = np.logspace(-3, 1, 50)
    ax.hist(d, bins=bins, color=RED, alpha=.75, label=f'spurious components ({len(d)})')
    for x, lab, ls in [(1 / T_SECTOR, '1/T_sector', '--'), (1 / T_ORBIT, '1/T_orbit (mid-sector gap)', '-.'),
                       (2 / T_SECTOR, '2/T_sector', ':')]:
        ax.axvline(x, color='black', ls=ls, lw=1.1)
        ax.text(x * 1.05, ax.get_ylim()[1] * .9, lab, rotation=90, fontsize=7.5, va='top', color='#3d3d39')
    frac_sl = float(((d > 0.5 / T_SECTOR) & (d < 2.5 / T_ORBIT)).mean())
    ax.text(.97, .80, f'{frac_sl:.0%} of spurious components sit at\n0.5/T_sector < |Δf| < 2.5/T_orbit:\nwindow-function sidelobes',
            transform=ax.transAxes, va='top', ha='right', fontsize=8.5, color='#3d3d39')
    ax.set(xscale='log', xlabel='|Δf| to nearest injection (1/day)', ylabel='spurious components')
    ax.set_title('(b) Spurious seeds sit at the sidelobe spacings of the window', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (c) why the peak finder's mask does not catch them ------------------------------
    ax = axs[0, 2]
    f_inj = 1 / spur.nearest_inj
    ok = np.isfinite(f_inj) & np.isfinite(spur.delta_f)
    ax.scatter(f_inj[ok], spur.delta_f[ok], s=12, alpha=.5, lw=0, color=RED, label='spurious component')
    fgrid = np.logspace(-1, np.log10(12), 100)
    ax.plot(fgrid, MASK_FRAC * fgrid, '-', color='black', lw=1.3, label=f'peak-finder mask ±{MASK_FRAC:.0%} of f')
    ax.axhline(1 / T_SECTOR, color='#5f5e5a', ls='--', lw=1, label='1/T_sector')
    ax.axhline(1 / T_ORBIT, color='#5f5e5a', ls='-.', lw=1, label='1/T_orbit')
    ax.fill_between(fgrid, MASK_FRAC * fgrid, 1e-3, color='black', alpha=.06)
    ax.text(.15, 2e-3, 'masked', fontsize=8, color='#3d3d39')
    ax.set(xscale='log', yscale='log', xlabel='frequency of nearest injection (1/day)',
           ylabel='|Δf| of spurious component (1/day)', ylim=(1e-3, 10))
    below = float((spur.delta_f[ok] < 1 / T_ORBIT * 1.5).mean())
    ax.text(.03, .97, f'a ±5% mask is narrower than one\nsidelobe spacing for f < {1 / (T_ORBIT * MASK_FRAC):.1f}/day\n'
                      f'(P > {T_ORBIT * MASK_FRAC:.2f} d), so sidelobes are seeded there',
            transform=ax.transAxes, va='top', fontsize=8.5, color='#3d3d39')
    ax.set_title('(c) The fractional mask is too narrow at low frequency', loc='left', fontweight='bold')
    ax.legend(fontsize=7.5, loc='lower right')

    # (d) period ratio to nearest injection ------------------------------------------
    ax = axs[0, 3]
    r_ = spur.ratio[np.isfinite(spur.ratio)]
    bins = np.logspace(-1.2, 1.2, 70)
    ax.hist(r_, bins=bins, color=RED, alpha=.75)
    for h in (0.5, 1, 2, 3):
        ax.axvline(h, color='black', ls=':', lw=1)
    ax.set(xscale='log', xlabel='P_component / P_nearest injection', ylabel='spurious components')
    near = float((np.abs(r_ - 1) < 0.3).mean())
    ax.text(.97, .97, f'{near:.0%} within 30% of an injected period\n(the 10% match tolerance is a frequency\nwindow of 0.1·f: narrower than a sidelobe\nspacing for P > {T_ORBIT * 0.1:.1f} d)',
            transform=ax.transAxes, va='top', ha='right', fontsize=8.5, color='#3d3d39')
    ax.set_title('(d) Spurious components are near-duplicates, not new periods', loc='left', fontweight='bold')

    # (e) purity against crowding ----------------------------------------------------
    ax = axs[1, 0]
    g = rt.groupby('n_inj_band').agg(seeds=('n_seeds', 'sum'), matched=('n_seeds_matched', 'sum'), rows=('row', 'size'))
    g['spurious'] = g.seeds - g.matched
    ax.bar(g.index, g.matched, color=GREEN, label='matched')
    ax.bar(g.index, g.spurious, bottom=g.matched, color=RED, label='spurious')
    for n, row in g.iterrows():
        pur = f'{row.matched / row.seeds:.0%}' if row.seeds else 'no seeds'
        ax.text(n, row.seeds + 3, f'{pur}\n({int(row.rows)} rows)', ha='center', fontsize=7.5, color='#3d3d39')
    ax.set(xlabel='injected oscillators in band', ylabel='fitted components')
    ax.set_title('(e) Purity by crowding (label = purity, rows)', loc='left', fontweight='bold')
    ax.legend(fontsize=8)

    # (f) purity against sector count ------------------------------------------------
    ax = axs[1, 1]
    g = rt.groupby('n_sectors').agg(seeds=('n_seeds', 'sum'), matched=('n_seeds_matched', 'sum'),
                                     inj=('n_inj_band', 'sum'), rows=('row', 'size'))
    x = np.arange(len(g))
    ax.bar(x - .2, g.matched / g.seeds, width=.4, color=GREEN, label='purity')
    ax.bar(x + .2, g.seeds / g.inj, width=.4, color=GREY, label='seeds per injection')
    for k, (ns, row) in enumerate(g.iterrows()):
        ax.text(k - .2, row.matched / row.seeds + .02, f'{row.matched / row.seeds:.0%}', ha='center', fontsize=8)
        ax.text(k + .2, row.seeds / row.inj + .02, f'{row.seeds / row.inj:.2f}', ha='center', fontsize=8)
    ax.set_xticks(x, [f'{ns} sector{"s" if ns > 1 else ""}\n({int(row.rows)} rows)' for ns, row in g.iterrows()])
    ax.set(ylim=(0, max(1.1, float((g.seeds / g.inj).max()) * 1.15)))
    ax.set_title('(f) Multi-sector windows seed more sidelobes', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')

    # (g) the GP already knows: amplitude of spurious vs matched -----------------------
    ax = axs[1, 2]
    bins = np.logspace(-5, 2, 50)
    ax.hist(fp.snr[fp.matched].clip(1e-5), bins=bins, color=GREEN, alpha=.65, label='matched')
    ax.hist(fp.snr[~fp.matched].clip(1e-5), bins=bins, color=RED, alpha=.65, label='spurious')
    ax.axvline(SNR_MIN, color='black', ls='--', lw=1.2)
    ax.text(SNR_MIN * 1.2, ax.get_ylim()[1] * .9, f'filter σ/WN ≥ {SNR_MIN:g}', fontsize=8)
    ax.set(xscale='log', yscale='log', xlabel='σ / white noise', ylabel='components')
    kept = fp[fp.survives_filter]
    ax.text(.03, .97, f'after filter: {len(kept)} components, purity {kept.matched.mean():.1%}\n'
                      f'the GP gives duplicate seeds ~no amplitude',
            transform=ax.transAxes, va='top', fontsize=8.5, color='#3d3d39')
    ax.set_title('(g) The fit itself separates them: duplicates get no amplitude', loc='left', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # (h) example periodogram --------------------------------------------------------
    ax = axs[1, 3]
    ex = example_rows(df, fp, n=1)
    if ex:
        r = ex[0]
        f = np.asarray(r['LSP_freq'], float); p = np.asarray(r['LSP_power'], float)
        cont = 10.0 ** float(r['rn_log10_N']) * f ** (-float(r['rn_alpha']))
        gam = -np.log(1 - 0.99 ** (1 / len(f)))
        ax.plot(f, p, color='#3d3d39', lw=.7, label='LSP')
        ax.plot(f, gam * cont, color=GREY, lw=1, ls='--', label='significance threshold')
        inj = np.atleast_1d(np.asarray(r['periods'], float))
        for Pi in inj:
            ax.axvline(1 / Pi, color=GREEN, lw=.9, alpha=.6)
        comps = np.atleast_1d(np.asarray(r['gp_periods'], float))
        # match the components of THIS row (a cluster can have several single-sector rows,
        # so selecting fp by name would mix them)
        matched = [bool(len(inj)) and bool(np.any(np.abs(inj / per - 1) < TOL)) for per in comps]
        for per, m in zip(comps, matched):
            fc = 1 / per
            ax.plot(fc, np.interp(fc, f, p), marker='v', ms=7, lw=0, color=GREEN if m else RED,
                    markeredgecolor='black', markeredgewidth=.4)
        ax.set(xscale='log', yscale='log', xlabel='frequency (1/day)', ylabel='LSP power',
               ylim=(np.nanpercentile(p, 5), np.nanmax(p) * 3))
        n_sp = int(len(comps) - sum(matched))
        ax.set_title(f'(h) {r["name"]}, {r["n_sectors"]} sector(s): {len(inj)} injected, '
                     f'{len(comps)} seeded, {n_sp} spurious', loc='left', fontweight='bold')
        ax.legend(handles=[plt.Line2D([], [], color=GREEN, lw=1, label='injected frequency'),
                           plt.Line2D([], [], marker='v', ls='', color=GREEN, label='seed, matched'),
                           plt.Line2D([], [], marker='v', ls='', color=RED, label='seed, spurious'),
                           plt.Line2D([], [], color=GREY, ls='--', label='threshold')],
                  fontsize=7.5, loc='lower left')

    fig.suptitle('Why raw GP component purity is 68%: the peak finder seeds window-function '
                 'sidelobes of real signals, and the fit keeps every seed',
                 fontsize=14, fontweight='bold', x=.005, ha='left')
    fig.text(.006, .002, stamp, fontsize=7.5, color=GREY)
    plt.tight_layout(rect=(0, .01, 1, .97))
    out = os.path.join(args.figdir, 'purity_diagnostics.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)

    # numbers for the docs
    print(f'seeds {rt.n_seeds.sum()}, in-band injections {rt.n_inj_band.sum()}, '
          f'seed purity {seed_pur:.3f}, seeds/injection {rt.n_seeds.sum() / rt.n_inj_band.sum():.2f}')
    print(f'spurious with 0.5/T_sector < df < 2.5/T_orbit: {frac_sl:.3f}; within 30% in period: {near:.3f}')
    print('spurious kinds:'); print(spur.kind.value_counts().to_string())
    print('by n_sectors:'); print(g.assign(purity=g.matched / g.seeds, seeds_per_inj=g.seeds / g.inj).round(3).to_string())
    return rt, fp


if __name__ == '__main__':
    main()

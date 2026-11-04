"""Figures for the synthetic-population light-curve set.

Every figure is regenerated from the truth table + star-parameter files + FITS on disk,
and carries its provenance (data dir, number of clusters, timestamp) in a footer.

    python -m sythn_real_variability.make_figs [--data-dir ...] [--out-lc-dir ...]
                                               [--ages 7.0 8.0 9.5] [--feh 0.0] [--inst 0]

Figures (written to ``results/``):

* ``fig_lc_examples.png``        clean signal vs TESS-sampled noisy light curve, one sector,
                                 with the construction parameters of each cluster
* ``fig_lc_full_baseline.png``   one cluster over every sector, clean vs noisy
* ``fig_population_hr.png``      HR diagrams of the example populations coloured by status
* ``fig_missing_light.png``      fraction of light the empirical grid cannot inform, vs age
* ``fig_variability_vs_age.png`` clean rms and variable flux fraction vs age
* ``fig_empirical_grid.png``     the empirical grid (counts, variable fraction, A, P)
* ``fig_variability_draws.png``  drawn amplitudes/periods vs the catalogue
* ``fig_construction_parameters.png`` + ``construction_parameters.md``  the run configuration
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
from matplotlib.patches import Rectangle
from astropy.io import fits

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, '..')), os.path.abspath(os.path.join(_HERE, '..', '..'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sythn_real_variability.config import RunConfig, TESS_ZP, LMC_DISTANCE_MODULUS            # noqa: E402
from sythn_real_variability import populations as P                                            # noqa: E402
from sythn_real_variability import variability as V                                            # noqa: E402
from sythn_real_variability.empirical import get_empirical_grid                                # noqa: E402
from sythn_real_variability import tess_sampling as T                                          # noqa: E402


# ---------------------------------------------------------------------------
# Style (same conventions as ../make_figs.py; categorical hues in fixed order)
# ---------------------------------------------------------------------------

SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6', '#e34948']
INK, INK2, GRID, SURFACE, MUTED = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb', '#8f8e88'
STATUS_COLOR = {'variable': SERIES[0], 'quiet': MUTED, 'unsampled_sparse': SERIES[1],
                'unsampled_empty': SERIES[3], 'outside_grid': SERIES[6]}
FEH_COLOR = {-1.0: SERIES[0], -0.5: SERIES[1], 0.0: SERIES[2]}

plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': SURFACE,
                     'axes.grid': True, 'grid.color': GRID, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'legend.frameon': False,
                     'figure.dpi': 120, 'font.size': 9.5})


def _footer(fig, cfg, truth, extra=''):
    fig.text(0.005, 0.004, f'sythn_real_variability | data: {os.path.basename(cfg.paths.data_dir)} | '
             f'{len(truth)} clusters | {datetime.now():%Y-%m-%d %H:%M} {extra}',
             fontsize=7, color=INK2, ha='left', va='bottom')


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def load_truth(cfg):
    return pd.read_pickle(cfg.paths.truth_file)


def pick_examples(truth, ages, feh, inst):
    rows = []
    for a in ages:
        m = (np.isclose(truth['age'], a)) & (np.isclose(truth['feh'], feh)) & (truth['inst'] == inst)
        if m.any():
            rows.append(truth[m].iloc[0])
    if not rows:                                 # whatever exists
        rows = [truth.iloc[i] for i in np.linspace(0, len(truth) - 1, min(3, len(truth))).astype(int)]
    return rows


def rebuild_signal(row):
    df, _ = P.load_components(row['star_params_path'])
    return V.build_clean_signal(df), df


def read_sectors(path):
    out = []
    with fits.open(path) as h:
        for hdu in h[1:]:
            if hdu.data is None:
                continue
            out.append(dict(sector=int(hdu.header.get('SECTOR', -1)),
                            E=float(hdu.header.get('SYN_EXPD', np.nan)),
                            t=np.asarray(hdu.data['time'], float), f=np.asarray(hdu.data['flux'], float),
                            e=np.asarray(hdu.data['flux_err'], float)))
    return out


def params_text(row, cfg):
    pc = cfg.population
    lines = [
        f"log age {row['age']:.2f}   [Fe/H] {row['feh']:+.2f}   instance {int(row['inst'])}",
        f"{int(row.get('n_systems', pc.n_systems)):,} systems -> {int(row['n_components']):,} stars; "
        f"M_init = {row.get('total_initial_mass_msun', np.nan):,.0f} Msun",
        f"IMF {pc.imf} {pc.imf_mass_bounds[0]}-{pc.imf_mass_bounds[1]:.0f} Msun, f_B = {pc.binary_fraction}, "
        f"gamma = {pc.mass_ratio_gamma}; MIST via {row.get('source', '?')}, BC_T {row.get('bc_source', '?')}",
        f"d = {pc.distance_pc / 1e3:.2f} kpc (DM {5 * np.log10(pc.distance_pc / 10):.2f}), A_V = {pc.A_V}; "
        f"integrated Tmag {row['Tmag_syn']:.2f}",
        f"template {row['template_name']} (Tmag {row['Tmag_template']:.2f}, F_syn/F_tmpl {row['flux_ratio']:.2f}); "
        f"noise = template flux_err, {cfg.noise.brightness_mode} flux",
        f"variable {int(row['n_var']):,} stars = {row['fluxfrac_var']:.0%} of flux; "
        f"{int(row['n_sinusoids']):,} sinusoids, P {row['period_min']:.3f}-{row['period_max']:.1f} d",
        f"missing light {row['fluxfrac_missing']:.1%}  (outside grid {row['fluxfrac_outside_grid']:.1%}, "
        f"sparse bins {row['fluxfrac_unsampled_sparse']:.1%}, empty bins {row['fluxfrac_unsampled_empty']:.1%})",
        f"clean rms {row['clean_rms'] * 1e3:.3f} ppt; red noise alpha {row['rn_alpha']:.2f}, "
        f"log10N {row['rn_log10_N_target']:.2f}, rms {row['rn_rms_realized'] * 1e3:.2f} ppt",
        f"grid {cfg.grid.teff_min:.0f}-{cfg.grid.teff_max:.0f} K x {cfg.grid.teff_step:.0f} K, "
        f"logL {cfg.grid.logl_min}-{cfg.grid.logl_max} x {cfg.grid.logl_step}; min {cfg.variability.min_bin_stars} stars/bin; "
        f"method {cfg.variability.method}, fallback {cfg.variability.fallback}; seed {int(row['seed'])}",
    ]
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_lc_examples(rows, cfg, truth, out, prefer_exposure_min=30.0):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.6 * n), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1.35], wspace=0.08, hspace=0.55))
    for i, row in enumerate(rows):
        sig, _ = rebuild_signal(row)
        secs = read_sectors(row['path'])
        cand = [s for s in secs if abs(s['E'] * 1440 - prefer_exposure_min) < 2] or secs
        s = max(cand, key=lambda d: np.isfinite(d['f']).sum())
        g = np.isfinite(s['t']) & np.isfinite(s['f'])
        t, f = s['t'][g], s['f'][g]
        med = np.median(f)
        tf = np.arange(t.min(), t.max(), 2 / 1440)
        clean_fine = (sig(tf, 0.0) - 1) * 1e3
        clean_obs = (sig(t, s['E']) - 1) * 1e3
        noisy = (f / med - 1) * 1e3

        ax = axes[i, 0]
        ax.plot(tf, clean_fine, color=SERIES[0], lw=0.9, label='clean signal (2-min grid)')
        ax.set_ylabel('flux - 1  [ppt]')
        ax.set_title(f"{row['name']}  -  pre-noise, sector {s['sector']}", loc='left', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax = axes[i, 1]
        ax.plot(t, noisy, '.', ms=2.2, color=INK2, alpha=0.6, label=f"TESS-sampled + noise ({s['E'] * 1440:.0f} min)")
        ax.plot(t, clean_obs, color=SERIES[0], lw=1.0, label='clean, exposure-averaged')
        ax.set_title('post-noise', loc='left', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.text(1.02, 0.98, params_text(row, cfg), transform=ax.transAxes, va='top', ha='left',
                fontsize=7.3, family='monospace', color=INK,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=GRID))
        for ax in axes[i]:
            ax.set_xlabel('time  [BTJD]')
    fig.subplots_adjust(top=0.9 if n == 1 else 0.95)
    fig.suptitle('Synthetic cluster light curves: population signal before and after TESS sampling + noise',
                 x=0.01, ha='left', fontsize=12, y=0.985)
    _footer(fig, cfg, truth)
    _save(fig, out)


def fig_lc_full_baseline(row, cfg, truth, out):
    sig, _ = rebuild_signal(row)
    secs = read_sectors(row['path'])
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    for k, s in enumerate(secs):
        g = np.isfinite(s['t']) & np.isfinite(s['f'])
        t, f = s['t'][g], s['f'][g]
        med = np.median(f)
        axes[0].plot(t, (sig(t, s['E']) - 1) * 1e3, color=SERIES[0], lw=0.6)
        axes[1].plot(t, (f / med - 1) * 1e3, '.', ms=1.2, color=INK2, alpha=0.5)
        axes[1].annotate(f"S{s['sector']}", (t.mean(), 1.0 + 0.06 * (k % 2)), xycoords=('data', 'axes fraction'),
                         ha='center', va='bottom', fontsize=6.5, color=INK2)
    axes[0].set_ylabel('clean flux - 1  [ppt]')
    axes[1].set_ylabel('observed flux / median - 1  [ppt]')
    axes[1].set_xlabel('time  [BTJD]')
    expo = sorted(set(float(np.round(e, 1)) for e in row['exposure_min_list']))
    axes[0].set_title(f"{row['name']}: every sector of template {row['template_name']} "
                      f"(exposures {expo} min)", loc='left', fontsize=10.5)
    fig.text(0.125, -0.02, params_text(row, cfg), va='top', ha='left', fontsize=7.2, family='monospace',
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=GRID))
    _footer(fig, cfg, truth)
    _save(fig, out)


def fig_population_hr(rows, grid, cfg, truth, out):
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.2), squeeze=False, sharey=True)
    sampled = grid.map2d(grid.sampled)
    for ax, row in zip(axes[0], rows):
        _, df = rebuild_signal(row)
        # sampled-bin shading
        ax.pcolormesh(grid.teff_edges, grid.logl_edges, np.where(sampled, 1.0, np.nan),
                      cmap=matplotlib.colors.ListedColormap(['#e8f1fb']), vmin=0, vmax=1, shading='flat', zorder=0)
        ax.add_patch(Rectangle((grid.teff_edges[0], grid.logl_edges[0]), grid.teff_edges[-1] - grid.teff_edges[0],
                               grid.logl_edges[-1] - grid.logl_edges[0], fill=False, ec=INK2, lw=0.8, ls='--', zorder=1))
        F = df['F_es'].to_numpy(); Ft = F.sum()
        for code, name in V.STATUS.items():
            m = df['status'].to_numpy() == code
            if not m.any():
                continue
            ax.scatter(df.loc[m, 'Teff'], df.loc[m, 'logL'], s=4, color=STATUS_COLOR[name], alpha=0.75, lw=0,
                       label=f"{name.replace('_', ' ')}: {m.sum():,} stars, {F[m].sum() / Ft:.0%} of flux", zorder=2)
        ax.set_xscale('log'); ax.invert_xaxis()
        ax.set_xlim(60000, 2500)
        ax.set_xlabel('Teff  [K]')
        ax.set_title(f"log age {row['age']:.2f}, [Fe/H] {row['feh']:+.1f}: Tmag {row['Tmag_syn']:.2f}, "
                     f"missing light {row['fluxfrac_missing']:.0%}", loc='left', fontsize=9.5)
        ax.legend(loc='lower left', fontsize=7.2, markerscale=3)
        ax.grid(True, which='both')
    axes[0, 0].set_ylabel('log L / Lsun')
    fig.suptitle('Synthetic populations on the empirical grid (dashed = grid extent, shaded = bins with '
                 f'>= {cfg.variability.min_bin_stars} catalogue stars)', x=0.01, ha='left', fontsize=11)
    _footer(fig, cfg, truth)
    _save(fig, out)


def _band(ax, g, col, color, label):
    med = g[col].median().reset_index(); lo = g[col].quantile(.16).reset_index(); hi = g[col].quantile(.84).reset_index()
    ax.fill_between(med['age'], lo[col], hi[col], color=color, alpha=0.18, lw=0)
    ax.plot(med['age'], med[col], color=color, lw=2, marker='o', ms=4, label=label)


def fig_missing_light(truth, cfg, out, templates=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(wspace=0.28))
    for feh in sorted(truth['feh'].unique()):
        sub = truth[np.isclose(truth['feh'], feh)]
        g = sub.groupby('age')
        c = FEH_COLOR.get(float(feh), INK2)
        _band(axes[0], g, 'fluxfrac_missing', c, f'[Fe/H] = {feh:+.1f}')
        _band(axes[1], g, 'fluxfrac_outside_grid', c, f'[Fe/H] = {feh:+.1f}')
        _band(axes[2], g, 'Tmag_syn', c, f'[Fe/H] = {feh:+.1f}')
    axes[0].set_ylabel('fraction of TESS flux in unsampled or out-of-grid bins')
    axes[0].set_title('Light the empirical grid cannot inform ("missing light")', loc='left', fontsize=10)
    axes[1].set_ylabel('fraction of TESS flux outside the grid (Teff > 12,500 K)')
    axes[1].set_title('...of which: hot stars beyond the catalogue', loc='left', fontsize=10)
    axes[2].set_ylabel('integrated TESS magnitude at the LMC')
    axes[2].set_title(f'Brightness of a {cfg.population.n_systems:,}-system population', loc='left', fontsize=10)
    axes[2].invert_yaxis()
    if templates is not None and len(templates):
        lo, hi = np.percentile(templates['Tmag'], [5, 95])
        axes[2].axhspan(lo, hi, color=MUTED, alpha=0.15, lw=0)
        axes[2].text(axes[2].get_xlim()[0] + 0.05, hi - 0.05, f'real {cfg.noise.template_origin} clusters (p5-p95)',
                     fontsize=7.5, color=INK2, va='bottom')
    for ax in axes:
        ax.set_xlabel('log age  [yr]')
        ax.legend(fontsize=8)
    for ax in axes[:2]:
        ax.set_ylim(0, 1)
    fig.text(0.01, -0.04, 'Bands: 16th-84th percentile over instances. Stars in unsampled bins are kept at constant '
             'flux; the number reported is the share of light whose variability the model cannot describe.',
             fontsize=8, color=INK2)
    _footer(fig, cfg, truth)
    _save(fig, out)


def fig_variability_vs_age(truth, cfg, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(wspace=0.28))
    truth = truth.assign(clean_rms_ppt=truth['clean_rms'] * 1e3)
    for feh in sorted(truth['feh'].unique()):
        sub = truth[np.isclose(truth['feh'], feh)]
        g = sub.groupby('age')
        c = FEH_COLOR.get(float(feh), INK2)
        _band(axes[0], g, 'clean_rms_ppt', c, f'[Fe/H] = {feh:+.1f}')
        _band(axes[1], g, 'fluxfrac_var', c, f'[Fe/H] = {feh:+.1f}')
        _band(axes[2], g, 'period_of_max_amp', c, f'[Fe/H] = {feh:+.1f}')
    axes[0].set_ylabel('rms of the clean integrated signal  [ppt]')
    axes[0].set_title('Predicted integrated variability amplitude', loc='left', fontsize=10)
    axes[0].set_yscale('log')
    axes[1].set_ylabel('fraction of TESS flux from variable stars'); axes[1].set_ylim(0, 1)
    axes[1].set_title('Variable light fraction', loc='left', fontsize=10)
    axes[2].set_ylabel('period of the strongest sinusoid  [d]'); axes[2].set_yscale('log')
    axes[2].set_title('Dominant period', loc='left', fontsize=10)
    for ax in axes:
        ax.set_xlabel('log age  [yr]'); ax.legend(fontsize=8)
    _footer(fig, cfg, truth)
    _save(fig, out)


def fig_empirical_grid(grid, cfg, truth, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw=dict(wspace=0.25, hspace=0.3))
    panels = [('number of catalogue stars (log10)', np.log10(np.where(grid.n_total > 0, grid.n_total, np.nan)), None),
              ('variable fraction', grid.frac_var, (0, 1)),
              ('median amplitude A1 (log10 ppm)', np.log10(grid.stats['A1']['med']), None),
              ('median period P1 (log10 d)', np.log10(grid.stats['P1']['med']), None)]
    unsampled = grid.map2d((grid.n_total > 0) & ~grid.sampled)
    for ax, (title, vals, rng) in zip(axes.ravel(), panels):
        img = grid.map2d(vals)
        kw = dict(vmin=rng[0], vmax=rng[1]) if rng else {}
        pm = ax.pcolormesh(grid.teff_edges, grid.logl_edges, img, cmap='Blues', shading='flat', **kw)
        ax.pcolormesh(grid.teff_edges, grid.logl_edges, np.where(unsampled, 1.0, np.nan),
                      cmap=matplotlib.colors.ListedColormap([SERIES[1]]), vmin=0, vmax=1, alpha=0.45, shading='flat')
        fig.colorbar(pm, ax=ax, fraction=0.04, pad=0.02)
        ax.invert_xaxis(); ax.set_xlabel('Teff  [K]'); ax.set_ylabel('log L / Lsun')
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_ylim(-3.2, 5.0)
    fig.suptitle(f'Empirical TESS variability grid ({len(grid.sample):,} stars; orange = populated bins with '
                 f'< {cfg.variability.min_bin_stars} stars, treated as unsampled)', x=0.01, ha='left', fontsize=11)
    _footer(fig, cfg, truth)
    _save(fig, out)


def fig_variability_draws(rows, grid, cfg, truth, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(wspace=0.25))
    s = grid.sample; isvar = s['cls'] == 'var'
    bA = np.linspace(1, 6, 51); bP = np.linspace(-2.5, 1.2, 51)
    axes[0].hist(np.log10(s.loc[isvar, 'A1']), bins=bA, color=MUTED, alpha=0.4, density=True, label='catalogue variables')
    axes[1].hist(np.log10(s.loc[isvar, 'P1']), bins=bP, color=MUTED, alpha=0.4, density=True, label='catalogue variables')
    for k, row in enumerate(rows):
        _, df = rebuild_signal(row)
        v = df[df['is_var']]
        c = SERIES[k % len(SERIES)]
        lab = f"log age {row['age']:.2f} ({len(v):,} variable stars)"
        axes[0].hist(np.log10(v['A1'] * 1e6), bins=bA, histtype='step', lw=1.6, color=c, density=True, label=lab)
        axes[1].hist(np.log10(v['P1']), bins=bP, histtype='step', lw=1.6, color=c, density=True, label=lab)
    axes[0].set_xlabel('log10 amplitude  [ppm]'); axes[1].set_xlabel('log10 period  [d]')
    axes[0].set_ylabel('density')
    axes[0].set_title('Drawn sinusoid amplitudes', loc='left', fontsize=10)
    axes[1].set_title('Drawn periods', loc='left', fontsize=10)
    for ax in axes:
        ax.legend(fontsize=8)
    _footer(fig, cfg, truth)
    _save(fig, out)


def construction_parameters_md(cfg, truth):
    d = cfg.to_dict()
    lines = ['# Construction parameters', '',
             f'Written by `sythn_real_variability.make_figs` on {datetime.now():%Y-%m-%d %H:%M}; '
             f'{len(truth)} clusters in the truth table.', '']
    for section in ('grid', 'population', 'variability', 'noise', 'constants'):
        lines += [f'## {section}', '', '| parameter | value |', '|---|---|']
        for k, v in d[section].items():
            if isinstance(v, (list, tuple)) and len(v) > 6:
                v = f'{v[0]} .. {v[-1]} ({len(v)} values)'
            lines.append(f'| `{k}` | {v} |')
        lines.append('')
    lines += ['## run', '', f'| `seed` | {d["seed"]} |', f'| `name_prefix` | {d["name_prefix"]} |', '']
    if len(truth):
        ages = [float(a) for a in sorted(truth['age'].unique())]
        fehs = [float(f) for f in sorted(truth['feh'].unique())]
        lines += ['## realised set', '', '| quantity | value |', '|---|---|',
                  f"| clusters | {len(truth)} |",
                  f"| log ages | {ages[0]} .. {ages[-1]} ({len(ages)} values) |" if len(ages) > 6 else f"| log ages | {ages} |",
                  f"| [Fe/H] | {fehs} |",
                  f"| instances per population | {truth.groupby(['age', 'feh']).size().min()}-{truth.groupby(['age', 'feh']).size().max()} |",
                  f"| population source | {sorted(truth['source'].unique())} |",
                  f"| BC source | {sorted(truth['bc_source'].unique())} |",
                  f"| integrated Tmag | {truth['Tmag_syn'].min():.2f}-{truth['Tmag_syn'].max():.2f} |",
                  f"| variable stars per cluster | {truth['n_var'].min()}-{truth['n_var'].max()} |",
                  f"| clean rms [ppt] | {truth['clean_rms'].min() * 1e3:.3f}-{truth['clean_rms'].max() * 1e3:.3f} |",
                  f"| missing light fraction | {truth['fluxfrac_missing'].min():.1%}-{truth['fluxfrac_missing'].max():.1%} |", '']
    return '\n'.join(lines)


def fig_construction_parameters(cfg, truth, out):
    md = construction_parameters_md(cfg, truth)
    text = [l for l in md.splitlines() if not l.startswith('|---') and l.strip() and not l.startswith('Written by')]
    text = [l.replace('| ', '').replace(' |', '').replace('`', '').replace('# ', '') for l in text]
    fig = plt.figure(figsize=(9, 0.125 * len(text) + 0.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.text(0.02, 0.98, '\n'.join(text), va='top', ha='left', family='monospace', fontsize=7.6, color=INK,
            transform=ax.transAxes)
    _footer(fig, cfg, truth)
    _save(fig, out)
    return md


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_all(cfg, ages=(7.0, 8.0, 9.5), feh=0.0, inst=0, figdir=None):
    figdir = figdir or cfg.paths.results_dir
    truth = load_truth(cfg)
    grid, _ = get_empirical_grid(cfg, verbose=False)
    templates = T.list_templates(cfg.paths.real_lc_dir, cfg.noise.template_origin,
                                 cache_csv=os.path.join(cfg.paths.data_dir, f'templates_{cfg.noise.template_origin}.csv'))
    rows = pick_examples(truth, ages, feh, inst)
    fig_lc_examples(rows, cfg, truth, os.path.join(figdir, 'fig_lc_examples.png'))
    fig_lc_full_baseline(rows[min(1, len(rows) - 1)], cfg, truth, os.path.join(figdir, 'fig_lc_full_baseline.png'))
    fig_population_hr(rows, grid, cfg, truth, os.path.join(figdir, 'fig_population_hr.png'))
    fig_missing_light(truth, cfg, os.path.join(figdir, 'fig_missing_light.png'), templates)
    fig_variability_vs_age(truth, cfg, os.path.join(figdir, 'fig_variability_vs_age.png'))
    fig_empirical_grid(grid, cfg, truth, os.path.join(figdir, 'fig_empirical_grid.png'))
    fig_variability_draws(rows, grid, cfg, truth, os.path.join(figdir, 'fig_variability_draws.png'))
    md = fig_construction_parameters(cfg, truth, os.path.join(figdir, 'fig_construction_parameters.png'))
    with open(os.path.join(figdir, 'construction_parameters.md'), 'w') as f:
        f.write(md)
    print(f'  wrote {os.path.join(figdir, "construction_parameters.md")}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out-lc-dir', default=None)
    ap.add_argument('--figdir', default=None)
    ap.add_argument('--ages', type=float, nargs='*', default=(7.0, 8.0, 9.5))
    ap.add_argument('--feh', type=float, default=0.0)
    ap.add_argument('--inst', type=int, default=0)
    a = ap.parse_args(argv)
    cfg = RunConfig.from_json(a.config) if a.config else RunConfig()
    if a.data_dir:
        cfg.paths.data_dir = os.path.abspath(a.data_dir)
    if a.out_lc_dir:
        cfg.paths.out_lc_dir = os.path.abspath(a.out_lc_dir)
    make_all(cfg, ages=a.ages, feh=a.feh, inst=a.inst, figdir=a.figdir)


if __name__ == '__main__':
    main()

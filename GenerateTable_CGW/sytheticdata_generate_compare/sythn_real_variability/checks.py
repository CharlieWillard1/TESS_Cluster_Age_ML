"""Physical-robustness checks for the synthetic-population light curves.

Runs against whatever has been generated (``data_dir`` / ``out_lc_dir`` from the config or
command line) and writes ``results/checks_report.md``.  Each check prints PASS / WARN / FAIL
with the numbers that decided it, so a reader can judge the thresholds.

    python -m sythn_real_variability.checks [--data-dir ...] [--out-lc-dir ...]
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import pandas as pd
from astropy.io import fits

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, '..')), os.path.abspath(os.path.join(_HERE, '..', '..'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sythn_real_variability.config import (RunConfig, TESS_ZP, LMC_DISTANCE_PC, LMC_DISTANCE_MODULUS,  # noqa: E402
                                           M_BOL_SUN)
from sythn_real_variability import populations as P                                            # noqa: E402
from sythn_real_variability import variability as V                                            # noqa: E402
from sythn_real_variability.empirical import get_empirical_grid                                # noqa: E402
from sythn_real_variability import tess_sampling as T                                          # noqa: E402


class Report:
    def __init__(self):
        self.lines = []
        self.n = dict(PASS=0, WARN=0, FAIL=0)

    def add(self, status, title, detail):
        self.n[status] += 1
        self.lines.append(f'- **{status}** {title}  \n  {detail}')
        print(f'[{status}] {title}: {detail}')

    def section(self, title):
        self.lines.append(f'\n## {title}\n')

    def write(self, path, header):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(header + '\n\n')
            f.write(f"**{self.n['PASS']} pass, {self.n['WARN']} warn, {self.n['FAIL']} fail**\n")
            f.write('\n'.join(self.lines) + '\n')


def _status(ok, warn=False):
    return 'PASS' if ok else ('WARN' if warn else 'FAIL')


def run_checks(cfg: RunConfig, out_md=None, verbose=True):
    rep = Report()
    paths = cfg.paths

    # ------------------------------------------------------------------ constants
    rep.section('Constants and photometric conventions')
    f10 = P.tess_flux_from_mag(10.0)
    rep.add(_status(abs(f10 / 15000 - 1) < 0.02), 'TESS zero point',
            f'Tmag 10 -> {f10:.0f} e/s (Sullivan et al. 2015 calibration: 15,000 e/s); zp = {TESS_ZP}')
    rep.add(_status(abs(LMC_DISTANCE_MODULUS - 18.477) < 0.01), 'LMC distance modulus',
            f'd = {LMC_DISTANCE_PC / 1e3:.2f} kpc -> DM = {LMC_DISTANCE_MODULUS:.3f} (Pietrzynski+2019: 18.477)')
    bc_sun = P.bc_tess_poly(np.log10(5772.0), 4.44)
    M_T_sun = M_BOL_SUN - bc_sun
    rep.add(_status(4.0 < M_T_sun < 4.5), 'Solar absolute TESS magnitude from the BC polynomial',
            f'BC_T(sun) = {bc_sun:.3f} -> M_T(sun) = {M_T_sun:.2f} (expected ~4.2-4.4; V = 4.83 and V-T ~ 0.5)')
    try:
        mbc = P.MistBC()
        rep.add(_status(abs(mbc.sun_bc - bc_sun) < 0.1, warn=True), 'BC polynomial vs MIST grid at the Sun',
                f'grid {mbc.sun_bc:.3f} vs polynomial {bc_sun:.3f} (difference {mbc.sun_bc - bc_sun:+.3f} mag)')
    except Exception as e:
        rep.add('WARN', 'MIST BC grid not available in this environment',
                f'{type(e).__name__}; the polynomial fallback is used for magnitudes')
    if os.path.isdir(paths.legacy_isos_dir):
        coef, fit = P.fit_bc_poly(paths.legacy_isos_dir)
        rep.add(_status(fit['rms'] < 0.05), 'BC polynomial reproduces MIST magnitudes of the legacy tables',
                f"n = {fit['n']}, rms = {fit['rms']:.3f} mag, p99 |resid| = {fit['p99']:.3f} mag; "
                f"max |coef change| vs stored = {np.max(np.abs(coef - P.BC_T_POLY_COEF)):.1e}")

    # ------------------------------------------------------------------ signal maths
    rep.section('Signal construction')
    rng = np.random.default_rng(1)
    P_test = np.array([0.1, 0.5, 2.0, 7.0]); E = 30 / 1440
    tfine = np.arange(-E / 2, E / 2, E / 2000) + E / 4000
    num = [np.mean(np.sin(2 * np.pi * (5.0 + tfine) / p + 0.3)) / np.sin(2 * np.pi * 5.0 / p + 0.3) for p in P_test]
    ana = np.sinc(E / P_test)
    rep.add(_status(np.max(np.abs(np.array(num) - ana)) < 1e-4), 'Exposure attenuation is sinc(E/P)',
            'numerical box average vs analytic: ' + ', '.join(f'P={p} d: {n:.5f}/{a:.5f}' for p, n, a in zip(P_test, num, ana)))
    sig = V.CleanSignal(a=rng.uniform(1e-4, 1e-3, 50), P=rng.uniform(0.1, 9, 50), phi=rng.uniform(0, 2 * np.pi, 50))
    t = np.arange(0, 2000, 1 / 48)
    x = sig(t)
    rep.add(_status(abs(x.mean() - 1) < 1e-4 and abs(x.var() / sig.variance() - 1) < 0.03),
            'Clean signal has mean 1 and the analytic variance',
            f'mean - 1 = {x.mean() - 1:+.1e}; numerical/analytic variance = {x.var() / sig.variance():.4f} over 2000 d')

    # ------------------------------------------------------------------ populations
    rep.section('Populations')
    pop_files = sorted(os.listdir(paths.populations_dir)) if os.path.isdir(paths.populations_dir) else []
    if pop_files:
        comp, meta = P.load_components(os.path.join(paths.populations_dir, pop_files[0]))
        Fsum = comp['F_es'].sum()
        rep.add(_status(abs(P.tess_mag_from_flux(Fsum) - meta.get('Tmag_integrated', np.nan)) < 1e-3),
                'Flux conservation (sum of component fluxes = integrated magnitude)',
                f"{pop_files[0]}: sum F = {Fsum:.1f} e/s -> Tmag {P.tess_mag_from_flux(Fsum):.3f}; stored {meta.get('Tmag_integrated')}")
        rep.add(_status(np.isfinite(comp['T_app']).all() and np.isfinite(comp['F_es']).all()),
                'Every surviving component has a finite TESS magnitude',
                f"{len(comp)} components, {int((~np.isfinite(comp['T_app'])).sum())} non-finite "
                f"(the legacy tables had ~50 % NaN TESS_mag); BC source = {meta.get('bc_source')}")
        dm = 5 * np.log10(cfg.population.distance_pc / 10)
        rep.add(_status(abs(np.median(comp['T_app'] - comp['M_T']) - dm) < 1e-6), 'Distance modulus applied',
                f"median(T_app - M_T) = {np.median(comp['T_app'] - comp['M_T']):.3f} vs DM = {dm:.3f}")

    # ------------------------------------------------------------------ empirical grid
    rep.section('Empirical grid')
    grid, _ = get_empirical_grid(cfg, verbose=False)
    s = grid.sample
    isvar = (s['cls'] == 'var').to_numpy()
    rep.add('PASS' if (s.loc[isvar, 'A1'] <= cfg.variability.amp_max_ppm).all() else 'FAIL',
            'Amplitude / period sanity of the catalogue variables',
            f"A1 in [{s.loc[isvar, 'A1'].min():.0f}, {s.loc[isvar, 'A1'].max():.0f}] ppm (cap {cfg.variability.amp_max_ppm:.0f}); "
            f"P1 in [{s.loc[isvar, 'P1'].min():.4f}, {s.loc[isvar, 'P1'].max():.2f}] d")
    cov = grid.sampled[s['bin'].to_numpy()[s['bin'].to_numpy() >= 0]].mean()
    rep.add(_status(cov > 0.95, warn=True), 'Catalogue stars that fall in sampled bins',
            f'{cov:.1%} of in-grid catalogue stars are in bins with >= {cfg.variability.min_bin_stars} stars; '
            f'{int(grid.sampled.sum())} sampled bins of {int((grid.n_total > 0).sum())} populated')
    tm = s['Tmag'].to_numpy(); med = np.nanmedian(tm)
    fv_b = isvar[tm <= med].mean(); fv_f = isvar[tm > med].mean()
    rep.add(_status(abs(fv_b - fv_f) < 0.1, warn=True), 'Variable fraction vs brightness (selection bias probe)',
            f'frac_var = {fv_b:.2f} for Tmag <= {med:.1f}, {fv_f:.2f} for fainter stars. A large difference means '
            f'frac_var is partly a detection-threshold effect, not astrophysics (documented caveat).')
    rep.add('PASS', 'Catalogue magnitude limit',
            f'Tmag range {np.nanmin(tm):.1f}-{np.nanmax(tm):.1f}; the model therefore describes variability of '
            f'bright (Tmag < 14) TESS targets, applied to every synthetic star of the same Teff/logL')

    # ------------------------------------------------------------------ noise convention
    rep.section('Noise convention of the existing synthetic pipeline')
    tmpl_files = sorted(T.glob.glob(os.path.join(paths.real_lc_dir, cfg.noise.template_origin,
                                                 'hlsp_elk_tess_ffi_*_tess_v1_llc.fits')))
    ratios = []
    for f in tmpl_files[:40]:
        with fits.open(f) as h:
            for hdu in h[1:]:
                if hdu.data is None:
                    continue
                fl = np.asarray(hdu.data['flux'], float); e = np.asarray(hdu.data['flux_err'], float)
                g = np.isfinite(fl) & np.isfinite(e)
                if g.sum() < 100:
                    continue
                d = np.diff(fl[g])
                p2p = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
                ratios.append(np.median(e[g]) / p2p)
    ratios = np.array(ratios)
    rep.add(_status(0.7 < np.median(ratios) < 1.3, warn=True),
            'Template flux_err matches the real point-to-point scatter',
            f'median(flux_err) / p2p-scatter over {len(ratios)} real sectors: median {np.median(ratios):.2f}, '
            f'p5-p95 {np.percentile(ratios, 5):.2f}-{np.percentile(ratios, 95):.2f}. The synthetic noise is drawn '
            f'from flux_err, so a ratio near 1 means the injected white noise is the real noise level.')

    # ------------------------------------------------------------------ generated set
    rep.section('Generated light curves')
    if os.path.exists(paths.truth_file):
        tr = pd.read_pickle(paths.truth_file)
        rep.add('PASS', 'Truth table', f'{len(tr)} clusters; ages {tr.age.min()}-{tr.age.max()}, '
                f"[Fe/H] {sorted(tr.feh.unique())}, instances {tr.inst.nunique()}")
        # star params sanity
        bad = 0; n_var = 0; a_big = 0
        for pth in tr['star_params_path'].head(20):
            if not os.path.exists(pth):
                continue
            d, _ = P.load_components(pth)
            iv = d['is_var'].to_numpy(bool)
            n_var += iv.sum()
            bad += int((~np.isfinite(d.loc[iv, 'P1']) | (d.loc[iv, 'P1'] < cfg.variability.period_min_d)
                        | (d.loc[iv, 'P1'] > cfg.variability.period_max_d) | ~np.isfinite(d.loc[iv, 'A1'])
                        | (d.loc[iv, 'A1'] <= 0) | (d.loc[iv, 'A1'] > cfg.variability.amp_max_ppm / 1e6)).sum())
            a_big += int((d.loc[iv, 'A1'] > 0.1).sum())
        rep.add(_status(bad == 0), 'Drawn sinusoid parameters are physical (first 20 clusters)',
                f'{n_var} variable components, {bad} with invalid P or A; {a_big} with A > 10 %')
        # fits structure vs template
        r = tr.iloc[0]
        with fits.open(r['path']) as sh, fits.open(r['template_path']) as th:
            same = len(sh) == len(th)
            for i in range(1, min(len(sh), len(th))):
                a, b = sh[i].data, th[i].data
                same &= list(a.columns.names) == list(b.columns.names)
                same &= np.allclose(np.nan_to_num(a['time']), np.nan_to_num(b['time']))
                same &= np.array_equal(np.isnan(a['flux']), np.isnan(b['time']))
                if cfg.noise.brightness_mode == 'template':
                    same &= np.allclose(np.nan_to_num(a['flux_err']), np.nan_to_num(b['flux_err']))
            rep.add(_status(bool(same)), 'FITS structure identical to the template (HDUs, columns, times, NaN pattern, errors)',
                    f"{os.path.basename(r['path'])} vs template {r['template_name']}: {len(sh)} HDUs")
            a = sh[1].data; g = np.isfinite(a['flux'])
            magchk = np.nanmax(np.abs(a['mag'][g] - (TESS_ZP - 2.5 * np.log10(a['flux'][g]))))
            rep.add(_status(magchk < 1e-6), 'mag column consistent with flux', f'max |mag - (zp - 2.5 log F)| = {magchk:.1e}')
        # pipeline resolution
        from table_pipeline.lc_lsp import get_lc_path
        nres = 0
        for _, rr in tr.iterrows():
            try:
                get_lc_path(rr['name'], rr['origin'], paths.out_lc_dir); nres += 1
            except FileNotFoundError:
                pass
        rep.add(_status(nres == len(tr)), 'Every synthetic cluster resolves through table_pipeline.lc_lsp.get_lc_path',
                f'{nres}/{len(tr)} resolve under {paths.out_lc_dir}')
        # brightness realism
        tmpl = T.list_templates(paths.real_lc_dir, cfg.noise.template_origin,
                                cache_csv=os.path.join(paths.data_dir, f'templates_{cfg.noise.template_origin}.csv'))
        real_med = float(np.median(tmpl['Tmag']))
        per_age = tr.groupby('age')['Tmag_syn'].median()
        n_needed = {a: cfg.population.n_systems * 10 ** (0.4 * (m - real_med)) for a, m in per_age.items()}
        rep.add(_status(tr['Tmag_syn'].median() < tmpl['Tmag'].max() + 1.0, warn=True),
                'Integrated brightness vs real LMC clusters',
                f"synthetic Tmag median {tr['Tmag_syn'].median():.2f} (range {tr['Tmag_syn'].min():.2f}-{tr['Tmag_syn'].max():.2f}) "
                f"vs real {cfg.noise.template_origin} clusters {tmpl['Tmag'].min():.2f}-{tmpl['Tmag'].max():.2f} (median {real_med:.2f}). "
                f"Systems needed to match the median real cluster: " +
                ', '.join(f'log age {a}: {n:,.0f}' for a, n in list(n_needed.items())[::max(1, len(n_needed) // 5)]))
        rep.add(_status((tr['flux_ratio'] > 0.1).all() and (tr['flux_ratio'] < 10).all(), warn=True),
                'Template brightness match', f"F_syn/F_template: median {tr['flux_ratio'].median():.2f}, "
                f"range {tr['flux_ratio'].min():.2f}-{tr['flux_ratio'].max():.2f} (templates are chosen among the "
                f"{cfg.noise.template_k_nearest} brightness-nearest real clusters)")
        # red noise
        if cfg.noise.add_red_noise:
            rr = tr['rn_rms_realized']
            rep.add(_status(np.isfinite(rr).all() and (rr < 0.05).all()), 'Red-noise calibration produced finite, modest amplitudes',
                    f'fractional rms: median {rr.median():.2e}, max {rr.max():.2e}; alpha median {tr.rn_alpha.median():.2f}')
        # missing light
        ml = tr.groupby('age')['fluxfrac_missing'].mean()
        rep.add('WARN' if (ml > 0.2).any() else 'PASS', 'Light the empirical grid cannot inform ("missing light")',
                'mean flux fraction in unsampled/outside bins by log age: ' +
                ', '.join(f'{a}: {v:.0%}' for a, v in ml.items()) +
                '. Those stars are kept at constant flux (the original code dropped them).')
        # variance accounting
        ratio = tr['true_excess_var'] / (tr['clean_rms'] ** 2)
        ratio = ratio[np.isfinite(ratio) & (tr['clean_rms'] > 0)]
        rep.add(_status(((ratio > 0.5) & (ratio < 1.5)).mean() > 0.9, warn=True),
                'Truth variance matches the analytic sinusoid variance',
                f'Var(truth grid)/analytic: median {ratio.median():.3f}, p5-p95 {ratio.quantile(.05):.2f}-{ratio.quantile(.95):.2f} '
                f'(finite 27-d windows undersample long periods, so some scatter is expected)')
    else:
        rep.add('WARN', 'No truth table found', f'{paths.truth_file} missing; run stage "lcs" first')

    out_md = out_md or os.path.join(paths.results_dir, 'checks_report.md')
    rep.write(out_md, '# Physical robustness checks\n\n'
              f'Generated by `sythn_real_variability.checks` on {pd.Timestamp.now():%Y-%m-%d %H:%M}; '
              f'data_dir = `{paths.data_dir}`, out_lc_dir = `{paths.out_lc_dir}`.')
    if verbose:
        print(f'[checks] report -> {out_md}  ({rep.n})')
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--out-lc-dir', default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    cfg = RunConfig.from_json(a.config) if a.config else RunConfig()
    if a.data_dir:
        cfg.paths.data_dir = os.path.abspath(a.data_dir)
    if a.out_lc_dir:
        cfg.paths.out_lc_dir = os.path.abspath(a.out_lc_dir)
    run_checks(cfg, out_md=a.out)


if __name__ == '__main__':
    main()

"""Driver: populations -> variability -> TESS-sampled light curves -> truth table.

Two stages, because they need different Python environments:

    # stage 1 (env synth_pop, has isochrones + MIST grids): draw the stellar populations
    python -m sythn_real_variability.run_generate populations [--n-systems N] [--instances K] ...

    # stage 2 (env tess_gp, the pipeline env): variability, sampling, noise, truth, figures
    python -m sythn_real_variability.run_generate lcs [--limit N] [--no-red-noise] ...

Run both from ``GenerateTable_CGW/sytheticdata_generate_compare`` (or put that directory and
``GenerateTable_CGW`` on ``sys.path``).  Stage 1 falls back to the legacy Dropbox populations
(bootstrap-resampled, re-scaled to the LMC) when ``isochrones`` is not importable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, '..')), os.path.abspath(os.path.join(_HERE, '..', '..'))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sythn_real_variability.config import RunConfig, cluster_name, population_tag       # noqa: E402
from sythn_real_variability import populations as P                                      # noqa: E402


# ---------------------------------------------------------------------------
# Stage 1: populations
# ---------------------------------------------------------------------------

def run_populations(cfg: RunConfig, ages=None, fehs=None, instances=None, overwrite=False,
                    verbose=True):
    ages = list(cfg.population.ages if ages is None else ages)
    fehs = list(cfg.population.fehs if fehs is None else fehs)
    instances = list(range(cfg.population.n_instances) if instances is None else instances)
    bc_func, bc_label = P.get_bc_function(prefer_mist=cfg.population.source != 'legacy', verbose=verbose)
    os.makedirs(cfg.paths.populations_dir, exist_ok=True)
    cfg.to_json(os.path.join(cfg.paths.data_dir, 'run_config_populations.json'))
    metas = []
    t0 = time.time()
    for feh in fehs:
        for age in ages:
            base = None
            for inst in instances:
                path = P.population_file(cfg, age, feh, inst)
                if os.path.exists(path) and not overwrite:
                    if verbose:
                        print(f'  [populations] exists, skipping {os.path.basename(path)}')
                    continue
                comp, meta, systems = P.build_population(age, feh, inst, cfg, bc_func, bc_label,
                                                         base_systems=base, verbose=verbose)
                if base is None:
                    base = systems
                P.save_components(comp, meta, path)
                metas.append(meta)
    if metas:
        pd.DataFrame(metas).to_csv(os.path.join(cfg.paths.data_dir, 'population_summary.csv'), index=False,
                                   mode='a', header=not os.path.exists(os.path.join(cfg.paths.data_dir, 'population_summary.csv')))
    if verbose:
        print(f'[run_populations] {len(metas)} populations written in {time.time() - t0:.0f}s')
    return metas


# ---------------------------------------------------------------------------
# Stage 2: light curves
# ---------------------------------------------------------------------------

def run_lcs(cfg: RunConfig, ages=None, fehs=None, instances=None, limit=None, overwrite=False,
            verbose=True):
    from sythn_real_variability.empirical import get_empirical_grid
    from sythn_real_variability import variability as V
    from sythn_real_variability import tess_sampling as T
    from sytheticdata_generate_compare.synth.ground_truth import truth_statistics

    ages = list(cfg.population.ages if ages is None else ages)
    fehs = list(cfg.population.fehs if fehs is None else fehs)
    instances = list(range(cfg.population.n_instances) if instances is None else instances)
    paths = cfg.paths
    os.makedirs(paths.data_dir, exist_ok=True)
    os.makedirs(paths.star_params_dir, exist_ok=True)
    cfg.to_json(os.path.join(paths.data_dir, 'run_config.json'))
    cfg.to_json(os.path.join(paths.results_dir, 'run_config.json'))

    grid, emp_report = get_empirical_grid(cfg, verbose=verbose)
    if emp_report is not None:
        with open(os.path.join(paths.results_dir, 'empirical_sample_report.json'), 'w') as f:
            json.dump(emp_report, f, indent=2)

    templates = T.list_templates(paths.real_lc_dir, cfg.noise.template_origin,
                                 cache_csv=os.path.join(paths.data_dir, f'templates_{cfg.noise.template_origin}.csv'))
    rn_pairs = T.load_rn_pairs(paths.rn_table, cfg.noise.rn_origin) if cfg.noise.add_red_noise else None
    if verbose:
        print(f'  [lcs] {len(templates)} {cfg.noise.template_origin} templates; '
              f'red noise: {"on, %d real pairs" % len(rn_pairs) if rn_pairs is not None else "off"}')

    rows, n_done = [], 0
    t_start = time.time()
    for feh in fehs:
        for age in ages:
            for inst in instances:
                if limit is not None and n_done >= limit:
                    break
                name = cluster_name(cfg.name_prefix, age, feh, inst)
                assert T.slug_ok(name), name
                pop_path = P.population_file(cfg, age, feh, inst)
                if not os.path.exists(pop_path):
                    print(f'  [lcs] MISSING population {pop_path}; run stage "populations" first')
                    continue
                out_path = T.output_path(paths.out_lc_dir, cfg.noise.template_origin, name)
                if os.path.exists(out_path) and not overwrite:
                    if verbose:
                        print(f'  [lcs] exists, skipping {name}')
                    continue
                seed = int(np.random.SeedSequence([cfg.seed + 1, int(round(age * 100)),
                                                   int(round((feh + 5) * 100)), inst]).generate_state(1)[0])
                rng = np.random.default_rng(seed)

                comp, meta = P.load_components(pop_path)
                assigned = V.assign_variability(comp, grid, cfg.variability, rng)
                budget = V.light_budget(assigned)
                signal = V.build_clean_signal(assigned)
                F_syn = float(np.nansum(assigned['F_es']))
                tmpl = T.select_template(F_syn, templates, rng, cfg.noise)
                rn_pair = rn_pairs[rng.integers(0, len(rn_pairs))] if rn_pairs is not None else None
                extra = {'SYN_AGE': (float(age), 'log10 age [yr]'), 'SYN_FEH': (float(feh), '[Fe/H]'),
                         'SYN_INST': (int(inst), 'instance'), 'SYN_NSYS': (int(meta.get('n_systems', -1)), 'systems drawn'),
                         'SYN_MASS': (float(meta.get('total_initial_mass_msun', np.nan)), 'total initial mass [Msun]'),
                         'SYN_DIST': (float(cfg.population.distance_pc), 'distance [pc]'),
                         'SYN_TMAG': (float(P.tess_mag_from_flux(F_syn)), 'integrated TESS mag'),
                         'SYN_NVAR': (int(budget['n_var']), 'variable components'),
                         'SYN_MISS': (float(budget['fluxfrac_missing']), 'flux fraction in unsampled/outside bins'),
                         'SYN_SEED': (seed, 'rng seed')}
                rec, tt, xx = T.write_sampled_fits(tmpl['path'], out_path, name, age, signal, F_syn,
                                                   cfg.noise, rng, rn_pair=rn_pair, header_extra=extra)

                # star-level parameters, for reproducibility and the figures
                sp_path = os.path.join(paths.star_params_dir, f'{name}.fits')
                P.save_components(assigned, dict(name=name, **{k: v for k, v in meta.items()
                                                                if isinstance(v, (int, float, str))}), sp_path)

                row = dict(name=name, origin=cfg.noise.template_origin, age=float(age), feh=float(feh), inst=int(inst),
                           seed=seed, F_syn_es=F_syn, Tmag_syn=float(P.tess_mag_from_flux(F_syn)),
                           star_params_path=sp_path, population_path=pop_path)
                row.update({k: v for k, v in meta.items() if k not in ('age', 'feh', 'inst', 'seed')})
                row.update(budget)
                row.update(V.signal_summary(signal))
                sectors = rec.pop('sectors')
                row.update(rec)
                row['exposure_min_list'] = [s['exposure_min'] for s in sectors]
                e = rec.get('err_frac_med', np.nan)
                err = np.full(len(xx), e) if np.isfinite(e) else None
                row.update(truth_statistics(tt, xx, err))
                rows.append(row)
                n_done += 1
                if verbose:
                    print(f"  [lcs] {name}: Tmag={row['Tmag_syn']:.2f} tmpl={rec['template_name']} "
                          f"(Tmag {rec['Tmag_template']:.2f}) nvar={budget['n_var']} "
                          f"rms={row['clean_rms']:.2e} missing={budget['fluxfrac_missing']:.1%}  "
                          f"[{n_done}, {time.time() - t_start:.0f}s]")

    if rows:
        new = pd.DataFrame(rows)
        # Always merge by name: --overwrite regenerates the selected clusters, it never
        # discards the truth rows of clusters that were not part of this call.
        if os.path.exists(paths.truth_file):
            old = pd.read_pickle(paths.truth_file)
            old = old[~old['name'].isin(new['name'])]
            truth = pd.concat([old, new], ignore_index=True)
        else:
            truth = new
        truth = truth.sort_values(['feh', 'age', 'inst']).reset_index(drop=True)
        truth.to_pickle(paths.truth_file)
        # small, human-readable companions (CSV per user preference)
        light_cols = [c for c in truth.columns if not c.startswith('true_') and c not in
                      ('exposure_min_list', 'star_params_path', 'population_path', 'template_path', 'path')]
        truth[light_cols].to_csv(os.path.join(paths.results_dir, 'ground_truth_summary.csv'), index=False)
        write_missing_light_table(truth, paths.results_dir)
        truth[['name', 'age', 'origin']].to_csv(paths.base_table_file, index=False)
        if verbose:
            print(f'[run_lcs] {n_done} clusters written; truth table {len(truth)} rows -> {paths.truth_file}')
    return rows


def write_missing_light_table(truth: pd.DataFrame, results_dir):
    """Per (age, feh): mean fraction of TESS flux the empirical grid cannot inform."""
    cols = ['fluxfrac_missing', 'fluxfrac_outside_grid', 'fluxfrac_unsampled_sparse',
            'fluxfrac_unsampled_empty', 'fluxfrac_missing_hot_gt12500K', 'fluxfrac_missing_cool_lt12500K',
            'lbolfrac_missing', 'fluxfrac_var', 'Tmag_syn', 'n_var', 'n_components']
    g = truth.groupby(['age', 'feh'])[cols].mean().reset_index()
    g['n_instances'] = truth.groupby(['age', 'feh']).size().to_numpy()
    g.to_csv(os.path.join(results_dir, 'missing_light_by_population.csv'), index=False,
             float_format='%.4f')
    return g


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['populations', 'lcs', 'all'])
    ap.add_argument('--ages', type=float, nargs='*', default=None, help='log10 ages (default: config grid)')
    ap.add_argument('--fehs', type=float, nargs='*', default=None)
    ap.add_argument('--instances', type=int, nargs='*', default=None, help='instance indices to run')
    ap.add_argument('--n-instances', type=int, default=None)
    ap.add_argument('--n-systems', type=int, default=None)
    ap.add_argument('--limit', type=int, default=None, help='stop after N light curves (stage lcs)')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--source', choices=['auto', 'isochrones', 'legacy'], default=None)
    ap.add_argument('--no-red-noise', action='store_true')
    ap.add_argument('--brightness-mode', choices=['template', 'physical'], default=None)
    ap.add_argument('--fallback', choices=['constant', 'nearest'], default=None)
    ap.add_argument('--method', choices=['resample', 'truncnorm'], default=None)
    ap.add_argument('--out-lc-dir', default=None)
    ap.add_argument('--data-dir', default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--config', default=None, help='JSON written by a previous run (overrides defaults)')
    return ap.parse_args(argv)


def config_from_args(a) -> RunConfig:
    cfg = RunConfig.from_json(a.config) if a.config else RunConfig()
    if a.n_instances is not None: cfg.population.n_instances = a.n_instances
    if a.n_systems is not None: cfg.population.n_systems = a.n_systems
    if a.seed is not None: cfg.seed = a.seed
    if a.source is not None: cfg.population.source = a.source
    if a.no_red_noise: cfg.noise.add_red_noise = False
    if a.brightness_mode is not None: cfg.noise.brightness_mode = a.brightness_mode
    if a.fallback is not None: cfg.variability.fallback = a.fallback
    if a.method is not None: cfg.variability.method = a.method
    if a.out_lc_dir is not None: cfg.paths.out_lc_dir = os.path.abspath(a.out_lc_dir)
    if a.data_dir is not None: cfg.paths.data_dir = os.path.abspath(a.data_dir)
    if a.ages is not None: cfg.population.ages = tuple(float(x) for x in a.ages)
    if a.fehs is not None: cfg.population.fehs = tuple(float(x) for x in a.fehs)
    return cfg


def main(argv=None):
    a = _parse(argv)
    cfg = config_from_args(a)
    verbose = not a.quiet
    if a.stage in ('populations', 'all'):
        run_populations(cfg, instances=a.instances, overwrite=a.overwrite, verbose=verbose)
    if a.stage in ('lcs', 'all'):
        run_lcs(cfg, instances=a.instances, limit=a.limit, overwrite=a.overwrite, verbose=verbose)


if __name__ == '__main__':
    main()

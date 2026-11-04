"""Run the current GP pipeline on the synthetic-population light curves.

Two stages, both parallel (multiprocessing, spawn):

    python -m sythn_real_variability.gp_eval_run rows   # sector-combination rows: LC + LSP + red noise
    python -m sythn_real_variability.gp_eval_run gp     # GP fit (recommended configuration)

Run from ``GenerateTable_CGW/sytheticdata_generate_compare`` in the ``tess_gp`` env.

Why not ``expand_table`` on everything
--------------------------------------
The LMC templates have ~15 sectors each, and ``expand_table`` (f_max=0.7, include_full) makes
116,123 sector-combination rows for the 675 clusters -- thousands of CPU-hours of GP fitting.
This evaluation keeps TWO rows per cluster, both taken from the pipeline's own deterministic
combination list (``make_distinct_sector_combinations``, seed 0, no shuffle) for the cadence
group with the most sectors: the first single-sector combination and the first three-sector
combination.  One sector is the pipeline's basic unit; three sectors is the largest
combination common in the real table and resolves periods three times finer.

LC / LSP / red-noise settings are those of ``generate_table.ipynb`` (cell 10); GP settings are
the validated configuration (prewhitened seeds, decay-time cap, power-law red noise), the same
as the injection-recovery refits behind ``figs/figure_of_merit.png``.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
GT = os.path.abspath(os.path.join(HERE, '..', '..'))                  # GenerateTable_CGW
if GT not in sys.path:
    sys.path.insert(0, GT)

from sythn_real_variability.config import MAIN_REPO  # noqa: E402  (main checkout; data survives worktrees)

LC_DIR = os.path.join(MAIN_REPO, 'light_curves_synthetic_populations')
BASE_CSV = os.path.join(MAIN_REPO, 'GenerateTable_CGW', 'sytheticdata_generate_compare',
                        'sythn_real_variability', 'data', 'base_table_synthetic_populations.csv')
DATA = os.path.join(MAIN_REPO, 'GenerateTable_CGW', 'data')
ROWS_OUT = os.path.join(DATA, 'cluster_table_Nov04_synthpop_1and3sector_rows.pkl')
GP_OUT = os.path.join(DATA, 'cluster_table_Nov04_synthpop_1and3sector_gp_prewhite.pkl')

COMBO_SIZES = (1, 3)
ROW_KW = dict(cadence_bin_min=30.0, P_min=(30 / 60 / 24) * 4, P_max=10.0, alpha=10,
              n_wn_boot=200, wn_percentile=99.0, nsigma=4, rn_mask_peaks=True)
GP_KW = dict(
    freq_lim=(0.1, 12.0), n_bins=10, max_components=16,
    Q_min=0.7071, tau_cap_days=27.0, init_Q=20.0,
    fit_full_only=True, debug_fitalln=True, bic_improvement_threshold=10.0,
    threshold_mode="red_noise", harmonic_tolerance=0.05, harmonic_masking=False,
    prewhiten_seeds=True,
    two_phase_fitting=True, use_lognormal_period_prior=True, fit_red_noise_component=True,
    rn_model="powerlaw", rn_exposure_min=30.0,
    use_q_prior=True, q_prior_weight=1.0, q_prior_threshold=3.141592653589793,
    rn_period_cap=10.0, verbose=0,
)


def choose_combos(hdul):
    """(combo HDU indices, cadence) for each size in COMBO_SIZES, from the largest cadence group."""
    from table_pipeline.sector_expansion import _get_sector_cadences
    from table_pipeline.random_sector_combos import make_distinct_sector_combinations
    groups = {}
    for idx, cad in _get_sector_cadences(hdul):
        groups.setdefault(cad, []).append(idx)
    if not groups:
        return []
    cad, g = max(groups.items(), key=lambda kv: (len(kv[1]), kv[0]))
    _, combos = make_distinct_sector_combinations(len(g), f_max=0.7, seed=0, shuffle=False, include_full=True)
    out = []
    for size in COMBO_SIZES:
        c = next((c for c in combos if len(c) == size), None)
        if c is not None:
            out.append(([g[i] for i in c], cad))
    return out


def _rows_one(rec):
    import warnings
    warnings.simplefilter('ignore')
    sys.path.insert(0, GT)
    from astropy.io import fits
    from table_pipeline.lc_lsp import get_lc_path, load_and_process_sectors
    rows, msgs = [], []
    with fits.open(get_lc_path(rec['name'], rec['origin'], LC_DIR)) as hdul:
        for combo, cad in choose_combos(hdul):
            try:
                t, f, e, lsp, wn, rn = load_and_process_sectors(hdul, combo, **ROW_KW)
            except Exception as exc:                                  # noqa: BLE001
                msgs.append(f"WARNING [{rec['name']}] combo={combo}: {exc}")
                continue
            rows.append(dict(name=rec['name'], age=rec['age'], origin=rec['origin'], sectors=combo,
                             n_sectors=len(combo), cadence=cad, LC_t=t, LC_flux=f, LC_flux_err=e,
                             LSP_freq=lsp['freqs'], LSP_power=lsp['power'],
                             LSP_FAP_power=lsp['fap_threshold'], LSP_WN_threshold=wn,
                             rn_log10_N=rn[0], rn_alpha=rn[1], rn_mask_frac=rn[2]))
    return rows, msgs


def stage_rows(n_workers=48):
    import multiprocessing as mp
    import pandas as pd
    base = pd.read_csv(BASE_CSV)
    t0 = time.time()
    rows, n_warn = [], 0
    with mp.get_context('spawn').Pool(n_workers) as pool:
        for k, (r, msgs) in enumerate(pool.imap(_rows_one, base.to_dict('records'), chunksize=4)):
            rows += r
            for m in msgs:
                print(m, flush=True); n_warn += 1
            if (k + 1) % 75 == 0:
                print(f'  {k + 1}/{len(base)} clusters, {len(rows)} rows  [{(time.time() - t0) / 60:.1f} min]', flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(DATA, exist_ok=True)
    df.to_pickle(ROWS_OUT)
    print(f'wrote {ROWS_OUT}: {len(df)} rows from {len(base)} clusters, {n_warn} warnings, '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)


def _gp_chunk(args):
    k, idx = args
    import io, contextlib, warnings
    warnings.simplefilter('ignore')
    sys.path.insert(0, GT)
    import pandas as pd
    from gp_pipeline.gp_table_io import add_gp_fits
    t0 = time.time()
    sub = pd.read_pickle(ROWS_OUT).loc[idx].copy()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        add_gp_fits(sub, **GP_KW)
    sub = sub.drop(columns=['gp_result'], errors='ignore')
    log = buf.getvalue()
    return k, sub, time.time() - t0, log.count('WARNING'), log


def stage_gp(n_workers=64, rows_per_chunk=6):
    import multiprocessing as mp
    import numpy as np
    import pandas as pd
    rows = pd.read_pickle(ROWS_OUT)
    # small chunks, longest light curves first, so the pool finishes evenly
    order = rows.LC_t.apply(len).sort_values(ascending=False).index.to_numpy()
    chunks = [(k, list(order[i:i + rows_per_chunk])) for k, i in enumerate(range(0, len(order), rows_per_chunk))]
    print(f'{len(rows)} rows -> {len(chunks)} chunks on {n_workers} workers', flush=True)
    t0 = time.time()
    parts, logs, n_fail = {}, [], 0
    with mp.get_context('spawn').Pool(n_workers) as pool:
        for j, (k, sub, dt, nf, log) in enumerate(pool.imap_unordered(_gp_chunk, chunks)):
            parts[k] = sub; logs.append(log); n_fail += nf
            if (j + 1) % 20 == 0 or j + 1 == len(chunks):
                print(f'  {j + 1}/{len(chunks)} chunks, {n_fail} failed fits  '
                      f'[{(time.time() - t0) / 60:.1f} min]', flush=True)
    out = pd.concat(parts.values()).loc[rows.index]
    out.to_pickle(GP_OUT)
    with open(GP_OUT.replace('.pkl', '.log'), 'w') as fh:
        fh.write('\n'.join(logs))
    print(f'wrote {GP_OUT}: {len(out)} rows, {n_fail} failed, {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else ''
    if stage == 'rows':
        stage_rows()
    elif stage == 'gp':
        stage_gp()
    else:
        sys.exit('usage: python -m sythn_real_variability.gp_eval_run {rows|gp}')

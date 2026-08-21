"""Driver: build the whole synthetic light-curve directory + ground-truth table."""

import os
import numpy as np
import pandas as pd

from table_pipeline.wainer_table import build_base_table
from .generate import (generate_cluster, copy_layout_extras, draw_parameters,
                       load_rn_pairs)
from .ground_truth import truth_statistics, injected_signal_summary


def run(data_dir, real_lc_dir, out_lc_dir, rn_table_path, out_truth_path,
        seed=0, limit=None, verbose=True):
    """Generate synthetic LCs for every cluster and return the ground-truth table.

    One row per cluster.  The noise-free full-baseline signal is used for the truth
    statistics and then discarded; only the numbers are kept.
    """
    base = build_base_table(data_dir)
    if limit:
        base = base.head(limit)
    rn_pairs = load_rn_pairs(rn_table_path)
    rng = np.random.default_rng(seed)

    copy_layout_extras(real_lc_dir, out_lc_dir)

    rows, n_ok, n_fail = [], 0, 0
    for i, cr in enumerate(base.itertuples(index=False)):
        try:
            params = draw_parameters(rng, rn_pairs)
            rec = generate_cluster(cr.name, cr.origin, real_lc_dir, out_lc_dir,
                                   params, rng)
            if rec is None:
                n_fail += 1
                continue
            t_grid = rec.pop('t_grid'); x_grid = rec.pop('x_grid')
            # stetson_j is error-normalised and diverges as err -> 0, so truth for it
            # uses the real (median) fractional uncertainty rather than zero.
            e = rec.get('err_frac_med', np.nan)
            err_frac = (np.full(len(x_grid), e) if np.isfinite(e) else None)
            row = dict(age=cr.age, **rec)
            row.update(truth_statistics(t_grid, x_grid, err_frac))
            row.update(injected_signal_summary(params))
            rows.append(row)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            if verbose:
                print(f"  WARNING [{cr.name}]: {type(e).__name__}: {e}")
        if verbose and (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(base)}] ok={n_ok} failed={n_fail}", flush=True)

    truth = pd.DataFrame(rows)
    if out_truth_path:
        os.makedirs(os.path.dirname(out_truth_path), exist_ok=True)
        truth.to_pickle(out_truth_path)
        if verbose:
            print(f"[run_generate] wrote {len(truth)} rows -> {out_truth_path}")
    print(f"[run_generate] done | ok={n_ok} failed={n_fail}")
    return truth

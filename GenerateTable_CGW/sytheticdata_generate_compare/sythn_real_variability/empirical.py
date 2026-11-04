"""Empirical TESS variability catalogue and its (Teff, log L) grid.

Re-implements ``v2_science/make_empirical_grid.ipynb`` + ``bin_empirical_stars.py``:

1. combine the TF2026 variable catalogue (``combined_catalog_all.csv``), the TF2026 quiet
   catalogue (``quiet_stars_catalog.csv``) and the TF2023 catalogue
   (``binned_fetherolf_gaia_data.fits``, SSV + DSV) into one star table;
2. bin it on the shared Teff x log(L) grid;
3. per bin keep the variable fraction, amplitude/period moments AND the index list of the
   real variable stars, so a synthetic star can copy a real (A, P) pair from its bin.

Differences from the original, all deliberate (see README "Fixes"):

* amplitude and period sanity cuts before anything is computed (the catalogue contains
  A > 100 % "sinusoids" and periods of a few minutes);
* a bin is only *sampled* if it holds at least ``min_bin_stars`` catalogue stars;
* bin assignment is vectorised (the original re-stacked the whole table 4160 times);
* the ``subclass`` bookkeeping bug (``table[mask]['col'] = ...`` on a copy) is gone.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from astropy.table import Table

from .config import GridConfig, VariabilityConfig


# ---------------------------------------------------------------------------
# Catalogue assembly
# ---------------------------------------------------------------------------

COLS = ['TIC', 'Tmag', 'Teff', 'logL', 'P1', 'A1', 'P2', 'A2', 'cls', 'source']


def _finite(a):
    return np.asarray(pd.to_numeric(np.asarray(a), errors='coerce'), dtype=float)


def _v2_variables(path):
    c = Table.read(path, format='ascii.csv').to_pandas()
    with np.errstate(divide='ignore', invalid='ignore'):
        logL = np.log10(_finite(c['lum']))
    df = pd.DataFrame({
        'TIC': c['TIC'].astype('int64'),
        'Tmag': _finite(c['Tmag']),
        'Teff': _finite(c['Teff']),
        'logL': logL,
        'P1': _finite(c['P [d] pk1']), 'A1': _finite(c['A [ppm] pk1']),
        'P2': _finite(c['P [d] pk2']), 'A2': _finite(c['A [ppm] pk2']),
    })
    df['cls'] = 'var'
    df['source'] = 'TF2026'
    return df


def _v2_quiet(path):
    q = Table.read(path, format='ascii.csv').to_pandas()
    with np.errstate(divide='ignore', invalid='ignore'):
        logL = np.log10(_finite(q['lum']))
    df = pd.DataFrame({
        'TIC': q['TIC'].astype('int64'),
        'Tmag': _finite(q['Tmag']),
        'Teff': _finite(q['Teff']),
        'logL': logL,
        'P1': np.nan, 'A1': np.nan, 'P2': np.nan, 'A2': np.nan,
    })
    df['cls'] = 'quiet'
    df['source'] = 'TF2026_quiet'
    return df


def _v1_variables(path):
    v = Table.read(path)
    v = v[['TIC', 'Tmag', 'Teff', 'Ls', 'type', 'Pvar', 'Amp', 'P1', 'P2', 'A1', 'A2']].to_pandas()
    typ = v['type'].astype(str).str.strip()
    ssv = (typ == 'SSV').to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        logL = np.log10(_finite(v['Ls']))
    df = pd.DataFrame({
        'TIC': v['TIC'].astype('int64'),
        'Tmag': _finite(v['Tmag']),
        'Teff': _finite(v['Teff']),
        'logL': logL,
        'P1': np.where(ssv, _finite(v['Pvar']), _finite(v['P1'])),
        'A1': np.where(ssv, _finite(v['Amp']), _finite(v['A1'])),
        'P2': np.where(ssv, np.nan, _finite(v['P2'])),
        'A2': np.where(ssv, np.nan, _finite(v['A2'])),
    })
    df['cls'] = 'var'
    df['source'] = np.where(ssv, 'TF2023_SSV', 'TF2023_DSV')
    return df


def build_empirical_sample(dropbox_synth_dir, grid: GridConfig, var: VariabilityConfig,
                           verbose=True):
    """Assemble, clean and deduplicate the empirical star sample.

    Returns ``(sample, report)``: a DataFrame with columns ``COLS`` and a dict of counts
    describing what was dropped and why.
    """
    v2 = _v2_variables(os.path.join(dropbox_synth_dir, 'v2_files', 'combined_catalog_all.csv'))
    q2 = _v2_quiet(os.path.join(dropbox_synth_dir, 'v2_files', 'quiet_stars_catalog.csv'))
    v1 = _v1_variables(os.path.join(dropbox_synth_dir, 'v1', 'fits',
                                    'binned_fetherolf_gaia_data.fits'))
    report = dict(n_TF2026_var=len(v2), n_TF2026_quiet=len(q2), n_TF2023=len(v1))

    # TF2026 first so it wins the TIC de-duplication; variables before quiet.
    allstars = pd.concat([v2, v1, q2], ignore_index=True)
    n0 = len(allstars)
    allstars = allstars[np.isfinite(allstars['logL']) & np.isfinite(allstars['Teff'])]
    report['dropped_nonfinite_teff_logl'] = n0 - len(allstars)
    n0 = len(allstars)
    allstars = allstars.drop_duplicates('TIC', keep='first')
    report['dropped_duplicate_TIC'] = n0 - len(allstars)
    n0 = len(allstars)
    allstars = allstars[allstars['Teff'] < grid.teff_catalog_max]
    report['dropped_teff_above_catalog_max'] = n0 - len(allstars)

    # Variable-star sanity: the primary sinusoid must be a plausible sinusoid.  A star that
    # fails is removed from the sample entirely (unknown status), NOT reclassified as quiet.
    isvar = (allstars['cls'] == 'var').to_numpy()
    A1 = allstars['A1'].to_numpy(); P1 = allstars['P1'].to_numpy()
    ok1 = (np.isfinite(A1) & np.isfinite(P1)
           & (A1 >= var.amp_min_ppm) & (A1 <= var.amp_max_ppm)
           & (P1 >= var.period_min_d) & (P1 <= var.period_max_d))
    bad = isvar & ~ok1
    report['dropped_variables_failing_sanity'] = int(bad.sum())
    report['dropped_variables_amp_gt_cap'] = int((isvar & np.isfinite(A1) & (A1 > var.amp_max_ppm)).sum())
    report['dropped_variables_period_lt_min'] = int((isvar & np.isfinite(P1) & (P1 < var.period_min_d)).sum())
    allstars = allstars[~bad].copy()

    # Secondary component: keep only if it also passes; otherwise blank it (the star stays).
    A2 = allstars['A2'].to_numpy(); P2 = allstars['P2'].to_numpy()
    ok2 = (np.isfinite(A2) & np.isfinite(P2)
           & (A2 >= var.amp_min_ppm) & (A2 <= var.amp_max_ppm)
           & (P2 >= var.period_min_d) & (P2 <= var.period_max_d))
    report['n_with_second_component'] = int(ok2.sum())
    allstars.loc[~ok2, ['A2', 'P2']] = np.nan

    allstars = allstars.reset_index(drop=True)
    report['n_final'] = len(allstars)
    report['n_final_var'] = int((allstars['cls'] == 'var').sum())
    report['n_final_quiet'] = int((allstars['cls'] == 'quiet').sum())
    if verbose:
        for k, v in report.items():
            print(f'  [empirical] {k:40s} {v}')
    return allstars[COLS], report


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

class EmpiricalGrid:
    """Teff x log(L) grid with per-bin variability statistics and member lists.

    Bin index = ``i_teff * n_logl + i_logl`` (same ordering as the original ``bin``
    column); -1 means outside the grid extent.
    """

    def __init__(self, grid: GridConfig, var: VariabilityConfig, sample: pd.DataFrame):
        self.grid = grid
        self.var = var
        self.teff_edges = grid.teff_edges
        self.logl_edges = grid.logl_edges
        self.n_teff = len(self.teff_edges) - 1
        self.n_logl = len(self.logl_edges) - 1
        self.nbins = self.n_teff * self.n_logl
        self.sample = sample.reset_index(drop=True).copy()
        self.sample['bin'] = self.assign(self.sample['Teff'].to_numpy(),
                                         self.sample['logL'].to_numpy())
        self._build()

    # -- geometry ---------------------------------------------------------------------
    def assign(self, teff, logl):
        teff = np.asarray(teff, float)
        logl = np.asarray(logl, float)
        it = np.searchsorted(self.teff_edges, teff, side='right') - 1
        il = np.searchsorted(self.logl_edges, logl, side='right') - 1
        ok = ((it >= 0) & (it < self.n_teff) & (il >= 0) & (il < self.n_logl)
              & np.isfinite(teff) & np.isfinite(logl))
        return np.where(ok, it * self.n_logl + il, -1)

    def bin_centers(self, b):
        b = np.asarray(b)
        it, il = b // self.n_logl, b % self.n_logl
        return (self.teff_edges[it] + 0.5 * self.grid.teff_step,
                self.logl_edges[il] + 0.5 * self.grid.logl_step)

    # -- statistics -------------------------------------------------------------------
    def _build(self):
        s = self.sample
        b = s['bin'].to_numpy()
        inb = b >= 0
        isvar = (s['cls'] == 'var').to_numpy()
        self.n_total = np.bincount(b[inb], minlength=self.nbins)
        self.n_var = np.bincount(b[inb & isvar], minlength=self.nbins)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.frac_var = np.where(self.n_total > 0,
                                     self.n_var / np.maximum(self.n_total, 1), np.nan)
        self.sampled = self.n_total >= self.var.min_bin_stars

        # member index lists for resampling (variables only), sorted by bin
        sel = np.where(inb & isvar)[0]
        order = np.argsort(b[sel], kind='stable')
        idx_var = sel[order]
        bins_sorted = b[idx_var]
        starts = np.searchsorted(bins_sorted, np.arange(self.nbins), side='left')
        stops = np.searchsorted(bins_sorted, np.arange(self.nbins), side='right')
        self._var_members = (idx_var, starts, stops)

        # per-bin moments of the variables
        A1 = s['A1'].to_numpy(); P1 = s['P1'].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            arrays = (('A1', A1), ('P1', P1), ('logA1', np.log10(A1)), ('logP1', np.log10(P1)))
        self.stats = {}
        for name, arr in arrays:
            out = {k: np.full(self.nbins, np.nan) for k in ('med', 'mean', 'std', 'min', 'max')}
            for bi in np.unique(bins_sorted):
                v = arr[idx_var[starts[bi]:stops[bi]]]
                v = v[np.isfinite(v)]
                if len(v):
                    out['med'][bi] = np.median(v); out['mean'][bi] = v.mean(); out['std'][bi] = v.std()
                    out['min'][bi] = v.min(); out['max'][bi] = v.max()
            self.stats[name] = out

        # nearest sampled bin (fallback='nearest'); distance in units of bin widths
        sb = np.where(self.sampled)[0]
        if len(sb):
            tc, lc = self.bin_centers(sb)
            tcs = tc / self.grid.teff_step; lcs = lc / self.grid.logl_step
            allb = np.arange(self.nbins)
            ta, la = self.bin_centers(allb)
            ta = ta / self.grid.teff_step; la = la / self.grid.logl_step
            d2 = (ta[:, None] - tcs[None, :]) ** 2 + (la[:, None] - lcs[None, :]) ** 2
            j = np.argmin(d2, axis=1)
            self.nearest_sampled = sb[j]
            self.nearest_sampled_dist = np.sqrt(d2[np.arange(self.nbins), j])
        else:
            self.nearest_sampled = np.full(self.nbins, -1)
            self.nearest_sampled_dist = np.full(self.nbins, np.inf)

    def variable_members(self, bi):
        idx_var, starts, stops = self._var_members
        return idx_var[starts[bi]:stops[bi]]

    def summary_table(self):
        """One row per populated bin (same spirit as ``binned_empirical_summary_data.fits``)."""
        b = np.where(self.n_total > 0)[0]
        tc, lc = self.bin_centers(b)
        df = pd.DataFrame(dict(bin=b, center_teff=tc, center_logl=lc, n_total=self.n_total[b],
                               n_var=self.n_var[b], frac_var=self.frac_var[b],
                               sampled=self.sampled[b]))
        for name in ('A1', 'P1'):
            for k in ('med', 'std', 'min', 'max'):
                df[f'{k}_{name}'] = self.stats[name][k][b]
        return df

    def map2d(self, values):
        """Reshape a per-bin vector to (n_logl, n_teff) for imshow (rows = log L)."""
        return np.asarray(values).reshape(self.n_teff, self.n_logl).T

    # -- persistence ------------------------------------------------------------------
    def save(self, sample_csv_path):
        os.makedirs(os.path.dirname(sample_csv_path), exist_ok=True)
        self.sample[COLS].to_csv(sample_csv_path, index=False)

    @classmethod
    def load(cls, grid: GridConfig, var: VariabilityConfig, sample_csv_path):
        sample = pd.read_csv(sample_csv_path)
        return cls(grid, var, sample[COLS])


def get_empirical_grid(cfg, rebuild=False, verbose=True):
    """Load the cached cleaned sample (fast) or build it from the Dropbox catalogues."""
    p = cfg.paths
    report = None
    if not rebuild and os.path.exists(p.empirical_sample_file):
        g = EmpiricalGrid.load(cfg.grid, cfg.variability, p.empirical_sample_file)
        if verbose:
            print(f'  [empirical] loaded cached sample: {len(g.sample)} stars')
    else:
        sample, report = build_empirical_sample(p.dropbox_synth, cfg.grid, cfg.variability, verbose)
        g = EmpiricalGrid(cfg.grid, cfg.variability, sample)
        g.save(p.empirical_sample_file)
    g.summary_table().to_csv(os.path.join(p.data_dir, 'empirical_grid_summary.csv'), index=False)
    if verbose:
        print(f'  [empirical] grid: {g.nbins} bins, {int((g.n_total > 0).sum())} populated, '
              f'{int(g.sampled.sum())} sampled (>= {cfg.variability.min_bin_stars} stars)')
    return g, report

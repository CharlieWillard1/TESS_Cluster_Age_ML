"""Isochrone populations -> component-level star tables with TESS magnitudes.

Two sources, same output:

* ``isochrones`` (preferred, env ``synth_pop``): the recipe of
  ``v1/make_synthetic_populations.ipynb`` -- MIST, Salpeter IMF, binary fraction 0.4,
  mass-ratio exponent 0.3, delta-function SFH, Gaussian [Fe/H] -- but at the LMC distance
  and with a fresh IMF draw per instance.
* legacy (fallback, any env): the 45 ``isos/age_*_feh_*_population.fits`` tables from the
  Dropbox, re-scaled from 10 pc to the LMC and bootstrap-resampled to ``n_systems``.

TESS magnitudes are **never** taken from the ``TESS_mag_*`` columns: in the legacy tables a
random half of all stars (carrying up to 94 % of the light in young populations) has NaN
there.  Instead ``m_T = M_bol - BC_T(Teff, logg) + DM + A_T`` with ``M_bol = 4.74 - 2.5 log L``
and ``BC_T`` from the MIST bolometric-correction grid (``isochrones``) or, when that is
unavailable, from the polynomial below fitted to 285,963 MIST stars (rms 0.026 mag).
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from .config import (TESS_ZP, M_BOL_SUN, A_TESS_OVER_A_V, PopulationConfig, RunConfig,
                     population_tag)


# ---------------------------------------------------------------------------
# Photometry
# ---------------------------------------------------------------------------

# BC_T(logTeff, logg) = sum_k c_k x^k + c_7 g + c_8 g x,  x = logTeff - 4, g = logg - 4.
# Fitted to the TESS_mag / logL / logTeff / logg of every star with a finite magnitude in
# Amaya_Dropbox/tess_clusters/synthetic_populations/isos/*.fits (MIST v1.2 via isochrones):
# n = 285,963, rms residual 0.026 mag, 99th percentile |resid| 0.091 mag,
# valid for logTeff in [3.46, 4.85], logg in [-0.2, 5.3].  Re-derive with fit_bc_poly().
BC_T_POLY_COEF = np.array([
    -0.2631952425982946, -5.011103179558816, -6.833753523098242, 4.740119860894082,
    12.457510768306065, 7.008859485473743, -24.034475743829553,
    -0.05061767538691339, -0.20839453649453415])
BC_T_POLY_RANGE = dict(logTeff=(3.46, 4.85), logg=(-0.2, 5.3))


def bc_tess_poly(logTeff, logg):
    """Polynomial TESS bolometric correction (see BC_T_POLY_COEF). Clipped to its fit range."""
    x = np.clip(np.asarray(logTeff, float), *BC_T_POLY_RANGE['logTeff']) - 4.0
    g = np.clip(np.asarray(logg, float), *BC_T_POLY_RANGE['logg']) - 4.0
    c = BC_T_POLY_COEF
    bc = sum(c[k] * x ** k for k in range(7)) + c[7] * g + c[8] * g * x
    return bc


def fit_bc_poly(isos_dir):
    """Re-derive BC_T_POLY_COEF from the legacy population tables (provenance helper)."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(isos_dir, '*_population.fits'))):
        t = Table.read(f)
        for c in ('0', '1'):
            m = ~np.asarray(getattr(t['logTeff_' + c], 'mask', np.zeros(len(t), bool)))
            rows.append(np.column_stack([np.asarray(t[k + c], float)[m] for k in
                                         ('TESS_mag_', 'logTeff_', 'logL_', 'logg_', 'distance_')]))
    tm, lt, ll, lg, d = np.vstack(rows).T
    ok = np.isfinite(tm) & np.isfinite(ll) & np.isfinite(lt) & np.isfinite(lg)
    bc = (M_BOL_SUN - 2.5 * ll) - (tm - 5 * np.log10(d / 10))
    x = lt[ok] - 4.0; g = lg[ok] - 4.0
    A = np.column_stack([x ** k for k in range(7)] + [g, g * x])
    coef = np.linalg.lstsq(A, bc[ok], rcond=None)[0]
    resid = bc[ok] - A @ coef
    return coef, dict(n=int(ok.sum()), rms=float(resid.std()), p99=float(np.percentile(abs(resid), 99)))


class MistBC:
    """TESS bolometric correction from the MIST grid through ``isochrones`` (if importable)."""

    def __init__(self):
        from isochrones import get_ichrone          # noqa: raises ImportError if unavailable
        self.mist = get_ichrone('mist', bands=['TESS'])
        self.grid = self.mist.bc_grid
        # force the table to load now so a broken download fails loudly, here
        self.sun_bc = float(np.asarray(self.grid.interp([5772.0, 4.44, 0.0, 0.0], ['TESS'])).ravel()[0])

    def __call__(self, teff, logg, feh, av):
        # isochrones' DFInterpolator takes the parameters as a list of COLUMN arrays
        pars = [np.asarray(teff, float), np.asarray(logg, float),
                np.asarray(feh, float), np.asarray(av, float)]
        out = np.asarray(self.grid.interp(pars, ['TESS']), float).reshape(len(pars[0]), -1)[:, 0]
        return out


def get_bc_function(prefer_mist=True, verbose=True):
    """Return ``(bc_func(teff, logg, feh, av) -> BC_T, label)``, MIST grid if possible."""
    if prefer_mist:
        try:
            mbc = MistBC()
            if verbose:
                print('  [populations] BC_T from the MIST grid (isochrones)')
            return mbc, 'mist_grid'
        except Exception as e:                     # ImportError or a failed table download
            if verbose:
                print(f'  [populations] MIST BC grid unavailable ({type(e).__name__}); '
                      f'using the fitted polynomial')
    return (lambda teff, logg, feh, av: bc_tess_poly(np.log10(teff), logg)), 'polynomial'


def tess_flux_from_mag(tmag, zp=TESS_ZP):
    """Apparent TESS magnitude -> count rate [e-/s]."""
    return 10.0 ** ((zp - np.asarray(tmag, float)) / 2.5)


def tess_mag_from_flux(flux, zp=TESS_ZP):
    return zp - 2.5 * np.log10(np.asarray(flux, float))


def apparent_tess_mag(logL, bc, distance_pc, A_V=0.0):
    """Absolute and apparent TESS magnitude from log(L/Lsun) and a bolometric correction."""
    M_T = (M_BOL_SUN - 2.5 * np.asarray(logL, float)) - bc
    m_T = M_T + 5.0 * np.log10(distance_pc / 10.0) + A_TESS_OVER_A_V * A_V
    return M_T, m_T


# ---------------------------------------------------------------------------
# System tables -> component tables
# ---------------------------------------------------------------------------

_SYS_COLS = ['mass', 'initial_mass', 'logTeff', 'logL', 'logg', 'eep', 'phase', 'feh']


def components_from_systems(systems: pd.DataFrame, pcfg: PopulationConfig, bc_func):
    """Split the isochrones system table (``*_0`` primaries, ``*_1`` secondaries) into one
    row per stellar component and attach TESS photometry.

    Components with non-finite logL or logTeff (stars that have evolved off the MIST grid --
    they are dead) are dropped; the count is returned in ``meta``.
    """
    parts = []
    for comp in (0, 1):
        sfx = f'_{comp}'
        if 'logL' + sfx not in systems:
            continue
        d = pd.DataFrame({c: pd.to_numeric(systems.get(c + sfx, np.nan), errors='coerce')
                          .to_numpy(dtype=float) for c in _SYS_COLS})
        d['sys_id'] = np.arange(len(systems))
        d['comp'] = comp
        d['is_binary_system'] = np.isfinite(pd.to_numeric(systems.get('logL_1', np.nan),
                                                          errors='coerce').to_numpy(dtype=float))
        parts.append(d)
    comp = pd.concat(parts, ignore_index=True)
    n_slots = len(comp)
    present = np.isfinite(comp['logL']) & np.isfinite(comp['logTeff'])
    # secondaries that do not exist are NaN rows for single systems: not "dead", just absent
    absent_secondary = (comp['comp'] == 1) & ~comp['is_binary_system']
    dead = (~present) & ~absent_secondary
    comp = comp[present].reset_index(drop=True)

    comp['Teff'] = 10.0 ** comp['logTeff']
    feh = comp['feh'].where(np.isfinite(comp['feh']), 0.0).to_numpy()
    comp['BC_T'] = bc_func(comp['Teff'].to_numpy(), comp['logg'].to_numpy(), feh,
                           np.full(len(comp), pcfg.A_V))
    comp['M_T'], comp['T_app'] = apparent_tess_mag(comp['logL'].to_numpy(), comp['BC_T'].to_numpy(),
                                                   pcfg.distance_pc, pcfg.A_V)
    comp['F_es'] = tess_flux_from_mag(comp['T_app'].to_numpy())
    comp['L_bol'] = 10.0 ** comp['logL']

    meta = dict(n_systems=int(len(systems)), n_components=int(len(comp)),
                n_dead_dropped=int(dead.sum()), n_binary_systems=int(comp.loc[comp['comp'] == 0, 'is_binary_system'].sum()),
                n_bc_nonfinite=int((~np.isfinite(comp['BC_T'])).sum()))
    return comp, meta


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def generate_isochrones_population(age, feh, pcfg: PopulationConfig, seed):
    """Draw ``n_systems`` systems with ``isochrones`` (needs the ``synth_pop`` env)."""
    from scipy.stats import norm
    from isochrones import get_ichrone
    from isochrones.priors import SalpeterPrior, ChabrierPrior, GaussianPrior
    from isochrones.populations import StarFormationHistory, StarPopulation

    np.random.seed(int(seed) % (2 ** 32 - 1))     # isochrones draws from the global RNG
    mist = get_ichrone('mist', bands=['TESS'])
    if pcfg.imf == 'salpeter':
        imf = SalpeterPrior(bounds=tuple(pcfg.imf_mass_bounds))
    elif pcfg.imf == 'chabrier':
        imf = ChabrierPrior(bounds=tuple(pcfg.imf_mass_bounds))
    else:
        raise ValueError(f'unknown IMF {pcfg.imf!r}')
    age_gyr = 10.0 ** float(age) / 1e9
    sfh = StarFormationHistory(dist=norm(age_gyr, 1e-6 * max(age_gyr, 1e-3)))
    pop = StarPopulation(mist, imf=imf, fB=pcfg.binary_fraction, gamma=pcfg.mass_ratio_gamma,
                         sfh=sfh, feh=GaussianPrior(float(feh), pcfg.feh_sigma),
                         distance=GaussianPrior(pcfg.distance_pc, 1.0),
                         AV=GaussianPrior(pcfg.A_V, 1e-6))
    df = pop.generate(int(pcfg.n_systems))
    df = df.reset_index(drop=True)
    return df


def legacy_population_path(isos_dir, age, feh):
    feh_s = '-1' if float(feh) == -1.0 else f'{float(feh):.1f}'
    return os.path.join(isos_dir, f'age_{float(age):.1f}_feh_{feh_s}_population.fits'
                        if (float(age) * 4) % 2 == 0 else
                        f'age_{float(age):.2f}_feh_{feh_s}_population.fits')


def load_legacy_population(isos_dir, age, feh):
    """Read one of the 45 Dropbox population tables as a system DataFrame."""
    cands = [os.path.join(isos_dir, f'age_{a}_feh_{f}_population.fits')
             for a in (f'{float(age):.1f}', f'{float(age):.2f}', str(float(age)))
             for f in ('-1' if float(feh) == -1.0 else f'{float(feh):.1f}', str(float(feh)))]
    for c in cands:
        if os.path.exists(c):
            t = Table.read(c)
            df = t.to_pandas()
            return df
    raise FileNotFoundError(f'no legacy population for age={age} feh={feh} in {isos_dir}')


def bootstrap_systems(systems: pd.DataFrame, n_systems, rng):
    """Resample whole systems with replacement (keeps primary/secondary pairs intact)."""
    idx = rng.integers(0, len(systems), int(n_systems))
    return systems.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def population_file(cfg: RunConfig, age, feh, inst):
    return os.path.join(cfg.paths.populations_dir, f'{population_tag(age, feh)}_i{inst:02d}.fits')


def save_components(comp: pd.DataFrame, meta: dict, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = Table.from_pandas(comp)
    for k, v in meta.items():
        key = ('M_' + k.upper())[:8]
        t.meta[key] = v if not isinstance(v, str) else v[:68]
    t.meta['comment'] = [f'{k} = {v}' for k, v in meta.items()]
    t.write(path, overwrite=True)


def load_components(path):
    with fits.open(path) as h:
        t = Table(h[1].data)
        hdr = h[1].header
    df = t.to_pandas()
    meta = {}
    for c in hdr.get('COMMENT', []):
        if ' = ' in str(c):
            k, v = str(c).split(' = ', 1)
            meta[k.strip()] = _parse(v.strip())
    return df, meta


def _parse(s):
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def summarize_components(comp: pd.DataFrame):
    F = comp['F_es'].to_numpy()
    prim = comp['comp'] == 0
    return dict(
        n_components=int(len(comp)),
        n_primaries=int(prim.sum()),
        n_secondaries=int((~prim).sum()),
        total_initial_mass_msun=float(np.nansum(comp['initial_mass'])),
        total_present_mass_msun=float(np.nansum(comp['mass'])),
        max_mass_msun=float(np.nanmax(comp['mass'])),
        F_total_es=float(np.nansum(F)),
        Tmag_integrated=float(tess_mag_from_flux(np.nansum(F))),
        logL_total=float(np.log10(np.nansum(comp['L_bol']))),
        teff_min=float(comp['Teff'].min()), teff_max=float(comp['Teff'].max()),
        logl_min=float(comp['logL'].min()), logl_max=float(comp['logL'].max()),
    )


# ---------------------------------------------------------------------------
# Driver for one population
# ---------------------------------------------------------------------------

def build_population(age, feh, inst, cfg: RunConfig, bc_func, bc_label, base_systems=None,
                     verbose=True):
    """Return ``(components, meta)`` for one (age, feh, instance).

    ``base_systems`` (legacy mode): a system table to bootstrap from, so the expensive read
    happens once per (age, feh).
    """
    pcfg = cfg.population
    seed = int(np.random.SeedSequence([cfg.seed, int(round(age * 100)), int(round((feh + 5) * 100)), inst]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    src = pcfg.source
    if src == 'auto':
        try:
            import isochrones  # noqa: F401
            src = 'isochrones'
        except ImportError:
            src = 'legacy'

    if src == 'isochrones':
        if pcfg.fresh_population_per_instance or base_systems is None:
            systems = generate_isochrones_population(age, feh, pcfg, seed)
        else:
            systems = bootstrap_systems(base_systems, pcfg.n_systems, rng)
    elif src == 'legacy':
        if base_systems is None:
            base_systems = load_legacy_population(cfg.paths.legacy_isos_dir, age, feh)
        systems = bootstrap_systems(base_systems, pcfg.n_systems, rng) \
            if (pcfg.n_systems != len(base_systems) or inst > 0) else base_systems
    else:
        raise ValueError(f'unknown population source {src!r}')

    comp, meta = components_from_systems(systems, pcfg, bc_func)
    meta.update(dict(age=float(age), feh=float(feh), inst=int(inst), seed=seed, source=src,
                     bc_source=bc_label, distance_pc=pcfg.distance_pc, A_V=pcfg.A_V,
                     imf=pcfg.imf, imf_mass_min=pcfg.imf_mass_bounds[0], imf_mass_max=pcfg.imf_mass_bounds[1],
                     binary_fraction=pcfg.binary_fraction, mass_ratio_gamma=pcfg.mass_ratio_gamma,
                     bootstrapped=int(src == 'legacy' or not pcfg.fresh_population_per_instance)))
    meta.update(summarize_components(comp))
    if verbose:
        print(f"  [populations] {population_tag(age, feh)} i{inst:02d} ({src}): "
              f"{meta['n_components']} components, {meta['total_initial_mass_msun']:.0f} Msun, "
              f"Tmag_int={meta['Tmag_integrated']:.2f}, dead dropped={meta['n_dead_dropped']}")
    return comp, meta, systems

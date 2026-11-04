"""Configuration and paths for the synthetic-population light-curve generator.

Every number that shapes the construction lives here so a run can be reproduced from
the ``run_config.json`` it writes.  Nothing in this file touches data.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SDGC_DIR = os.path.abspath(os.path.join(HERE, '..'))           # sytheticdata_generate_compare
GT_DIR = os.path.abspath(os.path.join(SDGC_DIR, '..'))         # GenerateTable_CGW
REPO_ROOT = os.path.abspath(os.path.join(GT_DIR, '..'))        # TESS_Cluster_Age_ML (this checkout)

# A git worktree lives under <main>/.claude/worktrees/<name>/.  Large, regenerable data
# products are written to the MAIN checkout so they survive the worktree being removed.
MAIN_REPO = REPO_ROOT.split(os.sep + '.claude' + os.sep + 'worktrees')[0]

# Small, tracked results (CSV / markdown / JSON / PNG) live with the code.
RESULTS_DIR = os.path.join(HERE, 'results')


@dataclass
class Paths:
    dropbox_synth: str = os.path.join(MAIN_REPO, 'Amaya_Dropbox', 'tess_clusters',
                                      'synthetic_populations')
    real_lc_dir: str = os.path.join(MAIN_REPO, 'light_curves')
    rn_table: str = os.path.join(MAIN_REPO, 'GenerateTable_CGW', 'data',
                                 'cluster_table_Sep21_3sectorcombos_real.pkl')
    out_lc_dir: str = os.path.join(MAIN_REPO, 'light_curves_synthetic_populations')
    data_dir: str = os.path.join(MAIN_REPO, 'GenerateTable_CGW', 'sytheticdata_generate_compare',
                                 'sythn_real_variability', 'data')
    results_dir: str = RESULTS_DIR

    @property
    def legacy_isos_dir(self):
        return os.path.join(self.dropbox_synth, 'isos')

    @property
    def populations_dir(self):
        return os.path.join(self.data_dir, 'populations')

    @property
    def star_params_dir(self):
        return os.path.join(self.data_dir, 'star_params')

    @property
    def empirical_sample_file(self):
        return os.path.join(self.data_dir, 'empirical_sample.csv')

    @property
    def truth_file(self):
        return os.path.join(self.data_dir, 'ground_truth_populations.pkl')

    @property
    def base_table_file(self):
        return os.path.join(self.data_dir, 'base_table_synthetic_populations.csv')


# ---------------------------------------------------------------------------
# Physical constants (single source of truth; verified in checks.py)
# ---------------------------------------------------------------------------

TESS_ZP = 20.44                # Tmag = 20.44 - 2.5 log10(e/s); Tmag 10 <-> 15,000 e/s (Sullivan+2015)
M_BOL_SUN = 4.74               # IAU 2015 resolution B2
LMC_DISTANCE_PC = 49_590.0     # Pietrzynski et al. 2019, Nature 567, 200: 49.59 +- 0.09 (stat) +- 0.54 (sys) kpc
LMC_DISTANCE_MODULUS = 5.0 * np.log10(LMC_DISTANCE_PC / 10.0)   # 18.477
A_TESS_OVER_A_V = 0.61         # TESS-band extinction per unit A_V (Stassun+2019, TIC-8); only used if A_V > 0


# ---------------------------------------------------------------------------
# Construction parameters
# ---------------------------------------------------------------------------

@dataclass
class GridConfig:
    """Teff x log(L) grid shared by the empirical catalogue and the synthetic stars.

    Identical to the v2_science grid (150 K x 0.15 dex) by default.  The catalogue is cut
    at ``teff_catalog_max`` before binning (TIC Teffs above that are unreliable), so the
    grid's upper edge of 12,500 K is also where the model stops having any information.
    """
    teff_min: float = 2700.0
    teff_max: float = 12500.0
    teff_step: float = 150.0
    logl_min: float = -3.2
    logl_max: float = 6.5
    logl_step: float = 0.15
    teff_catalog_max: float = 15000.0

    @property
    def teff_edges(self):
        return np.arange(self.teff_min, self.teff_max, self.teff_step)

    @property
    def logl_edges(self):
        return np.arange(self.logl_min, self.logl_max, self.logl_step)


@dataclass
class PopulationConfig:
    """Isochrone population recipe (mirrors v1/make_synthetic_populations.ipynb)."""
    ages: tuple = tuple(float(a) for a in np.round(np.arange(6.5, 10.25, 0.25), 2))   # log10(age/yr)
    fehs: tuple = (-1.0, -0.5, 0.0)
    n_instances: int = 15
    n_systems: int = 10_000            # primary systems drawn per population
    imf: str = 'salpeter'              # 'salpeter' | 'chabrier'
    imf_mass_bounds: tuple = (0.1, 300.0)   # MIST tracks start at 0.1 Msun; the old (0.008, 300)
                                            # bound was silently truncated at 0.1 by the grid anyway
    binary_fraction: float = 0.4
    mass_ratio_gamma: float = 0.3
    feh_sigma: float = 0.01
    distance_pc: float = LMC_DISTANCE_PC
    A_V: float = 0.0
    fresh_population_per_instance: bool = True   # new IMF draw per instance (needs isochrones);
                                                 # False -> bootstrap-resample one base population
    source: str = 'auto'               # 'isochrones' | 'legacy' | 'auto' (isochrones if importable)


@dataclass
class VariabilityConfig:
    """How a synthetic star gets its variability from the empirical grid."""
    method: str = 'resample'           # 'resample': copy (A,P) of a random real variable in the bin
                                       # 'truncnorm': truncated-normal draws from the bin moments
    min_bin_stars: int = 5             # bins with fewer catalogue stars count as UNSAMPLED
    fallback: str = 'constant'         # 'constant': unsampled stars shine at constant flux
                                       # 'nearest': borrow the nearest sampled bin's statistics
    include_second_component: bool = True   # use (A2, P2) of DSV catalogue stars when present
    amp_max_ppm: float = 5.0e5         # sinusoid amplitude sanity cap (50 %); larger values in the
                                       # catalogue are eclipsing binaries / failed fits, not sinusoids
    amp_min_ppm: float = 1.0
    period_min_d: float = 0.005        # ~7 min; below this a 2-min-cadence LSP is meaningless
    period_max_d: float = 15.0


@dataclass
class NoiseConfig:
    """TESS sampling + noise, reusing sytheticdata_generate_compare.synth.generate."""
    template_origin: str = 'LMC'
    template_select: str = 'brightness'   # 'brightness': nearest real cluster in median flux
                                          # 'random': any real cluster of that origin
    template_k_nearest: int = 3           # pick randomly among the k brightness-nearest templates
    brightness_mode: str = 'template'     # 'template': scale x(t) to the template's median flux and
                                          #   use its flux_err verbatim (exact pipeline convention)
                                          # 'physical': write the population's own e/s and scale
                                          #   flux_err by sqrt(F_syn/F_template) (photon-noise approx.)
    add_red_noise: bool = True            # calibrated (alpha, log10_N) red noise resampled from real rows
    rn_origin: str = 'LMC'
    rn_rms_max: float = 0.02              # cap on the calibrated red-noise fractional rms; a few real
                                          # (alpha ~1.9, log10_N ~3) rows calibrate to 5-9 %, which is
                                          # astrophysical signal absorbed by the fit, not noise
    grid_cadence_min: float = 1.0
    truth_cadence_min: float = 30.0
    lsp_P_min: float = (30 / 60 / 24) * 4
    lsp_P_max: float = 10.0
    lsp_alpha: int = 10


@dataclass
class RunConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    variability: VariabilityConfig = field(default_factory=VariabilityConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    paths: Paths = field(default_factory=Paths)
    seed: int = 0
    name_prefix: str = 'synthpop'

    def to_dict(self):
        d = asdict(self)
        d['constants'] = dict(TESS_ZP=TESS_ZP, M_BOL_SUN=M_BOL_SUN, LMC_DISTANCE_PC=LMC_DISTANCE_PC,
                              LMC_DISTANCE_MODULUS=float(LMC_DISTANCE_MODULUS),
                              A_TESS_OVER_A_V=A_TESS_OVER_A_V)
        return d

    def to_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        d.pop('constants', None)
        return cls(grid=GridConfig(**d['grid']), population=PopulationConfig(**d['population']),
                   variability=VariabilityConfig(**d['variability']), noise=NoiseConfig(**d['noise']),
                   paths=Paths(**d['paths']), seed=d['seed'], name_prefix=d['name_prefix'])


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f'not JSON serialisable: {type(o)}')


def population_tag(age, feh):
    sign = 'm' if feh < 0 else 'p'
    return f'a{age:.2f}_f{sign}{abs(feh):.2f}'


def cluster_name(prefix, age, feh, inst):
    """Name that survives ``lc_lsp._slug`` (lowercase; spaces/hyphens/brackets stripped).

    ``synthpop_a8.00_fm0.50_i03`` -> file
    ``hlsp_elk_tess_ffi_synthpop_a8.00_fm0.50_i03_tess_v1_llc.fits``.
    """
    return f'{prefix}_{population_tag(age, feh)}_i{inst:02d}'

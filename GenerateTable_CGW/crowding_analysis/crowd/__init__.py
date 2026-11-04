"""Crowding / light-budget analysis for the ELK integrated cluster apertures.

Modules
-------
catalogs       cluster list, measured aperture flux, literature cross-matches (VizieR TAP)
ssp            PARSEC single-burst population magnitudes per solar mass (CMD web service)
galaxy_models  smooth field-light models: LMC/SMC disks, MW line-of-sight integral, zodi
gaia_field     local field light from Gaia DR3 (radial-bin aggregates + bright-star rows)
budget         assembles the per-cluster light budget
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))          # TESS_Cluster_Age_ML
LC_DIR = os.path.join(ROOT, 'light_curves')
CACHE = os.path.join(ROOT, 'GenerateTable_CGW', 'data', 'crowding_cache')
os.makedirs(CACHE, exist_ok=True)

# ELK / TESS zero point: T = -2.5 log10(flux [e/s]) + 20.44.  Verified against the `mag`
# column of the ELK files, so every flux in this package is in ELK e/s.
T_ZP = 20.44

# Cardelli (R_V = 3.1) extinction ratios A_X / A_V at the band effective wavelengths.
# TESS (lambda_eff ~ 0.79 um) sits next to I_C (0.80 um).
A_RATIO = {'V': 1.0, 'R': 0.75, 'I': 0.59, 'T': 0.60, 'J': 0.28, 'G': 0.84}


def mag_to_flux(T):
    """TESS magnitude -> ELK flux in e/s."""
    return 10 ** (-0.4 * (T - T_ZP))


def flux_to_mag(F):
    import numpy as np
    F = np.asarray(F, float)
    return np.where(F > 0, -2.5 * np.log10(np.where(F > 0, F, 1)) + T_ZP, np.nan)

"""Local field light from Gaia DR3 around each cluster.

Two queries per cluster, both cached:

1. **Radial profile** (server-side aggregate).  Flux in the TESS band summed in rings of
   0.25 r_ap out to R_OUT r_ap.  Aggregating in ADQL keeps a 0.75 deg cone in the Galactic
   plane to a few dozen returned rows instead of ~1e6 stars.
2. **Bright stars** (row level) inside the aperture plus a margin: positions, photometry,
   parallax and proper motion, for PSF-weighted lumpiness statistics and to flag
   unambiguous Milky Way foreground stars in front of the Magellanic Clouds.

TESS magnitudes follow Stassun et al. (2019, AJ 158, 138, eq. 1):
    T = G - 0.00522555 c^3 + 0.0891337 c^2 - 0.633923 c + 0.0324473,   c = BP - RP
and T = G - 0.430 when no colour is available.  Gaia's ADQL accepts neither CASE nor
COALESCE, so the colour / no-colour parts are summed separately (aggregates skip NULLs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import T_ZP
from .catalogs import tap, GAIA_TAP

BIN = 0.25                                  # ring width in units of r_ap
R_OUT = {'MW': 3.0, 'LMC': 5.0, 'SMC': 5.0}
BRIGHT = {'MW': dict(rmax=1.0, gmax=12.0), 'LMC': dict(rmax=2.0, gmax=19.0),
          'SMC': dict(rmax=2.0, gmax=19.0)}

_POLY = '(-0.00522555*POWER(bp_rp,3) + 0.0891337*POWER(bp_rp,2) - 0.633923*bp_rp + 0.0324473)'
NOCOLOR_T_MINUS_G = -0.430


def stassun_T(G, bp_rp):
    c = np.asarray(bp_rp, float)
    poly = -0.00522555 * c ** 3 + 0.0891337 * c ** 2 - 0.633923 * c + 0.0324473
    return np.where(np.isfinite(c), G + poly, G + NOCOLOR_T_MINUS_G)


def _key(name):
    return ''.join(ch if ch.isalnum() else '_' for ch in name)


def radial_profile(name, origin, ra, dec, r_ap_deg, refresh=False):
    """Rings of width BIN*r_ap: count, and TESS-band flux (e/s) of all Gaia sources."""
    rout = R_OUT[origin] * r_ap_deg
    w = BIN * r_ap_deg
    dist = f'DISTANCE(POINT(ra, dec), POINT({ra}, {dec}))'
    fG = f'POWER(10, -0.4*(phot_g_mean_mag - {T_ZP}))'
    q = (f'SELECT FLOOR({dist} / {w}) AS ibin, COUNT(*) AS n, COUNT(bp_rp) AS n_col, '
         f'SUM(POWER(10, -0.4*(phot_g_mean_mag + {_POLY} - {T_ZP}))) AS fT_col, '
         f'SUM({fG}) AS fG_all, SUM({fG} + 0*bp_rp) AS fG_col '
         f'FROM gaiadr3.gaia_source '
         f'WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {rout})) '
         f'GROUP BY ibin ORDER BY ibin')
    df = tap(q, url=GAIA_TAP, cache_name=f'gaia_prof_{_key(name)}', refresh=refresh)
    df.columns = [c.lower() for c in df.columns]
    df = df.fillna(0.0)
    nocol = (df.fg_all - df.fg_col).clip(lower=0) * 10 ** (-0.4 * NOCOLOR_T_MINUS_G)
    df['fT'] = df.ft_col + nocol
    df['r_in'] = df.ibin * BIN                         # in units of r_ap
    df['r_out'] = (df.ibin + 1) * BIN
    df['area_arcsec2'] = np.pi * ((df.r_out * r_ap_deg * 3600) ** 2 - (df.r_in * r_ap_deg * 3600) ** 2)
    return df[['ibin', 'r_in', 'r_out', 'n', 'n_col', 'fT', 'area_arcsec2']]


def bright_stars(name, origin, ra, dec, r_ap_deg, refresh=False):
    """Row-level Gaia sources brighter than BRIGHT[origin]['gmax'] within rmax*r_ap."""
    b = BRIGHT[origin]
    q = (f'SELECT TOP 5000 source_id, ra, dec, phot_g_mean_mag AS G, bp_rp, parallax, '
         f'parallax_error, pmra, pmra_error, pmdec, pmdec_error, ruwe, '
         f'DISTANCE(POINT(ra, dec), POINT({ra}, {dec})) AS sep_deg '
         f'FROM gaiadr3.gaia_source '
         f'WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {b["rmax"] * r_ap_deg})) '
         f'AND phot_g_mean_mag < {b["gmax"]}')
    df = tap(q, url=GAIA_TAP, cache_name=f'gaia_bright_{_key(name)}', refresh=refresh)
    df.columns = [c.lower() for c in df.columns]
    if len(df):
        df['T'] = stassun_T(df['g'].values, df['bp_rp'].values)
        df['fT'] = 10 ** (-0.4 * (df['T'] - T_ZP))
        df['r_rel'] = df.sep_deg / r_ap_deg
    return df


# Systemic proper motions (mas/yr) of the Clouds, Gaia EDR3 (Luri+2021 / Gaia Collab.).
PM_CLOUD = {'LMC': (1.85, 0.23), 'SMC': (0.69, -1.23)}


def flag_foreground(stars, origin):
    """Unambiguous MW foreground stars in an MC field.

    At 50-60 kpc an MC star has parallax ~0.02 mas, so parallax > 3 sigma (and > 0.2 mas) is
    foreground.  Independently, a proper motion more than 1.5 mas/yr (and 5 sigma) from the
    Cloud's systemic motion is foreground; internal MC motions are < ~0.5 mas/yr.
    """
    if origin not in PM_CLOUD or not len(stars):
        return np.zeros(len(stars), bool)
    plx = (stars.parallax > 3 * stars.parallax_error) & (stars.parallax > 0.2)
    dpa, dpd = stars.pmra - PM_CLOUD[origin][0], stars.pmdec - PM_CLOUD[origin][1]
    dpm = np.hypot(dpa, dpd)
    err = np.hypot(stars.pmra_error, stars.pmdec_error)
    pm = (dpm > 1.5) & (dpm > 5 * err)
    return (plx | pm).fillna(False).values


def psf_weight(r_rel, r_ap_px, sigma_px):
    """Fraction of a point source's flux at offset r_rel*r_ap that falls inside the aperture.

    For a circular Gaussian PSF the enclosed fraction of a source at offset d inside radius R
    is the non-central chi-square CDF with 2 dof: P(chi2 < (R/s)^2; nc = (d/s)^2).
    """
    from scipy.stats import ncx2
    R = r_ap_px / sigma_px
    d = np.asarray(r_rel) * r_ap_px / sigma_px
    return ncx2.cdf(R ** 2, 2, d ** 2)

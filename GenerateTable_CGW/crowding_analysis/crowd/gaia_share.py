"""Cluster share of the resolved (Gaia) light in each ELK aperture -- no TESS fluxes used.

    f_cl,Gaia = (F_in - mu_field * A_ap) / F_in

F_in     TESS-band flux (Stassun+2019 T(G, BP-RP)) of every Gaia source in the aperture.
         For the Magellanic Clouds each source is weighted by the fraction of its PSF that
         falls inside the aperture, since the apertures are only 2-3 TESS pixels across;
         sources outside the aperture contribute through the same weights.
mu_field field surface brightness from an annulus (flux / area, unweighted: a uniform
         field keeps its surface brightness under PSF convolution).

Data access differs by galaxy:
* Field (all galaxies): server-side sums grouped by ring (0.25 r_ap) x integer G magnitude
  out to 4 r_ap (MW) or 6 r_ap (MCs), keeping sum f and sum f^2, so magnitude cuts, colour
  ablations and stochastic errors are computed locally.  MW cones hold up to ~1e6 stars.
* In-aperture flux: MW from the same ring sums (PSF irrelevant at r_ap ~ 27 px); LMC/SMC
  from Gaia rows within 3 r_ap, PSF-weighted.

Uncertainty: the field inside the aperture is one random draw of field stars.  Its variance
is the compound-Poisson value  Var = (A_ap / A_ann) * sum_ann f_i^2 , which is dominated by the
few brightest field stars and is usually much larger than the error on the mean field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import T_ZP
from .catalogs import tap, GAIA_TAP
from .gaia_field import stassun_T, psf_weight, _POLY, NOCOLOR_T_MINUS_G, _key

RING = 0.25                                    # ring width, units of r_ap
R_QUERY = {'MW': 4.0, 'LMC': 6.0, 'SMC': 6.0}   # query radius, units of r_ap
ANNULI = {                                     # (inner, outer) in r_ap; first = default
    'MW': [(1.5, 2.5), (2.0, 3.0), (2.5, 4.0), (1.25, 2.0)],
    'LMC': [(2.0, 4.0), (1.5, 2.5), (3.0, 5.0), (4.0, 6.0)],
    'SMC': [(2.0, 4.0), (1.5, 2.5), (3.0, 5.0), (4.0, 6.0)],
}
R_ROWS = 3.0                                   # row-level radius for MC PSF weighting, r_ap
G_CUTS = [17.0, 18.0, 19.0, 20.0, 99.0]
G_BRIGHT = [10.0, 12.0, 13.0, 14.0, 15.0, 16.0]  # bright-end cuts: drop sources with G < gmin
PSF_SIGMAS = [0.5, 0.75, 1.0]                  # TESS PSF Gaussian sigma, pixels
PIX = 21.0


# ---------------------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------------------

def ring_mag_table(name, origin, ra, dec, r_ap_deg, refresh=False):
    """Server-side sums per (ring, integer G) out to R_QUERY r_ap (all galaxies)."""
    rq = R_QUERY[origin] * r_ap_deg
    w = RING * r_ap_deg
    dist = f'DISTANCE(POINT(ra, dec), POINT({ra}, {dec}))'
    fG = f'POWER(10, -0.4*(phot_g_mean_mag - {T_ZP}))'
    fTc = f'POWER(10, -0.4*(phot_g_mean_mag + {_POLY} - {T_ZP}))'
    q = (f'SELECT FLOOR({dist} / {w}) AS ibin, FLOOR(phot_g_mean_mag) AS gbin, '
         f'COUNT(*) AS n, COUNT(bp_rp) AS n_col, '
         f'SUM({fTc}) AS ft_col, SUM(POWER({fTc}, 2)) AS ft2_col, '
         f'SUM({fG}) AS fg_all, SUM({fG} + 0*bp_rp) AS fg_col, '
         f'SUM(POWER({fG}, 2)) AS fg2_all, SUM(POWER({fG}, 2) + 0*bp_rp) AS fg2_col '
         f'FROM gaiadr3.gaia_source '
         f'WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {rq})) '
         f'GROUP BY ibin, gbin')
    df = tap(q, url=GAIA_TAP, cache_name=f'gaiashare_ring_{_key(name)}', refresh=refresh)
    df.columns = [c.lower() for c in df.columns]
    df = df.fillna(0.0)
    k = 10 ** (-0.4 * NOCOLOR_T_MINUS_G)                  # no-colour: T = G - 0.43
    df['f_nocol'] = (df.fg_all - df.fg_col).clip(lower=0) * k
    df['f2_nocol'] = (df.fg2_all - df.fg2_col).clip(lower=0) * k ** 2
    df['f'] = df.ft_col + df.f_nocol
    df['f2'] = df.ft2_col + df.f2_nocol
    df['r_in'] = df.ibin * RING
    df['r_out'] = (df.ibin + 1) * RING
    return df


def mc_rows(name, ra, dec, r_ap_deg, origin, refresh=False, top=200000):
    """Gaia sources within R_ROWS r_ap of an LMC/SMC cluster (enough for PSF weights:
    beyond r_ap + 4 sigma_PSF a source puts < 1e-3 of its light in the aperture)."""
    rq = R_ROWS * r_ap_deg
    q = (f'SELECT TOP {top} source_id, ra, dec, phot_g_mean_mag AS g, bp_rp, '
         f'DISTANCE(POINT(ra, dec), POINT({ra}, {dec})) AS sep_deg '
         f'FROM gaiadr3.gaia_source '
         f'WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {rq}))')
    df = tap(q, url=GAIA_TAP, cache_name=f'gaiashare_mcrows_{_key(name)}', refresh=refresh)
    df.columns = [c.lower() for c in df.columns]
    df = df[np.isfinite(df.g)].copy()                     # a few sources lack G photometry
    if len(df) >= top:
        raise RuntimeError(f'{name}: Gaia query hit TOP {top}; widen the limit')
    df['T'] = stassun_T(df.g.values, df.bp_rp.values)
    df['f'] = 10 ** (-0.4 * (df['T'] - T_ZP))
    df['has_col'] = np.isfinite(df.bp_rp.values)
    df['r_rel'] = df.sep_deg / r_ap_deg
    return df


# ---------------------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------------------

def _share(F_in, mu, A_ap, var_field, F_bright=0.0):
    """f = (F_in - mu A_ap) / F_in ; f_drop1 reassigns the brightest in-aperture source to
    the field (it stays in the denominator)."""
    F_field = mu * A_ap
    if not F_in > 0:
        return dict(f=np.nan, F_in=F_in, F_field=F_field, sig=np.nan, f_drop1=np.nan, frac_bright=np.nan)
    return dict(f=(F_in - F_field) / F_in, F_in=F_in, F_field=F_field,
                sig=np.sqrt(var_field) / F_in, f_drop1=(F_in - F_bright - F_field) / F_in,
                frac_bright=F_bright / F_in)


def _field(tab, r_ap_arcsec, annulus, gmax, nocol_shift, drop_nocol, gmin=-99.0):
    """Field surface brightness and the stochastic variance of an aperture-sized patch."""
    t = tab[(tab.gbin < gmax) & (tab.gbin >= gmin)]
    fn = 0.0 if drop_nocol else 10 ** (-0.4 * nocol_shift)
    ff = t.ft_col + fn * t.f_nocol
    ff2 = t.ft2_col + fn ** 2 * t.f2_nocol
    lo, hi = annulus
    m = (t.r_in >= lo - 1e-9) & (t.r_out <= hi + 1e-9)
    A_ann = np.pi * r_ap_arcsec ** 2 * (hi ** 2 - lo ** 2)
    return ff[m].sum() / A_ann, ff2[m].sum() / A_ann, t, ff


def share_mw(tab, bright, r_ap_arcsec, annulus=(1.5, 2.5), gmax=99.0, nocol_shift=0.0,
             drop_nocol=False, gmin=-99.0):
    """Cluster share for a MW cluster (ring x G table; PSF irrelevant at r_ap ~ 27 px)."""
    mu, mu2, t, ff = _field(tab, r_ap_arcsec, annulus, gmax, nocol_shift, drop_nocol, gmin)
    A_ap = np.pi * r_ap_arcsec ** 2
    F_in = ff[t.r_out <= 1 + 1e-9].sum()
    b = bright[(bright.r_rel <= 1) & (bright.g < gmax) & (bright.g >= gmin)] if len(bright) else bright
    F_b = float(b.fT.max()) if len(b) else 0.0
    return _share(F_in, mu, A_ap, A_ap * mu2, F_b)


def share_mc(tab, rows, r_ap_arcsec, annulus=(2.0, 4.0), gmax=99.0, sigma_px=0.75,
             radius_scale=1.0, nocol_shift=0.0, drop_nocol=False, gmin=-99.0):
    """PSF-weighted cluster share for an LMC/SMC cluster.

    radius_scale rescales the aperture (0.5 = the true Bica radius for the SMC); the annulus
    stays defined in units of the ELK radius so only the aperture changes.
    """
    mu, mu2, _, _ = _field(tab, r_ap_arcsec, annulus, gmax, nocol_shift, drop_nocol, gmin)
    d = rows[(rows.g < gmax) & (rows.g >= gmin)]
    f = d.f.values * np.where(d.has_col.values, 1.0, 0.0 if drop_nocol else 10 ** (-0.4 * nocol_shift))
    r_ap = r_ap_arcsec * radius_scale
    w = psf_weight(d.r_rel.values / radius_scale, r_ap / PIX, sigma_px)
    wf = w * f
    A_ap = np.pi * r_ap ** 2
    # stochastic variance of the PSF-weighted field in the aperture: mu2 * integral(w^2 dA);
    # integral(w^2) ~ A_ap for an aperture much larger than the PSF, smaller otherwise.
    return _share(np.sum(wf), mu, A_ap, A_ap * mu2, float(wf.max()) if len(wf) else 0.0)


def nocol_flux_fraction(tab):
    t = tab[tab.r_out <= 1 + 1e-9]
    return t.f_nocol.sum() / t.f.sum() if t.f.sum() > 0 else np.nan


def measure_all(clusters, verbose=True):
    """Default measurement + every ablation, one row per cluster."""
    from .gaia_field import bright_stars
    out = []
    for _, r in clusters.iterrows():
        o = r.origin
        tab = ring_mag_table(r['name'], o, r.ra, r.dec, r.r_ap_deg)
        row = dict(name=r['name'], origin=o, age=r.age, r_ap_arcsec=r.r_ap_arcsec,
                   nocol_frac=nocol_flux_fraction(tab))
        if o == 'MW':
            bright = bright_stars(r['name'], o, r.ra, r.dec, r.r_ap_deg)
            fn = lambda **kw: share_mw(tab, bright, r.r_ap_arcsec, **kw)
        else:
            rows = mc_rows(r['name'], r.ra, r.dec, r.r_ap_deg, o)
            row['n_rows'] = len(rows)
            fn = lambda **kw: share_mc(tab, rows, r.r_ap_arcsec, **kw)
        a0 = ANNULI[o][0]
        res = fn(annulus=a0)
        row.update(f_gaia=res['f'], F_in_gaia=res['F_in'], F_field_gaia=res['F_field'],
                   sig_stoch=res['sig'], f_drop1=res['f_drop1'], frac_brightest=res['frac_bright'])
        for a in ANNULI[o][1:]:
            row[f'f_ann_{a[0]:g}_{a[1]:g}'] = fn(annulus=a)['f']
        for g in G_CUTS[:-1]:
            row[f'f_G{g:g}'] = fn(annulus=a0, gmax=g)['f']
        for g in G_BRIGHT:
            rb = fn(annulus=a0, gmin=g)
            row[f'f_Gmin{g:g}'] = rb['f']
            row[f'sig_Gmin{g:g}'] = rb['sig']
        row['f_nocol_drop'] = fn(annulus=a0, drop_nocol=True)['f']
        row['f_nocol_p05'] = fn(annulus=a0, nocol_shift=+0.5)['f']
        row['f_nocol_m05'] = fn(annulus=a0, nocol_shift=-0.5)['f']
        if o != 'MW':
            for sgm in PSF_SIGMAS:
                if sgm != 0.75:
                    row[f'f_psf_{sgm:g}'] = fn(annulus=a0, sigma_px=sgm)['f']
            if o == 'SMC':
                rt = fn(annulus=a0, radius_scale=0.5)
                row['f_true_radius'] = rt['f']
                row['sig_true_radius'] = rt['sig']
        out.append(row)
    df = pd.DataFrame(out)
    if verbose:
        print(df.groupby('origin').f_gaia.describe()[['count', '25%', '50%', '75%']].round(3))
    return df

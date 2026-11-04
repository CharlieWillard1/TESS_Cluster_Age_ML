"""Smooth models of the field surface brightness behind / in front of each cluster (method A).

Everything returns a TESS-band surface brightness in ELK e/s per arcsec^2, so the expected
field flux in an aperture of area A is simply  mu * A  (a uniform field keeps its surface
brightness under PSF convolution, so no PSF term is needed).

These are order-of-magnitude models: smooth exponentials normalised to the galaxies' total
luminosities.  They deliberately ignore structure (the LMC bar, spiral arms, star-forming
complexes); the Gaia measurement (method B) captures that, and the ratio B/A is itself a
diagnostic.  Parameter sources are given next to each value; all are rounded literature
values, not fits.
"""
from __future__ import annotations

import numpy as np

from . import T_ZP

ARCSEC2_SR = (np.pi / 180 / 3600) ** 2           # sr per arcsec^2
M_SUN_T = 4.10                                   # solar absolute mag, ~ I_C (Willmer 2018)


def _mag_to_flux(m):
    return 10 ** (-0.4 * (m - T_ZP))


# ---------------------------------------------------------------------------------------
# Magellanic Clouds: exponential disks normalised to total luminosity
# ---------------------------------------------------------------------------------------

LMC = dict(
    ra0=81.90, dec0=-69.87,        # centre of the outer isophotes (van der Marel 2001)
    incl=34.7, theta=122.5,        # inclination, line-of-nodes PA (van der Marel & Cioni 2001)
    D_kpc=50.0,                    # (m-M)_0 = 18.5, as Glatt+2010 / Popescu+2012 assume
    Rd_kpc=1.5,                    # exponential scale length (van der Marel 2001: ~1.5 kpc)
    MV=-18.1,                      # McConnachie 2012
    V_minus_T=0.80,                # composite population, ~ (V-I) of the LMC disk
)
SMC = dict(
    ra0=13.19, dec0=-72.83,        # optical centre (McConnachie 2012)
    D_kpc=62.0,                    # (m-M)_0 = 18.96
    Rd_kpc=0.7,                    # projected exponential scale of the older population
    MV=-16.8,                      # McConnachie 2012
    V_minus_T=0.70,
)


def _sky_offsets(ra, dec, ra0, dec0):
    """Angular distance rho and position angle phi (N through E) from (ra0, dec0), radians."""
    a, d = np.radians(ra), np.radians(dec)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    cosr = np.cos(d) * np.cos(d0) * np.cos(a - a0) + np.sin(d) * np.sin(d0)
    rho = np.arccos(np.clip(cosr, -1, 1))
    phi = np.arctan2(np.cos(d) * np.sin(a - a0),
                     np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(a - a0))
    return rho, phi


def lmc_disk_radius(ra, dec, p=LMC):
    """In-plane galactocentric radius (kpc) of the LMC disk point seen along (ra, dec).

    van der Marel & Cioni (2001, eqs. 1-3, 7): the line of sight at (rho, phi) meets an
    inclined plane at distance D = D0 cos i / (cos i cos rho - sin i sin rho sin(phi-theta)).
    The near side is at phi = theta - 90 deg.
    """
    rho, phi = _sky_offsets(ra, dec, p['ra0'], p['dec0'])
    i, th = np.radians(p['incl']), np.radians(p['theta'])
    D0 = p['D_kpc']
    D = D0 * np.cos(i) / (np.cos(i) * np.cos(rho) - np.sin(i) * np.sin(rho) * np.sin(phi - th))
    # cartesian sky-plane coordinates (x along line of nodes), then deproject
    x = D * np.sin(rho) * np.cos(phi - th)
    y = D * (np.sin(rho) * np.cos(i) * np.sin(phi - th) + np.cos(rho) * np.sin(i)) - D0 * np.sin(i)
    z = D * (np.sin(rho) * np.sin(i) * np.sin(phi - th) - np.cos(rho) * np.cos(i)) + D0 * np.cos(i)
    return np.sqrt(x ** 2 + y ** 2 + z ** 2), D


def lmc_sb(ra, dec, p=LMC):
    """LMC field surface brightness (e/s/arcsec^2) from an inclined exponential disk."""
    R, D = lmc_disk_radius(ra, dec, p)
    Rd = p['Rd_kpc']
    frac_per_kpc2 = np.exp(-R / Rd) / (2 * np.pi * Rd ** 2)          # face-on, in-plane
    # one arcsec^2 on the sky covers D^2 * dOmega / cos(i) of disk
    frac_per_arcsec2 = frac_per_kpc2 * (D ** 2) * ARCSEC2_SR / np.cos(np.radians(p['incl']))
    m_tot = p['MV'] - p['V_minus_T'] + 5 * np.log10(p['D_kpc'] * 1e3 / 10)
    return _mag_to_flux(m_tot) * frac_per_arcsec2


def smc_sb(ra, dec, p=SMC):
    """SMC field surface brightness (e/s/arcsec^2), projected circular exponential."""
    rho, _ = _sky_offsets(ra, dec, p['ra0'], p['dec0'])
    R = p['D_kpc'] * rho                                             # projected kpc
    Rd = p['Rd_kpc']
    frac_per_arcsec2 = np.exp(-R / Rd) / (2 * np.pi * Rd ** 2) * p['D_kpc'] ** 2 * ARCSEC2_SR
    m_tot = p['MV'] - p['V_minus_T'] + 5 * np.log10(p['D_kpc'] * 1e3 / 10)
    return _mag_to_flux(m_tot) * frac_per_arcsec2


# ---------------------------------------------------------------------------------------
# Milky Way: line-of-sight integral through thin + thick disks with an exponential dust layer
# ---------------------------------------------------------------------------------------

MW = dict(
    R0=8.2, z_sun=0.02,                    # kpc (Bland-Hawthorn & Gerhard 2016)
    hR=2.6, hz_thin=0.30, hz_thick=0.90,   # kpc (Juric+2008; BH&G 2016)
    f_thick=0.12,                          # thick/thin local density ratio (Juric+2008)
    Sigma_sun=29.0,                        # local disk surface brightness, Lsun,I / pc^2
                                           # (Flynn+2006, I band ~ TESS band)
    aV0=0.7, hz_dust=0.12, hR_dust=3.0,    # dust: A_V per kpc in the plane at R0, scales
    AT_AV=0.60,
    s_max=60.0,                            # integrate to 60 kpc (covers the Clouds)
)


def mw_sb(l_deg, b_deg, s_max=None, p=MW, n=4000):
    """MW starlight surface brightness (e/s/arcsec^2) along (l, b), integrated over the whole
    line of sight (foreground and background) with dust attenuation.

    j(R,z) = j0 exp(-(R-R0)/hR) [exp(-|z|/hz_thin) + f_thick exp(-|z|/hz_thick)], with j0
    fixed by Sigma_sun = 2 j0 (hz_thin + f_thick hz_thick).  The flux per steradian is
    (1/4pi) Int j e^-tau ds; dividing by a 1-Lsun source at 10 pc gives the magnitude.
    """
    s_max = p['s_max'] if s_max is None else s_max
    l, b = np.radians(l_deg), np.radians(b_deg)
    s = np.geomspace(1e-4, s_max, n)                                    # kpc
    x = p['R0'] - s * np.cos(b) * np.cos(l)
    y = -s * np.cos(b) * np.sin(l)
    z = s * np.sin(b) + p['z_sun']
    R = np.hypot(x, y)
    j0 = p['Sigma_sun'] / (2 * (p['hz_thin'] + p['f_thick'] * p['hz_thick']) * 1e3)  # Lsun/pc^3
    j = j0 * np.exp(-(R - p['R0']) / p['hR']) * (np.exp(-abs(z) / p['hz_thin'])
                                                + p['f_thick'] * np.exp(-abs(z) / p['hz_thick']))
    dAds = p['aV0'] * p['AT_AV'] * np.exp(-(R - p['R0']) / p['hR_dust']) * np.exp(-abs(z) / p['hz_dust'])
    tau = 0.4 * np.log(10) * np.concatenate([[0], np.cumsum(0.5 * (dAds[1:] + dAds[:-1]) * np.diff(s))])
    integrand = j * np.exp(-tau)                                        # Lsun / pc^3
    col = np.trapezoid(integrand, s * 1e3)                                  # Lsun / pc^2
    ratio = col * ARCSEC2_SR * 100.0                                    # vs 1 Lsun at 10 pc
    return _mag_to_flux(M_SUN_T - 2.5 * np.log10(ratio))


# ---------------------------------------------------------------------------------------
# Zodiacal light
# ---------------------------------------------------------------------------------------

def zodi_sb(ecl_lat_deg):
    """Zodiacal light (e/s/arcsec^2) vs ecliptic latitude, Sullivan+2015 eq. 18:
    V = 23.345 - 1.148 x^2,  x = (|beta| - 90)/90, converted to T with the solar V - I ~ 0.7.
    """
    x = (np.abs(ecl_lat_deg) - 90.0) / 90.0
    return _mag_to_flux(23.345 - 1.148 * x ** 2 - 0.70)


def field_model_sb(origin, ra, dec, l_deg, b_deg):
    """Method A total: host-galaxy disk (MCs) + Milky Way starlight along the sightline."""
    mw = np.array([mw_sb(l, b) for l, b in zip(np.atleast_1d(l_deg), np.atleast_1d(b_deg))])
    host = np.zeros_like(mw)
    o = np.atleast_1d(origin)
    host = np.where(o == 'LMC', lmc_sb(np.atleast_1d(ra), np.atleast_1d(dec)), host)
    host = np.where(o == 'SMC', smc_sb(np.atleast_1d(ra), np.atleast_1d(dec)), host)
    return host, mw

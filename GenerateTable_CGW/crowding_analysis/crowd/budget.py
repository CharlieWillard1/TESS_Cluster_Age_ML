"""Assemble the per-cluster light budget: cluster light, field light (A and B), sky, closure.

All fluxes are TESS-band ELK e/s inside the ELK aperture.  Column naming:
    F_meas          measured: median over sectors of the per-sector median ELK flux
    F_cl_phot       cluster light from literature integrated photometry (primary)
    F_cl_mass       cluster light from mass x SSP light-per-mass (cross-check)
    F_cl_gaia       MW only: Gaia flux inside r_ap minus annulus field (cross-check)
    F_field_A       smooth galaxy model (host disk + MW starlight), mu * A_ap
    F_field_B       Gaia annulus surface brightness * A_ap (resolved sources only)
    F_sky           zodiacal light * A_ap, if ELK's median flux kept it
    f_cl            F_cl_adopted / F_meas
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import A_RATIO, mag_to_flux
from . import catalogs as C
from . import gaia_field as G
from . import galaxy_models as M
from .ssp import SSP

PIX_ARCSEC = 21.0
PSF_SIGMA_PX = 0.75                 # TESS PSF ~ Gaussian, FWHM ~ 1.8 px (range 0.5-1.0 px)
ANNULUS = {'MW': (1.5, 2.5), 'LMC': (2.0, 4.0), 'SMC': (2.0, 4.0)}
DM0 = {'LMC': 18.50, 'SMC': 18.90}  # moduli adopted by Glatt+2010 / Popescu+2012


# ---------------------------------------------------------------------------------------
# Aperture enclosure of the cluster light (Magellanic Clouds only)
# ---------------------------------------------------------------------------------------

def plummer_psf_enclosed(a_rel, sigma_rel, n=200_000, seed=0):
    """Fraction of a projected Plummer cluster (scale a = a_rel * r_ap) that lands inside the
    aperture after convolution with a Gaussian PSF (sigma = sigma_rel * r_ap).

    Returns (enclosed_after_psf, enclosed_intrinsic).  The intrinsic Plummer enclosed fraction
    is R^2 / (R^2 + a^2); positions are drawn from its inverse CDF, smeared by the PSF, and
    counted inside R = 1.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    r = a_rel * np.sqrt(u / (1 - u))
    th = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(th) + rng.normal(0, sigma_rel, n)
    y = r * np.sin(th) + rng.normal(0, sigma_rel, n)
    return np.mean(x ** 2 + y ** 2 < 1.0), 1.0 / (1.0 + a_rel ** 2)


def enclosure_factors(r_ap_px, aperture_light):
    """Multiplicative factor converting a literature cluster luminosity into the light the ELK
    aperture actually collects, with a (nominal, low, high) range.

    aperture_light=True : the literature value already counts only light inside r_ap
                          (Glatt+2010 sums stars within R_app), so factor = after/intrinsic.
    aperture_light=False: the literature value is total light, so factor = after-PSF fraction.
    Grid: Plummer a in r_ap * {1/4, 1/3, 1/2}; PSF sigma in {0.5, 0.75, 1.0} px.
    """
    vals = []
    for a in (0.25, 1 / 3, 0.5):
        for s_px in (0.5, PSF_SIGMA_PX, 1.0):
            after, intr = plummer_psf_enclosed(a, s_px / r_ap_px)
            vals.append(after / intr if aperture_light else after)
    nominal = plummer_psf_enclosed(1 / 3, PSF_SIGMA_PX / r_ap_px)
    nominal = nominal[0] / nominal[1] if aperture_light else nominal[0]
    return nominal, min(vals), max(vals)


# ---------------------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------------------

def build_budget(verbose=True):
    cl = C.cluster_list().copy()
    ssp = SSP()
    cl['r_ap_px'] = cl.r_ap_arcsec / PIX_ARCSEC

    # ---------------- cluster light: literature photometry --------------------------
    glatt = C.glatt2010()
    raf = C.rafelski2005()
    k16 = C.kharchenko2016()
    pop = C.popescu2012()
    hr = C.hunt_reffert2024()

    out = []
    for origin, s in cl.groupby('origin'):
        s = s.copy()
        age = s.age.values
        if origin in ('LMC', 'SMC'):
            g = C.crossmatch(s, glatt[glatt.galaxy == origin], 5, frac_of_radius=0.25,
                             name_col=['SimbadName', 'Cross-ID'])
            AV_g = 3.1 * g['E(B-V)'].astype(float).values
            T_g = (g.Vmag.astype(float).values - ssp.color('V', 'T', age, origin)
                   - (A_RATIO['V'] - A_RATIO['T']) * AV_g)
            s['T_cl_glatt'] = T_g
            s['EBV_glatt'] = g['E(B-V)'].astype(float).values
            src = np.where(np.isfinite(T_g), 'glatt2010', '')
            if origin == 'SMC':
                rz = C.crossmatch(s, raf, 10, frac_of_radius=0.25, name_col='Name')
                I_obs = rz.Vmag.astype(float).values - rz['V-I'].astype(float).values
                T_rz = I_obs + ssp.color('T', 'I', age, origin)       # A_T ~ A_I
                s['T_cl_rafelski'] = T_rz
                use_rz = ~np.isfinite(T_g) & np.isfinite(T_rz)
                T_g = np.where(use_rz, T_rz, T_g)
                src = np.where(use_rz, 'rafelski2005', src)
            s['T_cl_phot'] = T_g
            s['phot_source'] = src
            # Glatt counts light inside R_app (= ELK radius); Rafelski is a total magnitude.
            fac = [enclosure_factors(rpx, src_i == 'glatt2010') for rpx, src_i in zip(s.r_ap_px, src)]
            s['enc'], s['enc_lo'], s['enc_hi'] = np.array(fac).T
            # ---- mass route (LMC: Popescu+2012) ----
            if origin == 'LMC':
                pm = C.crossmatch(s, pop, 10, frac_of_radius=0.25, name_col='Name')
                mass = pm.Mass.astype(float).values
                AV = np.where(np.isfinite(AV_g), AV_g, 0.31)
                T_m = (ssp.mag('T', age, origin) - 2.5 * np.log10(mass) + DM0[origin]
                       + A_RATIO['T'] * AV)
                s['mass'] = mass
                s['T_cl_mass'] = T_m
                encm = [enclosure_factors(rpx, False)[0] for rpx in s.r_ap_px]
                s['F_cl_mass'] = mag_to_flux(T_m) * np.array(encm)
            s['F_cl_phot'] = mag_to_flux(s.T_cl_phot) * s.enc
        else:   # MW
            k = C.crossmatch(s, k16, 120, frac_of_radius=0, name_col='Name')
            J_obs = k['I(M_J)'].astype(float).values + k['DM(J)'].astype(float).values
            AV = 3.1 * k['E(B-V)'].astype(float).values
            T_k = J_obs + ssp.color('T', 'J', age, origin) + (A_RATIO['T'] - A_RATIO['J']) * AV
            s['T_cl_phot'] = T_k
            s['phot_source'] = np.where(np.isfinite(T_k), 'kharchenko2016', '')
            s['EBV_k16'] = k['E(B-V)'].astype(float).values
            s['enc'] = s['enc_lo'] = s['enc_hi'] = 1.0      # r_ap = r2 >> PSF
            s['F_cl_phot'] = mag_to_flux(T_k)
            h = C.crossmatch(s, hr, 120, frac_of_radius=0, name_col='AllNames')
            mass = h.MassTot.astype(float).values
            T_m = (ssp.mag('T', age, origin) - 2.5 * np.log10(mass)
                   + h.MOD50.astype(float).values + A_RATIO['T'] * h.AV50.astype(float).values)
            s['mass'] = mass
            s['T_cl_mass'] = T_m
            s['F_cl_mass'] = mag_to_flux(T_m)
        out.append(s)
    cl = pd.concat(out).sort_index()

    # ---------------- field light ------------------------------------------------------
    host, mw = M.field_model_sb(cl.origin.values, cl.ra.values, cl.dec.values,
                                cl.gal_l.values, cl.gal_b.values)
    cl['mu_host_A'] = host
    cl['mu_mw_A'] = mw
    cl['F_field_A'] = (host + mw) * cl.A_ap_arcsec2
    cl['F_sky'] = M.zodi_sb(cl.ecl_lat.values) * cl.A_ap_arcsec2

    gaia_cols = []
    for i, r in cl.iterrows():
        p = G.radial_profile(r['name'], r.origin, r.ra, r.dec, r.r_ap_deg)
        lo, hi = ANNULUS[r.origin]
        ann = p[(p.r_in >= lo - 1e-9) & (p.r_out <= hi + 1e-9)]
        mu_B = ann.fT.sum() / ann.area_arcsec2.sum()
        F_in = p[p.r_out <= 1 + 1e-9].fT.sum()
        n_in = p[p.r_out <= 1 + 1e-9].n.sum()
        b = G.bright_stars(r['name'], r.origin, r.ra, r.dec, r.r_ap_deg)
        row = dict(mu_B=mu_B, F_field_B=mu_B * r.A_ap_arcsec2, F_gaia_in=F_in, n_gaia_in=n_in,
                   n_field_in=ann.n.sum() / ann.area_arcsec2.sum() * r.A_ap_arcsec2)
        if len(b):
            if r.origin == 'MW':
                w = (b.r_rel <= 1).astype(float).values
            else:
                w = G.psf_weight(b.r_rel.values, r.r_ap_px, PSF_SIGMA_PX)
            fw = w * b.fT.values
            fg = G.flag_foreground(b, r.origin)
            k = int(np.argmax(fw))
            row.update(F_brightest=fw[k], T_brightest=b['T'].values[k],
                       r_brightest=b.r_rel.values[k], brightest_is_fg=bool(fg[k]),
                       F_foreground=float(fw[fg].sum()), n_foreground=int((fg & (w > 0.1)).sum()))
        else:
            row.update(F_brightest=0.0, T_brightest=np.nan, r_brightest=np.nan,
                       brightest_is_fg=False, F_foreground=0.0, n_foreground=0)
        gaia_cols.append(row)
    gaia = pd.DataFrame(gaia_cols, index=cl.index)
    cl = pd.concat([cl, gaia], axis=1)

    # MW: statistical cluster light from Gaia (in-aperture minus annulus field)
    cl['F_cl_gaia'] = np.where(cl.origin == 'MW', cl.F_gaia_in - cl.F_field_B, np.nan)

    # ---------------- adopted cluster light, f_cl, closure ----------------------------
    cl['F_cl'] = cl.F_cl_phot
    routes = cl[['F_cl_phot', 'F_cl_mass', 'F_cl_gaia']].where(lambda d: d > 0)
    enc_lo = (cl.enc_lo / cl.enc).fillna(1.0)
    enc_hi = (cl.enc_hi / cl.enc).fillna(1.0)
    cl['F_cl_lo'] = np.fmin(routes.min(axis=1), cl.F_cl * enc_lo)
    cl['F_cl_hi'] = np.fmax(routes.max(axis=1), cl.F_cl * enc_hi)
    cl['f_cl'] = cl.F_cl / cl.F_meas
    cl['f_cl_lo'] = cl.F_cl_lo / cl.F_meas
    cl['f_cl_hi'] = cl.F_cl_hi / cl.F_meas
    # Share of the *astrophysical* light (cluster + field stars): the zodiacal pedestal is
    # constant within a sector and carries no variability, so remove it from the denominator.
    # Clusters where the sky model exceeds half of F_meas are left NaN (model too uncertain).
    F_astro = (cl.F_meas - cl.F_sky).where(cl.F_meas - cl.F_sky > 0.5 * cl.F_meas)
    cl['F_astro'] = F_astro
    cl['f_cl_astro'] = cl.F_cl / F_astro
    cl['f_cl_astro_lo'] = cl.F_cl_lo / F_astro
    cl['f_cl_astro_hi'] = cl.F_cl_hi / F_astro
    cl['closure_B'] = cl.F_meas / (cl.F_cl + cl.F_field_B)
    cl['closure_A'] = cl.F_meas / (cl.F_cl + cl.F_field_A)
    cl['closure_A_sky'] = cl.F_meas / (cl.F_cl + cl.F_field_A + cl.F_sky)
    cl['B_over_A'] = cl.F_field_B / cl.F_field_A
    cl['frac_brightest'] = cl.F_brightest / cl.F_meas
    cl['frac_foreground'] = cl.F_foreground / cl.F_meas

    if verbose:
        cov = cl.groupby('origin').agg(n=('name', 'size'),
                                       phot=('F_cl_phot', lambda x: np.isfinite(x).sum()),
                                       mass=('F_cl_mass', lambda x: np.isfinite(x).sum()))
        print(cov)
    return cl

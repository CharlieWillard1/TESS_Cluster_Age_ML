"""MW cluster share from catalogued members (Hunt & Reffert 2024) -- no field subtraction.

    f_mem = F_members(< r_ap) / F_Gaia,all(< r_ap)

Both fluxes use the same Gaia DR3 photometry and the same Stassun+2019 T(G, BP-RP) conversion
as `gaia_share`, so the only difference from the annulus method is how cluster light is
identified: by astrometric membership rather than by subtracting an average field.  f_mem
cannot be negative and carries no bright-field-star sampling noise.  Its errors are
membership completeness (missed members -> low) and contamination (field stars labelled
members -> high).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import T_ZP
from .catalogs import tap, crossmatch, hunt_reffert2024
from .gaia_field import stassun_T, _key
from .gaia_share import ring_mag_table, ANNULI

HR24_MEMBERS = '"J/A+A/686/A42/members"'
PROB_CUTS = [0.0, 0.5, 0.7, 0.9]


def match_hr24(clusters):
    """HR24 entry for each MW cluster (name match within 2 r_ap, else nearest within 120")."""
    mw = clusters[clusters.origin == 'MW'].copy()
    hr = hunt_reffert2024()
    m = crossmatch(mw, hr, 120, frac_of_radius=0, name_col='AllNames')
    mw['hr_name'] = m.Name.values
    mw['hr_sep_arcsec'] = m.sep_arcsec.values
    mw['hr_r50_deg'] = m.r50.values
    mw['hr_rt_deg'] = m.rt.values
    mw['hr_logage'] = m.logAge50.values
    mw['hr_mod'] = m.MOD50.values
    return mw


def members(hr_name, refresh=False):
    """All HR24 members of one cluster with Gaia photometry and membership probability."""
    q = (f'SELECT GaiaDR3, Prob, inrt, RA_ICRS AS ra, DE_ICRS AS dec, Gmag AS g, '
         f'"BP-RP" AS bp_rp FROM {HR24_MEMBERS} WHERE Name = \'{hr_name}\'')
    df = tap(q, cache_name=f'hr24mem_{_key(hr_name)}', refresh=refresh)
    df = df[np.isfinite(df.g)].copy()
    df['T'] = stassun_T(df.g.values, df.bp_rp.values)
    df['f'] = 10 ** (-0.4 * (df['T'] - T_ZP))
    return df


def _sep_rel(df, ra0, dec0, r_ap_deg):
    d2r = np.pi / 180
    c = (np.sin(dec0 * d2r) * np.sin(df.dec.values * d2r)
         + np.cos(dec0 * d2r) * np.cos(df.dec.values * d2r) * np.cos((df.ra.values - ra0) * d2r))
    return np.degrees(np.arccos(np.clip(c, -1, 1))) / r_ap_deg


def member_share(r, mem, tab, prob_min=0.0, gmax=99.0, inrt_only=False, annulus=None, gmin=-99.0):
    """Member share of the resolved light in the ELK aperture, plus a field diagnostic.

    Returns F_in (all Gaia), F_mem (members in aperture), f_mem, and the non-member flux in
    the aperture compared with the non-member field expected from the annulus
    (members removed from both), which tests for a field gradient.
    """
    annulus = annulus or ANNULI['MW'][0]
    m = mem[(mem.Prob >= prob_min) & (mem.g < gmax) & (mem.g >= gmin)]
    if inrt_only:
        m = m[m.inrt == 1]
    rr = _sep_rel(m, r.ra, r.dec, r.r_ap_deg)
    t = tab[(tab.gbin < gmax) & (tab.gbin >= gmin)]
    F_in = t.f[t.r_out <= 1 + 1e-9].sum()
    lo, hi = annulus
    ring = (t.r_in >= lo - 1e-9) & (t.r_out <= hi + 1e-9)
    F_ann = t.f[ring].sum()
    F_mem = m.f[rr <= 1].sum()
    F_mem_ann = m.f[(rr >= lo) & (rr <= hi)].sum()
    area_ratio = hi ** 2 - lo ** 2                       # A_ann / A_ap
    field_pred = max(F_ann - F_mem_ann, 0.0) / area_ratio
    field_in = F_in - F_mem
    return dict(F_in=F_in, F_mem=F_mem, f_mem=F_mem / F_in if F_in > 0 else np.nan,
                n_mem_in=int((rr <= 1).sum()), F_mem_ann_frac=F_mem_ann / F_ann if F_ann > 0 else np.nan,
                field_in=field_in, field_pred=field_pred,
                field_ratio=field_in / field_pred if field_pred > 0 else np.nan,
                f_mem_brightest=m.f[rr <= 1].max() / F_mem if F_mem > 0 else np.nan)


def measure_all(clusters, verbose=True):
    """One row per MW cluster: default member share + ablations."""
    mw = match_hr24(clusters)
    out = []
    for _, r in mw.iterrows():
        row = dict(name=r['name'], age=r.age, r_ap_arcsec=r.r_ap_arcsec, hr_name=r.hr_name,
                   hr_sep_arcsec=r.hr_sep_arcsec, hr_r50_over_rap=r.hr_r50_deg / r.r_ap_deg,
                   hr_logage=r.hr_logage)
        if not isinstance(r.hr_name, str):
            out.append(row)
            continue
        mem = members(r.hr_name)
        tab = ring_mag_table(r['name'], 'MW', r.ra, r.dec, r.r_ap_deg)
        row['n_mem_total'] = len(mem)
        row['g_mem_faintest'] = mem.g.max() if len(mem) else np.nan
        row.update(member_share(r, mem, tab))
        for p in PROB_CUTS[1:]:
            row[f'f_mem_p{p:g}'] = member_share(r, mem, tab, prob_min=p)['f_mem']
        row['f_mem_inrt'] = member_share(r, mem, tab, inrt_only=True)['f_mem']
        for g in [16.0, 17.0, 18.0, 19.0]:
            row[f'f_mem_G{g:g}'] = member_share(r, mem, tab, gmax=g)['f_mem']
        r14 = member_share(r, mem, tab, gmin=14.0)
        row['f_mem_Gmin14'], row['field_ratio_Gmin14'] = r14['f_mem'], r14['field_ratio']
        for a in ANNULI['MW'][1:]:
            row[f'field_ratio_{a[0]:g}_{a[1]:g}'] = member_share(r, mem, tab, annulus=a)['field_ratio']
        out.append(row)
    df = pd.DataFrame(out)
    if verbose:
        print(f'{df.hr_name.notna().sum()} / {len(df)} MW clusters matched to HR24')
        print(df.f_mem.describe().round(3).to_string())
    return df


def nonmember_profile(r, mem, tab, gmin=-99.0, annulus=None):
    """Surface brightness of all and of non-member light in rings of 0.25 r_ap out to 4 r_ap,
    normalised to the non-member surface brightness in the annulus.  A peak at the centre in
    the non-member curve is cluster light that the member list misses."""
    from .gaia_share import RING
    annulus = annulus or ANNULI['MW'][0]
    m = mem[mem.g >= gmin]
    t = tab[tab.gbin >= gmin]
    nb = int(round(4.0 / RING))
    edges = np.arange(nb + 1) * RING
    F = t.groupby('ibin').f.sum().reindex(range(nb), fill_value=0.0).values
    Fm = np.histogram(_sep_rel(m, r.ra, r.dec, r.r_ap_deg), edges, weights=m.f.values)[0]
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    ref_m = (edges[:-1] >= annulus[0] - 1e-9) & (edges[1:] <= annulus[1] + 1e-9)
    sb_non = (F - Fm) / area
    ref = sb_non[ref_m].mean()
    return edges, (F / area) / ref, sb_non / ref

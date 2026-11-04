"""Cluster list, measured ELK aperture flux, and literature catalogues via TAP.

Only `requests` is used for network access (astroquery / pyvo are not installed in the
tess_gp environment).  Every query result is cached as CSV under data/crowding_cache/.
"""
from __future__ import annotations

import glob
import hashlib
import io
import os

import numpy as np
import pandas as pd

from . import CACHE, LC_DIR

VIZIER_TAP = 'https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync'
GAIA_TAP = 'https://gea.esac.esa.int/tap-server/tap/sync'


# ---------------------------------------------------------------------------------------
# TAP helper
# ---------------------------------------------------------------------------------------

def tap(query, url=VIZIER_TAP, cache_name=None, refresh=False, timeout=600):
    """Run a synchronous ADQL query and return a DataFrame, cached on disk by query hash."""
    import requests
    key = cache_name or ('tap_' + hashlib.md5((url + query).encode()).hexdigest()[:16])
    path = os.path.join(CACHE, key + '.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    import time
    for attempt in range(5):                       # TAP services return transient 503s
        try:
            r = requests.post(url, data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                                         'QUERY': query}, timeout=timeout)
            if r.status_code in (502, 503, 504) or 'too busy' in r.text[:2000]:
                raise requests.HTTPError(f'{r.status_code} / busy')
            break
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if attempt == 4:
                raise
            time.sleep(5 * 2 ** attempt)
    r.raise_for_status()
    if r.text.lstrip().startswith('<'):
        raise RuntimeError(f'TAP error for query:\n{query}\n{r.text[:1500]}')
    df = pd.read_csv(io.StringIO(r.text))
    df.to_csv(path, index=False)
    return df


def tap_async(query, url=GAIA_TAP, cache_name=None, refresh=False, poll=5, max_wait=1800):
    """Asynchronous TAP job (no sync time limit, no silent truncation), same cache as `tap`.
    `url` is the service's /sync endpoint; the job goes to the matching /async endpoint."""
    import requests
    import time
    key = cache_name or ('tap_' + hashlib.md5((url + query).encode()).hexdigest()[:16])
    path = os.path.join(CACHE, key + '.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    base = url.rsplit('/', 1)[0] + '/async'
    r = requests.post(base, data={'REQUEST': 'doQuery', 'LANG': 'ADQL', 'FORMAT': 'csv',
                                  'PHASE': 'RUN', 'EXECUTIONDURATION': max_wait, 'QUERY': query},
                      allow_redirects=False, timeout=120)
    job = r.headers.get('Location')
    if r.status_code not in (200, 303) or not job:
        raise RuntimeError(f'TAP async submit failed ({r.status_code}):\n{r.text[:1500]}')
    t0 = time.time()
    while True:
        phase = requests.get(job + '/phase', timeout=60).text.strip()
        if phase == 'COMPLETED':
            break
        if phase in ('ERROR', 'ABORTED'):
            raise RuntimeError(f'TAP async job {phase}:\n{requests.get(job, timeout=60).text[:1500]}')
        if time.time() - t0 > max_wait:
            raise TimeoutError(f'TAP async job still {phase} after {max_wait} s: {job}')
        time.sleep(poll)
    res = requests.get(job + '/results/result', timeout=600)
    res.raise_for_status()
    df = pd.read_csv(io.StringIO(res.text))
    df.to_csv(path, index=False)
    return df


def vizier_asu(source, columns, cache_name, refresh=False):
    """Whole VizieR table through the ASU-TSV interface (used when TAP is busy or a table's
    coordinates are sexagesimal; `_RAJ2000/_DEJ2000` are added in decimal degrees)."""
    import requests
    path = os.path.join(CACHE, cache_name + '.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    r = requests.get('https://vizier.cds.unistra.fr/viz-bin/asu-tsv',
                     params={'-source': source, '-out.max': 'unlimited',
                             '-out': ','.join(columns), '-out.add': '_RAJ,_DEJ'}, timeout=600)
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if l and not l.startswith('#')]
    # header, units, dashes, then data
    df = pd.read_csv(io.StringIO('\n'.join([lines[0]] + lines[3:])), sep='\t')
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    df = df.rename(columns={'_RAJ2000': 'ra', '_DEJ2000': 'dec'})
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------------------
# Clusters and their measured aperture flux
# ---------------------------------------------------------------------------------------

def _decode(x):
    return x.decode() if isinstance(x, bytes) else x


def cluster_list(lc_dir=LC_DIR, refresh=False):
    """One row per cluster from the ELK files: position, radius, age, and measured flux.

    F_meas is the median over sectors of each sector's median ELK flux (e/s) -- the
    quantity every normalised light-curve statistic is divided by.  `sector_spread` is the
    16-84% half-range of the per-sector medians over their median: an additive background
    that varies between sectors shows up here, a pure multiplicative effect does not.
    """
    path = os.path.join(CACHE, 'cluster_list.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    from astropy.io import fits
    from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
    import astropy.units as u

    rows = []
    for p in sorted(glob.glob(f'{lc_dir}/*/*.fits')):
        origin = os.path.basename(os.path.dirname(p))
        with fits.open(p) as h:
            hdr = h[0].header
            ra, dec = [float(v) for v in str(hdr['LOCATION']).split(',')]
            meds, mags = [], []
            for ext in h[1:]:
                d = ext.data
                if d is None or len(d) == 0:
                    continue
                meds.append(np.nanmedian(d['flux']))
                mags.append(np.nanmedian(d['mag']))
        meds = np.array(meds)
        F = np.nanmedian(meds) if len(meds) else np.nan
        q16, q84 = np.nanpercentile(meds, [16, 84]) if len(meds) > 1 else (F, F)
        rows.append(dict(name=_decode(hdr['NAME']), origin=origin, ra=ra, dec=dec,
                         r_ap_deg=float(hdr['RADIUS']), age=float(hdr['LOG_AGE']),
                         n_sectors=len(meds), F_meas=F, T_meas=np.nanmedian(mags),
                         sector_spread=0.5 * (q84 - q16) / F if F > 0 else np.nan,
                         file=os.path.relpath(p, lc_dir)))
    df = pd.DataFrame(rows)
    c = SkyCoord(ra=df.ra.values * u.deg, dec=df.dec.values * u.deg)
    df['gal_l'] = c.galactic.l.deg
    df['gal_b'] = c.galactic.b.deg
    df['ecl_lat'] = c.transform_to(BarycentricMeanEcliptic()).lat.deg
    df['r_ap_arcsec'] = df.r_ap_deg * 3600
    df['A_ap_arcsec2'] = np.pi * df.r_ap_arcsec ** 2
    df.to_csv(path, index=False)
    return df


# ---------------------------------------------------------------------------------------
# Literature catalogues
# ---------------------------------------------------------------------------------------

def glatt2010():
    """LMC + SMC (ages < 1 Gyr): integrated APPARENT V of the resolved MCPS stars inside
    R_app, FIELD-SUBTRACTED with an area-scaled 1-2' (or 2-3') annulus (Glatt+2010, Sect. 5).

    R_app is the same radius ELK adopted for the LMC (Glatt is the LMC base catalogue), so
    this is the closest literature analogue of "cluster light inside the ELK aperture".
    MCPS reaches V ~ 23, i.e. well below the turnoff of every cluster here, so the missing
    faint-star light is negligible.  Also provides E(B-V) per cluster.
    """
    df = vizier_asu('J/A+A/517/A50/clusters',
                    ['Name', 'E(B-V)', 'R', 'Age', 'Vmag', 'e_Vmag', 'SimbadName', 'Cross-ID'],
                    cache_name='glatt2010')
    df['galaxy'] = df.Name.str[:3]
    return df


def popescu2012():
    """LMC: integrated ABSOLUTE, dereddened V (Hunter+2003 photometry) + MASSCLEAN mass.

    Table 1 (288 clusters with CMD ages; has Glatt+2010 E(B-V)) and table 2 (632 clusters).
    Popescu+ convert with the LMC distance modulus (m-M)_0 = 18.5.
    """
    t1 = tap('SELECT Name, VMag, e_VMag, "V-R", "E(B-V)", Age, Mass, G10, _RA, _DE '
             'FROM "J/ApJ/751/122/table1"', cache_name='popescu2012_t1')
    t2 = tap('SELECT Name, VMag, e_VMag, "V-R", Age, Mass, _RA, _DE '
             'FROM "J/ApJ/751/122/table2"', cache_name='popescu2012_t2')
    t1['table'] = 1
    t2['table'] = 2
    df = pd.concat([t1, t2], ignore_index=True).rename(columns={'_RA': 'ra', '_DE': 'dec'})
    return df


def rafelski2005():
    """SMC: integrated APPARENT, observed V and V-I (Rafelski & Zaritsky 2005, MCPS)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    df = tap('SELECT Seq, RAJ2000, DEJ2000, Vmag, e_Vmag, "V-I", "e_V-I", Name '
             'FROM "J/AJ/129/2701/table1"', cache_name='rafelski2005_t1')
    # TAP returns decimal degrees already for this table; keep a guard for sexagesimal.
    if df.RAJ2000.dtype == object:
        c = SkyCoord(df.RAJ2000, df.DEJ2000, unit=(u.hourangle, u.deg))
        df['ra'], df['dec'] = c.ra.deg, c.dec.deg
    else:
        df['ra'], df['dec'] = df.RAJ2000, df.DEJ2000
    df = df[(df.Vmag < 90) & df['V-I'].notna()]
    return df


def kharchenko2016():
    """MW: integrated ABSOLUTE J/H/Ks of MWSC clusters + apparent distance moduli.

    DM(J) includes extinction (it is the apparent modulus in J), so the observed integrated
    magnitude is simply J_obs = I(M.J) + DM(J).
    """
    return tap('SELECT MWSC, Name, "E(B-V)", d, logt, "DM(J)", "I(M.J)", "DM(Ks)", "I(M.Ks)", '
               '"_RA.icrs" AS ra, "_DE.icrs" AS dec FROM "J/A+A/585/A101/catalog"',
               cache_name='kharchenko2016')


def hunt_reffert2024():
    """MW: Gaia-based cluster parameters incl. total mass, distance modulus and A_V."""
    return tap('SELECT Name, AllNames, RA_ICRS AS ra, DE_ICRS AS dec, r50, rt, MassTot, '
               'e_MassTot, MOD50, AV50, logAge50 FROM "J/A+A/686/A42/clusters"',
               cache_name='hunt_reffert2024')


def _norm(name):
    return ''.join(ch for ch in str(name).upper() if ch.isalnum())


def crossmatch(clusters, cat, max_sep_arcsec, frac_of_radius=0.5, name_col=None):
    """Nearest catalogue entry within max(max_sep, frac * aperture radius).

    Returns a DataFrame indexed like `clusters` with the matched catalogue columns (prefixed
    by nothing; caller renames) plus `sep_arcsec`.  When `name_col` is given, an exact
    normalised-name match takes precedence over the positional one.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c1 = SkyCoord(clusters.ra.values * u.deg, clusters.dec.values * u.deg)
    c2 = SkyCoord(cat.ra.values * u.deg, cat.dec.values * u.deg)
    idx, sep, _ = c1.match_to_catalog_sky(c2)
    sep = sep.arcsec
    lim = np.maximum(max_sep_arcsec, frac_of_radius * clusters.r_ap_arcsec.values)
    ok = sep <= lim
    if name_col is not None:
        import re
        lookup = {}
        cols = [name_col] if isinstance(name_col, str) else list(name_col)
        for i in range(len(cat)):
            for col in cols:
                for part in re.split(r'[,.;]', str(cat[col].values[i])):
                    if _norm(part):
                        lookup.setdefault(_norm(part), i)
        # A name match is trusted only within 2 aperture radii: short designations
        # (e.g. Lindsay / Kron numbers) collide across catalogues.
        rmax = np.maximum(max_sep_arcsec, 2 * clusters.r_ap_arcsec.values)
        for k, n in enumerate(clusters.name.values):
            j = lookup.get(_norm(n))
            if j is not None:
                s = c1[k].separation(c2[j]).arcsec
                if s <= rmax[k]:
                    idx[k], sep[k], ok[k] = j, s, True
    out = cat.iloc[idx].reset_index(drop=True).copy()
    out['sep_arcsec'] = sep
    out[~ok] = np.nan
    out.index = clusters.index
    return out

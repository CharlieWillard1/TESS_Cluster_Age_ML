"""PARSEC single-burst stellar population (SSP) magnitudes per solar mass.

Queried once from the CMD 3.9 web form (output_kind=4: "integrated magnitudes, for 1 Msun")
and cached.  Two photometric systems are needed because the literature cluster photometry is
in V/R/I (Magellanic Clouds) or J (Milky Way), while the light curves are in the TESS band.

The magnitudes are ABSOLUTE magnitudes of the stars still alive per 1 Msun of INITIAL mass,
with no extinction.  The IMF is Chabrier (2001) lognormal; CMD offers no Kroupa option, and
the two differ by < ~0.1 dex in M/L, far below the precision this analysis needs.
"""
from __future__ import annotations

import gzip
import io
import os
import re

import numpy as np
import pandas as pd

from . import CACHE

CMD_URL = 'https://stev.oapd.inaf.it/cgi-bin/cmd'

# Metallicity adopted per galaxy (mass fraction Z).  MW: PARSEC solar; LMC ~ -0.3 dex;
# SMC ~ -0.6 dex.  Clusters are not individually assigned metallicities.
Z_GALAXY = {'MW': 0.0152, 'LMC': 0.008, 'SMC': 0.004}

PHOTSYS = {
    'tess': 'YBC_tab_mag_odfnew/tab_mag_TESS_2mass.dat',     # TESS, J, H, Ks
    'bessell': 'YBC_tab_mag_odfnew/tab_mag_bessell.dat',     # U B V R I J H K
}


def _query_cmd(photsys_file, Z, lage_lo=6.0, lage_hi=10.15, dlage=0.05):
    import requests
    import urllib3
    urllib3.disable_warnings()
    form = {
        'cmd_version': '3.9', 'track_parsec': 'parsec_CAF09_v1.2S',
        'track_colibri': 'parsec_CAF09_v1.2S_S_LMC_08_web', 'track_postagb': 'no',
        'n_inTPC': '10', 'eta_reimers': '0.2', 'kind_interp': '1', 'kind_postagb': '-1',
        'photsys_file': photsys_file, 'photsys_version': 'YBCnewVega',
        'dust_sourceM': 'dpmod60alox40', 'dust_sourceC': 'AMCSIC15', 'kind_mag': '2',
        'kind_dust': '0', 'extinction_av': '0.0', 'extinction_coeff': 'constant',
        'extinction_curve': 'cardelli', 'kind_LPV': '4',
        'imf_file': 'tab_imf/imf_chabrier_lognormal.dat',
        'isoc_isagelog': '1', 'isoc_lagelow': str(lage_lo), 'isoc_lageupp': str(lage_hi),
        'isoc_dlage': str(dlage), 'isoc_ismetlog': '0', 'isoc_zlow': str(Z),
        'isoc_zupp': str(Z), 'isoc_dz': '0.0', 'output_kind': '4', 'output_evstage': '1',
        'lf_maginf': '-15', 'lf_magsup': '20', 'lf_deltamag': '0.5', 'sim_mtot': '1.0e4',
        'submit_form': 'Submit',
    }
    # The CMD host serves an incomplete certificate chain (curl exit 60), so verification is
    # disabled for this one public, read-only download.
    r = requests.post(CMD_URL, data=form, verify=False, timeout=300)
    links = re.findall(r'href="?([^"> ]*output\d+\.dat)', r.text)
    if not links:
        raise RuntimeError('CMD returned no output file:\n' + re.sub('<[^>]+>', '', r.text)[-800:])
    url = 'https://stev.oapd.inaf.it/' + links[0].replace('../', '')
    raw = requests.get(url, verify=False, timeout=300).content
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    text = raw.decode()
    header = [l for l in text.splitlines() if l.startswith('# age')][0]
    cols = header.lstrip('# ').split()
    body = '\n'.join(l for l in text.splitlines() if l.strip() and not l.startswith('#'))
    df = pd.read_csv(io.StringIO(body), sep=r'\s+', names=cols)
    df = df.replace(999.999, np.nan)
    df['logage'] = np.round(np.log10(df['age']), 3)
    return df


def load_ssp(refresh=False):
    """SSP table: one row per (logage, Z) with absolute mags per 1 Msun in T,J,H,Ks,U,B,V,R,I."""
    path = os.path.join(CACHE, 'parsec_ssp_per_msun.csv')
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path)
    parts = []
    for Z in sorted(set(Z_GALAXY.values())):
        t = _query_cmd(PHOTSYS['tess'], Z)[['logage', 'Z', 'TESSmag', 'Jmag', 'Hmag', 'Ksmag']]
        b = _query_cmd(PHOTSYS['bessell'], Z)
        b = b[['logage', 'UXmag', 'Bmag', 'Vmag', 'Rmag', 'Imag']].rename(columns={'UXmag': 'Umag'})
        parts.append(t.merge(b, on='logage', how='inner'))
    ssp = pd.concat(parts, ignore_index=True).rename(columns={'TESSmag': 'Tmag'})
    ssp.to_csv(path, index=False)
    return ssp


class SSP:
    """Interpolate SSP magnitudes in log age, per galaxy metallicity.

    Grid points where CMD failed (999.999 -> NaN, a few very young ages) are skipped, so the
    interpolation bridges them.  Ages outside the grid are clipped to its ends.
    """

    def __init__(self, table=None):
        self.t = load_ssp() if table is None else table

    def mag(self, band, logage, origin):
        Z = Z_GALAXY[origin]
        g = self.t[np.isclose(self.t.Z, Z)].sort_values('logage')
        ok = g[f'{band}mag'].notna()
        x, y = g.logage[ok].values, g[f'{band}mag'][ok].values
        return np.interp(np.clip(logage, x.min(), x.max()), x, y)

    def color(self, b1, b2, logage, origin):
        return self.mag(b1, logage, origin) - self.mag(b2, logage, origin)

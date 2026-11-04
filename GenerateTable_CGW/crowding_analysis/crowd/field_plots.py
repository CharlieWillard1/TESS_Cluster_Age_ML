"""Diagnostic plots of the Gaia light around one ELK aperture.

* `plot_radial_profile`  TESS-band surface brightness of all Gaia sources in rings of
  0.25 r_ap, against the field level measured in the background annulus.  The shaded band is
  the stochastic (compound-Poisson) scatter of the field expected in a patch the size of each
  ring, sqrt(<f^2>_ann / A_ring), so a ring outside the band is unlikely to be pure field.
* `plot_star_map`  synthetic image of the Gaia sources (flux convolved with a point-like PSF),
  linear or asinh stretch.  For MW clusters, HR24 members can be coloured green.
* `plot_cluster`  profile + linear (top) and asinh (bottom) images, plus a members column
  for the MW.

All fluxes are ELK e/s (Stassun+2019 T(G, BP-RP), `T_ZP`), as in `gaia_share`.  The
profile is the raw (not PSF-weighted) Gaia light, so for LMC/SMC clusters its in-aperture
sum differs slightly from the PSF-weighted f_gaia in `gaia_cluster_share.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import T_ZP
from .catalogs import tap, cluster_list
from .gaia_field import stassun_T, _key
from .gaia_share import ring_mag_table, ANNULI, RING

MEMBER_GREEN = (0.25, 0.90, 0.35)


# ---------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------

def cluster_row(name, clusters=None):
    cl = clusters if clusters is not None else cluster_list()
    m = cl.name.str.upper().str.strip() == name.upper().strip()
    if not m.any():
        raise KeyError(f'{name} not in the ELK cluster list')
    return cl[m].iloc[0]


G_SLICES = [(1, 14), (14, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 99)]


def field_rows(name, ra, dec, r_ap_deg, rmax, refresh=False, top=1_000_000, origin='MW'):
    """Every Gaia source (with G) within rmax * r_ap, with T-band flux and tangent-plane offsets
    (arcmin, xi positive to the east).

    Rows come from VizieR's copy of Gaia DR3 (the ESA archive truncated or timed out on dense
    fields), split into G slices, and each slice's per-ring counts are checked against the
    server-side aggregate (`ring_mag_table`, which uses the same G bins).  A mismatch triggers
    one fresh query of that slice, then an error.
    """
    tab = ring_mag_table(name, origin, ra, dec, r_ap_deg)
    nr = int(np.floor(min(rmax, tab.r_out.max()) / RING + 1e-9))
    redges = np.arange(nr + 1) * RING
    out = []
    for g0, g1 in G_SLICES:
        t = tab[(tab.gbin >= g0) & (tab.gbin < g1)]
        want = t.groupby('ibin').n.sum().reindex(range(nr), fill_value=0).values
        for attempt in range(2):
            df = _field_rows(name, ra, dec, r_ap_deg, rmax, g0, g1, refresh or attempt > 0, top)
            got = np.histogram(df.r_rel.values, redges)[0]
            if np.abs(got - want).sum() <= max(5, 1e-3 * want.sum()):
                break
            print(f'{name} G[{g0},{g1}): Gaia rows incomplete ({got.sum()} vs {want.sum()})'
                  + ('; re-querying' if attempt == 0 else ''))
        else:
            raise RuntimeError(f'{name}: Gaia rows for G[{g0},{g1}) still incomplete')
        out.append(df)
    return _add_flux_offsets(pd.concat(out, ignore_index=True), ra, dec)


def _field_rows(name, ra, dec, r_ap_deg, rmax, g0, g1, refresh, top):
    # Gaia DR3 rows from VizieR (I/355/gaiadr3): the ESA archive's sync service truncates and
    # its anonymous async jobs are capped at 60 s, too short for dense MW fields.
    q = (f'SELECT TOP {top} Source AS source_id, RA_ICRS AS ra, DE_ICRS AS dec, Gmag AS g, '
         f'"BP-RP" AS bp_rp FROM "I/355/gaiadr3" '
         f"WHERE 1 = CONTAINS(POINT('ICRS', RA_ICRS, DE_ICRS), "
         f"CIRCLE('ICRS', {ra}, {dec}, {rmax * r_ap_deg})) "
         f'AND Gmag >= {g0} AND Gmag < {g1}')
    df = tap(q, cache_name=f'fieldmap_{_key(name)}_r{rmax:g}_G{g0}-{g1}', refresh=refresh)
    df.columns = [c.lower() for c in df.columns]
    if len(df) >= top:
        raise RuntimeError(f'{name}: Gaia query hit TOP {top}; widen the limit or shrink rmax')
    d2r = np.pi / 180
    c = (np.sin(dec * d2r) * np.sin(df.dec.values * d2r)
         + np.cos(dec * d2r) * np.cos(df.dec.values * d2r) * np.cos((df.ra.values - ra) * d2r))
    df['r_rel'] = np.degrees(np.arccos(np.clip(c, -1, 1))) / r_ap_deg
    return df


def _add_flux_offsets(df, ra, dec):
    df['T'] = stassun_T(df.g.values, df.bp_rp.values)
    df['f'] = 10 ** (-0.4 * (df['T'] - T_ZP))
    d2r = np.pi / 180
    a, d, a0, d0 = df.ra.values * d2r, df.dec.values * d2r, ra * d2r, dec * d2r
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    df['xi'] = np.degrees(np.cos(d) * np.sin(a - a0) / cosc) * 60
    df['eta'] = np.degrees((np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc) * 60
    return df


def hr24_member_ids(r):
    """Gaia DR3 source_ids of the HR24 members of a MW cluster (empty if HR24 has no match)."""
    from .mw_members import match_hr24, members
    m = match_hr24(pd.DataFrame([r]))
    hr = m.hr_name.iloc[0]
    if not isinstance(hr, str):
        return hr, np.array([], dtype=np.int64)
    return hr, members(hr).GaiaDR3.values.astype(np.int64)


def ring_profile(tab, r_ap_arcsec, annulus):
    """Per-ring surface brightness (e/s/arcmin^2) and the annulus field level with its
    per-ring stochastic scatter.  Also returns the aperture-level f_gaia and its sigma."""
    nb = int(tab.ibin.max()) + 1
    g = tab.groupby('ibin')[['f', 'f2', 'n']].sum().reindex(range(nb), fill_value=0.0)
    return _profile(np.arange(nb + 1) * RING, g.f.values, g.f2.values, g.n.values, r_ap_arcsec, annulus)


def profile_from_rows(rows, r_ap_arcsec, annulus, rmax):
    """Same as `ring_profile`, from row-level sources (e.g. one flux bin) out to rmax r_ap."""
    edges = np.arange(int(np.floor(rmax / RING + 1e-9)) + 1) * RING
    rr, f = rows.r_rel.values, rows.f.values
    F = np.histogram(rr, edges, weights=f)[0]
    F2 = np.histogram(rr, edges, weights=f ** 2)[0]
    n = np.histogram(rr, edges)[0]
    return _profile(edges, F, F2, n, r_ap_arcsec, annulus)


def _profile(edges, F, F2, n, r_ap_arcsec, annulus):
    area = np.pi * (r_ap_arcsec / 60) ** 2 * (edges[1:] ** 2 - edges[:-1] ** 2)   # arcmin^2
    lo, hi = annulus
    ann = (edges[:-1] >= lo - 1e-9) & (edges[1:] <= hi + 1e-9)
    A_ann = area[ann].sum()
    mu = F[ann].sum() / A_ann                           # field SB
    mu2 = F2[ann].sum() / A_ann                         # sum f^2 per unit area
    A_ap = np.pi * (r_ap_arcsec / 60) ** 2
    F_in = F[edges[1:] <= 1 + 1e-9].sum()
    ok = F_in > 0
    return dict(edges=edges, sb=F / area, n=n, area=area,
                mu=mu, sig_ring=np.sqrt(mu2 / area), F_in=F_in, F_field=mu * A_ap,
                f_gaia=(F_in - mu * A_ap) / F_in if ok else np.nan,
                sig_f=np.sqrt(mu2 * A_ap) / F_in if ok else np.nan)


# ---------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------

def plot_radial_profile(r, tab=None, annulus=None, ax=None, rmax=None, p=None, legend=True,
                        title=None):
    """Profile from the ring x G table (default), or a precomputed profile `p`."""
    import matplotlib.pyplot as plt
    annulus = annulus or ANNULI[r.origin][0]
    if p is None:
        tab = tab if tab is not None else ring_mag_table(r['name'], r.origin, r.ra, r.dec, r.r_ap_deg)
        p = ring_profile(tab, r.r_ap_arcsec, annulus)
    ax = ax or plt.subplots(figsize=(6, 4.5))[1]
    e = p['edges']
    if rmax is not None:
        keep = e[1:] <= rmax + 1e-9
        e, sb, sg = e[:keep.sum() + 1], p['sb'][keep], p['sig_ring'][keep]
    else:
        sb, sg = p['sb'], p['sig_ring']
    rap_min = r.r_ap_arcsec / 60
    lo, hi = annulus

    ax.axvspan(lo, hi, color='tab:orange', alpha=0.12, lw=0,
               label=f'background annulus {lo:g}–{hi:g} r_ap ({lo * rap_min:.1f}′–{hi * rap_min:.1f}′)')
    ax.fill_between(e, np.r_[p['mu'] - sg, p['mu'] - sg[-1]], np.r_[p['mu'] + sg, p['mu'] + sg[-1]],
                    step='post', color='0.5', alpha=0.3, lw=0, label='field ±1σ (stochastic, per ring)')
    ax.axhline(p['mu'], color='0.25', lw=1.5, label=f'field level μ = {p["mu"]:.3g} e/s/arcmin²')
    ax.stairs(sb, e, color='tab:blue', lw=2, label='Gaia surface brightness')
    ax.axvline(1, color='tab:red', lw=1.5, ls='--', label=f'ELK r_ap = {rap_min:.2f}′ ({r.r_ap_deg:.3f}°)')
    ax.set_xlim(0, e[-1])
    ax.set_xlabel('r / r_ap')
    ax.set_ylabel('TESS-band surface brightness [e/s/arcmin²]')
    sec = ax.secondary_xaxis('top', functions=(lambda x: x * rap_min, lambda x: x / rap_min))
    sec.set_xlabel('r [arcmin]')
    ax.text(0.98, 0.97, f'F_in = {p["F_in"]:.3g} e/s\nμ·A_ap = {p["F_field"]:.3g} e/s\n'
                        f'f_cl,Gaia = {p["f_gaia"]:.3f} ± {p["sig_f"]:.3f}'
                        + ('' if r.origin == 'MW' else '\n(unweighted by PSF)'),
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', ec='0.7', alpha=0.9))
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.5, frameon=False)
    if title:
        ax.set_title(title)
    return ax, p


def render(rows, ext, npix=500, fwhm_px=2.5, member_ids=None):
    """Synthetic sky image: each source's flux deposited on an npix x npix grid spanning
    +-ext arcmin, then convolved with a Gaussian PSF of FWHM fwhm_px image pixels.

    Gaia's real PSF (~0.1-0.2 arcsec) is far below one image pixel at these scales, so the
    default is the narrowest PSF that keeps stars visible: point sources whose apparent size
    grows with flux only through the stretch, as in a real image.  Set fwhm_px larger (e.g.
    the TESS PSF, ~40 arcsec) to see the field as TESS does.  Returns (all, members) images in
    e/s per image pixel; `members` is None without member_ids.
    """
    from scipy.ndimage import gaussian_filter
    edges = np.linspace(-ext, ext, npix + 1)
    sig = fwhm_px / 2.3548

    def img(m):
        h = np.histogram2d(rows.eta.values[m], rows.xi.values[m], [edges, edges],
                           weights=rows.f.values[m])[0]
        return gaussian_filter(h, sig, mode='constant')

    ok = np.ones(len(rows), bool)
    full = img(ok)
    mem = img(np.isin(rows.source_id.values, member_ids)) if member_ids is not None else None
    return full, mem


def psf_peak(fwhm_px):
    """Peak pixel of a unit-flux point source centred on a pixel, after `render`'s PSF."""
    from scipy.ndimage import gaussian_filter
    d = np.zeros((41, 41))
    d[20, 20] = 1.0
    return gaussian_filter(d, fwhm_px / 2.3548, mode='constant').max()


def flux_bins(f):
    """Integer log10(flux) edges from floor(min) up to floor(max) + 1."""
    lf = np.log10(f[f > 0])
    return np.arange(np.floor(lf.min()), np.floor(lf.max()) + 2)


def stretch(x, vmax, kind='linear', soft=3e-4):
    """Linear, or asinh with softening soft*vmax (linear below it, logarithmic above)."""
    if kind == 'linear':
        return np.clip(x / vmax, 0, 1)
    a = soft * vmax
    return np.clip(np.arcsinh(np.clip(x, 0, None) / a) / np.arcsinh(vmax / a), 0, 1)


def plot_star_map(r, rows, annulus=None, ax=None, member_ids=None, kind='linear', vmax=None,
                  soft=3e-4, npix=500, fwhm_px=2.5, title=None, vmax_flux=None):
    """Synthetic image of the Gaia field (see `render`), displayed with a linear or asinh
    stretch saturating at vmax (default: the brightest pixel inside r_ap).  With member_ids,
    each pixel is coloured by its member fraction: white = field, green = HR24 members."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    annulus = annulus or ANNULI[r.origin][0]
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    rap = r.r_ap_arcsec / 60
    lo, hi = annulus
    ext = 1.05 * hi * rap
    full, mem = render(rows, ext, npix, fwhm_px, member_ids)
    c = (np.arange(npix) + 0.5) / npix * 2 * ext - ext
    inap = np.hypot(*np.meshgrid(c, c)) <= rap
    if vmax_flux:                                    # saturate at the peak of a star of this flux
        vmax = vmax_flux * psf_peak(fwhm_px)
    vmax = vmax or full[inap].max() or 1.0
    L = stretch(full, vmax, kind, soft)
    rgb = np.repeat(L[..., None], 3, axis=2)
    if mem is not None:
        fm = np.clip(np.divide(mem, full, out=np.zeros_like(full), where=full > 0), 0, 1)[..., None]
        rgb = L[..., None] * ((1 - fm) * np.ones(3) + fm * np.array(MEMBER_GREEN))
    ax.imshow(rgb, origin='lower', extent=[-ext, ext, -ext, ext], interpolation='antialiased')

    ax.add_patch(Circle((0, 0), rap, fill=False, ec='tab:red', lw=1.5, ls='--'))
    for rr in (lo, hi):
        ax.add_patch(Circle((0, 0), rr * rap, fill=False, ec='tab:orange', lw=1.2, ls=':'))
    ax.text(0, -rap * 1.02, f'ELK r_ap = {rap:.2f}′', color='tab:red', ha='center', va='top', fontsize=8)
    ax.text(0, hi * rap * 1.01, f'annulus {lo:g}–{hi:g} r_ap ({lo * rap:.1f}′–{hi * rap:.1f}′)',
            color='tab:orange', ha='center', va='bottom', fontsize=8)
    ax.set_xlim(ext, -ext)                           # east to the left
    ax.set_ylim(-ext, ext)
    ax.set_xlabel('Δα cos δ [arcmin]  (E ←)')
    ax.set_ylabel('Δδ [arcmin]')
    note = (f'{kind} stretch' + (f' (soft {soft:g}·max)' if kind != 'linear' else '')
            + (f', saturates at f = {vmax_flux:.3g} e/s' if vmax_flux else '')
            + f'; PSF FWHM {fwhm_px:g} px = {fwhm_px * 2 * ext * 60 / npix:.1f}″')
    if member_ids is not None:
        d = rows[rows.r_rel <= 1]
        is_mem = np.isin(d.source_id.values, member_ids)
        note += (f'\nmembers in r_ap: {int(is_mem.sum())}, '
                 f'{d.f.values[is_mem].sum() / d.f.sum():.0%} of Gaia flux')
    ax.text(0.02, 0.02, note, transform=ax.transAxes, color='white', fontsize=7.5, va='bottom')
    if title:
        ax.set_title(title)
    return ax


def plot_cluster(name, clusters=None, annulus=None, soft=3e-4, npix=500, fwhm_px=2.5,
                 flux_rows=True, share_y=True, save=None):
    """Row 1: radial profile + synthetic Gaia images (linear); row 2: asinh images.  With
    flux_rows, one more row per integer log10(flux) bin (brightest first): that bin's profile
    against its own annulus field, and its images on a linear stretch spanning one decade
    (saturating at the bin's upper edge).  MW clusters get an extra column with HR24 members
    in green.  share_y gives every profile panel the same y limits."""
    import matplotlib.pyplot as plt
    r = cluster_row(name, clusters)
    annulus = annulus or ANNULI[r.origin][0]
    rmax = 1.05 * np.sqrt(2) * annulus[1]
    rows = field_rows(r['name'], r.ra, r.dec, r.r_ap_deg, rmax=rmax, origin=r.origin)
    mw = r.origin == 'MW'
    hr, ids = hr24_member_ids(r) if mw else (None, None)
    n = 3 if mw else 2
    edges = flux_bins(rows.f.values)[::-1] if flux_rows else []
    nrow = 2 + max(len(edges) - 1, 0)
    fig = plt.figure(figsize=(6.2 * n, 6 * nrow))
    gs = fig.add_gridspec(nrow, n, width_ratios=[1.25] + [1] * (n - 1))
    prof_axes = [plot_radial_profile(r, annulus=annulus, ax=fig.add_subplot(gs[0, 0]),
                                     title='All Gaia sources')[0]]
    kw = dict(soft=soft, npix=npix, fwhm_px=fwhm_px)
    for i, kind in enumerate(['linear', 'asinh']):
        plot_star_map(r, rows, annulus, fig.add_subplot(gs[i, 1]), kind=kind,
                      title=f'All Gaia sources ({kind})', **kw)
        if mw:
            t = f'HR24 members ({hr}) in green ({kind})' if len(ids) else 'no HR24 match'
            plot_star_map(r, rows, annulus, fig.add_subplot(gs[i, 2]), member_ids=ids, kind=kind,
                          title=t, **kw)
    for j, (hi, lo) in enumerate(zip(edges[:-1], edges[1:])):
        sub = rows[(rows.f >= 10 ** lo) & (rows.f < 10 ** hi)]
        tlo, thi = T_ZP - 2.5 * hi, T_ZP - 2.5 * lo
        lab = f'{10 ** lo:g} ≤ f < {10 ** hi:g} e/s  (T {tlo:.1f}–{thi:.1f}, N = {len(sub)})'
        p = profile_from_rows(sub, r.r_ap_arcsec, annulus, rmax)
        prof_axes.append(plot_radial_profile(r, annulus=annulus, ax=fig.add_subplot(gs[2 + j, 0]),
                                             p=p, legend=False, title=lab)[0])
        plot_star_map(r, sub, annulus, fig.add_subplot(gs[2 + j, 1]), kind='linear',
                      vmax_flux=10 ** hi, title=f'log f ∈ [{lo:g}, {hi:g})', **kw)
        if mw:
            plot_star_map(r, sub, annulus, fig.add_subplot(gs[2 + j, 2]), member_ids=ids,
                          kind='linear', vmax_flux=10 ** hi, title=f'log f ∈ [{lo:g}, {hi:g}), members green', **kw)
    if share_y:
        lims = np.array([a.get_ylim() for a in prof_axes])
        for a in prof_axes:
            a.set_ylim(lims[:, 0].min(), lims[:, 1].max())
    fig.suptitle(f'{r["name"]} ({r.origin}, log age {r.age:g})', fontsize=13, y=1.0)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110, bbox_inches='tight')
    return fig

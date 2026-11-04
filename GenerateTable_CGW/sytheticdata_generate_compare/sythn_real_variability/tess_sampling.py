"""Turn a noise-free cluster signal into a TESS-sampled, noisy light-curve FITS file.

This is the "existing synthetic pipeline" step.  Everything that touches the real data
layout is imported from ``sytheticdata_generate_compare.synth.generate``:

* ``exposure_average`` / ``sector_exposure_days`` -- box-exposure sampling at the real
  timestamps (used here only for the red-noise component; the sinusoid part is averaged
  analytically by ``CleanSignal``);
* ``red_noise_series`` / ``_calibrate_red_scale`` -- optional calibrated red noise with
  ``(alpha, log10_N)`` resampled jointly from real pipeline rows;
* the copy-the-real-FITS-and-replace-``flux`` convention, so the output tree is read by
  ``table_pipeline`` unchanged.

A *template* real cluster supplies timestamps, sector structure, per-point ``flux_err`` and
(in ``brightness_mode='template'``) the flux level.  Templates are chosen among real
clusters of ``template_origin`` by brightness, so the noise level is the one a real TESS
cluster of that brightness actually has.
"""

from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd
from astropy.io import fits

from table_pipeline.lc_lsp import _slug
from sytheticdata_generate_compare.synth.generate import (exposure_average, sector_exposure_days,
                                                          red_noise_series, _calibrate_red_scale)
from .config import TESS_ZP, NoiseConfig
from .populations import tess_mag_from_flux
from .variability import CleanSignal


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def list_templates(real_lc_dir, origin='LMC', cache_csv=None):
    """One row per real cluster FITS of ``origin``: name, path, median flux, sectors, baseline."""
    if cache_csv and os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    rows = []
    for f in sorted(glob.glob(os.path.join(real_lc_dir, origin, 'hlsp_elk_tess_ffi_*_tess_v1_llc.fits'))):
        with fits.open(f) as h:
            meds, errs, n = [], [], 0
            tmin, tmax = np.inf, -np.inf
            for hdu in h[1:]:
                if hdu.data is None:
                    continue
                fl = np.asarray(hdu.data['flux'], float); ee = np.asarray(hdu.data['flux_err'], float)
                tt = np.asarray(hdu.data['time'], float)
                good = np.isfinite(fl) & np.isfinite(ee) & np.isfinite(tt)
                if good.sum() < 10:
                    continue
                med = np.nanmedian(fl[good])
                meds.append(med); errs.append(np.nanmedian(ee[good]) / med); n += 1
                tmin = min(tmin, tt[good].min()); tmax = max(tmax, tt[good].max())
            if n == 0:
                continue
            rows.append(dict(name=h[0].header.get('NAME', os.path.basename(f)), path=f,
                             log_age=h[0].header.get('LOG_AGE', np.nan), n_sectors=n,
                             median_flux=float(np.median(meds)), err_frac=float(np.median(errs)),
                             baseline_days=float(tmax - tmin)))
    df = pd.DataFrame(rows)
    df['Tmag'] = tess_mag_from_flux(df['median_flux'])
    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


def select_template(F_syn_es, templates: pd.DataFrame, rng, ncfg: NoiseConfig):
    """Pick a template row: among the ``k`` nearest in log(median flux), or any at random."""
    if ncfg.template_select == 'random':
        return templates.iloc[int(rng.integers(0, len(templates)))]
    d = np.abs(np.log10(templates['median_flux'].to_numpy()) - np.log10(F_syn_es))
    k = max(1, min(int(ncfg.template_k_nearest), len(templates)))
    cand = np.argsort(d)[:k]
    return templates.iloc[int(rng.choice(cand))]


def load_rn_pairs(rn_table_path, origin=None):
    """Real ``(rn_alpha, rn_log10_N)`` pairs to resample jointly (optionally one origin)."""
    t = pd.read_pickle(rn_table_path)
    if origin and 'origin' in t:
        t = t[t['origin'] == origin]
    v = t[['rn_alpha', 'rn_log10_N']].astype(float).dropna()
    return v.to_numpy()


# ---------------------------------------------------------------------------
# Writing one cluster
# ---------------------------------------------------------------------------

def output_path(out_lc_dir, origin, name):
    return os.path.join(out_lc_dir, origin, f'hlsp_elk_tess_ffi_{name}_tess_v1_llc.fits')


def write_sampled_fits(template_path, out_path, name, log_age, signal: CleanSignal, F_syn_es,
                       ncfg: NoiseConfig, rng, rn_pair=None, header_extra=None):
    """Sample ``signal`` at the template's timestamps, add noise, write the FITS.

    Returns ``(record, truth_t, truth_x)`` where ``truth_*`` is the clean point-sampled
    signal on a ``truth_cadence_min`` grid covering each sector's span (for ground-truth
    statistics, same convention as ``synth.ground_truth``).
    """
    with fits.open(template_path) as hdul:
        sectors = []
        for i in range(1, len(hdul)):
            d = hdul[i].data
            if d is None:
                continue
            sectors.append((i, np.asarray(d['time'], float), np.asarray(d['flux_err'], float),
                            np.asarray(d['flux'], float)))
        if not sectors:
            raise ValueError(f'template {template_path} has no sector tables')

        finite = np.concatenate([t[np.isfinite(t)] for _, t, _, _ in sectors])
        t0, t1 = float(finite.min()), float(finite.max())

        # Template brightness (median over sectors of the per-sector median flux)
        sec_med = []
        for _, tt, ee, ff in sectors:
            good = np.isfinite(ff) & np.isfinite(tt)
            sec_med.append(np.nanmedian(ff[good]) if good.any() else np.nan)
        F_tmpl = float(np.nanmedian(sec_med))
        flux_ratio = float(F_syn_es / F_tmpl)

        # Optional red noise, calibrated on the longest sector exactly as synth.generate does.
        rn_rms = 0.0; red_grid = None; t_grid = None
        if ncfg.add_red_noise and rn_pair is not None:
            alpha, log10_N = float(rn_pair[0]), float(rn_pair[1])
            dt = ncfg.grid_cadence_min / 60.0 / 24.0
            t_grid = np.arange(t0, t1 + dt, dt)
            red_grid = red_noise_series(t_grid, alpha, np.random.default_rng(rng.integers(0, 2**32 - 1)))
            j = int(np.argmax([np.isfinite(t).sum() for _, t, _, _ in sectors]))
            _, t_ref, e_ref, f_ref = sectors[j]
            m = np.isfinite(t_ref) & np.isfinite(e_ref) & (e_ref > 0) & np.isfinite(f_ref)
            med_f = float(np.nanmedian(f_ref[m]))
            rn_rms_requested = _calibrate_red_scale(t_grid, red_grid, t_ref[m], e_ref[m] / med_f, log10_N,
                                                    ncfg.lsp_P_min, ncfg.lsp_P_max, ncfg.lsp_alpha,
                                                    sector_exposure_days(t_ref))
            rn_rms = float(min(rn_rms_requested, ncfg.rn_rms_max))
        else:
            alpha, log10_N = np.nan, np.nan
            rn_rms_requested = 0.0

        out = fits.HDUList([hdul[0].copy()])
        hdr = out[0].header
        hdr['NAME'] = name
        hdr['LOG_AGE'] = float(log_age)
        hdr['SYNTH'] = (True, 'synthetic population light curve (sythn_real_variability)')
        hdr['SYN_TMPL'] = (str(hdul[0].header.get('NAME', ''))[:60], 'real template cluster')
        hdr['SYN_FSYN'] = (float(F_syn_es), 'population TESS flux at the model distance [e/s]')
        hdr['SYN_FTPL'] = (F_tmpl, 'template median flux [e/s]')
        hdr['SYN_BMOD'] = (ncfg.brightness_mode, 'template | physical flux scaling')
        hdr['SYN_RNA'] = (alpha if np.isfinite(alpha) else -99.0, 'injected red-noise alpha')
        hdr['SYN_RNN'] = (log10_N if np.isfinite(log10_N) else -99.0, 'injected red-noise log10_N target')
        hdr['SYN_RNS'] = (float(rn_rms), 'red-noise fractional rms (full baseline)')
        for k, v in (header_extra or {}).items():
            hdr[k] = v

        truth_t, truth_x = [], []
        sector_records = []
        for i, tt, ee, ff in sectors:
            good = np.isfinite(tt)
            E = sector_exposure_days(tt)
            x = signal(np.where(good, tt, t0), E)                       # analytic exposure average
            if red_grid is not None:
                x = x + rn_rms * exposure_average(t_grid, red_grid, np.where(good, tt, t0), E)
            med = np.nanmedian(ff[good & np.isfinite(ff)])
            if not np.isfinite(med) or med == 0:
                med = F_tmpl
            if ncfg.brightness_mode == 'template':
                F0 = med; err = ee
            elif ncfg.brightness_mode == 'physical':
                F0 = F_syn_es; err = ee * np.sqrt(flux_ratio)
            else:
                raise ValueError(f'unknown brightness_mode {ncfg.brightness_mode!r}')
            flux_new = x * F0 + np.where(np.isfinite(err), rng.normal(0.0, np.abs(np.nan_to_num(err))), 0.0)
            flux_new[~good] = np.nan
            err_new = np.where(good, err, np.nan)
            with np.errstate(invalid='ignore', divide='ignore'):
                mag = TESS_ZP - 2.5 * np.log10(flux_new)
                mag_err = 1.0857 * err_new / flux_new

            h = hdul[i].copy()
            h.data['flux'] = flux_new
            h.data['flux_err'] = err_new
            h.data['mag'] = mag
            h.data['mag_err'] = mag_err
            h.header['SYN_EXPD'] = (float(E), 'exposure used for averaging [d]')
            out.append(h)

            # truth grid for this sector (point-sampled, clean signal only)
            dtt = ncfg.truth_cadence_min / 60.0 / 24.0
            tg = np.arange(np.nanmin(tt[good]), np.nanmax(tt[good]) + dtt, dtt)
            truth_t.append(tg); truth_x.append(signal(tg, 0.0))
            sector_records.append(dict(sector=int(h.header.get('SECTOR', -1)), n=int(good.sum()),
                                       exposure_min=E * 24 * 60, median_flux_template=float(med)))

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out.writeto(out_path, overwrite=True)

    rec = dict(path=out_path, template_name=str(hdr['SYN_TMPL']), template_path=template_path,
               F_template_es=F_tmpl, flux_ratio=flux_ratio, Tmag_template=float(tess_mag_from_flux(F_tmpl)),
               n_sectors=len(sectors), baseline_days=t1 - t0,
               rn_alpha=alpha, rn_log10_N_target=log10_N, rn_rms_realized=float(rn_rms),
               rn_rms_requested=float(rn_rms_requested), rn_rms_clipped=bool(rn_rms_requested > rn_rms),
               err_frac_med=float(np.nanmedian(np.concatenate([e[np.isfinite(e)] for _, _, e, _ in sectors]))
                                  / F_tmpl * (np.sqrt(flux_ratio) if ncfg.brightness_mode == 'physical' else 1.0)),
               sectors=sector_records)
    return rec, np.concatenate(truth_t), np.concatenate(truth_x)


def slug_ok(name):
    """The output basename must round-trip through lc_lsp._slug for the pipeline to find it."""
    base = f'hlsp_elk_tess_ffi_{name}_tess_v1_llc.fits'
    slug = base.replace('hlsp_elk_tess_ffi_', '').replace('_tess_v1_llc.fits', '')
    return _slug(slug) == _slug(name)

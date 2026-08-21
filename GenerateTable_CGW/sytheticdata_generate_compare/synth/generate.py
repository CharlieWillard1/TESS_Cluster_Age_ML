"""Synthetic TESS cluster light curves with known generating parameters.

The synthetic set is produced by copying each real FITS file and replacing only the
``flux`` column.  Timestamps, ``flux_err``, per-sector HDU structure, ``SECTOR`` header
keys and the primary header are inherited unchanged, so the output directory is
structurally identical to ``TESS_Cluster_Age_ML/light_curves/`` and the pipeline runs on
it with no modification -- only ``LC_DIR`` changes in the notebook.

Layout requirement
------------------
``lc_lsp.get_lc_path`` resolves a cluster with the fixed glob

    {lc_dir}/{origin}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits

and slug-matches the portion between the prefix and suffix.  Preserving each source
file's basename inside a same-named ``MW``/``LMC``/``SMC`` subdirectory is what keeps
that working; the basename is never reconstructed from the cluster name, because the
slug transform (lowercase, strip spaces/hyphens/brackets) is not invertible.

Generative model
----------------
Per cluster, on a regular 30-min grid spanning the full multi-year baseline::

    x(t) = 1 + red(t; alpha, rms) + sum_i A_i * sin(2*pi*t/P_i + phi_i)

with ``N ~ Uniform{0..10}`` oscillators, ``log10(A_i) ~ U(-4.5, -2.5)`` fractional
amplitude, ``P_i ~ Uniform(0.1, 9)`` days (linear, entirely inside the pipeline's
0.083-10 d search band), and ``(alpha, log10_N)`` drawn as *joint* pairs resampled from
the real table so their observed +0.87 correlation is preserved.

The red-noise amplitude is calibrated rather than assumed: ``rn_log10_N`` is a fit output
in psd-periodogram units, not a settable input.  Since periodogram power scales as
amplitude squared, generating unit-RMS red noise, measuring its recovered ``log10_N``,
and rescaling by ``10**((target - measured)/2)`` hits the target exactly in one step.
"""

import os
import shutil
import numpy as np
import pandas as pd
from astropy.io import fits

from table_pipeline.lc_lsp import (get_lc_path, compute_lsp, _fit_red_noise_vaughan,
                                   normalize_flux)


# Generative defaults -- see module docstring.
N_OSC_MAX      = 10
# Calibrated, not guessed.  With N ~ U{0..10} the summed sinusoid variance is
# sum(A_i^2)/2, so the range has to be set against the TOTAL, not against one oscillator:
#     log10A ~ U(-4.5,-2.5) -> median total RMS 1.34e-3   (4.2x the real 3.2e-4)
#     log10A ~ U(-5.0,-3.0) -> median total RMS 4.22e-4   <-- matches real
#     log10A ~ U(-5.5,-3.5) -> median total RMS 1.34e-4   (too quiet)
# U(-5,-3) also brackets the detectability transition cleanly: 1e-5 is far below the
# median fractional error (2.7e-4) and undetectable, 1e-3 is far above and obvious.
LOG10_AMP_RANGE = (-5.0, -3.0)
PERIOD_RANGE    = (0.1, 9.0)
GRID_CADENCE_MIN = 30.0


# ---------------------------------------------------------------------------
# Red noise
# ---------------------------------------------------------------------------

def red_noise_series(t_grid, alpha, rng, n_fft=None):
    """Unit-RMS red noise with PSD ~ f^-alpha, evaluated on ``t_grid``.

    Built in the Fourier domain: an amplitude envelope f^(-alpha/2) with uniform random
    phases, inverse-transformed and normalised.  ``t_grid`` must be regularly spaced.
    """
    n = int(n_fft or max(4096, 2 ** int(np.ceil(np.log2(max(len(t_grid), 2)))) * 2))
    f = np.fft.rfftfreq(n, d=1.0)
    f[0] = f[1]                       # avoid a divide-by-zero at DC
    env = f ** (-alpha / 2.0)
    env[0] = 0.0                      # no DC power; the mean is set explicitly
    phase = rng.uniform(0, 2 * np.pi, len(f))
    x = np.fft.irfft(env * np.exp(1j * phase), n)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        return np.zeros(len(t_grid))
    x = x / sd
    # Map the synthetic series onto the real time grid.
    src = np.linspace(t_grid[0], t_grid[-1], n)
    return np.interp(t_grid, src, x)


def _calibrate_red_scale(t_grid, unit_series, t_ref, err_ref, log10_N_target,
                         P_min, P_max, lsp_alpha):
    """Scale factor putting this realisation's SECTOR-level ``log10_N`` on target.

    The calibration must use the *same realisation and the same normalisation* as the
    data that will be written.  ``red_noise_series`` normalises to unit RMS over the full
    multi-year grid, but the pipeline measures ``log10_N`` on a single ~27 d sector, and
    for a red spectrum a short chunk carries far less variance than the whole baseline --
    a factor ~5 in RMS at alpha ~ 1.8.  Calibrating on a separately-normalised short
    realisation therefore under-scales the red noise badly, and the steeper the slope the
    worse it gets.

    Instead: interpolate the actual full-baseline realisation onto the reference sector,
    measure what ``log10_N`` it produces there, and rescale.  Periodogram power goes as
    amplitude^2, so one measurement fixes the scale exactly.
    """
    work = 1e-3                                   # arbitrary small working amplitude
    x_ref = np.interp(t_ref, t_grid, unit_series)
    flux = 1.0 + x_ref * work
    try:
        _, x, ex = normalize_flux(t_ref, flux, err_ref)
        lsp = compute_lsp(t_ref, x, ex, P_min=P_min, P_max=P_max, alpha=lsp_alpha,
                          n_bootstraps=1)
        log10_N_unit, _, _ = _fit_red_noise_vaughan(lsp['freqs'], lsp['power'], 1.0)
    except Exception:
        return 3.0e-4                             # fall back to the real-data median
    scale = work * 10.0 ** ((log10_N_target - log10_N_unit) / 2.0)
    return float(np.clip(scale, 1e-6, 1e-1))


# ---------------------------------------------------------------------------
# Parameter draws
# ---------------------------------------------------------------------------

def draw_parameters(rng, rn_pairs, n_osc_max=N_OSC_MAX,
                    log10_amp_range=LOG10_AMP_RANGE, period_range=PERIOD_RANGE):
    """Draw the generating parameters for one cluster.

    ``rn_pairs`` is an (M, 2) array of real ``(rn_alpha, rn_log10_N)`` pairs; drawing a
    row at a time preserves their joint distribution, including the +0.87 correlation
    that an independent draw of each would destroy.
    """
    n = int(rng.integers(0, n_osc_max + 1))       # inclusive of both 0 and n_osc_max
    periods = rng.uniform(*period_range, n)
    amps = 10.0 ** rng.uniform(*log10_amp_range, n)
    phases = rng.uniform(0, 2 * np.pi, n)
    alpha, log10_N = rn_pairs[rng.integers(0, len(rn_pairs))]
    return dict(n_osc=n, periods=periods, amplitudes=amps, phases=phases,
                rn_alpha=float(alpha), rn_log10_N_target=float(log10_N))


def load_rn_pairs(table_path):
    """Real (rn_alpha, rn_log10_N) pairs to resample from."""
    t = pd.read_pickle(table_path)
    v = t[['rn_alpha', 'rn_log10_N']].astype(float).dropna()
    return v.to_numpy()


# ---------------------------------------------------------------------------
# One cluster
# ---------------------------------------------------------------------------

def add_oscillators(t, x, params):
    """Add the injected sinusoids to an existing series."""
    for A, P, ph in zip(params['amplitudes'], params['periods'], params['phases']):
        x = x + A * np.sin(2.0 * np.pi * t / P + ph)
    return x


def generate_cluster(name, origin, real_lc_dir, out_lc_dir, params, rng,
                     P_min=(30/60/24)*4, P_max=10.0, lsp_alpha=10,
                     grid_cadence_min=GRID_CADENCE_MIN):
    """Write one synthetic FITS and return its ground-truth record.

    The noise-free full-baseline signal is built, used for the truth statistics, then
    sampled at the real timestamps and given Gaussian noise from the real ``flux_err``.
    The clean light curve itself is never written to disk.
    """
    src = get_lc_path(name, origin, real_lc_dir)
    dst_dir = os.path.join(out_lc_dir, origin)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))   # basename preserved verbatim

    with fits.open(src) as hdul:
        sectors = []
        for i in range(1, len(hdul)):
            d = hdul[i].data
            if d is None:
                continue
            tt = np.asarray(d['time'], float)
            ee = np.asarray(d['flux_err'], float)
            ff = np.asarray(d['flux'], float)
            sectors.append((i, tt, ee, ff))
        if not sectors:
            return None

        finite = np.concatenate([t[np.isfinite(t)] for _, t, _, _ in sectors])
        t0, t1 = float(finite.min()), float(finite.max())

        # Full-baseline grid; the noise-free signal on it is what truth is computed on.
        dt = grid_cadence_min / 60.0 / 24.0
        t_grid = np.arange(t0, t1 + dt, dt)
        rng_red = np.random.default_rng(rng.integers(0, 2**32 - 1))
        unit_red = red_noise_series(t_grid, params['rn_alpha'], rng_red)

        # Reference sector for the red-noise calibration: the longest one.  Calibrate
        # against THIS realisation so the sector-level log10_N lands on target.
        j = int(np.argmax([np.isfinite(t).sum() for _, t, _, _ in sectors]))
        _, t_ref, e_ref, f_ref = sectors[j]
        m = np.isfinite(t_ref) & np.isfinite(e_ref) & (e_ref > 0)
        med_f = float(np.nanmedian(f_ref[m])) if m.any() else 1.0
        err_frac_med = float(np.nanmedian(e_ref[m] / max(med_f, 1e-30))) if m.any() else np.nan
        rms = _calibrate_red_scale(t_grid, unit_red, t_ref[m],
                                   e_ref[m] / max(med_f, 1e-30),
                                   params['rn_log10_N_target'], P_min, P_max, lsp_alpha)

        x_grid = add_oscillators(t_grid, 1.0 + unit_red * rms, params)

        out = fits.HDUList([hdul[0].copy()])
        for i, tt, ee, ff in sectors:
            good = np.isfinite(tt)
            x = np.interp(tt, t_grid, x_grid)              # clean fractional flux
            med = np.nanmedian(ff[good & np.isfinite(ff)]) if good.any() else 1.0
            if not np.isfinite(med) or med == 0:
                med = 1.0
            flux_new = x * med
            noise = np.where(np.isfinite(ee), rng.normal(0.0, np.abs(ee)), 0.0)
            flux_new = flux_new + noise
            flux_new[~good] = np.nan                        # preserve the NaN pattern

            h = hdul[i].copy()
            h.data['flux'] = flux_new
            out.append(h)

        out.writeto(dst, overwrite=True)

    return dict(name=name, origin=origin, path=dst, rn_rms_realized=rms,
                err_frac_med=err_frac_med, t_grid=t_grid, x_grid=x_grid, **params)


def copy_layout_extras(real_lc_dir, out_lc_dir):
    """Copy non-FITS entries (e.g. the `readme`) so the tree mirrors the real one."""
    os.makedirs(out_lc_dir, exist_ok=True)
    for entry in os.listdir(real_lc_dir):
        p = os.path.join(real_lc_dir, entry)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(out_lc_dir, entry))

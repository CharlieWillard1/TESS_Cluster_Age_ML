"""Assign empirical variability to synthetic stars and build the noise-free cluster signal.

Model (same construction as ``v2_science/make_luminosity_lcs.py``, with the fixes):

* every stellar component is placed in its (Teff, log L) grid bin;
* if the bin is *sampled* (>= ``min_bin_stars`` catalogue stars) the component is variable
  with probability ``frac_var(bin)``; a variable component copies the ``(A1, P1[, A2, P2])``
  of a randomly chosen real variable star of that bin (``method='resample'``) or draws them
  from truncated log-normals fitted to the bin (``method='truncnorm'``);
* quiet components and components in unsampled bins contribute **constant** flux (the
  original dropped unsampled stars entirely -- their light simply vanished);
* the cluster signal is ``x(t) = 1 + sum_i f_i sum_k A_ik sin(2 pi t / P_ik + phi_ik)`` with
  ``f_i`` the component's share of the total TESS flux.  A box exposure of length ``E``
  multiplies each term by ``sinc(E / P_ik)`` exactly, so no fine time grid is needed.

The *light budget* reports what fraction of the cluster's TESS flux sits in each status, in
particular how much light the model cannot say anything about (unsampled bins + outside the
grid).  That is the "missing light" number asked for in the README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from .config import VariabilityConfig
from .empirical import EmpiricalGrid

STATUS = {0: 'variable', 1: 'quiet', 2: 'unsampled_sparse', 3: 'unsampled_empty', 4: 'outside_grid'}
STATUS_CODE = {v: k for k, v in STATUS.items()}


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def _truncated_lognormal(stats, bins, rng):
    """Draw 10**x with x ~ TruncNormal(mean, std, [min, max]) per bin (vectorised)."""
    mu = stats['mean'][bins]; sd = stats['std'][bins]
    lo = stats['min'][bins]; hi = stats['max'][bins]
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1e-3)
    a = (lo - mu) / sd; b = (hi - mu) / sd
    x = truncnorm.rvs(a, b, loc=mu, scale=sd, random_state=rng)
    return 10.0 ** x


def assign_variability(comp: pd.DataFrame, grid: EmpiricalGrid, vcfg: VariabilityConfig, rng):
    """Return a copy of ``comp`` with bin / status / sinusoid columns added.

    Added columns
    -------------
    bin             native grid bin of the star (-1 outside the grid extent)
    bin_used        bin whose statistics were applied (differs only for fallback='nearest')
    used_fallback   True if bin_used != bin
    status          one of STATUS (describes the star's own bin, before any fallback)
    is_var          bool
    A1, P1, phi1    primary sinusoid: fractional amplitude, period [d], phase [rad]
    A2, P2, phi2    secondary sinusoid (NaN if none)
    src_TIC         TIC of the real star copied (resample) or -1
    """
    out = comp.copy()
    n = len(out)
    teff = out['Teff'].to_numpy(float); logl = out['logL'].to_numpy(float)
    b = grid.assign(teff, logl)
    outside = b < 0
    bsafe = np.where(outside, 0, b)
    n_total = grid.n_total[bsafe]
    sampled = ~outside & grid.sampled[bsafe]
    empty = ~outside & (n_total == 0)
    sparse = ~outside & ~sampled & ~empty

    status = np.full(n, STATUS_CODE['quiet'], dtype=np.int16)   # int16: FITS has no signed byte
    status[sparse] = STATUS_CODE['unsampled_sparse']
    status[empty] = STATUS_CODE['unsampled_empty']
    status[outside] = STATUS_CODE['outside_grid']

    bin_used = b.copy()
    if vcfg.fallback == 'nearest':
        # clip outside stars onto the grid edge, then jump to the nearest sampled bin
        tcl = np.clip(teff, grid.teff_edges[0], grid.teff_edges[-1] - 1e-6)
        lcl = np.clip(logl, grid.logl_edges[0], grid.logl_edges[-1] - 1e-6)
        bclip = grid.assign(tcl, lcl)
        need = ~sampled
        bin_used[need] = grid.nearest_sampled[bclip[need]]
    elif vcfg.fallback != 'constant':
        raise ValueError(f'unknown fallback {vcfg.fallback!r}')
    used_fallback = bin_used != b
    eligible = bin_used >= 0
    eligible &= grid.sampled[np.where(eligible, bin_used, 0)]

    u = rng.uniform(size=n)
    fv = np.where(eligible, grid.frac_var[np.where(eligible, bin_used, 0)], 0.0)
    is_var = eligible & (u < fv)
    status[is_var & (status == STATUS_CODE['quiet'])] = STATUS_CODE['variable']
    # with fallback='nearest' an unsampled/outside star may become variable; its status keeps
    # recording the coverage problem, is_var records what was done.

    A1 = np.full(n, np.nan); P1 = np.full(n, np.nan); A2 = np.full(n, np.nan); P2 = np.full(n, np.nan)
    src = np.full(n, -1, dtype=np.int64)
    iv = np.where(is_var)[0]
    if len(iv):
        if vcfg.method == 'resample':
            s = grid.sample
            sA1 = s['A1'].to_numpy(float); sP1 = s['P1'].to_numpy(float)
            sA2 = s['A2'].to_numpy(float); sP2 = s['P2'].to_numpy(float); sTIC = s['TIC'].to_numpy()
            for bi in np.unique(bin_used[iv]):
                members = grid.variable_members(bi)
                sel = iv[bin_used[iv] == bi]
                pick = members[rng.integers(0, len(members), len(sel))]
                A1[sel] = sA1[pick]; P1[sel] = sP1[pick]; src[sel] = sTIC[pick]
                if vcfg.include_second_component:
                    A2[sel] = sA2[pick]; P2[sel] = sP2[pick]
        elif vcfg.method == 'truncnorm':
            A1[iv] = _truncated_lognormal(grid.stats['logA1'], bin_used[iv], rng)
            P1[iv] = _truncated_lognormal(grid.stats['logP1'], bin_used[iv], rng)
        else:
            raise ValueError(f'unknown method {vcfg.method!r}')

    out['bin'] = b
    out['bin_used'] = bin_used
    out['used_fallback'] = used_fallback
    out['status'] = status
    out['is_var'] = is_var
    out['A1'] = A1 / 1e6                   # ppm -> fractional
    out['P1'] = P1
    out['phi1'] = np.where(is_var, rng.uniform(0, 2 * np.pi, n), np.nan)
    out['A2'] = A2 / 1e6
    out['P2'] = P2
    out['phi2'] = np.where(np.isfinite(A2), rng.uniform(0, 2 * np.pi, n), np.nan)
    out['src_TIC'] = src
    return out


# ---------------------------------------------------------------------------
# Light budget ("missing light")
# ---------------------------------------------------------------------------

def light_budget(assigned: pd.DataFrame):
    """Fractions of the cluster's TESS flux (and bolometric luminosity) per status."""
    F = assigned['F_es'].to_numpy(float); L = assigned['L_bol'].to_numpy(float)
    Ft, Lt = np.nansum(F), np.nansum(L)
    st = assigned['status'].to_numpy()
    out = dict(F_total_es=float(Ft), n_components=int(len(assigned)))
    for code, name in STATUS.items():
        m = st == code
        out[f'n_{name}'] = int(m.sum())
        out[f'fluxfrac_{name}'] = float(np.nansum(F[m]) / Ft) if Ft > 0 else np.nan
        out[f'lbolfrac_{name}'] = float(np.nansum(L[m]) / Lt) if Lt > 0 else np.nan
    miss = np.isin(st, [STATUS_CODE['unsampled_sparse'], STATUS_CODE['unsampled_empty'],
                        STATUS_CODE['outside_grid']])
    out['fluxfrac_missing'] = float(np.nansum(F[miss]) / Ft) if Ft > 0 else np.nan
    out['lbolfrac_missing'] = float(np.nansum(L[miss]) / Lt) if Lt > 0 else np.nan
    out['n_missing'] = int(miss.sum())
    iv = assigned['is_var'].to_numpy(bool)
    out['n_var'] = int(iv.sum())
    out['fluxfrac_var'] = float(np.nansum(F[iv]) / Ft) if Ft > 0 else np.nan
    out['n_used_fallback'] = int(assigned['used_fallback'].sum())
    out['fluxfrac_used_fallback'] = float(np.nansum(F[assigned['used_fallback'].to_numpy(bool)]) / Ft) if Ft > 0 else np.nan
    # where is the missing light in the HR diagram?
    hot = miss & (assigned['Teff'].to_numpy(float) >= 12500.0)
    cool = miss & (assigned['Teff'].to_numpy(float) < 12500.0)
    out['fluxfrac_missing_hot_gt12500K'] = float(np.nansum(F[hot]) / Ft) if Ft > 0 else np.nan
    out['fluxfrac_missing_cool_lt12500K'] = float(np.nansum(F[cool]) / Ft) if Ft > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Clean signal
# ---------------------------------------------------------------------------

class CleanSignal:
    """Noise-free fractional cluster light curve, evaluated analytically at any times.

    ``a_j`` are the flux-weighted fractional amplitudes ``f_i * A_ik`` of every sinusoid term.
    ``__call__(t, exposure_days)`` returns the box-exposure-averaged signal, which for a
    sinusoid is exact: the average of ``sin(w t + phi)`` over ``[t - E/2, t + E/2]`` is
    ``sinc(E/P) sin(w t + phi)`` with ``sinc(x) = sin(pi x) / (pi x)``.
    """

    def __init__(self, a, P, phi):
        a = np.asarray(a, float); P = np.asarray(P, float); phi = np.asarray(phi, float)
        ok = np.isfinite(a) & np.isfinite(P) & np.isfinite(phi) & (a > 0) & (P > 0)
        self.a, self.P, self.phi = a[ok], P[ok], phi[ok]

    @property
    def n_terms(self):
        return len(self.a)

    def attenuation(self, exposure_days):
        return np.sinc(float(exposure_days) / self.P) if exposure_days > 0 else np.ones_like(self.P)

    def __call__(self, t, exposure_days=0.0, chunk=1024):
        t = np.asarray(t, float)
        x = np.ones_like(t)
        if self.n_terms == 0:
            return x
        amp = self.a * self.attenuation(exposure_days)
        w = 2.0 * np.pi / self.P
        for j0 in range(0, self.n_terms, chunk):
            sl = slice(j0, j0 + chunk)
            x += np.sin(t[:, None] * w[None, sl] + self.phi[None, sl]) @ amp[sl]
        return x

    def variance(self, exposure_days=0.0):
        """Analytic variance of the signal: sum (a_j sinc_j)^2 / 2 (periods assumed distinct)."""
        amp = self.a * self.attenuation(exposure_days)
        return float(np.sum(amp ** 2) / 2.0)

    def rms(self, exposure_days=0.0):
        return float(np.sqrt(self.variance(exposure_days)))


def build_clean_signal(assigned: pd.DataFrame):
    """Collect the sinusoid terms of all variable components into a CleanSignal."""
    F = assigned['F_es'].to_numpy(float)
    f = F / np.nansum(F)
    terms_a, terms_P, terms_phi = [], [], []
    for k in ('1', '2'):
        A = assigned['A' + k].to_numpy(float); P = assigned['P' + k].to_numpy(float)
        phi = assigned['phi' + k].to_numpy(float)
        m = np.isfinite(A) & np.isfinite(P) & assigned['is_var'].to_numpy(bool)
        terms_a.append(f[m] * A[m]); terms_P.append(P[m]); terms_phi.append(phi[m])
    return CleanSignal(np.concatenate(terms_a), np.concatenate(terms_P), np.concatenate(terms_phi))


def signal_summary(sig: CleanSignal):
    if sig.n_terms == 0:
        return dict(n_sinusoids=0, clean_rms=0.0, clean_rms_30min=0.0, amp_max=0.0,
                    period_of_max_amp=np.nan, period_min=np.nan, period_max=np.nan)
    j = int(np.argmax(sig.a))
    return dict(n_sinusoids=int(sig.n_terms), clean_rms=sig.rms(0.0),
                clean_rms_30min=sig.rms(30.0 / 60.0 / 24.0),
                amp_max=float(sig.a[j]), period_of_max_amp=float(sig.P[j]),
                period_min=float(sig.P.min()), period_max=float(sig.P.max()))

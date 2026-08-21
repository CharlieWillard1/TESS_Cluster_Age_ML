"""Fixed-basis Lorentzian (OU) sum for a power-law red-noise GP kernel.

Motivation
----------
A single fixed-Q SHO cannot represent ``S(f) ~ f^-alpha``: it is a *broken* power law
(flat below the break, ``f^-4`` above), so its only freedom is where the break sits and
its effective slope carries no information about the true alpha.

Instead, approximate the power law as a sum of Lorentzians with **fixed**, log-spaced
break frequencies and weights tied deterministically to alpha::

    S_RN(f; A, alpha)  =  A^2 * sum_j w_j(alpha) / (1 + (f/f_j)^2)

Only ``A`` and ``alpha`` are free, whatever N is.

The weights are closed-form.  In the continuum limit,

    integral w(f') / (1 + (f/f')^2)  dlog f'   proportional to   f^-alpha

when ``w(f') ~ f'^-alpha`` (substitute u = f/f'; the u-integral converges for
0 < alpha < 2).  Each ``quasisep.Exp`` term has a low-frequency plateau
``sigma_j^2 / (pi f_j)``, so a Lorentzian coefficient ``w_j`` corresponds to

    sigma_j^2  proportional to  f_j^(1 - alpha)

which is what ``basis_sigmas`` implements, normalised so the total variance is ``A^2``.

Sizing
------
Accuracy is limited by **truncation at the edges of the basis, not by density**.  At fixed
padding, adding terms does not help; extending the basis does.  Measured ripple (half the
peak-to-peak deviation from a pure power law, in dex, over 0.1-10 /day):

    pad (dec)   N    alpha=0.25   0.5     1.0     1.5    1.75
        0.0     6      0.311     0.206   0.078   0.206   0.311
        0.4     8      0.194     0.110   0.029   0.110   0.194
        1.2    13      0.093     0.039   0.005   0.039   0.093
        2.0    18      0.051     0.015   0.001   0.015   0.051

and at pad=1.2 the density barely matters (1.5/decade is as good as 6/decade).  So prefer
**wide and sparse** over narrow and dense.

Accuracy is strongly alpha-dependent, symmetric about alpha=1 and worst at the endpoints;
the construction only converges for 0 < alpha < 2.
"""

import numpy as np
import jax
import jax.numpy as jnp
from tinygp import GaussianProcess, kernels

jax.config.update("jax_enable_x64", True)

BAND = (0.1, 10.0)


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def basis_frequencies(band=BAND, pad_decades=1.2, per_decade=2.0):
    """Fixed log-spaced break frequencies, extended beyond the band."""
    lo = np.log10(band[0]) - pad_decades
    hi = np.log10(band[1]) + pad_decades
    n = max(3, int(round(per_decade * (hi - lo))))
    return np.logspace(lo, hi, n)


def basis_weights(fj, alpha):
    """Lorentzian coefficients w_j ~ f_j^-alpha, normalised to sum to 1."""
    w = fj ** (-alpha)
    return w / np.sum(w)


def basis_sigmas(fj, amp, alpha):
    """Per-term sigma_j such that the total process variance is ``amp**2``.

    sigma_j^2 ~ f_j^(1-alpha), from w_j ~ f_j^-alpha and the OU plateau ~ sigma^2/(pi f).
    """
    v = fj ** (1.0 - alpha)
    v = v / jnp.sum(v)
    return jnp.sqrt(amp ** 2 * v)


def rn_psd(f, fj, amp, alpha):
    """Analytic PSD of the summed basis (numpy, for plotting)."""
    f = np.atleast_1d(np.asarray(f, float))
    sig = np.asarray(basis_sigmas(jnp.asarray(fj), amp, alpha))
    # one-sided OU PSD: sigma^2 / (pi * f_j) / (1 + (f/f_j)^2)
    return np.sum((sig ** 2 / (np.pi * fj))[:, None]
                  / (1.0 + (f[None, :] / fj[:, None]) ** 2), axis=0)


def sho_psd(f, sigma, period, Q):
    """Analytic PSD of one quasisep.SHO component (matches gp_fit.sho_psd)."""
    f = np.atleast_1d(np.asarray(f, float))
    om = 2 * np.pi * f
    om0 = 2 * np.pi / period
    return 4.0 * sigma ** 2 * om0 ** 3 / (Q * ((om ** 2 - om0 ** 2) ** 2
                                               + om ** 2 * om0 ** 2 / Q ** 2))


# ---------------------------------------------------------------------------
# Kernel + fit
# ---------------------------------------------------------------------------

def build_kernel(theta, fj, n_sho):
    """Chained kernel: N fixed-frequency OU terms (2 free params) + n_sho SHOs.

    theta = [log_amp, alpha,  (log_sigma, log_period, log_Q) * n_sho]
    """
    amp = jnp.exp(theta[0])
    alpha = theta[1]
    sig = basis_sigmas(jnp.asarray(fj), amp, alpha)

    kernel = None
    for j, f_ in enumerate(fj):
        k = kernels.quasisep.Exp(scale=1.0 / (2.0 * np.pi * f_), sigma=sig[j])
        kernel = k if kernel is None else kernel + k

    for i in range(n_sho):
        s, p, q = theta[2 + 3 * i], theta[3 + 3 * i], theta[4 + 3 * i]
        kernel = kernel + kernels.quasisep.SHO(omega=2.0 * jnp.pi / jnp.exp(p),
                                               quality=jnp.exp(q), sigma=jnp.exp(s))
    return kernel


def neg_loglike(theta, t, y, yerr, fj, n_sho):
    gp = GaussianProcess(build_kernel(theta, fj, n_sho), t,
                         diag=yerr ** 2 + 1e-24, mean=0.0)
    return -gp.log_probability(y)


def fit(t, y, yerr, fj, n_sho=0, theta0=None, bounds=None):
    """Fit by L-BFGS-B with JAX gradients. Returns (theta, result)."""
    from scipy.optimize import minimize

    if theta0 is None:
        theta0 = [np.log(np.std(y)), 1.0]
        for _ in range(n_sho):
            theta0 += [np.log(np.std(y) * 0.3), np.log(1.0), np.log(20.0)]
        theta0 = np.array(theta0)
    if bounds is None:
        bounds = [(np.log(1e-8), np.log(1e3)), (0.05, 1.95)]
        for _ in range(n_sho):
            bounds += [(np.log(1e-8), np.log(1e3)),
                       (np.log(0.05), np.log(20.0)),
                       (np.log(0.5), np.log(500.0))]

    val_grad = jax.jit(jax.value_and_grad(neg_loglike),
                       static_argnums=(5,))

    def f(th):
        v, g = val_grad(jnp.asarray(th), t, y, yerr, jnp.asarray(fj), n_sho)
        return float(v), np.asarray(g, dtype=float)

    res = minimize(f, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
    return res.x, res


# ---------------------------------------------------------------------------
# Data generation (no observational noise, dense regular sampling)
# ---------------------------------------------------------------------------

def make_powerlaw_noise(t, alpha, rms, rng, n_fft=1 << 15):
    """Series whose PSD is a true f^-alpha power law, sampled at t."""
    T = float(np.ptp(t)) * 2.0
    f = np.fft.rfftfreq(n_fft, d=T / n_fft)
    f[0] = f[1]
    env = f ** (-alpha / 2.0)
    env[0] = 0.0
    x = np.fft.irfft(env * np.exp(1j * rng.uniform(0, 2 * np.pi, len(f))), n_fft)
    x = x / x.std() * rms
    return np.interp(t, np.linspace(t.min(), t.max(), n_fft), x)


def seed_sho_periods(t, y, fj, n_sho, band=BAND, n_freq=2000):
    """Seed SHO periods the way the pipeline does: peaks above the fitted continuum.

    Fit the red-noise basis alone first, then take the largest maxima of
    ``periodogram / fitted continuum``.  Seeding from the raw periodogram instead would
    just find the top of the red-noise ramp, and seeding from a fixed guess leaves the
    optimiser in a local minimum whenever the true period is far from it.
    """
    from astropy.timeseries import LombScargle

    th_rn, _ = fit(t, y, np.full_like(t, 1e-6), fj, n_sho=0)
    freq = np.linspace(band[0], band[1], n_freq)
    power = LombScargle(t, y).power(freq, normalization="psd")
    cont = rn_psd(freq, fj, float(np.exp(th_rn[0])), float(th_rn[1]))
    ratio = power / np.maximum(cont, 1e-300)

    periods, mask = [], np.ones(len(freq), bool)
    for _ in range(n_sho):
        if not mask.any():
            break
        i = int(np.argmax(np.where(mask, ratio, -np.inf)))
        periods.append(1.0 / freq[i])
        mask &= np.abs(freq / freq[i] - 1.0) > 0.15      # suppress the peak just taken
    return periods, th_rn


def fit_with_seeding(t, y, yerr, fj, n_sho=1, band=BAND):
    """Two-stage fit: red noise alone, then seed the SHOs from the residual peaks."""
    if n_sho == 0:
        return fit(t, y, yerr, fj, n_sho=0)

    periods, th_rn = seed_sho_periods(t, y, fj, n_sho, band=band)
    theta0 = [float(th_rn[0]), float(th_rn[1])]
    for P in periods:
        theta0 += [np.log(np.std(y) * 0.5), np.log(P), np.log(20.0)]
    bounds = [(np.log(1e-8), np.log(1e3)), (0.05, 1.95)]
    for _ in periods:
        bounds += [(np.log(1e-8), np.log(1e3)),
                   (np.log(1.0 / band[1]), np.log(1.0 / band[0])),   # period in [1/f_hi, 1/f_lo]
                   (np.log(0.5), np.log(500.0))]
    return fit(t, y, yerr, fj, n_sho=len(periods),
               theta0=np.array(theta0), bounds=bounds)

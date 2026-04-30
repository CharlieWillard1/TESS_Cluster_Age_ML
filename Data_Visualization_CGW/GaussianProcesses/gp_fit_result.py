import numpy as np
import jax.numpy as jnp


class GPFitResult:
    """Structured result from iterative_sho_gp_fit()."""

    def __init__(
        self,
        t,
        x,
        y,
        yerr,
        gp_mean,
        gp_std,
        residual,
        features,
        final_fit,
        all_accepted_fits,
        residual_lsps,
        white_noise_info,
    ):
        self.t = t
        self.x = x
        self.y = y
        self.yerr = yerr
        self.gp_mean = gp_mean
        self.gp_std = gp_std
        self.residual = residual
        self.features = features
        self.final_fit = final_fit
        self.all_accepted_fits = all_accepted_fits
        self.residual_lsps = residual_lsps
        self.white_noise_info = white_noise_info

    @property
    def n_components(self):
        return int(self.features["n_gp_components"])

    @property
    def periods(self):
        return [self.features[f"sho_{i}_period_days"] for i in range(1, self.n_components + 1)]

    @property
    def bic(self):
        return float(self.features["gp_bic"])

    @property
    def log_likelihood(self):
        return float(self.features["gp_log_likelihood"])

    def component(self, i):
        """Return parameter dict for 1-indexed component i (keys: sigma, period_days, omega, Q)."""
        if not (1 <= i <= self.n_components):
            raise IndexError(f"Component index {i} out of range [1, {self.n_components}]")
        return {
            "sigma":       self.features[f"sho_{i}_sigma"],
            "period_days": self.features[f"sho_{i}_period_days"],
            "omega":       self.features[f"sho_{i}_omega"],
            "Q":           self.features[f"sho_{i}_Q"],
        }

    def interp(self, x_new):
        """
        Predict GP posterior mean flux at new times x_new.

        x_new must be in the same units as self.x (days relative to self.t[0]).
        """
        cond_gp = self.final_fit["gp"].condition(
            jnp.asarray(self.y), jnp.asarray(x_new)
        ).gp
        return np.asarray(cond_gp.mean)

    def kspace_realized(self, n_uniform=4096, subtract_mean=True, window=True):
        """
        FFT-based power spectrum of the GP posterior mean (data-dependent realization).

        Samples the conditioned posterior mean on a uniform time grid, applies
        an optional Hann window, then computes the real FFT.  The result
        reflects both the kernel structure and the specific data that was fitted.

        Parameters
        ----------
        n_uniform : int
            Number of uniform grid points (default 4096).
        subtract_mean : bool
            Remove mean from GP samples before FFT to suppress the DC bin.
        window : bool
            Apply a Hann window before FFT to reduce spectral leakage.

        Returns
        -------
        dict with keys:
            freq       : ndarray, frequencies in 1/day (length n_uniform//2 + 1)
            power      : ndarray, FFT power |FFT|^2 (same length as freq)
            t_uniform  : ndarray, uniform absolute time grid
            gp_uniform : ndarray, GP posterior mean sampled on that grid
        """
        x_uniform = np.linspace(0.0, float(self.x.max()), n_uniform)
        gp_uniform = self.interp(x_uniform)

        if subtract_mean:
            gp_uniform = gp_uniform - np.nanmean(gp_uniform)

        dt = x_uniform[1] - x_uniform[0]
        fft_input = gp_uniform * np.hanning(n_uniform) if window else gp_uniform

        fft_vals = np.fft.rfft(fft_input)
        freq  = np.fft.rfftfreq(n_uniform, d=dt)
        power = np.abs(fft_vals) ** 2

        return {
            "freq":       freq,
            "power":      power,
            "t_uniform":  self.t.min() + x_uniform,
            "gp_uniform": gp_uniform,
        }

    def kspace_true(self, freq_min=0.1, freq_max=10.0, n_freq=1000):
        """
        Analytic prior power spectral density of the fitted GP kernel.

        For each SHO component with parameters (sigma, omega_0, Q), the PSD is:

            S(f) = sigma^2 * (omega_0/Q) * (omega_0^2 + omega^2)
                   / ((omega^2 - omega_0^2)^2 + omega^2 * omega_0^2 / Q^2)

        where omega = 2*pi*f (rad/day).  This integrates to sigma^2 over all f
        and is independent of the observed data.

        Parameters
        ----------
        freq_min, freq_max : float
            Frequency range in 1/day (default 0.1–10).
        n_freq : int
            Number of frequency grid points (default 1000).

        Returns
        -------
        dict with keys: freq (1/day), power (analytic PSD)
        """
        from GP_Fit import unpack_theta  # deferred — GP_Fit imports this module

        freq  = np.linspace(freq_min, freq_max, n_freq)
        omega = 2.0 * np.pi * freq

        comps, _ = unpack_theta(self.final_fit["theta"], self.n_components)

        power = np.zeros(n_freq)
        for comp in comps:
            sigma  = float(comp["sigma"])
            omega0 = float(comp["omega"])
            Q      = float(comp["Q"])

            numer  = sigma**2 * (omega0 / Q) * (omega0**2 + omega**2)
            denom  = (omega**2 - omega0**2)**2 + omega**2 * omega0**2 / Q**2
            power += numer / denom

        return {"freq": freq, "power": power}

    def to_dict(self):
        """Reconstruct the legacy dict return for backward compatibility."""
        return {
            "features":          self.features,
            "final_fit":         self.final_fit,
            "all_accepted_fits": self.all_accepted_fits,
            "residual_lsps":     self.residual_lsps,
            "white_noise_info":  self.white_noise_info,
            "t":                 self.t,
            "x":                 self.x,
            "y":                 self.y,
            "yerr":              self.yerr,
            "gp_mean":           self.gp_mean,
            "gp_std":            self.gp_std,
            "residual":          self.residual,
        }

    def __repr__(self):
        period_str = ", ".join(f"{p:.3f}" for p in self.periods)
        return (
            f"GPFitResult("
            f"n_components={self.n_components}, "
            f"periods=[{period_str}] days, "
            f"BIC={self.bic:.1f}, "
            f"loglike={self.log_likelihood:.1f}"
            f")"
        )

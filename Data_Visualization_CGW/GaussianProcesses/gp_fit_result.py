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
        all_fits=None,
        all_residual_lsps=None,
        peaks=None,
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
        # All attempted fits (may differ from all_accepted_fits if BIC stopped early)
        self.all_fits = all_fits if all_fits is not None else all_accepted_fits
        self.all_residual_lsps = all_residual_lsps if all_residual_lsps is not None else residual_lsps
        # Initial LSP peak list [{freq, period, power}, ...] sorted by power desc
        self.peaks = peaks if peaks is not None else []

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

    def kspace_compare_allfits(self, freq_min=0.1, freq_max=10.0, n_freq=1000):
        """
        Compute analytic PSD for every attempted m-component model in all_fits.

        Returns a list of dicts (one per fit) with keys:
            n_components, bic, periods, freq, power

        Pass the result to visualize_gp_afterfit.plot_kspace_compare_allfits
        to overlay them all in a single figure.
        """
        from GP_Fit import unpack_theta  # deferred

        freq  = np.linspace(freq_min, freq_max, n_freq)
        omega = 2.0 * np.pi * freq

        entries = []
        for fit in self.all_fits:
            m = fit["n_components"]
            comps, _ = unpack_theta(fit["theta"], m)

            power = np.zeros(n_freq)
            for comp in comps:
                sigma  = float(comp["sigma"])
                omega0 = float(comp["omega"])
                Q      = float(comp["Q"])
                numer  = sigma**2 * (omega0 / Q) * (omega0**2 + omega**2)
                denom  = (omega**2 - omega0**2)**2 + omega**2 * omega0**2 / Q**2
                power += numer / denom

            entries.append({
                "n_components": m,
                "bic": float(fit["bic"]),
                "periods": [float(c["period"]) for c in comps],
                "freq": freq,
                "power": power,
            })

        return entries

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
            "all_fits":          self.all_fits,
            "all_residual_lsps": self.all_residual_lsps,
            "peaks":             self.peaks,
        }

    def __getstate__(self):
        # Strip gp objects before pickling — theta is sufficient to rebuild them.
        state = self.__dict__.copy()
        def _slim(fit):
            return {k: v for k, v in fit.items() if k != "gp"}
        state["final_fit"] = _slim(state["final_fit"])
        state["all_accepted_fits"] = [_slim(f) for f in state["all_accepted_fits"]]
        state["all_fits"] = [_slim(f) for f in state["all_fits"]]
        return state

    def __setstate__(self, state):
        from GP_Fit import build_multi_sho_gp
        import jax.numpy as jnp
        self.__dict__.update(state)
        def _rebuild(fit):
            gp = build_multi_sho_gp(
                jnp.asarray(fit["theta"]),
                jnp.asarray(self.x),
                jnp.asarray(self.yerr),
                fit["n_components"],
            )
            return {**fit, "gp": gp}
        self.final_fit = _rebuild(self.final_fit)
        self.all_accepted_fits = [_rebuild(f) for f in self.all_accepted_fits]
        self.all_fits = [_rebuild(f) for f in self.all_fits]

    def __repr__(self):
        period_str = ", ".join(f"{p:.3f}" for p in self.periods)
        n_attempted = len(self.all_fits)
        best_bic_str = f"BIC={self.bic:.1f}"
        return (
            f"GPFitResult("
            f"best={self.n_components} components [{period_str}] d, "
            f"{best_bic_str}, "
            f"loglike={self.log_likelihood:.1f}, "
            f"n_attempted={n_attempted}"
            f")"
        )

# Gaussian Process Fitting — Module Overview

Fits multi-component SHO (Stochastically-driven Harmonic Oscillator) Gaussian
Processes to TESS light curves to extract stellar rotation period and spectral
shape features for cluster age analysis.

---

## Files

### `GP_Fit.py` — Core fitting engine (use this one)

The main entry point is `iterative_sho_gp_fit`, which runs an exhaustive-search
fit over 1..N SHO components and selects the best model by BIC.

**Pipeline overview:**
1. Clean LC → estimate white-noise LSP threshold (bootstrap) → compute LSP once
2. Detect all significant peaks with harmonic masking (`find_all_significant_peaks`)
3. For m = 1 … N peaks, fit an m-component GP seeded by the top-m LSP peaks
4. BIC early-stopping; return `GPFitResult` for the best model

**Key functions:**

| Function | Purpose |
|---|---|
| `clean_lc(t, flux, flux_err)` | Mask NaNs, sort, median-subtract, zero-shift time |
| `unpack_theta(theta, n_components)` | Decode log-space parameter vector → sigma, period, omega, Q per component |
| `build_multi_sho_gp(theta, x, yerr, n)` | Build `tinygp` GP with sum-of-SHO kernel |
| `fit_multi_sho_gp(x, y, yerr, period_guesses, ...)` | L-BFGS-B optimisation with JAX JIT + analytic gradient |
| `make_initial_theta(period_guesses, y, yerr)` | Sensible parameter initialisation from LSP periods and data variance |
| `make_bounds(n_components, ...)` | L-BFGS-B bounds in log space (sigma, period, Q, jitter) |
| `predict_gp_at_training_times(fit, x, y)` | Conditioned posterior mean + std at observation times |
| `compute_gp_residuals(fit, x, y)` | Returns (residual, mu, std) |
| `residual_lsp_peak(t, residual, ...)` | Lomb-Scargle on any flux array; returns freq, power, best period, FAP |
| `estimate_lsp_white_noise_level(t, flux_err, ...)` | Bootstrap threshold: 99th percentile of max LSP power from pure noise |
| `find_all_significant_peaks(freq, power, threshold, ...)` | Iteratively find peaks above threshold; mask harmonics/sub-harmonics |
| `compute_bic(loglike, n_params, n_data)` | Standard BIC formula |
| `summarize_components(theta, n_components)` | Returns flat dict of sigma/period/omega/Q per SHO component |
| `iterative_sho_gp_fit(t, flux, flux_err, ...)` | **Main entry point** — full pipeline, returns `GPFitResult` |
| `save_fits(path, table, results)` / `load_fits(path)` | Pickle a (table, list[GPFitResult]) pair |
| `load_gp_checkpoint(path)` | Load a single mid-run checkpoint GPFitResult |

**`iterative_sho_gp_fit` key parameters:**

| Parameter | Default | Meaning |
|---|---|---|
| `max_components` | 6 | Upper limit on SHO components |
| `min_period / max_period` | 0.05 / 50 days | Period search range |
| `bic_improvement_threshold` | 10.0 | Minimum ΔBIC to accept an extra component |
| `harmonic_masking` | True | Suppress harmonics when searching for next peak |
| `fit_full_only` | False | Skip 1..N-1 models, fit only N-component model |
| `debug_fitalln` | False | Ignore BIC early-stopping; fit all N models |
| `checkpoint_dir` | None | Save partial results after each m to pickle files |
| `verbose` | 0 | 0=silent, 1=progress text, 2=also diagnostic plots |

---

### `GP_Fit_Iterative.py` — Alternative iterative approach (Approach B, older)

Grows the model one component at a time by fitting a residual LSP after each
step, rather than detecting all peaks upfront.  Does not support harmonic
masking, checkpointing, or `find_all_significant_peaks`.  Kept for comparison.

Key difference from `GP_Fit.py`: also uses FAP as a stopping criterion via
`should_add_component(residual_lsp, white_noise_info, fap_threshold)`.

---

### `gp_fit_result.py` — `GPFitResult` data class

Structured container returned by `iterative_sho_gp_fit`.  Handles pickle
serialisation automatically (strips/rebuilds JAX GP objects via
`__getstate__` / `__setstate__`).

**Attributes:**

| Attribute | Content |
|---|---|
| `t`, `x`, `y`, `yerr` | Cleaned light curve (absolute times, zero-shifted times, median-subtracted flux, errors) |
| `gp_mean`, `gp_std` | Posterior mean and std at training times for the best-BIC model |
| `residual` | `y - gp_mean` |
| `features` | Flat dict of all extracted GP parameters + summary scalars |
| `final_fit` | Raw optimiser result dict for the best model |
| `all_fits` | List of raw fit dicts for every attempted m-component model |
| `all_accepted_fits` | Subset accepted before BIC early-stopping |
| `residual_lsps` | LSP dict for residuals after each m-component fit |
| `white_noise_info` | Bootstrap threshold dict |
| `peaks` | List of `{freq, period, power}` dicts from initial peak detection |

**Properties:** `n_components`, `periods`, `bic`, `log_likelihood`

**Methods:**

| Method | Purpose |
|---|---|
| `component(i)` | Parameter dict for 1-indexed SHO component i |
| `interp(x_new)` | Posterior mean at new time array (same units as `self.x`) |
| `kspace_realized(n_uniform, subtract_mean, window)` | FFT of GP posterior mean on uniform grid (data-dependent) |
| `kspace_true(freq_min, freq_max, n_freq)` | Analytic prior PSD of each SHO kernel; data-independent |
| `kspace_compare_allfits(...)` | Analytic PSD for every m-component model in `all_fits` |
| `to_dict()` | Legacy dict representation for backward compatibility |

---

### `gp_summary_stats.py` — Spectral feature extraction

Computes summary statistics from any frequency-power spectrum (LSP or GP PSD)
via integrals over d(log10 f) for scale-consistent results.

**`compute_spectral_stats(freq, power, bins, period_lim)`**

Returns a flat dict of:
- **Raw amplitude:** `peak_freq`, `peak_period`, `peak_amplitude`, `total_power`, `band_power_0..N`
- **Normalised shape:** `band_frac_0..N`, `spectral_centroid_logfreq`, `spectral_width_logfreq`, `power_entropy`, `peak_to_continuum`

`bins` controls frequency band layout: `None` → 3 log-spaced bins, `int` → that many bins, `array` → explicit period-space edges.

**`add_gp_summary_stats(subset_table, results, bins, period_lim)`**

Applies `compute_spectral_stats` to both the raw-data LSP and the analytic GP
PSD (`kspace_true`) for every row, adding `lsp_*` and `gp_kspace_*` columns
to an astropy Table or pandas DataFrame.

---

### `visualize_gp_afterfit.py` — Post-fit plots

| Function | Output |
|---|---|
| `plot_gp_fit(results, table_rows, plot_all_ncomponent_fits)` | LC + GP mean + ±1σ band per star; all GP means overlaid on final panel |
| `plot_kspace(results, ..., plot_all_ncomponent_fits)` | 3-row × 2-col figure: observed LSP / realised FFT kspace / analytic kernel PSD, in both frequency and period domains |
| `plot_summary_stats(table, cluster_name, stats)` | Scatter of `lsp_*` vs `gp_kspace_*` summary stats vs n_sectors for up to 4 clusters |
| `plot_kspace_compare_allfits(result, ...)` | Overlay analytic PSD for all m-component models; best-BIC model highlighted |

---

### `visualize_gp_duringfit.py` — Diagnostic plots during fitting (verbose=2)

Called automatically by `iterative_sho_gp_fit` when `verbose >= 2`.

| Function | Output |
|---|---|
| `plot_initial_lsp(freq, power, peaks, masked_windows, white_noise_info)` | Initial LSP in period domain with white-noise threshold, found peaks, and shaded harmonic masks |
| `plot_before_fit(x, y, yerr, lsp_dict, white_noise_info, component_idx)` | LC + LSP before fitting component m |
| `plot_after_fit(x, y, yerr, all_accepted_fits, component_idx)` | Row-per-component breakdown: full GP mean in row 1, each subsequent component isolated from residuals |
| `plot_fit_summary(x, y, yerr, fits, component_idx, resid, resid_lsp, white_noise_info, bic)` | Combined figure: component rows + residual flux + residual LSP, BIC in title |

---

### `SimulationComparisons/` — Simulation validation notebook

`SimulationComparison.ipynb` compares GP-recovered periods against simulation
ground truth, with output plots (`sim_comparison.png`,
`sim_comparison_overdense.png`, `sim_comparison_Qual.png`).

---

## Typical usage

```python
from GP_Fit import iterative_sho_gp_fit, load_fits, save_fits
from gp_summary_stats import add_gp_summary_stats
from visualize_gp_afterfit import plot_gp_fit, plot_kspace

# Fit a single star
result = iterative_sho_gp_fit(
    t, norm_flux, flux_err,
    max_components=4,
    checkpoint_dir="scratch/",
    verbose=1,
)

# Inspect result
print(result)                    # GPFitResult(best=2 components [3.141, 6.283] d, BIC=-1234.5, ...)
print(result.features)           # flat dict of all parameters + summary scalars
print(result.periods)            # [3.141, 6.283]

# Spectral content
kr = result.kspace_realized()    # FFT of GP posterior
kt = result.kspace_true()        # analytic PSD

# Plots
plot_gp_fit([result], [table_row])
plot_kspace([result], [table_row])

# Batch: save / reload
save_fits("my_fits.pkl", subset_table, results)
subset_table, results = load_fits("my_fits.pkl")

# Add ML feature columns
subset_table = add_gp_summary_stats(subset_table, results,
                                    bins=[0.5, 1, 2, 5, 10])
```

---

## Dependencies

`numpy`, `scipy`, `astropy`, `matplotlib`, `jax` (with x64 mode), `tinygp`

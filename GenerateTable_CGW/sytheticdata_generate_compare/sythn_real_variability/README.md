# Synthetic cluster light curves from stellar populations (`sythn_real_variability`)

A re-implementation of the synthetic-population light-curve construction in
`Amaya_Dropbox/tess_clusters/synthetic_populations/v2_science/`, with the problems found in the
review fixed, placed at the LMC distance, and pushed through the **existing** synthetic
pipeline (`../synth/generate.py`) so the output is a TESS-sampled, noisy light-curve set that
`table_pipeline` reads with no code changes.

> The directory name keeps the spelling it was requested with (`sythn_real_variability`).

## Construction (same technique as the Dropbox code)

```
MIST isochrone population  ->  bin every star on the empirical (Teff, log L) grid
   -> variable with prob. frac_var(bin); copy (A, P) of a real TF2023/TF2026 star in that bin
   -> x(t) = 1 + sum_i f_i sum_k A_ik sin(2 pi t / P_ik + phi_ik),  f_i = star's share of the TESS flux
   -> exposure-average onto a real LMC cluster's timestamps, add its flux_err noise (+ calibrated red noise)
   -> FITS file identical in structure to the real one
```

| stage | module | what it does |
|---|---|---|
| empirical grid | `empirical.py` | TF2026 variables + quiet stars + TF2023 (SSV/DSV) → one table, 150 K × 0.15 dex bins, per-bin variable fraction, (A, P) moments **and member lists** |
| populations | `populations.py` | `isochrones` (MIST; Salpeter 0.1–300 M⊙, f_B 0.4, γ 0.3, δ-SFH, σ[Fe/H] 0.01) at **49.59 kpc** (Pietrzyński+2019); fresh IMF draw per instance; TESS mags from the MIST BC grid |
| variability | `variability.py` | bin → Bernoulli(frac_var) → resample a real (A1, P1[, A2, P2]) pair → analytic integrated signal |
| TESS sampling | `tess_sampling.py` | brightness-matched real LMC template; sinc(E/P) exposure averaging; Gaussian noise from the template `flux_err`; optional red noise calibrated with `synth.generate._calibrate_red_scale`; FITS writing |
| driver | `run_generate.py` | `populations` (env `synth_pop`) and `lcs` (env `tess_gp`) stages; truth table; base table for the pipeline |
| QA | `checks.py`, `make_figs.py` | physical-robustness report and figures in `results/` |

All parameters live in `config.py` (`RunConfig`) and are written to `results/run_config.json`
and `results/construction_parameters.md` on every run.

## Fixes relative to the Dropbox code

| issue in `v2_science` | what this code does |
|---|---|
| `calc_lum_stats.py` paired an unsorted `glob` (minus its first element) with a fixed list of luminosity names → mislabeled statistics | no luminosity-step files; every cluster is one named FITS and the truth table carries its parameters |
| light-curve "error" column was `0.01 × magnitude`, then converted *as a magnitude* to a flux error | per-point errors are the real template `flux_err` (checked to match the real point-to-point scatter to 7 %) |
| stars whose bin had no catalogue star contributed **zero** flux (`np.zeros`) | they contribute constant flux; the light fraction they carry is reported (see *Missing light*) |
| periods drawn from an untruncated normal (can be ≤ 0); amplitudes from a uniform between bin min/max | a real star's (A, P) pair is copied (`method='resample'`), or truncated log-normals (`method='truncnorm'`); catalogue sanity cuts 1 ≤ A ≤ 5×10⁵ ppm, 0.005 ≤ P ≤ 15 d |
| only [Fe/H] = −1 and 0 were run | all three metallicities; ages 6.5–10.0 in 0.25 dex |
| 15 "instances" reused one population and only re-rolled the variability | a fresh IMF draw per instance (`fresh_population_per_instance=True`) |
| uniform 3.9-min cadence, 27 d, no gaps, no noise | real TESS timestamps (30 / 10 / 3.3 min sectors, gaps, multi-year baseline), real noise, exposure averaging |
| `TESS_mag_*` columns of the isochrone tables are NaN for a random half of the stars (up to 94 % of the light in young populations) | magnitudes from `M_bol = 4.74 − 2.5 log L` and the MIST BC grid (`isochrones`), or a polynomial fitted to 285,963 MIST stars (rms 0.026 mag) when `isochrones` is absent |
| distance 10 pc (absolute magnitudes), `fB` comment "minimum 1 Msun" stale, IMF lower bound 0.008 silently truncated at 0.1 | LMC distance modulus 18.477 (+ optional A_V, A_T/A_V = 0.61); IMF bounds (0.1, 300) stated |
| bare `except:` hid failures; bin assignment re-stacked the whole table 4160 times | vectorised binning; errors propagate |
| `table[mask]['subclass'] = ...` assigned to a copy (no-op) | gone |
| red/"day" band names in `calc_lum_stats` select frequency bands | not reproduced; statistics come from the pipeline itself |

## Missing light — read this before using young clusters

The empirical grid stops at **Teff = 12,500 K** (the catalogues are cut at 15,000 K and have
almost no stars above ~10,000 K), and bins with fewer than `min_bin_stars = 5` catalogue stars
are treated as unsampled.  Stars in those bins are kept at **constant flux**: the model has no
information about their variability, so it injects none.  The fraction of the cluster's TESS
flux that sits in such stars is recorded per cluster (`fluxfrac_missing`, split into
`fluxfrac_outside_grid`, `fluxfrac_unsampled_sparse`, `fluxfrac_unsampled_empty`) and summarised
per (age, [Fe/H]) in `results/missing_light_by_population.csv` and `results/fig_missing_light.png`.

Measured on the production set (10,000 systems, 15 instances, mean over [Fe/H]):

| log age | 6.5 | 6.75 | 7.0 | 7.25 | 7.5 | 7.75 | 8.0 | 8.25 | 8.5 | 8.75 | 9.0 | ≥ 9.25 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| missing light | 94 % | 98 % | 97 % | 87 % | 90 % | 75 % | 70 % | 44 % | 30 % | 15 % | 7 % | 2–7 % |

Below log age ≈ 7.5 the light is in OB stars beyond the grid's hot edge and in red supergiants
whose bins hold no catalogue stars; at log age 8 it is mostly B stars; from log age 8.75 on
the grid covers the population.  Integrated-variability predictions for young clusters are
therefore lower limits that ignore exactly the stars that dominate the light;
`fallback='nearest'` (borrow the nearest sampled bin) is available but is an extrapolation, not
a measurement.

## Other caveats (documented, not fixed)

* **Selection bias in `frac_var`.** The variable fraction is 0.71 for catalogue stars brighter
  than Tmag 10.2 and 0.37 for fainter ones: detectability, not astrophysics.  The grid applies
  the bright-star fractions to every synthetic star of the same Teff / log L.
* **Only sinusoids.** Each variable star is one (or two) sinusoid(s) with the catalogue's
  single-sector period; eclipses, flares, granulation and red-noise-like stellar signals are not
  modelled beyond the optional injected red noise.
* **Brightness.** 10,000 systems ≈ 3,500 M⊙ gives Tmag 11–17 at the LMC, fainter than most of
  the real LMC clusters (Tmag 7.8–12.1).  Templates are chosen among the brightness-nearest real
  clusters so the noise level is that of a comparably bright real cluster; `--n-systems` scales
  the population (`checks_report.md` lists the number needed to match the median real cluster).
* **Red noise** is resampled from the real LMC pipeline rows, exactly as in `../synth`; it is
  an instrumental/astrophysical stand-in, not a prediction of the population model.  A handful
  of real rows (α ≈ 1.9, log10 N ≈ 3) calibrate to 5–9 % rms, so the realised rms is capped at
  `rn_rms_max = 2 %` (`rn_rms_requested` / `rn_rms_clipped` in the truth table record it).  Use
  `--no-red-noise` for the pure population signal.
* **Variance of a sum of sinusoids** assumes distinct periods; identical (A, P) pairs copied
  from the same catalogue star add coherently only through their random phases.

## How to run

```bash
cd GenerateTable_CGW/sytheticdata_generate_compare

# stage 1: populations (env synth_pop: isochrones + MIST grids in ~/.isochrones)
/astro/users/cgwill/miniforge3/envs/synth_pop/bin/python -m sythn_real_variability.run_generate populations

# stage 2: variability + TESS sampling + truth (env tess_gp)
/astro/users/cgwill/miniforge3/envs/tess_gp/bin/python -m sythn_real_variability.run_generate lcs
/astro/users/cgwill/miniforge3/envs/tess_gp/bin/python -m sythn_real_variability.checks
/astro/users/cgwill/miniforge3/envs/tess_gp/bin/python -m sythn_real_variability.make_figs
```

Useful flags: `--ages 8.0 9.0 --fehs 0.0 --instances 0 1`, `--n-systems 50000`, `--limit 5`,
`--source legacy` (use the Dropbox `isos/` tables when `isochrones` is unavailable),
`--no-red-noise`, `--brightness-mode physical`, `--fallback nearest`, `--method truncnorm`,
`--overwrite`.  Every option is a field of `RunConfig`; `--config results/run_config.json`
reproduces a previous run.

The `synth_pop` environment was built with
`conda create -n synth_pop python=3.10 numpy=1.26 pandas=1.5 scipy matplotlib astropy numba pytables; pip install isochrones`.
The MIST tarballs `isochrones` wants have moved: fetch
`https://mist.science/BC_tables/v1/UBVRIplus.txz` → `~/.isochrones/BC/mist/` and
`https://mist.science/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_full_isos.txz` → `~/.isochrones/mist/`
by hand before the first run (the package's own download URL returns 404).

## Outputs

| what | where |
|---|---|
| light curves | `<repo>/light_curves_synthetic_populations/LMC/hlsp_elk_tess_ffi_synthpop_a<age>_f<m|p><feh>_i<nn>_tess_v1_llc.fits` |
| populations (one row per stellar component) | `data/populations/*.fits` |
| per-cluster star parameters (bin, status, is_var, A, P, phase, source TIC) | `data/star_params/<name>.fits` |
| truth table (one row per cluster: all construction parameters, light budget, injected-signal summary, `true_*` statistics from `synth.ground_truth`) | `data/ground_truth_populations.pkl`, light version `results/ground_truth_summary.csv` |
| base table for the pipeline | `data/base_table_synthetic_populations.csv` |
| missing-light table | `results/missing_light_by_population.csv` |
| checks | `results/checks_report.md` |
| figures | `results/fig_*.png`, parameters in `results/construction_parameters.md` |

`data/` and the light curves are regenerable and git-ignored; `results/` is tracked.  In a
worktree, `data/` and the light curves are written to the **main** checkout
(`config.MAIN_REPO`) so they survive the worktree.

FITS primary headers carry `SYN_*` keywords (age, [Fe/H], instance, N systems, mass, distance,
integrated Tmag, template, red-noise parameters, missing-light fraction, seed); each sector HDU
carries `SYN_EXPD`, the exposure length used for averaging.

### Running the real pipeline on this set

In `generate_table.ipynb` replace the Wainer base table and the LC directory:

```python
base_table = pd.read_csv('sytheticdata_generate_compare/sythn_real_variability/data/base_table_synthetic_populations.csv')
LC_DIR = '../light_curves_synthetic_populations'
```

Every name resolves through `lc_lsp.get_lc_path` (asserted at generation time and in
`checks.py`).  Truth is one row per cluster; the pipeline emits one row per sector
combination, so join on `name` many-to-one as in `../synth/compare.py`.

### GP pipeline evaluation (`gp_eval_run.py`, `gp_eval_figs.py`, `results/gp_eval.png`)

```bash
cd GenerateTable_CGW/sytheticdata_generate_compare      # env tess_gp
python -m sythn_real_variability.gp_eval_run rows       # 1- and 3-sector row per cluster (~1 min)
python -m sythn_real_variability.gp_eval_run gp         # GP fit, prewhitened seeds (~9 min on 64 workers)
python -m sythn_real_variability.gp_eval_figs           # scoring, CSVs and figure
```

The full `expand_table` would make 116,123 rows (≈ 15 sectors per template), so the
evaluation fits two per cluster from the pipeline's own combination list: the first one-sector
and the first three-sector combination of the largest cadence group (1,343 rows).  A cluster's
signal is thousands of sinusoids, almost all far below the noise, so an injection counts as
**detectable** when its predicted periodogram power, (a sinc(E/P))² Σσ⁻² / 4, puts the expected
power above the Vaughan 1% threshold of the row (checked: measured / predicted median 0.97,
94% of predicted-detectable injections clear the threshold).  429 injections are detectable in
1,343 rows; most rows, and every row younger than log age 7.25, have none.

| delivered-list filter (all with cycles ≥ 1.6) | purity | completeness |
|---|---|---|
| σ/WN ≥ 0.2 (current default) | 0.70 | 0.70 |
| ΔlogL_seq ≥ 10, no amplitude cut | **0.83** | **0.82** |

The amplitude cut ranks components the wrong way on this set: components with no signal
behind them (red-noise absorbers at 3–10 d) have median σ/WN 0.69, matched ones 0.31.  On the
clean-sinusoid set the same evidence cut gives purity 0.997 and completeness 0.90 (amplitude
cut: 0.993, 0.94), see `gp_eval_filter_scan.csv`.  **Confound:** brightness falls with age in
this set (Spearman 0.93 with integrated Tmag), so the template white noise (0.64) and the
realised red-noise rms (0.43) correlate with age; any variability feature can pick up age
through noise alone.

## Figures

* `fig_lc_examples.png` — clean population signal (2-min grid) and the TESS-sampled noisy light
  curve for one sector of three example clusters, with every construction parameter in the box.
* `fig_lc_full_baseline.png` — one cluster across all sectors, clean vs noisy.
* `fig_population_hr.png` — the example populations on the grid, coloured by status
  (variable / quiet / unsampled / outside), with star counts and flux shares.
* `fig_missing_light.png` — missing-light fraction and integrated brightness vs age.
* `fig_variability_vs_age.png` — clean rms, variable flux fraction and dominant period vs age.
* `fig_empirical_grid.png` — the empirical grid (counts, variable fraction, A, P).
* `fig_variability_draws.png` — drawn amplitudes and periods against the catalogue.
* `fig_construction_parameters.png` — the run configuration as a table.

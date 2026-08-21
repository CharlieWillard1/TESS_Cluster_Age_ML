# Synthetic light-curve generation and pipeline validation

## Context

The pipeline's variability statistics, LSP peak finders, and GP fits have no ground truth:
every conclusion so far rests on internal consistency and on injection tests run outside
the pipeline. This builds a synthetic dataset with known generating parameters, run
through the *unmodified* pipeline, so recovery can be measured directly — especially
whether the GP recovers injected oscillators.

The `age` column carries no injected signal, so this validates the *estimators*, not the
science.

**Constraint: no edits to the real pipeline.** Everything lives in
`GenerateTable_CGW/sytheticdata_generate_compare/` (user's spelling — flag once, then
keep) and imports from `table_pipeline/` and `gp_pipeline/`. Where a pipeline function
does not fit the synthetic use directly, write a thin wrapper here rather than changing it.

## Settled design

| Decision | Value |
|---|---|
| Datasets generated | **One** — noisy, real sampling. No clean LC set on disk. |
| Ground truth | Computed **in memory at generation time** on the noise-free full-baseline signal; only the stats are saved, not the LC. |
| Oscillators per cluster | `N ~ Uniform{0..10}` (N=0 is a useful pure-red-noise null) |
| Amplitudes | log-uniform, `log10(A) ~ U(-4.5, -2.5)` fractional |
| Periods | `P ~ Uniform(0.1, 9)` d, linear, entirely in-band |
| Red noise | joint `(rn_alpha, rn_log10_N)` pairs **resampled from the real table**, preserving their +0.87 correlation |
| LSP ground truth | **skipped** (expensive). Period recovery is judged against the *injected* periods instead, which is the better truth anyway. |
| GP ground truth | the generative parameters — free |

Real distributions for reference: `rn_alpha` p5/med/p95 = 0.18/1.00/1.72;
`rn_log10_N` = 0.15/1.17/2.54; intrinsic RMS median 3.2e-4; fractional error median
2.7e-4 (so signal and noise are comparable, which is the interesting regime).

## Folder layout

```
sytheticdata_generate_compare/
├── PLAN.md                  # copy of this plan (step 1)
├── README.md                # written last
├── synth/
│   ├── generate.py          # LC synthesis + FITS writing
│   ├── ground_truth.py      # in-memory truth stats on the noise-free signal
│   └── compare.py           # matching + comparison metrics
├── 01_generate_synthetic.ipynb
├── 02_compare_to_truth.ipynb
└── data/
    └── ground_truth.pkl     # one row per cluster
```

## Phase 1 — generator (`synth/generate.py`)

### Mirror the `light_curves/` layout exactly so the pipeline works unchanged

`get_lc_path` (`lc_lsp.py:21`) is the only thing that resolves a cluster to a file, and it
is rigid:

```python
pattern = f'{lc_dir}/{loc}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits'
slug = basename.replace('hlsp_elk_tess_ffi_', '').replace('_tess_v1_llc.fits', '')
if _slug(slug) == _slug(cluster_name):   # lowercase, strip spaces/hyphens/brackets
```

So the synthetic root must reproduce `TESS_Cluster_Age_ML/light_curves/` exactly:

```
LC_DIR_synthetic/
├── MW/    hlsp_elk_tess_ffi_<slug>_tess_v1_llc.fits    (124 files)
├── LMC/   hlsp_elk_tess_ffi_<slug>_tess_v1_llc.fits    (118 files)
├── SMC/   hlsp_elk_tess_ffi_<slug>_tess_v1_llc.fits    (106 files)
└── readme                                              (copied verbatim)
```

- Per-origin subdirectory names must be exactly `MW` / `LMC` / `SMC` — they come straight
  from the `origin` column.
- Filenames must keep the `hlsp_elk_tess_ffi_` prefix and `_tess_v1_llc.fits` suffix or
  the glob misses them entirely.
- **Preserving each real file's basename is what guarantees all of this**, so the
  generator copies the source path's basename rather than reconstructing it from the
  cluster name. That also sidesteps slug round-tripping (e.g. `OGLE CL SMC 133` ->
  `ogle-cl-smc-133`), which is lossy in the reverse direction.
- The glob is non-recursive, so the stray nested directory inside the real `LMC/` is
  irrelevant and need not be reproduced.

With that layout, running the pipeline on the synthetic set requires changing only
`LC_DIR` in `generate_table.ipynb` — no pipeline edits, which is the whole point.

### Generation

**Approach: copy each real FITS and replace only the `flux` column.** This inherits
timestamps, `flux_err`, per-sector HDU structure, `SECTOR` keywords, and the primary
header for free, guaranteeing the synthetic directory is structurally identical to the
real one. All 348 clusters have LC files (verified), so coverage is complete.

Real file structure: `PrimaryHDU` (NAME, LOCATION, LOG_AGE, BTJD-BEG/END, ...) then one
`BinTableHDU` per sector with columns `time, flux, flux_err, mag, mag_err` (all `D`) and
a `SECTOR` header key. Spans reach ~1143 d with 30-min and 3.33-min cadences mixed.

Per cluster:
1. Build the continuous signal on a regular 30-min grid over `[BTJD-BEG, BTJD-END]`:
   `x(t) = 1 + red(t; alpha, rms) + sum_i A_i sin(2*pi*t/P_i + phi_i)`
2. **Red-noise amplitude calibration.** `rn_log10_N` is a fit output in psd-periodogram
   units, not a settable input. Since periodogram power scales as amplitude², calibrate
   exactly in one step: generate unit-RMS red noise at slope `alpha`, run the pipeline's
   `compute_lsp` + `_fit_red_noise_vaughan` on the longest sector, recover
   `log10_N_unit`, then scale the amplitude by `10**((log10_N_target - log10_N_unit)/2)`.
   One extra LSP per cluster; accept ~0.1-0.3 dex residual scatter across sector combos
   (real `rn_log10_N` varies by combo too, so exact matching is not meaningful).
3. Interpolate `x(t)` onto each sector's real timestamps.
4. Multiply by the real per-sector median flux to restore physical units, then add
   Gaussian noise drawn from the real `flux_err` at each point.
5. Write the FITS with the flux column replaced; leave `flux_err`, `mag`, `mag_err`,
   times and headers untouched.

Reuse: `get_lc_path` (`lc_lsp.py:21`), `compute_lsp` / `_fit_red_noise_vaughan`
(`lc_lsp.py`). Red noise via FFT: build `f^(-alpha/2)` amplitude envelope with random
phases, `irfft`, normalise to unit RMS, interpolate.

## Phase 2 — ground truth (`synth/ground_truth.py`)

Computed on the **noise-free, full-baseline** signal from step 1 (in memory, never
written to disk), one row per cluster.

Error-convention split, because the statistics disagree about what "no noise" means:

- **Moment statistics** — call `_compute_stats(signal, tiny_err)` from
  `summary_stats.py`. With `sigma_phot -> 0` this yields `excess_var = Var(signal)`
  exactly, plus `sigma_mad`, `mean_median_offset`, `gamma_p`. Do **not** report `snr`
  as truth: it is `excess_var/0`.
- **`von_neumann_ratio_gap_aware`** — needs no errors, call directly.
- **`J_stetson_gap_aware`** — is error-normalised and diverges as `err -> 0`. Call it with
  the **real** per-point errors, defining truth as "J if the same error bars were
  attached to a noise-free signal". Document the convention.
- **`g1`** — no pipeline function exists; compute `mu3/sigma^3` here (3 lines, matching
  `statistics_formulas.tex`).

Saved columns: `name, age, origin, n_osc, periods[], amplitudes[], phases[], rn_alpha,
rn_log10_N_target, rn_rms_realized`, plus the truth statistics above.

## Phase 3 — comparison (`synth/compare.py` + notebook 02)

Input: the pipeline table produced from the synthetic directory, joined on `name`.

1. **Summary statistics** — pipeline vs truth, per statistic: scatter with 1:1 line,
   median ratio, and scatter as a function of injected amplitude and of `snr`.
2. **Period recovery** — pipeline peak periods vs **injected** periods. Compute recovery
   fraction binned by injected period for each finder: raw `LSP_peak_period`, subtracted
   `LSP_peak_period_diff`, and the ratio finder recomputed here from `LSP_freq`/
   `LSP_power` (it was removed from the pipeline; recomputing it locally keeps the
   three-way comparison without touching the pipeline). **A flat curve is unbiased** —
   this is the pipeline-level version of the injection test already run standalone.
3. **GP recovery** — match `gp_periods` to injected periods greedily by fractional
   distance; report matched fraction, period error, amplitude error, spurious-component
   rate, and `gp_n_components` vs `n_osc`.

## Phase 4 — README

Covers what is generated, the parameter distributions, how to run both notebooks, which
paths to change in `generate_table.ipynb`, and — prominently — the caveats below.

## Known caveats to document, not fix

1. **Full-baseline vs per-sector variance.** Truth is measured over ~1143 d; the pipeline
   sees 27-d sectors with per-sector median normalisation. For `alpha = 1` red-noise
   variance grows as `ln(f_max*T)`, so truth carries roughly **1.6x** the variance a
   single sector can reach (more for steeper alpha, less for shallower). Expect a
   consistent offset in `excess_var`, not agreement. Restricting periods to 0.1-9 d means
   the *sinusoid* variance is fully recoverable — only the red-noise part is affected.
2. **Normalisation differs.** Truth is globally median-normalised; the pipeline normalises
   per sector, which additionally removes inter-sector offsets and suppresses low
   frequencies.
3. **Sigma clipping.** `expand_table` runs `nsigma=4`; truth is unclipped, so
   high-amplitude injected signals are partially clipped in the pipeline.
4. **One truth row per cluster, many pipeline rows per cluster** (sector combos). Join is
   one-to-many; aggregate or facet by `n_sectors` when comparing.

## Verification

Cap ad-hoc runs at ~2 min; background anything touching all 348 clusters.

1. **Structural identity**: a synthetic FITS opened alongside its real counterpart has
   identical HDU count, column names/formats, `SECTOR` keys, timestamps, and `flux_err`
   arrays — only `flux` differs.
2. **Layout resolves**: `get_lc_path(name, origin, LC_DIR_synthetic)` succeeds for all
   **348** clusters in `build_base_table` — the same count that resolves against the real
   directory. This is the check that the pipeline will run unmodified; run it before
   generating anything expensive.
2. **Round-trip on one cluster**: recover injected `alpha` within ~0.2 and `log10_N`
   within ~0.3 dex by running `compute_lsp` + `_fit_red_noise_vaughan` on the generated
   file.
3. **Zero-oscillator rows** (`n_osc=0`) must show no significant LSP peak more often than
   rows with large injected amplitudes — a basic sanity check on the false-positive rate.
4. **A single loud oscillator** (`A = 1e-2`, `N=1`) must be recovered by all three period
   finders to within 5%.
5. **Timing**: generation cost per cluster measured on 5 clusters and extrapolated to 348
   before the full run.
6. **Pipeline untouched**: `git diff` over `table_pipeline/` and `gp_pipeline/` is empty
   after the work.

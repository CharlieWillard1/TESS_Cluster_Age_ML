# Synthetic light curves for pipeline validation

Self-contained tooling to generate a synthetic TESS cluster light-curve set with **known
generating parameters**, run the real pipeline on it unmodified, and measure what the
pipeline recovers.

Nothing here modifies `table_pipeline/` or `gp_pipeline/` — everything imports from them.

> The directory name preserves the spelling used when it was requested
> (`sytheticdata_generate_compare`). Rename freely; nothing depends on it.

## What this validates, and what it does not

It validates the **estimators**: variability statistics, LSP peak finders, and GP
component recovery. No age–variability relation is injected, so `age` is uncorrelated with
everything by construction — this says nothing about the science, only about whether the
machinery measures what it claims to.

## Layout

```
sytheticdata_generate_compare/
├── PLAN.md                      design decisions and rationale
├── README.md
├── synth/
│   ├── generate.py              LC synthesis + FITS writing
│   ├── ground_truth.py          truth statistics on the noise-free signal
│   ├── run_generate.py          driver over all clusters
│   └── compare.py               join, stat comparison, period + GP recovery
├── 01_generate_synthetic.ipynb
├── 02_compare_to_truth.ipynb
└── data/ground_truth.pkl        one row per cluster
```

## How to run

**1. Generate** — run `01_generate_synthetic.ipynb`. About a minute for all 348 clusters.
Produces `light_curves_synthetic/` and `data/ground_truth.pkl`.

**2. Run the pipeline** — in `generate_table.ipynb`, change only:

```python
LC_DIR = '../light_curves_synthetic'
OUTPUT = './data/cluster_table_synthetic'
```

Everything else is unchanged. Step 2 (`expand_table`) is the expensive part, as usual.

### Subsetting is a pipeline-time choice, not a generation-time one

The synthetic set is always generated **in full** (all 348 clusters, ~1 min). To compare
a subset, filter `base_table` in `generate_table.ipynb` — there is a commented line for
it right after `build_base_table`:

```python
ORIGINS = ['MW']                                                      # 124 clusters
base_table = base_table[base_table['origin'].isin(ORIGINS)].reset_index(drop=True)
```

That filter sits *upstream of* `LC_DIR`, so the identical line cuts a real run and a
synthetic run the same way — which is exactly what makes the two comparable. Changing
your mind about the subset costs nothing, because no regeneration is involved.

**For a row-matched comparison, seed the row-selection cell.** `expand_table` is
deterministic (`seed=0, shuffle=False`) and the synthetic files carry the same sectors as
the real ones, so both runs produce the same sector combinations. But the selection cell
calls `.sample()`; without `random_state` it picks different combos each run and the two
tables stop being row-matched. The notebook now sets `RNG_SEED = 0` there.

**3. Compare** — run `02_compare_to_truth.ipynb`, pointing `PIPELINE_TABLE` at the table
from step 2.

## Generative model

On a regular 30-min grid spanning each cluster's full (multi-year) baseline:

```
x(t) = 1 + red(t; alpha, rms) + sum_i A_i sin(2 pi t / P_i + phi_i)
```

| parameter | distribution | rationale |
|---|---|---|
| `N` oscillators | `Uniform{0..10}` | `N=0` gives a pure-red-noise null class (~9% of clusters) |
| amplitude `A_i` | `log10 A ~ U(-5, -3)` | calibrated (below) |
| period `P_i` | `Uniform(0.1, 9)` d | linear and entirely in-band, so recovery flatness is a clean bias test |
| `(alpha, log10_N)` | resampled **jointly** from the real table | independent draws would destroy their +0.87 correlation |
| phase | `U(0, 2π)` | |

### Two things that are calibrated rather than assumed

**Red-noise amplitude.** `rn_log10_N` is a *fit output* in psd-periodogram units, not a
settable input, so the generator measures rather than guesses: it builds the full-baseline
realisation, interpolates it onto the longest sector, measures the `log10_N` that
realisation actually produces there, and rescales by `10**((target - measured)/2)` — exact
in one step, since periodogram power goes as amplitude².

The calibration must use the *same realisation and normalisation* as the written data. An
earlier version normalised a separate short realisation to unit RMS and under-scaled the
red noise by up to ~5× at steep α, because a 27-day chunk of a red process carries far
less variance than the full 1143-day baseline.

Verified on pure-red-noise realisations (`N=0`): median |Δα| = **0.091**, median
|Δlog10_N| = **0.035**.

**Amplitude range.** Set against the *total* injected variance, `sum(A_i²)/2`, not against
a single oscillator:

| range | median total sinusoid RMS |
|---|---|
| `U(-4.5, -2.5)` | 1.34e-3 — 4.2× too loud |
| **`U(-5, -3)`** | **4.22e-4** — matches the real 3.2e-4 |
| `U(-5.5, -3.5)` | 1.34e-4 — too quiet |

`U(-5,-3)` also brackets the detectability transition: 1e-5 is far below the median
fractional error (2.7e-4) and undetectable; 1e-3 is far above and obvious.

## File structure and why it matters

Each real FITS is copied and **only the `flux` column is replaced**. Timestamps,
`flux_err`, `mag`, `mag_err`, per-sector HDU structure, `SECTOR` keys and the primary
header are inherited unchanged.

`lc_lsp.get_lc_path` resolves clusters with a fixed glob:

```
{lc_dir}/{origin}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits
```

then slug-matches the middle portion (lowercase, spaces/hyphens/brackets stripped). So the
synthetic tree must keep the `MW`/`LMC`/`SMC` subdirectories and the exact filename
prefix/suffix. The generator preserves each source basename verbatim rather than rebuilding
it from the cluster name — the slug transform is not invertible (`OGLE CL SMC 133` →
`ogle-cl-smc-133`).

`01_generate_synthetic.ipynb` asserts that all 348 clusters resolve against the synthetic
tree before you spend hours on Step 2.

## Ground truth

Computed **in memory at generation time** on the noise-free, full-baseline signal. The
clean light curve is never written to disk; only the numbers survive. All statistics call
the pipeline's own functions so the definitions cannot drift — the sole exception is `g1`
(moment skewness), which the pipeline does not compute.

The statistics disagree about what "no noise" means, so two conventions are used:

| statistic | error convention | why |
|---|---|---|
| `excess_var`, `sigma_mad`, `mean_median_offset`, `gamma_p`, `g1` | `err → 0` | then `sigma_phot → 0` and `excess_var` reduces exactly to `Var(signal)` |
| `vn_ratio` | none needed | not error-normalised |
| `stetson_j` | **real** per-point errors | it is error-normalised and diverges as `err → 0`; truth means "J if these error bars were attached to a noise-free signal" |

`snr` has **no** ground-truth value — it would be `excess_var / 0`.

**No LSP ground truth is computed.** Period recovery is judged against the *injected*
periods, which is the better truth anyway and avoids an expensive second LSP pass.

## Expected disagreements — read before concluding the pipeline is broken

1. **`excess_var` will be offset, by design.** Truth spans ~1143 d; the pipeline sees 27-d
   sectors, normalised per sector. Red-noise variance grows as `ln(f_max·T)`, so truth
   carries roughly **1.6×** what one sector can reach at α = 1 (more at steeper α). The
   *sinusoid* contribution is fully recoverable — all injected periods are in band.
2. **Normalisation differs.** Truth is globally median-normalised; the pipeline normalises
   per sector, which also removes inter-sector offsets and suppresses low frequencies.
3. **Sigma clipping.** `expand_table` runs `nsigma=4`; truth is unclipped, so loud
   injected signals are partially clipped in the pipeline but not in truth.
4. **One truth row, many pipeline rows.** The join is one-to-many over sector combinations
   — facet or aggregate by `n_sectors`.
5. **Loud sinusoids corrupt the red-noise fit.** `_fit_red_noise_vaughan` fits *all* bins
   with no peak masking, so a forest of injected oscillators biases α steep and `log10_N`
   high. This is a real property of the pipeline, not a generator artefact, and it is worth
   measuring: it propagates into every statistic that divides or subtracts the continuum.
   Measured on a 20-cluster run: median `Δα = +0.19` with oscillators present, versus
   `0.09` on pure-red-noise realisations.
6. **The skew statistics have nothing to recover.** Sinusoids are *symmetric*, so the
   injected signal has true skewness ≈ 0 and `mean_median_offset`, `gamma_p` and `g1` are
   measuring noise on both sides of the comparison. In a 20-cluster run their Spearman
   correlation with truth was ~0.21, against ~0.94 for every amplitude statistic — that is
   the model's limitation, not the pipeline's. Testing the skew estimators properly needs
   an asymmetric injection (a non-sinusoidal spot profile, or flare-like transients);
   until then, ignore those three rows of the comparison table.

## Reading the comparison notebook

- **Summary statistics** — `ratio_median` is pipeline/truth; see caveat 1 before reading an
  offset as failure.
- **Period recovery** — recovery fraction binned by injected period for the raw finder, the
  continuum-subtracted finder, and the ratio finder (recomputed locally, since it was
  removed from the pipeline for being biased). **A flat curve means unbiased**: only then
  does the recovered period distribution reproduce the injected one. This is the
  pipeline-level version of the standalone injection test in `docs/peak_finders.tex`.
- **GP recovery** — `gp_periods` greedily matched to injected periods in fractional
  distance, each injection used at most once, so duplicate components count as spurious.

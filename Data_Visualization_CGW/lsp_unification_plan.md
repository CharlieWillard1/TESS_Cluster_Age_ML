# LSP Unification Plan

## Current state: two separate LSP implementations

### 1. `extract_LC_LSP.py` — `compute_lsp`
- Frequency grid built by `ls_bins()` using the **gap-aware effective baseline**
  (`T_effective` = sum of contiguous segment durations, inter-sector gaps excluded).
  `df = 1 / (alpha * T_effective)`, default `alpha=5`.
- `P_max` capped at `T/2` to avoid aliasing on short baselines.
- Always passes `flux_err` to `LombScargle`.
- `normalization='standard'` (explicit).
- Period range: `P_min=0.1`, `P_max=10.0` days (configurable).

### 2. `GP_Fit.py` — `residual_lsp_peak`
- Frequency grid built by `ls.autopower()` using astropy's internal logic,
  which uses the full **wall-clock baseline** (`t.max() - t.min()`).
  Oversampling expressed as `samples_per_peak=10`.
- `P_max` not capped.
- `flux_err` optional (always passed in practice, but not enforced).
- `normalization='standard'` (astropy default — matches, but implicit).
- Period range: whatever `min_period`/`max_period` is passed in at call time.

---

## Key differences that matter

| | `compute_lsp` | `residual_lsp_peak` |
|---|---|---|
| Baseline for df | gap-aware effective T | wall-clock span |
| Oversampling | alpha=5 | samples_per_peak=10 |
| P_max cap | yes (T/2) | no |
| flux_err | always | optional |
| Period range | 0.1–10 d | caller-defined |

For TESS data the wall-clock span can be significantly longer than the effective
baseline (multi-day inter-sector gaps), so the two grids resolve peaks differently.
LSP band powers and peak periods from the two codes are **not directly comparable**.

---

## Decisions for unification

- **Gap-aware baseline is correct.** The `ls_bins` / `T_effective` approach should
  be the standard going forward. Do not use `autopower()` for the science LSP.
- **`df = 1 / (alpha * T_effective)` is the right formula.** Keep it.
- **Default `alpha = 10`** (finer grid than the current `alpha=5`; matches
  `samples_per_peak=10` intent in the GP code).
- **Period range: always 0.04–10 days.** Hard-code this as the project standard
  rather than leaving it caller-defined. The 0.04 d lower bound (≈1 hr) covers
  the fast-rotator regime; 10 d covers the slow end of cluster sequences.
- **Always pass `flux_err`.** Unweighted LSP should not be used.
- **`normalization='standard'` always explicit.**

---

## Future goal: single LSP computed once, shared across pipeline

Currently the GP pipeline recomputes the LSP internally (for period seed finding,
white-noise bootstrap, residual LSP, and verbose plots). The right design is:

1. Compute **one** LSP per object before `iterative_sho_gp_fit` is called,
   using the unified settings above. Store it alongside the light curve.
2. Pass the pre-computed `(freq, power)` arrays into the GP pipeline.
   - Period-seed finding (`find_all_significant_peaks`) reads from the stored LSP.
   - White-noise bootstrap uses the stored frequency grid.
   - **Residual LSPs** (post-fit diagnostics) are legitimately new — they must be
     recomputed on the GP residuals, not the stored data LSP.
   - **Afterfit plots** (`visualize_gp_afterfit.py`) should use the stored LSP,
     not recompute it.
   - **Verbose=1 initial-periodogram plots** (`plot_initial_lsp`) may need to
     call the LSP function if no stored spectrum is passed in; if the stored one
     is passed, use it directly. Verbose=2 fit-summary plots are fine either way.

No full implementation plan needed yet — just flag that `residual_lsp_peak`
inside `GP_Fit.py` should eventually be replaced by a pass-in of the pre-computed
spectrum, and `add_lsps` in `extract_LC_LSP.py` (or a wrapper) becomes the
single canonical LSP entry point for the whole pipeline.

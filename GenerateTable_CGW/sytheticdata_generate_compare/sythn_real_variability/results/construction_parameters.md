# Construction parameters

Written by `sythn_real_variability.make_figs` on 2026-11-04 03:14; 675 clusters in the truth table.

## grid

| parameter | value |
|---|---|
| `teff_min` | 2700.0 |
| `teff_max` | 12500.0 |
| `teff_step` | 150.0 |
| `logl_min` | -3.2 |
| `logl_max` | 6.5 |
| `logl_step` | 0.15 |
| `teff_catalog_max` | 15000.0 |

## population

| parameter | value |
|---|---|
| `ages` | 6.5 .. 10.0 (15 values) |
| `fehs` | (-1.0, -0.5, 0.0) |
| `n_instances` | 15 |
| `n_systems` | 10000 |
| `imf` | salpeter |
| `imf_mass_bounds` | (0.1, 300.0) |
| `binary_fraction` | 0.4 |
| `mass_ratio_gamma` | 0.3 |
| `feh_sigma` | 0.01 |
| `distance_pc` | 49590.0 |
| `A_V` | 0.0 |
| `fresh_population_per_instance` | True |
| `source` | auto |

## variability

| parameter | value |
|---|---|
| `method` | resample |
| `min_bin_stars` | 5 |
| `fallback` | constant |
| `include_second_component` | True |
| `amp_max_ppm` | 500000.0 |
| `amp_min_ppm` | 1.0 |
| `period_min_d` | 0.005 |
| `period_max_d` | 15.0 |

## noise

| parameter | value |
|---|---|
| `template_origin` | LMC |
| `template_select` | brightness |
| `template_k_nearest` | 3 |
| `brightness_mode` | template |
| `add_red_noise` | True |
| `rn_origin` | LMC |
| `rn_rms_max` | 0.02 |
| `grid_cadence_min` | 1.0 |
| `truth_cadence_min` | 30.0 |
| `lsp_P_min` | 0.08333333333333333 |
| `lsp_P_max` | 10.0 |
| `lsp_alpha` | 10 |

## constants

| parameter | value |
|---|---|
| `TESS_ZP` | 20.44 |
| `M_BOL_SUN` | 4.74 |
| `LMC_DISTANCE_PC` | 49590.0 |
| `LMC_DISTANCE_MODULUS` | 18.47697054145555 |
| `A_TESS_OVER_A_V` | 0.61 |

## run

| `seed` | 0 |
| `name_prefix` | synthpop |

## realised set

| quantity | value |
|---|---|
| clusters | 675 |
| log ages | 6.5 .. 10.0 (15 values) |
| [Fe/H] | [-1.0, -0.5, 0.0] |
| instances per population | 15-15 |
| population source | ['isochrones'] |
| BC source | ['mist_grid'] |
| integrated Tmag | 7.42-16.51 |
| variable stars per cluster | 701-11517 |
| clean rms [ppt] | 0.000-8.861 |
| missing light fraction | 0.0%-99.8% |

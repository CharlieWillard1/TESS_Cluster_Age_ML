import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Function 4 — add LSP scalar summaries to master table
# ---------------------------------------------------------------------------

def add_invariant_LSP_stats(table):
    """
    Compute scalar LSP summary statistics for every row from the frequency and
    power arrays stored by ``add_lsps``.

    Parameters
    ----------
    table : pd.DataFrame
        Must have columns ``LSP_freq`` and ``LSP_power`` (added by ``add_lsps``).

    Returns
    -------
    table : pd.DataFrame
        Same object, with new columns added in-place.

    New columns
    -----------
    LSP_max_power           float64   peak LSP power
    LSP_freq_at_max_power   float64   frequency of peak (1/day)
    LSP_period_at_max_power float64   period of peak (days)
    LSP_SumPow_10_7         float64   sum of power in 7–10 day period band
    LSP_SumPow_7_4          float64   sum of power in 4–7 day period band
    LSP_SumPow_4_1          float64   sum of power in 1–4 day period band
    LSP_SumPow_1_p5         float64   sum of power in 0.5–1 day period band
    """
    n_rows = len(table)
    print(f"[add_invariant_LSP_stats] processing {n_rows} rows")

    idx = table.index
    max_power           = pd.Series(np.nan, index=idx, dtype=float)
    freq_at_max_power   = pd.Series(np.nan, index=idx, dtype=float)
    period_at_max_power = pd.Series(np.nan, index=idx, dtype=float)
    sum_10_7 = pd.Series(np.nan, index=idx, dtype=float)
    sum_7_4  = pd.Series(np.nan, index=idx, dtype=float)
    sum_4_1  = pd.Series(np.nan, index=idx, dtype=float)
    sum_1_p5 = pd.Series(np.nan, index=idx, dtype=float)

    n_ok, n_skipped, n_failed = 0, 0, 0

    for label, row in table.iterrows():
        freqs = row.get('LSP_freq')
        power = row.get('LSP_power')

        if freqs is None or power is None:
            n_skipped += 1
            continue

        try:
            peak_idx = int(np.argmax(power))
            freq_peak = float(freqs[peak_idx])

            with np.errstate(divide='ignore'):
                periods = 1.0 / freqs

            def _band(P_lo, P_hi):
                mask = (periods >= P_lo) & (periods <= P_hi)
                return float(power[mask].sum())

            max_power[label]           = float(power[peak_idx])
            freq_at_max_power[label]   = freq_peak
            period_at_max_power[label] = 1.0 / freq_peak
            sum_10_7[label] = _band(7.0,  10.0)
            sum_7_4[label]  = _band(4.0,   7.0)
            sum_4_1[label]  = _band(1.0,   4.0)
            sum_1_p5[label] = _band(0.5,   1.0)
            n_ok += 1

        except Exception as e:
            name_str = row['name'].decode() if isinstance(row['name'], bytes) else row['name']
            print(f"  WARNING [{name_str}] row {label}: {e}")
            n_failed += 1

    print(f"[add_invariant_LSP_stats] done  |  ok={n_ok}  skipped={n_skipped}  failed={n_failed}")

    table['LSP_max_power']           = max_power
    table['LSP_freq_at_max_power']   = freq_at_max_power
    table['LSP_period_at_max_power'] = period_at_max_power
    table['LSP_SumPow_10_7']         = sum_10_7
    table['LSP_SumPow_7_4']          = sum_7_4
    table['LSP_SumPow_4_1']          = sum_4_1
    table['LSP_SumPow_1_p5']         = sum_1_p5
    return table

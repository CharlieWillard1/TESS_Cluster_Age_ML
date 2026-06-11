import ast
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


_DROP_COLS = ['gp_result', 'GP_PSD_binned', 'GP_PSD_bin_ratios']
_ARRAY_COLS = ['LC_t', 'LC_flux', 'LC_flux_err', 'LSP_freq', 'LSP_power', 'GP_freq', 'GP_PSD']


def save_as_fits(df, path):
    """Save cluster table DataFrame to a FITS binary table."""
    df = df.copy()

    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # sectors list → string (lists are object dtype, not FITS-compatible)
    if 'sectors' in df.columns:
        df['sectors'] = df['sectors'].apply(str)

    # Coerce None/object array cells to numpy float64 (None → empty array)
    _vla_cols = _ARRAY_COLS + [
        'gp_periods', 'gp_sho_sigmas', 'gp_sho_omegas', 'gp_sho_Qs',
        'gp_initial_lsp_peak_periods',
    ]
    for col in _vla_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda x: np.array([], dtype=np.float64) if x is None
                      else np.asarray(x, dtype=np.float64).ravel()
        )

    Table.from_pandas(df).write(path, format='fits', overwrite=True)
    print(f"[fits_io] saved {len(df)} rows → {path}")


def load_from_fits(path):
    """Load cluster table from FITS, returning a pandas DataFrame.

    Uses astropy.io.fits directly because Table.to_pandas() silently NaN-ifies
    variable-length array (PD) columns.
    """
    _fmt_dtype = {'D': np.float64, 'E': np.float32, 'K': np.int64,
                  'J': np.int32, 'I': np.int16, 'L': np.bool_}

    with fits.open(path) as hdul:
        hdu  = hdul[1]
        data = hdu.data
        n    = len(data)

        rows = {}
        for col in hdu.columns:
            name = col.name
            fmt  = col.format
            raw  = data[name]

            if fmt.startswith('P'):
                rows[name] = [np.array(raw[i], dtype=np.float64) for i in range(n)]
            elif 'A' in fmt:
                rows[name] = [
                    v.decode('utf-8').strip() if isinstance(v, bytes) else str(v).strip()
                    for v in raw
                ]
            else:
                dtype = _fmt_dtype.get(fmt.lstrip('0123456789'), np.float64)
                rows[name] = np.array(raw, dtype=dtype)

    df = pd.DataFrame(rows)

    if 'sectors' in df.columns:
        df['sectors'] = df['sectors'].apply(ast.literal_eval)

    print(f"[fits_io] loaded {len(df)} rows from {path}")
    return df

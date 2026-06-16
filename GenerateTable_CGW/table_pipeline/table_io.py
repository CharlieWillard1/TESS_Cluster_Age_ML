import ast
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


_DROP_COLS = ['gp_result', 'GP_PSD_binned', 'GP_PSD_bin_ratios']
_ARRAY_COLS = ['LC_t', 'LC_flux', 'LC_flux_err', 'LSP_freq', 'LSP_power', 'GP_freq', 'GP_PSD']

# FITS 'P' (32-bit) variable-length-array descriptors address at most ~4 GiB of
# heap. Once the *cumulative* heap (summed across all VLA columns, in column
# write order) crosses that boundary, rows whose data lands past the wrap point
# are silently corrupted on read — no error, just wrong numbers. Warn well
# before the real limit so there's room to switch to save_as_pickle.
_HEAP_WARN_BYTES  = 1_500_000_000   # 1.5 GB
_HEAP_ERROR_BYTES = 3_500_000_000   # 3.5 GB


def _estimate_heap_bytes(df, cols):
    """Cumulative heap size (bytes) for the given VLA columns, in column order."""
    total = 0
    per_col = {}
    for col in cols:
        if col not in df.columns:
            continue
        n_elem = df[col].apply(lambda x: 0 if x is None else len(x)).sum()
        nbytes = int(n_elem) * 8
        total += nbytes
        per_col[col] = total
    return total, per_col


def save_as_fits(df, path):
    """Save cluster table DataFrame to a FITS binary table.

    Raises if the cumulative variable-length-array heap is large enough to risk
    silent corruption from the 32-bit 'P' descriptor's heap-offset overflow
    (observed in practice around ~4 GiB total heap). Use save_as_pickle for
    tables this large — pickle has no such limit.
    """
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

    total_heap, per_col = _estimate_heap_bytes(df, _vla_cols)
    if total_heap > _HEAP_ERROR_BYTES:
        raise ValueError(
            f"[table_io] Refusing to save: estimated VLA heap is "
            f"{total_heap/1e9:.2f} GB, which risks silent 32-bit heap-offset "
            f"overflow in FITS 'P' columns (corrupted array data on read, no "
            f"error raised). Use save_as_pickle(df, path) instead — see "
            f"per-column cumulative sizes: "
            f"{ {k: f'{v/1e9:.2f} GB' for k, v in per_col.items()} }"
        )
    if total_heap > _HEAP_WARN_BYTES:
        import warnings
        warnings.warn(
            f"[table_io] VLA heap is {total_heap/1e9:.2f} GB — approaching the "
            f"~4 GB range where FITS 'P' columns can silently corrupt data. "
            f"Consider save_as_pickle(df, path) for this table.",
            RuntimeWarning, stacklevel=2,
        )

    Table.from_pandas(df).write(path, format='fits', overwrite=True)
    print(f"[table_io] saved {len(df)} rows → {path}")


def save_as_pickle(df, path):
    """Save cluster table DataFrame to pickle.

    Preferred over save_as_fits for tables carrying per-row LC/LSP/GP arrays —
    pickle has no heap-size limit, so none of the FITS-side array coercion is
    needed. Still drops _DROP_COLS (gp_result, GP_PSD_binned, GP_PSD_bin_ratios)
    just like save_as_fits: GPFitResult holds JAX objects that don't reliably
    unpickle, so it's never safe to round-trip through pickle either.
    """
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    df.to_pickle(path)
    print(f"[table_io] saved {len(df)} rows → {path}")


def load_from_pickle(path):
    """Load cluster table previously saved with save_as_pickle."""
    df = pd.read_pickle(path)
    print(f"[table_io] loaded {len(df)} rows from {path}")
    return df


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

    print(f"[table_io] loaded {len(df)} rows from {path}")
    return df

def save_as_hdf5(df, path):
    """Save cluster table DataFrame to HDF5.
    No VLA heap-size limit unlike FITS 'P' columns. Array columns are detected
    dynamically (any cell holding a ndarray or list) and stored as vlen float64
    datasets. Remaining object-dtype columns are stored as variable-length HDF5
    strings via an explicit string_dtype() to avoid h5py dtype('O') errors.
    """
    import h5py
    df = df.copy()
    df = df.drop(columns=[c for c in _DROP_COLS if c in df.columns])
    if 'sectors' in df.columns:
        df['sectors'] = df['sectors'].apply(str)

    vlen_float = h5py.vlen_dtype(np.float64)
    str_dtype = h5py.string_dtype()

    with h5py.File(path, 'w', track_order=True) as f:
        for col in df.columns:
            vals = df[col].values
            first = next((v for v in vals if v is not None), None)

            if isinstance(first, (np.ndarray, list)):
                # Per-row float array column (variable or fixed length)
                ds = f.create_dataset(col, shape=(len(df),), dtype=vlen_float)
                for i, v in enumerate(vals):
                    ds[i] = (np.array([], dtype=np.float64) if v is None
                             else np.asarray(v, dtype=np.float64).ravel())
            elif vals.dtype.kind == 'O':
                # Remaining object-dtype (strings, etc.) — explicit HDF5 string type
                f.create_dataset(col, data=[str(v) if v is not None else '' for v in vals],
                                 dtype=str_dtype)
            else:
                f.create_dataset(col, data=vals)

    print(f"[table_io] saved {len(df)} rows → {path}")
    return


def load_from_hdf5(path):
    """Load cluster table from HDF5, returning a pandas DataFrame."""
    import h5py
    rows = {}
    with h5py.File(path, 'r', track_order=True) as f:
        for key in f.keys():
            ds = f[key]
            if h5py.check_string_dtype(ds.dtype):
                # vlen string column — decode bytes to str
                rows[key] = [v.decode('utf-8') if isinstance(v, bytes) else str(v)
                             for v in ds[:]]
            elif h5py.check_vlen_dtype(ds.dtype):
                # vlen float array column
                rows[key] = [np.array(ds[i], dtype=np.float64) for i in range(len(ds))]
            else:
                raw = ds[:]
                if raw.dtype.kind in ('S', 'O'):
                    rows[key] = [v.decode('utf-8') if isinstance(v, bytes) else str(v)
                                 for v in raw]
                else:
                    rows[key] = raw
    df = pd.DataFrame(rows)
    if 'sectors' in df.columns:
        df['sectors'] = df['sectors'].apply(ast.literal_eval)
    print(f"[table_io] loaded {len(df)} rows from {path}")
    return df


import os
import pandas as pd


# Byte-range specs (1-indexed, inclusive) → converted to 0-based Python slices inside _parse_file
_WAINER_SPECS = {
    'MW':  {'name': (1, 17), 'age': (41, 46)},
    'SMC': {'name': (1, 17), 'age': (42, 46)},
    'LMC': {'name': (1, 10), 'age': (36, 39)},
}

_WAINER_FILENAMES = {
    'MW':  'Wainer2023_Table1_MW.txt',
    'SMC': 'Wainer2023_Table2_SMC.txt',
    'LMC': 'Wainer2023_Table3_LMC.txt',
}

_HEADER_ROWS = 28


def _parse_file(filepath, origin):
    """Parse name and logAge from a single Wainer CDS fixed-width file."""
    spec = _WAINER_SPECS[origin]
    # Convert 1-indexed inclusive byte ranges to 0-based pandas colspecs (start inclusive, end exclusive)
    name_cs = (spec['name'][0] - 1, spec['name'][1])
    age_cs  = (spec['age'][0]  - 1, spec['age'][1])

    df = pd.read_fwf(
        filepath,
        skiprows=_HEADER_ROWS,
        header=None,
        colspecs=[name_cs, age_cs],
        names=['name', 'age'],
        dtype={'name': str, 'age': float},
    )
    df['name'] = df['name'].str.strip()
    df = df[df['name'].notna() & (df['name'] != '')].copy()
    df['origin'] = origin
    return df[['name', 'age', 'origin']].reset_index(drop=True)


def build_base_table(data_dir):
    """
    Read the three Wainer 2023 CDS tables and return a pandas DataFrame with one
    row per cluster containing name, age (log10 yr), and origin (MW/SMC/LMC).

    Parameters
    ----------
    data_dir : str
        Directory containing the three Wainer table files.

    Returns
    -------
    pd.DataFrame
        Columns: name (str), age (float, log10 yr), origin (str: 'MW'/'SMC'/'LMC').
    """
    parts = []
    for origin, filename in _WAINER_FILENAMES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Wainer table not found: {filepath}")
        df = _parse_file(filepath, origin)
        print(f"  {origin}: {len(df)} clusters from {filename}")
        parts.append(df)

    result = pd.concat(parts, ignore_index=True)
    print(f"[build_base_table] total: {len(result)} clusters")
    return result

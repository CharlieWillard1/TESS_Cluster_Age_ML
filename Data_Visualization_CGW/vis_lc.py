import os
import glob as glob_module
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as astropy_fits

LC_BASE = '/astro/users/cgwill/TESS_Cluster_Age_ML/light_curves'


def _slug(name):
    """Canonical key for matching: lowercase, strip all spaces and hyphens."""
    s = name.decode() if isinstance(name, bytes) else name
    return s.lower().replace(' ', '').replace('-', '')


def get_lc_path(master_table, name):
    """
    Given a cluster name (str or bytes), return the path to its LC .fits file.
    Uses the LOC column from master_table to determine the subdirectory.
    Matches by stripping spaces/hyphens from both the name and filename slug,
    to handle complex multi-word catalog names (e.g. 'OGLE CL SMC 133' -> ogle-cl-smc-133).
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    row = master_table[master_table['name'] == name].iloc[0]
    loc = row['LOC'].decode() if isinstance(row['LOC'], bytes) else row['LOC']

    target = _slug(name_str)
    pattern = f'{LC_BASE}/{loc}/hlsp_elk_tess_ffi_*_tess_v1_llc.fits'
    for fpath in glob_module.glob(pattern):
        basename = os.path.basename(fpath)
        slug = basename.replace('hlsp_elk_tess_ffi_', '').replace('_tess_v1_llc.fits', '')
        if _slug(slug) == target:
            return fpath

    raise FileNotFoundError(f"No LC file found for '{name_str}' in {LC_BASE}/{loc}/")


def add_std_norm(master_table, cadence_bin_min=30.0):
    """
    Adds 'std_norm' column to master_table in-place.
    For each row, loads the light curve HDU(s) specified by the 'sectors' column,
    stitches if multiple sectors, bins to cadence_bin_min minutes, then
    returns sigma_int = sqrt(max(0, sigma_obs^2 - mean(sigma_phot^2))).
    Returns the modified master_table.
    """
    cadence_bin_days = cadence_bin_min / (24.0 * 60.0)
    std_norm_vals = np.full(len(master_table), np.nan)

    # Group by cluster name to open each FITS file once
    for name, group in master_table.groupby('name'):
        try:
            path = get_lc_path(master_table, name)
        except FileNotFoundError:
            continue  # leave NaN for all rows of this cluster

        try:
            with astropy_fits.open(path) as hdul:
                for idx, row in group.iterrows():
                    try:
                        sectors = row['sectors']
                        all_time, all_flux, all_ferr = [], [], []

                        for s in sectors:
                            hdu_idx = int(s) + 1
                            data = hdul[hdu_idx].data

                            time = np.asarray(data['time'], dtype=float)
                            flux = np.asarray(data['flux'], dtype=float)
                            ferr = np.asarray(data['flux_err'], dtype=float)

                            mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(ferr)
                            time, flux, ferr = time[mask], flux[mask], ferr[mask]

                            # Normalize each sector individually to remove inter-sector offsets
                            med = np.nanmedian(flux)
                            if med == 0 or not np.isfinite(med):
                                continue
                            flux = flux / med
                            ferr = ferr / med

                            all_time.append(time)
                            all_flux.append(flux)
                            all_ferr.append(ferr)

                        if not all_time:
                            continue

                        time = np.concatenate(all_time)
                        flux = np.concatenate(all_flux)
                        ferr = np.concatenate(all_ferr)

                        sort_idx = np.argsort(time)
                        time = time[sort_idx]
                        flux = flux[sort_idx]
                        ferr = ferr[sort_idx]

                        # Step 2 done per-sector above; Step 3: bin to common cadence if needed
                        native_cadence = np.nanmedian(np.diff(time)) * 24.0 * 60.0
                        if native_cadence < cadence_bin_min:
                            t_min, t_max = time[0], time[-1]
                            bins = np.arange(t_min, t_max + cadence_bin_days, cadence_bin_days)
                            bin_idx = np.digitize(time, bins)

                            flux_binned, ferr_binned = [], []
                            for b in range(1, len(bins) + 1):
                                mask = bin_idx == b
                                n = mask.sum()
                                if n == 0:
                                    continue
                                flux_binned.append(np.nanmedian(flux[mask]))
                                ferr_binned.append(
                                    np.sqrt(np.nanmean(ferr[mask] ** 2) / n)
                                )
                            flux_binned = np.array(flux_binned)
                            ferr_binned = np.array(ferr_binned)
                        else:
                            flux_binned = flux
                            ferr_binned = ferr

                        if len(flux_binned) < 2:
                            continue

                        # Step 4: noise subtraction
                        sigma_obs2 = np.nanvar(flux_binned)
                        sigma_phot2 = np.nanmean(ferr_binned ** 2)
                        sigma_int = np.sqrt(max(0.0, sigma_obs2 - sigma_phot2))

                        std_norm_vals[idx] = sigma_int

                    except Exception:
                        pass  # leave NaN for this row

        except Exception:
            pass  # leave NaN for all rows of this cluster

    master_table['std_norm'] = std_norm_vals
    return master_table


def plot_cluster_lcs(master_table, name):
    """
    Plot all LC sectors for a given cluster, one subplot per sector.
    Each subplot is labelled with the sector index and cadence.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str or bytes
        Cluster name matching the 'name' column in master_table.
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    path = get_lc_path(master_table, name)

    single_sector_rows = master_table[
        (master_table['name'] == name) & (master_table['n_sectors'] == 1)
    ]
    cadence_by_hdu = {}
    for _, row in single_sector_rows.iterrows():
        sectors = row['sectors']
        if hasattr(sectors, '__len__') and len(sectors) == 1:
            cadence_by_hdu[int(sectors[0])] = row['cadence']

    with astropy_fits.open(path) as hdul:
        n_sectors = len(hdul) - 1
        print(f"{name_str}  —  {n_sectors} HDU(s)  —  {path}")
        for j, hdu in enumerate(hdul):
            h = hdu.header
            print(f"  HDU {j}: {hdu.name:<12}  SECTOR={h.get('SECTOR','—')!s:<4}  "
                  f"TIMEDEL={h.get('TIMEDEL','—')!s:<10}  "
                  f"nrows={len(hdu.data) if hdu.data is not None else 0}")

        fig, axes = plt.subplots(n_sectors, 1, figsize=(12, 3 * n_sectors))
        if n_sectors == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            hdu_idx = i + 1
            hdu = hdul[hdu_idx]
            tess_sector = hdu.header.get('SECTOR', '?')
            cadence = cadence_by_hdu.get(i, '?')
            data = hdu.data
            if cadence == '?':
                diffs = np.diff(data['time'])
                cadence = round(float(np.median(diffs[np.isfinite(diffs)])) * 24 * 60, 1)
            ax.plot(data['time'], data['flux'], lw=0.5, color='steelblue')
            ax.set_ylabel('flux')
            ax.set_title(f'Sector {hdu_idx}  |  TESS sector {tess_sector}  |  cadence = {cadence} min',
                         fontsize=9)

        axes[-1].set_xlabel('time (days)')
        fig.suptitle(name_str, fontsize=13, y=1.001)
        fig.tight_layout()
        plt.show()

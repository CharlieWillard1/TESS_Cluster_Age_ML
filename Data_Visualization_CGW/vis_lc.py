import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as astropy_fits

from invariant_summary_stats import LC_BASE, _slug, get_lc_path


def plot_cluster_lcs(master_table, name, x_zoom=None, plot_errors=False):
    """
    Plot all LC sectors for a given cluster, one subplot per sector.
    Each subplot is labelled with the sector index and cadence.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str or bytes
        Cluster name matching the 'name' column in master_table.
    x_zoom : float or None
        If given, zoom each subplot to the first ``x_zoom`` fraction of its
        time range (e.g. 0.5 shows the first half of each sector).
    plot_errors : bool
        If True, plot flux uncertainties as a shaded band around the LC.
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
            t = data['time']
            f = data['flux']
            ax.plot(t, f/np.median(f), lw=0.5, color='steelblue')
            if plot_errors:
                err = data['flux_err']
                ax.fill_between(t, f/np.median(f) - err/np.median(f), f/np.median(f) + err/np.median(f), color='steelblue', alpha=0.3, linewidth=0)
            if x_zoom is not None:
                t_finite = t[np.isfinite(t)]
                t_min, t_max = t_finite.min(), t_finite.max()
                ax.set_xlim(t_min, t_min + x_zoom * (t_max - t_min))
            ax.set_ylabel('flux')
            ax.set_title(f'Sector {hdu_idx}  |  TESS sector {tess_sector}  |  cadence = {cadence} min',
                         fontsize=9)

        axes[-1].set_xlabel('time (days)')
        fig.suptitle(name_str, fontsize=13, y=1.001)
        fig.tight_layout()
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


def _apply_x_lim(x, pg, x_lim):
    """Slice x and pg to the specified range (index or frequency)."""
    if x_lim is None:
        return x, pg
    mask = (x >= x_lim[0]) & (x <= x_lim[1])
    return x[mask], pg[mask]


def plot_cluster_lsp(table, name, max_num_pgs=10,
                     yscale='log', x_lim=None, y_lim=None):
    """
    Plot LSP periodograms for a cluster, grouped by n_sectors.

    Reads the ``LSP_freq``, ``LSP_power``, and ``LSP_FAP`` columns produced by
    ``expand_table``. Because each row carries its own frequency grid (grid
    spacing depends on the row's time baseline), every line is plotted against
    its own x-axis rather than a shared bin index.

    A secondary x-axis showing period in days is drawn at the top of each
    subplot.

    Parameters
    ----------
    table : pd.DataFrame
        Must contain ``LSP_freq``, ``LSP_power``, ``LSP_FAP``, ``cadence`` columns.
    name : str, bytes, or None
        Cluster name matching the ``name`` column. If None, a random cluster
        is chosen.
    max_num_pgs : int
        Maximum number of periodogram lines per subplot.
    yscale : str
        Y-axis scale: ``'log'`` (default) or ``'linear'``.
    x_lim : tuple of (min, max) or None
        Frequency range (1/day) to display.
    y_lim : tuple of (min, max) or None
        Power range applied to all subplots.
    """
    if name is None:
        name = np.random.choice(table['name'].unique())
        print(f"Randomly selected cluster: '{name}'")

    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = table[table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name_str}'.")
        return

    if 'LSP_freq' not in table.columns:
        raise KeyError("LSP_freq column not found. Run expand_table first.")

    ns_vals = sorted(cluster_data['n_sectors'].unique())
    n_subplots = len(ns_vals)

    fig, axes = plt.subplots(n_subplots, 1, figsize=(16, 6 * n_subplots))
    if n_subplots == 1:
        axes = [axes]

    for ax, ns in zip(axes, ns_vals):
        rows = cluster_data[cluster_data['n_sectors'] == ns]
        n_total = len(rows)

        sampled = rows.sample(min(max_num_pgs, n_total), random_state=42)
        colors = plt.cm.tab10(np.linspace(0, 1, len(sampled)))

        for (_, row), color in zip(sampled.iterrows(), colors):
            freqs = np.asarray(row['LSP_freq'])
            power = np.asarray(row['LSP_power'])
            fap   = row['LSP_FAP']

            if x_lim is not None:
                mask = (freqs >= x_lim[0]) & (freqs <= x_lim[1])
                freqs, power = freqs[mask], power[mask]

            df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else float('nan')
            sectors_label = list(row['sectors'])
            cad = row.get('cadence', '?')
            label = f'{sectors_label} | {cad} min | df={df:.2e} day⁻¹'
            ax.plot(freqs, power, lw=1.5, alpha=0.7, color=color, label=label)

            raw_noise = row.get('LSP_noise_floor') if hasattr(row, 'get') else None
            if raw_noise is not None:
                nf_freqs = np.asarray(row['LSP_freq'])
                nf = np.asarray(raw_noise)
                if x_lim is not None:
                    nf_mask = (nf_freqs >= x_lim[0]) & (nf_freqs <= x_lim[1])
                    nf_freqs, nf = nf_freqs[nf_mask], nf[nf_mask]
                ax.plot(nf_freqs, nf, lw=1.0, alpha=0.6, color=color, linestyle='--')

            if np.isfinite(fap):
                ax.axhline(fap, color=color, lw=1.0, linestyle=':', alpha=0.6)

        ax.set_yscale(yscale)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)

        ax.set_ylabel('power', fontsize=14)
        ax.tick_params(labelsize=12)
        shown = len(sampled)
        ax.set_title(f'n_sectors={ns}  ({n_total} total, showing {shown})',
                     fontsize=14)
        ax.legend(fontsize=11, ncol=2, loc='upper right')

        # Secondary x-axis: period in days (1/frequency)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xscale('linear')

        f_lo, f_hi = ax.get_xlim()
        f_lo = max(f_lo, 1e-6)
        candidate_periods = np.array([0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 13.5])
        candidate_freqs   = 1.0 / candidate_periods
        in_range = candidate_freqs[(candidate_freqs >= f_lo) & (candidate_freqs <= f_hi)]
        if len(in_range):
            ax_top.set_xticks(in_range)
            ax_top.set_xticklabels(
                [f'{1/f:.4g}' for f in in_range], fontsize=10
            )
        ax_top.set_xlabel('period (days)', fontsize=12)

    axes[-1].set_xlabel('frequency (1/day)', fontsize=14)
    fig.suptitle(f'{name_str} — LSP', fontsize=18)
    fig.tight_layout()
    plt.show()


def plot_cluster_lsp_specific(table, name, sectors_list,
                               yscale='log', x_lim=None, y_lim=None):
    """
    Plot specific sector combinations for a cluster on a single axes.

    Parameters
    ----------
    table : pd.DataFrame
    name : str or bytes
        Cluster name.
    sectors_list : list of lists
        Each entry is a sector combination to look up and plot,
        e.g. [[0], [0, 1], [0, 1, 2]].
    yscale : str
        ``'log'`` (default) or ``'linear'``.
    x_lim : tuple or None
        Frequency range (1/day).
    y_lim : tuple or None
        Power range.
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = table[table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name_str}'.")
        return

    fig, ax = plt.subplots(figsize=(16, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors_list)))

    for sectors, color in zip(sectors_list, colors):
        match = cluster_data[cluster_data['sectors'].apply(
            lambda s: list(s) == list(sectors)
        )]
        if match.empty:
            print(f"No row found for sectors={sectors} in cluster '{name_str}'. Skipping.")
            continue
        row = match.iloc[0]
        freqs = np.asarray(row['LSP_freq'])
        power = np.asarray(row['LSP_power'])
        if x_lim is not None:
            mask = (freqs >= x_lim[0]) & (freqs <= x_lim[1])
            freqs, power = freqs[mask], power[mask]
        cad = row.get('cadence', '?')
        label = f'{list(sectors)} | {cad} min'
        ax.plot(freqs, power, lw=1.5, alpha=0.8, color=color, label=label)
        fap = row['LSP_FAP']
        if np.isfinite(fap):
            ax.axhline(fap, color=color, lw=1.0, linestyle='--', alpha=0.6,
                       label=f'FAP={fap:.2e}')

    ax.set_yscale(yscale)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel('power', fontsize=14)
    ax.set_xlabel('frequency (1/day)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(name_str, fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    plt.show()

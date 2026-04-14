import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker


def _apply_x_lim(x, pg, x_lim):
    """Slice x and pg to the specified range (index or frequency)."""
    if x_lim is None:
        return x, pg
    mask = (x >= x_lim[0]) & (x <= x_lim[1])
    return x[mask], pg[mask]


def plot_cluster_periodograms(master_table, name, frequency_bins=None, max_num_pgs=10,
                               yscale='log', xscale='linear', x_lim=None, y_lim=None):
    """
    Plot pre-computed periodograms for a cluster, grouped by n_sectors.
    One subplot per unique n_sectors value; each subplot shows up to max_num_pgs lines.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str, bytes, or None
        Cluster name matching the 'name' column. If None, a random cluster is chosen.
    frequency_bins : array-like or None
        X-axis values. If None, integer indices are used.
    max_num_pgs : int
        Maximum number of periodogram lines per subplot. Excess rows are randomly sampled.
    yscale : str
        Y-axis scale: 'log' (default) or 'linear'.
    xscale : str
        X-axis scale: 'linear' (default) or 'log'.
    x_lim : tuple of (min, max) or None
        X-axis range. Interpreted as bin indices if frequency_bins is None,
        or as frequency values if frequency_bins is provided.
    y_lim : tuple of (min, max) or None
        Y-axis range applied to all subplots.
    """
    if name is None:
        name = np.random.choice(master_table['name'].unique())
        print(f"Randomly selected cluster: '{name}'")

    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = master_table[master_table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name_str}'.")
        return

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
            pg = row['FullPeriodogram']
            x = np.array(frequency_bins) if frequency_bins is not None else np.arange(len(pg))
            x, pg = _apply_x_lim(x, pg, x_lim)
            sectors_label = list(row['sectors'])
            label = f'{sectors_label} | {row["cadence"]:.0f} min'
            ax.plot(x, pg, lw=1.5, alpha=0.7, color=color, label=label)

        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.set_ylabel('power', fontsize=14)
        ax.tick_params(labelsize=12)
        shown = len(sampled)
        ax.set_title(f'n_sectors={ns}  ({n_total} total, showing {shown})', fontsize=14)
        ax.legend(fontsize=11, ncol=2, loc='upper right')

    xlabel = 'frequency' if frequency_bins is not None else 'frequency (bin index)'
    axes[-1].set_xlabel(xlabel, fontsize=14)
    fig.suptitle(name_str, fontsize=18)
    fig.tight_layout()
    plt.show()


def plot_cluster_periodograms_specific(master_table, name, sectors_list,
                                       frequency_bins=None, yscale='log', xscale='linear',
                                       x_lim=None, y_lim=None):
    """
    Plot a specific set of periodograms for a cluster on a single axes.

    Parameters
    ----------
    master_table : pd.DataFrame
    name : str or bytes
        Cluster name matching the 'name' column.
    sectors_list : list of lists
        Each entry is a sector combination to look up and plot,
        e.g. [[1], [1, 2], [1, 2, 3]].
    frequency_bins : array-like or None
        X-axis values. If None, integer indices are used.
    yscale : str
        Y-axis scale: 'log' (default) or 'linear'.
    xscale : str
        X-axis scale: 'linear' (default) or 'log'.
    x_lim : tuple of (min, max) or None
        X-axis range. Interpreted as bin indices if frequency_bins is None,
        or as frequency values if frequency_bins is provided.
    y_lim : tuple of (min, max) or None
        Y-axis range.
    """
    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = master_table[master_table['name'] == name]

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
        pg = row['FullPeriodogram']
        x = np.array(frequency_bins) if frequency_bins is not None else np.arange(len(pg))
        x, pg = _apply_x_lim(x, pg, x_lim)
        label = f'{list(sectors)} | {row["cadence"]:.0f} min'
        ax.plot(x, pg, lw=1.5, alpha=0.8, color=color, label=label)
        ax.axhline(row['FAP'], color=color, lw=1.0, linestyle='--', alpha=0.6,
                   label=f'FAP={row["FAP"]:.2e}')

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.set_ylabel('power', fontsize=14)
    xlabel = 'frequency' if frequency_bins is not None else 'frequency (bin index)'
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(name_str, fontsize=16)
    ax.legend(fontsize=12, loc='upper right')
    fig.tight_layout()
    plt.show()


def plot_cluster_lsp(master_table, name, max_num_pgs=10,
                     yscale='log', x_lim=None, y_lim=None):
    """
    Plot invariant LSP periodograms for a cluster, grouped by n_sectors.

    Reads the ``LSP_freq``, ``LSP_power``, and ``LSP_FAP`` columns produced by
    ``add_invariant_LSP_stats``.  Because each row carries its own frequency
    grid (grid spacing depends on the row's time baseline), every line is
    plotted against its own x-axis rather than a shared bin index.

    A secondary x-axis showing period in days is drawn at the top of each
    subplot for easier interpretation.

    Parameters
    ----------
    master_table : pd.DataFrame
        Must contain ``LSP_freq``, ``LSP_power``, ``LSP_FAP`` columns.
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
        name = np.random.choice(master_table['name'].unique())
        print(f"Randomly selected cluster: '{name}'")

    name_str = name.decode() if isinstance(name, bytes) else name
    cluster_data = master_table[master_table['name'] == name]

    if cluster_data.empty:
        print(f"No data found for cluster '{name_str}'.")
        return

    if 'LSP_freq' not in master_table.columns:
        raise KeyError("LSP_freq column not found. Run add_invariant_LSP_stats first.")

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
            label = f'{sectors_label} | {row["cadence"]:.0f} min | df={df:.2e} day⁻¹'
            ax.plot(freqs, power, lw=1.5, alpha=0.7, color=color, label=label)

            if np.isfinite(fap):
                ax.axhline(fap, color=color, lw=1.0, linestyle='--', alpha=0.6)

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

        # Choose period tick locations that fall within the plotted frequency range
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
    fig.suptitle(f'{name_str} — invariant LSP', fontsize=18)
    fig.tight_layout()
    plt.show()


def plot_lsp_comparison(master_table, name, n_sectors,
                        x_lim=None, yscale='log', xscale='linear', seed=None):
    """
    Plot old and new periodograms for one randomly chosen row on the same axes.

    X-axis is frequency (1/day).  The old ``FullPeriodogram`` used linear-period
    bins (1 / np.arange(0.04, 11, 0.01)); the new ``LSP_power`` uses
    linear-frequency bins from ``ls_bins``.

    Parameters
    ----------
    master_table : pd.DataFrame
        Must contain ``FullPeriodogram``, ``LSP_freq``, ``LSP_power``,
        ``FAP``, and ``LSP_FAP`` columns.
    name : str or bytes
        Cluster name matching the ``name`` column.
    n_sectors : int
        Only rows with this ``n_sectors`` value are considered.
    x_lim : tuple of (f_min, f_max) or None
        Frequency range (1/day) to display.
    yscale : str
        ``'log'`` (default) or ``'linear'``.
    xscale : str
        ``'linear'`` (default) or ``'log'``.
    seed : int or None
        Random seed for reproducible row selection.
    """
    if 'LSP_freq' not in master_table.columns:
        raise KeyError("LSP_freq column not found. Run add_invariant_LSP_stats first.")

    name_str = name.decode() if isinstance(name, bytes) else name
    rows = master_table[
        (master_table['name'] == name) &
        (master_table['n_sectors'] == n_sectors)
    ]

    if rows.empty:
        print(f"No rows found for cluster '{name_str}' with n_sectors={n_sectors}.")
        return

    row = rows.sample(1, random_state=seed).iloc[0]

    # --- old periodogram -------------------------------------------------------
    # FullPeriodogram is stored in INCREASING frequency order (ELK sorted
    # internally), so FullPeriodogram[0] = power at the lowest frequency.
    # 1/np.arange(0.04,11,0.01) is DECREASING [25,...,0.09], so we reverse it
    # to get ascending frequencies that correctly pair with the stored array.
    old_power = np.asarray(row['FullPeriodogram'])
    pg_len    = len(old_power)
    old_freq  = (1.0 / np.arange(0.04, 11, 0.01))[:pg_len]#[::-1]  # ascending [0.09,…,25]

    # --- new periodogram: LSP_freq already increasing frequency ---------------
    new_freq  = np.asarray(row['LSP_freq'])
    new_power = np.asarray(row['LSP_power'])

    # --- apply frequency zoom ------------------------------------------------
    if x_lim is not None:
        old_mask = (old_freq >= x_lim[0]) & (old_freq <= x_lim[1])
        old_freq, old_power = old_freq[old_mask], old_power[old_mask]
        new_mask = (new_freq >= x_lim[0]) & (new_freq <= x_lim[1])
        new_freq, new_power = new_freq[new_mask], new_power[new_mask]

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(old_freq, old_power, lw=1.5, alpha=0.8, color='steelblue',
            label='old (linear-period bins)')
    ax.plot(new_freq, new_power, lw=1.5, alpha=0.8, color='tomato',
            label='new (linear-freq bins)')

    old_fap = row['FAP']
    new_fap = row['LSP_FAP']
    if np.isfinite(old_fap):
        ax.axhline(old_fap, color='steelblue', lw=1.0, ls='--', alpha=0.7,
                   label=f'old FAP = {old_fap:.3f}')
    if np.isfinite(new_fap):
        ax.axhline(new_fap, color='tomato',    lw=1.0, ls='--', alpha=0.7,
                   label=f'new FAP = {new_fap:.3f}')

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_xlabel('frequency (1/day)', fontsize=14)
    ax.set_ylabel('power', fontsize=14)
    ax.tick_params(labelsize=12)
    if x_lim is not None:
        ax.set_xlim(x_lim)

    sectors_label = list(row['sectors'])
    ax.set_title(
        f"{name_str}  |  n_sectors={n_sectors}  |  sectors={sectors_label}  "
        f"|  cadence={row['cadence']:.0f} min",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    fig.tight_layout()
    plt.show()

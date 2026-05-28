from itertools import combinations
import numpy as np
import random


def make_distinct_sector_combinations(
    n_sectors,
    f_max=0.6,
    seed=0,
    shuffle=True,
    include_full=True,
):
    """
    Make distinct combinations of TESS sectors.

    A candidate combination A of size k is accepted only if

        |A ∩ B| / k <= f_max

    for every previously accepted combination B of the same size k.

    Parameters
    ----------
    n_sectors : int
        Number of available sectors.

    f_max : float
        Maximum allowed fractional overlap between same-size combinations.
        Smaller values produce fewer, more distinct combinations.

    seed : int
        Random seed used if shuffle=True.

    shuffle : bool
        If True, randomly shuffle candidates before greedy selection.
        If False, combinations are selected in lexicographic order.

    include_full : bool
        If True, include the full n-sector combination.
        This only matters because the full combination has no same-size competitor.

    Returns
    -------
    n_per_k : dict
        Dictionary mapping k -> number of selected combinations.

    combos : list[list[int]]
        Selected sector combinations.
    """

    if not (0 <= f_max <= 1):
        raise ValueError("f_max must be between 0 and 1.")

    rng = random.Random(seed)

    all_selected = []
    n_per_k = {}

    sector_ids = list(range(n_sectors))

    for k in range(1, n_sectors + 1):

        if (k == n_sectors) and (not include_full):
            n_per_k[k] = 0
            continue

        candidates = list(combinations(sector_ids, k))

        if shuffle:
            rng.shuffle(candidates)

        selected_k = []

        max_shared = int(np.floor(f_max * k))

        for cand in candidates:
            cand_set = set(cand)

            keep = True

            for prev in selected_k:
                prev_set = set(prev)
                n_shared = len(cand_set & prev_set)

                if n_shared > max_shared:
                    keep = False
                    break

            if keep:
                selected_k.append(cand)

        selected_k = [list(c) for c in selected_k]

        all_selected.extend(selected_k)
        n_per_k[k] = len(selected_k)

    return n_per_k, all_selected
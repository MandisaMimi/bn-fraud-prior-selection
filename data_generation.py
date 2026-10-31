"""
data_generation.py
===================
Ancestral (forward) sampling: walk the DAG in topological order and draw
each node's value conditional on its already-sampled parents. Because the
ground-truth structure is tiered by construction, config.NODES is already
in a valid topological order -- no separate topological sort is needed.
"""

import numpy as np
import pandas as pd

import config


def generate_dataset(ground_truth: dict, n_rows: int, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Parameters
    ----------
    ground_truth : dict
        Output of ground_truth.build_ground_truth() -- contains 'parents',
        'marginals', 'cpts'.
    n_rows : int
        Number of rows to generate.
    rng : np.random.RandomState
        Random generator (kept separate from the structure/CPT rng so the
        SAME ground-truth model can be used to generate multiple independent
        datasets if ever needed).

    Returns
    -------
    pd.DataFrame with one integer-coded column per node in config.NODES.
    """
    parents = ground_truth["parents"]
    marginals = ground_truth["marginals"]
    cpts = ground_truth["cpts"]

    data = {}

    # config.NODES is already topologically sorted (tier 0 first, sink last)
    for node in config.NODES:
        plist = parents[node]
        k = config.STATE_COUNTS[node]

        if not plist:
            # Root node -> sample directly from its marginal, vectorised
            probs = marginals[node]
            data[node] = rng.choice(k, size=n_rows, p=probs)
            continue

        # Non-root node: group rows by their (already sampled) parent-state
        # tuple, then sample each group in one vectorised call. This avoids
        # a slow python loop over every single row.
        parent_cols = np.column_stack([data[p] for p in plist])
        # Build a single string key per row so we can group cheaply
        keys = [tuple(row) for row in parent_cols]

        out = np.empty(n_rows, dtype=int)
        # Group row-indices by parent configuration
        df_keys = pd.DataFrame({"key": keys, "idx": np.arange(n_rows)})
        for cfg, group in df_keys.groupby("key", sort=False):
            probs = cpts[node].get(cfg)
            if probs is None:
                # Shouldn't happen (all configs are enumerated up front),
                # but fall back to uniform just in case of a mismatch.
                probs = np.ones(k) / k
            idxs = group["idx"].to_numpy()
            out[idxs] = rng.choice(k, size=len(idxs), p=probs)

        data[node] = out

    return pd.DataFrame(data)

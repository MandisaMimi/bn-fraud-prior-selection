"""
ground_truth.py
================
Everything needed to AUTOMATICALLY construct one "true" Bayesian Network:
  1. sample_structure()  -> random DAG respecting the tier constraint
  2. sample_parameters() -> random CPTs (Dirichlet for normal nodes,
                             calibrated Beta for the isFraud node)

No edges or probabilities are ever hand-picked here -- every choice is
driven by a numpy RandomState seeded per-run, so results are reproducible
but not manually curated (this is the "automatic, avoid selection bias"
requirement from the design discussion).
"""

import itertools
import numpy as np

import config


def sample_structure(rng: np.random.RandomState):
    """
    Randomly sample a DAG that respects config.TIERS (edges only ever point
    from an earlier tier to a later tier) and config.MAX_PARENTS (in-degree cap).

    Returns
    -------
    parents : dict[str, list[str]]
        Maps each node -> its list of parent node names (possibly empty).
    """
    # Flatten tiers into an ordered list of (node, tier_index) pairs
    tier_of = {}
    for t, nodes_in_tier in enumerate(config.TIERS):
        for n in nodes_in_tier:
            tier_of[n] = t

    parents = {n: [] for n in config.NODES}

    for node in config.NODES:
        t = tier_of[node]
        if t == 0:
            continue  # root nodes never get parents

        # Candidate parents = every node in a strictly earlier tier
        candidates = [n for n in config.NODES if tier_of[n] < t]

        # Each candidate is independently included with probability EDGE_INCLUSION_PROB
        chosen = [c for c in candidates if rng.rand() < config.EDGE_INCLUSION_PROB]

        # Enforce the max in-degree cap by randomly subsampling if needed
        if len(chosen) > config.MAX_PARENTS:
            chosen = list(rng.choice(chosen, size=config.MAX_PARENTS, replace=False))

        parents[node] = sorted(chosen)

    # Safety net: force isFraud to have at least one parent, otherwise fraud
    # would be statistically independent of everything else and the whole
    # exercise (recovering fraud-relevant structure) becomes vacuous.
    if len(parents["isFraud"]) == 0:
        all_earlier = [n for n in config.NODES if tier_of[n] < tier_of["isFraud"]]
        parents["isFraud"] = [rng.choice(all_earlier)]

    return parents


def _parent_configurations(parent_list):
    """Enumerate every possible combination of parent STATES as tuples."""
    if not parent_list:
        return [()]
    ranges = [range(config.STATE_COUNTS[p]) for p in parent_list]
    return list(itertools.product(*ranges))


def sample_parameters(rng: np.random.RandomState, parents: dict, target_fraud_rate=None):
    """
    Sample ground-truth CPTs given a structure produced by sample_structure().

    Returns
    -------
    marginals : dict[str, np.ndarray]
        Root-node marginal probability vectors (only for nodes with no parents).
    cpts : dict[str, dict[tuple, np.ndarray]]
        For every non-root node: maps a parent-state tuple -> probability
        vector over that node's own states.
    """
    if target_fraud_rate is None:
        target_fraud_rate = config.TARGET_FRAUD_RATE

    marginals = {}
    cpts = {}

    for node in config.NODES:
        k = config.STATE_COUNTS[node]
        plist = parents[node]

        if not plist:
            # Root node: sample one Dirichlet marginal distribution
            alpha = np.ones(k) * config.CATEGORICAL_CONCENTRATION
            marginals[node] = rng.dirichlet(alpha)
            continue

        configs = _parent_configurations(plist)
        node_cpt = {}

        if node == "isFraud":
            # Special calibrated case: for each parent configuration, draw
            # P(fraud=1) ~ Beta(a, b) with mean == target_fraud_rate. Because
            # every configuration is centred on the SAME mean, the overall
            # marginal fraud rate lands near target_fraud_rate regardless of
            # how common each configuration is in the data -- no manual
            # calibration loop required.
            conc = config.FRAUD_BETA_CONCENTRATION
            a = target_fraud_rate * conc
            b = conc - a
            for cfg in configs:
                p_fraud = rng.beta(a, b)
                node_cpt[cfg] = np.array([1.0 - p_fraud, p_fraud])
        else:
            alpha = np.ones(k) * config.CATEGORICAL_CONCENTRATION
            for cfg in configs:
                node_cpt[cfg] = rng.dirichlet(alpha)

        cpts[node] = node_cpt

    return marginals, cpts


def build_ground_truth(seed: int, target_fraud_rate=None):
    """Convenience wrapper: one call -> a full ground-truth BN for this seed."""
    rng = np.random.RandomState(seed)
    parents = sample_structure(rng)
    marginals, cpts = sample_parameters(rng, parents, target_fraud_rate)
    return {
        "seed": seed,
        "parents": parents,
        "marginals": marginals,
        "cpts": cpts,
    }

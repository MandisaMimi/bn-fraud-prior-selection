"""
reporting.py
============
A Bayesian Network isn't just its DAG -- it's the DAG PLUS a Conditional
Probability Table (CPT) at every node. This module writes those CPTs out
in full, human-readable form for both the ground-truth network and the
learned network, so each picture in figures/ has a matching text file that
makes the network "complete" rather than just a skeleton diagram.

Both networks are formatted with the SAME function (_format_all_cpts),
using the SAME plain-dict representation (parents/marginals/cpts) that
evaluation.py already builds for the learned model. This matters because
pgmpy's own TabularCPD.__str__() TRUNCATES wide tables with "..." once a
node has more than a few parent-state columns -- which would silently
drop states from the printed CPT. Building both networks' text output from
the same untruncated dict format avoids that entirely, and also makes the
ground-truth and learned files directly, line-for-line comparable.
"""

import config
import evaluation


def _format_prob_row(parent_names, parent_state_tuple, probs, node, state_count):
    left = ", ".join(f"{p}={s}" for p, s in zip(parent_names, parent_state_tuple))
    right = "  ".join(f"P({node}={v})={probs[v]:.4f}" for v in range(state_count))
    if left:
        return f"  {left:40s} -> {right}"
    return f"  {right}"


def _format_all_cpts(parents: dict, marginals: dict, cpts: dict, header_lines: list):
    """Shared formatter: builds the full, untruncated CPT text for every
    node in config.NODES, given the plain-dict (parents/marginals/cpts)
    representation used throughout evaluation.py."""
    lines = list(header_lines)

    for node in config.NODES:
        plist = parents[node]
        k = config.STATE_COUNTS[node]
        lines.append(f"Node: {node}  (states: {k})")
        lines.append(f"Parents: {', '.join(plist) if plist else '(none -- root node)'}")
        lines.append("-" * 78)

        if not plist:
            probs = marginals[node]
            lines.append(_format_prob_row([], (), probs, node, k))
        else:
            for cfg, probs in cpts[node].items():
                lines.append(_format_prob_row(plist, cfg, probs, node, k))

        lines.append("")

    return lines


def _true_cpt_lines(ground_truth: dict):
    """Ground-truth CPTs, already in plain-dict form -- no conversion needed."""
    header = [
        f"GROUND-TRUTH BAYESIAN NETWORK -- FULL CONDITIONAL PROBABILITY TABLES (seed {ground_truth['seed']})",
        "=" * 78,
        "",
    ]
    return _format_all_cpts(ground_truth["parents"], ground_truth["marginals"], ground_truth["cpts"], header)


def _learned_cpt_lines(model, seed: int):
    """
    Learned CPTs, converted from pgmpy's fitted TabularCPD objects into the
    SAME plain-dict format via evaluation.pgmpy_model_to_dict(), then
    formatted with the identical _format_all_cpts() used for the ground
    truth. This is what avoids pgmpy's own "..." column truncation.
    """
    parents, marginals, cpts = evaluation.pgmpy_model_to_dict(model)
    header = [
        f"LEARNED BAYESIAN NETWORK -- FULL CONDITIONAL PROBABILITY TABLES (seed {seed})",
        "Hill Climbing + BIC structure, Maximum Likelihood Estimation parameters",
        "Note: parent-state combinations never seen in the training data are filled",
        "with a small epsilon probability (then renormalised) rather than left undefined,",
        "so every row sums to 1.0 -- see evaluation.pgmpy_model_to_dict().",
        "=" * 78,
        "",
    ]
    return _format_all_cpts(parents, marginals, cpts, header)


def write_true_cpts(ground_truth: dict, out_path: str):
    """
    Write the FULL ground-truth CPT for every node (parents, marginal
    distribution for roots, full conditional table for non-roots) to a
    standalone .txt file.
    """
    lines = _true_cpt_lines(ground_truth)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_learned_cpts(model, out_path: str, seed: int):
    """
    Write the FULL learned CPT for every node to a standalone .txt file.
    Uses the untruncated plain-dict conversion (see _learned_cpt_lines),
    NOT pgmpy's own str(cpd), which truncates wide tables.
    """
    lines = _learned_cpt_lines(model, seed)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))

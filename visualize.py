"""
visualize.py
============
All plotting lives here, kept separate from the experiment logic so that
main.py doesn't get cluttered with matplotlib details.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

import config


# Fixed tier colours so ground-truth and learned pictures are visually
# consistent (same node = same colour in both pictures).
TIER_COLORS = ["#cfe8ff", "#d7f5d1", "#ffe9b3", "#ffd0d0", "#e3d1ff"]


def _tiered_layout(nodes, x_gap=4.5, y_gap=2.2):
    """
    A clean, hand-built layered layout: x position = tier index (spaced far
    apart), y position = centred within the tier (so tiers line up
    symmetrically around the horizontal midline instead of stacking
    downward, which is what was causing the cramped/crossed-over look).
    """
    tier_of = {}
    for t, tier_nodes in enumerate(config.TIERS):
        for n in tier_nodes:
            tier_of[n] = t

    pos = {}
    for t, tier_nodes in enumerate(config.TIERS):
        present = [n for n in tier_nodes if n in nodes]
        m = len(present)
        for i, n in enumerate(present):
            y = (i - (m - 1) / 2.0) * y_gap
            pos[n] = (t * x_gap, y)
    return pos, tier_of


def _draw_dag(ax, edges, nodes, title, node_size=5200):
    """Draw a single DAG on a given matplotlib axis, large and uncluttered."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos, tier_of = _tiered_layout(nodes)
    node_colors = [TIER_COLORS[tier_of[n] % len(TIER_COLORS)] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size,
                            node_color=node_colors, edgecolors="#333333", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight="bold")

    # Give every edge its own slight curvature (varying by source/target pair)
    # so parallel/near-parallel edges don't overlap into an unreadable mess.
    for i, (u, v) in enumerate(G.edges()):
        rad = 0.12 if (hash((u, v)) % 2 == 0) else -0.12
        nx.draw_networkx_edges(
            G, pos, ax=ax, edgelist=[(u, v)], arrows=True, arrowsize=22, width=1.8,
            connectionstyle=f"arc3,rad={rad}",
            min_source_margin=32, min_target_margin=32,
            edge_color="#555555",
        )

    ax.set_title(title, fontsize=15, pad=14)
    ax.axis("off")
    ax.margins(0.15)


def plot_single_dag(edges, nodes, title, out_path, figsize=(16, 10)):
    """Save ONE large, standalone picture of a single DAG."""
    fig, ax = plt.subplots(figsize=figsize)
    _draw_dag(ax, edges, nodes, title)

    # Small legend explaining the tier colour coding
    handles = [
        mpatches.Patch(color=TIER_COLORS[t % len(TIER_COLORS)], label=f"Tier {t}: {', '.join(tier_nodes)}")
        for t, tier_nodes in enumerate(config.TIERS)
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=2, fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_example_comparison(true_edges, learned_edges, seed, true_path, learned_path):
    """
    Saves TWO separate, large pictures (not a cramped side-by-side panel):
    one for the ground-truth DAG, one for the learned DAG. Same node
    positions/colours in both so they're easy to compare by eye.
    """
    plot_single_dag(true_edges, config.NODES,
                     f"Ground-truth Bayesian Network structure (seed {seed})", true_path)
    plot_single_dag(learned_edges, config.NODES,
                     f"Learned Bayesian Network structure via Hill Climbing + BIC (seed {seed})", learned_path)


def plot_network_complete(edges, nodes, title, cpt_lines, out_pdf_path, lines_per_page=60):
    """
    THE "complete network" output: a single multi-page PDF containing
    (1) the DAG diagram on page 1, followed by (2) the FULL Conditional
    Probability Table for every node, paginated as plain monospace text.

    This is the file to look at if you want "the graph" and "the
    probability tables" together in one place -- the standalone PNG
    diagrams and standalone .txt CPT dumps still exist separately too,
    but this PDF is the one-stop, unambiguous "here is the complete
    Bayesian Network" artifact.

    CPT pages use a WIDE landscape page (15in) at a small font (7pt)
    because 3-parent conditional rows run to ~190 characters -- a normal
    portrait page would clip the right-hand columns.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(out_pdf_path) as pdf:
        # --- Page 1: the DAG diagram -----------------------------------
        fig, ax = plt.subplots(figsize=(16, 10))
        _draw_dag(ax, edges, nodes, title)
        handles = [
            mpatches.Patch(color=TIER_COLORS[t % len(TIER_COLORS)], label=f"Tier {t}: {', '.join(tier_nodes)}")
            for t, tier_nodes in enumerate(config.TIERS)
        ]
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
                  ncol=2, fontsize=8, frameon=False)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # --- Following pages: full CPTs, paginated as monospace text ---
        for page_start in range(0, len(cpt_lines), lines_per_page):
            chunk = cpt_lines[page_start:page_start + lines_per_page]
            fig, ax = plt.subplots(figsize=(15, 9))
            ax.axis("off")
            ax.text(0.01, 0.99, "\n".join(chunk), family="monospace", fontsize=7,
                     va="top", ha="left", transform=ax.transAxes)
            # No tight_layout() here -- the axis is entirely off (no ticks/
            # labels to fit), so there's nothing for it to adjust, and it
            # only produces a spurious "margins cannot be made large enough"
            # warning on some matplotlib versions.
            pdf.savefig(fig)
            plt.close(fig)


def plot_aggregate_results(results_df: pd.DataFrame, out_path: str):
    """Boxplots of SHD, precision, recall, and the KL estimate across all seeds."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    metrics = [
        ("shd", "Structural Hamming Distance", axes[0]),
        ("precision", "Edge Precision", axes[1]),
        ("recall", "Edge Recall", axes[2]),
        ("kl_estimate", "KL(true || learned) estimate", axes[3]),
    ]
    for col, label, ax in metrics:
        ax.boxplot(results_df[col].values, vert=True)
        ax.scatter(
            [1] * len(results_df),
            results_df[col].values,
            alpha=0.5, color="#1f5fa0", zorder=3,
        )
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])

    fig.suptitle(f"Structure recovery across N={len(results_df)} independent ground-truth seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_fraud_rate_sweep(sweep_df: pd.DataFrame, out_path: str):
    """Line plot: how recovery quality changes as fraud prevalence drops."""
    agg = sweep_df.groupby("fraud_rate").agg(
        shd_mean=("shd", "mean"), shd_std=("shd", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        kl_mean=("kl_estimate", "mean"), kl_std=("kl_estimate", "std"),
    ).sort_index(ascending=False).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, mean_col, std_col, label in [
        (axes[0], "shd_mean", "shd_std", "Structural Hamming Distance"),
        (axes[1], "recall_mean", "recall_std", "Edge Recall"),
        (axes[2], "kl_mean", "kl_std", "KL(true || learned) estimate"),
    ]:
        ax.errorbar(agg["fraud_rate"], agg[mean_col], yerr=agg[std_col],
                    marker="o", capsize=4, color="#1f5fa0")
        ax.set_xlabel("Target fraud rate")
        ax.set_title(label, fontsize=10)
        ax.invert_xaxis()  # rarer fraud (smaller rate) shown moving right

    fig.suptitle("Structure recovery quality as fraud prevalence decreases", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_cross_distribution_comparison(cross_df: pd.DataFrame, out_path: str):
    """
    Paired comparison: for every (P, Q) pair, plot KL(P||P_hat) [same
    distribution, P_hat learned from P's own data] against KL(P||Q)
    [different distribution, Q learned from an entirely independent
    ground truth's data]. A connecting line per pair makes the consistent
    gap between the two cases visually obvious -- this is the figure that
    directly answers "does KL divergence actually discriminate between a
    genuine recovery and an unrelated distribution?"
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: paired dot-and-line plot, one line per (P, Q) pair
    ax = axes[0]
    n = len(cross_df)
    for i, row in cross_df.iterrows():
        ax.plot([0, 1], [row["kl_same_distribution"], row["kl_different_distribution"]],
                color="#999999", alpha=0.6, linewidth=1, zorder=1)
    ax.scatter([0] * n, cross_df["kl_same_distribution"], color="#1f5fa0",
               s=60, zorder=2, label="KL(P || P_hat)\n(same distribution)")
    ax.scatter([1] * n, cross_df["kl_different_distribution"], color="#c0392b",
               s=60, zorder=2, label="KL(P || Q)\n(different distribution)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same\ndistribution", "Different\ndistribution"])
    ax.set_ylabel("KL divergence estimate")
    ax.set_title(f"Paired comparison across N={n} independent seed-pairs", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # Right panel: the ratio, showing how many times larger the
    # different-distribution KL is, per pair
    ax2 = axes[1]
    ax2.bar(range(n), cross_df["ratio"], color="#5a8fc7")
    ax2.axhline(1.0, color="#333333", linestyle="--", linewidth=1)
    ax2.set_xlabel("Seed pair")
    ax2.set_ylabel("KL(P||Q) / KL(P||P_hat)")
    ax2.set_title("How many times larger is the mismatched-distribution KL?", fontsize=11)

    fig.suptitle("KL divergence correctly separates \"recovered the right distribution\"\n"
                 "from \"learned an unrelated one\"", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_rq2_comparison(rq2_df: pd.DataFrame, sweep_df: pd.DataFrame, out_path: str):
    """
    Left panel: KL(P_true || P_method) by method (MLE / Jeffreys / Laplace /
    Expert), at the default row count -- the DIRECT answer to "which
    parameter-estimation method recovers the true distribution best?"
    Right panel: how that same KL-from-truth shrinks as training data grows,
    for every method -- testing the data-scarcity hypothesis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    method_order = ["mle", "jeffreys", "laplace", "expert"]
    agg = rq2_df.groupby("method")["kl_vs_true"].mean().loc[method_order]
    axes[0].bar(method_order, agg.values, color=["#888888", "#5a8fc7", "#5a8fc7", "#c0392b"])
    axes[0].set_ylabel("KL(P_true || P_method)")
    axes[0].set_title("Direct distance from the true distribution\n(mean across seeds, default row count)")

    sweep_agg = sweep_df.groupby(["rows", "method"])["kl_vs_true"].mean().unstack()
    for method in method_order:
        axes[1].plot(sweep_agg.index, sweep_agg[method], marker="o", label=method)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training rows (log scale)")
    axes[1].set_ylabel("KL(P_true || P_method) (log scale)")
    axes[1].set_title("Distance from truth shrinks as training data grows,\nfor every method")
    axes[1].legend()

    fig.suptitle("RQ2: Which parameter-estimation method recovers the true distribution best?", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_full_comparison(full_df: pd.DataFrame, out_path: str):
    """F1 score comparison: the three BN prior variants vs the three
    discriminative baselines, mean across seeds."""
    agg = full_df.groupby("model")[["precision", "recall", "f1", "auroc"]].mean()
    model_order = ["BN_jeffreys", "BN_laplace", "BN_expert",
                   "logistic_regression", "random_forest", "naive_bayes"]
    agg = agg.loc[model_order]
    colors = ["#5a8fc7"] * 3 + ["#c0392b"] * 3

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(agg))
    ax.bar(x, agg["f1"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=30, ha="right")
    ax.set_ylabel("F1 score (mean across seeds)")
    ax.set_title("Bayesian Network (blue) vs discriminative baselines (red)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ground_truth_pipeline_diagram(out_path: str):
    """The controlled ground-truth experimental framework diagram: True DAG
    + True CPTs -> sample data -> hide truth -> Hill Climbing -> learned
    structure -> compare vs actual truth. Illustrates the central
    methodological contribution of the revised (post-PaySim) study."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2)
    ax.axis("off")

    steps = ["True DAG\n+ True CPTs", "Sample\nsynthetic data", "Hide the\ntruth",
             "Hill Climbing\n+ BIC", "Learned\nstructure", "Compare vs.\nactual truth"]
    colors = ["#5a8fc7", "#5a8fc7", "#c0392b", "#5a8fc7", "#5a8fc7", "#2e8b57"]

    n = len(steps)
    box_w, box_h = 1.3, 1.0
    xs = [0.4 + i * 1.75 for i in range(n)]

    for i, (x, label, color) in enumerate(zip(xs, steps, colors)):
        box = FancyBboxPatch((x, 0.5), box_w, box_h, boxstyle="round,pad=0.05",
                              linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(box)
        ax.text(x + box_w / 2, 0.5 + box_h / 2, label, ha="center", va="center",
                 fontsize=9.5, weight="bold")
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1], 1.0), xytext=(x + box_w, 1.0),
                        arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"))

    ax.set_title("The controlled ground-truth experimental framework at the heart of this study",
                  fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

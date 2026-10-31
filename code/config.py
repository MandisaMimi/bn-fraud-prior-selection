"""
config.py
=========
Single source of truth for every tunable setting in the pipeline.
Nothing else in the codebase should hard-code a number that belongs here.

WHY A TIERED ORDERING?
-----------------------
A fully unconstrained random DAG over 9 nodes could produce nonsense edges
(e.g. isFraud -> oldbalanceOrg). To keep the automatic generation process
free of hand-picked edges while still respecting basic domain logic, nodes
are grouped into TIERS. An edge is only allowed to go from an EARLIER tier
to a LATER tier. Which edges actually exist within that constraint is still
decided by random sampling (see ground_truth.py) -- so the causal *skeleton*
is automatic, only the *direction envelope* is constrained.
"""

# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------
# Each node maps to its number of discrete states.
# Continuous-type variables (amount, balances, balance errors) are modelled
# as NATIVE discrete nodes with K=5 states (bin 0 = lowest quantile ... bin 4
# = highest quantile) -- there is no continuous data anywhere in this
# pipeline, so no separate discretisation step is needed (see chat discussion:
# "Option A").
NUM_BINS = 5

STATE_COUNTS = {
    "type":            5,          # CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER
    "oldbalanceOrg":   NUM_BINS,
    "oldbalanceDest":  NUM_BINS,
    "amount":          NUM_BINS,
    "newbalanceOrig":  NUM_BINS,
    "newbalanceDest":  NUM_BINS,
    "delta_orig":      NUM_BINS,   # |newbalanceOrig - (oldbalanceOrg - amount)|, binned
    "delta_dest":      NUM_BINS,   # |newbalanceDest - (oldbalanceDest + amount)|, binned
    "isFraud":         2,          # 0 = legitimate, 1 = fraud
}

NODES = list(STATE_COUNTS.keys())

# ---------------------------------------------------------------------------
# Tier structure -> defines which nodes are ALLOWED to be parents of which
# ---------------------------------------------------------------------------
# Tier index 0 = roots (no parents allowed at all)
# A node in tier t may draw parents from the union of all nodes in tiers < t.
TIERS = [
    ["type", "oldbalanceOrg", "oldbalanceDest"],   # tier 0: roots
    ["amount"],                                     # tier 1
    ["newbalanceOrig", "newbalanceDest"],           # tier 2
    ["delta_orig", "delta_dest"],                   # tier 3
    ["isFraud"],                                     # tier 4: sink
]

# ---------------------------------------------------------------------------
# Ground-truth DAG sampling parameters
# ---------------------------------------------------------------------------
EDGE_INCLUSION_PROB = 0.5   # probability each *candidate* parent edge is kept
MAX_PARENTS = 3              # hard cap on in-degree for any node (keeps CPTs small)

# ---------------------------------------------------------------------------
# Ground-truth CPT sampling parameters
# ---------------------------------------------------------------------------
# Dirichlet concentration used for all NON-fraud categorical nodes.
# Lower values -> more "peaked"/deterministic-looking conditional distributions.
# Higher values -> closer to uniform/noisy conditional distributions.
CATEGORICAL_CONCENTRATION = 2.0

# isFraud is handled specially: for every parent configuration we draw
# P(fraud=1 | config) ~ Beta(a, b) with mean = TARGET_FRAUD_RATE, so the
# AVERAGE fraud rate across configurations is calibrated automatically
# without any manual tuning or rejection loop.
TARGET_FRAUD_RATE = 0.005     # 0.5% -- matches severe real-world imbalance
FRAUD_BETA_CONCENTRATION = 40  # a + b for the Beta draw (higher = less variance across configs)

# ---------------------------------------------------------------------------
# Experiment scale
# ---------------------------------------------------------------------------
N_SEEDS = 15                  # number of independent ground-truth DAGs/datasets
ROWS_PER_DATASET = 20000      # rows used to LEARN the structure
HOLDOUT_ROWS = 5000           # extra rows (same ground truth) used ONLY to
                               # estimate KL divergence via held-out log-likelihood
RANDOM_SEED_BASE = 42         # seed i uses RANDOM_SEED_BASE + i, for reproducibility

# ---------------------------------------------------------------------------
# Structure learning parameters (Hill Climbing + BIC)
# ---------------------------------------------------------------------------
HC_MAX_INDEGREE = 4
HC_MAX_ITER = 100000

# ---------------------------------------------------------------------------
# Secondary experiment: fraud-rate sweep (Stage 6)
# ---------------------------------------------------------------------------
SWEEP_FRAUD_RATES = [0.02, 0.01, 0.005, 0.002, 0.001]
SWEEP_SEEDS_PER_RATE = 5

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------
OUTPUT_DIR = "outputs"
FIGURES_DIR = "outputs/figures"
DATASETS_DIR = "outputs/datasets"
CPTS_DIR = "outputs/cpts"
RESULTS_CSV = "outputs/main_experiment_results.csv"
SWEEP_CSV = "outputs/fraud_rate_sweep_results.csv"

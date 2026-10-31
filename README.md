# Ground-Truth Bayesian Network Recovery Pipeline

This is the code for the "ground truth -> dataset -> learned model" stage
we discussed: it automatically builds N independent ground-truth Bayesian
Networks, generates a synthetic transaction dataset from each one, learns
a structure back from that data with Hill Climbing + BIC, and reports how
well the learned structure/parameters match the (known) ground truth.

## File map (separation of concerns)

| File                    | Responsibility                                                        |
|--------------------------|------------------------------------------------------------------------|
| `config.py`              | Every tunable setting (variables, tiers, sample sizes, seeds, etc.)   |
| `ground_truth.py`        | Stage 1 & 2: automatic random DAG + CPT generation                    |
| `data_generation.py`     | Stage 3: ancestral sampling to turn a ground truth into a dataset     |
| `structure_learning.py`  | Stage 4: Hill Climbing/BIC structure learning + MLE parameter fitting |
| `evaluation.py`          | Stage 5: SHD, edge precision/recall, KL-divergence estimate           |
| `reporting.py`           | Full CPT text for both networks (untruncated, matching format)        |
| `visualize.py`           | All plotting — PNG diagrams AND the combined diagram+CPT PDFs         |
| `main.py`                | Orchestrates everything end to end and saves results                 |

## Setup

```bash
cd coding
pip install -r requirements.txt --break-system-packages
```

(Drop `--break-system-packages` if you're using a virtual environment instead
of a system Python — recommended for anything beyond a quick test:
`python3 -m venv venv && source venv/bin/activate` first.)

## Run

```bash
python main.py
```

**You will get the exact same results every time you run this**, on any
machine, as long as the pinned versions in `requirements.txt` are installed.
This is not automatic in general — Python's default string-hash
randomisation can silently change Hill Climbing's tie-breaking between
otherwise-identical runs. `main.py` guards against this itself (it
transparently re-launches with `PYTHONHASHSEED=0` if that isn't already
set — you'll see a one-line "Re-launching..." message the first time),
so you don't need to set anything by hand; `python main.py` is always
enough. This was verified by diffing two independent fresh runs — every
metric matched exactly.

**On Windows:** use `python main.py`, not `python3 main.py`. Windows
installs usually only provide a `python.exe` command, not `python3.exe` —
running `python3` when that alias doesn't exist can silently do nothing
(no error, no output, prompt just returns), which looks exactly like a
crash but isn't one.

This runs, in order:

1. **Main experiment** — `config.N_SEEDS` independent ground-truth DAGs
   (default 15), each generating a `config.ROWS_PER_DATASET`-row dataset
   (default 20,000), each learned back via Hill Climbing + BIC, each scored
   against its own ground truth.
2. **Fraud-rate sweep** — a smaller secondary experiment
   (`config.SWEEP_FRAUD_RATES` × `config.SWEEP_SEEDS_PER_RATE`, default
   5 rates × 5 seeds) testing whether recovery quality degrades as the
   fraud class gets rarer.

On the machine this was tested on, the full run (15 + 25 = 40 seeds total)
takes under a minute.

## Outputs

Everything lands in `outputs/`:

- `main_experiment_results.csv` — one row per seed: SHD, edge precision,
  edge recall, KL-divergence estimate, realised fraud rate, runtime.
- `fraud_rate_sweep_results.csv` — same columns, one row per (rate, seed).
- `datasets/seed_{N}_train.csv` — **the actual generated dataset** for
  every one of the 15 main-experiment seeds (20,000 rows each). This is
  the real synthetic data, not a summary — every row is one simulated
  transaction with all 9 variables in their discretised (0..k-1) form.
- `datasets/seed_42_holdout.csv` — the separate held-out sample used only
  for the KL-divergence estimate on the example seed.
- `cpts/seed_42_ground_truth_cpts.txt` — the **full** Conditional
  Probability Table for every node in the ground-truth network (not just
  the skeleton diagram) — this is what makes it a complete Bayesian
  Network rather than just a DAG picture.
- `cpts/seed_42_learned_cpts.txt` — the full, fitted CPTs for every node
  in the learned network, in the same style. **Not** generated from
  pgmpy's own table printer, which truncates wide tables with `...` and
  silently hides columns — both files are built from the same untruncated
  formatter so every state is always shown and the two are line-for-line
  comparable.
- `figures/seed_42_ground_truth_COMPLETE.pdf` /
  `figures/seed_42_learned_COMPLETE.pdf` — **the one-stop "complete
  network" files**: page 1 is the DAG diagram, every following page is
  that network's full CPTs, all in a single PDF. This is the file to open
  if you want the graph and its probability tables together rather than
  as two separate files.
- `figures/ground_truth_network.png` — a large, standalone picture of the
  ground-truth DAG (seed 42), colour-coded by tier, well spaced.
- `figures/learned_network.png` — the matching large, standalone picture
  of what Hill Climbing recovered from that seed's data. Same node
  positions and colours as the ground-truth picture, so the two are easy
  to compare side by side even though they're separate files.
- `figures/aggregate_recovery_metrics.png` — boxplots of SHD/precision/
  recall/KL across all N main-experiment seeds.
- `figures/fraud_rate_sweep.png` — how SHD/recall/KL trend as fraud
  prevalence drops.
- `cross_distribution_results.csv` — per-pair results from
  `cross_distribution.py`: `KL(P||P_hat)` vs `KL(P||Q)` for each of the
  N independent seed pairs (see the "Cross-distribution validation"
  section below).
- `figures/cross_distribution_comparison.png` — paired comparison plot
  for the same experiment.

## Adjusting scale

Everything tunable lives in `config.py`. The two settings that most affect
runtime are `N_SEEDS` and `ROWS_PER_DATASET` — increase `ROWS_PER_DATASET`
first if recovery looks noisy, since more rows per dataset gives Hill
Climbing more signal per seed (this generally matters more than adding
extra seeds).

## Cross-distribution validation (`cross_distribution.py`)

**Why this file exists:** the main experiment only ever computes
`KL(P_true || P_learned)`, where `P_learned` is learned from data
generated BY `P_true` itself. Even though Hill Climbing doesn't recover
`P_true` perfectly (see the non-trivial SHD/precision/recall already in
`main_experiment_results.csv`), `P_learned` is still always an *estimate
of the same underlying distribution*. Nothing in the main experiment shows
what the KL estimate looks like when the two distributions being compared
are genuinely, completely different -- so there was no way to tell whether
a small KL value meant "good recovery" or just "this metric is always
small regardless of input."

`cross_distribution.py` closes that gap. For `N` independent seed pairs
`(seed_p, seed_q)`:

1. Build two **entirely independent** ground-truth networks, `P` and `R`
   (different random DAG structure *and* different CPTs).
2. Learn `P_hat` from `P`'s own generated data (same as the main
   experiment) -- the same-distribution case.
3. Learn `Q` from `R`'s generated data instead, then evaluate it against
   `P`'s held-out data -- the different-distribution case.
4. Report `KL(P || P_hat)` and `KL(P || Q)` side by side.

**Run it:**

```bash
python cross_distribution.py
```

**Result from the run in this repository (10 seed pairs, `PYTHONHASHSEED=0`):**

| Comparison | Mean KL | Std |
|---|---|---|
| Same distribution -- `KL(P \|\| P_hat)` | 0.071 | 0.042 |
| Different distribution -- `KL(P \|\| Q)` | 3.858 | 0.591 |

The different-distribution KL was **26x to 137x larger** than the
same-distribution KL in every single one of the 10 pairs, with no overlap
at all between the two groups. This confirms the KL-divergence estimator
used throughout this pipeline genuinely discriminates between "recovered
the right distribution, imperfectly" and "learned something unrelated" --
it is not just trivially small for any input. See
`outputs/figures/cross_distribution_comparison.png` for the paired
comparison plot, and `outputs/cross_distribution_results.csv` for the
full per-pair numbers.

## Interpreting the KL-divergence number

`kl_estimate` is **not** computed by matching CPT tables directly (the
learned structure often has different parents than the true structure, so
there's nothing to line up node-for-node). Instead it's an unbiased
Monte-Carlo estimate:

```
KL(P_true || P_learned) ~= mean over held-out rows of
    [ log P_true(row) - log P_learned(row) ]
```

using rows sampled from the true model but never seen during structure
learning or parameter fitting. A value near 0 means the learned model
explains genuinely new ground-truth data almost as well as the true model
itself.

## Notes for the write-up

- The DAG is generated automatically (random edge sampling within a fixed
  tier ordering + max in-degree cap), not hand-authored — see the
  docstring at the top of `ground_truth.py` for the exact rationale.
- `isFraud`'s conditional probability is drawn from a `Beta` distribution
  centred on `config.TARGET_FRAUD_RATE` for every parent configuration, so
  the realised fraud rate lands near target automatically, without any
  manual calibration loop.
- Both the true and learned models are scored with the **same** log-
  probability function (`evaluation.row_log_prob`) after the learned
  pgmpy model is converted into the same plain-dict format as the ground
  truth (`evaluation.pgmpy_model_to_dict`) — this is what makes the KL
  estimate a fair, apples-to-apples comparison.
- Every node's CPT is exported in full for both networks (`outputs/cpts/`)
  — a Bayesian Network is its structure *and* its parameters together, so
  the diagrams and the CPT files should be read as one complete network,
  not the diagram alone.

## RQ2: parameter-estimation method and Dirichlet prior comparison (`priors.py`, `rq2_experiment.py`)

**RQ2 (revised wording):** "How does the choice of parameter-estimation
method and Dirichlet prior affect recovery of the true data-generating
probability distribution, and downstream fraud-detection performance,
under different levels of fraud imbalance?"

(Earlier drafts described this as recovering "the true CPT probabilities".
That overstated what is actually measured: `KL(P_true || P_method)` is
computed over complete observations using each model's full BN
factorisation -- it is distribution-level recovery, not a direct
cell-by-cell comparison of individual CPT entries. A literal CPT
comparison is not always even well-defined here, since the LEARNED
structure can give a node different parents than the TRUE structure does,
in which case the two CPTs are indexed by entirely different parent
configurations and cannot be compared entry-by-entry. Distribution-level
KL divergence sidesteps that problem by scoring whole rows instead.)

Given the structure G* learned in the main experiment, this parameterises
it using FOUR methods -- **MLE** (no prior), **Jeffreys** (`alpha=0.5`),
**Laplace** (`alpha=1.0`), and an **imbalanced-fraud-informed prior**
(previously labelled "expert-informed") -- and evaluates how close each
gets to the KNOWN TRUE distribution, plus the resulting fraud-detection
performance.

**On the "expert" prior's terminology and calibration:** this prior
centres `isFraud`'s pseudo-counts on `config.TARGET_FRAUD_RATE`, a FIXED
constant (0.5%) representing a known/published base rate -- analogous to
citing Nassiri et al. (2024) in the proposal's own methodology -- and
critically, it is **NOT** derived from the realised fraud rate of
whatever specific training sample happens to be fit on. An earlier
version used `train_df["isFraud"].mean()` for this, which was flagged
correctly as circular: an "expert belief" that just restates whatever the
current data sample happens to show is not an independent prior at all.
Using the same fixed constant that `ground_truth.py` itself uses to build
the true distribution keeps this prior genuinely external to any specific
draw, which is what the name is supposed to mean.

**A related bug was fixed earlier too:** the prior was originally applying
its alpha value *symmetrically* to both the fraud and legitimate states,
which only changes smoothing strength, not direction -- it never actually
encoded "fraud is rare" at all. Fixed to use an asymmetric Dirichlet
centred on the target rate, mirroring exactly how `ground_truth.py`
builds `isFraud`'s true CPT.

**Key finding (15 seeds, `PYTHONHASHSEED=0`, default 20,000 rows):**

| Method | Mean KL(true \|\| method) |
|---|---|
| MLE | worst |
| Jeffreys | mid |
| Imbalanced-fraud-informed | mid |
| Laplace | best |

(See `outputs/rq2_prior_comparison_results.csv` for the exact run's
numbers -- printed to stdout when `rq2_experiment.py` runs.)

**MLE is consistently the worst of the four at recovering the true
distribution, at every training-set size tested (500 to 20,000 rows)** --
clean, direct evidence that Bayesian parameter estimation with any of the
three priors genuinely outperforms raw MLE here. The data-scarcity sweep
shows this gap shrinking sharply as training data grows, confirming that
the choice of estimator matters most exactly when data is scarce.

**Downstream classification metrics (F1/AUROC) remain statistically
indistinguishable across all four methods** at the default row count --
reported honestly rather than hidden. Results are also split by whether
Hill Climbing found any learned edge into `isFraud` for that seed
(`has_fraud_edge`), since seeds with no such edge necessarily make every
method look identical -- this separates "did structure learning find a
fraud-relevant dependency at all" from "given that structure, how well
was it parameterised", which are two different questions.

**Run it:**
```bash
python rq2_experiment.py
```
Outputs `outputs/rq2_prior_comparison_results.csv`,
`outputs/rq2_rows_sweep_results.csv`, and
`outputs/figures/rq2_prior_comparison.png`.

## Competitor baselines and full comparison (`baselines.py`, `full_comparison.py`)

Implements the three competitor models from Section 3.4.2 of the
proposal -- Logistic Regression (l2, C=1.0), Random Forest (100 trees,
max depth 10, balanced class weights), and Naive Bayes (CategoricalNB,
the "Bayesian Network with no edges" structural lower bound) -- fit on
the exact same data as the Bayesian Network experiments, and evaluated
with the same metrics for a fair, apples-to-apples comparison.

**Result (15 seeds):** the discriminative baselines clearly outperform
the Bayesian Network on raw predictive power -- Random Forest achieves
roughly 4x the F1 score of any BN prior variant on average. This is the
expected accuracy/interpretability trade-off documented in Chapter 2 of
the proposal (Ali et al., 2022; West & Bhattacharya, 2016; Castaño & Mora,
2025) -- the Bayesian Network is not competitive on raw accuracy, but it
is the only model here whose reasoning can be read directly off the graph
structure. See `outputs/figures/full_comparison.png`.

**Run it:**
```bash
python full_comparison.py
```
Outputs `outputs/full_comparison_results.csv` and
`outputs/figures/full_comparison.png`.

## Running everything

`main.py` now runs all nine stages -- ground truth recovery, the fraud-rate
sweep, cross-distribution validation, RQ2's prior comparison (including the
data-scarcity sweep), and the full BN-vs-baselines comparison -- in one
command:

```bash
PYTHONHASHSEED=0 python main.py
```

This takes roughly 7-8 minutes end-to-end on a single core and regenerates
every CSV and figure in `outputs/` from scratch, bit-for-bit reproducibly.

## Learning curves (`learning_curves.py`, `prior_centre_check.py`, `plot_learning_curves.py`, `learning_curves.ipynb`)

Added after the supervisor asked for graphs of *KL divergence* and *F1* against the *number of
training samples*, one curve per estimator (MLE, Bayesian estimators with different priors),
and for RQ1 to be framed as "how good a surrogate for the true distribution is the learned
network" rather than "does it recover the true structure" (score-based learning can only find
an equivalence class of structures).

* Exact evaluation: the joint space is only 5^8 x 2 = 781,250 states, so KL(P_true || P_model) and the
  population F1 / AUROC of every fitted model are computed by enumerating all states (no sampling
  noise). The fitted probabilities were checked against `priors.py` / pgmpy (max difference 1e-16)
  and F1 / AUROC against scikit-learn's weighted metrics (difference 1e-17).
* 20 independent ground truths x 9 training sizes (100 ... 50,000). Training sets are nested prefixes.
* Estimators: MLE, Jeffreys, Laplace, and a fraud-informed prior of strength s in
  {1,2,3,4,10,25,50,100,200,500} centred on the base rate 0.005. Structures: the TRUE structure
  (isolates parameter estimation = RQ2), the structure learned by Hill Climbing + BIC, and the empty graph.
* `prior_centre_check.py` varies the believed base rate to test robustness of the fraud prior.

Run order: `learning_curves.py` (~7 min) -> `prior_centre_check.py` (~1 min) -> `plot_learning_curves.py`.
Outputs: `outputs/learning_curves_*.csv`, `outputs/figures/lc_*.png`, and the notebook walks through them.

### Added in the second round (uncertainty, prior strength, statistics)

* `strength_centre_grid.py` crosses prior strength s in {1,10,50,200,500} with the believed base rate in
  {0.0005, 0.005, 0.02, 0.05, 0.5} (true rate 0.005): is a stronger prior always better, and does its
  influence fade as n grows?  (`outputs/learning_curves_strength_centre_grid.csv`, `lc_strength_by_belief.png`)
* `learning_curve_analysis.py`: convergence speed, 108 paired tests (Wilcoxon, Holm-corrected, with mean
  differences, 95% CIs and paired effect sizes), a direction-aware "prior vs MLE" verdict table, and a summary of the
  learned-structure (Hill Climbing) results.
* All shaded bands in the learning-curve figures are 95% t-confidence intervals across the 20 ground truths
  (the RQ1 figure uses median and inter-quartile range because greedy search occasionally lands in a poor optimum).

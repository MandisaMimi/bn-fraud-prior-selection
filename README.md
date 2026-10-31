# Bayesian Network Structure Learning and Prior Selection Under Class Imbalance

Honours research project, School of Computer Science and Applied Mathematics, University of the Witwatersrand.
Author: Mandisa N. Mashinini. Supervisor: Ritesh Ajoodha.

Synthetic worlds with a known true distribution allow exact KL divergence, F1 and AUROC (all 781,250 transactions are enumerated).
The main findings are checked on a second synthetic mechanism and on the public credit-card fraud data.

## Main findings
1. Hill Climbing with BIC approximates the true distribution only from about 20,000 rows (0.5% fraud). A direct test (`code/mechanism.py bic`) shows the likelihood gain of a true fraud edge stays below the BIC penalty until roughly 20,000-50,000 rows.
2. Any prior beats maximum likelihood when data are scarce. With identical smoothing on all non-fraud nodes (`code/mechanism.py fair`) a fraud-rate prior is closer to the truth than Laplace smoothing at every size tested.
3. In the tested synthetic settings the benefit of the fraud prior follows k = n*pi/q (expected fraud rows per parent configuration of the fraud node): largest below 1, small up to 3, negligible above about 10. These are exploratory guidelines; the thresholds are not validated on real data.
4. Learned graphs give the fraud node few parents, so the prior and structure learning compete; naive Bayes/TAN need only ordinary smoothing on real data.

## Layout
- `report/` - `Research_Report.pdf`, LaTeX source `main.tex` (elsarticle; run `pdflatex main.tex` twice), tables `tab_*.tex`, figures `figs/`.
- `code/` - all experiments. `code/outputs/` holds the result CSVs used in the report.
- `data/` - how to obtain the one external dataset (not stored here).

## Where each result comes from
| Report part | Script | Output |
|---|---|---|
| Learning curves (structure and parameters) | `learning_curves.py`, `learning_curve_analysis.py` | `outputs/learning_curves_*.csv` |
| Fraud-rate sweep, Holm-corrected maps | `prior_by_imbalance.py`, `corrected_map.py` | `outputs/prior_by_imbalance_results.csv`, `map_*.csv` |
| Robustness (floor, MLE convention, thresholds, controlled predictability) | `robustness.py` | `outputs/robust_*.csv`, `ctrl_*.csv` |
| Other model families | `baselines_exact.py` | `outputs/baselines_exact_results.csv` |
| BIC test of fraud edges; matched smoothing | `mechanism.py bic` / `mechanism.py fair` | `outputs/bic_mechanism.csv`, `outputs/fair_smoothing.csv` |
| Step-by-step Hill Climbing trace (synthetic and real) | `search_trace.py synth` / `search_trace.py real` | `outputs/trace_synth_*.csv`, `outputs/trace_real_*.csv` |
| Validation of our search against pgmpy (BIC, equivalence, fitted models) | `validate_search.py` | `outputs/validate_search.csv` |
| Source of the prior's rate; per-configuration fraud counts vs k | `rate_and_k.py` | `outputs/rate_source.csv`, `outputs/kmeasures.csv` |
| Transition table and figure | `transition.py` | `outputs/transition_table.csv`, `report/tab_mech.tex`, `report/figs/mechanism.png` |
| k-collapse and strength analysis | `collapse.py` | `outputs/collapse_*.csv`, `strength_collapse.csv` |
| Learned structures, latent-class generator | `structalt.py orig` / `structalt.py latent` | `outputs/structalt_orig.csv`, `latent_results.csv` |
| Real data | `realdata.py` | `outputs/real_results.csv`, `real_tests.csv` |
| Report tables (guide, strength, structures, matched, real) | `make_tabs.py` | `report/tab_*.tex` |

## Reproducing
    python -m venv .venv && source .venv/bin/activate
    pip install -r code/requirements.txt scikit-learn
    cd code
    python mechanism.py bic outputs/bic_mechanism.csv        # seconds
    python mechanism.py fair outputs/fair_smoothing.csv      # about 3 minutes
    python search_trace.py synth outputs/trace_synth         # about 1 minute
    python validate_search.py outputs/validate_search.csv    # about 1 minute (use PYTHONHASHSEED=0)
    python rate_and_k.py outputs/rate_source.csv outputs/kmeasures.csv   # about 3 minutes
    python structalt.py orig outputs/structalt_orig.csv      # about 10 minutes
    python structalt.py latent outputs/latent_results.csv    # about 15 minutes
    python realdata.py ../data/creditcard.csv outputs/real_results.csv 30   # needs data/creditcard.csv
    python search_trace.py real ../data/creditcard.csv outputs/trace_real 30
    python transition.py && python make_tabs.py              # rebuild report tables and figure
Set `export PYTHONHASHSEED=0` first. Run all commands from `code/` (scripts use relative paths). Experiments use fixed seeds (42-61).

Reproduction check (clean virtual environment, `PYTHONHASHSEED=0`): `mechanism.py bic`, `mechanism.py fair`, `search_trace.py synth` (our own search) and `rate_and_k.py` reproduced their stored CSVs exactly, and `transition.py` and `make_tabs.py` regenerate every table in `report/` identically from the stored CSVs. `structalt.py orig`, `structalt.py latent` and `realdata.py` were rerun from scratch: results for naive Bayes, TAN and the fixed-parent structure were identical, but results for pgmpy's Hill Climbing change slightly with the hash seed (tie-breaking), so the stored CSVs of those three experiments are the clean-environment `PYTHONHASHSEED=0` runs and the report tables were rebuilt from them. Always set `PYTHONHASHSEED=0`. The learning-curve experiment (`learning_curves.py 20`, 3,240 rows, includes pgmpy Hill Climbing) and the fraud-rate experiment (`prior_by_imbalance.py 20`, 9,900 rows) were also rerun from scratch in the clean environment and reproduced the stored CSVs exactly (maximum difference 0 in every metric). `robustness.py` and `baselines_exact.py` were not rerun from scratch.
The columns `same_graph`, `skeleton_same` and `jaccard` in `trace_synth_runs.csv` compare our search with pgmpy and depend on that tie-breaking; `validate_search.py` (run with `PYTHONHASHSEED=0`) quantifies the agreement (Markov-equivalent graphs and BIC) and the fraud decision agreed in every run.

## Notes
The real credit-card data are anonymised principal components, so no ground truth exists there and KL is not computed on them.
Tests in the report that are not Holm-corrected are labelled exploratory.

"""
prior_centre_check.py
=====================
Robustness check for the fraud-informed prior: what if the base rate you BELIEVE
(the centre of the prior) is wrong?  Strength is fixed at 50 pseudo-samples; the
believed fraud rate r is varied around the true generating rate (0.005).
True structure, exact evaluation, same ground truths and data as learning_curves.py.
"""
import numpy as np
import pandas as pd
import config, ground_truth, data_generation
import learning_curves as lc

CENTRES = [0.0005, 0.001, 0.005, 0.02, 0.1, 0.5]
STRENGTH = 50

rows = []
for i in range(lc.N_SEEDS_LC):
    seed = lc.BASE_SEED + i
    gt = ground_truth.build_ground_truth(seed)
    ctx = lc.TruthContext(gt)
    X_all = data_generation.generate_dataset(gt, max(lc.N_GRID), np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy()
    for n in lc.N_GRID:
        X = X_all[:n]
        for r in CENTRES:
            kl, f1, auroc = ctx.score(lc.fit_tables(X, gt["parents"], lc.alpha_fraud_prior(STRENGTH, rate=r)), gt["parents"], ctx.cache)
            rows.append({"seed": seed, "n": n, "believed_rate": r, "kl": kl, "f1": f1, "auroc": auroc})
    print("seed", seed, "done", flush=True)
pd.DataFrame(rows).to_csv("outputs/learning_curves_prior_centre.csv", index=False)

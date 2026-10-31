"""
strength_centre_grid.py
=======================
Does an over-strong prior hurt when it is WRONG, and does its influence fade as n grows?
Crosses the prior STRENGTH s with the BELIEVED base rate r (true rate = 0.005).
True structure (isolates parameter estimation), exact evaluation, same ground truths/data
as learning_curves.py (20 seeds, nested training prefixes).
"""
import numpy as np, pandas as pd
import config, ground_truth, data_generation
import learning_curves as lc

N_SUB = [100, 500, 2000, 5000, 20000, 50000]
STRENGTHS = [1, 10, 50, 200, 500]
CENTRES = [0.0005, 0.005, 0.02, 0.05, 0.5]      # 10x low, true, 4x high, 10x high, 100x high

rows = []
for i in range(lc.N_SEEDS_LC):
    seed = lc.BASE_SEED + i
    gt = ground_truth.build_ground_truth(seed)
    ctx = lc.TruthContext(gt)
    X_all = data_generation.generate_dataset(gt, max(lc.N_GRID), np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy()
    for n in N_SUB:
        X = X_all[:n]
        for s in STRENGTHS:
            for r in CENTRES:
                kl, f1, auroc = ctx.score(lc.fit_tables(X, gt["parents"], lc.alpha_fraud_prior(s, rate=r)), gt["parents"], ctx.cache)
                rows.append({"seed": seed, "n": n, "strength": s, "believed_rate": r, "kl": kl, "f1": f1, "auroc": auroc})
    print("seed", seed, flush=True)
    pd.DataFrame(rows).to_csv("outputs/learning_curves_strength_centre_grid.csv", index=False)

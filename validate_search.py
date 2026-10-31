"""
validate_search.py -- is our step-by-step Hill Climbing equivalent to pgmpy's?  Same data as search_trace.py (20 worlds, 9 sizes, 0.5% fraud).
For each run: BIC of both learned graphs (our scorer, same formula as pgmpy's BIC), whether the graphs have the same skeleton,
whether they are Markov equivalent (same skeleton and same v-structures), and the KL / AUROC of the Laplace-fitted models from each graph.
Usage: python validate_search.py out.csv
"""
import sys, time, itertools, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import config, ground_truth, data_generation, learning_curves as lc
from structure_learning import learn_structure
from search_trace import hill_climb, make_scorer

NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
def vstructs(edges):
    par = {}
    for a, b in edges: par.setdefault(b, set()).add(a)
    sk = {frozenset(e) for e in edges}; out = set()
    for c, ps in par.items():
        for a, b in itertools.combinations(sorted(ps), 2):
            if frozenset((a, b)) not in sk: out.add((a, c, b))
    return out

def one(seed):
    gt = ground_truth.build_ground_truth(seed); ctx = lc.TruthContext(gt)
    Xall = data_generation.generate_dataset(gt, max(NS), np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy(); rows = []
    for n in NS:
        X = Xall[:n]; fam = make_scorer(X, lc.K)
        mine, _ = hill_climb(X, lc.K, lc.FRAUD)
        theirs = {(lc.IDX[a], lc.IDX[b]) for a, b in learn_structure(pd.DataFrame(X, columns=lc.NODES))}
        def bic(E):
            par = {v: tuple(sorted(a for a, b in E if b == v)) for v in range(len(lc.NODES))}; return sum(fam(v, par[v]) for v in par)
        def fit(E):
            pm = lc.parents_map_from_edges([(lc.NODES[a], lc.NODES[b]) for a, b in E]); return ctx.score(lc.fit_tables(X, pm, lc.alpha_uniform(1.0)), pm, lc.make_cache(pm))
        km, fm, am = fit(mine); kt, ft, at = fit(theirs)
        rows.append(dict(seed=seed, n=n, bic_mine=bic(mine), bic_pgmpy=bic(theirs), same_graph=mine == theirs,
                         same_skeleton={frozenset(e) for e in mine} == {frozenset(e) for e in theirs},
                         markov_equivalent=({frozenset(e) for e in mine} == {frozenset(e) for e in theirs}) and vstructs(mine) == vstructs(theirs),
                         edges_mine=len(mine), edges_pgmpy=len(theirs), kl_mine=km, kl_pgmpy=kt, auroc_mine=am, auroc_pgmpy=at,
                         fraud_linked_mine=any(lc.FRAUD in e for e in mine), fraud_linked_pgmpy=any(lc.FRAUD in e for e in theirs)))
    return rows

if __name__ == "__main__":
    t0 = time.time(); R = []
    with Pool(2) as p:
        for i, r in enumerate(p.imap_unordered(one, [42 + i for i in range(20)]), 1): R += r; print(i, 20, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(R).to_csv(sys.argv[1], index=False); print("done")

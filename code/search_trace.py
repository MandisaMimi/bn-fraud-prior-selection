"""
search_trace.py -- follow the actual Hill Climbing + BIC search step by step.
 Own implementation of greedy hill climbing (add / remove / reverse edge, max in-degree 4, empty start, stop when the best
 move improves BIC by less than 1e-4, short tabu list), recording at every step the accepted move and the best move that
 involves the fraud node.  Checked against pgmpy's HillClimbSearch on the same data (agreement is reported).
Usage: python search_trace.py synth out_prefix | real creditcard.csv out_prefix [n_repeats]
 synth: original worlds at 0.5% fraud, 20 worlds, nine sample sizes   -> out_prefix_runs.csv, out_prefix_steps.csv
 real : credit-card data, 30 repeats, five sizes (as realdata.py)      -> out_prefix_runs.csv, out_prefix_steps.csv
"""
import sys, time, itertools, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings("ignore")

def make_scorer(X, K):
    n = len(X); cache = {}
    def fam(v, ps):
        key = (v, ps)
        if key in cache: return cache[key]
        k = int(K[v]); nc = 1; ci = np.zeros(n, dtype=np.int64)
        for p in ps: ci = ci * int(K[p]) + X[:, p]; nc *= int(K[p])
        c = np.bincount(ci * k + X[:, v], minlength=nc * k).reshape(nc, k).astype(float)
        t = c.sum(1, keepdims=True); m = c > 0
        ll = float((c[m] * np.log((c / np.where(t > 0, t, 1))[m])).sum())
        cache[key] = ll - 0.5 * np.log(n) * nc * (k - 1)
        return cache[key]
    return fam

def reachable(par, src, dst):            # is dst an ancestor-descendant: path src -> ... -> dst following child links?
    ch = {}
    for c, ps in par.items():
        for p in ps: ch.setdefault(p, []).append(c)
    seen, st = set(), [src]
    while st:
        u = st.pop()
        if u == dst: return True
        for w in ch.get(u, []):
            if w not in seen: seen.add(w); st.append(w)
    return False

def hill_climb(X, K, fraud, max_in=4, eps=1e-4):
    d = X.shape[1]; fam = make_scorer(X, K); par = {v: () for v in range(d)}
    steps, tabu = [], []
    while True:
        cands = []
        for u, v in itertools.permutations(range(d), 2):
            if u in par[v]:
                cands.append((fam(v, tuple(p for p in par[v] if p != u)) - fam(v, par[v]), "remove", u, v))
                # reverse u->v into v->u
                np_v = tuple(p for p in par[v] if p != u); tmp = {**par, v: np_v}
                if len(par[u]) < max_in and not reachable(tmp, u, v):
                    cands.append((fam(v, np_v) - fam(v, par[v]) + fam(u, tuple(sorted(par[u] + (v,)))) - fam(u, par[u]), "flip", u, v))
            else:
                if len(par[v]) < max_in and not reachable(par, v, u):
                    cands.append((fam(v, tuple(sorted(par[v] + (u,)))) - fam(v, par[v]), "add", u, v))
        cands = [c for c in cands if (c[1], c[2], c[3]) not in tabu]
        if not cands: break
        best = max(cands, key=lambda c: c[0])
        fc = [c for c in cands if fraud in (c[2], c[3])]
        bf = max(fc, key=lambda c: c[0]) if fc else None
        steps.append(dict(step=len(steps), op=best[1], u=best[2], v=best[3], delta=best[0], best_fraud_op=bf[1] if bf else "", best_fraud_delta=bf[0] if bf else np.nan,
                          fraud_move=bool(fraud in (best[2], best[3]))))
        if best[0] <= eps:
            steps[-1]["accepted"] = False; break
        steps[-1]["accepted"] = True
        u, v, op = best[2], best[3], best[1]
        if op == "add": par[v] = tuple(sorted(par[v] + (u,)))
        elif op == "remove": par[v] = tuple(p for p in par[v] if p != u)
        else:
            par[v] = tuple(p for p in par[v] if p != u); par[u] = tuple(sorted(par[u] + (v,)))
        tabu.append(({"add": "remove", "remove": "add", "flip": "flip"}[op], u if op != "flip" else v, v if op != "flip" else u)); tabu = tabu[-100:]
        if len(steps) > 500: break
    edges = {(p, c) for c, ps in par.items() for p in ps}
    return edges, steps

def run_synth(seed):
    import config, ground_truth, data_generation, learning_curves as lc
    from structure_learning import learn_structure
    NS = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    gt = ground_truth.build_ground_truth(seed)
    Xall = data_generation.generate_dataset(gt, max(NS), np.random.RandomState(seed + 10_000))[lc.NODES].to_numpy()
    fr = lc.FRAUD; runs, steps = [], []
    for n in NS:
        X = Xall[:n]; edges, st = hill_climb(X, lc.K, fr)
        pg = {(lc.IDX[a], lc.IDX[b]) for a, b in learn_structure(pd.DataFrame(X, columns=lc.NODES))}
        acc = [s for s in st if s["accepted"]]
        first = next((s["step"] for s in acc if s["fraud_move"] and s["op"] == "add"), np.nan)
        last = st[-1]
        runs.append(dict(seed=seed, n=n, n_fraud=int(X[:, fr].sum()), n_steps=len(acc), fraud_linked=any(fr in e for e in edges), n_fraud_edges=sum(fr in e for e in edges),
                         first_fraud_step=first, stop_best_fraud_delta=last["best_fraud_delta"], stop_best_fraud_op=last["best_fraud_op"],
                         pgmpy_fraud_linked=any(fr in e for e in pg), same_graph=(edges == pg), skeleton_same=({frozenset(e) for e in edges} == {frozenset(e) for e in pg}),
                         jaccard=len(edges & pg) / max(1, len(edges | pg))))
        for s in st: steps.append(dict(seed=seed, n=n, **s))
    return runs, steps

def run_real(args):
    rep, csv = args
    from sklearn.metrics import mutual_info_score
    from sklearn.model_selection import train_test_split
    D = pd.read_csv(csv); COLS = [c for c in D.columns if c != "Class"]
    tr, te = train_test_split(D, test_size=0.3, stratify=D.Class, random_state=1000 + rep)
    def disc(c):
        e = np.unique(np.quantile(tr[c].values, [.2, .4, .6, .8])); return np.searchsorted(e, tr[c].values, side="right"), len(e) + 1
    d = {c: disc(c) for c in COLS}
    mi = {c: mutual_info_score(d[c][0], tr.Class) for c in COLS}
    feats = sorted(mi, key=mi.get, reverse=True)[:8]
    dtr = pd.DataFrame({c: d[c][0] for c in feats}, index=tr.index); K = [d[c][1] for c in feats] + [2]; dtr["Class"] = tr.Class.values
    names = feats + ["Class"]; fr = 8; rng = np.random.RandomState(rep); runs, steps = [], []
    for n in [1000, 5000, 20000, 60000, 0]:
        sub = dtr.sample(n=len(dtr) if n == 0 else n, random_state=rng.randint(1 << 30)); X = sub[names].to_numpy()
        edges, st = hill_climb(X, np.array(K), fr)
        acc = [s for s in st if s["accepted"]]; last = st[-1]
        conn = sorted(feats.index(names[a if b == fr else b]) + 1 for a, b in edges if fr in (a, b))   # MI rank of connected features
        runs.append(dict(rep=rep, n=len(sub), n_fraud=int(sub.Class.sum()), n_steps=len(acc), fraud_linked=bool(conn), n_fraud_edges=len(conn), connected_ranks=" ".join(map(str, conn)),
                         stop_best_fraud_delta=last["best_fraud_delta"], stop_best_fraud_op=last["best_fraud_op"]))
        for s in st: steps.append(dict(rep=rep, n=len(sub), **s))
    return runs, steps

if __name__ == "__main__":
    mode = sys.argv[1]; t0 = time.time(); R, S = [], []
    if mode == "synth":
        out = sys.argv[2]; tasks = [42 + i for i in range(20)]; fn = run_synth
    else:
        csv, out = sys.argv[2], sys.argv[3]; nrep = int(sys.argv[4]) if len(sys.argv) > 4 else 30; tasks = [(r, csv) for r in range(nrep)]; fn = run_real
    with Pool(2) as p:
        for k, (r, s) in enumerate(p.imap_unordered(fn, tasks), 1):
            R += r; S += s; print(k, len(tasks), f"{time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(R).to_csv(out + "_runs.csv", index=False); pd.DataFrame(S).to_csv(out + "_steps.csv", index=False)

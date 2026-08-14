"""E7 — inference fidelity against exact marginals (PROTOCOL.md §C.4, §I.5).

Scores mean-field, damped loopy BP and chromatic Gibbs against EXACT marginals on graphs small
enough to solve exactly, as a function of coupling strength and graph structure. This is
simultaneously (a) the validation layer for every inference routine used at scale — no
real-data run is trusted until BP reproduces exact marginals on trees and every engine
reproduces softmax(s) at beta=0 — and (b) a measurement of the bias/variance split between
deterministic approximations (MF, BP: bias that no extra compute removes) and Monte-Carlo
approximation (Gibbs: variance that shrinks with sweeps).

Framing note: this replicates well-known behaviour (Murphy, Weiss & Jordan 1999); it is
reported as validation and characterisation, never as a novel finding.

Usage:  python -m experiments.run_fidelity [--structure NAME] [--draws N]
"""
import argparse
import time

import torch

from src.inference.exact import exact_marginals
from src.inference.gibbs import gibbs
from src.inference.loopy_bp import loopy_bp
from src.inference.mean_field import mean_field
from src.runlog import ROOT, append_fidelity
from src.synthetic.small_graphs import (balanced_tree, chain, cycle, dense_sbm, grid,
                                        make_unaries, star)

BETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.3, 1.75, 2.5, 4.0]
ALPHAS = [0.5, 1.5]
DRAW_SEED0 = 200
ENUM_CUTOFF = 5e6

STRUCTURES = {
    "chain": (lambda s: (*chain(20), None), 3),
    "tree":  (lambda s: (*balanced_tree(15), None), 3),
    "star":  (lambda s: (*star(16), None), 3),
    "cycle": (lambda s: (*cycle(16), None), 3),
    "grid":  (lambda s: (*grid(4, 4), None), 3),
    "grid7": (lambda s: (*grid(4, 4), None), 7),
    "dense": (lambda s: dense_sbm(14, seed=s), 3),
}


def tv_stats(q, ex, informative):
    tv = 0.5 * (q - ex).abs().sum(1)
    return {
        "tv_mean": float(tv.mean()), "tv_max": float(tv.max()),
        "tv_mean_informative": float(tv[informative].mean()) if informative.any() else "",
        "tv_mean_uninformative": float(tv[~informative].mean()) if (~informative).any() else "",
    }


def _shard(name):
    return ROOT / "results" / "raw" / f"fidelity_{name}.csv"


def run_structure(name: str, draws: int):
    out = _shard(name)
    builder, C = STRUCTURES[name]
    for draw in range(draws):
        seed = DRAW_SEED0 + draw
        ei, n, labels = builder(seed)
        exact_method = "enumerate" if C**n <= ENUM_CUTOFF else "ve"
        for alpha in ALPHAS:
            s, y, informative = make_unaries(n, C, alpha, seed=seed, labels=labels)
            for beta in BETAS:
                t0 = time.perf_counter()
                ex = exact_marginals(s, ei, beta, method=exact_method)
                t_exact = time.perf_counter() - t0
                base = dict(structure=name, n=n, C=C, beta=beta, alpha=alpha, draw_seed=seed,
                            exact_method=exact_method)
                append_fidelity(path=out, row={**base, "method": "exact", "tv_mean": 0.0, "tv_max": 0.0,
                                 "converged": 1, "wall_seconds": round(t_exact, 4)})

                t0 = time.perf_counter()
                q, info = mean_field(s, ei, beta, check_elbo=False)
                append_fidelity(path=out, row={**base, "method": "mf", **tv_stats(q, ex, informative),
                                 "converged": int(info["converged"]), "iters": info["iters"],
                                 "wall_seconds": round(time.perf_counter() - t0, 4)})

                t0 = time.perf_counter()
                q, info = loopy_bp(s, ei, beta)
                append_fidelity(path=out, row={**base, "method": "lbp", **tv_stats(q, ex, informative),
                                 "converged": int(info["converged"]), "iters": info["iters"],
                                 "final_residual": info["final_residual"],
                                 "wall_seconds": round(time.perf_counter() - t0, 4)})

                t0 = time.perf_counter()
                q, info = gibbs(s, ei, beta, chains=4, burn_in=1000, kept=2000, seed=seed)
                append_fidelity(path=out, row={**base, "method": "gibbs", **tv_stats(q, ex, informative),
                                 "converged": int(info["converged"]), "iters": info["iters"],
                                 "ess_min": round(info["ess_min"], 1),
                                 "rhat_max": round(info["rhat_max"], 5), "gibbs_kept": 2000,
                                 "wall_seconds": round(time.perf_counter() - t0, 4)})
        print(f"  {name}: draw {draw + 1}/{draws} done", flush=True)


def run_budget_sweep(draws: int = 10, beta: float = 1.75):
    out = _shard("budget")
    """Bias-vs-variance panel: Gibbs error against sweep count, with MF/LBP as bias floors."""
    ei, n = grid(4, 4)
    for draw in range(draws):
        seed = DRAW_SEED0 + draw
        s, y, informative = make_unaries(n, 3, 1.5, seed=seed)
        ex = exact_marginals(s, ei, beta, method="ve")
        base = dict(structure="grid_budget", n=n, C=3, beta=beta, alpha=1.5, draw_seed=seed,
                    exact_method="ve")
        for engine, fn in (("mf", lambda: mean_field(s, ei, beta, check_elbo=False)),
                           ("lbp", lambda: loopy_bp(s, ei, beta))):
            q, info = fn()
            append_fidelity(path=out, row={**base, "method": engine, **tv_stats(q, ex, informative),
                             "converged": int(info["converged"]), "gibbs_kept": ""})
        for kept in (10, 30, 100, 300, 1000, 3000):
            t0 = time.perf_counter()
            q, info = gibbs(s, ei, beta, chains=4, burn_in=1000, kept=kept, seed=seed)
            append_fidelity(path=out, row={**base, "method": "gibbs", **tv_stats(q, ex, informative),
                             "converged": int(info["converged"]), "gibbs_kept": kept,
                             "ess_min": round(info["ess_min"], 1),
                             "rhat_max": round(info["rhat_max"], 5),
                             "wall_seconds": round(time.perf_counter() - t0, 4)})
        print(f"  budget sweep: draw {draw + 1}/{draws} done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--structure", default=None, choices=list(STRUCTURES) + ["budget"])
    ap.add_argument("--draws", type=int, default=20)
    a = ap.parse_args()
    if a.structure == "budget":
        run_budget_sweep(draws=min(a.draws, 10))
    elif a.structure:
        run_structure(a.structure, a.draws)
    else:
        for k in STRUCTURES:
            run_structure(k, a.draws)
        run_budget_sweep()

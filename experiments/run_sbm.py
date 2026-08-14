"""E5 — the contextual-SBM homophily sweep (PROTOCOL.md §I.4). The study's centrepiece.

A fresh graph is drawn per seed, so unlike the real-graph blocks this experiment has genuine
graph-level replication. Expected degree, class balance, feature dimension and feature noise
are identical at every h; only the *truth of the homophily prior* changes.

Both prior regimes come from the same runs: every beta on the grid is logged, and analysis
derives (a) the TUNED regime — beta selected on validation per cell, which measures the prior
as an *option* — and (b) the DOGMATIC regime — beta fixed at the value tuned at (h=0.9, m=5)
and applied everywhere, which measures the prior as an *assumption*. The distinction matters:
a tuned prior can always retreat towards beta=0 when it is wrong, so a harm region can only
appear when the prior is committed to in advance.

Reference points logged per cell: Oracle-F (closed-form feature-only Bayes rule) and Oracle-G
(true Gaussian unaries + the generator-implied coupling beta_gen = log(p_in/p_out)).

Usage: python -m experiments.run_sbm --h 0.9 [--seeds 0-9] [--m 1,5,20]
"""
import argparse

import torch

from experiments.common import run_posthoc, run_trained
from experiments.run_real import parse_seeds
from src.baselines.label_prop import label_prop
from src.config import BETA_GRID, LAMBDA_GRID, SBM_H, SBM_M, SIGMA_X, STAGE1
from src.infer import posterior
from src.metrics import all_metrics
from src.runlog import append_run, new_run_id
from src.splits import masks_sbm
from src.synthetic.oracle import oracle_f_accuracy, oracle_f_predictions, oracle_g_unaries
from src.synthetic.sbm import check_graph, make_csbm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--m", default=None)
    a = ap.parse_args()

    hs = [a.h] if a.h is not None else SBM_H
    ms = [int(x) for x in a.m.split(",")] if a.m else SBM_M
    seeds = parse_seeds(a.seeds)
    hp = STAGE1["sbm"]

    for h in hs:
        for seed in seeds:
            data = make_csbm(h, seed=seed, sigma_x=SIGMA_X)
            chk = check_graph(data)
            gstats = dict(target_h=h, empirical_h=round(chk["empirical_h"], 5),
                          adjusted_h=round(chk["adjusted_h"], 5),
                          label_informativeness=round(chk["label_informativeness"], 5),
                          sigma_x=SIGMA_X, beta_gen=round(data.beta_gen, 5))
            if not (chk["degree_ok"] and chk["homophily_ok"] and chk["balance_ok"]):
                print(f"  !! graph check FAILED h={h} seed={seed}: {chk}", flush=True)

            for m in ms:
                tr, va, te = masks_sbm(data.y, m, seed)
                masks = {"train": tr, "val": va, "test": te}
                kw = dict(data=data, masks=masks, seed=seed, m=m, dataset="sbm")

                logits_mlp, _ = run_trained("mlp", model_label="M0_mlp", hp=hp["mlp"],
                                            extra=gstats, **kw)
                logits_gcn, _ = run_trained("gcn", model_label="M1_gcn", hp=hp["gcn"],
                                            extra=gstats, **kw)

                q = label_prop(data.edge_index, data.y, tr, data.x.shape[0], data.num_classes)
                mt = all_metrics(q, data.y, {"val": va, "test": te}, data.edge_index)
                append_run(dict(run_id=new_run_id("LP"), dataset="sbm", model="LP",
                                regime="closed_form", inference_method="label_prop",
                                label_per_class=m, split_seed=seed, **gstats,
                                notes="uses train labels as inference-time evidence",
                                **{k: round(v, 6) for k, v in mt.items()}))

                for lam in LAMBDA_GRID:
                    run_trained("gcn", model_label="M2_lap", lam=lam, reg="laplacian",
                                hp=hp["gcn"], save=False, extra=gstats, **kw)

                for beta in BETA_GRID:
                    run_posthoc(logits_gcn, model_label="M4_mrf_gcn", beta=beta,
                                extra=gstats, **kw)
                    run_posthoc(logits_mlp, model_label="M4_mrf_mlp", beta=beta,
                                extra=gstats, **kw)

                # --- reference ceilings (no training, generative parameters known) ---
                pred = oracle_f_predictions(data)
                append_run(dict(run_id=new_run_id("OF"), dataset="sbm", model="Oracle_F",
                                regime="oracle", inference_method="closed_form",
                                label_per_class=m, split_seed=seed, **gstats,
                                val_acc=round(float((pred[va] == data.y[va]).double().mean()), 6),
                                test_acc=round(float((pred[te] == data.y[te]).double().mean()), 6),
                                notes="feature-only Bayes rule, known generative parameters"))

                sg = oracle_g_unaries(data)
                qg, ig = posterior(sg, data.edge_index, data.beta_gen, engine="mf")
                mg = all_metrics(qg, data.y, {"val": va, "test": te}, data.edge_index)
                append_run(dict(run_id=new_run_id("OG"), dataset="sbm", model="Oracle_G",
                                regime="oracle", inference_method="mf", beta=round(data.beta_gen, 5),
                                label_per_class=m, split_seed=seed, **gstats,
                                converged=int(ig["converged"]), iters=ig["iters"],
                                notes="true Gaussian unaries + generator-implied coupling",
                                **{k: round(v, 6) for k, v in mg.items()}))
            print(f"[sbm] h={h} seed={seed} done", flush=True)


if __name__ == "__main__":
    main()

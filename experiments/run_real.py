"""E1–E4, E6 — the real-graph block (Cora, CiteSeer).

For every (m, seed) the SAME split and the SAME trained checkpoints feed every model, so all
comparisons are paired and the post-hoc MRF variants carry zero extra training variance.

Every grid point (each lambda, each beta) is written to runs.csv; hyperparameter selection
happens later in analysis as a pure function of validation accuracy. That keeps the full
sensitivity curves (E3) as a by-product and makes selection auditable rather than buried in a
runner.

Models: M0 MLP · M1 GCN · LP (labels+graph, no features) · M2 GCN+Laplacian · M3 GCN+rule
(Cora only) · M4-GCN and M4-MLP (Potts MRF, mean-field, on frozen logits) · clamped variant
(reported separately — it uses training labels as inference-time evidence).

Usage: python -m experiments.run_real --dataset cora [--seeds 0-9] [--m 1,2,5,10,20]
"""
import argparse

import torch

from experiments.common import run_posthoc, run_trained
from src.baselines.label_prop import label_prop
from src.config import BETA_GRID, CITESEER_M, CORA_M, LAMBDA_GRID, STAGE1
from src.data import load_planetoid
from src.metrics import all_metrics
from src.runlog import append_run, new_run_id
from src.splits import train_mask_real


def parse_seeds(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora", choices=["cora", "citeseer"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--m", default=None)
    ap.add_argument("--models", default="all",
                    help="comma list from mlp,gcn,lp,lap,rule,mrf_gcn,mrf_mlp,clamped")
    a = ap.parse_args()

    data = load_planetoid(a.dataset)
    ms = ([int(x) for x in a.m.split(",")] if a.m else
          (CORA_M if a.dataset == "cora" else CITESEER_M))
    seeds = parse_seeds(a.seeds)
    want = set(a.models.split(",")) if a.models != "all" else {
        "mlp", "gcn", "lp", "lap", "rule", "mrf_gcn", "mrf_mlp", "clamped"}
    if a.dataset != "cora":
        want.discard("rule")            # M3 is a Cora-only ablation (protocol scope decision)
    hp = STAGE1[a.dataset]

    for m in ms:
        for seed in seeds:
            masks = {"train": train_mask_real(data, m, seed), "val": data.val_mask,
                     "test": data.test_mask}
            kw = dict(data=data, masks=masks, seed=seed, m=m, dataset=a.dataset)

            logits_mlp = logits_gcn = None
            if want & {"mlp", "mrf_mlp"}:
                logits_mlp, _ = run_trained("mlp", model_label="M0_mlp", hp=hp["mlp"], **kw)
            if want & {"gcn", "mrf_gcn", "clamped"}:
                logits_gcn, _ = run_trained("gcn", model_label="M1_gcn", hp=hp["gcn"], **kw)

            if "lp" in want:
                q = label_prop(data.edge_index, data.y, masks["train"], data.x.shape[0],
                               data.num_classes)
                mt = all_metrics(q, data.y, {"val": masks["val"], "test": masks["test"]},
                                 data.edge_index)
                append_run(dict(run_id=new_run_id("LP"), dataset=a.dataset, model="LP",
                                regime="closed_form", inference_method="label_prop",
                                label_per_class=m, split_seed=seed,
                                notes="uses train labels as inference-time evidence",
                                **{k: round(v, 6) for k, v in mt.items()}))

            for lam in (LAMBDA_GRID if "lap" in want else []):
                run_trained("gcn", model_label="M2_lap", lam=lam, reg="laplacian",
                            hp=hp["gcn"], save=False, **kw)
            for lam in (LAMBDA_GRID if "rule" in want else []):
                run_trained("gcn", model_label="M3_rule", lam=lam, reg="rule",
                            hp=hp["gcn"], save=False, **kw)

            for beta in BETA_GRID:
                if "mrf_gcn" in want:
                    run_posthoc(logits_gcn, model_label="M4_mrf_gcn", beta=beta, **kw)
                if "mrf_mlp" in want:
                    run_posthoc(logits_mlp, model_label="M4_mrf_mlp", beta=beta, **kw)
                if "clamped" in want:
                    run_posthoc(logits_gcn, model_label="M4_mrf_gcn_clamped", beta=beta,
                                clamped=True, **kw)
        print(f"[{a.dataset}] m={m} done (seeds {seeds[0]}..{seeds[-1]})", flush=True)


if __name__ == "__main__":
    main()

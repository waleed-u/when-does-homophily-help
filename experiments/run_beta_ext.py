"""EXPLORATORY (post-freeze): extend the beta grid beyond the pre-registered maximum.

Validation selection lands on the grid boundary beta=2.0 for the MRF-on-MLP cells, which means
the pre-registered grid may be truncating the optimum and the reported gains are lower bounds.
This run measures how much is being left on the table.

Labelled exploratory per PROTOCOL.md §M: it is reported separately and never used to restate a
confirmatory endpoint.
"""
import argparse

from experiments.common import run_posthoc, run_trained
from experiments.run_real import parse_seeds
from src.config import SIGMA_X, STAGE1
from src.data import load_planetoid
from src.splits import masks_sbm, train_mask_real
from src.synthetic.sbm import make_csbm

EXT = [3.0, 4.0, 6.0, 8.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cora")
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--m", default="2,20")
    ap.add_argument("--h", type=float, default=0.90)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    ms = [int(x) for x in a.m.split(",")]

    if a.dataset == "cora":
        data = load_planetoid("cora")
        hp = STAGE1["cora"]
        for m in ms:
            for seed in seeds:
                masks = {"train": train_mask_real(data, m, seed), "val": data.val_mask,
                         "test": data.test_mask}
                kw = dict(data=data, masks=masks, seed=seed, m=m, dataset="cora_betaext")
                lm, _ = run_trained("mlp", model_label="M0_mlp", hp=hp["mlp"],
                                    extra={"notes": "exploratory beta extension"}, **kw)
                lg, _ = run_trained("gcn", model_label="M1_gcn", hp=hp["gcn"],
                                    extra={"notes": "exploratory beta extension"}, **kw)
                for beta in EXT:
                    run_posthoc(lm, model_label="M4_mrf_mlp", beta=beta, regime="exploratory",
                                extra={"notes": "exploratory beta extension"}, **kw)
                    run_posthoc(lg, model_label="M4_mrf_gcn", beta=beta, regime="exploratory",
                                extra={"notes": "exploratory beta extension"}, **kw)
            print(f"[betaext cora] m={m} done", flush=True)
    else:
        hp = STAGE1["sbm"]
        for seed in seeds:
            data = make_csbm(a.h, seed=seed, sigma_x=SIGMA_X)
            for m in ms:
                tr, va, te = masks_sbm(data.y, m, seed)
                masks = {"train": tr, "val": va, "test": te}
                kw = dict(data=data, masks=masks, seed=seed, m=m, dataset="sbm_betaext")
                lm, _ = run_trained("mlp", model_label="M0_mlp", hp=hp["mlp"],
                                    extra={"target_h": a.h, "notes": "exploratory"}, **kw)
                for beta in EXT:
                    run_posthoc(lm, model_label="M4_mrf_mlp", beta=beta, regime="exploratory",
                                extra={"target_h": a.h, "notes": "exploratory"}, **kw)
            print(f"[betaext sbm h={a.h}] seed={seed} done", flush=True)


if __name__ == "__main__":
    main()

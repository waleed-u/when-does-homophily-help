"""Stage-1 backbone tuning (PROTOCOL.md §F).

lr x weight-decay is tuned ONCE per (dataset, architecture) on validation accuracy at m=5, then
frozen for every model built on that architecture. The MLP gets its OWN budget rather than
inheriting the GCN's: a handicapped feature-only baseline would inflate every "the graph helps"
conclusion downstream.

Writes nothing to runs.csv — this is protocol setup, recorded in CHANGELOG addendum M3.
"""
import sys

import torch

from src.data import load_planetoid
from src.splits import train_mask_real
from src.synthetic.sbm import make_csbm
from src.splits import masks_sbm
from src.train import train_model

GRID = [(lr, wd) for lr in (0.01, 0.005) for wd in (5e-4, 1e-3)]
SEEDS = [0, 1, 2, 3, 4]


def tune(data, masks_fn, tag):
    print(f"\n{tag}")
    print(f"{'arch':>5} {'lr':>7} {'wd':>8} {'mean val acc':>13}")
    best = {}
    for arch in ("mlp", "gcn"):
        scores = []
        for lr, wd in GRID:
            vals = []
            for seed in SEEDS:
                masks = masks_fn(seed)
                logits, _ = train_model(arch, data if not callable(data) else data(seed),
                                        masks["train"], masks["val"], seed=seed, lr=lr,
                                        weight_decay=wd)
                d = data if not callable(data) else data(seed)
                vals.append((logits[masks["val"]].argmax(1) == d.y[masks["val"]])
                            .float().mean().item())
            mv = sum(vals) / len(vals)
            scores.append((mv, lr, wd))
            print(f"{arch:>5} {lr:>7} {wd:>8} {mv:>13.4f}")
        mv, lr, wd = max(scores)
        best[arch] = {"lr": lr, "weight_decay": wd, "val_acc": round(mv, 4)}
        print(f"  -> {arch}: lr={lr} wd={wd} (val {mv:.4f})")
    return best


def main():
    out = {}
    for name in ("cora", "citeseer"):
        data = load_planetoid(name)
        out[name] = tune(data, lambda s, d=data: {"train": train_mask_real(d, 5, s),
                                                  "val": d.val_mask, "test": d.test_mask},
                         f"Stage-1 tuning: {name} (m=5, seeds {SEEDS})")

    # SBM: tune at the middle homophily level; one graph per seed
    def sbm_data(seed):
        return make_csbm(0.50, seed=seed, sigma_x=0.50)

    def sbm_masks(seed):
        d = sbm_data(seed)
        tr, va, te = masks_sbm(d.y, 5, seed)
        return {"train": tr, "val": va, "test": te}

    out["sbm"] = tune(sbm_data, sbm_masks, "Stage-1 tuning: CSBM h=0.50 (m=5)")

    print("\nFROZEN Stage-1 hyperparameters (record in CHANGELOG addendum M3):")
    for ds, v in out.items():
        for arch, hp in v.items():
            print(f"  {ds:>9} {arch:>4}: lr={hp['lr']} weight_decay={hp['weight_decay']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

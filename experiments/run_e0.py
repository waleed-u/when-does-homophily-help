"""E0 — pipeline sanity gate (PROTOCOL.md §I.1).

Cora m=20, seeds 0-2, M0/M1 at the protocol's default hyperparameters. This is a GATE, not an
experiment: it writes nothing to runs.csv. Nothing scientific runs until it is clean.
"""
import sys

import torch

from src.data import load_planetoid
from src.splits import assert_disjoint, train_mask_real
from src.train import train_model

EXPECTED = {"gcn": (0.75, 0.82), "mlp": (0.55, 0.60)}   # protocol's stated expectation
HARD = {"gcn": (0.70, 0.87), "mlp": (0.45, 0.68)}       # failure bands


def main() -> int:
    data = load_planetoid("cora")
    print(f"Cora: n={data.x.shape[0]} d={data.x.shape[1]} C={data.num_classes} "
          f"val={int(data.val_mask.sum())} test={int(data.test_mask.sum())}")
    ok = True
    for name in ("mlp", "gcn"):
        vals, tests = [], []
        for seed in (0, 1, 2):
            tm = train_mask_real(data, 20, seed)
            assert_disjoint(tm, data.val_mask, data.test_mask)
            logits, info = train_model(name, data, tm, data.val_mask, seed=seed,
                                       lr=0.01, weight_decay=5e-4)
            pred = logits.argmax(1)
            va = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            ta = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            vals.append(va)
            tests.append(ta)
            print(f"  {name} seed={seed}: val={va:.4f} test={ta:.4f} "
                  f"best_epoch={info['best_epoch']} ({info['train_seconds']:.1f}s)")
        mv = sum(vals) / len(vals)
        mt = sum(tests) / len(tests)
        lo, hi = EXPECTED[name]
        hlo, hhi = HARD[name]
        flag = "OK" if lo <= mv <= hi else "WARNING (outside protocol's expected band)"
        print(f"{name}: mean val={mv:.4f} test={mt:.4f}  [{flag}]")
        if not (hlo <= mv <= hhi):
            print(f"  FAIL: mean val {mv:.4f} outside hard band [{hlo}, {hhi}]")
            ok = False
    print("\nE0", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

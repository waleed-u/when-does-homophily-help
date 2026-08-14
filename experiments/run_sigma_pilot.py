"""CSBM feature-noise calibration (PROTOCOL.md §C.3) — the ONLY pre-freeze tuning allowed.

sigma_x sets how much of the label signal lives in the features, i.e. how much room a
relational prior has to help. It is therefore chosen ONCE, on quarantined pilot seeds
(100-109) that never appear in any reported result, by a criterion fixed in advance:

    closed-form feature-only Bayes oracle accuracy in [70%, 80%]

and then frozen for the entire study. The criterion deliberately uses only Oracle-F — never
any model's accuracy — so the choice cannot be tuned to favour the prior.
"""
import sys

import torch

from src.synthetic.oracle import oracle_f_accuracy
from src.synthetic.sbm import beta_gen, check_graph, make_csbm

PILOT_SEEDS = [100, 101, 102]
BAND = (0.70, 0.80)
CANDIDATES = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main() -> int:
    print("sigma_x pilot on quarantined seeds", PILOT_SEEDS, f"— acceptance band {BAND}\n")
    print(f"{'sigma_x':>8} {'OracleF acc':>12}  (mean over pilot seeds, h=0.5)")
    mid = sum(BAND) / 2
    in_band = []
    for sx in CANDIDATES:
        accs = [oracle_f_accuracy(make_csbm(0.5, seed=s, sigma_x=sx)) for s in PILOT_SEEDS]
        mean = sum(accs) / len(accs)
        if BAND[0] <= mean <= BAND[1]:
            in_band.append((abs(mean - mid), sx, mean))
        print(f"{sx:>8.2f} {mean:>12.4f}{'   [in band]' if BAND[0] <= mean <= BAND[1] else ''}")

    if not in_band:
        print("\nNo candidate in band — widen the grid before running anything else.")
        return 1
    # Tie-break among qualifying candidates: closest to the band midpoint. The protocol fixes
    # the band but not the tie-break; midpoint is chosen over "first in band" so the frozen
    # value is not sitting on a boundary where seed noise could push it out of spec.
    _, chosen, chosen_acc = min(in_band)
    print(f"\nband midpoint rule -> sigma_x = {chosen} (Oracle-F = {chosen_acc:.4f})")

    print(f"\nFROZEN sigma_x = {chosen}")
    print("\nGenerator check across the h grid at the frozen sigma_x (pilot seed 100):")
    print(f"{'h':>6} {'emp_h':>8} {'adj_h':>8} {'LI':>7} {'degree':>8} {'beta_gen':>9}  checks")
    for h in (0.05, 0.15, 0.30, 0.50, 0.70, 0.90):
        d = make_csbm(h, seed=100, sigma_x=chosen)
        c = check_graph(d)
        ok = "OK" if (c["degree_ok"] and c["homophily_ok"] and c["balance_ok"]) else "FAIL"
        print(f"{h:>6.2f} {c['empirical_h']:>8.4f} {c['adjusted_h']:>8.4f} "
              f"{c['label_informativeness']:>7.4f} {c['mean_degree']:>8.3f} "
              f"{beta_gen(h):>9.3f}  {ok}")
    print("\nOracle-F accuracy is h-independent by construction (features do not depend on h):")
    for h in (0.05, 0.90):
        print(f"  h={h}: {oracle_f_accuracy(make_csbm(h, seed=100, sigma_x=chosen)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

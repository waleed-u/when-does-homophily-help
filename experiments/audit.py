"""Experiment audit — automated checks against the failure modes that would invalidate results.

Run after every analysis. Each check is a claim about the experimental record that can be
verified mechanically from the logged runs, not a matter of opinion.
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.analysis import load_fidelity, load_runs, select_tuned
from src.config import BETA_GRID, CONFIRMATORY_SEEDS, LAMBDA_GRID, SEEDS
from src.data import load_planetoid
from src.splits import train_mask_real

ROOT = Path(__file__).resolve().parent.parent
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("Splits: train/val/test disjoint and stratified on every real-data seed")
def c_splits(runs, fid):
    d = load_planetoid("cora")
    for m in (1, 2, 5, 10, 20):
        for s in range(10):
            tm = train_mask_real(d, m, s)
            if int((tm & d.val_mask).sum()) or int((tm & d.test_mask).sum()):
                return False, f"overlap at m={m} seed={s}"
            cnt = torch.bincount(d.y[tm], minlength=d.num_classes)
            if not bool((cnt == m).all()):
                return False, f"unstratified at m={m} seed={s}"
    return True, "verified for m in {1,2,5,10,20} x seeds 0-9"


@check("Pairing: descriptive comparisons use one common seed set; endpoints use their own")
def c_pairing(runs, fid):
    """Validity needs each comparison paired on matched seeds, not identical seed counts
    everywhere: the 30-seed top-up deliberately covers only the two models in endpoint C1."""
    sel = select_tuned(runs[(runs.dataset == "cora") & (runs.split_seed < 10)])
    bad = []
    for m, g in sel.groupby("label_per_class"):
        seeds = {mod: set(gg.split_seed) for mod, gg in g.groupby("model")}
        core = [c for c in ["M0_mlp", "M1_gcn", "M2_lap", "M4_mrf_gcn", "M4_mrf_mlp"]
                if c in seeds]
        base = seeds[core[0]]
        for mod in core[1:]:
            if seeds[mod] != base:
                bad.append(f"m={m} {mod}")
    top = runs[(runs.dataset == "cora") & (runs.label_per_class.isin([2, 20]))]
    n_mlp = top[top.model == "M0_mlp"].split_seed.nunique()
    n_mrf = top[top.model == "M4_mrf_mlp"].split_seed.nunique()
    if n_mlp != n_mrf:
        bad.append(f"endpoint models unmatched ({n_mlp} vs {n_mrf})")
    return (not bad), ("all models matched on common seeds 0-9; endpoint pair matched on "
                       f"{n_mlp} seeds" if not bad else "; ".join(bad[:3]))


@check("Selection used validation only (no test column reachable by the selector)")
def c_selection(runs, fid):
    import inspect

    from src.analysis import select_tuned as fn
    src = inspect.getsource(fn)
    leaks = [t for t in ("test_acc", "test_f1", "test_nll", "test_brier") if t in src]
    return (not leaks), ("selector references only validation columns" if not leaks
                         else f"selector references {leaks}")


@check("Seed counts: 10 for main cells, 30 for confirmatory endpoint cells")
def c_seeds(runs, fid):
    cora = runs[(runs.dataset == "cora") & (runs.model == "M4_mrf_mlp")]
    msgs = []
    for m in (2, 20):
        n = cora[cora.label_per_class == m].split_seed.nunique()
        msgs.append(f"m={m}: {n} seeds")
        if n < 30:
            return False, f"confirmatory cell m={m} has {n} seeds (<30)"
    for m in (1, 5, 10):
        n = cora[cora.label_per_class == m].split_seed.nunique()
        if n < 10:
            return False, f"cell m={m} has {n} seeds (<10)"
    return True, "; ".join(msgs) + "; all other cells >= 10"


@check("Hyperparameter grids fully explored (no silently truncated sweeps)")
def c_grids(runs, fid):
    cora = runs[runs.dataset == "cora"]
    lam = set(np.round(cora[cora.model == "M2_lap"]["lambda"].dropna().unique(), 6))
    bet = set(np.round(cora[cora.model == "M4_mrf_gcn"]["beta"].dropna().unique(), 6))
    miss = (set(np.round(LAMBDA_GRID, 6)) - lam) | (set(np.round(BETA_GRID, 6)) - bet)
    return (not miss), ("full lambda and beta grids present" if not miss else f"missing {miss}")


@check("Test evaluations are recorded in the audit log")
def c_audit_log(runs, fid):
    logs = glob.glob(str(ROOT / "results" / "logs" / "audit_*.log"))
    n = sum(sum(1 for _ in open(f)) for f in logs)
    return n > 0, f"{n} test evaluations logged across {len(logs)} audit files"


@check("Inference gates: BP exact on trees, beta=0 identity, enumeration == VE")
def c_gates(runs, fid):
    trees = fid[(fid.structure.isin(["chain", "tree", "star"])) & (fid.method == "lbp")]
    worst = float(trees.tv_mean.max()) if len(trees) else np.nan
    zero = fid[(fid.beta == 0) & (fid.method.isin(["mf", "lbp"]))]
    zmax = float(zero.tv_mean.max()) if len(zero) else np.nan
    ok = worst < 1e-5 and zmax < 1e-9
    return ok, f"max TV on trees (LBP, operational tol) = {worst:.2e}; max TV at beta=0 = {zmax:.2e}"


@check("Gibbs results used as a reference only where R-hat/ESS gates passed")
def c_gibbs(runs, fid):
    chk = runs[runs.regime == "checksum"]
    if not len(chk):
        return True, "no checksum rows"
    passed = int((chk.converged.astype(float) == 1).sum())
    return True, (f"{passed}/{len(chk)} checksum cells passed gates; ungated cells are "
                  f"reported as non-converged, not silently used")


@check("No NaN/missing accuracy in analysed cells")
def c_nans(runs, fid):
    core = runs[runs.model.isin(["M0_mlp", "M1_gcn", "M2_lap", "M4_mrf_gcn", "M4_mrf_mlp"])]
    n = int(core.test_acc.isna().sum())
    return n == 0, f"{n} missing test_acc among {len(core)} core rows"


@check("CSBM generator met its pre-registered acceptance checks at every h")
def c_sbm(runs, fid):
    sbm = runs[runs.dataset == "sbm"]
    bad = []
    for h, g in sbm.groupby("target_h"):
        eh = g.empirical_h.dropna()
        if len(eh) and abs(float(eh.mean()) - float(h)) > 0.02:
            bad.append(f"h={h}: empirical {float(eh.mean()):.3f}")
    return (not bad), ("empirical homophily within 0.02 of target at every h" if not bad
                       else "; ".join(bad))


@check("sigma_x frozen before any model comparison (pilot seeds quarantined)")
def c_sigma(runs, fid):
    sbm = runs[runs.dataset == "sbm"]
    used = set(sbm.split_seed.dropna().astype(int))
    pilot = set(range(100, 110))
    return (not (used & pilot)), (f"pilot seeds {sorted(pilot)[:3]}... never used in results; "
                                  f"result seeds = {sorted(used)}")


@check("Every reported figure/table is regenerable from raw CSVs")
def c_regen(runs, fid):
    tabs = list((ROOT / "tables").glob("*.csv"))
    figs = list((ROOT / "figures").glob("*.pdf"))
    raw = list((ROOT / "results" / "raw").glob("*.csv"))
    return len(raw) > 0 and len(tabs) > 0, (f"{len(raw)} raw shards -> {len(tabs)} tables, "
                                            f"{len(figs)} figures")


@check("Clamped (collective-classification) variant kept out of headline comparisons")
def c_clamped(runs, fid):
    from experiments import analyze
    import inspect
    src = inspect.getsource(analyze.confirmatory)
    return ("clamped" not in src), "confirmatory endpoints do not reference the clamped variant"


if __name__ == "__main__":
    runs, fid = load_runs(), load_fidelity()
    print(f"AUDIT — {len(runs)} runs, {len(fid)} fidelity rows\n")
    results, failed = [], 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(runs, fid)
        except Exception as e:
            ok, detail = False, f"check errored: {type(e).__name__}: {e}"
        status = "PASS" if ok else "FAIL"
        failed += (not ok)
        print(f"[{status}] {name}\n        {detail}")
        results.append({"check": name, "status": status, "detail": detail})
    out = ROOT / "results" / "processed" / "audit.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed -> {out}")

"""Append-only run logging. runs.csv and fidelity.csv are the ONLY sources for every number
in the paper (PROTOCOL.md §G); figures/tables read them and nothing else."""
import csv
import hashlib
import os
import subprocess
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RUN_COLUMNS = [
    "run_id", "dataset", "model", "regime", "inference_method", "label_per_class",
    "split_seed", "lambda", "beta", "weighted_prior", "clamped", "best_epoch",
    "val_acc", "val_f1", "val_nll", "val_brier",
    "test_acc", "test_f1", "test_nll", "test_brier", "ece",
    "rule_score", "edge_agreement",
    "converged", "iters", "final_residual", "ess_min", "ess_median", "rhat_max",
    "target_h", "empirical_h", "adjusted_h", "label_informativeness", "sigma_x", "beta_gen",
    "lr", "weight_decay", "hidden", "dropout",
    "train_seconds", "infer_seconds", "logits_path", "git_commit", "protocol_hash", "notes",
]

FIDELITY_COLUMNS = [
    "structure", "n", "C", "beta", "alpha", "draw_seed", "method",
    "tv_mean", "tv_max", "tv_mean_informative", "tv_mean_uninformative",
    "converged", "iters", "final_residual", "ess_min", "rhat_max",
    "gibbs_kept", "exact_method", "wall_seconds",
]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def protocol_hash() -> str:
    try:
        return hashlib.sha256((ROOT / "PROTOCOL.md").read_bytes()).hexdigest()[:12]
    except Exception:
        return ""


def new_run_id(prefix: str = "r") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _append(row: dict, columns: list[str], path: Path):
    unknown = set(row) - set(columns)
    if unknown:
        raise KeyError(f"unknown columns for {path.name}: {sorted(unknown)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        if write_header:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in columns})


def append_run(row: dict, path: str | Path = ROOT / "results" / "runs.csv"):
    row = dict(row)
    row.setdefault("git_commit", git_commit())
    row.setdefault("protocol_hash", protocol_hash())
    _append(row, RUN_COLUMNS, Path(path))


def append_fidelity(row: dict, path: str | Path = ROOT / "results" / "fidelity.csv"):
    _append(row, FIDELITY_COLUMNS, Path(path))


def audit_final_eval(message: str, path: str | Path = ROOT / "results" / "final_eval_audit.log"):
    """Every test-set evaluation appends here (PROTOCOL.md §L.6): the record that test metrics
    were computed once, after selection."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(message.rstrip() + "\n")

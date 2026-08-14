"""Shared experiment machinery.

Every grid point is written to runs.csv; hyperparameter SELECTION happens later in analysis as
a pure function of `val_acc` (PROTOCOL.md §F, §K). Test metrics are computed for all grid
points but are never read by the selection code — the selection function literally only sees
validation columns, and every test evaluation is recorded in the audit log. This keeps
selection reproducible and auditable instead of hidden inside a runner.
"""
import torch

from src.infer import flip_rate, posterior
from src.metrics import all_metrics
from src.runlog import append_run, audit_final_eval, new_run_id
from src.train import save_logits, train_model

BASE = {"lr": 0.01, "weight_decay": 5e-4, "hidden": 64, "dropout": 0.5}


def evaluate(p, data, masks):
    m = all_metrics(p, data.y, masks, data.edge_index)
    return m


def log_row(**kw):
    append_run(kw)


def run_trained(model_name, data, masks, *, seed, m, dataset, model_label, lam=0.0,
                reg="none", layers=2, hp=None, save=True, extra=None):
    """Train one model, evaluate, log a runs.csv row. Returns (logits, row)."""
    hp = {**BASE, **(hp or {})}
    logits, info = train_model(model_name, data, masks["train"], masks["val"], seed=seed,
                               lr=hp["lr"], weight_decay=hp["weight_decay"],
                               hidden=hp["hidden"], dropout=hp["dropout"], layers=layers,
                               lam=lam, reg=reg)
    p = torch.softmax(logits.double(), dim=1)
    mt = evaluate(p, data, {"val": masks["val"], "test": masks["test"]})
    rid = new_run_id(model_label)
    path = save_logits(logits, rid) if save else ""
    row = dict(run_id=rid, dataset=dataset, model=model_label, regime="train",
               inference_method="none", label_per_class=m, split_seed=seed,
               **{"lambda": lam}, beta="", best_epoch=info["best_epoch"],
               lr=hp["lr"], weight_decay=hp["weight_decay"], hidden=hp["hidden"],
               dropout=hp["dropout"], train_seconds=round(info["train_seconds"], 3),
               logits_path=path, **{k: round(v, 6) for k, v in mt.items()})
    row.update(extra or {})          # caller-supplied fields win (e.g. per-graph diagnostics)
    log_row(**row)
    audit_final_eval(f"{rid} {dataset} {model_label} m={m} seed={seed} test_acc={mt['test_acc']:.4f}")
    return logits, row


def run_posthoc(logits, data, masks, *, seed, m, dataset, model_label, beta, engine="mf",
                w=None, weighted="", clamped=False, regime="tuned_grid", extra=None,
                engine_kwargs=None):
    """Run MRF inference on saved logits, evaluate, log. Returns (q, row)."""
    cm = cy = None
    if clamped:
        cm, cy = masks["train"], data.y
    q, info = posterior(logits, data.edge_index, beta, engine=engine, w=w,
                        clamp_mask=cm, clamp_y=cy, **(engine_kwargs or {}))
    mt = evaluate(q, data, {"val": masks["val"], "test": masks["test"]})
    rid = new_run_id(model_label)
    row = dict(run_id=rid, dataset=dataset, model=model_label, regime=regime,
               inference_method=engine, label_per_class=m, split_seed=seed,
               beta=beta, weighted_prior=weighted, clamped=int(clamped),
               converged=int(bool(info.get("converged", True))),
               iters=info.get("iters", ""), final_residual=info.get("final_residual", ""),
               ess_min=info.get("ess_min", ""), rhat_max=info.get("rhat_max", ""),
               infer_seconds=round(info["infer_seconds"], 4),
               notes=f"flip={flip_rate(logits, q, masks['test']):.4f}",
               **{k: round(v, 6) for k, v in mt.items()})
    extra = dict(extra or {})
    if "notes" in extra:             # keep the flip-rate diagnostic, append the caller's note
        extra["notes"] = f"{row['notes']} {extra['notes']}"
    row.update(extra)
    log_row(**row)
    audit_final_eval(f"{rid} {dataset} {model_label} beta={beta} m={m} seed={seed} "
                     f"test_acc={mt['test_acc']:.4f}")
    return q, row

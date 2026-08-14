"""Training loop (PROTOCOL.md §F, §G): Adam, early stopping on validation accuracy with
patience, best-checkpoint restore, deterministic CPU execution, logits saved for every run so
that all post-hoc inference is a pure function of saved logits (infer.py never trains)."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .losses.regularizers import REGULARIZERS
from .models.gcn import GCN
from .models.mlp import MLP

torch.use_deterministic_algorithms(True, warn_only=True)

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"mlp": MLP, "gcn": GCN}


def train_model(model_name: str, data, train_mask: torch.Tensor, val_mask: torch.Tensor, *,
                seed: int, lr: float, weight_decay: float, hidden: int = 64,
                dropout: float = 0.5, layers: int = 2, lam: float = 0.0, reg: str = "none",
                max_epochs: int = 500, patience: int = 50):
    """Returns (logits [n, C] detached float32, info dict).

    lam == 0 skips the regularizer term entirely, so reg='laplacian' with lam=0 follows the
    exact optimization path of reg='none' (the pre-registered lambda=0 equivalence gate).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MODELS[model_name](data.x.shape[1], data.num_classes, hidden=hidden,
                               dropout=dropout, layers=layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_reg = reg != "none" and lam != 0.0
    reg_fn = REGULARIZERS[reg] if use_reg else None

    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])
        if use_reg:
            loss = loss + lam * reg_fn(torch.softmax(out, dim=1), data.edge_index)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            va = (logits[val_mask].argmax(1) == data.y[val_mask]).float().mean().item()
        if va > best["val_acc"]:  # strict: keeps the FIRST best on ties
            best = {"val_acc": va, "epoch": epoch,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index).detach()
    return logits, {"best_epoch": best["epoch"], "val_acc_best": best["val_acc"],
                    "train_seconds": time.perf_counter() - t0, "epochs_run": epoch + 1}


def save_logits(logits: torch.Tensor, run_id: str, out_dir: str | Path = ROOT / "results" / "logits") -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.npz"
    np.savez_compressed(path, logits=logits.cpu().numpy())
    return str(path.relative_to(ROOT))


def load_logits(path: str | Path) -> torch.Tensor:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return torch.from_numpy(np.load(p)["logits"])

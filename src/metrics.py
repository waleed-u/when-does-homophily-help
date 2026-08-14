"""Metric definitions (PROTOCOL.md §J). All take probabilities p [n, C] and labels y [n]."""
import torch

from .data import undirected_edges


def accuracy(p: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    return (p[mask].argmax(1) == y[mask]).float().mean().item()


def macro_f1(p: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred, true = p[mask].argmax(1), y[mask]
    C = p.shape[1]
    f1s = []
    for c in range(C):
        tp = int(((pred == c) & (true == c)).sum())
        fp = int(((pred == c) & (true != c)).sum())
        fn = int(((pred != c) & (true == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / C


def nll(p: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    q = p[mask].clamp(min=1e-12)
    return float(-q[torch.arange(len(q)), y[mask]].log().mean())


def brier(p: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Multiclass Brier = mean squared error vs one-hot (PROTOCOL.md §J)."""
    q = p[mask]
    onehot = torch.zeros_like(q).scatter_(1, y[mask].unsqueeze(1), 1.0)
    return float(((q - onehot) ** 2).sum(1).mean())


def ece(p: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, bins: int = 15) -> float:
    """Expected calibration error, equal-mass bins."""
    conf, pred = p[mask].max(1)
    correct = (pred == y[mask]).float()
    order = conf.argsort()
    conf, correct = conf[order], correct[order]
    n = len(conf)
    edges = torch.linspace(0, n, bins + 1).long()
    e = 0.0
    for b in range(bins):
        sl = slice(int(edges[b]), int(edges[b + 1]))
        if edges[b + 1] > edges[b]:
            e += (edges[b + 1] - edges[b]) / n * float((conf[sl].mean() - correct[sl].mean()).abs())
    return float(e)


def rule_satisfaction(p: torch.Tensor, edge_index: torch.Tensor) -> float:
    """R = (1/|E|) sum over undirected edges of p_i . p_j."""
    e = undirected_edges(edge_index)
    return float((p[e[0]] * p[e[1]]).sum(1).mean())


def argmax_edge_agreement(p: torch.Tensor, edge_index: torch.Tensor) -> float:
    e = undirected_edges(edge_index)
    pred = p.argmax(1)
    return float((pred[e[0]] == pred[e[1]]).float().mean())


def all_metrics(p, y, masks: dict, edge_index) -> dict:
    out = {}
    for split, mask in masks.items():
        out[f"{split}_acc"] = accuracy(p, y, mask)
        out[f"{split}_f1"] = macro_f1(p, y, mask)
        out[f"{split}_nll"] = nll(p, y, mask)
        out[f"{split}_brier"] = brier(p, y, mask)
    out["ece"] = ece(p, y, masks["test"]) if "test" in masks else float("nan")
    out["rule_score"] = rule_satisfaction(p, edge_index)
    out["edge_agreement"] = argmax_edge_agreement(p, edge_index)
    return out

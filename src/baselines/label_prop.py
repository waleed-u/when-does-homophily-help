"""Label propagation / harmonic functions (Zhu, Ghahramani & Lafferty 2003).

The historical ancestor of this whole project: the homophily assumption as a probabilistic
model over labels, with observed labels clamped as evidence and the rest filled in by
propagation. Graph + labels only, no features — the corner of the evidence factorial that the
MLP (features, no graph) and GCN (both) leave empty.
"""
import torch


def label_prop(edge_index: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, n: int,
               C: int, iters: int = 1000, tol: float = 1e-8) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(n, dtype=torch.float64).index_add_(
        0, dst, torch.ones(edge_index.shape[1], dtype=torch.float64)).clamp(min=1.0)

    f = torch.full((n, C), 1.0 / C, dtype=torch.float64)
    onehot = torch.zeros(n, C, dtype=torch.float64)
    onehot[train_mask, y[train_mask]] = 1.0
    f[train_mask] = onehot[train_mask]

    for _ in range(iters):
        agg = torch.zeros(n, C, dtype=torch.float64).index_add_(0, dst, f[src]) / deg.unsqueeze(1)
        agg[train_mask] = onehot[train_mask]                     # clamp observed labels
        if float((agg - f).abs().max()) < tol:
            f = agg
            break
        f = agg
    return f / f.sum(1, keepdim=True).clamp(min=1e-300)

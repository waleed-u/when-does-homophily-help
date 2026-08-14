"""Edge potentials for the Potts MRF (PROTOCOL.md §E).

The unweighted prior trusts every edge equally. The similarity-weighted variant (M5) makes the
rule context-dependent — Edge(i,j) AND Similar(i,j) => SameClass(i,j) — by down-weighting edges
between dissimilar nodes, which is the natural repair when some edges cross class boundaries.
The transformation g is frozen (clipped cosine); only beta is tuned.
"""
import torch


def similarity_weights(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """w_ij = clip(cosine(x_i, x_j), 0, 1), aligned with the directed edge columns."""
    xn = torch.nn.functional.normalize(x.double(), dim=1, eps=1e-12)
    return (xn[edge_index[0]] * xn[edge_index[1]]).sum(1).clamp(0.0, 1.0)


def permuted_similarity_weights(x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor,
                                seed: int = 0) -> torch.Tensor:
    """Control for M5: features shuffled *within* class, so the similarity signal is destroyed
    while its marginal distribution is preserved. If M5's benefit survives this, the benefit was
    not coming from the similarity mechanism."""
    g = torch.Generator().manual_seed(seed)
    x2 = x.clone()
    for c in range(int(y.max()) + 1):
        idx = torch.nonzero(y == c, as_tuple=False).view(-1)
        x2[idx] = x[idx[torch.randperm(len(idx), generator=g)]]
    return similarity_weights(x2, edge_index)


def edge_similarity_report(x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor) -> dict:
    """M5 pre-check: is cosine similarity actually informative about same-class edges here?"""
    from ..data import undirected_edges
    e = undirected_edges(edge_index)
    w = similarity_weights(x, e)
    same = y[e[0]] == y[e[1]]
    return {
        "sim_same_mean": float(w[same].mean()), "sim_same_std": float(w[same].std()),
        "sim_cross_mean": float(w[~same].mean()), "sim_cross_std": float(w[~same].std()),
        "separation": float(w[same].mean() - w[~same].mean()),
        "n_same": int(same.sum()), "n_cross": int((~same).sum()),
    }

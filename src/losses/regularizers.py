"""Training-time impositions of the homophily prior (PROTOCOL.md §E).

Both sum over ALL undirected edges, including edges with unlabelled endpoints — that is the
semi-supervised mechanism, and it is identical across prior models so comparisons stay fair.

Proposition 1 (see paper appendix): with agreement A(p) = sum_edges p_i . p_j,
    rule_violation_loss = 1 - A/|E|
    laplacian_loss      = (sum_edges ||p_i||^2 + ||p_j||^2)/|E| - 2A/|E|
i.e. both are driven by the same sufficient statistic; the Laplacian form adds a
confidence penalty.
"""
import torch

from ..data import undirected_edges


def laplacian_loss(p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    e = undirected_edges(edge_index)
    return ((p[e[0]] - p[e[1]]) ** 2).sum(1).mean()


def rule_violation_loss(p: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    e = undirected_edges(edge_index)
    return (1.0 - (p[e[0]] * p[e[1]]).sum(1)).mean()


REGULARIZERS = {"laplacian": laplacian_loss, "rule": rule_violation_loss}

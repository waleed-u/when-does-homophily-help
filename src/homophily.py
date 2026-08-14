"""Graph label diagnostics: edge homophily, adjusted homophily, label informativeness
(Platonov et al., NeurIPS 2023). Computed on the undirected edge set."""
import torch

from .data import undirected_edges


def edge_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    e = undirected_edges(edge_index)
    return (y[e[0]] == y[e[1]]).float().mean().item()


def _degree_weighted_class_dist(edge_index: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # p_c = D_c / 2|E| over the undirected graph == class distribution of a random edge endpoint
    C = int(y.max()) + 1
    ends = torch.cat([y[edge_index[0]], y[edge_index[1]]])  # both directions present
    return torch.bincount(ends, minlength=C).double() / len(ends)


def adjusted_homophily(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    h = edge_homophily(edge_index, y)
    p = _degree_weighted_class_dist(edge_index, y)
    p2 = float((p**2).sum())
    return (h - p2) / (1.0 - p2)


def label_informativeness(edge_index: torch.Tensor, y: torch.Tensor) -> float:
    """LI = 2 - H(y_u, y_v) / H(y_u) for a uniformly sampled edge (degree-weighted marginal)."""
    e = undirected_edges(edge_index)
    C = int(y.max()) + 1
    yu, yv = y[e[0]], y[e[1]]
    # symmetrize the joint since edges are unordered
    idx = torch.cat([yu * C + yv, yv * C + yu])
    joint = torch.bincount(idx, minlength=C * C).double()
    joint = joint / joint.sum()
    p = _degree_weighted_class_dist(edge_index, y)

    def H(q):
        q = q[q > 0]
        return float(-(q * q.log()).sum())

    return 2.0 - H(joint) / H(p)


def diagnostics(edge_index: torch.Tensor, y: torch.Tensor) -> dict:
    return {
        "edge_homophily": edge_homophily(edge_index, y),
        "adjusted_homophily": adjusted_homophily(edge_index, y),
        "label_informativeness": label_informativeness(edge_index, y),
        "mean_degree": edge_index.shape[1] / len(y),  # both directions / n
    }

"""Small graph builders for the inference-fidelity suite (PROTOCOL.md §C.4).

Structures span the axes that matter for approximate inference: acyclic (chain, tree, star),
single-loop (cycle), many short loops (grid), and dense/high-degree (2-block SBM). The star is
included because mean-field's behaviour is degree-dependent, and every other structure here is
roughly degree-homogeneous.
"""
import torch


def _undirected(pairs, n):
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long), n
    e = torch.tensor(pairs, dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1), n


def chain(n: int = 20):
    return _undirected([(i, i + 1) for i in range(n - 1)], n)


def balanced_tree(n: int = 15):
    return _undirected([(i, 2 * i + 1) for i in range((n - 1) // 2)] +
                       [(i, 2 * i + 2) for i in range((n - 1) // 2)], n)


def star(n: int = 16):
    return _undirected([(0, i) for i in range(1, n)], n)


def cycle(n: int = 16):
    return _undirected([(i, (i + 1) % n) for i in range(n)], n)


def grid(rows: int = 4, cols: int = 4):
    pairs = []
    for r in range(rows):
        for c in range(cols):
            v = r * cols + c
            if c + 1 < cols:
                pairs.append((v, v + 1))
            if r + 1 < rows:
                pairs.append((v, v + cols))
    return _undirected(pairs, rows * cols)


def dense_sbm(n: int = 14, p_in: float = 0.8, p_out: float = 0.3, seed: int = 0):
    """2 equal blocks; returns (edge_index, n, block_labels)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.zeros(n, dtype=torch.long)
    y[n // 2:] = 1
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if float(torch.rand(1, generator=g)) < p:
                pairs.append((i, j))
    ei, n = _undirected(pairs, n)
    return ei, n, y


def make_unaries(n: int, C: int, alpha: float, seed: int, uninformative_frac: float = 0.25,
                 labels: torch.Tensor | None = None):
    """s_i = alpha * onehot(y_i) + N(0, I); a fixed fraction of nodes get alpha=0 so that some
    nodes carry no evidence of their own — those are exactly the nodes where relational
    inference has to do the work."""
    g = torch.Generator().manual_seed(seed)
    y = labels if labels is not None else torch.randint(0, C, (n,), generator=g)
    s = torch.randn(n, C, generator=g, dtype=torch.float64)
    scale = torch.full((n,), float(alpha), dtype=torch.float64)
    k = int(round(uninformative_frac * n))
    if k:
        scale[torch.randperm(n, generator=g)[:k]] = 0.0
    s[torch.arange(n), y] += scale
    return s, y, (scale > 0)


BUILDERS = {"chain": chain, "tree": balanced_tree, "star": star, "cycle": cycle, "grid": grid}

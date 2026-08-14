"""Mean-field variational inference — the study's primary inference engine (PROTOCOL.md §H).

Fully factorized q(y) = prod_i q_i(y_i); coordinate ascent update

    q_i(c)  proportional to  exp( s_i(c) + beta * sum_{j in N(i)} w_ij q_j(c) )

executed one graph-colour class at a time. Nodes sharing a colour share no edge, so a whole
class updates simultaneously and this is still exact block coordinate ascent — vectorised AND
monotone in the ELBO (asserted), unlike naive synchronous updates which can oscillate.

ELBO:  F(q) = sum_i q_i.s_i + beta * sum_{undirected} w_ij (q_i.q_j) + sum_i H(q_i)

Note the update is structurally APPNP-style propagation with the unary evidence re-injected
every sweep; the aggregation uses the raw adjacency, not the symmetric normalisation a GCN
applies, which is why the two smooth differently (see Proposition 1 in the paper).
"""
import torch

from .coloring import greedy_coloring

_COLOR_CACHE: dict = {}


def _colors(edge_index: torch.Tensor, n: int):
    key = (n, int(edge_index.shape[1]), hash(edge_index[:, :64].numpy().tobytes()))
    if key not in _COLOR_CACHE:
        _COLOR_CACHE[key] = greedy_coloring(edge_index, n)
    return _COLOR_CACHE[key]


def _aggregate(q, edge_index, w):
    """agg[i] = sum_{j in N(i)} w_ij q_j, using the directed edge list (both directions)."""
    src, dst = edge_index[0], edge_index[1]
    msg = q[src] if w is None else q[src] * w.unsqueeze(1)
    return torch.zeros_like(q).index_add_(0, dst, msg)


def elbo(q, s, edge_index, beta, w=None):
    agg = _aggregate(q, edge_index, w)
    pair = 0.5 * float((q * agg).sum())            # directed sum double-counts each edge
    unary = float((q * s).sum())
    ent = float(-(q.clamp(min=1e-300) * q.clamp(min=1e-300).log()).sum())
    return unary + beta * pair + ent


def mean_field(s: torch.Tensor, edge_index: torch.Tensor, beta: float, w=None,
               tol: float = 1e-6, max_sweeps: int = 500, clamp_mask=None, clamp_y=None,
               check_elbo: bool = True, init: torch.Tensor | None = None):
    """Returns (q [n, C] float64, info). Runs to convergence; the sweep count is never tuned."""
    s = s.double()
    w = None if w is None else w.double()
    n, C = s.shape
    q = torch.softmax(s, dim=1) if init is None else init.double().clone()

    free = torch.ones(n, dtype=torch.bool)
    if clamp_mask is not None:
        idx = torch.nonzero(clamp_mask, as_tuple=False).view(-1)
        q[idx] = 0.0
        q[idx, clamp_y[idx]] = 1.0
        free[idx] = False

    classes = [c[free[c]] for c in _colors(edge_index, n)]
    classes = [c for c in classes if len(c)]
    trace = [elbo(q, s, edge_index, beta, w)] if check_elbo else []

    converged, sweeps = False, 0
    for sweep in range(max_sweeps):
        sweeps = sweep + 1
        delta = 0.0
        for cls in classes:
            agg = _aggregate(q, edge_index, w)[cls]
            new = torch.softmax(s[cls] + beta * agg, dim=1)
            delta = max(delta, float(0.5 * (new - q[cls]).abs().sum(1).max()))
            q[cls] = new
            if check_elbo:
                f = elbo(q, s, edge_index, beta, w)
                assert f >= trace[-1] - 1e-9, f"ELBO decreased: {trace[-1]:.12f} -> {f:.12f}"
                trace.append(f)
        if delta < tol:
            converged = True
            break

    return q, {"converged": converged, "iters": sweeps,
               "elbo": trace[-1] if trace else elbo(q, s, edge_index, beta, w),
               "elbo_trace": trace}

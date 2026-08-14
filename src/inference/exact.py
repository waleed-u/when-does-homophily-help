"""Exact marginals for small pairwise MRFs — the ground truth the approximate engines are
scored against (PROTOCOL.md §H).

Joint:  p(y) proportional to exp( sum_i s_i(y_i) + beta * sum_{undirected (i,j)} w_ij 1[y_i=y_j] )

Two exact routes, and each validates the other in tests:
  * brute-force enumeration when C**n <= enum_cutoff (two-pass log-sum-exp, float64)
  * repeated variable elimination with a min-fill ordering otherwise (one run per query node)

Junction trees would give the same answer; repeated VE is used because it is far easier to
verify and these graphs have n <= 20.
"""
import itertools

import torch

from ..data import undirected_edges

NEG = -1e12   # log-space stand-in for an impossible assignment (exp underflows to 0 in float64)


# --------------------------------------------------------------------------------------- setup
def _prepare(s, edge_index, w, clamp_mask, clamp_y):
    s = s.double().clone()
    if clamp_mask is not None:
        idx = torch.nonzero(clamp_mask, as_tuple=False).view(-1)
        s[idx] = NEG
        s[idx, clamp_y[idx]] = 0.0
    e = undirected_edges(edge_index)
    if w is None:
        we = torch.ones(e.shape[1], dtype=torch.float64)
    else:
        # w is given per directed edge; take the value on the (i<j) direction
        mask = edge_index[0] < edge_index[1]
        we = w.double()[mask]
    return s, e, we


# --------------------------------------------------------------------------------- enumeration
def _enumerate_marginals(s, e, we, beta, chunk=1_000_000):
    n, C = s.shape
    total = C**n
    ei, ej = e[0], e[1]

    def log_potentials(lo, hi):
        idx = torch.arange(lo, hi, dtype=torch.int64)
        # mixed-radix decode: assignment [B, n]
        y = torch.stack([(idx // (C**k)) % C for k in range(n)], dim=1)
        lp = s[torch.arange(n).unsqueeze(0), y].sum(1)
        if beta != 0.0:
            agree = (y[:, ei] == y[:, ej]).double() * we.unsqueeze(0)
            lp = lp + beta * agree.sum(1)
        return y, lp

    m = -float("inf")                                  # pass 1: global max for stability
    for lo in range(0, total, chunk):
        _, lp = log_potentials(lo, min(lo + chunk, total))
        m = max(m, float(lp.max()))

    acc = torch.zeros(n, C, dtype=torch.float64)       # pass 2: accumulate exp(lp - max)
    for lo in range(0, total, chunk):
        y, lp = log_potentials(lo, min(lo + chunk, total))
        wgt = (lp - m).exp()
        for i in range(n):
            acc[i].index_add_(0, y[:, i], wgt)
    return acc / acc.sum(1, keepdim=True)


# ------------------------------------------------------------------------ variable elimination
class _Factor:
    """log-space table over an ordered tuple of variables."""

    __slots__ = ("vars", "table")

    def __init__(self, vars_, table):
        self.vars = tuple(vars_)
        self.table = table

    def align(self, order):
        """expand this factor's table to broadcast against variable list `order`."""
        shape = [self.table.shape[self.vars.index(v)] if v in self.vars else 1 for v in order]
        perm = [self.vars.index(v) for v in order if v in self.vars]
        return self.table.permute(perm).reshape(shape)


def _multiply(factors):
    order = sorted(set().union(*[set(f.vars) for f in factors]))
    table = sum(f.align(order) for f in factors)
    return _Factor(order, table)


def _sum_out(f, v):
    ax = f.vars.index(v)
    return _Factor([u for u in f.vars if u != v], torch.logsumexp(f.table, dim=ax))


def _min_fill_order(adj, keep):
    """greedy min-fill elimination ordering over all variables except `keep`."""
    adj = {v: set(ns) for v, ns in adj.items()}
    order = []
    remaining = set(adj) - {keep}
    while remaining:
        best, best_fill = None, None
        for v in remaining:
            nb = adj[v] & (remaining | {keep})
            fill = sum(1 for a, b in itertools.combinations(nb, 2) if b not in adj[a])
            if best_fill is None or fill < best_fill:
                best, best_fill = v, fill
        nb = adj[best] & (remaining | {keep})
        for a, b in itertools.combinations(nb, 2):
            adj[a].add(b)
            adj[b].add(a)
        for u in adj[best]:
            adj[u].discard(best)
        remaining.discard(best)
        order.append(best)
    return order


def _ve_marginals(s, e, we, beta):
    n, C = s.shape
    adj = {i: set() for i in range(n)}
    for k in range(e.shape[1]):
        i, j = int(e[0, k]), int(e[1, k])
        adj[i].add(j)
        adj[j].add(i)

    pair_tables = []
    eye = torch.eye(C, dtype=torch.float64)
    for k in range(e.shape[1]):
        i, j = int(e[0, k]), int(e[1, k])
        pair_tables.append(_Factor((i, j) if i < j else (j, i),
                                   beta * float(we[k]) * eye))

    out = torch.zeros(n, C, dtype=torch.float64)
    for q in range(n):
        factors = [_Factor((i,), s[i].clone()) for i in range(n)] + \
                  [_Factor(f.vars, f.table.clone()) for f in pair_tables]
        for v in _min_fill_order(adj, keep=q):
            involved = [f for f in factors if v in f.vars]
            factors = [f for f in factors if v not in f.vars]
            if involved:
                factors.append(_sum_out(_multiply(involved), v))
        final = _multiply(factors)
        logp = final.table.reshape(-1) if final.vars == (q,) else final.table
        out[q] = torch.softmax(logp, dim=0)
    return out


# ---------------------------------------------------------------------------------- public API
def exact_marginals(s: torch.Tensor, edge_index: torch.Tensor, beta: float, w=None,
                    enum_cutoff: float = 5e6, clamp_mask=None, clamp_y=None,
                    method: str | None = None):
    """Exact node marginals [n, C] (float64). `method` forces 'enumerate' or 've'."""
    s, e, we = _prepare(s, edge_index, w, clamp_mask, clamp_y)
    n, C = s.shape
    if method is None:
        method = "enumerate" if C**n <= enum_cutoff else "ve"
    if method == "enumerate":
        return _enumerate_marginals(s, e, we, beta)
    return _ve_marginals(s, e, we, beta)

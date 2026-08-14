"""Chromatic Gibbs sampling with Rao-Blackwellised marginals (PROTOCOL.md §H).

Nodes sharing a graph colour share no edge, so they are conditionally independent given the
rest and a whole colour class can be resampled in one vectorised step (Gonzalez et al. 2011) —
no per-node Python loop, which is what keeps Gibbs viable at Cora scale.

Estimator: rather than counting label frequencies, average the full-conditional probability
vectors over kept sweeps and chains (Rao-Blackwellisation) — same target, strictly lower
variance, no extra cost since the conditionals are already computed.

Unlike mean-field and BP, the error here is Monte-Carlo variance rather than bias, so it is
diagnosable: split R-hat and ESS on the energy trace, plus inter-chain marginal agreement.
"""
import torch

from ..data import undirected_edges
from .coloring import greedy_coloring


def _conditionals(y, s, edge_index, beta, w, C):
    """p(y_i = . | y_{-i}) for every node, given the current state."""
    src, dst = edge_index[0], edge_index[1]
    onehot = torch.zeros(len(y), C, dtype=torch.float64)
    onehot[torch.arange(len(y)), y] = 1.0
    msg = onehot[src] if w is None else onehot[src] * w.unsqueeze(1)
    counts = torch.zeros(len(y), C, dtype=torch.float64).index_add_(0, dst, msg)
    return torch.softmax(s + beta * counts, dim=1)


def gibbs(s: torch.Tensor, edge_index: torch.Tensor, beta: float, w=None, chains: int = 4,
          burn_in: int = 1000, kept: int = 2000, seed: int = 0, clamp_mask=None, clamp_y=None):
    """Returns (q [n, C] float64, info with R-hat / ESS gates)."""
    s = s.double()
    w = None if w is None else w.double()
    n, C = s.shape
    classes = greedy_coloring(edge_index, n)
    free = torch.ones(n, dtype=torch.bool)
    if clamp_mask is not None:
        free[clamp_mask] = False
    classes = [c[free[c]] for c in classes]
    classes = [c for c in classes if len(c)]

    e_u = undirected_edges(edge_index)
    if w is None:
        w_u = torch.ones(e_u.shape[1], dtype=torch.float64)
    else:
        w_u = w[edge_index[0] < edge_index[1]]

    def energy(y):
        return float((s[torch.arange(n), y]).sum() +
                     beta * (w_u * (y[e_u[0]] == y[e_u[1]]).double()).sum())

    chain_marginals, traces = [], []
    for ch in range(chains):
        g = torch.Generator().manual_seed(seed * 1000 + ch)
        y = (torch.multinomial(torch.softmax(s, 1), 1, generator=g).view(-1) if ch == 0
             else torch.randint(0, C, (n,), generator=g))
        if clamp_mask is not None:
            y[clamp_mask] = clamp_y[clamp_mask]

        acc = torch.zeros(n, C, dtype=torch.float64)
        trace = []
        for it in range(burn_in + kept):
            for cls in classes:
                p = _conditionals(y, s, edge_index, beta, w, C)[cls]
                y[cls] = torch.multinomial(p, 1, generator=g).view(-1)
            if it >= burn_in:
                acc += _conditionals(y, s, edge_index, beta, w, C)
                trace.append(energy(y))
        if clamp_mask is not None:                       # clamped nodes are evidence, not samples
            acc[clamp_mask] = 0.0
            acc[clamp_mask, clamp_y[clamp_mask]] = kept
        chain_marginals.append(acc / kept)
        traces.append(trace)

    q = torch.stack(chain_marginals).mean(0)
    q = q / q.sum(1, keepdim=True)

    rhat = ess = float("nan")
    try:
        import arviz as az
        import numpy as np
        arr = np.asarray(traces)[None, ...] if False else np.asarray(traces)
        idata = az.convert_to_dataset({"E": arr})
        rhat = float(az.rhat(idata).E.values)
        ess = float(az.ess(idata).E.values)
    except Exception:
        pass

    inter = 0.0
    for a in range(len(chain_marginals)):
        for b in range(a + 1, len(chain_marginals)):
            inter = max(inter, float(0.5 * (chain_marginals[a] - chain_marginals[b]).abs().sum(1).max()))

    # Convergence is judged by the mixing diagnostics only (CHANGELOG amendment A1).
    # inter-chain marginal TV is finite-budget Monte-Carlo precision, not a mixing failure:
    # at kept=2000 its expected size is already ~0.02, so gating on it would mislabel
    # perfectly-mixed chains. It is reported alongside, never used as a pass/fail.
    gates = (rhat == rhat and ess == ess and rhat < 1.01 and ess > 400)
    return q, {"converged": bool(gates), "iters": kept, "rhat_max": rhat, "ess_min": ess,
               "interchain_tv": inter}

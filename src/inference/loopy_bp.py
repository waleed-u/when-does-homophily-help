"""Damped loopy belief propagation (PROTOCOL.md §H).

Sum-product on the pairwise Potts model. The Potts structure collapses the usual O(C^2)
message-matrix product to O(C): with a normalised cavity belief b,

    m_{i->j}(c)  proportional to  1 + (exp(beta * w_ij) - 1) * b_{i\\j}(c)

BP is exact on trees; on loopy graphs it is a fixed-point iteration that may oscillate, which
is why damping and a non-convergence reporting rule are pre-registered rather than improvised:
non-convergence is recorded and reported, never silently dropped.
"""
import torch


def _reverse_index(edge_index: torch.Tensor) -> torch.Tensor:
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(edge_index.t().tolist())}
    return torch.tensor([key[(int(b), int(a))] for a, b in edge_index.t().tolist()],
                        dtype=torch.long)


def loopy_bp(s: torch.Tensor, edge_index: torch.Tensor, beta: float, w=None,
             damping: float = 0.5, tol: float = 1e-6, max_iters: int = 500,
             clamp_mask=None, clamp_y=None, _retry: bool = True):
    """Returns (q [n, C] float64, info)."""
    s = s.double().clone()
    if clamp_mask is not None:
        idx = torch.nonzero(clamp_mask, as_tuple=False).view(-1)
        s[idx, clamp_y[idx]] += 30.0          # dominant but numerically safe evidence
    n, C = s.shape
    src, dst = edge_index[0], edge_index[1]
    rev = _reverse_index(edge_index)
    fac = (torch.expm1(torch.tensor(beta, dtype=torch.float64)) if w is None
           else torch.expm1(beta * w.double()))
    fac = fac.expand(edge_index.shape[1]) if fac.dim() == 0 else fac

    logm = torch.zeros(edge_index.shape[1], C, dtype=torch.float64)   # uniform init
    stable, residual, tail = 0, float("inf"), []

    for it in range(1, max_iters + 1):
        total = s.clone().index_add_(0, dst, logm)                    # sum of incoming logs
        cavity = torch.softmax(total[src] - logm[rev], dim=1)         # exclude reverse message
        new = 1.0 + fac.unsqueeze(1) * cavity
        new = new / new.sum(1, keepdim=True)
        old = torch.softmax(logm, dim=1)
        damped = (1 - damping) * new + damping * old
        damped = damped / damped.sum(1, keepdim=True)

        residual = float((damped - old).abs().max())
        logm = damped.clamp(min=1e-300).log()
        if it > max_iters - 50:
            tail.append(torch.softmax(s.clone().index_add_(0, dst, logm), dim=1))
        stable = stable + 1 if residual < tol else 0
        if stable >= 3:
            q = torch.softmax(s.clone().index_add_(0, dst, logm), dim=1)
            return q, {"converged": True, "iters": it, "final_residual": residual,
                       "damping_used": damping}

    if _retry and damping < 0.9:      # oscillating: one pre-registered retry with more damping
        return loopy_bp(s, edge_index, beta, w, damping=0.9, tol=tol, max_iters=max_iters,
                        _retry=False)
    q = torch.stack(tail).mean(0) if tail else torch.softmax(s, dim=1)
    q = q / q.sum(1, keepdim=True)
    return q, {"converged": False, "iters": max_iters, "final_residual": residual,
               "damping_used": damping}

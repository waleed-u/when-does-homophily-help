"""Contextual SBM generator (PROTOCOL.md §C.2) — the study's centrepiece environment.

An SBM over labels with class-conditional Gaussian features (Deshpande et al. 2018). Edge
homophily h is the manipulated variable — the *truth of the prior* — while expected degree,
class balance, feature dimension and feature noise are held fixed:

    p_in = h*k/(n_c - 1),   p_out = (1-h)*k/(n - n_c)

so the expected degree stays k for every h. Degree and features therefore cannot co-vary with
h, which is what makes the sweep a controlled experiment rather than a correlation.

Because the generative process is known, it also yields a closed-form feature-only Bayes
oracle and the generator-implied coupling beta_gen = log(p_in/p_out) — the value an ideal
Potts prior "should" use, which lets us ask whether validation tuning recovers it.
"""
import math
from types import SimpleNamespace

import torch

from ..homophily import diagnostics


def edge_probs(h: float, n: int = 2800, C: int = 7, k: float = 8.0):
    n_c = n // C
    return h * k / (n_c - 1), (1.0 - h) * k / (n - n_c)


def beta_gen(h: float, n: int = 2800, C: int = 7, k: float = 8.0) -> float:
    """Generator-implied Potts coupling: the log-odds of an edge given same vs different class."""
    p_in, p_out = edge_probs(h, n, C, k)
    return math.log(p_in / p_out)


def class_means(C: int, d: int, mu_seed: int = 7, spacing: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(mu_seed)
    q, _ = torch.linalg.qr(torch.randn(d, C, generator=g))
    return (q[:, :C].t() * spacing).contiguous()          # [C, d], orthonormal rows


def make_csbm(h: float, seed: int, n: int = 2800, C: int = 7, k: float = 8.0, d: int = 32,
              sigma_x: float = 1.0, mu_seed: int = 7, block: int = 256) -> SimpleNamespace:
    assert n % C == 0
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % C                                # exactly balanced classes
    y = y[torch.randperm(n, generator=g)]
    p_in, p_out = edge_probs(h, n, C, k)

    rows, cols = [], []
    for lo in range(0, n, block):                          # chunked upper-triangular Bernoulli
        hi = min(lo + block, n)
        idx = torch.arange(lo, hi)
        same = y[idx].unsqueeze(1) == y.unsqueeze(0)       # [B, n]
        p = torch.where(same, torch.tensor(p_in), torch.tensor(p_out))
        upper = idx.unsqueeze(1) < torch.arange(n).unsqueeze(0)
        draw = (torch.rand(hi - lo, n, generator=g) < p) & upper
        r, c = torch.nonzero(draw, as_tuple=True)
        rows.append(idx[r])
        cols.append(c)
    ei_u = torch.stack([torch.cat(rows), torch.cat(cols)])
    edge_index = torch.cat([ei_u, ei_u.flip(0)], dim=1)

    mu = class_means(C, d, mu_seed)
    x = mu[y] + sigma_x * torch.randn(n, d, generator=g)

    data = SimpleNamespace(
        name=f"sbm_h{h:g}", x=x.float(), y=y.long(), edge_index=edge_index, num_classes=C,
        normalized=False, target_h=h, sigma_x=sigma_x, mu=mu, p_in=p_in, p_out=p_out,
        beta_gen=beta_gen(h, n, C, k), n=n, k=k,
    )
    data.stats = diagnostics(edge_index, data.y)
    return data


def check_graph(data, degree_tol: float = 0.5, homophily_tol: float = 0.02) -> dict:
    """Per-graph acceptance checks (PROTOCOL.md §C.2), logged for every generated graph."""
    st = data.stats
    counts = torch.bincount(data.y, minlength=data.num_classes)
    return {
        "empirical_h": st["edge_homophily"],
        "adjusted_h": st["adjusted_homophily"],
        "label_informativeness": st["label_informativeness"],
        "mean_degree": st["mean_degree"],
        "degree_ok": abs(st["mean_degree"] - data.k) <= degree_tol,
        "homophily_ok": abs(st["edge_homophily"] - data.target_h) <= homophily_tol,
        "balance_ok": bool((counts == counts[0]).all()),
    }

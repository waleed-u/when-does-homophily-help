"""Post-hoc inference: turn saved logits into MRF posteriors (PROTOCOL.md §E, §G).

Architectural rule of the project: `train.py` trains and saves logits; `infer.py` NEVER trains.
Every MRF variant (unweighted, similarity-weighted, clamped) and every engine (mean-field,
loopy BP, Gibbs) is therefore a pure function of the *same* frozen unary evidence — which is
exactly what makes "the prior imposed at inference time" comparable to "the prior imposed at
training time", and what makes engine-vs-engine differences attributable to the engine alone.
"""
import time

import torch

from .inference.gibbs import gibbs
from .inference.loopy_bp import loopy_bp
from .inference.mean_field import mean_field

ENGINES = {"mf": mean_field, "lbp": loopy_bp, "gibbs": gibbs}


def posterior(logits: torch.Tensor, edge_index: torch.Tensor, beta: float, engine: str = "mf",
              w=None, clamp_mask=None, clamp_y=None, **kwargs):
    """Returns (q [n, C] float64, info dict incl. infer_seconds)."""
    s = logits.double()
    t0 = time.perf_counter()
    if engine == "mf":
        q, info = mean_field(s, edge_index, beta, w=w, clamp_mask=clamp_mask, clamp_y=clamp_y,
                             check_elbo=False, **kwargs)
    elif engine == "lbp":
        q, info = loopy_bp(s, edge_index, beta, w=w, clamp_mask=clamp_mask, clamp_y=clamp_y,
                           **kwargs)
    elif engine == "gibbs":
        q, info = gibbs(s, edge_index, beta, w=w, clamp_mask=clamp_mask, clamp_y=clamp_y,
                        **kwargs)
    else:
        raise ValueError(f"unknown engine {engine}")
    info["infer_seconds"] = time.perf_counter() - t0
    return q, info


def flip_rate(logits: torch.Tensor, q: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of evaluated nodes whose argmax changes between beta=0 and the fitted posterior
    — a one-number answer to "how much work is the relational prior actually doing?"."""
    return float((logits[mask].argmax(1) != q[mask].argmax(1)).double().mean())

"""Reference ceilings available only because the CSBM's generative process is known
(PROTOCOL.md §E).

Oracle-F: the Bayes-optimal *feature-only* classifier. With equal spherical class-conditional
Gaussians this is the nearest-class-mean rule, so it bounds what any amount of feature
learning could achieve without the graph.

Oracle-G: true Gaussian log-likelihood unaries, to be combined with the generator-implied
coupling beta_gen — an approximate graph+features ceiling, so prior benefit can be reported as
a fraction of what is attainable rather than only as a delta over a GCN.
"""
import torch


def _neg_sq_dist(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    return -torch.cdist(x.double(), mu.double()) ** 2


def oracle_f_predictions(data) -> torch.Tensor:
    return _neg_sq_dist(data.x, data.mu).argmax(1)


def oracle_f_accuracy(data, mask: torch.Tensor | None = None) -> float:
    pred = oracle_f_predictions(data)
    if mask is not None:
        pred, y = pred[mask], data.y[mask]
    else:
        y = data.y
    return float((pred == y).double().mean())


def oracle_g_unaries(data) -> torch.Tensor:
    """s_i(c) = -||x_i - mu_c||^2 / (2 sigma_x^2)  (constants dropped)."""
    return _neg_sq_dist(data.x, data.mu) / (2.0 * data.sigma_x**2)

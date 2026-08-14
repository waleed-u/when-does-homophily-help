"""Pre-registered inference gates (PROTOCOL.md §H). No real-data run may launch until these pass.

Chain: enumeration validates variable elimination; VE validates BP on trees; the validated
exact engine then scores mean-field and Gibbs.
"""
import torch

from src.inference.exact import exact_marginals
from src.inference.gibbs import gibbs
from src.inference.loopy_bp import loopy_bp
from src.inference.mean_field import mean_field
from src.synthetic.small_graphs import chain, cycle, dense_sbm, grid, make_unaries, star


def tv_max(a, b):
    return float(0.5 * (a - b).abs().sum(1).max())


# ------------------------------------------------------------------ exact engine self-consistency
def test_enumeration_equals_ve():
    for builder in (lambda: chain(8), lambda: cycle(8), lambda: grid(3, 3)):
        ei, n = builder()
        s, _, _ = make_unaries(n, 3, 1.5, seed=0)
        for beta in (0.0, 0.5, 2.0, 4.0):
            a = exact_marginals(s, ei, beta, method="enumerate")
            b = exact_marginals(s, ei, beta, method="ve")
            assert tv_max(a, b) < 1e-10


def test_enumeration_equals_ve_with_clamping():
    ei, n = chain(8)
    s, _, _ = make_unaries(n, 3, 1.5, seed=1)
    cm = torch.zeros(n, dtype=torch.bool)
    cm[0] = True
    cy = torch.zeros(n, dtype=torch.long)
    cy[0] = 2
    a = exact_marginals(s, ei, 1.0, clamp_mask=cm, clamp_y=cy, method="enumerate")
    b = exact_marginals(s, ei, 1.0, clamp_mask=cm, clamp_y=cy, method="ve")
    assert tv_max(a, b) < 1e-10
    assert abs(float(a[0, 2]) - 1.0) < 1e-12          # clamped node is certain


def test_dense_graph_enumerable():
    ei, n, y = dense_sbm(14, seed=0)
    s, _, _ = make_unaries(n, 3, 1.0, seed=2, labels=y)
    a = exact_marginals(s, ei, 1.0, method="enumerate")
    b = exact_marginals(s, ei, 1.0, method="ve")
    assert tv_max(a, b) < 1e-10


# ---------------------------------------------------------------------------- BP exact on trees
def test_bp_exact_on_trees():
    """BP's tree fixed point is exact; accuracy tracks the message tolerance, so the gate is
    run at a tighter tolerance than the operational 1e-6."""
    for builder in (lambda: chain(15), lambda: star(16)):
        ei, n = builder()
        s, _, _ = make_unaries(n, 3, 1.5, seed=0)
        for beta in (0.5, 2.0):
            ex = exact_marginals(s, ei, beta, method="ve")
            bp, info = loopy_bp(s, ei, beta, tol=1e-13, max_iters=2000)
            assert info["converged"]
            assert tv_max(ex, bp) < 1e-8


# ------------------------------------------------------------------------ beta = 0 identities
def test_beta_zero_recovers_unary_softmax():
    ei, n = cycle(12)
    s, _, _ = make_unaries(n, 3, 1.0, seed=1)
    target = torch.softmax(s, dim=1)
    q_mf, _ = mean_field(s, ei, 0.0)
    q_bp, _ = loopy_bp(s, ei, 0.0)
    q_ex = exact_marginals(s, ei, 0.0, method="ve")
    assert tv_max(q_mf, target) < 1e-12
    assert tv_max(q_bp, target) < 1e-12
    assert tv_max(q_ex, target) < 1e-12


# ---------------------------------------------------------------------------- mean-field ELBO
def test_mean_field_elbo_monotone():
    ei, n = cycle(16)
    s, _, _ = make_unaries(n, 3, 1.0, seed=2)
    for beta in (0.5, 2.0):
        _, info = mean_field(s, ei, beta, check_elbo=True)
        tr = info["elbo_trace"]
        assert all(tr[i + 1] >= tr[i] - 1e-9 for i in range(len(tr) - 1))
    q, info = mean_field(s, ei, 0.5)
    assert info["converged"]
    assert torch.allclose(q.sum(1), torch.ones(n, dtype=torch.float64))


# ------------------------------------------------------------------------------------- Gibbs
def test_gibbs_matches_exact_small():
    ei, n = cycle(12)
    s, _, _ = make_unaries(n, 3, 1.0, seed=1)
    ex = exact_marginals(s, ei, 1.0, method="ve")
    q, info = gibbs(s, ei, 1.0, kept=2000, burn_in=1000, seed=0)
    assert tv_max(ex, q) < 0.02
    assert info["converged"]                       # R-hat / ESS gates (CHANGELOG A1)
    assert info["rhat_max"] < 1.01 and info["ess_min"] > 400


def test_gibbs_beta_zero():
    ei, n = cycle(12)
    s, _, _ = make_unaries(n, 3, 1.0, seed=1)
    q, _ = gibbs(s, ei, 0.0, kept=500, burn_in=200, seed=0)
    assert tv_max(q, torch.softmax(s, dim=1)) < 1e-12    # Rao-Blackwellised: exact at beta=0

"""Pre-registered gates that must pass before any experiment runs (PROTOCOL.md §H, §L).

Covered here: lambda=0 equivalence, regularizer properties, split/leakage invariants.
Inference gates live in test_inference_exact.py.
"""
import torch

from src.data import load_planetoid
from src.losses.regularizers import laplacian_loss, rule_violation_loss
from src.splits import assert_disjoint, train_mask_real
from src.train import train_model

CORA = load_planetoid("cora")


def _fit(reg, lam, seed=0, m=20):
    tm = train_mask_real(CORA, m, seed)
    return train_model("gcn", CORA, tm, CORA.val_mask, seed=seed, lr=0.01,
                       weight_decay=5e-4, lam=lam, reg=reg, max_epochs=60, patience=60)[0]


def test_lambda0_equals_plain_gcn():
    """lambda=0 must reproduce the baseline GCN exactly (playbook §19.1)."""
    a = _fit("none", 0.0)
    b = _fit("laplacian", 0.0)
    assert (a - b).abs().max().item() < 1e-5


def test_regularizer_changes_solution():
    """Sanity in the other direction: a nonzero lambda must actually do something."""
    a = _fit("none", 0.0)
    c = _fit("laplacian", 1.0)
    assert (a - c).abs().max().item() > 1e-4


def test_laplacian_loss_properties():
    ei = torch.tensor([[0, 1], [1, 0]])
    same = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
    assert laplacian_loss(same, ei).item() == 0.0
    diff = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert abs(laplacian_loss(diff, ei).item() - 2.0) < 1e-9   # ||(1,-1)||^2
    assert laplacian_loss(diff, ei).item() >= 0.0


def test_rule_violation_loss_properties():
    ei = torch.tensor([[0, 1], [1, 0]])
    p = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    assert abs(rule_violation_loss(p, ei).item() - (1 - (0.7 * 0.6 + 0.3 * 0.4))) < 1e-6
    onehot_same = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert abs(rule_violation_loss(onehot_same, ei).item()) < 1e-6


def test_splits_disjoint_and_stratified():
    for m in (1, 20):
        for seed in (0, 1, 2):
            tm = train_mask_real(CORA, m, seed)
            assert_disjoint(tm, CORA.val_mask, CORA.test_mask)
            counts = torch.bincount(CORA.y[tm], minlength=CORA.num_classes)
            assert bool((counts == m).all())


def test_test_labels_never_influence_training():
    """Permuting test labels must leave training output bit-identical."""
    tm = train_mask_real(CORA, 20, 0)
    base = train_model("gcn", CORA, tm, CORA.val_mask, seed=0, lr=0.01, weight_decay=5e-4,
                       max_epochs=40, patience=40)[0]

    import copy
    scrambled = copy.copy(CORA)
    y2 = CORA.y.clone()
    idx = torch.nonzero(CORA.test_mask).view(-1)
    g = torch.Generator().manual_seed(123)
    y2[idx] = y2[idx][torch.randperm(len(idx), generator=g)]
    scrambled.y = y2
    other = train_model("gcn", scrambled, tm, CORA.val_mask, seed=0, lr=0.01,
                        weight_decay=5e-4, max_epochs=40, patience=40)[0]
    assert torch.equal(base, other)

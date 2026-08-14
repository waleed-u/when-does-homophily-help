"""Split generation (PROTOCOL.md §D).

Real data: val/test are the fixed Planetoid masks; the m-labels/class train sample is
stratified from the remaining nodes and NESTED across m within a seed (the m-level
sample extends the smaller-m sample). SBM: per-graph val/test sampled after train.
"""
import torch


def _class_orders(y: torch.Tensor, eligible: torch.Tensor, seed: int) -> list[torch.Tensor]:
    """Per-class permutation of eligible node ids, deterministic in seed. Nesting across m
    follows from always taking the first m of the same permutation."""
    g = torch.Generator().manual_seed(seed)
    orders = []
    for c in range(int(y.max()) + 1):
        ids = torch.nonzero((y == c) & eligible, as_tuple=False).view(-1)
        orders.append(ids[torch.randperm(len(ids), generator=g)])
    return orders


def train_mask_real(data, m: int, seed: int) -> torch.Tensor:
    eligible = ~(data.val_mask | data.test_mask)
    orders = _class_orders(data.y, eligible, seed)
    mask = torch.zeros_like(data.val_mask)
    for ids in orders:
        if len(ids) < m:
            raise ValueError(f"class has only {len(ids)} eligible nodes < m={m}")
        mask[ids[:m]] = True
    return mask


def masks_sbm(y: torch.Tensor, m: int, seed: int, val_size: int = 500, test_size: int = 1000):
    """Train (stratified, nested) then val/test from the remainder; deterministic in seed."""
    n = len(y)
    eligible = torch.ones(n, dtype=torch.bool)
    orders = _class_orders(y, eligible, seed)
    train = torch.zeros(n, dtype=torch.bool)
    for ids in orders:
        train[ids[:m]] = True
    g = torch.Generator().manual_seed(seed + 10_000)
    rest = torch.nonzero(~train, as_tuple=False).view(-1)
    rest = rest[torch.randperm(len(rest), generator=g)]
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    val[rest[:val_size]] = True
    test[rest[val_size : val_size + test_size]] = True
    return train, val, test


def small_val_mask(data, train_mask: torch.Tensor, m_val: int, seed: int) -> torch.Tensor:
    """E13 small-validation ablation: m_val labels/class disjoint from train (and test)."""
    eligible = ~(train_mask | data.test_mask)
    orders = _class_orders(data.y, eligible, seed + 20_000)
    mask = torch.zeros_like(train_mask)
    for ids in orders:
        mask[ids[:m_val]] = True
    return mask


def assert_disjoint(*masks: torch.Tensor):
    total = sum(m.long() for m in masks)
    assert int(total.max()) <= 1, "split masks overlap"

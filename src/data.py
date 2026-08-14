"""Dataset loading. Real graphs come from PyG Planetoid; SBM graphs from src/synthetic/sbm.py.

Returned object (SimpleNamespace) is the single data interface used everywhere:
  x           float32 [n, d]   row-normalized features
  y           long    [n]
  edge_index  long    [2, 2|E|]  both directions, no self-loops
  num_classes int
  val_mask, test_mask  bool [n]  (fixed across seeds for real data; per-graph for SBM)
"""
import os
import ssl
from types import SimpleNamespace

import torch


def _certifi_ssl():
    # macOS framework Python ships without certs; point at the venv's certifi bundle.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    s = x.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return x / s


def load_planetoid(name: str, root: str = "data/") -> SimpleNamespace:
    _certifi_ssl()
    from torch_geometric.datasets import Planetoid

    ds = Planetoid(root=root, name={"cora": "Cora", "citeseer": "CiteSeer"}[name.lower()])
    d = ds[0]
    return SimpleNamespace(
        name=name.lower(),
        x=row_normalize(d.x.float()),
        y=d.y.long(),
        edge_index=d.edge_index,
        num_classes=ds.num_classes,
        # Planetoid's public val/test masks: fixed across seeds (PROTOCOL.md §D)
        val_mask=d.val_mask.clone(),
        test_mask=d.test_mask.clone(),
    )


def undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """[2, |E|] with each undirected edge counted once (i < j)."""
    ei = edge_index
    mask = ei[0] < ei[1]
    return ei[:, mask]

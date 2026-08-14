"""M1: standard GCN (Kipf & Welling 2017). The convolution is itself Laplacian smoothing,
so this baseline already carries an implicit homophily prior — the reason the study measures
the *marginal* effect of imposing the prior explicitly."""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 64, dropout: float = 0.5,
                 layers: int = 2):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [n_classes]
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(layers))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            if i > 0:
                x = torch.relu(x)
                x = torch.dropout(x, self.dropout, self.training)
            x = conv(x, edge_index)
        return x

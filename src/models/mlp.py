"""M0: feature-only MLP. Assumes nodes are i.i.d. given features — deliberately wrong,
which is what makes it the reference for how much the graph contributes."""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in: int, n_classes: int, hidden: int = 64, dropout: float = 0.5,
                 layers: int = 2):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [n_classes]
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            if i > 0:
                x = torch.relu(x)
                x = torch.dropout(x, self.dropout, self.training)
            x = lin(x)
        return x

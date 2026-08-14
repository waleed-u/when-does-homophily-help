"""Greedy graph coloring shared by the color-blocked MF updates and chromatic Gibbs.
Nodes within a color class share no edge, so block updates/resampling are valid."""
import networkx as nx
import torch


def greedy_coloring(edge_index: torch.Tensor, n: int) -> list[torch.Tensor]:
    """Returns a list of LongTensors (one per color) partitioning range(n)."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(edge_index.t().tolist())
    colors = nx.coloring.greedy_color(g, strategy="largest_first")
    k = max(colors.values()) + 1
    return [
        torch.tensor([v for v, c in colors.items() if c == ci], dtype=torch.long)
        for ci in range(k)
    ]

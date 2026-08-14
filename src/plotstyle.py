"""Consistent publication styling: one model -> one colour -> one label, in every figure."""
import matplotlib as mpl
import matplotlib.pyplot as plt

MODEL_LABEL = {
    "M0_mlp": "MLP (features only)",
    "M1_gcn": "GCN",
    "LP": "Label prop. (graph only)",
    "M2_lap": "GCN + Laplacian penalty",
    "M3_rule": "GCN + rule penalty",
    "M4_mrf_gcn": "MRF on GCN (mean-field)",
    "M4_mrf_mlp": "MRF on MLP (mean-field)",
    "M4_mrf_gcn_clamped": "MRF on GCN, labels clamped",
    "Oracle_F": "Oracle: features only",
    "Oracle_G": "Oracle: features + graph",
}
MODEL_COLOR = {
    "M0_mlp": "#8c8c8c", "M1_gcn": "#1f77b4", "LP": "#bcbd22", "M2_lap": "#ff7f0e",
    "M3_rule": "#d62728", "M4_mrf_gcn": "#2ca02c", "M4_mrf_mlp": "#9467bd",
    "M4_mrf_gcn_clamped": "#17becf", "Oracle_F": "#4d4d4d", "Oracle_G": "#000000",
}
MODEL_MARKER = {
    "M0_mlp": "s", "M1_gcn": "o", "LP": "v", "M2_lap": "^", "M3_rule": "D",
    "M4_mrf_gcn": "P", "M4_mrf_mlp": "X", "M4_mrf_gcn_clamped": "*",
}
ENGINE_LABEL = {"mf": "Mean-field (VI)", "lbp": "Loopy BP", "gibbs": "Gibbs (MCMC)"}
ENGINE_COLOR = {"mf": "#9467bd", "lbp": "#ff7f0e", "gibbs": "#2ca02c", "exact": "#000000"}


def use_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "legend.fontsize": 7.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "legend.frameon": False, "figure.autolayout": False,
    })


def save(fig, name, outdir="figures"):
    from pathlib import Path
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(p / f"{name}.{ext}")
    plt.close(fig)
    return str(p / f"{name}.pdf")

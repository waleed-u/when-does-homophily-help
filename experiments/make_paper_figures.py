"""Figures sized for the ICLR 2026 text block (5.5 inch line width).

The exploratory figures in make_figures.py are laid out for screen reading; these are the
versions that go in the report, drawn at final size so fonts render at their intended point
size rather than being shrunk by \\includegraphics.

Outputs to report_figures/ as PDF (vector) for direct upload to Overleaf.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.analysis import crossover, load_fidelity, load_runs, paired_summary, select_tuned
from src.config import BETA_GRID
from src.plotstyle import ENGINE_COLOR, ENGINE_LABEL, MODEL_COLOR, MODEL_LABEL, MODEL_MARKER

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_figures"
OUT.mkdir(exist_ok=True)
W = 5.5   # ICLR text width in inches

mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "lines.linewidth": 1.2, "lines.markersize": 3.6, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  report_figures/{name}.pdf")


# ------------------------------------------------------------------ Fig 1: method schematic
def fig1_schematic():
    fig, ax = plt.subplots(figsize=(W, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=0.8, facecolor=fc, edgecolor=ec))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.35)

    def arrow(x1, y1, x2, y2, color="#333333", style="-|>"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, linewidth=0.9, color=color))

    box(0.05, 1.25, 1.5, 0.95, "Graph $G$\nfeatures $X$", "#f0f0f0")
    box(1.95, 1.25, 1.65, 0.95, "Encoder\nMLP or GCN", "#dbe9f6")
    box(4.0, 1.25, 1.6, 0.95, "unary logits\n$s_i(c)$", "#f0f0f0")
    arrow(1.55, 1.72, 1.95, 1.72)
    arrow(3.60, 1.72, 4.00, 1.72)

    # upper branch: training-time prior
    box(2.35, 2.62, 3.4, 0.78, "training-time prior:  $\\lambda \\cdot \\mathcal{L}_{\\rm prior}(p)$\n"
        "acts on the network parameters", "#fde3cd", fs=6.8)
    arrow(2.78, 2.20, 2.78, 2.62, color="#c8721c")
    arrow(4.70, 2.62, 4.70, 2.20, color="#c8721c")

    # lower branch: inference-time prior
    box(6.05, 1.25, 2.15, 0.95,
        "Potts MRF over labels\n"
        "$p(y)\\!\\propto\\! \\exp\\{\\sum_i s_i(y_i)$\n"
        "$+\\,\\beta\\sum_{(i,j)\\in E}\\mathbf{1}[y_i{=}y_j]\\}$",
        "#d9ecd9", fs=5.8)
    arrow(5.60, 1.72, 6.05, 1.72)
    box(8.55, 1.25, 1.4, 0.95, "posterior\n$q_i(c)$", "#f0f0f0")
    arrow(8.20, 1.72, 8.55, 1.72)

    box(6.05, 0.06, 3.9, 0.72,
        "inference-time prior: mean-field / loopy BP / Gibbs\nvalidated against exact marginals",
        "#e8e2f2", fs=6.6)
    arrow(7.1, 0.78, 7.1, 1.25, color="#5b4b8a")

    ax.text(0.05, 0.30, "one assumption, two places to impose it", fontsize=7.4,
            style="italic", color="#444444")
    save(fig, "fig1_schematic")


# ---------------------------------------------------------------- Fig 2: Cora low-label curves
def fig2_low_label(runs):
    cora = select_tuned(runs[(runs.dataset == "cora") & (runs.split_seed < 10)])
    piv = cora.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                           values="test_acc")
    ms = sorted(piv.index.get_level_values(0).unique())
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.25))

    ax = axes[0]
    for mod in ["M0_mlp", "LP", "M1_gcn", "M2_lap", "M4_mrf_mlp", "M4_mrf_gcn"]:
        mean = piv[mod].groupby("label_per_class").mean() * 100
        sd = piv[mod].groupby("label_per_class").std() * 100
        ax.errorbar(ms, mean.loc[ms], yerr=sd.loc[ms], label=MODEL_LABEL[mod],
                    color=MODEL_COLOR[mod], marker=MODEL_MARKER.get(mod, "o"),
                    capsize=1.5, elinewidth=0.7)
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels(ms)
    ax.set_xlabel("labels per class $m$")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("(a) accuracy vs. label budget")
    ax.legend(loc="lower right", ncol=1, handlelength=1.4)

    ax = axes[1]
    for mod, base in (("M2_lap", "M1_gcn"), ("M4_mrf_gcn", "M1_gcn"), ("M4_mrf_mlp", "M0_mlp")):
        means, los, his = [], [], []
        for m in ms:
            p = piv.xs(m, level="label_per_class")
            r = paired_summary(p[mod].values, p[base].values)
            means.append(r["mean_diff"] * 100)
            los.append((r["mean_diff"] - r["ci_lo"]) * 100)
            his.append((r["ci_hi"] - r["mean_diff"]) * 100)
        short = {"M2_lap": "Laplacian $-$ GCN", "M4_mrf_gcn": "MRF(GCN) $-$ GCN",
                 "M4_mrf_mlp": "MRF(MLP) $-$ MLP"}[mod]
        ax.errorbar(ms, means, yerr=[los, his], color=MODEL_COLOR[mod],
                    marker=MODEL_MARKER.get(mod, "o"), capsize=1.5, elinewidth=0.7, label=short)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels(ms)
    ax.set_xlabel("labels per class $m$")
    ax.set_ylabel("paired $\\Delta$ accuracy (pts)")
    ax.set_title("(b) benefit of the prior (paired, 95% CI)")
    ax.legend(loc="upper left", handlelength=1.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig2_low_label")


# ------------------------------------------------------------------- Fig 3: CSBM heatmaps
def sbm_frames(runs):
    sbm = runs[runs.dataset == "sbm"]
    mrf = sbm[sbm.model == "M4_mrf_mlp"]
    tuned = select_tuned(mrf, extra_group=["target_h"])
    ref = tuned[(tuned.target_h == 0.90) & (tuned.label_per_class == 5)]["beta"].dropna()
    beta_dog = float(np.sort(ref.values)[(len(ref) - 1) // 2])
    dog = mrf[mrf.beta == beta_dog]
    base = sbm[sbm.model == "M0_mlp"]
    return sbm, tuned, dog, base, beta_dog


def _delta_grid(frame, base, ms, hs):
    g = np.full((len(ms), len(hs)), np.nan)
    for i, m in enumerate(ms):
        for j, h in enumerate(hs):
            a = frame[(frame.label_per_class == m) & (frame.target_h == h)] \
                .set_index("split_seed")["test_acc"]
            b = base[(base.label_per_class == m) & (base.target_h == h)] \
                .set_index("split_seed")["test_acc"]
            c = a.index.intersection(b.index)
            if len(c) >= 3:
                g[i, j] = paired_summary(a.loc[c].values, b.loc[c].values)["mean_diff"] * 100
    return g


def fig3_heatmap(runs):
    sbm, tuned, dog, base, beta_dog = sbm_frames(runs)
    ms = sorted(sbm.label_per_class.unique())
    hs = sorted(sbm.target_h.dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15))
    for ax, (frame, title) in zip(axes, [
            (dog, f"(a) dogmatic prior ($\\beta$ fixed $=\\,${beta_dog:g})"),
            (tuned, "(b) tuned prior ($\\beta$ chosen on validation)")]):
        g = _delta_grid(frame, base, ms, hs)
        v = 56.0
        im = ax.imshow(g, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", origin="lower")
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([f"{h:g}" for h in hs])
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels([int(m) for m in ms])
        ax.set_xlabel("edge homophily $h$")
        ax.set_ylabel("labels/class $m$")
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(ms)):
            for j in range(len(hs)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i,j]:+.0f}", ha="center", va="center", fontsize=6.2,
                            color="white" if abs(g[i, j]) > 33 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("$\\Delta$ acc. vs MLP (pts)", fontsize=6.6)
        cb.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.4)
    save(fig, "fig3_sbm_heatmap")


# ------------------------------------------- Fig 4: delta-vs-h with crossover, beta calibration
def fig4_crossover(runs):
    sbm, tuned, dog, base, beta_dog = sbm_frames(runs)
    hs = sorted(sbm.target_h.dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1))

    ax = axes[0]
    for m, ls, col in zip([1, 5, 20], ["-", "--", ":"], ["#9467bd", "#2ca02c", "#1f77b4"]):
        ys, los, his = [], [], []
        for h in hs:
            a = dog[(dog.label_per_class == m) & (dog.target_h == h)] \
                .set_index("split_seed")["test_acc"]
            b = base[(base.label_per_class == m) & (base.target_h == h)] \
                .set_index("split_seed")["test_acc"]
            c = a.index.intersection(b.index)
            r = paired_summary(a.loc[c].values, b.loc[c].values)
            ys.append(r["mean_diff"] * 100)
            los.append((r["mean_diff"] - r["ci_lo"]) * 100)
            his.append((r["ci_hi"] - r["mean_diff"]) * 100)
        ax.errorbar(hs, ys, yerr=[los, his], ls=ls, marker="o", color=col, capsize=1.5,
                    elinewidth=0.7, label=f"$m={m}$")
        hx = crossover(np.array(hs), np.array(ys))
        if not np.isnan(hx):
            ax.plot([hx], [0], marker="v", color=col, ms=5, clip_on=False)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    ax.set_xlabel("edge homophily $h$")
    ax.set_ylabel("$\\Delta$ acc. vs MLP (pts)")
    ax.set_title("(a) dogmatic prior: benefit, harm, $h^\\star$")
    ax.legend(loc="upper left", handlelength=1.8)

    ax = axes[1]
    bg = sbm.groupby("target_h")["beta_gen"].first()
    sb = tuned[tuned.label_per_class == 5].groupby("target_h")["beta"].median()
    ax.plot(bg.index, bg.values, "k-", marker="s", label=r"$\beta_{\rm gen}=\log(p_{\rm in}/p_{\rm out})$")
    ax.plot(sb.index, sb.values, color=MODEL_COLOR["M4_mrf_mlp"], marker="X",
            label=r"validation-selected $\beta^\star$")
    ax.axhline(max(BETA_GRID), color="grey", lw=0.6, ls=":")
    ax.text(0.055, max(BETA_GRID) + 0.12, "grid max", fontsize=5.8, color="grey")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("edge homophily $h$")
    ax.set_ylabel(r"coupling $\beta$")
    ax.set_title(r"(b) tuning does not recover $\beta_{\rm gen}$")
    ax.legend(loc="upper left", handlelength=1.8)
    fig.tight_layout(pad=0.4)
    save(fig, "fig4_crossover_beta")


# ------------------------------------------------------- Fig 5: strength sensitivity/consistency
def fig5_sensitivity(runs):
    cora = runs[(runs.dataset == "cora") & (runs.split_seed < 10)]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.95))

    ax = axes[0]
    for m, ls in ((2, "--"), (5, "-")):
        g = cora[(cora.model == "M2_lap") & (cora.label_per_class == m)] \
            .groupby("lambda")["test_acc"].agg(["mean", "std"]) * 100
        ax.errorbar(g.index, g["mean"], yerr=g["std"], ls=ls, marker="^",
                    color=MODEL_COLOR["M2_lap"], capsize=1.5, elinewidth=0.7, label=f"$m={m}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\lambda$ (training-time)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("(a) Laplacian strength")
    ax.legend(handlelength=1.6)

    ax = axes[1]
    for mod, ls in (("M4_mrf_gcn", "-"), ("M4_mrf_mlp", "--")):
        g = cora[(cora.model == mod) & (cora.label_per_class == 5)] \
            .groupby("beta")["test_acc"].agg(["mean", "std"]) * 100
        lab = "MRF(GCN)" if mod == "M4_mrf_gcn" else "MRF(MLP)"
        ax.errorbar(g.index, g["mean"], yerr=g["std"], ls=ls, marker="P",
                    color=MODEL_COLOR[mod], capsize=1.5, elinewidth=0.7, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ (inference-time)")
    ax.set_title(r"(b) MRF coupling, $m=5$")
    ax.legend(handlelength=1.6)

    ax = axes[2]
    g = runs[(runs.dataset == "sbm") & (runs.model == "M4_mrf_mlp") &
             (runs.label_per_class == 5) & (runs.target_h == 0.05)]
    acc = g.groupby("beta")["test_acc"].mean() * 100
    agree = g.groupby("beta")["edge_agreement"].mean()
    ax.plot(acc.index, acc.values, color="#d62728", marker="P", label="accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot(agree.index, agree.values, color="k", ls=":", marker=".",
             label="edge agreement")
    ax2.set_ylabel("edge agreement", fontsize=7)
    ax2.tick_params(labelsize=6.5)
    ax2.grid(False)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_title("(c) CSBM $h{=}0.05$: collapse")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=5.8, handlelength=1.4)
    fig.tight_layout(pad=0.4)
    save(fig, "fig5_sensitivity")


# ----------------------------------------------------------------- Fig 6: inference fidelity
def fig6_fidelity(fid):
    d = fid[(fid.method != "exact") & (fid.structure != "grid_budget")]
    order = ["chain", "star", "cycle", "grid", "dense"]
    titles = {"chain": "chain (tree)", "star": "star (hub tree)", "cycle": "single cycle",
              "grid": "$4{\\times}4$ grid", "dense": "dense 2-block"}
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.5), sharey=True)
    for ax, st in zip(axes, order):
        sub = d[d.structure == st]
        for eng in ("mf", "lbp", "gibbs"):
            g = sub[sub.method == eng].groupby("beta")["tv_mean"].mean()
            ax.plot(g.index, g.clip(lower=1e-9), color=ENGINE_COLOR[eng], marker="o", ms=2.2,
                    label=ENGINE_LABEL[eng])
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 2)
        ax.set_title(titles[st], fontsize=7)
        ax.set_xlabel(r"$\beta$", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("mean TV to exact", fontsize=7)
    axes[0].legend(loc="lower right", fontsize=5.4, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig6_fidelity")


def fig7_bias_variance(fid):
    b = fid[fid.structure == "grid_budget"]
    fig, ax = plt.subplots(figsize=(2.7, 1.9))
    gb = b[b.method == "gibbs"].groupby("gibbs_kept")["tv_mean"].agg(["mean", "std"])
    ax.errorbar(gb.index, gb["mean"], yerr=gb["std"], color=ENGINE_COLOR["gibbs"],
                marker="o", capsize=1.5, elinewidth=0.7, label="Gibbs (MC variance)")
    for eng in ("mf", "lbp"):
        v = b[b.method == eng]["tv_mean"].mean()
        ax.axhline(v, color=ENGINE_COLOR[eng], ls="--",
                   label=f"{ENGINE_LABEL[eng]}: bias floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("kept Gibbs sweeps")
    ax.set_ylabel("mean TV to exact")
    ax.set_title(r"$4{\times}4$ grid, $\beta=1.75$", fontsize=7.5)
    ax.legend(fontsize=5.8, handlelength=1.4)
    fig.tight_layout(pad=0.3)
    save(fig, "fig7_bias_variance")


if __name__ == "__main__":
    runs, fid = load_runs(), load_fidelity()
    print("Report figures (sized for the ICLR 5.5in text block):")
    fig1_schematic()
    fig2_low_label(runs)
    fig3_heatmap(runs)
    fig4_crossover(runs)
    fig5_sensitivity(runs)
    fig6_fidelity(fid)
    fig7_bias_variance(fid)

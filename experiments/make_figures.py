"""Generate every figure and table from results/raw/*.csv ONLY (PROTOCOL.md §G).

No number in the paper is typed by hand; each is produced here from the logged runs, so every
figure and table is traceable to raw experimental output.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import (bca_ci, crossover, load_fidelity, load_runs, paired_summary,
                          select_tuned)
from src.config import BETA_GRID, SBM_H
from src.plotstyle import (ENGINE_COLOR, ENGINE_LABEL, MODEL_COLOR, MODEL_LABEL, MODEL_MARKER,
                           save, use_paper_style)

ROOT = Path(__file__).resolve().parent.parent
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True)
use_paper_style()


def write_table(df, name, float_fmt="%.3f", **kw):
    df.to_csv(TAB / f"{name}.csv", index=kw.pop("index", True))
    with open(TAB / f"{name}.tex", "w") as f:
        f.write(df.to_latex(float_format=float_fmt, **kw))
    print(f"  wrote tables/{name}.csv|.tex")


# ------------------------------------------------------------------ F2: Cora low-label curves
def fig_low_label(runs):
    # common seed set across all models (the 30-seed top-up covers only the endpoint models)
    cora = select_tuned(runs[(runs.dataset == "cora") & (runs.split_seed < 10)])
    piv = cora.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                           values="test_acc")
    ms = sorted(piv.index.get_level_values(0).unique())
    order = ["M0_mlp", "LP", "M1_gcn", "M2_lap", "M4_mrf_mlp", "M4_mrf_gcn"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for mod in order:
        if mod not in piv:
            continue
        mean = piv[mod].groupby("label_per_class").mean() * 100
        sd = piv[mod].groupby("label_per_class").std() * 100
        ax.errorbar(ms, mean.loc[ms], yerr=sd.loc[ms], label=MODEL_LABEL[mod],
                    color=MODEL_COLOR[mod], marker=MODEL_MARKER.get(mod, "o"), capsize=2)
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels(ms)
    ax.set_xlabel("labels per class $m$ (log scale)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("(a) Cora: accuracy vs. label budget")
    ax.legend(loc="lower right", ncol=1)

    ax = axes[1]
    for mod, base in (("M2_lap", "M1_gcn"), ("M4_mrf_gcn", "M1_gcn"), ("M4_mrf_mlp", "M0_mlp")):
        means, los, his = [], [], []
        for m in ms:
            p = piv.xs(m, level="label_per_class")
            r = paired_summary(p[mod].values, p[base].values)
            means.append(r["mean_diff"] * 100)
            los.append((r["mean_diff"] - r["ci_lo"]) * 100)
            his.append((r["ci_hi"] - r["mean_diff"]) * 100)
        ax.errorbar(ms, means, yerr=[los, his], color=MODEL_COLOR[mod],
                    marker=MODEL_MARKER.get(mod, "o"), capsize=2,
                    label=f"{MODEL_LABEL[mod]}\n  $-$ {MODEL_LABEL[base]}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(ms)
    ax.set_xticklabels(ms)
    ax.set_xlabel("labels per class $m$ (log scale)")
    ax.set_ylabel("paired $\\Delta$ accuracy (pts)")
    ax.set_title("(b) Benefit of the prior, paired by seed")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "F2_low_label_cora")


# ------------------------------------------------- F3: CSBM homophily heatmap (dogmatic regime)
def sbm_regimes(runs):
    """Returns (tuned, dogmatic) frames for CSBM prior models."""
    sbm = runs[runs.dataset == "sbm"].copy()
    mrf = sbm[sbm.model.isin(["M4_mrf_mlp", "M4_mrf_gcn"])]
    tuned = select_tuned(mrf, group=["dataset", "model", "label_per_class", "split_seed"],
                         extra_group=["target_h"])
    # dogmatic beta: lower median of the tuned betas at the reference cell (h=0.9, m=5)
    dog = {}
    for mod in ("M4_mrf_mlp", "M4_mrf_gcn"):
        ref = tuned[(tuned.model == mod) & (tuned.target_h == 0.90) &
                    (tuned.label_per_class == 5)]["beta"].dropna().sort_values().values
        dog[mod] = float(ref[(len(ref) - 1) // 2]) if len(ref) else np.nan
    dogmatic = pd.concat([mrf[(mrf.model == mod) & (mrf.beta == b)] for mod, b in dog.items()])
    return tuned, dogmatic, dog


def fig_sbm_heatmap(runs):
    sbm = runs[runs.dataset == "sbm"]
    tuned, dogmatic, dog = sbm_regimes(runs)
    base = sbm[sbm.model.isin(["M0_mlp", "M1_gcn"])]
    ms = sorted(sbm.label_per_class.unique())
    hs = sorted(sbm.target_h.unique())

    def delta_grid(frame, mod, base_mod):
        g = np.full((len(ms), len(hs)), np.nan)
        ci = np.full((len(ms), len(hs)), np.nan)
        for i, m in enumerate(ms):
            for j, h in enumerate(hs):
                a = frame[(frame.model == mod) & (frame.label_per_class == m) &
                          (frame.target_h == h)].set_index("split_seed")["test_acc"]
                b = base[(base.model == base_mod) & (base.label_per_class == m) &
                         (base.target_h == h)].set_index("split_seed")["test_acc"]
                common = a.index.intersection(b.index)
                if len(common) >= 3:
                    r = paired_summary(a.loc[common].values, b.loc[common].values)
                    g[i, j] = r["mean_diff"] * 100
                    ci[i, j] = r["ci_lo"] * 100
        return g, ci

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for ax, (frame, title, tag) in zip(axes, [
            (dogmatic, f"(a) Dogmatic prior ($\\beta$ fixed = {dog['M4_mrf_mlp']:g})", "dog"),
            (tuned, "(b) Tuned prior ($\\beta$ chosen per cell)", "tuned")]):
        g, _ = delta_grid(frame, "M4_mrf_mlp", "M0_mlp")
        v = np.nanmax(np.abs(g))
        im = ax.imshow(g, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", origin="lower")
        ax.set_xticks(range(len(hs)))
        ax.set_xticklabels([f"{h:g}" for h in hs])
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels(ms)
        ax.set_xlabel("edge homophily $h$")
        ax.set_ylabel("labels per class $m$")
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(ms)):
            for j in range(len(hs)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i,j]:+.1f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(g[i, j]) > 0.6 * v else "black")
        fig.colorbar(im, ax=ax, label="$\\Delta$ acc. vs MLP (pts)")
    fig.tight_layout()
    path = save(fig, "F3_sbm_heatmap")

    # companion: delta vs h curve with crossover, plus beta* vs beta_gen
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for m, ls in zip(ms, ["-", "--", ":"]):
        ys, los, his = [], [], []
        for h in hs:
            a = dogmatic[(dogmatic.model == "M4_mrf_mlp") & (dogmatic.label_per_class == m) &
                         (dogmatic.target_h == h)].set_index("split_seed")["test_acc"]
            b = base[(base.model == "M0_mlp") & (base.label_per_class == m) &
                     (base.target_h == h)].set_index("split_seed")["test_acc"]
            c = a.index.intersection(b.index)
            r = paired_summary(a.loc[c].values, b.loc[c].values)
            ys.append(r["mean_diff"] * 100)
            los.append((r["mean_diff"] - r["ci_lo"]) * 100)
            his.append((r["ci_hi"] - r["mean_diff"]) * 100)
        ax.errorbar(hs, ys, yerr=[los, his], ls=ls, marker="o", capsize=2, label=f"$m$={m}")
        hstar = crossover(np.array(hs), np.array(ys))
        if not np.isnan(hstar):
            ax.axvline(hstar, color="grey", lw=0.7, ls=":")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("edge homophily $h$")
    ax.set_ylabel("$\\Delta$ acc. vs MLP (pts)")
    ax.set_title("(a) Dogmatic prior: benefit and harm")
    ax.legend()

    ax = axes[1]
    bg = runs[(runs.dataset == "sbm")].groupby("target_h")["beta_gen"].first()
    sel_beta = tuned[(tuned.model == "M4_mrf_mlp") & (tuned.label_per_class == 5)] \
        .groupby("target_h")["beta"].median()
    ax.plot(bg.index, bg.values, "k-", marker="s", label=r"generator $\beta_{\rm gen}=\log(p_{in}/p_{out})$")
    ax.plot(sel_beta.index, sel_beta.values, color=MODEL_COLOR["M4_mrf_mlp"], marker="X",
            label=r"validation-selected $\beta^*$ (median)")
    ax.axhline(max(BETA_GRID), color="grey", lw=0.7, ls=":")
    ax.text(0.06, max(BETA_GRID) * 1.03, "grid max", fontsize=6.5, color="grey")
    ax.set_xlabel("edge homophily $h$")
    ax.set_ylabel(r"coupling $\beta$")
    ax.set_title(r"(b) Does tuning recover the true coupling?")
    ax.legend()
    fig.tight_layout()
    save(fig, "F3b_sbm_curve_and_beta")
    return path, dog


# ------------------------------------------------------- F4: strength sensitivity / consistency
def fig_sensitivity(runs):
    cora = runs[(runs.dataset == "cora") & (runs.label_per_class.isin([2, 5]))]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))

    ax = axes[0]
    for m, ls in ((2, "--"), (5, "-")):
        g = cora[(cora.model == "M2_lap") & (cora.label_per_class == m)] \
            .groupby("lambda")["test_acc"].agg(["mean", "std"]) * 100
        ax.errorbar(g.index, g["mean"], yerr=g["std"], ls=ls, marker="^",
                    color=MODEL_COLOR["M2_lap"], capsize=2, label=f"Laplacian, $m$={m}")
    ax.set_xscale("log")
    ax.set_xlabel("$\\lambda$ (training-time)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("(a) Regularizer strength")
    ax.legend()

    ax = axes[1]
    for mod, ls in (("M4_mrf_gcn", "-"), ("M4_mrf_mlp", "--")):
        g = cora[(cora.model == mod) & (cora.label_per_class == 5)] \
            .groupby("beta")["test_acc"].agg(["mean", "std"]) * 100
        ax.errorbar(g.index, g["mean"], yerr=g["std"], ls=ls, marker="P",
                    color=MODEL_COLOR[mod], capsize=2, label=MODEL_LABEL[mod])
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ (inference-time)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title(r"(b) MRF coupling strength, $m$=5")
    ax.legend()

    ax = axes[2]
    g = cora[(cora.model == "M4_mrf_gcn") & (cora.label_per_class == 5)]
    acc = g.groupby("beta")["test_acc"].mean() * 100
    rule = g.groupby("beta")["rule_score"].mean()
    agree = g.groupby("beta")["edge_agreement"].mean()
    ax.plot(acc.index, acc.values, color=MODEL_COLOR["M4_mrf_gcn"], marker="P", label="accuracy (%)")
    ax2 = ax.twinx()
    ax2.plot(rule.index, rule.values, color="grey", ls="--", marker=".", label="rule satisfaction $R$")
    ax2.plot(agree.index, agree.values, color="k", ls=":", marker=".", label="argmax edge agreement")
    ax2.set_ylabel("edge consistency")
    ax2.grid(False)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("(c) Consistency $\\neq$ accuracy")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=6.5)
    fig.tight_layout()
    return save(fig, "F4_sensitivity")


# ------------------------------------------------------------- F5: inference fidelity vs exact
def fig_fidelity(fid):
    d = fid[fid.method != "exact"].copy()
    d = d[d.structure != "grid_budget"]
    order = ["chain", "tree", "star", "cycle", "grid", "dense"]
    titles = {"chain": "chain (tree)", "tree": "binary tree", "star": "star (hub)",
              "cycle": "single cycle", "grid": "4x4 grid", "dense": "dense 2-block"}
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.2), sharex=True, sharey=True)
    for ax, st in zip(axes.ravel(), order):
        sub = d[d.structure == st]
        for eng in ("mf", "lbp", "gibbs"):
            g = sub[sub.method == eng].groupby("beta")["tv_mean"].agg(["mean", "std"])
            ax.plot(g.index, g["mean"].clip(lower=1e-9), color=ENGINE_COLOR[eng],
                    marker="o", ms=3, label=ENGINE_LABEL[eng])
            ax.fill_between(g.index, (g["mean"] - g["std"]).clip(lower=1e-9),
                            g["mean"] + g["std"], color=ENGINE_COLOR[eng], alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(titles[st])
        ax.set_ylim(1e-9, 1)
    for ax in axes[-1]:
        ax.set_xlabel(r"coupling $\beta$")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean TV to exact")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Approximate marginals vs exact inference (20 draws; band = $\\pm$1 sd)",
                 fontsize=9.5)
    fig.tight_layout()
    p1 = save(fig, "F5_fidelity")

    # bias vs variance panel
    b = fid[fid.structure == "grid_budget"]
    if len(b):
        fig, ax = plt.subplots(figsize=(3.6, 2.7))
        gb = b[b.method == "gibbs"].groupby("gibbs_kept")["tv_mean"].agg(["mean", "std"])
        ax.errorbar(gb.index, gb["mean"], yerr=gb["std"], color=ENGINE_COLOR["gibbs"],
                    marker="o", capsize=2, label="Gibbs (MC variance)")
        for eng in ("mf", "lbp"):
            v = b[b.method == eng]["tv_mean"].mean()
            ax.axhline(v, color=ENGINE_COLOR[eng], ls="--",
                       label=f"{ENGINE_LABEL[eng]} bias floor")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("kept Gibbs sweeps")
        ax.set_ylabel("mean TV to exact")
        ax.set_title(r"Bias vs variance (4x4 grid, $\beta$=1.75)")
        ax.legend(fontsize=6.8)
        fig.tight_layout()
        save(fig, "F6_bias_variance")
    return p1


if __name__ == "__main__":
    runs = load_runs()
    fid = load_fidelity()
    print("Figures:")
    print(" ", fig_low_label(runs))
    print(" ", fig_sbm_heatmap(runs)[0])
    print(" ", fig_sensitivity(runs))
    print(" ", fig_fidelity(fid))

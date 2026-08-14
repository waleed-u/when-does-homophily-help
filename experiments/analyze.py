"""Full statistical analysis: tables, confirmatory verdicts, and processed result files.

Reads only results/raw/*.csv. Selection is validation-only (src.analysis.select_tuned never
reads a test column). The three pre-registered endpoints are tested with Holm-corrected
Wilcoxon; everything else is reported as an estimate with a bootstrap interval and a sign
count, explicitly labelled exploratory.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import (bca_ci, crossover, load_fidelity, load_runs, holm, paired_summary,
                          select_tuned)
from src.config import BETA_GRID
from src.data import load_planetoid
from src.homophily import diagnostics
from src.plotstyle import MODEL_LABEL

ROOT = Path(__file__).resolve().parent.parent
TAB, PROC = ROOT / "tables", ROOT / "results" / "processed"
TAB.mkdir(exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)


def write(df, name, index=True, float_format="%.3f"):
    df.to_csv(TAB / f"{name}.csv", index=index)
    with open(TAB / f"{name}.tex", "w") as f:
        f.write(df.to_latex(index=index, float_format=float_format))
    print(f"    tables/{name}.csv")


# ------------------------------------------------------------------------------------- T1
def table_datasets(runs):
    rows = []
    for name in ("cora", "citeseer"):
        d = load_planetoid(name)
        dg = diagnostics(d.edge_index, d.y)
        rows.append({"dataset": name.capitalize(), "nodes": d.x.shape[0],
                     "undirected edges": d.edge_index.shape[1] // 2,
                     "features": d.x.shape[1], "classes": d.num_classes,
                     "mean degree": round(dg["mean_degree"], 2),
                     "edge homophily": round(dg["edge_homophily"], 3),
                     "adjusted homophily": round(dg["adjusted_homophily"], 3),
                     "label informativeness": round(dg["label_informativeness"], 3)})
    sbm = runs[runs.dataset == "sbm"]
    for h in sorted(sbm.target_h.dropna().unique()):
        s = sbm[sbm.target_h == h].iloc[0]
        rows.append({"dataset": f"CSBM h={h:g}", "nodes": 2800, "undirected edges": "~11.2k",
                     "features": 32, "classes": 7,
                     "mean degree": 8.0,
                     "edge homophily": round(float(s.empirical_h), 3),
                     "adjusted homophily": round(float(s.adjusted_h), 3),
                     "label informativeness": round(float(s.label_informativeness), 3)})
    df = pd.DataFrame(rows).set_index("dataset")
    write(df, "T1_datasets")
    return df


# ------------------------------------------------------------------------------------- T2
COMMON_SEEDS = set(range(10))   # every model was run on these; the 30-seed top-up covers
                                # only the two models in confirmatory endpoint C1, so
                                # descriptive tables use the common set for comparability.


def table_main(runs):
    out = []
    for ds in ("cora", "citeseer"):
        sel = select_tuned(runs[(runs.dataset == ds) & runs.split_seed.isin(COMMON_SEEDS)])
        piv = sel.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                              values="test_acc")
        for m in sorted(piv.index.get_level_values(0).unique()):
            p = piv.xs(m, level="label_per_class")
            row = {"dataset": ds, "m": m}
            for mod in ["M0_mlp", "LP", "M1_gcn", "M2_lap", "M3_rule", "M4_mrf_mlp",
                        "M4_mrf_gcn"]:
                if mod in p and p[mod].notna().sum():
                    row[mod] = f"{p[mod].mean()*100:.1f} ± {p[mod].std()*100:.1f}"
            out.append(row)
    df = pd.DataFrame(out).set_index(["dataset", "m"])
    df.columns = [MODEL_LABEL.get(c, c) for c in df.columns]
    write(df, "T2_main_results", float_format="%s")
    return df


# ------------------------------------------------------------------------------------- T3
def table_mechanism(runs, floor):
    sel = select_tuned(runs[(runs.dataset == "cora") & runs.split_seed.isin(COMMON_SEEDS)])
    piv = sel.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                          values="test_acc")
    rows = []
    for m in sorted(piv.index.get_level_values(0).unique()):
        p = piv.xs(m, level="label_per_class")
        for mod, base in (("M2_lap", "M1_gcn"), ("M3_rule", "M1_gcn"),
                          ("M4_mrf_gcn", "M1_gcn"), ("M4_mrf_gcn_clamped", "M1_gcn"),
                          ("M4_mrf_mlp", "M0_mlp")):
            if mod not in p:
                continue
            r = paired_summary(p[mod].values, p[base].values)
            rows.append({"m": m, "model": MODEL_LABEL[mod], "baseline": MODEL_LABEL[base],
                         "Δ acc (pts)": round(r["mean_diff"] * 100, 2),
                         "95% CI": f"[{r['ci_lo']*100:+.2f}, {r['ci_hi']*100:+.2f}]",
                         "seeds +": f"{r['n_positive']}/{r['n']}",
                         "p (exploratory)": round(r["p_wilcoxon"], 4)})
    df = pd.DataFrame(rows).set_index(["m", "model"])
    write(df, "T3_mechanism", float_format="%s")
    return df


# ------------------------------------------------------------ confirmatory endpoints C1-C3
def confirmatory(runs):
    res, pvals, detail = {}, {}, {}

    # C1 (H1): Cora, [Acc(M4_mrf_mlp) - Acc(M0_mlp)](m=2) - [same](m=20) > 0
    sel = select_tuned(runs[runs.dataset == "cora"])
    piv = sel.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                          values="test_acc")
    d = {}
    for m in (2, 20):
        p = piv.xs(m, level="label_per_class")
        d[m] = (p["M4_mrf_mlp"] - p["M0_mlp"]).dropna()
    common = d[2].index.intersection(d[20].index)
    theta = (d[2].loc[common] - d[20].loc[common]).values
    r = paired_summary(theta + 0.0, np.zeros_like(theta), "C1")
    res["C1"] = {"hypothesis": "H1 scarcity: benefit larger at m=2 than m=20 (MRF on MLP vs MLP)",
                 "n_seeds": int(len(common)), "theta_mean_pts": r["mean_diff"] * 100,
                 "ci_pts": [r["ci_lo"] * 100, r["ci_hi"] * 100],
                 "seeds_positive": f"{r['n_positive']}/{r['n']}", "p": r["p_wilcoxon"],
                 "delta_m2_pts": float(d[2].loc[common].mean() * 100),
                 "delta_m20_pts": float(d[20].loc[common].mean() * 100)}
    pvals["C1"] = r["p_wilcoxon"]
    detail["C1"] = theta

    # C2/C3 (H2): CSBM dogmatic regime, m=5, Delta = Acc(M4_mrf_mlp) - Acc(M0_mlp)
    sbm = runs[runs.dataset == "sbm"]
    mrf = sbm[sbm.model == "M4_mrf_mlp"]
    tuned = select_tuned(mrf, extra_group=["target_h"])
    ref = tuned[(tuned.target_h == 0.90) & (tuned.label_per_class == 5)]["beta"].dropna()
    beta_dog = float(np.sort(ref.values)[(len(ref) - 1) // 2]) if len(ref) else np.nan
    dog = mrf[(mrf.beta == beta_dog) & (mrf.label_per_class == 5)]
    base = sbm[(sbm.model == "M0_mlp") & (sbm.label_per_class == 5)]

    def delta_at(h):
        a = dog[dog.target_h == h].set_index("split_seed")["test_acc"]
        b = base[base.target_h == h].set_index("split_seed")["test_acc"]
        c = a.index.intersection(b.index)
        return (a.loc[c] - b.loc[c])

    d09, d05, d005 = delta_at(0.90), delta_at(0.50), delta_at(0.05)
    c = d09.index.intersection(d05.index)
    t2 = (d09.loc[c] - d05.loc[c]).values
    r2 = paired_summary(t2, np.zeros_like(t2), "C2")
    res["C2"] = {"hypothesis": "H2-benefit: dogmatic-prior benefit larger at h=0.9 than h=0.5",
                 "beta_dogmatic": beta_dog, "n_seeds": int(len(c)),
                 "theta_mean_pts": r2["mean_diff"] * 100,
                 "ci_pts": [r2["ci_lo"] * 100, r2["ci_hi"] * 100],
                 "seeds_positive": f"{r2['n_positive']}/{r2['n']}", "p": r2["p_wilcoxon"],
                 "delta_h09_pts": float(d09.mean() * 100), "delta_h05_pts": float(d05.mean() * 100)}
    pvals["C2"] = r2["p_wilcoxon"]

    r3 = paired_summary(d005.values, np.zeros_like(d005.values), "C3")
    res["C3"] = {"hypothesis": "H2-harm: dogmatic prior HURTS at h=0.05 (delta < 0)",
                 "beta_dogmatic": beta_dog, "n_seeds": int(len(d005)),
                 "theta_mean_pts": r3["mean_diff"] * 100,
                 "ci_pts": [r3["ci_lo"] * 100, r3["ci_hi"] * 100],
                 "seeds_negative": f"{r3['n'] - r3['n_positive']}/{r3['n']}", "p": r3["p_wilcoxon"]}
    pvals["C3"] = r3["p_wilcoxon"]

    hol = holm(pvals)
    for k in res:
        res[k]["holm"] = hol.get(k, {})
        # directional verdict: the pre-registered direction must hold AND Holm must reject
        want_pos = k in ("C1", "C2")
        obs = res[k]["theta_mean_pts"]
        res[k]["direction_as_predicted"] = bool(obs > 0) if want_pos else bool(obs < 0)
        res[k]["verdict"] = ("SUPPORTED" if res[k]["direction_as_predicted"]
                             and hol.get(k, {}).get("reject_null") else
                             "NOT SUPPORTED")
    return res, beta_dog


# ---------------------------------------------------------------------------- SBM summaries
def sbm_summary(runs, beta_dog):
    sbm = runs[runs.dataset == "sbm"]
    hs = sorted(sbm.target_h.dropna().unique())
    rows = []
    for m in sorted(sbm.label_per_class.unique()):
        for h in hs:
            base = sbm[(sbm.model == "M0_mlp") & (sbm.target_h == h) &
                       (sbm.label_per_class == m)].set_index("split_seed")["test_acc"]
            dog = sbm[(sbm.model == "M4_mrf_mlp") & (sbm.beta == beta_dog) &
                      (sbm.target_h == h) & (sbm.label_per_class == m)] \
                .set_index("split_seed")["test_acc"]
            tun = select_tuned(sbm[(sbm.model == "M4_mrf_mlp") & (sbm.target_h == h) &
                                   (sbm.label_per_class == m)], extra_group=["target_h"]) \
                .set_index("split_seed")["test_acc"]
            gcn = sbm[(sbm.model == "M1_gcn") & (sbm.target_h == h) &
                      (sbm.label_per_class == m)].set_index("split_seed")["test_acc"]
            og = sbm[(sbm.model == "Oracle_G") & (sbm.target_h == h) &
                     (sbm.label_per_class == m)].set_index("split_seed")["test_acc"]
            of = sbm[(sbm.model == "Oracle_F") & (sbm.target_h == h) &
                     (sbm.label_per_class == m)].set_index("split_seed")["test_acc"]
            c = base.index.intersection(dog.index)
            rd = paired_summary(dog.loc[c].values, base.loc[c].values)
            ct = base.index.intersection(tun.index)
            rt = paired_summary(tun.loc[ct].values, base.loc[ct].values)
            rows.append({"m": m, "h": h, "MLP": base.mean() * 100, "GCN": gcn.mean() * 100,
                         "Oracle-F": of.mean() * 100, "Oracle-G": og.mean() * 100,
                         "MRF dogmatic": dog.mean() * 100, "MRF tuned": tun.mean() * 100,
                         "Δ dogmatic": rd["mean_diff"] * 100,
                         "Δ dog CI": f"[{rd['ci_lo']*100:+.2f},{rd['ci_hi']*100:+.2f}]",
                         "Δ tuned": rt["mean_diff"] * 100})
    df = pd.DataFrame(rows)
    write(df.set_index(["m", "h"]), "T4_sbm_sweep", float_format="%.2f")

    hstars = {}
    for m in sorted(df.m.unique()):
        sub = df[df.m == m].sort_values("h")
        per_seed = []
        for seed in sorted(sbm.split_seed.unique()):
            ds = []
            for h in hs:
                b = sbm[(sbm.model == "M0_mlp") & (sbm.target_h == h) &
                        (sbm.label_per_class == m) & (sbm.split_seed == seed)]["test_acc"]
                d = sbm[(sbm.model == "M4_mrf_mlp") & (sbm.beta == beta_dog) &
                        (sbm.target_h == h) & (sbm.label_per_class == m) &
                        (sbm.split_seed == seed)]["test_acc"]
                ds.append(float(d.iloc[0] - b.iloc[0]) if len(b) and len(d) else np.nan)
            hx = crossover(np.array(hs), np.array(ds))
            if not np.isnan(hx):
                per_seed.append(hx)
        hstars[m] = {"median": float(np.median(per_seed)) if per_seed else None,
                     "min": float(np.min(per_seed)) if per_seed else None,
                     "max": float(np.max(per_seed)) if per_seed else None,
                     "n_seeds_with_crossover": len(per_seed)}
    return df, hstars


# --------------------------------------------------------------------------------- controls
def controls_summary(runs):
    out = {}
    # rewired null
    rw = runs[runs.dataset == "cora_rewired"]
    if len(rw):
        sel = select_tuned(rw)
        piv = sel.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                              values="test_acc")
        out["rewired"] = {}
        for m in sorted(piv.index.get_level_values(0).unique()):
            p = piv.xs(m, level="label_per_class")
            for mod in ("M2_lap", "M4_mrf_gcn"):
                if mod in p:
                    r = paired_summary(p[mod].values, p["M1_gcn"].values)
                    out["rewired"][f"m={m} {mod}"] = {
                        "delta_pts": r["mean_diff"] * 100,
                        "ci": [r["ci_lo"] * 100, r["ci_hi"] * 100],
                        "signs": f"{r['n_positive']}/{r['n']}"}
    # deeper GCN
    dp = runs[runs.model == "M1_gcn_3layer"]
    if len(dp):
        cora = select_tuned(runs[(runs.dataset == "cora")])
        out["deeper_gcn"] = {}
        for m in sorted(dp.label_per_class.unique()):
            a = dp[dp.label_per_class == m].set_index("split_seed")["test_acc"]
            b = cora[(cora.model == "M1_gcn") & (cora.label_per_class == m)] \
                .set_index("split_seed")["test_acc"]
            c = a.index.intersection(b.index)
            r = paired_summary(a.loc[c].values, b.loc[c].values)
            out["deeper_gcn"][f"m={m}"] = {"delta_vs_2layer_pts": r["mean_diff"] * 100,
                                           "ci": [r["ci_lo"] * 100, r["ci_hi"] * 100]}
    # retrain noise floor
    fl = runs[runs.model == "M1_retrain_floor"]
    if len(fl):
        fl = fl.copy()
        fl["dis"] = fl.notes.str.extract(r"disagreement=([\d.]+)").astype(float)
        out["retrain_floor"] = {f"m={int(m)}": float(g.dis.mean())
                                for m, g in fl.groupby("label_per_class")}
    # small validation
    sv = runs[runs.dataset == "cora_smallval"]
    if len(sv):
        sel = select_tuned(sv)
        piv = sel.pivot_table(index=["label_per_class", "split_seed"], columns="model",
                              values="test_acc")
        out["small_validation"] = {}
        for m in sorted(piv.index.get_level_values(0).unique()):
            p = piv.xs(m, level="label_per_class")
            r = paired_summary(p["M4_mrf_mlp"].values, p["M0_mlp"].values)
            out["small_validation"][f"m={m} MRF(MLP)-MLP"] = {
                "delta_pts": r["mean_diff"] * 100,
                "ci": [r["ci_lo"] * 100, r["ci_hi"] * 100],
                "signs": f"{r['n_positive']}/{r['n']}"}
    # MF vs Gibbs checksum
    chk = runs[runs.regime == "checksum"]
    if len(chk):
        chk = chk.copy()
        chk["tv"] = chk.notes.str.extract(r"tv_mf_gibbs_mean=([\d.]+)").astype(float)
        chk["mf_acc"] = chk.notes.str.extract(r"mf_acc=([\d.]+)").astype(float)
        gated = chk[chk.converged.astype(float) == 1]
        ungated = chk[chk.converged.astype(float) != 1]
        out["mf_vs_gibbs"] = {
            "mean_TV_marginals": float(chk.tv.mean()), "max_TV": float(chk.tv.max()),
            "mean_acc_gibbs": float(chk.test_acc.mean()),
            "mean_acc_mf": float(chk.mf_acc.mean()),
            "n_gates_passed": int(len(gated)), "n_cells": int(len(chk)),
            # cells where the Gibbs reference itself passed its diagnostics: only these
            # licence a statement about mean-field's accuracy
            "gated_mean_TV": float(gated.tv.mean()) if len(gated) else None,
            "gated_betas": sorted(gated.beta.dropna().unique().tolist()) if len(gated) else [],
            "ungated_mean_TV": float(ungated.tv.mean()) if len(ungated) else None,
            "ungated_betas": sorted(ungated.beta.dropna().unique().tolist()) if len(ungated) else [],
            "ungated_rhat_max": float(ungated.rhat_max.astype(float).max()) if len(ungated) else None,
            "ungated_ess_min": float(ungated.ess_min.astype(float).min()) if len(ungated) else None,
            "gated_acc_gap_mf_minus_gibbs": (float((gated.mf_acc - gated.test_acc).mean())
                                             if len(gated) else None)}
    return out


# ------------------------------------------------------------------------------ fidelity
def fidelity_summary(fid):
    d = fid[(fid.method != "exact") & (fid.structure != "grid_budget")]
    piv = d.pivot_table(index=["structure", "beta"], columns="method", values="tv_mean")
    write(piv, "T5_fidelity", float_format="%.2e")
    conv = d[d.method == "lbp"].groupby("beta")["converged"].mean()
    trees = d[(d.structure.isin(["chain", "tree", "star"])) & (d.method == "lbp")]
    out = {"lbp_convergence_by_beta": {float(k): float(v) for k, v in conv.items()},
           "lbp_max_tv_on_trees": float(trees.tv_mean.max()),
           "structures": sorted(d.structure.unique())}
    b = fid[fid.structure == "grid_budget"]
    if len(b):
        gb = b[b.method == "gibbs"].groupby("gibbs_kept")["tv_mean"].mean()
        out["gibbs_tv_by_budget"] = {int(k): float(v) for k, v in gb.items()}
        out["mf_bias_floor"] = float(b[b.method == "mf"].tv_mean.mean())
        out["lbp_bias_floor"] = float(b[b.method == "lbp"].tv_mean.mean())
    return out


def table_controls(runs, ctrl):
    """T6 — the alternative-explanation controls, in one place."""
    rows = []
    for k, v in ctrl.get("rewired", {}).items():
        rows.append({"control": "Degree-preserving rewiring (Cora)", "condition": k,
                     "quantity": "Δ vs GCN on rewired graph (pts)",
                     "value": f"{v['delta_pts']:+.2f}",
                     "95% CI": f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]", "seeds +": v["signs"]})
    for k, v in ctrl.get("deeper_gcn", {}).items():
        rows.append({"control": "3-layer GCN (propagation depth)", "condition": k,
                     "quantity": "Δ vs 2-layer GCN (pts)",
                     "value": f"{v['delta_vs_2layer_pts']:+.2f}",
                     "95% CI": f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]", "seeds +": ""})
    for k, v in ctrl.get("retrain_floor", {}).items():
        rows.append({"control": "Retrain-noise floor (seed only)", "condition": k,
                     "quantity": "test-prediction disagreement", "value": f"{v:.3f}",
                     "95% CI": "", "seeds +": ""})
    for k, v in ctrl.get("small_validation", {}).items():
        rows.append({"control": "Small validation (5 labels/class)", "condition": k,
                     "quantity": "Δ (pts)", "value": f"{v['delta_pts']:+.2f}",
                     "95% CI": f"[{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]", "seeds +": v["signs"]})
    mg = ctrl.get("mf_vs_gibbs")
    if mg:
        rows.append({"control": "Mean-field vs gated Gibbs (Cora)", "condition": "m=5, tuned β",
                     "quantity": "mean TV between marginals", "value": f"{mg['mean_TV_marginals']:.4f}",
                     "95% CI": f"acc MF {mg['mean_acc_mf']:.3f} vs Gibbs {mg['mean_acc_gibbs']:.3f}",
                     "seeds +": f"gates passed: {mg['n_gates_passed']}/{mg['n_cells']}"})
    df = pd.DataFrame(rows)
    if len(df):
        write(df.set_index(["control", "condition"]), "T6_controls", float_format="%s")
    return df


def table_beta_ext(runs):
    """T7 — exploratory: does the pre-registered beta grid truncate the optimum?"""
    ext = runs[runs.dataset.isin(["cora_betaext", "sbm_betaext"])]
    if not len(ext):
        return None
    rows = []
    for ds, base_ds in (("cora_betaext", "cora"), ("sbm_betaext", "sbm")):
        e = ext[ext.dataset == ds]
        if not len(e):
            continue
        for m in sorted(e.label_per_class.dropna().unique()):
            for mod in ("M4_mrf_mlp", "M4_mrf_gcn"):
                sub = e[(e.model == mod) & (e.label_per_class == m)]
                if not len(sub):
                    continue
                base = runs[(runs.dataset == base_ds) & (runs.model == mod) &
                            (runs.label_per_class == m)]
                in_grid = base.groupby("beta")["test_acc"].mean()
                out_grid = sub.groupby("beta")["test_acc"].mean()
                both = pd.concat([in_grid, out_grid])
                rows.append({"dataset": ds.replace("_betaext", ""), "m": int(m),
                             "model": MODEL_LABEL.get(mod, mod),
                             "best β in pre-registered grid": f"{in_grid.idxmax():g}",
                             "acc there (%)": round(in_grid.max() * 100, 2),
                             "best β overall": f"{both.idxmax():g}",
                             "acc there (%) ": round(both.max() * 100, 2),
                             "gain from extension (pts)": round((both.max() - in_grid.max()) * 100, 2)})
    df = pd.DataFrame(rows)
    if len(df):
        write(df.set_index(["dataset", "m", "model"]), "T7_beta_extension", float_format="%s")
    return df


if __name__ == "__main__":
    runs = load_runs()
    fid = load_fidelity()
    print(f"Loaded {len(runs)} runs, {len(fid)} fidelity rows\n")

    print("Tables:")
    table_datasets(runs)
    table_main(runs)
    table_mechanism(runs, None)

    print("\nConfirmatory endpoints:")
    conf, beta_dog = confirmatory(runs)
    for k, v in conf.items():
        print(f"  {k}: {v['verdict']}  theta={v['theta_mean_pts']:+.2f} pts "
              f"CI[{v['ci_pts'][0]:+.2f},{v['ci_pts'][1]:+.2f}] p={v['p']:.4g} "
              f"holm_reject={v['holm'].get('reject_null')}")
        print(f"      {v['hypothesis']}")

    print(f"\nSBM (dogmatic beta = {beta_dog}):")
    sbm_df, hstars = sbm_summary(runs, beta_dog)
    print(f"  crossover h*: {hstars}")

    ctrl = controls_summary(runs)
    fidsum = fidelity_summary(fid)
    table_controls(runs, ctrl)
    table_beta_ext(runs)

    payload = {"confirmatory": conf, "beta_dogmatic": beta_dog, "crossover": hstars,
               "controls": ctrl, "fidelity": fidsum,
               "n_runs": int(len(runs)), "n_fidelity_rows": int(len(fid))}
    with open(PROC / "analysis.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote results/processed/analysis.json")

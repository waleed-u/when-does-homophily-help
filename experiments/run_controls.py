"""Tier-2 controls — the experiments that decide whether the headline claims survive scrutiny.

Each answers an alternative explanation a sceptical reader would raise:

  rewired   Does the prior exploit the graph's *label structure*, or would any smoothing over
            any graph of the same degree do? Degree-preserving rewiring destroys the label
            structure while preserving the degree sequence: benefit must vanish.
  deeper    Is the prior just "more propagation"? A 3-layer GCN propagates further at zero
            probabilistic cost; if it matches the MRF, the MRF earns nothing.
  floor     H3 asks whether training-time and inference-time imposition differ. Differences
            are only meaningful above the disagreement between two *identically specified*
            GCNs that differ solely by initialisation seed.
  simcheck  M5's precondition: is feature cosine similarity actually informative about whether
            an edge joins same-class nodes? Reported whether or not M5 runs.
  checksum  Is mean-field trustworthy at scale? Compare converged MF against diagnostics-gated
            Gibbs on identical logits — MF's bias is invisible unless something better is run.
  smallval  Conclusions are tuned on a ~500-label validation set while the training budget is
            7-140 labels. Re-select with 5 labels/class and see whether the sign survives.

Usage: python -m experiments.run_controls --control rewired|deeper|floor|simcheck|checksum|smallval
"""
import argparse

import numpy as np
import torch

from experiments.common import BASE, run_posthoc, run_trained
from src.config import BETA_GRID, LAMBDA_GRID, SIGMA_X, STAGE1
from src.data import load_planetoid
from src.infer import posterior
from src.metrics import all_metrics
from src.mrf.potts import edge_similarity_report
from src.runlog import append_run, new_run_id
from src.splits import small_val_mask, train_mask_real
from src.synthetic.sbm import make_csbm
from src.splits import masks_sbm

SEEDS = list(range(10))


def rewire_degree_preserving(edge_index: torch.Tensor, seed: int) -> torch.Tensor:
    """Configuration-model shuffle: keep every node's degree, destroy which pairs are joined."""
    e = edge_index[:, edge_index[0] < edge_index[1]]
    stubs = torch.cat([e[0], e[1]])
    g = torch.Generator().manual_seed(seed)
    stubs = stubs[torch.randperm(len(stubs), generator=g)]
    half = len(stubs) // 2
    a, b = stubs[:half], stubs[half:2 * half]
    keep = a != b                                    # drop self-loops from the shuffle
    a, b = a[keep], b[keep]
    ei = torch.stack([a, b])
    return torch.cat([ei, ei.flip(0)], dim=1)


def ctrl_rewired(ms=(2, 5)):
    data = load_planetoid("cora")
    hp = STAGE1["cora"]
    for m in ms:
        for seed in SEEDS:
            masks = {"train": train_mask_real(data, m, seed), "val": data.val_mask,
                     "test": data.test_mask}
            import copy
            rw = copy.copy(data)
            rw.edge_index = rewire_degree_preserving(data.edge_index, seed)
            from src.homophily import edge_homophily
            note = f"rewired_h={edge_homophily(rw.edge_index, rw.y):.4f}"
            kw = dict(data=rw, masks=masks, seed=seed, m=m, dataset="cora_rewired")
            lg, _ = run_trained("gcn", model_label="M1_gcn", hp=hp["gcn"],
                                extra={"notes": note}, **kw)
            for lam in LAMBDA_GRID:
                run_trained("gcn", model_label="M2_lap", lam=lam, reg="laplacian",
                            hp=hp["gcn"], save=False, extra={"notes": note}, **kw)
            for beta in BETA_GRID:
                run_posthoc(lg, model_label="M4_mrf_gcn", beta=beta, extra={"notes": note}, **kw)
        print(f"[rewired] m={m} done", flush=True)


def ctrl_deeper(ms=(1, 2, 5)):
    data = load_planetoid("cora")
    hp = STAGE1["cora"]
    for m in ms:
        for seed in SEEDS:
            masks = {"train": train_mask_real(data, m, seed), "val": data.val_mask,
                     "test": data.test_mask}
            run_trained("gcn", data=data, masks=masks, seed=seed, m=m, dataset="cora",
                        model_label="M1_gcn_3layer", layers=3, hp=hp["gcn"], save=False,
                        extra={"notes": "propagation-depth control"})
        print(f"[deeper] m={m} done", flush=True)


def ctrl_floor(ms=(2, 5, 20)):
    """Retrain-noise floor: prediction disagreement between GCNs differing only by seed."""
    data = load_planetoid("cora")
    hp = STAGE1["cora"]
    rows = []
    for m in ms:
        for seed in SEEDS:
            masks = {"train": train_mask_real(data, m, seed), "val": data.val_mask,
                     "test": data.test_mask}
            preds = []
            for rep in range(2):
                from src.train import train_model
                lg, _ = train_model("gcn", data, masks["train"], masks["val"],
                                    seed=seed + 1000 * (rep + 1), lr=hp["gcn"]["lr"],
                                    weight_decay=hp["gcn"]["weight_decay"])
                preds.append(lg[data.test_mask].argmax(1))
            dis = float((preds[0] != preds[1]).double().mean())
            rows.append(dis)
            append_run(dict(run_id=new_run_id("FLOOR"), dataset="cora", model="M1_retrain_floor",
                            regime="control", label_per_class=m, split_seed=seed,
                            notes=f"disagreement={dis:.4f}", test_acc=""))
        print(f"[floor] m={m}: mean retrain disagreement={np.mean(rows[-len(SEEDS):]):.4f}",
              flush=True)


def ctrl_simcheck():
    print("M5 pre-check — is cosine similarity informative about same-class edges?\n")
    rows = []
    d = load_planetoid("cora")
    rows.append(("cora", edge_similarity_report(d.x, d.edge_index, d.y)))
    for h in (0.05, 0.50, 0.90):
        s = make_csbm(h, seed=0, sigma_x=SIGMA_X)
        rows.append((f"sbm_h{h}", edge_similarity_report(s.x, s.edge_index, s.y)))
    print(f"{'dataset':>10} {'sim same':>10} {'sim cross':>10} {'separation':>11} "
          f"{'n_same':>8} {'n_cross':>8}")
    for name, r in rows:
        print(f"{name:>10} {r['sim_same_mean']:>10.4f} {r['sim_cross_mean']:>10.4f} "
              f"{r['separation']:>11.4f} {r['n_same']:>8} {r['n_cross']:>8}")
        append_run(dict(run_id=new_run_id("SIM"), dataset=name, model="M5_precheck",
                        regime="control",
                        notes=f"same={r['sim_same_mean']:.4f} cross={r['sim_cross_mean']:.4f} "
                              f"sep={r['separation']:.4f}"))


def ctrl_checksum():
    """Converged mean-field vs diagnostics-gated Gibbs on identical unaries."""
    from src.analysis import load_runs, select_tuned
    from src.train import load_logits
    sel = select_tuned(load_runs("results/raw/runs_cora_m*.csv"))
    sel = sel[(sel.model == "M4_mrf_gcn") & (sel.label_per_class == 5)]
    data = load_planetoid("cora")
    src_runs = load_runs("results/raw/runs_cora_m*.csv")
    print(f"{'seed':>5} {'beta':>6} {'TV(MF,Gibbs) mean':>18} {'max':>8} "
          f"{'acc MF':>8} {'acc Gibbs':>10} {'rhat':>7} {'ess':>8} {'gate':>5}")
    for _, r in sel.iterrows():
        seed = int(r.split_seed)
        gcn_run = src_runs[(src_runs.model == "M1_gcn") & (src_runs.split_seed == seed)
                           & (src_runs.label_per_class == 5) & (src_runs.dataset == "cora")]
        logits = load_logits(gcn_run.iloc[0].logits_path)
        masks = {"train": train_mask_real(data, 5, seed), "val": data.val_mask,
                 "test": data.test_mask}
        qm, _ = posterior(logits, data.edge_index, float(r.beta), engine="mf")
        qg, ig = posterior(logits, data.edge_index, float(r.beta), engine="gibbs",
                           chains=4, burn_in=1000, kept=2000, seed=seed)
        tv = 0.5 * (qm - qg).abs().sum(1)
        am = all_metrics(qm, data.y, {"test": masks["test"]}, data.edge_index)["test_acc"]
        ag = all_metrics(qg, data.y, {"test": masks["test"]}, data.edge_index)["test_acc"]
        print(f"{seed:>5} {float(r.beta):>6.2f} {float(tv.mean()):>18.5f} "
              f"{float(tv.max()):>8.4f} {am:>8.4f} {ag:>10.4f} {ig['rhat_max']:>7.4f} "
              f"{ig['ess_min']:>8.0f} {int(ig['converged']):>5}")
        append_run(dict(run_id=new_run_id("CHK"), dataset="cora", model="M4_mrf_gcn",
                        regime="checksum", inference_method="gibbs", label_per_class=5,
                        split_seed=seed, beta=float(r.beta), test_acc=round(ag, 6),
                        converged=int(ig["converged"]), rhat_max=round(ig["rhat_max"], 5),
                        ess_min=round(ig["ess_min"], 1),
                        notes=f"tv_mf_gibbs_mean={float(tv.mean()):.5f} "
                              f"max={float(tv.max()):.4f} mf_acc={am:.4f}"))


def ctrl_smallval(ms=(1, 2)):
    """Re-select hyperparameters with a 5-labels/class validation set."""
    data = load_planetoid("cora")
    hp = STAGE1["cora"]
    for m in ms:
        for seed in SEEDS:
            tm = train_mask_real(data, m, seed)
            sv = small_val_mask(data, tm, 5, seed)
            masks = {"train": tm, "val": sv, "test": data.test_mask}
            kw = dict(data=data, masks=masks, seed=seed, m=m, dataset="cora_smallval")
            lm, _ = run_trained("mlp", model_label="M0_mlp", hp=hp["mlp"],
                                extra={"notes": "5 labels/class validation"}, **kw)
            lg, _ = run_trained("gcn", model_label="M1_gcn", hp=hp["gcn"],
                                extra={"notes": "5 labels/class validation"}, **kw)
            for beta in BETA_GRID:
                run_posthoc(lm, model_label="M4_mrf_mlp", beta=beta,
                            extra={"notes": "5 labels/class validation"}, **kw)
                run_posthoc(lg, model_label="M4_mrf_gcn", beta=beta,
                            extra={"notes": "5 labels/class validation"}, **kw)
        print(f"[smallval] m={m} done", flush=True)


CONTROLS = {"rewired": ctrl_rewired, "deeper": ctrl_deeper, "floor": ctrl_floor,
            "simcheck": ctrl_simcheck, "checksum": ctrl_checksum, "smallval": ctrl_smallval}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, choices=list(CONTROLS))
    CONTROLS[ap.parse_args().control]()

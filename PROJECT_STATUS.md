# PROJECT_STATUS.md

**Last updated:** start of implementation phase (after workspace audit).
**Rule:** this file records what has *actually* been built and run. Nothing here is aspirational; planned work lives in `FINAL_RESEARCH_DESIGN.md`.

## 1. Original research goal

From `CMPT727_ProjectProposal.pdf`: implement a GCN baseline for node classification on the Cora citation network, encode the homophily assumption ("connected papers share a topic") as a differentiable Laplacian "logic" regularizer L_logic = (1/|E|)Σ‖p_i − p_j‖² added to cross-entropy with weight λ, and test whether this symbolic prior improves accuracy in low-label regimes (1–5% labelled). Framed as MAP estimation and as a first step toward neuro-symbolic learning.

## 2. Current refined research question

> When does an explicit homophily prior improve semi-supervised node classification — as a function of label scarcity and of how well the prior matches the graph — and does it matter whether the prior is imposed at training time (as a regularizer) or at inference time (via an explicit MRF)?

Supporting (validation, not headline): is the approximate inference faithful enough that conclusions are about the *model* rather than the *inference*?

The proposal's question is nested inside this one (its λ-sweep on Cora is experiment E1).

## 3. Current hypotheses

Frozen in `PROTOCOL.md` §B. Confirmatory (the only hypothesis tests): **C1** H1 scarcity, **C2** H2-benefit, **C3** H2-harm — all on the clean MLP-unary cell, 30 seeds, Holm-corrected Wilcoxon. Estimation-only: H2-crossover h*, H3 mechanism, V4 inference validity, H5 similarity weighting, probability-quality reporting.

## 4. Datasets currently available

| Dataset | Status | Verified statistics (recomputed from loader, not copied) |
|---|---|---|
| Cora | **downloaded**, `data/Cora` | 2,708 nodes · 5,278 undirected edges · 1,433 features · 7 classes · mean degree 3.898 · edge homophily **0.810** · adjusted homophily 0.771 · label informativeness 0.590 |
| CiteSeer | **downloaded**, `data/CiteSeer` | 3,327 nodes · 4,552 undirected edges · 3,703 features · 6 classes · mean degree 2.736 · edge homophily **0.736** · adjusted homophily 0.671 · label informativeness 0.451 |

Both match the playbook's expected values, confirming the loader path is correct.

## 5. Datasets still required

- **Contextual SBM** (generated, not downloaded): generator not yet implemented; σ_x still to be frozen by the quarantined pilot (PROTOCOL §C.3, CHANGELOG addendum M1).
- **Fidelity-suite small graphs** (generated): builders not yet implemented.
- Roman-empire: **cut to future work** by the approved design (not required).

## 6. Methods already implemented

| Module | Status |
|---|---|
| `src/data.py` | Planetoid loading (certifi SSL fix), row normalization, undirected-edge helper — **working** |
| `src/splits.py` | stratified train masks, nesting across m, SBM masks, small-validation masks, disjointness assert — **working, invariants verified** |
| `src/homophily.py` | edge/adjusted homophily, label informativeness, mean degree — **working, values verified** |
| `src/metrics.py` | accuracy, macro-F1, NLL, Brier, ECE, rule satisfaction R, argmax edge-agreement — **written, not yet exercised** |
| `src/inference/coloring.py` | greedy graph coloring for MF blocks / chromatic Gibbs — **written, not yet exercised** |

Infrastructure: venv with torch 2.13.0 (CPU) + PyG 2.8.0; git repo with `PROTOCOL.md` committed and tagged `protocol-freeze` (the pre-registration boundary).

Day-1 microbenchmark (measured): GCN epoch on Cora-shaped data **3.7 ms** (≈0.9 s per 250-epoch run); mean-field sweep **0.22 ms** (≈22 ms per 100 sweeps). Compute is not a constraint.

## 7. Methods still missing

Models (MLP, GCN), regularizer losses (Laplacian, rule-violation), training loop with early stopping and logit saving, run logging (`runs.csv`), exact inference (enumeration + variable elimination), mean-field, loopy BP, chromatic Gibbs + MCMC diagnostics, CSBM generator + oracles, small-graph builders, label propagation baseline, similarity weights, all analysis/figure/table scripts.

## 8. Experiments already completed

**None.** No experimental result exists yet; `results/runs.csv` and `results/fidelity.csv` do not exist. The only executed computations are the dataset-statistics verification and the timing microbenchmark reported above.

## 9. Experiments still required

E0 (sanity gate), E1 (proposal reproduction), E2 (Cora low-label factorial), E3 (strength sensitivity), E4 (mechanism ladder), E5 (CSBM sweep, tuned + dogmatic), E6 (CiteSeer replication), E7 (fidelity suite), E8r (Gibbs checksum), E13 (small-validation ablation), plus controls: rewired-graph null, deeper-GCN row, M3 row, clamped variant, M5 pre-check and variant. Specified in `FINAL_RESEARCH_DESIGN.md`.

## 10. Existing results

None (see §8). Verified dataset statistics in §4 are descriptive facts about the data, not experimental results.

## 11. Existing figures/tables

None.

## 12. Known problems

1. **SSL certificate failure** on first Planetoid download (macOS framework Python lacks a cert bundle). Fixed in `src/data.py` by pointing `SSL_CERT_FILE` at `certifi`.
2. `torch.use_deterministic_algorithms(True)` interacts with scatter-based PyG ops; must be validated once the GCN exists (fallback: `warn_only=True`, documented if used).
3. Cora at m=20 needs ≥20 eligible nodes/class outside val/test — holds, but `splits.py` raises explicitly if violated (guards future datasets).

## 13. Reproducibility issues

None outstanding. In place: frozen tagged protocol; deterministic seeding; pinned `requirements.txt`; append-only `runs.csv`/`fidelity.csv` as the only sources for figures/tables; `results/logits/` per-run artifacts; `--final` test-evaluation audit log (to be implemented with `evaluate.py`).

## 14. Recommended next steps

1. Models + losses + training loop + run logging; gate `λ=0 ≡ GCN` and leakage tests. → E0, E1.
2. Exact kit + mean-field with ELBO-monotonicity assertion; gates `β=0 ≡ softmax`, `enumeration ≡ VE`.
3. Loopy BP + chromatic Gibbs; gate `BP ≡ exact on trees`; run the fidelity suite (E7).
4. Post-hoc inference pipeline + LP baseline; then the Cora block (E2/E3/E4).
5. CSBM generator + σ_x pilot → freeze → SBM grid (E5), CiteSeer (E6), Tier-2 controls.
6. Analysis (paired CIs, C1–C3), figures/tables, audit, paper material.

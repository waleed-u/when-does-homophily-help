# PROTOCOL.md — Frozen Scientific Contract

**Project:** When Does a Homophily Prior Help? Regularization versus Probabilistic Inference in Low-Label Node Classification
**Course:** CMPT 727 Statistical Machine Learning, SFU Spring 2026 (Waleed Ahmed, 301663286)
**Status:** FROZEN at git tag `protocol-freeze`. Amendments only via CHANGELOG.md (§M). Results affected by any post-freeze change are demoted to exploratory.

This file is the pre-registration for the study. Everything in it was fixed **before** any main experimental run. Exploratory work permitted before freeze: the E0 sanity gate and the quarantined SBM σ_x pilot (§C.3).

---

## A. Research question

**Primary.** When does an explicit homophily prior improve semi-supervised node classification — as a function of label scarcity and of how well the prior matches the graph — and does it matter whether the prior is imposed at training time (as a regularizer) or at inference time (via an explicit MRF)?

**Supporting (validation, not headline).** Is the approximate inference faithful enough that conclusions are about the model rather than the inference? (Exact-marginal fidelity suite + audited Gibbs checksum at scale.)

Secondary: RQ-A (how much of the prior does GCN message passing capture implicitly? — 2×2 unary factorial), RQ-B (does validation-tuned β track the generator-implied coupling β_gen = log(p_in/p_out) on the CSBM?), RQ-C (probability quality: NLL/Brier), RQ-D (conditional: similarity weighting).

## B. Hypotheses (direction, condition, status)

| ID | Statement | Condition | Status |
|---|---|---|---|
| H1 | Prior benefit is largest when labels are scarcest: θ = [Acc(M4-MLP)−Acc(M0)](m=2) − [same](m=20) > 0 | Cora | **Confirmatory (C1)** |
| H2-benefit | Dogmatic-regime benefit grows with homophily: Δ(h=0.9) − Δ(h=0.5) > 0, Δ = Acc(M4-MLP)−Acc(M0) | CSBM, m=5, dogmatic β | **Confirmatory (C2)** |
| H2-harm | Dogmatic-regime prior harms under misspecification: Δ(h=0.05) < 0 | CSBM, m=5, dogmatic β | **Confirmatory (C3)** |
| H2-crossover | A crossover h* ∈ (0.05, 0.9) exists (dogmatic regime) | CSBM | Estimation only (median [min,max] over seeds; interpolation of Δ(h) zero-crossing; never tested) |
| H3 | M2 (training-time) and M4 (inference-time) are correlated but not identical; differences exceed the M1-retrain-noise floor and are structured (degree/boundary) | Cora m∈{2,5,20}; SBM h∈{0.3,0.9} m=5 | Estimation only (explicitly no significance test) |
| V4 | MF/LBP/Gibbs match exact marginals at small–moderate β; error grows with coupling/cyclicity; Gibbs error is MC variance, MF/LBP error is bias | fidelity suite | Validation criterion + descriptive appendix (gates in §H) |
| H5 | Similarity weighting reduces dogmatic-regime harm at h=0.05, ≈neutral at h=0.9 | CSBM + Cora; gated by pre-check §I.5 | Exploratory (SBM result labeled best-case by construction) |
| PQ | Probability-quality reporting: NLL/Brier columns everywhere; "accuracy up, NLL worse at large β" is a pre-registered reportable outcome; calibration claims only on gated-Gibbs marginals | all | Reporting obligation |

Tuned-regime companion expectation (not a hypothesis): with the baseline nested, tuned-regime Δ ⪆ 0 everywhere; deviations measure selection (winner's-curse) noise. The harm claim attaches **only** to the dogmatic regime.

Cut to future work (pre-registered): EM-learned compatibility (M6), Roman-empire, joint/unrolled training, LBP at real-data scale.

## C. Datasets

**C.1 Cora (primary), CiteSeer (replication).** PyG `Planetoid` loader; all reported statistics recomputed from the loaded objects. Transductive setting: val/test features and edges are visible during training (standard; disclosed). Preprocessing: row-normalize features only. CiteSeer runs m ∈ {1,5,20} only.

**C.2 Contextual SBM (centerpiece).** n=2800, C=7 equal classes, expected degree k=8 via p_in = h·k/(n_c−1), p_out = (1−h)·k/(n−n_c); h ∈ {0.05, 0.15, 0.30, 0.50, 0.70, 0.90}; features x_i = μ_{y_i} + σ_x·ε, ε∼N(0,I_d), d=32, μ_c = orthonormal-ish class means (random rotation of scaled identity block, generated once with seed 7, frozen). New graph per experiment seed. Per-graph acceptance checks (logged): empirical mean degree within ±0.5 of 8; empirical edge homophily within ±0.02 of target; class balance exact by construction. Results are plotted against realized h; adjusted homophily and label informativeness logged per graph.

**C.3 σ_x pilot (the only pre-freeze tuning).** Pilot on quarantined seeds {100–109}, excluded from all reported results. Acceptance criterion (fixed in advance): closed-form feature-only Bayes oracle accuracy in **[70%, 80%]** at the chosen σ_x. σ_x is frozen by addendum §M1 after the pilot and never retuned per condition. σ_x is a designer-chosen effect-size dial: heatmap magnitudes and h* are generator-conditional (stated in the paper). One coarse second σ_x level is run at h ∈ {0.05, 0.5, 0.9}, m=5 as an appendix robustness check.

**C.4 Fidelity-suite graphs.** chain n=20, balanced tree n=15, star n=16, single cycle n=16, 4×4 grid, dense 2-block SBM n=14 (p_in=0.8, p_out=0.3). C=3 everywhere, plus one C=7 grid case via VE. Unaries s_i = α·e_{y_i} + N(0, I_C) with α ∈ {0.5, 1.5} and 25% of nodes set to α=0 (uninformative); α scale calibrated once to the Cora GCN logit-margin distribution at m=5 (calibration factor recorded in §M addendum), then frozen. 20 draws per (structure, β, α) cell. β grid: {0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.3, 1.75, 2.5, 4.0}. Critical-coupling reference markers per q: β_c(3) = ln(1+√3) ≈ 1.005; β_c(7) = ln(1+√7) ≈ 1.299 (qualitative markers only).

## D. Splits and seeds

- Experiment seeds: **{0..9}** everywhere; **{0..29}** on confirmatory cells only (C1: Cora m∈{2,20}; C2/C3: CSBM h∈{0.05,0.5,0.9}, m=5, dogmatic).
- Quarantined pilot seeds: {100–109}. Fidelity-suite draw seeds: {200–219}.
- Seed s bundles: train-label sample (stratified, **nested across m within seed**: the m-labels/class sample is a superset of the (m−1)-level sample), backbone init stream, and (SBM) the graph draw.
- Real data: validation = 500 nodes, test = 1000 nodes, fixed across seeds (Planetoid-standard); only train sample + init vary per seed. SBM: per-graph val 500 / test 1000, sampled disjointly after train.
- Split generation pseudocode: stratified sample m nodes/class for train from nodes outside val∪test; deterministic given (dataset, seed, m). Same split shared by every model within (seed, m).
- Pre-committed forks: disagreement-curve seed = **0**; SBM Stage-2 selection uses each seed's own graph/validation; VAC analyses use seed 0.

## E. Models

- **M0** MLP: 2×(Linear-ReLU-Dropout0.5)→Linear, hidden 64. **M1** GCN: 2-layer, hidden 64, dropout 0.5 (Kipf & Welling). Softmax outputs.
- **M2** M1 + λ·L_lap, L_lap = (1/|E|)·Σ_(i,j)∈E ‖p_i − p_j‖²  (undirected edges counted once).
- **M3** M1 + λ·L_rule, L_rule = (1/|E|)·Σ_(i,j)∈E (1 − p_i·p_j). Cora only; appendix row.
- **M4** Potts MRF: p(y|X,G) ∝ exp(Σ_i s_i(y_i) + β·Σ_(i,j)∈E w_ij·1[y_i=y_j]), w_ij = 1; unaries s = frozen logits of M1 (**M4-GCN**) or M0 (**M4-MLP**); post-hoc convergent mean-field inference (§H). All MRF variants are pure functions of saved logits (`infer.py` never trains).
- **M5** M4 with w_ij = clip(cosine(x_i, x_j), 0, 1) (g frozen; no tuning of g). Gated by pre-check §I.5.
- **LP** label propagation / harmonic functions (Zhu-Ghahramani-Lafferty 2003), labels+graph only, closed-form; uses train labels as evidence (noted in its table row).
- **Oracle-F** (SBM only): closed-form Gaussian Bayes classifier on features with known generative parameters. **Oracle-G** (SBM only): Potts(β_gen = log(p_in/p_out)) on true Gaussian log-likelihood unaries, MF inference — approximate attainable ceiling.
- **Clamped variant** of M4 (appendix only): train labels clamped as one-hot evidence; excluded from all endpoints and Δ-vs-baseline figures.

## F. Hyperparameters, tuning budget, selection policy

- **Stage 1** (backbone; per dataset, per architecture — M1 and M0 tuned separately): lr ∈ {0.01, 0.005} × weight decay ∈ {5e-4, 1e-3}, selected on validation accuracy at m=5, 10 seeds, then frozen for all models of that architecture on that dataset. Hidden=64, dropout=0.5 fixed.
- **Stage 2** (prior knob; per model, dataset, m; equal cap 6 configs; validation only): λ ∈ {0.001, 0.01, 0.1, 0.5, 1, 5}; β ∈ {0.05, 0.1, 0.25, 0.5, 1, 2}. Grids exclude 0 (the baseline is the separate M0/M1 row). Selection: max validation accuracy; ties → lower validation NLL.
- **Dogmatic regime (SBM)**: β^dog (and λ^dog) = median over seeds {0..9} of the tuned value at (h=0.9, m=5) for the same model; then fixed across all h. β_gen = log(p_in/p_out) is additionally reported per h as the generator-implied reference.
- **Early stopping**: max 500 epochs, patience 50 on validation accuracy, restore best-validation checkpoint. Identical rule for every trained model. Adam optimizer.
- Winner's-curse asymmetry disclosure: baselines have no Stage-2 knob; full-grid test metrics are reported in the appendix.

## G. Training / infrastructure invariants

CPU float32 training with `torch.use_deterministic_algorithms(True)`; float64 for all exact-inference computations. Every training run saves logits to `results/logits/{run_id}.npz`. Test metrics computed exactly once per final configuration behind `evaluate.py --final`, which appends to `results/final_eval_audit.log`. Two results files: `results/runs.csv` (task runs) and `results/fidelity.csv` (exactness suite). All figures/tables generated by scripts reading only these CSVs.

## H. Inference engines (single reconciled protocol)

- **Mean-field (primary engine everywhere):** coordinate ascent q_i(c) ∝ exp(s_i(c) + β·Σ_{j∈N(i)} w_ij·q_j(c)), executed color-block-wise (greedy graph coloring; block updates are exact coordinate ascent ⇒ ELBO monotone — asserted every block at tol 1e-9). Init q_i = softmax(s_i). Converged when max_i TV(q_i^{t+1}, q_i^t) < 1e-6; cap 500 sweeps. **Always run to convergence; the iteration count is never a tuned hyperparameter.**
- **Loopy BP (fidelity suite only):** damped synchronous sum-product; Potts O(C) messages m_{i→j}(c) ∝ 1 + (e^{β·w_ij}−1)·b_i^{\j}(c) with normalized cavity beliefs; log-domain accumulation. Damping γ=0.5 (retry once with γ=0.9 if residual plateaus 100 iters). Converged: max directed-edge message Δ_∞ < 1e-6 for 3 consecutive iters; cap 500. Non-convergence: flag, report final-50-iteration belief average, plot fraction-converged vs β; never drop.
- **Gibbs (fidelity suite + checksum cells only):** chromatic blocked sampling (greedy coloring; Gonzalez et al. 2011); 4 chains (1 init from softmax(s), 3 random); burn-in 1000 sweeps, kept 2000; **Rao-Blackwellized marginals** (average full-conditional probability vectors). Reduced budget (kept 100) additionally in the fidelity suite for the variance panel. Gates: rank-normalized split R-hat < 1.01 and ESS > 400 on the energy trace E(y) = Σ_i s_i(y_i) + β·Σ w_ij·1[y_i=y_j]; max inter-chain marginal TV < 0.02. Failing cells flagged/hatched; never silently included or extended.
- **Exact:** brute-force enumeration (float64) when C^n ≤ 5×10^6; repeated min-fill variable elimination otherwise. Validation chain (pytest gates, must pass before any real-data run): enumeration ≡ VE on every enumerable graph (TV < 1e-10); BP ≡ exact on trees (TV < 1e-8); β=0 ≡ softmax(unaries) for every engine (TV < 1e-12).
- **Checksum cells (E8r):** Cora m=5 (tuned β), SBM (h=0.9, m=5), SBM (h=0.05, m=5, dogmatic): gated Gibbs vs converged MF on identical logits; marginal TV, Δacc, ΔNLL. Demotion trigger: >30 s per 1500-sweep Cora chain after vectorization ⇒ checksum restricted to SBM cells.
- **Clamping semantics (identical across engines):** MF: q_i frozen one-hot; Gibbs: clamped nodes never resampled; LBP: +30 added to the observed-class logit.

## I. Experiments (manipulated / fixed / outputs)

1. **E0 sanity gate** (Cora m=20, M0/M1, seeds 0–2): loss decreases; GCN val ≈ 75–82%, MLP ≈ 55–60%; leakage tests pass. Nothing runs until clean.
2. **E1 proposal reproduction** (Cora m=20, M1 vs M2 over λ grid, 10 seeds). Gate: λ=0 run ≡ M1 (max |logit diff| < 1e-5).
3. **E2 Cora low-label factorial** (m ∈ {1,2,5,10,20}; M0, M1, LP, M2, M4-GCN, M4-MLP; 10 seeds; 30 on C1 cells). Headroom-normalized Δ (relative error reduction) reported alongside raw. → F2, T2.
4. **E5 CSBM sweep** (h-grid × m ∈ {1,5,20}; models + oracles; tuned AND dogmatic regimes; 10 seeds; 30 on C2/C3 cells). β-vs-β_gen curve; fraction-of-attainable vs Oracle-G. → F3.
5. **E7 fidelity suite** (§C.4 grids; MF/LBP/Gibbs vs exact; TV mean/max; Gibbs two budgets). Appendix-first figure F-A; promotable at D14 layout freeze only.
6. **E4 mechanism** (Cora m ∈ {2,5,20}; SBM h ∈ {0.3,0.9} m=5; shared checkpoints; M1-retrain-noise disagreement floor from 3 extra M1 runs per cell; disagreement location by degree strata and boundary nodes; flip-rate = % test nodes whose argmax changes between β=0 and tuned β). → T3.
7. **E3 strength sensitivity** (free from E1/E2 grids: acc, NLL, R, argmax edge-agreement vs λ/β). → F4.
8. **E6 CiteSeer replication** (m ∈ {1,5,20}; M0, M1, LP, M2, M4-GCN; +M4-MLP at m=5).
9. **Tier-2 controls:** E8r checksum; E13 small-validation ablation (Stage-2 re-run with 5-labels/class validation at Cora m∈{1,2} and SBM h∈{0.3,0.9} m=1); rewired-graph null (degree-preserving configuration-model Cora, M2/M4 at m=5 — benefit expected to vanish); deeper-GCN row (3-layer at Cora m∈{1,2}); M3 row; clamped table; M5 (§I.5); ECE (15 equal-mass bins) + reliability diagrams (appendix).
10. **I.5 M5 pre-check (gate):** distributions of cosine similarity on same- vs cross-class edges (Cora + SBM h∈{0.05,0.9}). M5 runs only if visibly separated (report the distributions regardless); feature-permutation control (within-class shuffle ⇒ M5 must collapse to M4).

## J. Metrics

Test/val accuracy; macro-F1; NLL; Brier (mean squared error vs one-hot); rule satisfaction R = (1/|E|)Σ p_i·p_j; argmax edge-agreement (fraction of edges with equal predicted labels); ECE (15 equal-mass bins; appendix); edge homophily, adjusted homophily, label informativeness (per dataset/graph); inference columns (engine, converged, iterations, residual, ESS min/median, R-hat max, infer_seconds); realized SBM stats; fidelity: per-node TV to exact (mean primary, max secondary).

## K. Statistical analysis plan (verbatim)

1. **Pairing:** per-seed paired differences d_s between models sharing (seed, split, checkpoint lineage). Never compare unpaired means.
2. **Reporting:** mean ± sd of absolute scores; for every claimed comparison: mean(d), 95% BCa bootstrap CI (B=10,000; percentile fallback if BCa degenerate, stated), sign count k/n, per-seed dot plot for headline claims. Caption every real-data CI as "variability over paired train-split/init seeds on a fixed test set"; SBM CIs as "over graph/split seeds."
3. **Confirmatory family (the only hypothesis tests):** C1, C2, C3 (§B), exact two-sided Wilcoxon signed-rank on 30 paired seeds, Holm-corrected at α=0.05; paired t as sensitivity check. Everything else is exploratory: estimates + CIs, no stars, no "significant."
4. **Trends:** endpoint contrasts only (no per-cell testing). h* estimated per seed by linear interpolation of Δ(h) crossing zero (dogmatic arm), reported median [min, max].
5. **Failed runs:** NaN → rerun once same-seed with logged note, else cell reported missing; no seed replacement. Claims with <8/10 consistent signs are flagged in-text.
6. **Exploratory/confirmatory boundary:** this file's git tag. Pre-freeze: E0 + σ_x pilot only.

## L. Leakage checklist (verified per run by tests/test_leakage.py where automatable)

(1) test labels never touch loss/stopping/selection/edge-weights/clamping; (2) val labels never in training loss, never clamped; (3) transductive disclosure in paper; (4) similarity weights from features only, g frozen; (5) regularizers sum over all edges incl. unlabeled endpoints — identical edge set for all prior models (disclosed as the semi-supervised mechanism); (6) engine settings selected on validation predictions only; test marginals once, behind `--final` audit; (7) SBM pilot quarantined; generator frozen before comparisons; (8) row-normalization only, no statistics fit on val/test; (9) control-condition seeds logged; (10) paper phrases conditions by total label budget (m·C + validation size).

## M. Amendment policy and cut order

Any post-freeze change: dated CHANGELOG.md entry with rationale; affected results demoted to exploratory. Addenda slots (pre-declared, to be filled by their pre-registered procedures, not free changes): **M1** frozen σ_x from the pilot; **M2** fidelity-unary α calibration factor; **M3** Stage-1 selected hyperparameters; **M4** β^dog/λ^dog transfer values.

**Cut order under time pressure (mechanical, pre-registered):** (1) ECE/reliability detail (2) M5 (3) rewired/deeper controls → appendix stubs (4) E8r checksum (5) E13 (6) CiteSeer m-grid 3→2 (7) M3 row (8) grid trims. **Never cut:** 10 seeds, paired splits, the SBM heatmap, M2-vs-M4, the M4-MLP cell, validation-only tuning.

## N. Decision rules for conflicting results (pre-registered interpretations)

- **D1** Cora improves, CiteSeer doesn't → never average; place both on F3's diagnostics axis; qualitative-consistency claim only (n=2 forbids "prediction").
- **D2** M4 ties M2 (paired CI within ±0.5 acc points) → "the simple relaxation captures the useful inductive bias," explained via Proposition 1; agreement analysis shown.
- **D3** M4 worse than M2 → consult checksum: gated Gibbs closes the gap ⇒ MF bias (reported); otherwise the pairwise Potts prior is too rigid ⇒ negative result, reported.
- **D4** improvements vanish at m=20 → confirms H1.
- **D5** accuracy up, NLL worse at large β → both reported; overconfident smoothing (MF-calibration caveat; Gibbs-side numbers where available).
- **D6** null everywhere → verify SBM extremes (h=0.9, m=1, dogmatic); if still null, headline = "two-layer GCN aggregation already implements the useful part of the homophily prior," backed by the M4-MLP cell and the rewired null.

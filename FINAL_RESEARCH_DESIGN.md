# FINAL_RESEARCH_DESIGN.md

The design is **frozen** in `PROTOCOL.md` (git tag `protocol-freeze`); this document is the readable companion — same content, organized for a human rather than for pre-registration. Where the two disagree, `PROTOCOL.md` governs.

---

## Research Question

**Primary.** When does an explicit homophily prior improve semi-supervised node classification — as a function of (i) label scarcity and (ii) how well the prior matches the graph — and does it matter whether the prior is imposed **at training time** (as a regularizer on the parameters) or **at inference time** (via posterior inference in an explicit MRF)?

**Supporting, methodological.** Is the approximate inference faithful enough that the conclusions are about the model rather than about the inference algorithm?

**Why this and not the proposal's question.** The proposal asks "does the Laplacian regularizer raise Cora accuracy?" That question is confounded: a GCN's convolution *is* Laplacian smoothing (Li, Han & Wu 2018), so the baseline already contains the prior, and a null result would be uninterpretable. The refined question keeps the proposal's model and experiment (they become E1/E2) and adds the two axes that make the answer interpretable: the *truth* of the prior (manipulable only in synthetic graphs) and the *mechanism* by which it is imposed.

## Hypotheses

| ID | Prediction | Status | Test |
|---|---|---|---|
| **H1** scarcity | benefit largest when labels are scarcest: θ = [Acc(M4-MLP)−Acc(M0)](m=2) − [same](m=20) > 0 | **confirmatory C1** | E2, Cora, 30 seeds |
| **H2-benefit** | dogmatic-regime benefit grows with homophily: Δ(h=0.9) − Δ(h=0.5) > 0 | **confirmatory C2** | E5, CSBM m=5, 30 seeds |
| **H2-harm** | dogmatic-regime prior *harms* under misspecification: Δ(h=0.05) < 0 | **confirmatory C3** | E5, CSBM m=5, 30 seeds |
| H2-crossover | a crossover h* ∈ (0.05, 0.9) exists | estimation only | interpolation, median [min,max] |
| **H3** mechanism | training-time and inference-time imposition are correlated but not identical; differences exceed the retrain-noise floor and are structured | estimation only (no test) | E4 |
| **V4** inference validity | MF/LBP/Gibbs match exact marginals at small–moderate β; error grows with coupling/cyclicity; Gibbs error is MC variance, MF/LBP error is bias | validation criterion | E7 + gates |
| H5 | similarity weighting reduces dogmatic harm at low h | exploratory, gated | E10 |
| PQ | probability-quality reporting (NLL/Brier everywhere; "accuracy up, NLL worse at large β" is a pre-registered outcome) | reporting obligation | all |

Null forms are the directional complements. **No hypothesis is assumed true**; `PROTOCOL.md` §N pre-registers the interpretation of every major failure pattern, so negative results are deliverable results.

## Datasets

1. **Cora** (primary, real): 2,708 nodes, 5,278 undirected edges, 1,433 BoW features, 7 classes, edge homophily 0.810 (verified from loader). Continuity with the proposal.
2. **CiteSeer** (replication, real): 3,327 / 4,552 / 3,703 / 6, edge homophily 0.736 (verified). Tests transfer to a graph with measurably different diagnostics; m ∈ {1,5,20} only.
3. **Contextual SBM** (centerpiece, synthetic): n=2800, C=7, expected degree 8 held fixed while edge homophily h ∈ {0.05,…,0.90} varies; class-conditional Gaussian features with σ_x frozen by a quarantined pilot. **Only place where the prior's truth is manipulated.** Because the generator is known, it also supplies a closed-form feature-only Bayes oracle (Oracle-F) and an approximate graph+features ceiling (Oracle-G).
4. **Small exact-inference graphs** (synthetic): chain, tree, star, cycle, 4×4 grid, dense 2-block SBM; C=3 (+ one C=7 case) so exact marginals are computable.

**Declared scope for the CSBM.** Conditioned on labels, the SBM edge process has log-odds proportional to the Potts sufficient statistic — at high h *the prior is true by construction*. E5 is therefore framed as a **known-truth manipulation check of the measurement instrument**, never as evidence about real graphs; real-data results carry generalization alone. Two additions convert this circularity into findings: β_gen tracking (does validation-tuned β recover the generator-implied coupling log(p_in/p_out)?) and the Oracle-G ceiling (benefit reported as fraction-of-attainable).

## Methods

One MRF unifies everything: p(y | X, G) ∝ exp( Σ_i s_i(y_i) + β Σ_(i,j)∈E w_ij·1[y_i = y_j] ), with neural logits s as unary potentials.

**The mechanism ladder** — the same assumption at three levels of probabilistic seriousness:
1. **M2** Laplacian penalty λ·(1/|E|)Σ‖p_i−p_j‖² — smoothness heuristic on parameters (the proposal).
2. **M3** expected rule-violation λ·(1/|E|)Σ(1−p_i·p_j) — the probability that SameClass(i,j) fails under the factorized approximation (a semantic-loss-style construction).
3. **M4** explicit Potts MRF over labels, frozen logits as unaries, posterior computed by convergent mean-field.

**Proposition 1** (proved in the appendix) makes H3 precise: with edgewise agreement A(p)=Σ_(i,j)∈E p_i·p_j, L_rule = 1 − A/|E|; L_lap = (Σ‖p_i‖²+‖p_j‖²)/|E| − 2A/|E| (agreement **plus a confidence penalty**, optimized over *parameters*); and the MF objective = Σ q_i·s_i + β·A(q) + Σ H(q_i) (agreement **plus entropy**, optimized over the *posterior*). Same sufficient statistic, different optimized variable, different accompanying terms.

**Inference** (the CMPT 727 core): convergent color-blocked mean-field (primary engine, ELBO monotonicity asserted); damped loopy BP and chromatic Rao-Blackwellized Gibbs with R-hat/ESS gates (fidelity suite + at-scale checksum); exact marginals by brute-force enumeration or repeated min-fill variable elimination (ground truth).

## Baselines

| Model | Assumption | Role |
|---|---|---|
| **M0 MLP** | nodes i.i.d. (deliberately wrong) | feature-only floor; unary source for the clean M4-MLP cell |
| **M1 GCN** | homophily *implicitly*, via smoothing | the structural baseline every Δ is measured against |
| **LP** (harmonic functions, ZGL 2003) | labels+graph only, no features | the missing corner of the features×graph factorial |
| **Oracle-F / Oracle-G** (CSBM only) | known generative process | attainable ceilings |
| public-split GCN row | — | implementation sanity vs. literature (~81%) |

## Evaluation Metrics

Accuracy and macro-F1 (predictive); NLL and Brier (probability quality; ECE + reliability in appendix); rule satisfaction R = mean p_i·p_j **and** argmax edge-agreement (R alone conflates consistency with confidence); edge/adjusted homophily and label informativeness (explanatory variables); inference diagnostics (convergence, iterations, residual, R-hat, ESS, TV to exact); wall-clock.

## Main Experiments

- **E0** sanity gate — pipeline correctness before anything scientific.
- **E1** proposal reproduction — M1 vs M2 over the λ grid at Cora m=20.
- **E2** Cora low-label factorial — M0/M1/LP/M2/M4-GCN/M4-MLP × m∈{1,2,5,10,20}; the 2×2 unary factorial answers "how much does the GCN already encode?" → **F2, T2**.
- **E5** CSBM homophily sweep — h×m×model, **tuned and dogmatic** regimes, oracles, β_gen tracking → **F3 (centerpiece)**.
- **E6** CiteSeer replication → T2 rows, F3 overlay.
- **E7** inference fidelity vs exact → gates + **F-A**.
- **E4** mechanism ladder with the M1-retrain-noise disagreement floor → **T3**.

## Ablations

Prior on/off; implicit vs explicit (2×2 unaries); mechanism (M2 vs M3 vs M4); prior strength (λ/β grids → **F4**, the "consistency ≠ accuracy" figure); option vs dogma (tuned vs dogmatic); prior form (M3 row); edge weighting (M5 + feature-permutation control); label evidence (clamped variant, reported separately); inference engine (MF vs gated Gibbs).

## Robustness Experiments

Paired 10 seeds everywhere (30 on confirmatory cells) with per-seed dot plots and sign counts; **E13** small-validation ablation (5 labels/class selection budget — the standard critique of low-label GNN papers); **rewired-graph null** (degree-preserving configuration model: prior benefit must vanish); **deeper-GCN row** (is the prior just "more smoothing"?); **E8r Gibbs checksum** at scale; σ_x second level; Gibbs-budget halving; MF restarts at high β.

## Expected Evidence

Paired differences with 95% BCa bootstrap CIs and sign counts for every claim; three Holm-corrected confirmatory verdicts; a crossover estimate h* with seed spread; a mechanism table with disagreement measured against a retrain-noise floor; TV-to-exact curves validating the inference; and honest reporting of whichever way each result falls. **No expected outcome is treated as a result.**

## Potential Limitations

n=10 (30 on endpoints) seeds ⇒ approximate CI coverage; real-data error bars describe train-split/initialization variability on a *fixed* graph and test set, not cross-graph generalization (only the CSBM has graph-level replication); single backbone; transductive setting; one pairwise prior family; CSBM magnitudes and h* are generator-conditional (σ_x is a designer-chosen effect-size dial); tuning uses a validation set far larger than the training budget (quantified by E13); "logic" here is one pairwise rule, not general symbolic knowledge.

**Novelty stance.** Every ingredient is established (ZGL 2003; Hu et al. 2016; Xu et al. 2018 semantic loss; GMNN 2019; APPNP 2019; Correct & Smooth 2021; Jia & Benson 2022; Zhu et al. 2020; Ma et al. 2022; Platonov et al. 2023; Deshpande et al. 2018; Murphy-Weiss-Jordan 1999). The contribution is the **controlled, pre-registered measurement** under matched conditions — stated as such in the paper, with ten explicit "do not claim as new" guardrails.

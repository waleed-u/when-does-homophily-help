# PAPER_BLUEPRINT.md

Structure for the 6–10 page conference-style report (ICLR format; main text excludes
references, appendix allowed beyond). Every claim below is mapped to evidence that already
exists in `figures/`, `tables/` and `results/processed/analysis.json`.

**Working title:** *When Does a Homophily Prior Help? Regularization versus Probabilistic
Inference in Low-Label Node Classification*

**One-sentence thesis:** A relational prior pays according to how much relational information
the base model has *not* already extracted, and it is safe only when its strength is chosen by
validation rather than assumed — as shown by a controlled homophily sweep, a degree-preserving
rewiring control, and inference validated against exact marginals.

---

## Page budget (9 pages main text)

| § | Section | Pages | Carries |
|---|---|---|---|
| — | Abstract | 0.25 | three findings, stated as measured |
| 1 | Introduction | 1.0 | gap + three contributions |
| 2 | Background & related work | 0.9 | GCN=smoothing, graph-SSL lineage, novelty disclaimers |
| 3 | Methods | 2.0 | ladder, Potts MRF, mean-field, **Proposition 1** |
| 4 | Experimental setup | 0.9 | datasets, CSBM generator, protocol, statistics |
| 5 | Results | 2.6 | four claims, F2–F5, T2–T4 |
| 6 | Discussion & limitations | 0.9 | mechanism, threats to validity |
| 7 | Future work + Conclusion | 0.45 | learned compatibility; take-home |
| — | References | (excl.) | ~20 verified |
| — | Appendix | (excl.) | proofs, grids, per-seed tables, controls, fidelity detail |

---

## §1 Introduction

Five paragraphs: (1) relational data + label scarcity; (2) GNNs already encode homophily
*implicitly* — a GCN convolution is Laplacian smoothing (Li et al. 2018) — so the value of
stating the assumption explicitly is unmeasured; (3) we impose one assumption three ways of
increasing probabilistic seriousness and measure when each pays; (4) the missing questions:
*when* does it help, and does the mechanism matter; (5) contributions.

**Contributions (descriptive, not inflated):**
1. A pre-registered, matched-budget factorial over label scarcity × prior correctness × how the
   prior is imposed, on one fixed backbone.
2. A controlled separation of *relational reasoning* from *generic regularization* via
   degree-preserving rewiring — showing the popular Laplacian penalty is mostly the latter.
3. Inference validated against exact marginals on 11,280 comparisons, including an honest
   negative: at the couplings models actually use on Cora, no trustworthy MCMC reference exists.

## §2 Background and related work

GCN and Laplacian smoothing (Kipf & Welling 2017; Li, Han & Wu 2018); graph-based SSL and
harmonic functions (Zhu, Ghahramani & Lafferty 2003); logic-rule and semantic-loss
regularization (Hu et al. 2016; Xu et al. 2018); GNN+MRF/CRF hybrids (GMNN, Qu et al. 2019; Gao
et al. 2019); decoupled predict-then-propagate (APPNP, Gasteiger et al. 2019; Correct & Smooth,
Huang et al. 2021); smoothing/inference unifications (Wang & Leskovec 2020; Jia & Benson 2022);
homophily measures and heterophily (Zhu et al. 2020; Ma et al. 2022; Platonov et al. 2023 ×2);
CSBM (Deshpande et al. 2018); approximate-inference benchmarking (Murphy, Weiss & Jordan 1999);
chromatic Gibbs (Gonzalez et al. 2011); evaluation protocol (Shchur et al. 2018).

**Explicit disclaimers (two sentences):** the model class, the post-hoc correction, the losses,
the heterophily failure mode and the inference behaviour are all established; the contribution
is the controlled measurement under matched conditions.

## §3 Methods

1. Setup and notation; transductive node classification.
2. The three rungs: Laplacian penalty (M2) → expected rule-violation / semantic loss (M3) →
   explicit Potts MRF over labels with frozen logits as unary potentials (M4).
3. **Proposition 1** (3-line proof; appendix): with A(p) = Σ_(i,j)∈E p_i·p_j,
   L_rule = 1 − A/|E|; L_lap = (Σ‖p_i‖²+‖p_j‖²)/|E| − 2A/|E|; F_MF = Σq_i·s_i + βA(q) + ΣH(q_i).
   Same sufficient statistic; different optimised variable (parameters vs posterior) and
   different companion term (confidence penalty vs entropy). *This proposition predicts
   Finding 4 and the rewiring control tests it.*
4. Mean-field: colour-blocked coordinate ascent, ELBO monotone by construction, run to
   convergence (never tuned).
5. Validation: exact marginals by enumeration / min-fill variable elimination; loopy BP and
   chromatic Rao-Blackwellised Gibbs with R-hat/ESS gates.
6. The two prior regimes: **tuned** (β selected on validation) vs **dogmatic** (β transferred
   from the reference cell) — the distinction that makes harm observable.

## §4 Experimental setup

Datasets and diagnostics (**T1**); CSBM generator with p_in/p_out holding expected degree fixed,
σ_x frozen by a quarantined pilot; splits, 10 paired seeds (30 on endpoints), equal tuning
budgets, validation-only selection; metrics; the pre-registered endpoints C1–C3 with Holm
correction; one sentence on what the intervals mean (fixed graph/test set on real data; graph
redrawn per seed on CSBM).

## §5 Results — organised by claim

### 5.1 The prior's value tracks unary quality, not label scarcity  → **F2**, **T2**, C1
- C1 **refuted and significantly reversed**: θ = −7.07 pts, CI [−10.26, −3.91], p=6.6e-4 (30 seeds).
- Opposite trends: Laplacian−GCN falls +4.38→+0.59 as m grows; MRF(MLP)−MLP rises +4.45→+19.55.
- Mechanism: at m=1 the MLP is 29.2% (chance 14.3%), so β* is small (0.375) and there is little
  to propagate; by m≥2, β* saturates the grid and propagation converts features into +19–22 pts.
- Replicated on CiteSeer (+5.34/+12.46/+10.09, 10/10 seeds each).

### 5.2 Prior correctness governs benefit — but only a committed prior can be harmed  → **F3**, **F3b**, **T4**, C2/C3
- C2 supported (+78.77 pts, 10/10); C3 supported (−29.72 pts, 10/10).
- Dogmatic β=2: −45 pts at worst; **posterior collapse** (edge agreement 0.966, accuracy 16.9% ≈ chance).
- Tuned β: worst cell −0.5 pts — validation retreats toward β=0 when the prior is wrong.
- Crossover h* = 0.76 / 0.57 / 0.43 at m = 1 / 5 / 20 (10/10 seeds, per-seed interpolation).
- **A low-h graph is not uninformative:** β_gen = −1.15 at h=0.05, and the true-model oracle
  reaches 77.0% > 73.4% (features only) while the positive-β prior collapses to 15%.

### 5.3 The Laplacian penalty is mostly generic regularization; the MRF is relational  → rewiring control (**T6**), Prop. 1
- On degree-preserving rewired Cora: Laplacian keeps +3.24 [+1.82,+5.17]; MRF falls to
  +0.11 [−0.15,+0.74] (m=2), −0.06 (m=5).
- Interpreted through Proposition 1's confidence-penalty term.
- Honest caveat: a 3-layer GCN gains +2.9/+3.7 pts at m=1/2 — comparable to the explicit prior
  in that regime, so no claim of superiority over deeper propagation at very low label rates.
- Mechanism claims are made only above the retrain-noise floor (16.2%/14.2%/6.5% disagreement).

### 5.4 The inference is validated, and its failure mode is diagnosable  → **F5**, **F6**, **T5**
- BP exact on all trees (max TV 4.8e-7); β=0 identity to 2.1e-14 for every engine.
- MF bias grows with coupling everywhere; bias floors (MF 0.176, LBP 0.060) vs Gibbs variance
  decay (0.077 → 0.007 over 10 → 3000 sweeps).
- At scale: where Gibbs passed its gates (β ≤ 1) MF agreed to TV 0.035; at β=2 Gibbs **failed to
  mix** (R-hat 2.58, ESS 4.7) — reported as "no trustworthy reference at this coupling".
- Consistency ≠ accuracy: R and edge agreement rise monotonically with β while accuracy collapses.

## §6 Discussion and limitations

Why GCNs blunt the explicit prior (double-counted relational evidence); over-imposition ↔
posterior collapse; why tuning converts a dangerous assumption into a safe option; what the
symbolic-rule framing does and does not buy.
**Threats to validity:** n=10 seeds (30 on endpoints), approximate CI coverage; real-data
intervals describe protocol variability on a fixed graph/test set; single backbone; transductive
only; one pairwise prior family; CSBM magnitudes generator-conditional (σ_x is an effect-size
dial) and the Potts prior is correct *by construction* at high h; β-grid maximum binding, so
MRF gains are lower bounds (exploratory extension in **T7**); validation set larger than the
training budget (quantified by the small-validation ablation).

## §7 Future work and conclusion

Learned class-compatibility (EM / GMNN-style) — motivated directly by Finding 3, since the
failure at low h is a *sign* error a learned compatibility could fix; structured variational
families or annealed sampling for the β≥2 regime where Gibbs fails; richer rules than a single
pairwise predicate.

**Conclusion (one paragraph):** the take-home from §12 of `SCIENTIFIC_FINDINGS.md`.

---

## Claim → evidence map (for the writer)

| Claim | Figure | Table | Stat |
|---|---|---|---|
| Benefit tracks unary quality, not scarcity | F2b | T2, T3 | C1: −7.07 [−10.26,−3.91] |
| Prior correctness governs benefit | F3a | T4 | C2: +78.77 [+55.53,+84.87] |
| Dogmatic prior harms; tuned prior is safe | F3a vs F3b | T4 | C3: −29.72 [−31.81,−28.16] |
| Crossover exists and moves with m | F3b | T4 | h*=0.76/0.57/0.43 |
| Low-h graph still informative (wrong sign) | F3b | T4 | Oracle-G 77.0% vs Oracle-F 73.4% |
| Laplacian ≈ generic regularization | — | T6 | rewired +3.24 vs MRF +0.11 |
| Consistency ≠ accuracy | F4c | — | edge agreement 0.966 at 16.9% acc |
| Inference validated | F5 | T5 | BP-tree TV 4.8e-7; β=0 TV 2.1e-14 |
| Bias vs variance | F6 | T5 | floors 0.176/0.060 vs 0.077→0.007 |
| MF trustworthy only where Gibbs mixes | — | T6 | gated TV 0.035; ungated R-hat 2.58 |

## Figure inventory (5 main + 1 appendix)

F2 low-label curves + paired Δ · F3 CSBM heatmap (dogmatic vs tuned) · F3b Δ-vs-h with crossover
and β* vs β_gen · F4 sensitivity and consistency · F5 inference fidelity · F6 bias-vs-variance
(appendix). A method schematic (F1) remains to be drawn for §3.

## Remaining work before submission

1. Draw F1 (method schematic: unaries → prior as regularizer vs as posterior inference).
2. Write the LaTeX in `report/` using the ICLR style; import figures from `figures/`, tables
   from `tables/*.tex`.
3. Appendix: Proposition 1 proof, mean-field derivation, full grids and selected values,
   per-seed tables, control details, fidelity per-structure tables.

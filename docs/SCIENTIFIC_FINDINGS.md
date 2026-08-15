# SCIENTIFIC_FINDINGS.md

Evidence-based conclusions from the executed experiments. Every number here is produced by
`experiments/analyze.py` from `results/raw/*.csv` and is reproducible with the commands in
`RESULTS_MANIFEST.md`. **7,744 model runs and 11,280 exact-inference comparisons.** All 13
automated audit checks pass (`results/processed/audit.json`).

Where a hypothesis was pre-registered (`PROTOCOL.md`, git tag `protocol-freeze`), the verdict
below is the pre-registered one — including the one that went against the prediction.

---

## 1. Headline

> A homophily prior is worth almost nothing on top of a GCN, is worth a great deal on top of
> feature-only evidence, and is catastrophic when it is *committed to* on a graph that does not
> satisfy it. What governs the outcome is not label scarcity, as the proposal assumed, but
> (i) how much relational information the base model already extracts and (ii) whether the
> prior's strength is chosen by validation or asserted in advance.

## 2. Pre-registered confirmatory verdicts (Holm-corrected, α=0.05)

| Endpoint | Prediction | Result | Verdict |
|---|---|---|---|
| **C1** (H1, scarcity) | benefit larger at m=2 than m=20 | θ = **−7.07 pts**, 95% CI [−10.26, −3.91], p=6.6e-4, 7/30 seeds positive | **REFUTED — and significantly reversed** |
| **C2** (H2, benefit) | dogmatic benefit larger at h=0.9 than h=0.5 | θ = **+78.77 pts**, CI [+55.53, +84.87], p=0.002, 10/10 seeds | **SUPPORTED** |
| **C3** (H2, harm) | dogmatic prior *hurts* at h=0.05 | Δ = **−29.72 pts**, CI [−31.81, −28.16], p=0.002, 10/10 seeds | **SUPPORTED** |

C1 is the scientifically interesting one: the Holm-corrected test rejects the null, but in the
**opposite direction to the pre-registered prediction**. The MRF's benefit over a feature-only
MLP is *larger* with more labels (+19.7 pts at m=20) than with fewer (+12.6 pts at m=2).

## 3. Finding 1 — "Priors help when labels are scarce" is false as stated; the real variable is how good the unary evidence is

The proposal's core assumption was that a relational prior compensates for missing supervision.
Measured directly, the two arms move in opposite directions (Fig. F2b):

| m | GCN+Laplacian − GCN | MRF(GCN) − GCN | MRF(MLP) − MLP |
|---|---|---|---|
| 1 | **+4.38** | +2.80 | +4.45 |
| 2 | +2.17 | +2.37 | +11.23 |
| 5 | +1.08 | +4.30 | +19.05 |
| 10 | +0.75 | +1.40 | +22.50 |
| 20 | +0.59 | +0.74 | **+19.55** |

(Cora, paired over 10 common seeds, validation-selected hyperparameters.)

**Mechanism.** Relational inference *propagates* the label information already present in the
unary evidence; it does not create it. At m=1 the MLP is 29.2% accurate — barely above the
14.3% chance level — so there is almost nothing to propagate, and validation selects a weak
coupling (median β* = 0.375). By m≥2 the MLP carries real signal, validation selects the
strongest coupling available (β* = 2.0, the grid maximum), and propagation converts feature
evidence into a 19–22 point gain. Against a GCN the pattern inverts because the GCN has
*already* propagated: what remains for the explicit prior shrinks as supervision grows.

**Conclusion.** The benefit of an explicit relational prior is governed by the informativeness
of the unary evidence and by how much relational structure the base model has already
exploited — not by label scarcity per se.

## 4. Finding 2 — Prior correctness governs everything, but only a *committed* prior can be harmed by it

The contextual-SBM sweep manipulates the truth of the prior while holding expected degree (8.0),
class balance, feature dimension and feature noise fixed (Fig. F3, Table T4, m=5):

| h | 0.05 | 0.15 | 0.30 | 0.50 | 0.70 | 0.90 |
|---|---|---|---|---|---|---|
| Δ **dogmatic** (β fixed = 2.0) | −29.7 | −29.6 | −29.6 | −24.1 | +50.1 | **+54.7** |
| Δ **tuned** (β chosen on validation) | −0.5 | −0.3 | +2.0 | +12.2 | +50.1 | +54.7 |

Two conclusions, and the contrast between them is the point:

1. **A dogmatic prior is dangerous.** Fixing β at the value that is optimal on a homophilous
   graph and applying it everywhere costs up to **45 points** (m=20, h=0.05). The failure is
   not gradual degradation but **posterior collapse**: at β=2, h=0.05, argmax edge agreement
   reaches 0.966 and accuracy falls to 16.9% ≈ chance — the coupling term overwhelms the
   unaries and the MRF assigns nearly every node the same label.
2. **A tuned prior is nearly free.** Because β=0 nests the baseline, validation simply retreats
   to a weak coupling when the prior is wrong; the worst observed tuned cell is −0.5 pts.

**Crossover** (dogmatic regime, per-seed interpolation, 10/10 seeds): h* = 0.76 [0.52, 0.87] at
m=1, **0.57 [0.40, 0.59]** at m=5, 0.43 [0.42, 0.43] at m=20. More supervision lets the model
tolerate a wronger prior, and shifts the break-even homophily downward.

*This experiment was only possible because the design carries both regimes.* The originally
planned tuned-only sweep would have shown a flat, benign heatmap and the harm region — the
scientifically important half of the result — would have been invisible by construction.

## 5. Finding 3 — A low-homophily graph is not an uninformative graph; the prior is simply wrong-signed

At h=0.05 the generator implies a **negative** coupling, β_gen = log(p_in/p_out) = **−1.15**.
The oracle that uses the true generative model with that coupling reaches **77.0%** — *better*
than the feature-only Bayes oracle (73.4%) — while the positive-β Potts prior collapses to 15%.

The graph at h=0.05 carries genuinely useful information (dissimilar neighbours are predictive);
what fails is the *same-label* assumption, not the relational evidence. This is direct,
controlled evidence for the distinction between a misspecified prior and an uninformative graph.

Validation tuning also fails to recover the true coupling: β* saturates at the grid maximum
2.0 while β_gen rises to 3.99 at h=0.9, and at low h no positive β can represent an
anti-homophilous compatibility at all (Fig. F3b).

## 6. Finding 4 — The Laplacian penalty is largely *not* a relational prior; the MRF is

The mechanism question was answered by a control, not by a comparison of accuracies. Rewiring
Cora with a degree-preserving configuration model destroys label structure while preserving
every node's degree (edge homophily falls from 0.810 to ≈0.15):

| Model on **rewired** Cora | Δ vs GCN | 95% CI | seeds + |
|---|---|---|---|
| GCN + Laplacian penalty, m=2 | **+3.24** | [+1.82, +5.17] | 9/10 |
| MRF on GCN (mean-field), m=2 | **+0.11** | [−0.15, +0.74] | 4/10 |
| GCN + Laplacian penalty, m=5 | **+2.22** | [+0.52, +4.40] | 7/10 |
| MRF on GCN (mean-field), m=5 | **−0.06** | [−0.23, +0.15] | 3/10 |

The MRF's benefit **vanishes** when label structure is destroyed — its interval covers zero at
both label rates, so it is genuinely relational. The Laplacian penalty's benefit **survives**
on a graph whose edges now carry no label information (interval excludes zero at both rates):
much of what it contributes on Cora is generic regularization, not relational reasoning.

This is exactly what the algebra predicts. With edgewise agreement A(p) = Σ_(i,j)∈E p_i·p_j,

  L_rule = 1 − A/|E|,  L_lap = (Σ‖p_i‖² + ‖p_j‖²)/|E| − 2A/|E|,  F_MF = Σ q_i·s_i + βA(q) + ΣH(q_i)

All three are driven by the same pairwise statistic, but the Laplacian form carries an extra
**confidence penalty** Σ‖p_i‖², which discourages confident predictions regardless of what the
graph says. The rewired control isolates that term empirically.

**Caveat, reported honestly:** a 3-layer GCN — pure extra propagation, no probabilistic
machinery — gains +2.91 pts (m=1) and +3.69 pts (m=2) over the 2-layer baseline, comparable to
the explicit prior's gains in that regime. At very low label rates on Cora we cannot claim the
explicit prior does anything a deeper propagation scheme would not.

## 7. Finding 5 — The inference is validated, and where it fails, it fails diagnosably

Against exact marginals on 11,280 controlled comparisons (Fig. F5):

- **Loopy BP is exact on all three tree structures** at every coupling (max TV = 4.8e-7, at the
  message-tolerance floor) — the gate that licenses trusting the implementation.
- **Every engine reproduces softmax(s) at β=0** to 2.1e-14.
- **Mean-field error grows steeply with coupling on every structure**, reaching TV ≈ 0.3–0.5 at
  β=4: its factorized family cannot represent the pairwise dependence that strong coupling
  creates.
- **Bias vs variance** (4×4 grid, β=1.75): Gibbs error falls monotonically with budget
  (0.077 → 0.007 from 10 to 3000 kept sweeps) while MF (0.176) and LBP (0.060) sit at fixed
  bias floors no extra compute removes.

**At scale (Cora, m=5, 10 seeds), the honest result is a diagnostic failure, not a comparison.**
Where the Gibbs reference passed its R-hat/ESS gates (β ≤ 1), mean-field agreed with it closely
(mean TV = 0.035). Where validation had selected β = 2, Gibbs **failed to mix** — R-hat up to
2.58 and ESS as low as 4.7 from 8,000 samples — so at exactly the coupling the models use, *no
trustworthy reference exists*. Reporting mean-field as "validated" there would have been
unsupportable; reporting Gibbs as ground truth would have been worse. The correct statement is
that the Potts posterior on Cora at β=2 is multimodal enough to defeat a chromatic Gibbs sampler
at this budget.

## 8. Finding 6 — Consistency is not accuracy

Rule satisfaction R = mean p_i·p_j and argmax edge agreement both increase monotonically with β,
while accuracy peaks and then collapses (Fig. F4c). At h=0.05, β=2 the model is maximally
*self-consistent* (edge agreement 0.966) and minimally *correct* (16.9%). Optimising a model's
agreement with a symbolic rule is therefore not a proxy for optimising its correctness — a
caution that applies directly to the neuro-symbolic framing the proposal started from.

## 9. Robustness

- **Small validation budget.** Re-selecting hyperparameters with 5 labels/class (rather than
  ~500 validation labels) preserves the MRF-over-MLP effect: +5.64 pts (m=1, 8/10 seeds),
  +7.82 pts (m=2, 9/10). The conclusions do not depend on an unrealistically large tuning set.
- **Retrain-noise floor.** Two GCNs differing only by initialisation disagree on 16.2% (m=2),
  14.2% (m=5), 6.5% (m=20) of test predictions. Mechanism claims are only made where the
  measured effect exceeds this floor.
- **Replication.** CiteSeer (h=0.736) reproduces every qualitative Cora finding. MRF(MLP)−MLP
  again *grows* with supervision (+5.34, +12.46, +10.09 pts at m=1, 5, 20; 10/10 seeds each),
  confirming that C1's reversal is not a Cora artefact. MRF(GCN)−GCN stays small
  (+0.42, +2.04, +0.78). The Laplacian penalty's edge is largest at m=1 (+5.61, 10/10) and gone
  by m=20 (−0.04, 3/10) — consistent with a mostly-generic regulariser that matters when the
  model is least constrained by data.
- **Grid truncation (exploratory).** Validation selects the β-grid maximum (2.0) in most
  MRF-on-MLP cells, so reported gains are **lower bounds**; the exploratory extension to β≤8 is
  in `tables/T7_beta_extension.csv` and is labelled exploratory per the amendment policy.

## 10. What the evidence does *not* support

- Not that "logic/symbolic priors improve GNNs" — on a GCN backbone the explicit prior's gain is
  ≤1 pt at m≥10 and is not distinguishable from deeper propagation at m≤2.
- Not that heterophily failure is a discovery: it is established (Zhu et al. 2020; Ma et al.
  2022; Platonov et al. 2023). The contribution is locating the crossover under controlled
  degree/feature conditions and separating the tuned from the dogmatic regime.
- Not that mean-field is "good enough" in general — only that it agreed with a *gated* Gibbs
  reference at β ≤ 1 on Cora.
- Not any cross-graph generalisation from the real-data intervals: on Cora and CiteSeer the
  seeds vary the training sample and initialisation on a **fixed** graph and test set. Only the
  CSBM redraws the graph per seed.
- Not that the CSBM results transfer directly to real graphs: at high h the Potts prior is
  correct *by construction* (the SBM's edge log-odds are exactly the Potts sufficient
  statistic), so that sweep is a known-truth manipulation check of the measurement instrument,
  not evidence about citation networks.

## 11. Trade-offs surfaced

| Axis | Finding |
|---|---|
| Accuracy vs. probability quality | MF and gated Gibbs differ in *marginals* (TV 0.035) far more than in *accuracy* (1.6 pts); argmax is robust to inference error that would matter for any downstream use of the probabilities. |
| Compute vs. fidelity | MF converges on Cora in 0.07 s with irreducible bias; Gibbs costs ~3.3 s/chain and buys accuracy only while it mixes. Above β≈1 the extra compute buys nothing diagnosable. |
| Safety vs. strength | The dogmatic regime attains the same peak benefit as the tuned regime at high h (+54.7) but risks −45 pts elsewhere. Tuning is what converts a dangerous assumption into a safe option. |
| Interpretability vs. performance | The most interpretable mechanism (an explicit MRF with one coupling parameter) is also the one whose benefit provably disappears on rewired graphs — its gains are attributable, unlike the Laplacian penalty's. |

## 12. Answer to the project's central question

**A relational prior is most useful when the base model has not already exploited the graph and
the unary evidence is informative enough to propagate; it becomes harmful when its strength is
fixed in advance on a graph that violates it (below h* ≈ 0.57 at m=5, costing up to 45 points
through posterior collapse); and the controlled experiments show this because the contextual-SBM
sweep manipulates the prior's correctness at fixed degree and features, the rewiring control
separates relational reasoning from generic regularisation, and the exact-inference suite
certifies that these conclusions reflect the model rather than the approximate inference.**

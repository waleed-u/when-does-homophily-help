# CHANGELOG — protocol amendments

Post-freeze changes to anything governed by PROTOCOL.md are recorded here with date and
rationale. Results affected by an amendment are demoted to exploratory (PROTOCOL.md §M).

## A1 — Gibbs "converged" flag drops the inter-chain-TV criterion (implementation phase, before any experimental run)

**Changed.** PROTOCOL.md §H defined the Gibbs pass/fail gate as `R-hat < 1.01 AND ESS > 400 AND
max inter-chain marginal TV < 0.02`. The third criterion is removed from the pass/fail decision;
it is still computed, logged (`interchain_tv`) and reported.

**Rationale (empirical, from the implementation smoke test).** On a 12-node cycle at β=1 with
4 chains × 2000 kept sweeps, the sampler reached R-hat = 1.0004 and ESS = 5540 — textbook good
mixing — yet max-over-nodes inter-chain TV was 0.0258. That statistic is dominated by
finite-budget Monte-Carlo noise (per-chain standard error ≈ √(p(1−p)/n_eff) ≈ 0.01, and a max
over nodes and chain pairs inflates it further), so it measures *precision at a given budget*,
not convergence. Keeping it as a gate would have flagged well-mixed chains as failures and
systematically hollowed out the fidelity figure at exactly the couplings of interest — an
artifact, not a finding. R-hat and ESS are the diagnostics that actually target mixing.

**Scope.** Diagnostic-definition only. No hypothesis, endpoint, model, metric, or comparison
changes. The amendment was made *before any experimental run produced a result*, so no results
are affected and none are demoted. Inter-chain TV remains in `fidelity.csv`/`runs.csv` and is
discussed as a precision statistic.

---

Pre-declared addenda slots (filled by their pre-registered procedures; not free changes):

- [x] **M1** frozen **σ_x = 0.50** (quarantined pilot seeds 100–102, `results/logs/sigma_pilot.log`).
      Oracle-F accuracy = 0.7444, inside the pre-registered [0.70, 0.80] band. Candidates 0.45
      (0.7999) and 0.50 (0.7444) both qualified; the protocol fixes the band but not the
      tie-break, so the qualifying candidate closest to the band **midpoint** was taken rather
      than the first in the grid — 0.45 sat on the boundary where seed noise could push it out
      of spec. Selection used only the model-free Oracle-F, never any model's accuracy.
      Generator checks pass at every h (degree 7.93–7.98 vs target 8, empirical h within 0.003
      of target, classes exactly balanced). Recorded facts for later interpretation: at h=0.15
      label informativeness is ≈0 (the graph carries almost no label signal at chance homophily
      1/C≈0.143), and at h=0.05 adjusted homophily is **negative** (−0.107) with
      β_gen = −1.15, i.e. the generator implies an *anti*-homophily coupling — a positive-β
      Potts prior is not merely weak there, it has the wrong sign.
- [ ] **M2** fidelity-suite unary α calibration factor (matched to Cora GCN margins at m=5)
- [ ] **M3** Stage-1 selected backbone hyperparameters (per dataset, per architecture)
- [ ] **M4** dogmatic-transfer values β^dog, λ^dog (median tuned value at h=0.9, m=5)

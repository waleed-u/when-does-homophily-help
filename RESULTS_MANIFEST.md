# RESULTS_MANIFEST.md

Traceability map: every figure and table in the report comes from a script reading only
`results/raw/*.csv`. No number is transcribed by hand.

## Provenance chain

```
experiments/run_*.py  ->  results/raw/{runs_*,fidelity_*}.csv   (append-only, one row per run)
                      ->  experiments/analyze.py                 (selection + statistics)
                      ->  tables/*.csv|.tex, results/processed/analysis.json
                      ->  experiments/make_figures.py            (figures)
                      ->  figures/*.pdf|.png
```

Supporting records: `results/final_eval_audit.log` (one line per test evaluation),
`results/logs/*.log` (stdout of every run), `PROTOCOL.md` (pre-registration, git tag
`protocol-freeze`), `CHANGELOG.md` (every post-freeze amendment).

## Experiment -> raw output -> figure/table -> paper section

| Experiment | What it answers | Raw output | Figure / Table | Paper section |
|---|---|---|---|---|
| **E0** sanity gate | is the pipeline trustworthy? | `results/logs/` (gate only, no rows) | — | §4 setup (one sentence) |
| **E1** proposal reproduction (λ sweep, Cora m=20) | does the original proposal's regularizer help? | `runs_cora_m20.csv` | F4a, T3 | §5.1, §5.3 |
| **E2** Cora low-label factorial | when does the prior pay? | `runs_cora_m{1,2,5,10,20}.csv` | **F2**, **T2** | §5.1 |
| **E2c** 30-seed confirmatory cells | endpoint C1 | `runs_cora_conf_m{2,20}.csv` | C1 verdict | §5.1 |
| **E3** strength sensitivity | is there an over-imposition regime? | same as E1/E2 (full grids logged) | **F4** | §5.3 |
| **E4** mechanism ladder | does *how* the prior is imposed matter? | `runs_cora_*.csv` + floor control | **T3** | §5.3 |
| **E5** CSBM homophily sweep | does prior *correctness* govern benefit? | `runs_sbm_h*.csv` | **F3**, F3b, **T4** | §5.2 |
| **E5c** endpoints C2/C3 | H2 benefit and harm | same | C2/C3 verdicts | §5.2 |
| **E6** CiteSeer replication | does the Cora pattern transfer? | `runs_citeseer_m*.csv` | T2 rows | §5.1 |
| **E7** inference fidelity | are the approximations trustworthy? | `fidelity_*.csv` | **F5**, F6, T5 | §5.4 / appendix |
| **E8r** MF-vs-Gibbs checksum | is mean-field trustworthy at scale? | `runs_ctrl_checksum.csv` | T6 | §5.4 |
| **E13** small-validation ablation | do conclusions survive realistic tuning? | `runs_ctrl_smallval.csv` | T6 | §6 limitations |
| **C-rewired** degree-preserving null | does the prior exploit *label* structure? | `runs_ctrl_rewired.csv` | T6 | §5.3 |
| **C-deeper** 3-layer GCN | is the prior just more propagation? | `runs_ctrl_deeper.csv` | T6 | §5.3 |
| **C-floor** retrain-noise floor | is a mechanism difference above seed noise? | `runs_ctrl_floor.csv` | T6 | §5.3 |
| **C-simcheck** M5 precondition | is cosine similarity informative here? | `runs_ctrl_simcheck.csv` | T6 | §5.3 |
| **X-betaext** β-grid extension | is the pre-registered grid truncating the optimum? | `runs_betaext_*.csv` | T7 (exploratory) | appendix |
| σ_x pilot | freeze the CSBM effect-size dial | `results/logs/sigma_pilot.log` | — | §4 setup |
| Stage-1 tuning | freeze backbone hyperparameters | `results/logs/stage1_tuning.log` | — | appendix |

## Figures

| File | Content | Built by |
|---|---|---|
| `figures/F2_low_label_cora.pdf` | accuracy vs label budget + paired Δ with CIs | `make_figures.fig_low_label` |
| `figures/F3_sbm_heatmap.pdf` | Δ over (h, m): dogmatic vs tuned prior | `make_figures.fig_sbm_heatmap` |
| `figures/F3b_sbm_curve_and_beta.pdf` | Δ vs h with crossover; β* vs β_gen | same |
| `figures/F4_sensitivity.pdf` | λ/β sensitivity; consistency ≠ accuracy | `make_figures.fig_sensitivity` |
| `figures/F5_fidelity.pdf` | TV to exact marginals per structure | `make_figures.fig_fidelity` |
| `figures/F6_bias_variance.pdf` | Gibbs variance decay vs MF/LBP bias floors | same |

## Tables

| File | Content |
|---|---|
| `tables/T1_datasets.csv` | dataset statistics and homophily diagnostics (recomputed from loaders) |
| `tables/T2_main_results.csv` | Cora + CiteSeer accuracy by label budget, mean ± sd over paired seeds |
| `tables/T3_mechanism.csv` | paired Δ with CIs and sign counts for each imposition mechanism |
| `tables/T4_sbm_sweep.csv` | CSBM sweep: models, oracles, dogmatic and tuned Δ per (h, m) |
| `tables/T5_fidelity.csv` | mean TV to exact by structure, β and engine |
| `tables/T6_controls.csv` | rewired null, deeper GCN, retrain floor, small validation, MF-vs-Gibbs |
| `tables/T7_beta_extension.csv` | exploratory β-grid extension |

## Reproduction

```bash
.venv/bin/pytest tests/ -q                       # pre-registered gates
.venv/bin/python -m experiments.run_e0           # sanity gate
.venv/bin/python -m experiments.run_sigma_pilot  # frozen sigma_x
.venv/bin/python -m experiments.tune_stage1      # frozen backbone hyperparameters
# main blocks (shardable; see README for the parallel form)
.venv/bin/python -m experiments.run_real --dataset cora --seeds 0-9
.venv/bin/python -m experiments.run_sbm --seeds 0-9
.venv/bin/python -m experiments.run_fidelity
.venv/bin/python -m experiments.run_controls --control rewired   # etc.
.venv/bin/python -m experiments.analyze
.venv/bin/python -m experiments.make_figures
```

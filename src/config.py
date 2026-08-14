"""Frozen protocol constants. Stage-1 values come from experiments/tune_stage1.py
(CHANGELOG addendum M3); sigma_x from the quarantined pilot (addendum M1)."""

STAGE1 = {
    "cora":     {"mlp": {"lr": 0.01,  "weight_decay": 1e-3}, "gcn": {"lr": 0.01,  "weight_decay": 1e-3}},
    "citeseer": {"mlp": {"lr": 0.01,  "weight_decay": 1e-3}, "gcn": {"lr": 0.01,  "weight_decay": 5e-4}},
    "sbm":      {"mlp": {"lr": 0.01,  "weight_decay": 5e-4}, "gcn": {"lr": 0.005, "weight_decay": 1e-3}},
}

LAMBDA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]
BETA_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
SIGMA_X = 0.50

SEEDS = list(range(10))
CONFIRMATORY_SEEDS = list(range(30))

CORA_M = [1, 2, 5, 10, 20]
CITESEER_M = [1, 5, 20]
SBM_M = [1, 5, 20]
SBM_H = [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]

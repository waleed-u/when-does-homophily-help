"""Statistical analysis primitives (PROTOCOL.md §K).

Design commitments encoded here rather than left to the analyst:

* Hyperparameter selection is a pure function of VALIDATION accuracy. `select_tuned` never
  reads a test column — selection cannot be contaminated even by accident.
* Everything is PAIRED by seed: models sharing a seed share the split (and, for post-hoc
  variants, the trained checkpoint), so per-seed differences remove split and initialisation
  variance from the comparison.
* Estimation first: paired mean differences with BCa bootstrap intervals and sign counts. Only
  the three pre-registered endpoints get a hypothesis test, Holm-corrected.
* What the intervals mean is stated wherever they are produced: on the real graphs the seeds
  vary the training sample and initialisation on a FIXED graph and test set, so intervals
  describe protocol variability, not generalisation to new graphs. Only the CSBM redraws the
  graph per seed.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
GROUP = ["dataset", "model", "label_per_class", "split_seed"]


# ------------------------------------------------------------------------------------ loading
def load_runs(pattern: str = "results/raw/runs_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / pattern)))
    if not files:
        raise FileNotFoundError(f"no run shards matching {pattern}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("beta", "lambda", "target_h", "empirical_h", "adjusted_h", "beta_gen",
              "val_acc", "test_acc", "val_nll", "test_nll", "test_f1", "test_brier",
              "rule_score", "edge_agreement", "ece", "val_f1"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_fidelity(pattern: str = "results/raw/fidelity_*.csv") -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / pattern)))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------------- selection
def select_tuned(df: pd.DataFrame, group=GROUP, extra_group=()) -> pd.DataFrame:
    """Validation-only model selection: per group, keep the row with the highest val_acc
    (ties -> lower val_nll, then lower beta/lambda for determinism)."""
    keys = list(group) + list(extra_group)
    d = df.copy()
    d["_nll"] = d["val_nll"].fillna(np.inf)
    d["_tie"] = d["beta"].fillna(d["lambda"]).fillna(0.0)
    d = d.sort_values(["val_acc", "_nll", "_tie"], ascending=[False, True, True])
    return d.groupby(keys, as_index=False, dropna=False).first().drop(columns=["_nll", "_tie"])


def pivot_metric(df: pd.DataFrame, metric="test_acc", index=("label_per_class", "split_seed"),
                 columns="model") -> pd.DataFrame:
    return df.pivot_table(index=list(index), columns=columns, values=metric)


# -------------------------------------------------------------------------------- uncertainty
def bca_ci(d: np.ndarray, alpha=0.05, B=10000, seed=0):
    """Bias-corrected and accelerated bootstrap CI for the mean of paired differences.

    n is small (10, or 30 on the confirmatory cells), so coverage is approximate; these are
    reported as descriptive precision, not exact guarantees. Falls back to the percentile
    interval when the BCa correction is degenerate (e.g. all differences identical).
    """
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    theta = d.mean()
    if n < 2 or np.allclose(d, d[0]):
        return theta, theta, theta, "degenerate"
    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(B, n), replace=True).mean(axis=1)
    prop = float((boot < theta).mean())
    if prop <= 0 or prop >= 1:
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return theta, float(lo), float(hi), "percentile"
    z0 = stats.norm.ppf(prop)
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jbar = jack.mean()
    denom = 6.0 * ((jbar - jack) ** 2).sum() ** 1.5
    a = 0.0 if denom == 0 else ((jbar - jack) ** 3).sum() / denom
    out = []
    for z in (stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)):
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.percentile(boot, 100 * stats.norm.cdf(adj))))
    return theta, out[0], out[1], "bca"


def paired_summary(a: np.ndarray, b: np.ndarray, label="", alpha=0.05) -> dict:
    """Paired difference a - b with CI, sign count and (secondary) Wilcoxon p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    d = a[ok] - b[ok]
    mean, lo, hi, method = bca_ci(d)
    p = np.nan
    if len(d) >= 5 and not np.allclose(d, 0):
        try:
            p = float(stats.wilcoxon(d, alternative="two-sided").pvalue)
        except ValueError:
            p = np.nan
    return {"comparison": label, "n": int(len(d)), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "ci_method": method, "n_positive": int((d > 0).sum()), "p_wilcoxon": p,
            "diffs": d}


def holm(pvals: dict, alpha=0.05) -> dict:
    """Holm-Bonferroni over the pre-registered confirmatory family."""
    items = sorted(((k, v) for k, v in pvals.items() if not np.isnan(v)), key=lambda kv: kv[1])
    m = len(items)
    out, reject_further = {}, True
    for i, (k, p) in enumerate(items):
        thr = alpha / (m - i)
        rej = bool(p <= thr and reject_further)
        if not rej:
            reject_further = False
        out[k] = {"p": p, "threshold": thr, "reject_null": rej}
    for k, v in pvals.items():
        out.setdefault(k, {"p": v, "threshold": np.nan, "reject_null": False})
    return out


# ------------------------------------------------------------------------------------ helpers
def crossover(hs: np.ndarray, deltas: np.ndarray):
    """First h at which the paired delta crosses zero from below, by linear interpolation."""
    hs, deltas = np.asarray(hs, float), np.asarray(deltas, float)
    o = np.argsort(hs)
    hs, deltas = hs[o], deltas[o]
    for i in range(len(hs) - 1):
        if deltas[i] < 0 <= deltas[i + 1]:
            t = -deltas[i] / (deltas[i + 1] - deltas[i])
            return float(hs[i] + t * (hs[i + 1] - hs[i]))
    return float("nan")


def fmt_ci(row) -> str:
    return f"{row['mean_diff']:+.4f} [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]"

"""Selective prediction: can reliance / consistency flag answers that should be withheld?

Compared against the standard baseline, the model's own confidence. A score is 'good' when higher
score <=> more likely correct. AUROC and AURC get image-clustered 95% CIs, and every score gets a
paired difference vs. the confidence baseline (same questions, same bootstrap resamples' images),
which is what a claim like "PMI flags errors better than confidence" has to rest on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .stats import cluster_bootstrap

BASELINE = "confidence (baseline)"


def risk_coverage(correct: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the highest-scoring first; risk = error rate among answered at each coverage level."""
    order = np.argsort(-score, kind="stable")
    err = 1.0 - correct[order].astype(float)
    n = np.arange(1, len(err) + 1)
    return n / len(err), np.cumsum(err) / n


def aurc(correct: np.ndarray, score: np.ndarray) -> float:
    """Area under the risk-coverage curve; lower is better (random ~ overall error rate)."""
    _, risk = risk_coverage(correct, score)
    return float(risk.mean())


def _auroc(d: pd.DataFrame, col: str) -> float:
    c = d["c"].to_numpy(bool)
    return float(roc_auc_score(c, d[col].to_numpy(float))) if 0 < c.sum() < len(c) else float("nan")


def default_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Baseline confidence vs. our reliance / consistency signals (no learned parameters)."""
    s: dict[str, pd.Series] = {}
    if "conf_orig" in df:
        s[BASELINE] = df["conf_orig"]
    if "entropy_orig" in df:
        s["neg_entropy (baseline)"] = -df["entropy_orig"].astype(float)
    if "pmi_noimg" in df:
        s["image_pmi (noimg)"] = df["pmi_noimg"]
    if "pmi_shuf" in df:
        s["image_pmi (shuf)"] = df["pmi_shuf"]
    if "consistent" in df:
        s["consistent"] = df["consistent"].astype(float)
    if "consistent" in df and "reliant_noimg" in df:
        s["consistent & reliant"] = df["consistent"].astype(float) + df["reliant_noimg"].astype(float)
    return s


def evaluate_scores(df: pd.DataFrame, scores: dict[str, pd.Series], correct_col: str = "correct_orig",
                    cluster_col: str = "image_id", B: int = 0, seed: int = 0) -> pd.DataFrame:
    """One row per score: AUROC, AURC, and (B > 0) cluster-bootstrap CIs plus the paired AUROC
    difference vs. the confidence baseline on the questions where both scores exist."""
    rows = []
    base = scores.get(BASELINE)
    if cluster_col not in df:                      # no cluster ids: every row is its own cluster
        df = df.assign(**{cluster_col: np.arange(len(df))})
    for name, s in scores.items():
        d = pd.DataFrame({"c": df[correct_col].astype(float), "s": s, cluster_col: df[cluster_col]}).dropna()
        if len(d) < 10 or d["c"].nunique() < 2:
            continue
        c, v = d["c"].to_numpy(bool), d["s"].to_numpy(float)
        row = {"score": name, "n": len(d), "n_images": d[cluster_col].nunique(),
               "auroc": roc_auc_score(c, v), "aurc": aurc(c, v), "error_rate": 1 - c.mean()}
        if B:
            a = cluster_bootstrap(d, lambda x: _auroc(x, "s"), cluster_col, B, seed)
            r = cluster_bootstrap(d, lambda x: aurc(x["c"].to_numpy(bool), x["s"].to_numpy(float)), cluster_col, B, seed)
            row.update(auroc_lo=a["lo"], auroc_hi=a["hi"], aurc_lo=r["lo"], aurc_hi=r["hi"])
            if base is not None and name != BASELINE:
                p = pd.DataFrame({"c": df[correct_col].astype(float), "s": s, "b": base,
                                  cluster_col: df[cluster_col]}).dropna()
                if len(p) >= 10 and p["c"].nunique() == 2:
                    diff = cluster_bootstrap(p, lambda x: _auroc(x, "s") - _auroc(x, "b"), cluster_col, B, seed)
                    row.update(d_auroc_vs_conf=diff["est"], d_auroc_lo=diff["lo"], d_auroc_hi=diff["hi"],
                               d_auroc_p=diff["p"], d_auroc_n=len(p))
        rows.append(row)
    return pd.DataFrame(rows)

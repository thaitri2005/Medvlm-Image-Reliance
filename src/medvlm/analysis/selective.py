"""Selective prediction: can reliance / consistency flag answers that should be withheld?

Compared against the standard baseline of the model's own confidence. A score is 'good' when
higher score <=> more likely correct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


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


def evaluate_scores(df: pd.DataFrame, scores: dict[str, pd.Series], correct_col: str = "correct_orig") -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        d = pd.DataFrame({"c": df[correct_col], "s": s}).dropna()
        if len(d) < 10 or d["c"].nunique() < 2:
            continue
        c, v = d["c"].to_numpy(bool), d["s"].to_numpy(float)
        rows.append({"score": name, "n": len(d), "auroc": roc_auc_score(c, v), "aurc": aurc(c, v),
                     "error_rate": 1 - c.mean()})
    return pd.DataFrame(rows)


def default_scores(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Baseline confidence vs. our reliance / consistency signals (no learned parameters)."""
    s: dict[str, pd.Series] = {}
    if "conf_orig" in df:
        s["confidence (baseline)"] = df["conf_orig"]
    if "entropy_orig" in df:
        s["neg_entropy (baseline)"] = -df["entropy_orig"].astype(float)
    if "pmi_noimg" in df:
        s["image_pmi (noimg)"] = df["pmi_noimg"]
    if "pmi_shuf" in df:
        s["image_pmi (shuf)"] = df["pmi_shuf"]
    if "consistent" in df:
        s["consistent"] = df["consistent"].astype(float)
    if "consistent" in df and "reliant_noimg" in df:
        s["consistent & reliant"] = (df["consistent"].astype(float) + df["reliant_noimg"].astype(float))
    return s

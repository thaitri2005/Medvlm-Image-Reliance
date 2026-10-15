"""Cluster bootstrap. Questions on the same image are correlated (SLAKE test has 96 images for
1,061 questions), so confidence intervals resample IMAGES, not questions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scoring import normalize


def majority_prior(answers: pd.Series) -> float:
    """Accuracy of always giving the most common gold answer: the chance baseline a blind model
    can reach from answer frequencies alone (50-60% for yes/no)."""
    a = answers.map(normalize)
    return float(a.value_counts(normalize=True).iloc[0]) if len(a) else float("nan")


def cluster_mean_ci(df: pd.DataFrame, col: str, cluster_col: str = "image_id", B: int = 2000,
                    seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) of a numeric/boolean column, bootstrapping over clusters."""
    d = df[[col, cluster_col]].dropna()
    n = len(d)
    if n == 0:
        return (float("nan"),) * 3 + (0,)
    g = d.groupby(cluster_col)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    mean = s.sum() / c.sum()
    if len(g) < 2:
        return mean, float("nan"), float("nan"), n
    idx = np.random.default_rng(seed).integers(0, len(g), size=(B, len(g)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(mean), float(lo), float(hi), n


def summarize(df: pd.DataFrame, by: list[str], cols: list[str], cluster_col: str = "image_id",
              B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per group with mean [lo, hi] for each column."""
    out = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(g)
        row["n_images"] = g[cluster_col].nunique()
        if "answer" in g and "answer_type" in by:        # only meaningful within one answer type
            row["prior_acc"] = majority_prior(g["answer"])
        for c in cols:
            if c in g:
                m, lo, hi, k = cluster_mean_ci(g.assign(**{c: g[c].astype(float)}), c, cluster_col, B, seed)
                row[c], row[f"{c}_lo"], row[f"{c}_hi"], row[f"{c}_n"] = m, lo, hi, k   # k = samples with a value
        out.append(row)
    return pd.DataFrame(out)

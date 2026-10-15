"""Statistics, all clustered by image.

Questions on the same image are correlated (SLAKE test: 1,061 questions on 96 images), so every interval
and test resamples IMAGES, not questions (cluster / block bootstrap). Two-sided bootstrap p-values are
reported for the pre-registered primary tests and Holm-adjusted as one family; everything else is
descriptive (estimate + 95% CI) and labelled exploratory when a cell has fewer than MIN_IMAGES images.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..scoring import normalize

MIN_IMAGES = 30          # below this a cell is reported but flagged exploratory


def majority_prior(answers: pd.Series) -> float:
    """Accuracy of always giving the most common gold answer: the chance baseline a blind model
    can reach from answer frequencies alone (50-60% for yes/no)."""
    a = answers.map(normalize)
    return float(a.value_counts(normalize=True).iloc[0]) if len(a) else float("nan")


def cluster_mean_ci(df: pd.DataFrame, col: str, cluster_col: str = "image_id", B: int = 2000,
                    seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float, int]:
    """(mean, lo, hi, n) of a numeric/boolean column, bootstrapping over clusters (fast path)."""
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


def cluster_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], cluster_col: str = "image_id",
                      B: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Generic cluster bootstrap for any statistic (AUROC, AUROC difference, kappa, ...).

    Returns est, lo, hi (percentile CI) and p: the two-sided bootstrap p-value for H0: stat = 0
    (2 x the smaller tail mass at 0). Resamples where the statistic is undefined (e.g. one class only)
    are dropped and counted in `b_valid`."""
    est = stat(df)
    codes, uniq = pd.factorize(df[cluster_col])
    groups = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    out = {"est": est, "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "b_valid": 0,
           "n": len(df), "n_images": len(groups)}
    if len(groups) < 2 or not np.isfinite(est):
        return out
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(B):
        pick = rng.integers(0, len(groups), len(groups))
        v = stat(df.iloc[np.concatenate([groups[i] for i in pick])])
        if np.isfinite(v):
            boots.append(v)
    if len(boots) < B // 2:
        return out
    b = np.asarray(boots)
    out.update(lo=float(np.quantile(b, alpha / 2)), hi=float(np.quantile(b, 1 - alpha / 2)),
               p=float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))), b_valid=len(b))
    return out


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control), NaNs passed through."""
    p = np.asarray(pvals, float)
    ok = np.flatnonzero(np.isfinite(p))
    adj = np.full_like(p, np.nan)
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa for two binary/categorical ratings; NaN when undefined (no variation)."""
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) == 0:
        return float("nan")
    cats = sorted(set(d["a"]) | set(d["b"]))
    po = float((d["a"] == d["b"]).mean())
    pe = sum(float((d["a"] == c).mean()) * float((d["b"] == c).mean()) for c in cats)
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def summarize(df: pd.DataFrame, by: list[str], cols: list[str], cluster_col: str = "image_id",
              B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """One row per group: n, n_images, exploratory flag, prior, and mean [lo, hi] (+ n) per column.
    `blind_gain` = blind accuracy minus the majority-answer prior (prior is fixed within a group)."""
    out = []
    for key, g in df.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(g)
        row["n_images"] = g[cluster_col].nunique()
        row["exploratory"] = row["n_images"] < MIN_IMAGES
        if "answer" in g and "answer_type" in by:        # only meaningful within one answer type
            row["prior_acc"] = majority_prior(g["answer"])
        for c in cols:
            if c in g:
                m, lo, hi, k = cluster_mean_ci(g.assign(**{c: g[c].astype(float)}), c, cluster_col, B, seed)
                row[c], row[f"{c}_lo"], row[f"{c}_hi"], row[f"{c}_n"] = m, lo, hi, k
        if "prior_acc" in row and "correct_noimg" in row:
            for s in ("", "_lo", "_hi"):
                row[f"blind_gain{s}"] = row[f"correct_noimg{s}"] - row["prior_acc"]
        out.append(row)
    return pd.DataFrame(out)

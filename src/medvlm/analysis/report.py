"""`medvlm analyze`: turn a finished (or partial) run into the committed result tables.

Pre-registered primary endpoints (PLAN.md section 5) are computed with image-clustered bootstrap CIs and
two-sided p-values, Holm-adjusted as one family. Everything else is descriptive (estimate + 95% CI) and
cells with < 30 images are flagged `exploratory`.

Outputs in results/<model>/<profile>/:
  primary.csv          the pre-registered tests (the only p-values we interpret)
  by_dataset*.csv, by_modality.csv, by_qgroup.csv   descriptive tables with CIs and blind_gain
  selective.csv        AUROC/AURC with CIs + paired difference vs. confidence
  agreement.csv        do the two image probes (removed vs. wrong image) agree? Cohen's kappa
  sensitivity.csv      key numbers under alternative thresholds (F1 0.3/0.5/0.8, 1-3 paraphrases, strong flips)
  reference_check.csv  token-F1 vs. the MedGemma model card (pipeline sanity, not a finding)
  timing.csv, per_sample.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from ..schema import load_items
from .selective import BASELINE, _auroc, default_scores, evaluate_scores
from .stats import MIN_IMAGES, cluster_bootstrap, cohen_kappa, holm, majority_prior, summarize
from .table import QUADRANT, build_sample_table, load_results

MODEL_CARD_F1 = {"slake": 72.3, "vqa_rad": 49.9}   # MedGemma-4B model card, tokenized F1 (VQA-RAD: "balanced split")
METRICS = ["yn_mass_orig", "f1_orig", "correct_orig", "correct_noimg", "correct_shuf", "drop_noimg", "drop_shuf",
           "reliant_noimg", "reliant_shuf", "flip", "pmi_noimg", "pmi_shuf",
           "q_ideal", "q_fragile", "q_dangerous", "q_worst", "q_dangerous_shuf"]


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    if "quadrant" not in df:
        df["quadrant"] = None
    for q in ("ideal", "fragile", "dangerous", "worst"):
        df[f"q_{q}"] = (df["quadrant"] == q).astype(float).where(df["quadrant"].notna())
    if "quadrant_shuf" in df:
        df["q_dangerous_shuf"] = (df["quadrant_shuf"] == "dangerous").astype(float).where(df["quadrant_shuf"].notna())
    df["flip"] = 1 - df["consistent"]
    for c in ("noimg", "shuf"):
        if f"correct_{c}" in df:
            df[f"drop_{c}"] = df["correct_orig"].astype(float) - df[f"correct_{c}"].astype(float)
    return df


def primary_endpoints(df: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """P1: blind accuracy above the majority prior, yes/no, per dataset (H0: gain = 0).
    P2: AUROC(image PMI) - AUROC(confidence) for predicting correctness, yes/no pooled (H0: diff = 0)."""
    rows = []
    yn = df[df["answer_type"] == "yesno"]
    for ds, g in yn.groupby("dataset"):
        if "correct_noimg" not in g:
            continue
        prior = majority_prior(g["answer"])
        g = g.dropna(subset=["correct_noimg"])
        r = cluster_bootstrap(g, lambda x: float(x["correct_noimg"].astype(float).mean()) - prior, B=B, seed=seed)
        rows.append({"endpoint": "P1 blind accuracy - majority prior (yes/no)", "dataset": ds, **r, "prior": prior})
    if {"pmi_noimg", "conf_orig"} <= set(yn.columns):
        p = pd.DataFrame({"c": yn["correct_orig"].astype(float), "s": yn["pmi_noimg"], "b": yn["conf_orig"],
                          "image_id": yn["image_id"]}).dropna()
        if len(p) >= 10 and p["c"].nunique() == 2:
            r = cluster_bootstrap(p, lambda x: _auroc(x, "s") - _auroc(x, "b"), B=B, seed=seed)
            rows.append({"endpoint": "P2 AUROC(image PMI) - AUROC(confidence), yes/no", "dataset": "pooled", **r})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p"].tolist())
        out["significant_0.05"] = out["p_holm"] < 0.05
        out["exploratory"] = out["n_images"] < MIN_IMAGES
    return out


def agreement(df: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    """A5: removed-image vs wrong-image probes. Low kappa = 'reliance' depends on how you probe it."""
    rows = []
    if not {"reliant_noimg", "reliant_shuf"} <= set(df.columns):
        return pd.DataFrame()
    for (ds, at), g in df.groupby(["dataset", "answer_type"]):
        g = g.dropna(subset=["reliant_noimg", "reliant_shuf"])
        if len(g) < 10:
            continue
        r = cluster_bootstrap(g, lambda x: cohen_kappa(x["reliant_noimg"].astype(bool), x["reliant_shuf"].astype(bool)),
                              B=B, seed=seed)
        rows.append({"dataset": ds, "answer_type": at, "kappa": r["est"], "kappa_lo": r["lo"], "kappa_hi": r["hi"],
                     "raw_agreement": float((g["reliant_noimg"].astype(bool) == g["reliant_shuf"].astype(bool)).mean()),
                     "n": len(g), "n_images": r["n_images"]})
    return pd.DataFrame(rows)


def sensitivity(items, res, base_kw: dict) -> pd.DataFrame:
    """Point estimates of the headline numbers under alternative analysis choices. A finding that
    flips sign across this grid is reported as fragile."""
    rows = []
    for f1 in (0.3, 0.5, 0.8):
        for k in (1, 2, 3):
            df = build_sample_table(items, res, open_correct_f1=f1, same_f1=f1, min_paraphrases=k)
            if df.empty:
                continue
            if "reliant_noimg_strong" in df:
                ok = df["consistent"].notna() & df["reliant_noimg_strong"].notna()
                df["dangerous_strong"] = np.where(ok, (df["consistent"] == 1) & ~df["reliant_noimg_strong"].eq(True), np.nan)
            df = _derive(df)
            for (ds, at), g in df.groupby(["dataset", "answer_type"]):
                rows.append({"f1_threshold": f1, "min_paraphrases": k, "dataset": ds, "answer_type": at,
                             "is_default": f1 == base_kw["open_correct_f1"] and k == base_kw["min_paraphrases"],
                             "correct_orig": g["correct_orig"].astype(float).mean(),
                             "correct_noimg": g["correct_noimg"].astype(float).mean() if "correct_noimg" in g else np.nan,
                             "q_dangerous": g["q_dangerous"].mean(), "q_dangerous_n": int(g["q_dangerous"].notna().sum()),
                             "dangerous_strong": g["dangerous_strong"].mean() if "dangerous_strong" in g else np.nan})
    return pd.DataFrame(rows)


def reference_check(df: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    rows = []
    for ds, g in df.groupby("dataset"):
        r = cluster_bootstrap(g, lambda x: 100 * float(x["f1_orig"].mean()), B=B, seed=seed)
        rows.append({"dataset": ds, "token_f1_ours": r["est"], "lo": r["lo"], "hi": r["hi"], "n": len(g),
                     "model_card": MODEL_CARD_F1.get(ds, np.nan),
                     "note": "different prompt and split; a gap > ~15 points means a bug, not a finding"})
    return pd.DataFrame(rows)


def analyze(cfg: dict, paths: Paths) -> dict[str, pd.DataFrame]:
    profile, model = cfg["profile"], cfg["model"]["name"]
    items = load_items(paths.manifest_dir(profile) / "items.jsonl")
    res = load_results(paths.run_dir(model, profile))
    sc, an = cfg["scoring"], cfg["analysis"]
    kw = dict(open_correct_f1=sc["open_correct_f1"], same_f1=sc["same_answer_f1"], min_paraphrases=an["min_paraphrases"])
    df = build_sample_table(items, res, **kw)
    out_dir = paths.result_dir(model, profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("[analyze] no results yet")
        return {}
    df = _derive(df)
    B, seed = an["bootstrap_B"], cfg["seed"]
    skw = dict(cluster_col=an["cluster_col"], B=B, seed=seed)

    # Primary analysis set = official TEST splits. Other splits only enter descriptive tables that keep
    # `split` as a column, plus the pre-specified pooled modality table.
    test = df[df["split"] == "test"]
    tables = {}
    for name, by in {"by_dataset": ["dataset", "split"],
                     "by_dataset_type": ["dataset", "split", "answer_type"],
                     "by_modality": ["dataset", "split", "modality", "answer_type"],
                     "by_qgroup": ["dataset", "split", "qgroup", "answer_type"]}.items():
        tables[name] = summarize(df, by, METRICS, **skw)
    tables["by_modality_pooled"] = summarize(df, ["dataset", "modality", "answer_type"], METRICS, **skw)
    tables["selective"] = pd.concat(
        [evaluate_scores(g, default_scores(g), cluster_col=an["cluster_col"], B=an.get("selective_B", B), seed=seed)
         .assign(answer_type=k) for k, g in test.groupby("answer_type")], ignore_index=True)
    tables["primary"] = primary_endpoints(test, B, seed)
    tables["agreement"] = agreement(test, B, seed)
    tables["sensitivity"] = sensitivity(items, res, {"open_correct_f1": sc["open_correct_f1"],
                                                     "min_paraphrases": an["min_paraphrases"]})
    tables["reference_check"] = reference_check(test, B, seed)

    if "sec" in res:                       # profiling data written by the runner (seconds per job)
        cond = res["condition"].str.replace(r"para\d+", "para", regex=True)
        tables["timing"] = res.groupby(cond)["sec"].agg(["count", "mean", "median"]).round(3).reset_index()
    df.to_csv(out_dir / "per_sample.csv", index=False)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"[analyze] {len(df)} samples -> {out_dir}")
    if len(tables["primary"]):
        cols = ["endpoint", "dataset", "est", "lo", "hi", "p_holm", "n_images"]
        print("[analyze] primary endpoints:\n" + tables["primary"][cols].round(3).to_string(index=False))
    return {"per_sample": df, **tables}


__all__ = ["analyze", "QUADRANT", "BASELINE"]

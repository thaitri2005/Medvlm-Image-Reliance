"""`medvlm analyze`: turn a finished (or partial) run into the committed result tables.

Tables only; figures are produced in Phase 5 of PLAN.md once the numbers are stable.
"""
from __future__ import annotations

import pandas as pd

from ..config import Paths
from ..schema import load_items
from .selective import default_scores, evaluate_scores
from .stats import summarize
from .table import build_sample_table, load_results


def analyze(cfg: dict, paths: Paths) -> dict[str, pd.DataFrame]:
    profile, model = cfg["profile"], cfg["model"]["name"]
    items = load_items(paths.manifest_dir(profile) / "items.jsonl")
    res = load_results(paths.run_dir(model, profile))
    sc, an = cfg["scoring"], cfg["analysis"]
    df = build_sample_table(items, res, open_correct_f1=sc["open_correct_f1"], same_f1=sc["same_answer_f1"],
                             min_paraphrases=an["min_paraphrases"])
    out_dir = paths.result_dir(model, profile)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("[analyze] no results yet")
        return {}

    if "quadrant" not in df:
        df["quadrant"] = None
    kw = dict(cluster_col=an["cluster_col"], B=an["bootstrap_B"], seed=cfg["seed"])
    for q in ("ideal", "fragile", "dangerous", "worst"):
        df[f"q_{q}"] = (df["quadrant"] == q).astype(float).where(df["quadrant"].notna())
    df["flip"] = 1 - df["consistent"]
    for c in ("noimg", "shuf"):
        if f"correct_{c}" in df:
            df[f"drop_{c}"] = df["correct_orig"].astype(float) - df[f"correct_{c}"].astype(float)

    metric_cols = ["yn_mass_orig", "correct_orig", "correct_noimg", "correct_shuf", "drop_noimg", "drop_shuf",
                   "reliant_noimg", "reliant_shuf", "flip", "pmi_noimg", "pmi_shuf",
                   "q_ideal", "q_fragile", "q_dangerous", "q_worst"]
    tables = {}
    for name, by in {"by_dataset": ["dataset", "split"],
                     "by_dataset_type": ["dataset", "split", "answer_type"],
                     "by_modality": ["dataset", "split", "modality", "answer_type"],
                     "by_qgroup": ["dataset", "split", "qgroup", "answer_type"]}.items():
        tables[name] = summarize(df, by, metric_cols, **kw)
    tables["selective"] = pd.concat(
        [evaluate_scores(g, default_scores(g)).assign(answer_type=k) for k, g in df.groupby("answer_type")],
        ignore_index=True)

    if "sec" in res:                       # profiling data written by the runner (seconds per job)
        cond = res["condition"].str.replace(r"para\d+", "para", regex=True)
        tm = res.groupby(cond)["sec"].agg(["count", "mean", "median"]).round(3)
        tm.to_csv(out_dir / "timing.csv")
        print("[analyze] seconds/job by condition:\n" + tm.to_string())
    df.to_csv(out_dir / "per_sample.csv", index=False)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"[analyze] {len(df)} samples -> {out_dir}")
    return {"per_sample": df, **tables}

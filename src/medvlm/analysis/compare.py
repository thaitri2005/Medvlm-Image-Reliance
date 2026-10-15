"""`medvlm compare --model A --model-b B`: paired comparison of two models on the SAME questions.

Both models see identical manifests, conditions and paraphrases, so each question contributes a paired
difference (A - B). Means and 95% CIs come from the image-clustered bootstrap. The one pre-registered
test here (P3) is the difference in the Dangerous share on yes/no questions, pooled over datasets.
"""
from __future__ import annotations

import pandas as pd

from ..config import Paths
from ..schema import load_items
from .report import _derive
from .stats import MIN_IMAGES, cluster_bootstrap
from .table import build_sample_table, load_results

PAIRED = ["correct_orig", "correct_noimg", "reliant_noimg", "flip", "q_dangerous", "pmi_noimg"]


def compare(cfg: dict, paths: Paths, model_a: str, model_b: str) -> dict[str, pd.DataFrame]:
    profile = cfg["profile"]
    items = load_items(paths.manifest_dir(profile) / "items.jsonl")
    sc, an = cfg["scoring"], cfg["analysis"]
    kw = dict(open_correct_f1=sc["open_correct_f1"], same_f1=sc["same_answer_f1"], min_paraphrases=an["min_paraphrases"])
    tabs = {}
    for m in (model_a, model_b):
        res = load_results(paths.run_dir(m, profile))
        if res.empty:
            raise SystemExit(f"[compare] no results for {m}/{profile}")
        tabs[m] = _derive(build_sample_table(items, res, **kw)).set_index("sample_id")
    a, b = tabs[model_a], tabs[model_b]
    common = a.index.intersection(b.index)
    meta = a.loc[common, ["dataset", "answer_type", "image_id"]]
    B, seed = an["bootstrap_B"], cfg["seed"]

    rows = []
    for (ds, at), g in meta.groupby(["dataset", "answer_type"]):
        row = {"dataset": ds, "answer_type": at, "n": len(g), "n_images": g["image_id"].nunique()}
        row["exploratory"] = row["n_images"] < MIN_IMAGES
        for col in PAIRED:
            if col not in a or col not in b:
                continue
            d = pd.DataFrame({"d": a.loc[g.index, col].astype(float) - b.loc[g.index, col].astype(float),
                              "image_id": g["image_id"]}).dropna()
            if len(d) < 5:
                continue
            r = cluster_bootstrap(d, lambda x: float(x["d"].mean()), B=B, seed=seed)
            row.update({f"d_{col}": r["est"], f"d_{col}_lo": r["lo"], f"d_{col}_hi": r["hi"], f"d_{col}_n": len(d)})
        rows.append(row)
    by_cell = pd.DataFrame(rows)

    yn = meta[meta["answer_type"] == "yesno"].index
    d = pd.DataFrame({"d": a.loc[yn, "q_dangerous"] - b.loc[yn, "q_dangerous"],
                      "image_id": meta.loc[yn, "image_id"]}).dropna()
    r = cluster_bootstrap(d, lambda x: float(x["d"].mean()), B=B, seed=seed)
    primary = pd.DataFrame([{"endpoint": f"P3 Dangerous share {model_a} - {model_b} (yes/no, pooled)", **r}])

    out_dir = paths.results / f"compare_{model_a}_vs_{model_b}" / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cell.to_csv(out_dir / "paired_differences.csv", index=False)
    primary.to_csv(out_dir / "primary.csv", index=False)
    print(f"[compare] {len(common)} shared questions -> {out_dir}")
    print(primary[["endpoint", "est", "lo", "hi", "p", "n", "n_images"]].round(3).to_string(index=False))
    return {"paired_differences": by_cell, "primary": primary}

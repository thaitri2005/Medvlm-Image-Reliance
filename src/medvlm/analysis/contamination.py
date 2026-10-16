"""`medvlm contamination`: which evaluated questions could a model have seen in training?

Two parts, written to results/contamination/:
  overlap.csv        for every evaluated set: how many of its images also occur in that dataset's TRAIN split
                     (image identity = dataset image id; PathVQA ids are SHA-1 of the image bytes)
  training_use.csv   what each model's own documentation says about training on each dataset (checked by hand,
                     with source and date; 'unknown' when the training data is not published)

Image overlap matters because a model trained on a dataset's train split has seen those images, with other
questions, even if the evaluated question itself is new.
"""
from __future__ import annotations

import pandas as pd

from ..config import Paths
from ..data.sources import get_source
from ..schema import load_items

CHECKED = "2026-10-16"
MEDGEMMA_REPORT = "MedGemma technical report, arXiv 2507.05201 (Table 1)"
TRAINING_USE = [
    ("medgemma4b", "slake", "trained", "SLAKE (450) in vision, pre-training and RL stages; 450 = SLAKE train images", MEDGEMMA_REPORT),
    ("medgemma4b", "vqa_rad", "trained", "VQA-RAD (1,391 QA) in the RL stage", MEDGEMMA_REPORT),
    ("medgemma4b", "pathvqa", "not trained", "PathVQA removed from training for data-quality issues", MEDGEMMA_REPORT),
    ("medgemma15_4b", "slake", "trained", "listed among training sources", "model card google/medgemma-1.5-4b-it"),
    ("medgemma15_4b", "vqa_rad", "not stated", "listed under data ownership and citation, not under training sources",
     "model card google/medgemma-1.5-4b-it"),
    ("medgemma15_4b", "pathvqa", "not stated", "not mentioned", "model card google/medgemma-1.5-4b-it"),
    ("medvlthinker_3b", "slake", "not stated", "RL on PMC-VQA only", "model card UCSC-VLAA/MedVLThinker-3B-RL_PMC"),
    ("medvlthinker_3b", "vqa_rad", "not stated", "RL on PMC-VQA only", "model card UCSC-VLAA/MedVLThinker-3B-RL_PMC"),
    ("medvlthinker_3b", "pathvqa", "not stated", "RL on PMC-VQA only", "model card UCSC-VLAA/MedVLThinker-3B-RL_PMC"),
    ("gemma3_4b", "all", "unknown", "pre-training data not published", "model card google/gemma-3-4b-it"),
    ("qwen25vl_3b", "all", "unknown", "pre-training data not published", "model card Qwen/Qwen2.5-VL-3B-Instruct"),
]


def train_overlap(evaluated: pd.DataFrame, train_images: dict[str, set]) -> pd.DataFrame:
    """evaluated: rows with profile, dataset, split, image_id, answer_type. One row per (profile, dataset, split)."""
    rows = []
    for (prof, ds, sp), g in evaluated.groupby(["profile", "dataset", "split"]):
        tr = train_images.get(ds, set())
        yn = g[g.answer_type == "yesno"]
        rows.append({"profile": prof, "dataset": ds, "split": sp, "questions": len(g), "images": g.image_id.nunique(),
                     "images_in_train_split": len(set(g.image_id) & tr),
                     "yesno_questions_on_train_images": int(yn.image_id.isin(tr).sum()), "yesno_questions": len(yn)})
    return pd.DataFrame(rows)


def contamination(paths: Paths, profiles=("main", "contrast")) -> dict[str, pd.DataFrame]:
    evaluated = []
    for prof in profiles:
        f = paths.manifest_dir(prof) / "items.jsonl"
        if f.exists():
            evaluated += [{"profile": prof, "dataset": it.dataset, "split": it.split, "image_id": it.image_id,
                           "answer_type": it.answer_type} for it in load_items(f)]
    ev = pd.DataFrame(evaluated)
    train = {ds: {r["image_id"] for r in get_source(ds, paths).rows("train")} for ds in sorted(set(ev.dataset))}
    out = {"overlap": train_overlap(ev, train),
           "training_use": pd.DataFrame(TRAINING_USE, columns=["model", "dataset", "status", "detail", "source"])
           .assign(checked=CHECKED)}
    d = paths.results / "contamination"
    d.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(d / f"{name}.csv", index=False)
    print(out["overlap"].to_string(index=False))
    print(f"[contamination] -> {d}")
    return out

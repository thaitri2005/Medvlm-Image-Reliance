"""Long per-(sample, condition) results -> one wide row per sample, with the quantities every
analysis needs: correctness, consistency under paraphrase, binary + continuous image reliance,
quadrant, and confidence scores."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Item, read_jsonl
from ..scoring import is_correct, same_answer

# (consistent, image-reliant) -> quadrant, as defined in arXiv 2603.20985
QUADRANT = {(True, True): "ideal", (False, True): "fragile", (True, False): "dangerous", (False, False): "worst"}
IMG_CONDS = ("noimg", "shuf", "blank")


def load_results(run_dir: Path) -> pd.DataFrame:
    rows = [r for f in sorted(run_dir.glob("results*.jsonl")) for r in read_jsonl(f, tolerant=True)]
    if not rows:
        return pd.DataFrame(columns=["key", "sample_id", "condition", "text"])
    return pd.DataFrame(rows).drop_duplicates("key", keep="last")


def build_sample_table(items: list[Item], results: pd.DataFrame, *, open_correct_f1: float = 0.5,
                       same_f1: float = 0.5, reliance_cond: str = "noimg", min_paraphrases: int = 2) -> pd.DataFrame:
    """`consistent` uses the FIRST `min_paraphrases` valid paraphrases of every sample, so all samples
    face the same number of chances to flip; samples with fewer get NaN and leave the quadrant
    analysis (they are counted in `n` vs `consistent_n`). `flip_frac` uses all valid paraphrases."""
    by = {(r.sample_id, r.condition): r for r in results.itertuples(index=False)}
    recs = []
    for it in items:
        orig = by.get((it.sample_id, "orig"))
        if orig is None:
            continue
        kind = it.answer_type
        rec = {**{k: getattr(it, k) for k in ("sample_id", "dataset", "split", "image_id", "modality", "region",
                                                "qgroup", "answer_type", "answer", "question")},
               "pred_orig": orig.text, "correct_orig": is_correct(kind, orig.text, it.answer, open_correct_f1)}
        for c in IMG_CONDS:
            r = by.get((it.sample_id, c))
            if r is None:
                continue
            rec[f"correct_{c}"] = is_correct(kind, r.text, it.answer, open_correct_f1)
            rec[f"reliant_{c}"] = not same_answer(kind, orig.text, r.text, same_f1)
            if r.gold_logp is not None and orig.gold_logp is not None and not _nan(r.gold_logp):
                rec[f"pmi_{c}"] = orig.gold_logp - r.gold_logp      # log p(gold|img) - log p(gold|no/other img)
        paras = [by[(it.sample_id, f"para{k}")] for k in range(1, 20) if (it.sample_id, f"para{k}") in by]
        rec["n_para"] = len(paras)
        rec["consistent"] = np.nan
        if paras:
            flips = [not same_answer(kind, orig.text, p.text, same_f1) for p in paras]
            rec["flip_frac"] = float(np.mean(flips))
            if len(paras) >= min_paraphrases:
                rec["consistent"] = float(not any(flips[:min_paraphrases]))   # strict: all must agree
        # baseline confidence in the original answer
        if kind == "yesno" and orig.p_yes is not None:
            rec["conf_orig"] = max(orig.p_yes, 1 - orig.p_yes)
        elif orig.mean_logprob is not None and not _nan(orig.mean_logprob):
            rec["conf_orig"] = math.exp(orig.mean_logprob)
        rec["entropy_orig"] = orig.entropy
        rec["yn_mass_orig"] = getattr(orig, "yn_mass", None)    # format-compliance diagnostic (yes/no only)
        recs.append(rec)
    df = pd.DataFrame(recs)
    rc = f"reliant_{reliance_cond}"
    if len(df) and rc in df:
        ok = df["consistent"].notna() & df[rc].notna()
        df["quadrant"] = None
        df.loc[ok, "quadrant"] = [QUADRANT[(bool(c), bool(r))] for c, r in zip(df.loc[ok, "consistent"], df.loc[ok, rc])]
    return df


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))

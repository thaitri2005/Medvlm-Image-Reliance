from types import SimpleNamespace

import numpy as np
import pandas as pd

from medvlm.analysis.posthoc import p3_sensitivity, selective


def _fake(n=400, seed=0):
    """Random yes/no outputs with no relation between the image and the gold answer."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i}" for i in range(n)]
    gold = rng.choice(["yes", "no"], n)
    items = [SimpleNamespace(sample_id=s, answer=g) for s, g in zip(ids, gold)]
    samples = pd.DataFrame({"answer_type": "yesno", "split": "test", "image_id": [f"i{i // 2}" for i in range(n)],
                            "q_dangerous": rng.integers(0, 2, n).astype(float),
                            "q_dangerous_shuf": rng.integers(0, 2, n).astype(float)}, index=pd.Index(ids, name="sample_id"))
    raws = pd.DataFrame([{"sample_id": s, "condition": c, "p_yes": float(rng.uniform(0.01, 0.99)),
                          "yn_mass": float(rng.uniform(0.3, 1.0))} for s in ids for c in ("orig", "noimg", "shuf")])
    return items, samples, raws


def test_gold_based_scores_leak_the_label_and_label_free_pmi_does_not():
    items, samples, raws = _fake()
    sel = selective({"m": samples}, {"m": raws}, items, B=50, seed=1).set_index("score")
    assert sel.loc["log p(gold | image)  [leaks label]", "auroc"] == 1.0          # correct <=> p(gold) > .5
    assert sel.loc["image PMI, gold answer  [pre-registered P2, leaks label]", "auroc"] > 0.75   # leaks even here
    assert abs(sel.loc["image PMI, predicted answer  [label-free]", "auroc"] - 0.5) < 0.1  # pure noise -> chance


def test_p3_uses_the_test_split_only():
    items, a, raws = _fake(seed=1)
    _, b, _ = _fake(seed=2)
    a.loc[a.index[:100], "split"] = "validation"
    b.loc[b.index[:100], "split"] = "validation"
    out = p3_sensitivity({"A": a, "B": b}, {"A": raws, "B": raws}, "A", "B", B=50, seed=1)
    assert out.loc[0, "n"] == 300


def test_kappa_is_zero_for_independent_answers_even_when_raw_agreement_is_high():
    from medvlm.analysis.posthoc import _kappa

    rng = np.random.default_rng(0)
    o = pd.Series(rng.random(20000) < 0.8)          # mostly "yes" with the image
    n = pd.Series(rng.random(20000) < 0.8)          # independent default "yes" without it
    assert (o == n).mean() > 0.6                    # 68% "unchanged" -> would look like image-blind answers
    assert abs(_kappa(o, n)) < 0.03                 # but none of it is beyond chance


def test_removal_split_counts_forced_answers_on_refused_questions_and_partitions_the_set():
    from medvlm.analysis.posthoc import removal_split

    _, samples, raws = _fake(seed=3)
    out = removal_split({"m": samples}, {"m": raws}, B=50, seed=1).set_index("subset")
    assert out.loc["all", "n"] == 400
    assert out.loc["refuses (mass < 0.5)", "n"] + out.loc["answers (mass >= 0.5)", "n"] == 400
    # random outputs: about half of the answers "change" and none of it is beyond chance
    assert abs(out.loc["all", "changed_noimg"] - 0.5) < 0.1
    assert abs(out.loc["all", "kappa_orig_noimg"]) < 0.15

import numpy as np
import pandas as pd

from medvlm.analysis.selective import aurc, evaluate_scores, risk_coverage
from medvlm.analysis.stats import cluster_mean_ci
from medvlm.analysis.table import QUADRANT, build_sample_table


def test_quadrant_definitions_match_the_reference_paper():
    # arXiv 2603.20985: Ideal = consistent & image-reliant; Dangerous = consistent & NOT reliant
    assert QUADRANT[(True, True)] == "ideal"
    assert QUADRANT[(False, True)] == "fragile"
    assert QUADRANT[(True, False)] == "dangerous"
    assert QUADRANT[(False, False)] == "worst"


def test_cluster_ci_is_wider_than_naive_when_questions_on_one_image_agree():
    rng = np.random.default_rng(0)
    per_image = rng.integers(0, 2, 40)                          # outcome fixed per image...
    df = pd.DataFrame({"image_id": np.repeat(np.arange(40), 5), "x": np.repeat(per_image, 5)})
    m, lo, hi, n = cluster_mean_ci(df, "x", B=1000)
    assert n == 200 and lo <= m <= hi
    naive = df.assign(image_id=np.arange(200))                  # ...so treating 200 rows as independent
    _, nlo, nhi, _ = cluster_mean_ci(naive, "x", B=1000)       # is over-confident
    assert (hi - lo) > 1.5 * (nhi - nlo)


def test_cluster_ci_single_cluster_has_no_interval():
    m, lo, hi, n = cluster_mean_ci(pd.DataFrame({"image_id": [1, 1], "x": [0, 1]}), "x")
    assert m == 0.5 and np.isnan(lo) and np.isnan(hi)


def test_selective_prediction_perfect_score_beats_random():
    rng = np.random.default_rng(1)
    correct = rng.random(500) < 0.6
    perfect = correct.astype(float) + rng.random(500) * 0.01
    noise = rng.random(500)
    assert aurc(correct, perfect) < aurc(correct, noise)
    cov, risk = risk_coverage(correct, perfect)
    assert risk[int(0.55 * 500)] == 0.0 and cov[-1] == 1.0
    ev = evaluate_scores(pd.DataFrame({"correct_orig": correct}), {"perfect": pd.Series(perfect), "noise": pd.Series(noise)})
    assert ev.set_index("score").loc["perfect", "auroc"] > 0.99


def test_sample_table_flags_dangerous_when_answer_ignores_image(paths, cfg):
    from medvlm.paraphrase import STYLES  # noqa: F401
    from tests.conftest import make_items

    items = make_items(paths, n_images=1, per_image=1)          # one yes/no item, gold "yes"
    sid = items[0].sample_id
    rows = [dict(key=f"{sid}|{c}", sample_id=sid, condition=c, text=t, p_yes=0.9, entropy=0.1, mean_logprob=None, gold_logp=-0.1)
            for c, t in (("orig", "yes"), ("noimg", "yes"), ("shuf", "yes"), ("para1", "yes"), ("para2", "yes"))]
    df = build_sample_table(items, pd.DataFrame(rows))
    r = df.iloc[0]
    assert r["consistent"] and not r["reliant_noimg"] and r["quadrant"] == "dangerous"
    assert r["correct_orig"] and r["pmi_noimg"] == 0.0

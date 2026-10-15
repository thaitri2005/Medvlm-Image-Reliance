"""Tests of the statistical methods themselves (not just that the code runs)."""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from medvlm.analysis.selective import _auroc, evaluate_scores
from medvlm.analysis.stats import cluster_bootstrap, cohen_kappa, holm


def _clustered(n_img, per_img, icc_sd, mean, rng):
    """Binary outcomes whose probability varies by image (within-image correlation)."""
    p = np.clip(mean + rng.normal(0, icc_sd, n_img), 0.02, 0.98)
    return pd.DataFrame({"image_id": np.repeat(np.arange(n_img), per_img),
                         "x": (rng.random(n_img * per_img) < np.repeat(p, per_img)).astype(float)})


def test_kappa_matches_sklearn():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 300).astype(bool)
    b = np.where(rng.random(300) < 0.8, a, ~a)
    assert abs(cohen_kappa(pd.Series(a), pd.Series(b)) - cohen_kappa_score(a, b)) < 1e-12
    assert np.isnan(cohen_kappa(pd.Series([True] * 5), pd.Series([True] * 5)))       # undefined, not 1.0


def test_holm_known_values():
    adj = holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert np.isnan(holm([0.01, float("nan")])[1]) and holm([0.01, float("nan")])[0] == 0.01


def test_cluster_bootstrap_has_nominal_coverage_where_naive_intervals_do_not():
    """Simulation: 60 datasets with correlated questions; the true mean is 0.6."""
    rng = np.random.default_rng(1)
    hit_cluster = hit_naive = 0
    sims = 60
    for s in range(sims):
        df = _clustered(40, 8, 0.25, 0.6, rng)
        true = 0.6
        c = cluster_bootstrap(df, lambda d: float(d["x"].mean()), B=300, seed=s)
        n = cluster_bootstrap(df.assign(image_id=np.arange(len(df))), lambda d: float(d["x"].mean()), B=300, seed=s)
        hit_cluster += c["lo"] <= true <= c["hi"]
        hit_naive += n["lo"] <= true <= n["hi"]
    assert hit_cluster / sims >= 0.85             # ~0.95 expected; slack for 60 sims / clipping
    assert hit_naive / sims < hit_cluster / sims  # ignoring clustering under-covers


def test_bootstrap_p_value_behaves_under_null_and_alternative():
    rng = np.random.default_rng(2)
    null = pd.DataFrame({"image_id": np.repeat(np.arange(50), 4), "d": rng.normal(0, 1, 200)})
    alt = null.assign(d=null["d"] + 0.8)
    assert cluster_bootstrap(null, lambda x: float(x["d"].mean()), B=400)["p"] > 0.01
    assert cluster_bootstrap(alt, lambda x: float(x["d"].mean()), B=400)["p"] < 0.01


def test_auroc_difference_test():
    rng = np.random.default_rng(3)
    n = 400
    c = rng.random(n) < 0.6
    df = pd.DataFrame({"c": c.astype(float), "image_id": np.arange(n) // 4,
                       "good": c + rng.normal(0, 0.5, n), "noise": rng.random(n)})
    assert abs(_auroc(df.rename(columns={"good": "s"}), "s") - roc_auc_score(c, df["good"])) < 1e-12
    diff = cluster_bootstrap(df, lambda x: _auroc(x, "good") - _auroc(x, "noise"), B=300)
    same = cluster_bootstrap(df, lambda x: _auroc(x, "noise") - _auroc(x, "noise"), B=300)
    assert diff["est"] > 0.2 and diff["p"] < 0.01 and diff["lo"] > 0
    assert same["est"] == 0 and same["p"] == 1.0


def test_selective_table_reports_cis_and_paired_difference():
    rng = np.random.default_rng(4)
    n = 300
    c = rng.random(n) < 0.7
    df = pd.DataFrame({"correct_orig": c, "image_id": np.arange(n) // 3,
                       "conf": rng.random(n), "pmi": c + rng.normal(0, 0.4, n)})
    t = evaluate_scores(df, {"confidence (baseline)": df["conf"], "image_pmi (noimg)": df["pmi"]}, B=200).set_index("score")
    assert t.loc["image_pmi (noimg)", "auroc_lo"] <= t.loc["image_pmi (noimg)", "auroc"] <= t.loc["image_pmi (noimg)", "auroc_hi"]
    assert t.loc["image_pmi (noimg)", "d_auroc_vs_conf"] > 0 and t.loc["image_pmi (noimg)", "d_auroc_p"] < 0.05
    assert np.isnan(t.loc["confidence (baseline)"].get("d_auroc_vs_conf", np.nan))

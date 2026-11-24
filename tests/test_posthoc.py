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


def test_replication_summary_holm_across_pairs_and_decision_rule(tmp_path):
    from medvlm.analysis.posthoc import replication_summary

    def write(a, b, rows):
        d = tmp_path / f"posthoc_{a}_vs_{b}" / "main"
        d.mkdir(parents=True)
        pd.DataFrame([{"pair": f"{a} - {b}", "hypothesis": h, "predicted": pr, "est": e, "p": p, "sign_ok": ok}
                      for h, pr, e, p, ok in rows]).to_csv(d / "replication_tests.csv", index=False)

    write("m1", "g1", [("H9", "> 0", .4, .001, True), ("H10", "< 0", -.05, .004, True), ("H11", "< 0", -.1, .001, True)])
    write("m2", "g2", [("H9", "> 0", .3, .001, True), ("H10", "< 0", .02, .3, False), ("H11", "< 0", -.1, .02, True)])
    T = replication_summary(tmp_path, [("m1", "g1"), ("m2", "g2"), ("m3", "g3")])
    rep = T.groupby("pair").reversal_replicates.first()
    assert rep["m1 - g1"] and not rep["m2 - g2"]               # m2: image removal does not rank the other way
    assert (T.p_holm >= T.p).all() and len(T) == 6            # Holm over all six tests; missing pair skipped


def test_probe_disagreement_is_a_difference_of_differences_on_the_same_questions():
    from medvlm.analysis.posthoc import did_row

    rng = np.random.default_rng(0)
    n = 600
    img = [f"i{i // 3}" for i in range(n)]
    same = rng.choice([-1.0, 0.0, 1.0], n)
    agree = did_row(pd.DataFrame({"x": same, "y": same, "image_id": img}), B=200, seed=1)
    assert agree["did"] == 0 and agree["lo"] == 0 == agree["hi"] and not agree["reversal"]
    # probe x says "medical changes 20 points less", probe y says "10 points more": did = -30, and a reversal
    x = (rng.random(n) < .3).astype(float) - (rng.random(n) < .5).astype(float)
    y = (rng.random(n) < .5).astype(float) - (rng.random(n) < .4).astype(float)
    r = did_row(pd.DataFrame({"x": x, "y": y, "image_id": img}), B=200, seed=1)
    assert abs(r["did"] - (r["d_x"] - r["d_y"])) < 1e-12 and r["hi"] < 0 and r["n_images"] == 200 and r["reversal"]
    # same sign, different size: the probes report different differences but do not order the models oppositely
    z = (rng.random(n) < .1).astype(float) - (rng.random(n) < .5).astype(float)
    r = did_row(pd.DataFrame({"x": x, "y": z, "image_id": img}), B=200, seed=1)
    assert r["lo"] > 0 and not r["reversal"]


def test_removal_difference_survives_excluding_refusals_only_if_it_is_not_made_by_them(monkeypatch, tmp_path):
    """Two models that answer identically except that B's answer is noise wherever B refuses: the removal
    difference exists on all questions and vanishes once refused questions are excluded."""
    import medvlm.analysis.posthoc as ph

    rng = np.random.default_rng(0)
    n = 2000
    ids = pd.Index([f"s{i}" for i in range(n)], name="sample_id")
    samples = pd.DataFrame({"answer_type": "yesno", "split": "test", "dataset": "slake",
                            "image_id": [f"i{i // 4}" for i in range(n)]}, index=ids)
    orig = rng.random(n) < .6
    blind = np.where(rng.random(n) < .7, orig, ~orig)            # both models keep 70% of answers without the image
    refuse = rng.random(n) < .4                                  # B refuses on 40%; its forced answer there is noise
    blind_b = np.where(refuse, rng.random(n) < .5, blind)

    def raw(blind_answers, mass):
        return pd.DataFrame([{"sample_id": s, "condition": c, "p_yes": float(v), "yn_mass": float(m)}
                             for s, o, bl, mm in zip(ids, orig, blind_answers, mass)
                             for c, v, m in (("orig", o, 1.0), ("noimg", bl, mm))])

    raws = {"A": raw(blind, np.ones(n)), "B": raw(blind_b, np.where(refuse, 0.2, 1.0))}
    monkeypatch.setattr(ph, "_tables", lambda cfg, paths, models: (None, {"A": samples, "B": samples}, raws))
    out = ph.removal_without_refusals({}, None, [("A", "B")], B=200, seed=1)
    out = out[out.dataset == "all"].set_index("subset")
    assert out.loc["all questions", "hi"] < 0                                  # A changes less: B's refusals add noise
    assert out.loc["neither refuses (mass >= 0.5)", "d_removal"] == 0          # and nothing is left without them
    assert out.loc["neither refuses (mass >= 0.5)", "n"] == int((~refuse).sum())

"""`medvlm posthoc`: analyses added after the main run, each motivated by something the run showed.
Logged in PLAN.md section 13 (items 12-15). Nothing here replaces a pre-registered endpoint; it
qualifies them.

  abstention      without the image, models often answer "please provide the image" instead of yes/no;
                  rate per model and dataset (first-token yes+no mass < 0.5).
  selective       P2 used log p(GOLD answer), which leaks the label (log p(gold|image) alone has
                  AUROC 1.0 for yes/no). Label-free image PMI uses the model's own predicted answer.
  partner         how often the wrong-image partner has the same gold answer for the same question
                  (only where that pair exists in the data), and whether the model follows real changes.
  p3_sensitivity  P3 with abstaining questions removed, and with the wrong-image probe.
  chance          the quadrant metric counts "unchanged without the image" raw; on yes/no ~half of that is
                  chance. Chance agreement, the chance-only Dangerous floor, Cohen's kappa and paired d(kappa).
  no_pilot        key endpoints re-estimated without the 330 pilot questions the design was tuned on.
  harm            Dangerous split into correct and wrong answers; share of each model's errors that are Dangerous.
  probe_*         the robustness probes registered in PLAN.md section 15 (blank, abstain, opp) and their
                  Holm-adjusted tests H6-H8, when `medvlm probes` outputs exist.
  reversal        one measure (answer changes) under every probe, both models: the probes disagree.
  removal_split   answer changes without the image, split by whether the model would rather refuse.
  figures         fig1-fig7, twice: titled (results/, README) and paper-sized without titles (report/figures/).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from ..schema import load_items
from ..scoring import normalize
from .report import _derive, majority_prior
from .stats import cluster_bootstrap
from .table import build_sample_table, load_results

EPS = 1e-6
ABSTAIN_MASS = 0.5        # first-token P(yes)+P(no) below this: the model mostly says something else
ANSWER_MASS = 0.9         # the pre-registered format threshold
BLUE, ORANGE = "#2a78d6", "#eb6834"                                  # MedGemma, Gemma-3 (validated pair)
QUAD_COLORS = {"ideal": "#1baf7a", "fragile": "#6250d6", "dangerous": "#e34948", "worst": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DS_LABEL = {"pathvqa": "PathVQA", "slake": "SLAKE", "vqa_rad": "VQA-RAD"}


def _auroc(c: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(c, s)) if len(np.unique(c)) == 2 else float("nan")


def _ci(df: pd.DataFrame, fn, B: int, seed: int) -> dict:
    return cluster_bootstrap(df, fn, B=B, seed=seed)


def _tables(cfg: dict, paths: Paths, models: list[str]):
    items = load_items(paths.manifest_dir(cfg["profile"]) / "items.jsonl")
    sc, an = cfg["scoring"], cfg["analysis"]
    samples, raws = {}, {}
    for m in models:
        res = load_results(paths.run_dir(m, cfg["profile"]))
        if res.empty:
            raise SystemExit(f"[posthoc] no results for {m}")
        samples[m] = _derive(build_sample_table(items, res, open_correct_f1=sc["open_correct_f1"],
                                                same_f1=sc["same_answer_f1"],
                                                min_paraphrases=an["min_paraphrases"])).set_index("sample_id")
        raws[m] = res
    return items, samples, raws


def _wide(res: pd.DataFrame, col: str) -> pd.DataFrame:
    return res.pivot_table(index="sample_id", columns="condition", values=col, aggfunc="first")


def abstention(samples, raws, B, seed) -> pd.DataFrame:
    rows = []
    for m, t in samples.items():
        mass = _wide(raws[m], "yn_mass")
        yn = t[(t.answer_type == "yesno") & (t.split == "test")]
        for ds, g in yn.groupby("dataset"):
            d = pd.DataFrame({"a": (mass.loc[g.index, "noimg"] < ABSTAIN_MASS).astype(float), "image_id": g.image_id})
            r = _ci(d, lambda x: float(x.a.mean()), B, seed)
            rows.append({"model": m, "dataset": ds, "abstain_noimg": r["est"], "lo": r["lo"], "hi": r["hi"],
                         "n": len(d), "n_images": g.image_id.nunique(),
                         "mean_mass_noimg": float(mass.loc[g.index, "noimg"].mean()),
                         "mean_mass_orig": float(mass.loc[g.index, "orig"].mean())})
    return pd.DataFrame(rows)


def selective(samples, raws, items, B, seed) -> pd.DataFrame:
    """AUROC for predicting correct_orig on test yes/no: confidence vs gold-based (leaky) vs label-free PMI."""
    by_id = {it.sample_id: it for it in items}
    rows = []
    for m, t in samples.items():
        yn = t[(t.answer_type == "yesno") & (t.split == "test")]
        p = _wide(raws[m], "p_yes").loc[yn.index].clip(EPS, 1 - EPS)
        gold_yes = np.array([by_id[s].answer.strip().lower() == "yes" for s in yn.index])
        po, pn, ps = p["orig"].to_numpy(), p["noimg"].to_numpy(), p["shuf"].to_numpy()
        pred_yes = po >= 0.5

        def lp(prob, yes):                      # log p(answer) for a yes/no answer
            return np.where(yes, np.log(prob), np.log(1 - prob))

        d = pd.DataFrame({
            "c": (pred_yes == gold_yes).astype(float), "image_id": yn.image_id.to_numpy(),
            "confidence (baseline)": np.maximum(po, 1 - po),
            "log p(gold | image)  [leaks label]": lp(po, gold_yes),
            "image PMI, gold answer  [pre-registered P2, leaks label]": lp(po, gold_yes) - lp(pn, gold_yes),
            "image PMI, predicted answer  [label-free]": lp(po, pred_yes) - lp(pn, pred_yes),
            "wrong-image PMI, predicted answer  [label-free]": lp(po, pred_yes) - lp(ps, pred_yes)})
        base = "confidence (baseline)"
        for name in [c for c in d.columns if c not in ("c", "image_id")]:
            r = _ci(d, lambda x, n=name: _auroc(x.c.to_numpy(), x[n].to_numpy()), B, seed)
            row = {"model": m, "score": name, "auroc": r["est"], "lo": r["lo"], "hi": r["hi"], "n": len(d)}
            if name != base:
                dr = _ci(d, lambda x, n=name: _auroc(x.c.to_numpy(), x[n].to_numpy())
                         - _auroc(x.c.to_numpy(), x[base].to_numpy()), B, seed)
                row.update(d_vs_conf=dr["est"], d_lo=dr["lo"], d_hi=dr["hi"], d_p=dr["p"])
            rows.append(row)
    return pd.DataFrame(rows)


def partner(samples, raws, items) -> pd.DataFrame:
    """Wrong-image confound: is the partner image's gold answer (same question) the same as ours?"""
    gold = {(it.image_id, normalize(it.question)): it.answer.strip().lower() for it in items}
    img_of = {it.image_file: it.image_id for it in items}
    by_id = {it.sample_id: it for it in items}
    rows = []
    for m, t in samples.items():
        p = _wide(raws[m], "p_yes")
        for s in t[t.answer_type == "yesno"].index:
            it = by_id[s]
            pg = gold.get((img_of.get(it.shuffle_image_file), normalize(it.question)))
            if pg is None:
                continue
            rows.append({"model": m, "sample_id": s, "dataset": it.dataset, "modality": it.modality,
                         "partner_same_gold": pg == it.answer.strip().lower(),
                         "answer_changed": bool((p.at[s, "orig"] >= .5) != (p.at[s, "shuf"] >= .5))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    by_same = (df.groupby(["model", "partner_same_gold"]).agg(n=("answer_changed", "size"),
                                                              answer_changed=("answer_changed", "mean"))
               .reset_index().assign(breakdown="answer change by whether the partner's gold answer is the same"))
    by_mod = (df.groupby(["model", "modality"]).agg(n=("partner_same_gold", "size"),
                                                    partner_same_gold_rate=("partner_same_gold", "mean"))
              .reset_index().assign(breakdown="share of partners with the same gold answer, by modality"))
    return pd.concat([by_same, by_mod], ignore_index=True)


def p3_sensitivity(samples, raws, model_a, model_b, B, seed) -> pd.DataFrame:
    a, b = samples[model_a], samples[model_b]
    idx = a.index[(a.answer_type == "yesno") & (a.split == "test")]
    ma, mb = _wide(raws[model_a], "yn_mass")["noimg"], _wide(raws[model_b], "yn_mass")["noimg"]
    variants = {"pre-registered (image-removed probe)": (idx, "q_dangerous"),
                f"both models answer without image (mass >= {ANSWER_MASS})":
                    (idx[(ma.reindex(idx) >= ANSWER_MASS) & (mb.reindex(idx) >= ANSWER_MASS)], "q_dangerous"),
                f"neither model abstains (mass >= {ABSTAIN_MASS})":
                    (idx[(ma.reindex(idx) >= ABSTAIN_MASS) & (mb.reindex(idx) >= ABSTAIN_MASS)], "q_dangerous"),
                "wrong-image probe instead": (idx, "q_dangerous_shuf")}
    rows = []
    for name, (ix, col) in variants.items():
        d = pd.DataFrame({"d": a.loc[ix, col] - b.loc[ix, col], "image_id": a.loc[ix, "image_id"]}).dropna()
        r = _ci(d, lambda x: float(x.d.mean()), B, seed)
        rows.append({"variant": name, "est": r["est"], "lo": r["lo"], "hi": r["hi"], "p": r["p"], "n": len(d),
                     "n_images": d.image_id.nunique(),
                     f"dangerous_{model_a}": float(a.loc[d.index, col].mean()),
                     f"dangerous_{model_b}": float(b.loc[d.index, col].mean())})
    return pd.DataFrame(rows)


def p1_sensitivity(samples, raws, B, seed) -> pd.DataFrame:
    rows = []
    for m, t in samples.items():
        mass = _wide(raws[m], "yn_mass")["noimg"]
        yn = t[(t.answer_type == "yesno") & (t.split == "test")]
        for ds, g in yn.groupby("dataset"):
            prior = majority_prior(g["answer"])
            for name, gg in {"all (pre-registered)": g, "model answers without image": g[mass.reindex(g.index) >= ANSWER_MASS]}.items():
                gg = gg.dropna(subset=["correct_noimg"])
                if len(gg) < 10:
                    continue
                r = _ci(gg, lambda x: float(x.correct_noimg.astype(float).mean()) - prior, B, seed)
                rows.append({"model": m, "dataset": ds, "subset": name, "blind_minus_prior": r["est"], "lo": r["lo"],
                             "hi": r["hi"], "p": r["p"], "n": len(gg), "prior": prior})
    return pd.DataFrame(rows)


def _kappa(o: pd.Series, n: pd.Series) -> float:
    o, n = o.astype(bool), n.astype(bool)
    po, pn = o.mean(), n.mean()
    exp = po * pn + (1 - po) * (1 - pn)
    return float(((o == n).mean() - exp) / (1 - exp)) if exp < 1 else float("nan")


def chance(samples, raws, model_a, model_b, B, seed) -> pd.DataFrame:
    """On the P3 set: how much of 'unchanged without the image' (and so of Dangerous) is chance agreement."""
    a, b = samples[model_a], samples[model_b]
    idx = a.index[(a.answer_type == "yesno") & (a.split == "test")]
    idx = idx[a.loc[idx, "consistent"].notna() & b.loc[idx, "consistent"].notna()]
    cols = {}
    for tag, m in (("a", model_a), ("b", model_b)):
        w = _wide(raws[m], "p_yes").loc[idx]
        cols.update({f"o{tag}": w["orig"] >= .5, f"n{tag}": w["noimg"] >= .5,
                     f"c{tag}": samples[m].loc[idx, "consistent"].astype(float),
                     f"d{tag}": samples[m].loc[idx, "q_dangerous"].astype(float)})
    D = pd.DataFrame({**cols, "image_id": a.loc[idx, "image_id"]})
    rows = []
    for tag, m in (("a", model_a), ("b", model_b)):
        o, n = D[f"o{tag}"], D[f"n{tag}"]
        exp = o.mean() * n.mean() + (1 - o.mean()) * (1 - n.mean())
        k = _ci(D, lambda x, t=tag: _kappa(x[f"o{t}"], x[f"n{t}"]), B, seed)
        rows.append({"model": m, "n": len(D), "consistent": D[f"c{tag}"].mean(), "unchanged_noimg": float((o == n).mean()),
                     "unchanged_by_chance": float(exp), "dangerous": D[f"d{tag}"].mean(),
                     "dangerous_chance_floor": float(D[f"c{tag}"].mean() * exp),
                     "kappa_orig_noimg": k["est"], "kappa_lo": k["lo"], "kappa_hi": k["hi"]})
    dk = _ci(D, lambda x: _kappa(x.oa, x.na) - _kappa(x.ob, x.nb), B, seed)
    rows.append({"model": f"{model_a} - {model_b}", "n": len(D), "kappa_orig_noimg": dk["est"], "kappa_lo": dk["lo"],
                 "kappa_hi": dk["hi"], "p": dk["p"]})
    return pd.DataFrame(rows)


def no_pilot(cfg, paths, samples, raws, items, model_a, model_b, B, seed) -> pd.DataFrame:
    from .report import primary_endpoints

    pfile = paths.manifest_dir("pilot") / "items.jsonl"
    if not pfile.exists():
        return pd.DataFrame()
    pilot = {it.sample_id for it in load_items(pfile)}
    S = {m: t[~t.index.isin(pilot)] for m, t in samples.items()}
    R = {m: r[~r.sample_id.isin(pilot)] for m, r in raws.items()}
    out = []
    for m in S:
        p = primary_endpoints(S[m][S[m].split == "test"].reset_index(), B, seed)
        out.append(p.assign(model=m)[["model", "endpoint", "dataset", "est", "lo", "hi", "p_holm", "n"]])
    p3 = p3_sensitivity(S, R, model_a, model_b, B, seed)
    out.append(p3.rename(columns={"variant": "dataset"}).assign(model=f"{model_a} - {model_b}", endpoint="P3")
               [["model", "endpoint", "dataset", "est", "lo", "hi", "n"]])
    sel = selective(S, R, [i for i in items if i.sample_id not in pilot], B, seed)
    out.append(sel.rename(columns={"score": "dataset", "auroc": "est"}).assign(endpoint="AUROC")
               [["model", "endpoint", "dataset", "est", "lo", "hi", "n"]])
    return pd.concat(out, ignore_index=True)


def harm(samples, model_a, model_b, B, seed) -> pd.DataFrame:
    """Is Dangerous dangerous? Split Dangerous answers into correct and wrong (the harmful cell), and give the
    share of each model's errors that are Dangerous (stable and unchanged without the image: hard to catch)."""
    a, b = samples[model_a], samples[model_b]
    idx = a.index[(a.answer_type == "yesno") & (a.split == "test") & a.quadrant.notna()]
    idx = idx[b.loc[idx, "quadrant"].notna()]
    D = pd.DataFrame({"image_id": a.loc[idx, "image_id"]})
    for tag, t in (("a", a.loc[idx]), ("b", b.loc[idx])):
        dang, ok = t.quadrant == "dangerous", t.correct_orig.astype(float) == 1
        D[f"ok{tag}"], D[f"bad{tag}"], D[f"err{tag}"] = (dang & ok).astype(float), (dang & ~ok).astype(float), (~ok).astype(float)

    def share(x, t):
        return float(x[f"bad{t}"].sum() / max(x[f"err{t}"].sum(), 1))

    rows = []
    for tag, m in (("a", model_a), ("b", model_b)):
        r = {k: _ci(D, f, B, seed) for k, f in {"dangerous_correct": lambda x, t=tag: float(x[f"ok{t}"].mean()),
                                                  "dangerous_wrong": lambda x, t=tag: float(x[f"bad{t}"].mean()),
                                                  "errors_that_are_dangerous": lambda x, t=tag: share(x, t),
                                                  "error_rate": lambda x, t=tag: float(x[f"err{t}"].mean())}.items()}
        for k, v in r.items():
            rows.append({"model": m, "measure": k, "est": v["est"], "lo": v["lo"], "hi": v["hi"], "n": len(D)})
    for k, f in {"dangerous_correct": lambda x: float((x.oka - x.okb).mean()),
                 "dangerous_wrong": lambda x: float((x.bada - x.badb).mean()),
                 "errors_that_are_dangerous": lambda x: share(x, "a") - share(x, "b"),
                 "error_rate": lambda x: float((x.erra - x.errb).mean())}.items():
        v = _ci(D, f, B, seed)
        rows.append({"model": f"{model_a} - {model_b}", "measure": k, "est": v["est"], "lo": v["lo"], "hi": v["hi"],
                     "p": v["p"], "n": len(D)})
    return pd.DataFrame(rows)


def probe_tables(cfg, paths, samples, raws, items, model_a, model_b, B, seed) -> dict[str, pd.DataFrame]:
    """The registered robustness probes (PLAN.md section 15), if their outputs exist."""
    from ..probes import opposite_partners
    from .stats import holm

    def load(m, p):
        return load_results(paths.run_dir(f"{m}__{p}", cfg["profile"]))

    have = {p: all(not load(m, p).empty for m in (model_a, model_b)) for p in ("blank", "abstain", "opp")}
    out, tests = {}, []
    a = samples[model_a]
    test_yn = a.index[(a.answer_type == "yesno") & (a.split == "test")]

    if have["blank"]:                                  # R1 / H6
        cols = {}
        for tag, m in (("a", model_a), ("b", model_b)):
            pb = _wide(load(m, "blank"), "p_yes")["blank"]
            cols.update({f"o{tag}": _wide(raws[m], "p_yes")["orig"].reindex(test_yn) >= .5,
                         f"k{tag}": pb.reindex(test_yn) >= .5,
                         f"c{tag}": samples[m].loc[test_yn, "consistent"],
                         f"m{tag}": _wide(load(m, "blank"), "yn_mass")["blank"].reindex(test_yn)})
        D = pd.DataFrame({**cols, "image_id": a.loc[test_yn, "image_id"]}).dropna(subset=["ma", "mb"])
        both = D.ca.notna() & D.cb.notna()
        rows = []
        for tag, m in (("a", model_a), ("b", model_b)):
            o, k = D[f"o{tag}"], D[f"k{tag}"]
            kap = _ci(D, lambda x, t=tag: _kappa(x[f"o{t}"], x[f"k{t}"]), B, seed)
            dang = ((D.loc[both, f"c{tag}"] == 1) & (o[both] == k[both])).astype(float)
            exp = o.mean() * k.mean() + (1 - o.mean()) * (1 - k.mean())
            rows.append({"model": m, "n": len(D), "unchanged_blank": float((o == k).mean()), "unchanged_by_chance": float(exp),
                         "kappa_orig_blank": kap["est"], "kappa_lo": kap["lo"], "kappa_hi": kap["hi"],
                         "dangerous_blank": float(dang.mean()), "abstain_blank": float((D[f"m{tag}"] < ABSTAIN_MASS).mean())})
        dk = _ci(D, lambda x: _kappa(x.oa, x.ka) - _kappa(x.ob, x.kb), B, seed)
        Db = D[both].assign(da=lambda x: ((x.ca == 1) & (x.oa == x.ka)).astype(float),
                            db=lambda x: ((x.cb == 1) & (x.ob == x.kb)).astype(float))
        dd = _ci(Db, lambda x: float((x.da - x.db).mean()), B, seed)
        rows.append({"model": f"{model_a} - {model_b}", "n": len(D), "kappa_orig_blank": dk["est"], "kappa_lo": dk["lo"],
                     "kappa_hi": dk["hi"], "p": dk["p"], "dangerous_blank": dd["est"], "dangerous_lo": dd["lo"],
                     "dangerous_hi": dd["hi"], "dangerous_p": dd["p"], "dangerous_n": len(Db)})
        out["probe_blank"] = pd.DataFrame(rows)
        tests.append(("H6: kappa(orig, blank), MedGemma - Gemma-3 (> 0 predicted)", dk, len(D)))

    if have["abstain"]:                                # R2 / H7
        rows, cols = [], {}
        for tag, m in (("a", model_a), ("b", model_b)):
            cols[f"x{tag}"] = (_wide(load(m, "abstain"), "yn_mass")["noimg"].reindex(test_yn) < ABSTAIN_MASS).astype(float)
            cols[f"f{tag}"] = (_wide(raws[m], "yn_mass")["noimg"].reindex(test_yn) < ABSTAIN_MASS).astype(float)
        D = pd.DataFrame({**cols, "image_id": a.loc[test_yn, "image_id"], "dataset": a.loc[test_yn, "dataset"]})
        for tag, m in (("a", model_a), ("b", model_b)):
            for ds, g in [("all", D)] + list(D.groupby("dataset")):
                r = _ci(g, lambda x, t=tag: float(x[f"x{t}"].mean()), B, seed)
                rows.append({"model": m, "dataset": ds, "abstain_allowed": r["est"], "lo": r["lo"], "hi": r["hi"],
                             "abstain_forced_main_run": float(g[f"f{tag}"].mean()), "n": len(g)})
        d = _ci(D, lambda x: float((x.xa - x.xb).mean()), B, seed)
        rows.append({"model": f"{model_a} - {model_b}", "dataset": "all", "abstain_allowed": d["est"], "lo": d["lo"],
                     "hi": d["hi"], "p": d["p"], "n": len(D)})
        out["probe_abstain"] = pd.DataFrame(rows)
        tests.append(("H7: abstention with 'unknown' allowed, MedGemma - Gemma-3 (< 0 predicted)", d, len(D)))

    if have["opp"]:                                    # R3 / H8
        partners = opposite_partners(items)
        by_id = {it.sample_id: it for it in items}
        cols = {}
        for tag, m in (("a", model_a), ("b", model_b)):
            po = _wide(raws[m], "p_yes")["orig"]
            pp = _wide(load(m, "opp"), "p_yes")["opp"].dropna()
            cols[f"o{tag}"], cols[f"p{tag}"] = po.reindex(pp.index) >= .5, pp >= .5
        idx = cols["pa"].index.intersection(cols["pb"].index)
        D = pd.DataFrame({k: v.reindex(idx) for k, v in cols.items()})
        D["image_id"] = [by_id[s].image_id for s in idx]
        D["modality"] = [by_id[s].modality for s in idx]
        D["partner_yes"] = [partners[s].answer.strip().lower() == "yes" for s in idx]
        D["dataset"] = [by_id[s].dataset for s in idx]
        D_all, D = D, D[D.dataset == "slake"]          # H8 was registered on SLAKE (test+val); others exploratory
        rows = []
        for tag, m in (("a", model_a), ("b", model_b)):
            for mod, g in [("all", D)] + list(D.groupby("modality")) + [("all datasets (exploratory)", D_all)]:
                f = _ci(g, lambda x, t=tag: float((x[f"o{t}"] != x[f"p{t}"]).mean()), B, seed)
                acc = float((g[f"p{tag}"] == g.partner_yes).mean())
                rows.append({"model": m, "modality": mod, "follow_rate": f["est"], "lo": f["lo"], "hi": f["hi"],
                             "accuracy_on_partner_image": acc, "n": len(g), "n_images": g.image_id.nunique()})
        d = _ci(D, lambda x: float(((x.oa != x.pa).astype(float) - (x.ob != x.pb).astype(float)).mean()), B, seed)
        rows.append({"model": f"{model_a} - {model_b}", "modality": "all", "follow_rate": d["est"], "lo": d["lo"],
                     "hi": d["hi"], "p": d["p"], "n": len(D)})
        out["probe_opp"] = pd.DataFrame(rows)
        tests.append(("H8: follow rate on opposite-answer images, MedGemma - Gemma-3 (> 0 predicted)", d, len(D)))

    if tests:
        ph = holm([t[1]["p"] for t in tests])
        out["probe_tests"] = pd.DataFrame([{"hypothesis": h, "est": r["est"], "lo": r["lo"], "hi": r["hi"], "p": r["p"],
                                            "p_holm": p, "n": n} for (h, r, n), p in zip(tests, ph)])
    return out


def removal_split(samples, raws, B, seed) -> pd.DataFrame:
    """Why image removal misranks: split 'answer changes without the image' (test yes/no) by whether the model
    would rather refuse (first-token yes/no mass < 0.5). Forced answers on refused questions are unrelated to the
    with-image answer (kappa ~ 0), so they count as 'image-reliant' about half the time."""
    rows = []
    for m, t in samples.items():
        idx = t.index[(t.answer_type == "yesno") & (t.split == "test")]
        p, mass = _wide(raws[m], "p_yes").loc[idx], _wide(raws[m], "yn_mass").loc[idx, "noimg"]
        D = pd.DataFrame({"o": p["orig"] >= .5, "n": p["noimg"] >= .5, "image_id": t.loc[idx, "image_id"]})
        for name, sel in (("all", mass.notna()), ("refuses (mass < 0.5)", mass < ABSTAIN_MASS),
                          ("answers (mass >= 0.5)", mass >= ABSTAIN_MASS)):
            g = D[sel]
            ch = _ci(g, lambda x: float((x.o != x.n).mean()), B, seed)
            k = _ci(g, lambda x: _kappa(x.o, x.n), B, seed)
            rows.append({"model": m, "subset": name, "n": len(g), "changed_noimg": ch["est"], "lo": ch["lo"],
                         "hi": ch["hi"], "kappa_orig_noimg": k["est"], "kappa_lo": k["lo"], "kappa_hi": k["hi"],
                         "yes_rate_noimg": float(g.n.mean())})
    return pd.DataFrame(rows)


REVERSAL_PROBES = (("noimg", "Image removed"), ("shuf", "Wrong image"), ("blank", "Black image"),
                   ("opp", "Opposite-answer image"))


def reversal(cfg, paths, samples, raws, items, model_a, model_b, B, seed) -> pd.DataFrame:
    """One measure across every probe: the share of yes/no answers that change when the image is altered.
    Test yes/no for image removed / wrong / black; the registered SLAKE set of probe R3 for opposite-answer images.
    Descriptive (the confirmatory tests are P3 and H6-H8); shows that the probes rank the two models differently."""
    a = samples[model_a]
    test_yn = a.index[(a.answer_type == "yesno") & (a.split == "test")]
    by_id = {it.sample_id: it for it in items}

    def p_yes(m, probe):
        if probe in ("noimg", "shuf"):
            return _wide(raws[m], "p_yes")[probe]
        res = load_results(paths.run_dir(f"{m}__{probe}", cfg["profile"]))
        return None if res.empty else _wide(res, "p_yes")[probe]

    rows = []
    for probe, label in REVERSAL_PROBES:
        pp = {m: p_yes(m, probe) for m in (model_a, model_b)}
        if any(v is None for v in pp.values()):
            continue
        idx = test_yn if probe != "opp" else pd.Index([s for s in pp[model_a].dropna().index
                                                        if s in pp[model_b].dropna().index and by_id[s].dataset == "slake"])
        cols = {}
        for tag, m in (("a", model_a), ("b", model_b)):
            po, pr = _wide(raws[m], "p_yes")["orig"].reindex(idx), pp[m].reindex(idx)
            cols[tag] = ((po >= .5) != (pr >= .5)).astype(float).where(po.notna() & pr.notna())
        D = pd.DataFrame({**cols, "image_id": [by_id[s].image_id for s in idx]}, index=idx).dropna()
        for tag, m in (("a", model_a), ("b", model_b)):
            r = _ci(D, lambda x, t=tag: float(x[t].mean()), B, seed)
            rows.append({"probe": probe, "label": label, "model": m, "changed": r["est"], "lo": r["lo"], "hi": r["hi"],
                         "n": len(D), "n_images": D.image_id.nunique()})
        d = _ci(D, lambda x: float((x.a - x.b).mean()), B, seed)
        rows.append({"probe": probe, "label": label, "model": f"{model_a} - {model_b}", "changed": d["est"],
                     "lo": d["lo"], "hi": d["hi"], "p": d["p"], "n": len(D), "n_images": D.image_id.nunique()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
# Each figure is drawn twice: "web" (titled, for README/RESULTS/poster) and "paper" (no title, sized for the
# CVPR column: 3.25 in single column, 6.9 in full width; the caption lives in LaTeX).
LABEL = {"medgemma4b": "MedGemma-4B", "gemma3_4b": "Gemma-3-4B"}


def _style(ax, title: str | None = None, fs: float = 9):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=fs)
    if title:
        ax.set_title(title, loc="left", fontsize=fs + 1.5, color=INK, pad=6)


def _bars(ax, groups, series, color, label, fs, pct=True, fmt="{:.0%}", ylim=None, hline=None):
    """Grouped bars with CIs: series = {model: (est[], lo[], hi[])}; value labels above the CI."""
    import matplotlib

    n = len(series)
    w = 0.8 / n
    for k, (m, (est, lo, hi)) in enumerate(series.items()):
        x = np.arange(len(groups)) + (k - (n - 1) / 2) * w
        ax.bar(x, est, width=w * 0.92, color=color[m], label=label[m], linewidth=0)
        ax.errorbar(x, est, yerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)], fmt="none",
                    ecolor=INK2, elinewidth=0.9)
        for xi, v, h in zip(x, est, hi):
            ax.text(xi, h + (0.012 if pct else 0.01), fmt.format(v), ha="center", va="bottom", fontsize=fs - 1.2, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    if pct:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    if ylim:
        ax.set_ylim(*ylim)
    if hline is not None:
        ax.axhline(hline, color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figures(out_dir, samples, models, tabs, results_dir, profile, paper_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    color = {models[0]: BLUE, models[1]: ORANGE}
    written = []
    for mode, target in (("web", out_dir), ("paper", paper_dir)):
        if target is None:
            continue
        target.mkdir(parents=True, exist_ok=True)
        paper = mode == "paper"
        fs = 7.2 if paper else 9
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                             "font.size": fs})
        col, full = 3.25, 6.9

        def save(fig, name, title=None):
            if title and not paper:
                # place the title above everything already drawn (panel titles, top tick labels)
                fig.canvas.draw()
                bb = fig.get_tightbbox(fig.canvas.get_renderer())
                fig.suptitle(title, x=bb.x0 / fig.get_figwidth(), y=bb.y1 / fig.get_figheight() + 0.015,
                             ha="left", va="bottom", fontsize=fs + 3, color=INK)
            fig.savefig(target / name, dpi=300 if paper else 200, bbox_inches="tight", pad_inches=0.03)
            plt.close(fig)
            written.append(target / name)

        # 1: quadrant composition (full width)
        fig, axes = plt.subplots(1, 2, figsize=(full, 2.35) if paper else (11, 3.9), sharey=True)
        ylabels = []
        for ax, (probe, c) in zip(axes, [("image removed", "quadrant"), ("wrong image", "quadrant_shuf")]):
            ylabels, y = [], 0
            for ds in ("pathvqa", "slake", "vqa_rad"):
                for m in models:
                    t = samples[m]
                    share = t[(t.answer_type == "yesno") & (t.split == "test") & (t.dataset == ds)][c].dropna() \
                        .value_counts(normalize=True)
                    left = 0.0
                    for q in ("ideal", "fragile", "dangerous", "worst"):
                        w = float(share.get(q, 0.0))
                        ax.barh(y, max(w - 0.004, 0), left=left, height=0.72, color=QUAD_COLORS[q], linewidth=0)
                        if w >= 0.08:
                            ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=fs - 1,
                                    color="white" if q in ("fragile", "dangerous") else INK)
                        left += w
                    ylabels.append(f"{DS_LABEL[ds]} · {LABEL[m]}")
                    y += 1
                y += 0.4
            ax.set_yticks([0, 1, 2.4, 3.4, 4.8, 5.8])
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            _style(ax, f"Probe: {probe}", fs)
        axes[0].set_yticklabels(ylabels)
        axes[0].invert_yaxis()
        axes[1].tick_params(axis="y", labelleft=False)
        fig.legend([plt.Rectangle((0, 0), 1, 1, color=QUAD_COLORS[q]) for q in QUAD_COLORS],
                   ["Ideal (stable, image-reliant)", "Fragile (flips, reliant)", "Dangerous (stable, not reliant)",
                    "Worst (flips, not reliant)"], loc="upper center", ncol=4, frameon=False, fontsize=fs,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout()
        save(fig, "fig1_quadrants.png", "Quadrant composition of yes/no answers (test split)")

        # 2: Dangerous share by SLAKE modality, both probes (single column in the paper)
        fig, axes = plt.subplots(1, 2, figsize=(col, 1.85) if paper else (9.5, 3.4), sharey=True)
        mods = ["ct", "mri", "xray"]
        for ax, (probe, c) in zip(axes, [("image removed", "q_dangerous"), ("wrong image", "q_dangerous_shuf")]):
            for k, m in enumerate(models):
                tb = pd.read_csv(results_dir / m / profile / "by_modality_pooled.csv")
                tb = tb[(tb.dataset == "slake") & (tb.answer_type == "yesno")].set_index("modality")
                x = np.arange(len(mods)) + (k - 0.5) * 0.26
                est, lo, hi = (tb.loc[mods, s].to_numpy() for s in (c, f"{c}_lo", f"{c}_hi"))
                ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", ms=4.5 if paper else 7, color=color[m],
                            ecolor=color[m], elinewidth=1.4 if paper else 2, capsize=0, mec=SURFACE, mew=1)
            ax.set_xticks(range(len(mods)), ["CT", "MRI", "X-ray"])
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            _style(ax, probe, fs)
        axes[0].set_ylabel("Dangerous share", color=INK2, fontsize=fs)
        fig.legend([Line2D([], [], marker="o", ls="", ms=4.5 if paper else 7, color=color[m]) for m in models],
                   [LABEL[m] for m in models], frameon=False, fontsize=fs - 0.5, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.55, 0.0))
        fig.tight_layout()
        save(fig, "fig2_modality.png", "Dangerous share by modality (SLAKE yes/no, test+val, 95% CI by image)")

        # 3: does image reliance flag wrong answers? label-free scores only
        sel = tabs["selective_labelfree"]
        order = ["confidence (baseline)", "image PMI, predicted answer  [label-free]",
                 "wrong-image PMI, predicted answer  [label-free]"]
        groups = ["Confidence", "Image reliance\n(image removed)", "Image reliance\n(wrong image)"]
        fig, ax = plt.subplots(figsize=(col, 1.95) if paper else (7.5, 3.4))
        series = {m: tuple(sel[sel.model == m].set_index("score").loc[order, k].to_numpy() for k in ("auroc", "lo", "hi"))
                  for m in models}
        _bars(ax, groups, series, color, LABEL, fs, pct=False, fmt="{:.2f}", ylim=(0.4, 0.82), hline=0.5)
        ax.text(-0.5, 0.505, "chance", fontsize=fs - 1.5, color=INK2, ha="left", va="bottom")
        ax.set_ylabel("AUROC, correct answer", color=INK2, fontsize=fs)
        ax.legend(frameon=False, fontsize=fs - 0.5, loc="upper right")
        _style(ax, None, fs)
        fig.tight_layout()
        save(fig, "fig3_selective.png", "Does image reliance flag wrong answers? (test yes/no)")

        # 4: abstention without the image: forced yes/no (main run) and with 'unknown' allowed (probe R2)
        abst = tabs["abstention"]
        dss = ["pathvqa", "slake", "vqa_rad"]
        pa = tabs.get("probe_abstain")
        npan = 2 if pa is not None else 1
        shape = (npan, 1) if paper else (1, npan)
        fig, axes = plt.subplots(*shape, figsize=(col, 1.55 * npan + 0.1) if paper else (5.5 * npan, 3.3),
                                 sharey=True, squeeze=False)
        axes = axes.reshape(1, -1)
        series = {m: tuple(abst[abst.model == m].set_index("dataset").loc[dss, k].to_numpy()
                           for k in ("abstain_noimg", "lo", "hi")) for m in models}
        _bars(axes[0, 0], [DS_LABEL[d] for d in dss], series, color, LABEL, fs, ylim=(0, 1.05))
        _style(axes[0, 0], "Instruction: answer yes or no (main run)", fs)
        axes[0, 0].legend(frameon=False, fontsize=fs - 0.5, loc="upper left")
        if pa is not None:
            series = {m: tuple(pa[(pa.model == m)].set_index("dataset").loc[dss, k].to_numpy()
                               for k in ("abstain_allowed", "lo", "hi")) for m in models}
            _bars(axes[0, 1], [DS_LABEL[d] for d in dss], series, color, LABEL, fs, ylim=(0, 1.05))
            _style(axes[0, 1], "Instruction allows 'unknown' (probe R2)", fs)
        fig.tight_layout()
        save(fig, "fig4_abstention.png", "Declines to answer when the image is removed")

        # 5: the MedGemma - Gemma-3 Dangerous gap under every variant (forest plot, one axis: share difference)
        p3 = tabs["p3_sensitivity"].copy()
        names = {"pre-registered (image-removed probe)": "Image removed (registered)",
                 f"both models answer without image (mass >= {ANSWER_MASS})": "  both answer blind",
                 f"neither model abstains (mass >= {ABSTAIN_MASS})": "  no abstentions",
                 "wrong-image probe instead": "Wrong image"}
        rows = [(names[v], e, lo, hi, n) for v, e, lo, hi, n in p3[["variant", "est", "lo", "hi", "n"]].itertuples(index=False)]
        pb = tabs.get("probe_blank")
        if pb is not None:
            r = pb.iloc[-1]
            rows.append(("Black image (R1)", r.dangerous_blank, r.dangerous_lo, r.dangerous_hi, int(r.dangerous_n)))
        hm = tabs["harm"]
        hm = hm[hm.model.str.contains(" - ")].set_index("measure")
        for meas, nm in (("dangerous_correct", "Removed: correct only"), ("dangerous_wrong", "Removed: wrong only")):
            rows.append((nm, hm.loc[meas, "est"], hm.loc[meas, "lo"], hm.loc[meas, "hi"], int(hm.loc[meas, "n"])))
        fig, ax = plt.subplots(figsize=(col, 0.32 * len(rows) + 0.5) if paper else (8.5, 0.45 * len(rows) + 1))
        y = np.arange(len(rows))[::-1]
        for yi, (nm, e, lo, hi, n) in zip(y, rows):
            ax.errorbar(e, yi, xerr=[[e - lo], [hi - e]], fmt="o", ms=4 if paper else 7, color=BLUE, ecolor=BLUE,
                        elinewidth=1.3 if paper else 2, mec=SURFACE, mew=1)
        ax.axvline(0, color=INK2, lw=0.9, ls=(0, (3, 3)))
        ax.axhline(1.5, color=GRID, lw=0.8)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
        ax.set_yticks(y, [f"{nm}  {n}" for nm, *_, n in rows], fontsize=fs - 0.5)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.0f}" if v else "0"))
        ax.set_xlabel("Dangerous share, MedGemma − Gemma-3 (points, 95% CI)", color=INK2, fontsize=fs)
        _style(ax, None, fs)
        fig.tight_layout()
        save(fig, "fig5_p3.png", "The Dangerous gap, by probe and by correctness")

        # 6: opposite-answer images (probe R3): follow rate by modality
        po = tabs.get("probe_opp")
        if po is not None:
            mods = ["all", "ct", "mri", "xray"]
            fig, ax = plt.subplots(figsize=(col, 1.85) if paper else (6.5, 3.2))
            series = {m: tuple(po[po.model == m].set_index("modality").reindex(mods)[k].to_numpy()
                               for k in ("follow_rate", "lo", "hi")) for m in models}
            _bars(ax, ["All", "CT", "MRI", "X-ray"], series, color, LABEL, fs, ylim=(0, 1.1))
            ax.set_ylabel("answer changes", color=INK2, fontsize=fs)
            fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=fs - 0.5, loc="upper center", ncol=2,
                       bbox_to_anchor=(0.55, 0.0))
            _style(ax, None, fs)
            fig.tight_layout()
            save(fig, "fig6_opposite.png", "Image swapped for one whose correct answer is opposite (SLAKE)")

        # 7: the headline: one measure (answer changes) under every probe, both models
        rv = tabs.get("reversal")
        if rv is not None and not rv.empty:
            probes = [p for p, _ in REVERSAL_PROBES if p in set(rv.probe)]
            sub = {"noimg": "the reference probe", "shuf": "same dataset and modality",
                   "blank": "probe R1", "opp": "probe R3, SLAKE"}
            fig, ax = plt.subplots(figsize=(col, 2.6) if paper else (8.0, 4.3))
            y = np.arange(len(probes))[::-1].astype(float)
            ms, lw = (4.5 if paper else 8), (1.4 if paper else 2.2)
            off = {models[0]: 0.0, models[1]: -0.26}
            for m in models:
                g = rv[rv.model == m].set_index("probe").loc[probes]
                ax.errorbar(g.changed, y + off[m], xerr=[g.changed - g.lo, g.hi - g.changed], fmt="o", ms=ms,
                            color=color[m], ecolor=color[m], elinewidth=lw, mec=SURFACE, mew=1)
            d = rv[rv.model.str.contains(" - ")].set_index("probe").loc[probes]
            for yi, p in zip(y, probes):
                head = ax.text(0.005, yi + 0.2, dict(REVERSAL_PROBES)[p], fontsize=fs, color=INK, ha="left",
                               va="baseline", fontweight="bold")
                ax.annotate(f"{sub[p]} · n={int(d.loc[p, 'n']):,}", xy=(1, 0), xycoords=head, xytext=(4, 0),
                            textcoords="offset points", fontsize=fs - 1.5, color=INK2, ha="left", va="bottom")
                ax.text(1.02, yi - 0.13, f"{d.loc[p, 'changed'] * 100:+.0f}".replace("-", "−"),
                        transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=fs, color=INK)
            ax.text(1.02, y[0] + 0.55, "MedGemma\n− Gemma-3", transform=ax.get_yaxis_transform(), ha="left",
                    va="bottom", fontsize=fs - 1.5, color=INK2, linespacing=1.05)
            ax.axhline(y[0] - 0.55, color=GRID, lw=0.8)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.set_ylim(y[-1] - 0.5, y[0] + 0.62)
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_xlabel("yes/no answers that change when the image is altered (95% CI)", color=INK2, fontsize=fs)
            _style(ax, None, fs)
            ax.spines["left"].set_visible(False)
            fig.legend([Line2D([], [], marker="o", ls="", ms=ms, color=color[m]) for m in models],
                       [LABEL[m] for m in models], frameon=False, fontsize=fs - 0.5, loc="upper center", ncol=2,
                       bbox_to_anchor=(0.5, 0.0))
            fig.tight_layout()
            save(fig, "fig7_reversal.png", "Which model uses the image? The probes disagree")
    return written


def posthoc(cfg: dict, paths: Paths, model_a: str, model_b: str) -> dict[str, pd.DataFrame]:
    profile, B, seed = cfg["profile"], cfg["analysis"]["bootstrap_B"], cfg["seed"]
    items, samples, raws = _tables(cfg, paths, [model_a, model_b])
    out = {"abstention": abstention(samples, raws, B, seed),
           "selective_labelfree": selective(samples, raws, items, B, seed),
           "partner_confound": partner(samples, raws, items),
           "p3_sensitivity": p3_sensitivity(samples, raws, model_a, model_b, B, seed),
           "p1_sensitivity": p1_sensitivity(samples, raws, B, seed),
           "chance": chance(samples, raws, model_a, model_b, B, seed),
           "harm": harm(samples, model_a, model_b, B, seed),
           "no_pilot": no_pilot(cfg, paths, samples, raws, items, model_a, model_b, B, seed)}
    out.update(probe_tables(cfg, paths, samples, raws, items, model_a, model_b, B, seed))
    out["reversal"] = reversal(cfg, paths, samples, raws, items, model_a, model_b, B, seed)
    out["removal_split"] = removal_split(samples, raws, B, seed)
    out_dir = paths.results / f"posthoc_{model_a}_vs_{model_b}" / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    paper_dir = paths.root / "report" / "figures"
    figs = figures(out_dir, samples, [model_a, model_b], out, paths.results, profile, paper_dir=paper_dir)
    print(f"[posthoc] {len(out)} tables + {len(figs)} figures -> {out_dir} (paper versions -> {paper_dir})")
    for k in ("p3_sensitivity", "harm", "probe_tests"):
        if k in out:
            print(out[k].round(3).to_string(index=False))
    return out

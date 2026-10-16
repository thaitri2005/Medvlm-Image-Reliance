"""Opposite-answer (contrast-set) analysis: probe R3 on SLAKE and the registered out-of-distribution replication
R4 on PathVQA (docs/PREREGISTRATION.md section 17).

Each question is asked about its own image and about a partner image on which the same question has the opposite
gold answer. Because the two gold answers are opposite, every model outcome falls in exactly one cell:

    both right   answer changes, and correctly        (image read correctly twice)
    both wrong   answer changes, but wrongly           (image read, consistently wrong)
    stuck        answer does not change                (exactly one of the two is right)

follow rate = both right + both wrong. **Pair accuracy** (both right) is the contrast-set metric: a model that
ignores the image always gives the same answer to both images, so it scores exactly 0 by construction, and a model
answering independently at random scores 25%. It separates "uses the image" from "is more accurate".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Paths
from ..data.sampling import opposite_key
from ..schema import load_items
from .stats import cluster_bootstrap, holm
from .table import load_results

ABSTAIN_MASS = 0.5
EQUIV_MARGIN = 0.05            # H12c: "no difference" if the 95% CI lies inside +-5 points (registered)
MODELS = ("medgemma4b", "medgemma15_4b", "gemma3_4b", "medvlthinker_3b", "qwen25vl_3b")
PAIRS = (("medgemma4b", "gemma3_4b"), ("medgemma15_4b", "gemma3_4b"), ("medvlthinker_3b", "qwen25vl_3b"))
# H12: predicted sign of (medical - general) follow rate on the R4 set, per pair (registered before running R4)
H12 = {("medgemma4b", "gemma3_4b"): ("H12a", "> 0"), ("medgemma15_4b", "gemma3_4b"): ("H12b", "> 0"),
       ("medvlthinker_3b", "qwen25vl_3b"): ("H12c", "= 0 (CI within +-5 points)")}
# (set id, profile, dataset filter, label). R3 = registered on SLAKE; R4 = registered on PathVQA.
SETS = (("r3_slake", "main", "slake", "SLAKE, probe R3"),
        ("main_other", "main", "other", "VQA-RAD + PathVQA swaps in main (exploratory)"),
        ("r4_pathvqa", "contrast", None, "PathVQA, probe R4"))
METRICS = ("follow", "both_right", "both_wrong", "stuck", "correct_orig", "correct_partner", "refuses")


def _wide(res: pd.DataFrame, cond: str) -> pd.DataFrame:
    r = res[res.condition == cond].drop_duplicates("sample_id", keep="last").set_index("sample_id")
    return r[["p_yes", "yn_mass"]] if len(r) else pd.DataFrame(columns=["p_yes", "yn_mass"])


def contrast_frame(paths: Paths, model: str, profile: str) -> pd.DataFrame:
    """One row per question with both answers and the outcome cells; empty if either run is missing."""
    from ..probes import opposite_partners

    mf = paths.manifest_dir(profile) / "items.jsonl"
    o = load_results(paths.run_dir(model, profile))
    p = load_results(paths.run_dir(f"{model}__opp", profile))
    if not mf.exists() or o.empty or p.empty:
        return pd.DataFrame()
    items = load_items(mf)
    by_id, partners = {it.sample_id: it for it in items}, opposite_partners(items)
    o, p = _wide(o, "orig"), _wide(p, "opp")
    idx = [s for s in p.index if s in o.index and s in partners]
    if not idx:
        return pd.DataFrame()
    it = [by_id[s] for s in idx]
    d = pd.DataFrame({
        "dataset": [x.dataset for x in it], "split": [x.split for x in it], "modality": [x.modality for x in it],
        "image_id": [x.image_id for x in it], "partner_image_id": [partners[s].image_id for s in idx],
        "template": ["|".join(opposite_key(x.dataset, x.modality, x.question)) for x in it],
        "gold_yes": [x.answer.strip().lower() == "yes" for x in it],
        "yes_orig": (o.loc[idx, "p_yes"].astype(float) >= .5).to_numpy(),
        "yes_partner": (p.loc[idx, "p_yes"].astype(float) >= .5).to_numpy(),
        "mass_orig": o.loc[idx, "yn_mass"].astype(float).to_numpy(),
        "mass_partner": p.loc[idx, "yn_mass"].astype(float).to_numpy()}, index=pd.Index(idx, name="sample_id"))
    d["correct_orig"] = d.yes_orig == d.gold_yes
    d["correct_partner"] = d.yes_partner != d.gold_yes          # the partner's gold is the opposite answer
    d["follow"] = d.yes_orig != d.yes_partner
    d["both_right"] = d.correct_orig & d.correct_partner
    d["both_wrong"] = ~d.correct_orig & ~d.correct_partner
    d["stuck"] = ~d.follow
    d["refuses"] = (d.mass_orig < ABSTAIN_MASS) | (d.mass_partner < ABSTAIN_MASS)
    assert (d.follow == (d.both_right | d.both_wrong)).all()    # holds because the two gold answers are opposite
    return d


def _subset(d: pd.DataFrame, which: str | None) -> pd.DataFrame:
    if d.empty or which is None:
        return d
    return d[d.dataset == "slake"] if which == "slake" else d[d.dataset != "slake"]


def _mean(col):
    return lambda x: float(x[col].astype(float).mean())


def _paired(col):
    return lambda x: float((x[f"{col}_a"].astype(float) - x[f"{col}_b"].astype(float)).mean())


def summarize_contrast(paths: Paths, models=MODELS, pairs=PAIRS, sets=SETS, B: int = 2000, seed: int = 1234
                       ) -> dict[str, pd.DataFrame]:
    """Per-model rates and per-pair paired differences for every set that has outputs, plus the H12 tests."""
    frames = {(sid, m): _subset(contrast_frame(paths, m, prof), which) for sid, prof, which, _ in sets for m in models}
    label = {sid: lab for sid, _, _, lab in sets}
    rows = []
    for (sid, m), d in frames.items():
        if d.empty:
            continue
        row = {"set": sid, "set_label": label[sid], "model": m, "n": len(d), "n_images": d.image_id.nunique(),
               "n_templates": d.template.nunique()}
        for col in METRICS:
            r = cluster_bootstrap(d, _mean(col), B=B, seed=seed)
            row.update({col: r["est"], f"{col}_lo": r["lo"], f"{col}_hi": r["hi"]})
        rows.append(row)
    per_model = pd.DataFrame(rows)

    rows = []
    for sid, _, _, _ in sets:
        for a, b in pairs:
            da, db = frames.get((sid, a), pd.DataFrame()), frames.get((sid, b), pd.DataFrame())
            if da.empty or db.empty:
                continue
            idx = da.index.intersection(db.index)
            D = da.loc[idx, ["image_id", "template"]].copy()
            for col in ("follow", "both_right", "both_wrong"):
                D[f"{col}_a"], D[f"{col}_b"] = da.loc[idx, col], db.loc[idx, col]
            row = {"set": sid, "set_label": label[sid], "pair": f"{a} - {b}", "n": len(D), "n_images": D.image_id.nunique()}
            for col in ("follow", "both_right", "both_wrong"):
                r = cluster_bootstrap(D, _paired(col), B=B, seed=seed)
                row.update({f"d_{col}": r["est"], f"d_{col}_lo": r["lo"], f"d_{col}_hi": r["hi"], f"d_{col}_p": r["p"]})
            # sensitivity: resample question templates instead of images (questions repeat across images)
            r = cluster_bootstrap(D, _paired("follow"), cluster_col="template", B=B, seed=seed)
            row.update(d_follow_lo_template=r["lo"], d_follow_hi_template=r["hi"], d_follow_p_template=r["p"],
                       n_templates=D.template.nunique())
            rows.append(row)
    per_pair = pd.DataFrame(rows)

    tests = pd.DataFrame()
    if len(per_pair):
        r4 = per_pair[per_pair.set == "r4_pathvqa"].set_index("pair")
        keys = [f"{a} - {b}" for a, b in pairs if (a, b) in H12 and f"{a} - {b}" in r4.index]
        if len(keys) == len(H12):                               # the registered family is only tested when complete
            rows = []
            for (a, b) in H12:
                k = f"{a} - {b}"
                hid, pred = H12[(a, b)]
                x = r4.loc[k]
                rows.append({"hypothesis": hid, "pair": k, "predicted": pred, "est": x.d_follow, "lo": x.d_follow_lo,
                             "hi": x.d_follow_hi, "p": x.d_follow_p, "n": x.n, "n_images": x.n_images})
            tests = pd.DataFrame(rows)
            tests["p_holm"] = holm(tests.p.tolist())
            tests["supported"] = [
                (r.lo > 0 and r.p_holm < 0.05) if r.predicted == "> 0"
                else (-EQUIV_MARGIN <= r.lo and r.hi <= EQUIV_MARGIN) for r in tests.itertuples()]
    return {"contrast_models": per_model, "contrast_pairs": per_pair, "contrast_tests": tests}


def contrast_figure(per_model: pd.DataFrame, pairs, out_dir, paper_dir=None) -> list:
    """fig9: follow rate and pair accuracy on each opposite-answer set, one panel per pair (blue = medical model of
    the pair, orange = its general base). Pair accuracy is 0 for a model that ignores the image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from .posthoc import BLUE, GRID, INK, INK2, LABEL, ORANGE, SURFACE, _style

    sets = [s for s in ("r3_slake", "r4_pathvqa") if s in set(per_model.set)]
    pairs = [(a, b) for a, b in pairs if {a, b} <= set(per_model.model)]
    if not sets or not pairs:
        return []
    short = {"r3_slake": "SLAKE (R3)", "r4_pathvqa": "PathVQA (R4)"}
    rows = [(m, s) for m in ("follow", "both_right") for s in sets]
    names = {"follow": "follows the image", "both_right": "pair accuracy"}
    written = []
    for mode, target in (("web", out_dir), ("paper", paper_dir)):
        if target is None:
            continue
        paper = mode == "paper"
        fs = 7.2 if paper else 9
        plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                             "font.size": fs})
        fig, axes = plt.subplots(1, len(pairs), figsize=(6.9, 0.45 * len(rows) + 1.0) if paper
                                 else (13, 0.75 * len(rows) + 1.6), sharey=True, squeeze=False)
        ms, lw = (4 if paper else 7), (1.3 if paper else 2)
        y = np.arange(len(rows))[::-1].astype(float)
        for ax, (a, b) in zip(axes[0], pairs):
            for m, col, off in ((a, BLUE, 0.14), (b, ORANGE, -0.14)):
                g = per_model[per_model.model == m].set_index("set")
                for yi, (met, s) in zip(y, rows):
                    if s in g.index:
                        v = g.loc[s]
                        ax.errorbar(v[met], yi + off, xerr=[[v[met] - v[f"{met}_lo"]], [v[f"{met}_hi"] - v[met]]],
                                    fmt="o", ms=ms, color=col, ecolor=col, elinewidth=lw, mec=SURFACE, mew=1)
            ax.axhline(y[len(sets) - 1] - 0.5, color=GRID, lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(y[-1] - 0.5, y[0] + 0.5)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.25))
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
            ax.grid(axis="x", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            _style(ax, f"{LABEL.get(a, a)}\nvs {LABEL.get(b, b)}", fs)
            ax.tick_params(axis="y", length=0)
        axes[0, 0].set_yticks(y, [f"{names[m]}, {short[s]}" for m, s in rows])
        fig.supxlabel("share of opposite-answer pairs (95% CI). Pair accuracy = right on both images; "
                      "0% for a model that ignores the image", fontsize=fs, color=INK2)
        fig.legend([Line2D([], [], marker="o", ls="", ms=ms, color=c) for c in (BLUE, ORANGE)],
                   ["medical model of the pair", "its general-purpose base"], frameon=False, fontsize=fs - 0.5,
                   loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(w_pad=2.5)
        target.mkdir(parents=True, exist_ok=True)
        if not paper:
            fig.canvas.draw()
            bb = fig.get_tightbbox(fig.canvas.get_renderer())
            title = ("Opposite-answer images inside and outside the medical models' training data"
                     if "r4_pathvqa" in sets else "Opposite-answer images (SLAKE)")
            fig.suptitle(title,
                         x=bb.x0 / fig.get_figwidth(), y=bb.y1 / fig.get_figheight() + 0.015, ha="left",
                         va="bottom", fontsize=fs + 3, color=INK)
        fig.savefig(target / "fig9_contrast.png", dpi=300 if paper else 200, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        written.append(target / "fig9_contrast.png")
    return written


def analyze_contrast(paths: Paths, B: int = 2000, seed: int = 1234, paper_dir=None) -> dict[str, pd.DataFrame]:
    out = summarize_contrast(paths, B=B, seed=seed)
    out_dir = paths.results / "contrast"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        if len(t):
            t.to_csv(out_dir / f"{name}.csv", index=False)
    if len(out["contrast_models"]):
        figs = contrast_figure(out["contrast_models"], PAIRS, out_dir, paper_dir)
        cols = ["set", "model", "n", "follow", "both_right", "both_wrong", "stuck"]
        print("[contrast] per model:\n" + out["contrast_models"][cols].round(3).to_string(index=False))
        if len(out["contrast_pairs"]):
            cols = ["set", "pair", "n", "d_follow", "d_follow_lo", "d_follow_hi", "d_both_right"]
            print("[contrast] medical - general:\n" + out["contrast_pairs"][cols].round(3).to_string(index=False))
        if len(out["contrast_tests"]):
            print("[contrast] registered H12 (R4):\n" + out["contrast_tests"].round(4).to_string(index=False))
        else:
            print("[contrast] H12 not tested yet: R4 outputs missing for some model (run `medvlm contrast`)")
        print(f"[contrast] -> {out_dir}" + (f"; figures: {', '.join(str(f) for f in figs)}" if figs else ""))
    else:
        print("[contrast] no opposite-answer outputs found")
    return out

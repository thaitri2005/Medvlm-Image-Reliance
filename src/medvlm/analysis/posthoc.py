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
  figures         fig1-fig5 PNGs for the report and poster.
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


# ------------------------------------------------------------------ figures

def _style(ax, title: str | None = None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)


def figures(out_dir, samples, models, abst, sel, p3, results_dir, profile):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    color = {models[0]: BLUE, models[1]: ORANGE}
    label = {"medgemma4b": "MedGemma-4B", "gemma3_4b": "Gemma-3-4B"}
    paths = []

    # fig1: quadrant composition, test yes/no, both probes (small multiples, 100% stacked bars)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, (probe, col) in zip(axes, [("image removed", "quadrant"), ("wrong image", "quadrant_shuf")]):
        ylabels, y = [], 0
        for ds in ("pathvqa", "slake", "vqa_rad"):
            for m in models:
                t = samples[m]
                g = t[(t.answer_type == "yesno") & (t.split == "test") & (t.dataset == ds)][col].dropna()
                share = g.value_counts(normalize=True)
                left = 0.0
                for q in ("ideal", "fragile", "dangerous", "worst"):
                    w = float(share.get(q, 0.0))
                    ax.barh(y, w - 0.004 if w > 0.004 else w, left=left, height=0.72, color=QUAD_COLORS[q], linewidth=0)
                    if w >= 0.08:
                        ax.text(left + w / 2, y, f"{w:.0%}", ha="center", va="center", fontsize=8,
                                color="white" if q in ("fragile", "dangerous") else INK)
                    left += w
                ylabels.append(f"{DS_LABEL[ds]} · {label[m]}")
                y += 1
            y += 0.4
        ax.set_yticks([0, 1, 2.4, 3.4, 4.8, 5.8])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        _style(ax, f"Probe: {probe}")
    axes[0].set_yticklabels(ylabels)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=QUAD_COLORS[q]) for q in QUAD_COLORS]
    fig.legend(handles, ["Ideal (stable, image-reliant)", "Fragile (flips, reliant)", "Dangerous (stable, not reliant)",
                         "Worst (flips, not reliant)"], loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Fig. 1  Quadrant composition of yes/no answers (test split)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    paths.append(out_dir / "fig1_quadrants.png"); fig.savefig(paths[-1], dpi=200); plt.close(fig)

    # fig2: Dangerous share by SLAKE modality (test+val pooled, pre-specified), both probes, CIs
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    mods = ["ct", "mri", "xray"]
    for ax, (probe, col) in zip(axes, [("image removed", "q_dangerous"), ("wrong image", "q_dangerous_shuf")]):
        for k, m in enumerate(models):
            tb = pd.read_csv(results_dir / m / profile / "by_modality_pooled.csv")
            tb = tb[(tb.dataset == "slake") & (tb.answer_type == "yesno")].set_index("modality")
            x = np.arange(len(mods)) + (k - 0.5) * 0.22
            est = tb.loc[mods, col].to_numpy()
            lo, hi = tb.loc[mods, f"{col}_lo"].to_numpy(), tb.loc[mods, f"{col}_hi"].to_numpy()
            ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", ms=7, color=color[m], ecolor=color[m],
                        elinewidth=2, capsize=0, label=label[m], mec=SURFACE, mew=1.5)
        ax.set_xticks(range(len(mods)), ["CT", "MRI", "X-ray"])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        _style(ax, f"Probe: {probe}")
    axes[0].set_ylabel("Dangerous share", color=INK2)
    from matplotlib.lines import Line2D
    axes[0].legend([Line2D([], [], marker="o", ls="", ms=7, color=color[m]) for m in models], [label[m] for m in models],
                   frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Fig. 2  Dangerous share by modality (SLAKE yes/no, test+val, 95% CI by image)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    paths.append(out_dir / "fig2_modality.png"); fig.savefig(paths[-1], dpi=200); plt.close(fig)

    # fig3: label leakage in the pre-registered P2
    order = ["confidence (baseline)", "log p(gold | image)  [leaks label]",
             "image PMI, gold answer  [pre-registered P2, leaks label]", "image PMI, predicted answer  [label-free]",
             "wrong-image PMI, predicted answer  [label-free]"]
    short = ["Confidence\n(baseline)", "log p(gold|image)\n[leaks label]", "Image PMI, gold\n[pre-reg. P2, leaks]",
             "Image PMI, predicted\n[label-free]", "Wrong-image PMI,\npredicted [label-free]"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for k, m in enumerate(models):
        s = sel[sel.model == m].set_index("score").loc[order]
        x = np.arange(len(order)) + (k - 0.5) * 0.36
        ax.bar(x, s.auroc - 0.5, bottom=0.5, width=0.34, color=color[m], label=label[m], linewidth=0)
        ax.errorbar(x, s.auroc, yerr=[s.auroc - s.lo, s.hi - s.auroc], fmt="none", ecolor=INK2, elinewidth=1)
        for xi, v, h in zip(x, s.auroc, s.hi):
            ax.text(xi, max(v, h) + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(-0.62, 0.508, "chance", fontsize=8, color=INK2, ha="left")
    ax.set_xticks(range(len(order)), short, fontsize=8.5)
    ax.set_ylim(0.4, 1.08)
    ax.set_ylabel("AUROC for a correct answer", color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, "Fig. 3  Does image reliance flag wrong answers? Only when the score is allowed to see the answer key")
    fig.tight_layout()
    paths.append(out_dir / "fig3_leakage.png"); fig.savefig(paths[-1], dpi=200); plt.close(fig)

    # fig4: abstention without the image
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    dss = ["pathvqa", "slake", "vqa_rad"]
    for k, m in enumerate(models):
        s = abst[abst.model == m].set_index("dataset").loc[dss]
        x = np.arange(len(dss)) + (k - 0.5) * 0.36
        ax.bar(x, s.abstain_noimg, width=0.34, color=color[m], label=label[m], linewidth=0)
        ax.errorbar(x, s.abstain_noimg, yerr=[s.abstain_noimg - s.lo, s.hi - s.abstain_noimg], fmt="none",
                    ecolor=INK2, elinewidth=1)
        for xi, v, h in zip(x, s.abstain_noimg, s.hi):
            ax.text(xi, h + 0.015, f"{v:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(dss)), [DS_LABEL[d] for d in dss])
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, 'Fig. 4  Abstains when the image is removed ("please provide the image")')
    fig.tight_layout()
    paths.append(out_dir / "fig4_abstention.png"); fig.savefig(paths[-1], dpi=200); plt.close(fig)

    # fig5: P3 under each variant (forest plot)
    fig, ax = plt.subplots(figsize=(8.5, 2.9))
    y = np.arange(len(p3))[::-1]
    ax.errorbar(p3.est, y, xerr=[p3.est - p3.lo, p3.hi - p3.est], fmt="o", ms=7, color=BLUE, ecolor=BLUE, elinewidth=2,
                mec=SURFACE, mew=1.5)
    ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.set_yticks(y, [f"{v}  (n={n})" for v, n in zip(p3.variant, p3.n)], fontsize=8.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Dangerous share, MedGemma-4B minus Gemma-3-4B (paired, 95% CI by image)", color=INK2, fontsize=9)
    _style(ax)
    fig.suptitle("Fig. 5  The MedGemma - Gemma-3 gap (P3) depends on the probe", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    paths.append(out_dir / "fig5_p3.png"); fig.savefig(paths[-1], dpi=200); plt.close(fig)
    return paths


def posthoc(cfg: dict, paths: Paths, model_a: str, model_b: str) -> dict[str, pd.DataFrame]:
    profile, B, seed = cfg["profile"], cfg["analysis"]["bootstrap_B"], cfg["seed"]
    items, samples, raws = _tables(cfg, paths, [model_a, model_b])
    out = {"abstention": abstention(samples, raws, B, seed),
           "selective_labelfree": selective(samples, raws, items, B, seed),
           "partner_confound": partner(samples, raws, items),
           "p3_sensitivity": p3_sensitivity(samples, raws, model_a, model_b, B, seed),
           "p1_sensitivity": p1_sensitivity(samples, raws, B, seed)}
    out_dir = paths.results / f"posthoc_{model_a}_vs_{model_b}" / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    figs = figures(out_dir, samples, [model_a, model_b], out["abstention"], out["selective_labelfree"],
                   out["p3_sensitivity"], paths.results, profile)
    print(f"[posthoc] {len(out)} tables + {len(figs)} figures -> {out_dir}")
    print(out["p3_sensitivity"][["variant", "est", "lo", "hi", "n"]].round(3).to_string(index=False))
    return out

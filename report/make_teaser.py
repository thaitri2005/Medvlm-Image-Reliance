"""Build report/figures/fig0_teaser.png from report/teaser_examples.json (real model outputs, committed)
and the SLAKE images (CC-BY-4.0) fetched by `medvlm prepare`. Run from the repo root:

    python report/make_teaser.py

Everything is placed in inch coordinates on one canvas (7.0 in = CVPR \\textwidth) so nothing overlaps.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = json.loads((ROOT / "report" / "teaser_examples.json").read_text(encoding="utf-8"))
IMG = ROOT / "data" / "images"
BLUE, ORANGE = "#2a78d6", "#eb6834"
RED, AQUA = "#e34948", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d3", "#ffffff"
W, H = 7.0, 3.0
S = 0.78                                           # thumbnail size (in)
COL = {"model": 1.78, "orig": 2.78, "para": 3.42, "noimg": 4.18, "shuf": 5.0, "q_noimg": 6.02, "q_shuf": 6.66}
CHIP_W = 0.6

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(W, H), dpi=300)
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, W), ax.set_ylim(0, H), ax.axis("off")


def text(x, y, s, size=6.8, color=INK, **kw):
    ax.text(x, y, s, fontsize=size, color=color, va="center", **kw)


def thumb(x, y, path, label, sub=None):
    ax.imshow(Image.open(IMG / path).convert("L"), cmap="gray", extent=(x, x + S, y, y + S), zorder=1)
    ax.add_patch(Rectangle((x, y), S, S, fill=False, ec=GRID, lw=0.6, zorder=2))
    text(x + S / 2, y + S + 0.08, label, 6.6, INK, ha="center")
    if sub:
        text(x + S / 2, y - 0.08, sub, 6.0, INK2, ha="center")


def blank(x, y, label):
    ax.add_patch(Rectangle((x, y), S, S, fc="#f2f1ee", ec=GRID, lw=0.6, ls=(0, (3, 2))))
    text(x + S / 2, y + S / 2, "no image", 6.2, INK2, ha="center")
    text(x + S / 2, y + S + 0.08, label, 6.6, INK, ha="center")


def chip(xc, y, label):
    fc, tc = (RED, "white") if label == "Dangerous" else (AQUA, INK)
    ax.add_patch(FancyBboxPatch((xc - CHIP_W / 2, y - 0.075), CHIP_W, 0.15,
                                boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec="none"))
    text(xc, y, label, 6.4, tc, ha="center", weight="bold")


def model_label(y, name, color):
    ax.add_patch(Rectangle((COL["model"], y - 0.045), 0.09, 0.09, fc=color, ec="none"))
    text(COL["model"] + 0.14, y, name, 6.8, INK, weight="bold")


def row(y_top, title, imgs, cells, quads, footnote):
    """y_top: top of the row block. imgs: two callables placing thumbnails at (x, y)."""
    text(0.02, y_top, title, 7.6, INK, weight="bold")
    y_img = y_top - 0.32 - S
    imgs[0](0.02, y_img)
    imgs[1](0.02 + S + 0.1, y_img)
    y_h = y_top - 0.3
    heads = {"orig": "with image", "para": "2 paraphrases", "noimg": "image removed", "shuf": "wrong image",
             "q_noimg": "quadrant\n(removed)", "q_shuf": "quadrant\n(wrong)"}
    for k, h in heads.items():
        if k in cells["MedGemma-4B"] or k in quads["MedGemma-4B"]:
            text(COL[k] if not k.startswith("q_") else COL[k], y_h, h, 6.0, INK2,
                 ha="center" if k.startswith("q_") else "left", linespacing=1.0)
    ax.plot([COL["model"], W - 0.02], [y_h - 0.13, y_h - 0.13], color=GRID, lw=0.6)
    for i, (name, color) in enumerate((("MedGemma-4B", BLUE), ("Gemma-3-4B", ORANGE))):
        y = y_h - 0.33 - i * 0.27
        model_label(y, name, color)
        for k, v in cells[name].items():
            text(COL[k], y, v, 6.8, INK, linespacing=1.0)
        for k, v in quads[name].items():
            chip(COL[k], y, v)
    if footnote:
        text(COL["model"], y_h - 0.94, footnote, 5.9, INK2, style="italic")


def ans(m, cond):
    return m[cond]["text"] + ("*" if m[cond]["mass"] < 0.9 else "")


A, B = EX["slake-test-12941"], EX["slake-test-12880"]
ma, ga, mb, gb = A["models"]["medgemma4b"], A["models"]["gemma3_4b"], B["models"]["medgemma4b"], B["models"]["gemma3_4b"]

row(H - 0.12, f"(a)  “{A['question']}”   (SLAKE CT, gold: {A['gold'].lower()})",
    [lambda x, y: thumb(x, y, A["image"], "with image"), lambda x, y: blank(x, y, "image removed")],
    {"MedGemma-4B": {"orig": ans(ma, "orig"), "para": "yes, yes", "noimg": ans(ma, "noimg")},
     "Gemma-3-4B": {"orig": ans(ga, "orig"), "para": "yes, yes", "noimg": "“Please provide\nthe picture.”"}},
    {"MedGemma-4B": {"q_noimg": "Dangerous"}, "Gemma-3-4B": {"q_noimg": "Ideal"}},
    "Gemma-3 asks for the image; its forced yes/no choice is “no”, so the probe counts it as image-reliant.")

row(H - 1.62, f"(b)  “{B['question']}”   (SLAKE CT, gold: {B['gold'].lower()})",
    [lambda x, y: thumb(x, y, B["image"], "with image"),
     lambda x, y: thumb(x, y, B["partner"], "wrong image", "its answer: no")],
    {"MedGemma-4B": {"orig": ans(mb, "orig"), "para": "yes, yes", "noimg": ans(mb, "noimg"), "shuf": ans(mb, "shuf")},
     "Gemma-3-4B": {"orig": ans(gb, "orig"), "para": "yes, yes", "noimg": ans(gb, "noimg"), "shuf": ans(gb, "shuf")}},
    {"MedGemma-4B": {"q_noimg": "Dangerous", "q_shuf": "Ideal"}, "Gemma-3-4B": {"q_noimg": "Ideal", "q_shuf": "Dangerous"}},
    f"* only {gb['noimg']['mass']:.0%} of first-token probability on yes/no.   "
    "Same question, opposite verdicts from the two probes.")

out = ROOT / "report" / "figures" / "fig0_teaser.png"
fig.savefig(out, dpi=300, facecolor=SURFACE)
print("wrote", out)

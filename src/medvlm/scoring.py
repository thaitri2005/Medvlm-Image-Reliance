"""Answer normalisation, correctness and answer-equivalence.

Open-ended correctness uses token-level F1 (the metric Google reports for MedGemma on
VQA-RAD / SLAKE), thresholded so that we can speak of "correct" per sample, with two safeguards
that plain F1 lacks: surface-form canonicalisation ("Two" = "2", "Lungs" = "lung", "Axial" =
"Transverse") and an opposite-term guard (an answer with "left" never matches one with "right").
"""
from __future__ import annotations

import re
import string
from collections import Counter

from .schema import YESNO

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans({c: " " for c in string.punctuation})
_NUMBERS = {"zero": "0", "none": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
_SYNONYMS = {"axial": "transverse"}      # SLAKE's gold for the axial plane is "Transverse Plane"
_XRAY = re.compile(r"\bx[\s-]?rays?\b")
# If prediction and gold each contain a different member of one pair, the answers contradict.
_OPPOSITES = [{"left", "right"}, {"upper", "lower"}, {"anterior", "posterior"},
              {"superior", "inferior"}, {"medial", "lateral"}, {"proximal", "distal"}]


def _canon(tok: str) -> str:
    """Applied identically to prediction and gold."""
    tok = _NUMBERS.get(tok, tok)
    tok = _SYNONYMS.get(tok, tok)
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


def normalize(s: str) -> str:
    s = _XRAY.sub("xray", str(s).lower()).translate(_PUNCT)
    return " ".join(_canon(w) for w in s.split() if w not in _ARTICLES)


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def _contradicts(p: list[str], g: list[str]) -> bool:
    ps, gs = set(p), set(g)
    return any((ps & pair) and (gs & pair) and not (ps & gs & pair) for pair in _OPPOSITES)


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g or _contradicts(p, g):
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def parse_yesno(text: str) -> str | None:
    """Label from the first yes/no word; None if the model said neither."""
    for w in tokens(text):
        if w in ("yes", "no"):
            return w
        break
    m = re.search(r"\b(yes|no)\b", normalize(text))
    return m.group(1) if m else None


def is_correct(kind: str, pred: str, gold: str, f1_thresh: float = 0.5) -> bool:
    if kind == YESNO:
        return parse_yesno(pred) == normalize(gold)
    return token_f1(pred, gold) >= f1_thresh


def same_answer(kind: str, a: str, b: str, f1_thresh: float = 0.5) -> bool:
    """Equivalence used for consistency (paraphrase) and reliance (no-image) comparisons."""
    if kind == YESNO:
        la, lb = parse_yesno(a), parse_yesno(b)
        return la == lb if (la is not None and lb is not None) else normalize(a) == normalize(b)
    return normalize(a) == normalize(b) or token_f1(a, b) >= f1_thresh

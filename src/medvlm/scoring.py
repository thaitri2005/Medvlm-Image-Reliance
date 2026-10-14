"""Answer normalisation, correctness and answer-equivalence.

Open-ended correctness uses token-level F1 (the metric Google reports for MedGemma on
VQA-RAD / SLAKE), thresholded so that we can speak of "correct" per sample.
"""
from __future__ import annotations

import re
import string
from collections import Counter

from .schema import OPEN, YESNO

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalize(s: str) -> str:
    s = str(s).lower().translate(_PUNCT)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def tokens(s: str) -> list[str]:
    return normalize(s).split()


def token_f1(pred: str, gold: str) -> float:
    p, g = tokens(pred), tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
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

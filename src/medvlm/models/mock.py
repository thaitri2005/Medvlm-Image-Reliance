"""Deterministic fake VLM: answers depend on a hash of (question, image thumbnail). It has no
weights, so the whole pipeline (runner, resume, analysis) can be exercised in milliseconds."""
from __future__ import annotations

import hashlib
import math

from ..schema import YESNO
from .base import VLM, Output, Query, yesno_gold_logp

_WORDS = ["lung", "liver", "brain", "kidney", "heart", "spleen", "chest", "abdomen"]


def _u(*parts: str) -> tuple[float, int]:
    h = hashlib.sha1("|".join(parts).encode()).digest()
    return h[0] / 255.0, h[1]


def _sig(image) -> str:
    return "noimg" if image is None else hashlib.sha1(image.convert("RGB").resize((8, 8)).tobytes()).hexdigest()


class MockVLM(VLM):
    name = "mock"

    def generate(self, queries: list[Query]) -> list[Output]:
        outs = []
        for q in queries:
            u_img, k = _u(q.question, _sig(q.image))
            if q.kind == YESNO:
                p_yes = 0.05 + 0.9 * u_img
                out = Output("yes" if p_yes >= 0.5 else "no", p_yes=p_yes, yn_mass=1.0, entropy=-p_yes * math.log(p_yes)
                             - (1 - p_yes) * math.log(1 - p_yes))
                if q.gold:
                    out.gold_logp = yesno_gold_logp(p_yes, q.gold)
            else:
                out = Output(_WORDS[k % len(_WORDS)], entropy=1.0 + u_img, mean_logprob=-u_img * 2)
                if q.gold:
                    out.gold_logp = -1.0 - 4 * u_img
            outs.append(out)
        return outs

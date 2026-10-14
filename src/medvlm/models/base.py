"""Model-agnostic interface. The runner only ever sees Query -> Output."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

EPS = 1e-6


@dataclass
class Query:
    question: str
    kind: str                       # "yesno" | "open"
    image: object | None            # PIL.Image or None (text-only: image tokens omitted entirely)
    gold: str | None = None         # if set, the backend also returns log p(gold | image, question)


@dataclass
class Output:
    text: str                       # decoded answer ("yes"/"no" for yes/no questions)
    p_yes: float | None = None      # yes/no only: P(yes) renormalised over {yes, no} at the first token
    yn_mass: float | None = None    # yes/no only: raw P(yes)+P(no) at the first token; low = model not answering in format
    entropy: float | None = None    # first-token entropy over the full vocabulary (nats)
    mean_logprob: float | None = None   # open only: mean log-prob of the generated tokens
    gold_logp: float | None = None  # log p(gold | image, question); the continuous reliance signal


def yesno_gold_logp(p_yes: float, gold: str) -> float:
    p = min(max(p_yes, EPS), 1 - EPS)
    return math.log(p if gold.strip().lower() == "yes" else 1 - p)


class VLM(ABC):
    name = "vlm"

    @abstractmethod
    def generate(self, queries: list[Query]) -> list[Output]:
        """One Output per Query, same order."""

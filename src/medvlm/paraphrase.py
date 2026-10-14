"""Paraphrase generation + automatic validation + human audit sheet.

A paraphrase is only useful if it cannot change the correct answer. Validation therefore rejects
anything that drops/adds laterality, direction, negation, quantity words or numbers, leaks the gold
answer into the question, or turns a yes/no question into something else. Rejected paraphrases are
kept (valid=false, with reasons) so the rejection rate is itself reportable.
"""
from __future__ import annotations

import csv
import difflib
import random
import re
from pathlib import Path

from .schema import YESNO, Item, read_jsonl, write_jsonl
from .scoring import normalize

STYLES = ["lexical", "syntactic", "formal", "plain", "concise"]
_STYLE_HINT = {
    "lexical": "Replace words with natural synonyms but keep the sentence structure.",
    "syntactic": "Restructure the sentence (word order, voice) using mostly the same words.",
    "formal": "Phrase it the way a radiologist or pathologist would in a formal report request.",
    "plain": "Phrase it in plain, simple language that a non-specialist would use.",
    "concise": "Make it as short as possible without losing any meaning.",
}
_PROTECTED = {
    "left", "right", "upper", "lower", "middle", "anterior", "posterior", "medial", "lateral",
    "superior", "inferior", "proximal", "distal", "bilateral", "unilateral", "above", "below",
    "no", "not", "without", "none", "never", "any", "all", "both", "more", "less", "larger",
    "smaller", "most", "least", "only", "normal", "abnormal",
}
_AUX = {"is", "are", "does", "do", "did", "can", "could", "was", "were", "has", "have", "will",
        "would", "should"}


def _protected(text: str) -> set[str]:
    return {t for t in normalize(text).split() if t in _PROTECTED or any(c.isdigit() for c in t)}


def validate(orig: str, para: str, kind: str, gold: str, min_edit_ratio: float = 0.15) -> list[str]:
    """Return the list of reasons a paraphrase is rejected (empty list = valid)."""
    reasons = []
    o, p = normalize(orig), normalize(para)
    if not p:
        return ["empty"]
    if "?" in orig and not para.strip().endswith("?"):
        reasons.append("not_a_question")
    if para.count("?") > 1:
        reasons.append("multiple_questions")
    if 1 - difflib.SequenceMatcher(None, o, p).ratio() < min_edit_ratio:
        reasons.append("too_similar")
    if not 0.4 <= len(p) / max(len(o), 1) <= 3.0:
        reasons.append("length_ratio")
    if _protected(orig) != _protected(para):
        reasons.append("protected_terms_changed")
    g = normalize(gold)
    if g and kind != YESNO and re.search(rf"\b{re.escape(g)}\b", p) and not re.search(rf"\b{re.escape(g)}\b", o):
        reasons.append("leaks_answer")
    of, pf = o.split()[:1], p.split()[:1]
    if kind == YESNO and of and of[0] in _AUX and (not pf or pf[0] not in _AUX):
        reasons.append("yesno_form_lost")
    return reasons


# ---------------------------------------------------------------- backends

def _lowerfirst(q: str) -> str:
    head = q.split(" ", 1)[0]
    return q if head.isupper() and len(head) > 1 else q[:1].lower() + q[1:]


_SYN = {"image": "scan", "picture": "image", "show": "depict", "shows": "depicts",
        "visible": "seen", "present": "seen", "organ": "structure", "main": "primary",
        "located": "found", "abnormality": "abnormal finding"}
_SYNTAX = [
    (r"^is there (a |an |any )?(.+)\?$", r"Can you see \1\2 in this image?"),
    (r"^are there (.+)\?$", r"Can you see \1 in this image?"),
    (r"^does (the|this) (image|picture|scan) (show|contain) (.+)\?$", r"Is \4 shown in \1 \2?"),
    (r"^what is the (.+) of (.+)\?$", r"What \1 does \2 have?"),
]


class TemplateParaphraser:
    """Shallow rule-based rewrites. For tests and as a fallback; real runs use HFParaphraser."""

    def __call__(self, question: str, style: str) -> str:
        q = question.strip()
        if style == "lexical":
            words = []
            for w in q.split():
                core = w.strip("?,.")
                words.append(w.replace(core, _SYN[core.lower()]) if core.lower() in _SYN else w)
            return " ".join(words)
        if style == "syntactic":
            for pat, rep in _SYNTAX:
                if re.match(pat, q, flags=re.I):
                    out = re.sub(pat, rep, q, flags=re.I)
                    return out[:1].upper() + out[1:]
            return q
        if style == "formal":
            return f"Based on the image, {_lowerfirst(q)}"
        if style == "plain":
            return f"Looking at this picture, {_lowerfirst(q)}"
        return re.sub(r"^(in|on) (the|this) (image|scan|picture),? ", "", q, flags=re.I)


class HFParaphraser:
    """Text-only instruction-tuned LM; greedy decoding so the paraphrase file is reproducible."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype if dtype == "auto" else getattr(torch, dtype), device_map=device)

    def __call__(self, question: str, style: str) -> str:
        msg = [{"role": "user", "content":
                "Rewrite this medical imaging question so it keeps exactly the same meaning, the same "
                "anatomical terms, laterality, negations and numbers. Do NOT answer it. "
                f"{_STYLE_HINT[style]}\nOutput only the rewritten question.\n\nQuestion: {question}"}]
        ids = self.tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True).to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=48, do_sample=False)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip().split("\n")[0].strip().strip('"').strip()


def make_backend(cfg: dict):
    kind = cfg["paraphrase"]["backend"]
    if kind == "template":
        return TemplateParaphraser()
    if kind == "hf":
        return HFParaphraser(cfg["paraphrase"]["hf_model"])
    raise ValueError(f"backend {kind!r} generates nothing; use `medvlm import-paraphrases` for 'file'")


# ---------------------------------------------------------------- pipeline

def generate(items: list[Item], cfg: dict, out_path: Path, backend=None) -> int:
    """Resumable: (sample_id, k) pairs already in out_path are kept and skipped."""
    backend = backend or make_backend(cfg)
    n = cfg["paraphrase"]["n"]
    done = {(r["sample_id"], r["k"]): r for r in read_jsonl(out_path)} if out_path.exists() else {}
    for it in items:
        # a paraphrase that repeats an earlier valid one would count as a second, independent test
        seen = {normalize(done[(it.sample_id, k)]["text"]) for k in range(1, n + 1)
                if (it.sample_id, k) in done and done[(it.sample_id, k)]["valid"]}
        for k, style in enumerate(STYLES[:n], start=1):
            if (it.sample_id, k) in done:
                continue
            text = backend(it.question, style)
            reasons = validate(it.question, text, it.answer_type, it.answer, cfg["paraphrase"]["min_edit_ratio"])
            if not reasons and normalize(text) in seen:
                reasons.append("duplicate")
            if not reasons:
                seen.add(normalize(text))
            done[(it.sample_id, k)] = {"sample_id": it.sample_id, "k": k, "style": style, "text": text,
                                       "valid": not reasons, "reasons": reasons}
    return write_jsonl(out_path, (done[key] for key in sorted(done)))


def load_valid(path: Path, n: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if path.exists():
        for r in read_jsonl(path):
            if r["valid"] and r["k"] <= n:
                out.setdefault(r["sample_id"], []).append(r)
    return out


def export_audit_sheet(items: list[Item], para_path: Path, out_csv: Path, n: int, seed: int) -> int:
    """Random sample of paraphrases for two teammates to judge 'same meaning? y/n' independently."""
    by_id = {it.sample_id: it for it in items}
    rows = [r for r in read_jsonl(para_path) if r["valid"]]
    random.Random(seed).shuffle(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "k", "style", "original", "paraphrase", "same_meaning_A", "same_meaning_B", "notes"])
        for r in rows[:n]:
            w.writerow([r["sample_id"], r["k"], r["style"], by_id[r["sample_id"]].question, r["text"], "", "", ""])
    return min(n, len(rows))

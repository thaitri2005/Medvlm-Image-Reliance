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
    "no", "not", "without", "none", "never", "all", "both", "more", "less", "larger",
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
    """Text-only instruction-tuned LM; greedy decoding so the paraphrase file is reproducible.
    `batch` is what the pipeline uses (left-padded, shortest prompts together)."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.batch_size = torch, batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        device_map = {"": 0} if device == "auto" and torch.cuda.is_available() and torch.cuda.device_count() == 1 else device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype if dtype == "auto" else getattr(torch, dtype), device_map=device_map)
        self.model.eval()

    def _prompt(self, question: str, style: str) -> str:
        msg = [{"role": "user", "content":
                "Rewrite this medical imaging question so it keeps exactly the same meaning, the same "
                "anatomical terms, laterality, negations and numbers. Do NOT answer it. "
                f"{_STYLE_HINT[style]}\nOutput only the rewritten question.\n\nQuestion: {question}"}]
        return self.tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)

    def batch(self, questions: list[str], style: str) -> list[str]:
        order = sorted(range(len(questions)), key=lambda i: len(questions[i]))
        out: list[str] = [""] * len(questions)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([self._prompt(questions[i], style) for i in idx], return_tensors="pt",
                           padding=True).to(self.model.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=self.tok.pad_token_id)
            for row, i in zip(gen[:, enc["input_ids"].shape[1]:], idx):
                text = self.tok.decode(row, skip_special_tokens=True)
                out[i] = text.strip().split("\n")[0].strip().strip('"').strip()
        return out

    def __call__(self, question: str, style: str) -> str:
        return self.batch([question], style)[0]


def make_backend(cfg: dict):
    kind = cfg["paraphrase"]["backend"]
    if kind == "template":
        return TemplateParaphraser()
    if kind == "hf":
        return HFParaphraser(cfg["paraphrase"]["hf_model"], batch_size=cfg["paraphrase"].get("batch_size", 16))
    raise ValueError(f"backend {kind!r} generates nothing; use `medvlm import-paraphrases` for 'file'")


# ---------------------------------------------------------------- pipeline

def _candidates(backend, pairs: list[tuple[Item, int, str]]) -> dict[tuple[str, int], str]:
    """Raw paraphrase text for (item, k, style) triples, batched per style when the backend can."""
    texts: dict[tuple[str, int], str] = {}
    if hasattr(backend, "batch"):
        by_style: dict[str, list[tuple[Item, int]]] = {}
        for it, k, style in pairs:
            by_style.setdefault(style, []).append((it, k))
        for style, members in by_style.items():
            for (it, k), text in zip(members, backend.batch([it.question for it, _ in members], style)):
                texts[(it.sample_id, k)] = text
    else:
        for it, k, style in pairs:
            texts[(it.sample_id, k)] = backend(it.question, style)
    return texts


def generate(items: list[Item], cfg: dict, out_path: Path, backend=None, block: int = 256) -> int:
    """Resumable: (sample_id, k) pairs already in out_path are kept and skipped. Written to disk
    after every block of `block` questions, so a crash loses at most one block."""
    n = cfg["paraphrase"]["n"]
    done = {(r["sample_id"], r["k"]): r for r in read_jsonl(out_path, tolerant=True)} if out_path.exists() else {}
    if all((it.sample_id, k) in done for it in items for k in range(1, n + 1)):
        return len(done)                      # nothing pending: do not even load the paraphrase model
    backend = backend or make_backend(cfg)
    for start in range(0, len(items), block):
        part = items[start:start + block]
        pairs = [(it, k, style) for it in part for k, style in enumerate(STYLES[:n], start=1)
                 if (it.sample_id, k) not in done]
        if not pairs:
            continue
        texts = _candidates(backend, pairs)
        for it in part:
            # a paraphrase that repeats an earlier valid one would count as a second, independent test
            seen = {normalize(done[(it.sample_id, k)]["text"]) for k in range(1, n + 1)
                    if (it.sample_id, k) in done and done[(it.sample_id, k)]["valid"]}
            for k, style in enumerate(STYLES[:n], start=1):
                if (it.sample_id, k) in done:
                    continue
                text = texts[(it.sample_id, k)]
                reasons = validate(it.question, text, it.answer_type, it.answer, cfg["paraphrase"]["min_edit_ratio"])
                if not reasons and normalize(text) in seen:
                    reasons.append("duplicate")
                if not reasons:
                    seen.add(normalize(text))
                done[(it.sample_id, k)] = {"sample_id": it.sample_id, "k": k, "style": style, "text": text,
                                           "valid": not reasons, "reasons": reasons}
        write_jsonl(out_path, (done[key] for key in sorted(done)))
        print(f"[paraphrase] {min(start + block, len(items))}/{len(items)} questions", flush=True)
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

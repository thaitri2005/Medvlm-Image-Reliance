"""Paraphrase generation + automatic validation + human audit sheet.

A paraphrase is only useful if it cannot change the correct answer. Validation therefore rejects
anything that drops/adds laterality, direction, negation, quantity words or numbers, leaks the gold
answer into the question, or turns a yes/no question into something else. Rejected paraphrases are
kept (valid=false, with reasons) so the rejection rate is itself reportable.
"""
from __future__ import annotations

import collections
import csv
import difflib
import random
import re
from pathlib import Path

import pandas as pd

from .schema import YESNO, Item, read_jsonl, write_jsonl
from .scoring import normalize

# Append only: k is the 1-based position here and is part of every result key (para<k>).
STYLES = ["lexical", "syntactic", "formal", "plain", "concise", "frame", "template_a", "template_b"]
# Deterministic rewrites of the common yes/no forms (no LM): meaning cannot drift by construction.
RULE_STYLES = {"template_a", "template_b"}
_STYLE_HINT = {
    "lexical": "Replace words with natural synonyms but keep the sentence structure.",
    "syntactic": "Restructure the sentence (word order, voice) using mostly the same words.",
    "formal": "Phrase it the way a radiologist or pathologist would in a formal report request.",
    "plain": "Phrase it in plain, simple language that a non-specialist would use.",
    "concise": "Make it as short as possible without losing any meaning.",
    "frame": "Change only how the question is introduced or framed (for example 'Is there X?' -> "
             "'Can X be seen?'), keeping every medical and anatomical word exactly as it is.",
}
_PROTECTED = {
    "left", "right", "upper", "lower", "middle", "anterior", "posterior", "medial", "lateral",
    "superior", "inferior", "proximal", "distal", "bilateral", "unilateral", "above", "below",
    "no", "not", "without", "none", "never", "all", "both", "more", "less", "larger",
    "smaller", "most", "least", "only", "normal", "abnormal",
    # v2 (audit round 1): extent / side words that paraphrasers like to add ...
    "side", "either", "each", "same", "opposite", "largest", "smallest", "main", "mainly",
    # ... and absence words, which invert a yes/no answer ("present?" -> "absent?")
    "absent", "absence", "lack", "lacking", "missing", "devoid", "free", "neither", "nor", "nothing",
    # v3 (full LLM-judge pass): antonym pairs that flip a yes/no answer, and imaging planes
    "healthy", "unhealthy", "homogeneous", "heterogeneous", "adequate", "inadequate", "involved",
    "uninvolved", "obscured", "circular", "coronal", "sagittal", "transverse",
}
# Words that mean the same for the protected check; the canonical form is what gets counted.
_CANON = {
    "bilaterally": "bilateral", "unilaterally": "unilateral", "sides": "side", "lacks": "lack",
    "biggest": "largest", "greatest": "largest", "bigger": "larger", "greater": "larger",
    "primary": "main", "principal": "main", "chief": "main", "predominant": "main",
    "primarily": "mainly", "predominantly": "mainly", "chiefly": "mainly", "mostly": "mainly",
    "isnt": "not", "arent": "not", "doesnt": "not", "dont": "not", "cannot": "not",
    "homogenous": "homogeneous", "affected": "involved", "unaffected": "uninvolved", "round": "circular",
    "axial": "transverse", "horizontal": "transverse", "laterally": "lateral",
}
_AUX = {"is", "are", "does", "do", "did", "can", "could", "was", "were", "has", "have", "will",
        "would", "should"}
_WH = {"what", "which", "where", "how", "who", "whom", "whose", "when", "why"}
_TYPOGRAPHIC = set("‘’“”–— ")


def _protected(text: str) -> collections.Counter:
    """Protected terms as a multiset: adding a second 'right' or a 'bilaterally' is a change."""
    toks = (_CANON.get(t, t) for t in normalize(text).replace("'", "").split())
    return collections.Counter(t for t in toks if t in _PROTECTED or any(c.isdigit() for c in t))


def _foreign_chars(orig: str, para: str) -> bool:
    """Non-ASCII characters the original does not have (e.g. a CJK word leaking from the LM)."""
    return any(ord(c) > 127 and c not in orig and c not in _TYPOGRAPHIC for c in para)


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
    # first word from the raw text: normalize() strips plural 's' and would turn "does" into "doe"
    of, pf = re.findall(r"[a-z]+", orig.lower())[:1], re.findall(r"[a-z]+", para.lower())[:1]
    if kind == YESNO and of and of[0] in _AUX and (not pf or pf[0] not in _AUX):
        reasons.append("yesno_form_lost")
    if kind != YESNO and _WH & set(o.split()) and not _WH & set(p.split()):
        reasons.append("open_form_lost")           # "which two ventricles?" -> "can you see any?"
    if _foreign_chars(orig, para):
        reasons.append("foreign_characters")
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


_IMG = r"(?:the |this )?(?:image|picture|scan|photo)"
# (pattern, template_a, template_b); groups are pasted verbatim, so every content word is kept.
_RULES = [
    (rf"^does {_IMG} contain (.+?)\?$", r"Is \1 shown in the image?", r"Can \1 be seen in the image?"),
    (rf"^does this image show (.+?)\?$", r"Is \1 shown in this image?", r"Can \1 be seen in this image?"),
    (rf"^does the (.+?) (?:appear|exist) in {_IMG}\?$", r"Is the \1 visible in the image?",
     r"Can the \1 be seen in the image?"),
    (r"^do the organs in the image exist in (.+?)\?$", r"Are the organs in the image located in \1?",
     r"Do the organs in the image lie in \1?"),
    (r"^do any of the organs in the picture belong to (.+?)\?$", r"Are any of the organs in the image part of \1?",
     r"Do any of the organs shown in the image belong to \1?"),
    (r"^is there (.+?)\?$", r"Is \1 present?", r"Can \1 be seen?"),
    (r"^are there (.+?)\?$", r"Are \1 present?", r"Can \1 be seen?"),
    (r"^is (.+?) present\?$", r"Is there \1?", r"Can \1 be seen?"),
    (r"^are (.+?) present\?$", r"Are there \1?", r"Can \1 be seen?"),
    (r"^does (?!this image\b|the picture\b)(.+?) show (.+?)\?$", r"Is \2 shown in \1?", r"Can \2 be seen in \1?"),
]


def rule_paraphrase(question: str, style: str) -> str:
    """template_a / template_b rewrite of a yes/no question, or "" when no rule applies."""
    q = " ".join(question.strip().split())
    for pat, ta, tb in _RULES:
        m = re.match(pat, q, flags=re.I)
        if m:
            out = m.expand(ta if style == "template_a" else tb)
            return out[:1].upper() + out[1:]
    return ""


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
        if style == "frame":
            return f"In this image, {_lowerfirst(q)}"
        if style == "reorder":
            return f"{q.rstrip('?')}, in this image?"
        if style == "opening":
            return f"Tell me: {_lowerfirst(q)}"
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
    for it, k, style in pairs:
        if style in RULE_STYLES:
            texts[(it.sample_id, k)] = rule_paraphrase(it.question, style)
    pairs = [(it, k, style) for it, k, style in pairs if style not in RULE_STYLES]
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


def load_judgments(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Semantic-equivalence verdicts per unique (original, paraphrase) pair: {pair: (y|n, reason)}.
    Written once by an LLM judge over every rule-valid pair (docs/PREREGISTRATION.md 4.5) and committed with the manifest."""
    with open(path, encoding="utf-8", newline="") as f:
        return {(r["original"].strip(), r["paraphrase"].strip()): (r["verdict"].strip().lower(), r["reason"])
                for r in csv.DictReader(f)}


def _validate_item(it: Item, rows: dict[int, dict], cfg: dict,
                   judgments: dict[tuple[str, str], tuple[str, str]] | None = None) -> None:
    """(Re)apply the current rules to every candidate of one question, in k order, in place.
    Validity is recomputed from the stored text each time, so a stricter validator applies to an
    existing paraphrase file without regenerating it. With `judgments`, a rule-valid paraphrase must
    also have been judged equivalent; one that was never judged is rejected too (fail closed)."""
    excluded = set(cfg["paraphrase"].get("exclude_styles") or [])
    seen: set[str] = set()                    # a repeat of an earlier valid one is not a second, independent test
    for k in sorted(rows):
        r = rows[k]
        reasons = validate(it.question, r["text"], it.answer_type, it.answer, cfg["paraphrase"]["min_edit_ratio"])
        if r["style"] in excluded:
            reasons.append("style_excluded")
        if judgments is not None and not reasons:
            verdict = judgments.get((it.question.strip(), r["text"].strip()))
            if verdict is None:
                reasons.append("not_judged")
            elif verdict[0] != "y":
                reasons.append("judge_rejected")
        if not reasons and normalize(r["text"]) in seen:
            reasons.append("duplicate")
        if not reasons:
            seen.add(normalize(r["text"]))
        r["valid"], r["reasons"] = not reasons, reasons


def _revalidate(items: list[Item], done: dict, cfg: dict, judgments=None) -> None:
    by_item: dict[str, dict[int, dict]] = {}
    for (sid, k), r in done.items():
        by_item.setdefault(sid, {})[k] = r
    for it in items:
        if it.sample_id in by_item:
            _validate_item(it, by_item[it.sample_id], cfg, judgments)


def generate(items: list[Item], cfg: dict, out_path: Path, backend=None, block: int = 256) -> int:
    """Resumable: (sample_id, k) pairs already in out_path are kept and skipped. Written to disk
    after every block of `block` questions, so a crash loses at most one block. Every call
    re-validates all stored candidates with the current rules (no model needed for that)."""
    n = cfg["paraphrase"]["n"]
    jname = cfg["paraphrase"].get("judgments")
    judgments = load_judgments(out_path.parent / jname) if jname else None
    done = {(r["sample_id"], r["k"]): r for r in read_jsonl(out_path, tolerant=True)} if out_path.exists() else {}
    pending = [it for it in items if any((it.sample_id, k) not in done for k in range(1, n + 1))]
    needs_lm = any((it.sample_id, k) not in done for it in pending
                   for k, style in enumerate(STYLES[:n], start=1) if style not in RULE_STYLES)
    if needs_lm:                              # otherwise do not even load the paraphrase model
        backend = backend or make_backend(cfg)
    for start in range(0, len(pending), block):
        part = pending[start:start + block]
        pairs = [(it, k, style) for it in part for k, style in enumerate(STYLES[:n], start=1)
                 if (it.sample_id, k) not in done]
        texts = _candidates(backend, pairs)
        for it, k, style in pairs:
            done[(it.sample_id, k)] = {"sample_id": it.sample_id, "k": k, "style": style,
                                       "text": texts[(it.sample_id, k)], "valid": False, "reasons": []}
        _revalidate(part, done, cfg, judgments)
        write_jsonl(out_path, (done[key] for key in sorted(done)))
        print(f"[paraphrase] {min(start + block, len(pending))}/{len(pending)} questions generated", flush=True)
    _revalidate(items, done, cfg, judgments)
    return write_jsonl(out_path, (done[key] for key in sorted(done)))


def load_valid(path: Path, n: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if path.exists():
        for r in read_jsonl(path):
            if r["valid"] and r["k"] <= n:
                out.setdefault(r["sample_id"], []).append(r)
    return out


def export_audit_sheet(items: list[Item], para_path: Path, out_csv: Path, n: int, seed: int) -> int:
    """Random sample of valid paraphrases to judge 'same meaning? y/n'."""
    by_id = {it.sample_id: it for it in items}
    # Never re-show paraphrases rated in an earlier round (paraphrase_audit_round*.csv next to out_csv):
    # a round that judges a changed validator must be a fresh sample.
    rated = set()
    for prev in sorted(out_csv.parent.glob("paraphrase_audit_round*.csv")):
        rated |= {(r["sample_id"], int(r["k"])) for r in csv.DictReader(open(prev, encoding="utf-8"))}
    rows = [r for r in read_jsonl(para_path) if r["valid"] and (r["sample_id"], r["k"]) not in rated]
    random.Random(seed).shuffle(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "k", "style", "original", "paraphrase", "same_meaning_A", "same_meaning_B", "notes"])
        for r in rows[:n]:
            w.writerow([r["sample_id"], r["k"], r["style"], by_id[r["sample_id"]].question, r["text"], "", "", ""])
    return min(n, len(rows))


def score_audit(csv_path: Path) -> dict:
    """Read a filled audit sheet. Columns same_meaning_A / same_meaning_B take y/n (anything else = blank).
    Reports the acceptance rate (share judged 'same meaning') and, when B is filled, agreement and
    Cohen's kappa. Working alone, B is a blind second pass by the same person a few days later
    (intra-rater agreement) -- say so in the report."""
    from .analysis.stats import cohen_kappa

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    norm = lambda s: s.str.strip().str.lower().str[:1].map({"y": True, "n": False})  # noqa: E731
    a, b = norm(df["same_meaning_A"]), norm(df["same_meaning_B"])
    rate = lambda s: float(s.dropna().astype(bool).mean()) if s.notna().any() else float("nan")  # noqa: E731
    res = {"n_rated_A": int(a.notna().sum()), "accept_A": rate(a), "n_rated_B": int(b.notna().sum())}
    both = a.notna() & b.notna()
    if both.any():
        res.update(accept_B=rate(b), n_both=int(both.sum()),
                   agreement=float((a[both] == b[both]).mean()), kappa=cohen_kappa(a[both], b[both]))
    if "style" in df and a.notna().any():
        res["accept_A_by_style"] = (df.assign(a=a).dropna(subset=["a"]).assign(a=lambda d: d["a"].astype(float))
                                    .groupby("style")["a"].mean().round(3).to_dict())
    return res

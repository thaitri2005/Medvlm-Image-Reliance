"""Generic Hugging Face image-text-to-text backend (MedGemma, Gemma 3, SmolVLM, Qwen-VL, ...).

Everything model-specific is configuration: hf_id, dtype, quantization, whether a system turn is
supported. For yes/no questions we do ONE forward pass and read the logits of the yes/no tokens
(as arXiv 2603.20985 does); for open questions we greedy-decode.
"""
from __future__ import annotations

import math

from ..prompts import build_messages
from ..schema import YESNO
from .base import VLM, Output, Query, yesno_gold_logp

_YES = ["yes", "Yes", " yes", " Yes", "YES"]
_NO = ["no", "No", " no", " No", "NO"]


class HFChatVLM(VLM):
    def __init__(self, hf_id: str, name: str, prompt_cfg: dict, *, dtype: str = "bfloat16",
                 quantization: str | None = None, device: str = "auto", use_system: bool = True,
                 batch_size: int = 4, max_new_tokens: int = 16, revision: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.name = torch, name
        self.prompt_cfg, self.use_system = prompt_cfg, use_system
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else None

        kwargs: dict = {}
        if quantization:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16, bnb_4bit_quant_type="nf4")
        self.processor = AutoProcessor.from_pretrained(hf_id, revision=revision)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):   # keeps CPU smoke tests fast
            ip.do_image_splitting = False
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, torch_dtype=torch_dtype, revision=revision,
            device_map=None if device == "cpu" else device, **kwargs)
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

        self.yes_ids = self._single_token_ids(_YES)
        self.no_ids = self._single_token_ids(_NO)
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.pad_token_id}

    # ------------------------------------------------------------------ helpers
    def _single_token_ids(self, words: list[str]) -> list[int]:
        ids = {self.tok.encode(w, add_special_tokens=False)[0] for w in words
               if len(self.tok.encode(w, add_special_tokens=False)) == 1}
        if not ids:
            raise RuntimeError("tokenizer has no single-token yes/no variants")
        return sorted(ids)

    def _prompt(self, q: Query) -> str:
        msgs = build_messages(q.question, q.kind, q.image is not None, self.prompt_cfg, self.use_system)
        return self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def _encode(self, qs: list[Query], texts: list[str] | None = None):
        texts = texts or [self._prompt(q) for q in qs]
        kw = dict(text=texts, return_tensors="pt", padding=True)
        if qs[0].image is not None:
            kw["images"] = [[q.image.convert("RGB")] for q in qs]
        enc = self.processor(**kw)
        return enc.to(self.model.device, dtype=self.model.dtype)

    # ------------------------------------------------------------------ yes/no
    def _yesno(self, qs: list[Query]) -> list[Output]:
        torch = self.torch
        with torch.no_grad():
            logits = self.model(**self._encode(qs)).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        ent = -(lp.exp() * lp).sum(-1)
        ly = torch.logsumexp(lp[:, self.yes_ids], dim=-1)
        ln = torch.logsumexp(lp[:, self.no_ids], dim=-1)
        p_yes = torch.sigmoid(ly - ln)
        mass = torch.logaddexp(ly, ln).exp()
        outs = []
        for i, q in enumerate(qs):
            p = float(p_yes[i])
            outs.append(Output("yes" if p >= 0.5 else "no", p_yes=p, yn_mass=float(mass[i]), entropy=float(ent[i]),
                               gold_logp=yesno_gold_logp(p, q.gold) if q.gold else None))
        return outs

    # ------------------------------------------------------------------ open-ended
    def _open(self, qs: list[Query]) -> list[Output]:
        torch = self.torch
        enc = self._encode(qs)
        with torch.no_grad():
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      output_scores=True, return_dict_in_generate=True)
        new = gen.sequences[:, enc["input_ids"].shape[1]:]
        trans = self.model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
        p0 = torch.softmax(gen.scores[0].float(), dim=-1)
        ent = -(p0 * torch.log(p0.clamp_min(1e-12))).sum(-1)
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        outs = []
        for i, q in enumerate(qs):
            keep = [j for j, t in enumerate(new[i].tolist()) if t not in self.stop_ids]
            mlp = float(trans[i, keep].float().mean()) if keep else None
            out = Output(texts[i].strip(), entropy=float(ent[i]), mean_logprob=mlp)
            if q.gold:
                out.gold_logp = self._gold_logp(q)
            outs.append(out)
        return outs

    def _gold_logp(self, q: Query) -> float:
        """Teacher-forced log p(gold tokens | image, question). Excludes the end-of-turn token."""
        torch = self.torch
        prompt = self._prompt(q)
        n_gold = len(self.tok(prompt + q.gold, add_special_tokens=False).input_ids) - \
            len(self.tok(prompt, add_special_tokens=False).input_ids)
        enc = self._encode([q], texts=[prompt + q.gold])
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -n_gold - 1:-1].float()
        ids = enc["input_ids"][0, -n_gold:]
        return float(torch.log_softmax(logits, dim=-1).gather(-1, ids[:, None]).sum())

    # ------------------------------------------------------------------ public
    def generate(self, queries: list[Query]) -> list[Output]:
        results: list[Output | None] = [None] * len(queries)
        # Homogeneous sub-batches: same answer kind, and all-with or all-without an image.
        groups: dict[tuple, list[int]] = {}
        for i, q in enumerate(queries):
            groups.setdefault((q.kind == YESNO, q.image is not None), []).append(i)
        for (is_yesno, _), idx in groups.items():
            for s in range(0, len(idx), self.batch_size):
                chunk = idx[s:s + self.batch_size]
                outs = (self._yesno if is_yesno else self._open)([queries[i] for i in chunk])
                for i, o in zip(chunk, outs):
                    results[i] = o
        return results  # type: ignore[return-value]

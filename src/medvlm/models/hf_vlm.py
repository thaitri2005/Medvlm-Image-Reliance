"""Generic Hugging Face image-text-to-text backend (MedGemma, Gemma 3, SmolVLM, Qwen-VL, ...).

Everything model-specific is configuration: hf_id, dtype, quantization, whether a system turn is
supported. For yes/no questions we do ONE forward pass and read the logits of the yes/no tokens
(as arXiv 2603.20985 does); for open questions we greedy-decode and then score the gold answer.

Speed-ups (all verified to leave results unchanged by tests/test_hf_backend.py and `medvlm regress`):
  * logits_to_keep: the LM head runs on the last 1..k positions only. With a 262k-token vocabulary,
    computing logits for all ~300 prompt positions is most of the output-layer cost and memory.
  * batched teacher-forced gold scoring (was one forward per question).
  * sub-batches sorted by prompt length (less padding), original order restored.
  * fast torchvision image processor when available; whole model placed on one GPU (no dispatch hooks).
  * inference_mode, SDPA attention (selects the fused kernels on Ampere/Ada GPUs in bf16).
Deliberately NOT used: FlashAttention-2 (no padded-batch support), torch.compile (variable shapes ->
recompiles), 4-bit on GPUs that fit bf16 (changes the numbers being audited).
"""
from __future__ import annotations

from ..prompts import build_messages
from ..schema import YESNO
from .base import VLM, Output, Query, yesno_gold_logp

_YES = ["yes", "Yes", " yes", " Yes", "YES"]
_NO = ["no", "No", " no", " No", "NO"]


class HFChatVLM(VLM):
    def __init__(self, hf_id: str, name: str, prompt_cfg: dict, *, dtype: str = "bfloat16",
                 quantization: str | None = None, device: str = "auto", use_system: bool = True,
                 batch_size: int = 4, max_new_tokens: int = 16, revision: str | None = None,
                 attn_implementation: str | None = "sdpa", fast_processor: bool = True,
                 logits_to_keep: bool = True):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.name = torch, name
        self.prompt_cfg, self.use_system = prompt_cfg, use_system
        self.batch_size, self.max_new_tokens = batch_size, max_new_tokens
        self.use_ltk = logits_to_keep
        torch_dtype = getattr(torch, dtype) if dtype != "auto" else None

        kwargs: dict = {}
        if quantization:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit",
                bnb_4bit_compute_dtype=torch_dtype or torch.bfloat16, bnb_4bit_quant_type="nf4")
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation

        self.processor = self._load_processor(AutoProcessor, hf_id, revision, fast_processor)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "left"
        ip = getattr(self.processor, "image_processor", None)
        if ip is not None and hasattr(ip, "do_image_splitting"):   # keeps CPU smoke tests fast
            ip.do_image_splitting = False

        # One GPU that fits the model: load straight onto it (no accelerate dispatch hooks).
        if device == "auto" and not quantization and torch.cuda.is_available() and torch.cuda.device_count() == 1:
            device_map = {"": 0}
        else:
            device_map = None if device == "cpu" else device
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, torch_dtype=torch_dtype, revision=revision, device_map=device_map, **kwargs)
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

        self.yes_ids = self._single_token_ids(_YES)
        self.no_ids = self._single_token_ids(_NO)
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {self.tok.pad_token_id}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _load_processor(auto_processor, hf_id, revision, fast):
        if fast:
            try:
                return auto_processor.from_pretrained(hf_id, revision=revision, use_fast=True)
            except Exception:   # torchvision missing or processor has no fast variant
                pass
        return auto_processor.from_pretrained(hf_id, revision=revision)

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

    def _last_logits(self, enc, keep: int):
        """Logits of the last `keep` positions: [B, keep, V] (float32)."""
        torch = self.torch
        with torch.inference_mode():
            if self.use_ltk:
                try:
                    return self.model(**enc, logits_to_keep=keep).logits[:, -keep:, :].float()
                except TypeError:          # model without logits_to_keep: fall back, once
                    self.use_ltk = False
            return self.model(**enc).logits[:, -keep:, :].float()

    # ------------------------------------------------------------------ yes/no
    def _yesno(self, qs: list[Query]) -> list[Output]:
        torch = self.torch
        logits = self._last_logits(self._encode(qs), 1)[:, 0, :]
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
        with torch.inference_mode():
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
            outs.append(Output(texts[i].strip(), entropy=float(ent[i]), mean_logprob=mlp))
        with_gold = [i for i, q in enumerate(qs) if q.gold]
        if with_gold:
            for i, lp in zip(with_gold, self._gold_logp([qs[i] for i in with_gold])):
                outs[i].gold_logp = lp
        return outs

    def _gold_logp(self, qs: list[Query]) -> list[float | None]:
        """Teacher-forced log p(gold tokens | image, question), batched. Excludes the end-of-turn token.
        Rows are left-padded, so every gold continuation ends at the last position."""
        torch = self.torch
        prompts = [self._prompt(q) for q in qs]
        n_gold = [len(self.tok(p + q.gold, add_special_tokens=False).input_ids)
                  - len(self.tok(p, add_special_tokens=False).input_ids) for p, q in zip(prompts, qs)]
        enc = self._encode(qs, texts=[p + q.gold for p, q in zip(prompts, qs)])
        keep = max(max(n_gold), 1) + 1
        logits = self._last_logits(enc, keep)                      # [B, keep, V]
        lps = torch.log_softmax(logits, dim=-1)
        out: list[float | None] = []
        for i, n in enumerate(n_gold):
            if n <= 0:
                out.append(None)
                continue
            ids = enc["input_ids"][i, -n:]
            rows = lps[i, keep - n - 1:keep - 1]                  # predictions for the n gold tokens
            out.append(float(rows.gather(-1, ids[:, None]).sum()))
        return out

    # ------------------------------------------------------------------ public
    def generate(self, queries: list[Query]) -> list[Output]:
        results: list[Output | None] = [None] * len(queries)
        # Homogeneous sub-batches (same answer kind, all-with or all-without an image), shortest
        # prompts together so padding is minimal.
        groups: dict[tuple, list[int]] = {}
        for i, q in enumerate(queries):
            groups.setdefault((q.kind == YESNO, q.image is not None), []).append(i)
        for (is_yesno, _), idx in groups.items():
            idx.sort(key=lambda i: len(queries[i].question))
            for s in range(0, len(idx), self.batch_size):
                chunk = idx[s:s + self.batch_size]
                outs = (self._yesno if is_yesno else self._open)([queries[i] for i in chunk])
                for i, o in zip(chunk, outs):
                    results[i] = o
        return results  # type: ignore[return-value]

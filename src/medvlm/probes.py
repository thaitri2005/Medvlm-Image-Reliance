"""`medvlm probes`: robustness probes registered after the main run (docs/PREREGISTRATION.md section 15), each answering a
question a skeptical reader would ask. Outputs go to separate run folders (outputs/<model>__<probe>/main), so
the frozen main results are never touched.

  blank    the question with a BLACK image instead of no image: keeps the image format, removes the content.
           Separates "answers without visual evidence" from "confused by missing image tokens".
  abstain  no image, but the instruction allows "unknown": is answering blind a real tendency or just
           compliance with "Answer with only 'yes' or 'no'"?
  opp      the question with another image of the same dataset and modality on which the SAME question has
           the OPPOSITE correct answer: an image-reliant model must change its answer.
"""
from __future__ import annotations

import copy

from .config import Paths
from .data.sampling import is_opposite, opposite_key
from .run import Job, run
from .schema import YESNO, Item, load_items, stable_int

PROBES = ("blank", "abstain", "opp")
ABSTAIN_SUFFIX = "Answer 'yes' or 'no'. If this cannot be determined without seeing the image, answer 'unknown'."


def opposite_partners(items: list[Item]) -> dict[str, Item]:
    """sample_id -> an item on a DIFFERENT image of the same dataset and modality whose normalised question is
    identical and whose gold answer is the opposite (yes/no only). Chosen by hash, never by model output."""
    by_q: dict[tuple, list[Item]] = {}
    for it in items:
        if it.answer_type == YESNO:
            by_q.setdefault(opposite_key(it.dataset, it.modality, it.question), []).append(it)
    out = {}
    for it in items:
        if it.answer_type != YESNO:
            continue
        pool = sorted((o for o in by_q[opposite_key(it.dataset, it.modality, it.question)]
                       if o.image_id != it.image_id and is_opposite(o.answer, it.answer)),
                      key=lambda o: o.sample_id)
        if pool:
            out[it.sample_id] = pool[stable_int(f"opp|{it.sample_id}") % len(pool)]
    return out


def probe_jobs(probe: str, items: list[Item]) -> list[Job]:
    yn = [it for it in items if it.answer_type == YESNO]
    if probe == "blank":
        jobs = [Job(f"{it.sample_id}|blank", it, "blank", it.question) for it in yn]
    elif probe == "abstain":
        jobs = [Job(f"{it.sample_id}|noimg", it, "noimg", it.question) for it in yn]
    elif probe == "opp":
        partners = opposite_partners(items)
        jobs = [Job(f"{it.sample_id}|opp", it, "opp", it.question, image_file=partners[it.sample_id].image_file)
                for it in yn if it.sample_id in partners]
    else:
        raise ValueError(f"unknown probe {probe!r}; choose from {PROBES}")
    return sorted(jobs, key=lambda j: stable_int("order|" + j.item.sample_id))


def probe_config(cfg: dict, probe: str) -> dict:
    c = copy.deepcopy(cfg)
    c["model"]["name"] = f"{cfg['model']['name']}__{probe}"
    c["run"]["score_gold_logprob"] = False             # probes only need the answer and its probabilities
    if probe == "abstain":
        c["prompt"]["yesno_suffix"] = ABSTAIN_SUFFIX
        c["run"]["warn_low_mass"] = False                # abstaining is what this probe measures
    return c


def run_probes(cfg: dict, paths: Paths, model, which: list[str], max_minutes: float | None = None) -> None:
    items = load_items(paths.manifest_dir(cfg["profile"]) / "items.jsonl")
    for probe in which:
        pc = probe_config(cfg, probe)
        if hasattr(model, "prompt_cfg"):
            model.prompt_cfg = pc["prompt"]              # the abstain probe changes the instruction text
        jobs = probe_jobs(probe, items)
        print(f"[probes] === {probe}: {len(jobs)} yes/no jobs -> {paths.run_dir(pc['model']['name'], pc['profile'])}",
              flush=True)
        run(pc, paths, model, jobs=jobs, max_minutes=max_minutes)
    if hasattr(model, "prompt_cfg"):
        model.prompt_cfg = cfg["prompt"]


def run_contrast(models: list[str], overrides: list[str], hardware: str | None,
                 max_minutes: float | None = None, loader=None, paths: Paths | None = None) -> None:
    """Probe R4 (docs/PREREGISTRATION.md section 17), one model after another: fetch + verify the committed
    `contrast` manifest's images, run every question on its own image (`orig`), then on its opposite-answer
    partner (`opp`). Resumable like every run; `loader` and `paths` are for tests."""
    from .config import get_paths, load_config
    from .data.prepare import prepare, verify
    from .pipeline import _free_gpu

    if loader is None:
        from .models import load_model as loader
    for name in models:
        cfg = load_config("contrast", name, overrides, hardware=hardware)
        paths = paths or get_paths(cfg)
        if not (paths.manifest_dir("contrast") / "items.jsonl").exists():
            raise SystemExit("[contrast] manifests/contrast/items.jsonl missing: `git pull` (it is committed)")
        if verify(cfg, paths):
            prepare(cfg, paths)
        print(f"[contrast] === {name} ===", flush=True)
        model = loader(cfg)
        run(cfg, paths, model, max_minutes=max_minutes)                 # own image
        run_probes(cfg, paths, model, ["opp"], max_minutes=max_minutes)  # opposite-answer partner image
        del model
        _free_gpu()

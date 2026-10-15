"""`medvlm pipeline`: prepare -> paraphrase -> run in N parts -> analyze, in ONE process.

The model is loaded once and the parts run back to back. Everything is resumable: if the process dies,
run the same command again and finished work is skipped. `--part i` runs a single part (for spreading
parts across days or machines); a checkpoint analysis is written after every part.
"""
from __future__ import annotations

import gc

from .analysis.report import analyze
from .config import Paths
from .data.prepare import prepare, verify
from .models import load_model
from .paraphrase import STYLES, generate, load_valid
from .run import done_keys, run, select_jobs
from .schema import load_items


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def pipeline(cfg: dict, paths: Paths, *, parts: int = 2, part: int | None = None,
             max_minutes: float | None = None, paraphrase_backend=None) -> None:
    profile = cfg["profile"]
    items_path = paths.manifest_dir(profile) / "items.jsonl"

    # 1. images (skip the network entirely when the local cache already verifies against the manifest)
    if not items_path.exists() or verify(cfg, paths):
        print("[pipeline] step 1/3: images missing or unverified, fetching", flush=True)
        prepare(cfg, paths)
    else:
        print("[pipeline] step 1/3: images verified, skipping download")
    items = load_items(items_path)

    # 2. paraphrases (own model, freed before the VLM is loaded)
    para_path = paths.manifest_dir(profile) / "paraphrases.jsonl"
    print("[pipeline] step 2/3: paraphrases (skipped if already complete)", flush=True)
    generate(items, cfg, para_path, backend=paraphrase_backend)
    _free_gpu()
    ready = sum(len(v) >= cfg["analysis"]["min_paraphrases"] for v in load_valid(para_path, cfg["paraphrase"]["n"]).values())
    print(f"[pipeline] {ready}/{len(items)} questions have >= {cfg['analysis']['min_paraphrases']} valid paraphrases "
          f"(of {len(STYLES[:cfg['paraphrase']['n']])} candidates each)")

    # 3. the model, once
    print(f"[pipeline] step 3/3: loading {cfg['model'].get('hf_id', cfg['model']['name'])}", flush=True)
    model = load_model(cfg)
    todo = [part] if part is not None else list(range(parts))
    for i in todo:
        print(f"[pipeline] === part {i + 1} of {parts} ===", flush=True)
        run(cfg, paths, model, shard=(i, parts), max_minutes=max_minutes)
        print("[pipeline] checkpoint analysis", flush=True)
        analyze(cfg, paths)                       # checkpoint tables from whatever is finished so far
    jobs = select_jobs(cfg, paths, None)
    finished = done_keys(paths.run_dir(cfg["model"]["name"], profile))
    left = sum(j.key not in finished for j in jobs)
    if left:
        print(f"[pipeline] stopped with {left}/{len(jobs)} jobs left; re-run the same command to continue")
    else:
        print(f"[pipeline] done: all {len(jobs)} jobs finished; tables in results/{cfg['model']['name']}/{profile}")

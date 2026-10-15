"""Resumable inference runner.

Every (sample, condition) is one job with the key "<sample_id>|<condition>". Results are appended to
outputs/<model>/<profile>/results.<host>.jsonl after each chunk. On restart (Colab disconnect, a
different machine, an extra shard) any key already present in any results*.jsonl is skipped.
"""
from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .config import ROOT, Paths
from .models.base import VLM, Query
from .paraphrase import load_valid
from .schema import Item, load_items, read_jsonl, stable_int

IMAGE_CONDITIONS = ("orig", "noimg", "shuf", "blank")


@dataclass
class Job:
    key: str
    item: Item
    condition: str
    question: str


def build_jobs(items: list[Item], paras: dict[str, list[dict]], conditions: list[str]) -> list[Job]:
    jobs = []
    for it in items:
        for c in conditions:
            if c == "shuf" and not it.shuffle_image_file:
                continue
            jobs.append(Job(f"{it.sample_id}|{c}", it, c, it.question))
        for p in paras.get(it.sample_id, []):
            jobs.append(Job(f"{it.sample_id}|para{p['k']}", it, f"para{p['k']}", p["text"]))
    return jobs


def done_keys(run_dir: Path) -> set[str]:
    keys: set[str] = set()
    for f in run_dir.glob("results*.jsonl"):
        keys.update(r["key"] for r in read_jsonl(f))
    return keys


@lru_cache(maxsize=64)
def _open(path: str) -> Image.Image:
    im = Image.open(path).convert("RGB")
    im.load()
    return im


def make_query(job: Job, paths: Paths, score_gold: bool) -> Query:
    it, c = job.item, job.condition
    orig = _open(str(paths.images / it.image_file))
    if c == "noimg":
        image = None
    elif c == "shuf":
        image = _open(str(paths.images / it.shuffle_image_file))
    elif c == "blank":
        image = Image.new("RGB", orig.size)
    else:  # orig, paraK
        image = orig
    gold = it.answer if (score_gold and c in IMAGE_CONDITIONS) else None
    return Query(job.question, it.answer_type, image, gold)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def run(cfg: dict, paths: Paths, model: VLM, *, limit: int | None = None, shard: tuple[int, int] | None = None,
        chunk: int = 16) -> int:
    profile, mname = cfg["profile"], cfg["model"]["name"]
    run_dir = paths.run_dir(mname, profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    items = load_items(paths.manifest_dir(profile) / "items.jsonl")
    if shard:
        i, n = shard
        items = [it for it in items if stable_int(it.sample_id) % n == i]
    paras = load_valid(paths.manifest_dir(profile) / "paraphrases.jsonl", cfg["paraphrase"]["n"])
    jobs = build_jobs(items, paras, list(cfg["conditions"]))

    finished = done_keys(run_dir)
    todo = [j for j in jobs if j.key not in finished]
    # Group by (has image?, answer kind) so batches are full; stable order keeps runs reproducible.
    todo.sort(key=lambda j: (j.condition == "noimg", j.item.answer_type, j.item.sample_id, j.condition))
    if limit:
        todo = todo[:limit]
    print(f"[run] {mname}/{profile}: {len(jobs)} jobs, {len(finished)} already done, {len(todo)} to run")

    meta = run_dir / "meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({
            "model": cfg["model"], "prompt": cfg["prompt"], "run": cfg["run"], "conditions": cfg["conditions"],
            "paraphrase": cfg["paraphrase"], "git_commit": _git_commit(), "python": platform.python_version(),
            "host": socket.gethostname(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2, default=str), encoding="utf-8")

    host = re.sub(r"[^A-Za-z0-9_-]", "", socket.gethostname()) or "host"
    out_path = run_dir / f"results.{host}.jsonl"
    score_gold = bool(cfg["run"].get("score_gold_logprob", True))
    n_done = 0
    with open(out_path, "a", encoding="utf-8", newline="\n") as f, tqdm(total=len(todo), unit="job") as bar:
        for s in range(0, len(todo), chunk):
            batch = todo[s:s + chunk]
            queries = [make_query(j, paths, score_gold) for j in batch]
            t0 = time.perf_counter()
            outs = model.generate(queries)
            sec = (time.perf_counter() - t0) / len(batch)      # mean seconds per job in this chunk
            for j, o in zip(batch, outs):
                f.write(json.dumps({
                    "key": j.key, "sample_id": j.item.sample_id, "condition": j.condition, "text": o.text,
                    "p_yes": o.p_yes, "yn_mass": o.yn_mass, "entropy": o.entropy, "mean_logprob": o.mean_logprob,
                    "gold_logp": o.gold_logp, "sec": round(sec, 3)}) + "\n")
            f.flush()
            n_done += len(batch)
            bar.update(len(batch))
    return n_done


def estimate(cfg: dict, paths: Paths, sec_per_job: list[float], valid_frac: float = 0.52) -> None:
    """Job counts for a profile and wall-clock at given seconds/job. Uses the real paraphrase file if
    present, otherwise assumes `valid_frac` of the candidates survive validation (pilot: 52%)."""
    profile = cfg["profile"]
    items = load_items(paths.manifest_dir(profile) / "items.jsonl")
    conds = list(cfg["conditions"])
    base = sum(1 for it in items for c in conds if not (c == "shuf" and not it.shuffle_image_file))
    para_file = paths.manifest_dir(profile) / "paraphrases.jsonl"
    if para_file.exists():
        n_para, how = sum(len(v) for v in load_valid(para_file, cfg["paraphrase"]["n"]).values()), "actual"
    else:
        n_para, how = int(len(items) * cfg["paraphrase"]["n"] * valid_frac), f"assumed {valid_frac:.0%} valid"
    print(f"[estimate] {profile}: {len(items)} questions, {base} image-condition jobs + {n_para} paraphrase jobs ({how})")
    total = base + n_para
    for s in sec_per_job:
        print(f"   at {s:g} s/job: {total * s / 3600:.1f} h  ({total * s / 3600 / 24:.1f} days)")

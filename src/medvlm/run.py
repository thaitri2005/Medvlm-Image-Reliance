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
from .scoring import same_answer

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
        keys.update(r["key"] for r in read_jsonl(f, tolerant=True))
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


def select_jobs(cfg: dict, paths: Paths, shard: tuple[int, int] | None) -> list[Job]:
    """All jobs of this profile (optionally one shard), in a fixed pseudo-random order by question.
    Whole questions stay together, so any prefix of the work is an unbiased, complete subset."""
    items = load_items(paths.manifest_dir(cfg["profile"]) / "items.jsonl")
    if shard:
        i, n = shard
        items = [it for it in items if stable_int(it.sample_id) % n == i]
    para_file = paths.manifest_dir(cfg["profile"]) / "paraphrases.jsonl"
    if not para_file.exists():
        print(f"[warn] {para_file} not found: paraphrase jobs are NOT included "
              f"(run `medvlm paraphrase` or `git pull` first)")
    paras = load_valid(para_file, cfg["paraphrase"]["n"])
    jobs = build_jobs(items, paras, list(cfg["conditions"]))
    return sorted(jobs, key=lambda j: (stable_int("order|" + j.item.sample_id), j.condition))


def status(cfg: dict, paths: Paths, n_shards: int) -> None:
    """Progress per shard, so back-to-back sessions know what is left."""
    finished = done_keys(paths.run_dir(cfg["model"]["name"], cfg["profile"]))
    print(f"[status] {cfg['model']['name']}/{cfg['profile']}, split into {n_shards} shards")
    tot_done = tot = 0
    for i in range(n_shards):
        jobs = select_jobs(cfg, paths, (i, n_shards))
        d = sum(j.key in finished for j in jobs)
        tot_done, tot = tot_done + d, tot + len(jobs)
        print(f"   shard {i}/{n_shards}: {d}/{len(jobs)} jobs ({d / max(len(jobs), 1):.0%})"
              + ("  <- done" if d == len(jobs) else ""))
    print(f"   total: {tot_done}/{tot} ({tot_done / max(tot, 1):.0%})")


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def _gpu_peak_gib() -> float | None:
    """Peak GPU memory since the last call (then reset), or None without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        return peak
    except ImportError:
        return None


class _Progress:
    """Timestamped progress lines for the log (the tqdm bar only shows in the terminal), plus the
    stop-the-line format check from PLAN.md section 11 (yn_mass < 0.9) and empty open answers."""

    def __init__(self, total: int, every_sec: float, min_yn_mass: float = 0.9):
        self.total, self.every, self.min_yn_mass = total, every_sec, min_yn_mass
        self.t0 = self.t_last = time.perf_counter()
        self.done = self.done_last = 0
        self.win_yn_min, self.win_empty, self.win_jobs = 1.0, 0, 0
        self.low_mass_total = self.empty_total = 0

    def add(self, batch: list[Job], outs) -> None:
        self.done += len(batch)
        for j, o in zip(batch, outs):
            self.win_jobs += 1
            if o.yn_mass is not None:
                self.win_yn_min = min(self.win_yn_min, o.yn_mass)
                if o.yn_mass < self.min_yn_mass:
                    self.low_mass_total += 1
                    tqdm.write(f"[warn] {j.key}: yn_mass {o.yn_mass:.3f} < {self.min_yn_mass} "
                          f"(model not answering yes/no); answer {o.text!r}")
            elif not (o.text or "").strip():
                self.win_empty += 1
                self.empty_total += 1
        now = time.perf_counter()
        if now - self.t_last >= self.every or self.done == len(batch) or self.done == self.total:
            self.report(now)

    def report(self, now: float) -> None:
        el = now - self.t0
        rate = el / max(self.done, 1)
        recent = (now - self.t_last) / max(self.done - self.done_last, 1)
        left = (self.total - self.done) * recent
        peak = _gpu_peak_gib()
        tqdm.write(f"[progress] {self.done}/{self.total} ({self.done / max(self.total, 1):.1%}) | "
              f"{recent:.2f} s/job now, {rate:.2f} avg | elapsed {_fmt_dur(el)}, ETA {_fmt_dur(left)} "
              f"(~{time.strftime('%H:%M', time.localtime(time.time() + left))})"
              + (f" | GPU peak {peak:.1f} GiB" if peak is not None else "")
              + f" | yn_mass min {self.win_yn_min:.3f}, empty open {self.win_empty}/{self.win_jobs}")
        self.t_last, self.done_last = now, self.done
        self.win_yn_min, self.win_empty, self.win_jobs = 1.0, 0, 0

    def summary(self) -> None:
        el = time.perf_counter() - self.t0
        print(f"[run] finished {self.done} jobs in {_fmt_dur(el)} ({el / max(self.done, 1):.2f} s/job); "
              f"yn_mass < {self.min_yn_mass}: {self.low_mass_total}, empty open answers: {self.empty_total}")


def run(cfg: dict, paths: Paths, model: VLM, *, limit: int | None = None, shard: tuple[int, int] | None = None,
        chunk: int | None = None, max_minutes: float | None = None, log_every_sec: float = 120) -> int:
    profile, mname = cfg["profile"], cfg["model"]["name"]
    chunk = chunk or max(16, 8 * int(cfg["run"]["batch_size"]))
    run_dir = paths.run_dir(mname, profile)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(cfg, paths, shard)
    finished = done_keys(run_dir)
    todo = [j for j in jobs if j.key not in finished]
    if limit:
        todo = todo[:limit]
    print(f"[run] {mname}/{profile}" + (f" shard {shard[0]}/{shard[1]}" if shard else "")
          + f": {len(jobs)} jobs, {sum(j.key in finished for j in jobs)} already done, "
          f"{len(todo)} to run (batch {cfg['run']['batch_size']}, chunk {chunk})")
    by_cond: dict[str, int] = {}
    for j in todo:
        c = "paraK" if j.condition.startswith("para") else j.condition
        by_cond[c] = by_cond.get(c, 0) + 1
    if todo:
        print(f"[run] to run by condition: {by_cond}")

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
    print(f"[run] writing to {out_path}")
    n_done, t_start = 0, time.perf_counter()
    if out_path.exists() and out_path.stat().st_size and not out_path.read_bytes().endswith(b"\n"):
        with open(out_path, "ab") as f:          # previous run died mid-line: close it so records never glue together
            f.write(b"\n")
        print("[run] repaired a half-written last line from an interrupted run")
    prog = _Progress(len(todo), log_every_sec)
    with open(out_path, "a", encoding="utf-8", newline="\n") as f, tqdm(total=len(todo), unit="job") as bar:
        for s in range(0, len(todo), chunk):
            if max_minutes and time.perf_counter() - t_start > max_minutes * 60:
                print(f"[run] stopped after {max_minutes:g} min budget; {len(todo) - n_done} jobs left. "
                      f"Re-run the same command to continue.")
                break
            batch = todo[s:s + chunk]
            t0 = time.perf_counter()
            try:
                queries = [make_query(j, paths, score_gold) for j in batch]
                outs = model.generate(queries)
            except BaseException as e:
                # Everything before this chunk is already on disk; say exactly what was in flight.
                print(f"\n[run] {type(e).__name__} in chunk {s // chunk + 1} ({len(batch)} jobs, "
                      f"{batch[0].key} .. {batch[-1].key}); {n_done} jobs saved this session")
                if "OutOfMemory" in type(e).__name__:
                    print("[run] GPU out of memory: lower the batch with --set run.batch_size=<smaller>")
                raise
            sec = (time.perf_counter() - t0) / len(batch)      # mean seconds per job in this chunk
            for j, o in zip(batch, outs):
                f.write(json.dumps({
                    "key": j.key, "sample_id": j.item.sample_id, "condition": j.condition, "text": o.text,
                    "p_yes": o.p_yes, "yn_mass": o.yn_mass, "entropy": o.entropy, "mean_logprob": o.mean_logprob,
                    "gold_logp": o.gold_logp, "sec": round(sec, 3)}) + "\n")
            f.flush()
            n_done += len(batch)
            bar.update(len(batch))
            prog.add(batch, outs)
    prog.summary()
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


def regress(cfg: dict, paths: Paths, model: VLM, n: int = 100, min_agree: float = 0.97, max_dp: float = 0.03) -> bool:
    """Re-run `n` jobs that already have stored results and compare. Proves that a new model build,
    batch size or optimisation did not change the numbers. Needs an existing run for this model/profile."""
    stored = {}
    for f in sorted(paths.run_dir(cfg["model"]["name"], cfg["profile"]).glob("results*.jsonl")):
        for r in read_jsonl(f, tolerant=True):
            stored[r["key"]] = r
    jobs = {j.key: j for j in select_jobs(cfg, paths, None) if j.key in stored}
    keys = sorted(jobs, key=lambda k: stable_int("regress|" + k))[:n]
    if not keys:
        print("[regress] no stored results to compare against")
        return False
    outs = model.generate([make_query(jobs[k], paths, bool(cfg["run"].get("score_gold_logprob", True))) for k in keys])
    agree, dps, dlp = 0, [], []
    for k, o in zip(keys, outs):
        r = stored[k]
        agree += same_answer(jobs[k].item.answer_type, r["text"], o.text)
        if o.p_yes is not None and r.get("p_yes") is not None:
            dps.append(abs(o.p_yes - r["p_yes"]))
        if o.gold_logp is not None and r.get("gold_logp") is not None:
            dlp.append(abs(o.gold_logp - r["gold_logp"]))
    rate = agree / len(keys)
    mdp = sum(dps) / len(dps) if dps else 0.0
    ok = rate >= min_agree and mdp <= max_dp
    print(f"[regress] {len(keys)} jobs: same answer {rate:.1%} (need >= {min_agree:.0%}); "
          f"mean |dP(yes)| {mdp:.4f} over {len(dps)} yes/no (need <= {max_dp}); "
          f"mean |d gold_logp| {sum(dlp) / len(dlp) if dlp else 0.0:.3f} over {len(dlp)}")
    print("[regress] PASS" if ok else "[regress] FAIL: results differ from the stored run")
    return ok

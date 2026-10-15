"""`medvlm bench`: find the fastest batch size that fits THIS GPU (about 2-3 minutes).

Uses real manifest questions in the `orig` condition (with gold scoring), warms up first, then times
each batch size and records peak GPU memory. Stops at the first out-of-memory size. The numbers
answer "how long will `main` take here" far better than any spec-sheet guess.
"""
from __future__ import annotations

import time

from .config import Paths
from .models.base import VLM
from .run import Job, make_query
from .schema import load_items, stable_int


def bench(cfg: dict, paths: Paths, model: VLM, sizes: list[int] | None = None, n: int = 32) -> dict | None:
    sizes = sizes or [1, 2, 4, 8, 16]
    items = sorted(load_items(paths.manifest_dir(cfg["profile"]) / "items.jsonl"),
                   key=lambda it: stable_int("bench|" + it.sample_id))
    half = n // 2
    chosen = [it for it in items if it.answer_type == "yesno"][:half] + [it for it in items if it.answer_type == "open"][:half]
    queries = [make_query(Job(it.sample_id, it, "orig", it.question), paths, True) for it in chosen]
    try:
        import torch
        cuda = torch.cuda.is_available()
    except ImportError:
        torch, cuda = None, False

    def sync():
        if cuda:
            torch.cuda.synchronize()

    if hasattr(model, "batch_size"):
        model.batch_size = sizes[0]
    model.generate(queries[:4])                       # warm-up (kernel selection, caches), not timed
    sync()
    rows = []
    for bs in sizes:
        if hasattr(model, "batch_size"):
            model.batch_size = bs
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            t0 = time.perf_counter()
            model.generate(queries)
            sync()
            dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            if "out of memory" not in str(e).lower():
                raise
            rows.append({"batch_size": bs, "sec_per_job": None, "peak_gb": None, "status": "OOM"})
            if cuda:
                torch.cuda.empty_cache()
            break
        peak = torch.cuda.max_memory_allocated() / 2**30 if cuda else None
        rows.append({"batch_size": bs, "sec_per_job": dt / len(queries), "peak_gb": peak, "status": "ok"})
    print(f"[bench] {len(queries)} real questions (half yes/no, half open, with gold scoring)")
    for r in rows:
        sp = f"{r['sec_per_job']:.3f}" if r["sec_per_job"] else "  -  "
        pk = f"{r['peak_gb']:.1f} GB" if r["peak_gb"] else "  -  "
        print(f"   batch {r['batch_size']:>2}: {sp} s/job   peak {pk}   {r['status']}")
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return None
    best = min(ok, key=lambda r: r["sec_per_job"])
    print(f"[bench] fastest: batch_size={best['batch_size']} at {best['sec_per_job']:.3f} s/job "
          f"-> use  --set run.batch_size={best['batch_size']}")
    print(f"        (mix here is 50% open; `main` is ~45% open, so expect a similar figure)")
    return best

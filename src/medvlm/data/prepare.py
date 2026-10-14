"""`medvlm prepare`: sample questions -> manifest (committed) -> export images (not committed).

Re-running with an existing manifest never re-samples; it only fetches images and verifies their
sha1, which is what makes "git pull on another machine and continue" safe.
"""
from __future__ import annotations

import collections
import json
import platform
from pathlib import Path

from ..config import Paths
from ..schema import Item, load_items, write_jsonl
from .sampling import assign_shuffle_partners, stratified_sample
from .sources import Source, export_image, get_source


def _pool_index(cfg: dict, paths: Paths, sources: dict[str, Source]) -> dict[str, dict]:
    """image_id -> a pool row that knows how to fetch that image, over every dataset spec."""
    index: dict[str, dict] = {}
    for spec in cfg["datasets"]:
        src = sources.setdefault(spec["name"], get_source(spec["name"], paths))
        for row in src.rows(spec["split"], spec.get("pool_limit")):
            index.setdefault(row["image_id"], row)
    return index


def prepare(cfg: dict, paths: Paths, rebuild: bool = False) -> list[Item]:
    items_path = paths.manifest_dir(cfg["profile"]) / "items.jsonl"
    sources: dict[str, Source] = {}
    fresh = rebuild or not items_path.exists()

    if fresh:
        rows: list[dict] = []
        for spec in cfg["datasets"]:
            src = sources.setdefault(spec["name"], get_source(spec["name"], paths))
            pool = src.rows(spec["split"], spec.get("pool_limit"))
            rows += stratified_sample(pool, spec.get("n"), spec.get("stratify", []), cfg["seed"],
                                      spec.get("allocation", "equal"))
        assign_shuffle_partners(rows, cfg["seed"])
        items = [Item.from_dict({**r, "image_file": "", "shuffle_image_id": r["_shuffle_image_id"]}) for r in rows]
    else:
        items = load_items(items_path)

    index = _pool_index(cfg, paths, sources)
    exported: dict[str, tuple[str, str]] = {}

    def ensure(image_id: str) -> tuple[str, str]:
        if image_id not in exported:
            row = index[image_id]
            exported[image_id] = export_image(sources[row["dataset"]], row, paths)
        return exported[image_id]

    for it in items:
        rel, sha = ensure(it.image_id)
        if it.image_sha1 and it.image_sha1 != sha:
            raise RuntimeError(f"{it.image_id}: image bytes differ from the manifest "
                               f"(dataset mirror changed?). expected {it.image_sha1}, got {sha}")
        it.image_file, it.image_sha1 = rel, sha
        if it.shuffle_image_id:
            it.shuffle_image_file = ensure(it.shuffle_image_id)[0]

    if fresh:
        write_jsonl(items_path, (it.to_dict() for it in items))
        meta = {
            "profile": cfg["profile"], "seed": cfg["seed"], "n_items": len(items),
            "n_images": len({it.image_id for it in items}),
            "datasets": cfg["datasets"], "python": platform.python_version(),
            "counts": summarize(items),
        }
        (items_path.parent / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return items


def summarize(items: list[Item]) -> dict:
    c = collections.Counter((it.dataset, it.split, it.modality, it.answer_type) for it in items)
    return {"|".join(k): v for k, v in sorted(c.items())}


def verify(cfg: dict, paths: Paths) -> list[str]:
    """Problems found with the local image cache vs. the manifest (empty list = healthy)."""
    import hashlib

    problems = []
    items = load_items(paths.manifest_dir(cfg["profile"]) / "items.jsonl")
    seen: set[str] = set()
    for it in items:
        for rel, want in ((it.image_file, it.image_sha1), (it.shuffle_image_file, "")):
            if not rel or rel in seen:
                continue
            seen.add(rel)
            f = Path(paths.images / rel)
            if not f.exists():
                problems.append(f"missing: {rel}")
            elif want and hashlib.sha1(f.read_bytes()).hexdigest() != want:
                problems.append(f"sha1 mismatch: {rel}")
    return problems

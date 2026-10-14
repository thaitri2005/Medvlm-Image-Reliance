"""Deterministic stratified sampling and shuffle-partner assignment.

Everything is seeded through stable_int (sha1), never hash(), so the same config yields the same
manifest on every machine and Python version.
"""
from __future__ import annotations

import random
from collections import defaultdict

from ..schema import stable_int


def allocate(sizes: dict[tuple, int], n: int, mode: str = "equal") -> dict[tuple, int]:
    """Split a budget n over strata without exceeding any stratum's size."""
    keys = sorted(sizes)
    quota = {k: 0 for k in keys}
    open_keys = [k for k in keys if sizes[k] > 0]
    remaining = min(n, sum(sizes.values()))
    while remaining > 0 and open_keys:
        if mode == "proportional":
            total = sum(sizes[k] - quota[k] for k in open_keys)
            share = {k: remaining * (sizes[k] - quota[k]) / total for k in open_keys}
        else:
            share = {k: remaining / len(open_keys) for k in open_keys}
        given = 0
        for k in open_keys:
            take = min(int(share[k]), sizes[k] - quota[k])
            quota[k] += take
            given += take
        remaining -= given
        if given == 0:  # hand out the leftovers one by one in a fixed order
            for k in open_keys:
                if remaining and quota[k] < sizes[k]:
                    quota[k] += 1
                    remaining -= 1
        open_keys = [k for k in open_keys if quota[k] < sizes[k]]
    return quota


def stratified_sample(rows: list[dict], n: int | None, keys: list[str], seed: int,
                      mode: str = "equal") -> list[dict]:
    rows = sorted(rows, key=lambda r: r["sample_id"])
    if n is None or n >= len(rows):
        return rows
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        strata[tuple(r[k] for k in keys)].append(r)
    quota = allocate({k: len(v) for k, v in strata.items()}, n, mode)
    picked: list[dict] = []
    for k, members in strata.items():
        rng = random.Random(seed ^ stable_int("|".join(map(str, k))))
        picked.extend(rng.sample(members, quota[k]))
    return sorted(picked, key=lambda r: r["sample_id"])


def assign_shuffle_partners(rows: list[dict], seed: int) -> None:
    """Set _shuffle_image_id on each row: a different image from the same dataset and modality
    (falling back to the same dataset). Chosen only among images present in the sample so that no
    extra images need downloading."""
    by_group: dict[tuple, set] = defaultdict(set)
    by_dataset: dict[str, set] = defaultdict(set)
    for r in rows:
        by_group[(r["dataset"], r["modality"], r["region"] if r["modality"] == "radiology" else "")].add(r["image_id"])
        by_dataset[r["dataset"]].add(r["image_id"])
    for r in rows:
        grp = (r["dataset"], r["modality"], r["region"] if r["modality"] == "radiology" else "")
        pool = sorted(by_group[grp] - {r["image_id"]}) or sorted(by_dataset[r["dataset"]] - {r["image_id"]})
        r["_shuffle_image_id"] = pool[stable_int(f"{seed}:{r['sample_id']}") % len(pool)] if pool else ""

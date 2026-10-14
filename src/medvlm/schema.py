"""Data records shared by every stage. Manifests are plain JSONL so they diff and review well."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator

YESNO = "yesno"
OPEN = "open"


def stable_int(s: str) -> int:
    """Process-independent hash. Python's built-in hash() is salted per run and would silently
    change sampling / shuffle partners between machines."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:12], 16)


@dataclass
class Item:
    """One question about one image. A manifest row; images live outside git."""

    sample_id: str            # "<dataset>-<split>-<qid>", stable across machines
    dataset: str              # slake | vqa_rad | pathvqa
    split: str
    image_id: str
    image_file: str           # relative to <data>/images
    question: str
    answer: str
    answer_type: str          # YESNO | OPEN  (SLAKE "CLOSED" multiple-choice-ish answers count as OPEN)
    modality: str             # ct | mri | xray | pathology | radiology (VQA-RAD, until labelled)
    region: str               # body region / organ if the dataset provides it, else ""
    qgroup: str               # presence | recognition | localization | abnormality | attribute | knowledge | other
    qtype_raw: str            # dataset-native question type, kept for traceability
    shuffle_image_id: str = ""    # partner image for the `shuf` condition (same dataset + modality)
    shuffle_image_file: str = ""
    image_sha1: str = ""      # integrity check for re-downloaded images (`medvlm verify`)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_items(path: Path) -> list[Item]:
    return [Item.from_dict(d) for d in read_jsonl(path)]

"""Dataset sources: fetch public annotations, normalise them to the unified row schema, and
export individual images on demand.

Each source returns *rows* (dicts with the Item fields minus shuffle/sha) and can write the bytes
of any image it described. Nothing here is model-specific.

Licences: SLAKE CC-BY-4.0, VQA-RAD CC0 1.0, PathVQA MIT. Cite the original papers in the report.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import requests

from ..config import Paths
from ..schema import OPEN, YESNO

SLAKE_BASE = "https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main"
VQARAD_JSON = "https://osf.io/download/6qdas/"
VQARAD_IMG_LIST = "https://api.osf.io/v2/nodes/89kps/files/osfstorage/5b21453986d8510011c277bc/"
PATHVQA_REPO = "flaviagiammarino/path-vqa"

_SLAKE_QGROUP = {
    "Modality": "recognition", "Plane": "recognition", "Organ": "recognition",
    "Position": "localization", "Abnormality": "abnormality",
    "Size": "attribute", "Color": "attribute", "Shape": "attribute", "Quantity": "attribute",
    "KG": "knowledge",
}
_VQARAD_QGROUP = {
    "PRES": "presence", "POS": "localization", "ABN": "abnormality",
    "MODALITY": "recognition", "PLANE": "recognition", "ORGAN": "recognition",
    "SIZE": "attribute", "COLOR": "attribute", "ATTRIB": "attribute", "COUNT": "attribute",
    "OTHER": "other",
}
_SLAKE_MODALITY = {"CT": "ct", "MRI": "mri", "X-Ray": "xray"}


def download(url: str, dest: Path, *, attempts: int = 5) -> Path:
    """Atomic, cached, retrying download (OSF in particular returns sporadic 5xx)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return dest
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout):
            if attempt == attempts:
                raise
            time.sleep(2 ** attempt)
    return dest  # unreachable


def answer_kind(answer: str) -> str:
    return YESNO if str(answer).strip().lower() in ("yes", "no") else OPEN


def _ext_from_bytes(data: bytes) -> str:
    from PIL import Image

    fmt = (Image.open(io.BytesIO(data)).format or "PNG").lower()
    return {"jpeg": "jpg"}.get(fmt, fmt)


class Source:
    name = ""

    def __init__(self, paths: Paths):
        self.paths = paths
        self._rows: dict[tuple, list[dict]] = {}

    def rows(self, split: str, pool_limit: int | None = None) -> list[dict]:
        key = (split, pool_limit)
        if key not in self._rows:
            self._rows[key] = self._load(split, pool_limit)
        return self._rows[key]

    def _load(self, split: str, pool_limit: int | None) -> list[dict]:  # pragma: no cover
        raise NotImplementedError

    def image_bytes(self, row: dict) -> bytes:  # pragma: no cover
        raise NotImplementedError


class Slake(Source):
    name = "slake"

    def _json(self, split: str) -> list[dict]:
        fname = {"train": "train.json", "validation": "validation.json", "test": "test.json"}[split]
        p = download(f"{SLAKE_BASE}/{fname}", self.paths.raw / "slake" / fname)
        return json.loads(p.read_text(encoding="utf-8"))

    def _load(self, split, pool_limit):
        out = []
        for r in self._json(split):
            if r.get("q_lang") != "en":
                continue
            ans = str(r["answer"]).strip()
            out.append({
                "sample_id": f"slake-{split}-{r['qid']}",
                "dataset": self.name, "split": split,
                "image_id": f"slake-{r['img_id']}",
                "question": r["question"].strip(), "answer": ans,
                "answer_type": answer_kind(ans),
                "modality": _SLAKE_MODALITY.get(r["modality"], r["modality"].lower()),
                "region": str(r.get("location", "")),
                "qgroup": _SLAKE_QGROUP.get(r["content_type"], "other"),
                "qtype_raw": r["content_type"],
                "_src": r["img_name"],
            })
        return out[:pool_limit] if pool_limit else out

    def image_bytes(self, row):
        z = download(f"{SLAKE_BASE}/imgs.zip", self.paths.raw / "slake" / "imgs.zip")
        with zipfile.ZipFile(z) as zf:
            return zf.read(f"imgs/{row['_src']}")


class VqaRad(Source):
    name = "vqa_rad"

    def _load(self, split, pool_limit):
        p = download(VQARAD_JSON, self.paths.raw / "vqa_rad" / "VQA_RAD_Dataset_Public.json")
        out = []
        for r in json.loads(p.read_text(encoding="utf-8")):
            is_test = str(r["phrase_type"]).startswith("test_")
            if (split == "test") != is_test:
                continue
            ans = str(r["answer"]).strip()
            stem = Path(r["image_name"]).stem
            qt = str(r["question_type"]).split(",")[0].strip()
            out.append({
                "sample_id": f"vqa_rad-{split}-{r['qid']}",
                "dataset": self.name, "split": split,
                "image_id": f"vqa_rad-{stem}",
                "question": r["question"].strip(), "answer": ans,
                "answer_type": answer_kind(ans),
                "modality": "radiology",   # not annotated in VQA-RAD; labelled later (see docs/PREREGISTRATION.md, Phase 2)
                "region": str(r["image_organ"]).lower(),
                "qgroup": _VQARAD_QGROUP.get(qt, "other"),
                "qtype_raw": str(r["question_type"]),
                "phrase_type": str(r["phrase_type"]),
                "_src": r["image_name"],
            })
        return out[:pool_limit] if pool_limit else out

    def _image_urls(self) -> dict[str, str]:
        cache = self.paths.raw / "vqa_rad" / "image_urls.json"
        if cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        urls: dict[str, str] = {}
        page = f"{VQARAD_IMG_LIST}?page%5Bsize%5D=100"
        while page:
            r = requests.get(page, timeout=60)
            r.raise_for_status()
            body = r.json()
            for e in body["data"]:
                if e["attributes"]["kind"] == "file":
                    urls[e["attributes"]["name"]] = e["links"]["download"]
            page = body["links"].get("next")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(urls), encoding="utf-8")
        return urls

    def image_bytes(self, row):
        name = row["_src"]
        dest = download(self._image_urls()[name], self.paths.raw / "vqa_rad" / "images" / name)
        return dest.read_bytes()


class PathVqa(Source):
    """HF parquet mirror. It has no image ids, so ids are the sha1 of the image bytes."""

    name = "pathvqa"

    def _dataset(self, split):
        from datasets import Image, load_dataset

        return load_dataset(PATHVQA_REPO, split=split).cast_column("image", Image(decode=False))

    def _load(self, split, pool_limit):
        ds = self._dataset(split)
        n = min(len(ds), pool_limit) if pool_limit else len(ds)
        out, self._bytes = [], getattr(self, "_bytes", {})
        for i in range(n):
            r = ds[i]
            data = r["image"]["bytes"]
            iid = "pathvqa-" + hashlib.sha1(data).hexdigest()[:12]
            self._bytes.setdefault(iid, data)
            ans = str(r["answer"]).strip()
            kind = answer_kind(ans)
            first = r["question"].strip().lower().split()[0] if r["question"].strip() else ""
            if kind == YESNO:
                qgroup = "presence"
            else:
                qgroup = {"what": "recognition", "where": "localization", "how": "attribute"}.get(first, "other")
            out.append({
                "sample_id": f"pathvqa-{split}-{i}",
                "dataset": self.name, "split": split, "image_id": iid,
                "question": r["question"].strip(), "answer": ans,
                "answer_type": kind, "modality": "pathology", "region": "",
                "qgroup": qgroup, "qtype_raw": first, "_src": iid,
            })
        return out

    def image_bytes(self, row):
        return self._bytes[row["_src"]]


SOURCES = {"slake": Slake, "vqa_rad": VqaRad, "pathvqa": PathVqa}


def get_source(name: str, paths: Paths) -> Source:
    return SOURCES[name](paths)


def export_image(source: Source, row: dict, paths: Paths) -> tuple[str, str]:
    """Write one image under <data>/images and return (relative file name, sha1)."""
    data = source.image_bytes(row)
    rel = f"{row['dataset']}/{row['image_id']}.{_ext_from_bytes(data)}"
    dest = paths.images / rel
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return rel, hashlib.sha1(data).hexdigest()

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from medvlm.config import Paths, load_config
from medvlm.schema import Item, write_jsonl


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(root=tmp_path, data=tmp_path / "data", raw=tmp_path / "data" / "raw",
                 images=tmp_path / "data" / "images", manifests=tmp_path / "manifests",
                 outputs=tmp_path / "outputs", results=tmp_path / "results")


@pytest.fixture
def cfg() -> dict:
    c = load_config("smoke", "mock")
    c["analysis"]["bootstrap_B"] = 60          # keeps tests fast; statistics are tested separately
    return c


def make_items(paths: Paths, profile: str = "smoke", n_images: int = 6, per_image: int = 3) -> list[Item]:
    """Synthetic manifest with real (tiny) image files so the whole runner can execute."""
    items = []
    for i in range(n_images):
        rel = f"slake/img{i}.png"
        (paths.images / "slake").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (i * 30, 10, 200 - i * 20)).save(paths.images / rel)
        for q in range(per_image):
            yesno = q % 2 == 0
            items.append(Item(
                sample_id=f"slake-test-{i}-{q}", dataset="slake", split="test", image_id=f"slake-{i}",
                image_file=rel, question=("Is there a lesion in the left lung?" if yesno else f"What organ is shown {q}?"),
                answer="yes" if yesno else "lung", answer_type="yesno" if yesno else "open",
                modality="ct" if i % 2 else "mri", region="", qgroup="presence" if yesno else "recognition",
                qtype_raw="x", shuffle_image_id=f"slake-{(i + 1) % n_images}",
                shuffle_image_file=f"slake/img{(i + 1) % n_images}.png"))
    write_jsonl(paths.manifest_dir(profile) / "items.jsonl", (it.to_dict() for it in items))
    return items

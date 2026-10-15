"""Layered YAML config. Later layers win; see the header of configs/base.yaml."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("MEDVLM_ROOT") or Path(__file__).resolve().parents[2])


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _set_dotted(cfg: dict, dotted: str, raw: str) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(profile: str, model: str | None = None, overrides: list[str] | None = None,
                hardware: str | None = None) -> dict:
    cfg = _read(ROOT / "configs" / "base.yaml")
    cfg = _merge(cfg, _read(ROOT / "configs" / "profiles" / f"{profile}.yaml"))
    if model:
        cfg = _merge(cfg, _read(ROOT / "configs" / "models" / f"{model}.yaml"))
    hardware = hardware or os.environ.get("MEDVLM_HARDWARE")
    if hardware:
        cfg = _merge(cfg, _read(ROOT / "configs" / "hardware" / f"{hardware}.yaml"))
        cfg["hardware"] = hardware
    local = ROOT / "configs" / "local.yaml"
    if local.exists():
        cfg = _merge(cfg, _read(local))
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set_dotted(cfg, key.strip(), raw)
    cfg["profile"] = profile
    return cfg


@dataclass(frozen=True)
class Paths:
    root: Path
    data: Path
    raw: Path          # downloaded archives / annotation files
    images: Path       # exported images referenced by the manifests
    manifests: Path    # committed: sampled items + paraphrases per profile
    outputs: Path      # NOT committed: raw per-sample model outputs
    results: Path      # committed: small aggregated tables / figures

    def manifest_dir(self, profile: str) -> Path:
        return self.manifests / profile

    def run_dir(self, model: str, profile: str) -> Path:
        return self.outputs / model / profile

    def result_dir(self, model: str, profile: str) -> Path:
        return self.results / model / profile


def get_paths(cfg: dict) -> Paths:
    p = cfg["paths"]
    data = Path(os.environ.get("MEDVLM_DATA_DIR") or ROOT / p["data_dir"])
    outputs = Path(os.environ.get("MEDVLM_OUTPUTS_DIR") or ROOT / p["outputs_dir"])
    return Paths(
        root=ROOT,
        data=data,
        raw=data / "raw",
        images=data / "images",
        manifests=ROOT / p["manifests_dir"],
        outputs=outputs,
        results=ROOT / p["results_dir"],
    )

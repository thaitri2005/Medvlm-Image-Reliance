from __future__ import annotations

from .base import VLM


def load_model(cfg: dict) -> VLM:
    m = cfg["model"]
    if m["name"] == "mock":
        from .mock import MockVLM

        return MockVLM()
    from .hf_vlm import HFChatVLM

    return HFChatVLM(
        m["hf_id"], m["name"], cfg["prompt"], dtype=m.get("dtype", "bfloat16"),
        quantization=m.get("quantization"), device=m.get("device", "auto"),
        use_system=m.get("system_prompt", True), batch_size=cfg["run"]["batch_size"],
        max_new_tokens=cfg["run"]["max_new_tokens"], revision=m.get("revision"))

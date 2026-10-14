"""One prompt template for every condition: only the image is added or removed."""
from __future__ import annotations

from .schema import YESNO


def user_text(question: str, kind: str, prompt_cfg: dict) -> str:
    suffix = prompt_cfg["yesno_suffix"] if kind == YESNO else prompt_cfg["open_suffix"]
    return f"{question.strip()}\n{suffix}"


def build_messages(question: str, kind: str, has_image: bool, prompt_cfg: dict, use_system: bool) -> list[dict]:
    content: list[dict] = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": user_text(question, kind, prompt_cfg)})
    messages = [{"role": "user", "content": content}]
    if use_system and prompt_cfg.get("system"):
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": prompt_cfg["system"]}]})
    return messages

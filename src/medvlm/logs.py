"""Session log files: everything printed to the terminal is also written, timestamped, to
logs/<YYYYmmdd-HHMMSS>_<command>_<model>_<profile>.log.

Progress bars redraw with carriage returns; only the final state of each bar reaches the file, so the
log stays readable while the terminal keeps its live bars.
"""
from __future__ import annotations

import platform
import socket
import sys
import time
from pathlib import Path

from .config import ROOT


class _Tee:
    def __init__(self, stream, log, tag: str = ""):
        self.stream, self.log, self.tag, self.buf = stream, log, tag, ""

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.buf += s
        while True:
            i = self.buf.find("\n")
            if i < 0:
                break
            line, self.buf = self.buf[:i], self.buf[i + 1:]
            line = line.rsplit("\r", 1)[-1]         # keep only the last redraw of a progress bar
            if line.strip():
                self.log.write(f"{time.strftime('%H:%M:%S')} {self.tag}{line}\n")
                self.log.flush()
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):                    # isatty, encoding, fileno, ... from the real stream
        return getattr(self.stream, name)


def _versions() -> str:
    parts = [f"python {platform.python_version()}"]
    try:
        import torch

        parts.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            parts.append(f"gpu {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
    try:
        import transformers

        parts.append(f"transformers {transformers.__version__}")
    except ImportError:
        pass
    return ", ".join(parts)


def start_log(command: str, cfg: dict, argv: list[str], log_dir: Path | None = None) -> Path:
    from .run import _git_commit

    log_dir = log_dir or ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{command}_{cfg['model']['name']}_{cfg['profile']}.log"
    path = log_dir / name
    log = open(path, "a", encoding="utf-8", newline="\n", buffering=1)
    log.write(f"# medvlm {' '.join(argv)}\n# host {socket.gethostname()}, git {_git_commit()[:12]}, "
              f"{_versions()}\n# batch_size {cfg['run'].get('batch_size')}, started "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout = _Tee(sys.stdout, log)
    sys.stderr = _Tee(sys.stderr, log, "[stderr] ")
    print(f"[log] {path}")
    return path

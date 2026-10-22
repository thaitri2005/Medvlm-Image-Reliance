"""Build report/arxiv.zip: the minimal source bundle arXiv needs to compile the paper.

Run after compiling main.tex with intermediates kept (so main.bbl exists):
    tectonic --keep-intermediates main.tex      (or: latexmk -pdf main.tex)
    python make_arxiv.py
Contains main.tex, main.bbl (arXiv does not reliably run BibTeX), the style files and only the figures
that main.tex includes. Upload the zip at https://arxiv.org/submit (see ARXIV.md).
"""
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    tex = (HERE / "main.tex").read_text(encoding="utf-8")
    bbl = HERE / "main.bbl"
    if not bbl.exists():
        print("main.bbl missing: compile with `tectonic --keep-intermediates main.tex` first")
        return 1
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    files = ["main.tex", "main.bbl", "cvpr.sty", "ieeenat_fullname.bst", *figs]
    missing = [f for f in files if not (HERE / f).exists()]
    if missing:
        print("missing:", missing)
        return 1
    out = HERE / "arxiv.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(HERE / f, f)
    size = out.stat().st_size / 2**20
    print(f"{out} ({size:.1f} MB): {', '.join(files)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

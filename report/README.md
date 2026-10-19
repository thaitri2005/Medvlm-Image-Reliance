# Report

The paper and the poster.

| File | What it is |
|---|---|
| [main.pdf](main.pdf) | The paper (CVPR author kit: `cvpr.sty`, `ieeenat_fullname.bst`), source `main.tex` + `refs.bib` |
| [poster/poster.pdf](poster/poster.pdf) | A0 portrait poster, source `poster/poster.tex` |
| `figures/` | Figures used by both, written by the analysis commands (below) |

**Building.** `tectonic main.tex` (or `latexmk -pdf main.tex`) in this folder; `tectonic poster.tex` in `poster/`.
On Overleaf, upload this folder and compile with pdfLaTeX.

**Figures.** Every figure is generated, never edited by hand:

| Figure | Written by |
|---|---|
| `fig0_teaser.png` | `python report/make_teaser.py` (examples in `teaser_examples.json`) |
| `fig1`-`fig7` | `medvlm posthoc --profile main --model medgemma4b --model-b gemma3_4b` |
| `fig8_pairs`, `fig10_by_dataset` | `medvlm replication --profile main` (after `medvlm contrast --analyze`) |
| `fig9_contrast` | `medvlm contrast --analyze` |
| `<pair>/` subfolders | `medvlm posthoc` for the replication pairs |

Each analysis figure is written as a 300 dpi PNG and a vector PDF; the paper and the poster use the PDFs. Titled
versions for the README and RESULTS go to `results/`. Every number in the text comes from the committed tables
under `results/` ([docs/RESULTS.md](../docs/RESULTS.md)).

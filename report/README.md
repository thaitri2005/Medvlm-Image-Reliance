# Report

LaTeX source of the paper, on the official CVPR author kit (`cvpr.sty`, `ieeenat_fullname.bst`).

- **Overleaf:** New Project -> Upload Project -> zip this folder -> Recompile (pdfLaTeX).
- **Locally:** `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).
  [Tectonic](https://tectonic-typesetting.github.io) also works: `tectonic main.tex`.

`figures/fig1..fig6` are the untitled, column-sized versions that `medvlm posthoc --profile main --model
medgemma4b --model-b gemma3_4b` writes here directly (titled versions go to `results/posthoc_*/main/`).
`figures/fig0_teaser.png` comes from `python make_teaser.py` (examples in `teaser_examples.json`). Every
number in the text is taken from the committed tables under `results/` (see RESULTS.md).

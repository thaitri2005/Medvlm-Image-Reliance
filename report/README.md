# Report

LaTeX source of the term-project report. Compile with pdfLaTeX + BibTeX:

- **Overleaf:** New Project -> Upload Project -> zip this folder (`main.tex`, `refs.bib`, `figures/`) -> Recompile.
- **Locally:** `latexmk -pdf main.tex` (or `pdflatex main && bibtex main && pdflatex main && pdflatex main`).

Figures are copies of `results/posthoc_medgemma4b_vs_gemma3_4b/main/fig*.png`; regenerate them with
`medvlm posthoc --profile main --model medgemma4b --model-b gemma3_4b` and copy again. Every number in the
text is taken from the committed tables under `results/` (see RESULTS.md).

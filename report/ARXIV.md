# arXiv submission

## Build the bundle

```bash
cd report
tectonic --keep-intermediates main.tex     # or: latexmk -pdf main.tex (writes main.bbl)
python make_arxiv.py                       # -> report/arxiv.zip (gitignored)
```

`arxiv.zip` holds `main.tex`, `main.bbl`, `cvpr.sty`, `ieeenat_fullname.bst` and the six included figures. It has
been checked to compile on its own (11 pages).

## Submit (https://arxiv.org/submit)

1. Log in (new authors may need an endorsement for cs.CV; arXiv shows this at the category step).
2. **License:** CC BY 4.0.
3. **Primary category:** cs.CV. **Cross-lists:** cs.CL, cs.LG.
4. **Upload** `arxiv.zip`; arXiv compiles it with pdfLaTeX. Check the generated PDF before continuing.
5. **Metadata** (paste as plain text):

**Title**

Consistent but Blind? Image Removal Does Not Measure Image Reliance in Medical Vision-Language Models

**Authors**

Tri Thai Huu

**Abstract**

Whether a medical vision-language model (VLM) uses the image is commonly assessed by removing it: the
consistent-but-dangerous taxonomy labels an answer Dangerous when it is stable under paraphrase and unchanged
without the image. We ask whether this criterion can rank models. Three medical VLMs and the general-purpose models
they were built from (MedGemma-4B and MedGemma-1.5-4B vs. Gemma-3-4B; MedVLThinker-3B vs. Qwen2.5-VL-3B) answer
3,365 questions on 907 CT, MRI, X-ray and pathology images without the image and under three in-format probes that
keep an image in the input: a different image, a black image, and an image for which the same question has the
opposite answer. The main analysis was pre-registered, and every later probe and model pair was registered before
it was run. Under image removal, MedGemma-4B has a 13.7-point [10.0, 17.5] higher Dangerous share than Gemma-3-4B.
No in-format probe corroborates this or any other image-removal difference: of nine model-pair x dataset
comparisons, image removal separates the two models in five, and an in-format probe agrees in none. The
image-removal differences track refusal (without an image, Gemma-3 declines on 29% of questions, and its forced
answers there are unrelated to its answers with the image) and the agreement between a model's priors and the
image. In-format probes have a confound of their own: both MedGemma models follow opposite-answer images far more
than Gemma-3 on SLAKE, which they were trained on (+48.0 and +36.5 points), but not on held-out PathVQA (+0.8 and
-5.6). Image removal should therefore not be used to compare image reliance between models; such comparisons
require in-format probes on data outside each model's training set.

**Comments**

11 pages, 6 figures. Pre-registered; code, data manifests and all model outputs:
https://github.com/thaitri2005/Medvlm-Image-Reliance

After it is announced, add the arXiv link to the README and to `CITATION` information if any.

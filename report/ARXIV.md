# arXiv submission

## Build the bundle

```bash
cd report
tectonic --keep-intermediates main.tex     # or: latexmk -pdf main.tex (writes main.bbl)
python make_arxiv.py                       # -> report/arxiv.zip (gitignored)
```

`arxiv.zip` holds `main.tex`, `main.bbl`, `cvpr.sty`, `ieeenat_fullname.bst` and the seven included figures.
Rebuild it after every change to `main.tex`, and check that the bundle compiles on its own before uploading.

## Submit (https://arxiv.org/submit)

1. Log in (new authors may need an endorsement for cs.CV; arXiv shows this at the category step).
2. **License:** CC BY 4.0.
3. **Primary category:** cs.CV. **Cross-lists:** cs.CL, cs.LG.
4. **Upload** `arxiv.zip`; arXiv compiles it with pdfLaTeX. Check the generated PDF before continuing.
5. **Metadata** (paste as plain text):

**Title**

Consistent but Blind? Probes of Image Reliance Disagree in Medical Vision-Language Models

**Authors**

Tri Thai Huu, Hung Nguyen Le (both VinUniversity; 23tri.th@vinuni.edu.vn, 23hung.nl@vinuni.edu.vn)

**Abstract**

Whether a medical vision-language model (VLM) uses the image is commonly assessed by removing it: the
consistent-but-dangerous taxonomy labels an answer Dangerous when it is stable under paraphrase and unchanged
without the image. We ask whether this criterion and in-format probes, which keep an image in the input (a
different image, a black image, or an image for which the same question has the opposite answer), report the same
differences between models. Three medical VLMs and the general-purpose models they were built from (MedGemma-4B and
MedGemma-1.5-4B vs. Gemma-3-4B; MedVLThinker-3B vs. Qwen2.5-VL-3B) answer the same yes/no questions on CT, MRI,
X-ray and pathology images under every probe. By image removal, MedGemma-4B appears less image-reliant than
Gemma-3-4B (Dangerous share higher by 13.7 points [10.0, 17.5]; pre-registered). Probes that keep an image do not
agree. Tested directly on the same questions (post hoc), image removal and an in-format probe measure a different
gap between a medical model and its base in six of nine model-pair x dataset cells, and on SLAKE the sign flips: by
the opposite-answer image, both MedGemma models are far more image-reliant than Gemma-3 (+48.0 and +36.5 points).
Refusal under a forced answer format accounts for part of the image-removal differences, not all. We show that the
opposite-answer probe equals accuracy on pairs for which the answer prior is useless, up to the share of pairs
answered wrongly on both images. Its MedGemma advantage appears on SLAKE, which both models were trained on, and
not on held-out PathVQA (+0.8 and -5.6; registered before it was run), where no probe that changes image content
shows the medical models to be more image-reliant. Probes of image reliance are not interchangeable: a comparison
between models should name its probe, use images that require a different answer, and be run on data outside each
model's training set.

**Comments**

14 pages, 7 figures, 5 tables. Pre-registered; code, data manifests and all model outputs:
https://github.com/thaitri2005/Medvlm-Image-Reliance

After it is announced, add the arXiv link to the README and to `CITATION` information if any.

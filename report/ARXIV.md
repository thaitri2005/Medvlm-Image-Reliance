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

Consistent but Blind? Image Removal Does Not Rank Medical Vision-Language Models by Image Reliance

**Authors**

Tri Thai Huu, Hung Nguyen Le (both VinUniversity; 23tri.th@vinuni.edu.vn, 23hung.nl@vinuni.edu.vn)

**Abstract**

Whether a medical vision-language model (VLM) uses the image is commonly assessed by removing it: the
consistent-but-dangerous taxonomy labels an answer Dangerous when it is stable under paraphrase and unchanged without the
image. We test whether this criterion can rank models. Three medical VLMs and the general-purpose models they were
built from (MedGemma-4B and MedGemma-1.5-4B vs. Gemma-3-4B; MedVLThinker-3B vs. Qwen2.5-VL-3B) answer the same
yes/no questions on CT, MRI, X-ray and pathology images without the image and under three in-format probes that
keep an image in the input: a different image, a black image, and an image for which the same question has the
opposite answer. By image removal, MedGemma-4B appears less image-reliant than Gemma-3-4B (Dangerous share higher
by 13.7 points [10.0, 17.5]; pre-registered). Tested directly, as a difference of differences on the same
questions, image removal ranks a medical model against its base differently from an in-format probe in 13 of 21
comparisons after Holm correction, in 6 of 9 model-pair x dataset cells, and by up to 60.5 points [51.3, 68.8]
against the opposite-answer image. The image-removal differences track refusal (without an image, Gemma-3 declines
on 29% of questions, and its forced answers there are unrelated to its answers with the image) and the agreement
between a model's priors and the image. In-format probes have a confound of their own: both MedGemma models follow
opposite-answer images far more than Gemma-3 on SLAKE, which they were trained on (+48.0 and +36.5 points), but not
on held-out PathVQA (+0.8 and -5.6; registered before it was run), where every model's result is within 6 points of
what its accuracy alone predicts. Image removal should not be used to compare image reliance between models; such
comparisons require opposite-answer images on data outside each model's training set, read against accuracy.

**Comments**

14 pages, 7 figures, 5 tables. Pre-registered; code, data manifests and all model outputs:
https://github.com/thaitri2005/Medvlm-Image-Reliance

After it is announced, add the arXiv link to the README and to `CITATION` information if any.

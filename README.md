<h1 align="center">Consistent but Blind?</h1>
<p align="center"><b>Removing the image misranks how much a medical vision-language model uses it</b></p>
<p align="center">
3 medical VLMs vs their general-purpose bases &nbsp;·&nbsp; SLAKE · VQA-RAD · PathVQA &nbsp;·&nbsp; CT · MRI · X-ray · pathology &nbsp;·&nbsp; pre-registered
</p>
<p align="center">
<a href="report/main.pdf"><b>Paper (PDF)</b></a> &nbsp;·&nbsp; <a href="docs/RESULTS.md">All results</a> &nbsp;·&nbsp;
<a href="docs/PREREGISTRATION.md">Pre-registration &amp; lab notebook</a> &nbsp;·&nbsp; <a href="docs/REPRODUCE.md">Reproduce</a>
</p>

<p align="center"><img src="report/figures/fig0_teaser.png" width="100%"></p>

> **TL;DR.** A recent taxonomy flags a medical VLM as *consistent but dangerous* when its answer survives
> rephrasing **and** survives removing the image. We ran that test, and three tests that keep an image in the
> input, on the same questions for three medical VLMs and their general-purpose bases. For MedGemma, removing
> the image says the medical model uses the image **less**; every in-format test says it uses the image
> **more**, by up to 48 points. A pre-registered replication with MedGemma-1.5 shows the same reversal; a Qwen
> pair whose models never refuse to answer does not. Image removal measures whether a model will answer
> without an image, and whether its priors agree with the image, not whether it looks.

---

## The finding in one figure

<p align="center"><img src="results/replication_main/fig8_pairs.png" width="100%"></p>

Each row alters the image in a different way and counts how often the yes/no answer changes; blue is the
medical model, orange its general-purpose base. A model that uses the image should change its answer when the
image changes. For **both MedGemma releases**, image removal, the probe that defines the taxonomy, ranks
Gemma-3 as the more image-reliant model (−6, −7 points), while the in-format probes rank MedGemma higher (up to
+48 on an image whose correct answer is the opposite). **MedVLThinker and its Qwen base** do not differ under
any in-format probe, and neither ever refuses to answer, so there is no ranking to reverse. Same questions,
prompts and scoring for all five models; 95% CIs clustered by image.

## Why this matters

Medical VQA benchmarks score whether an answer is right, not whether the model looked. A model can reach a
correct "yes" from language priors and be wrong on exactly the patient whose image is unusual. *Consistent but
Dangerous* ([Sadanandan & Behzadan, 2026](https://arxiv.org/abs/2603.20985)) addressed this by crossing two
per-sample tests:

|  | **Answer changes without the image** | **Answer unchanged without the image** |
|---|---|---|
| **Stable under paraphrase** | **Ideal** | **Dangerous**: trustworthy-looking, image-independent |
| **Flips under paraphrase** | **Fragile** | **Worst** |

On chest X-rays they found that fine-tuning for consistency pushes most answers into *Dangerous*. The
taxonomy is a good idea, and a natural thing to use when comparing models. This project asks whether its
reliance axis, removing the image, can be trusted to **rank** models, and finds that it can be reversed.

## What is new here, and what is not

The quadrant taxonomy, the individual probes and the observation that VLMs answer medical questions without
images all predate this project. Our contribution is to run the probes **side by side** on the same questions
and three medical/general model pairs, and to explain why they disagree.

| | Prior work | This project |
|---|---|---|
| Quadrant taxonomy | Introduced on chest X-ray yes/no ([CbD](https://arxiv.org/abs/2603.20985)) | Reused unchanged as the reference; extended to CT, MRI and pathology on three VQA datasets |
| Black-image, swapped and hard-negative images | Used to corroborate image removal ([CbD](https://arxiv.org/abs/2603.20985)) or to define reliance metrics and train against them ([Zafar et al.](https://arxiv.org/abs/2607.03647), [Lotfinia et al.](https://arxiv.org/abs/2606.17710)) | Compared as **rankings** of model pairs: image removal ranks two of three pairs opposite to the in-format probes |
| Refusing without an image | Studied as a safety outcome ([MedVIGIL](https://arxiv.org/abs/2605.07919), [Hearsay](https://arxiv.org/abs/2607.26886)) | Shown to be a **confound** of the image-removal probe: a forced yes/no turns refusals into scored answers |
| *Dangerous* share | Raw share; accuracy per quadrant | Chance floor and Cohen's κ; paired split into correct and wrong *Dangerous* answers |
| Medical vs general checkpoint | Model class does not predict image use ([Lotfinia et al.](https://arxiv.org/abs/2606.17710)) | Three pairs, each a medical model and the exact general model it was trained from, so no comparison is confounded by architecture or size |

## Results

*18,896 model calls per model in the main run plus 3,535 per model for the probes, five models. Primary pair:
MedGemma-4B vs Gemma-3-4B; results 1–5 are for it, result 6 for the two replication pairs. Primary set: 1,206
yes/no test questions on 670 images. All intervals are 95% cluster-bootstrap CIs by image. Every number:
[RESULTS.md](docs/RESULTS.md).*

**1. By the reference test, MedGemma looks more "consistent but blind".** With the image removed, its
*Dangerous* share is **+13.7 points** higher than Gemma-3's [10.0, 17.5]: in every dataset and every cell of
the pre-registered sensitivity grid.

<p align="center"><img src="results/posthoc_medgemma4b_vs_gemma3_4b/main/fig1_quadrants.png" width="92%"></p>

**2. Every probe that keeps an image in the input says the opposite.**

<p align="center"><img src="results/posthoc_medgemma4b_vs_gemma3_4b/main/fig7_reversal.png" width="78%"></p>

The cleanest is the
opposite-answer image: the same question asked about another image of the same modality whose correct answer
is the opposite. MedGemma follows the image **73%** of the time, Gemma-3 **25%** (+48.0 [41.9, 53.8]), in CT,
MRI and X-ray alike. Given a black image, Gemma-3 keeps its real-image answer far more often (κ 0.33 vs
0.11), and becomes the more *Dangerous* model by 14 points.

<p align="center"><img src="results/posthoc_medgemma4b_vs_gemma3_4b/main/fig6_opposite.png" width="55%"></p>

**3. Why image removal gets it backwards: two confounds.**

- **Refusal.** Told to answer yes or no about an image it was not given, Gemma-3 would rather ask for the image
  on 354 of 1,206 questions (MedGemma 219). Its forced answer on those is mostly "no" and unrelated to its
  with-image answer (κ ≈ 0), so 55% of them count as "image-reliant". Allowed to say "unknown", **both models
  decline on at least 93%**.
- **Agreement between prior and image.** On questions both models answer blind, MedGemma's gap persists
  (+8.0), because its blind answers are informative (above the answer-frequency prior on PathVQA, P1a). When
  the image confirms what the prior predicts, a model that looks gives the same answer with and without it.
  The opposite-answer probe removes this case, and there MedGemma follows the image.

<p align="center"><img src="results/posthoc_medgemma4b_vs_gemma3_4b/main/fig4_abstention.png" width="78%"></p>

**4. What the *Dangerous* share contains.** On yes/no questions about 36–42% is a **chance floor**: answers
that would match with and without the image if the two were unrelated (Cohen's κ corrects for it; the
chance-corrected gap is Δκ = +0.11 [0.04, 0.18]). And MedGemma's excess is **entirely correct answers**:
*Dangerous*-and-correct +14.1 [11.1, 17.1], *Dangerous*-and-wrong 11.3% vs 11.7%. The residual concern is
subtler: MedGemma errs less, so more of its remaining errors (44% vs 31%) are the stable, image-independent
kind that no label-free check would catch.

<p align="center"><img src="results/posthoc_medgemma4b_vs_gemma3_4b/main/fig5_p3.png" width="70%"></p>

**5. A negative result.** Image reliance, measured as image PMI of the model's own answer, does **not** flag
wrong answers (AUROC 0.51, at chance; the model's confidence reaches 0.71).

**6. The reversal replicates where refusal differs** (pre-registered, [docs/PREREGISTRATION.md §16](docs/PREREGISTRATION.md)).

| | MedGemma-4B | MedGemma-1.5-4B | Gemma-3-4B | MedVLThinker-3B | Qwen2.5-VL-3B |
|---|---|---|---|---|---|
| Follows an opposite-answer image | 72.8% | 61.3% | 24.8% | 47.1% | 48.8% |
| Answer changes when the image is removed | 42.0% | 40.6% | 48.0% | 46.4% | 42.5% |
| Refuses without the image (forced yes/no) | 18.2% | 0.7% | 29.4% | 0% | 0% |
| Accuracy with the image, SLAKE yes/no | 86.5% | 81.1% | 68.7% | 78.9% | 76.6% |

- **MedGemma-1.5 vs Gemma-3: replicates.** +36.5 points on opposite-answer images, −7.4 under image removal;
  it refuses on 8 questions to Gemma-3's 354, and on questions both answer the *Dangerous* gap disappears
  (+3.0 [−3.6, 9.5]).
- **MedVLThinker vs Qwen2.5-VL: no reversal.** The medical fine-tune changed neither accuracy nor image use
  (opposite-answer −1.6 [−4.9, 1.4]) and neither model refuses, so image removal has nothing to reverse.
- Whether a medical model uses the image more than its base is model-specific: both MedGemma releases do,
  MedVLThinker does not.

### Confirmatory tests

Endpoints P1–P3 were frozen before the main run; probe hypotheses H6–H8 after it, but before the probes were
run ([docs/PREREGISTRATION.md §15](docs/PREREGISTRATION.md)); replication hypotheses H9–H11 before the new models were run ([§16](docs/PREREGISTRATION.md)).
Failed predictions are reported as such.

| | Test | Estimate [95% CI] | Outcome |
|---|---|---|---|
| P1a | MedGemma blind accuracy − prior, PathVQA | +5.0 [0.9, 8.8], p_Holm = 0.042 | above the prior |
| P1b–c | … SLAKE, VQA-RAD | −3.1 [−7.8, 1.8]; +7.6 [0.4, 14.5], p_Holm = 0.072 | not significant |
| P2 | AUROC(image PMI) − AUROC(confidence) | −0.197 [−0.250, −0.145] | reliance is worse than confidence |
| P3 | *Dangerous* share, MedGemma − Gemma-3 (image removed) | +13.7 [10.0, 17.5], p < 0.001 | MedGemma higher |
| H6 | κ(real, black image), MedGemma − Gemma-3 > 0 | −0.22 [−0.30, −0.14], p_Holm < 0.001 | **failed, reversed** |
| H7 | Declines with "unknown" allowed, MedGemma − Gemma-3 < 0 | −6.7 [−8.2, −5.3], p_Holm < 0.001 | supported; both ≥ 93% |
| H8 | Follows an opposite-answer image, MedGemma − Gemma-3 > 0 | +48.0 [41.9, 53.8], p_Holm < 0.001 | supported |
| H9–H11 | Replication: medical − general follows the opposite image more, changes less without the image, refuses less | MedGemma-1.5: +36.5, −7.4, −28.7 (all p_Holm < 0.001); MedVLThinker: −1.6, +3.9, 0.0 | **replicates for MedGemma-1.5, not for MedVLThinker** |

## Method

```mermaid
flowchart LR
    Q[Question + image] --> O[original image]
    Q --> P[validated paraphrases]
    Q --> N[image removed]
    Q --> S[wrong image]
    Q --> K[black image]
    Q --> X[opposite-answer image]
    Q --> U[image removed,<br/>'unknown' allowed]
    O & P & N & S & K & X & U --> M[MedGemma-4B / Gemma-3-4B<br/>bf16, greedy, batch 1]
    M --> A["P(yes), answer,<br/>yes/no probability mass"]
    A --> QD[quadrants, agreement κ,<br/>abstention, follow rate]
```

| Probe | What changes | Role |
|---|---|---|
| **Image removed** | image tokens omitted, text identical | the taxonomy's reference probe (main run) |
| **Wrong image** | another image, same dataset and modality | second probe (main run) |
| **Black image** (R1) | a black image in the image slot | in format, no content |
| **"Unknown" allowed** (R2) | image removed, instruction allows "unknown" | separates refusal from blind answering |
| **Opposite-answer image** (R3) | another image on which the same question has the opposite gold answer | an image-reliant model must change its answer |

- **Models.** [`google/medgemma-4b-it`](https://huggingface.co/google/medgemma-4b-it) and its
  architecture-matched general twin [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it), both
  unquantized (bf16), because quantization would perturb the probabilities being audited. The two differ in
  their whole post-training, so we compare *checkpoints*, not "medical data" in isolation. Replication pairs:
  [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) vs the same Gemma-3, and
  [`UCSC-VLAA/MedVLThinker-3B-RL_PMC`](https://huggingface.co/UCSC-VLAA/MedVLThinker-3B-RL_PMC) vs its base
  [`Qwen/Qwen2.5-VL-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). Every model gets the same
  prompts, precision and scoring, and each passed a registered format check before its run.
- **Data.** [SLAKE](https://www.med-vqa.com/slake/) (CT, MRI, X-ray; the only dataset with modality labels),
  [VQA-RAD](https://osf.io/89kps/) (radiology), [PathVQA](https://github.com/UCSD-AI4H/PathVQA) (pathology):
  3,365 questions on 907 images, sampled once with SHA-1 seeds and fixed in `manifests/main/`.
- **Scoring.** Yes/no answers from one forward pass,
  `P(yes) = sigmoid(logsumexp(yes-token logits) - logsumexp(no-token logits))`
  over token variants; the yes+no probability mass detects refusals. Open answers by
  token-F1 with an opposite-term guard (left/right, upper/lower, … never match).
- **Paraphrases.** Qwen2.5-3B-Instruct plus content-copying templates, a rule validator, and **every one of
  4,890 unique (question, paraphrase) pairs judged for equivalence** against a written rubric (1,208
  rejected, each with a reason; unjudged pairs are rejected by construction).
- **Statistics.** Cluster bootstrap over images (B = 2,000), Holm correction within families, Cohen's κ
  for chance agreement. A finding is "robust" only if its sign survives the pre-specified sensitivity grid.

## Engineering

The audit runs on one consumer GPU from a fresh clone, and every number in the paper is regenerated by a
command.

**Stack:** Python · PyTorch · Hugging Face Transformers · bf16 inference on one 16 GB GPU (RTX 4080 SUPER) ·
pandas · NumPy · scikit-learn · matplotlib · pytest · LaTeX (CVPR kit)

- **Resumable inference engine.** Each job is one (question, condition) pair keyed by ID and appended to a
  per-host JSONL log, so a crashed or interrupted run continues where it stopped, and runs split across
  machines merge. 18,896 calls per model in ~75 min (0.23 s/call, 8.2 GB peak).
- **Deterministic inputs.** Dataset loaders, seeded sampling, SHA-checked images and committed manifests: every
  run on every machine sees byte-identical prompts and images.
- **Numerical determinism.** Pre-flight showed that in bf16 the **batch size** alone shifts teacher-forced
  log-probabilities by up to 1.09 nats on identical, unpadded inputs, and an fp32 output head does not fix it.
  All runs use batch size 1; a regression check reproduces the T4 pilot on the 4080 (98.5% identical answers).
- **A paraphrase pipeline** with generation, rule validation and LLM-judged equivalence, enforced fail-closed.
- **Statistics as code.** Cluster bootstrap, Holm, κ and all post-hoc analyses in one tested module; a
  simulation test shows the clustered intervals reach nominal coverage where naive ones do not.
- **CLI and configs.** `medvlm prepare | paraphrase | pipeline | probes | analyze | compare | posthoc | regress
  | status | doctor`; configs layer base → profile → model → hardware; every command writes a timestamped log
  with progress, ETA and GPU memory.
- **97 unit tests + 7 real-model GPU tests**, including scoring edge cases, crash-safe resume, batching
  equivalence and the construction of opposite-answer partners.

## Reproduce

```bash
pip install -e ".[model,dev]" && pytest                         # 97 fast tests, no GPU needed
medvlm pipeline --profile main --model medgemma4b --hardware rtx4080 --parts 2
medvlm pipeline --profile main --model gemma3_4b  --hardware rtx4080 --parts 2
medvlm probes   --profile main --model medgemma4b --hardware rtx4080      # and gemma3_4b
medvlm posthoc  --profile main --model medgemma4b --model-b gemma3_4b     # all tables and figures
```

Full instructions, including gated-model access and pre-flight checks: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Repository

```
report/            the paper (CVPR author kit): main.tex, main.pdf, refs.bib, figures/
RESULTS.md         every result with its CI, the hypotheses, threats to validity and a post-run audit
docs/PREREGISTRATION.md            pre-registration, design decisions and lab notebook (frozen sections marked)
docs/REPRODUCE.md  how to run everything from a fresh clone
src/medvlm/        data/ (loaders, sampler), models/ (HF backend), paraphrase.py, run.py, probes.py,
                   scoring.py, analysis/ (stats, tables, post-hoc, figures), cli.py
configs/           base -> profile -> model -> hardware layering
manifests/         exact questions, image hashes, paraphrases, equivalence judgments, audit sheets
results/           analysis tables and figures per model and comparison
outputs/           raw per-call model outputs of the finished runs
tests/             fast unit tests and -m slow real-model tests
```

## Limitations

- **Three pairs, two families.** Enough to show that image removal *can* misrank models and that refusal
  predicts when; not enough to say how often it happens across the field. The refusal account was formed after
  the primary results and fits all three pairs, but three pairs cannot test it formally.
- **Checkpoints, not causes.** Each medical model differs from its base in its whole post-training, not only in
  medical data. MedGemma-1.5's training data includes SLAKE and VQA-RAD.
- **Probe timing.** The probes were registered after the main results were known (before running them); one of
  three predictions failed.
- **Paraphrases** were judged by an LLM and not validated by human raters. Quadrant results cover questions
  with at least two valid paraphrases (84% of the primary set).
- SLAKE is in MedGemma's training data (split unstated). PathVQA questions are auto-generated from captions.
- Benchmark VQA is not clinical use. Nothing here is a clinical claim.

## Related work

- **Consistent but Dangerous** ([arXiv 2603.20985](https://arxiv.org/abs/2603.20985)): the quadrant
  taxonomy on chest X-rays; image removal, corroborated by black-image and random-swap checks.
- **PSF-Med** ([arXiv 2602.21428](https://arxiv.org/abs/2602.21428)): paraphrase sensitivity; low flip rates
  can mean the image is ignored.
- **Do Medical VLMs Actually See?** ([arXiv 2607.03647](https://arxiv.org/abs/2607.03647)): blank, shuffled,
  absent and hard-negative images as reliance metrics and a training signal.
- **VLMs for chest radiography do not always need the image** ([arXiv 2606.17710](https://arxiv.org/abs/2606.17710)):
  image swaps and occlusion; model class does not predict image use.
- **MedVIGIL** ([arXiv 2605.07919](https://arxiv.org/abs/2605.07919)) and **Hearsay**
  ([arXiv 2607.26886](https://arxiv.org/abs/2607.26886)): whether models refuse when visual evidence is
  broken or missing.
- **How Do Medical MLLMs Fail?** ([ICLR 2026](https://arxiv.org/abs/2603.14323)): grounding measured on
  expert-annotated regions.

## Licences

SLAKE (CC-BY-4.0), VQA-RAD (CC0 1.0), PathVQA (MIT); MedGemma and Gemma-3 under Google's terms. Cite the
original dataset papers in any use of this work.

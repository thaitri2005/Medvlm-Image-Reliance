# Results: Consistent but Blind?

MedGemma-4B and its architecture-matched control Gemma-3-4B, 3,365 questions × 5.6 conditions on average =
**18,896 model calls per model**, all completed (RTX 4080 SUPER, bf16, batch 1, ~75 min per model). Primary
set: the 1,206 yes/no **test** questions on 670 images. All intervals are 95% cluster-bootstrap CIs by image
(B = 2,000). **Pre-registered** results and **post-hoc** analyses are labelled as such; every change after the
analysis plan was frozen is logged in [PLAN.md §13](PLAN.md), and the robustness probes were registered in
[PLAN.md §15](PLAN.md) before they were run. Tables: `results/<model>/main/`,
`results/compare_medgemma4b_vs_gemma3_4b/main/`, `results/posthoc_medgemma4b_vs_gemma3_4b/main/`
(`medvlm analyze`, `compare`, `posthoc`). Paper: [report/main.tex](report/main.tex).

## Summary

1. **By the reference test, MedGemma is "consistent but dangerous" more often** than its architecture twin
   Gemma-3: with the image removed, its Dangerous share is +13.7 points higher [10.0, 17.5] (P3, robust across
   the pre-registered sensitivity grid and every dataset).
2. **By every in-format test, MedGemma is the *more* image-reliant model** (robustness probes registered before
   they were run, PLAN.md §15):
   - shown an image on which the same question has the **opposite** answer, MedGemma changes its answer
     **73%** of the time, Gemma-3 **25%** (H8: +48.0 points [41.9, 53.8]);
   - shown a **black** image, Gemma-3 keeps its real-image answer far more often (κ 0.33 vs 0.11), and under
     this probe Gemma-3 is the more Dangerous model by 14.0 points (H6 predicted the opposite and **failed**);
   - MedGemma is 18 points more accurate with the image on SLAKE yes/no (86.5% vs 68.7%).
3. **What image removal measures is willingness to answer without an image.** Forced to say yes or no,
   Gemma-3 asks for the image on up to 50% of questions and MedGemma on 26–27%; allowed to say "unknown",
   **both decline on at least 93%** (MedGemma 93.0%, Gemma-3 99.7%; H7). MedGemma's residual blind answers
   beat the answer-frequency prior (PathVQA +5.0 points, P1a).
4. **MedGemma's extra "Dangerous" answers are correct answers.** Dangerous-and-correct: +14.1 [11.1, 17.1];
   Dangerous-and-wrong: 11.3% vs 11.7%. The residual concern: MedGemma errs less, so more of its remaining
   errors (44% vs 31%) are the stable, image-independent kind no label-free check can catch.
5. **The Dangerous share needs correcting before it means anything**: a chance floor on yes/no (41.5% / 35.8%;
   Δκ = +0.11 [0.04, 0.18]), its probe (it flips sign between image removal and a black image), and the
   correctness split above. **Image reliance does not flag wrong answers** (AUROC 0.51, confidence 0.71).

## 1. Sanity checks

| | MedGemma-4B | Gemma-3-4B | Model card |
|---|---|---|---|
| SLAKE token-F1 (test, all questions) | **70.2** [67.5, 72.7] | 45.1 [42.1, 48.2] | MedGemma 72.3 · Gemma-3 40.2 |
| VQA-RAD token-F1 (test) | 64.6 [59.5, 69.4] | 45.0 [39.9, 50.3] | 49.9 · 33.6 (balanced split) |
| yes/no format mass, with image | ≥ 0.997 in every dataset | 1.000 | |

The pipeline reproduces MedGemma's published SLAKE score within 2.1 points. VQA-RAD is higher than the card,
which evaluates a harder "balanced" split. Both were anticipated in the pre-registration.

## 2. Accuracy with, without and with the wrong image (test, yes/no)

| | prior | MedGemma: image | no image | wrong image | Gemma-3: image | no image | wrong image |
|---|---|---|---|---|---|---|---|
| PathVQA | 52.2 | 65.5 | 57.2 | 51.0 | 63.2 | 53.5 | 46.7 |
| SLAKE | 50.7 | **86.5** | 47.6 | 67.0 | 68.7 | 47.6 | 62.3 |
| VQA-RAD | 53.0 | **81.3** | 60.6 | 70.5 | 56.2 | 55.4 | 60.2 |

## 3. Pre-registered endpoints

| | Endpoint | MedGemma-4B | Gemma-3-4B |
|---|---|---|---|
| **P1a** | Blind accuracy − majority prior, PathVQA | **+5.0** [0.9, 8.8], p_Holm = 0.042 | +1.3 [−2.8, 5.3] |
| **P1b** | … SLAKE | −3.1 [−7.8, 1.8], p_Holm = 0.21 | −3.1 [−8.3, 2.3] |
| **P1c** | … VQA-RAD | +7.6 [0.4, 14.5], p_Holm = 0.072 | +2.4 [−4.5, 9.2] |
| **P2** | AUROC(image PMI) − AUROC(confidence), PMI of the model's answer | **−0.197** [−0.250, −0.145] | −0.071 [−0.116, −0.026] |
| **P3** | Dangerous share, MedGemma − Gemma-3 (paired) | **+13.7** [10.0, 17.5], p < 0.001 (1,012 questions, 600 images) | |

P2 is computed on the model's own answer; the registered formula used the gold answer, which is not available
at deployment and predicts correctness by construction (PLAN.md §13, item 13).

**Q1 (blind accuracy above the prior).** Yes for MedGemma on PathVQA (Holm-significant); VQA-RAD is in the
same direction but not after correction; SLAKE is not. Gemma-3 is never above the prior. Post hoc, among
questions where MedGemma *does* answer without the image (format mass ≥ 0.9), the margin grows to **+11.9**
[7.1, 16.6] on PathVQA and **+15.7** [5.1, 25.3] on VQA-RAD: its blind answers are informative, i.e. it has
learned useful language priors. Memorising the test set is unlikely to be the reason: the margin appears on
PathVQA and VQA-RAD, which MedGemma's model card does not list as training data, and **not** on SLAKE, which
it does list.

## 4. Quadrants, modalities and probes (Q2)

![Fig. 1](results/posthoc_medgemma4b_vs_gemma3_4b/main/fig1_quadrants.png)

Raw shares, as the taxonomy defines them. On yes/no questions they include chance agreement and correct
answers (§6), so a share near 50% is not "half the answers ignore the image".

| Dangerous share, test yes/no | PathVQA | SLAKE | VQA-RAD |
|---|---|---|---|
| MedGemma, image-removed probe | 50.6 | 46.8 | 56.5 |
| Gemma-3, image-removed probe | 43.2 | 28.1 | 31.0 |
| MedGemma, wrong-image probe | 46.6 | 64.7 | 61.9 |
| Gemma-3, wrong-image probe | 44.1 | 70.8 | 63.7 |

MedGemma is also the more **consistent** model (paraphrase flip rate 6.8% vs 13.6% on SLAKE, 19.3% vs
25.9% on PathVQA).

**Modality** (SLAKE, test+validation pooled as pre-specified):

![Fig. 2](results/posthoc_medgemma4b_vs_gemma3_4b/main/fig2_modality.png)

Under the image-removed probe MedGemma's Dangerous share is ~50% for CT (50.8 [46.6, 55.6]) and MRI
(50.0 [42.6, 57.4]) but lower for X-ray (33.1 [24.8, 41.2]). Under the wrong-image probe the order
**reverses** and X-ray becomes the most "Dangerous" (75.4%). The reversal has a mechanical explanation:
where the swapped image carries the same question we can check its gold answer, and **all 23 chest X-ray
partners had the same correct answer** (CT 58%, MRI 60%), so an unchanged answer there is correct, not
blind.

**Probe agreement** (Cohen's κ between "reliant under image removal" and "reliant under wrong image",
yes/no): MedGemma 0.23 (PathVQA), −0.11 (SLAKE), 0.18 (VQA-RAD); Gemma-3 0.07, 0.03, 0.07. H5 holds: the two
probes measure different things, and a Dangerous share without its probe is underspecified.

## 5. Does image reliance flag wrong answers? (Q3) No

![Fig. 3](results/posthoc_medgemma4b_vs_gemma3_4b/main/fig3_selective.png)

Each yes/no answer is scored by its image PMI, log p(answer | image) − log p(answer | no image) for the answer
the model gave: how much the image raised the probability of that answer. It ranks correct above incorrect
answers at chance (AUROC **0.513** [0.473, 0.552] for MedGemma, 0.532 for Gemma-3), while the model's own
confidence reaches 0.709 and 0.603 (**better by 0.197** [0.145, 0.250] and 0.071 [0.026, 0.116]). The
wrong-image variant is no better (0.52 / 0.54). *Whether a model leaned on the image does not tell you whether
it was right.*

## 6. What medical post-training changes (Q4)

![Fig. 4](results/posthoc_medgemma4b_vs_gemma3_4b/main/fig4_abstention.png)

**Abstention.** The pre-registered format check (yes+no first-token mass ≥ 0.9) failed in one condition only:
**image removed** (MedGemma 37–67% of questions below 0.9, Gemma-3 18–76%; with the image ≥ 99.8%). Free
generation shows why: *"Please provide the picture. I need to see the image to determine if there is a spinal
cord present."* (Gemma-3), *"I am unable to answer this question as I do not have access to medical images."*
(MedGemma). Counting mass < 0.5 as abstaining, Gemma-3 abstains on 14 / 41 / 50% (PathVQA / SLAKE / VQA-RAD)
and MedGemma on 10 / 26 / 27%. Paired difference: **−11.2 points** [−14.0, −8.6].

![Fig. 5](results/posthoc_medgemma4b_vs_gemma3_4b/main/fig5_p3.png)

**P3.** MedGemma's Dangerous share is +13.7 points higher [10.0, 17.5], in every dataset (PathVQA +7.5
[2.6, 12.3], SLAKE +18.6 [12.2, 24.9], VQA-RAD +25.6 [14.7, 36.5]) and in every cell of the pre-registered
sensitivity grid. Removing questions on which either model abstains shrinks it to **+8.0** [2.9, 13.3] (both
answer, mass ≥ 0.9; n = 436) or +12.6 [8.1, 17.0] (neither abstains, mass ≥ 0.5; n = 699). Under the
wrong-image probe it is **−0.7** [−4.4, 3.2].

**How much of the gap is chance?** "Dangerous" counts an answer as image-independent whenever the no-image
answer happens to equal the with-image answer, which on yes/no questions happens often by chance. On the P3
set, MedGemma's answers are unchanged without the image 58.6% of the time against 49.1% expected by chance from
its yes-rates (Gemma-3: 49.7% vs 45.7%). Multiplying by each model's consistency gives a **chance-only
Dangerous floor of 41.5% (MedGemma) and 35.8% (Gemma-3)**, against observed 50.5% and 36.8%. Gemma-3 is at its
floor; MedGemma is 9 points above it. About 6 of the 13.7 points are the difference in floors, partly because
MedGemma is more consistent (84.5% vs 78.4%), a desirable property the metric counts against it. The
chance-corrected effect is Δκ = +0.11 [0.04, 0.18], p = 0.002 (`chance.csv`).

**Is Dangerous dangerous?** (post hoc, `harm.csv`) A Dangerous answer is stable and image-independent, but it
may be correct. Split by correctness:

| test yes/no, P3 set (n = 1,012) | MedGemma | Gemma-3 | Paired difference |
|---|---|---|---|
| Dangerous **and correct** | 39.2% | 25.1% | **+14.1** [11.1, 17.1] |
| Dangerous **and wrong** | 11.3% | 11.7% | −0.4 [−2.7, 2.0] |
| All errors | 25.7% | 37.6% | −12.0 [−15.5, −8.5] |
| Share of errors that are Dangerous | 43.8% | 31.0% | **+12.9** [6.0, 19.9] |

**Answer to Q4.** MedGemma's post-training makes it more accurate, more consistent, and more willing to answer
without an image, with blind answers that beat the prior and are usually right. The extra "Dangerous" answers
are these correct blind answers; the harmful cell is unchanged. The cost is subtler: MedGemma's remaining errors
look like its correct answers to every label-free check (stable, unchanged without the image, confident), so
more of them would go unnoticed. The robustness probes (§7) confirm that this is a willingness to answer
without an image, not neglect of an image that is present. One model pair cannot separate medical data from
the rest of the post-training recipe.

## 7. Robustness probes (registered, PLAN.md §15)

Registered after the main run and before running the probes (commit `44131a5`), each with a directional
prediction; one Holm-adjusted family (`probe_tests.csv`). R1/R2 use the 1,206 yes/no test questions, R3 the
367 SLAKE yes/no questions (test+validation, 130 images) that have an opposite-answer partner, as registered.

| | Probe | MedGemma | Gemma-3 | Difference | Prediction |
|---|---|---|---|---|---|
| R1 | Black image: answer unchanged | 52.6% | 72.9% | | |
| | κ(real image, black image) | 0.11 [0.06, 0.17] | 0.33 [0.28, 0.39] | **−0.22** [−0.30, −0.14], p_Holm < 0.001 | H6: > 0 → **failed (reversed)** |
| | Dangerous share under black image | 44.2% | 58.2% | −14.0 [−18.6, −9.4] | |
| R2 | "unknown" allowed: declines to answer | 93.0% | 99.7% | **−6.7** [−8.2, −5.3], p_Holm < 0.001 | H7: < 0 → **supported** |
| R3 | Opposite-answer image: answer changes | 72.8% [68.1, 77.0] | 24.8% [20.4, 29.3] | **+48.0** [41.9, 53.8], p_Holm < 0.001 | H8: > 0 → **supported** |
| | Accuracy on the opposite-answer image | 86.1% | 61.0% | | |

![Fig. 6](results/posthoc_medgemma4b_vs_gemma3_4b/main/fig6_opposite.png)

**R3: MedGemma follows the image.** It changes its answer on 73% of opposite-answer swaps and Gemma-3 on 25%,
for CT (75 vs 26%), MRI (57 vs 26%) and X-ray (81 vs 8%; exploratory), and with the 40 PathVQA/VQA-RAD swaps
added (69.8 vs 24.8%).

**R1: with a black image, the ranking reverses.** Gemma-3 keeps the answer it gave with the real image on 73%
of questions, MedGemma on 53%. The answers explain it: on a black image MedGemma's yes-rate drops from 58% to
28% (it reads an empty image as absent findings), while Gemma-3's rises from 64% to 84%; its answers track the
question rather than the image. The registered prediction (H6) was the opposite; it failed, and we report it.

**R2: given the option, both models decline.** Without an image and with "unknown" allowed, MedGemma declines on
93% (PathVQA 88%, SLAKE 98%, VQA-RAD 97%) and Gemma-3 on 99.7%. H7's direction holds and the registered
"compliance-only" rule (gap < 3 points) is not met, so MedGemma keeps a small tendency to answer blind, mostly
on PathVQA. But most of the forced-format gap (11.2 points) disappears: the main run's image-removed Dangerous
gap largely reflects how each model handles being told to answer about an image it was not given.

**Conclusion of the probes.** Image removal, the probe that defines the consistent-but-dangerous taxonomy,
ranks the two models opposite to both in-format probes. It measures willingness to answer without an image, not
reliance on image content.

## 8. Hypotheses

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Blind accuracy exceeds the prior somewhere, most likely PathVQA | **Supported** for MedGemma on PathVQA (P1a); not for Gemma-3 |
| H2 | Dangerous share differs by modality | **Probe-dependent:** X-ray is lowest under image removal and highest under wrong image, explained by answer-preserving swaps |
| H3 | Image PMI flags errors better than confidence | **Not supported:** at chance, below confidence |
| H4 | MedGemma has a larger Dangerous share than Gemma-3 | **Supported only under image removal** (robust to the grid and to abstention); null under a wrong image, **reversed** under a black image; the excess is correct answers |
| H5 | The two probes agree only weakly | **Supported** (κ −0.11 to 0.23) |
| H6 | κ(real, black image) higher for MedGemma | **Failed, reversed:** −0.22 [−0.30, −0.14] |
| H7 | With "unknown" allowed, MedGemma still abstains less | **Supported**, but both decline on ≥ 93% |
| H8 | MedGemma follows opposite-answer images more often | **Supported:** +48.0 [41.9, 53.8] |

## 9. Threats to validity

- One model pair; the two differ in their whole post-training, not only in medical data.
- The probes were registered after the main results were known (before running them); one of three
  predictions failed.
- Abstention is inferred from first-token probability mass, confirmed by free generation on a sample, not
  annotated for every question.
- SLAKE is in MedGemma's training data (split unstated); VQA-RAD and PathVQA are not listed.
- Paraphrase equivalence was judged by an LLM (PLAN.md §4.5) and not validated by human raters.
- Post-hoc analyses (chance correction, correctness split, abstention, partner check) were motivated by the data
  and are reported as such.

## 10. Audit: leaks, confounds and wrong assumptions

No model is trained in this study; both are frozen and only queried. After the run we checked every path by
which the answer, or something correlated with it, could reach a model or a score, and every assumption the
conclusions rest on:

| Check | Finding | Effect on conclusions |
|---|---|---|
| Gold answer in the model input | None. The prompt is the question plus a fixed instruction; gold answers are used only *after* generation | none |
| Answer type (yes/no vs open) is derived from the gold answer and picks the instruction | Reveals the answer *format*, not the value; disagrees with the question's phrasing for 4 of 3,365 questions (dataset label errors) | negligible |
| Conditions differ only by the image | Verified on the rendered prompts: identical text for both models; with-image adds exactly 256 image tokens | none |
| Wrong-image partner | Never the same image, never another dataset or modality, chosen by a hash of the question ID | none; content confound in §4 |
| Paraphrase screening saw model outputs | No. Judged before the main run, from question text only | none |
| P2 formula | The registered version used the gold answer (log p(gold \| image) alone has AUROC 1.0); recomputed on the model's answer from stored outputs, no rerun needed | P2 reported on the model's answer |
| Dangerous share is not chance-corrected | On yes/no, 49% / 46% of answers would be "unchanged without the image" by chance | corrected with κ (§6) |
| Dangerous mixes correct and wrong answers | 78% of MedGemma's Dangerous answers are correct | split by correctness (§6) |
| Pilot questions (design tuned on them) are part of `main` | Without them: P1 PathVQA +6.3 [2.4, 10.6], p_Holm = 0.003; P3 +12.9 [8.9, 17.1]; wrong-image P3 −0.3 | none (`no_pilot.csv`) |
| Training-data contamination | Model card lists SLAKE, not VQA-RAD or PathVQA | SLAKE caveat; P1 pattern argues against it |
| Answer capitalisation, duplicates, prior | Case normalised; 5 duplicate pairs with conflicting gold (dataset noise); the test-set prior is the strongest constant baseline | none |
| `compare` pooled the validation split into P3 | Bug, fixed before reporting (+14.1 → +13.7) | fixed |

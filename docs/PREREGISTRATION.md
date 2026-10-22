# Consistent but Blind? Pre-registration and lab notebook

A modality-by-modality audit of image reliance in a small medical VLM. Solo project, ~2 weeks, all
runs on one GPU machine. This file is the single source of truth for *what we claim and how we test it*;
[README.md](../README.md) is the paper-style overview and [docs/REPRODUCE.md](REPRODUCE.md) the how-to-run. Last updated 2026-10-16.
Sections 5-6 are the **analysis pre-registration**: frozen before `main` runs; any later change is
listed in section 13 with its reason.

> **Registrations are tagged in git**, so anyone can check that a plan predates its results (this file was
> `PLAN.md` at the root until 2026-10-16): `prereg-main` (the plan when the main run started, sections 0-14),
> `prereg-probes` (section 15), `prereg-replication` (section 16), `prereg-contrast` (section 17). For example
> `git show prereg-main:PLAN.md`. Corrections of facts found later are listed in section 18, never edited into
> the frozen text.

> **Framing after the results.** Sections 0 and 3 keep the framing the study was planned with (a cross-modality
> audit of MedGemma, with Gemma-3 as control). The probe results (section 15.1) moved the headline to the comparison
> between probes; the held-out probe R4 (section 17.1) then withdrew the claim that the MedGemmas rely on the image
> more. The final title is "Image Removal Does Not Measure Image Reliance in Medical Vision-Language Models". No
> frozen section was edited; post-hoc summaries are listed in section 13, item 17, and section 17.1.

## 0. Objective, target, scope

**Objective.** Determine, for a medical vision-language model, *when its answers actually depend on the
image*, across imaging modalities, and whether that dependence can be measured cheaply enough to flag
answers that should not be trusted.

**What we are working on.** An audit of **MedGemma-4B** (`google/medgemma-4b-it`) on SLAKE, VQA-RAD and
PathVQA. Each question is asked with the image, without it (image tokens omitted), with a different image
of the same modality, and in validated paraphrases. Crossing *stable under paraphrase* with *changes when
the image changes* gives four quadrants: Ideal, Fragile, **Dangerous** (consistent but identical without
the image), Worst. Gemma-3-4B (same architecture, no medical post-training) is the control.

**Target: four questions, answered with intervals and pre-registered tests.**

| | Question | Evidence |
|---|---|---|
| Q1 | Is MedGemma's blind accuracy above what answer frequencies alone give? | P1 (primary, per dataset) |
| Q2 | How large is the Dangerous share, by dataset and modality, and does it depend on how image reliance is probed? | descriptive with CIs; probe agreement (kappa) |
| Q3 | Does image reliance (log-prob shift) flag wrong answers better than the model's own confidence? | P2 (primary) |
| Q4 | Does medical post-training change any of this? | P3 (primary, MedGemma vs Gemma-3) |

**Out of scope (deferred, listed so nobody expects them):** more models, the train-vs-test contamination
analysis (profile `full`), the `blank` control, VQA-RAD modality labels, per-question-type claims inside
a modality (reported as exploratory only), training anything, clinical claims.

## 1. Status

- Built: loaders + deterministic sampler + image hashing, batched paraphrasing + validation, resumable
  runner, scoring, statistics (cluster bootstrap, Holm, kappa), `analyze`, `compare`, `bench`, `regress`,
  `pipeline`, hardware presets. **103 fast tests + 7 real-model tests pass** (CPU stand-ins and MedGemma on the RTX 4080), including
  a simulation showing the clustered intervals reach nominal coverage where naive ones do not.
- Pilot done (330 questions, Colab T4, 1,510 jobs). Its role was to test the pipeline and to tune the design
  (section 13), **not** to estimate effects.
- Big machine (RTX 4080 SUPER 16 GB): pre-flight 1-7 done 2026-10-15 (section 9 results). Batch 1 for all runs
  (section 13, item 10). Paraphrases + full LLM-judge screen done (4.5): 84% of the primary set covered.
  Main runs done 2026-10-16 (18,896 calls per model, both complete). Results: [RESULTS.md](RESULTS.md).
  Post-run deviations and post-hoc analyses: section 13, items 12-16.
- Robustness probes (section 15) registered at commit `44131a5`, then run 2026-10-16 (3,535 calls per model):
  H6 failed (reversed), H7 and H8 supported. The report is in `report/` (CVPR format, `report/main.pdf`).
- Prior work re-checked 2026-10-16 (section 14) and the paper reframed around the probe comparison.
- Replication pairs (section 16) registered at commit `ef39ebf`, run 2026-10-16: pair A replicates the reversal,
  pair B does not (section 16.1).
- Probe R4 (section 17) registered at tag `prereg-contrast`, run 2026-10-17 for all five models: H12a not
  supported, H12b reversed, H12c supported (section 17.1). Under the registered decision rule the claim that the
  MedGemmas use the image more is withdrawn; the paper's claim is the disagreement between probes.
- **Final (2026-10-17): no further experiments.** Remaining work is presentation only (poster). The 50-row
  paraphrase audit sheet (`manifests/main/paraphrase_audit.csv`) was rated by the first author **with AI assistance**:
  44/50 accepted (88%, pre-specified threshold 85%). It is reported as an AI-assisted check, not as independent human
  validation; pass B (re-rating) was not done.

## 2. Prior work and our delta

| Prior work | What it did | Limit we address |
|---|---|---|
| *Consistent but Dangerous* ([arXiv 2603.20985](https://arxiv.org/abs/2603.20985), CVPR-W 2026) | Quadrants from K=5 paraphrases (strict) x binary reliance (answer changes when image tokens are omitted); 5 model configs; MIMIC-CXR + PadChest; yes/no only. Fine-tuning lowers flip rate but moves samples into Dangerous. | Chest X-ray only; yes/no only; binary reliance (authors note it misses confidence shifts); single probe. |
| PSF-Med (arXiv 2602.21428) | Chest-X-ray paraphrase-sensitivity benchmark. | Single modality. |
| Why Does Grounding Hurt Medical VQA? (2604.27720) | MedGemma-4B / Lingshu-7B on SLAKE with grounding inputs. | Different question. |
| HEAL-MedVQA (2505.00744), MedHEval (2503.02157) | Shortcut / hallucination benchmarks. | No per-sample consistency x reliance taxonomy. |

Delta: CT / MRI / X-ray / pathology and open-ended answers; a continuous reliance score; a second probe
(wrong image) and its agreement with the first; selective prediction against confidence with paired tests;
a same-architecture control for medical tuning. **Novelty search still to do** (day 4): arXiv / Scholar for
"image reliance" / "text-only baseline" / "paraphrase consistency" x SLAKE / VQA-RAD / PathVQA / MedGemma.

## 3. Hypotheses (directional guesses; nulls are reported as findings)

*Outcomes: [RESULTS.md section 10](RESULTS.md). H1 supported (MedGemma, PathVQA); H2 probe-dependent; H3 not
supported (image reliance at chance for flagging errors); H4 supported under the pre-registered probe (image
removed) only: null under a wrong image, reversed under a black image (section 15); H5 supported. Probe
hypotheses H6-H8: section 15.*

- H1: blind accuracy exceeds the majority prior somewhere (most likely PathVQA, whose questions name the entity). *Pilot hint: it did not (SLAKE -0.12).*
- H2: the Dangerous share differs by modality.
- H3: image PMI predicts correctness better than confidence on yes/no. *Pilot: +0.11 AUROC, CI [-0.08, 0.28], inconclusive.*
- H4: MedGemma has a larger Dangerous share than Gemma-3-4B.
- H5: the two probes agree only weakly. *Pilot: kappa -0.18 to 0.24.*

## 4. Design

### 4.1 Data (profile `main`: 3,365 questions on 907 images)

| Dataset | Source | In `main` | Notes |
|---|---|---|---|
| SLAKE | `BoKelvin/SLAKE` (CC-BY-4.0) | **test** 1,061 Q / 96 images + **validation** 1,053 Q / 96 images (English) | Modality labels (CT / MRI / X-ray). Test alone has MRI 26 / X-ray 30 / CT 40 images, too few for modality comparisons, so validation is added (test+val: CT 82, MRI 53, X-ray 57 images). Listed by MedGemma's model card as **training data** (split unspecified): contamination is a stated confound. |
| VQA-RAD | OSF `89kps` original JSON (CC0) | test 451 Q (`test_*`) | 2,248 QA on 314 images in total (not 3,515). Organ + question type; no modality. |
| PathVQA | `flaviagiammarino/path-vqa` (MIT) | test, 600 yes/no + 200 open, seeded | Auto-generated from captions. Open questions near-unanswerable in the pilot: capped, exploratory. |

Yes/no in `main`, test split: 1,206 questions on 670 images. Sampling and shuffle partners are seeded
through SHA-1 and committed in `manifests/main/items.jsonl` with image hashes.

### 4.2 Model, prompt, decoding

`google/medgemma-4b-it` (model card v1.0.1), control `google/gemma-3-4b-it`; bf16, no quantization.
One prompt for every condition: question + "Answer with only 'yes' or 'no'." / "Answer with a single word
or short phrase." Greedy. Yes/no from one forward pass, `P(yes) = sigmoid(logsumexp(yes tokens) -
logsumexp(no tokens))`; format check `yn_mass` = raw P(yes)+P(no) (pilot: 1.0).

### 4.3 Conditions

`orig` (image), `noimg` (image tokens omitted, not zeros), `shuf` (another image of the same dataset and
modality; fixed partner), `para1..5` (validated paraphrases, original image).

### 4.4 Definitions (code: `scoring.py`, `analysis/table.py`; all unit-tested)

- Normalisation of prediction and gold: case, punctuation, articles; number words -> digits; plurals;
  "x-ray"; "axial" = "transverse". **Opposite-term guard**: left/right, upper/lower, anterior/posterior,
  superior/inferior, medial/lateral, proximal/distal mismatches score 0.
- `correct`: yes/no label = gold; open token-F1 >= 0.5. `same_answer`: same label / F1 >= 0.5.
- `consistent`: same answer as `orig` on each of the **first 2** valid paraphrases (fixed N for all).
- `reliant_noimg` / `reliant_shuf`: answer differs from `orig` under that probe. `reliant_*_strong` (yes/no):
  also moves the log-odds by >= 1 nat (sensitivity).
- Quadrant from (consistent, reliant) under **each** probe: `quadrant` (noimg, the reference paper's) and `quadrant_shuf`.
- PMI = log p(gold given image, q) - log p(gold given no/other image, q); yes/no via renormalised P(yes), open via
  teacher-forced token log-prob of the gold string; never pooled across answer types.
- `prior_acc`: share of the most common gold answer; `blind_gain = correct_noimg - prior_acc`.
- Confidence: yes/no `max(P(yes), 1-P(yes))`; open `exp(mean token log-prob)`.

### 4.5 Paraphrases and their audit

Five styles (lexical, syntactic, formal, plain, concise) from `Qwen/Qwen2.5-3B-Instruct`, greedy, batched.
Rejected automatically if they change laterality, direction, negation, quantity/comparative words,
normal/abnormal or numbers; leak the gold answer; duplicate an earlier one; are near-copies (<10% edit);
change length >3x; or stop being yes/no questions. Rejections are kept with reasons.

**Audit (before the run):** 100 random valid paraphrases, rated "same meaning? y/n" (`medvlm audit-sheet`,
column A), then rated again **at least 48 h later with column A hidden** (column B): intra-rater agreement,
since there is no second person. `medvlm audit-score` gives acceptance and kappa. **Gate: acceptance >= 90%**;
below 85%, tighten the validator and regenerate before running.

**What actually happened (main, 2026-10-15; the rater was an LLM, Claude, at the first author's request).**
Round 1 (100 rows): 81% accepted, below the 85% line, so the validator was tightened as pre-specified
(absence words, side/extent words counted as a multiset, open->yes/no form loss, foreign characters; the
`plain` style excluded, 18/28). Round 2 (300 fresh rows, none seen before): 82.7%; `lexical` 70% and a new
`frame` style 74%, PathVQA and VQA-RAD yes/no 72% / 68%. Rules had reached their ceiling (remaining errors
were semantic: carcinoma->cancer, artery->vessel, fact->possibility, "does A show B"->"is A associated with B"),
so the audit became a **full screen**: every unique rule-valid (question, paraphrase) pair was judged against a
written rubric (below), and `manifests/main/paraphrase_judgments.csv` (4,890 pairs: 3,682 y, 1,208 n, each
rejection with a reason) is part of validation: a rule-valid paraphrase must also be judged `y`; an unjudged one
is rejected (fail closed). Two deterministic styles (`template_a/b`: "Is there X?" -> "Is X present?" / "Can X
be seen?", "does A show B?" -> "Is B shown in A?", ...) were added to recover coverage; they copy every content
word, so meaning cannot drift by construction (still judged). The judge pass also found a validator bug
(`normalize` turned "does" into "doe", so yes/no form loss was never detected for "Does ..." questions), fixed
and tested. **Result: 84% of test yes/no questions (primary set) have >= 2 valid paraphrases** (PathVQA 92%,
SLAKE 83%, VQA-RAD 67%); open questions 32-59% (secondary). Rubric: `y` = asks exactly the same thing, so the
correct answer cannot differ (perception framing "can X be seen" and exact synonyms allowed); `n` = fact turned
into possibility/capability, broadened or narrowed term (mass->lesion, organs->structures), changed scope or
presupposition, added qualifier/finding/modality, inverted or changed relation, antonym, two questions, or
malformed text. A human spot-check of 50 accepted pairs (human-LLM agreement) is planned and reported.

## 5. Pre-registered analysis

**Primary analysis set:** official test splits. **Unit of resampling:** the image (cluster bootstrap,
B = 2,000, seed 1234, percentile 95% CI, two-sided bootstrap p for H0: effect = 0).

**Primary endpoints** (the only p-values interpreted):

| ID | Endpoint | Set | Family |
|---|---|---|---|
| P1a-c | `correct_noimg - prior_acc`, yes/no, per dataset (3 tests) | test | A |
| P2 | AUROC(`pmi_noimg`) - AUROC(confidence) for predicting `correct_orig`, yes/no pooled | test | A |
| P3 | Dangerous share MedGemma - Gemma-3-4B, yes/no pooled, paired by question | test | B |

Family A (4 tests) is Holm-adjusted at alpha = 0.05 (`primary.csv`, `p_holm`); family B is one test.

**Secondary, descriptive (estimate + 95% CI, no p-values):** accuracy with/without image and drop; Dangerous
share by dataset and modality under both probes (`by_modality_pooled.csv` pools SLAKE test+val for modality
comparisons, as pre-specified in 4.1); probe agreement (Cohen's kappa); PMI distributions; open-ended
analyses; selective-prediction table for all scores; paired MedGemma-Gemma-3 differences per cell.

**Pre-specified exclusions and flags:** questions with < 2 valid paraphrases leave quadrant analyses (counts
reported as `*_n`); PathVQA open is exploratory; any cell with < 30 images is flagged `exploratory`.

**Robustness rule:** a finding is called robust only if its sign holds across the sensitivity grid
(`sensitivity.csv`: F1 threshold 0.3 / 0.5 / 0.8 x first 1 / 2 / 3 paraphrases, and strong flips).
Otherwise it is reported as threshold-dependent.

**Sanity check (not a finding):** token-F1 of `orig` vs the model card (SLAKE 72.3, VQA-RAD 49.9;
`reference_check.csv`). Pilot: SLAKE 76.3 [69.5, 82.6]; VQA-RAD 66.3, plausibly because the card uses a harder
"balanced" split. A SLAKE gap > 15 points would mean a pipeline bug.

## 6. Power (projected from the pilot, before running)

CI half-width scales roughly with 1/sqrt(images). P2 in the pilot: half-width ~0.18 on 137 images; `main`
yes/no test has 670 images, so **~0.08**: an AUROC gain of about 0.08 or more is detectable, smaller ones
will be reported as inconclusive. Per-dataset P1 cells have 96 (SLAKE), 135 (VQA-RAD), 439 (PathVQA) images.
Pooled SLAKE modality cells have 53-82 images.

## 7. Pilot (design-tuning run; not used for estimates)

330 questions, 1,510 jobs, ~2 h on a T4 (4.5 s/job). With the final scoring:

| | prior | with image | without image | Dangerous (n) |
|---|---|---|---|---|
| SLAKE yes/no | 0.55 | 0.92 | 0.43 | 0.51 (35) |
| VQA-RAD yes/no | 0.51 | 0.82 | 0.60 | 0.58 (26) |
| PathVQA yes/no | 0.58 | 0.69 | 0.47 | 0.45 (31) |
| SLAKE / VQA-RAD / PathVQA open | - | 0.67 / 0.49 / 0.16 | 0.29 / 0.22 / 0.00 | - |

Clustered tests on the pilot: P1 all CIs include or fall below 0 (no evidence of beyond-prior blind
answering); P2 +0.11 [-0.08, 0.28], p = 0.27 (an earlier claim that "PMI beats confidence" from point
estimates was premature); probe agreement kappa between -0.18 and 0.24.

## 8. Compute

`main` = 3,365 questions: 10,095 image-condition jobs + ~8.7k paraphrase jobs (if ~52% of 5 candidates validate)
= **~19k jobs per model**; Gemma-3 repeats them. `medvlm estimate` prints wall-clock for a given s/job (its pilot
prediction, 1,504 jobs, matched 1,510). At 4.5 s/job (T4) that is ~24 h; at 1 s/job ~5 h; at 0.5 s/job ~2.6 h.
Speed on the big machine is measured by `medvlm bench`, not assumed. **Measured (RTX 4080 SUPER):** 0.31 s/job at
batch 1, 0.21 at batch 8-16, so ~1.6 h per model at batch 1, the setting used.

Applied speed-ups, each with an equivalence test and re-checked by `regress` on the real model:
`logits_to_keep` (LM head on the last positions only; 262k-token vocabulary), batched gold scoring,
length-sorted batches, batched paraphrasing, fast image processor, single-GPU placement, `inference_mode`,
SDPA. Not applied: FlashAttention-2 (no padded batches in the HF path), `torch.compile` (variable shapes),
quantization (would change the audited numbers). Options if too slow: vision-feature caching across jobs that
share an image, a vLLM backend; both need a `regress` pass first.

## 9. Pre-flight on the big machine (no Colab)

| # | Check | Command | Pass if |
|---|---|---|---|
| 1 | Environment | `medvlm doctor --model medgemma4b` | `cuda=True`, bf16=True, HF access OK |
| 2 | Code | `pytest` | 78 passed |
| 3 | Real model, real batching, bf16 | `MEDVLM_TEST_MODEL=medgemma4b pytest -m slow` | all pass (incl. padding, `logits_to_keep`, batched-gold equivalence) |
| 4 | Images | `medvlm prepare --profile pilot` and `--profile main` | `[verify] OK` |
| 5 | Same numbers as the pilot | `medvlm regress --profile pilot --model medgemma4b --hardware <preset> --set run.batch_size=1 -n 200` | PASS (>= 97% same answer, mean abs dP(yes) <= 0.03) |
| 6 | Speed | `medvlm bench --profile main --model medgemma4b --hardware <preset>` | prints the fastest batch that fits |
| 7 | Same numbers at that batch | step 5 with `--set run.batch_size=<best>` | PASS |
| 8 | Paraphrases | `medvlm paraphrase --profile main`, then the audit (4.5) | >= 80% of questions with >= 2 valid; acceptance >= 90% (done: full judge screen replaces sampling; primary set 84%) |
| 9 | Runtime | `medvlm estimate --profile main --model medgemma4b --sec-per-job <bench>` | fits the schedule |

**Results, RTX 4080 SUPER, 2026-10-15:** 1 OK; 2 81 passed; 3 six of seven passed, the batched-gold test
failed (see section 13, item 10) and was reframed as a misalignment guard; 4 OK; 5 PASS at batch 1 (98.5% same
answer, mean |dP(yes)| 0.004, mean |d gold_logp| 0.14 over 200); 6 batch 1: 0.31 s/job, batch 16: 0.21 s/job;
7 not needed (runs stay at batch 1).

Step 5 compares against results from a different GPU (T4), so tiny bf16 differences are expected; the
thresholds allow for them. If step 5 fails but 3 passes, rerun 200 pilot jobs at batch 1 on the big machine
under another name (`--set model.name=medgemma4b_ref --limit 200`) and use that as the reference for step 7.

## 10. Runs

```bash
medvlm pipeline --profile main --model medgemma4b --hardware <preset> --parts 2     # presets run at batch 1
medvlm pipeline --profile main --model gemma3_4b  --hardware <preset> --parts 2
medvlm analyze  --profile main --model medgemma4b     # also run for gemma3_4b
medvlm compare  --profile main --model medgemma4b --model-b gemma3_4b
medvlm probes   --profile main --model medgemma4b --hardware <preset>    # section 15; also gemma3_4b
medvlm posthoc  --profile main --model medgemma4b --model-b gemma3_4b   # post-hoc + probe tables, figures
```

Each `pipeline` = images (skipped if verified) -> paraphrases (skipped if complete; shared by both models)
-> model loaded once -> part 1 -> checkpoint analysis -> part 2 -> analysis. Resume after any crash by
re-running the same command. Commit `results/` and `git add -f outputs/<model>/main`.

## 11. 14-day schedule (solo)

| Day | Work | Exit |
|---|---|---|
| 1 | Pre-flight 1-7; generate paraphrases (step 8) | all PASS |
| 2 | Audit pass A (100 items, ~1 h); start MedGemma part 1 | acceptance known |
| 3 | MedGemma part 2; `analyze`; start Gemma-3 | MedGemma tables |
| 4 | Gemma-3 finishes; `compare`; audit pass B (>= 48 h after A); novelty search | primary.csv x2 |
| 5-6 | Error analysis: read 30 Dangerous + 30 Ideal cases per dataset with images; write Results skeleton | failure taxonomy |
| 7-8 | Figures: Dangerous share by modality under both probes; risk-coverage with CIs; probe agreement; sensitivity | 4 figures |
| 9-11 | Report (problem, related work, method, results, limitations, implementation) | full draft |
| 12-13 | Poster, README, code release | public repository |
| 14 | Buffer | |

**Stop-the-line rules:** a failed pre-flight check, audit acceptance < 85%, or `yn_mass` < 0.9 in any dataset
halts the run until fixed. If day 4 slips, drop Gemma-3 (Q4) and keep Q1-Q3.

## 12. Limitations (to state in the report)

- One primary model; Gemma-3 is a control, not a population of models.
- SLAKE is in MedGemma's training data (split unspecified); dataset comparisons are confounded.
- Paraphrase equivalence was judged by one LLM (Claude), not by humans; a 50-pair human spot-check bounds its
  error. Quadrant results apply to questions with >= 2 judged-valid paraphrases (84% of the primary set);
  compare their accuracy with the excluded questions before generalising.
- Image removal is out-of-distribution input; the wrong-image probe addresses this but does not settle it.
- Yes/no and open answers use different scales and are never pooled; open-ended correctness depends on a
  token-F1 threshold (sensitivity-tested).
- Reference-paper thresholds were read from its HTML; check against the PDF before claiming equivalence.

## 13. Change log (decisions made after looking at data, with reasons)

All from the pilot, before `main`: (1) scoring normalisation + opposite-term guard (pilot showed
"Two"/"2" errors and "right lung" = "left lung" passing); (2) consistency on a fixed first-2 paraphrases
(was biased by paraphrase count); (3) `prior_acc` / `blind_gain` added; (4) PathVQA capped at 200 open
(16% / 0% accuracy); (5) validator: "any" no longer protected, edit threshold 0.15 -> 0.10 (~52% yield);
(6) quadrants under both probes (pilot kappa near 0); (7) SLAKE validation added for modality power; (8)
`blank` postponed; (9) clustered CIs + paired tests for selective prediction (pilot point estimates were
over-read). From the paraphrase audit (section 4.5): (11) validator v2/v3 rules, `plain` excluded, LLM
judgments as a validation step, `template_a/b` styles added as k=7, 8 (an LM attempt at k=7, 8 failed and was
discarded before any run), `paraphrase.n` 5 -> 8. From the big-machine pre-flight: (10) **batch size 1 for all runs.** On MedGemma in bf16 (RTX
4080), teacher-forced gold log-probs depend on the batch size even with no padding (identical queries
batched: up to 0.99 nats; real pilot questions, batch 8 vs 1: median 0.17, max 1.09), and an fp32 LM head
does not remove it (it is in the network, not the output rounding). Yes/no P(yes) was unaffected (max 0.008,
no flips). Batch 1 costs ~1.5x speed, matches the pilot's settings, and makes every job's numbers
independent of which questions it was batched with.

## 14. Verified facts

SLAKE splits / fields / licence / `imgs.zip` layout; VQA-RAD OSF schema (2,248 rows, 314 images, 451 test);
PathVQA fields and splits; MedGemma model card (v1.0.1; SLAKE in training list; contamination warning;
tokenized F1 VQA-RAD 49.9 / SLAKE 72.3 vs Gemma-3-4B 33.6 / 40.2); reference-paper definitions; `logits_to_keep`
in the installed Gemma 3 forward signature. **Unverified:** novelty of the cross-modality angle;
`medgemma-1.5-4b-it`; big-machine throughput.

**After the main run (2026-10-16), reported in RESULTS.md, none changes a pre-registered estimate:**
(12) **Stop-the-line rule triggered by model behaviour, not a bug.** yn_mass < 0.9 occurred only in the
image-removed condition (MedGemma 37-67%, Gemma-3 18-76% of yes/no questions; >= 99.8% elsewhere). Free
generation showed abstentions ("please provide the image"). The runs were complete and the inputs correct, so
instead of halting: forced-choice P(yes) stays the pre-registered definition, abstention (mass < 0.5) is
reported as an outcome, and P1/P3 are re-estimated without abstaining questions (`medvlm posthoc`).
(13) **P2 is confounded by label leakage.** It scored image PMI on the gold answer; for yes/no, log p(gold |
image) alone predicts correctness perfectly (AUROC 1.0). The pre-registered number is reported but not
interpreted; the label-free version (PMI of the model's own answer) is reported next to it. Design error, caught
after unblinding.
(14) **Bug fix in `compare`: P3 had pooled the SLAKE validation split** into the primary test set (+14.1, n =
1,315); fixed to the test split as pre-registered (+13.7, n = 1,012) with a unit test.
(15) **Wrong-image confound check**: where the partner image carries the same question, compare gold answers
(112 swaps; exploratory).
(16) **Post-run audit** (RESULTS.md section 12): no answer reaches a model's input; prompts verified
identical across models; the answer-type instruction is gold-derived but disagrees with question phrasing
for 4/3,365; **the Dangerous share has a large chance floor on yes/no** (41.5% / 35.8%), so P3 is
additionally reported chance-corrected (d-kappa +0.11 [0.04, 0.18]); results unchanged without the pilot
questions; model card lists SLAKE (not VQA-RAD/PathVQA) as training data.
(17) **After the probes: two descriptive summaries** (`medvlm posthoc`). `reversal.csv` puts every probe on
one scale (share of yes/no answers that change when the image is altered): image removed -6.1 points
(MedGemma - Gemma-3), wrong image +5.1, black image +20.3, opposite-answer image +48.0. `removal_split.csv`
splits image-removed changes by refusal (mass < 0.5): on refused questions the forced answer is unrelated to
the with-image answer (kappa 0.04 / -0.01), and Gemma-3 refuses on 354 vs 219 questions. Both summarise runs
that were already registered; neither is a new test.

**Prior work re-checked 2026-10-16** (before reframing the paper). The reference study (arXiv 2603.20985)
corroborated image removal with a KL score, a black image and random swaps, which agreed on its chest X-ray
models. Zafar et al. (arXiv 2607.03647) use blank, shuffled, absent and hard-negative images to define reliance
metrics and train against them; Lotfinia et al. (arXiv 2606.17710) use image swaps and occlusion and find model
class does not predict image use; MedVIGIL (arXiv 2605.07919) and Hearsay (arXiv 2607.26886) study refusal
without evidence. None of these, as far as we could find, compares the probes' *rankings* of models on the
same questions; that comparison and the two confounds behind the disagreement are what we claim as new.

## 15. Robustness probes (registered 2026-10-16, after the main run, BEFORE running these probes)

Each probe answers a question a skeptical reader would ask of the main results. Outputs go to separate run
folders (`outputs/<model>__<probe>/main`), so the frozen main run is untouched. Command:
`medvlm probes --profile main --model <m> --hardware rtx4080` (all three probes, resumable).

| | Probe | Question it answers | Measure | Prediction (directional) |
|---|---|---|---|---|
| R1 | **blank**: black image instead of no image (all 1,564 yes/no) | Is "answers without the image" just confusion from missing image tokens? | chance-corrected kappa(orig, blank) per model; Dangerous share under blank; blank yes/no mass | **H6**: kappa(orig, blank) MedGemma > Gemma-3 |
| R2 | **abstain**: no image, instruction allows "unknown" | Is answering blind real, or compliance with "Answer with only yes or no"? | abstention (first-token yes/no mass < 0.5) per model and dataset | **H7**: Gemma-3 abstains more than MedGemma. If the paired difference falls below 3 points with a CI including 0, we conclude the main-run difference was format compliance |
| R3 | **opp**: another image of the same dataset and modality on which the same question has the OPPOSITE gold answer (SLAKE, test+validation, 367 questions) | Does the model use the image when one is present? (main-run evidence: 35 swaps) | follow rate (answer changes vs orig); accuracy against the partner's gold; by modality (exploratory) | **H8**: follow rate MedGemma > Gemma-3 |

**Inference.** Paired differences, cluster bootstrap by image (B = 2,000, seed 1234), 95% percentile CIs,
two-sided p; H6-H8 form one family, Holm-adjusted. These are confirmatory for the probes but registered after
the main results were seen, and are reported as such (not as part of the original P1-P3 family). R1/R2 use the
test split; R3 uses SLAKE test+validation (pooled, as for the modality analyses).

**Already observed, post hoc (reported as such):** the Dangerous gap decomposes into Dangerous-and-correct
(+14.1 [11.1, 17.1]) and Dangerous-and-wrong (-0.4 [-2.7, 2.0]); a larger share of MedGemma's errors are
Dangerous (44% vs 31%).

### 15.1 Outcome (run 2026-10-16; `results/posthoc_medgemma4b_vs_gemma3_4b/main/probe_*.csv`)

| | Result (MedGemma vs Gemma-3) | Paired difference [95% CI], p_Holm | Hypothesis |
|---|---|---|---|
| R1 | kappa(orig, blank) 0.11 vs 0.33; answer unchanged 52.6% vs 72.9% | -0.22 [-0.30, -0.14], < 0.001 | **H6 failed (reversed)** |
| R2 | declines with "unknown" allowed: 93.0% vs 99.7% | -6.7 points [-8.2, -5.3], < 0.001 | **H7 supported**; the "format compliance" rule (gap < 3 points) is not met |
| R3 | follow rate 72.8% vs 24.8% (367 SLAKE questions, 130 images) | +48.0 points [41.9, 53.8], < 0.001 | **H8 supported** |

Notes. R3 is analysed on the registered SLAKE set; the runner also produced 40 opposite-answer swaps on
PathVQA/VQA-RAD, reported only as an exploratory "all datasets" row (69.8% vs 24.8%). On a black image
MedGemma's yes-rate falls from 58% to 28% and Gemma-3's rises from 64% to 84%, which is why H6 reversed.
Reading: image removal (the reference probe) ranks the two models opposite to both in-format probes; it
measures willingness to answer without an image rather than reliance on image content.

## 16. Replication pairs (registered 2026-10-16, BEFORE any output of these models was seen)

**Why.** One pair shows that image removal *can* rank two models opposite to in-format probes, not how often.
Two more medical-vs-general pairs test whether the reversal and its mechanism repeat.

| Pair | Medical model | General model | Notes |
|---|---|---|---|
| A | `medgemma15_4b` (`google/medgemma-1.5-4b-it`, released 2026-01) | `gemma3_4b` (outputs reused, not re-run) | Same family as the primary pair; model card lists SLAKE **and VQA-RAD** as training data |
| B | `medvlthinker_3b` (`UCSC-VLAA/MedVLThinker-3B-RL_PMC`) | `qwen25vl_3b` (`Qwen/Qwen2.5-VL-3B-Instruct`, its base) | Different family; RL on PMC-VQA only; trained to reason in `<think>` tags |
| C (fallback) | Med-R1 (Qwen2-VL-2B, RL) | `Qwen/Qwen2-VL-2B-Instruct` | Run only if a pair-B model fails the format rule below |

**Protocol: identical to the primary pair.** Same manifests, images, paraphrases, wrong-image and
opposite-answer partners, prompts and system line, conditions (main run + probes R1-R3), bf16 without
quantization, batch size 1, greedy decoding, scoring and statistics. The only per-model parts are the model's
own Hugging Face processor and chat template (Qwen: native dynamic resolution, default pixel limits). **No
prompt is changed for any model.**

**Format rule (validity of yes/no scoring), fixed now.** Each new model first runs the `pilot` profile (330
questions; a design check, not an estimate) and the slow backend tests. Its yes/no scoring is valid only if the
mean first-token yes+no mass *with the image* is >= 0.9 in every dataset (`results/<model>/pilot/by_dataset.csv`,
`yn_mass_orig`; the primary run's stop-the-line threshold). A model that fails is reported as incompatible with
the shared protocol, its pair is reported as not evaluable, and (pair B only) pair C is run under the same rule.

**Hypotheses** (per pair, medical - general, paired, cluster bootstrap by image, B = 2,000):

| | Hypothesis | Measure |
|---|---|---|
| H9 | The medical model follows opposite-answer images more | R3 follow rate (SLAKE, 367 questions), > 0 |
| H10 | Image removal ranks the pair the other way | share of yes/no answers that change without the image (test, 1,206), < 0 |
| H11 | The general model refuses more without the image | refusal rate, first-token mass < 0.5 (test yes/no), < 0 |

H9-H11 over the evaluable pairs form one Holm family. **Decision rule:** the reversal *replicates* in a pair if
H9 and H10 both hold (Holm-adjusted p < 0.05, predicted signs). The paper reports, for all three pairs
(primary, A, B), the four-probe answer-change comparison (`reversal.csv`), P3, the refusal split, the chance
floor and the correctness split, as for the primary pair; anything beyond H9-H11 is descriptive. Results are
reported whatever they are, including a pair in which the reversal does not replicate.

**Commands** (per new model; resumable): `pytest -m slow` with `MEDVLM_TEST_MODEL=<model>`; `medvlm pipeline
--profile pilot`; check the format rule; `medvlm pipeline --profile main --parts 2`; `medvlm probes`; then
`medvlm posthoc --model <medical> --model-b <general>`. Replication-pair figures go to
`report/figures/<medical>_vs_<general>/` and never overwrite the primary pair's.

### 16.1 Outcome (run 2026-10-16; `results/replication_main.csv`, `results/posthoc_<pair>/main/`)

All three new models passed the format rule on the pilot (yes/no mass with the image: MedGemma-1.5 0.93-0.98,
Qwen2.5-VL 0.998-0.999, MedVLThinker 0.999); the fallback pair C was not needed.

| Pair | H9 (> 0) | H10 (< 0) | H11 (< 0) | Reversal replicates |
|---|---|---|---|---|
| A: MedGemma-1.5 - Gemma-3 | +36.5 [30.6, 42.1], p_Holm < 0.001 | -7.4 [-11.2, -3.7], p_Holm < 0.001 | -28.7 [-31.6, -26.0], p_Holm < 0.001 | **yes** |
| B: MedVLThinker - Qwen2.5-VL | -1.6 [-4.9, 1.4], p_Holm = 0.64 | +3.9 [1.7, 6.1], wrong sign | 0.0 (neither refuses) | **no** |

Reading (post hoc): the reversal appears in both pairs whose models differ in forced-format refusal and not in the
pair whose models never refuse; in pair B the medical fine-tune does not change image use under any in-format
probe, so there is no ranking to reverse.

## 17. Out-of-distribution opposite-answer probe R4 (registered 2026-10-16, BEFORE running it)

**Why.** R3 (+48.0 points, the strongest evidence that MedGemma uses the image more than Gemma-3) is on SLAKE
only, and SLAKE is training data of both MedGemma releases (section 18) but not of MedVLThinker, the one medical
model that showed no gap. "Trained on SLAKE" therefore predicts the R3 pattern as well as "uses the image more"
does. SLAKE's test and validation images are disjoint from its train split (`results/contamination/overlap.csv`),
so the R3 images were not themselves seen in training, but their distribution and question templates were.
R4 repeats R3 on PathVQA, which none of the three medical models lists as training data: MedGemma's technical
report says PathVQA was removed from training; the MedGemma-1.5 and MedVLThinker cards do not mention it.

**Set** (`manifests/contrast/items.jsonl`, committed with this registration; `configs/profiles/contrast.yaml`).
Every PathVQA test and validation yes/no question that is also asked about a different image with the opposite
gold answer: **1,192 questions on 682 images** (test 667, validation 525). Partners are chosen by hash with the
same rule as R3 (`probes.opposite_partners`); 326 partners cross test/validation. Most questions are organ-system
templates (e.g. "is cardiovascular present?"); 65% of gold answers are "yes". None of the 682 images occurs in
PathVQA's train split.

**Protocol.** Identical to R3 for all five models: each question on its own image (`orig`), then on the partner
image (`opp`); same prompt and system line, bf16, batch 1, greedy, answer = P(yes) >= 0.5.
Command: `medvlm contrast --model all --hardware rtx4080` (resumable), then `medvlm contrast --analyze`.

**Measures** (per model and per pair, `results/contrast/`): follow rate (answer changes); pair accuracy (right
on both images; 0 for a model that ignores the image); both wrong; stuck (answer unchanged); refusal (yes/no mass
< 0.5 on either image). Paired medical - general differences, cluster bootstrap by the question's image
(B = 2,000, seed 1234), 95% percentile CIs, two-sided p.

**Hypotheses** (medical - general follow rate; H12a-c form one Holm family):

| | Pair | Predicted | Supported if |
|---|---|---|---|
| H12a | MedGemma-4B - Gemma-3-4B | > 0 | CI lower bound > 0 and p_Holm < 0.05 |
| H12b | MedGemma-1.5-4B - Gemma-3-4B | > 0 | CI lower bound > 0 and p_Holm < 0.05 |
| H12c | MedVLThinker-3B - Qwen2.5-VL-3B | no difference | 95% CI inside [-5, +5] points |

**Decision rule.** (i) H12a and H12b supported: the follow gap is not specific to the medical models' training
distribution, and the paper states that both MedGemma releases use the image more than Gemma-3 on data outside
their listed training sets, reporting the SLAKE and PathVQA gaps side by side. (ii) Either fails with a CI that
includes 0: the SLAKE gap is reported as possibly a training-distribution effect, and the paper's claim is
restricted to the disagreement between probes on the same questions. (iii) A gap below 0: reported as such, and
the claim that this MedGemma release uses the image more is withdrawn. Pair accuracy, per-split results and a
sensitivity analysis that resamples question templates instead of images are descriptive.

**Already seen before registering (disclosed).** The 40 non-SLAKE opposite-answer swaps of the main run
(PathVQA and VQA-RAD; exploratory in section 15.1): MedGemma-4B - Gemma-3 +17.5 [0.0, 34.1], MedGemma-1.5 +12.5
[-5.1, 32.5], MedVLThinker 0.0 [-16.7, 17.1] (`results/contrast/contrast_pairs.csv`, set `main_other`). 108
of the 667 PathVQA test questions in R4 are also in the main run, so their with-image answers (not their partner
answers) have been seen.

### 17.1 Outcome (run 2026-10-17; `results/contrast/`)

| | Pair | Predicted | Follow-rate difference [95% CI] | p_Holm | Outcome |
|---|---|---|---|---|---|
| H12a | MedGemma-4B - Gemma-3 | > 0 | +0.8 [-2.7, 4.2] | 0.87 | not supported |
| H12b | MedGemma-1.5 - Gemma-3 | > 0 | -5.6 [-9.1, -2.0] | 0.003 | reversed |
| H12c | MedVLThinker - Qwen2.5-VL | within +-5 | -0.9 [-3.0, 1.3] | | supported |

Template-clustered sensitivity (154 templates): +0.8 [-6.8, 7.3], -5.6 [-12.0, -0.2], -0.9 [-5.3, 2.9]; same
conclusions. Per-model follow rates 42.7-49.3%, pair accuracy 35.7-43.0% (0 for a model that ignores the image).

**Decision rule applied as registered.** (ii) for MedGemma-4B: the SLAKE gap (+48.0) is reported as possibly a
training-distribution effect and the claim is restricted to the disagreement between probes on the same
questions. (iii) for MedGemma-1.5: the claim that it uses the image more is withdrawn. A per-dataset breakdown of
every probe (RESULTS.md section 15, `results/replication_main/by_dataset.csv`) was added after this outcome and is
post hoc: image removal separates the medical model from its base in 5 of 9 pair-dataset cells, and in none of
them does an in-format probe agree.

## 18. Errata (found 2026-10-16 when re-checking training-data documentation)

Corrections of facts, not of the analysis plan; no pre-registered estimate changes. RESULTS.md and the paper
carry the corrected interpretation.

1. **MedGemma-4B was trained on VQA-RAD.** The MedGemma technical report (arXiv 2507.05201, Table 1) lists
   VQA-RAD (1,391 QA pairs) in the RL stage. Section 13 item 16, RESULTS.md and the paper said the model card
   does not list it. Moreover **202 of the 203 VQA-RAD test images also occur in VQA-RAD's train split** (250 of
   251 yes/no test questions; `medvlm contamination`), so MedGemma-4B may have seen our VQA-RAD test images.
   Consequence: P1c (VQA-RAD) cannot be separated from training exposure, and the argument that MedGemma's
   blind-answer margin appears on data it was not trained on rests on PathVQA (P1a) alone, which the report
   says was removed from training.
2. **MedGemma-1.5 and VQA-RAD.** Section 16 says its model card lists SLAKE and VQA-RAD as training data. The
   card lists SLAKE among training sources; VQA-RAD appears only under data ownership and citation. Corrected
   to "not stated".
3. **Which SLAKE data MedGemma-4B saw.** The report lists 450 SLAKE items, the number of images in SLAKE's train
   split; test and validation images are disjoint from it. If the train split was used, the evaluated SLAKE
   images were not training images, but their distribution was (motivating R4, section 17).


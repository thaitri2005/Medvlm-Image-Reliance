# Consistent but blind? An image-reliance audit of a medical VLM across imaging modalities

Course project (computer vision, groups of up to 3). This file is the single source of truth for
*what we are doing and why*; `README.md` is the how-to-run. Last updated 2026-10-15.

## 0. Objective, target, scope

**Objective (one sentence).** Find out, for a medical vision-language model, *when its answers actually
depend on the image*, across imaging modalities and question types, and whether that can be measured
cheaply enough to flag answers we should not trust.

**What we are working on.** An audit of **MedGemma-4B** (`google/medgemma-4b-it`) on three public medical
VQA benchmarks (SLAKE, VQA-RAD, PathVQA; test splits). Every question is asked under controlled
conditions: with the image, with the image removed, with a *different* image, and in several validated
paraphrases. Crossing "stable under paraphrase" with "changes when the image is removed" sorts each
answer into one of four quadrants (Ideal / Fragile / **Dangerous** / Worst); Dangerous = *consistent but
would be identical without the image*.

**The target (what "done" means).**

| Tier | Deliverable | Status |
|---|---|---|
| 1, must ship | Per dataset x modality x question type: how much is answerable blind (vs. the majority-answer prior), and the quadrant mix, with image-clustered confidence intervals | pipeline built; pilot (330 Q) done; `main` pending |
| 2, the differentiators | A continuous reliance score `log p(gold given image) - log p(gold given no image)` that also works for open answers; a wrong-image control; **does reliance/consistency flag wrong answers better than the model's confidence?** (AUROC, risk-coverage); MedGemma vs Gemma-3-4B (same architecture, no medical tuning); human audit of paraphrases | built, needs `main` |
| 3, bigger machine | More models (Lingshu-7B, MedGemma-27B, ...), train-vs-test (contamination) analysis, VQA-RAD modality labels | later |

Each tier is a valid stopping point. The report is a research paper (problem, approach, results,
implementation), plus a poster and the code zip.

**What we might do beyond that.** Workshop / arXiv tech report once Tier 2 is solid; add models;
prefix/vision-feature caching if runs become the bottleneck (section 8).

**What we are not doing.** Training or fine-tuning anything; chest-X-ray-only work (already covered by prior
work); claiming clinical validity.

## 1. Status snapshot (2026-10-15)

- Built and tested: data loaders + deterministic sampler + image hashing, paraphrase generation/validation,
  resumable runner, scoring, analysis, `bench`, `regress`, `pipeline`, hardware presets. **68 fast tests +
  7 real-model integration tests** pass on the laptop (CPU; SmolVLM-256M and Qwen-0.5B as stand-ins).
- Done on Colab T4: MedGemma loads and the integration tests pass; **pilot** (330 questions, 1,510 jobs, ~2 h).
- Next: pre-flight checks (section 9), then `main` in 2 parts on the big machine.

## 2. What is already known (so we do not claim it) and what we add

Verified by reading the sources:

| Prior work | What it did | Limit we address |
|---|---|---|
| *Consistent but Dangerous* ([arXiv 2603.20985](https://arxiv.org/abs/2603.20985), CVPR-W 2026) | Four quadrants: consistency under K=5 paraphrases (strict: all must match) x image reliance (answer changes when image tokens are omitted). 5 model configs, MIMIC-CXR + PadChest, yes/no logits only. LoRA fine-tuning lowers flip rate but moves samples into "Dangerous". Recommends a text-only baseline pass. | Chest X-ray only; yes/no only; reliance is **binary** (the authors note it misses confidence shifts). |
| PSF-Med (arXiv 2602.21428) | Paraphrase-sensitivity benchmark on chest X-ray; low flip rate does not imply grounding. | Same single modality. |
| Why Does Grounding Hurt Medical VQA? (2604.27720) | MedGemma-4B and Lingshu-7B on SLAKE with grounding inputs. | Different question (grounding aids). |
| HEAL-MedVQA (2505.00744), MedHEval (2503.02157) | Shortcut / hallucination benchmarks. | Not a per-sample consistency x reliance taxonomy. |

**Our delta** (separable contributions): (1) cross-modality, cross-dataset audit; (2) open-ended answers,
not only yes/no; (3) continuous reliance (PMI), fixing the stated limitation of the binary test;
(4) wrong-image control (`shuf`); (5) selective prediction vs. model confidence; (6) controlled
MedGemma-vs-Gemma-3 comparison.

> **Novelty is not yet verified.** Only a handful of searches so far. Week-1 task: search arXiv / Scholar for
> "image reliance" / "text-only baseline" / "paraphrase consistency" x "SLAKE" / "VQA-RAD" / "PathVQA" /
> "MedGemma". If items 1-2 exist already, items 3-6 still stand.

## 3. Hypotheses (written before seeing `main`)

Directions are guesses; a null or reversed result is still a finding for an audit.

- **H1 dataset leakage**: a large share of questions is answerable without the image, differing by dataset; PathVQA is the likeliest (auto-generated questions name the answer entity, e.g. *"is leiomyosarcoma present?"*).
- **H2 modality / question type**: the Dangerous share differs by modality and question type.
- **H3 confidence blindness**: confidence does not separate Dangerous from Ideal; reliance and consistency signals give better selective prediction (lower AURC).
- **H4 medical tuning**: MedGemma-4B has a larger Dangerous share than Gemma-3-4B on the same items.
- **H5 probe validity**: "image removed" and "wrong image" agree only moderately (Cohen's kappa).

## 4. Experimental design

### 4.1 Data (profile `main`: 2,312 questions on 811 images, test splits only)

| Dataset | Source (loader) | In `main` | Metadata | Notes |
|---|---|---|---|---|
| SLAKE | `BoKelvin/SLAKE` (CC-BY-4.0) | all 1,061 English test Q on **96 images**: CT 472 / X-ray 361 / MRI 228 | modality, location, content type | **Primary for the modality analysis.** 61 of 416 "closed" questions are not yes/no and count as open. |
| VQA-RAD | OSF `89kps` original JSON (CC0) | all 451 test Q (`test_*`) | organ, question type, phrase type | **2,248 QA on 314 images in total** (an earlier note said 3,515: wrong). Modality is not annotated. |
| PathVQA | `flaviagiammarino/path-vqa` (MIT) | **800 of 6,719**: 600 yes/no + 200 open | none (image id = sha1 of bytes) | Auto-generated from captions. |

- **Why PathVQA is capped at 200 open** (pilot-driven tuning): open PathVQA questions were near-unanswerable (*"what is present?"* -> gold *"endocrine"*; 16% accuracy with the image, 0% without), yet they are the most expensive jobs. Kept for coverage, treated as exploratory, never pooled into headline accuracy.
- **Contamination.** The MedGemma model card lists SLAKE as a training source and warns of "a risk of data contamination". Headline results therefore use test splits only; profile `full` adds train/val splits for a seen-vs-unseen analysis. Datasets are not directly comparable because of this.
- **Unit of analysis = the image.** SLAKE test has 96 images, so every interval is a **cluster bootstrap over images**; cells with fewer than ~30 images are exploratory.
- `qgroup` is a coarse mapping of dataset-native types (VQA-RAD "PRES" also contains open questions): always cross with `answer_type`.

### 4.2 Model and prompts

- Primary `google/medgemma-4b-it` (gated; model card v1.0.1). Control `google/gemma-3-4b-it`. A `medgemma-1.5-4b-it` was mentioned earlier but the model card does not mention it: **unverified**.
- One fixed prompt for every condition: question + *"Answer with only 'yes' or 'no'."* / *"Answer with a single word or short phrase."* Only the image is added or removed. Prompt changes only to fix **format** failures; the diagnostic is `yn_mass` (raw P(yes)+P(no) at the first token; pilot: 1.0 everywhere).
- Greedy decoding. Yes/no from one forward pass: `P(yes) = sigmoid(logsumexp(yes tokens) - logsumexp(no tokens))`.

### 4.3 Conditions

| Condition | Image | Question | Purpose |
|---|---|---|---|
| `orig` | original | original | reference answer |
| `noimg` | **tokens omitted** (not zeros) | original | text-only baseline (reference paper's probe) |
| `shuf` | other image, same dataset + modality (partner fixed in manifest) | original | wrong-image control |
| `blank` | black image | original | optional extra control; **off by default**, add later (only missing jobs run) |
| `para1..5` | original | validated paraphrase k | consistency |

### 4.4 Definitions (`scoring.py`, `analysis/table.py`; unit-tested)

- **Normalisation** (prediction and gold alike): lower-case, strip punctuation/articles, number words -> digits ("Two" = "2", "None" = "0"), plural stripping ("Lungs" = "lung"), x-ray unified, "axial" = "transverse" (SLAKE's gold wording). **Opposite-term guard**: different members of left/right, upper/lower, anterior/posterior, superior/inferior, medial/lateral, proximal/distal -> F1 = 0 (plain F1 gave "right lung" vs "left lung" a passing 0.5).
- `correct`: yes/no -> label equals gold; open -> token-F1 >= 0.5. `same_answer(a, b)`: yes/no -> same label; open -> normalised match or F1 >= 0.5 (swept 0.3-0.8 as sensitivity).
- `consistent`: same answer as `orig` for **each of the first `analysis.min_paraphrases` (=2) valid paraphrases**. Same number of chances to flip for every sample (an earlier version used "all valid", which penalised samples with more paraphrases). Samples with fewer get no quadrant and are reported (`*_n` columns).
- `reliant_noimg = not same_answer(orig, noimg)`; `reliant_shuf` likewise. **Quadrant** = (consistent, reliant): Ideal (T,T), Fragile (F,T), **Dangerous (T,F)**, Worst (F,F).
- `PMI = log p(gold | image, q) - log p(gold | no/other image, q)`. Yes/no uses the renormalised `P(yes)`, open uses the teacher-forced token log-prob of the gold string. **Scales differ; never pool yes/no with open.**
- `prior_acc`: accuracy of always giving the most common gold answer (yes/no priors are 51-58%). Blind accuracy only means something next to it.
- Baseline confidence: yes/no `max(P(yes), 1-P(yes))`; open `exp(mean token log-prob)`.

### 4.5 Paraphrases

Five styles (lexical, syntactic, formal, plain, concise) from a text-only instruct LM (`Qwen/Qwen2.5-3B-Instruct`, greedy, batched). Automatic validation rejects a paraphrase that changes/drops laterality, direction, negation, quantity words, normal/abnormal or numbers; leaks the gold answer; repeats an earlier paraphrase; is a near-copy (<10% edit; was 15%); changes length by >3x; or stops being a yes/no question. Rejected ones are kept with reasons (the rejection rate is reportable). **Tuning from the pilot:** "any" no longer counts as meaning-critical and the edit threshold went 0.15 -> 0.10, because the pilot kept only ~52% of candidates and good syntactic rewrites were being rejected. Check the printed yield before running. **Human audit:** `medvlm audit-sheet -n 100`, two teammates label "same meaning?" independently; report acceptance and Cohen's kappa. Negation-insertion paraphrases are excluded (they invert the answer). VQA-RAD's ~734 clinician `para` entries have noisy links (we found a mismatched pair): optional extra, not a dependency.

## 5. Analyses and outputs

`medvlm analyze` writes to `results/<model>/<profile>/` with image-clustered 95% CIs: `by_dataset`, `by_dataset_type`, `by_modality`, `by_qgroup`, `selective`, `timing`, `per_sample`.

| # | Question | Output |
|---|---|---|
| A1 | How much is answerable blind? `correct_noimg` vs `correct_orig` vs `prior_acc` | `by_dataset_type`, `by_modality` |
| A2 | Quadrant mix per dataset x modality x answer type x question group | same, `q_*` columns |
| A3 | Do Dangerous answers look confident? | `per_sample` |
| A4 | Does binary reliance hide signal? PMI inside "not reliant" answers | `pmi_*` |
| A5 | Is image removal a valid probe? kappa(`reliant_noimg`, `reliant_shuf`) | `per_sample` |
| A6 | Selective prediction: AUROC / AURC of confidence vs PMI vs consistency | `selective` |
| A7 | MedGemma vs Gemma-3-4B on identical items | second model's results |
| A8 | Qualitative: 30 Dangerous + 30 Ideal cases per dataset, with images | figure (Phase 5) |
| A9 | Seen vs unseen: SLAKE train vs test | `full` profile |

Figures are deferred to Phase 5, when numbers are stable.

## 6. Execution plan

Deadlines are not announced; weeks assume ~8 and should be scaled. **Cut rule:** if `main` has not started by week 4, finish Tier 1 and stop.

| Phase | Week | Work | Exit gate |
|---|---|---|---|
| 0 Setup | 0 | Repo, tests, loaders, manifests. **Done.** You: request Hugging Face access to MedGemma + Gemma-3. | `pytest` green |
| 1 GPU gate | 1 | MedGemma integration tests; pilot on Colab T4. **Done.** | tests pass; throughput known |
| 2 Pilot + tuning | 1-2 | Pilot analysed; scoring, consistency definition, PathVQA quota, validator tuned (sections 4.4, 4.5, 7). **Done.** Remaining: human paraphrase audit, novelty search, 2-page proposal. | audit acceptance rate known |
| 3 Pre-flight | 2 | Section 9 checklist (laptop / Colab T4 / big machine). | all checks green |
| 4 `main` run | 2-3 | `medvlm pipeline --parts 2` on the big machine (section 10). | all keys done |
| 5 Extras | 3-4 | `gemma3_4b` on the same manifest; `blank` control; threshold sensitivity. | A5, A6, A7 tables |
| 6 Analysis | 4-6 | Figures, qualitative panel (A8), error analysis. | story fits one slide |
| 7 Writing | 6-8 | Report, poster, code zip, README polish. | submitted |
| 8 Stretch | after | Tier 3 on the big machine; optional workshop / arXiv. | - |

## 7. Pilot findings (330 questions, free Colab T4; direction only, intervals are wide)

1,510 jobs in ~2 h (7.8 s/job for the first 200, 4.5 s/job after). Scores below are after the scoring fixes.

| | prior | with image | without image | Dangerous share (n) |
|---|---|---|---|---|
| SLAKE yes/no | 0.55 | 0.92 | 0.43 | 0.51 (35) |
| VQA-RAD yes/no | 0.51 | 0.82 | 0.60 | 0.58 (26) |
| PathVQA yes/no | 0.58 | 0.69 | 0.47 | 0.45 (31) |
| SLAKE open | - | 0.67 | 0.29 | 0.35 (43) |
| VQA-RAD open | - | 0.49 | 0.22 | 0.23 (22) |
| PathVQA open | - | 0.16 | 0.00 | 0.00 (22) |

- Format is fine (`yn_mass` = 1.0; open answers are 1-2 words).
- Selective prediction (n=165 each): for **yes/no**, image PMI predicts correctness better than confidence (AUROC 0.85 vs 0.74); for **open**, confidence wins (0.78 vs 0.69). Needs `main` before believing it.
- Consistency was defined for only 179/330 questions (3 candidates, ~52% validated). `main` generates 5 and the yield is printed before the run.
- What the audit changed: number/plural/synonym normalisation (SLAKE accuracy 0.73 -> 0.79), laterality guard (VQA-RAD 0.67 -> 0.66), fixed-N consistency, `prior_acc`, PathVQA quota, validator relaxation, per-job timing, crash-safe resume.
- The pilot's `paraphrases.jsonl` is not in the repo (only on Drive); `main` paraphrases will be committed.

## 8. Compute and acceleration

**Size.** `main` = 2,312 questions: 6,936 image-condition jobs (orig/noimg/shuf) + ~6.0k paraphrase jobs (assumed 52% of 5 candidates validate) = **~13k jobs**; open questions cost extra (generation + gold scoring). `medvlm estimate --profile main --model medgemma4b --sec-per-job X` prints the wall-clock (its pilot prediction, 1,504 jobs, matched reality: 1,510).

| Seconds per job | Whole `main` | Each of 2 parts |
|---|---|---|
| 4.5 (measured, free T4, batch 1, old code path) | ~16 h | ~8 h |
| 1.5 | ~5.4 h | ~2.7 h |
| 1.0 | ~3.6 h | ~1.8 h |
| 0.7 | ~2.5 h | ~1.3 h |

Speed on a 4080 / 4090 / A100 is **not measured**; `medvlm bench` measures it in 2-3 minutes. Reasoning only: they have native bf16 and fused attention kernels (the T4 emulates bf16 and needs the slow path, which is why it was stuck at batch 1), so several-fold faster is plausible. 16 GB (4080) holds the 8.6 GB weights with room for small batches; 24 GB (4090) more; the A100's main advantage is memory (MedGemma-27B needs ~54 GB in bf16).

**Applied (each has a test showing results unchanged; `medvlm regress` re-checks on the real model):**

1. `logits_to_keep`: the LM head runs on the last 1..k positions only. The vocabulary is 262k tokens, so logits for ~300 prompt positions were most of the output-layer cost and memory (supported by both Gemma 3 and SmolVLM; falls back automatically).
2. **Batched teacher-forced gold scoring** (was one forward per question), left-padded.
3. **Length-sorted sub-batches** (less padding), original order restored.
4. **Batched paraphrase generation** (was one LM call per paraphrase: ~12k sequential calls for `main`), written to disk per 256 questions.
5. Fast torchvision image processor when available; whole model loaded straight onto the single GPU (no accelerate dispatch hooks); `inference_mode`; SDPA attention (fused kernels on Ampere/Ada in bf16).
6. Per-question job ordering and chunk size tied to the batch size, so batches stay full and any prefix of the work is a complete subset.
7. Hardware presets + `bench`: batch size is measured, not guessed.

**Deliberately not applied, and why.** FlashAttention-2: no padded-batch support in the HF implementation ([HF GPU inference guide](https://huggingface.co/docs/transformers/perf_infer_gpu_one)), SDPA already picks fused kernels. `torch.compile`: variable sequence lengths cause recompiles. 4-bit/8-bit on GPUs that fit bf16: changes the very numbers being audited. **Kept as measured options** if `bench` shows `main` is too slow: (a) caching the vision-encoder output across jobs that share an image (orig + paraphrases reuse it; the 4096-patch encoder is a large part of each job), (b) a vLLM backend with prefix caching ([vLLM docs](https://docs.vllm.ai/)). Both change code paths that cannot be tested without a GPU, so they need a `regress` pass first. A reported risk worth remembering: left-padded VLM batches have been seen to shift logits for specific pad lengths ([example report](https://huggingface.co/google/gemma-4-12B-it/discussions/50), unverified) which is why the batching tests force real padding and `regress` compares against the unbatched stored run.

## 9. Pre-flight checklist (what runs where)

| Where | Check | Command | Why |
|---|---|---|---|
| **Laptop** (CPU) | Everything not needing MedGemma | `pytest`; `pytest -m slow` | 68 + 7 tests; real-model equivalence (padding, `logits_to_keep`, batched gold, batched paraphrasing) on SmolVLM / Qwen-0.5B. **Re-run after any code change.** |
| **Colab T4** | MedGemma path | `MEDVLM_TEST_MODEL=medgemma4b pytest -m slow` | Gemma 3 processor, bf16, `logits_to_keep` on the real model (the batched test falls back to batch 2 if 3 does not fit). |
| **Colab T4** | Optimised code == old numbers | `medvlm regress --profile pilot --model medgemma4b -n 100` | Pilot results are in git. Must print PASS (>=97% same answer, mean abs dP(yes) <= 0.03). ~7 min. |
| **Colab T4** (or big machine) | Paraphrase yield | `medvlm paraphrase --profile main` | Look at "have >= 2 valid paraphrases" (want >= ~80%) and the rejection reasons; commit the file so the big machine skips this step. |
| **Big machine** | Environment | `medvlm doctor --model medgemma4b` | CUDA torch (the laptop has CPU torch), HF access, suggested preset. |
| **Big machine** | Batched numerics on the real GPU | `MEDVLM_TEST_MODEL=medgemma4b pytest -m slow` | Real batching on bf16: this is the check the T4 cannot do. |
| **Big machine** | Best batch size | `medvlm bench --hardware rtx4090` | Prints s/job and peak memory per batch size; gives the real runtime. |
| **Big machine** | Still the same numbers at that batch | `medvlm regress --profile pilot --model medgemma4b -n 100 --hardware ... --set run.batch_size=<best>` | Final gate before the long run. |

If any check fails, stop and fix before the long run.

## 10. Running `main` on the big machine (2 parts)

```bash
git clone https://github.com/thaitri2005/Medvlm-Image-Reliance.git && cd Medvlm-Image-Reliance
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[model,dev]"                              # install the CUDA build of torch first if pip gave you CPU torch
huggingface-cli login                                      # token with access to the gated models
medvlm doctor --model medgemma4b
medvlm prepare --profile pilot --model medgemma4b          # images for bench/regress; manifests are already in git
medvlm prepare --profile main  --model medgemma4b
# ... then the checks from section 9 ...
medvlm pipeline --profile main --model medgemma4b --hardware rtx4090 --set run.batch_size=<best> --parts 2
```

`pipeline` = prepare images (skipped if already verified) -> paraphrases (skipped if the file is complete; the paraphrase model is freed before MedGemma loads) -> MedGemma loaded **once** -> part 1 of 2 -> checkpoint `analyze` -> part 2 of 2 -> `analyze`. Each part is a fixed random half of the **questions** (all their conditions together). Resume after any crash or kill by running the **same command**: finished jobs are skipped and a half-written last line is repaired. Single part: `--part 0` / `--part 1`; stop cleanly: `--max-minutes M`; progress: `medvlm status --profile main --model medgemma4b --shards 2`.

Results go to `results/medgemma4b/main/` (commit them) and raw outputs to `outputs/medgemma4b/main/` (small; `git add -f` it to move between machines). On Colab, set `MEDVLM_OUTPUTS_DIR` to a Drive folder and use the notebook's push cell.

## 11. Team split (up to 3)

| Role | Owns |
|---|---|
| A, data and language | paraphrase validation, the 100-item human audit (with a second person), VQA-RAD modality labelling (Tier 3), novelty search |
| B, infrastructure and models | big-machine runs, HF access, `bench`/`regress`, Gemma-3 comparison, `full` profile |
| C, analysis and writing | tables, figures, qualitative panel, report, poster |

Everyone reviews each other's changes and takes part in the audit. Working alone: follow the tiers in order. (Team size not yet confirmed.)

## 12. What a researcher will ask, and our answer

- *Is deleting the image a fair probe?* -> `shuf` and `blank` controls, kappa between probes (A5).
- *Contamination?* -> test splits only; SLAKE is the known training source; A9 measures it.
- *Are your paraphrases equivalent?* -> automatic constraints + two-rater audit with kappa.
- *Open-ended scoring is noisy.* -> normalisation + laterality guard, threshold sweep; yes/no and open never pooled.
- *Is blind accuracy just the prior?* -> `prior_acc` next to every accuracy.
- *Are the intervals honest?* -> bootstrap over images, not questions.
- *Did your speed-ups change the results?* -> equivalence tests + `regress` against stored runs.
- *One model is anecdote.* -> Gemma-3-4B control now, larger models later.

Portfolio value: a public, tested, reproducible repo, a pre-specified analysis plan, honest negative results.

## 13. Known limitations and risks

- MedGemma batched numerics (bf16, real batch) are only verifiable on a bigger GPU (section 9). T4 bf16/`logits_to_keep` verified only via the pilot regression.
- Paraphrase quality is the most delicate piece: validated by rules, not yet by humans; real yield on `main` unknown until the paraphrase step runs.
- VQA-RAD has no modality labels; modality conclusions rest on SLAKE and PathVQA.
- `shuf` partners can share the gold answer by chance (more likely for yes/no); bounded and reported.
- For yes/no, a binary "reliant" flip can come from a tiny shift near P=0.5; the continuous PMI is the check (A4).
- Reference-paper thresholds were read from its HTML; re-check against the PDF before claiming to "follow" them.
- Gated access and session limits are the schedule risks; request access now.

## 14. Verified facts log

Checked: SLAKE splits / fields / licence / `imgs.zip` layout; VQA-RAD OSF schema (2,248 rows, 314 images, 451 `test_*`); PathVQA fields and split sizes; MedGemma model card (v1.0.1, SLAKE in training list, contamination warning, tokenized-F1 VQA-RAD 49.9 / SLAKE 72.3 vs Gemma-3-4B 33.6 / 40.2); reference-paper definitions (K=5, strict consistency, binary text-only reliance, yes/no logits); `logits_to_keep` exists in the installed Gemma 3 and Idefics3 forward signatures; HF docs on fast image processors ([link](https://huggingface.co/docs/transformers/main/image_processors)) and FlashAttention padding limits.
**Corrections to earlier notes:** VQA-RAD has 2,248 QA pairs (not 3,515). **Unverified:** `medgemma-1.5-4b-it`; novelty of the cross-modality angle; speed on 4080 / 4090 / A100; vLLM and feature-caching gains.

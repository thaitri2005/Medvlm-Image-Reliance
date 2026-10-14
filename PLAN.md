# Consistent but blind? An image-reliance audit of a medical VLM across imaging modalities

Course project (computer vision, groups of up to 3). This file is the single source of truth for
*what we are doing and why*. `README.md` covers how to run it. Last updated 2026-10-14.

## 1. One-paragraph pitch

Medical VQA benchmarks report accuracy, but accuracy does not tell you whether the model *looked at
the image*. Recent work shows that a medical VLM can be perfectly stable under rephrased questions
and still give the same answer with the image deleted ("consistent but dangerous"). That work covers
chest X-ray yes/no questions only. We audit **MedGemma-4B across CT, MRI, X-ray and pathology, on
yes/no and open-ended questions**, and add a **continuous** measure of image reliance (how much the
image changes the log-probability of the correct answer), a **wrong-image control**, and a test of
whether these signals can be used to **abstain** on unreliable answers.

## 2. What is already known (so we do not claim it) and what we add

Verified by reading the sources on 2026-10-14:

| Prior work | What it did | Limit we address |
|---|---|---|
| *Consistent but Dangerous* (arXiv 2603.20985, CVPR-W 2026) | Four quadrants: consistency under K=5 paraphrases (strict: all must match) x image reliance (answer changes when image tokens are omitted). 5 model configs, MIMIC-CXR + PadChest, yes/no logits only. LoRA fine-tuning lowers flip rate but moves samples into "Dangerous". Recommends a text-only baseline pass. | Chest X-ray only; yes/no only; reliance is **binary** (the authors note it misses confidence shifts). |
| PSF-Med (arXiv 2602.21428) | Paraphrase-sensitivity benchmark on chest X-ray; low flip rate does not imply grounding. | Same single modality. |
| Why Does Grounding Hurt Medical VQA? (2604.27720) | MedGemma-4B and Lingshu-7B on SLAKE with grounding inputs. | Different question (grounding aids), no reliance audit. |
| HEAL-MedVQA (2505.00744), MedHEval (2503.02157) | Shortcut / hallucination benchmarks. | Not a per-sample consistency x reliance taxonomy. |

**Our delta** (each is a separable contribution):

1. **Cross-modality, cross-dataset audit**: quadrant distribution per modality (CT / MRI / X-ray from SLAKE, pathology from PathVQA, radiology from VQA-RAD) and per question type.
2. **Open-ended answers**, not just yes/no (consistency and reliance defined via token-F1 equivalence; sensitivity-tested).
3. **Continuous reliance**: `PMI = log p(gold | image, q) - log p(gold | no image, q)`. Fixes the stated limitation of the binary test and works for open answers.
4. **Wrong-image control** (`shuf`: another image of the same dataset+modality). Tests whether "image removed" is a valid probe or just an out-of-distribution input.
5. **Selective prediction**: do reliance/consistency signals flag wrong answers better than the model's own confidence? (AUROC, risk-coverage.)
6. **Controlled comparison**: MedGemma-4B vs Gemma-3-4B (same architecture, no medical post-training): does medical tuning change reliance?

> **Novelty is not yet verified.** We have only run a handful of searches. Week-1 task: search arXiv /
> Google Scholar for "image reliance" / "text-only baseline" / "paraphrase consistency" x "SLAKE" /
> "VQA-RAD" / "PathVQA" / "MedGemma". If someone already did items 1-2, items 3-6 still stand.

## 3. Hypotheses (written before seeing results)

Directions are guesses; a null or reversed result is still a finding for an audit.

- **H1 dataset leakage**: a large share of questions is answerable without the image, and this differs by dataset. PathVQA is the likeliest to leak because its questions are auto-generated from captions and name the answer entity (e.g. *"is leiomyosarcoma present?"*).
- **H2 modality / question type**: the "Dangerous" share differs by modality and question type (priors are strong for organ/modality/plane questions, weaker for location/size/abnormality).
- **H3 confidence blindness**: the model's confidence does not separate Dangerous from Ideal answers, while reliance and consistency signals give better selective prediction (lower AURC) than confidence.
- **H4 medical tuning**: MedGemma-4B has a larger Dangerous share than Gemma-3-4B on the same questions (the reference paper reports fine-tuning moves samples into Dangerous).
- **H5 probe validity**: "image removed" and "wrong image" agree only moderately (Cohen's kappa); removal overstates reliance for some question types.

## 4. Experimental design

### 4.1 Data

| Dataset | Source (loader) | Test split used | Metadata we get | Notes |
|---|---|---|---|---|
| SLAKE | `BoKelvin/SLAKE` (CC-BY-4.0) | English test: 1,061 Q on **96 images**; CT 472, X-ray 361, MRI 228; 645 open / 416 "closed" (355 yes/no, 61 are not yes/no and are treated as open) | modality, body location, content type | **Primary dataset for the modality analysis.** |
| VQA-RAD | OSF `89kps` original JSON (CC0) | 451 Q (`test_*` entries) | organ (head/chest/abdomen), question type, phrase type | **2,248 QA on 314 images**, not 3,515 (that is the paper's total incl. template questions; an earlier note of ours was wrong). The HF mirror has no metadata, hence OSF. Modality is *not* annotated. |
| PathVQA | `flaviagiammarino/path-vqa` (MIT) | 6,719 Q / 858 images; we sample 1,000 | none (image ids = sha1 of bytes; types by rule) | Auto-generated from captions; strong text priors expected. |

- **Contamination.** The MedGemma model card lists SLAKE among its training sources and says "there is a risk of data contamination". So headline results use **test splits only**; train/val splits are reserved for an explicit seen-vs-unseen analysis (profile `full`). SLAKE's training exposure is a confound when comparing datasets; say so in the report.
- **Unit of analysis.** The image, not the question. SLAKE test has only 96 images, so every confidence interval is a **cluster bootstrap over images**; per-modality x question-type cells with fewer than ~30 images are labelled exploratory.
- **Unified schema.** Answers that are not yes/no are `open` even when the dataset says CLOSED. `qgroup` is a coarse mapping of each dataset's native types (VQA-RAD "PRES" also contains open questions), so always cross it with `answer_type`.

### 4.2 Model and prompts

- Primary: `google/medgemma-4b-it` (gated; model card version 1.0.1). Control: `google/gemma-3-4b-it`. A newer `medgemma-1.5-4b-it` was mentioned earlier but the model card does not mention it: **unverified**.
- One fixed prompt for every condition (`configs/base.yaml`): the question plus *"Answer with only 'yes' or 'no'."* / *"Answer with a single word or short phrase."* Only the image is added or removed.
- Prompt changes are allowed **only to fix format failures**, never to raise accuracy. Freeze after Phase 2. The format diagnostic is `yn_mass` (raw P(yes)+P(no) at the first answer token; reported per dataset in `by_*.csv`): if it is low, the model is not answering yes/no and the renormalised `P(yes)` is not trustworthy for that slice.
- Greedy decoding. Yes/no from one forward pass: `P(yes) = sigmoid(logsumexp(yes tokens) - logsumexp(no tokens))` at the first answer position, as in the reference paper.

### 4.3 Conditions (`configs/base.yaml`)

| Condition | Image | Question | Purpose |
|---|---|---|---|
| `orig` | original | original | reference answer |
| `noimg` | **tokens omitted** (not zeros) | original | text-only baseline (reference paper's probe) |
| `shuf` | other image, same dataset + modality (fixed partner in manifest) | original | wrong-image control |
| `blank` | black image, same size | original | optional second control (profile `main`+) |
| `para1..K` | original | validated paraphrase k | consistency (K=5; pilot 3) |

### 4.4 Definitions (implemented in `analysis/table.py`, unit-tested)

- `same_answer(a, b)`: yes/no -> same parsed label; open -> normalised exact match **or** token-F1 >= 0.5 (swept 0.3-0.8 as sensitivity).
- `correct`: yes/no -> label equals gold; open -> token-F1(pred, gold) >= 0.5.
- `consistent`: `same_answer(orig, para_k)` for **every** valid paraphrase (strict, as in the reference paper). Computed over the valid paraphrases that survived validation; samples with none are excluded and counted.
- `reliant_noimg = not same_answer(orig, noimg)`; `reliant_shuf` likewise.
- **Quadrant** = (consistent, reliant): Ideal (T,T), Fragile (F,T), **Dangerous (T,F)**, Worst (F,F).
- `PMI_noimg = log p(gold | img, q) - log p(gold | no img, q)`; same for `shuf`. Yes/no uses the binary-renormalised `P(yes)`, open uses the teacher-forced token log-prob of the gold string. **The two scales differ; never pool yes/no with open.**
- Baseline confidence: yes/no -> `max(P(yes), 1-P(yes))`; open -> `exp(mean token log-prob)`.

### 4.5 Paraphrases

Five styles (lexical, syntactic, formal, plain, concise) from a text-only instruct LM (default `Qwen/Qwen2.5-3B-Instruct`, greedy). Automatic validation rejects a paraphrase that changes/drops laterality, direction, negation, quantity words or numbers; leaks the gold answer; is a near-copy (<15% edit); changes length by >3x; or stops being a yes/no question. Rejected ones are kept with reasons (the rejection rate is reportable). **Human audit**: export 100 with `medvlm audit-sheet`; two teammates label "same meaning?" independently; report acceptance rate and Cohen's kappa. Negation-insertion paraphrases (used by the reference paper) are excluded because they invert the correct answer. VQA-RAD ships ~734 clinician-written `para` entries, but their `qid_linked_id` links are noisy (we found a mismatched pair), so they are an optional extra after validation, not a dependency.

## 5. Analyses and outputs

All tables are written to `results/<model>/<profile>/` by `medvlm analyze` with image-clustered 95% CIs.

| # | Question | Output |
|---|---|---|
| A1 | How much is answerable blind? `correct_noimg` vs `correct_orig`, vs the majority-answer prior for yes/no | `by_dataset.csv`, `by_modality.csv` |
| A2 | Quadrant mix per dataset x modality x answer type x question group | same files, `q_*` columns |
| A3 | Do Dangerous answers look confident? Confidence/entropy by quadrant | `per_sample.csv` |
| A4 | Is binary reliance hiding signal? PMI distribution within "not reliant" answers | `pmi_*` columns |
| A5 | Is image removal a valid probe? kappa(`reliant_noimg`, `reliant_shuf`), by question type | `per_sample.csv` |
| A6 | Selective prediction: AUROC / AURC of confidence vs PMI vs consistency | `selective.csv` |
| A7 | MedGemma vs Gemma-3-4B on identical items | second model's results dir |
| A8 | Qualitative: 30 Dangerous and 30 Ideal cases per dataset, with images | figure (Phase 5) |
| A9 | (Phase 7) seen vs unseen: SLAKE train vs test | `full` profile |

Figures are deliberately deferred to Phase 5, when numbers are stable (use the `dataviz` skill).

## 6. Execution plan

Deadlines are not announced yet; weeks below assume ~8 weeks and should be scaled. **Every tier is a
valid stopping point** (this resolves the earlier "which variant?" question).

- **Tier 1 (must ship):** A1 + A2 on MedGemma-4B, three datasets, test splits, K=3 paraphrases.
- **Tier 2 (the differentiators):** continuous PMI (A4), wrong-image control (A5), selective prediction (A6), Gemma-3 comparison (A7), human paraphrase audit.
- **Tier 3 (stronger machine):** more models (Lingshu-7B, MedGemma-27B, LLaVA-Med / HuatuoGPT-Vision), A9, VQA-RAD modality labels, K=5 everywhere.

| Phase | Week | Work | Exit gate |
|---|---|---|---|
| 0 Setup | 0 | **Done in this repo:** package, configs, loaders, sampler, runner, analysis, 36 unit tests, 4 real-model integration tests (SmolVLM, CPU), manifests reproducible. **You:** request MedGemma + Gemma-3 access on Hugging Face (only step that can make you wait); push the repo. | `pytest` green; `medvlm doctor` OK on Colab |
| 1 GPU gate | 1 | Colab: `MEDVLM_TEST_MODEL=medgemma4b pytest -m slow`; `prepare --profile pilot`; `run --limit 200` to **measure throughput**. Decide dtype: bf16 -> else 4-bit (a T4 has no native bf16 and Gemma 3 is fp16-fragile: *to be tested, not assumed*). Sanity: orig-condition token-F1 on SLAKE/VQA-RAD should be in the neighbourhood of the model card (72.3 / 49.9; different prompt and split, so a large gap means a bug, not a finding). | integration tests pass on MedGemma; throughput known |
| 2 Pilot | 1-2 | Generate + validate paraphrases, run human audit (100), run `pilot` (~330 Q), `analyze`, read 30 raw outputs per dataset, fix **format** failures only, **freeze prompt + analysis plan**. Novelty search. Draft 2-page proposal. | parse rate high; audit acceptance rate known; proposal done |
| 3 Main run | 2-4 | `main` profile (see budget) on MedGemma-4B, sharded across sessions/machines; commit paraphrases + small results. | all keys done; `analyze` clean |
| 4 Extras | 4-5 | `gemma3_4b` on the same manifest; shuffled/blank controls; threshold sensitivity (F1 0.3-0.8). | A5, A6, A7 tables |
| 5 Analysis | 5-6 | Figures (quadrant bars per modality, blind-vs-with-image accuracy, risk-coverage), qualitative panel (A8), error analysis. | story fits on one slide |
| 6 Writing | 6-8 | Report (research-paper format: problem, approach, results, implementation), poster, code `.zip` for Canvas, README polish. | submitted |
| 7 Stretch | after | Stronger machine: Tier 3. Optional workshop / arXiv tech report once Tier 2 is solid. | - |

**Cut rule:** if Phase 3 has not started by week 4, drop to Tier 1 and finish; do not start Tier 2 work on unfinished Tier 1 numbers.

## 7. Compute budget (counts, not guesses about speed)

`main` profile: 1,061 + 451 + 1,000 = **2,512 questions**. With 4 image conditions + up to 5 paraphrases that is about **23k generation jobs**, plus about **6k teacher-forced scoring passes** for open questions (4 per open question; ~1,400 open) => **~29k forward passes**. `pilot` is ~330 questions, ~2k jobs. Seconds per pass on a T4 / L4 is **unmeasured**: measure in Phase 1 with `--limit 200`, then extrapolate. Runs are resumable and shardable (`--shard i/n`), so Colab disconnects only cost the current chunk.

Hardware ladder: laptop (CPU, mock + SmolVLM, development and tests) -> Colab / Kaggle GPU (pilot, main) -> stronger machine (bigger models, `full`). Moving up = change `--model` / `--profile` and `run.batch_size`; nothing else.

## 8. Moving to another machine (the "git pull and continue" protocol)

What is in git: code, configs, **manifests** (exact sampled questions, gold answers, shuffle partners, image hashes, validated paraphrases), small result tables. What is not: images (re-fetched and hash-checked by `medvlm prepare`), raw per-sample outputs (`outputs/`, copy by hand or via Drive; the runner skips whatever it finds there).

```bash
git clone https://github.com/thaitri2005/Medvlm-Image-Reliance.git && cd Medvlm-Image-Reliance
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[model,gpu,dev]"
huggingface-cli login                                      # token with access to the gated models
medvlm doctor --model medgemma4b
medvlm prepare --profile main --model medgemma4b           # manifest exists -> only fetches + verifies images
pytest && MEDVLM_TEST_MODEL=medgemma4b pytest -m slow
medvlm run --profile main --model medgemma4b               # resumes; add --shard 0/3 on three sessions
medvlm analyze --profile main --model medgemma4b
```

Per-machine settings (batch size, dtype, 4-bit) go in `configs/local.yaml` (gitignored) or `--set run.batch_size=16`. Set `MEDVLM_OUTPUTS_DIR` to a Drive path on Colab so outputs survive disconnects. To merge runs from several machines, drop their `results.*.jsonl` into one `outputs/<model>/<profile>/` folder.

## 9. Team split (up to 3)

| Role | Owns |
|---|---|
| A, data and language | paraphrase generation + validation, the 100-item human audit (with a second person), VQA-RAD modality labelling (Tier 3), novelty search |
| B, infrastructure and models | Colab/Kaggle runs, HF access, throughput + dtype decisions, second model, `full` profile on the stronger machine |
| C, analysis and writing | tables, figures, qualitative panel, report, poster |

Everyone reviews each other's pull requests and takes part in the audit. Working alone: follow the tiers in order.

## 10. What a researcher will ask, and our answer

- *Is deleting the image a fair probe?* -> `shuf` and `blank` controls, kappa between probes (A5).
- *Contamination?* -> test splits only; SLAKE is the known training source, so datasets are not directly comparable; A9 measures it.
- *Are your paraphrases really equivalent?* -> automatic constraints + two-rater audit with kappa.
- *Open-ended scoring is noisy.* -> threshold sweep; yes/no and open never pooled.
- *Why should I trust the intervals?* -> bootstrap over images, not questions.
- *One model is anecdote.* -> Gemma-3-4B control now, larger models on the stronger machine.

Portfolio value: a public, tested, reproducible repo (rare for course work), a pre-specified analysis plan, and honest negative results.

## 11. Known limitations and risks

- Gemma-3/MedGemma code path is unverified until Phase 1 (we tested the same backend on SmolVLM only). bf16/4-bit on a T4 untested.
- The LLM paraphraser was only smoke-tested with a 0.5B model; paraphrase quality is the most delicate piece.
- VQA-RAD has no modality labels; until labelled, modality conclusions rest on SLAKE (CT/MRI/X-ray) and PathVQA (pathology).
- `shuf` partners can share the gold answer by chance (more likely for yes/no); the effect is bounded and reported.
- `qgroup` is coarse; cells with few images are exploratory.
- Reference-paper definitions were read from its HTML text; re-check thresholds against the PDF before the report claims to "follow" them.
- Gated access, Colab quotas and session limits are the main schedule risks; start the access request today.

## 12. Verified facts log

Checked on 2026-10-14: SLAKE splits/fields/licence and `imgs.zip` layout; VQA-RAD OSF JSON schema, 2,248 rows, 314 images, 451 `test_*`; PathVQA HF fields and split sizes; MedGemma model card (version 1.0.1, SLAKE in training list, contamination warning, tokenized-F1 VQA-RAD 49.9 / SLAKE 72.3 vs Gemma-3-4B 33.6 / 40.2); reference-paper definitions (K=5, strict consistency, binary text-only reliance, yes/no logits).
**Corrections to earlier notes:** VQA-RAD has 2,248 QA pairs (not 3,515). **Unverified:** `medgemma-1.5-4b-it`, novelty of the cross-modality angle, MedGemma throughput.

# Reproducing the audit

Everything below runs from a fresh clone on one GPU with at least 16 GB (all reported runs: RTX 4080 SUPER, bf16,
batch size 1). The study design and every decision are in [PREREGISTRATION.md](PREREGISTRATION.md); the numbers
are in [RESULTS.md](RESULTS.md); this file is only *how to run it*.

## 1. Environment

```bash
git clone https://github.com/thaitri2005/Medvlm-Image-Reliance.git && cd Medvlm-Image-Reliance
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build first
pip install -e ".[model,dev]"
pytest                                                  # 103 fast tests, no GPU / network needed
```

MedGemma and Gemma-3 are gated: accept the terms on their Hugging Face pages
([MedGemma-4B](https://huggingface.co/google/medgemma-4b-it), [MedGemma-1.5-4B](https://huggingface.co/google/medgemma-1.5-4b-it),
[Gemma-3-4B](https://huggingface.co/google/gemma-3-4b-it)), then `hf auth login`. Keep the token out of the repo.

## 2. Pre-flight (about 30 min)

```bash
medvlm doctor   --model medgemma4b                                   # cuda, bf16, HF access, suggested preset
medvlm prepare  --profile pilot --model medgemma4b                   # fetch + SHA-check images
medvlm prepare  --profile main  --model medgemma4b
MEDVLM_TEST_MODEL=medgemma4b pytest -m slow                          # real model, real GPU
medvlm regress  --profile pilot --model medgemma4b --hardware rtx4080 -n 200   # must PASS against the stored pilot
```

`regress` re-runs stored pilot jobs and compares answers and probabilities; it is what makes results from
different GPUs comparable. Every hardware preset (`t4`, `rtx4080`, `rtx4090`, `a100`) runs at **batch size 1**:
in bf16, batching shifts teacher-forced log-probabilities by up to ~1 nat (PREREGISTRATION.md section 13, item 10).

## 3. Primary pair: main run, probes, post-hoc analyses (~1.5 h per model)

```bash
medvlm pipeline --profile main --model medgemma4b --hardware rtx4080 --parts 2
medvlm pipeline --profile main --model gemma3_4b  --hardware rtx4080 --parts 2
medvlm compare  --profile main --model medgemma4b --model-b gemma3_4b   # P3
medvlm probes   --profile main --model medgemma4b --hardware rtx4080    # R1-R3, ~7 min per model
medvlm probes   --profile main --model gemma3_4b  --hardware rtx4080
medvlm posthoc  --profile main --model medgemma4b --model-b gemma3_4b   # post-hoc tables + figures 1-7
```

`pipeline` = verify images -> validate the committed paraphrases (no LM needed) -> load the model once -> part 1
-> checkpoint analysis -> part 2 -> analysis. `probes` writes to separate folders (`outputs/<model>__<probe>/main`)
so the main run is never touched. `posthoc` writes titled figures to `results/posthoc_*/main/` and paper versions
to `report/figures/`.

**Interrupted?** Run the same command again: finished jobs are skipped and a half-written last line is repaired.
`medvlm status --profile main --model medgemma4b --shards 2` shows progress. Every long command writes
`logs/<time>_<command>_<model>_<profile>.log` with progress, ETA, GPU memory and format warnings.

## 4. Replication pairs (PREREGISTRATION.md section 16)

Pair A: `medgemma15_4b` vs `gemma3_4b` (Gemma-3 outputs are reused). Pair B: `medvlthinker_3b` vs `qwen25vl_3b`.
Same protocol; each new model first passes the registered format check.

```bash
MEDVLM_TEST_MODEL=medvlthinker_3b pytest -m slow tests/test_hf_backend.py
medvlm pipeline     --profile pilot --model medvlthinker_3b --hardware rtx4080
medvlm format-check --profile pilot --model medvlthinker_3b          # must PASS before the main run
medvlm pipeline     --profile main  --model medvlthinker_3b --hardware rtx4080 --parts 2
medvlm probes       --profile main  --model medvlthinker_3b --hardware rtx4080
medvlm posthoc      --profile main  --model medvlthinker_3b --model-b qwen25vl_3b
medvlm replication  --profile main                                   # H9-H11, figure 8; per-dataset table + figure 10 (run after contrast --analyze)
```

## 5. Probe R4: opposite-answer images outside the training data (PREREGISTRATION.md section 17)

```bash
medvlm contrast --model all --hardware rtx4080      # five models, one after another; resumable (~1 h)
medvlm contrast --analyze                           # tables + figure 9; no GPU
```

`--model` also takes a comma-separated list (`--model medgemma4b,gemma3_4b`). The manifest
(`manifests/contrast/`, 1,192 PathVQA questions) is committed; the command only fetches and SHA-checks its 682
images. Each model answers every question on its own image (`outputs/<model>/contrast`) and on the
opposite-answer partner (`outputs/<model>__opp/contrast`). `--analyze` writes `results/contrast/`: per-model rates,
paired differences, and the registered H12 tests, which are computed only when all five models are complete.
It also covers the SLAKE probe R3 from the main run.

## 6. Training-data check

```bash
medvlm contamination        # image overlap with each dataset's train split + documented training use
```

Writes `results/contamination/`. It downloads the train splits (SLAKE, VQA-RAD, PathVQA); no model is loaded.

## 7. Commands

| Command | What it does |
|---|---|
| `doctor` | Python / torch / GPU / HF access; suggests a hardware preset |
| `prepare` / `verify` | Sample questions into `manifests/<profile>` (first time only); fetch + hash-verify images |
| `paraphrase` | Generate candidates (LM + templates) and (re)validate all of them against rules and judgments |
| `audit-sheet -n N` / `audit-score` | Export a fresh sample for human rating / score acceptance and kappa |
| `bench`, `estimate` | Time batch sizes on this GPU; job count and wall-clock at given seconds/job |
| `regress -n N` | Re-run stored jobs; PASS if answers and probabilities did not change |
| `run --shard i/n` | Inference only; resumable, `--limit`, `--max-minutes` |
| `pipeline` | The whole main run for one model, in N back-to-back parts |
| `status`, `analyze`, `compare` | Progress; tables in `results/<model>/<profile>/`; paired model comparison (P3) |
| `probes --probe blank\|abstain\|opp\|all` | Robustness probes R1-R3 |
| `posthoc --model-b B` | Post-hoc analyses, probe tests and figures for one pair |
| `format-check`, `replication` | Format rule for a new model; replication tests H9-H11 across pairs |
| `contrast [--analyze]` | Probe R4 and the opposite-answer tables (R3 + R4) |
| `contamination` | Train-split image overlap and documented training use |

Config layering: `configs/base.yaml` -> profile -> model -> hardware -> `configs/local.yaml` (gitignored) ->
`--set a.b=c`.

| Profile | Purpose |
|---|---|
| `smoke` | ~40 questions, CPU pipeline checks (`--model mock` or `smolvlm`) |
| `pilot` | 330 questions: the design-tuning run, and the reference for `regress` and `format-check` |
| `main` | 3,365 questions: SLAKE test + validation, VQA-RAD test, PathVQA test (600 yes/no + 200 open) |
| `contrast` | 1,192 PathVQA test + validation yes/no questions with an opposite-answer partner (probe R4) |

## 8. What is in git, and why it is reproducible

- **Manifests** (`manifests/<profile>/`): the exact sampled questions, gold answers, wrong-image partners and
  SHA hashes of every image; all paraphrase candidates with their rejection reasons; the semantic-equivalence
  judgments; the audit sheets. `prepare` never re-samples an existing manifest.
- **Determinism:** sampling and partners use SHA-1 (never Python's `hash()`); greedy decoding; batch size 1;
  every run writes `meta.json` (config, git commit, versions, host).
- **Registrations** are git tags (`prereg-main`, `prereg-probes`, `prereg-replication`, `prereg-contrast`).
- **Results:** `results/` (index in [results/README.md](../results/README.md)); raw per-call outputs are
  force-added after a finished run (`git add -f outputs/<model>/<profile>`).
- **Not in git:** images (re-fetched and hash-checked), model weights, logs.

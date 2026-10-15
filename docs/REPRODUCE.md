# Reproducing the audit

Everything below runs from a fresh clone. The study design and every decision are in
[PLAN.md](../PLAN.md); this file is only *how to run it*.

## 1. Environment

```bash
git clone https://github.com/thaitri2005/Medvlm-Image-Reliance.git && cd Medvlm-Image-Reliance
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA build first
pip install -e ".[model,dev]"
pytest                                                  # 95 fast tests, no GPU / network needed
```

MedGemma and Gemma-3 are gated: accept the terms on their Hugging Face pages
([MedGemma-4B](https://huggingface.co/google/medgemma-4b-it), [Gemma-3-4B](https://huggingface.co/google/gemma-3-4b-it)),
then `hf auth login`.

## 2. Pre-flight (about 30 min on one GPU)

```bash
medvlm doctor   --model medgemma4b                                   # cuda, bf16, HF access, suggested preset
medvlm prepare  --profile pilot --model medgemma4b                   # fetch + SHA-check images
medvlm prepare  --profile main  --model medgemma4b
MEDVLM_TEST_MODEL=medgemma4b pytest -m slow                          # real model, real GPU
medvlm regress  --profile pilot --model medgemma4b --hardware rtx4080 -n 200   # must PASS against the stored pilot
```

`regress` re-runs stored pilot jobs and compares answers and probabilities; it is what makes results from
different GPUs comparable. Hardware presets (`t4`, `rtx4080`, `rtx4090`, `a100`) all run at **batch size 1**:
in bf16, batching shifts teacher-forced log-probabilities by up to ~1 nat (PLAN.md section 13, item 10).

## 3. The main runs (~1.5 h per model on an RTX 4080 SUPER)

```bash
medvlm pipeline --profile main --model medgemma4b --hardware rtx4080 --parts 2
medvlm pipeline --profile main --model gemma3_4b  --hardware rtx4080 --parts 2
medvlm compare  --profile main --model medgemma4b --model-b gemma3_4b
medvlm probes   --profile main --model medgemma4b --hardware rtx4080    # robustness probes, ~7 min per model
medvlm probes   --profile main --model gemma3_4b  --hardware rtx4080
medvlm posthoc  --profile main --model medgemma4b --model-b gemma3_4b   # all post-hoc tables + figures
```

`probes` runs the three robustness probes registered in PLAN.md section 15 (black image, "unknown" allowed,
opposite-answer image) into separate folders (`outputs/<model>__<probe>/main`). `posthoc` writes tables to
`results/posthoc_*/main/` and figures twice: titled versions there, paper versions to `report/figures/`.
The teaser figure is built by `python report/make_teaser.py`.

`pipeline` = verify images -> validate paraphrases (no LM needed: they are committed) -> load the model once
-> part 1 -> checkpoint analysis -> part 2 -> analysis. Every command writes a timestamped
`logs/<time>_<command>_<model>_<profile>.log` with progress, ETA, GPU memory and format warnings.

**Interrupted?** Run the same command again: finished jobs are skipped, a half-written last line is repaired.
`medvlm status --profile main --model medgemma4b --shards 2` shows progress at any time.

## 4. Commands

| Command | What it does |
|---|---|
| `doctor` | Python / torch / GPU / HF access; suggests a hardware preset |
| `prepare` / `verify` | Sample questions into `manifests/<profile>` (first time only); fetch + hash-verify images |
| `paraphrase` | Generate candidates (LM + templates) and (re)validate all of them against rules and judgments |
| `audit-sheet -n N` / `audit-score` | Export a fresh sample for human rating / score acceptance and kappa |
| `bench` | Time batch sizes on this GPU |
| `regress -n N` | Re-run stored jobs; PASS if answers and probabilities did not change |
| `estimate` | Job count and wall-clock at given seconds/job |
| `run --shard i/n` | Inference only; resumable, `--limit`, `--max-minutes` |
| `pipeline` | The whole study for one model, in N back-to-back parts |
| `status`, `analyze`, `compare` | Progress; tables in `results/<model>/<profile>/`; paired model comparison |
| `probes --probe blank\|abstain\|opp\|all` | Registered robustness probes (PLAN.md section 15) |
| `posthoc --model-b B` | Post-hoc analyses, probe tests and all figures (web + paper versions) |

Config layering: `configs/base.yaml` -> profile -> model -> hardware -> `configs/local.yaml` (gitignored) ->
`--set a.b=c`.

| Profile | Purpose |
|---|---|
| `smoke` | ~40 questions, CPU pipeline checks (`--model mock` or `smolvlm`) |
| `pilot` | 330 questions, design-tuning run on a Colab T4 (done) |
| `main` | 3,365 questions: SLAKE test + validation, VQA-RAD test, PathVQA test (600 yes/no + 200 open) |
| `full` | adds train splits for a contamination analysis (deferred) |

## 5. What is in git, and why it is reproducible

- **Manifests** (`manifests/<profile>/`): the exact sampled questions, gold answers, wrong-image partners and
  SHA hashes of every image; all paraphrase candidates with their rejection reasons; the semantic-equivalence
  judgments; the audit sheets. `prepare` never re-samples an existing manifest.
- **Determinism:** sampling and partners use SHA-1 (never Python's `hash()`); greedy decoding; batch size 1;
  every run writes `meta.json` (config, git commit, versions, host).
- **Results:** `results/<model>/<profile>/*.csv`; raw per-job outputs are force-added after a finished run
  (`git add -f outputs/<model>/main`).
- **Not in git:** images (re-fetched and hash-checked) and model weights.

## 6. Colab

[notebooks/colab_run.ipynb](../notebooks/colab_run.ipynb) wraps the same commands (secret `HF_TOKEN`, GPU
runtime, `--hardware t4`). A free T4 runs at ~4.5 s/job, so split `main` into more parts.

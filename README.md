# Medvlm-Image-Reliance

**Does a medical vision-language model actually use the image?** An audit of MedGemma-4B on SLAKE,
VQA-RAD and PathVQA. Every question is asked with the image, without it, with a *different* image and in
several validated paraphrases; crossing **stability under paraphrase** with **image reliance** sorts each
answer into Ideal / Fragile / **Dangerous** (consistent but identical without the image) / Worst. A
continuous reliance score and selective-prediction tests ask whether this can flag untrustworthy answers.

Objective, target tiers, hypotheses, design and risks: **[PLAN.md](PLAN.md)**.

## Quick start (any machine)

```bash
git clone https://github.com/thaitri2005/Medvlm-Image-Reliance.git && cd Medvlm-Image-Reliance
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[model,dev]"                          # add ,gpu for 4-bit loading on Linux/Colab
pytest                                                 # 78 fast tests, no GPU / network needed
```

No GPU (laptop): everything except real MedGemma runs; use `--model mock` or `--model smolvlm` (CPU).
`pytest -m slow` runs the real-model equivalence tests on SmolVLM / Qwen-0.5B (downloads ~1.5 GB).

## The big run: `main` in 2 parts

MedGemma and Gemma-3 are gated: accept the terms on their Hugging Face pages, then `huggingface-cli login`.
If pip gave you CPU-only torch, install the CUDA build first (`medvlm doctor` must show `cuda=True`).

```bash
medvlm doctor     --model medgemma4b                                # GPU, versions, HF access, suggested preset
medvlm prepare    --profile pilot --model medgemma4b                # fetch + hash-check images (bench/regress use them)
medvlm prepare    --profile main  --model medgemma4b
MEDVLM_TEST_MODEL=medgemma4b pytest -m slow                         # real batching / bf16 on the real GPU
medvlm bench      --model medgemma4b --hardware rtx4090 --profile pilot   # fastest batch size + s/job
medvlm regress    --profile pilot --model medgemma4b --hardware rtx4090 --set run.batch_size=8 -n 100   # must PASS
medvlm estimate   --profile main --model medgemma4b --sec-per-job <from bench>
medvlm pipeline   --profile main --model medgemma4b --hardware rtx4090 --set run.batch_size=8 --parts 2
```

`pipeline` runs prepare -> paraphrases -> MedGemma (loaded once) -> part 1 -> checkpoint analysis -> part 2
-> analysis. Re-running the same command resumes (finished jobs skipped, torn last lines repaired).
`--part 0|1` runs one part; `--max-minutes M` stops cleanly; `medvlm status --shards 2` shows progress.
Hardware presets (`--hardware t4|rtx4080|rtx4090|a100`) set a starting batch size; `bench` finds the real one.

## Commands

| Command | What it does |
|---|---|
| `doctor` | Python / torch / GPU / HF access; suggests a hardware preset |
| `prepare` | Sample questions into `manifests/<profile>` (first time only) and fetch + hash-verify images |
| `paraphrase` | Batched LLM paraphrases + validation; prints yield and rejection reasons |
| `audit-sheet -n 100` | CSV for two humans to check paraphrases |
| `bench` | Time batch sizes 1..16 on this GPU (2-3 min) |
| `regress -n 100` | Re-run stored jobs; PASS if answers/probabilities did not change |
| `estimate` | Job count and wall-clock at given seconds/job |
| `run --shard i/n` | Inference only; resumable, shardable, `--limit`, `--max-minutes` |
| `pipeline` | The whole thing in N back-to-back parts, one process |
| `status`, `analyze` | Progress; tables in `results/<model>/<profile>/` (primary tests, CIs, sensitivity) |
| `compare --model-b` | Paired comparison of two models on the same questions |
| `audit-score` | Paraphrase-audit acceptance and kappa |

## Profiles, models, hardware

| Profile | Purpose |
|---|---|
| `smoke` | ~40 questions, laptop pipeline checks only |
| `pilot` | 330 questions, first MedGemma run (done) |
| `main` | 3,365 questions: SLAKE test + validation (2,114), VQA-RAD test (451), PathVQA test 600 yes/no + 200 open |
| `full` | adds train/val splits (contamination analysis), for a stronger machine |

Models (`--model`): `medgemma4b` (primary), `gemma3_4b` (same architecture, no medical tuning),
`smolvlm` (CPU stand-in), `mock`. Config layering: `base` -> profile -> model -> hardware -> `configs/local.yaml`
(gitignored) -> `--set a.b=c`.

## Colab

[notebooks/colab_run.ipynb](notebooks/colab_run.ipynb) wraps the same commands: set a Colab secret
`HF_TOKEN`, use a GPU runtime, run top to bottom. A free T4 needs `--hardware t4` (batch 1, ~4.5 s/job),
so split into more parts there (`--shards 8`, `--max-minutes 170`). The last cell pushes results to GitHub
(needs a `GH_TOKEN` secret).

## What is in git

Code, configs, **manifests** (the exact sampled questions, gold answers, shuffle partners, image hashes,
validated paraphrases), small result tables, and the raw outputs of finished runs (`git add -f outputs/...`).
Not in git: images (re-fetched and hash-checked by `prepare`), model weights.

```
configs/    base.yaml, profiles/, models/, hardware/
manifests/  <profile>/items.jsonl, paraphrases.jsonl
results/    <model>/<profile>/*.csv
src/medvlm/ data/  models/ (hf, mock)  paraphrase.py  run.py  bench.py  pipeline.py  analysis/
tests/      fast unit tests + `-m slow` real-model tests
```

## Reproducibility

Sampling and shuffle partners use SHA-1, never `hash()` (tested across processes). `prepare` never
re-samples an existing manifest (`--rebuild` is deliberate). Every run writes `meta.json` (config, git
commit, versions). Every speed-up is covered by an equivalence test, and `regress` compares a new
build/batch size against stored results.

## Data and licences

SLAKE (CC-BY-4.0), VQA-RAD (CC0 1.0), PathVQA (MIT); MedGemma / Gemma-3 under their Google terms. Cite the
original dataset papers in any write-up. VQA-RAD is loaded from its original OSF release, PathVQA from the
`flaviagiammarino/path-vqa` Hugging Face mirror.

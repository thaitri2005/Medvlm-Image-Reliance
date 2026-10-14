# Medvlm-Image-Reliance

Does a medical vision-language model actually *use the image*? An audit of MedGemma-4B on SLAKE,
VQA-RAD and PathVQA that crosses **consistency under paraphrase** with **image reliance**
(answer with vs. without the image, plus a wrong-image control and a continuous log-probability
signal). Read [PLAN.md](PLAN.md) for the research design, hypotheses, phases and risks.

## Quick start (any machine)

```bash
git clone https://github.com/thaitri2005/Medvlm-Image-Reliance.git && cd Medvlm-Image-Reliance
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[model,dev]"                          # add ,gpu for 4-bit loading on Linux/Colab
pytest                                                 # fast tests, no GPU, no network needed
```

Laptop without a GPU: everything except the real MedGemma runs. Use `--model mock` (no weights) or
`--model smolvlm` (256M params, CPU) to exercise the pipeline.

```bash
huggingface-cli login                                  # MedGemma / Gemma-3 are gated: accept terms on their HF pages first
medvlm doctor  --model medgemma4b                      # GPU, versions, HF access
medvlm prepare --profile pilot                         # fetch images for the committed manifest and verify hashes
MEDVLM_TEST_MODEL=medgemma4b pytest -m slow            # real-model integration tests (run this first on a GPU)
medvlm paraphrase --profile pilot                      # LLM paraphrases + automatic validation
medvlm audit-sheet --profile pilot -n 100              # CSV for two humans to check paraphrases
medvlm run     --profile pilot --model medgemma4b --limit 200   # measure throughput, then drop --limit
medvlm analyze --profile pilot --model medgemma4b      # tables in results/medgemma4b/pilot/
```

## Profiles and models

| Profile | Purpose |
|---|---|
| `smoke` | ~40 questions, laptop pipeline checks only |
| `pilot` | ~330 questions, first MedGemma run on Colab |
| `main` | all test-split questions (SLAKE 1,061; VQA-RAD 451; PathVQA 1,000 sampled), headline results |
| `full` | adds train/val splits (contamination analysis) for a stronger machine |

Models (`--model`): `medgemma4b` (primary), `gemma3_4b` (same architecture, no medical tuning),
`smolvlm` (CPU stand-in), `mock`. Add a model by dropping a YAML in `configs/models/`; any Hugging Face
image-text-to-text model with a chat template works through the same backend.

## Continuing on a stronger machine

Everything that defines the experiment is in git (configs, sampled questions, shuffle partners, image
hashes, validated paraphrases). Images are re-fetched and hash-checked; raw outputs are not in git. The
runner is **resumable and shardable**, so an interrupted or relocated run picks up where it stopped:

```bash
medvlm run --profile main --model medgemma4b --shard 0/2    # machine 1
medvlm run --profile main --model medgemma4b --shard 1/2    # machine 2
```

Per-machine settings go in `configs/local.yaml` (gitignored), e.g. `run: {batch_size: 16}` or
`model: {quantization: 4bit}`. On Colab set `MEDVLM_OUTPUTS_DIR` to a Drive folder
(see [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb)). Full protocol: PLAN.md section 8.

## Layout

```
configs/    base.yaml, profiles/*.yaml, models/*.yaml        (layered; later wins)
manifests/  <profile>/items.jsonl, paraphrases.jsonl          committed: the exact experiment
results/    <model>/<profile>/*.csv                           committed: small aggregated tables
src/medvlm/ data/ (sources, sampling, prepare)  models/ (hf, mock)  paraphrase.py  run.py  analysis/
tests/      unit tests (fast) + test_hf_backend.py (real model, `-m slow`)
data/ outputs/                                                gitignored: images, raw per-sample outputs
```

## Reproducibility rules

Sampling and shuffle partners use SHA-1, never `hash()`, so manifests are identical across machines
and Python versions (tested across processes). `prepare` never re-samples an existing manifest; use
`--rebuild` deliberately. Every run writes `meta.json` (config, git commit, versions) next to its outputs.

## Data and licences

SLAKE (CC-BY-4.0), VQA-RAD (CC0 1.0), PathVQA (MIT). MedGemma and Gemma-3 are under the Health AI
Developer Foundations / Gemma terms. Cite the original dataset papers in any write-up; we load VQA-RAD
from its original OSF release and PathVQA from the `flaviagiammarino/path-vqa` Hugging Face mirror.

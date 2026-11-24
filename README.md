<h1 align="center">Consistent but Blind?</h1>
<p align="center"><b>Probes of image reliance disagree in medical vision-language models</b></p>
<p align="center">Tri Thai Huu, Hung Nguyen Le · VinUniversity</p>
<p align="center">
<a href="report/main.pdf"><b>Paper</b></a> ·
<a href="report/poster/poster.pdf">Poster</a> ·
<a href="docs/RESULTS.md">All results</a> ·
<a href="docs/PREREGISTRATION.md">Pre-registration</a> ·
<a href="docs/REPRODUCE.md">Reproduce</a>
</p>

**Does a medical vision-language model look at the image? The answer depends on how you ask.**

Remove the image, and MedGemma appears to rely on it *less* than the general model it was built from. Swap the
image for one that requires the opposite answer, and MedGemma appears to rely on it *far more*. Same models, same
questions.

<p align="center"><img src="report/figures/fig0_teaser.png" width="100%"></p>

## What we did

A recent taxonomy, [*Consistent but Dangerous*](https://arxiv.org/abs/2603.20985), flags a medical VLM answer as
**dangerous** when it survives rephrasing and does not change when the image is removed. That makes image removal
the test of whether a model uses the image.

We ran that test next to three tests that keep an image in the input, on the same 1,206 yes/no questions (CT,
MRI, X-ray, pathology), for three medical models and the general-purpose models they were built from:

| Probe | What the model sees |
|---|---|
| Image removed | the question, no image |
| Different image | a random image of the same kind |
| Black image | a black image |
| **Opposite-answer image** | an image on which the same question has the opposite answer |

Models: MedGemma-4B and MedGemma-1.5-4B vs Gemma-3-4B; MedVLThinker-3B vs Qwen2.5-VL-3B.

## What we found

<p align="center"><img src="results/replication_main/fig10_by_dataset.png" width="100%"></p>

*Each point is a medical model minus its base: how much more often its answer changes when the image is altered.
Right of zero, the medical model looks more image-reliant. If the probes measured one thing, each dataset's
markers would line up vertically.*

**1. The probes are not interchangeable.** In 6 of 9 model-pair × dataset combinations, image removal and the
probes that keep an image measure a different gap between the two models. On SLAKE the sign flips: image removal
puts MedGemma-4B 10 points behind Gemma-3, the opposite-answer image puts it 48 points ahead.

**2. Refusing to answer is part of it, not all of it.** Without an image, Gemma-3 asks for one on 29% of
questions. Its forced yes/no answers there are noise, which makes it look more image-reliant. Dropping those
questions explains part of one model pair's gap and little of the others.

**3. The most direct probe is accuracy in disguise.** On opposite-answer pairs, a model that ignores the image is
right exactly half the time. How often a model follows the image is therefore its accuracy where guessing cannot
help. That makes it the best of the four probes, and it means the probe rewards whatever a model was trained on.

**4. The medical models' advantage does not show up on held-out data.** We fixed this test in a git tag before
running it.

| Follows an opposite-answer image, medical − general | SLAKE (in MedGemma's training data) | PathVQA (held out) |
|---|---|---|
| MedGemma-4B − Gemma-3 | **+48.0** [41.9, 53.8] | +0.8 [−2.7, 4.2] |
| MedGemma-1.5 − Gemma-3 | **+36.5** [30.6, 42.1] | −5.6 [−9.1, −2.0] |
| MedVLThinker − Qwen2.5-VL | −1.6 [−4.9, 1.4] | −0.9 [−3.0, 1.3] |

*Points, with 95% confidence intervals.*

## What to take away

- Name the probe. "Does not rely on the image" means different things under different probes.
- Prefer images that require the opposite answer, and read the result as accuracy without priors.
- Compare models on data none of them was trained on.
- Let models abstain. Declining to answer about a missing image is the right behaviour.

## What this does not show

- Three model pairs of 3–4B parameters show that probes *can* disagree, not how often they do.
- Every sign flip is on SLAKE, which MedGemma was trained on. On held-out data no disagreement between probes is
  shown.
- The held-out test is pathology only, where every model is weak, so "no difference" may mean the task is equally
  hard for all of them.
- The paraphrases behind the "consistent" label were judged by an LLM (Claude), not by independent human raters.
  The comparison between probes does not use them.

Full list: section 6 of the [paper](report/main.pdf).

## How it was built

- **Pre-registered.** The main tests were fixed before the first run, and every later probe and model was
  registered in a git tag (`prereg-*`) before it was run. Failed predictions are reported, and one analysis was
  withdrawn after review; the [pre-registration](docs/PREREGISTRATION.md) logs both.
- **One consumer GPU.** About 120,000 model calls on a 16 GB card, with a resumable inference engine: an
  interrupted run continues where it stopped.
- **Reproducible without a GPU.** Raw model outputs are committed, so every table and figure regenerates on a
  laptop.
- **Deterministic.** In bf16, batch size alone shifts log-probabilities by up to 1 nat on identical inputs, so
  every run uses batch size 1.
- **Tested.** 105 unit tests, including a simulation showing that the confidence intervals reach nominal coverage.

## Run it

```bash
pip install -e ".[dev]" && pytest          # 105 tests, no GPU needed
medvlm contrast --analyze                  # rebuild the held-out tables and figure from committed outputs
medvlm replication --profile main          # rebuild the probe comparison (10+ minutes)
```

Running the models themselves needs a 16 GB GPU and access to the gated MedGemma and Gemma-3 weights:
[docs/REPRODUCE.md](docs/REPRODUCE.md).

| Folder | Contents |
|---|---|
| `report/` | paper, poster, figures |
| `docs/` | every result, the pre-registration, how to reproduce |
| `src/medvlm/` | data loaders, inference, scoring, analysis, CLI |
| `manifests/`, `outputs/`, `results/` | exact questions and image hashes, raw model outputs, tables and figures |

## Cite

```bibtex
@misc{thaihuu2026consistentblind,
  title  = {Consistent but Blind? Probes of Image Reliance Disagree in Medical Vision-Language Models},
  author = {Thai Huu, Tri and Nguyen Le, Hung},
  year   = {2026},
  note   = {VinUniversity. https://github.com/thaitri2005/Medvlm-Image-Reliance}
}
```

Contact: 23tri.th@vinuni.edu.vn, 23hung.nl@vinuni.edu.vn, or open an issue.

Code: MIT. Data: SLAKE (CC-BY-4.0), VQA-RAD (CC0), PathVQA (MIT). Models are under their own licences. Benchmark
VQA is not clinical use, and nothing here is a clinical claim.

# Results index

Every table and figure is written by a command and committed; [docs/RESULTS.md](../docs/RESULTS.md) reports them
with their meaning. Intervals are 95% cluster-bootstrap CIs by image (B = 2,000, seed 1234) unless a file says
otherwise. Models: `medgemma4b`, `gemma3_4b` (primary pair), `medgemma15_4b`, `medvlthinker_3b`, `qwen25vl_3b`.

| Folder | Written by | Contents |
|---|---|---|
| `<model>/main/` | `medvlm analyze` (also run by `pipeline`) | `primary.csv` (P1, P2 as registered), `by_dataset*.csv`, `by_modality.csv`, `by_modality_pooled.csv` (SLAKE test + validation), `by_qgroup.csv`, `selective.csv`, `agreement.csv` (probe kappa), `sensitivity.csv` (grid), `reference_check.csv` (vs the model card), `timing.csv`, `per_sample.csv` |
| `<model>/pilot/` | `medvlm pipeline --profile pilot` | the design-tuning run; reference for `regress` and `format-check` |
| `compare_<a>_vs_<b>/main/` | `medvlm compare` | `primary.csv` (P3), `paired_differences.csv` |
| `posthoc_<a>_vs_<b>/main/` | `medvlm posthoc` | registered probe tests (`probe_tests.csv`, `probe_blank.csv`, `probe_abstain.csv`, `probe_opp.csv`), replication tests for the pair (`replication_tests.csv`), and post-hoc analyses: `reversal.csv` (every probe on one scale), `removal_split.csv`, `abstention.csv`, `chance.csv`, `harm.csv` (correctness split), `selective_labelfree.csv` (P2 on the model's own answer), `partner_confound.csv`, `no_pilot.csv`, `p1_sensitivity.csv`, `p3_sensitivity.csv`; figures 1-7 |
| `replication_main.csv`, `replication_main/` | `medvlm replication` (after `contrast --analyze`) | H9-H11 across pairs (Holm), figure 8; `by_dataset.csv` (post hoc: medical - general under every probe, per dataset and pair, with refusal gaps), figure 10 |
| `contrast/` | `medvlm contrast --analyze` | opposite-answer probes R3 (SLAKE) and R4 (PathVQA): `contrast_models.csv` (follow rate, pair accuracy, both wrong, stuck), `contrast_pairs.csv` (paired gaps, template-clustered sensitivity), `contrast_tests.csv` (H12a-c), figure 9 |
| `contamination/` | `medvlm contamination` | `overlap.csv` (evaluated images also in the dataset's train split), `training_use.csv` (what each model's documentation says, with source and date) |

Paper-sized versions of figures 1-10 are written to [report/figures/](../report/figures/).

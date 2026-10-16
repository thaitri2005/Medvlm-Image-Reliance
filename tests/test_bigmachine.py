"""The pieces that matter on the big machine: hardware presets, bench, regress, the 2-part pipeline."""
from medvlm.analysis.table import load_results
from medvlm.bench import bench
from medvlm.config import load_config
from medvlm.data.sampling import stratified_sample
from medvlm.models.base import Output
from medvlm.models.mock import MockVLM
from medvlm.paraphrase import TemplateParaphraser
from medvlm.pipeline import pipeline
from medvlm.run import done_keys, regress, run, select_jobs

from .conftest import make_items


def test_hardware_presets_keep_experiments_at_batch_1():
    base = load_config("main", "medgemma4b")
    assert base["run"]["batch_size"] == 1                                   # safe default
    for hw in ("t4", "rtx4080", "rtx4090", "a100"):                         # docs/PREREGISTRATION.md section 13, item 10
        assert load_config("main", "medgemma4b", hardware=hw)["run"]["batch_size"] == 1
    assert load_config("main", "medgemma4b", ["run.batch_size=2"], hardware="rtx4090")["run"]["batch_size"] == 2  # --set wins
    assert load_config("main", "medgemma4b", hardware="a100")["hardware"] == "a100"


def test_per_stratum_quota_for_sampling():
    rows = [{"sample_id": f"s{i}", "answer_type": "open" if i % 2 else "yesno"} for i in range(100)]
    got = stratified_sample(rows, {"open": 10, "yesno": 30}, ["answer_type"], seed=1)
    assert sum(r["answer_type"] == "open" for r in got) == 10 and sum(r["answer_type"] == "yesno" for r in got) == 30
    assert got == stratified_sample(rows, {"open": 10, "yesno": 30}, ["answer_type"], seed=1)


def test_main_profile_uses_the_tuned_pathvqa_quota_and_no_blank():
    cfg = load_config("main", "medgemma4b")
    path = next(d for d in cfg["datasets"] if d["name"] == "pathvqa")
    assert path["n"] == {"open": 200, "yesno": 600}
    assert "blank" not in cfg["conditions"] and cfg["paraphrase"]["min_edit_ratio"] == 0.10


def test_bench_runs_and_reports_best_batch(paths, cfg, capsys):
    make_items(paths, n_images=6, per_image=6)
    best = bench(cfg, paths, MockVLM(), sizes=[1, 2], n=8)
    assert best is not None and best["batch_size"] in (1, 2)
    assert "fastest" in capsys.readouterr().out


def test_regress_passes_on_identical_model_and_fails_when_results_drift(paths, cfg, capsys):
    from tests.test_runner import _setup

    _setup(paths, cfg)
    run(cfg, paths, MockVLM())
    assert regress(cfg, paths, MockVLM(), n=20)

    class Drifted(MockVLM):
        def generate(self, queries):
            return [Output("yes" if o.text != "yes" else "no", p_yes=1 - (o.p_yes or 0.5), gold_logp=o.gold_logp)
                    for o in super().generate(queries)]

    assert not regress(cfg, paths, Drifted(), n=20)
    assert "FAIL" in capsys.readouterr().out


def test_pipeline_runs_two_parts_back_to_back_and_is_resumable(paths, cfg, capsys):
    make_items(paths)
    cfg["paraphrase"]["n"] = 3
    pipeline(cfg, paths, parts=2, paraphrase_backend=TemplateParaphraser())
    out = capsys.readouterr().out
    assert "part 1 of 2" in out and "part 2 of 2" in out
    everything = {j.key for j in select_jobs(cfg, paths, None)}
    assert done_keys(paths.run_dir("mock", "smoke")) == everything
    assert (paths.result_dir("mock", "smoke") / "per_sample.csv").exists()
    assert len(load_results(paths.run_dir("mock", "smoke"))) == len(everything)
    pipeline(cfg, paths, parts=2)                      # second call: nothing to do, no paraphrase backend needed
    assert done_keys(paths.run_dir("mock", "smoke")) == everything


def test_pipeline_can_run_a_single_part(paths, cfg):
    make_items(paths)
    cfg["paraphrase"]["n"] = 2
    pipeline(cfg, paths, parts=2, part=1, paraphrase_backend=TemplateParaphraser())
    done = done_keys(paths.run_dir("mock", "smoke"))
    assert done == {j.key for j in select_jobs(cfg, paths, (1, 2))}


def test_analyze_writes_every_research_table(paths, cfg):
    from medvlm.analysis.report import analyze
    from tests.test_runner import _setup

    _setup(paths, cfg)
    run(cfg, paths, MockVLM())
    out = analyze(cfg, paths)
    for name in ("primary", "selective", "agreement", "sensitivity", "reference_check", "by_modality"):
        assert (paths.result_dir("mock", "smoke") / f"{name}.csv").exists(), name
    s = out["sensitivity"]
    assert set(s["f1_threshold"]) == {0.3, 0.5, 0.8} and set(s["min_paraphrases"]) == {1, 2, 3}
    assert s["is_default"].any()
    assert "exploratory" in out["by_modality"] and "blind_gain" in out["by_dataset_type"]
    if len(out["primary"]):
        assert "p_holm" in out["primary"]


def test_compare_two_models_paired_on_the_same_questions(paths, cfg):
    from medvlm.analysis.compare import compare
    from tests.test_runner import _setup

    class Other(MockVLM):
        name = "other"

        def generate(self, queries):
            outs = super().generate(queries)
            for o in outs:
                if o.p_yes is not None:
                    o.p_yes = 1 - o.p_yes
                    o.text = "yes" if o.p_yes >= 0.5 else "no"
            return outs

    _setup(paths, cfg)
    run(cfg, paths, MockVLM())
    cfg_b = {**cfg, "model": {"name": "other", "hf_id": None}}
    run(cfg_b, paths, Other())
    out = compare(cfg, paths, "mock", "other")
    assert len(out["paired_differences"]) > 0 and "P3" in out["primary"]["endpoint"].iloc[0]
    assert (paths.results / "compare_mock_vs_other" / "smoke" / "primary.csv").exists()


def test_strong_flip_ignores_tiny_moves_near_one_half(paths, cfg):
    import pandas as pd

    from medvlm.analysis.table import build_sample_table

    items = make_items(paths, n_images=1, per_image=1)          # one yes/no item
    sid = items[0].sample_id
    rows = [dict(key=f"{sid}|orig", sample_id=sid, condition="orig", text="yes", p_yes=0.55, entropy=0.1, mean_logprob=None, gold_logp=-0.6),
            dict(key=f"{sid}|noimg", sample_id=sid, condition="noimg", text="no", p_yes=0.45, entropy=0.1, mean_logprob=None, gold_logp=-0.8)]
    r = build_sample_table(items, pd.DataFrame(rows)).iloc[0]
    assert r["reliant_noimg"] and not r["reliant_noimg_strong"]     # flipped, but only 0.4 nats


def test_audit_scoring_acceptance_and_kappa(tmp_path):
    import pandas as pd

    from medvlm.analysis.stats import cohen_kappa
    from medvlm.paraphrase import score_audit

    a = ["y", "y", "n", "y", "y", "n", "y", "y", "", "y"]
    b = ["y", "n", "n", "y", "y", "n", "y", "y", "y", ""]
    f = tmp_path / "audit.csv"
    pd.DataFrame({"sample_id": range(10), "k": 1, "style": "formal", "original": "q", "paraphrase": "p",
                  "same_meaning_A": a, "same_meaning_B": b, "notes": ""}).to_csv(f, index=False)
    r = score_audit(f)
    assert r["n_rated_A"] == 9 and abs(r["accept_A"] - 7 / 9) < 1e-12 and r["n_both"] == 8
    both = [(x == "y", y == "y") for x, y in zip(a, b) if x and y]
    assert abs(r["kappa"] - cohen_kappa(pd.Series([p for p, _ in both]), pd.Series([q for _, q in both]))) < 1e-12

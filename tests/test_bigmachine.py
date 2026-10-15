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


def test_hardware_preset_overrides_the_model_default():
    base = load_config("main", "medgemma4b")
    assert base["run"]["batch_size"] == 1                                   # safe default
    assert load_config("main", "medgemma4b", hardware="rtx4090")["run"]["batch_size"] == 8
    assert load_config("main", "medgemma4b", hardware="rtx4080")["run"]["batch_size"] == 4
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

from medvlm.analysis.report import analyze
from medvlm.models.mock import MockVLM
from medvlm.paraphrase import TemplateParaphraser, generate
from medvlm.run import done_keys, run
from medvlm.schema import load_items

from .conftest import make_items


def _setup(paths, cfg):
    items = make_items(paths)
    cfg["paraphrase"]["n"] = 3
    generate(items, cfg, paths.manifest_dir("smoke") / "paraphrases.jsonl", backend=TemplateParaphraser())
    return items


def test_run_resumes_after_interruption_and_never_duplicates(paths, cfg):
    _setup(paths, cfg)
    first = run(cfg, paths, MockVLM(), limit=10, chunk=4)
    assert first == 10
    run_dir = paths.run_dir("mock", "smoke")
    assert len(done_keys(run_dir)) == 10

    rest = run(cfg, paths, MockVLM(), chunk=4)           # simulates restart / another machine
    assert rest > 0 and run(cfg, paths, MockVLM()) == 0  # nothing left
    keys = [line for f in run_dir.glob("results*.jsonl") for line in f.read_text().splitlines()]
    assert len(keys) == len(set(keys))                   # no job was run twice


def test_conditions_cover_every_sample(paths, cfg):
    items = _setup(paths, cfg)
    run(cfg, paths, MockVLM())
    keys = done_keys(paths.run_dir("mock", "smoke"))
    for it in items:
        for c in ("orig", "noimg", "shuf"):
            assert f"{it.sample_id}|{c}" in keys


def test_shards_partition_the_work(paths, cfg):
    _setup(paths, cfg)
    a = run(cfg, paths, MockVLM(), shard=(0, 2))
    b = run(cfg, paths, MockVLM(), shard=(1, 2))
    total = len(done_keys(paths.run_dir("mock", "smoke")))
    assert a > 0 and b > 0 and a + b == total


def test_end_to_end_analysis_produces_quadrants_and_tables(paths, cfg):
    _setup(paths, cfg)
    run(cfg, paths, MockVLM())
    out = analyze(cfg, paths)
    df = out["per_sample"]
    assert len(df) == len(load_items(paths.manifest_dir("smoke") / "items.jsonl"))
    assert set(df["quadrant"].dropna()) <= {"ideal", "fragile", "dangerous", "worst"}
    assert {"by_dataset", "by_modality", "by_qgroup", "selective"} <= set(out)
    assert (paths.result_dir("mock", "smoke") / "per_sample.csv").exists()


def test_runner_records_seconds_per_job_and_analyze_reports_timing(paths, cfg):
    _setup(paths, cfg)
    run(cfg, paths, MockVLM())
    analyze(cfg, paths)
    timing = paths.result_dir("mock", "smoke") / "timing.csv"
    assert timing.exists() and "orig" in timing.read_text()


def test_estimate_uses_real_paraphrase_counts(paths, cfg, capsys):
    from medvlm.run import estimate

    _setup(paths, cfg)
    estimate(cfg, paths, [2.0])
    out = capsys.readouterr().out
    assert "actual" in out and "at 2 s/job" in out


def test_paraphrase_summary_reports_yield(paths, cfg, capsys):
    from medvlm.cli import _paraphrase_summary

    _setup(paths, cfg)
    _paraphrase_summary(paths.manifest_dir("smoke") / "paraphrases.jsonl", cfg)
    out = capsys.readouterr().out
    assert "valid" in out and "rejected for" in out and "valid by style" in out

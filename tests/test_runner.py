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


def test_sequential_shards_cover_everything_exactly_once(paths, cfg):
    """Back-to-back sessions: shard 0/3, then 1/3, then 2/3 -> every job done once, none twice."""
    from medvlm.run import select_jobs

    _setup(paths, cfg)
    everything = {j.key for j in select_jobs(cfg, paths, None)}
    seen = []
    for i in range(3):
        seen += [j.key for j in select_jobs(cfg, paths, (i, 3))]
        run(cfg, paths, MockVLM(), shard=(i, 3))
    assert sorted(seen) == sorted(everything) and len(seen) == len(set(seen))
    assert done_keys(paths.run_dir("mock", "smoke")) == everything


def test_time_budget_stops_cleanly_and_rerun_finishes(paths, cfg):
    _setup(paths, cfg)
    n1 = run(cfg, paths, MockVLM(), chunk=4, max_minutes=1e-9)    # budget already spent
    assert n1 <= 4                                                 # at most the chunk in flight (coarse clocks)
    rest = run(cfg, paths, MockVLM(), chunk=4)
    assert rest > 0 and run(cfg, paths, MockVLM()) == 0
    assert len(done_keys(paths.run_dir("mock", "smoke"))) == n1 + rest


def test_partial_run_gives_complete_questions(paths, cfg):
    """Interrupted runs must leave whole questions (all conditions), not all-orig-no-noimg."""
    from medvlm.run import select_jobs

    _setup(paths, cfg)
    jobs = select_jobs(cfg, paths, None)
    first_ids = []
    for j in jobs:
        if j.item.sample_id not in first_ids:
            first_ids.append(j.item.sample_id)
    first = first_ids[0]
    prefix = [j.key for j in jobs[:len([j for j in jobs if j.item.sample_id == first])]]
    assert all(k.startswith(first + "|") for k in prefix)


def test_status_reports_progress(paths, cfg, capsys):
    from medvlm.run import status

    _setup(paths, cfg)
    run(cfg, paths, MockVLM(), shard=(0, 2))
    status(cfg, paths, 2)
    out = capsys.readouterr().out
    assert "shard 0/2" in out and "<- done" in out and "total:" in out

"""medvlm <command> --profile <p> --model <m> [--set a.b=c ...]

  doctor        check Python / torch / GPU / HF access on this machine
  prepare       sample questions into manifests/<profile> (first time) and fetch + verify images
  verify        check the local image cache against the manifest
  paraphrase    generate + validate paraphrases into manifests/<profile>/paraphrases.jsonl
  audit-sheet   export a CSV of paraphrases for human spot-checking
  run           run the model on every (sample, condition); resumable
  estimate      job count and wall-clock for a profile at given seconds/job
  analyze       aggregate a run into results/<model>/<profile>/*.csv
"""
from __future__ import annotations

import argparse
import sys

from .config import get_paths, load_config


def _doctor(cfg, paths, args):
    import platform

    print(f"python      {platform.python_version()} on {platform.system()}")
    try:
        import torch

        print(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print(f"gpu         {p.name}  {p.total_memory / 2**30:.1f} GiB  bf16={torch.cuda.is_bf16_supported()}")
    except ImportError:
        print("torch       NOT INSTALLED (pip install -e .[model])")
    try:
        import transformers

        print(f"transformers {transformers.__version__}  (Gemma 3 / MedGemma needs >= 4.50)")
    except ImportError:
        print("transformers NOT INSTALLED")
    hf_id = (cfg.get("model") or {}).get("hf_id")
    if hf_id:
        try:
            from huggingface_hub import HfApi

            HfApi().auth_check(hf_id)
            print(f"hf access   OK for {hf_id}")
        except Exception as e:  # noqa: BLE001
            print(f"hf access   FAILED for {hf_id}: {type(e).__name__} (accept the terms on the model page, "
                  f"then `huggingface-cli login`)")
    print(f"data dir    {paths.data}\noutputs dir {paths.outputs}")


def _prepare(cfg, paths, args):
    from .data.prepare import prepare, summarize

    items = prepare(cfg, paths, rebuild=args.rebuild)
    print(f"[prepare] {len(items)} items, {len({i.image_id for i in items})} images")
    for k, v in summarize(items).items():
        print(f"   {k:45s} {v}")


def _verify(cfg, paths, args):
    from .data.prepare import verify

    problems = verify(cfg, paths)
    print("\n".join(problems) if problems else "[verify] OK")
    return 1 if problems else 0


def _paraphrase(cfg, paths, args):
    from .paraphrase import generate
    from .schema import load_items

    items = load_items(paths.manifest_dir(cfg["profile"]) / "items.jsonl")
    n = generate(items, cfg, paths.manifest_dir(cfg["profile"]) / "paraphrases.jsonl")
    print(f"[paraphrase] wrote {n} rows")
    _paraphrase_summary(paths.manifest_dir(cfg["profile"]) / "paraphrases.jsonl", cfg)


def _paraphrase_summary(path, cfg):
    """Check this BEFORE the long run: low yield means consistency is measured on too few samples."""
    import collections

    from .schema import read_jsonl

    rows = list(read_jsonl(path))
    per = collections.Counter(r["sample_id"] for r in rows if r["valid"])
    ids = {r["sample_id"] for r in rows}
    need = cfg["analysis"]["min_paraphrases"]
    ok = sum(1 for i in ids if per[i] >= need)
    print(f"[paraphrase] valid {sum(r['valid'] for r in rows)}/{len(rows)}; "
          f"{ok}/{len(ids)} questions ({ok / max(len(ids), 1):.0%}) have >= {need} valid paraphrases")
    print("   rejected for:", dict(collections.Counter(x for r in rows for x in r["reasons"]).most_common()))
    by = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        by[r["style"]][0] += r["valid"]
        by[r["style"]][1] += 1
    print("   valid by style:", {k: f"{v[0]}/{v[1]}" for k, v in by.items()})


def _audit(cfg, paths, args):
    from .paraphrase import export_audit_sheet
    from .schema import load_items

    mdir = paths.manifest_dir(cfg["profile"])
    n = export_audit_sheet(load_items(mdir / "items.jsonl"), mdir / "paraphrases.jsonl",
                           mdir / "paraphrase_audit.csv", args.n, cfg["seed"])
    print(f"[audit-sheet] {n} rows -> {mdir / 'paraphrase_audit.csv'}")


def _run(cfg, paths, args):
    from .models import load_model
    from .run import run

    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    run(cfg, paths, load_model(cfg), limit=args.limit, shard=shard)


def _estimate(cfg, paths, args):
    from .run import estimate

    estimate(cfg, paths, args.sec_per_job or [4.5, 1.5, 0.5])


def _analyze(cfg, paths, args):
    from .analysis.report import analyze

    analyze(cfg, paths)


COMMANDS = {"doctor": _doctor, "prepare": _prepare, "verify": _verify, "paraphrase": _paraphrase,
            "audit-sheet": _audit, "run": _run, "estimate": _estimate, "analyze": _analyze}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="medvlm", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--profile", default="pilot")
    ap.add_argument("--model", default="medgemma4b")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override any config key")
    ap.add_argument("--rebuild", action="store_true", help="prepare: re-sample and overwrite the manifest")
    ap.add_argument("--limit", type=int, help="run: stop after N jobs (smoke tests)")
    ap.add_argument("--shard", help="run: 'i/n' to split samples across machines/sessions")
    ap.add_argument("--sec-per-job", type=float, action="append", help="estimate: seconds per job (repeatable)")
    ap.add_argument("-n", type=int, default=100, help="audit-sheet: number of paraphrases to export")
    args = ap.parse_args(argv)
    cfg = load_config(args.profile, args.model, args.set)
    return COMMANDS[args.command](cfg, get_paths(cfg), args) or 0


if __name__ == "__main__":
    sys.exit(main())

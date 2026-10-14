"""Command-line interface: ``milner run | report | make-toy | list``."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__


def _cmd_run(args) -> int:
    from .config import load_config
    from .report import build_report, build_table, to_markdown
    from .runner import read_logs, run_experiment

    cfg = load_config(args.config)
    if args.limit:
        cfg.limit = args.limit

    def progress(label, done, total):
        if done == total:
            print(f"  {label}: {total} examples", file=sys.stderr)

    print(f"Running {cfg.name!r} -> {cfg.run_dir}", file=sys.stderr)
    run_experiment(cfg, progress=progress)
    written = build_report(cfg.run_dir)
    print(to_markdown(build_table(read_logs(cfg.run_dir))))
    for kind, path in written.items():
        print(f"{kind}: {path}", file=sys.stderr)
    if "plot" not in written:
        print("plot skipped (pip install 'milner[plots]')", file=sys.stderr)
    return 0


def _cmd_report(args) -> int:
    from .report import build_report, build_table, to_markdown
    from .runner import read_logs

    written = build_report(args.run_dir)
    print(to_markdown(build_table(read_logs(args.run_dir))))
    for kind, path in written.items():
        print(f"{kind}: {path}", file=sys.stderr)
    return 0


def _cmd_make_toy(args) -> int:
    from .schema import dump_jsonl
    from .toy import make_toy

    examples = make_toy(n=args.n, turns=args.turns, facts=args.facts, seed=args.seed)
    dump_jsonl(examples, args.output)
    print(f"wrote {len(examples)} examples to {args.output}", file=sys.stderr)
    return 0


def _cmd_list(_args) -> int:
    from .memory import MEMORIES
    from .models import MODELS

    print("memories:", ", ".join(sorted(MEMORIES)))
    print("models:  ", ", ".join(sorted(MODELS)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="milner", description=__doc__)
    p.add_argument("--version", action="version", version=f"milner {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment from a YAML config and report")
    run.add_argument("config")
    run.add_argument("--limit", type=int, help="only the first N examples")
    run.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="rebuild table and plots from raw logs")
    rep.add_argument("run_dir")
    rep.set_defaults(func=_cmd_report)

    toy = sub.add_parser("make-toy", help="generate the synthetic toy benchmark")
    toy.add_argument("-o", "--output", default="toy.jsonl")
    toy.add_argument("-n", type=int, default=20)
    toy.add_argument("--turns", type=int, default=30)
    toy.add_argument("--facts", type=int, default=3)
    toy.add_argument("--seed", type=int, default=0)
    toy.set_defaults(func=_cmd_make_toy)

    sub.add_parser("list", help="list registered memories and models").set_defaults(func=_cmd_list)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

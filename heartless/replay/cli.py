"""`heartless replay`: argument parsing and the CLI entry point (kept out of heartless/cli.py on purpose)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def add_replay_parser(sub) -> argparse.ArgumentParser:
    rp = sub.add_parser("replay", help="full-system replay soak: the real bot on stored history in virtual time")
    rp.add_argument("--start", required=True, help="ISO date/time (UTC unless an offset is given)")
    rp.add_argument("--end", required=True, help="ISO date/time; a date-only end is exclusive")
    rp.add_argument("--symbols", default=None, help="comma separated (default: every symbol in the source store)")
    rp.add_argument("--source", default=None, help="store to replay (default: <HEARTLESS_DATA_DIR>/heartless.db)")
    rp.add_argument("--work-dir", default=None, help="data dir of the replayed bot (default: a new temp dir)")
    rp.add_argument("--overwrite", action="store_true", help="wipe a non-empty --work-dir first")
    rp.add_argument("--speed", default="max", help="'max' or virtual seconds per real second (e.g. 600)")
    rp.add_argument("--seed-days", type=int, default=60, help="history before --start copied into the bot's store")
    rp.add_argument("--seed-metrics", action="store_true", help="also copy the 5m metrics (default: none, like a bot that only starts polling them now)")
    rp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="bot setting override, e.g. RESEARCH_INTERVAL_MINUTES=1440 (env-style names)")
    rp.add_argument("--params", default=None, help="StrategyParams JSON installed as the starting champion")
    rp.add_argument("--command", action="append", default=[], metavar="WHEN=TEXT",
                    help="scripted owner command added to the built-in script, e.g. '+2d=/pause' or "
                         "'2026-07-03T06:00=/resume' (repeatable)")
    rp.add_argument("--no-default-script", action="store_true", help="send only the --command commands")
    rp.add_argument("--memory-limit-mb", type=float, default=300.0)
    rp.add_argument("--memory-slope-mb-day", type=float, default=5.0)
    rp.add_argument("--tracemalloc", action="store_true", help="also trace Python allocations (slower)")
    rp.add_argument("--no-flatten", action="store_true", help="skip the final close-all")
    rp.add_argument("--out", default=None, help="summary JSON (a human-readable .txt is written next to it)")
    rp.add_argument("--messages-out", default=None, help="every captured Telegram message as JSONL")
    rp.add_argument("--trades-out", default=None, help="every stored trade as JSON")
    rp.add_argument("--log-level", default="WARNING")
    rp.add_argument("--profile", action="store_true",
                    help="profile the event-loop thread and list the hottest functions in the report (slower)")
    return rp


def _value(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def main_cli(settings, args) -> None:
    from heartless.learning.lab import parse_when
    from heartless.replay.runner import ReplayConfig, run_replay
    from heartless.replay.telegram import parse_command

    start, end = parse_when(args.start), parse_when(args.end, end=True)
    overrides = {}
    for item in args.set:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"--set {item!r}: expected KEY=VALUE")
        overrides[k.strip().upper()] = _value(v)
    from heartless.replay.runner import DEFAULT_SCRIPT

    specs = ([] if args.no_default_script else list(DEFAULT_SCRIPT)) + list(args.command)
    commands = [parse_command(c, start) for c in specs]
    params = json.loads(Path(args.params).read_text()) if args.params else None
    cfg = ReplayConfig(source_db=args.source or str(settings.db_path), start_ms=start, end_ms=end,
                       work_dir=args.work_dir, overwrite=args.overwrite,
                       symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None,
                       speed=None if str(args.speed).lower() == "max" else float(args.speed), seed_days=args.seed_days,
                       seed_metrics=args.seed_metrics, settings=overrides, params=params, commands=commands,
                       memory_limit_mb=args.memory_limit_mb, memory_slope_mb_per_day=args.memory_slope_mb_day,
                       tracemalloc=args.tracemalloc, final_flatten=not args.no_flatten, out_json=args.out,
                       messages_out=args.messages_out, trades_out=args.trades_out, log_level=args.log_level,
                       profile=bool(getattr(args, "profile", False)))
    summary = run_replay(cfg)
    print(summary["report"], flush=True)
    if args.out:
        print(f"summary: {args.out}", flush=True)
    sys.exit(0 if summary.get("passed") else 1)

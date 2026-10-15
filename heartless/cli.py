"""Command line entry point: run the bot, run offline backtests/research, or check the setup."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys

from heartless import __version__
from heartless.config import load_settings
from heartless.util.logging_setup import setup_logging
from heartless.util.timeutil import MS_DAY, now_ms

log = logging.getLogger("heartless")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heartless", description="Heartless - emotionless Binance futures scalping bot")
    parser.add_argument("--version", action="version", version=f"heartless {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("run", help="run the bot (default)")
    bt = sub.add_parser("backtest", help="backtest the champion (or one alpha) on stored/downloaded candles; with "
                                        "--start/--end: one shared account over a stored-history window (offline)")
    bt.add_argument("--days", type=int, default=14, help="the last N days (ignored with --start)")
    bt.add_argument("--start", default=None,
                    help="ISO date: joint-portfolio mode, all symbols in ONE account with the production risk settings "
                         "over [start, end] of stored history (seal-checked; a sealed period only with --unseal)")
    bt.add_argument("--end", default=None, help="ISO date; a date-only end is exclusive (default: last stored candle)")
    bt.add_argument("--unseal", action="append", default=[], metavar="PERIOD",
                    help="with --start: deliberately evaluate a sealed period (e.g. 2025); loud warning + seal ledger "
                         "entry, like lab")
    bt.add_argument("--alpha", default=None)
    bt.add_argument("--symbols", default=None,
                    help="comma separated, default: current universe (with --start: all stored symbols)")
    bt.add_argument("--params", default=None, help="StrategyParams JSON file (default: stored champion)")
    bt.add_argument("--download", action="store_true", help="download missing candles from Binance first")
    bt.add_argument("--json", action="store_true")
    bt.add_argument("--out", default=None, help="with --start: write the result JSON here (e.g. docs/results/<name>.json)")
    rs = sub.add_parser("research", help="run one walk-forward research cycle offline and print the summary")
    rs.add_argument("--days", type=int, default=None)
    rs.add_argument("--candidates", type=int, default=None)
    rs.add_argument("--apply", action="store_true", help="promote the best candidates straight to champion (offline use)")
    fe = sub.add_parser("fetch", help="download history from the Binance public data archive (no API weight)")
    fe.add_argument("--symbols", default=None, help="comma separated (default: ALWAYS_INCLUDE + stored symbols)")
    fe.add_argument("--days", type=int, default=90, help="the last N days (default mode, ignored with --start)")
    fe.add_argument("--start", default=None, help="ISO date: long-range mode, fetched in ascending calendar-year chunks")
    fe.add_argument("--end", default=None, help="ISO date, exclusive (default: now); --end 2026-10-09 ends 10-08 23:59")
    fe.add_argument("--no-metrics", action="store_true", help="skip open-interest / long-short metrics")
    fe.add_argument("--retry-holes", action="store_true",
                    help="re-request candle ranges the archive previously did not have")
    cv = sub.add_parser("coverage", help="report stored history per symbol (rows, gaps, funding, metrics) to coverage.json")
    cv.add_argument("--symbols", default=None, help="comma separated (default: all stored symbols)")
    cv.add_argument("--start", default=None, help="ISO date: only count history from here")
    cv.add_argument("--end", default=None, help="ISO date, exclusive")
    cv.add_argument("--out", default=None, help="JSON path (default: <data_dir>/coverage.json)")
    lb = sub.add_parser("lab", help="research lab: parallel per-symbol backtests on stored history with train/valid/holdout splits")
    lb.add_argument("--alpha", default=None, help="evaluate a single alpha in isolation")
    lb.add_argument("--split", default="train", choices=["train", "valid", "holdout", "all"])
    lb.add_argument("--start", default=None, help="ISO date, overrides --split")
    lb.add_argument("--end", default=None, help="ISO date; a date-only end is exclusive")
    lb.add_argument("--years", default=None, help="contiguous years, e.g. 2022,2023,2024 (window = their union)")
    lb.add_argument("--by-side", action="store_true", help="add long/short breakdowns (overall and per year)")
    lb.add_argument("--chunk-days", type=int, default=None,
                    help="split the window into N-day jobs per symbol (default: calendar years; 0 = no split)")
    lb.add_argument("--unseal", action="append", default=[], metavar="PERIOD",
                    help="deliberately evaluate a sealed period (e.g. 2025); loud warning + seal ledger entry")
    lb.add_argument("--out", default=None, help="write the full result JSON here (e.g. docs/results/<name>.json)")
    lb.add_argument("--symbols", default=None)
    lb.add_argument("--params", default=None, help="StrategyParams JSON file (default: stored champion)")
    lb.add_argument("--set", action="append", default=[], help="override, e.g. trend_pullback.adx_min=24")
    lb.add_argument("--workers", type=int, default=None)
    lb.add_argument("--stress", action="store_true", help="1.5x fees and 2x slippage")
    lb.add_argument("--meta", action="store_true", help="enable the walk-forward meta-labeling filter")
    lb.add_argument("--json", action="store_true")
    lb.add_argument("--trades-out", default=None)
    # P2a research options
    lb.add_argument("--universe", default=None, type=_universe_name,
                    help="symbol set instead of --symbols: core (12), extended (core + 8 research symbols), all stored, "
                         "or topN (F0: every symbol top-N by liquidity in a month of the window; docs/UNIVERSE.md)")
    lb.add_argument("--tag", default=None, metavar="FAMILY",
                    help="append this run to the P2 trial ledger docs/results/p2_trials.jsonl under FAMILY")
    dc = sub.add_parser("discover", help="mine rule-based alphas on stored history (TRAIN) and validate them (VALID)")
    dc.add_argument("--tf", default="15m", help="decision grid(s), comma separated: 15m, 1h, 1d (00:00 UTC daily)")
    dc.add_argument("--grid", default=None, help="same as --tf (e.g. 1h,1d); the t bar counts every grid's rules")
    dc.add_argument("--templates", default="classic", choices=["classic", "time", "all"],
                    help="exit templates: stop/target/timeout (classic), 8h/24h/72h time exits behind a 4 ATR stop "
                         "(time), or both")
    dc.add_argument("--context", action="store_true",
                    help="add the market-context features (built on demand, see `heartless context build`)")
    dc.add_argument("--tag", default=None, metavar="FAMILY",
                    help="append the search (rules evaluated) to the P2 trial ledger; the t bar then also counts the "
                         "family's earlier trials")
    dc.add_argument("--universe", default=None, type=_universe_name,
                    help="symbol set instead of --symbols (core, extended, all, or topN resolved over --train only: "
                         "the VALID period never chooses the symbols)")
    dc.add_argument("--symbols", default=None)
    dc.add_argument("--beam", type=int, default=30)
    dc.add_argument("--depth", type=int, default=3)
    dc.add_argument("--min-trades", type=int, default=150)
    dc.add_argument("--workers", type=int, default=2)
    dc.add_argument("--out", default=None, help="write the full result JSON here")
    dc.add_argument("--train", default=None, metavar="START:END", help="TRAIN window as ISO dates (default: stored split)")
    dc.add_argument("--valid", default=None, metavar="START:END", help="VALID window as ISO dates (default: stored split)")
    dc.add_argument("--unseal", action="append", default=[], metavar="PERIOD",
                    help="deliberately mine/validate on a sealed period; loud warning + seal ledger entry")
    cx = sub.add_parser("context", help="market context: hourly cross-sectional features (build / status)")
    cx.add_argument("action", choices=["build", "status"])
    cx.add_argument("--start", default=None, help="ISO date (default: first stored candle)")
    cx.add_argument("--end", default=None, help="ISO date, exclusive (default: last complete stored hour)")
    cx.add_argument("--rebuild", action="store_true",
                    help="recompute everything, also a store built from another source / universe (default: resume, "
                         "skip built years, refuse a foreign store)")
    un = sub.add_parser("universe", help="F0: exchange-wide symbol inventory, 1h / spot datasets, research universe")
    un.add_argument("action", choices=["inventory", "build", "show", "resolve", "status"])
    un.add_argument("--start", default=None, help="ISO date (build default 2022-01-01; resolve: window start)")
    un.add_argument("--end", default=None, help="ISO date, exclusive")
    un.add_argument("--spot", action="store_true", help="build: spot 1h klines of the perpetuals with a spot market")
    un.add_argument("--symbols", default=None, help="build: only these (comma separated)")
    un.add_argument("--rebuild", action="store_true", help="build: download stored months again")
    un.add_argument("--repair", action="store_true",
                    help="build: download again the stored months with whole days missing (truncated monthly archive "
                         "files, filled from the daily files) and futures months stored as not served")
    un.add_argument("--month", default=None, help="show: the ranked month, YYYY-MM")
    un.add_argument("--top", type=int, default=40)
    un.add_argument("--out", default=None, help="write the report / resolution JSON here")
    tr = sub.add_parser("trials", help="P2 trial ledger: evaluated configurations per family and the t_day bar")
    tr.add_argument("--family", default=None)
    tr.add_argument("--ledger", default=None, help="ledger path (default docs/results/p2_trials.jsonl)")
    from heartless.replay.cli import add_replay_parser

    add_replay_parser(sub)
    sub.add_parser("doctor", help="check configuration and connectivity")
    sub.add_parser("params", help="print the champion parameters")
    from heartless.learning.forward import add_forward_parser, main_cli as forward_main  # F4, ROADMAP 7.9

    add_forward_parser(sub)
    args = parser.parse_args(argv)
    args.argv = ["heartless", *argv] if argv is not None else list(sys.argv)
    cmd = args.cmd or "run"
    settings = load_settings()
    setup_logging(settings.log_level, settings.data_dir)
    if cmd == "forward":
        forward_main(settings, args)
    elif cmd == "run":
        _run(settings)
    elif cmd == "backtest":
        asyncio.run(_backtest(settings, args))
    elif cmd == "research":
        _research(settings, args)
    elif cmd == "fetch":
        asyncio.run(_fetch(settings, args))
    elif cmd == "coverage":
        _coverage(settings, args)
    elif cmd == "lab":
        from heartless.learning.lab import main_cli

        top = _top_universe(settings, args, "lab")
        main_cli(settings, args)
        _note_top_universe(args, top)
    elif cmd == "discover":
        top = _top_universe(settings, args, "discover")
        _discover(settings, args)
        _note_top_universe(args, top)
    elif cmd == "universe":
        from heartless.data.symbols_inventory import main_cli as universe_main

        universe_main(settings, args)
    elif cmd == "context":
        _context(settings, args)
    elif cmd == "trials":
        from heartless.learning import trials

        print(trials.report(trials.load(args.ledger), args.family))
    elif cmd == "replay":
        from heartless.replay.cli import main_cli as replay_main

        replay_main(settings, args)
    elif cmd == "doctor":
        asyncio.run(_doctor(settings))
    elif cmd == "params":
        from heartless.core.store import Store
        from heartless.learning.research import ResearchManager

        class _Stub:
            s = settings
            store = Store(settings.db_path)

        print(ResearchManager(_Stub()).load_champion().json())


def _universe_name(text: str) -> str:
    from heartless.data.universe import parse_top

    name = text.strip().lower()
    if name not in ("core", "extended", "all") and parse_top(name) is None:
        raise argparse.ArgumentTypeError(f"{text!r}: core, extended, all or topN (e.g. top40)")
    return name


def _top_universe(settings, args, cmd: str) -> dict | None:
    """lab / discover --universe topN (F0): resolve the research universe of the run's window from the 1h universe
    database and register the symbols that have stored 1m candles in it under that name in lab.UNIVERSES for this run.
    The others are reported as skipped (never silently dropped) and recorded in the --out JSON (_note_top_universe).

    discover resolves over --train only: the run mines TRAIN and validates on VALID with one symbol list, and members
    picked by their VALID-period liquidity would let the VALID period choose the symbols (often the ones that surged
    there). A universe frozen at the end of TRAIN is point-in-time for VALID (its later delistings included)."""
    from heartless.data import universe as U
    from heartless.learning import lab

    n = U.parse_top(getattr(args, "universe", None))
    if n is None:
        return None
    try:
        if cmd == "lab":
            if args.years:
                start, end, _ = lab.year_window(args.years)
            elif args.start and args.end:
                start, end = lab.parse_when(args.start), lab.parse_when(args.end, end=True)
            else:
                raise ValueError(f"--universe {args.universe} needs the window: --years or --start and --end")
        else:
            if not args.train:
                raise ValueError(f"--universe {args.universe} needs --train (the universe is resolved over it)")
            start, end = _window_arg(args.train)
        res = U.resolve_for_run(settings.universe_db_path, settings.db_path, start, end, n)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    lab.UNIVERSES[args.universe] = res["symbols"]
    print(f"universe {args.universe}: {len(res['research'])} symbols were top {n} in a month of the "
          f"{'window' if cmd == 'lab' else 'TRAIN window'}; {len(res['symbols'])} have stored 1m candles in it; "
          f"skipped {len(res['skipped'])} without 1m candles"
          + (f": {', '.join(res['skipped'])}" if res["skipped"] else ""), file=sys.stderr, flush=True)
    if res["symbols"]:
        print(f"note: every member is evaluated over the whole window, also in the months it was not top {n} (members "
              f"cover {res['member_share']:.0%} of the symbol-hours): results include that look-ahead until the "
              f"{cmd} filters by membership; universe_resolution.windows in --out has the spans "
              f"(heartless.data.universe.member_trades)", file=sys.stderr, flush=True)
    if not res["symbols"]:
        print("error: none of them has stored 1m candles in this database", file=sys.stderr)
        sys.exit(2)
    return res


def _note_top_universe(args, res: dict | None) -> None:
    out = getattr(args, "out", None)
    if res is None or not out or not os.path.exists(out):
        return
    with open(out) as f:
        doc = json.load(f)
    doc["universe_resolution"] = {k: res[k] for k in ("rule", "research", "symbols", "skipped", "unranked", "members",
                                                      "member_share")}
    # membership spans as [first ms, last ms] per evaluated symbol: the trades entered inside them are the ones free of
    # the union's look-ahead (heartless.data.universe.member_trades)
    doc["universe_resolution"]["windows"] = {s: [list(w) for w in ws] for s, ws in res["windows"].items()}
    with open(out, "w") as f:
        json.dump(doc, f, default=str, indent=1)


def _run(settings) -> None:
    from heartless.app import Heartless

    async def runner() -> BaseException | None:
        app = Heartless(settings)
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:  # pragma: no cover - windows
                pass
        main_task = asyncio.create_task(app.run())
        stopper = asyncio.create_task(stop.wait())
        done, _ = await asyncio.wait({main_task, stopper}, return_when=asyncio.FIRST_COMPLETED)
        err = main_task.exception() if (main_task in done and not main_task.cancelled()) else None
        if err is not None:
            log.error("fatal: %s", err, exc_info=err)
        try:
            await app.stop()
        except Exception as e:  # noqa: BLE001
            log.error("stop failed: %s", e)
        main_task.cancel()
        try:
            await main_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        return err

    if asyncio.run(runner()) is not None:
        sys.exit(1)


async def _load_symbols_and_universe(settings, store, download: bool, symbols_arg: str | None):
    from heartless.data.universe import select_universe
    from heartless.exchange.binance_rest import BinanceRest
    from heartless.exchange.symbols import parse_exchange_info

    rest = BinanceRest(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
    symbols = parse_exchange_info(await rest.exchange_info())
    if symbols_arg:
        universe = [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]
    else:
        stored = store.candle_symbols()
        if stored and not download:
            universe = stored
        else:
            universe = select_universe(symbols, await rest.ticker_24h(), settings.universe_size,
                                       settings.universe_min_quote_volume, settings.always_include_list)
    return rest, symbols, universe


async def _ensure_candles(rest, store, universe: list[str], days: int) -> None:
    from heartless.data.archive import MAX_FUNDING_INTERVAL_MS, settlement_interval
    from heartless.util.timeutil import MS_HOUR, MS_MINUTE

    now = now_ms()
    since = now - days * MS_DAY
    for sym in universe:
        lo, hi, n = store.candle_range(sym)
        start = since if (hi is None or hi < since) else hi + MS_MINUTE
        if lo is not None and lo > since + MS_HOUR:
            start = since  # extend backwards
        if start < now - MS_MINUTE:
            print(f"downloading {sym} 1m candles ...", file=sys.stderr)
            rows = await rest.klines_range(sym, start, now, "1m")
            rows = [c for c in rows if c.close_time <= now]
            if rows:
                store.save_candles(sym, rows)
        f_lo, f_hi = store.funding_range(sym)
        # the next settlement is due one of THIS symbol's intervals after the last stored one (8 h for most, 4 h or
        # shorter for some), derived from its stored settlement times
        f_iv = settlement_interval([t for t, _ in store.load_funding(sym, f_hi - MAX_FUNDING_INTERVAL_MS, f_hi)]) \
            if f_hi is not None else MAX_FUNDING_INTERVAL_MS
        if f_hi is None or f_hi < now - f_iv:
            fr = await rest.funding_rate_history(sym, start=since, end=now)
            store.save_funding(sym, [(int(r["fundingTime"]), float(r["fundingRate"]), float(r.get("markPrice", 0) or 0)) for r in fr])


async def _backtest(settings, args) -> None:
    from heartless.core.store import Store
    from heartless.execution.stats import by_alpha
    from heartless.learning import seal
    from heartless.learning.backtester import load_backtester
    from heartless.learning.research import ResearchManager

    if getattr(args, "start", None) or getattr(args, "end", None):
        await _backtest_window(settings, args)
        return
    if getattr(args, "out", None):
        print("error: --out needs --start (joint-portfolio window mode)", file=sys.stderr)
        sys.exit(2)
    if getattr(args, "unseal", None):  # the rolling --days lookback must never be a way into a sealed period
        print("error: --unseal needs --start (joint-portfolio window mode)", file=sys.stderr)
        sys.exit(2)
    try:
        seal.check_window(now_ms() - args.days * MS_DAY, now_ms())
    except seal.SealedError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    store = Store(settings.db_path)
    rest, symbols, universe = await _load_symbols_and_universe(settings, store, args.download, args.symbols)
    if args.download:
        await _ensure_candles(rest, store, universe, args.days)
    await rest.close()

    class _Stub:
        s = settings

    _Stub.store = store
    if getattr(args, "params", None):
        params = _params_file(args.params)
    else:
        params = ResearchManager(_Stub()).load_champion()
    since = now_ms() - args.days * MS_DAY
    bt = load_backtester(settings, store, symbols, since, symbol_list=universe)
    if not bt.candles:
        print("no candles stored. Run with --download or start the bot once.", file=sys.stderr)
        sys.exit(1)
    res = await bt.arun(params, only_alpha=args.alpha, initial_balance=settings.paper_initial_balance)
    if args.json:
        print(json.dumps({"stats": res.stats, "by_alpha": by_alpha(res.trades), "skipped": res.skipped}, indent=2, default=str))
        return
    s = res.stats
    print(f"\nBacktest {args.days}d on {len(bt.candles)} symbols, params {params.version}" + (f", alpha={args.alpha}" if args.alpha else ""))
    print(f"  trades {s['n']}  win {s['win_rate']:.1f}%  PF {min(s['profit_factor'], 99):.2f}  avgR {s['avg_r']:+.3f}  t {s['t_stat']:+.2f}")
    print(f"  net {s['net']:+.2f} USDT ({s['return_pct']:+.2f}%)  fees {s['fees']:.2f}  maxDD {s['max_dd']:.2f} ({s['max_dd_pct']:.2f}%)")
    print(f"  avg hold {s['avg_hold_min']:.0f} min  best {s['best']:+.2f}  worst {s['worst']:+.2f}")
    print("  by alpha:")
    for a, st in sorted(by_alpha(res.trades).items(), key=lambda kv: -kv[1]["net"]):
        print(f"    {a:18s} n={st['n']:4d} win {st['win_rate']:5.1f}% PF {min(st['profit_factor'], 99):5.2f} avgR {st['avg_r']:+.3f} net {st['net']:+9.2f}")
    if res.skipped:
        print("  skipped entries:", res.skipped)


def _params_file(path: str):
    from heartless.strategy.params import StrategyParams

    with open(path) as f:
        return StrategyParams.from_dict(json.load(f))


# risk settings reported with every joint-portfolio result (the values actually used)
JOINT_RISK_KEYS = ("risk_per_trade_pct", "max_risk_per_trade_pct", "max_positions", "max_same_direction",
                   "max_gross_leverage", "daily_loss_limit_pct", "weekly_loss_limit_pct", "max_drawdown_halt_pct",
                   "paper_initial_balance", "taker_fee", "maker_fee", "backtest_slippage_bps")


def _window_symbol_infos(store, symbols: list[str], lo: int, hi: int) -> dict:
    """Exchange filters inferred offline from the newest stored closes in [lo, hi], like the lab (period-true).

    `Store.load_candles(limit=...)` already returns the newest rows in time order; `infer_symbol_info` takes the
    quantity step from the median of the LAST closes, so the array must stay chronological (reversed, the step came
    from the oldest part of the tail, about a month before `hi`)."""
    import numpy as np

    from heartless.learning import lab

    infos = {}
    for sym in symbols:
        tail = store.load_candles(sym, start=lo, end=hi, limit=50_000)
        if tail:
            infos[sym] = lab.infer_symbol_info(sym, np.array([c.close for c in tail], dtype=float))
    return infos


async def _backtest_window(settings, args) -> None:
    """`backtest --start/--end`: one shared account over a stored-history window (docs/RESEARCH.md section 6).

    The lab backtests every symbol in its own account with the circuit breakers off, which measures the signals; this
    runs all symbols through ONE account with the production risk settings, so symbols compete for the position limits
    and the daily/weekly loss halts and the drawdown pause act - what a single live account would have seen. It is
    offline and reproducible: the risk settings are the shipped Settings() defaults (the local .env is not applied),
    symbol filters are inferred from the data as in the lab, indicators warm up on lab.WARMUP_MS of history before
    `start`, and nothing stamped after `end` is read. The window must pass the seal check. `--unseal PERIOD` opens a
    sealed period as the lab does (loud warning, one seal-ledger line per period the window touches, nothing for one
    it does not), because a pre-registered final exam may include the joint portfolio over the sealed period
    (docs/ROADMAP.md section 6, 2026-10-11)."""
    import time
    from datetime import datetime, timezone

    from heartless.config import Settings
    from heartless.core.store import Store
    from heartless.learning import lab, seal
    from heartless.learning.backtester import load_backtester
    from heartless.strategy.params import StrategyParams
    from heartless.util.timeutil import MS_MINUTE

    def fail(msg: str) -> None:
        print(f"error: {msg}", file=sys.stderr)
        sys.exit(2)

    if args.download:
        fail("--download does not combine with --start/--end: the window mode reads stored history only")
    if not args.start:
        fail("--end needs --start")
    try:
        start = lab.parse_when(args.start)
        end = lab.parse_when(args.end, end=True) if args.end else None
        unseal = sorted(set(getattr(args, "unseal", None) or []))
        seal.overlapping(start, start, unseal)  # an unknown period name is an error, not a silent no-op
    except ValueError as e:
        fail(str(e))
    if end is not None:  # refuse a sealed window before touching the database
        try:
            seal.check_window(start, end, unseal)
        except seal.SealedError as e:
            fail(str(e))
    store = Store(settings.db_path)
    try:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols \
            else sorted(store.candle_symbols())
        if end is None:
            lasts = [hi for _, hi, _ in (store.candle_range(s) for s in symbols) if hi is not None]
            if not lasts:
                fail("no candles stored for the requested symbols; run `heartless fetch` first")
            end = min(lasts) + MS_MINUTE - 1
        if end <= start:
            fail(f"empty window {_iso(start)} .. {_iso(end)}")
        try:
            seal.check_window(start, end, unseal)
        except seal.SealedError as e:
            fail(str(e))
        touched = seal.overlapping(start, end)
        for period in unseal:
            if period in touched:
                print(f"\n{'!' * 78}\n!!! UNSEALING {period}: this joint backtest evaluates the sealed period {period}. "
                      f"It is the one-time\n!!! final exam (docs/ROADMAP.md section 1); re-tuning after seeing it is "
                      f"forbidden. Recorded in\n!!! {seal.LEDGER}\n{'!' * 78}\n", file=sys.stderr, flush=True)
                seal.record_unseal(period, list(getattr(args, "argv", None) or sys.argv))
            else:
                print(f"note: --unseal {period} has no effect: the window does not overlap it (not recorded)",
                      file=sys.stderr)
        # read-only parameter lookup (ResearchManager.load_champion would persist a default champion into the DB)
        if args.params:
            params, source = _params_file(args.params), f"file {args.params}"
        else:
            rows = store.load_params_versions(role="champion", limit=1)
            if rows:
                params, reset = StrategyParams.from_stored(rows[0]["params"])
                params.version = rows[0]["id"]
                source = f"stored champion {rows[0]['id']}" + (
                    f" (shipped defaults for {', '.join(reset)}: their design changed)" if reset else "")
            else:
                params, source = StrategyParams.default(), "shipped defaults (no stored champion)"
        warm = lab.warmup_for(params, args.alpha)  # longer when an evaluated alpha declares lookback_days
        infos = _window_symbol_infos(store, symbols, start - warm, end)
        run_settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(settings.data_dir))
        bt = load_backtester(run_settings, store, infos, start, end, symbol_list=symbols, warmup_ms=warm,
                             lookahead=False)
    finally:
        store.close()
    if not bt.candles:
        fail(f"no stored candles for {', '.join(symbols)} in {_iso(start)} .. {_iso(end)}")
    t0 = time.time()
    commit, dirty = seal.git_commit(), lab.git_dirty()  # the code state the run starts from
    res = await bt.arun(params, start=start, end=end, only_alpha=args.alpha,
                        initial_balance=run_settings.paper_initial_balance)
    st, trades = res.stats, res.trades
    loaded = sorted(bt.candles)
    risk = {k: getattr(run_settings, k) for k in JOINT_RISK_KEYS}
    # The drawdown pause latches until an owner /resume, so nothing trades after it: in a backtest every signal refused
    # as "paused" comes after it. Its time is read off the hourly equity curve; the engine checks every bar, so a
    # crossing between two hourly samples leaves it unknown (the last entry still bounds it).
    peak, pause_at, daily = run_settings.paper_initial_balance, None, {}
    for ts, eq in res.equity_curve:
        peak = max(peak, eq)
        if pause_at is None and peak > 0 and (eq - peak) / peak * 100 <= -risk["max_drawdown_halt_pct"]:
            pause_at = ts
        daily[_iso(ts)[:10]] = round(eq, 2)
    entries = sorted(int(t.get("entry_time") or 0) for t in trades)
    summary = {**lab.group_stats(trades), "return_pct": round(st["return_pct"], 3), "max_dd_pct": round(st["max_dd_pct"], 3),
               "final_equity": round(st["final_equity"], 2), "fees": round(st["fees"], 1), "funding": round(st["funding"], 1),
               "avg_hold_min": round(st["avg_hold_min"], 1),
               "boundary": sum(1 for t in trades if t.get("exit_reason") == lab.BOUNDARY_REASON)}
    doc = {"mode": "joint portfolio: one shared account, production risk settings",
           "window": {"start": _iso(start), "end": _iso(end), "start_ms": start, "end_ms": end},
           "symbols": symbols, "symbols_loaded": loaded, "symbols_missing": [s for s in symbols if s not in bt.candles],
           "alpha": args.alpha, "params_version": params.version, "params_source": source,
           "enabled": sorted(a for a, on in params.enabled.items() if on), "risk_settings": risk, "summary": summary,
           "by_alpha": {k: lab.group_stats(v) for k, v in sorted(lab._bucket(trades, lambda t: t.get("alpha") or "?").items())},
           "by_side": {k: lab.group_stats(v) for k, v in sorted(lab._bucket(trades, lambda t: t.get("side") or "?").items())},
           "by_year": {k: lab.group_stats(v) for k, v in sorted(lab._bucket(trades, lab._year).items())},
           "by_symbol": {k: lab.group_stats(v) for k, v in sorted(lab._bucket(trades, lambda t: t.get("symbol") or "?").items())},
           "by_month": {k: lab.group_stats(v) for k, v in sorted(lab._bucket(trades, lab._month).items())},
           "rejected": dict(sorted(res.skipped.items(), key=lambda kv: -kv[1])),
           "first_entry": _iso(entries[0]) if entries else None, "last_entry": _iso(entries[-1]) if entries else None,
           "drawdown_pause": {"latched": res.skipped.get("paused", 0) > 0, "refused_signals": res.skipped.get("paused", 0),
                              "hourly_crossing": _iso(pause_at) if pause_at is not None else None},
           "equity_daily": daily,
           "seconds": round(time.time() - t0, 1), "git_commit": commit, "git_dirty": dirty,
           "seal": seal.status(start, end, unseal), "argv": list(getattr(args, "argv", None) or sys.argv),
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if args.json:
        print(json.dumps(lab._clean(doc), indent=1, default=str))
    else:
        missing = f" (no data: {', '.join(doc['symbols_missing'])})" if doc["symbols_missing"] else ""
        print(f"\nJoint backtest {_iso(start)} .. {_iso(end)} UTC, {len(loaded)}/{len(symbols)} symbols{missing}, "
              f"one {risk['paper_initial_balance']:,.0f} USDT account")
        print(f"  params {params.version} ({source}); enabled: {', '.join(doc['enabled']) or '-'}"
              + (f"; only alpha {args.alpha}" if args.alpha else ""))
        print(f"  risk: {risk['risk_per_trade_pct']}% per trade, max {risk['max_positions']} positions / "
              f"{risk['max_same_direction']} same direction, loss halts daily {risk['daily_loss_limit_pct']}% / weekly "
              f"{risk['weekly_loss_limit_pct']}%, drawdown pause {risk['max_drawdown_halt_pct']}%")
        print(f"  return {summary['return_pct']:+.2f}%  maxDD {summary['max_dd_pct']:.2f}%  PF {summary['profit_factor']:.2f}  "
              f"avgR {summary['avg_r']:+.3f}  n {summary['n']}  win {summary['win_rate']:.1f}%  t_day {summary['t_day']:+.2f}  "
              f"boundary {summary['boundary']}")
        for a, g in doc["by_alpha"].items():
            print(f"  alpha {a:18s} n={g['n']:5d} avgR={g['avg_r']:+.3f} PF={g['profit_factor']:5.2f} net={g['net']:+.1f}")
        print("  rejected / skipped entries: " + (", ".join(f"{k} {v}" for k, v in doc["rejected"].items()) or "none"))
        dp = doc["drawdown_pause"]
        if dp["latched"]:
            when = f"first seen on the hourly equity curve {dp['hourly_crossing']} UTC" if dp["hourly_crossing"] else \
                "between two hourly equity samples"
            print(f"  drawdown pause (-{risk['max_drawdown_halt_pct']}% from the peak) latched, {when}; it waits for an "
                  f"owner /resume: {dp['refused_signals']} signals refused, last entry {doc['last_entry']} UTC")
    if args.out:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(lab._clean(doc), f, indent=1, default=str)
        print(f"wrote {args.out}", file=sys.stderr)


def _research(settings, args) -> None:
    from dataclasses import asdict

    from heartless.core.store import Store
    from heartless.exchange.binance_rest import BinanceRest
    from heartless.exchange.symbols import parse_exchange_info
    from heartless.learning.optimizer import run_research_cycle
    from heartless.learning.research import ResearchManager

    store = Store(settings.db_path)
    symbols = asyncio.run(_symbols_offline(settings))

    class _Stub:
        s = settings

    _Stub.store = store
    rm = ResearchManager(_Stub())
    params = rm.load_champion()
    settings_dict = {k: (str(v) if hasattr(v, "__fspath__") else v) for k, v in settings.model_dump(by_alias=True).items()
                     if k not in ("BINANCE_API_KEY", "BINANCE_API_SECRET", "TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY")}
    result = run_research_cycle(str(settings.db_path), settings_dict, params.to_dict(), {k: asdict(v) for k, v in symbols.items()},
                                args.days or settings.research_lookback_days, args.candidates or settings.research_candidates)
    if result.get("window", {}).get("sealed_clipped"):
        print(f"note: research window clipped to stay out of sealed period(s) {result['window']['sealed_clipped']}",
              file=sys.stderr)
    if "error" in result:
        print(f"research: {result['error']}", file=sys.stderr)
    print(json.dumps({a: {"base": r["base"]["test"], "best": r["best"]["test"], "best_params": r["best"]["params"],
                          "improved": not r["best"].get("is_base")} for a, r in result.get("alphas", {}).items()},
                     indent=2, default=str))
    if args.apply:
        new = params
        for a, r in result.get("alphas", {}).items():
            if not r["best"].get("is_base") and r["best"]["score"] > r["base"]["score"]:
                new = new.with_alpha(a, r["best"]["params"], note=f"offline research {a}")
        if new is not params:
            store.set_params_role(params.version, "retired")
            store.save_params_version(new.version, new.created, "champion", "offline-research", new.note, new.to_dict())
            print(f"champion updated -> {new.version}")


async def _symbols_offline(settings):
    from heartless.exchange.binance_rest import BinanceRest
    from heartless.exchange.symbols import parse_exchange_info

    rest = BinanceRest(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
    try:
        return parse_exchange_info(await rest.exchange_info())
    finally:
        await rest.close()


def _window_arg(text: str) -> tuple[int, int]:
    from heartless.learning.lab import parse_when

    a, sep, b = text.partition(":")
    if not sep or not a.strip() or not b.strip():
        raise ValueError(f"expected START:END (ISO dates), got {text!r}")
    return parse_when(a), parse_when(b, end=True)


def _discover(settings, args) -> None:
    from heartless.core.store import Store
    from heartless.learning import discovery as D
    from heartless.learning import seal
    from heartless.learning.lab import store_splits, universe_symbols

    store = Store(settings.db_path)
    unseal = sorted(set(args.unseal or []))
    tag = (getattr(args, "tag", None) or "").strip() or None
    try:
        if getattr(args, "universe", None) and args.symbols:
            raise ValueError("--symbols and --universe are mutually exclusive")
        if getattr(args, "universe", None):
            symbols = universe_symbols(args.universe, store)
        else:
            symbols = [x.strip().upper() for x in args.symbols.split(",")] if args.symbols else store.candle_symbols()
        grids = D.parse_grids(getattr(args, "grid", None) or args.tf)
        if getattr(args, "tag", None) is not None and not tag:
            raise ValueError("--tag needs a family name")
        sp = store_splits(store, symbols) if not (args.train and args.valid) else {}
        train = _window_arg(args.train) if args.train else sp["train"]
        valid = _window_arg(args.valid) if args.valid else sp["valid"]
        for w in (train, valid):
            seal.check_window(w[0], w[1], unseal)
    except (ValueError, seal.SealedError) as e:
        store.close()
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    touched = set(seal.overlapping(*train)) | set(seal.overlapping(*valid))
    for period in unseal:
        if period in touched:
            print(f"\n{'!' * 78}\n!!! UNSEALING {period} for discovery: the sealed final-exam period is being mined "
                  f"or validated on.\n!!! Recorded in {seal.LEDGER}\n{'!' * 78}\n", file=sys.stderr, flush=True)
            seal.record_unseal(period, args.argv)
        else:
            print(f"note: --unseal {period} has no effect: neither window overlaps it (not recorded)", file=sys.stderr)
    cfg = D.SearchConfig(min_trades=args.min_trades, beam=args.beam, depth=args.depth)
    templates = getattr(args, "templates", "classic") or "classic"
    context = None
    if getattr(args, "context", False):
        context = {"db": str(settings.context_db_path), "universe": settings.context_universe_list,
                   "build": bool(settings.context_auto_build), "top_n": settings.context_top_n,
                   "progress": lambda m: print(m, file=sys.stderr, flush=True)}
    n_prior, conf = 0, None
    if tag:
        from heartless.learning import trials

        conf = {"tool": "discover", "grids": grids, "templates": templates, "context": bool(context),
                "features": len(D.FEATURES_X if context else D.FEATURES), "symbols": sorted(symbols),
                "search": {"min_trades": cfg.min_trades, "beam": cfg.beam, "depth": cfg.depth},
                "train": [int(train[0]), int(train[1])]}
        # the family's other configurations: an identical rerun replaces its own earlier count (the ledger keeps the
        # largest count per configuration), so its bar must not add that count on top of this run's
        n_prior = trials.family_tests(tag, exclude=trials.config_hash(conf))
    kw = {"workers": args.workers, "progress": lambda m: print(m, file=sys.stderr, flush=True),
          "unseal": [p for p in unseal if p in touched]}
    if templates != "classic" or context or n_prior:  # the defaults keep the call the live discovery cycle makes
        kw.update(templates=templates, context=context, n_prior=n_prior)
    from heartless.data.context import ContextMismatch

    try:
        res = D.mine(str(settings.db_path), symbols, train, valid, ",".join(grids), cfg, **kw)
    except ContextMismatch as e:  # a context store of another source / universe: refused before any dataset is built
        store.close()
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    store.set("discovery.last", {"tf": res.get("tf", ",".join(grids)), "tested": res["tested"], "t_bar": res["t_bar"],
                                 "passed": res["passed"], "ts": now_ms()})
    store.close()
    trial = None
    if tag:
        best = res.get("best_train") or {}
        trial = trials.record("discover", tag, conf, train, n=best.get("n"), avg_r=best.get("avg_r"),
                              t_day=best.get("t_ex"), count=res["tested"],
                              extra={"valid": [_iso(valid[0]), _iso(valid[1])], "argv": list(args.argv)})
        res["trial"] = trial
        n_fam = trials.family_tests(tag)
        print(f"trial ledger: {tag} search {trial['config_hash']} with {res['tested']} rules recorded; family N={n_fam}, "
              f"TRAIN bar t >= {trials.threshold(n_fam):.2f}", file=sys.stderr)
    else:
        from heartless.learning.trials import period_of

        if period_of(*train) == "TRAIN":
            print("note: not recorded in the P2 trial ledger (no --tag FAMILY)", file=sys.stderr)
    rows = res.get("rows", {})
    print(f"tested {res['tested']} rules on {rows.get('train', 0)} TRAIN rows ({res.get('tf', ','.join(grids))}); "
          f"family-wise t bar {res['t_bar']:.2f}" + (f" (N={res['n_tests']})" if res.get("n_tests") else "")
          + f"; {len(res['passed'])} rule(s) passed VALID ({res['seconds']:.0f}s)")
    for r in res["results"]:
        st = r["rule"]["stats"]
        tr, va = st["train"], st["valid"]
        tp = r["rule"]["tp_r"]
        print(f"{'PASS' if r['passed'] else 'fail'} {r['rule']['tf']:3s} {r['rule']['side']:5s} sl={r['rule']['sl_atr']} "
              f"tp={'-' if tp is None else str(tp) + 'R'} hold={r['rule']['hold_min']}m {r['rule']['conds']}")
        print(f"     TRAIN n={tr['n']} days={tr.get('days', 0)} avgR={tr['avg_r']:+.3f} excess={tr.get('avg_ex', 0):+.3f} "
              f"t={tr['t']:.2f} t_excess={tr.get('t_ex', 0):.2f} | VALID n={va['n']} avgR={va['avg_r']:+.3f} "
              f"excess={va.get('avg_ex', 0):+.3f} t={va['t']:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, default=str, indent=1)


def _context(settings, args) -> None:
    """`heartless context build|status`: the market_context table in its own store (CONTEXT_DB, default
    <data_dir>/context.db; heartless/data/context.py). A store of another source or universe is refused unless
    `--rebuild`."""
    import os
    import time

    from heartless.core.store import Store
    from heartless.data import context as C
    from heartless.learning.lab import parse_when

    universe = settings.context_universe_list
    ctx_path = settings.context_db_path
    if os.path.realpath(ctx_path) == os.path.realpath(settings.db_path):
        print(f"error: CONTEXT_DB {ctx_path} is the candle database; the market context needs its own file "
              f"(default <data_dir>/{C.DB_NAME})", file=sys.stderr)
        sys.exit(2)
    if args.action == "status" and not ctx_path.exists():
        print(f"context store {ctx_path}: not built (run `heartless context build`)")
        return
    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    src = Store(settings.db_path)
    dst = Store(ctx_path)
    try:
        if args.action == "status":
            meta = dst.get("context.meta") or {}
            st = dst.market_context_stats()
            ok, status, stable, why = C.usable(src, dst, universe, settings.context_top_n)
            verdict = {"same": "usable (same source, universe and definition)",
                       "extended": f"usable up to {_iso(stable)} UTC, newer source history is built on the next run "
                                   f"({why})",
                       "none": "empty (built on the next build)"}.get(status, f"NOT usable: {why}")
            print(f"context store {ctx_path}: {st['hours']} hours, {st['rows']} rows, {_iso(st['first'])} .. "
                  f"{_iso(st['last'])} UTC; version {meta.get('version', '-')} "
                  f"({'current' if C.meta_ok(meta) else 'outdated'}); {verdict}")
            src_meta = meta.get("source") or {}
            print(f"  source {src_meta.get('path', '-')} (fingerprint {src_meta.get('fingerprint', '-')}, "
                  f"{len(src_meta.get('ranges') or {})} symbols); this run reads {os.path.realpath(settings.db_path)}")
            for a, b in meta.get("built", []):
                print(f"  built {_iso(a)} .. {_iso(b)}")
            print(f"  universe ({len(universe)}): {', '.join(universe)}; recorded: "
                  f"{','.join(meta.get('universe') or []) or '-'}; stored candles: "
                  f"{', '.join(C.stored_symbols(src, universe)) or '-'}")
            return
        start = parse_when(args.start) if args.start else None
        end = parse_when(args.end, end=True) if args.end else None
        t0 = time.time()
        stats = C.build(src, dst, start, end, universe, rebuild=args.rebuild, top_n=settings.context_top_n,
                        progress=lambda m: print(m, file=sys.stderr, flush=True))
        built = [u for u in stats["units"] if not u.get("skipped")]
        print(f"context: {stats['hours']} hours / {stats['rows']} rows built in {len(built)} year(s), "
              f"{len(stats['units']) - len(built)} already built; {len(stats['stored_symbols'])}/{len(universe)} "
              f"universe symbols stored; {time.time() - t0:.0f}s -> {ctx_path}")
    except RuntimeError as e:  # ContextMismatch included: another source / universe, refused until --rebuild
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        dst.close()
        src.close()


async def _fetch(settings, args) -> None:
    """Archive download. --days N fetches the recent tail; --start/--end fetches a long range symbol by symbol in
    ascending calendar-year chunks (at most about a year of 1m klines in memory), skipping what is stored, so an
    interrupted run simply resumes. Interior holes are re-requested once and reported."""
    import time

    from heartless.core.store import Store
    from heartless.data.archive import BinanceArchive, sync_symbol, year_spans
    from heartless.learning.lab import parse_when

    now = now_ms()
    try:
        if args.start:
            start = parse_when(args.start)
            end = min(parse_when(args.end, end=True), now) if args.end else now
        elif args.end:
            raise ValueError("--end needs --start")
        else:
            start, end = now - args.days * MS_DAY, now
        if end <= start:
            raise ValueError("empty fetch window")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    store = Store(settings.db_path)
    if args.symbols:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = list(dict.fromkeys(settings.always_include_list + store.candle_symbols()))
    chunks = year_spans(start, end)
    print(f"fetching {len(symbols)} symbol(s) {_iso(start)} .. {_iso(end)} UTC in {len(chunks)} chunk(s)", flush=True)
    async with BinanceArchive() as archive:
        for sym in symbols:
            for a, b in chunks:
                tag = f"{sym} {_iso(a)[:4] if len(chunks) > 1 else ''}".rstrip()
                t0 = time.time()
                try:
                    res = await sync_symbol(store, archive, sym, a, b, now, metrics=not args.no_metrics,
                                            retry_holes=args.retry_holes)
                except Exception as e:  # noqa: BLE001
                    print(f"{tag}: failed ({e!r}); rerun to resume", file=sys.stderr, flush=True)
                    continue
                gaps = res["gaps"]
                extra = ""
                if gaps:
                    big = max(gaps, key=lambda g: g[2])
                    extra = (f", {len(gaps)} gap(s) >= 5m ({sum(g[2] for g in gaps)} min, largest {big[2]}m at "
                             f"{_iso(big[0])})")
                if res["known_holes"]:
                    extra += f", {res['known_holes']} known archive hole(s) skipped"
                print(f"{tag}: +{res['candles']} candles, +{res['funding']} funding, +{res['metrics']} metrics{extra} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            lo, hi, n = store.candle_range(sym)
            span = f"{_iso(lo)} .. {_iso(hi)}" if lo else "-"
            print(f"{sym}: total {n} candles ({span})", flush=True)
        print(f"downloaded {archive.downloaded_bytes / 1e6:.1f} MB from {archive._good or '-'}")
    store.close()


def _iso(ms: int | None) -> str:
    from heartless.util.timeutil import fmt_ts

    return fmt_ts(ms, "UTC", "%Y-%m-%d %H:%M") if ms is not None else "-"


def _coverage(settings, args) -> None:
    from pathlib import Path

    from heartless.core.store import Store
    from heartless.data.archive import coverage, coverage_table
    from heartless.learning.lab import parse_when

    store = Store(settings.db_path)
    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()] if args.symbols else sorted(store.candle_symbols())
    start = parse_when(args.start) if args.start else None
    end = parse_when(args.end, end=True) if args.end else None
    cov = coverage(store, symbols, start, end)
    store.close()
    cov["db"] = str(settings.db_path)
    out = Path(args.out) if args.out else settings.data_dir / "coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cov, indent=1))
    print(coverage_table(cov))
    tot = sum(e["rows"] for e in cov["symbols"].values())
    exp = sum(e["expected"] for e in cov["symbols"].values())
    print(f"{len(symbols)} symbol(s), {tot:,} candles, missing {((exp - tot) / exp * 100) if exp else 0:.4f}% of the "
          f"expected minutes between each symbol's first and last candle; wrote {out}")


async def _doctor(settings) -> None:
    from heartless.exchange.binance_rest import BinanceError, BinanceRest

    ok = True
    print(f"Heartless {__version__} doctor")
    print(f"  mode: {settings.mode}  testnet: {settings.binance_testnet}  data: {settings.data_dir}")
    print(f"  binance keys: {'present' if settings.live_capable else 'MISSING (paper only)'}")
    print(f"  telegram token: {'present' if settings.telegram_bot_token else 'missing (web only)'}")
    print(f"  anthropic key: {'present (AI advisor on)' if settings.anthropic_api_key else 'absent (advisor off)'}")
    rest = BinanceRest(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
    try:
        await rest.sync_time()
        print(f"  binance public API: OK (time offset {rest.time_offset}ms)")
        info = await rest.exchange_info()
        print(f"  exchange info: {len(info.get('symbols', []))} symbols")
        if settings.live_capable:
            acc = await rest.account()
            print(f"  account: wallet {float(acc.get('totalWalletBalance', 0)):.2f} USDT, canTrade={acc.get('canTrade')}")
            dual = await rest.get_position_mode()
            print(f"  position mode: {'HEDGE (will switch to one-way if flat)' if dual else 'one-way OK'}")
            algos = await rest.open_algo_orders()
            print(f"  open algo orders: {len(algos)}")
    except BinanceError as e:
        ok = False
        print(f"  binance error: {e}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  connectivity error: {e}")
    finally:
        await rest.close()
    if settings.telegram_bot_token:
        import httpx

        try:
            r = await httpx.AsyncClient(timeout=15).get(f"https://api.telegram.org/bot{settings.telegram_bot_token}/getMe")
            d = r.json()
            print(f"  telegram: {'OK @' + d['result']['username'] if d.get('ok') else 'FAILED ' + str(d)}")
        except Exception as e:  # noqa: BLE001
            print(f"  telegram error: {e}")
    print("  result:", "OK" if ok else "PROBLEMS FOUND")

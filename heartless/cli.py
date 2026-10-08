"""Command line entry point: run the bot, run offline backtests/research, or check the setup."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
    bt = sub.add_parser("backtest", help="backtest the champion (or one alpha) on stored/downloaded candles")
    bt.add_argument("--days", type=int, default=14)
    bt.add_argument("--alpha", default=None)
    bt.add_argument("--symbols", default=None, help="comma separated, default: current universe")
    bt.add_argument("--download", action="store_true", help="download missing candles from Binance first")
    bt.add_argument("--json", action="store_true")
    rs = sub.add_parser("research", help="run one walk-forward research cycle offline and print the summary")
    rs.add_argument("--days", type=int, default=None)
    rs.add_argument("--candidates", type=int, default=None)
    rs.add_argument("--apply", action="store_true", help="promote the best candidates straight to champion (offline use)")
    fe = sub.add_parser("fetch", help="download history from the Binance public data archive (no API weight)")
    fe.add_argument("--symbols", default=None, help="comma separated (default: ALWAYS_INCLUDE + stored symbols)")
    fe.add_argument("--days", type=int, default=90)
    fe.add_argument("--no-metrics", action="store_true", help="skip open-interest / long-short metrics")
    lb = sub.add_parser("lab", help="research lab: parallel per-symbol backtests on stored history with train/valid/holdout splits")
    lb.add_argument("--alpha", default=None, help="evaluate a single alpha in isolation")
    lb.add_argument("--split", default="train", choices=["train", "valid", "holdout", "all"])
    lb.add_argument("--start", default=None, help="ISO date, overrides --split")
    lb.add_argument("--end", default=None)
    lb.add_argument("--symbols", default=None)
    lb.add_argument("--params", default=None, help="StrategyParams JSON file (default: stored champion)")
    lb.add_argument("--set", action="append", default=[], help="override, e.g. trend_pullback.adx_min=24")
    lb.add_argument("--workers", type=int, default=None)
    lb.add_argument("--json", action="store_true")
    lb.add_argument("--trades-out", default=None)
    sub.add_parser("doctor", help="check configuration and connectivity")
    sub.add_parser("params", help="print the champion parameters")
    args = parser.parse_args(argv)
    cmd = args.cmd or "run"
    settings = load_settings()
    setup_logging(settings.log_level, settings.data_dir)
    if cmd == "run":
        _run(settings)
    elif cmd == "backtest":
        asyncio.run(_backtest(settings, args))
    elif cmd == "research":
        _research(settings, args)
    elif cmd == "fetch":
        asyncio.run(_fetch(settings, args))
    elif cmd == "lab":
        from heartless.learning.lab import main_cli

        main_cli(settings, args)
    elif cmd == "doctor":
        asyncio.run(_doctor(settings))
    elif cmd == "params":
        from heartless.core.store import Store
        from heartless.learning.research import ResearchManager

        class _Stub:
            s = settings
            store = Store(settings.db_path)

        print(ResearchManager(_Stub()).load_champion().json())


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
        if f_hi is None or f_hi < now - 8 * MS_HOUR:
            fr = await rest.funding_rate_history(sym, start=since, end=now)
            store.save_funding(sym, [(int(r["fundingTime"]), float(r["fundingRate"]), float(r.get("markPrice", 0) or 0)) for r in fr])


async def _backtest(settings, args) -> None:
    from heartless.core.store import Store
    from heartless.execution.stats import by_alpha
    from heartless.learning.backtester import load_backtester
    from heartless.learning.research import ResearchManager

    store = Store(settings.db_path)
    rest, symbols, universe = await _load_symbols_and_universe(settings, store, args.download, args.symbols)
    if args.download:
        await _ensure_candles(rest, store, universe, args.days)
    await rest.close()

    class _Stub:
        s = settings

    _Stub.store = store
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


async def _fetch(settings, args) -> None:
    from heartless.core.store import Store
    from heartless.data.archive import BinanceArchive, sync_symbol

    store = Store(settings.db_path)
    if args.symbols:
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    else:
        symbols = list(dict.fromkeys(settings.always_include_list + store.candle_symbols()))
    now = now_ms()
    start = now - args.days * MS_DAY
    async with BinanceArchive() as archive:
        for sym in symbols:
            try:
                res = await sync_symbol(store, archive, sym, start, now, now, metrics=not args.no_metrics)
            except Exception as e:  # noqa: BLE001
                print(f"{sym}: failed ({e})", file=sys.stderr)
                continue
            lo, hi, n = store.candle_range(sym)
            span = f"{(hi - lo) / MS_DAY:.1f}d" if lo else "-"
            print(f"{sym}: +{res['candles']} candles (total {n}, {span}), +{res['funding']} funding, +{res['metrics']} metrics")
        print(f"downloaded {archive.downloaded_bytes / 1e6:.1f} MB from {archive._good or '-'}")
    store.close()


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

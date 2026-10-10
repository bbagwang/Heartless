"""Replay runner: wires the real Heartless orchestrator to the replay market, the Telegram capture and the invariant
checker on a virtual-time loop, runs it over [start, end] and returns a JSON-serialisable summary."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, fmt_ts, now_ms

log = logging.getLogger(__name__)

# Settings every replay forces: a paper-only, closed-world bot (no keys, no web, no archive, no advisor) whose Telegram
# goes to the capture stub and whose reporting day is the UTC day.
FORCED_SETTINGS: dict[str, Any] = {
    "HEARTLESS_MODE": "paper", "WEB_ENABLED": False, "ARCHIVE_BACKFILL": False,
    "BINANCE_API_KEY": "", "BINANCE_API_SECRET": "", "ANTHROPIC_API_KEY": "",
    "TELEGRAM_BOT_TOKEN": "replay-capture", "TELEGRAM_OWNER_ID": 1,
}
# Defaults a replay changes unless overridden (`--set KEY=VALUE`). The learning cadence is compressed: a research
# cycle freezes virtual time while its worker runs, so the production 4-hourly walk-forward with 10 candidates x 3
# folds would dominate the wall time of a months-long replay. The cycles still run for real, on smaller windows.
REPLAY_DEFAULTS: dict[str, Any] = {
    "TIMEZONE": "UTC", "RESEARCH_INTERVAL_MINUTES": 7 * 24 * 60, "RESEARCH_CANDIDATES": 3, "RESEARCH_FOLDS": 2,
    "RESEARCH_LOOKBACK_DAYS": 21, "DISCOVERY_INTERVAL_HOURS": 14 * 24,
}
# owner commands exercised by default (offsets from the start; only those inside the window are sent)
DEFAULT_SCRIPT = ("+1h=/status", "+6h=/positions", "+1d=/pnl", "+1d1h=/pnl week", "+1d12h=/report", "+2d=/pause",
                  "+2d12h=/resume", "+3d=/close *", "+3d1h=/trades", "+4d=/challengers", "+5d=/research",
                  "+6d=/pnl all", "+7d=/health")


@dataclass
class ReplayConfig:
    source_db: str
    start_ms: int
    end_ms: int  # the last emitted bar closes at or before this instant
    work_dir: str | None = None  # the replayed bot's data dir (fresh); default: a new temp dir
    symbols: list[str] | None = None
    speed: float | None = None  # virtual seconds per real second; None = as fast as possible
    seed_days: int = 60  # history copied into the bot's store before start (research/discovery need it)
    seed_metrics: bool = False  # the production bot never stores metrics; True gives research the archive metrics
    settings: dict[str, Any] = field(default_factory=dict)
    params: dict | None = None  # StrategyParams dict installed as the starting champion (default: shipped defaults)
    commands: list | None = None  # ScriptedCommand list; None = DEFAULT_SCRIPT
    check_every_ms: int = MS_MINUTE
    memory_every_ms: int = MS_HOUR
    memory_limit_mb: float = 300.0
    memory_slope_mb_per_day: float = 5.0
    memory_warmup_ms: int = 2 * MS_DAY
    tracemalloc: bool = False
    final_flatten: bool = True
    overwrite: bool = False
    rest_latency_ms: int = 0
    idle_grace_s: float = 300.0
    log_level: str = "WARNING"
    out_json: str | None = None
    out_report: str | None = None
    messages_out: str | None = None
    trades_out: str | None = None
    # the bot starts this long after start_ms, so its 5 s scheduler grid is not aligned to minutes and midnights (a real
    # bot is not either; an aligned grid would hide races between the scheduler and the market data)
    app_start_offset_ms: int = 2_500
    # Thompson-sampling RNG seed for the shared and every challenger bandit (production leaves them unseeded); with it a
    # replay is reproducible run to run. None keeps them unseeded.
    seed: int | None = 1
    # fault injection for tests: (virtual epoch ms, fn(app)) called once at that instant (fn may be async)
    inject: list = field(default_factory=list)
    # extra trades stored before start (dicts in TradeRecord shape), e.g. a challenger's record to evaluate at once
    seed_trades: list = field(default_factory=list)
    # challengers stored before start: (slot name, StrategyParams dict, started ms, source)
    seed_challengers: list = field(default_factory=list)


class _LogCapture(logging.Handler):
    """Error-level records are invariant violations; warnings are counted for the report."""

    def __init__(self, checker) -> None:
        super().__init__(level=logging.WARNING)
        self.checker = checker
        self.warnings: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("heartless.replay.invariants"):
            return
        if record.levelno >= logging.ERROR:
            self.checker.error_log(record)
        else:
            key = f"{record.name}: {record.msg}" if isinstance(record.msg, str) else record.name
            self.warnings[key[:160]] = self.warnings.get(key[:160], 0) + 1


def _prepare_store(cfg: ReplayConfig, work: Path, symbols: list[str]) -> None:
    """Fresh bot store holding the `seed_days` of history before start, as a bot that has been collecting data would
    have (candles, funding; metrics only on request: the production bot keeps polled metrics in memory only)."""
    from heartless.core.store import Store

    db = work / "heartless.db"
    Store(db).close()  # schema
    lo = cfg.start_ms - cfg.seed_days * MS_DAY
    con = sqlite3.connect(str(db), uri=True)
    try:
        con.execute("ATTACH DATABASE ? AS src", (f"file:{Path(cfg.source_db).resolve()}?mode=ro",))
        marks = ",".join("?" * len(symbols))
        cols = "symbol, open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, close_time"
        con.execute(f"INSERT INTO candles({cols}) SELECT {cols} FROM src.candles WHERE symbol IN ({marks}) AND "
                    f"open_time>=? AND close_time<?", (*symbols, lo, cfg.start_ms))
        con.execute(f"INSERT INTO funding(symbol, funding_time, rate, mark) SELECT symbol, funding_time, rate, mark FROM "
                    f"src.funding WHERE symbol IN ({marks}) AND funding_time>=? AND funding_time<?",
                    (*symbols, lo, cfg.start_ms))
        if cfg.seed_metrics:
            mcols = "symbol, ts, oi, oi_value, top_ls_accounts, top_ls_positions, ls_accounts, taker_ls_vol"
            con.execute(f"INSERT INTO metrics({mcols}) SELECT {mcols} FROM src.metrics WHERE symbol IN ({marks}) AND "
                        f"ts>=? AND ts<=?", (*symbols, lo, cfg.start_ms - 5 * MS_MINUTE))
        con.commit()
        con.execute("DETACH DATABASE src")
    finally:
        con.close()
    if cfg.params or cfg.seed_trades or cfg.seed_challengers:
        from heartless.core.models import TradeRecord
        from heartless.strategy.params import StrategyParams

        st = Store(db)
        try:
            if cfg.params:
                p = StrategyParams.from_dict(cfg.params)
                st.save_params_version(p.version, p.created, "champion", "replay", p.note or "replay start", p.to_dict())
            for name, params, started, source in cfg.seed_challengers:
                p = StrategyParams.from_dict(params)
                st.save_params_version(p.version, p.created, "challenger", source, p.note or source, p.to_dict(),
                                       {"slot": name, "started": int(started)})
            for t in cfg.seed_trades:
                st.save_trade(TradeRecord(**t))
        finally:
            st.close()


def _settings(cfg: ReplayConfig, work: Path, symbols: list[str]):
    from heartless.config import Settings

    always = [s for s in "BTCUSDT,ETHUSDT,SOLUSDT".split(",") if s in symbols]
    values = {**REPLAY_DEFAULTS, "ALWAYS_INCLUDE": ",".join(always) or symbols[0], **cfg.settings, **FORCED_SETTINGS,
              "HEARTLESS_DATA_DIR": str(work), "LOG_LEVEL": cfg.log_level}
    s = Settings(_env_file=None, **values)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


def run_replay(cfg: ReplayConfig) -> dict:
    """Replay [cfg.start_ms, cfg.end_ms] with the real orchestrator; returns the summary (also written to out_json)."""
    from heartless.learning import seal
    from heartless.replay.clock import run_virtual
    from heartless.replay.market import ReplaySource
    from heartless.replay.telegram import parse_command

    if cfg.end_ms <= cfg.start_ms + MS_MINUTE:
        raise ValueError("replay window is empty")
    # the replayed bot reads the seed history (research / discovery evaluate it) and the window itself
    seal.check_window(cfg.start_ms - cfg.seed_days * MS_DAY, cfg.end_ms)
    source = ReplaySource(cfg.source_db, cfg.symbols)
    try:
        symbols = source.symbols
        for s in symbols:
            first, last, n = source.coverage(s, cfg.start_ms, cfg.end_ms)
            if not n:
                raise ValueError(f"{s}: no stored candles in the replay window")
        if cfg.work_dir:
            work = Path(cfg.work_dir)
            if work.exists() and any(work.iterdir()):
                if not cfg.overwrite:
                    raise FileExistsError(f"work dir {work} is not empty (pass overwrite / --overwrite)")
                shutil.rmtree(work)
            work.mkdir(parents=True, exist_ok=True)
        else:
            work = Path(tempfile.mkdtemp(prefix="heartless-replay-"))
        _prepare_store(cfg, work, symbols)
        settings = _settings(cfg, work, symbols)
        commands = cfg.commands
        if commands is None:
            commands = [parse_command(c, cfg.start_ms) for c in DEFAULT_SCRIPT]
        commands = [c for c in commands if cfg.start_ms <= c.at_ms <= cfg.end_ms]
        ctx = _Run(cfg, settings, source, symbols, commands, work)
        wall0 = time.perf_counter()
        old_factory = logging.getLogRecordFactory()
        root = logging.getLogger()
        old_level = root.level
        root.setLevel(getattr(logging, cfg.log_level.upper(), logging.WARNING))
        ctx.install_log_time(old_factory)
        try:
            summary = run_virtual(ctx.main, cfg.start_ms, speed=cfg.speed, idle_grace_s=cfg.idle_grace_s,
                                  exception_handler=ctx.loop_exception)
        finally:
            logging.setLogRecordFactory(old_factory)
            root.setLevel(old_level)
            if ctx.log_handler is not None:
                root.removeHandler(ctx.log_handler)
        summary["wall_seconds"] = round(time.perf_counter() - wall0, 1)
        summary["speedup"] = round(summary["virtual_seconds"] / max(summary["wall_seconds"], 1e-9), 1)
    finally:
        source.close()
    if cfg.out_json:
        Path(cfg.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out_json).write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
    report = format_report(summary)
    summary_report_path = cfg.out_report or (str(Path(cfg.out_json).with_suffix(".txt")) if cfg.out_json else None)
    if summary_report_path:
        Path(summary_report_path).write_text(report)
    summary["report"] = report
    return summary


class _Run:
    """State of one replay run (lives on the virtual loop)."""

    def __init__(self, cfg: ReplayConfig, settings, source, symbols: list[str], commands: list, work: Path):
        self.cfg, self.settings, self.source, self.symbols, self.commands, self.work = cfg, settings, source, symbols, commands, work
        self.checker = None
        self.log_handler: _LogCapture | None = None
        self.events: dict[str, int] = {}
        self.opened: dict[str, list[int]] = {}
        self.closed: dict[str, int] = {}
        self.research: list[dict] = []
        self.discovery: list[dict] = []
        self.promotions: list[dict] = []
        self.installed: list[dict] = []
        self.retired: list[dict] = []
        self.evaluations = {"calls": 0, "compared": 0, "early_checks": 0}
        self.pending_errors: list[str] = []  # violations raised before the checker exists
        self.wall0 = time.perf_counter()

    # --- infrastructure hooks ----------------------------------------------------------------------
    def install_log_time(self, old_factory) -> None:
        """Log records carry the virtual time (log files of a replay read like the bot's own logs)."""
        def factory(*a, **k):
            rec = old_factory(*a, **k)
            ms = now_ms()
            rec.created = ms / 1000
            rec.msecs = ms % 1000
            return rec

        logging.setLogRecordFactory(factory)

    def loop_exception(self, loop, context: dict) -> None:
        # recorded as a violation instead of asyncio's default ERROR log (which the log scan would count twice)
        if self.checker is not None:
            self.checker.loop_exception(context)
        else:
            self.pending_errors.append(f"{context.get('message')}: {context.get('exception')!r}")

    # --- the run -----------------------------------------------------------------------------------
    async def main(self, loop) -> dict:
        from heartless.app import Heartless
        from heartless.replay.invariants import InvariantChecker
        from heartless.replay.market import ReplayMarketStream, ReplayRest, symbol_infos
        from heartless.replay.netguard import block_network
        from heartless.replay.telegram import CaptureTelegramBot, TelegramCapture

        cfg = self.cfg
        if cfg.tracemalloc:
            import tracemalloc

            tracemalloc.start()
        infos = symbol_infos(self.source, cfg.start_ms)
        await loop.sleep_until_ms(cfg.start_ms + cfg.app_start_offset_ms)
        app = Heartless(self.settings)
        checker = self.checker = InvariantChecker(app)
        for e in self.pending_errors:
            checker.violation("exceptions", f"before start: {e}")
        self.log_handler = _LogCapture(checker)
        logging.getLogger().addHandler(self.log_handler)
        with block_network(checker.network):
            real_rest = app.rest
            app.rest = ReplayRest(self.source, infos, latency_ms=cfg.rest_latency_ms)
            await real_rest.close()
            market = ReplayMarketStream(self.source, app._on_kline, app._on_book, app._on_mark, cfg.start_ms, cfg.end_ms)
            app.market = market
            capture = TelegramCapture(commands=list(self.commands))
            capture.listeners.append(checker.on_message)
            capture.resolve_symbol = lambda: self._open_symbol(app)
            app.telegram_factory = lambda a: CaptureTelegramBot(a, capture)
            self._hook(app)
            app_task = asyncio.create_task(app.run(), name="heartless")
            checker_task = asyncio.create_task(self._check_loop(loop, app, market), name="replay-checker")
            inject_task = asyncio.create_task(self._inject_loop(loop, app), name="replay-inject")
            finished = asyncio.create_task(market.finished.wait())
            done, _ = await asyncio.wait({app_task, finished}, return_when=asyncio.FIRST_COMPLETED)
            finished.cancel()
            crashed = app_task in done
            if crashed:
                exc = app_task.exception() if not app_task.cancelled() else None
                checker.violation("exceptions", f"orchestrator stopped early: {type(exc).__name__ if exc else 'cancelled'}"
                                                f": {exc}")
            checker_task.cancel()
            inject_task.cancel()
            # never abandon a research / discovery worker mid-flight (its result would be lost and the pool hang); the
            # short sleep lets a cycle the scheduler has just created start and raise its `running` flag
            await asyncio.sleep(0.01)
            while app.research.running:
                await asyncio.sleep(1)
            if not crashed:
                await checker.check_minute()
            pre_flatten = self._engine_stats(app)
            if cfg.final_flatten and not crashed:
                for eng in list(app.engines.values()):
                    await eng.close_all("리플레이 종료(replay end)")
                await asyncio.sleep(1)
                app._persist_paper_accounts()
                await checker.check_minute()
                checker.check_trades(final=True)
            else:
                checker.check_trades()
            checker.sample_memory(cfg.tracemalloc)
            memory = checker.check_memory(cfg.memory_limit_mb, cfg.memory_slope_mb_per_day, cfg.memory_warmup_ms)
            summary = self._summary(loop, app, market, capture, memory, pre_flatten)
            if cfg.trades_out:
                rows = [dict(r) for r in app.store.query("SELECT * FROM trades ORDER BY exit_time")]
                Path(cfg.trades_out).write_text(json.dumps(rows, indent=0, default=str))
            if cfg.messages_out:
                with open(cfg.messages_out, "w") as f:
                    for m in capture.messages:
                        f.write(json.dumps({"ts": m.ts, "time": fmt_ts(m.ts), "topic": m.topic, "text": m.text},
                                           ensure_ascii=False) + "\n")
            await app.stop()
            try:
                await asyncio.wait_for(app_task, timeout=30)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
            # violations recorded during shutdown (a failing stop) still count
            summary["invariants"] = checker.summary()
            summary["passed"] = checker.total_violations == 0
        if cfg.tracemalloc:
            import tracemalloc

            tracemalloc.stop()
        return summary

    async def _check_loop(self, loop, app, market) -> None:
        cfg, checker = self.cfg, self.checker
        nxt = cfg.start_ms + cfg.check_every_ms + 500  # between a bar's close (:59.999) and the next open tick (+1 s)
        next_mem = cfg.start_ms
        next_trades = cfg.start_ms + MS_HOUR
        next_progress = cfg.start_ms + MS_DAY
        while not market.finished.is_set():
            await loop.sleep_until_ms(nxt)
            if app.engines:
                await checker.check_minute()
            now = now_ms()
            if now >= next_mem:
                checker.sample_memory(cfg.tracemalloc)
                next_mem += cfg.memory_every_ms
            if now >= next_trades:
                checker.check_trades()
                next_trades += MS_HOUR
            if now >= next_progress:
                next_progress += MS_DAY
                self._progress(app, market)
            nxt += cfg.check_every_ms

    def _progress(self, app, market) -> None:
        """One line per virtual day on stdout (long soaks are watched from a log file)."""
        cfg = self.cfg
        done = (now_ms() - cfg.start_ms) / MS_DAY
        total = (cfg.end_ms - cfg.start_ms) / MS_DAY
        wall = time.perf_counter() - self.wall0
        n_trades = app.store.query("SELECT COUNT(*) n FROM trades")[0]["n"]
        rss = self.checker.memory[-1]["rss_mb"] if self.checker.memory else 0.0
        print(f"[replay] {fmt_ts(now_ms(), fmt='%Y-%m-%d %H:%M')} day {done:.0f}/{total:.0f} wall {wall:,.0f}s "
              f"(x{done * 86400 / max(wall, 1e-9):,.0f}) bars {market.bars_emitted:,} trades {n_trades} "
              f"research {len(self.research)} discovery {len(self.discovery)} challengers {len(app.challengers)} "
              f"violations {self.checker.total_violations} rss {rss:.0f}MB", flush=True)

    async def _inject_loop(self, loop, app) -> None:
        for at, fn in sorted(self.cfg.inject, key=lambda x: x[0]):
            await loop.sleep_until_ms(at)
            res = fn(app)
            if asyncio.iscoroutine(res):
                await res

    @staticmethod
    def _open_symbol(app) -> str:
        prim = app.primary_engine
        open_syms = sorted(p.symbol for p in prim.open_positions()) if prim else []
        return open_syms[0] if open_syms else (app.universe[0] if app.universe else "BTCUSDT")

    # --- observation (no behaviour change) ---------------------------------------------------------
    def _hook(self, app) -> None:
        topics = ("position_opened", "position_closed", "partial_tp", "entry_cancelled", "risk_event", "error",
                  "research_started", "research_done", "promotion", "graduation_ready", "daily_report", "mode_changed",
                  "control", "started", "universe", "advisor_review", "reconcile", "challenger_retired")
        for t in topics:
            app.bus.subscribe(t, self._make_counter(t))
        research = app.research
        orig_cycle, orig_disc, orig_eval = research.cycle, research.discovery_cycle, research.evaluate_challengers
        orig_install = app.install_challenger
        orig_start = app._start_challenger
        seed = self.cfg.seed
        if seed is not None:
            app.bandit.rng.seed(seed)

        async def start_challenger(slot, restore: bool = False):
            await orig_start(slot, restore)
            eng = app.engines.get(slot.name)
            if eng is not None and seed is not None:  # before its first decision: start() itself decides nothing
                eng.bandit.rng.seed(f"{seed}:{slot.name}:{slot.params.version}")

        async def cycle(force: bool = False):
            t0, ts = time.perf_counter(), now_ms()
            res = await orig_cycle(force)
            if res is not None:
                self.research.append({"ts": ts, "time": fmt_ts(ts), "wall_s": round(time.perf_counter() - t0, 1),
                                      "error": res.get("error"), "window": res.get("window"),
                                      "alphas": {a: {"best_score": round(r["best"]["score"], 3),
                                                     "base_score": round(r["base"]["score"], 3),
                                                     "is_base": r["best"].get("is_base"),
                                                     "test_n": r["best"]["test"].get("n", 0)}
                                                 for a, r in (res.get("alphas") or {}).items()}})
            return res

        async def discovery_cycle(force: bool = False):
            t0, ts = time.perf_counter(), now_ms()
            res = await orig_disc(force)
            if res is not None:
                self.discovery.append({"ts": ts, "time": fmt_ts(ts), "wall_s": round(time.perf_counter() - t0, 1),
                                       "tested": res.get("tested"), "passed": len(res.get("passed") or []),
                                       "errors": res.get("errors"), "rows": res.get("rows")})
            return res

        async def evaluate_challengers(force: bool = False):
            # mirror the promotion gate on the same store state to count evaluations that actually compared
            if force or now_ms() - research.last_promotion_check >= 60 * 60_000:
                self.evaluations["calls"] += 1
                s = app.s
                for slot in list(app.challengers):
                    n_ch = len(app.store.load_trades(engine=slot.name, since=slot.started))
                    n_champ = len(app.store.load_trades(engine="paper", since=slot.started))
                    if n_ch >= s.promotion_min_trades and n_champ >= max(10, s.promotion_min_trades // 2):
                        self.evaluations["compared"] += 1
                    elif n_ch >= 15:
                        self.evaluations["early_checks"] += 1
            return await orig_eval(force)

        async def install_challenger(params, source):
            slot = await orig_install(params, source)
            self.installed.append({"ts": now_ms(), "time": fmt_ts(now_ms()), "source": source,
                                   "slot": slot.name if slot else None, "version": params.version, "note": params.note})
            return slot

        research.cycle, research.discovery_cycle, research.evaluate_challengers = cycle, discovery_cycle, evaluate_challengers
        app.install_challenger = install_challenger
        app._start_challenger = start_challenger

    def _make_counter(self, topic: str):
        async def handler(p: dict) -> None:
            self.events[topic] = self.events.get(topic, 0) + 1
            eng = p.get("engine", "system")
            if topic == "position_opened":
                self.opened.setdefault(eng, []).append(int(p.get("ts") or now_ms()))
            elif topic == "position_closed":
                self.closed[eng] = self.closed.get(eng, 0) + 1
            elif topic == "promotion":
                self.promotions.append({"ts": now_ms(), "time": fmt_ts(now_ms()), "version": p.get("version"),
                                        "message": p.get("message")})
            elif topic == "challenger_retired":
                self.retired.append({"ts": now_ms(), "time": fmt_ts(now_ms()), "message": p.get("message")})
        return handler

    # --- results -----------------------------------------------------------------------------------
    def _engine_stats(self, app) -> dict:
        out = {}
        for name, eng in app.engines.items():
            acc = eng.account
            out[name] = {"open_positions": len(eng.open_positions()), "equity": round(acc.equity(), 4)
                         if hasattr(acc, "equity") else None, "params_version": eng.params.version,
                         "skipped": dict(sorted(eng.stats.skipped.items()))}
        return out

    def _summary(self, loop, app, market, capture, memory: dict, pre_flatten: dict) -> dict:
        from heartless.execution.stats import by_alpha, summarize

        cfg = self.cfg
        engines: dict[str, dict] = {}
        all_trades = app.store.load_trades()
        for name in sorted({t["engine"] for t in all_trades} | set(app.engines)):
            trades = [t for t in all_trades if t["engine"] == name]
            st = summarize(trades)
            eng = app.engines.get(name)
            acc = eng.account if eng else None
            row = {"trades": st["n"], "net": round(st["net"], 4), "win_rate": round(st["win_rate"], 1),
                   "avg_r": round(st["avg_r"], 4), "profit_factor": round(min(st["profit_factor"], 999.0), 3),
                   "fees": round(st["fees"], 4), "funding": round(st["funding"], 4), "max_dd": round(st["max_dd"], 4),
                   "active": eng is not None, "params_version": eng.params.version if eng else None,
                   "by_alpha": {a: {"n": s["n"], "net": round(s["net"], 2), "avg_r": round(s["avg_r"], 3)}
                                for a, s in by_alpha(trades).items()},
                   "exit_reasons": _count(t["exit_reason"] for t in trades)}
            if eng is not None:
                row["skipped"] = dict(sorted(eng.stats.skipped.items()))
            if acc is not None and hasattr(acc, "wallet"):
                row.update(wallet=round(acc.wallet, 4), initial=acc.initial_balance,
                           return_pct=round((acc.wallet / acc.initial_balance - 1) * 100, 3),
                           fees_paid=round(acc.fees_paid, 4), funding_paid=round(acc.funding_paid, 4))
            engines[name] = row
        span_ms = now_ms() - cfg.start_ms
        champ = engines.get("paper", {})
        mem_curve = [{"time": fmt_ts(r["ts"], fmt="%m-%d %H:%M"), **{k: v for k, v in r.items() if k != "ts"}}
                     for r in _thin(self.checker.memory, 120)]
        opened = {k: len(v) for k, v in self.opened.items()}
        return {
            "window": {"start": fmt_ts(cfg.start_ms), "end": fmt_ts(cfg.end_ms), "start_ms": cfg.start_ms,
                       "end_ms": cfg.end_ms, "virtual_days": round((cfg.end_ms - cfg.start_ms) / MS_DAY, 3)},
            "virtual_seconds": round(span_ms / 1000, 1),
            "source_db": str(cfg.source_db), "work_dir": str(self.work), "symbols": self.symbols,
            "universe_final": list(app.universe), "settings": {k: getattr(app.s, k.lower(), None)
                                                               for k in REPLAY_DEFAULTS} | {"timezone": app.s.timezone},
            "loop": {"jumps": loop.jumps, "executor_jobs": loop.executor_jobs,
                     "frozen_wall_seconds": round(loop.frozen_wall_s, 1), "deadlocked": loop.deadlocked},
            "market": {"bars": market.bars_emitted, "minutes": market.minutes, "messages": market.messages,
                       "gaps": market.gaps, "rest_calls": dict(sorted(app.rest.calls.items()))},
            "trades": {"total": len(all_trades), "opened": opened, "closed": dict(self.closed)},
            "pnl": {"champion_net": champ.get("net"), "champion_return_pct": champ.get("return_pct"),
                    "champion_trades": champ.get("trades")},
            "engines": engines, "before_final_flatten": pre_flatten,
            "telegram": {"messages": len(capture.messages), "by_type": capture.counts(),
                         "api_calls": dict(sorted(capture.api_calls.items())),
                         "commands": [{"time": fmt_ts(t), "text": x} for t, x in capture.delivered]},
            "events": dict(sorted(self.events.items())),
            "learning": {"research_cycles": self.research, "discovery_cycles": self.discovery,
                         "challengers_installed": self.installed, "challengers_retired": self.retired,
                         "promotions": self.promotions, "challenger_evaluations": self.evaluations,
                         "challengers_final": [{"name": c.name, "source": c.source, "version": c.params.version,
                                                "started": fmt_ts(c.started)} for c in app.challengers],
                         "champion_final": app.params.version},
            "memory": {**memory, "curve": mem_curve},
            "warnings": dict(sorted(self.log_handler.warnings.items(), key=lambda kv: -kv[1])[:40])
            if self.log_handler else {},
            "invariants": self.checker.summary(),
        }


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _thin(rows: list, n: int) -> list:
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)] + [rows[-1]]


def format_report(s: dict) -> str:
    """Human-readable summary of a replay run."""
    inv = s["invariants"]
    lines = [f"Heartless replay {s['window']['start']} -> {s['window']['end']} ({s['window']['virtual_days']:g} days, "
             f"{len(s['symbols'])} symbols)",
             f"result: {'PASS' if s.get('passed') else 'FAIL'} - {inv['violations_total']} invariant violations",
             f"wall {s.get('wall_seconds', 0):.0f}s, speed-up x{s.get('speedup', 0):,.0f}, "
             f"{s['loop']['frozen_wall_seconds']:.0f}s waiting for executor work (virtual time frozen)",
             f"bars {s['market']['bars']:,} ({s['market']['minutes']:,} minutes, {s['market']['gaps']} gaps), "
             f"stream messages {s['market']['messages']:,}",
             "", "invariants (runs / violations):"]
    for k, c in inv["checks"].items():
        lines.append(f"  {k:<12} {c['runs']:>9,} / {c['violations']:<5} {c['what']}")
    for v in inv["violations"][:25]:
        lines.append(f"  ! {v['time']} {v['check']} [{v['engine'] or '-'}] {v['detail']}")
    lines += ["", "engines:"]
    for name, e in s["engines"].items():
        ret = f", wallet {e['wallet']:,.2f} ({e['return_pct']:+.2f}%)" if "wallet" in e else ""
        lines.append(f"  {name:<14} {e['trades']:>4} trades, net {e['net']:+,.2f}, avgR {e['avg_r']:+.3f}, "
                     f"PF {e['profit_factor']:.2f}, win {e['win_rate']:.0f}%{ret}{'' if e['active'] else ' (retired)'}")
    learn = s["learning"]
    lines += ["", f"learning: {len(learn['research_cycles'])} research cycles, {len(learn['discovery_cycles'])} "
                  f"discovery cycles, {len(learn['challengers_installed'])} challengers installed, "
                  f"{len(learn['challengers_retired'])} retired, {len(learn['promotions'])} promotions; challenger "
                  f"evaluations {learn['challenger_evaluations']}; champion {learn['champion_final']}"]
    for r in learn["research_cycles"]:
        lines.append(f"  research {r['time']} ({r['wall_s']}s){' error: ' + str(r['error']) if r.get('error') else ''}")
    for r in learn["discovery_cycles"]:
        lines.append(f"  discovery {r['time']} ({r['wall_s']}s): {r['tested']} rules tested, {r['passed']} passed")
    for r in learn["challengers_installed"]:
        lines.append(f"  challenger {r['time']} {r['slot']} <- {r['source']}: {r['note']}")
    for r in learn["promotions"]:
        lines.append(f"  promotion {r['time']} {r['version']}")
    tg = s["telegram"]
    lines += ["", f"telegram: {tg['messages']} messages; " + ", ".join(f"{k} {v}" for k, v in tg["by_type"].items())]
    for c in tg["commands"]:
        lines.append(f"  command {c['time']} {c['text']}")
    m = s["memory"]
    if m.get("samples", 0) >= 3:
        lines += ["", f"memory: RSS {m['rss_start_mb']:.0f} -> {m['rss_end_mb']:.0f} MB (peak {m['rss_peak_mb']:.0f}), "
                      f"growth after warm-up {m['growth_after_warmup_mb']:+.0f} MB, slope {m['slope_mb_per_day']:+.2f} "
                      f"MB/day (limits {m['limit_mb']:.0f} MB, {m['slope_limit_mb_per_day']:.1f} MB/day)"]
        curve = m.get("curve") or []
        for r in curve[:: max(1, len(curve) // 12)]:
            lines.append(f"  {r['time']}  rss {r['rss_mb']:>7.1f} MB  candles {r['candle_rows']:>9,}")
    if s.get("warnings"):
        lines += ["", "warnings (count):"] + [f"  {n:>6}  {k}" for k, n in list(s["warnings"].items())[:15]]
    return "\n".join(lines)

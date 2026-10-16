"""Heartless orchestrator: market data, engines (live + paper champion + challengers), learning, UIs."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Candle, PositionStatus, Regime, SymbolInfo, Ticker
from heartless.core.store import Store
from heartless.data.candles import CandleArrays
from heartless.data import context as CTX
from heartless.data.context import EMPTY as NO_MARKET_CONTEXT
from heartless.data.context import WARMUP_HOURS as CONTEXT_WARMUP_HOURS
from heartless.data.context import MarketContextTracker, hour_close, hour_of
from heartless.data.extras import MetricsSeries
from heartless.data.features import LIVE_WINDOW, MarketView
from heartless.data.universe import select_universe
from heartless.exchange.binance_rest import BinanceError, BinanceRest
from heartless.exchange.binance_ws import MarketStream
from heartless.exchange.live import LiveAccount
from heartless.exchange.orderbook import EMPTY_BOOK  # F7: ctx.book when no order book runs
from heartless.exchange.paper import PaperAccount
from heartless.exchange.symbols import parse_exchange_info
from heartless.execution.engine import TradingEngine
from heartless.execution.stats import by_alpha, exec_quality, r_stats, slippage_bps, summarize
from heartless.learning import research as R
from heartless.learning.advisor import Advisor
from heartless.learning.bandit import AlphaBandit
from heartless.learning.research import AlphaKillSwitch, ChallengerSlot, ResearchManager
from heartless.strategy import evidence
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.strategy.regime import detect_regime
from heartless.util import perf  # F7: latency budget, clock drift
from heartless.util.ids import short_id, token_hex
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, fmt_ts, local_day_start, next_local_time, now_ms

log = logging.getLogger(__name__)

# how often the scheduler re-runs the history fetch for symbols whose backfill failed
BACKFILL_RETRY_MS = 5 * MS_MINUTE
# 1m history kept in memory per symbol: the longest live frame (1h x LIVE_WINDOW["1h"] = 21.7 days) plus margin. Older
# bars can never reach a feature frame (each frame is the tail of the resampled history), so dropping them changes no
# feature, while keeping them grew memory ~1.4 MB/day for 12 symbols and the per-boundary resample cost without bound.
LIVE_HISTORY_DAYS = math.ceil(LIVE_WINDOW["1h"] / 24) + 2
# universe hysteresis: a member stays until its 24h quote volume drops below this fraction of the entry threshold. A
# symbol hovering around UNIVERSE_MIN_QUOTE_VOLUME was otherwise added and removed at hourly refreshes (a 2-day replay
# of July 2026 flipped AVAXUSDT 6 times), each time reloading its history, rebuilding its views and messaging the owner.
UNIVERSE_KEEP_FRACTION = 0.8
# how often the scheduler stores the funding settlements published since the last stored one. It only asks REST for a
# symbol once a settlement is due, so the short period costs nothing between settlements; it bounds how late a new
# settlement reaches the market context (the snapshot at the next hour close must see it, as the offline build does)
FUNDING_REFRESH_MS = 10 * MS_MINUTE
# market context (data/context.py): how long past an hour close the snapshot waits for the context symbols' bars
# closing at it (live klines arrive within a second or two) before it is published without the stragglers, and how
# often that wait looks again
CONTEXT_GRACE_MS = 10_000
CONTEXT_POLL_S = 0.05
# stored 1m history every context symbol needs at start-up (the tracker's longest window plus the carry), backfilled
# even when HISTORY_DAYS / DATA_RETENTION_DAYS are shorter (the retention prune keeps DATA_RETENTION_DAYS + 5 days)
CONTEXT_HISTORY_DAYS = math.ceil(CONTEXT_WARMUP_HOURS / 24) + 1
# funding settlements (C2-B): a symbol that Binance's fundingInfo does not list settles every 8 h; others every
# fundingIntervalHours (4 h, sometimes 1-2 h when the rate hits its cap). The next settlement time itself comes from
# premiumIndex nextFundingTime (and the markPrice stream for traded symbols).
DEFAULT_FUNDING_INTERVAL_MS = 8 * MS_HOUR
# a settlement that the funding history still did not show this long after it was due is not asked for again (its
# interval changed under us); a new expectation (the next settlement) is asked for as usual
FUNDING_MISS_GRACE_MS = 15 * MS_MINUTE
# P3 signal batching (PORTFOLIO_ALLOCATOR, heartless/execution/allocator.py): the entry candidates of one bar close are
# admitted together once every traded symbol's bar at that close has been processed; a batch still incomplete this long
# after its close (a symbol's stream lagging or dead) is admitted without the stragglers by the scheduler
ENTRY_BATCH_GRACE_MS = 15_000


def live_context_path(settings: Settings) -> Path:
    """The store the bot appends its hourly market-context snapshots to and its research worker reads them from (C2-B).
    Resolved by Settings.context_db_path, the one resolution lab, discovery and the backtester use as well (CONTEXT_DB,
    else <data_dir>/context.db). The live and the offline stores stay apart because the bot and the research commands
    run on different data dirs, and each refuses the other's store (_ctx_meta here, context.source_status there)."""
    return settings.context_db_path


# truthful status line while the champion has no alpha switched on (startup message, /status, daily report)
NO_ALPHA_TEXT = "⚠️ 활성 알파 없음(0개): 거래하지 않고 연구·발굴·챌린저 검증만 진행"
# ROADMAP P2 (final exam rule): the sealed-2025 joint portfolio lost (-0.77%, docs/SEALED_2025.md), so the startup
# message keeps "live trading not recommended" while no alpha has passed a pre-registered out-of-sample test
LIVE_NOT_RECOMMENDED_TEXT = ("⚠️ 실거래 비권장: 사전 등록 표본 외 시험(P1·P2·2025 확인 시험)을 통과한 알파가 없고, "
                             "2025 정보용 공동 포트폴리오도 손실(-0.77%)이었습니다 (docs/SEALED_2025.md)")


class Heartless:
    # next local midnight (set in __init__); the class default keeps the day roll inert on instances built without
    # __init__ (tests drive _tick on such partial apps)
    _next_day_ms: int = 2**62
    # long-lookback alphas (C2-A): last hourly history-depth check, its running task, the depth it last asked for, and
    # the depth already attempted per symbol (a symbol listed more recently cannot get deeper; it is not retried)
    _last_depth_check: int = 0
    _depth_task: asyncio.Task | None = None
    _depth_need: int = 0
    _depth_tried: dict | None = None
    # why the live context store is not used (it records another candle database), "" while it is (integration)
    _ctx_off: str = ""
    # P3 signal batching: close time of the open entry batch and the traded symbols whose bar at it was processed
    _batch_t: int | None = None
    _batch_seen: set | None = None
    # F7: local order books (ORDERBOOK_STREAM) and their depth streams; None when off
    books: Any = None
    book_stream: Any = None

    def __init__(self, settings: Settings):
        self.s = settings
        self.store = Store(settings.db_path)
        self.bus = EventBus()
        self.rest = BinanceRest(settings.binance_api_key, settings.binance_api_secret, settings.binance_testnet)
        self.symbols: dict[str, SymbolInfo] = {}
        self.universe: list[str] = []
        self.candles: dict[str, CandleArrays] = {}
        self.views: dict[str, MarketView] = {}
        self.tickers: dict[str, Ticker] = {}
        self.book_qty: dict[str, tuple[float, float]] = {}
        self.oi_hist: dict[str, deque] = {}
        self.metrics_series: dict[str, MetricsSeries] = {}  # 5m positioning data per symbol
        # live market context (P2a, data/context.py): the hourly cross-sectional snapshot over CONTEXT_UNIVERSE that
        # every engine receives as Context.market, computed exactly like the offline market_context table. Members of
        # that universe the bot does not trade ("context-only") are streamed and stored as 1m bars only: no feature
        # frames, no engines, one REST series (open interest) in the metrics poll.
        self.context = MarketContextTracker(settings.context_universe_list)
        self.context_only: list[str] = []
        self._ctx_last_open: dict[str, int] = {}  # context-only symbol -> open time of its newest stored bar
        self._ctx_polled: set[str] = set()  # context-only symbols whose first (2-day) open-interest poll succeeded
        self.context_stats: dict[str, Any] = {"snapshots": 0, "late": 0, "errors": 0, "last": None, "symbols": 0,
                                              "stored": 0, "filled": 0, "fill_errors": 0}
        # C2-B: every published snapshot is appended to the live context store (CONTEXT_DB or <data_dir>/context.db),
        # pruned with the data retention; hours of the research window it lacks are built from the stored candles in
        # the background at start-up. The research worker's backtests read it (ResearchManager.worker_settings).
        self.live_context_path = live_context_path(settings)
        self._ctx_store: Store | None = None
        self._ctx_off = ""
        self._ctx_fill_task: asyncio.Task | None = None
        # C2-B: funding schedule per symbol (see DEFAULT_FUNDING_INTERVAL_MS)
        self.funding_next: dict[str, int] = {}  # symbol -> next settlement time as the exchange last reported it
        self.funding_interval: dict[str, int] = {}  # symbol -> settlement interval (ms) when not 8 h
        self._funding_miss: dict[str, int] = {}  # symbol -> expected settlement the history did not show
        self.regimes: dict[str, tuple[Regime, dict]] = {}
        self.bandit = AlphaBandit(list(StrategyParams.default().alphas), store=self.store, engine="shared")
        self.research = ResearchManager(self)
        self.meta = None
        if settings.meta_label:
            from heartless.learning.metalabel import MetaLabeler

            self.meta = MetaLabeler(min_samples=settings.meta_min_samples, store=self.store, key="metalabel")
        self.params: StrategyParams = self.research.load_champion()
        # per-alpha kill switch on live closed trades (ROADMAP P4); its state is derived from the stored trades
        self.killswitch = AlphaKillSwitch(self)
        self.killswitch.sync_champion(self.params)
        self.bus.subscribe("position_closed", self.killswitch.on_trade)
        # execution-quality monitor: rolling fill rows per engine (slippage vs the decision reference, fee vs model)
        self.fill_rows: dict[str, deque] = {}
        self._entry_refs: dict[str, float] = {}  # position id -> reference price of its first entry fill
        self._order_refs: dict[str, tuple[float, str, str]] = {}  # client id -> (decision reference, kind, position id)
        self.challengers: list[ChallengerSlot] = []
        self.engines: dict[str, TradingEngine] = {}
        self.paper_accounts: dict[str, PaperAccount] = {}
        self.live_account: LiveAccount | None = None
        self.market = MarketStream(self.rest.ws_base, self._on_kline, self._on_book, self._on_mark)
        # --- F7: speed and optimization (docs/PERFORMANCE.md) -----------------------------------------------------
        perf.DRIFT.warn_ms = settings.clock_drift_warn_ms
        self._f7_recv: dict[str, int] = {}  # symbol -> local time its latest closed kline arrived
        self._f7_bar: dict[str, int] = {}  # symbol -> close time of the bar its engines are deciding on right now
        self.startup_timings: dict[str, float] = {}  # startup check -> milliseconds
        self.books = None  # exchange.orderbook.OrderBooks (ORDERBOOK_STREAM); the replay installs its 1m-bar proxy
        self.book_stream = None  # exchange.binance_ws.BookStream feeding it
        if settings.orderbook_stream:
            from heartless.exchange.binance_ws import BookStream
            from heartless.exchange.orderbook import OrderBooks

            self.books = OrderBooks(settings.orderbook_mode, settings.orderbook_depth_bps,
                                    fetch_snapshot=self._f7_depth_snapshot)
            self.book_stream = BookStream(self.rest.ws_base, self.books.on_message, settings.orderbook_mode,
                                          settings.orderbook_speed_ms, on_reset=self.books.on_reset)
        # --- end F7 ---
        self.advisor = Advisor(settings.anthropic_api_key)
        self.mode_notice: str = ""  # startup warning when HEARTLESS_MODE and the persisted mode disagree
        self.mode: str = self._resolve_mode(settings)
        if self.mode == "live" and not settings.live_capable:
            log.warning("HEARTLESS_MODE=live but no Binance keys; falling back to paper")
            self.mode = "paper"
        self.web_token: str = self.store.get("web.token") or self._new_web_token()
        self.started_at = now_ms()
        self.tasks: list[asyncio.Task] = []
        self._kline_queue: asyncio.Queue = asyncio.Queue()
        self._stopping = False
        self.last_bar_ts: dict[str, int] = {}
        self.telegram = None
        # builds the Telegram bot in run(); None = the production TelegramBot (the replay harness injects its capture bot)
        self.telegram_factory: Callable[["Heartless"], Any] | None = None
        self.web = None
        self._backfilling: set[str] = set()
        self.archive = None  # BinanceArchive, created lazily for bulk backfill
        self._backfill_failed: set[str] = set()  # symbols whose history fetch failed; retried by the scheduler
        self._last_backfill_retry = 0
        self._next_report_ts = next_local_time(settings.daily_report_hour, 0, settings.timezone)
        self._last_prune = 0
        self._last_oi_poll = 0
        self._last_funding_refresh = 0
        self._last_universe = 0
        self._last_reconcile = 0
        self._last_equity_tick = 0
        self._last_day_key = local_day_start(now_ms(), settings.timezone)
        self._next_day_ms = next_local_time(0, 0, settings.timezone, now_ms())
        self.health: dict[str, Any] = {}

    # --- helpers -------------------------------------------------------------------------------
    def _new_web_token(self) -> str:
        t = token_hex(16)
        self.store.set("web.token", t)
        return t

    def _resolve_mode(self, settings: Settings) -> str:
        """Startup mode. The mode last chosen via Telegram/web is persisted and normally wins over HEARTLESS_MODE: an
        unchanged .env (the shipped example sets paper) must not undo /mode live on a crash-restart and leave real
        positions unmanaged. An operator who CHANGES HEARTLESS_MODE between two starts means it, so that edit
        overrides the persisted mode. Either disagreement is logged and reported in the startup message."""
        stored = self.store.get("mode")
        env_mode = settings.mode if "mode" in settings.model_fields_set else ""  # explicitly configured?
        last_env = self.store.get("mode.env")
        self.store.set("mode.env", env_mode)
        mode = stored or settings.mode
        if env_mode and stored and stored != env_mode:
            if last_env is not None and env_mode != last_env:
                log.warning("HEARTLESS_MODE changed (%s -> %s): overrides persisted mode %s", last_env or "unset",
                            env_mode, stored)
                self.store.set("mode", env_mode)
                self.mode_notice = (f"⚠️ HEARTLESS_MODE 변경({last_env or '미설정'} → {env_mode}) 적용: "
                                    f"저장된 {stored.upper()} 모드 대신 {env_mode.upper()} 모드로 시작")
                mode = env_mode
            else:
                log.warning("persisted mode %s overrides HEARTLESS_MODE=%s (switch with /mode or the web dashboard)",
                            stored, env_mode)
                self.mode_notice = (f"⚠️ .env HEARTLESS_MODE={env_mode} 는 무시됨 — 저장된 {stored.upper()} 모드 사용 중 "
                                    f"(/mode {env_mode} 로 전환)")
        return mode

    async def emit(self, topic: str, payload: dict | None = None) -> None:
        payload = dict(payload or {})
        payload.setdefault("ts", now_ms())
        payload.setdefault("engine", "system")
        payload.setdefault("notify", True)
        await self.bus.publish(topic, payload)
        if topic in ("risk_event", "error", "mode_changed", "promotion", "research_done", "graduation_ready", "universe",
                     "kill_switch", "challenger_retired"):
            self.store.log_event(payload["ts"], "INFO", topic, {k: v for k, v in payload.items() if k not in ("result",)})

    @property
    def primary_engine(self) -> TradingEngine | None:
        """The engine whose activity is reported in detail (live if live, else paper champion)."""
        return self.engines.get("live") or self.engines.get("paper")

    def marks(self) -> dict[str, float]:
        return {s: (t.mark or t.mid or t.last) for s, t in self.tickers.items()}

    # --- startup -------------------------------------------------------------------------------
    def enabled_alphas(self) -> list[str]:
        return [a for a, on in self.params.enabled.items() if on and a in self.params.alphas]

    async def run(self) -> None:
        log.info("Heartless starting (mode=%s, testnet=%s)", self.mode, self.s.binance_testnet)
        active = self.enabled_alphas()
        if active:
            log.info("enabled alphas (champion %s): %s", self.params.version, ", ".join(active))
        else:
            log.warning("no alpha is enabled (champion %s): the bot opens no trades; it collects data and runs research, "
                        "discovery and challenger validation, and starts trading only when a challenger clears the "
                        "absolute promotion bar on live paper results (docs/SELF_IMPROVEMENT.md)", self.params.version)
        from heartless.strategy.alphas import FAILED_ALPHAS

        for mod, err in FAILED_ALPHAS.items():
            log.error("alpha module %s failed to import and is unavailable: %s", mod, err)
        # --- F7: the environment checks run concurrently and are timed (were four sequential round trips) ---------
        await self._f7_startup_checks()
        # --- end F7 ---
        await self.refresh_universe(initial=True)
        self.context_only = self._context_only_symbols()
        await self._backfill(self.universe + self.context_only)
        self._build_views(self.universe)
        await self._create_engines()
        metrics_task = None
        if self.s.archive_backfill:
            metrics_task = asyncio.create_task(self._backfill_metrics(list(self.universe)), name="metrics-backfill")
            self.tasks.append(metrics_task)
        self._warm_context()  # after the backfill (the store holds the history), before the first streamed bar
        # hours of the research window the live context store lacks, built in the background (after the metrics
        # backfill, so the open-interest features of those hours see the archive rows too)
        self._ctx_fill_task = asyncio.create_task(self._fill_context_gaps(metrics_task), name="context-fill")
        self.market.set_symbols(self.universe, kline_only=self.context_only)
        self.tasks.append(asyncio.create_task(self.market.run(), name="market-stream"))
        # --- F7: depth streams for the local order books (ORDERBOOK_STREAM) --------------------------------------
        self._f7_book_symbols()
        if self.book_stream is not None:
            self.tasks.append(asyncio.create_task(self.book_stream.run(), name="depth-stream"))
        # --- end F7 ---
        self.tasks.append(asyncio.create_task(self._kline_worker(), name="kline-worker"))
        self.tasks.append(asyncio.create_task(self._scheduler(), name="scheduler"))
        if self.s.telegram_bot_token:
            if self.telegram_factory is not None:
                self.telegram = self.telegram_factory(self)
            else:
                from heartless.notify.telegram import TelegramBot

                self.telegram = TelegramBot(self)
            self.tasks.append(asyncio.create_task(self.telegram.run(), name="telegram"))
        if self.s.web_enabled:
            from heartless.web.app import WebServer

            self.web = WebServer(self)
            self.tasks.append(asyncio.create_task(self.web.run(), name="web"))
        # a kill-switch trip the previous process stored the trade for but did not get to act on is acted on now (after
        # the Telegram bot subscribed, so the owner is told; the engines' parameters change before anything can yield)
        await self.killswitch.check_all()
        await self.emit("started", {"message": self.startup_summary()})
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            pass

    def learning_status(self) -> dict:
        """What the self-improvement loop is doing and what has to happen for trading to start (startup message,
        /status, daily report). The criteria are the promotion rules themselves (heartless/learning/research.py)."""
        active = self.enabled_alphas()
        bar = R.promotion_bar(self.s, "absolute")
        failed_days = max(R.FAILED_MIN_DAYS, self.s.promotion_min_days)
        sched = "·".join(str(c) for c in R.PROMOTION_CHECKPOINTS)
        ts = "/".join(f"{R.ABS_T_AT[c]:g}" for c in R.PROMOTION_CHECKPOINTS)
        fsched = "·".join(str(c) for c in R.FAILED_CHECKPOINTS)
        fts = "/".join(f"{R.FAILED_ABS_T_AT[c]:g}" for c in R.FAILED_CHECKPOINTS)
        criteria = (f"챌린저가 실시간 페이퍼에서 절대 기준을 넘으면 승격되어 그 알파로 거래를 시작합니다. 판정은 시작 후 "
                    f"{sched}일 판정점에서만 합니다: 거래 {bar['min_trades']}건 이상·청산일 {bar['min_days']}일 이상, 평균R > 0, "
                    f"일 단위 HAC(뉴이-웨스트 {R.PROMOTION_HAC_LAG}일) t ≥ {ts}, PF ≥ {bar['min_pf_r']:g}, "
                    f"최대낙폭 ≤ {bar['max_dd_r']:g}R×√(n/{R.ABS_MIN_TRADES}) "
                    f"(OOS 탈락 알파를 켜는 챌린저는 {fsched}일 판정점·청산일 {failed_days}일·t ≥ {fts}). "
                    f"마지막 판정점까지 승격되지 않으면 종료")
        judged = [a for a in self.params.alphas if evidence.entry(a) is not None]
        counts = evidence.summary(judged)
        # the documents of the judgements behind those verdicts (P1: docs/OOS_2022_2024.md, P2: docs/RESEARCH_P2.md)
        docs = list(dict.fromkeys(evidence.entry(a).get("source") for a in judged if evidence.entry(a).get("source")))
        ev = (f"OOS 판정: {len(judged)}개 중 탈락 {counts['failed']}, 통과 {counts['passed']}"
              + (f" ({', '.join(docs)})" if docs else ""))
        return {"active": active, "no_alpha": not active, "headline": NO_ALPHA_TEXT if not active else
                f"활성 알파 {len(active)}개: {', '.join(active)}", "criteria": criteria if not active else "",
                "evidence": ev, "evidence_counts": counts,
                "active_evidence": {a: evidence.verdict(a) for a in active},
                "kill_switch": (f"킬 스위치 {'켜짐' if self.s.kill_switch else '꺼짐'} (알파별 CUSUM k={self.s.kill_switch_k:g}σ, "
                                f"-{self.s.kill_switch_h:g}σ (σ = 그 알파 R의 표준편차) / 최근 {self.s.kill_switch_window}건 합 < "
                                f"{self.s.kill_switch_window_r:g}R)")}

    def startup_summary(self) -> str:
        lines = [f"Heartless 시작 — 모드: {'🔴 LIVE' if self.mode == 'live' else '🟢 PAPER'}",
                 f"유니버스 {len(self.universe)}종목: {', '.join(self.universe)}",
                 f"챔피언 파라미터: {self.params.version}", f"챌린저: {', '.join(c.name for c in self.challengers) or '-'}"]
        ls = self.learning_status()
        if ls["active"]:
            lines.append(f"활성 알파: {', '.join(ls['active'])}")
            tags = [f"{a} {evidence.short(a)}" for a in ls["active"]]
            lines.append("알파 증거: " + "; ".join(tags))
        else:
            lines += [NO_ALPHA_TEXT, "거래 시작 조건: " + ls["criteria"]]
        lines += [ls["evidence"], ls["kill_switch"]]
        if not ls["evidence_counts"].get("passed"):
            lines.append(LIVE_NOT_RECOMMENDED_TEXT)
        pf = self.portfolio_status()
        lines.append(f"포트폴리오 할당기 켜짐: 위험 예산 총 {pf['max_total_risk_pct']:g}%·같은 방향 "
                     f"{pf['max_same_direction_risk_pct']:g}%(BTC 베타 가중) (docs/PORTFOLIO.md)"
                     if pf["enabled"] else
                     f"포트폴리오 할당기 꺼짐(출하 기본): 개수 한도 최대 {pf['max_positions']}개·같은 방향 "
                     f"{pf['max_same_direction']}개 안에서 봉 처리 순서대로 진입 (docs/PORTFOLIO.md)")
        if self.mode_notice:
            lines.append(self.mode_notice)
        if self.store.get("paused", False):
            lines.append("⏸ 신규 진입 일시정지 상태로 시작 (재시작 전 /pause 또는 /kill 유지; /resume 로 재개)")
        if self.s.web_enabled:
            lines.append(f"웹 대시보드: http://<host>:{self.s.web_port}/?token={self.web_token}")
        return "\n".join(lines)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        log.info("shutting down")
        await self.market.stop()
        for eng in list(self.engines.values()):
            try:
                await eng.account.stop()
            except Exception:  # noqa: BLE001
                pass
        self.research.shutdown()
        for t in self.tasks:
            t.cancel()
        try:
            # the scheduler writes the paper wallets only every 30 s; trades are written immediately, so without
            # this flush a restart restores a wallet that disagrees with the trades booked since the last tick
            self._persist_paper_accounts()
        except Exception:  # noqa: BLE001
            log.exception("paper account flush failed")
        if self.archive is not None:
            try:
                await self.archive.close()
            except Exception:  # noqa: BLE001
                pass
        await self.rest.close()
        if self._ctx_fill_task is not None:
            self._ctx_fill_task.cancel()
        if self._ctx_store is not None and self._ctx_store is not self.store:
            try:
                self._ctx_store.close()
            except Exception:  # noqa: BLE001
                pass
        self.store.close()

    async def _load_symbols(self) -> None:
        info = await self.rest.exchange_info()
        self.symbols = parse_exchange_info(info)
        log.info("loaded %d symbols", len(self.symbols))

    async def refresh_universe(self, initial: bool = False) -> None:
        try:
            tickers = await self.rest.ticker_24h()
        except Exception as e:  # noqa: BLE001
            log.warning("ticker fetch failed: %s", e)
            if initial:
                raise
            return
        new = select_universe(self.symbols, tickers, self.s.universe_size, self.s.universe_min_quote_volume,
                              self.s.always_include_list)
        if not initial:
            self._keep_members(new, tickers)
        # keep symbols with open positions
        held = {p.symbol for eng in self.engines.values() for p in eng.open_positions()}
        for sym in held:
            if sym not in new:
                new.append(sym)
        added = [s for s in new if s not in self.universe]
        removed = [s for s in self.universe if s not in new]
        self._last_universe = now_ms()
        if initial:
            self.universe = new
            return
        if added or removed:
            self.universe = new
            if added:
                await self._backfill(added)
                self._build_views(added)
                if self.s.archive_backfill:
                    asyncio.create_task(self._backfill_metrics(added))
            for s in removed:
                self.views.pop(s, None)
                self.candles.pop(s, None)
            self._set_context_only(self._context_only_symbols())
            self.market.set_symbols(self.universe, kline_only=self.context_only)
            self._f7_book_symbols()  # F7
            await self.emit("universe", {"message": f"유니버스 갱신: +{','.join(added) or '-'} / -{','.join(removed) or '-'}",
                                         "added": added, "removed": removed, "universe": self.universe})

    def _keep_members(self, new: list[str], tickers: list[dict]) -> None:
        """Universe hysteresis (UNIVERSE_KEEP_FRACTION): current members that fell out of `new` only because their 24h
        volume dipped a little under the entry threshold are kept, best volume first, within the universe size."""
        vol: dict[str, float] = {}
        for t in tickers:
            try:
                vol[t["symbol"]] = float(t.get("quoteVolume", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
        keep_min = self.s.universe_min_quote_volume * UNIVERSE_KEEP_FRACTION
        for sym in sorted((s for s in self.universe if s not in new), key=lambda s: -vol.get(s, 0.0)):
            info = self.symbols.get(sym)
            if len(new) >= self.s.universe_size:
                break
            if info is not None and info.status == "TRADING" and vol.get(sym, 0.0) >= keep_min:
                new.append(sym)

    async def _backfill(self, symbols: list[str]) -> set[str]:
        """Fetch the missing 1m history (and funding) into the store. Returns the symbols whose fetch failed; they
        are remembered in `_backfill_failed` and retried by the scheduler, because a symbol left without history
        would otherwise trade on minutes of data for the whole process lifetime (the stream gap fill only repairs
        holes after the first live bar)."""
        now = now_ms()
        # keep the longer research history in the store (cheap via the archive); live views use history_days, or the
        # longest lookback of the enabled alphas when that is longer
        keep_days = max(self.s.history_days, self.s.data_retention_days if self.s.archive_backfill else self.s.history_days,
                        self._alpha_history_days())
        sem = asyncio.Semaphore(3)
        failed: set[str] = set()

        async def one(sym: str) -> None:
            # a market-context symbol needs the tracker's whole warm-up window stored, traded or not
            since = now - max(keep_days, CONTEXT_HISTORY_DAYS if sym in self.context.symbols else 0) * MS_DAY
            async with sem:
                self._backfilling.add(sym)
                try:
                    lo, hi, n = self.store.candle_range(sym)
                    ranges: list[tuple[int, int]] = []
                    if hi is None or hi < since:
                        ranges.append((since, now))
                    else:
                        if lo - MS_MINUTE > since:  # history before the oldest stored bar is missing (failed/partial fetch)
                            ranges.append((since, lo - MS_MINUTE))
                        ranges.append((hi + MS_MINUTE, now))
                    for start, end in ranges:
                        if self.s.archive_backfill and start <= end and start < now - MS_MINUTE and end - start > 2 * MS_DAY:
                            # bulk history from the public archive first (no API weight); REST only fills the tail
                            try:
                                if self.archive is None:
                                    from heartless.data.archive import BinanceArchive

                                    self.archive = BinanceArchive()
                                arch = await self.archive.klines(sym, start, end, now)
                                arch = [c for c in arch if c.close_time <= now]
                                if arch:
                                    self.store.save_candles(sym, arch)
                                    log.info("archive backfill %s: %d bars", sym, len(arch))
                                    start = max(start, arch[-1].open_time + MS_MINUTE)
                            except Exception as e:  # noqa: BLE001
                                log.info("archive backfill %s unavailable (%s); using REST", sym, e)
                        if start <= end and start < now - MS_MINUTE:
                            rows = await self.rest.klines_range(sym, start, end, "1m")
                            rows = [c for c in rows if c.close_time <= now]
                            if rows:
                                self.store.save_candles(sym, rows)
                                log.info("backfilled %s: %d bars", sym, len(rows))
                    f_lo, f_hi = self.store.funding_range(sym)
                    f_start = since if (f_hi is None or f_hi < since) else f_hi + 1
                    if f_start < now - self._funding_interval_ms(sym):
                        fr = await self.rest.funding_rate_history(sym, start=f_start, end=now)
                        self.store.save_funding(sym, [(int(r["fundingTime"]), float(r["fundingRate"]), float(r.get("markPrice", 0) or 0)) for r in fr])
                except Exception as e:  # noqa: BLE001
                    log.warning("backfill %s failed: %s", sym, e)
                    failed.add(sym)
                finally:
                    self._backfilling.discard(sym)

        await asyncio.gather(*(one(s) for s in symbols))
        self._backfill_failed -= set(symbols)
        self._backfill_failed |= failed
        return failed

    async def _backfill_metrics(self, symbols: list[str]) -> None:
        """Fill the 5-minute positioning metrics of the retention window from the public archive (daily files), in the
        background: the bot otherwise only stores what it polls from its own start on, so the research cycle's
        120-day window evaluated the metrics alphas (positioning, funding_fade) on days of data. Holes of more than two
        days before the oldest / after the newest stored row are fetched; failures are logged and left for the next
        start (never raised: this task runs next to the trading loop)."""
        now = now_ms()
        since = now - max(self.s.history_days, self.s.data_retention_days) * MS_DAY
        for sym in symbols:
            try:
                lo, hi = self.store.metrics_range(sym)
                ranges: list[tuple[int, int]] = []
                if hi is None or hi < since:
                    ranges.append((since, now))
                else:
                    if lo - since > 2 * MS_DAY:
                        ranges.append((since, lo - 5 * MS_MINUTE))
                    if now - hi > 2 * MS_DAY:
                        ranges.append((hi + 5 * MS_MINUTE, now))
                for a, b in ranges:
                    if self.archive is None:
                        from heartless.data.archive import BinanceArchive

                        self.archive = BinanceArchive()
                    rows = await self.archive.metrics(sym, a, b, now)
                    if rows:
                        self.store.save_metrics(sym, rows)
                        log.info("metrics archive backfill %s: %d rows", sym, len(rows))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.info("metrics archive backfill %s unavailable: %s", sym, e)
            await asyncio.sleep(0)

    async def _retry_backfill(self) -> None:
        """Re-run the history fetch for symbols whose startup/universe backfill failed and rebuild their views."""
        self._backfill_failed &= set(self.universe) | set(self.context_only)
        pending = sorted(self._backfill_failed)
        if not pending:
            return
        await self._backfill(pending)
        fixed = [s for s in pending if s not in self._backfill_failed]
        if fixed:
            # a repaired context-only symbol only needs its stored history (the next start warms the tracker from it;
            # this process keeps what it already streamed)
            self._build_views([s for s in fixed if s in self.universe])
            log.info("backfill retry repaired %s", ", ".join(fixed))

    def _build_views(self, symbols: list[str]) -> None:
        since = now_ms() - max(self.s.history_days, self._alpha_history_days()) * MS_DAY
        for sym in symbols:
            rows = self.store.load_candles(sym, start=since)
            ca = CandleArrays("1m", capacity=len(rows) + 4096)
            ca.extend(rows)
            self.candles[sym] = ca
            view = MarketView(sym, incremental=self.s.incremental_features)  # F7: incremental live frames
            view.rebuild(ca, live=True)
            if ca.n:
                view.seek(int(ca.close_time[ca.n - 1]))
            self.views[sym] = view
            self.regimes[sym] = detect_regime(view)

    async def _create_engines(self) -> None:
        # paper champion (always on: it is the learning benchmark)
        pa = PaperAccount("paper", self.s.paper_initial_balance, self.s.taker_fee, self.s.maker_fee)
        pa.set_symbols(self.symbols)
        saved = self.store.get("paper.account")
        if saved:
            pa.wallet = float(saved.get("wallet", pa.wallet))
            pa.initial_balance = float(saved.get("initial", pa.initial_balance))
            pa.realized = float(saved.get("realized", 0.0))
            pa.fees_paid = float(saved.get("fees", 0.0))
            pa.funding_paid = float(saved.get("funding", 0.0))
        self.paper_accounts["paper"] = pa
        watch = self._watch_fills(pa, "paper")  # before the engine: fill handlers run in registration order
        self.engines["paper"] = TradingEngine("paper", pa, self.params, self.s, self.symbols, self.store, self.bus,
                                              self.bandit, notify=(self.mode != "live"), meta=self.meta)
        watch["engine"] = self.engines["paper"]
        self._apply_persisted_pause(self.engines["paper"])
        await self._restore_paper_positions(self.engines["paper"])
        await self.engines["paper"].start()
        await self._rearm_paper_positions(self.engines["paper"])
        # challengers
        for slot in self.research.load_challengers():
            await self._start_challenger(slot, restore=True)
        # live
        if self.mode == "live":
            await self._start_live()

    def _apply_persisted_pause(self, eng: TradingEngine) -> None:
        """Honour the owner's /pause or /kill across restarts. pause()/resume() persist the flag, but every engine
        starts with a fresh RiskState, so it is re-applied to each engine as it is created: the paper champion,
        challengers and a live engine started later via /mode live alike (the deployments auto-restart the process,
        so without this a kill switch would silently re-arm entries on the next crash or reboot)."""
        if self.store.get("paused", False):
            eng.risk.state.paused = True
            eng.risk.state.halt_reason = (self.store.get("halt_reason") or eng.risk.state.halt_reason
                                          or "재시작 전 일시정지 상태 복원 (/resume 로 재개)")

    async def _restore_paper_positions(self, eng: TradingEngine) -> None:
        """Recreate open paper positions inside the simulated account after a restart."""
        acc: PaperAccount = eng.account  # type: ignore[assignment]
        for p in self.store.load_open_positions(eng.name):
            # a CLOSING row is a close interrupted mid-flight: its quantity is still held in the simulated account
            if p.status in (PositionStatus.OPEN, PositionStatus.CLOSING) and p.qty > 0:
                from heartless.exchange.paper import PaperPosition

                acc.positions[p.symbol] = PaperPosition(qty=p.qty * p.side.sign, entry=p.entry_price, leverage=p.leverage)
                acc.orders.clear()

    async def _rearm_paper_positions(self, eng: TradingEngine) -> None:
        """After `eng.start()` loaded a paper engine's persisted positions, give them simulated brackets again: the
        fresh PaperAccount holds no algo orders and the rows still carry the previous process's algo ids, so without
        this take-profits would never fill and stops would degrade to the on_tick software backstop (biasing the
        paper record that drives promotion and the go-live offer). A CLOSING row (close interrupted mid-flight) is
        handed back as OPEN so it is managed and protected again instead of blocking its symbol forever."""
        acc = eng.account
        if not acc.is_paper:
            return
        for pos in list(eng.positions.values()):
            if pos.status is PositionStatus.CLOSING and pos.qty > 0:
                pos.status = PositionStatus.OPEN
            if pos.status is not PositionStatus.OPEN or pos.qty <= 0 or pos.symbol not in eng.symbols:
                continue
            pp = getattr(acc, "positions", {}).get(pos.symbol)
            if pp is None or abs(pp.qty) <= 0:
                continue  # not mirrored in the simulator: _paper_reconcile finalizes it
            # drop the dead process's ids WITHOUT cancelling them: the fresh simulator numbers its algos from 1
            # again, so a stale id can collide with (and would cancel) a bracket just placed for another symbol
            pos.sl_algo_id = pos.tp_algo_id = ""
            pos.extra.pop("tp1_algo_id", None)
            await eng._place_brackets(pos)  # fresh SL / TP1 / TP (TP1 is skipped when it already filled)
            eng._save(pos)

    async def _start_live(self) -> None:
        if "live" in self.engines:
            return
        # Build everything into locals and publish only once BOTH starts succeeded. A failure half-way (REST/auth
        # error in LiveAccount.start(), reconcile in eng.start()) must leave neither an account nor an engine behind:
        # otherwise a retry reports success without a live engine, or an unreconciled live engine trades real money
        # while mode/status still say paper.
        # --- F7: the account reads LiveAccount.start and the reconcile make, fetched at once and timed ----------
        await self._f7_prefetch_account()
        # --- end F7 ---
        acc = LiveAccount(self.rest, "live", self.s.taker_fee)
        acc.set_symbols(self.symbols)
        watch = self._watch_fills(acc, "live")
        eng = TradingEngine("live", acc, self.params, self.s, self.symbols, self.store, self.bus, self.bandit, notify=True,
                            meta=self.meta)
        watch["engine"] = eng
        self._apply_persisted_pause(eng)
        try:
            await acc.start()
            await eng.start()
        except BaseException:
            try:
                await acc.stop()
            except Exception:  # noqa: BLE001
                pass
            raise
        self.live_account = acc
        self.engines["live"] = eng
        # F7: keep a pooled REST connection warm for the order path (and the clock offset fresh); one task per process
        warm = getattr(self, "_f7_warm", None)
        if hasattr(self.rest, "keep_warm") and (warm is None or warm.done()):
            self._f7_warm = asyncio.create_task(self.rest.keep_warm(), name="rest-keep-warm")
            self.tasks.append(self._f7_warm)
        # the paper champion trades the same params on the same bars: while live is attached its outcomes alone are
        # booked into the shared bandit (weight 1.5), the paper duplicates are dropped
        self.bandit.live_attached = True
        if "paper" in self.engines:
            self.engines["paper"].notify = False
        log.info("live engine started")

    async def _start_challenger(self, slot: ChallengerSlot, restore: bool = False) -> None:
        pa = PaperAccount(slot.name, self.s.paper_initial_balance, self.s.taker_fee, self.s.maker_fee)
        pa.set_symbols(self.symbols)
        if restore:
            saved = self.store.get(f"{slot.name}.account")
            if saved:
                pa.wallet = float(saved.get("wallet", pa.wallet))
                pa.initial_balance = float(saved.get("initial", pa.initial_balance))
                pa.realized = float(saved.get("realized", 0.0))
                pa.fees_paid = float(saved.get("fees", 0.0))
                pa.funding_paid = float(saved.get("funding", 0.0))
        else:
            # a fresh challenger reuses a slot name: it must not inherit the retired one's record or its risk halts. The
            # finished run's trades are archived under "<slot>@<its params version>" (kept for the record), not deleted
            self.store.delete_positions(slot.name)
            archived = self.store.archive_trades(slot.name)
            if archived:
                log.info("archived %d trades of the previous %s run", archived, slot.name)
            self.store.delete_equity(slot.name)
            self.store.set(f"{slot.name}.risk", None)
        self.paper_accounts[slot.name] = pa
        watch = self._watch_fills(pa, slot.name, since=slot.started)
        # challengers consult the shared meta model (fair comparison with the champion) but do not train it
        # its Thompson-sampling RNG is seeded from the slot and its start (not the version id, which carries wall time
        # and random bits): a replay of the same window then reproduces the challenger's trades exactly
        eng = TradingEngine(slot.name, pa, slot.params, self.s, self.symbols, self.store, self.bus,
                            AlphaBandit(list(slot.params.alphas), seed=f"{slot.name}:{int(slot.started)}", store=None),
                            notify=False, meta=self.meta, meta_learn=False)
        watch["engine"] = eng
        self._apply_persisted_pause(eng)
        if restore:
            await self._restore_paper_positions(eng)
        self.engines[slot.name] = eng
        await eng.start()
        if restore:
            await self._rearm_paper_positions(eng)
        if slot not in self.challengers:
            self.challengers.append(slot)

    # --- execution quality ---------------------------------------------------------------------------
    def _watch_fills(self, acc, name: str, since: int | None = None) -> dict:
        """Execution-quality monitor for one engine: every entry / exit fill is compared with the reference price the
        engine decided on (slippage in bps, positive = cost) and its fee with the fee model.

        The reference of a market / limit order is recorded when the engine sends it (the touch on its side, or the
        limit price), keyed by client id: a live fill can reach the account's handlers after the engine already booked
        the REST result (entry price overwritten, a closed position gone). Stops and take-profits are measured against
        their trigger. The observer is registered on the account BEFORE the engine, so it runs first and sees the
        position as the engine left it. Returns a holder the caller fills with the engine once built."""
        holder: dict = {}
        self._record_order_refs(acc, holder)
        self._f7_time_orders(acc, name)  # F7: order sent -> ack, bar close -> sent
        rows = self.store.load_fill_quality(name, self.s.exec_quality_window)
        if since is not None:
            rows = [r for r in rows if (r.get("ts") or 0) >= since]
        self.fill_rows[name] = deque(rows, maxlen=max(1, self.s.exec_quality_window))

        async def observe(fill) -> None:
            try:  # never let the monitor break the fill dispatch to the engine
                eng = holder.get("engine")
                if eng is None or fill.kind == "FUNDING" or fill.qty <= 0 or fill.price <= 0:
                    return
                row = self._fill_row(name, eng.positions.get(fill.symbol), fill)
                if row is not None:
                    self.fill_rows.setdefault(name, deque(maxlen=max(1, self.s.exec_quality_window))).append(row)
                    self.store.save_fill_quality(row)
            except Exception:  # noqa: BLE001
                log.debug("fill-quality observer failed for %s", name, exc_info=True)

        acc.on_fill(observe)
        return holder

    def _record_order_refs(self, acc, holder: dict) -> None:
        """Wrap the account's market / limit order calls to remember each order's decision reference by client id."""
        market, limit = acc.market_order, acc.limit_order

        def remember(client_id: str, symbol: str, ref: float, reduce_only: bool) -> None:
            if not client_id or not ref or ref <= 0:
                return
            eng = holder.get("engine")
            pos = eng.positions.get(symbol) if eng is not None else None
            self._order_refs[client_id] = (float(ref), "CLOSE" if reduce_only else "ENTRY", pos.id if pos else "")
            while len(self._order_refs) > 5000:
                self._order_refs.pop(next(iter(self._order_refs)))

        async def market_order(symbol: str, side: str, qty: float, reduce_only: bool = False, client_id: str = ""):
            t = self.tickers.get(symbol)
            touch = ((t.ask if side == "BUY" else t.bid) or t.mark or t.last) if t else 0.0
            remember(client_id, symbol, touch, reduce_only)
            return await market(symbol, side, qty, reduce_only=reduce_only, client_id=client_id)

        async def limit_order(symbol: str, side: str, qty: float, price: float, post_only: bool = True,
                              reduce_only: bool = False, client_id: str = ""):
            remember(client_id, symbol, price, reduce_only)
            return await limit(symbol, side, qty, price, post_only=post_only, reduce_only=reduce_only,
                               client_id=client_id)

        acc.market_order, acc.limit_order = market_order, limit_order

    def _fill_row(self, name: str, pos, fill) -> dict | None:
        """Reference price per fill: a market order -> the touch on its side when the engine sent it (an entry: the
        price it sized at; a close: the price it decided to exit at); a post-only entry -> its limit price; a stop ->
        its trigger; a take-profit -> the nearer of TP1 / TP. Without a recorded order (e.g. after a restart) an entry
        falls back to the position's decision price and a close to the touch when the fill is processed."""
        known = self._order_refs.get(fill.client_id) if fill.client_id else None
        if fill.kind not in ("SL", "TP") and known is not None:
            ref, kind, pid = known
            return self._fill_quality_row(name, pid or (pos.id if pos is not None else ""), kind, ref, fill)
        if pos is None or pos.status is PositionStatus.CLOSED:
            return None
        opens = fill.order_side == pos.side.order_side and not fill.reduce_only
        kind = "ENTRY" if opens else (fill.kind if fill.kind in ("SL", "TP") else "CLOSE")
        if opens:
            if pos.entry_style == "LIMIT" and pos.limit_price:
                ref = float(pos.limit_price)
            elif pos.filled_qty <= 0:
                ref = float(pos.entry_price)
            else:
                ref = self._entry_refs.get(pos.id, float(pos.entry_price))
            if len(self._entry_refs) > 2000:
                self._entry_refs.clear()
            self._entry_refs.setdefault(pos.id, ref)
        elif kind == "SL":
            ref = float(pos.stop)
        elif kind == "TP":
            cands = [x for x in ((pos.tp1 if not pos.tp1_done else None), pos.take_profit) if x]
            ref = float(min(cands, key=lambda x: abs(x - fill.price))) if cands else 0.0
        else:
            t = self.tickers.get(fill.symbol)
            ref = float(((t.bid if fill.order_side == "SELL" else t.ask) if t else 0.0) or pos.extra.get("last_mark") or 0.0)
        return self._fill_quality_row(name, pos.id, kind, ref, fill)

    def _fill_quality_row(self, name: str, position_id: str, kind: str, ref: float, fill) -> dict | None:
        if not ref or ref <= 0:
            return None
        notional = fill.qty * fill.price
        model_fee = notional * (self.s.maker_fee if fill.maker else self.s.taker_fee)
        return {"engine": name, "ts": int(fill.ts or now_ms()), "symbol": fill.symbol, "position_id": position_id,
                "kind": kind, "side": fill.order_side, "maker": bool(fill.maker), "qty": fill.qty, "price": fill.price,
                "ref": float(ref), "notional": notional, "slip_bps": slippage_bps(fill.order_side, fill.price, ref),
                "fee": float(fill.fee or 0.0), "model_fee": model_fee}

    def exec_quality(self) -> dict[str, dict]:
        """Rolling execution-quality aggregates per engine (engines without fills omitted)."""
        out = {}
        for name, rows in self.fill_rows.items():
            if rows and (name in self.engines or name in ("paper", "live")):
                out[name] = exec_quality(list(rows), self.s.backtest_slippage_bps, self.s.exec_quality_min_fills)
        return out

    # --- challenger / champion management ------------------------------------------------------
    def challenger_trial_ms(self, slot: ChallengerSlot) -> int:
        """How long a challenger keeps its slot before a new candidate may replace it (unless it is losing): its whole
        trial, i.e. its last promotion checkpoint (research.trial_days). Research runs every few hours; evicting
        younger slots would churn challengers before the late checkpoints, where most of the power is."""
        return self.research.trial_days(slot) * MS_DAY

    async def install_challenger(self, params: StrategyParams, source: str) -> ChallengerSlot | None:
        """Place a new challenger into a free slot, or replace the weakest one that has had its fair trial. A candidate
        identical to a running challenger (same alpha values, switches and ensemble) is not installed: the research
        cycle proposes the same re-enable every few hours, and a copy would restart a trial already under way."""
        base = self.params
        diff, base_values = R.params_diff(base, params)
        for c in self.challengers:
            same = c.params.alphas == params.alphas and c.params.enabled == params.enabled and \
                c.params.ensemble == params.ensemble
            if same or (any(diff.values()) and self.research.slot_diff(c) == diff):
                log.info("challenger candidate from %s is identical to %s (%s); not installed", source, c.name,
                         c.params.version)
                return None
        tried = self._recently_tried(diff)
        if tried is not None:
            log.info("challenger candidate from %s repeats the change of %s, judged and retired %s (%s); not installed "
                     "for CHALLENGER_RETRY_DAYS=%d", source, tried.get("version"), fmt_ts(tried.get("at", 0)),
                     tried.get("rule"), self.s.challenger_retry_days)
            return None
        names = [f"challenger-{i + 1}" for i in range(self.s.challengers)]
        used = {c.name: c for c in self.challengers}
        free = [n for n in names if n not in used]
        if free:
            name = free[0]
        else:
            # replace the worst-performing challenger (by net pnl since start) among those that have had a fair
            # trial (challenger_trial_ms), or that are losing on a promotion-sized sample
            now = now_ms()

            def trades(slot: ChallengerSlot) -> list[dict]:
                return self.store.load_trades(engine=slot.name, since=slot.started)

            def score(slot: ChallengerSlot) -> float:
                st = summarize(trades(slot))
                return st["net"] if st["n"] else -1e-9

            def losing(slot: ChallengerSlot) -> bool:
                st = r_stats(trades(slot))
                return st["n"] >= self.s.promotion_min_trades and st["avg_r"] <= 0

            eligible = [c for c in self.challengers if now - c.started >= self.challenger_trial_ms(c) or losing(c)]
            if not eligible:
                log.info("no challenger slot for %s candidate: all %d challengers are still in their trial period",
                         source, len(self.challengers))
                return None
            victim = min(eligible, key=score)
            await self.retire_challenger(victim, reason="새 후보로 교체", silent=True)
            name = victim.name
        slot = ChallengerSlot(name=name, params=params, started=now_ms(), source=source, base=base.version,
                              base_created=int(base.created or 0), diff=diff, base_values=base_values,
                              base_params=base.to_dict())
        self.store.save_params_version(params.version, params.created, "challenger", source, params.note, params.to_dict(),
                                       {"slot": name, "started": slot.started, "base": slot.base,
                                        "base_created": slot.base_created, "diff": diff, "base_values": base_values,
                                        "base_params": slot.base_params})
        await self._start_challenger(slot)
        return slot

    # a challenger retirement that judged its change (not a replacement, not a conflict with a newer champion)
    TRIED_KEY = "challenger.tried"

    @staticmethod
    def _diff_key(diff: dict) -> str:
        return json.dumps(diff, sort_keys=True, default=str)

    def _recently_tried(self, diff: dict) -> dict | None:
        """The record of a challenger with the same change that was judged and retired within CHALLENGER_RETRY_DAYS
        (None if there is none). A trial that ended without promotion, a retirement on the statistics or a kill-switch
        trip all count; every fresh trial of the same change would be another chance for noise to clear the bar."""
        if not any(diff.values()) or self.s.challenger_retry_days <= 0:
            return None
        key = self._diff_key(diff)
        since = now_ms() - self.s.challenger_retry_days * MS_DAY
        return next((r for r in reversed(self.store.get(self.TRIED_KEY, []) or [])
                     if r.get("diff") == key and int(r.get("at", 0)) >= since), None)

    def _remember_tried(self, slot: ChallengerSlot, decision: dict, reason: str) -> None:
        diff = self.research.slot_diff(slot)
        if not any(diff.values()):
            return
        if decision.get("kind") == "kill_switch" and decision.get("alpha") not in R.diff_alphas(diff):
            # the trip judged an alpha the challenger inherited from its base (it trades the base's settings of every
            # alpha its change does not touch), not the change: a later trial of the change is no second look at it
            return
        since = now_ms() - max(1, self.s.challenger_retry_days) * MS_DAY
        rows = [r for r in (self.store.get(self.TRIED_KEY, []) or []) if int(r.get("at", 0)) >= since]
        rows.append({"diff": self._diff_key(diff), "at": now_ms(), "version": slot.params.version,
                     "rule": decision.get("rule") or decision.get("kind") or reason.split("\n")[0][:80]})
        self.store.set(self.TRIED_KEY, rows[-200:])

    async def retire_challenger(self, slot: ChallengerSlot, reason: str, silent: bool = False,
                                decision: dict | None = None) -> None:
        eng = self.engines.pop(slot.name, None)
        if eng:
            await eng.close_all("챌린저 종료")
        self.paper_accounts.pop(slot.name, None)
        if slot in self.challengers:
            self.challengers.remove(slot)
        self.research.last_decisions.pop(slot.name, None)
        if decision is not None:
            # the decision's inputs stay with the retired version (params_versions.metrics.decisions + its note)
            self.store.record_params_decision(slot.params.version, {**decision, "reason": reason}, role="retired",
                                              note=f"{slot.params.note} | {self._decision_note(decision)}")
            if decision.get("verdict") != "promote" and not decision.get("blockers"):
                self._remember_tried(slot, decision, reason)
        else:
            self.store.set_params_role(slot.params.version, "retired")
        if not silent:
            await self.emit("challenger_retired", {"message": f"챌린저 {slot.name} 종료: {reason}", "notify": True,
                                                   "slot": slot.name, "version": slot.params.version})

    @staticmethod
    def _decision_note(d: dict) -> str:
        """One-line English summary of a promotion / retirement / kill-switch decision for a version note."""
        if d.get("blockers"):
            return f"not promoted ({d.get('rule')})"
        if d.get("kind") == "kill_switch":
            return (f"kill switch {d.get('alpha')} ({d.get('rule')}): n={d.get('n', 0)} sum {d.get('sum_r', 0):+.2f}R "
                    f"CUSUM {d.get('cusum', 0):+.2f}sd (sd {d.get('scale', 1.0):.2f}R) last {d.get('window_n', 0)} "
                    f"{d.get('window_sum', 0):+.2f}R")
        ch = d.get("challenger") or {}
        return (f"{d.get('verdict')} via the {d.get('path')} path ({d.get('rule')}): n={ch.get('n', 0)} "
                f"days={ch.get('days', 0)} avgR {ch.get('avg_r', 0):+.3f} t_day {ch.get('t_day', 0):+.2f}"
                + (f" paired t {d['paired']['t']:+.2f}" if d.get("paired") else "")
                + f" champion n={(d.get('champion') or {}).get('n', 0)}")

    async def promote(self, slot: ChallengerSlot, decision: dict) -> None:
        """Make a challenger's CHANGE part of the champion: its diff relative to its base is applied onto the current
        champion, so champion changes since the base (another promotion, a kill-switch trip) are kept. `decision` is
        the verdict of research.judge_challenger: its inputs are stored with the new champion version (note + metrics)
        and shown in the Telegram message. A change that conflicts with the current champion is not applied (the
        challenger is retired with the reason; evaluate_challengers normally catches this first)."""
        old = self.params
        blockers = self.research.promotion_blockers(slot)
        if blockers:
            d = {**decision, "verdict": "retire", "blockers": blockers,
                 "rule": "change conflicts with the current champion: " + "; ".join(blockers)}
            await self.retire_challenger(slot, reason="승격 불가 (기준 챔피언 이후의 변경과 충돌): " + "; ".join(blockers),
                                         decision=d)
            return
        diff = self.research.slot_diff(slot)
        path = decision.get("path", "?")
        ch = decision.get("challenger", {})
        rebased = bool(slot.base) and slot.base != old.version
        note = (f"promoted from {slot.name} ({slot.source}) via the {path} path at the {decision.get('checkpoint')}-day "
                f"checkpoint: n={ch.get('n', 0)} days={ch.get('days', 0)} avgR {ch.get('avg_r', 0):+.3f} "
                f"t_day {ch.get('t_day', 0):+.2f}"
                + (f" paired t {decision['paired']['t']:+.2f}" if "paired" in decision else "")
                + f" champion n={(decision.get('champion') or {}).get('n', 0)}; its change applied onto {old.version}"
                + (f" (base {slot.base})" if rebased else "") + f" (was {old.version})")
        new = R.apply_diff(old, diff, version=short_id("v"), note=note)
        now = now_ms()
        self.store.set_params_role(old.version, "retired")
        self.store.save_params_version(new.version, new.created, "champion", slot.source, new.note, new.to_dict(),
                                       {"decisions": [decision], "challenger": slot.params.version, "base": slot.base,
                                        "diff": diff})
        self.params = new
        for name in ("live", "paper"):
            if name in self.engines:
                self.engines[name].set_params(new)
        self.killswitch.on_new_champion(old, new, now)
        await self.retire_challenger(slot, reason="챔피언으로 승격", silent=True, decision=decision)
        switched_on = [a for a, on in new.enabled.items() if on and not old.enabled.get(a, False)]
        switched_off = [a for a, on in old.enabled.items() if on and not new.enabled.get(a, False)]
        tuned = [a for a in (diff.get("alphas") or {}) if a not in switched_on]
        log.warning("promotion: %s (%s) -> champion %s via the %s path; change applied onto %s (base %s); on: %s off: %s "
                    "tuned: %s", slot.name, slot.params.version, new.version, path, old.version, slot.base or "-",
                    ", ".join(switched_on) or "-", ", ".join(switched_off) or "-", ", ".join(tuned) or "-")
        msg = (f"🏆 파라미터 승격: {slot.name} → 챔피언 {new.version}\n"
               f"변경 알파: {slot.source}" + (f" · 켜짐 {', '.join(switched_on)}" if switched_on else "")
               + (f" · 꺼짐 {', '.join(switched_off)}" if switched_off else "")
               + (f" · 파라미터 {', '.join(tuned)}" if tuned else "")
               + (f"\n기준 챔피언 {slot.base} 이후 바뀐 챔피언 {old.version}에 변경분만 적용" if rebased else "")
               + "\n" + R.describe_decision(decision))
        await self.emit("promotion", {"message": msg, "version": new.version, "path": path})
        await self.research.check_conflicts()

    async def kill_alpha(self, alpha: str, decision: dict) -> None:
        """Kill switch on a champion engine: a new champion version with `alpha` switched off, effective at once for
        the paper and live engines (open positions keep their stops and exits; no new entries from the alpha)."""
        old = self.params
        if not old.enabled.get(alpha, False):
            return
        new = old.clone(version=short_id("v"),
                        note=(f"kill switch: {alpha} off ({decision.get('engine')}: {decision.get('rule')}; "
                              f"n={decision.get('n', 0)} sum {decision.get('sum_r', 0):+.2f}R CUSUM "
                              f"{decision.get('cusum', 0):+.2f}sd (sd {decision.get('scale', 1.0):.2f}R) last "
                              f"{decision.get('window_n', 0)} {decision.get('window_sum', 0):+.2f}R; was {old.version})"))
        new.enabled[alpha] = False
        self.store.set_params_role(old.version, "retired")
        self.store.save_params_version(new.version, new.created, "champion", "kill-switch", new.note, new.to_dict(),
                                       {"decisions": [decision]})
        self.params = new
        for name in ("live", "paper"):
            if name in self.engines:
                self.engines[name].set_params(new)
        self.killswitch.sync_champion(new)
        log.warning("kill switch: %s switched off in the champion -> %s", alpha, new.version)
        left = self.enabled_alphas()
        msg = (f"🛑 알파 킬 스위치: {alpha} 끔 → 챔피언 {new.version}\n{R.describe_kill(decision)}\n"
               f"다시 켜려면 챌린저 승격 경로를 거쳐야 합니다. 남은 활성 알파: {', '.join(left) if left else '없음'}")
        await self.emit("kill_switch", {"message": msg, "alpha": alpha, "version": new.version})
        await self.research.check_conflicts()  # a challenger tuning the alpha can no longer be promoted

    # --- market data handlers ------------------------------------------------------------------
    async def _on_kline(self, data: dict) -> None:
        k = data.get("k", {})
        if not k.get("x"):
            return
        sym, c = data["s"], Candle.from_ws(k)
        # --- F7: latency budget (bar close -> received) ---
        recv = now_ms()
        self._f7_recv[sym] = recv
        perf.LATENCY.record("app", "close_to_recv", recv - perf.exchange_to_local(c.close_time))
        # --- end F7 ---
        # into the market context at once, not through the worker queue: the worker holds an hour-close bar until
        # every context symbol's bar closing at that hour has arrived (_ensure_context)
        self._context_bar(sym, c)
        self._kline_queue.put_nowait((sym, c))

    async def _on_book(self, data: dict) -> None:
        sym = data["s"]
        t = self.tickers.setdefault(sym, Ticker(sym))
        t.bid, t.ask = float(data["b"]), float(data["a"])
        t.ts = int(data.get("E", now_ms()))
        self.book_qty[sym] = (float(data.get("B", 0)), float(data.get("A", 0)))
        await self._tick(sym, t)

    async def _on_mark(self, items: list[dict]) -> None:
        for d in items:
            sym = d.get("s")
            if sym not in self.candles:
                continue
            t = self.tickers.setdefault(sym, Ticker(sym))
            t.mark = float(d.get("p", 0) or 0)
            t.index = float(d.get("i", 0) or 0)
            t.funding_rate = float(d.get("r", 0) or 0)
            t.next_funding_time = int(d.get("T", 0) or 0)
            if t.next_funding_time:
                self.funding_next[sym] = t.next_funding_time
            t.ts = int(d.get("E", now_ms()))
            await self._tick(sym, t)

    async def _tick(self, sym: str, t: Ticker) -> None:
        mark = t.mark or t.mid
        if not mark:
            return
        self._roll_day(now_ms())  # before a stop filled by this tick can book a trade into the previous day
        # snapshots: a live close inside on_tick yields to the loop, during which the scheduler may retire/install
        # a challenger or switch modes (dict mutation mid-iteration would tear down the market stream)
        for acc in list(self.paper_accounts.values()):
            await acc.on_ticker(t)
        for eng in list(self.engines.values()):
            pos = eng.positions.get(sym)
            if pos is not None:
                pos.extra["last_mark"] = mark
                try:
                    await eng.on_tick(sym, mark, t.bid, t.ask)
                except Exception:  # noqa: BLE001
                    # one engine's failure (a paper simulator / store error) must never skip the software backstop of
                    # the engines after it in the loop -- the live engine is created last
                    log.exception("engine %s on_tick failed for %s", eng.name, sym)

    async def _kline_worker(self) -> None:
        while True:
            sym, c = await self._kline_queue.get()
            try:
                await self._process_bar(sym, c)
            except Exception:  # noqa: BLE001
                log.exception("bar processing failed for %s", sym)

    async def _process_bar(self, sym: str, c: Candle) -> None:
        self._roll_day(now_ms())
        ca = self.candles.get(sym)
        if ca is None:
            if sym in self.context_only:
                await self._process_context_bar(sym, c)
            return
        # P3: a bar of a later close while an older close's batch is still open (a straggler): admit that batch first,
        # before this bar moves the symbol's view (a queued candidate of it is planned on the bar it was decided on)
        await self._entry_batch_roll(c.close_time)
        # gap detection: fetch missing bars if the stream skipped some
        missing: list[Candle] = []
        if ca.n and c.open_time - int(ca.open_time[ca.n - 1]) > MS_MINUTE and sym not in self._backfilling:
            try:
                missing = await self.rest.klines_range(sym, int(ca.open_time[ca.n - 1]) + MS_MINUTE, c.open_time - 1, "1m") or []
                if missing:
                    ca.extend(missing)
                    self.store.save_candles(sym, missing)
            except Exception as e:  # noqa: BLE001
                log.warning("gap fill %s failed: %s", sym, e)
        for b in (*missing, c):  # c again: callers that bypass _on_kline (a repeated bar replaces itself)
            self._context_bar(sym, b)
        ca.append(c)
        if c.open_time % MS_DAY == 0:
            # once a day per symbol, here and not in the scheduler: the feature rebuild thread reads `ca` and only this
            # worker mutates it, between rebuilds
            self._trim_history(ca, c.open_time)
        self.store.save_candles(sym, [c])
        self.last_bar_ts[sym] = c.close_time
        view = self.views.get(sym)
        if view is None:
            return
        # rebuild every timeframe whose period ended on ANY bar that entered the array in this call: when the
        # :04/:14/:59 bar arrived through the gap fill the 5m/15m/1h frame would otherwise stay stale until the
        # next boundary (up to an hour of regime/HTF features computed on the previous period)
        tfs = ["1m"]
        for b in (*missing, c):
            m = b.open_time // MS_MINUTE
            if (m + 1) % 5 == 0 and "5m" not in tfs:
                tfs.append("5m")
            if (m + 1) % 15 == 0 and "15m" not in tfs:
                tfs.append("15m")
            if m % 60 == 59 and "1h" not in tfs:
                tfs.append("1h")
        # --- F7: incremental feature frames (INCREMENTAL_FEATURES): an append on the loop, a (re)build off it ------
        t_feat = time.perf_counter()
        if view.live is not None:
            if view.advance_cheap(ca):
                view.advance(ca)
            else:
                await asyncio.to_thread(view.advance, ca)
        else:
            await asyncio.to_thread(self._rebuild_view, view, ca, tfs)
        perf.LATENCY.record("app", "feature_update", (time.perf_counter() - t_feat) * 1e3)
        # --- end F7 ---
        view.seek(c.close_time)
        self._f7_features_ready(sym)  # F7: latency budget
        # a 5m close inside the gap is not flagged by seek() (it closed more than a minute ago): refresh explicitly
        if missing or view.closed("5m") or sym not in self.regimes:
            self.regimes[sym] = detect_regime(view)
        await self._ensure_context(c.close_time)  # the hour's market snapshot exists before any engine decides
        ctx = self._context(sym, c.close_time)
        self._f7_bar[sym] = c.close_time  # F7: orders sent while deciding on this bar count against its close
        for eng in list(self.engines.values()):
            try:
                await eng.on_bar(view, ctx)
            except Exception:  # noqa: BLE001
                log.exception("engine %s on_bar failed for %s", eng.name, sym)
            perf.LATENCY.record(eng.name, "close_to_decision", now_ms() - perf.exchange_to_local(c.close_time))  # F7
        self._f7_bar.pop(sym, None)  # F7
        await self._entry_batch_done(sym, c.close_time)
        if sym == "BTCUSDT":
            await self._btc_shock_check(view)

    def _trim_history(self, ca: CandleArrays, now: int) -> None:
        """Drop 1m bars older than max(HISTORY_DAYS, LIVE_HISTORY_DAYS, the enabled alphas' lookback) days, cut at a
        UTC midnight so every 5m/15m/1h/4h/1d group that remains is whole."""
        keep = max(self.s.history_days, LIVE_HISTORY_DAYS, self._alpha_history_days()) * MS_DAY
        ca.trim_before(now - now % MS_DAY - keep)

    # --- history depth for long-lookback alphas (C2-A) ------------------------------------------------------------
    def _alpha_history_days(self) -> int:
        """Days of 1m history the champion's and the challengers' enabled alphas need: the longest Alpha.lookback_days
        plus the lab's warm-up margin (0 when no enabled alpha declares one: HISTORY_DAYS rules as before)."""
        from heartless.learning.lab import LOOKBACK_MARGIN_MS
        from heartless.strategy.base import required_lookback_days

        days = required_lookback_days([getattr(self, "params", None)]
                                      + [c.params for c in getattr(self, "challengers", None) or []])
        return math.ceil(days + LOOKBACK_MARGIN_MS / MS_DAY) if days > 0 else 0

    def _history_depth_tick(self, now: int) -> None:
        """Hourly from the scheduler: a newly enabled long-lookback alpha (promotion, a new challenger, a restored one at
        start-up) gets its history in the background (backfill into the store, then the views are reloaded). A need
        deeper than the one last handled runs at the next tick instead of up to an hour later, so a new challenger's
        daily alpha does not decide on a short history in between."""
        if self._depth_task is not None and not self._depth_task.done():
            return
        if now - self._last_depth_check < MS_HOUR and self._alpha_history_days() <= self._depth_need:
            return
        self._last_depth_check = now
        self._depth_task = asyncio.create_task(self._ensure_history_depth(), name="history-depth")

    async def _ensure_history_depth(self) -> None:
        need = self._alpha_history_days()
        if self._depth_tried is None or need < self._depth_need:
            self._depth_tried = {}  # a shorter need lets the daily trim shrink the arrays: a later rise re-checks
        self._depth_need = need
        if need <= self.s.history_days:
            return
        now = now_ms()
        short = [s for s in self.universe if s in self.candles and need > self._depth_tried.get(s, 0)
                 and (not self.candles[s].n or int(self.candles[s].open_time[0]) > now - (need - 1) * MS_DAY)]
        if not short:
            return
        for s in short:
            self._depth_tried[s] = need
        log.info("history depth %d days for long-lookback alphas: loading %s", need, ", ".join(short))
        await self._backfill(short)
        self._build_views([s for s in short if s in self.universe and s not in self._backfill_failed])

    @staticmethod
    def _rebuild_view(view: MarketView, ca: CandleArrays, tfs: list[str]) -> None:
        from heartless.data.candles import resample_tail
        from heartless.data.features import LIVE_WINDOW, FeatureFrame

        for tf in tfs:
            # only the 1m tail the frame's bars come from: the same bars at a cost that does not grow with the history
            # a long-lookback alpha keeps in memory
            bars = ca.tail(LIVE_WINDOW[tf]) if tf == "1m" else resample_tail(ca, tf, LIVE_WINDOW[tf])
            view.frames[tf] = FeatureFrame(tf, bars)
        view.base = ca

    # --- F7: speed and optimization (docs/PERFORMANCE.md) --------------------------------------------------------------
    async def _f7_startup_checks(self) -> dict[str, float]:
        """The environment checks of the start-up at once instead of one after another: the server time (clock offset),
        exchangeInfo (symbols), and premiumIndex, fundingInfo and the 24h tickers prefetched for the funding schedule
        and the universe selection that read them next. Each is timed; the total is logged and kept in startup_timings.
        Failures surface as before: the time sync and exchangeInfo raise, the funding schedule only logs, a ticker
        failure is raised by the universe selection's own request."""
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        async def timed(name: str, coro):
            t = time.perf_counter()
            try:
                return await coro
            finally:
                timings[name] = round((time.perf_counter() - t) * 1e3, 1)

        res = await asyncio.gather(
            timed("time", self.rest.sync_time()), timed("exchangeInfo", self._load_symbols()),
            timed("prefetch", self._f7_prefetch([("GET", "/fapi/v1/premiumIndex", None, False),
                                                 ("GET", "/fapi/v1/fundingInfo", None, False),
                                                 ("GET", "/fapi/v1/ticker/24hr", None, False)])),
            return_exceptions=True)
        for r in res[:2]:
            if isinstance(r, BaseException):
                raise r
        if isinstance(res[2], dict):
            timings.update({f"prefetch {k}": round(v, 1) for k, v in res[2].items()})
        await timed("funding_schedule", self._refresh_funding_schedule())
        timings["total"] = round((time.perf_counter() - t0) * 1e3, 1)
        self.startup_timings.update(timings)
        log.info("startup checks in %.0f ms (%s); runtime %s", timings["total"],
                 ", ".join(f"{k} {v:.0f}" for k, v in timings.items() if k != "total"),
                 ", ".join(f"{k} {v}" for k, v in perf.runtime_info().items()))
        return timings

    async def _f7_prefetch_account(self) -> dict[str, float]:
        """LiveAccount.start and the engine's reconcile read the position mode, the account, the positions and the open
        (algo) orders one after another: fetched here at once (BinanceRest.prefetch, answers kept PREFETCH_TTL_S), they
        then read the answers instead of waiting for five round trips."""
        t0 = time.perf_counter()
        res = await self._f7_prefetch([("GET", "/fapi/v1/positionSide/dual", None, True),
                                        ("GET", "/fapi/v3/account", None, True),
                                        ("GET", "/fapi/v3/positionRisk", None, True),
                                        ("GET", "/fapi/v1/openOrders", None, True),
                                        ("GET", "/fapi/v1/openAlgoOrders", None, True)])
        timings = {f"live {k}": round(v, 1) for k, v in res.items()}
        timings["live total"] = round((time.perf_counter() - t0) * 1e3, 1)
        self.startup_timings.update(timings)
        if res:
            log.info("live account checks in %.0f ms (%s)", timings["live total"],
                     ", ".join(f"{k} {v:.0f}" for k, v in res.items()))
        return timings

    async def _f7_prefetch(self, calls: list) -> dict[str, float]:
        fn = getattr(self.rest, "prefetch", None)  # a stand-in REST client (tests) may not have it
        return await fn(calls) if fn is not None else {}

    def _f7_time_orders(self, acc, name: str) -> None:
        """Latency budget: time every order request of this engine's account until its answer (sent_to_ack) and, for
        an order decided on a bar, that bar's close until the request leaves (close_to_sent)."""
        for kind in ("market_order", "limit_order", "place_stop", "place_take_profit"):
            fn = getattr(acc, kind, None)
            if fn is not None:
                setattr(acc, kind, self._f7_timed(fn, name, kind in ("market_order", "limit_order")))

    def _f7_timed(self, fn, engine: str, decision: bool):
        async def call(symbol, *a, **k):
            if decision:
                bar = self._f7_bar.get(symbol)
                if bar is not None:
                    perf.LATENCY.record(engine, "close_to_sent", now_ms() - perf.exchange_to_local(bar))
            t0 = time.perf_counter()
            try:
                return await fn(symbol, *a, **k)
            finally:
                perf.LATENCY.record(engine, "sent_to_ack", (time.perf_counter() - t0) * 1e3)

        return call

    def _f7_features_ready(self, sym: str) -> None:
        recv = self._f7_recv.pop(sym, None)
        if recv is not None:
            perf.LATENCY.record("app", "recv_to_features", now_ms() - recv)

    def _f7_book(self, sym: str, now: int):
        if self.books is None:
            return EMPTY_BOOK
        try:
            return self.books.features(sym, now)
        except Exception:  # noqa: BLE001 - a book problem must never stop a decision
            log.debug("order book features failed for %s", sym, exc_info=True)
            return EMPTY_BOOK

    def _f7_book_symbols(self) -> None:
        if self.books is not None:
            self.books.set_symbols(self.universe)
        if self.book_stream is not None:
            self.book_stream.set_symbols(self.universe)

    async def _f7_depth_snapshot(self, symbol: str, limit: int) -> dict:
        return await self.rest.request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})

    def _f7_health(self) -> dict[str, Any]:
        """The F7 part of self.health, refreshed at most once a minute (the scheduler updates the health every few
        seconds; summarising the latency reservoirs that often cost more than the measured work)."""
        now = now_ms()
        cached = getattr(self, "_f7_health_cache", None)
        if cached is not None and now - cached[0] < MS_MINUTE:
            return cached[1]
        out = self._f7_health_now()
        self._f7_health_cache = (now, out)
        return out

    def _f7_health_now(self) -> dict[str, Any]:
        views = list(self.views.values())
        # each view's LiveFrames read once: a feature step in a worker thread may drop it (fallback to batch) between
        # a check and a second attribute read
        inc = [lf for lf in (getattr(v, "live", None) for v in views) if lf is not None]
        reanchors = sum(lf.stats["reanchors"] for lf in inc)
        fallback = sum(1 for v in views if self.s.incremental_features and not getattr(v, "incremental", False))
        d = perf.DRIFT.snapshot()
        out = {"latency": " | ".join(perf.LATENCY.lines()) or "-",
               "clock": (f"offset {', '.join(f'{k} {v:+.0f}ms' for k, v in d['offset_ms'].items()) or '-'}, "
                         f"worst {d['worst_ms']:.0f}ms, warnings {d['warnings']}"),
               "features": (f"incremental {len(inc)}/{len(views)} symbols, {reanchors} re-anchors checked"
                            + (f", {fallback} back on batch" if fallback else "") if self.s.incremental_features
                            else "batch rebuild (INCREMENTAL_FEATURES=false)")}
        if self.startup_timings:
            out["startup_ms"] = self.startup_timings.get("total")
        out["runtime"] = ", ".join(f"{k} {v}" for k, v in perf.runtime_info().items())
        if self.books is not None:
            out["orderbook"] = ", ".join(f"{k} {v}" for k, v in self.books.status().items())
        return out

    def _context(self, sym: str, now: int) -> Context:
        t = self.tickers.get(sym) or Ticker(sym)
        if not t.mark and self.candles.get(sym) is not None and self.candles[sym].n:
            t.mark = float(self.candles[sym].close[self.candles[sym].n - 1])
        reg, info = self.regimes.get(sym, (Regime.RANGE, {}))
        btc = self.regimes.get("BTCUSDT", (None, {}))[0] if sym != "BTCUSDT" else None
        bq = self.book_qty.get(sym)
        imb = (bq[0] - bq[1]) / (bq[0] + bq[1]) if bq and (bq[0] + bq[1]) > 0 else 0.0
        ms = self.metrics_series.get(sym)
        extras = ms.snapshot(now) if ms is not None else {}
        oi4 = extras.get("oi_chg_4h") if extras else None
        oi_change = oi4 if oi4 is not None and oi4 == oi4 else None
        if oi_change is None:  # fall back to the 15m OI history poll
            oi = self.oi_hist.get(sym)
            if oi and len(oi) >= 2:
                old = next((v for ts, v in oi if ts <= now - 4 * MS_HOUR), oi[0][1])
                if old:
                    oi_change = oi[-1][1] / old - 1
        iv = self._funding_interval_ms(sym)
        nft = t.next_funding_time or self.funding_next.get(sym) or ((now // iv) + 1) * iv
        return Context(symbol=sym, info=self.symbols[sym], ticker=t, regime=reg, regime_info=info, btc_regime=btc,
                       oi_change=oi_change, now=now, funding_rate=t.funding_rate,
                       minutes_to_funding=max((nft - now) / MS_MINUTE, 0.0), funding_interval_min=iv / MS_MINUTE,
                       book_imbalance=imb, extras=extras, market=self._market_context(sym, now),
                       book=self._f7_book(sym, now))  # F7: order-book features (EMPTY_BOOK unless ORDERBOOK_STREAM)

    # --- P3 signal batching (PORTFOLIO_ALLOCATOR, heartless/execution/allocator.py) ----------------------------------
    # Each engine queues the entry decisions of a bar close (TradingEngine.on_bar) and admits them together in
    # flush_entries, ranked by expected value under the risk budgets: the same code the joint backtester runs after each
    # step of its timeline. Here a close's batch is admitted once every traded symbol's bar at it has been processed,
    # when a bar of a later close arrives first, or ENTRY_BATCH_GRACE_MS after the close (scheduler).
    def _batching(self) -> bool:
        return any(getattr(e, "allocator", None) is not None for e in self.engines.values())

    async def _entry_batch_roll(self, close_time: int) -> None:
        """A bar of a later close is about to be decided while the previous close's batch is still open (a symbol's
        bar is late or missing): admit that batch first, so a candidate never waits for or competes with a later close."""
        if self._batch_t is not None and close_time > self._batch_t:
            await self._flush_entry_batch()

    async def _entry_batch_done(self, sym: str, close_time: int) -> None:
        """`sym`'s bar at `close_time` went through every engine: admit the batch once all traded symbols reported."""
        if not self._batching():
            return
        if self._batch_t is None:
            self._batch_t, self._batch_seen = close_time, set()
        elif close_time != self._batch_t:
            return  # an older bar (gap fill) rides with the open batch; a newer one was rolled before the engines ran
        seen = self._batch_seen if self._batch_seen is not None else set()
        seen.add(sym)
        self._batch_seen = seen
        if all(s in seen for s in self.views):
            await self._flush_entry_batch()

    async def _entry_batch_due(self, now: int) -> None:
        """Scheduler backstop: a batch still open ENTRY_BATCH_GRACE_MS after its close is admitted without the
        stragglers."""
        if self._batch_t is not None and now - self._batch_t > ENTRY_BATCH_GRACE_MS:
            await self._flush_entry_batch()

    async def _flush_entry_batch(self) -> None:
        self._batch_t, self._batch_seen = None, set()
        for eng in list(self.engines.values()):
            if getattr(eng, "allocator", None) is None or not eng.queued_entries:
                continue
            try:
                self._prime_beta(eng)
                await eng.flush_entries()
            except Exception:  # noqa: BLE001
                log.exception("engine %s entry batch failed", eng.name)

    def _prime_beta(self, eng: TradingEngine) -> None:
        """Seed an engine's BTC-beta estimator with the stored hourly closes of the traded symbols and BTC (once per
        symbol), so a fresh process ranks and limits with the whole look-back rather than the hours it observed; the
        joint backtester seeds it from the history before its window the same way."""
        from heartless.execution.allocator import hourly_closes

        beta = eng.allocator.beta
        now = now_ms()
        for sym in sorted(set(self.views) | {beta.btc}):
            if sym in beta.tried:  # once per estimator: a re-created engine of the same name is seeded again
                continue
            beta.tried.add(sym)
            try:
                rows = self.store.query("SELECT open_time, close FROM candles WHERE symbol=? AND open_time>=? AND "
                                        "open_time<? AND open_time % 3600000 = 3540000 ORDER BY open_time",
                                        (sym, now - (beta.hours + 2) * MS_HOUR, now))
            except Exception as e:  # noqa: BLE001 - a missing history only means beta 1.0 until it is observed
                log.debug("beta history %s unavailable: %s", sym, e)
                continue
            if rows:
                import numpy as np

                hours, closes = hourly_closes(np.array([r[0] for r in rows], dtype=np.int64),
                                              np.array([r[1] for r in rows], dtype=float), before=now,
                                              limit=beta.hours + 1)
                beta.prime(sym, hours, closes)

    # --- market context (P2a, data/context.py) ----------------------------------------------------------------
    def _context_only_symbols(self) -> list[str]:
        """Context-universe members trading on the exchange that the bot does not trade."""
        return [s for s in self.context.symbols if s not in self.universe and s in self.symbols
                and self.symbols[s].status == "TRADING"]

    def _set_context_only(self, symbols: list[str]) -> None:
        """New context-only set after a universe change: a symbol that left the universe continues as a 1m-only stream
        from its newest stored bar; one that joined is handled by the trading path from now on."""
        for s in symbols:
            if s not in self._ctx_last_open:
                _, hi = self.store.context_candle_bounds(s)
                if hi is not None:
                    self._ctx_last_open[s] = int(hi)
        for s in [s for s in self._ctx_last_open if s not in symbols]:
            del self._ctx_last_open[s]
        self._ctx_polled &= set(symbols)
        self.context_only = list(symbols)

    def _warm_context(self) -> None:
        """Feed the tracker the stored history its windows need (1m bars, funding, metrics of the last WARMUP_HOURS)
        so the first snapshot after a start or restart already equals the offline build. Synchronous: a few seconds
        once at start-up, before the market stream runs."""
        now = now_ms()
        self._set_context_only(self.context_only)
        try:
            fed = self.context.warm(self.store, now)
        except Exception:  # noqa: BLE001 - the context must never stop the bot from trading
            self._context_error("warm-up")
            return
        log.info("market context warmed: %d bars of %d symbols (%d context-only: %s)", fed,
                 len(self.context.symbols), len(self.context_only), ",".join(self.context_only) or "-")
        # the hour the warm-up finalised is never published by a streamed bar (its decisions read it lazily): store it
        # now, it is what every engine decides on until the next hour close
        h = self.context.last_hour
        if h is not None:
            try:
                self._persist_context(self.context.snapshot(CTX.hour_close(h)))
            except Exception:  # noqa: BLE001
                self._context_error("snapshot")

    def _context_error(self, what: str) -> None:
        self.context_stats["errors"] += 1
        if self.context_stats["errors"] <= 5:  # a systematic failure must not flood the log every minute
            log.exception("market context %s failed (ctx.market falls back to empty)", what)

    def _context_bar(self, sym: str, c: Candle) -> None:
        try:
            self.context.on_bar(sym, c)
        except Exception:  # noqa: BLE001
            self._context_error("bar")

    async def _process_context_bar(self, sym: str, c: Candle) -> None:
        """A closed 1m bar of a context-only symbol: a stream gap is filled from REST first, and the bars are stored
        (a restart warms the tracker from the store; research can build the same context offline from it)."""
        last = self._ctx_last_open.get(sym)
        if last is not None and c.open_time <= last:
            return
        bars: list[Candle] = []
        if last is not None and c.open_time - last > MS_MINUTE and sym not in self._backfilling:
            try:
                got = await self.rest.klines_range(sym, last + MS_MINUTE, c.open_time - 1, "1m") or []
                bars = [b for b in got if last < b.open_time < c.open_time and b.closed]
            except Exception as e:  # noqa: BLE001
                log.warning("gap fill %s failed: %s", sym, e)
        bars.append(c)
        for b in bars:
            self._context_bar(sym, b)
        self.store.save_candles(sym, bars)
        self._ctx_last_open[sym] = c.open_time

    async def _ensure_context(self, t: int) -> None:
        """Publish the snapshot of the hour close at or before t before an engine decides on a bar of that hour. The
        first bar closing at T waits (at most CONTEXT_GRACE_MS past T) until every context symbol that traded in the
        previous hour delivered its bar closing at T; those bars reach the tracker from the stream callback, so they
        arrive while this worker waits. From then on the snapshot is fixed for the hour: the champion, the challengers
        and a live engine all decide on the same one. A bar processed long after its close (backlog, gap fill) does
        not wait."""
        tr = self.context
        h = hour_of(t)
        if tr.last_hour is not None and h <= tr.last_hour:
            return
        T = hour_close(h)
        loop = asyncio.get_running_loop()
        # bounded by the loop clock as well, so a frozen test clock can never make this spin
        until = loop.time() + max(0, T + CONTEXT_GRACE_MS - now_ms()) / 1000
        try:
            while not tr.ready(T) and loop.time() < until:
                await asyncio.sleep(CONTEXT_POLL_S)
                if tr.last_hour is not None and h <= tr.last_hour:
                    return  # published meanwhile (the scheduler's fallback)
            self._publish_context(T, late=not tr.ready(T))
        except Exception:  # noqa: BLE001
            self._context_error("snapshot")

    def _publish_context(self, T: int, late: bool = False) -> None:
        snap = self.context.snapshot(T)
        self._persist_context(snap)
        st = self.context_stats
        st["snapshots"] += 1
        st["last"] = T
        st["symbols"] = int(snap.present.sum()) if snap is not None else 0
        if late:
            st["late"] += 1
            log.info("market context %s UTC published after the %.0fs grace without every symbol's bar",
                     fmt_ts(T, fmt="%m-%d %H:%M"), CONTEXT_GRACE_MS / 1000)

    def _context_tick(self, now: int) -> None:
        """Scheduler fallback: an hour no traded bar has published once its grace ran out is published here, so the
        hourly snapshot (and the tracker's bounded buffers) keep moving even without a traded symbol's bar."""
        tr = self.context
        h = hour_of(now - CONTEXT_GRACE_MS)
        if tr.last_hour is None or h <= tr.last_hour:
            return
        try:
            self._publish_context(hour_close(h), late=not tr.ready(hour_close(h)))
        except Exception:  # noqa: BLE001
            self._context_error("snapshot")

    def _market_context(self, sym: str, now: int):
        """Context.market for a decision at `now`: the snapshot of the last hour close at or before it."""
        try:
            return self.context.view(sym, now)
        except Exception:  # noqa: BLE001
            self._context_error("view")
            return NO_MARKET_CONTEXT

    # --- live market-context store (C2-B) ----------------------------------------------------------------------
    def _ctx_db(self) -> Store:
        """The live context store (opened on first use; the main store when CONTEXT_DB names the main database).
        Checked once when opened (_ctx_meta): a store recorded for another candle database raises ContextMismatch and
        is not used for the rest of the process (_ctx_off)."""
        if self._ctx_off:
            raise CTX.ContextMismatch(self._ctx_off)
        if self._ctx_store is None:
            path = self.live_context_path
            if path.resolve() == Path(self.s.db_path).resolve():
                st = self.store
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                st = Store(path)
            try:
                self._ctx_meta(st)
            except CTX.ContextMismatch as e:
                if st is not self.store:
                    st.close()
                self._ctx_refuse(e)
                raise
            self._ctx_store = st
        return self._ctx_store

    def _ctx_refuse(self, err: Exception) -> None:
        if not self._ctx_off:
            self._ctx_off = str(err)
            log.error("live context store not used: %s. Point CONTEXT_DB at a file of its own (or leave it empty for "
                      "<data_dir>/context.db); until then the hourly snapshots are not stored and the research "
                      "backtests run without market context", err)

    def _ctx_meta(self, st: Store) -> dict:
        """The store's context meta for this bot's universe, recording this bot's candle database as its live source
        (data/context.py live_source; research builds refuse such a store). Rows of another universe or feature version
        are stale (another cross-section) and dropped, as the offline build does. A store recorded for ANOTHER candle
        database (an offline research build, e.g. CONTEXT_DB exported for lab commands) is never touched: appending
        and the retention prune would cut its years of history to the bot's 185 days, a universe mismatch would wipe
        it. ContextMismatch instead."""
        meta = st.get("context.meta")
        own = CTX.live_source(self.s.db_path)
        rec = (meta.get("source") or {}).get("path") if isinstance(meta, dict) else None
        if rec and rec != own["path"]:
            raise CTX.ContextMismatch(f"{self.live_context_path} holds the market context built from {rec}, not from "
                                      f"this bot's candle database {own['path']}")
        if not CTX.meta_ok(meta, self.context.symbols):
            if st.has_market_context():
                log.warning("live context store %s holds another context version or universe: rebuilt",
                            self.live_context_path)
                st.delete_market_context()
            meta = CTX.meta_for(self.context.symbols, own)
            st.set("context.meta", meta)
        elif meta.get("source") != own:  # a store from before the live flag, or an offline build of this very database
            meta["source"] = own
            st.set("context.meta", meta)
        return meta

    def _persist_context(self, snap) -> None:
        """Append one published snapshot to the live context store (its hour close joins the built spans)."""
        if snap is None or self._ctx_off:
            return
        try:
            st = self._ctx_db()
            meta = self._ctx_meta(st)
            st.add_market_context(snap.rows(), snap.ts, snap.ts, meta)
            self.context_stats["stored"] += 1
        except CTX.ContextMismatch as e:
            self._ctx_refuse(e)
        except Exception:  # noqa: BLE001 - persistence must never stop the snapshot from reaching the engines
            self._context_error("persist")

    def _context_gaps(self, lo: int, hi: int) -> list[tuple[int, int]]:
        """Hour-close spans inside [lo, hi] the live context store has not built (per its meta)."""
        built = sorted((int(a), int(b)) for a, b in (self._ctx_meta(self._ctx_db()).get("built") or []))
        out: list[tuple[int, int]] = []
        cur = lo
        for a, b in built:
            if b < cur:
                continue
            if a > hi:
                break
            if a > cur:
                out.append((cur, min(hi, a - MS_HOUR)))
            cur = max(cur, b + MS_HOUR)
            if cur > hi:
                break
        if cur <= hi:
            out.append((cur, hi))
        return out

    def _compute_context(self, a: int, b: int) -> list:
        """Offline builder (data/context.py compute_unit) over the stored candles for the hour closes [a, b], on its own
        connection (runs in a worker thread; the main store's lock stays free for the trading loop)."""
        src = Store(self.s.db_path)
        try:
            return CTX.compute_unit(src, self.context.symbols, a, b)
        finally:
            src.close()

    def _build_context_gap(self, a: int, b: int) -> int:
        """Worker thread: compute the hour closes [a, b] and store them under the store's exclusive build lock (the
        lock data/context.py builds and readers take), so the research worker's ContextTable.load never reads a span
        half written. Returns the number of snapshots stored."""
        st = self._ctx_db()
        with CTX.build_lock(st.path):
            snaps = self._compute_context(a, b)
            st.add_market_context([r for sn in snaps for r in sn.rows()], a, b, self._ctx_meta(st))
        return len(snaps)

    async def _fill_context_gaps(self, wait_for: asyncio.Task | None = None) -> int:
        """Start-up, in the background: build the hour closes of the research window (RESEARCH_LOOKBACK_DAYS before now
        up to the hour the warm-up finalised) that the live context store lacks, from the stored candles with the
        offline builder. Returns the number of snapshots stored. A failure is logged and left for the next start."""
        if wait_for is not None:
            try:
                await asyncio.shield(wait_for)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                pass
        h = self.context.last_hour
        if h is None or self._ctx_off:
            return 0
        now = now_ms()
        lo = CTX.hour_close(CTX.hour_of(now - self.s.research_lookback_days * MS_DAY))
        hi = CTX.hour_close(h)
        stored = 0
        try:
            for a, b in self._context_gaps(lo, hi):
                n = await asyncio.to_thread(self._build_context_gap, a, b)
                stored += n
                log.info("live context store: built %d hour(s) %s .. %s UTC from stored candles", n,
                         fmt_ts(a, fmt="%m-%d %H:%M"), fmt_ts(b, fmt="%m-%d %H:%M"))
        except asyncio.CancelledError:
            raise
        except CTX.ContextMismatch as e:
            self._ctx_refuse(e)
        except Exception:  # noqa: BLE001
            self.context_stats["fill_errors"] += 1
            log.exception("live context store: building the missing hours failed")
        self.context_stats["filled"] += stored
        return stored

    async def _btc_shock_check(self, view: MarketView) -> None:
        m1 = view.tf("1m")
        if not m1.ok or m1.idx < 6:
            return
        move = abs(m1.v("close") - m1.v("close", 5)) / m1.v("close", 5)
        if move >= 0.03:
            for eng in self.engines.values():
                eng.risk.register_market_shock(now_ms(), 15)
            await self.emit("risk_event", {"message": f"⚠️ BTC 5분 내 {move * 100:.1f}% 급변 → 15분간 신규 진입 중단"})

    # --- scheduler -----------------------------------------------------------------------------
    async def _scheduler(self) -> None:
        await asyncio.sleep(5)
        while True:
            now = now_ms()
            try:
                self._roll_day(now)  # first: the reconcile and challenger checks below can book trades
                if now - self._last_equity_tick >= 30_000:
                    self._last_equity_tick = now
                    for eng in list(self.engines.values()):
                        await eng.on_equity_tick()
                    self._persist_paper_accounts()
                if now - self._last_reconcile >= 45_000:
                    self._last_reconcile = now
                    if "live" in self.engines:
                        await self.engines["live"].reconcile()
                    await self._paper_reconcile()
                if self._backfill_failed and now - self._last_backfill_retry >= BACKFILL_RETRY_MS:
                    self._last_backfill_retry = now
                    await self._retry_backfill()
                if now - self._last_oi_poll >= 5 * MS_MINUTE:
                    self._last_oi_poll = now
                    asyncio.create_task(self._poll_metrics())
                if now - self._last_funding_refresh >= FUNDING_REFRESH_MS:
                    self._last_funding_refresh = now
                    asyncio.create_task(self._funding_tick())
                if now - self._last_universe >= self.s.universe_refresh_minutes * MS_MINUTE:
                    await self.refresh_universe()
                self._context_tick(now)
                await self._entry_batch_due(now)  # P3: an entry batch whose stragglers never came
                self._history_depth_tick(now)  # long-lookback alphas of a new champion / challenger (C2-A)
                if self.research.due(now) and not self.research.running:
                    asyncio.create_task(self.research.cycle())
                elif self.research.discovery_due(now) and not self.research.running:
                    asyncio.create_task(self.research.discovery_cycle())
                await self.research.evaluate_challengers()
                await self.research.check_graduation()
                if now >= self._next_report_ts:
                    self._next_report_ts = next_local_time(self.s.daily_report_hour, 0, self.s.timezone, now)
                    asyncio.create_task(self.daily_report())
                if now - self._last_prune >= 6 * MS_HOUR:
                    self._last_prune = now
                    self._prune(now)
                self._update_health(now)
            except Exception:  # noqa: BLE001
                log.exception("scheduler iteration failed")
            await asyncio.sleep(5)

    def _prune(self, now: int) -> None:
        """Retention: candles, funding, metrics, equity rows and fill-quality rows older than
        max(HISTORY_DAYS, DATA_RETENTION_DAYS) + 5 days. Only candles were pruned before; the 5-minute metrics and the
        per-minute equity rows of every engine grew without bound."""
        cutoff = now - (max(self.s.history_days, self.s.data_retention_days, self._alpha_history_days()) + 5) * MS_DAY
        self.store.prune_candles(cutoff)
        self.store.prune_funding(cutoff)
        self.store.prune_metrics(cutoff)
        self.store.prune_equity(cutoff)
        self.store.prune_fill_quality(cutoff)
        if not self._ctx_off:
            # one DELETE (atomic for a reader) of hours far behind the research window: no build lock needed
            try:
                self._ctx_db().prune_market_context(cutoff)
            except CTX.ContextMismatch as e:
                self._ctx_refuse(e)
            except Exception:  # noqa: BLE001
                log.warning("live context store prune failed", exc_info=True)

    def _roll_day(self, now: int) -> None:
        """Start a new reporting day (the engines' daily counters) once `now` passed the local midnight. Runs before
        any tick, bar or owner close reaches an engine, not only in the 5 s scheduler: a trade closed in between was
        booked into the old day's counter and then wiped, so the day's "today" PnL in close messages missed it."""
        if now < self._next_day_ms:
            return
        self._next_day_ms = next_local_time(0, 0, self.s.timezone, now)
        day_key = local_day_start(now, self.s.timezone)
        if day_key != self._last_day_key:
            self._last_day_key = day_key
            for eng in self.engines.values():
                eng.reset_daily()

    def _persist_paper_accounts(self) -> None:
        for name, acc in self.paper_accounts.items():
            self.store.set(f"{name}.account", {"wallet": acc.wallet, "initial": acc.initial_balance, "realized": acc.realized,
                                               "fees": acc.fees_paid, "funding": acc.funding_paid})

    async def _paper_reconcile(self) -> None:
        """Keep paper engines consistent with their simulated accounts (e.g. after restarts)."""
        for name, acc in list(self.paper_accounts.items()):
            eng = self.engines.get(name)
            if eng is None:
                continue
            for sym, pos in list(eng.positions.items()):
                # a CLOSING row whose simulated quantity is gone was closed by the previous process: book it
                if pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING):
                    continue
                pp = acc.positions.get(sym)
                if pp is None or abs(pp.qty) <= 0:
                    mark = self.marks().get(sym) or pos.entry_price
                    await eng._finalize(pos, mark, now_ms(), "시뮬 계정과 동기화")

    async def _poll_metrics(self) -> None:
        """Poll the 5-minute positioning series (OI, top-trader / global long-short, taker ratio) for every
        symbol. The first poll loads ~2 days (enough for the 24h change and the OI z-score); later polls fetch
        the newest points and merge them. Rows are archive-shaped, so alphas see exactly what backtests saw."""
        from heartless.data.extras import rows_from_rest

        for sym in list(self.universe):
            have = self.metrics_series.get(sym)
            limit = 500 if have is None or len(have) < 100 else 6
            try:
                oi, tp, ta, ga, tk = await asyncio.gather(
                    self.rest.open_interest_hist(sym, "5m", limit), self.rest.top_long_short_ratio(sym, "5m", limit),
                    self.rest.top_long_short_account_ratio(sym, "5m", limit),
                    self.rest.global_long_short_account_ratio(sym, "5m", limit),
                    self.rest.taker_long_short_ratio(sym, "5m", limit))
                rows = rows_from_rest(oi, tp, ta, ga, tk)
                fresh = MetricsSeries.from_rows(rows)
                merged = fresh if have is None else have.merge(fresh)
                self._context_metrics(sym, self._store_metrics(sym, rows))
                self.metrics_series[sym] = merged.tail(1200)
                # keep the 15m OI deque fed for anything still reading it
                dq = self.oi_hist.setdefault(sym, deque(maxlen=64))
                if oi:
                    dq.clear()
                    for r in oi[-64:]:
                        dq.append((int(r["timestamp"]), float(r["sumOpenInterest"])))
            except Exception as e:  # noqa: BLE001
                log.debug("metrics poll %s failed: %s", sym, e)
            await asyncio.sleep(0.25)
        # context-only symbols: the market context reads open interest only (one series instead of five)
        for sym in list(self.context_only):
            try:
                oi = await self.rest.open_interest_hist(sym, "5m", 6 if sym in self._ctx_polled else 500)
                self._context_metrics(sym, self._store_metrics(sym, rows_from_rest(oi, [], [], [], [])))
                if oi:
                    self._ctx_polled.add(sym)
            except Exception as e:  # noqa: BLE001
                log.debug("metrics poll %s failed: %s", sym, e)
            await asyncio.sleep(0.25)

    def _context_metrics(self, sym: str, rows: list) -> None:
        """Hand the rows just stored (merged with what was stored before, so the tracker holds what the store holds)
        to the market context."""
        try:
            for r in rows:
                self.context.on_metrics(sym, r)
        except Exception:  # noqa: BLE001
            self._context_error("metrics")

    def _store_metrics(self, sym: str, rows: list[dict]) -> list:
        """Persist polled positioning rows (archive-shaped, keyed by bucket start) and return them as stored. The
        research worker's backtests read metrics from the store like the lab does; with polled rows kept in memory only,
        every research cycle evaluated the metrics alphas (positioning, funding_fade) without data, so they could never
        pass and be switched back on. A bucket polled while still incomplete is completed by the next polls (each
        re-reads the newest buckets). A value missing from this poll keeps the stored one: a bucket that only some
        series answered (one series lagging a bucket behind the others, or answering empty) would otherwise erase what
        an earlier poll stored, and with a steady one-bucket lag every bucket would end up holding a single series once
        it left the poll window."""
        from types import SimpleNamespace

        cols = ("oi", "oi_value", "top_ls_accounts", "top_ls_positions", "ls_accounts", "taker_ls_vol")
        out: list = []
        try:
            if not rows:
                return out
            ts = [int(r["ts"]) for r in rows]
            stored = {int(r["ts"]): r for r in self.store.load_metrics(sym, min(ts), max(ts))}
            for t, r in zip(ts, rows):
                old = stored.get(t) or {}
                out.append(SimpleNamespace(ts=t, **{c: r.get(c) if r.get(c) is not None else old.get(c) for c in cols}))
            self.store.save_metrics(sym, out)
        except Exception as e:  # noqa: BLE001 - persistence must never break the live series
            log.warning("metrics store %s failed: %s", sym, e)
        return out

    async def _refresh_funding(self) -> None:
        """Store the funding settlements published since the last stored one, for every universe symbol. _backfill
        only fetched them at start-up and on universe changes, so on a long-running bot the funding table stopped at
        the start: research backtests charge the latest stored rate at every later settlement, i.e. they kept applying
        the start-up rate for weeks (and the funding-avoidance exit saw it too). The context-only symbols are refreshed
        too, and every new settlement is handed to the market context (its snapshots read the settled rates)."""
        now = now_ms()
        for sym in list(self.universe) + list(self.context_only):
            _, f_hi = self.store.funding_range(sym)
            if f_hi is None:
                continue  # never backfilled (the backfill owns that)
            due = self._funding_due(sym, f_hi, now)
            if due is None:
                continue  # no settlement due since the last stored one
            try:
                fr = await self.rest.funding_rate_history(sym, start=f_hi + 1, end=now)
                rows = [(int(r["fundingTime"]), float(r["fundingRate"]), float(r.get("markPrice", 0) or 0)) for r in fr]
                self.store.save_funding(sym, rows)
                for t, rate, _ in rows:
                    self.context.on_funding(sym, t, rate)
                if not any(t >= due for t, _, _ in rows):
                    self._funding_miss[sym] = due
            except Exception as e:  # noqa: BLE001
                log.debug("funding refresh %s failed: %s", sym, e)

    def _funding_interval_ms(self, sym: str) -> int:
        return int(self.funding_interval.get(sym, DEFAULT_FUNDING_INTERVAL_MS))

    def _funding_due(self, sym: str, f_hi: int, now: int) -> int | None:
        """The settlement time after the last stored one (f_hi) that is due at `now`, or None. With the exchange's next
        settlement time T known, the latest settlement at or before now is T (when it has passed) or T minus the
        symbol's interval; without it, f_hi plus the interval. A 4-hourly symbol's settlements were otherwise fetched
        only every 8 h, i.e. up to 4 h late for the market context (the offline build has them on time) and for the
        funding-avoidance exit. An expectation the history did not show FUNDING_MISS_GRACE_MS after it was due is not
        asked for again (an interval that changed under us)."""
        iv = self._funding_interval_ms(sym)
        nxt = self.funding_next.get(sym)
        if nxt:
            last = nxt if nxt <= now else nxt - iv
        else:
            last = f_hi + iv
        if last <= f_hi or last > now:
            return None
        if self._funding_miss.get(sym) == last and now - last > FUNDING_MISS_GRACE_MS:
            return None
        return last

    async def _funding_tick(self) -> None:
        await self._refresh_funding_schedule()
        await self._refresh_funding()

    async def _refresh_funding_schedule(self) -> None:
        """Every symbol's next settlement time (premiumIndex nextFundingTime, all symbols in one call) and settlement
        interval (fundingInfo fundingIntervalHours; a symbol it does not list settles every 8 h). Failures keep the
        previous schedule (the funding refresh then falls back to the 8-hour rule for symbols without one)."""
        try:
            rows = await self.rest.premium_index()
            for r in rows if isinstance(rows, list) else [rows]:
                t = int(r.get("nextFundingTime") or 0)
                if r.get("symbol") and t > 0:
                    self.funding_next[r["symbol"]] = t
        except Exception as e:  # noqa: BLE001
            log.debug("premium index refresh failed: %s", e)
        try:
            info = await self.rest.request("GET", "/fapi/v1/fundingInfo")
            if isinstance(info, list):
                self.funding_interval = {r["symbol"]: int(float(r["fundingIntervalHours"]) * MS_HOUR) for r in info
                                         if r.get("symbol") and float(r.get("fundingIntervalHours") or 0) > 0}
        except Exception as e:  # noqa: BLE001
            log.debug("funding info refresh failed: %s", e)

    async def _poll_open_interest(self) -> None:
        for sym in list(self.universe):
            try:
                rows = await self.rest.open_interest_hist(sym, "15m", 20)
                dq = self.oi_hist.setdefault(sym, deque(maxlen=64))
                dq.clear()
                for r in rows:
                    dq.append((int(r["timestamp"]), float(r["sumOpenInterest"])))
            except Exception as e:  # noqa: BLE001
                log.debug("OI poll %s failed: %s", sym, e)
            await asyncio.sleep(0.2)

    def _update_health(self, now: int) -> None:
        stale = [s for s in self.universe if now - self.last_bar_ts.get(s, self.started_at) > 3 * MS_MINUTE]
        short = [s for s in self.universe if s in self.candles and self.candles[s].n < LIVE_WINDOW["1m"]]
        st = self.context_stats
        ctx = (f"{st['symbols']}/{len(self.context.symbols)} symbols at {fmt_ts(st['last'], fmt='%m-%d %H:%M')} UTC, "
               f"{st['snapshots']} snapshots ({st['late']} after the grace, {st['errors']} errors), "
               f"{len(self.context_only)} context-only streams; "
               + (f"stored {st['stored']} + built {st['filled']} hours in {self.live_context_path.name}"
                  if not self._ctx_off else f"NOT STORED ({self._ctx_off})"))
        self.health = {"market_stream": self.market.connected, "stale_symbols": stale, "queue": self._kline_queue.qsize(),
                       "short_history": short, "backfill_pending": sorted(self._backfill_failed), "market_context": ctx,
                       "user_stream": self.live_account.user_stream.connected if self.live_account else None,
                       "uptime_min": (now - self.started_at) / MS_MINUTE, "used_weight": self.rest.used_weight}
        self.health.update(self._f7_health())  # F7: latency budget, clock drift, feature updates, order books

    # --- control API (Telegram / web) ----------------------------------------------------------
    async def pause(self, reason: str = "owner") -> None:
        for eng in self.engines.values():
            eng.risk.pause()  # also writes the per-engine risk blob now, so a restart right after still sees the pause
        self.store.set("paused", True)  # read back by _apply_persisted_pause for every engine created later
        self.store.set("halt_reason", f"일시정지 ({reason}) — /resume 로 재개")
        await self.emit("control", {"message": f"⏸ 신규 진입 일시정지 ({reason}). 기존 포지션은 계속 관리됩니다"})

    async def resume(self) -> None:
        for eng in self.engines.values():
            # clears the flag in the per-engine blob immediately and re-bases the drawdown peak to the current equity,
            # so the max-drawdown pause re-arms after a further decline instead of staying latched until a new high
            eng.risk.resume(eng.stats.equity)
        self.store.set("paused", False)
        self.store.set("halt_reason", "")
        await self.emit("control", {"message": "▶️ 거래 재개"})

    async def close_symbol(self, symbol: str, engine: str | None = None) -> int:
        self._roll_day(now_ms())
        n = 0
        targets = [self.engines[engine]] if engine and engine in self.engines else [e for e in (self.primary_engine,) if e]
        for eng in targets:
            if await eng.close_position(symbol.upper(), "수동 청산(owner)"):
                n += 1
        return n

    async def close_all(self, engine: str | None = None) -> int:
        self._roll_day(now_ms())
        targets = [self.engines[engine]] if engine and engine in self.engines else [e for e in (self.primary_engine,) if e]
        n = 0
        for eng in targets:
            n += await eng.close_all("전체 수동 청산(owner)")
        return n

    async def kill(self) -> int:
        n = await self.close_all()
        await self.pause("kill switch")
        return n

    async def set_credentials(self, api_key: str, api_secret: str, testnet: bool | None = None) -> str:
        """Owner supplied Binance keys (web settings / Telegram). Verify them, persist them, keep the mode unchanged."""
        from heartless.core.secrets import save_secrets
        from heartless.exchange.binance_rest import BinanceRest

        api_key, api_secret = (api_key or "").strip(), (api_secret or "").strip()
        if not api_key or not api_secret:
            return "API 키와 시크릿을 모두 입력해야 합니다"
        if self.mode == "live":
            return "라이브 모드 중에는 키를 바꿀 수 없습니다. 먼저 /mode paper 로 전환하세요"
        tn = self.s.binance_testnet if testnet is None else bool(testnet)
        probe = BinanceRest(api_key, api_secret, tn)
        try:
            await probe.sync_time()
            acc = await probe.account()
        except Exception as e:  # noqa: BLE001
            await probe.close()
            return f"키 검증 실패: {e}"
        await probe.close()
        if acc.get("canTrade") is False:
            return "이 키에는 선물 거래 권한이 없습니다 (Binance API 관리에서 Futures 권한을 켜세요)"
        save_secrets(self.s.data_dir, {"BINANCE_API_KEY": api_key, "BINANCE_API_SECRET": api_secret,
                                       "BINANCE_TESTNET": "true" if tn else "false"})
        self.s.binance_api_key, self.s.binance_api_secret, self.s.binance_testnet = api_key, api_secret, tn
        self.rest.set_credentials(api_key, api_secret, tn)
        wallet = float(acc.get("totalWalletBalance", 0.0) or 0.0)
        await self.emit("control", {"message": f"🔑 Binance 키 등록 완료 ({'테스트넷' if tn else '메인넷'}, 지갑 {wallet:,.2f} USDT). "
                                               "페이퍼 모드 유지 중 — 실거래는 /golive 로 확인 후 전환됩니다"})
        return f"키 등록 완료 (지갑 {wallet:,.2f} USDT). 실거래 전환은 /golive"

    async def set_mode(self, mode: str) -> str:
        if mode == self.mode:
            return f"이미 {mode} 모드입니다"
        if mode == "live":
            if not self.s.live_capable:
                return "Binance API 키가 설정되지 않아 라이브 모드를 켤 수 없습니다"
            try:
                await self._start_live()
            except Exception as e:  # noqa: BLE001
                # _start_live published nothing: mode stays paper, nothing is persisted, and the owner is told
                # (Telegram/web show this string) instead of a silent log line
                log.exception("live start failed")
                await self.emit("error", {"message": f"라이브 전환 실패: {e}"})
                return f"라이브 전환 실패: {e}"
            self.mode = "live"
            self.store.set("mode", "live")
            await self.emit("mode_changed", {"message": "🔴 LIVE 모드 전환: 실제 자금으로 거래를 시작합니다", "mode": "live"})
            return "라이브 모드로 전환했습니다"
        # live -> paper: flatten, then remove the live engine -- but only once nothing is left open, on our side or
        # on the exchange. Dropping the engine with a position still open (a rejected/timed-out reduce-only order)
        # would orphan real money: no trailing/time stop, no reconcile, no software backstop.
        eng = self.engines.get("live")
        n = 0
        if eng:
            self._roll_day(now_ms())  # the flatten below reaches the engine directly, not through close_all
            eng.entries_enabled = False
            err = ""
            remaining: set[str] = set()
            try:
                n = await eng.close_all("모드 전환(paper)")
                await asyncio.sleep(1.0)
                remaining = {p.symbol for p in eng.open_positions()}
                remaining |= {s for s, snap in (await eng.account.get_positions()).items() if abs(snap.qty) > 0}
            except Exception as e:  # noqa: BLE001
                log.exception("live -> paper flatten failed")
                err = str(e)
            if err or remaining:
                eng.entries_enabled = True  # back to the pre-call state: nothing stays half-switched
                msg = (f"라이브 포지션 청산 미완료 ({', '.join(sorted(remaining)) or err}): 전환 취소, 라이브 모드 유지. "
                       f"잠시 후 /mode paper 를 다시 시도하세요")
                await self.emit("error", {"message": msg})
                return msg
            self.engines.pop("live", None)
        acc = eng.account if eng else self.live_account
        if acc is not None:
            try:
                await acc.stop()
            except Exception:  # noqa: BLE001
                pass
        self.live_account = None  # also releases an account left behind by an earlier failed switch
        self.mode = "paper"
        self.store.set("mode", "paper")
        self.bandit.live_attached = False  # live's final outcomes were booked above; paper learns again at weight 1.0
        if "paper" in self.engines:
            self.engines["paper"].notify = True
        await self.emit("mode_changed", {"message": f"🟢 PAPER 모드 전환 (라이브 포지션 {n}개 청산)", "mode": "paper"})
        return "페이퍼 모드로 전환했습니다"

    # --- reporting -----------------------------------------------------------------------------
    def status(self) -> dict:
        marks = self.marks()
        engines = {name: eng.snapshot(marks) for name, eng in self.engines.items()}
        prim = self.primary_engine
        day_start = local_day_start(now_ms(), self.s.timezone)
        today = self.store.load_trades(engine=prim.name, since=day_start) if prim else []
        st_today = summarize(today)
        return {"mode": self.mode, "paused": prim.risk.state.paused if prim else bool(self.store.get("paused", False)),
                "universe": self.universe,
                "params_version": self.params.version, "engines": engines, "primary": prim.name if prim else None,
                "today": st_today, "health": self.health, "research": {"last_cycle": self.research.last_cycle_ts,
                                                                         "running": self.research.running,
                                                                         "failures": self.research.research_failures, "last_error": self.research.last_error,
                                                                         "next_due": self.research.last_cycle_ts + self.research.interval_ms()},
                "portfolio": self.portfolio_status(),
                "challengers": [self._challenger_status(c) for c in self.challengers],
                "graduation": self.research.graduation_status() if self.mode != "live" else None,
                "alphas": self.bandit.snapshot(), "regimes": {s: r[0].value for s, r in self.regimes.items()},
                "web_token": self.web_token, "started_at": self.started_at, "live_capable": self.s.live_capable,
                "learning": self.learning_status(), "exec_quality": self.exec_quality(),
                "kill_switch": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}
                                for st in self.killswitch.champion_state()]}

    def portfolio_status(self) -> dict:
        """The P3 portfolio allocator for /status, the daily report and the start message: whether the primary engine
        batches entries (PORTFOLIO_ALLOCATOR; off by default, docs/PORTFOLIO.md), the budgets and count caps, and while
        it is on that engine's open risk against the budgets and the entries they rejected since it started."""
        from heartless.execution.allocator import SAME_REASON, TOTAL_REASON

        prim = self.primary_engine
        alloc = getattr(prim, "allocator", None) if prim is not None else None
        out: dict[str, Any] = {"enabled": alloc is not None if prim is not None else bool(self.s.portfolio_allocator),
                               "max_total_risk_pct": self.s.max_total_risk_pct,
                               "max_same_direction_risk_pct": self.s.max_same_direction_risk_pct,
                               "max_positions": self.s.max_positions, "max_same_direction": self.s.max_same_direction,
                               "beta_lookback_days": self.s.beta_lookback_days}
        if alloc is not None:
            out["exposure"] = {k: round(v, 4) for k, v in
                               alloc.exposure(prim.open_positions(), prim.stats.equity).items()}
            out["rejected"] = {"total": prim.stats.skipped.get(TOTAL_REASON, 0),
                               "same": prim.stats.skipped.get(SAME_REASON, 0)}
            out["queued"] = prim.queued_entries
        return out

    def _challenger_status(self, c: ChallengerSlot) -> dict:
        row = {"name": c.name, "version": c.params.version, "source": c.source, "started": c.started,
               "trial_days": self.research.trial_days(c)}
        d = self.research.last_decisions.get(c.name)
        if d:
            row["decision"] = {"path": d.get("path"), "verdict": d.get("verdict"), "rule": d.get("rule"),
                               "at": d.get("at"), "challenger": d.get("challenger"), "bar": d.get("bar"),
                               "paired": d.get("paired"), "checks": d.get("checks"), "checkpoint": d.get("checkpoint"),
                               "next_checkpoint": d.get("next_checkpoint"), "current": d.get("current")}
        row["base"] = c.base
        row["diff"] = self.research.slot_diff(c)
        return row

    def pnl_report(self, period: str = "today") -> dict:
        prim = self.primary_engine
        if prim is None:
            return {}
        now = now_ms()
        if period == "today":
            since = local_day_start(now, self.s.timezone)
        elif period == "week":
            since = now - 7 * MS_DAY
        elif period == "month":
            since = now - 30 * MS_DAY
        else:
            since = None
        trades = self.store.load_trades(engine=prim.name, since=since)
        curve = self.store.load_equity(prim.name, since=since, limit=3000)
        return {"period": period, "engine": prim.name, "stats": summarize(trades), "by_alpha": by_alpha(trades),
                "trades": trades[:50], "equity": curve, "equity_now": prim.stats.equity,
                "unrealized": sum(p.unrealized(self.marks().get(p.symbol) or p.entry_price) for p in prim.open_positions()
                                  if p.status.value == "OPEN")}

    async def daily_report(self) -> None:
        rep = self.pnl_report("today")
        week = self.pnl_report("week")
        await self.emit("daily_report", {"today": rep, "week": week, "status": self.status()})
        if self.advisor.available:
            try:
                payload = {"today": {"stats": rep["stats"], "by_alpha": rep["by_alpha"], "trades": rep["trades"][:40]},
                           "week": {"stats": week["stats"], "by_alpha": week["by_alpha"]},
                           "alpha_trust": self.bandit.snapshot(), "params": self.params.to_dict(),
                           "regimes": {s: r[0].value for s, r in self.regimes.items()}, "mode": self.mode}
                text, seeds = await self.advisor.review(payload)
                if seeds:
                    self.store.set("advisor.seeds", seeds)
                if text:
                    await self.emit("advisor_review", {"message": text, "seeds": seeds})
            except Exception:  # noqa: BLE001
                log.exception("advisor review failed")

    # --- read-only views for the owner's UIs (Telegram /why /alphas, the daily report, the dashboard research page) ---
    def killed_alphas(self, limit: int = 20) -> list[dict]:
        """Kill-switch trips that switched an alpha off in the champion, newest first: the version they created, when,
        the rule and record that tripped, and whether the alpha is still off (only a promotion switches it back on)."""
        out = []
        for row in self.store.params_versions_since(0, source="kill-switch")[:limit]:
            for d in (row.get("metrics") or {}).get("decisions") or []:
                if d.get("alpha"):
                    out.append({"alpha": d["alpha"], "version": row.get("id"), "at": int(row.get("created") or 0),
                                "engine": d.get("engine"), "rule": d.get("rule"), "n": d.get("n", 0),
                                "sum_r": d.get("sum_r", 0.0), "still_off": not self.params.enabled.get(d["alpha"], False)})
        return out

    def why(self, symbol: str) -> dict:
        """What /why shows for one symbol: the primary engine's open position (its signal's reasons and tags, levels,
        risk, R so far, expected R, the alpha's evidence, live record and kill-switch state), other engines' positions
        in it, the latest entry decisions of every engine (TradingEngine.decision_log, memory only) and what blocks a
        new entry on the primary engine right now."""
        sym = symbol.upper()
        now = now_ms()
        prim = self.primary_engine
        mark = self.marks().get(sym)
        out: dict[str, Any] = {"symbol": sym, "engine": prim.name if prim else None, "mode": self.mode,
                               "known": sym in self.symbols, "in_universe": sym in self.universe,
                               "active": self.enabled_alphas(), "mark": mark,
                               "regime": self.regimes[sym][0].value if sym in self.regimes else None,
                               "position": None, "others": [], "decisions": [], "blockers": [],
                               "killed": [k for k in self.killed_alphas() if k["still_off"]]}
        live = (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.CLOSING)
        for name, eng in self.engines.items():
            p = eng.positions.get(sym)
            if p is None or p.status not in live:
                continue
            m = mark or p.extra.get("last_mark") or p.entry_price
            if prim is None or name != prim.name:
                out["others"].append({"engine": name, "side": p.side.value, "status": p.status.value, "alpha": p.alpha,
                                      "r_now": p.r_now(m)})
                continue
            q = p.original_qty or p.qty
            out["position"] = {
                "side": p.side.value, "status": p.status.value, "alpha": p.alpha, "alphas": list(p.alphas),
                "reason": p.reason, "tags": {k: v for k, v in (p.extra.get("tags") or {}).items() if k != "ref_price"},
                "regime": p.regime, "confidence": p.confidence, "timeframe": p.timeframe, "entry": p.entry_price,
                "entry_time": p.entry_time, "mark": m, "stop": p.stop, "initial_stop": p.initial_stop, "tp1": p.tp1,
                "tp1_done": p.tp1_done, "tp": p.take_profit, "trail_atr_mult": p.trail_atr_mult,
                "max_hold_bars": p.max_hold_bars, "bars_held": p.bars_held, "qty": p.qty, "notional": p.notional,
                "leverage": p.leverage, "risk": p.risk_amount, "r_unit": p.r_unit, "r_now": p.r_now(m),
                "stop_r": (p.stop - p.entry_price) * p.side.sign / p.r_unit if p.r_unit else 0.0,
                "unrealized": p.unrealized(m) if p.status is PositionStatus.OPEN else 0.0, "realized": p.realized,
                "expected_r": p.expected_profit / (p.r_unit * q) if p.r_unit and q else None,
                "params_version": p.params_version, "evidence": evidence.entry(p.alpha),
                "evidence_stale": evidence.entry(p.alpha) is None and p.alpha in evidence.registry().get("alphas", {}),
                "live": {"engine": name, **r_stats(self.store.load_trades(engine=name, alpha=p.alpha))},
                "kill_switch": next((k for k in self.killswitch.champion_state()
                                     if k.get("engine") == name and k.get("alpha") == p.alpha), None)}
        order = ([prim.name] if prim else []) + [n for n in self.engines if not prim or n != prim.name]
        for name in order:
            log_ = getattr(self.engines[name], "decision_log", None) or {}
            out["decisions"] += list(reversed(log_.get(sym, ())))
        if prim is not None:
            if not prim.entries_enabled:
                out["blockers"].append({"code": "entries disabled"})
            blocked = prim.risk.entries_blocked(now)
            if blocked:
                out["blockers"].append({"code": blocked.split(":")[0], "detail": blocked.partition(":")[2]})
            until = prim.risk.state.symbol_cooldown.get(sym, 0)
            if until > now:
                out["blockers"].append({"code": "cooldown", "until": until})
            if len(prim.open_positions()) >= self.s.max_positions:
                out["blockers"].append({"code": "max positions", "detail": str(self.s.max_positions)})
        return out

    def challenger_outlook(self) -> list[dict]:
        """Per challenger: its checkpoint schedule, the next checkpoint and its date, and the current record against
        that checkpoint's bar on the path the champion's record selects now (the retirement statistic of the latest
        hourly decision, or before the first one judge_challenger's own: a re-derivation here would drift from the
        statistic the bar is calibrated for whenever the judge changes)."""
        now = now_ms()
        out = []
        for c in self.challengers:
            d = self.research.last_decisions.get(c.name) or {}
            failed = evidence.failed(self.research.slot_new_alphas(c))
            sched = list(R.checkpoints(failed))
            passed = [x for x in sched if now >= c.started + x * MS_DAY]
            nxt = next((x for x in sched if x not in passed), None)
            if d:
                path, cur, stat = d.get("retire_path") or d.get("path"), d.get("current") or {}, d.get("retire_stat")
            else:
                ch = [t for t in self.store.load_trades(engine=c.name, since=c.started)
                      if (t.get("entry_time") or 0) >= c.started]
                champ = [t for t in self.store.load_trades(engine="paper", since=c.started)
                         if (t.get("entry_time") or 0) >= c.started]
                j = R.judge_challenger(ch, champ, self.s, c.started, now, self.research.slot_new_alphas(c))
                path, cur, stat = j["retire_path"], j["current"], j["retire_stat"]
            bar = R.promotion_bar(self.s, path or "absolute", failed, nxt if nxt is not None else sched[-1])
            out.append({"name": c.name, "source": c.source, "version": c.params.version, "started": c.started,
                        "checkpoints": sched, "passed": passed, "next_checkpoint": nxt,
                        "next_at": c.started + nxt * MS_DAY if nxt is not None else None, "path": path,
                        "current": cur, "stat": stat, "hac_lag": bar.get("hac_lag"),
                        "need": bar.get("min_paired_t", bar.get("min_t_hac", bar.get("min_t_day"))),
                        "min_trades": bar["min_trades"], "min_days": bar["min_days"], "failed_alphas": failed,
                        "verdict": d.get("verdict"), "rule": d.get("rule"), "checkpoint": d.get("checkpoint"),
                        "diff": self.research.slot_diff(c)})
        return out

    def alpha_overview(self) -> dict:
        """One row per registered alpha for /alphas and the research page: switched on in the champion, the challengers
        that change or trade it, its evidence (verdict, stage, numbers), its live record on the primary engine and its
        kill-switch state; plus the learning status that explains a champion without any alpha."""
        prim = self.primary_engine.name if self.primary_engine else "paper"
        ks = {k["alpha"]: k for k in self.killswitch.champion_state() if k.get("engine") == prim}
        killed: dict[str, dict] = {}
        for k in self.killed_alphas():
            killed.setdefault(k["alpha"], k)
        trust = {r["alpha"]: r.get("trust") for r in self.bandit.snapshot()}
        trades: dict[str, list[dict]] = {}
        for t in self.store.load_trades(engine=prim):
            trades.setdefault(t.get("alpha") or "?", []).append(t)
        judged = evidence.registry().get("alphas", {})
        rows = []
        for a in self.params.alphas:
            chs = []
            for c in self.challengers:
                diff = self.research.slot_diff(c)
                if a in (diff.get("enabled") or {}):
                    role = "on" if diff["enabled"][a] else "off"
                elif a in (diff.get("alphas") or {}):
                    role = "tuned"
                elif c.params.enabled.get(a, False):
                    role = "trades"
                else:
                    continue
                chs.append({"name": c.name, "role": role})
            e = evidence.entry(a)
            rows.append({"alpha": a, "enabled": bool(self.params.enabled.get(a, False)), "challengers": chs,
                         "verdict": evidence.verdict(a), "evidence": e, "evidence_stale": e is None and a in judged,
                         "live": r_stats(trades.get(a, [])), "kill_switch": ks.get(a), "killed": killed.get(a),
                         "trust": trust.get(a)})
        return {"engine": prim, "champion": self.params.version, "rows": rows, "learning": self.learning_status()}

    def report_insight(self) -> dict:
        """What the daily report adds to the PnL lines: per-alpha performance on the primary engine (7 days and all
        time), the latest research and discovery cycles, and every challenger's outlook (challenger_outlook)."""
        now = now_ms()
        prim = self.primary_engine.name if self.primary_engine else "paper"
        groups: dict[str, list[dict]] = {}
        for t in self.store.load_trades(engine=prim):
            groups.setdefault(t.get("alpha") or "?", []).append(t)
        perf = []
        for a in sorted(set(groups) | set(self.enabled_alphas())):
            ts = groups.get(a, [])
            wk = [t for t in ts if (t.get("exit_time") or 0) >= now - 7 * MS_DAY]
            perf.append({"alpha": a, "enabled": bool(self.params.enabled.get(a, False)), "week": r_stats(wk),
                         "week_net": sum(float(t.get("pnl") or 0.0) for t in wk), "all": r_stats(ts),
                         "all_net": sum(float(t.get("pnl") or 0.0) for t in ts)})
        last = self.research.last_summary or {}
        alphas = last.get("alphas") or {}
        improved = [a for a, r in alphas.items()
                    if not (r.get("best") or {}).get("is_base") and (r.get("best") or {}).get("score", -1e9) >
                    (r.get("base") or {}).get("score", -1e9)]
        outcome = disc_outcome = ""
        for ev in self.store.load_events(30, topic="research_done"):
            msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
            text = str(msg.get("message") or "")
            if not disc_outcome and text.startswith("알파 발굴"):
                disc_outcome = text
            elif not outcome and "summary" in msg:
                outcome = text
        disc = self.store.get("discovery.last") or {}
        disc_last = int(self.store.get("discovery.last_cycle", 0) or 0)
        return {"engine": prim, "alphas": perf,
                "research": {"last": self.research.last_cycle_ts or None, "running": self.research.running,
                             "seconds": last.get("seconds"), "symbols": len(last.get("symbols") or []),
                             "evaluated": len(alphas), "improved": improved, "outcome": outcome,
                             "failures": self.research.research_failures, "last_error": self.research.last_error,
                             "selection": R.describe_selection(last["selection_detail"])
                             if last.get("selection_detail") else "",
                             "next_due": self.research.next_due_ts()},
                "discovery": {"last": disc.get("ts"), "tested": disc.get("tested", 0), "t_bar": disc.get("t_bar"),
                              "passed": len(disc.get("passed") or []), "seconds": disc.get("seconds"),
                              "outcome": disc_outcome,
                              "next_due": disc_last + self.s.discovery_interval_hours * 3_600_000 if disc_last else None},
                "challengers": self.challenger_outlook()}

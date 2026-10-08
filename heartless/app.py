"""Heartless orchestrator: market data, engines (live + paper champion + challengers), learning, UIs."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from typing import Any

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Candle, PositionStatus, Regime, SymbolInfo, Ticker
from heartless.core.store import Store
from heartless.data.candles import CandleArrays
from heartless.data.extras import MetricsSeries
from heartless.data.features import LIVE_WINDOW, MarketView
from heartless.data.universe import select_universe
from heartless.exchange.binance_rest import BinanceError, BinanceRest
from heartless.exchange.binance_ws import MarketStream
from heartless.exchange.live import LiveAccount
from heartless.exchange.paper import PaperAccount
from heartless.exchange.symbols import parse_exchange_info
from heartless.execution.engine import TradingEngine
from heartless.execution.stats import by_alpha, summarize
from heartless.learning.advisor import Advisor
from heartless.learning.bandit import AlphaBandit
from heartless.learning.research import ChallengerSlot, ResearchManager
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.strategy.regime import detect_regime
from heartless.util.ids import short_id, token_hex
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, local_day_start, next_local_time, now_ms

log = logging.getLogger(__name__)

# a challenger keeps its slot at least this long unless it already has a promotion-sized trade sample: research runs
# every few hours and would otherwise evict the youngest slots before any challenger can reach promotion_min_trades
CHALLENGER_MIN_AGE_MS = 2 * MS_DAY
# how often the scheduler re-runs the history fetch for symbols whose backfill failed
BACKFILL_RETRY_MS = 5 * MS_MINUTE


class Heartless:
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
        self.regimes: dict[str, tuple[Regime, dict]] = {}
        self.bandit = AlphaBandit(list(StrategyParams.default().alphas), store=self.store, engine="shared")
        self.research = ResearchManager(self)
        self.meta = None
        if settings.meta_label:
            from heartless.learning.metalabel import MetaLabeler

            self.meta = MetaLabeler(min_samples=settings.meta_min_samples, store=self.store, key="metalabel")
        self.params: StrategyParams = self.research.load_champion()
        self.challengers: list[ChallengerSlot] = []
        self.engines: dict[str, TradingEngine] = {}
        self.paper_accounts: dict[str, PaperAccount] = {}
        self.live_account: LiveAccount | None = None
        self.market = MarketStream(self.rest.ws_base, self._on_kline, self._on_book, self._on_mark)
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
        self.web = None
        self._backfilling: set[str] = set()
        self.archive = None  # BinanceArchive, created lazily for bulk backfill
        self._backfill_failed: set[str] = set()  # symbols whose history fetch failed; retried by the scheduler
        self._last_backfill_retry = 0
        self._next_report_ts = next_local_time(settings.daily_report_hour, 0, settings.timezone)
        self._last_prune = 0
        self._last_oi_poll = 0
        self._last_universe = 0
        self._last_reconcile = 0
        self._last_equity_tick = 0
        self._last_day_key = local_day_start(now_ms(), settings.timezone)
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
        if topic in ("risk_event", "error", "mode_changed", "promotion", "research_done", "graduation_ready", "universe"):
            self.store.log_event(payload["ts"], "INFO", topic, {k: v for k, v in payload.items() if k not in ("result",)})

    @property
    def primary_engine(self) -> TradingEngine | None:
        """The engine whose activity is reported in detail (live if live, else paper champion)."""
        return self.engines.get("live") or self.engines.get("paper")

    def marks(self) -> dict[str, float]:
        return {s: (t.mark or t.mid or t.last) for s, t in self.tickers.items()}

    # --- startup -------------------------------------------------------------------------------
    async def run(self) -> None:
        log.info("Heartless starting (mode=%s, testnet=%s)", self.mode, self.s.binance_testnet)
        await self.rest.sync_time()
        await self._load_symbols()
        await self.refresh_universe(initial=True)
        await self._backfill(self.universe)
        self._build_views(self.universe)
        await self._create_engines()
        self.market.set_symbols(self.universe)
        self.tasks.append(asyncio.create_task(self.market.run(), name="market-stream"))
        self.tasks.append(asyncio.create_task(self._kline_worker(), name="kline-worker"))
        self.tasks.append(asyncio.create_task(self._scheduler(), name="scheduler"))
        if self.s.telegram_bot_token:
            from heartless.notify.telegram import TelegramBot

            self.telegram = TelegramBot(self)
            self.tasks.append(asyncio.create_task(self.telegram.run(), name="telegram"))
        if self.s.web_enabled:
            from heartless.web.app import WebServer

            self.web = WebServer(self)
            self.tasks.append(asyncio.create_task(self.web.run(), name="web"))
        await self.emit("started", {"message": self.startup_summary()})
        try:
            await asyncio.gather(*self.tasks)
        except asyncio.CancelledError:
            pass

    def startup_summary(self) -> str:
        lines = [f"Heartless 시작 — 모드: {'🔴 LIVE' if self.mode == 'live' else '🟢 PAPER'}",
                 f"유니버스 {len(self.universe)}종목: {', '.join(self.universe)}",
                 f"챔피언 파라미터: {self.params.version}", f"챌린저: {', '.join(c.name for c in self.challengers) or '-'}"]
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
            for s in removed:
                self.views.pop(s, None)
                self.candles.pop(s, None)
            self.market.set_symbols(self.universe)
            await self.emit("universe", {"message": f"유니버스 갱신: +{','.join(added) or '-'} / -{','.join(removed) or '-'}",
                                         "added": added, "removed": removed, "universe": self.universe})

    async def _backfill(self, symbols: list[str]) -> set[str]:
        """Fetch the missing 1m history (and funding) into the store. Returns the symbols whose fetch failed; they
        are remembered in `_backfill_failed` and retried by the scheduler, because a symbol left without history
        would otherwise trade on minutes of data for the whole process lifetime (the stream gap fill only repairs
        holes after the first live bar)."""
        now = now_ms()
        # keep the longer research history in the store (cheap via the archive); live views use history_days
        keep_days = max(self.s.history_days, self.s.data_retention_days if self.s.archive_backfill else self.s.history_days)
        since = now - keep_days * MS_DAY
        sem = asyncio.Semaphore(3)
        failed: set[str] = set()

        async def one(sym: str) -> None:
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
                    if f_start < now - 8 * MS_HOUR:
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

    async def _retry_backfill(self) -> None:
        """Re-run the history fetch for symbols whose startup/universe backfill failed and rebuild their views."""
        self._backfill_failed &= set(self.universe)
        pending = sorted(self._backfill_failed)
        if not pending:
            return
        await self._backfill(pending)
        fixed = [s for s in pending if s not in self._backfill_failed]
        if fixed:
            self._build_views(fixed)
            log.info("backfill retry repaired %s", ", ".join(fixed))

    def _build_views(self, symbols: list[str]) -> None:
        since = now_ms() - self.s.history_days * MS_DAY
        for sym in symbols:
            rows = self.store.load_candles(sym, start=since)
            ca = CandleArrays("1m", capacity=len(rows) + 4096)
            ca.extend(rows)
            self.candles[sym] = ca
            view = MarketView(sym)
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
        self.engines["paper"] = TradingEngine("paper", pa, self.params, self.s, self.symbols, self.store, self.bus,
                                              self.bandit, notify=(self.mode != "live"), meta=self.meta)
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
        acc = LiveAccount(self.rest, "live", self.s.taker_fee)
        acc.set_symbols(self.symbols)
        eng = TradingEngine("live", acc, self.params, self.s, self.symbols, self.store, self.bus, self.bandit, notify=True,
                            meta=self.meta)
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
            # a fresh challenger reuses a slot name: it must not inherit the retired one's record or its risk halts
            self.store.delete_positions(slot.name)
            self.store.delete_trades(slot.name)
            self.store.delete_equity(slot.name)
            self.store.set(f"{slot.name}.risk", None)
        self.paper_accounts[slot.name] = pa
        # challengers consult the shared meta model (fair comparison with the champion) but do not train it
        eng = TradingEngine(slot.name, pa, slot.params, self.s, self.symbols, self.store, self.bus,
                            AlphaBandit(list(slot.params.alphas), store=None), notify=False, meta=self.meta, meta_learn=False)
        self._apply_persisted_pause(eng)
        if restore:
            await self._restore_paper_positions(eng)
        self.engines[slot.name] = eng
        await eng.start()
        if restore:
            await self._rearm_paper_positions(eng)
        if slot not in self.challengers:
            self.challengers.append(slot)

    # --- challenger / champion management ------------------------------------------------------
    async def install_challenger(self, params: StrategyParams, source: str) -> ChallengerSlot | None:
        """Place a new challenger into a free slot, or replace the weakest one."""
        names = [f"challenger-{i + 1}" for i in range(self.s.challengers)]
        used = {c.name: c for c in self.challengers}
        free = [n for n in names if n not in used]
        if free:
            name = free[0]
        else:
            # replace the worst-performing challenger (by net pnl since start) among those that have had a fair
            # trial: a promotion-sized sample, or an age of CHALLENGER_MIN_AGE_MS. Evicting younger slots at every
            # research cycle would churn them before any challenger can reach promotion_min_trades.
            now = now_ms()

            def trades(slot: ChallengerSlot) -> list[dict]:
                return self.store.load_trades(engine=slot.name, since=slot.started)

            def score(slot: ChallengerSlot) -> float:
                st = summarize(trades(slot))
                return st["net"] if st["n"] else -1e-9

            eligible = [c for c in self.challengers
                        if len(trades(c)) >= self.s.promotion_min_trades or now - c.started >= CHALLENGER_MIN_AGE_MS]
            if not eligible:
                log.info("no challenger slot for %s candidate: all %d challengers are still in their trial period",
                         source, len(self.challengers))
                return None
            victim = min(eligible, key=score)
            await self.retire_challenger(victim, reason="새 후보로 교체", silent=True)
            name = victim.name
        slot = ChallengerSlot(name=name, params=params, started=now_ms(), source=source)
        self.store.save_params_version(params.version, params.created, "challenger", source, params.note, params.to_dict(),
                                       {"slot": name, "started": slot.started})
        await self._start_challenger(slot)
        return slot

    async def retire_challenger(self, slot: ChallengerSlot, reason: str, silent: bool = False) -> None:
        eng = self.engines.pop(slot.name, None)
        if eng:
            await eng.close_all("챌린저 종료")
        self.paper_accounts.pop(slot.name, None)
        if slot in self.challengers:
            self.challengers.remove(slot)
        self.store.set_params_role(slot.params.version, "retired")
        if not silent:
            await self.emit("challenger_retired", {"message": f"챌린저 {slot.name} 종료: {reason}", "notify": True})

    async def promote(self, slot: ChallengerSlot, st_ch: dict, st_champ: dict) -> None:
        old = self.params
        new = slot.params.clone(version=short_id("v"), note=f"promoted from {slot.name} ({slot.source})")
        self.store.set_params_role(old.version, "retired")
        self.store.save_params_version(new.version, new.created, "champion", slot.source, new.note, new.to_dict(),
                                       {"challenger": st_ch, "champion": st_champ})
        self.params = new
        for name in ("live", "paper"):
            if name in self.engines:
                self.engines[name].set_params(new)
        await self.retire_challenger(slot, reason="챔피언으로 승격", silent=True)
        msg = (f"🏆 파라미터 승격: {slot.name} → 챔피언 {new.version}\n"
               f"변경 알파: {slot.source}\n"
               f"챌린저 n={st_ch['n']} avgR {st_ch['avg_r']:+.2f} PF {st_ch['profit_factor']:.2f} net {st_ch['net']:+.1f}\n"
               f"기존 챔피언 n={st_champ['n']} avgR {st_champ['avg_r']:+.2f} PF {st_champ['profit_factor']:.2f} net {st_champ['net']:+.1f}")
        await self.emit("promotion", {"message": msg, "version": new.version})

    # --- market data handlers ------------------------------------------------------------------
    async def _on_kline(self, data: dict) -> None:
        k = data.get("k", {})
        if not k.get("x"):
            return
        self._kline_queue.put_nowait((data["s"], Candle.from_ws(k)))

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
            t.ts = int(d.get("E", now_ms()))
            await self._tick(sym, t)

    async def _tick(self, sym: str, t: Ticker) -> None:
        mark = t.mark or t.mid
        if not mark:
            return
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
        ca = self.candles.get(sym)
        if ca is None:
            return
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
        ca.append(c)
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
        await asyncio.to_thread(self._rebuild_view, view, ca, tfs)
        view.seek(c.close_time)
        # a 5m close inside the gap is not flagged by seek() (it closed more than a minute ago): refresh explicitly
        if missing or view.closed("5m") or sym not in self.regimes:
            self.regimes[sym] = detect_regime(view)
        ctx = self._context(sym, c.close_time)
        for eng in list(self.engines.values()):
            try:
                await eng.on_bar(view, ctx)
            except Exception:  # noqa: BLE001
                log.exception("engine %s on_bar failed for %s", eng.name, sym)
        if sym == "BTCUSDT":
            await self._btc_shock_check(view)

    @staticmethod
    def _rebuild_view(view: MarketView, ca: CandleArrays, tfs: list[str]) -> None:
        from heartless.data.candles import resample
        from heartless.data.features import LIVE_WINDOW, FeatureFrame

        for tf in tfs:
            bars = ca if tf == "1m" else resample(ca, tf)
            view.frames[tf] = FeatureFrame(tf, bars.tail(LIVE_WINDOW[tf]))
        view.base = ca

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
        nft = t.next_funding_time or ((now // (8 * MS_HOUR)) + 1) * 8 * MS_HOUR
        return Context(symbol=sym, info=self.symbols[sym], ticker=t, regime=reg, regime_info=info, btc_regime=btc,
                       oi_change=oi_change, now=now, funding_rate=t.funding_rate,
                       minutes_to_funding=max((nft - now) / MS_MINUTE, 0.0), book_imbalance=imb, extras=extras)

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
                if now - self._last_universe >= self.s.universe_refresh_minutes * MS_MINUTE:
                    await self.refresh_universe()
                if self.research.due(now) and not self.research.running:
                    asyncio.create_task(self.research.cycle())
                elif self.research.discovery_due(now) and not self.research.running:
                    asyncio.create_task(self.research.discovery_cycle())
                await self.research.evaluate_challengers()
                await self.research.check_graduation()
                if now >= self._next_report_ts:
                    self._next_report_ts = next_local_time(self.s.daily_report_hour, 0, self.s.timezone, now)
                    asyncio.create_task(self.daily_report())
                day_key = local_day_start(now, self.s.timezone)
                if day_key != self._last_day_key:
                    self._last_day_key = day_key
                    for eng in self.engines.values():
                        eng.reset_daily()
                if now - self._last_prune >= 6 * MS_HOUR:
                    self._last_prune = now
                    self.store.prune_candles(now - (max(self.s.history_days, self.s.data_retention_days) + 5) * MS_DAY)
                self._update_health(now)
            except Exception:  # noqa: BLE001
                log.exception("scheduler iteration failed")
            await asyncio.sleep(5)

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
                fresh = MetricsSeries.from_rows(rows_from_rest(oi, tp, ta, ga, tk))
                merged = fresh if have is None else have.merge(fresh)
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
        self.health = {"market_stream": self.market.connected, "stale_symbols": stale, "queue": self._kline_queue.qsize(),
                       "short_history": short, "backfill_pending": sorted(self._backfill_failed),
                       "user_stream": self.live_account.user_stream.connected if self.live_account else None,
                       "uptime_min": (now - self.started_at) / MS_MINUTE, "used_weight": self.rest.used_weight}

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
        n = 0
        targets = [self.engines[engine]] if engine and engine in self.engines else [e for e in (self.primary_engine,) if e]
        for eng in targets:
            if await eng.close_position(symbol.upper(), "수동 청산(owner)"):
                n += 1
        return n

    async def close_all(self, engine: str | None = None) -> int:
        targets = [self.engines[engine]] if engine and engine in self.engines else [e for e in (self.primary_engine,) if e]
        n = 0
        for eng in targets:
            n += await eng.close_all("전체 수동 청산(owner)")
        return n

    async def kill(self) -> int:
        n = await self.close_all()
        await self.pause("kill switch")
        return n

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
                                                                         "next_due": self.research.last_cycle_ts + self.s.research_interval_minutes * 60_000},
                "challengers": [{"name": c.name, "version": c.params.version, "source": c.source, "started": c.started}
                                for c in self.challengers],
                "graduation": self.research.graduation_status() if self.mode != "live" else None,
                "alphas": self.bandit.snapshot(), "regimes": {s: r[0].value for s, r in self.regimes.items()},
                "web_token": self.web_token, "started_at": self.started_at, "live_capable": self.s.live_capable}

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

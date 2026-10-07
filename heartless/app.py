"""Heartless orchestrator: market data, engines (live + paper champion + challengers), learning, UIs."""
from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from typing import Any

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Candle, Regime, SymbolInfo, Ticker
from heartless.core.store import Store
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
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
        self.regimes: dict[str, tuple[Regime, dict]] = {}
        self.bandit = AlphaBandit(list(StrategyParams.default().alphas), store=self.store, engine="shared")
        self.research = ResearchManager(self)
        self.params: StrategyParams = self.research.load_champion()
        self.challengers: list[ChallengerSlot] = []
        self.engines: dict[str, TradingEngine] = {}
        self.paper_accounts: dict[str, PaperAccount] = {}
        self.live_account: LiveAccount | None = None
        self.market = MarketStream(self.rest.ws_base, self._on_kline, self._on_book, self._on_mark)
        self.advisor = Advisor(settings.anthropic_api_key)
        self.mode: str = self.store.get("mode", settings.mode)
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
        if self.s.web_enabled:
            lines.append(f"웹 대시보드: http://<host>:{self.s.web_port}/?token={self.web_token}")
        return "\n".join(lines)

    async def stop(self) -> None:
        if self._stopping:
            return
        self._stopping = True
        log.info("shutting down")
        await self.market.stop()
        for eng in self.engines.values():
            try:
                await eng.account.stop()
            except Exception:  # noqa: BLE001
                pass
        self.research.shutdown()
        for t in self.tasks:
            t.cancel()
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

    async def _backfill(self, symbols: list[str]) -> None:
        now = now_ms()
        since = now - self.s.history_days * MS_DAY
        sem = asyncio.Semaphore(3)

        async def one(sym: str) -> None:
            async with sem:
                self._backfilling.add(sym)
                try:
                    lo, hi, n = self.store.candle_range(sym)
                    start = since if (hi is None or hi < since) else hi + MS_MINUTE
                    if start < now - MS_MINUTE:
                        rows = await self.rest.klines_range(sym, start, now, "1m")
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
                finally:
                    self._backfilling.discard(sym)

        await asyncio.gather(*(one(s) for s in symbols))

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
                                              self.bandit, notify=(self.mode != "live"))
        await self._restore_paper_positions(self.engines["paper"])
        await self.engines["paper"].start()
        # challengers
        for slot in self.research.load_challengers():
            await self._start_challenger(slot, restore=True)
        # live
        if self.mode == "live":
            await self._start_live()

    async def _restore_paper_positions(self, eng: TradingEngine) -> None:
        """Recreate open paper positions inside the simulated account after a restart."""
        acc: PaperAccount = eng.account  # type: ignore[assignment]
        for p in self.store.load_open_positions(eng.name):
            if p.status.value == "OPEN" and p.qty > 0:
                from heartless.exchange.paper import PaperPosition

                acc.positions[p.symbol] = PaperPosition(qty=p.qty * p.side.sign, entry=p.entry_price, leverage=p.leverage)
                acc.orders.clear()

    async def _start_live(self) -> None:
        if self.live_account is not None:
            return
        self.live_account = LiveAccount(self.rest, "live", self.s.taker_fee)
        self.live_account.set_symbols(self.symbols)
        await self.live_account.start()
        eng = TradingEngine("live", self.live_account, self.params, self.s, self.symbols, self.store, self.bus,
                            self.bandit, notify=True)
        self.engines["live"] = eng
        await eng.start()
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
        self.paper_accounts[slot.name] = pa
        eng = TradingEngine(slot.name, pa, slot.params, self.s, self.symbols, self.store, self.bus,
                            AlphaBandit(list(slot.params.alphas), store=None), notify=False)
        if restore:
            await self._restore_paper_positions(eng)
        else:
            self.store.delete_positions(slot.name)
            self.store.delete_trades(slot.name)
            self.store.delete_equity(slot.name)
        self.engines[slot.name] = eng
        await eng.start()
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
            # replace the worst-performing challenger (by net pnl since start)
            def score(slot: ChallengerSlot) -> float:
                st = summarize(self.store.load_trades(engine=slot.name, since=slot.started))
                return st["net"] if st["n"] else -1e-9
            victim = min(self.challengers, key=score)
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
        for acc in self.paper_accounts.values():
            await acc.on_ticker(t)
        for eng in self.engines.values():
            pos = eng.positions.get(sym)
            if pos is not None:
                pos.extra["last_mark"] = mark
                await eng.on_tick(sym, mark, t.bid, t.ask)

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
        if ca.n and c.open_time - int(ca.open_time[ca.n - 1]) > MS_MINUTE and sym not in self._backfilling:
            try:
                missing = await self.rest.klines_range(sym, int(ca.open_time[ca.n - 1]) + MS_MINUTE, c.open_time - 1, "1m")
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
        minute = (c.open_time // MS_MINUTE) % 60
        tfs = ["1m"]
        if (c.open_time // MS_MINUTE + 1) % 5 == 0:
            tfs.append("5m")
        if (c.open_time // MS_MINUTE + 1) % 15 == 0:
            tfs.append("15m")
        if minute == 59:
            tfs.append("1h")
        await asyncio.to_thread(self._rebuild_view, view, ca, tfs)
        view.seek(c.close_time)
        if view.closed("5m") or sym not in self.regimes:
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
        oi = self.oi_hist.get(sym)
        oi_change = None
        if oi and len(oi) >= 2:
            old = next((v for ts, v in oi if ts <= now - 4 * MS_HOUR), oi[0][1])
            if old:
                oi_change = oi[-1][1] / old - 1
        nft = t.next_funding_time or ((now // (8 * MS_HOUR)) + 1) * 8 * MS_HOUR
        return Context(symbol=sym, info=self.symbols[sym], ticker=t, regime=reg, regime_info=info, btc_regime=btc,
                       oi_change=oi_change, now=now, funding_rate=t.funding_rate,
                       minutes_to_funding=max((nft - now) / MS_MINUTE, 0.0), book_imbalance=imb)

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
                if now - self._last_oi_poll >= 15 * MS_MINUTE:
                    self._last_oi_poll = now
                    asyncio.create_task(self._poll_open_interest())
                if now - self._last_universe >= self.s.universe_refresh_minutes * MS_MINUTE:
                    await self.refresh_universe()
                if self.research.due(now) and not self.research.running:
                    asyncio.create_task(self.research.cycle())
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
                    self.store.prune_candles(now - (self.s.history_days + 10) * MS_DAY)
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
        for name, acc in self.paper_accounts.items():
            eng = self.engines.get(name)
            if eng is None:
                continue
            for sym, pos in list(eng.positions.items()):
                if pos.status.value != "OPEN":
                    continue
                pp = acc.positions.get(sym)
                if pp is None or abs(pp.qty) <= 0:
                    mark = self.marks().get(sym) or pos.entry_price
                    await eng._finalize(pos, mark, now_ms(), "시뮬 계정과 동기화")

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
        self.health = {"market_stream": self.market.connected, "stale_symbols": stale, "queue": self._kline_queue.qsize(),
                       "user_stream": self.live_account.user_stream.connected if self.live_account else None,
                       "uptime_min": (now - self.started_at) / MS_MINUTE, "used_weight": self.rest.used_weight}

    # --- control API (Telegram / web) ----------------------------------------------------------
    async def pause(self, reason: str = "owner") -> None:
        for eng in self.engines.values():
            eng.risk.state.paused = True
        self.store.set("paused", True)
        await self.emit("control", {"message": f"⏸ 신규 진입 일시정지 ({reason}). 기존 포지션은 계속 관리됩니다"})

    async def resume(self) -> None:
        for eng in self.engines.values():
            eng.risk.state.paused = False
            eng.risk.state.halt_reason = ""
        self.store.set("paused", False)
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
            await self._start_live()
            self.mode = "live"
            self.store.set("mode", "live")
            await self.emit("mode_changed", {"message": "🔴 LIVE 모드 전환: 실제 자금으로 거래를 시작합니다", "mode": "live"})
            return "라이브 모드로 전환했습니다"
        # live -> paper: flatten and remove the live engine
        eng = self.engines.get("live")
        n = 0
        if eng:
            eng.entries_enabled = False
            n = await eng.close_all("모드 전환(paper)")
            await asyncio.sleep(1.0)
            self.engines.pop("live", None)
            try:
                await eng.account.stop()
            except Exception:  # noqa: BLE001
                pass
            self.live_account = None
        self.mode = "paper"
        self.store.set("mode", "paper")
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
        return {"mode": self.mode, "paused": prim.risk.state.paused if prim else False, "universe": self.universe,
                "params_version": self.params.version, "engines": engines, "primary": prim.name if prim else None,
                "today": st_today, "health": self.health, "research": {"last_cycle": self.research.last_cycle_ts,
                                                                         "running": self.research.running,
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

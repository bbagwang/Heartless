"""Event-driven backtester that reuses the live TradingEngine + PaperAccount on stored 1m candles."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from heartless.config import Settings
from heartless.core.models import Candle, Regime, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.exchange.base import OrderResult
from heartless.exchange.paper import PaperAccount, PaperOrder
from heartless.execution.engine import TradingEngine
from heartless.execution.stats import summarize
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.strategy.regime import detect_regime
from heartless.util.timeutil import MS_HOUR, MS_MINUTE

log = logging.getLogger(__name__)
FUNDING_INTERVAL = 8 * MS_HOUR


@dataclass
class BacktestResult:
    stats: dict
    trades: list[dict]
    equity_curve: list[tuple[int, float]]
    final_equity: float
    params_version: str
    alpha: str | None = None
    skipped: dict = field(default_factory=dict)


class BarPaperAccount(PaperAccount):
    """PaperAccount for the bar-driven backtest.

    The engine evaluates bar i only once it has closed, so an order it sends can at the earliest execute on the
    book that follows that close, i.e. at bar i+1's open. A non-reduce-only MARKET order (entries, including the
    post-only -> market fallback) therefore rests as a NEW PaperOrder until the next on_bar for its symbol and
    fills at that bar's open (+ half spread + slippage) before the bar's triggers are evaluated, instead of at the
    close that produced the signal. Reduce-only market orders (exits, the final flatten) stay synchronous, so an
    open position is always flattened on the last bar. A resting entry that is cancelled (close_all / cancel
    of a pending position) or never sees another bar simply never fills."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.defer_market = True
        self._pending_market: dict[str, list[PaperOrder]] = {}

    async def market_order(self, symbol: str, side: str, qty: float, reduce_only: bool = False,
                           client_id: str = "") -> OrderResult:
        if reduce_only or not self.defer_market:
            return await super().market_order(symbol, side, qty, reduce_only=reduce_only, client_id=client_id)
        oid = str(next(self._ids))
        if self._ref(symbol, side) <= 0:
            return OrderResult(oid, client_id, "REJECTED", raw={"msg": "no price"})
        o = PaperOrder(oid, client_id, symbol, side, qty, None, False, False, created=self.now())
        self.orders[oid] = o
        self._pending_market.setdefault(symbol, []).append(o)
        return OrderResult(oid, client_id, "NEW")

    async def on_bar(self, symbol: str, c: Candle) -> None:
        queue = self._pending_market.pop(symbol, None)
        if queue:
            t = self.tickers.get(symbol) or Ticker(symbol)
            half = self.spread_bps / 2e4
            t.mark = t.last = c.open
            t.bid, t.ask = c.open * (1 - half), c.open * (1 + half)
            t.ts = c.open_time
            self.tickers[symbol] = t
            for o in queue:
                if o.status != "NEW" or self.orders.get(o.order_id) is not o:
                    continue  # cancelled (or the account was reset) while resting
                ref = t.ask if o.side == "BUY" else t.bid
                slip = self._slip(symbol, o.qty * ref)
                px = ref * (1 + slip) if o.side == "BUY" else ref * (1 - slip)
                await self._fill_order(o, px, maker=False, ts=c.open_time)
        await super().on_bar(symbol, c)


class Backtester:
    def __init__(self, settings: Settings, symbols: dict[str, SymbolInfo], candles: dict[str, CandleArrays],
                 funding: dict[str, list[tuple[int, float]]] | None = None):
        self.s = settings
        self.symbols = symbols
        self.candles = {s: c for s, c in candles.items() if c.n > 300}
        self.funding = funding or {}
        self.views: dict[str, MarketView] = {}
        self.now = 0
        self._timeline: np.ndarray | None = None
        self._prepare()

    def _prepare(self) -> None:
        all_times = []
        for sym, ca in self.candles.items():
            v = MarketView(sym)
            v.rebuild(ca, live=False)
            self.views[sym] = v
            all_times.append(ca.view("close_time"))
        self._timeline = np.unique(np.concatenate(all_times)) if all_times else np.array([], dtype=np.int64)
        self._fund_idx: dict[str, np.ndarray] = {}
        for sym, rows in self.funding.items():
            if rows:
                self._fund_idx[sym] = np.array([r[0] for r in rows], dtype=np.int64)

    def _funding_rate(self, sym: str, t: int) -> float:
        """Settled-rate lookup: the rate of the most recent settlement at or before `t` (what apply_funding charges)."""
        rows = self.funding.get(sym)
        if not rows:
            return 0.0
        idx = int(np.searchsorted(self._fund_idx[sym], t, side="right")) - 1
        return rows[idx][1] if idx >= 0 else 0.0

    def _upcoming_funding_rate(self, sym: str, t: int) -> float:
        """What the live markPrice stream's `r` shows at `t`: the rate for the NEXT settlement (Context.funding_rate).

        Binance publishes `r` as a running estimate that converges to the charged value over the 8h window; the
        stored history only has the final values, so the estimate is approximated by a linear blend from the last
        settled rate (window start) to the rate actually charged at the next settlement (window end). Hence the
        funding-avoidance exit (<= 2 min before settlement) is evaluated against the rate that is then charged,
        while the early window does not see the final value outright. Without a next row (end of history / gap)
        the settled rate is used."""
        rows = self.funding.get(sym)
        if not rows:
            return 0.0
        idx = int(np.searchsorted(self._fund_idx[sym], t, side="right"))  # first row strictly after t
        prev = rows[idx - 1][1] if idx >= 1 else 0.0
        next_due = (t // FUNDING_INTERVAL + 1) * FUNDING_INTERVAL
        if idx >= len(rows) or rows[idx][0] >= next_due + FUNDING_INTERVAL // 2:
            return prev
        w = (t % FUNDING_INTERVAL) / FUNDING_INTERVAL
        return (1 - w) * prev + w * rows[idx][1]

    def run(self, params: StrategyParams, start: int | None = None, end: int | None = None, only_alpha: str | None = None,
            initial_balance: float = 10_000.0, bandit: AlphaBandit | None = None) -> BacktestResult:
        return asyncio.run(self.arun(params, start, end, only_alpha, initial_balance, bandit))

    async def arun(self, params: StrategyParams, start: int | None = None, end: int | None = None,
                   only_alpha: str | None = None, initial_balance: float = 10_000.0,
                   bandit: AlphaBandit | None = None) -> BacktestResult:
        tl = self._timeline
        if tl is None or len(tl) == 0:
            return BacktestResult(summarize([]), [], [], initial_balance, params.version, only_alpha)
        lo = int(np.searchsorted(tl, start if start is not None else tl[0]))
        hi = int(np.searchsorted(tl, end if end is not None else tl[-1], side="right"))
        if lo >= hi:  # start after the last candle (tl[lo] would raise) or end before start: nothing to simulate
            stats = summarize([])
            stats.update(final_equity=initial_balance, return_pct=0.0, max_dd_pct=0.0)
            return BacktestResult(stats, [], [], initial_balance, params.version, only_alpha)
        account = BarPaperAccount(name="bt", initial_balance=initial_balance, taker_fee=self.s.taker_fee,
                                  maker_fee=self.s.maker_fee, clock=lambda: self.now)
        account.set_symbols(self.symbols)
        engine = TradingEngine("bt", account, params, self.s, self.symbols, store=None, bus=None,
                               bandit=bandit or AlphaBandit(list(params.alphas), seed=7), only_alpha=only_alpha,
                               clock=lambda: self.now, persist=False)
        self.now = int(tl[lo])
        await engine.start()
        # per-symbol cursors into candle arrays
        cursors = {s: int(np.searchsorted(ca.view("close_time"), tl[lo])) for s, ca in self.candles.items()}
        regimes: dict[str, tuple[Regime, dict]] = {s: (Regime.RANGE, {}) for s in self.candles}
        btc_regime: Regime | None = None
        equity_curve: list[tuple[int, float]] = []
        peak = initial_balance
        dd_pct = 0.0
        last_funding_bucket = tl[lo] // FUNDING_INTERVAL
        for i in range(lo, hi):
            t = int(tl[i])
            self.now = t
            # funding settlement at 00:00 / 08:00 / 16:00 UTC
            bucket = t // FUNDING_INTERVAL
            if bucket != last_funding_bucket:
                last_funding_bucket = bucket
                for sym in self.candles:
                    rate = self._funding_rate(sym, t)
                    if rate:
                        await account.apply_funding(sym, rate, account.mark(sym), t)
            for sym, ca in self.candles.items():
                ci = cursors[sym]
                if ci >= ca.n or int(ca.close_time[ci]) != t:
                    continue
                cursors[sym] = ci + 1
                c = ca.candle_at(ci)
                await account.on_bar(sym, c)
                view = self.views[sym]
                view.seek(t)
                if view.closed("5m") or sym not in regimes:
                    regimes[sym] = detect_regime(view)
                    if sym == "BTCUSDT":
                        btc_regime = regimes[sym][0]
                reg, reg_info = regimes[sym]
                half = account.spread_bps / 2e4
                tk = Ticker(sym, bid=c.close * (1 - half), ask=c.close * (1 + half), mark=c.close, last=c.close, ts=t,
                            funding_rate=self._upcoming_funding_rate(sym, t), next_funding_time=(bucket + 1) * FUNDING_INTERVAL)
                ctx = Context(symbol=sym, info=self.symbols[sym], ticker=tk, regime=reg, regime_info=reg_info,
                              btc_regime=btc_regime if sym != "BTCUSDT" else None, oi_change=None, now=t,
                              funding_rate=tk.funding_rate, minutes_to_funding=((bucket + 1) * FUNDING_INTERVAL - t) / MS_MINUTE)
                pos = engine.positions.get(sym)
                if pos is not None:
                    pos.extra["last_mark"] = c.close
                await engine.on_bar(view, ctx)
            # risk anchors / loss halts / drawdown are observed every bar (live: on_equity_tick every 30 s), so a
            # limit breached mid-hour blocks entries from the next bar on; the stored curve stays hourly
            eq = account.equity()
            engine._equity_cache = (t, eq)
            for _ in engine.risk.observe_equity(eq, t):
                pass
            peak = max(peak, eq)
            dd_pct = max(dd_pct, (peak - eq) / peak * 100 if peak else 0.0)
            if i % 60 == 0:
                equity_curve.append((t, eq))
        # flatten at the end so every trade is accounted for
        await engine.close_all("backtest end")
        trades = [r.__dict__ if not hasattr(r, "to_dict") else r.to_dict() for r in engine.closed]
        for tr in trades:
            if isinstance(tr.get("alphas"), str):
                import json

                tr["alphas"] = json.loads(tr["alphas"])
        stats = summarize(trades)
        stats["final_equity"] = account.equity()
        stats["return_pct"] = (account.equity() / initial_balance - 1) * 100
        peak = initial_balance
        dd_pct = 0.0
        for _, eq in equity_curve:
            peak = max(peak, eq)
            dd_pct = max(dd_pct, (peak - eq) / peak * 100 if peak else 0.0)
        stats["max_dd_pct"] = dd_pct
        return BacktestResult(stats, trades, equity_curve, account.equity(), params.version, only_alpha,
                              skipped=dict(engine.stats.skipped))


def load_backtester(settings: Settings, store, symbols: dict[str, SymbolInfo], since: int, until: int | None = None,
                    symbol_list: list[str] | None = None) -> Backtester:
    """Build a Backtester from candles stored in the DB."""
    syms = symbol_list or store.candle_symbols()
    candles: dict[str, CandleArrays] = {}
    funding: dict[str, list[tuple[int, float]]] = {}
    for s in syms:
        if s not in symbols:
            continue
        rows = store.load_candles(s, start=since, end=until)
        if len(rows) < 500:
            continue
        ca = CandleArrays("1m", capacity=len(rows) + 16)
        ca.extend(rows)
        candles[s] = ca
        # one interval past `until` so the last bucket's upcoming (next-settlement) rate is known to the Context
        funding[s] = store.load_funding(s, since - FUNDING_INTERVAL, until + FUNDING_INTERVAL if until else 2**62)
    return Backtester(settings, symbols, candles, funding)

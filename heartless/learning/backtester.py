"""Event-driven backtester that reuses the live TradingEngine + PaperAccount on stored 1m candles."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from heartless.config import Settings
from heartless.core.models import Candle, Regime, SymbolInfo, Ticker
from heartless.data.candles import COLUMNS, CandleArrays, resample
from heartless.data.context import EMPTY, ContextTable
from heartless.data.extras import AVAILABLE_AFTER_MS, MetricsSeries
from heartless.data.features import LONG_TIMEFRAMES, TIMEFRAMES, FeatureFrame, MarketView
from heartless.exchange.base import OrderResult
from heartless.exchange.orderbook import proxy_from_frame
from heartless.exchange.paper import PaperAccount, PaperOrder
from heartless.execution.engine import TradingEngine
from heartless.execution.stats import summarize
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.strategy.regime import detect_regime
from heartless.util.timeutil import MS_HOUR, MS_MINUTE

log = logging.getLogger(__name__)
# The longest settlement interval Binance uses: most symbols settle every 8 h, some every 4 h, and 1-2 h while the rate
# sits at its cap (SOLUSDT settled every 2-4 h in November 2022). The settlements themselves are charged at their stored
# times (funding_schedule); this constant only bounds the funding history loaded around a window and marks a hole in
# the stored settlements (a spacing longer than it).
FUNDING_INTERVAL = 8 * MS_HOUR
_NO_TIMES = np.zeros(0, dtype=np.int64)


def funding_schedule(times: np.ndarray, ts: int) -> tuple[int, int]:
    """(next settlement time after `ts`, settlement interval) from a symbol's stored settlement times (ascending): the
    next stored settlement and its spacing from the one before, as the exchange announces them live (premiumIndex
    nextFundingTime, fundingInfo fundingIntervalHours). Past the stored span, or across a hole in it (a spacing over
    FUNDING_INTERVAL), the settlements continue on the last interval from the last settlement before `ts`, and on the
    8-hour UTC grid without any. The same rule as the replay market (heartless/replay/market.py funding_schedule), so a
    backtest, a replay and the live bot see one schedule; for 8-hourly settlements on the UTC grid it is the grid."""
    t = times
    n = len(t)
    i = int(np.searchsorted(t, ts, side="right"))  # first settlement strictly after ts
    if 1 <= i < n and int(t[i]) - int(t[i - 1]) <= FUNDING_INTERVAL:
        return int(t[i]), int(t[i]) - int(t[i - 1])
    if i == 0:
        if n >= 2 and int(t[1]) - int(t[0]) <= FUNDING_INTERVAL and int(t[0]) - ts <= int(t[1]) - int(t[0]):
            return int(t[0]), int(t[1]) - int(t[0])
        return (ts // FUNDING_INTERVAL + 1) * FUNDING_INTERVAL, FUNDING_INTERVAL
    last = int(t[i - 1])
    iv = int(t[i - 1]) - int(t[i - 2]) if i >= 2 else FUNDING_INTERVAL
    iv = iv if 0 < iv <= FUNDING_INTERVAL else FUNDING_INTERVAL
    return last + ((ts - last) // iv + 1) * iv, iv


# The bar columns a FeatureFrame carries as views of its bars (no computation); every other column is an indicator.
_RAW_COLUMNS = ("open", "high", "low", "close", "volume", "quote_volume")
# Frames whose indicator columns a backtest computes on first read: over a year plus a 92-day warm-up the eager 1m and 5m
# frames held about 330 of a symbol's 400 MB of frames. The 15m .. 1d frames stay eager: about 30 MB per symbol, and a
# run reads many of their columns at different bars (each new read would recompute the frame).
_LAZY_TIMEFRAMES = ("1m", "5m")
# A lazy frame computed more often than this keeps every column from then on: a run that reads it widely pays at most
# _KEEP_ALL_AFTER + 1 computations of it, not one per newly read column.
_KEEP_ALL_AFTER = 2
# Candles are read from the store this much at a time: one query over a year held about 0.5 GB of row objects per
# symbol at once (sqlite rows + Candle objects) on top of the arrays they fill.
_LOAD_CHUNK_MS = 30 * 86_400_000


class _LazyState:
    """What the lazy frames of one Backtester share: the indicator columns read so far per timeframe (a symbol's
    first recompute keeps every column another symbol already needed), and the frames holding a spare full
    computation until the current bar is done (release, called by the bar loop)."""

    __slots__ = ("reads", "spared")

    def __init__(self) -> None:
        self.reads: dict[str, set[str]] = {}
        self.spared: list[_LazyColumns] = []

    def release(self) -> None:
        for cols in self.spared:
            cols.spare = None
        self.spared.clear()


class _LazyColumns(dict):
    """A backtest frame's column mapping holding the bar columns and only the indicator columns the run has read.

    A missing indicator is filled from FeatureFrame computed again on the same bars: the same code on the same input,
    so every value is the one the eager frame held. The recompute is kept as a spare until the bar is done (the
    regime reads two 5m columns in a row: one computation), and the columns kept are every one of that timeframe
    read so far by any symbol; after _KEEP_ALL_AFTER recomputes the frame keeps them all. Unknown names behave as in
    a plain dict (get -> default, [] -> KeyError)."""

    __slots__ = ("tf", "bars", "names", "state", "spare", "computes")

    def __init__(self, tf: str, bars: CandleArrays, names: tuple[str, ...], state: _LazyState):
        super().__init__((k, bars.view(k)) for k in _RAW_COLUMNS)
        self.tf, self.bars, self.names, self.state = tf, bars, names, state
        self.spare: dict | None = None
        self.computes = 0

    def __missing__(self, name: str) -> np.ndarray:
        if name not in self.names:
            raise KeyError(name)
        reads = self.state.reads.setdefault(self.tf, set())
        reads.add(name)
        if self.spare is None:
            self.spare = FeatureFrame(self.tf, self.bars).f
            self.computes += 1
            self.state.spared.append(self)
        keep = self.spare if self.computes > _KEEP_ALL_AFTER else reads
        for k in keep:
            if not dict.__contains__(self, k) and k in self.spare:
                dict.__setitem__(self, k, self.spare[k])
        return dict.__getitem__(self, name)

    def get(self, name: str, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def __contains__(self, name) -> bool:
        return name in self.names

    def __iter__(self):
        return iter(self.names)

    def __len__(self) -> int:
        return len(self.names)

    def keys(self):
        return list(self.names)

    def values(self):
        return [self[k] for k in self.names]

    def items(self):
        return [(k, self[k]) for k in self.names]


class _LazyFrame(FeatureFrame):
    """FeatureFrame over a backtest's whole history whose indicator columns are computed on first read (see
    Backtester._build_view)."""

    def __init__(self, tf: str, bars: CandleArrays, names: tuple[str, ...], state: _LazyState):
        self._lazy = (names, state)
        super().__init__(tf, bars)

    def _compute(self, b: CandleArrays) -> None:
        names, state = self._lazy
        self.f = _LazyColumns(self.tf, b, names if b.n else _RAW_COLUMNS, state)  # empty: bar columns only, as eager


def _union_sorted(arrays: list[np.ndarray]) -> np.ndarray:
    """np.unique(np.concatenate(arrays)) folded one array at a time: the same sorted values, without a copy of every
    symbol's close times at once (20 symbols x 3 years: 250 MB, and unique's sort copies of it)."""
    if not arrays:
        return np.array([], dtype=np.int64)
    out = np.unique(arrays[0])
    for a in arrays[1:]:
        out = np.union1d(out, a)
    return out


def _frame_names(ca: CandleArrays) -> tuple[str, ...]:
    """Every column a FeatureFrame holds, in its order (computed on two bars: the set does not depend on the data)."""
    return tuple(FeatureFrame("1m", ca.slice(0, min(ca.n, 2))).f)


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
                 funding: dict[str, list[tuple[int, float]]] | None = None, metrics: dict | None = None,
                 market: ContextTable | None = None):
        self.s = settings
        self.symbols = symbols
        self.candles = {s: c for s, c in candles.items() if c.n > 300}
        self.funding = funding or {}
        self.metrics = metrics or {}  # symbol -> MetricsSeries (open interest / long-short ratios)
        # stored market context rows (data/context.py) -> Context.market; None: alphas see an empty mapping
        self.market = market
        self._extras_cache: dict[str, tuple[int, dict]] = {}
        self._extras_bounds: dict[str, tuple[int, int, int]] = {}  # symbol -> (usable row, valid from, valid until)
        self._fund_cache: dict[str, tuple] = {}  # symbol -> cached settlement neighbourhood (_funding_view)
        self.views: dict[str, MarketView] = {}
        self._lazy = _LazyState()
        self.now = 0
        self._timeline: np.ndarray | None = None
        self._prepare()

    def _prepare(self) -> None:
        all_times = []
        for sym, ca in self.candles.items():
            self.views[sym] = self._build_view(sym, ca)
            all_times.append(ca.view("close_time"))
        self._timeline = _union_sorted(all_times)
        self._fund_idx: dict[str, np.ndarray] = {}
        for sym, rows in self.funding.items():
            if rows:
                self._fund_idx[sym] = np.array([r[0] for r in rows], dtype=np.int64)

    def _build_view(self, sym: str, ca: CandleArrays) -> MarketView:
        """MarketView.rebuild(ca, live=False), except that the 1m and 5m frames compute an indicator column only when
        the run first reads it (_LazyColumns, docs/PORTFOLIO.md section 4.4).

        Eagerly every frame held all 55 indicator columns over the whole loaded history: about 400 MB per symbol for a
        year plus a 92-day warm-up, 276 MB of it in the 1m frame and 55 MB in the 5m frame, so a 20-symbol joint
        backtest needed about 9 GB. No alpha, the regime or the engine reads a 1m indicator in a backtest (the 1m bars
        are read), and of the 5m frame the regime and the engine read atr and atr_rank; the 1m indicators also cost a
        few seconds of computation per symbol. The values are the eager ones: each column is computed by FeatureFrame
        over the same bars. The 15m .. 1d frames are built as rebuild builds them (the long ones up front too: the
        engine looks a position's timeframe up in view.frames)."""
        v = MarketView(sym)
        v.base = ca
        names = _frame_names(ca)
        for tf in TIMEFRAMES:
            bars = ca if tf == "1m" else resample(ca, tf)
            v.frames[tf] = (_LazyFrame(tf, bars, names, self._lazy) if tf in _LAZY_TIMEFRAMES
                            else FeatureFrame(tf, bars))
        for tf in LONG_TIMEFRAMES:
            v.tf(tf)
        return v

    def _make_meta(self):
        if not getattr(self.s, "meta_label", False):
            return None
        from heartless.learning.metalabel import MetaLabeler

        return MetaLabeler(min_samples=getattr(self.s, "meta_min_samples", 120))  # starts empty: walk-forward

    def _extras(self, sym: str, t: int) -> dict:
        """Positioning snapshot at t, recomputed only when a new 5-minute metrics row becomes usable.

        The usable row index (ms._idx(t)) only changes when t passes the next row's availability time, so the bar
        loop keeps that interval per symbol instead of searching the metrics series on every bar."""
        ms = self.metrics.get(sym)
        if ms is None or len(ms) == 0:
            return {}
        b = self._extras_bounds.get(sym)
        if b is not None and b[1] <= t < b[2]:
            idx = b[0]
        else:
            idx = ms._idx(t)
            lo = int(ms.ts[idx]) + AVAILABLE_AFTER_MS if idx >= 0 else -(2**62)
            hi = int(ms.ts[idx + 1]) + AVAILABLE_AFTER_MS if idx + 1 < len(ms) else 2**62
            self._extras_bounds[sym] = (idx, lo, hi)
        cached = self._extras_cache.get(sym)
        if cached is not None and cached[0] == idx and idx >= 0:
            snap = cached[1]
            if snap and not snap.get("stale"):
                snap = dict(snap)
                snap["age_min"] = (t - int(ms.ts[idx]) - 300_000) / 60_000
                if snap["age_min"] > 30:
                    return {"stale": True, "age_min": snap["age_min"]}
            return snap
        snap = ms.snapshot(t)
        self._extras_cache[sym] = (idx, snap)
        return snap

    def _funding_rate(self, sym: str, t: int) -> float:
        """Settled-rate lookup: the rate of the most recent settlement at or before `t` (what apply_funding charges)."""
        rows = self.funding.get(sym)
        if not rows:
            return 0.0
        idx = int(np.searchsorted(self._fund_idx[sym], t, side="right")) - 1
        return rows[idx][1] if idx >= 0 else 0.0

    def _upcoming_funding_rate(self, sym: str, t: int) -> float:
        """What the live markPrice stream's `r` shows at `t`: the rate for the NEXT settlement (Context.funding_rate)."""
        return self._funding_view(sym, t)[0]

    def _funding_view(self, sym: str, t: int) -> tuple[float, int, int]:
        """(upcoming rate, next settlement time, settlement interval) of `sym` at `t`, from its stored settlements.

        The schedule is funding_schedule (each symbol's own interval). Binance publishes `r` as a running estimate that
        converges to the charged value over the interval; the stored history only has the final values, so the
        estimate is approximated by a linear blend from the last settled rate (interval start) to the rate actually
        charged at the next settlement (interval end). Hence the funding-avoidance exit (<= 2 min before settlement) is
        evaluated against the rate that is then charged, while the early interval does not see the final value
        outright. When the next settlement is not stored (end of history / a hole) the settled rate is used."""
        c = self._fund_cache.get(sym)
        if c is None or not (c[0] <= t < c[1]):
            # the neighbourhood (previous settled rate, next settlement and its rate) is constant between two stored
            # settlements and within one scheduled interval, so it is looked up once per change, not on every bar
            times = self._fund_idx.get(sym, _NO_TIMES)
            rows = self.funding.get(sym) or []
            nxt, iv = funding_schedule(times, t)
            i = int(np.searchsorted(times, t, side="right"))  # first settlement strictly after t
            prev = rows[i - 1][1] if i >= 1 else 0.0
            nrate = rows[i][1] if i < len(rows) and abs(int(times[i]) - nxt) <= MS_MINUTE else None
            lo = max(int(times[i - 1]) if i >= 1 else -(2**62), nxt - iv)
            hi = min(int(times[i]) if i < len(times) else 2**62, nxt)
            c = self._fund_cache[sym] = (lo, hi, nxt, iv, prev, nrate)
        _, _, nxt, iv, prev, nrate = c
        if nrate is None:
            return prev, nxt, iv
        w = (t - (nxt - iv)) / iv
        return (1 - w) * prev + w * nrate, nxt, iv

    def _prime_beta(self, beta, start: int) -> None:
        """Seed the allocator's BTC-beta estimator with the hourly closes before `start` (as the live app seeds it from
        its stored candles), at most its look-back; later hours are observed by the engine as the run proceeds."""
        from heartless.execution.allocator import hourly_closes

        for sym, ca in self.candles.items():
            hours, closes = hourly_closes(ca.view("open_time"), ca.view("close"), before=start, limit=beta.hours + 1)
            beta.prime(sym, hours, closes)

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
                                  maker_fee=self.s.maker_fee, clock=lambda: self.now,
                                  slippage_bps=getattr(self.s, "backtest_slippage_bps", 1.5))
        account.set_symbols(self.symbols)
        engine = TradingEngine("bt", account, params, self.s, self.symbols, store=None, bus=None,
                               bandit=bandit or AlphaBandit(list(params.alphas), seed=7), only_alpha=only_alpha,
                               clock=lambda: self.now, persist=False, meta=self._make_meta())
        self.now = int(tl[lo])
        await engine.start()
        # per-symbol cursors into candle arrays
        cursors = {s: int(np.searchsorted(ca.view("close_time"), tl[lo])) for s, ca in self.candles.items()}
        regimes: dict[str, tuple[Regime, dict]] = {s: (Regime.RANGE, {}) for s in self.candles}
        btc_regime: Regime | None = None
        equity_curve: list[tuple[int, float]] = []
        peak = initial_balance
        dd_pct = 0.0
        # funding is charged at every stored settlement of each symbol (its own schedule: 8 h, or 4 / 2 / 1 h while
        # Binance settles it more often), at that settlement's rate, on the first bar closing at or after it; for
        # 8-hourly settlements on the UTC grid that is the first bar of 00:00 / 08:00 / 16:00 as before
        fund_ptr = {s: int(np.searchsorted(self._fund_idx[s], tl[lo], side="right")) for s in self.candles
                    if s in self._fund_idx}
        next_fund = min((int(self._fund_idx[s][k]) for s, k in fund_ptr.items() if k < len(self._fund_idx[s])),
                        default=2**62)
        if engine.allocator is not None:
            self._prime_beta(engine.allocator.beta, int(tl[lo]))
        # F7: with ORDERBOOK_STREAM on, ctx.book is the conservative 1m-bar proxy the replay serves (ProxyBooks); off
        # (the default, as in live trading without the stream) it stays EMPTY_BOOK and nothing is computed
        book_on = bool(getattr(self.s, "orderbook_stream", False))
        for i in range(lo, hi):
            t = int(tl[i])
            self.now = t
            if t >= next_fund:
                next_fund = 2**62
                for sym in self.candles:
                    k = fund_ptr.get(sym)
                    if k is None:
                        continue
                    times, rows = self._fund_idx[sym], self.funding[sym]
                    while k < len(times) and int(times[k]) <= t:
                        rate = rows[k][1]
                        if rate:
                            await account.apply_funding(sym, rate, account.mark(sym), t)
                        k += 1
                    fund_ptr[sym] = k
                    if k < len(times):
                        next_fund = min(next_fund, int(times[k]))
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
                fr, nft, f_iv = self._funding_view(sym, t)
                tk = Ticker(sym, bid=c.close * (1 - half), ask=c.close * (1 + half), mark=c.close, last=c.close, ts=t,
                            funding_rate=fr, next_funding_time=nft)
                extras = self._extras(sym, t)
                oi4 = extras.get("oi_chg_4h") if extras else None
                ctx = Context(symbol=sym, info=self.symbols[sym], ticker=tk, regime=reg, regime_info=reg_info,
                              btc_regime=btc_regime if sym != "BTCUSDT" else None,
                              oi_change=oi4 if oi4 is not None and oi4 == oi4 else None, now=t,
                              funding_rate=fr, minutes_to_funding=(nft - t) / MS_MINUTE,
                              funding_interval_min=f_iv / MS_MINUTE,
                              extras=extras, market=self.market.view(sym, t) if self.market is not None else EMPTY)
                if book_on:
                    ctx.book = proxy_from_frame(view.frames["1m"], t, self.s.orderbook_depth_bps,
                                                self.symbols[sym].tick_size)
                pos = engine.positions.get(sym)
                if pos is not None:
                    pos.extra["last_mark"] = c.close
                await engine.on_bar(view, ctx)
            if engine.allocator is not None:
                # P3: every symbol's bar at t has been processed: rank and admit this close's entry candidates
                await engine.flush_entries()
            if self._lazy.spared:
                self._lazy.release()  # a frame recomputed for this bar's first reads keeps only the columns read
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
        self._lazy.release()
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
                    symbol_list: list[str] | None = None, warmup_ms: int = 0, lookahead: bool = True,
                    market: ContextTable | str | None = "auto") -> Backtester:
    """Build a Backtester from candles stored in the DB.

    Window use (an evaluation over [since, until], e.g. `heartless backtest --start/--end`): `warmup_ms` also loads
    that much history before `since` so indicators are warm when the run starts at `since` (pass it as `arun(start=)`);
    `lookahead=False` reads nothing stamped after `until` - not even the next settlement's funding rate, which the
    default loads one interval past `until` - so a window that ends at a sealed period never touches it. With the
    defaults (no warm-up, look-ahead on) the behaviour is unchanged.

    `market="auto"` attaches the stored market context of the window when a current table built from this candle
    database exists (CONTEXT_DB, default <data_dir>/context.db; nothing is built here), None runs without it, or pass a
    ContextTable."""
    syms = symbol_list or store.candle_symbols()
    lo = since - max(0, int(warmup_ms))
    candles: dict[str, CandleArrays] = {}
    funding: dict[str, list[tuple[int, float]]] = {}
    metrics: dict = {}
    for s in syms:
        if s not in symbols:
            continue
        ca = _load_1m(store, s, lo, until)
        if ca.n < 500:
            continue
        candles[s] = ca
        # by default one interval past `until`, so the last bucket's upcoming (next-settlement) rate is known to the
        # Context; without look-ahead the last bucket sees the settled rate instead (Backtester._upcoming_funding_rate)
        f_end = (until + FUNDING_INTERVAL if lookahead else until) if until else 2**62
        funding[s] = store.load_funding(s, lo - FUNDING_INTERVAL, f_end)
        if hasattr(store, "load_metrics"):
            mrows = store.load_metrics(s, lo - 2 * 86_400_000, until)
            if mrows:
                metrics[s] = MetricsSeries.from_rows(mrows)
            del mrows  # a year of 5-minute rows is about 70 MB of row objects: not held while the next symbol loads
    if isinstance(market, str):
        market = stored_context(settings, store, list(candles), lo, until if until is not None else 2**62)
    return Backtester(settings, symbols, candles, funding, metrics, market)


def _load_1m(store, symbol: str, lo: int, until: int | None) -> CandleArrays:
    """The symbol's stored 1m candles in [lo, until]: the rows of one query, read _LOAD_CHUNK_MS at a time (a window
    shorter than that is one query, as before) and copied into arrays sized like one CandleArrays.extend over all of
    them. Open-ended windows (until None) stay one query."""
    if until is None or until - lo < _LOAD_CHUNK_MS:
        rows = store.load_candles(symbol, start=lo, end=until)
        ca = CandleArrays("1m", capacity=len(rows) + 16)
        ca.extend(rows)
        return ca
    pieces: list[CandleArrays] = []
    a = lo
    while a <= until:
        b = min(until, a + _LOAD_CHUNK_MS - 1)
        rows = store.load_candles(symbol, start=a, end=b)  # open_time in [a, b]: the pieces neither overlap nor gap
        if rows:
            piece = CandleArrays("1m", capacity=len(rows) + 16)
            piece.extend(rows)
            pieces.append(piece)
        del rows
        a = b + 1
    total = sum(p.n for p in pieces)
    ca = CandleArrays("1m", capacity=total + 16)
    k = 0
    for piece in pieces:
        if ca.n and int(piece.open_time[0]) <= int(ca.open_time[ca.n - 1]):
            raise ValueError(f"{symbol}: candle pieces out of order")  # cannot happen with ORDER BY open_time
        for name in COLUMNS:
            getattr(ca, name)[k:k + piece.n] = piece.view(name)
        k += piece.n
        ca.n = k
    return ca


def stored_context(settings: Settings, store, symbols: list[str], start: int, end: int) -> ContextTable | None:
    """Stored context rows for a window from the context store (CONTEXT_DB, default <data_dir>/context.db) when it
    exists and was built from this candle database (`store`), else None. Nothing is built here."""
    from pathlib import Path

    path = getattr(settings, "context_db_path", None) or getattr(settings, "context_db", "") or ""
    if not path or not Path(path).exists():
        return None
    from heartless.core.store import Store

    ctx_store = Store(path)
    try:
        return ContextTable.load(ctx_store, symbols, start, end, source=getattr(store, "path", None))
    finally:
        ctx_store.close()

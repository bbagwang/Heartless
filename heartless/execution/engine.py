"""The trading engine: one instance per account/parameter-set (live, paper champion, paper challengers, backtest).

Every code path (entries, brackets, trailing, time stops, reconciliation) is identical across modes;
only the Account implementation differs.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import (Decision, EntryStyle, Fill, Position, PositionStatus, Side, SymbolInfo,
                                   TradeRecord)
from heartless.core.store import Store
from heartless.data.features import MarketView
from heartless.exchange.base import Account, OrderResult
from heartless.exchange.symbols import min_qty_for_notional, norm_price, norm_qty
from heartless.execution.allocator import Candidate, PortfolioAllocator
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.base import Context
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.strategy.risk import RiskManager, SizeResult
from heartless.util.ids import client_order_id, short_id
from heartless.util.timeutil import MS_MINUTE, TF_MS, now_ms

log = logging.getLogger(__name__)

# a reduce-only close that has not been confirmed within this window may be re-sent (on_tick / reconcile / owner)
CLOSE_RETRY_MS = 10_000
# order states after which executedQty can no longer change
TERMINAL_ORDER_STATES = ("FILLED", "CANCELED", "EXPIRED", "REJECTED")
# entry decisions kept per symbol for the owner's /why and the dashboard (memory only, newest last)
DECISION_LOG_PER_SYMBOL = 5
# F-pre-B execution modes (ROADMAP 7.7.2): how long a maker entry rests (decision bars of the alpha, capped)
MAKER_TTL = {"maker_m1": (1, 60 * MS_MINUTE), "maker_m2": (3, 240 * MS_MINUTE)}
MAKER_M2_ATR = 0.25  # maker_m2 rests 0.25 signal ATR behind the decision close (long below, short above)


class BookGuard:
    """ROADMAP 7.11: hold an entry while the symbol's book is abnormal against its own last 24 hours: spread >=
    BOOK_GUARD_SPREAD_MULT x its prior-24h median and >= BOOK_GUARD_MIN_SPREAD_BPS, or the depth within
    +-ORDERBOOK_DEPTH_BPS (both sides) < BOOK_GUARD_DEPTH_FRAC x its prior-24h median.

    The figures are the live stream book's when it is usable (usable_stream_book: synced, fresh, its levels reaching
    +-k on both sides; one sample per minute is kept per symbol) or, everywhere else (backtests, replays, live without
    ORDERBOOK_STREAM, a stale or uncovered stream book), the conservative 1m-bar proxy of
    exchange/orderbook.bar_book_proxy computed over the symbol's 1m bars on demand (book_proxy_series), current and
    prior alike. The thresholds are ratios to the symbol's own usual values, so they mean the same on the proxy.
    Fewer than MIN_SAMPLES prior minutes: no verdict (the entry is not held)."""

    MIN_SAMPLES = 720  # half the window: a fresh stream or a short history judges nothing
    WINDOW = 1440  # prior-24h = the 1,440 minutes before the current one

    def __init__(self, settings: Settings):
        self.mult = float(getattr(settings, "book_guard_spread_mult", 3.0))
        self.min_bps = float(getattr(settings, "book_guard_min_spread_bps", 5.0))
        self.frac = float(getattr(settings, "book_guard_depth_frac", 0.25))
        self.k_bps = float(getattr(settings, "orderbook_depth_bps", 10.0))
        self.samples: dict[str, deque] = {}  # symbol -> (minute, spread bps, depth both sides) of the stream book
        self.held = 0

    def observe(self, ctx: Context, now: int) -> None:
        """Keep the minute's stream-book sample (called on every 1m bar; nothing for a proxy or no book)."""
        b = ctx.book
        if not usable_stream_book(b):
            return
        minute = int(now) // MS_MINUTE
        dq = self.samples.setdefault(ctx.symbol, deque(maxlen=self.WINDOW + 60))
        if dq and dq[-1][0] == minute:
            return
        dq.append((minute, float(b.spread_bps), float(b.depth_bid_k) + float(b.depth_ask_k)))

    def figures(self, view: MarketView, ctx: Context, now: int, tick: float) -> tuple[float, float, list, list]:
        """(current spread, current depth, prior spreads, prior depths) of the symbol at `now`."""
        b = ctx.book
        if usable_stream_book(b):
            minute = int(now) // MS_MINUTE
            prior = [(s, d) for m, s, d in self.samples.get(ctx.symbol, ()) if minute - self.WINDOW <= m < minute]
            return (float(b.spread_bps), float(b.depth_bid_k) + float(b.depth_ask_k), [p[0] for p in prior],
                    [p[1] for p in prior])
        spread, depth = book_proxy_series(view.frames.get("1m"), now, self.k_bps, tick, self.WINDOW + 1)
        if not len(spread):
            return math.nan, math.nan, [], []
        return float(spread[-1]), float(depth[-1]), spread[:-1].tolist(), depth[:-1].tolist()

    def check(self, view: MarketView, ctx: Context, now: int, tick: float) -> str:
        """Why the entry is held ('' = not held)."""
        s, d, ps, pd = self.figures(view, ctx, now, tick)
        ps = [x for x in ps if x == x]
        pd = [x for x in pd if x == x]
        if len(ps) >= self.MIN_SAMPLES and s == s:
            med = float(np.median(ps))
            if s >= self.mult * med and s >= self.min_bps:
                self.held += 1
                return "book guard: spread"
        if len(pd) >= self.MIN_SAMPLES and d == d:
            med = float(np.median(pd))
            if med > 0 and d < self.frac * med:
                self.held += 1
                return "book guard: depth"
        return ""


# a stream book older than this is not the market (exchange/orderbook.MAX_BOOK_AGE_MS of the red-team fix F-M6)
BOOK_MAX_AGE_MS = 3_000


def usable_stream_book(b) -> bool:
    """Whether a ctx.book can back the book guard as a live stream book: source "stream", synced and covering +-k on both
    sides (finite spread, depths and imbalance: a book whose levels stop short of k reads NaN depth) and fresh
    (staleness <= BOOK_MAX_AGE_MS). The same test as exchange/orderbook.usable_book (red-team F-M5 / F-M6); anything
    else falls back to the 1m-bar proxy: a guard comparing NaN figures would never hold anything."""
    if getattr(b, "source", "none") != "stream":
        return False
    figs = (b.spread_bps, b.depth_bid_k, b.depth_ask_k, b.imbalance, b.staleness_ms)
    return all(x == x for x in figs) and b.staleness_ms <= BOOK_MAX_AGE_MS


def book_proxy_series(frame, now: int, k_bps: float, tick: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The 1m-bar book proxy (exchange/orderbook.bar_book_proxy: spread = max(tick, 0.2 x the median 1m range of the
    last 15 bars), depth per side = 0.1 x their median quote volume x min(1, k / range)) at each of the last `n` bars
    closed at or before `now`, vectorised: (spread bps, depth of both sides). Empty arrays without bars."""
    from bisect import bisect_right

    from numpy.lib.stride_tricks import sliding_window_view

    from heartless.exchange.orderbook import PROXY_BARS, PROXY_DEPTH_FRAC, PROXY_SPREAD_FRAC

    empty = np.zeros(0)
    if frame is None or not frame.n:
        return empty, empty
    i = bisect_right(frame.close_times(), now)
    if i <= 0:
        return empty, empty
    a = max(0, i - n - PROXY_BARS + 1)
    f = frame.f
    h, lo, c, qv = (np.asarray(f[k][a:i], dtype=float) for k in ("high", "low", "close", "quote_volume"))
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.where(c > 0, (h - lo) / c * 1e4, np.nan)
    if len(rng) < PROXY_BARS:  # too short for one full window: the proxy over what there is, for the last bar only
        r_med, q_med, mid = np.array([np.nanmedian(rng)]), np.array([np.nanmedian(qv)]), c[-1:]
    else:
        r_med = np.nanmedian(sliding_window_view(rng, PROXY_BARS), axis=1)
        q_med = np.nanmedian(sliding_window_view(qv, PROXY_BARS), axis=1)
        mid = c[PROXY_BARS - 1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        tick_bps = np.where(mid > 0, tick / mid * 1e4, 0.0) if tick > 0 else np.zeros(len(mid))
        spread = np.maximum(tick_bps, PROXY_SPREAD_FRAC * r_med)
        scale = np.where(r_med > 0, np.minimum(1.0, k_bps / r_med), 1.0)
    depth = 2.0 * PROXY_DEPTH_FRAC * q_med * scale
    return spread[-n:], depth[-n:]


@dataclass
class EngineStats:
    realized_today: float = 0.0
    trades_today: int = 0
    wins_today: int = 0
    last_trade_ts: int = 0
    total_trades: int = 0
    equity: float = 0.0
    balance: float = 0.0
    start_equity: float = 0.0
    peak_equity: float = 0.0
    last_equity_ts: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


@dataclass
class EntryPlan:
    """A sized, bracket-checked entry that has not been sent yet (TradingEngine._plan_entry -> _place_entry)."""
    decision: Decision
    view: MarketView
    ctx: Context
    info: SymbolInfo
    entry_ref: float  # decision reference: the limit price, or the touch a market order is sent against
    size: SizeResult
    stop: float  # tick-rounded: the stop the order carries and the size was computed for
    tp: float | None
    tp1: float | None
    r_unit: float
    risk_pct: float  # per-trade budget in % of equity the entry was sized for
    budget: float  # that budget in USDT
    equity: float
    open_notional: float
    meta_x: list | None = None
    exec_mode: str = "taker"  # F-pre-B: the alpha's execution mode (EXEC_MODE)
    ttl_ms: int = 0  # how long a maker entry rests (0 = taker, or the signal's own rules)

    @property
    def symbol(self) -> str:
        return self.decision.symbol

    @property
    def side(self) -> Side:
        return self.decision.side


class TradingEngine:
    def __init__(self, name: str, account: Account, params: StrategyParams, settings: Settings, symbols: dict[str, SymbolInfo],
                 store: Store | None = None, bus: EventBus | None = None, bandit: AlphaBandit | None = None,
                 only_alpha: str | None = None, clock: Callable[[], int] | None = None, persist: bool = True,
                 notify: bool = False, meta=None, meta_learn: bool = True,
                 allocator: PortfolioAllocator | None = None):
        self.name = name
        self.meta = meta  # MetaLabeler or None (heartless/learning/metalabel.py)
        self.meta_learn = meta_learn
        self.account = account
        self.params = params
        self.s = settings
        self.symbols = symbols
        self.store = store
        self.bus = bus
        self.bandit = bandit or AlphaBandit([a for a in params.alphas])
        self.ensemble = Ensemble(params, self.bandit, only_alpha=only_alpha)
        # store-backed whenever this engine persists (paper/live/challengers): loss halts, the drawdown pause, the weekly
        # risk scale and symbol cool-downs survive a restart; backtests (persist=False) keep a memory-only RiskState
        self.risk = RiskManager(settings, store if (persist and store is not None) else None, name)
        self.clock = clock or now_ms
        self.persist = persist and store is not None
        self.notify = notify
        self.positions: dict[str, Position] = {}
        self.stats = EngineStats()
        self.closed: list[TradeRecord] = []  # in-memory (backtests + recent)
        self.account.on_fill(self.on_fill)
        if hasattr(self.account, "on_algo_event"):
            self.account.on_algo_event(self.on_algo_event)
        self.last_views: dict[str, MarketView] = {}
        self._equity_cache: tuple[int, float] = (0, 0.0)
        self.entries_enabled = True
        self.tag = "".join(ch for ch in name.upper() if ch.isalnum())[:4]
        self._last_exit_ts: dict[str, int] = {}  # symbol -> clock() of the last finalize/cancel (reconcile freshness)
        # P3 portfolio allocator (PORTFOLIO_ALLOCATOR, or injected to share a beta estimator): when set, on_bar queues
        # entry decisions and flush_entries ranks and admits a bar close's batch; None = FIFO, exactly as before
        self.allocator: PortfolioAllocator | None = allocator if allocator is not None else (
            PortfolioAllocator(settings) if getattr(settings, "portfolio_allocator", False) else None)
        self._batch: list[Candidate] = []
        # symbol -> the latest entry decisions that had at least one signal (what was considered, the outcome and why):
        # the answer to "why did / didn't it trade". Only persisted engines keep it; backtests pay nothing
        self.decision_log: dict[str, deque] | None = {} if self.persist else None
        # F-pre-B. The shock layer (heartless/strategy/shock.py): an object answering entry_halt(t) -> reason ('' =
        # none) - the backtester's ShockSchedule, the live bot's ShockDetector (wired by the app); None = no layer.
        self.shock = None
        # execution mode per alpha (EXEC_MODE, ROADMAP 7.7.2): taker = the orders as before F-pre-B
        self._exec_default, self._exec = getattr(settings, "exec_modes", ("taker", {}))
        # entry hold on an abnormal book (BOOK_GUARD, ROADMAP 7.11)
        self.book_guard: BookGuard | None = BookGuard(settings) if getattr(settings, "book_guard", False) else None
        self._exempt: dict[str, bool] = {}

    # --- lifecycle -----------------------------------------------------------------------------
    async def start(self) -> None:
        if self.persist:
            for p in self.store.load_open_positions(self.name):
                self.positions[p.symbol] = p
            if self.positions:
                log.info("[%s] restored %d open positions", self.name, len(self.positions))
        st = await self.account.get_state()
        self.stats.equity = self.stats.balance = st.equity
        self.stats.start_equity = self.stats.start_equity or st.equity
        self.stats.peak_equity = max(self.stats.peak_equity, st.equity)
        self.risk.observe_equity(st.equity, self.clock())
        self._equity_cache = (self.clock(), st.equity)
        if not self.account.is_paper:
            await self.reconcile()

    def set_params(self, params: StrategyParams) -> None:
        self.params = params
        self.ensemble.set_params(params)

    # --- F-pre-B: execution mode and shock layer -------------------------------------------------------------------
    def exec_mode(self, alpha: str) -> str:
        """The execution mode the engine trades `alpha` with (taker | maker_m1 | maker_m2): the parameter set's
        `exec[alpha]` (ROADMAP 7.7.2), else the EXEC_MODE setting (Settings.exec_for)."""
        mode = (getattr(self.params, "exec", None) or {}).get(alpha)
        return mode or self._exec.get(alpha, self._exec_default)

    def _shock_exempt(self, alpha: str) -> bool:
        ex = self._exempt.get(alpha)
        if ex is None:
            from heartless.strategy.shock import is_exempt

            ex = self._exempt[alpha] = is_exempt(alpha)
        return ex

    def shock_halt(self, now: int, alpha: str | None = None) -> str:
        """Why the shock layer halts new entries at `now` ('' = it does not): for `alpha` when given (a shock_exempt
        alpha is never halted), else for the non-exempt alphas."""
        if self.shock is None or (alpha is not None and self._shock_exempt(alpha)):
            return ""
        return self.shock.entry_halt(now)

    @property
    def queued_entries(self) -> int:
        """Entry candidates waiting for this bar close's flush_entries (portfolio allocator on)."""
        return len(self._batch)

    def open_positions(self) -> list[Position]:
        return [p for p in self.positions.values() if p.status in (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.CLOSING)]

    async def equity(self, max_age_ms: int = 5000) -> float:
        ts, eq = self._equity_cache
        if self.clock() - ts <= max_age_ms and eq > 0:
            return eq
        st = await self.account.get_state()
        self._equity_cache = (self.clock(), st.equity)
        self.stats.equity, self.stats.balance = st.equity, st.balance
        self.stats.peak_equity = max(self.stats.peak_equity, st.equity)
        return st.equity

    async def _emit(self, topic: str, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("engine", self.name)
        payload.setdefault("notify", self.notify)
        payload.setdefault("ts", self.clock())
        if self.bus is not None:
            await self.bus.publish(topic, payload)
        if self.persist and topic in ("position_opened", "position_closed", "risk_event", "reconcile", "error"):
            self.store.log_event(self.clock(), "INFO", topic, {k: v for k, v in payload.items() if k != "position"})

    def _save(self, p: Position) -> None:
        if self.persist:
            self.store.save_position(p)

    # --- bar handling --------------------------------------------------------------------------
    async def on_bar(self, view: MarketView, ctx: Context) -> None:
        """Called once per completed 1m bar per symbol (after the view has been sought to that bar).

        With the portfolio allocator on, an entry decision is queued instead of opened; the caller then runs
        flush_entries() once every symbol's bar at this close has been processed."""
        self.last_views[ctx.symbol] = view
        now = self.clock()
        if self.book_guard is not None:
            self.book_guard.observe(ctx, now)
        if self.allocator is not None and view.closed("1h"):
            h1 = view.tf("1h")
            if h1.ok:
                self.allocator.beta.observe(ctx.symbol, h1.open_time, h1.v("close"))
        pos = self.positions.get(ctx.symbol)
        if pos is not None and pos.status in (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.CLOSING):
            if not await self._manage(pos, view, ctx):
                return
            # an opposite-signal exit (Signal.exit_on_opposite) closed it on this bar: the bar may still decide
        if not view.ready() or not self.entries_enabled:
            return
        # the shock layer's entry halt drops the signals of every alpha but the shock_exempt ones (ROADMAP 7.8)
        halt = self.shock_halt(now)
        decision = self.ensemble.decide(view, ctx, veto=self._shock_exempt_not) if halt else self.ensemble.decide(view, ctx)
        if halt and self.ensemble.last_vetoed:
            self.stats.skipped["shock halt"] = self.stats.skipped.get("shock halt", 0) + len(self.ensemble.last_vetoed)
            if decision is None and self.decision_log is not None:
                self._log_decision(ctx, None, "risk", halt, self.ensemble.last_vetoed)
                return
        if decision is None:
            if self.decision_log is not None and self.ensemble.last_signals.get(ctx.symbol):
                self._log_decision(ctx, None, "ensemble", self._ensemble_pass_reason(ctx.symbol))
            return
        if self.allocator is not None:
            cand = Candidate(decision, view, ctx)
            if self.decision_log is not None:
                # this close's signals for the record flush_entries writes once the batch is decided
                cand.extra["signals"] = list(self.ensemble.last_signals.get(ctx.symbol) or [])
            self._batch.append(cand)
            return
        equity = await self.equity()
        allowed, why = self.risk.can_open(decision, self.open_positions(), now, equity)
        if not allowed:
            self.stats.skipped[why.split(":")[0]] = self.stats.skipped.get(why.split(":")[0], 0) + 1
            if self.decision_log is not None:
                self._log_decision(ctx, decision, "risk", why)
            return
        before, skipped = self.positions.get(ctx.symbol), dict(self.stats.skipped)
        await self._open(decision, view, ctx, equity)
        if self.decision_log is not None:
            self._log_outcome(ctx, decision, before, skipped)

    def _shock_exempt_not(self, alpha: str) -> bool:
        return not self._shock_exempt(alpha)

    def _log_outcome(self, ctx: Context, decision: Decision, before: Position | None, skipped: dict[str, int],
                     signals: list | None = None, allocation: dict | None = None) -> None:
        """Record what an entry attempt ended in: 'opened' when a new position stands, else 'execution' with the skip
        counter _plan_entry / _open moved (each skip is counted in stats.skipped)."""
        pos = self.positions.get(ctx.symbol)
        if pos is not None and pos is not before and pos.status is not PositionStatus.CANCELLED:
            self._log_decision(ctx, decision, "opened", pos.entry_style, signals, allocation)
        else:
            moved = [k for k, v in self.stats.skipped.items() if v != skipped.get(k, 0)]
            self._log_decision(ctx, decision, "execution", moved[0] if moved else "entry not placed", signals,
                               allocation)

    def _log_candidate(self, c: Candidate, stage: str, why: str, **extra) -> None:
        """Record one allocator batch candidate's outcome (flush_entries): the decision log entry of the FIFO path plus
        its place in the batch (`allocation`: rank / batch size, what it was ranked by and its BTC beta)."""
        self._log_decision(c.ctx, c.decision, stage, why, c.extra.get("signals"), self._allocation(c, **extra))

    def _allocation(self, c: Candidate, **extra) -> dict:
        rank, of = c.extra.get("rank", (0, 0))
        out = {"rank": rank, "of": of, "by": c.source, "value": round(float(c.value), 4),
               "beta": round(float(self.allocator.beta.beta(c.symbol)), 2) if self.allocator is not None else 1.0}
        out.update(extra)
        return out

    def _ensemble_pass_reason(self, symbol: str) -> str:
        """Why Ensemble.decide passed on the symbol's signals without recomputing its scores (the bandit weight is a
        Thompson draw: drawing again would move the RNG stream). One side only can fail just the entry threshold;
        signals on both sides may also have vetoed each other."""
        sides = {s.side for s in self.ensemble.last_signals.get(symbol) or []}
        return "threshold" if len(sides) <= 1 else "conflict or threshold"

    def _log_decision(self, ctx: Context, decision: Decision | None, stage: str, why: str, signals: list | None = None,
                      allocation: dict | None = None) -> None:
        """Append a compact record of one entry decision (stage: 'ensemble' | 'risk' | 'allocator' | 'execution'
        rejected it, or 'opened'). Plain values only: it is read by Telegram, the web JSON and tests. `signals`: the
        decision's signals when they are no longer the ensemble's latest (an allocator batch is decided after the
        close); `allocation`: the candidate's place in its batch (portfolio allocator on)."""
        sigs = signals if signals is not None else (self.ensemble.last_signals.get(ctx.symbol) or [])
        rec = {"ts": self.clock(), "engine": self.name, "symbol": ctx.symbol,
               "regime": getattr(ctx.regime, "value", ctx.regime),
               "stage": stage, "why": why, "threshold": self.params.ensemble.get("entry_threshold"),
               "signals": [{"alpha": s.alpha, "side": s.side.value, "confidence": round(float(s.confidence), 3),
                            "reason": str(s.reason)[:200]} for s in sigs[:6]]}
        if decision is not None:
            rec["decision"] = {"side": decision.side.value, "alphas": list(decision.alphas),
                               "score": round(float(decision.score), 3), "expected_r": round(float(decision.expected_r), 2)}
        if allocation is not None:
            rec["allocation"] = allocation
        self.decision_log.setdefault(ctx.symbol, deque(maxlen=DECISION_LOG_PER_SYMBOL)).append(rec)

    async def on_equity_tick(self) -> None:
        """Periodic: refresh equity, risk anchors, persist equity curve."""
        now = self.clock()
        try:
            eq = await self.equity(max_age_ms=0)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] equity refresh failed: %s", self.name, e)
            return
        for ev in self.risk.observe_equity(eq, now):
            await self._emit("risk_event", {"message": ev})
        if self.persist and now - self.stats.last_equity_ts >= 60_000:
            self.stats.last_equity_ts = now
            self.store.save_equity(self.name, now, self.stats.balance, eq)

    # --- entries -------------------------------------------------------------------------------
    async def _open(self, d: Decision, view: MarketView, ctx: Context, equity: float,
                    meta: tuple | None = None) -> bool:
        """Plan (size, check) and send one entry. Returns True when an entry was sent."""
        plan = await self._plan_entry(d, view, ctx, equity, meta)
        if plan is None:
            return False
        await self._place_entry(plan)
        return True

    def _meta_eval(self, view: MarketView, ctx: Context, d: Decision) -> tuple:
        """(meta vector, allow, size multiplier, expected R or None) of the meta model for a decision."""
        x = self._meta_vector(view, ctx, d)
        allow, mult, er = self.meta.decide(d.primary.alpha, x)
        return x, allow, mult, er

    def _fill_buffer(self, info: SymbolInfo, entry_ref: float) -> float:
        """Expected adverse fill of a market entry beyond its decision touch, as a fraction of it (E1): the modelled
        slippage plus MARKET_ENTRY_BUFFER_BPS, plus one tick for the move between the decision and the fill."""
        bps = float(getattr(self.s, "backtest_slippage_bps", 0.0) or 0.0) + \
            float(getattr(self.s, "market_entry_buffer_bps", 0.0) or 0.0)
        return bps / 1e4 + (info.tick_size / entry_ref if entry_ref > 0 else 0.0)

    async def _plan_entry(self, d: Decision, view: MarketView, ctx: Context, equity: float,
                          meta: tuple | None = None) -> EntryPlan | None:
        """Everything about an entry short of sending it: reference price, spread and meta-model gates, sizing and the
        bracket sanity checks. None (with the skip counted) when the entry must not be sent. `meta` is a precomputed
        _meta_eval result (the allocator ranks on it before planning)."""
        sig = d.primary
        info = self.symbols.get(d.symbol)
        if info is None:
            return None
        t = ctx.ticker
        limit = sig.entry_style is EntryStyle.LIMIT
        mode = self.exec_mode(sig.alpha)
        ttl_ms = 0
        if mode != "taker":
            # ROADMAP 7.7.2: a post-only limit at the decision close (maker_m1) or 0.25 signal ATR behind it
            # (maker_m2); a limit the alpha set itself keeps its price. Sized below for the limit and the stop.
            entry_ref = self._maker_price(mode, d, view, info)
            if entry_ref is None:
                self.stats.skipped["maker"] = self.stats.skipped.get("maker", 0) + 1
                return None
            limit = True
            bars, cap = MAKER_TTL[mode]
            tf = self.ensemble.alpha_timeframe(sig.alpha) or sig.timeframe
            ttl_ms = min(bars * int(TF_MS.get(tf, MS_MINUTE)), cap)
        elif limit and sig.limit_price:
            entry_ref = sig.limit_price
        else:
            entry_ref = (t.ask if d.side is Side.LONG else t.bid) or t.ref or view.price
        if not entry_ref or entry_ref <= 0:
            return None
        if t.spread_bps > 12:  # illiquid moment
            self.stats.skipped["spread"] = self.stats.skipped.get("spread", 0) + 1
            return None
        if self.book_guard is not None:
            held = self.book_guard.check(view, ctx, self.clock(), info.tick_size)
            if held:
                self.stats.skipped["book guard"] = self.stats.skipped.get("book guard", 0) + 1
                return None
        meta_x = None
        if self.meta is not None:
            meta_x, allow, mult, er = meta if meta is not None else self._meta_eval(view, ctx, d)
            if not allow:
                self.stats.skipped["meta veto"] = self.stats.skipped.get("meta veto", 0) + 1
                return None
            d.size_mult *= mult
            if er is not None:
                d.reason += f" | 메타모델 기대 {er:+.2f}R"
        open_notional = sum(p.notional for p in self.open_positions())
        # E1 sizing: the size is computed for the stop the order will carry (tick-rounded, up to half a tick further
        # than the signal's) and, for a market entry, for the expected fill beyond the decision touch, so the risk the
        # fill realizes stays within the budget. A post-only limit fills at its own (side-rounded) price or better.
        stop = norm_price(info, sig.stop)
        risk_pct = self.risk.budget_pct(d.size_mult)
        buffer = 0.0 if limit else self._fill_buffer(info, entry_ref)
        sz = self.risk.size(d, equity, info, entry_ref, open_notional, stop=stop, fill_buffer=buffer)
        if sz.qty <= 0:
            self.stats.skipped["size"] = self.stats.skipped.get("size", 0) + 1
            return None
        tp = norm_price(info, sig.take_profit) if sig.take_profit else None
        tp1 = norm_price(info, sig.tp1) if sig.tp1 else None
        # Bracket sanity against the price we will actually trade at (the signal's levels were computed from the bar
        # close; the touch can have moved since). A stop at or through the reference is already hit: Binance rejects
        # the STOP_MARKET (-2021), the software backstop flattens at once and the trade is a fee-only round trip with
        # seconds of no exchange stop. A take-profit behind the reference would self-trigger likewise, so drop it.
        if (stop - entry_ref) * d.side.sign >= 0:
            self.stats.skipped["stop"] = self.stats.skipped.get("stop", 0) + 1
            return None
        if tp is not None and (tp - entry_ref) * d.side.sign <= 0:
            tp = None
        if tp1 is not None and ((tp1 - entry_ref) * d.side.sign <= 0 or (tp is not None and abs(tp1 - entry_ref) >= abs(tp - entry_ref))):
            tp1 = None
        r_unit = abs(entry_ref - stop)
        if r_unit <= 0:
            return None
        return EntryPlan(decision=d, view=view, ctx=ctx, info=info, entry_ref=entry_ref, size=sz, stop=stop, tp=tp,
                         tp1=tp1, r_unit=r_unit, risk_pct=risk_pct, budget=equity * risk_pct / 100, equity=equity,
                         open_notional=open_notional, meta_x=meta_x, exec_mode=mode, ttl_ms=ttl_ms)

    @staticmethod
    def _maker_price(mode: str, d: Decision, view: MarketView, info: SymbolInfo) -> float | None:
        """The resting limit of a maker entry, rounded to the passive side of the tick (None: no valid limit)."""
        sig = d.primary
        if sig.entry_style is EntryStyle.LIMIT and sig.limit_price:
            px = float(sig.limit_price)
        elif mode == "maker_m1":
            px = float(view.price)
        else:
            atr = float(sig.atr or 0.0)
            if not (atr > 0 and math.isfinite(atr)):
                return None
            px = float(view.price) - d.side.sign * MAKER_M2_ATR * atr
            if (sig.stop - px) * d.side.sign >= 0:
                return None  # the retest limit lies beyond the stop: the signal is dropped
        if not (px > 0 and math.isfinite(px)):
            return None
        return norm_price(info, px, d.side.order_side)

    def _cooldown_ms(self, alpha: str, bars: int | None, timeframe: str) -> int | None:
        """Signal.cooldown_bars in ms of the alpha's decision timeframe (the signal's own as a fallback)."""
        if bars is None:
            return None
        tf = self.ensemble.alpha_timeframe(alpha) or timeframe
        return max(int(bars), 0) * int(TF_MS.get(tf, MS_MINUTE))

    async def _place_entry(self, plan: EntryPlan) -> None:
        d, view, ctx, info, sz = plan.decision, plan.view, plan.ctx, plan.info, plan.size
        sig = d.primary
        t = ctx.ticker
        entry_ref, stop, tp, tp1, r_unit = plan.entry_ref, plan.stop, plan.tp, plan.tp1, plan.r_unit
        pid = short_id("P")
        cid = client_order_id(f"E{self.tag}")
        maker = plan.exec_mode != "taker"
        style = EntryStyle.LIMIT if maker else sig.entry_style
        pos = Position(id=pid, engine=self.name, symbol=d.symbol, side=d.side, qty=0.0, entry_price=entry_ref,
                       entry_time=self.clock(), stop=stop, take_profit=tp, tp1=tp1, initial_stop=stop,
                       alpha=sig.alpha, alphas=list(d.alphas), reason=d.reason, confidence=d.score, regime=d.regime.value,
                       risk_amount=sz.risk_amount, r_unit=r_unit, notional=sz.notional, leverage=sz.leverage,
                       params_version=self.params.version, atr=sig.atr, trail_atr_mult=sig.trail_atr_mult,
                       max_hold_bars=sig.max_hold_bars, timeframe=sig.timeframe, status=PositionStatus.PENDING,
                       entry_client_id=cid, original_qty=sz.qty, entry_style=style.value,
                       limit_price=entry_ref if style is EntryStyle.LIMIT else None, pending_since=self.clock(),
                       entry_ttl_bars=d.entry_ttl_bars if style is EntryStyle.LIMIT and not maker else None,
                       exit_on_regime_change=d.exit_on_regime_change,
                       time_exit=sig.time_exit, allow_renewal=sig.allow_renewal, exit_on_opposite=sig.exit_on_opposite,
                       cooldown_ms=self._cooldown_ms(sig.alpha, sig.cooldown_bars, sig.timeframe),
                       avoid_funding=sig.avoid_funding,
                       expected_profit=(abs((tp or entry_ref + d.side.sign * d.expected_r * r_unit) - entry_ref) * sz.qty),
                       extra={"tp1_frac": self.params.alphas.get(sig.alpha, {}).get("tp1_frac", 0.5), "tags": sig.tags,
                              "size_mult": d.size_mult, "risk_pct": sz.risk_pct, "risk_budget": plan.budget})
        if plan.meta_x is not None:
            pos.extra["meta_x"] = [None if not (v == v) else float(v) for v in plan.meta_x]
        if maker:
            pos.extra["exec"] = plan.exec_mode
            pos.extra["exec_ttl_ms"] = int(plan.ttl_ms)
        self.positions[d.symbol] = pos
        self._save(pos)
        sent = False  # becomes True once an order call was reached (it may have been accepted despite an exception)
        try:
            await self.account.prepare_symbol(d.symbol, sz.leverage)
            sent = True
            if maker:
                # ROADMAP 7.7.2: post-only at the limit; one that would cross is dropped (no re-quote, no market order)
                kw = {"fill_rule": "maker"} if getattr(self.account, "maker_fill_rule", False) else {}
                res = await self.account.limit_order(d.symbol, d.side.order_side, sz.qty, entry_ref, post_only=True,
                                                     client_id=cid, **kw)
                if res.status in ("EXPIRED", "REJECTED"):
                    self.stats.skipped["maker cross"] = self.stats.skipped.get("maker cross", 0) + 1
                    res = None
            elif sig.entry_style is EntryStyle.LIMIT:
                price = norm_price(info, entry_ref, d.side.order_side)
                res = await self.account.limit_order(d.symbol, d.side.order_side, sz.qty, price, post_only=True, client_id=cid)
                if res.status in ("EXPIRED", "REJECTED"):
                    # post-only would cross: take liquidity if the price is still within a fraction of the ATR
                    ref_now = (t.ask if d.side is Side.LONG else t.bid) or entry_ref
                    qty = self._taker_fallback_qty(plan, ref_now) \
                        if abs(ref_now - entry_ref) <= 0.15 * max(sig.atr, r_unit) else 0.0
                    if qty > 0:
                        pos.original_qty = qty
                        res = await self.account.market_order(d.symbol, d.side.order_side, qty, client_id=cid)
                    else:
                        res = None
            else:
                res = await self.account.market_order(d.symbol, d.side.order_side, sz.qty, client_id=cid)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] entry failed for %s", self.name, d.symbol)
            res = None
            await self._emit("error", {"message": f"{d.symbol} 진입 주문 실패: {e}"})
            if sent:
                # a transport error / timeout can hit AFTER Binance accepted the order: ask before forgetting it
                try:
                    chk = await self.account.query_order(d.symbol, "", cid)
                except Exception as e2:  # noqa: BLE001
                    log.warning("[%s] cannot verify entry %s after failure (%s); keeping it pending", self.name, d.symbol, e2)
                    self._save(pos)
                    return  # stays PENDING: _manage_pending verifies / cancels it by client id on later bars
                if chk.status in ("NEW", "PARTIALLY_FILLED", "FILLED"):
                    res = chk
        if res is None or res.status == "REJECTED":
            if pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
                pos.status = PositionStatus.CANCELLED
                pos.exit_reason = "entry rejected"
                self._save(pos)
                self.positions.pop(d.symbol, None)
            return
        pos.entry_order_id = res.order_id
        if res.status == "FILLED" and pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
            # REST already reports the fill (live market order): synthesize if no stream fill arrived
            q = res.filled_qty or pos.original_qty
            fill = Fill(symbol=d.symbol, order_side=d.side.order_side, qty=q, price=res.avg_price or entry_ref,
                        fee=q * (res.avg_price or entry_ref) * self.s.taker_fee,
                        ts=self.clock(), client_id=cid, order_id=res.order_id, kind="ENTRY")
            await self.on_fill(fill)
            self._mark_settled(pos, res.order_id)
        self._save(pos)

    def _taker_fallback_qty(self, plan: EntryPlan, ref_now: float) -> float:
        """Quantity for the market order that replaces a post-only entry which would cross (E1): sized again for the
        touch it now takes plus the market-entry buffer, never above the planned quantity, never lifted to the
        exchange minimum (0 when what fits the budget is below it)."""
        info = plan.info
        sz = self.risk.size_at(plan.stop, ref_now, plan.equity, info, plan.risk_pct, plan.open_notional,
                               lift_to_min=False, fill_buffer=self._fill_buffer(info, ref_now))
        qty = min(sz.qty, plan.size.qty)
        return qty if sz.qty > 0 and qty >= min_qty_for_notional(info, ref_now) else 0.0

    async def flush_entries(self) -> int:
        """Admit the entry candidates queued at this bar close (portfolio allocator on; a no-op otherwise): ranked by
        expected value, each in turn passes the count caps and cool-downs (RiskManager.can_open), is sized
        (_plan_entry) and must fit the risk budgets (PortfolioAllocator.check) before it is sent. Returns the number of
        entries sent. Callers run it once every symbol's bar at the close has been processed (Backtester: each
        timeline step; app: when the batch is complete)."""
        batch, self._batch = self._batch, []
        if not batch or self.allocator is None:
            return 0
        # E3 decision record (persisted engines only; backtests keep none and take exactly the unlogged path)
        log_ = self.decision_log is not None
        # entries switched off after the close was queued (the live -> paper flatten sets entries_enabled False, then
        # closes everything and checks nothing is left): the FIFO path reads the flag right before it opens, so a
        # queued candidate must not open after it either
        if not self.entries_enabled:
            if log_:
                for c in batch:
                    self._log_candidate(c, "risk", "entries disabled")
            return 0
        now = self.clock()
        equity = await self.equity()
        if self.meta is not None:
            for c in batch:  # the meta model's expected R ranks; computed once and reused by the plan
                c.meta = self._meta_eval(c.view, c.ctx, c.decision)
        ranked = self.allocator.rank(batch, self.bandit)
        if log_:
            for i, c in enumerate(ranked):
                c.extra["rank"] = (i + 1, len(ranked))
        sent = 0
        for i, c in enumerate(ranked):
            if not self.entries_enabled:  # switched off while an earlier candidate's order was being sent
                if log_:
                    for rest in ranked[i:]:
                        self._log_candidate(rest, "risk", "entries disabled")
                break
            stage = "risk"
            allowed, why = self.risk.can_open(c.decision, self.open_positions(), now, equity)
            if allowed:
                skipped = dict(self.stats.skipped) if log_ else {}
                plan = await self._plan_entry(c.decision, c.view, c.ctx, equity, c.meta)
                if plan is None:  # spread / meta veto / size / stop: counted by _plan_entry
                    if log_:
                        self._log_outcome(c.ctx, c.decision, self.positions.get(c.symbol), skipped,
                                          c.extra.get("signals"), self._allocation(c))
                    continue
                allowed, why = self.allocator.check(plan.symbol, plan.side, plan.size.risk_amount,
                                                    self.open_positions(), equity)
                if allowed:
                    before = self.positions.get(plan.symbol)
                    await self._place_entry(plan)
                    sent += 1
                    if log_:
                        self._log_outcome(c.ctx, c.decision, before, skipped, c.extra.get("signals"),
                                          self._allocation(c))
                    continue
                stage = "allocator"
                if log_:  # the budget it did not fit: open risk before it, and what it would have added
                    ex = self.allocator.exposure(self.open_positions(), equity)
                    self._log_candidate(c, stage, why, **{
                        "total_pct": round(ex["total_pct"], 3), "long_pct": round(ex["long_pct"], 3),
                        "short_pct": round(ex["short_pct"], 3),
                        "add_pct": round(plan.size.risk_amount / equity * 100.0, 3) if equity > 0 else 0.0,
                        "max_total_pct": self.s.max_total_risk_pct,
                        "max_same_pct": self.s.max_same_direction_risk_pct})
            elif log_:
                self._log_candidate(c, stage, why)
            key = why.split(":")[0]
            self.stats.skipped[key] = self.stats.skipped.get(key, 0) + 1
        return sent

    # --- fills ---------------------------------------------------------------------------------
    async def on_fill(self, fill: Fill) -> None:
        pos = self.positions.get(fill.symbol)
        if fill.kind == "FUNDING":
            if pos is not None and pos.status is PositionStatus.OPEN:
                pos.funding -= fill.fee  # fee holds the payment (positive = paid)
                self._save(pos)
            return
        if pos is None:
            if not fill.reduce_only and fill.kind in ("ENTRY", "UNKNOWN") and not self.account.is_paper:
                log.warning("[%s] fill for unknown position %s %s %s", self.name, fill.symbol, fill.order_side, fill.qty)
            return
        opens_side = fill.order_side == pos.side.order_side
        if opens_side and not fill.reduce_only:
            await self._on_entry_fill(pos, fill)
        else:
            await self._on_exit_fill(pos, fill)

    @staticmethod
    def _mark_settled(pos: Position, order_id: str) -> None:
        """Record that an order's final executedQty has been credited from a REST query/result: every later stream
        fill for that order id is a duplicate."""
        if order_id:
            settled = pos.extra.setdefault("settled_orders", [])
            if order_id not in settled:
                settled.append(order_id)

    def _gone(self, pos: Position) -> bool:
        """True once a position stopped being the live OPEN position for its symbol (exited / replaced during an await)."""
        return pos.status is not PositionStatus.OPEN or self.positions.get(pos.symbol) is not pos

    async def _on_entry_fill(self, pos: Position, fill: Fill) -> None:
        if (fill.client_id and pos.entry_client_id and fill.client_id != pos.entry_client_id and fill.client_id.startswith("HLE")
                and fill.client_id not in pos.extra.get("entry_cids", [])):
            return  # a different (stale) entry order
        if fill.order_id and fill.order_id in pos.extra.get("settled_orders", []):
            return  # this order's final executed quantity was already credited from its REST state
        if pos.status is PositionStatus.CLOSED:
            # the entry's remainder filled under the cancel that _finalize sent: that quantity is live on the
            # exchange and no longer tracked here. reconcile adopts it with a protective stop on its next pass; say
            # so, because until then it is protected by nothing but that pass.
            log.warning("[%s] late entry fill %s %s after close; reconcile will adopt it", self.name, fill.symbol, fill.qty)
            await self._emit("error", {"message": f"{fill.symbol} 청산 후 진입 잔량 {fill.qty} 체결 — 다음 reconcile 에서 인수/보호"})
            return
        # de-duplicate: a REST-synthesized fill and a stream fill for the same order
        key = f"{fill.order_id}:{fill.ts}:{fill.qty}"
        seen = pos.extra.setdefault("fills", [])
        if fill.order_id and any(k.startswith(f"{fill.order_id}:") for k in seen) and fill.client_id == pos.entry_client_id and pos.filled_qty >= pos.original_qty - 1e-12:
            return
        seen.append(key)
        total = pos.filled_qty + fill.qty
        pos.entry_price = (pos.entry_price * pos.filled_qty + fill.price * fill.qty) / total if pos.filled_qty > 0 else fill.price
        pos.filled_qty = total
        pos.qty += fill.qty  # not `total`: a late entry fill must not resurrect quantity already exited
        pos.fees += fill.fee
        pos.notional = pos.qty * pos.entry_price
        # recompute R geometry from the real fills (every fill: a market order routinely executes as several trades);
        # measured against the initial stop so a late fill after a breakeven/trailing move cannot shrink the R unit
        pos.r_unit = abs(pos.entry_price - (pos.initial_stop or pos.stop)) or pos.r_unit
        pos.risk_amount = pos.filled_qty * (pos.r_unit + pos.entry_price * (2 * self.s.taker_fee + 0.0003))
        if pos.status is PositionStatus.PENDING:
            pos.status = PositionStatus.OPEN
            pos.entry_time = fill.ts or self.clock()
            await self._place_brackets(pos)
            self._save(pos)
            await self._emit("position_opened", {"position": pos, "symbol": pos.symbol})
        else:
            # additional partial fill of a limit entry: brackets must cover the new quantity (not while closing)
            if pos.status is PositionStatus.OPEN:
                await self._refresh_brackets(pos)
            self._save(pos)

    async def _place_brackets(self, pos: Position) -> None:
        if self._gone(pos):
            return
        info = self.symbols[pos.symbol]
        close_side = pos.side.close_order_side
        try:
            new_id = await self.account.place_stop(pos.symbol, close_side, pos.stop, qty=pos.qty, close_position=True,
                                                   client_id=client_order_id(f"S{self.tag}"))
        except Exception as e:  # noqa: BLE001
            new_id = None
            log.error("[%s] failed to place stop for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 손절 주문 실패 ({e}). 봇 내부 백스톱으로 보호 중"})
        if self._gone(pos):  # exited while the order was in flight: never leave a zombie closePosition stop behind
            await self._cancel_algo_quiet(pos.symbol, new_id)
            return
        if new_id is not None:
            pos.sl_algo_id = new_id
            pos.last_stop_update = self.clock()
        frac = float(pos.extra.get("tp1_frac", 0.5) or 0.0)
        placed_tp1 = 0.0  # TP1 quantity placed in THIS call (never a stale value from a previous bracket set)
        pos.extra.pop("tp1_qty", None)
        if pos.tp1 and not pos.tp1_done and frac > 0:
            q1 = norm_qty(info, pos.qty * frac)
            rest = norm_qty(info, pos.qty - q1)
            if q1 > 0 and rest > 0:
                try:
                    tp1_id = await self.account.place_take_profit(pos.symbol, close_side, pos.tp1, q1,
                                                                  client_id=client_order_id(f"T{self.tag}"))
                except Exception as e:  # noqa: BLE001
                    tp1_id = None
                    log.warning("[%s] TP1 placement failed for %s: %s", self.name, pos.symbol, e)
                if self._gone(pos):
                    await self._cancel_algo_quiet(pos.symbol, tp1_id)
                    return
                if tp1_id is not None:
                    pos.extra["tp1_algo_id"] = tp1_id
                    pos.extra["tp1_qty"] = q1
                    placed_tp1 = q1
            else:
                pos.tp1 = None
        if pos.take_profit:
            q_tp = norm_qty(info, pos.qty - placed_tp1)
            if q_tp > 0:
                try:
                    tp_id = await self.account.place_take_profit(pos.symbol, close_side, pos.take_profit, q_tp,
                                                                 client_id=client_order_id(f"T{self.tag}"))
                except Exception as e:  # noqa: BLE001
                    tp_id = None
                    log.warning("[%s] TP placement failed for %s: %s", self.name, pos.symbol, e)
                if self._gone(pos):
                    await self._cancel_algo_quiet(pos.symbol, tp_id)
                    return
                if tp_id is not None:
                    pos.tp_algo_id = tp_id

    async def _cancel_algo_quiet(self, symbol: str, algo_id: str | None) -> None:
        if not algo_id:
            return
        try:
            await self.account.cancel_algo(symbol, algo_id)
        except Exception:  # noqa: BLE001
            pass

    async def _ensure_stop(self, pos: Position) -> None:
        """Make sure an OPEN position has an exchange stop (used when a close failed and the position is handed back)."""
        if pos.sl_algo_id:
            return
        try:
            pos.sl_algo_id = await self.account.place_stop(pos.symbol, pos.side.close_order_side, pos.stop, qty=pos.qty,
                                                           close_position=True, client_id=client_order_id(f"S{self.tag}"))
            pos.last_stop_update = self.clock()
        except Exception as e:  # noqa: BLE001
            log.error("[%s] failed to restore stop for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 손절 재설정 실패 ({e}). 봇 내부 백스톱으로 보호 중"})

    def _meta_vector(self, view: MarketView, ctx: Context, d: Decision) -> list[float]:
        """Market context at signal time for the meta-labeler (same feature code as the discovery engine)."""
        import numpy as _np

        from heartless.learning.discovery import features_at
        from heartless.learning.metalabel import META_FEATURES, meta_vector

        feats: dict = {}
        try:
            arr = features_at(view.frames, None, _np.array([int(view.cursor_time)], dtype=_np.int64))
            feats = {n: float(arr[n][0]) for n in META_FEATURES if n in arr}
        except Exception:  # noqa: BLE001
            pass
        ex = ctx.extras or {}
        if ex and not ex.get("stale"):
            for n in META_FEATURES:
                if n.startswith("x.") and isinstance(ex.get(n[2:]), (int, float)):
                    feats[n] = float(ex[n[2:]])
        return meta_vector(feats, d.score, d.side.sign)

    async def on_algo_event(self, event: dict) -> None:
        """The exchange reports a conditional order's status. If our protective stop was rejected or expired
        without filling (margin check, price protection, system cancel), re-arm it right away instead of waiting for
        the next reconcile; a TP that died is simply forgotten (the stop still protects the position)."""
        pos = self.positions.get(event.get("symbol", ""))
        if pos is None or pos.status is not PositionStatus.OPEN:
            return
        status = event.get("status", "")
        if status not in ("REJECTED", "EXPIRED", "CANCELED"):
            return
        aid = event.get("algo_id", "")
        if aid and aid == pos.sl_algo_id:
            if status == "CANCELED" and self.clock() - pos.last_stop_update < 5_000:
                return  # our own replace/cancel sequence
            log.warning("[%s] exchange %s our stop on %s; re-arming", self.name, status.lower(), pos.symbol)
            pos.sl_algo_id = ""
            await self._ensure_stop(pos)
            self._save(pos)
            await self._emit("reconcile", {"message": f"{pos.symbol} 손절 주문이 거래소에서 {status} 되어 즉시 재설정했습니다"})
        elif aid and aid == pos.tp_algo_id and status in ("REJECTED", "EXPIRED"):
            pos.tp_algo_id = ""
            self._save(pos)
        elif aid and aid == pos.extra.get("tp1_algo_id") and status in ("REJECTED", "EXPIRED"):
            pos.extra.pop("tp1_algo_id", None)
            self._save(pos)

    async def _refresh_brackets(self, pos: Position) -> None:
        await self._cancel_brackets(pos)
        await self._place_brackets(pos)

    async def _cancel_brackets(self, pos: Position) -> None:
        for aid in (pos.sl_algo_id, pos.tp_algo_id, pos.extra.get("tp1_algo_id", "")):
            if aid:
                try:
                    await self.account.cancel_algo(pos.symbol, aid)
                except Exception:  # noqa: BLE001
                    pass
        pos.sl_algo_id = pos.tp_algo_id = ""
        pos.extra.pop("tp1_algo_id", None)

    async def _on_exit_fill(self, pos: Position, fill: Fill) -> None:
        if pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING):
            # a PENDING row holds nothing yet (the first entry fill flips it to OPEN), so a reducing fill on its
            # symbol belongs to something else: a late event of the previous trade (already finalized) or an
            # external position (reconcile's business). Booking it here would record a phantom trade against the
            # entry reference and cancel a working entry.
            if pos.status is PositionStatus.PENDING:
                log.warning("[%s] ignoring reducing fill %s %s %s on a pending entry", self.name, fill.symbol,
                            fill.order_side, fill.qty)
            return
        qty = min(fill.qty, pos.qty) if pos.qty > 0 else fill.qty
        if qty <= 0:
            return
        pnl = (fill.price - pos.entry_price) * qty * pos.side.sign
        pos.realized += pnl
        # a REST-synthesized close carries the whole order's fee; when part of that order was already booked from
        # stream trades only the uncredited share of the fee belongs here (otherwise fees are counted twice)
        pos.fees += fill.fee * (qty / fill.qty) if fill.qty > qty > 0 else fill.fee
        pos.qty = max(pos.qty - qty, 0.0)
        kind = fill.kind
        # the engine's own close orders (close_position sets CLOSING + exit_reason before sending; live ids start
        # with HLC, the REST-synthesized fill has no id) already carry the true reason: never relabel them
        engine_close = pos.status is PositionStatus.CLOSING or (fill.client_id or "").startswith("HLC")
        if kind in ("UNKNOWN", "CLOSE") and fill.reduce_only and not engine_close:
            # infer from price relative to brackets
            if pos.tp1 and not pos.tp1_done and abs(fill.price - pos.tp1) <= abs(fill.price - pos.stop):
                kind = "TP"
            elif (pos.side is Side.LONG and fill.price <= pos.stop * 1.002) or (pos.side is Side.SHORT and fill.price >= pos.stop * 0.998):
                kind = "SL"
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        remaining = pos.qty > step * 0.5 + 1e-12
        if remaining and self._is_dust(pos, fill.price):
            # a leftover below the exchange minimum (rounding residue, partial bracket fill) is not a position
            # worth managing and no bracket may cover it: flatten it now instead of holding it indefinitely
            main = {"SL": "손절(SL)" if not pos.be_moved else "본절 손절(BE)", "TP": "익절(TP)"}.get(kind, pos.exit_reason or "청산")
            self._save(pos)
            await self.close_position(pos.symbol, f"{main} (+잔량 정리)")
            return
        if remaining and kind == "TP" and pos.tp1 and not pos.tp1_done and not engine_close:
            pos.tp1_done = True
            pos.extra.pop("tp1_algo_id", None)
            pos.extra.pop("tp1_qty", None)
            await self._move_stop_to_breakeven(pos)
            self._save(pos)
            await self._emit("partial_tp", {"position": pos, "symbol": pos.symbol, "price": fill.price, "qty": qty, "pnl": pnl})
            return
        if remaining:
            self._save(pos)
            return
        reason = {"SL": "손절(SL)" if not pos.be_moved else "본절 손절(BE)", "TP": "익절(TP)"}.get(kind, pos.exit_reason or "청산")
        if kind == "SL" and pos.be_moved and pos.stop != pos.initial_stop and (
                (pos.side is Side.LONG and pos.stop > pos.entry_price) or (pos.side is Side.SHORT and pos.stop < pos.entry_price)):
            reason = "트레일링 스탑"
        await self._finalize(pos, fill.price, fill.ts or self.clock(), reason)

    def _is_dust(self, pos: Position, price: float) -> bool:
        info = self.symbols.get(pos.symbol)
        if info is None or pos.qty <= 0:
            return False
        px = price or pos.entry_price
        return pos.qty < info.min_qty * 0.999 or (px > 0 and pos.qty * px < info.min_notional * 0.999) or \
            (pos.original_qty > 0 and pos.qty < pos.original_qty * 0.02 and pos.qty <= info.step_size * 1.001)

    async def _move_stop_to_breakeven(self, pos: Position) -> None:
        info = self.symbols[pos.symbol]
        buffer = pos.entry_price * (self.s.taker_fee * 2 + 0.0002)
        be = pos.entry_price + pos.side.sign * buffer
        be = norm_price(info, be)
        better = be > pos.stop if pos.side is Side.LONG else be < pos.stop
        if better:
            if await self._replace_stop(pos, be):
                pos.be_moved = True
            # on failure leave be_moved False so the next bar retries instead of pretending protection exists
        else:
            pos.be_moved = True  # stop already at/above breakeven

    async def _replace_stop(self, pos: Position, new_stop: float) -> bool:
        """Move the exchange stop. Returns True when the position is protected at new_stop (or already better)."""
        info = self.symbols[pos.symbol]
        new_stop = norm_price(info, new_stop)
        if abs(new_stop - pos.stop) < info.tick_size * 0.5:
            return True
        if self._gone(pos):
            return False
        old_id = pos.sl_algo_id
        try:
            new_id = await self.account.place_stop(pos.symbol, pos.side.close_order_side, new_stop, qty=pos.qty,
                                                   close_position=True, client_id=client_order_id(f"S{self.tag}"))
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] replace stop failed on %s: %s", self.name, pos.symbol, e)
            return False
        if self._gone(pos):
            # the position exited (old stop / TP / manual close) while the new stop was in flight: _finalize already
            # cancelled the old ids, so the new closePosition stop would otherwise live on as a zombie
            await self._cancel_algo_quiet(pos.symbol, new_id)
            return False
        pos.sl_algo_id = new_id
        pos.stop = new_stop
        pos.last_stop_update = self.clock()
        if old_id:
            try:
                await self.account.cancel_algo(pos.symbol, old_id)
            except Exception:  # noqa: BLE001
                pass
        self._save(pos)
        return True

    # --- management ----------------------------------------------------------------------------
    async def _manage(self, pos: Position, view: MarketView, ctx: Context) -> bool:
        """Manage an open / pending position on a 1m bar. Returns True only when an opposite-signal exit
        (Signal.exit_on_opposite) has just closed it, i.e. when this bar may still decide a new entry."""
        now = self.clock()
        price = ctx.ticker.ref or view.price
        if pos.status is PositionStatus.PENDING:
            await self._manage_pending(pos, view, ctx)
            return False
        if pos.status is not PositionStatus.OPEN or not price:
            return False
        # a post-only entry that opened on a partial fill keeps its remainder working for two bars at most, or until
        # its signal TTL runs out (or new entries get blocked meanwhile)
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        if (pos.entry_style == EntryStyle.LIMIT.value and pos.filled_qty < pos.original_qty - step * 0.5 - 1e-12
                and (self._ttl_stop_reason(pos, now) if self._entry_ttl_ms(pos)
                     else now - pos.pending_since >= 2 * MS_MINUTE)):
            await self._cancel_entry_remainder(pos)
            if pos.status is not PositionStatus.OPEN:
                return False
        pos.bars_held += 1
        fav = (price - pos.entry_price) * pos.side.sign
        pos.max_fav = max(pos.max_fav, fav)
        pos.max_adv = max(pos.max_adv, -fav)
        r_now = pos.r_now(price)
        tf = view.tf(pos.timeframe if pos.timeframe in view.frames else "5m")
        atr = tf.v("atr") if tf.ok else pos.atr
        if not atr or math.isnan(atr):
            atr = pos.atr
        # 0) opt-ins (E1), on the alpha's own decision bars and before the time stop, so a signal on the deadline bar
        #    counts: the same alpha signalling the same side again re-arms the time stop from this bar
        #    (Signal.allow_renewal; not while new entries are blocked), the other side ends the trade
        #    (Signal.exit_on_opposite; the bar may then decide a new entry)
        if pos.allow_renewal or pos.exit_on_opposite:
            s = self.ensemble.evaluate_alpha(pos.alpha, view, ctx)
            if s is not None and s.side is pos.side:
                if (pos.allow_renewal and pos.max_hold_bars > 0 and s.max_hold_bars > 0 and self.entries_enabled
                        and not self.risk.entries_blocked(now) and not self.shock_halt(now, pos.alpha)):
                    pos.max_hold_bars = max(pos.max_hold_bars, pos.bars_held + s.max_hold_bars)
                    pos.extra["renewals"] = int(pos.extra.get("renewals", 0) or 0) + 1
                    pos.extra["renewed_at"] = now
            elif s is not None and pos.exit_on_opposite:
                await self.close_position(pos.symbol, "반대 신호 청산")
                return pos.status is PositionStatus.CLOSED and self.positions.get(pos.symbol) is None
        # 1) time stop
        if pos.max_hold_bars and pos.bars_held >= pos.max_hold_bars:
            if pos.time_exit:
                # Signal.time_exit: a fixed holding period ends here whatever the trade made
                await self.close_position(pos.symbol, "보유 기간 만료 청산(time exit)")
                return False
            if r_now < 0.5 or pos.bars_held >= 3 * pos.max_hold_bars:
                # not working at the deadline, or a winner held 3x past it: a scalp must not turn into a swing
                reason = "시간 초과 청산(time stop)" if r_now < 0.5 else "최대 보유시간 초과 청산"
                await self.close_position(pos.symbol, reason)
                return False
            if not pos.be_moved:
                await self._move_stop_to_breakeven(pos)
            if pos.trail_atr_mult <= 0:
                # a working trade past its deadline keeps running only under a protective trail
                pos.trail_atr_mult = 1.5
        # 2) funding avoidance: don't pay a large funding bill on a trade that is not working (Signal.avoid_funding)
        if pos.avoid_funding and ctx.minutes_to_funding <= 2 and r_now < 0.3:
            adverse = ctx.funding_rate * pos.side.sign  # long pays positive funding
            if adverse > 0.0003:
                await self.close_position(pos.symbol, f"펀딩 회피 청산 (펀딩 {ctx.funding_rate * 100:+.3f}%)")
                return False
        # 3) breakeven once 1R in profit even without a partial TP
        if not pos.be_moved and r_now >= 1.0 and pos.trail_atr_mult > 0:
            await self._move_stop_to_breakeven(pos)
        # 4) trailing stop (chandelier) on the decision timeframe once in profit
        if pos.trail_atr_mult > 0 and atr and (pos.be_moved or r_now >= 0.8):
            close = tf.v("close") if tf.ok else price
            hh = tf.v("hh10") if tf.ok else price
            ll = tf.v("ll10") if tf.ok else price
            if pos.side is Side.LONG:
                anchor = max(hh if not math.isnan(hh) else close, close)
                cand = anchor - pos.trail_atr_mult * atr
                if cand > pos.stop + max(self.symbols[pos.symbol].tick_size, 0.05 * atr):
                    await self._replace_stop(pos, cand)
            else:
                anchor = min(ll if not math.isnan(ll) else close, close)
                cand = anchor + pos.trail_atr_mult * atr
                if cand < pos.stop - max(self.symbols[pos.symbol].tick_size, 0.05 * atr):
                    await self._replace_stop(pos, cand)
        # 5) signals that opted in (fades / mean reversion): exit if the regime flips hard against us
        if pos.exit_on_regime_change and pos.bars_held % 5 == 0:
            m15 = view.tf("15m")
            adx = m15.v("adx") if m15.ok else float("nan")
            slope = m15.v("slope20") if m15.ok else float("nan")
            if not math.isnan(adx) and not math.isnan(slope) and adx > 32 and (slope * pos.side.sign) < 0 and r_now < 0:
                await self.close_position(pos.symbol, "레짐 전환(강한 역추세) 청산")
                return False
        self._save(pos)
        return False

    async def _manage_pending(self, pos: Position, view: MarketView, ctx: Context) -> None:
        """Re-quote or cancel a resting post-only entry.

        Default (no signal TTL): the order chases the touch. Two minutes after it was placed it is cancelled and
        re-quoted at the current bid/ask, at most twice (then the entry is abandoned), and abandoned at once if the
        touch ran more than 0.6 ATR away. Every re-quote re-sizes the order for its new price (_requote_size): a
        chased entry never risks more than the trade's budget at its stop and never grows beyond what the open sized;
        when what fits is below the exchange minimum the entry is cancelled.

        With Signal.entry_ttl_bars = N: the order rests at its limit price until N bars of the signal's timeframe
        have passed since it was placed, then it is cancelled. The deadline (pending_since + N * timeframe) is checked
        on every 1m bar after the simulator processed that bar, so the order works through exactly N such bars. It is
        never re-quoted, re-sized or abandoned for price drift in between (a retest limit), a partial fill keeps the
        remainder working until the same deadline (see _manage), and its state is checked every bar so an order the
        exchange no longer works (filled with the event missed, cancelled, unknown) is settled at once. It is also
        cancelled as soon as new entries are blocked (pause, loss halt, market shock, entries disabled): unlike the
        default entry, which is gone within minutes, it could otherwise open a position hours into a halt."""
        now = self.clock()
        age_bars = (now - pos.pending_since) / MS_MINUTE
        if pos.entry_style != EntryStyle.LIMIT.value:
            # a market order whose result never came back (status UNKNOWN / transport error) is verified on the very
            # next bar: if it did execute the position sits on the exchange without a stop until its fill is credited
            if age_bars >= 1:
                res = await self._credit_entry_fills(pos)  # credits a fill -> OPEN + brackets
                if res is None or pos.status is not PositionStatus.PENDING:
                    return  # unreachable (retry next bar) or it filled
                if res.status in ("CANCELED", "EXPIRED", "REJECTED", "UNKNOWN"):
                    await self._cancel_pending(pos, f"entry {res.status.lower()}")
            return
        # (a partially filled limit entry is OPEN, not PENDING: its remainder is handled in _manage / close / finalize)
        ttl = self._entry_ttl_ms(pos)
        if ttl:
            why = self._ttl_stop_reason(pos, now)
            if why:
                await self._cancel_pending(pos, why)
                return
            try:
                res = await self.account.query_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
            except Exception as e:  # noqa: BLE001
                log.debug("[%s] cannot check resting entry on %s: %s", self.name, pos.symbol, e)
                return
            if res.status in TERMINAL_ORDER_STATES or res.status == "UNKNOWN":
                # only a terminal order is settled here (credits a missed fill -> OPEN, else cancels the entry); a
                # working order's fills arrive on the stream, crediting them from a poll could count them twice
                await self._cancel_pending(pos, f"진입 주문이 거래소에서 종료됨 ({res.status})")
            return
        halt = self.shock_halt(now, pos.alpha)
        if halt:  # ROADMAP 7.8: resting entry limits are cancelled while the shock layer halts entries
            await self._cancel_pending(pos, f"신규 진입 중단({halt})으로 대기 주문 취소")
            return
        if age_bars < 2:
            return
        if pos.requotes >= 2:
            await self._cancel_pending(pos, "진입가 미도달로 주문 취소")
            return
        # re-quote at the new touch if price has not run away (> 0.6 ATR)
        t = ctx.ticker
        touch = t.bid if pos.side is Side.LONG else t.ask
        if not touch:
            return
        info = self.symbols[pos.symbol]
        if abs(touch - pos.limit_price) > 0.6 * max(pos.atr, pos.r_unit):
            await self._cancel_pending(pos, "가격 이탈로 진입 포기")
            return
        price = norm_price(info, touch, pos.side.order_side)
        qty, risk = await self._requote_size(pos, price)
        if qty <= 0:
            await self._cancel_pending(pos, "재호가 수량이 위험 예산 내 거래소 최소 수량 미만 → 진입 취소")
            return
        res = await self._settle_entry_order(pos)  # cancel the resting order and credit anything that filled meanwhile
        if pos.status is not PositionStatus.PENDING:
            return
        if res is None or res.status in ("NEW", "PARTIALLY_FILLED"):
            # exchange unreachable, or the cancel did not take (timeout, rate limit) and the old order still works:
            # quoting a second one on top could fill both (twice the size and the risk budget); retry next bar
            return
        cid = client_order_id(f"E{self.tag}")
        # remember every entry id this position ever used so fills are recognised whichever order they belong to,
        # whether they arrive before the REST response (new id) or late (old id, de-duplicated via settled_orders)
        cids = pos.extra.setdefault("entry_cids", [])
        for c in (pos.entry_client_id, cid):
            if c and c not in cids:
                cids.append(c)
        pos.requotes += 1  # counts attempts, so a flaky exchange cannot keep a position re-quoting forever
        old_qty, old_r = pos.original_qty, pos.r_unit
        pos.original_qty = pos.filled_qty + qty  # the entry is now this order: remainder / dust checks measure it
        try:
            res = await self.account.limit_order(pos.symbol, pos.side.order_side, qty, price, post_only=True, client_id=cid)
            if res.status in ("EXPIRED", "REJECTED"):
                # would cross: the ask (long) is at or below our price, so a taker fill is no worse than the quote
                res = await self.account.market_order(pos.symbol, pos.side.order_side, qty, client_id=cid)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] re-quote failed for %s: %s", self.name, pos.symbol, e)
            await self._emit("error", {"message": f"{pos.symbol} 재주문 실패: {e}"})
            # the old order is already cancelled; the newest id is the only one that might rest on the exchange
            pos.entry_client_id = cid
            pos.entry_order_id = ""
            try:
                res = await self.account.query_order(pos.symbol, "", cid)
            except Exception:  # noqa: BLE001
                res = None
            if res is None or res.status not in ("NEW", "PARTIALLY_FILLED", "FILLED"):
                self._save(pos)
                return  # nothing confirmed under the new id: the next bar settles it and tries again
        if res.status == "REJECTED":
            await self._cancel_pending(pos, "재주문 실패")
            return
        pos.entry_client_id = cid
        pos.entry_order_id = res.order_id
        pos.pending_since = now
        pos.limit_price = price
        if pos.status is PositionStatus.PENDING:
            # the geometry of the order now resting; the first fill recomputes entry, R and risk from the real fill
            # price (a crossing re-quote that a simulator filled synchronously as a taker already did: keep that)
            pos.r_unit = abs(price - pos.stop)  # keep the structural stop; R shrinks/grows with the new price
            pos.entry_price = price
            pos.risk_amount = risk
            pos.notional = pos.original_qty * price
            if pos.take_profit:
                pos.expected_profit = abs(pos.take_profit - price) * pos.original_qty
            elif old_qty > 0 and old_r > 0:
                pos.expected_profit *= (pos.original_qty * pos.r_unit) / (old_qty * old_r)
        if res.status == "FILLED" and pos.status is PositionStatus.PENDING and pos.filled_qty <= 0:
            await self.on_fill(Fill(pos.symbol, pos.side.order_side, res.filled_qty, res.avg_price,
                                    res.filled_qty * res.avg_price * self.s.taker_fee, now, cid, res.order_id, kind="ENTRY"))
            self._mark_settled(pos, res.order_id)
        self._save(pos)

    async def _requote_size(self, pos: Position, price: float) -> tuple[float, float]:
        """Quantity and stop-out risk for re-quoting the unfilled part of a post-only entry at `price`.

        Uses the open's sizing rule (RiskManager.size_at) with the new stop distance and the risk budget the open
        accepted (its risk_pct, of current equity); a filled part of the trade consumes that budget and the notional
        caps first. The result never exceeds the quantity still to fill (a re-quote only ever shrinks an entry) and is
        never lifted to the exchange minimum: (0, 0) when what fits the budget is below it."""
        info = self.symbols[pos.symbol]
        stop = pos.initial_stop or pos.stop
        pct = pos.extra.get("risk_pct") or self.risk.budget_pct(float(pos.extra.get("size_mult", 1.0) or 1.0))
        equity = await self.equity()
        others = sum(p.notional for p in self.open_positions() if p is not pos)
        filled = pos.filled_qty > 0
        sz = self.risk.size_at(stop, price, equity, info, float(pct), others,
                               filled_risk=pos.risk_amount if filled else 0.0,
                               filled_notional=pos.qty * pos.entry_price if filled else 0.0, lift_to_min=False)
        qty = min(sz.qty, norm_qty(info, max(pos.original_qty - pos.filled_qty, 0.0)))
        if sz.qty <= 0 or qty < min_qty_for_notional(info, price):
            return 0.0, 0.0
        return qty, sz.risk_amount * qty / sz.qty

    @staticmethod
    def _entry_ttl_ms(pos: Position) -> int:
        """How long a post-only entry with a signal TTL (or a maker entry's validity, F-pre-B) may rest (0 = no TTL: the
        default re-quote policy applies)."""
        if pos.entry_style != EntryStyle.LIMIT.value:
            return 0
        maker_ms = int((pos.extra or {}).get("exec_ttl_ms") or 0)
        if maker_ms > 0:
            return maker_ms
        if not pos.entry_ttl_bars or pos.entry_ttl_bars <= 0:
            return 0
        return int(pos.entry_ttl_bars) * TF_MS.get(pos.timeframe, MS_MINUTE)

    def _ttl_stop_reason(self, pos: Position, now: int) -> str:
        """Why a resting signal-TTL entry (or its partial-fill remainder) must stop working now ('' = keep resting):
        its deadline passed, or new entries are blocked account-wide."""
        ttl = self._entry_ttl_ms(pos)
        if not ttl:
            return ""
        if now - pos.pending_since >= ttl:
            if (pos.extra or {}).get("exec_ttl_ms"):
                return f"메이커 진입 유효 시간({ttl // MS_MINUTE}분) 만료로 주문 취소"
            return f"진입 유효 기간({pos.entry_ttl_bars}봉) 만료로 주문 취소"
        blocked = self.risk.entries_blocked(now) if self.entries_enabled else "entries disabled"
        blocked = blocked or self.shock_halt(now, pos.alpha)
        return f"신규 진입 중단({blocked})으로 대기 주문 취소" if blocked else ""

    async def _settle_entry_order(self, pos: Position) -> OrderResult | None:
        """Cancel the working entry order, then read its final state and credit any fill that raced the cancel.

        Returns the query result, or None when the exchange could not be asked (the order may then still rest).
        Once the order is terminal its id is marked settled so a late stream fill for it cannot be double counted."""
        try:
            await self.account.cancel_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] cancel entry order on %s failed: %s", self.name, pos.symbol, e)
        return await self._credit_entry_fills(pos)

    async def _credit_entry_fills(self, pos: Position) -> OrderResult | None:
        """Read the entry order's state from the exchange and credit any executed quantity the engine has not seen
        (a fill whose stream event was missed: restart, user-stream gap, REST timeout). Crediting a first fill turns
        the PENDING row into an OPEN position and places its brackets. Returns the order state, or None when the
        exchange could not be asked. A terminal order is marked settled so a late stream fill cannot double count."""
        if not (pos.entry_order_id or pos.entry_client_id):
            return None
        try:
            res = await self.account.query_order(pos.symbol, pos.entry_order_id, pos.entry_client_id)
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] query entry order on %s failed: %s", self.name, pos.symbol, e)
            return None
        oid = res.order_id or pos.entry_order_id
        if res.filled_qty > pos.filled_qty + 1e-12:
            delta = res.filled_qty - pos.filled_qty
            fee_rate = self.s.maker_fee if pos.entry_style == EntryStyle.LIMIT.value else self.s.taker_fee
            await self.on_fill(Fill(pos.symbol, pos.side.order_side, delta, res.avg_price, delta * res.avg_price * fee_rate,
                                    self.clock(), pos.entry_client_id, oid, kind="ENTRY"))
        if res.status in TERMINAL_ORDER_STATES:
            self._mark_settled(pos, oid)
        return res

    async def _cancel_entry_remainder(self, pos: Position) -> None:
        """Cancel the unfilled rest of a partially filled entry so it cannot fill into an unmanaged position later."""
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        if not (pos.entry_order_id or pos.entry_client_id) or pos.filled_qty >= pos.original_qty - step * 0.5 - 1e-12:
            return
        res = await self._settle_entry_order(pos)
        if res is not None and res.status not in ("NEW", "PARTIALLY_FILLED"):
            pos.original_qty = pos.filled_qty  # nothing rests any more: the position is what was filled
            self._save(pos)

    async def _cancel_pending(self, pos: Position, reason: str) -> None:
        res = await self._settle_entry_order(pos)
        if pos.status is not PositionStatus.PENDING:
            return  # it filled under the cancel (in either ordering): it is a live position now, brackets placed
        if res is None or res.status in ("NEW", "PARTIALLY_FILLED"):
            # the cancel could not be confirmed, or it did not take and the order still works on the exchange: keep
            # it pending rather than forgetting an order that may still fill into an untracked, unprotected
            # position; _manage_pending (or the caller) retries on the next bar
            await self._emit("error", {"message": f"{pos.symbol} 진입 주문 취소 확인 실패; 다음 봉에 재시도"})
            return
        if pos.filled_qty > 0:  # partially filled: manage what we have
            pos.original_qty = pos.filled_qty
            return
        pos.status = PositionStatus.CANCELLED
        pos.exit_reason = reason
        pos.exit_time = self.clock()
        self._last_exit_ts[pos.symbol] = self.clock()
        self._save(pos)
        self.positions.pop(pos.symbol, None)
        await self._emit("entry_cancelled", {"position": pos, "symbol": pos.symbol, "reason": reason})

    async def on_tick(self, symbol: str, mark: float, bid: float, ask: float) -> None:
        """High-frequency backstop: enforce stops in software if the exchange stop is missing or late."""
        pos = self.positions.get(symbol)
        if pos is None or pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING) or not mark:
            return
        fav = (mark - pos.entry_price) * pos.side.sign
        if fav > pos.max_fav:
            pos.max_fav = fav
        elif -fav > pos.max_adv:
            pos.max_adv = -fav
        through = (pos.side is Side.LONG and mark <= pos.stop * (1 - 0.0012)) or (pos.side is Side.SHORT and mark >= pos.stop * (1 + 0.0012))
        if through and self.clock() - pos.last_stop_update > 2500:
            await self.close_position(symbol, "소프트웨어 백스톱 손절")

    # --- closing -------------------------------------------------------------------------------
    async def close_position(self, symbol: str, reason: str) -> bool:
        pos = self.positions.get(symbol)
        if pos is None:
            return False
        if pos.status is PositionStatus.PENDING:
            await self._cancel_pending(pos, reason)
            if pos.status is PositionStatus.CANCELLED:
                return True
            if pos.status is not PositionStatus.OPEN:
                return False  # still pending (exchange unreachable); retried on the next bar
            # the entry filled under the cancel: flatten it like any open position
        if pos.status is PositionStatus.CLOSING:
            if self.clock() - int(pos.extra.get("close_sent", 0) or 0) < CLOSE_RETRY_MS:
                return False  # a close order is already in flight; its fill (or reconcile) settles it
        elif pos.status is not PositionStatus.OPEN:
            return False
        pos.status = PositionStatus.CLOSING
        pos.exit_reason = reason
        pos.extra["close_sent"] = self.clock()
        self._save(pos)  # a restart must see the close attempt (reconcile settles a CLOSING row either way)
        info = self.symbols[symbol]
        cid = client_order_id(f"C{self.tag}")
        # Send the reduce-only close FIRST and cancel the brackets only once it is accepted: if the order fails
        # (transport error, timeout, rejection) the exchange stop is still protecting the position. A closePosition
        # stop that outlives the position is harmless (and _finalize / reconcile cancel it).
        try:
            await self._cancel_entry_remainder(pos)
            qty = norm_qty(info, pos.qty) or pos.qty
            res = await self.account.market_order(symbol, pos.side.close_order_side, qty, reduce_only=True, client_id=cid)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] close order failed for %s", self.name, symbol)
            return await self._close_failed(pos, str(e))
        if res.status not in ("FILLED", "NEW", "PARTIALLY_FILLED"):
            # REJECTED, or UNKNOWN (the account could not tell whether the send reached Binance): never assume flat
            return await self._close_failed(pos, f"{res.status} {res.raw}")
        if res.status == "FILLED" and pos.status is PositionStatus.CLOSING and res.filled_qty > 0 and not self.account.is_paper:
            await self.on_fill(Fill(symbol, pos.side.close_order_side, res.filled_qty, res.avg_price,
                                    res.filled_qty * res.avg_price * self.s.taker_fee, self.clock(), cid, res.order_id,
                                    reduce_only=True, kind="CLOSE"))
        if pos.status is PositionStatus.CLOSING:  # fill not processed yet: the brackets are no longer needed
            await self._cancel_brackets(pos)
            self._save(pos)
        return True

    async def _close_failed(self, pos: Position, why: str) -> bool:
        """A reduce-only close did not go through. The brackets were left untouched, so the position is still protected
        on the exchange; find out what is left there and either book the exit or hand the position back."""
        symbol = pos.symbol
        try:
            snap = (await self.account.get_positions()).get(symbol)
        except Exception as e:  # noqa: BLE001
            log.error("[%s] close failed for %s (%s) and positions could not be read: %s", self.name, symbol, why, e)
            await self._emit("error", {"message": f"{symbol} 청산 실패 ({why}); 포지션 조회도 실패 ({e}). 손절 유지, 재시도 예정"})
            return False  # stays CLOSING: on_tick / reconcile / the owner retry after CLOSE_RETRY_MS
        if pos.status is PositionStatus.CLOSED or self.positions.get(symbol) is not pos:
            return True  # an exit fill arrived meanwhile and finalized it
        if snap is None or abs(snap.qty) <= 0:
            # nothing left on the exchange -> treat as closed at mark (the remainder is booked in _finalize)
            price = pos.extra.get("last_mark") or pos.entry_price
            await self._finalize(pos, price, self.clock(), (pos.exit_reason or "청산") + " (거래소 포지션 없음)")
            return True
        pos.status = PositionStatus.OPEN
        await self._ensure_stop(pos)
        self._save(pos)
        await self._emit("error", {"message": f"{symbol} 청산 주문 실패 ({why}); 포지션 유지, 손절 유지"})
        return False

    async def close_all(self, reason: str) -> int:
        n = 0
        for sym in list(self.positions):
            if await self.close_position(sym, reason):
                n += 1
        return n

    async def _finalize(self, pos: Position, exit_price: float, ts: int, reason: str) -> None:
        if pos.status is PositionStatus.CLOSED:
            return
        pos.status = PositionStatus.CLOSED
        pos.exit_price = exit_price
        pos.exit_time = ts
        pos.exit_reason = reason
        step = self.symbols[pos.symbol].step_size if pos.symbol in self.symbols else 0.0
        if pos.qty > step * 0.5 + 1e-12:
            # quantity that left the exchange without a fill we processed (reconcile, rejected close, restart):
            # book it at the exit estimate so the trade record, risk cool-down and learning see the real outcome
            pos.realized += (exit_price - pos.entry_price) * pos.qty * pos.side.sign
            pos.fees += pos.qty * exit_price * self.s.taker_fee
            pos.extra["exit_estimated"] = True
        pos.qty = 0.0
        net = pos.net_pnl()
        pos.r_multiple = net / pos.risk_amount if pos.risk_amount else 0.0
        self._last_exit_ts[pos.symbol] = self.clock()
        await self._cancel_entry_remainder(pos)
        await self._cancel_brackets(pos)
        self._save(pos)
        self.positions.pop(pos.symbol, None)
        rec = TradeRecord(position_id=pos.id, engine=self.name, symbol=pos.symbol, side=pos.side.value, alpha=pos.alpha,
                          alphas=list(pos.alphas), regime=pos.regime, entry_time=pos.entry_time, exit_time=ts,
                          entry_price=pos.entry_price, exit_price=exit_price, qty=pos.filled_qty or pos.original_qty,
                          notional=pos.notional, pnl=net, gross=pos.realized, fees=pos.fees, funding=pos.funding,
                          r_multiple=pos.r_multiple, risk_amount=pos.risk_amount, exit_reason=reason,
                          confidence=pos.confidence, params_version=pos.params_version, bars_held=pos.bars_held,
                          max_fav_r=pos.max_fav / pos.r_unit if pos.r_unit else 0.0,
                          max_adv_r=pos.max_adv / pos.r_unit if pos.r_unit else 0.0, reason=pos.reason)
        self.closed.append(rec)
        if len(self.closed) > 5000:
            self.closed = self.closed[-2500:]
        if self.persist:
            self.store.save_trade(rec)
        self.stats.total_trades += 1
        self.stats.realized_today += net
        self.stats.trades_today += 1
        self.stats.wins_today += 1 if net > 0 else 0
        self.stats.last_trade_ts = ts
        if net < 0:
            self.risk.register_loss(pos.symbol, ts, ms=pos.cooldown_ms)
        self.bandit.update(pos.alpha, pos.regime, pos.r_multiple, live=not self.account.is_paper)
        if self.meta is not None and self.meta_learn and pos.extra.get("meta_x"):
            try:
                self.meta.update(pos.alpha, [float("nan") if v is None else v for v in pos.extra["meta_x"]], pos.r_multiple)
            except Exception:  # noqa: BLE001
                log.exception("[%s] meta-label update failed", self.name)
        self._equity_cache = (0, 0.0)
        await self._emit("position_closed", {"position": pos, "trade": rec, "symbol": pos.symbol})

    def reset_daily(self) -> None:
        self.stats.realized_today = 0.0
        self.stats.trades_today = 0
        self.stats.wins_today = 0

    # --- reconciliation (live) -----------------------------------------------------------------
    async def reconcile(self) -> None:
        if self.account.is_paper:
            return
        snap_ts = self.clock()  # anything the engine finalized after this instant is newer than the snapshot
        try:
            ex_positions = await self.account.get_positions()
            algos = await self.account.open_algo_orders()
        except Exception as e:  # noqa: BLE001
            log.warning("[%s] reconcile skipped: %s", self.name, e)
            return
        algo_by_symbol: dict[str, list[dict]] = {}
        for a in algos:
            algo_by_symbol.setdefault(a.get("symbol", ""), []).append(a)
        now = self.clock()
        flipped: set[str] = set()
        # positions we think are open (or that a close left hanging)
        for sym, pos in list(self.positions.items()):
            if pos.status is PositionStatus.PENDING:
                # an entry that executed while its fill event was missed (restart, stream gap): the exchange holds
                # the position, nothing protects it and the symbol is ours so it is never adopted below -> credit
                # the fill now (which places the brackets) instead of waiting for _manage_pending's next bar
                snap = ex_positions.get(sym)
                if snap is not None and abs(snap.qty) > 0 and (snap.qty > 0) == (pos.side is Side.LONG):
                    await self._credit_entry_fills(pos)
                continue
            if pos.status not in (PositionStatus.OPEN, PositionStatus.CLOSING):
                continue
            snap = ex_positions.get(sym)
            if snap is None or abs(snap.qty) <= 0:
                mark = pos.extra.get("last_mark") or pos.stop
                reason = (f"{pos.exit_reason} (reconcile)" if pos.status is PositionStatus.CLOSING and pos.exit_reason
                          else "거래소에서 청산 확인(reconcile)")
                await self._finalize(pos, mark, now, reason)
                await self._emit("reconcile", {"message": f"{sym} 포지션이 거래소에서 이미 종료되어 정리했습니다"})
                continue
            if (snap.qty > 0) != (pos.side is Side.LONG):
                stale_ids = {str(x) for x in (pos.sl_algo_id, pos.tp_algo_id, pos.extra.get("tp1_algo_id", "")) if x}
                await self._finalize(pos, snap.mark or pos.entry_price, now, "방향 불일치(reconcile)")
                # the ids _finalize just cancelled must not count as protection for the position adopted below
                algo_by_symbol[sym] = [a for a in algo_by_symbol.get(sym, []) if self._algo_id(a) not in stale_ids]
                flipped.add(sym)
                continue
            retry_close = False
            if pos.status is PositionStatus.CLOSING:
                if now - int(pos.extra.get("close_sent", 0) or 0) < CLOSE_RETRY_MS:
                    continue  # the close order is still in flight
                # the close never executed (or we could not tell): hand the position back and try again
                log.warning("[%s] %s is still on the exchange after a close attempt; retrying", self.name, sym)
                pos.status = PositionStatus.OPEN
                retry_close = True
            if abs(abs(snap.qty) - pos.qty) > self.symbols[sym].step_size * 0.5:
                pos.qty = abs(snap.qty)
                pos.notional = pos.qty * pos.entry_price
            rows = self._stop_rows(pos, algo_by_symbol.get(sym, []))
            stop_row = rows[0] if rows else None
            if stop_row is not None:
                if self._algo_id(stop_row) and pos.sl_algo_id != self._algo_id(stop_row):
                    pos.sl_algo_id = self._algo_id(stop_row)  # keep the id so _cancel_brackets can clean it up
                # exactly one engine stop protects a position: every other same-side STOP_MARKET is a leftover (a
                # bracket whose cancel failed, the stop of the previous trade on this symbol that survived because
                # the symbol was re-entered before the flat-symbol sweep) and would close the position at a level
                # nobody chose
                for a in rows[1:]:
                    await self._cancel_algo_quiet(sym, self._algo_id(a))
            else:
                log.warning("[%s] %s has no exchange stop, re-placing", self.name, sym)
                try:
                    pos.sl_algo_id = await self.account.place_stop(sym, pos.side.close_order_side, pos.stop, qty=pos.qty,
                                                                   close_position=True, client_id=client_order_id(f"S{self.tag}"))
                    pos.last_stop_update = now
                except Exception as e:  # noqa: BLE001
                    await self._emit("error", {"message": f"{sym} 손절 재설정 실패: {e}"})
            # a STOP_MARKET on the wrong side can never protect this position: it is a zombie from an earlier one
            if not getattr(self.account, "hedge_mode", False):
                for a in algo_by_symbol.get(sym, []):
                    if a.get("orderType", a.get("type")) == "STOP_MARKET" and a.get("side") != pos.side.close_order_side:
                        await self._cancel_algo_quiet(sym, self._algo_id(a))
            self._save(pos)
            if retry_close:
                await self.close_position(sym, pos.exit_reason or "청산 재시도(reconcile)")
        # positions on the exchange we do not know about -> adopt with a protective stop
        candidates = [sym for sym in ex_positions if sym not in self.positions and sym in self.symbols]
        fresh = ex_positions
        if candidates:
            # adoption is rare and must never act on a stale snapshot (a stop may have fired during the awaits above)
            try:
                fresh = await self.account.get_positions()
            except Exception as e:  # noqa: BLE001
                log.warning("[%s] reconcile: could not refresh positions before adopting: %s", self.name, e)
                fresh = {}
        for sym in candidates:
            snap = fresh.get(sym)
            if snap is None or abs(snap.qty) <= 0 or sym in self.positions:
                continue
            if self._last_exit_ts.get(sym, 0) >= snap_ts and sym not in flipped:
                continue  # the engine closed this symbol during this pass: the snapshot predates that exit
            side = Side.LONG if snap.qty > 0 else Side.SHORT
            view = self.last_views.get(sym)
            atr = view.tf("5m").v("atr") if view and view.tf("5m").ok else snap.entry_price * 0.01
            if not atr or math.isnan(atr):
                atr = snap.entry_price * 0.01
            stop = norm_price(self.symbols[sym], snap.entry_price - side.sign * 2.0 * atr)
            pos = Position(id=short_id("P"), engine=self.name, symbol=sym, side=side, qty=abs(snap.qty),
                           entry_price=snap.entry_price, entry_time=now, stop=stop, take_profit=None, tp1=None,
                           initial_stop=stop, alpha="adopted", alphas=["adopted"], reason="거래소에서 발견된 외부 포지션을 인수",
                           confidence=0.0, regime="RANGE", risk_amount=abs(snap.qty) * 2.0 * atr, r_unit=2.0 * atr,
                           notional=abs(snap.qty) * snap.entry_price, leverage=snap.leverage or self.s.exchange_leverage,
                           params_version=self.params.version, atr=atr, trail_atr_mult=2.5, max_hold_bars=0,
                           status=PositionStatus.OPEN, filled_qty=abs(snap.qty), original_qty=abs(snap.qty),
                           extra={"tp1_frac": 0.0})
            self.positions[sym] = pos
            stop_row = self._protective_stop(pos, algo_by_symbol.get(sym, []))
            if stop_row is None:
                await self._place_brackets(pos)
            else:
                pos.sl_algo_id = self._algo_id(stop_row)
            self._save(pos)
            await self._emit("reconcile", {"message": f"{sym} {side.value} 외부 포지션 인수, 보호 손절 {stop}"})
        # zombie algo orders on flat symbols
        for sym, lst in algo_by_symbol.items():
            if sym and sym not in ex_positions and sym not in self.positions and lst:
                try:
                    await self.account.cancel_all(sym)
                    log.info("[%s] cancelled %d orphan algo orders on %s", self.name, len(lst), sym)
                except Exception:  # noqa: BLE001
                    pass
        # resting entry orders nobody owns (e.g. the rest of a partially filled entry after the position closed)
        try:
            orders = await self.account.open_orders()
        except Exception as e:  # noqa: BLE001
            log.debug("[%s] reconcile: open orders unavailable: %s", self.name, e)
            orders = []
        prefix = f"HLE{self.tag}"
        for o in orders or []:
            sym = o.get("symbol", "")
            cid = str(o.get("clientOrderId", "") or "")
            if not cid.startswith(prefix):
                continue
            pos = self.positions.get(sym)
            if pos is not None and pos.status in (PositionStatus.PENDING, PositionStatus.OPEN) and cid == pos.entry_client_id:
                continue  # the working entry of a live position (its remainder is managed by _manage / close)
            try:
                await self.account.cancel_order(sym, str(o.get("orderId", "") or ""), cid)
                log.info("[%s] cancelled orphan entry order %s on %s", self.name, cid, sym)
            except Exception:  # noqa: BLE001
                pass

    @staticmethod
    def _algo_id(a: dict) -> str:
        return str(a.get("algoId", a.get("algo_id", "")) or "")

    def _stop_rows(self, pos: Position, algos: list[dict]) -> list[dict]:
        """The STOP_MARKETs on this symbol that could protect pos (closing side, and position side in hedge mode),
        best first: the engine's own sl_algo_id, then by trigger distance to pos.stop. A stop on the other side is a
        leftover and counts for nothing."""
        want = pos.side.close_order_side
        hedge = getattr(self.account, "hedge_mode", False)
        rows = [a for a in algos if a.get("orderType", a.get("type")) == "STOP_MARKET" and a.get("side") == want
                and (not hedge or a.get("positionSide") in (None, "BOTH", pos.side.value))]

        def rank(a: dict) -> tuple[int, float]:
            own = bool(pos.sl_algo_id) and self._algo_id(a) == pos.sl_algo_id
            try:
                dist = abs(float(a.get("triggerPrice", a.get("trigger")) or pos.stop) - pos.stop)
            except (TypeError, ValueError):
                dist = float("inf")
            return (0 if own else 1, dist)

        return sorted(rows, key=rank)

    def _protective_stop(self, pos: Position, algos: list[dict]) -> dict | None:
        rows = self._stop_rows(pos, algos)
        return rows[0] if rows else None

    # --- snapshot for UIs ----------------------------------------------------------------------
    def snapshot(self, marks: dict[str, float] | None = None) -> dict:
        marks = marks or {}
        pos_rows = []
        for p in self.open_positions():
            m = marks.get(p.symbol) or p.extra.get("last_mark") or p.entry_price
            pos_rows.append({"id": p.id, "symbol": p.symbol, "side": p.side.value, "status": p.status.value, "qty": p.qty,
                             "entry": p.entry_price, "mark": m, "stop": p.stop, "tp": p.take_profit, "tp1": p.tp1,
                             "tp1_done": p.tp1_done, "unrealized": p.unrealized(m) if p.status is PositionStatus.OPEN else 0.0,
                             "pnl_pct": p.pnl_pct(m), "r": p.r_now(m), "alpha": p.alpha, "alphas": p.alphas,
                             "reason": p.reason, "regime": p.regime, "confidence": p.confidence,
                             "entry_time": p.entry_time, "bars_held": p.bars_held, "notional": p.notional,
                             "leverage": p.leverage, "risk": p.risk_amount, "expected_profit": p.expected_profit})
        return {"name": self.name, "paper": self.account.is_paper, "params_version": self.params.version,
                "equity": self.stats.equity, "balance": self.stats.balance, "start_equity": self.stats.start_equity,
                "realized_today": self.stats.realized_today, "trades_today": self.stats.trades_today,
                "wins_today": self.stats.wins_today, "total_trades": self.stats.total_trades,
                "paused": self.risk.state.paused, "halted": self.risk.state.halted_until > self.clock(),
                "halt_reason": self.risk.state.halt_reason, "risk_scale": self.risk.state.risk_scale,
                "positions": pos_rows, "skipped": self.stats.skipped, "entries_enabled": self.entries_enabled,
                # P3: open risk in % of equity (total, beta-weighted per side) while the portfolio allocator is on
                "risk_budget": (self.allocator.exposure(self.open_positions(), self.stats.equity)
                                if self.allocator is not None else None)}

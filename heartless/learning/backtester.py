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
from heartless.exchange.paper import PaperAccount
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
        rows = self.funding.get(sym)
        if not rows:
            return 0.0
        idx = int(np.searchsorted(self._fund_idx[sym], t, side="right")) - 1
        return rows[idx][1] if idx >= 0 else 0.0

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
        account = PaperAccount(name="bt", initial_balance=initial_balance, taker_fee=self.s.taker_fee,
                               maker_fee=self.s.maker_fee, clock=lambda: self.now)
        account.set_symbols(self.symbols)
        engine = TradingEngine("bt", account, params, self.s, self.symbols, store=None, bus=None,
                               bandit=bandit or AlphaBandit(list(params.alphas), seed=7), only_alpha=only_alpha,
                               clock=lambda: self.now, persist=False)
        self.now = int(tl[lo]) if lo < len(tl) else 0
        await engine.start()
        # per-symbol cursors into candle arrays
        cursors = {s: int(np.searchsorted(ca.view("close_time"), tl[lo])) for s, ca in self.candles.items()}
        regimes: dict[str, tuple[Regime, dict]] = {s: (Regime.RANGE, {}) for s in self.candles}
        btc_regime: Regime | None = None
        equity_curve: list[tuple[int, float]] = []
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
                            funding_rate=self._funding_rate(sym, t), next_funding_time=(bucket + 1) * FUNDING_INTERVAL)
                ctx = Context(symbol=sym, info=self.symbols[sym], ticker=tk, regime=reg, regime_info=reg_info,
                              btc_regime=btc_regime if sym != "BTCUSDT" else None, oi_change=None, now=t,
                              funding_rate=tk.funding_rate, minutes_to_funding=((bucket + 1) * FUNDING_INTERVAL - t) / MS_MINUTE)
                pos = engine.positions.get(sym)
                if pos is not None:
                    pos.extra["last_mark"] = c.close
                await engine.on_bar(view, ctx)
            if i % 60 == 0:
                eq = account.equity()
                equity_curve.append((t, eq))
                engine._equity_cache = (t, eq)
                for _ in engine.risk.observe_equity(eq, t):
                    pass
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
        funding[s] = store.load_funding(s, since - FUNDING_INTERVAL, until or 2**62)
    return Backtester(settings, symbols, candles, funding)

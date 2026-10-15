"""Portfolio allocator (ROADMAP P3): which of one bar close's entry candidates get the account's risk.

Without it (PORTFOLIO_ALLOCATOR off, the default) every symbol's entry is admitted the moment its bar is processed, so
on a market-wide move the symbols that happen to come first take the position limits (docs/RESEARCH.md section 6).
With it the engine queues the candidates of a bar close (TradingEngine.on_bar) and opens them together once every
symbol's bar at that close has been processed (TradingEngine.flush_entries): the joint backtester flushes after each
step of its timeline, the live app once every traded symbol reported the close (or a later close / a grace period
arrives first). Both run this module, so the backtest measures the exact admission rule the bot trades.

1. Ranking (rank): by expected value.
   * expected R where an estimate exists: the meta model's expected R when it is enabled and ready for the alpha,
     else the evidence registry's mean R of an alpha whose out-of-sample verdict is 'passed';
   * otherwise the bandit's posterior mean for (alpha, regime) (no exploration noise) x the signal's confidence.
   Candidates with an R estimate come first (by R), the rest by the bandit score. Equal values (e.g. one alpha, no
   evidence yet) go to the lower BTC beta first, the candidate that uses less of its side's budget, then by symbol,
   so a live bot and a backtest order the same candidates the same way regardless of stream arrival.
2. Limits (check), applied in rank order after the engine's count caps (MAX_POSITIONS, MAX_SAME_DIRECTION, gross
   leverage) and its sizing:
   * total open risk (every pending / open position's risk to its current stop, plus the candidate's planned risk)
     <= MAX_TOTAL_RISK_PCT of equity;
   * same-direction risk weighted by BTC beta (sum over the positions of that side of risk x beta, plus the
     candidate's) <= MAX_SAME_DIRECTION_RISK_PCT. Beta is the slope of the symbol's 1h log returns on BTC's over the
     last BETA_LOOKBACK_DAYS (BetaEstimator), bounded to [BETA_MIN, BETA_MAX]; 1.0 while too little history exists.
   A candidate that does not fit is skipped (never shrunk), and the next one in rank order is tried.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from heartless.core.models import Decision, Position, PositionStatus, Side
from heartless.util.timeutil import MS_HOUR

if TYPE_CHECKING:  # pragma: no cover
    from heartless.config import Settings
    from heartless.data.features import MarketView
    from heartless.learning.bandit import AlphaBandit
    from heartless.strategy.base import Context

BTC = "BTCUSDT"
BETA_MIN, BETA_MAX = 0.5, 2.5  # a 30-day estimate is noisy: no symbol counts as a hedge or as more than 2.5 BTCs
BETA_MIN_HOURS = 168  # fewer aligned hourly returns than a week: beta 1.0
TOTAL_REASON = "total risk budget"
SAME_REASON = "same-direction risk budget"
_ACTIVE = (PositionStatus.PENDING, PositionStatus.OPEN, PositionStatus.CLOSING)


@dataclass
class Candidate:
    """One entry decision waiting for its bar close's batch (TradingEngine.flush_entries)."""
    decision: Decision
    view: "MarketView"
    ctx: "Context"
    # (meta vector, allow, size multiplier, expected R) once the meta model was asked; None while it was not
    meta: tuple | None = None
    tier: int = 0  # 1: ranked by an expected R, 0: by the bandit score
    value: float = 0.0
    source: str = ""  # "meta" | "evidence" | "bandit"
    extra: dict = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.decision.symbol


class BetaEstimator:
    """BTC beta per symbol from the last `hours` 1h closes.

    Fed one closed hour at a time (observe; idempotent per (symbol, hour), so several engines may share one) and primed
    from stored history (prime). Beta is the OLS slope of the symbol's hourly log returns on BTC's over the hours both
    have, recomputed at most once per new hour."""

    def __init__(self, hours: int = 720, btc: str = BTC, min_hours: int = BETA_MIN_HOURS,
                 lo: float = BETA_MIN, hi: float = BETA_MAX):
        self.hours = int(hours)
        self.btc = btc
        self.min_hours = int(min_hours)
        self.lo, self.hi = float(lo), float(hi)
        self._bars: dict[str, deque] = {}  # symbol -> deque[(hour open ms, close)], oldest first
        self._cache: dict[str, tuple[int, int, float]] = {}  # symbol -> (its last hour, BTC's last hour, beta)
        # symbols whose stored history was already asked for (Heartless._prime_beta): kept on the estimator, not per
        # engine name, because a re-created engine (live after a mode switch, a fresh challenger in a reused slot)
        # starts with an empty estimator and must be seeded again
        self.tried: set[str] = set()

    def observe(self, symbol: str, hour_open: int, close: float) -> None:
        if not (close > 0 and math.isfinite(close)):
            return
        dq = self._bars.get(symbol)
        if dq is None:
            dq = self._bars[symbol] = deque(maxlen=self.hours + 1)
        if dq and int(hour_open) <= dq[-1][0]:
            return  # already seen (another engine, a re-sent bar) or older than what is held
        dq.append((int(hour_open), float(close)))

    def prime(self, symbol: str, hour_opens: Sequence[int], closes: Sequence[float]) -> None:
        """Seed with stored hourly closes. Merged with what is held (an hour already observed keeps its value), so it
        may run after the first live hours were observed; the newest `hours` + 1 are kept."""
        merged = {int(t): float(c) for t, c in zip(hour_opens, closes) if c > 0 and math.isfinite(c)}
        if not merged:
            return
        dq = self._bars.get(symbol)
        if dq:
            merged.update(dict(dq))
        self._bars[symbol] = deque(sorted(merged.items()), maxlen=self.hours + 1)
        self._cache.clear()  # older hours change every beta computed against this series (BTC's: all of them)

    def known(self, symbol: str) -> bool:
        return bool(self._bars.get(symbol))

    def raw_beta(self, symbol: str) -> float | None:
        """Unbounded slope, None while fewer than min_hours aligned returns exist."""
        if symbol == self.btc:
            return 1.0
        a, b = self._bars.get(symbol), self._bars.get(self.btc)
        if not a or not b:
            return None
        bt = dict(b)
        ra: list[float] = []
        rb: list[float] = []
        prev = None
        for t, c in a:
            if prev is not None and t - prev[0] == MS_HOUR:
                b0, b1 = bt.get(prev[0]), bt.get(t)
                if b0 and b1:
                    ra.append(math.log(c / prev[1]))
                    rb.append(math.log(b1 / b0))
            prev = (t, c)
        if len(ra) < self.min_hours:
            return None
        x, y = np.asarray(rb), np.asarray(ra)
        var = float(np.var(x))
        if var <= 0:
            return None
        return float(np.mean((x - x.mean()) * (y - y.mean())) / var)

    def beta(self, symbol: str) -> float:
        """Bounded beta used as the same-direction risk weight (1.0 without enough history)."""
        if symbol == self.btc:
            return 1.0
        a, b = self._bars.get(symbol), self._bars.get(self.btc)
        key = (a[-1][0] if a else 0, b[-1][0] if b else 0)
        hit = self._cache.get(symbol)
        if hit is not None and (hit[0], hit[1]) == key:
            return hit[2]
        raw = self.raw_beta(symbol)
        val = 1.0 if raw is None else min(self.hi, max(self.lo, raw))
        self._cache[symbol] = (key[0], key[1], val)
        return val


def hourly_closes(open_times: np.ndarray, closes: np.ndarray, before: int | None = None,
                  limit: int | None = None) -> tuple[list[int], list[float]]:
    """(hour open times, closes) of the hours in 1m bars (open time, close) whose last minute (xx:59) is stored, with
    that minute's close as the hour's close. `before` keeps the hours closed by then, `limit` the newest that many."""
    if len(open_times) == 0:
        return [], []
    ot = np.asarray(open_times, dtype=np.int64)
    cl = np.asarray(closes, dtype=float)
    last_min = (ot % MS_HOUR) == MS_HOUR - 60_000
    hours = ot[last_min] - (MS_HOUR - 60_000)
    vals = cl[last_min]
    if before is not None:
        keep = hours + MS_HOUR <= before
        hours, vals = hours[keep], vals[keep]
    if limit is not None and len(hours) > limit:
        hours, vals = hours[-limit:], vals[-limit:]
    return [int(h) for h in hours], [float(v) for v in vals]


class PortfolioAllocator:
    """Ranking and risk-budget admission for one engine's entry batches (module docstring)."""

    def __init__(self, settings: "Settings", beta: BetaEstimator | None = None):
        self.s = settings
        self.beta = beta or BetaEstimator(hours=int(getattr(settings, "beta_lookback_days", 30)) * 24)

    # --- ranking -------------------------------------------------------------------------------------------------
    @staticmethod
    def value(cand: Candidate, bandit: "AlphaBandit | None") -> tuple[int, float, str]:
        """(tier, value, source) of a candidate (module docstring, 1.)."""
        from heartless.strategy import evidence

        d = cand.decision
        alpha = d.primary.alpha
        er = cand.meta[3] if cand.meta is not None and len(cand.meta) >= 4 else None
        if er is not None and math.isfinite(float(er)):
            return 1, float(er), "meta"
        if evidence.verdict(alpha) == "passed":
            return 1, float(evidence.expected_r(alpha)), "evidence"
        w = bandit.weight(alpha, d.regime, explore=False) if bandit is not None else 1.0
        # weight = shrunk posterior mean / 0.5 (AlphaBandit.weight): back to the mean, times the signal confidence
        return 0, 0.5 * float(w) * float(d.confidence), "bandit"

    def rank(self, cands: Iterable[Candidate], bandit: "AlphaBandit | None" = None) -> list[Candidate]:
        out = list(cands)
        for c in out:
            c.tier, c.value, c.source = self.value(c, bandit)
        return sorted(out, key=lambda c: (-c.tier, -c.value, self.beta.beta(c.symbol), c.symbol))

    # --- limits --------------------------------------------------------------------------------------------------
    @staticmethod
    def open_risk(p: Position) -> float:
        """A position's remaining risk to its current stop (USDT): its risk at the entry scaled by the part still held
        and by how much of the initial stop distance is left (0 once the stop is at or past breakeven)."""
        if p.status not in _ACTIVE:
            return 0.0
        risk = float(p.risk_amount or 0.0)
        if p.status is not PositionStatus.PENDING and p.filled_qty > 0:
            risk *= min(1.0, max(p.qty, 0.0) / p.filled_qty)
        if p.r_unit > 0:
            risk *= min(1.0, max(0.0, (p.entry_price - p.stop) * p.side.sign / p.r_unit))
        return risk

    def exposure(self, positions: Iterable[Position], equity: float) -> dict[str, Any]:
        """Open risk in % of equity: total and beta-weighted per side (status UIs, check)."""
        tot, side = 0.0, {Side.LONG: 0.0, Side.SHORT: 0.0}
        for p in positions:
            r = self.open_risk(p)
            if r <= 0:
                continue
            tot += r
            side[p.side] += r * self.beta.beta(p.symbol)
        k = 100.0 / equity if equity > 0 else 0.0
        return {"total_pct": tot * k, "long_pct": side[Side.LONG] * k, "short_pct": side[Side.SHORT] * k}

    def check(self, symbol: str, side: Side, risk_amount: float, positions: Iterable[Position],
              equity: float) -> tuple[bool, str]:
        """May an entry risking `risk_amount` open next to `positions`? (module docstring, 2.)"""
        if equity <= 0:
            return False, TOTAL_REASON
        ex = self.exposure(positions, equity)
        add = risk_amount / equity * 100.0
        if ex["total_pct"] + add > self.s.max_total_risk_pct + 1e-9:
            return False, TOTAL_REASON
        same = ex["long_pct"] if side is Side.LONG else ex["short_pct"]
        if same + add * self.beta.beta(symbol) > self.s.max_same_direction_risk_pct + 1e-9:
            return False, SAME_REASON
        return True, ""

"""Two-leg neutral evaluator frame (Wave F, F-pre-A; docs/ROADMAP.md 7.4, 7.7.1, 7.7.3; docs/RESEARCH_TOOLS.md 6).

carry_neutral (spot long + perp short) and pairs_neutral (two perps) earn from a structure, not from a price forecast,
so they are judged on a daily mark-to-market equity in percent of the capital C instead of R per trade. This module
is the shared frame; the class rules (universe filter, decision times, entries, exits, rebalancing, ranking) are
plug-ins written by the class researcher as a NeutralStrategy.

Account (one pool of cash C, spot and futures wallets together; transfers are free and instant):
* a position has two or more legs (symbol, market "perp" | "spot", signed quantity in contract units; spot prices
  are scaled by the contract multiplier, so 1000PEPEUSDT's spot leg trades PEPEUSDT x 1000 like basis_series). Spot
  legs are long only (no borrowing, 7.7.1).
* fills at the close of the decision bar: the far side of the synthetic book (half of a 1 bp spread) plus the market's
  slippage, the market's fee on the fill notional (ROADMAP 7.4 cost table: perp taker 5 bp + 1.5 bp, spot 10 bp +
  3 bp; stress fee x1.5, slippage x2).
* perp legs are isolated: margin = notional / leverage at the fill, liquidation price where margin + unrealized =
  maintenance (1.5%) x notional. A leg whose 1h high (short) or low (long) reaches it is liquidated (the whole margin
  is lost, the leg's equity becomes 0) before anything at that bar's close; on_liquidation decides the rest (default:
  close the other legs at that close). Funding settles into the cash pool, not the margin, so the liquidation price
  only moves when the leg is re-struck.
* rebalance (a hook): the strategy returns new leg quantities; the frame trades the differences at the close with
  costs and re-strikes every changed perp leg at that close (its unrealized PnL is realized into cash, entry = close,
  margin = |qty| x close / leverage, a new liquidation price), so "restore the margin to notional / leverage" is exact.
* funding on every perp leg at its stored settlements by the time convention: a settlement s (on the open of bar k)
  is charged when entry close <= s < exit close, at the close before s (PaperAccount.apply_funding).
* delisting: an instrument's data ends at its last traded hour; positions with a leg on it are closed at that close,
  that leg with 5x the slippage. Positions still open at the end of the window are closed at its last close, and so
  are positions with a leg whose stored data stops earlier without a delisting ("data_end", nominal costs). Entries
  are never refused because a bar is an instrument's last one (that is only known when the next bar does not come):
  a position entered at that close is closed at the next hour close, its leg on the stopped instrument at the last
  close (with 5x the slippage for a delisting), the other legs at that next close. An instrument on which the close
  step has just settled a position takes no entry at that same close: that settlement is the end of its trading, and
  without it the key and the slot would still be held (the strategy that kept wanting the position re-entered at the
  close it was forced out at and paid the delisting close twice).
* entries only in the months the symbols are members of the topN research universe (or the given symbols); open
  positions keep their exit rules after membership ends. Decisions see only bars that closed by the decision.

Statistics (7.4): daily net PnL in % of C from the equity marked at every UTC day end (days without trades are 0),
the gating t = the Newey-West HAC t of that daily series (Bartlett kernel, fixed lag HAC_LAG_DAYS = 14; ROADMAP
section 6, 2026-10-16: carry PnL runs for days, so the plain daily t overstates the evidence), the plain daily t
mean / (sd / sqrt(days)) and the weekly-block t for information, annualized = daily mean x 365, max drawdown on the
daily equity curve, calendar years and quarters, liquidations. record() writes the run to the trial ledger (tool
"neutral", t_day = the HAC t; the window and costs stay out of the configuration); train_gate() and
confirmation_gate() apply the registered gates (docs/results/wave_f_registration.json "gates") to results.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from heartless.execution.stats import hac_t
from heartless.learning import seal
from heartless.learning.hourly import (COST_TABLE, DELIST_SLIP_MULT, HALF_SPREAD_BPS, MS_D, MS_H, STRESS, HourlyBars,
                                       ReadOnlyStore, alpha_code_hash, load_bars)

MAINTENANCE = 0.015  # ROADMAP 7.7.1: maintenance margin rate, fixed
LEVERAGE = 2.0  # ROADMAP 7.13: carry / neutral classes at 2x or less
# ROADMAP section 6 (2026-10-16): the neutral gates use the Newey-West t of the daily net PnL with this fixed lag (days)
HAC_LAG_DAYS = 14


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# --- costs -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LegCosts:
    """Fee and slippage fractions per market (ROADMAP 7.4), the synthetic half spread and the delisting multiple."""
    fee: dict
    slippage: dict
    half_spread: float = HALF_SPREAD_BPS / 1e4
    delist_mult: float = DELIST_SLIP_MULT

    @classmethod
    def of(cls, exec: str = "taker", stress: bool = False) -> "LegCosts":
        if exec != "taker":
            raise ValueError(f"exec={exec!r}: the neutral frame fills market orders (taker); a maker variant needs its "
                             f"own fill model and registration")
        fm, sm = (STRESS["fee"], STRESS["slippage"]) if stress else (1.0, 1.0)
        return cls(fee={m: row["taker"] / 1e4 * fm for m, row in COST_TABLE.items()},
                   slippage={m: row["slippage"] / 1e4 * sm for m, row in COST_TABLE.items()})

    def fill(self, market: str, close: float, buy: bool, slip_mult: float = 1.0) -> float:
        """Fill price of a market order at a close: far side of the synthetic book plus slippage."""
        s = 1.0 if buy else -1.0
        return close * (1 + s * self.half_spread) * (1 + s * self.slippage[market] * slip_mult)


# --- positions -------------------------------------------------------------------------------------------------------
@dataclass
class LegSpec:
    """A leg to open: notional in quote currency at the fill (quantity = notional / fill price), side +1 / -1."""
    symbol: str
    market: str
    side: int
    notional: float


@dataclass
class Candidate:
    """An entry the strategy proposes at a decision; `key` identifies the position (one per key), larger `score`
    first when there are more candidates than free slots."""
    key: str
    score: float
    legs: list[LegSpec]
    info: dict = field(default_factory=dict)


@dataclass
class Leg:
    symbol: str
    market: str
    qty: float  # signed contract units
    entry: float  # perp: the strike (fill or last re-strike); spot: the average fill (for reports)
    margin: float = 0.0  # perp isolated margin
    liq: float = float("nan")  # perp liquidation price
    fees: float = 0.0
    funding: float = 0.0
    cash: float = 0.0  # every cash flow of the leg (fills, fees, margin, funding): its realized PnL once closed
    alive: bool = True


@dataclass
class Position:
    key: str
    legs: list[Leg]
    opened: int  # entry close T
    info: dict = field(default_factory=dict)
    closed: int | None = None
    reason: str = ""
    rebalances: int = 0
    pnl: float = 0.0  # realized total (all legs, costs and funding included), set on close


# --- strategy plug-in ------------------------------------------------------------------------------------------------
class NeutralStrategy:
    """Class rules for the frame (override what the class needs):

    * markets: the data each symbol needs (("perp", "spot") for carry, ("perp",) for pairs);
    * eligible(ctx, symbol, T): the class's universe filter on top of membership (e.g. a spot market and its volume);
    * is_decision(ctx, T): whether the close at T is a decision time for exits and entries (default every hour);
      rebalance is asked at every close, so a margin rule never waits for the next decision;
    * candidates(ctx, T, book): entries proposed at T (Candidate: key, score, legs); the frame opens the best scores
      while slots (max_positions) are free and the key is not held;
    * exit_reason(ctx, T, pos): a reason to close an open position at T, or None;
    * rebalance(ctx, T, pos): new signed quantities per leg (same order) or None; the frame trades the differences;
    * on_liquidation(ctx, T, pos, leg): what to do after a leg was liquidated (default: close the rest at T);
    * prepare(ctx): once after loading, e.g. to precompute causal indicator arrays.
    Decisions must read only bars that closed by T (ctx.idx / ctx.upto); the gatekeeper audits that."""
    name: str = "neutral"
    family: str = ""
    defaults: dict = {}
    markets: tuple = ("perp",)
    leverage: float = LEVERAGE
    maintenance: float = MAINTENANCE
    max_positions: int = 8
    warmup_hours: int = 720
    shock_exempt: bool = False  # ROADMAP 7.8: the shock layer's entry halt does not apply (registered exemptions only)

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)} (known: {sorted(self.defaults)})")
        self.params = {**self.defaults, **params}

    def prepare(self, ctx: "NeutralContext") -> None:
        return None

    def eligible(self, ctx: "NeutralContext", symbol: str, T: int) -> bool:
        return True

    def is_decision(self, ctx: "NeutralContext", T: int) -> bool:
        return True

    def candidates(self, ctx: "NeutralContext", T: int, book: "Book") -> list[Candidate]:
        return []

    def exit_reason(self, ctx: "NeutralContext", T: int, pos: Position) -> str | None:
        return None

    def rebalance(self, ctx: "NeutralContext", T: int, pos: Position) -> list[float] | None:
        return None

    def on_liquidation(self, ctx: "NeutralContext", T: int, pos: Position, leg: Leg) -> str:
        """Return "close" (close the other legs at T, default) or "keep"."""
        return "close"

    def config(self) -> dict:
        return {"strategy": self.name, "params": dict(sorted(self.params.items())), "code": alpha_code_hash(self),
                "markets": list(self.markets), "leverage": self.leverage, "maintenance": self.maintenance,
                "max_positions": self.max_positions}


# --- data context ----------------------------------------------------------------------------------------------------
class NeutralContext:
    """Loaded series and the universe of a run. series[(symbol, market)] are HourlyBars (spot scaled by the contract
    multiplier); idx(symbol, market, T) is the bar whose close is T (None outside the data or after its last traded
    hour); members(T) the topN members of T's month (all symbols without top)."""

    def __init__(self, start: int, end: int, series: dict, members_by_month: dict | None, symbols: list[str],
                 last_bar: dict, delisted: dict, spot_symbol: dict):
        self.start, self.end = start, end
        self.series = series
        self.members_by_month = members_by_month
        self.symbols = symbols
        self.last_bar = last_bar  # (symbol, market) -> index of the last bar a position may live in
        self.delisted = delisted  # (symbol, market) -> True when that last bar is a delisting
        self.spot_symbol = spot_symbol
        self.cache: dict = {}  # for strategies (prepare)

    def bars(self, symbol: str, market: str = "perp") -> HourlyBars | None:
        return self.series.get((symbol, market))

    def idx(self, symbol: str, market: str, T: int) -> int | None:
        b = self.series.get((symbol, market))
        if b is None or not b.n:
            return None
        k = (int(T) - MS_H - int(b.t[0])) // MS_H
        if k < 0 or k > self.last_bar.get((symbol, market), b.n - 1):
            return None
        return int(k)

    def close(self, symbol: str, market: str, T: int) -> float:
        k = self.idx(symbol, market, T)
        return float("nan") if k is None else float(self.series[(symbol, market)].close[k])

    def upto(self, symbol: str, market: str, T: int) -> HourlyBars | None:
        k = self.idx(symbol, market, T)
        return None if k is None else self.series[(symbol, market)].upto(k)

    def members(self, T: int) -> set[str]:
        if self.members_by_month is None:
            return set(self.symbols)
        from heartless.data.symbols_inventory import month_str

        return set(self.members_by_month.get(month_str(int(T)), ()))


# --- the book --------------------------------------------------------------------------------------------------------
class Book:
    """Cash, open positions and the closed ones; every money movement goes through it."""

    def __init__(self, capital: float, costs: LegCosts, leverage: float, maintenance: float):
        self.capital = float(capital)
        self.cash = float(capital)
        self.costs = costs
        self.leverage = float(leverage)
        self.maintenance = float(maintenance)
        self.open: dict[str, Position] = {}
        self.closed: list[Position] = []
        self.fees = 0.0
        self.funding = 0.0
        self.liquidations: list[dict] = []
        self.min_cash = self.cash

    # leg arithmetic
    def liquidation_price(self, leg: Leg) -> float:
        """Price where the isolated leg's margin + unrealized = maintenance x notional."""
        q, e, m, mm = abs(leg.qty), leg.entry, leg.margin, self.maintenance
        if q <= 0:
            return float("nan")
        if leg.qty < 0:
            return (m + q * e) / (q * (1 + mm))
        return max((q * e - m) / (q * (1 - mm)), 0.0)

    def leg_value(self, leg: Leg, price: float) -> float:
        """What the leg is worth in the account at `price`: spot qty x price; perp margin + unrealized."""
        if not leg.alive:
            return 0.0
        if leg.market == "spot":
            return leg.qty * price
        return leg.margin + leg.qty * (price - leg.entry)

    def _trade(self, leg: Leg, delta: float, close: float, slip_mult: float = 1.0) -> float:
        """Trade `delta` contract units of the leg at the close; returns the fill price. Spot: cash pays / receives the
        fill notional. Perp: the leg is re-struck at the close (unrealized realized into cash) before the change, then
        the margin is reset to |qty| x close / leverage; the fill's spread and slippage are paid against the close."""
        if delta == 0:
            return close
        before = self.cash
        buy = delta > 0
        px = self.costs.fill(leg.market, close, buy, slip_mult)
        fee = abs(delta) * px * self.costs.fee[leg.market]
        self.cash -= fee
        self.fees += fee
        leg.fees += fee
        if leg.market == "spot":
            new = leg.qty + delta
            if new < -1e-12:
                raise ValueError(f"spot leg {leg.symbol} would go short ({new}): spot legs are long only")
            if buy:
                leg.entry = (leg.entry * leg.qty + px * delta) / new if new > 0 else px
            self.cash -= delta * px
            leg.qty = max(new, 0.0)
        else:
            # re-strike at the close: realize the unrealized PnL, release the margin
            self.cash += leg.margin + leg.qty * (close - leg.entry)
            # the fill itself: the traded units cost (px - close) per unit beyond the close
            self.cash -= delta * (px - close)
            leg.qty += delta
            leg.entry = close
            leg.margin = abs(leg.qty) * close / self.leverage
            self.cash -= leg.margin
            leg.liq = self.liquidation_price(leg)
        if abs(leg.qty) < 1e-15:
            leg.qty = 0.0
        leg.cash += self.cash - before
        self.min_cash = min(self.min_cash, self.cash)
        return px

    def restrike(self, leg: Leg, close: float) -> None:
        """Re-strike a perp leg at the close without trading: realize its unrealized PnL, reset the margin to
        |qty| x close / leverage (a top-up or a release through the cash pool) and recompute the liquidation price."""
        if leg.market != "perp" or not leg.alive:
            return
        before = self.cash
        self.cash += leg.margin + leg.qty * (close - leg.entry)
        leg.entry = close
        leg.margin = abs(leg.qty) * close / self.leverage
        self.cash -= leg.margin
        leg.liq = self.liquidation_price(leg)
        leg.cash += self.cash - before
        self.min_cash = min(self.min_cash, self.cash)

    def open_position(self, cand: Candidate, T: int, closes: dict) -> Position:
        legs = []
        for spec in cand.legs:
            c = closes[(spec.symbol, spec.market)]
            px = self.costs.fill(spec.market, c, spec.side > 0)
            leg = Leg(spec.symbol, spec.market, 0.0, c)
            self._trade(leg, spec.side * spec.notional / px, c)
            legs.append(leg)
        pos = Position(cand.key, legs, int(T), dict(cand.info))
        self.open[cand.key] = pos
        return pos

    def set_quantities(self, pos: Position, qtys: list[float], closes: dict) -> None:
        """A rebalance: trade every leg to its new signed quantity at the close (costs included) and re-strike every
        perp leg, also an unchanged one, so each holds margin = notional / leverage again."""
        if len(qtys) != len(pos.legs):
            raise ValueError(f"rebalance of {pos.key}: {len(qtys)} quantities for {len(pos.legs)} legs")
        for leg, q in zip(pos.legs, qtys):
            if not leg.alive:
                continue
            c = closes[(leg.symbol, leg.market)]
            if float(q) != leg.qty:
                self._trade(leg, float(q) - leg.qty, c)
            else:
                self.restrike(leg, c)
        pos.rebalances += 1

    def close_position(self, pos: Position, T: int, closes: dict, reason: str, slip: dict | None = None) -> None:
        for leg in pos.legs:
            if leg.alive and leg.qty != 0:
                mult = (slip or {}).get((leg.symbol, leg.market), 1.0)
                self._trade(leg, -leg.qty, closes[(leg.symbol, leg.market)], mult)
            leg.alive = False
        pos.closed, pos.reason = int(T), reason
        pos.pnl = sum(l.cash for l in pos.legs)
        self.open.pop(pos.key, None)
        self.closed.append(pos)

    def liquidate(self, pos: Position, leg: Leg, T: int) -> None:
        """The leg's equity is lost (margin and the unrealized loss up to the liquidation price): it leaves the book
        at 0."""
        self.liquidations.append({"key": pos.key, "symbol": leg.symbol, "market": leg.market, "bar_close": _iso(T),
                                  "liq": leg.liq, "margin": leg.margin})
        leg.alive, leg.qty, leg.margin = False, 0.0, 0.0

    def settle_funding(self, leg: Leg, rate: float, mark: float) -> None:
        pay = -leg.qty * mark * rate  # a short receives a positive rate
        self.cash += pay
        self.funding += pay
        leg.funding += pay
        leg.cash += pay

    def equity(self, prices: dict) -> float:
        return self.cash + sum(self.leg_value(l, prices[(l.symbol, l.market)]) for p in self.open.values()
                               for l in p.legs if l.alive)


# --- the run ---------------------------------------------------------------------------------------------------------
@dataclass
class NeutralResult:
    strategy: str
    family: str
    config: dict
    start: int
    end: int
    capital: float
    daily: dict  # UTC day index -> net PnL (currency) of that day, every day of the window
    equity: dict  # UTC day index -> equity at the day end
    positions: list[dict]
    liquidations: list[dict]
    fees: float
    funding: float
    min_cash: float
    symbols: list[str]
    skipped: list[str]
    membership: dict | None = None
    stress: bool = False
    seconds: float = 0.0

    def daily_pct(self) -> np.ndarray:
        return np.array([self.daily[d] for d in sorted(self.daily)], dtype=float) / self.capital * 100.0

    def stats(self) -> dict:
        """ROADMAP 7.4 neutral statistics (percent of C). "t" is the gating t: the Newey-West HAC t of the daily net
        PnL (Bartlett, lag HAC_LAG_DAYS; also "t_hac"); "t_daily" (the plain t of the mean) and "weekly_t" are for
        information only (ROADMAP section 6, 2026-10-16)."""
        v = self.daily_pct()
        days = sorted(self.daily)
        n = len(v)
        sd = float(np.std(v, ddof=1)) if n > 1 else 0.0
        mean = float(v.mean()) if n else 0.0
        eq = np.array([self.equity[d] for d in days], dtype=float)
        peak = np.maximum.accumulate(np.concatenate(([self.capital], eq)))[1:] if n else np.zeros(0)
        dd = float(np.max((peak - eq) / peak)) * 100.0 if n else 0.0
        t_hac = round(hac_t([float(x) for x in v], HAC_LAG_DAYS), 3)
        return {"days": n, "mean_daily_pct": round(mean, 6), "t": t_hac, "t_hac": t_hac, "hac_lag_days": HAC_LAG_DAYS,
                "t_daily": round(mean / (sd / math.sqrt(n)), 3) if sd > 0 else 0.0,
                "annualized_pct": round(mean * 365, 4), "total_pct": round(float(v.sum()), 4),
                "max_drawdown_pct": round(dd, 4), "weekly_t": self.weekly_t(), "liquidations": len(self.liquidations),
                # positions closed early because a leg's stored data stopped (not a delisting): a lagging market build
                # shows up here instead of as a silently flat tail
                "data_end_closes": sum(1 for p in self.positions if p.get("reason") == "data_end"),
                "positions": len(self.positions), "fees_pct": round(self.fees / self.capital * 100, 4),
                "funding_pct": round(self.funding / self.capital * 100, 4),
                "min_cash_pct": round(self.min_cash / self.capital * 100, 4), "seconds": round(self.seconds, 1)}

    def weekly_t(self) -> float:
        """t of 7-day block sums from the window's first day (information only: daily carry PnL is autocorrelated)."""
        v = self.daily_pct()
        k = len(v) // 7
        if k < 2:
            return 0.0
        w = v[:k * 7].reshape(k, 7).sum(axis=1)
        sd = float(np.std(w, ddof=1))
        return round(float(w.mean() / (sd / math.sqrt(k))), 3) if sd > 0 else 0.0

    def _calendar(self, fmt) -> dict:
        g: dict[str, float] = defaultdict(float)
        for d, pnl in self.daily.items():
            g[fmt(datetime.fromtimestamp(d * MS_D / 1000, tz=timezone.utc))] += pnl / self.capital * 100.0
        return {k: round(v, 4) for k, v in sorted(g.items())}

    def by_year(self) -> dict:
        return self._calendar(lambda dt: str(dt.year))

    def by_quarter(self) -> dict:
        return self._calendar(lambda dt: f"{dt.year}Q{(dt.month - 1) // 3 + 1}")

    def daily_series(self) -> list[list]:
        return [[_iso(d * MS_D)[:10], round(self.daily[d] / self.capital * 100.0, 6),
                 round(self.equity[d] / self.capital * 100.0, 6)] for d in sorted(self.daily)]

    def to_json(self) -> dict:
        return {"evaluator": "neutral", "strategy": self.strategy, "family": self.family, "config": self.config,
                "window": {"start": _iso(self.start), "end": _iso(self.end), "start_ms": self.start, "end_ms": self.end},
                "capital": self.capital, "stress": self.stress, "stats": self.stats(), "by_year": self.by_year(),
                "by_quarter": self.by_quarter(), "liquidations": self.liquidations, "positions": self.positions,
                "daily_equity": self.daily_series(), "symbols": self.symbols, "skipped": self.skipped,
                "membership": self.membership}

    def report(self) -> str:
        s = self.stats()
        lines = [f"[{self.strategy}] neutral {_iso(self.start)[:10]} -> {_iso(self.end)[:10]}"
                 f"{' STRESS' if self.stress else ''}  days={s['days']} mean={s['mean_daily_pct']:+.5f}%/day "
                 f"HAC{s['hac_lag_days']} t={s['t']:+.2f} (daily t={s['t_daily']:+.2f} weekly t={s['weekly_t']:+.2f}) "
                 f"ann={s['annualized_pct']:+.2f}% maxDD={s['max_drawdown_pct']:.2f}% "
                 f"positions={s['positions']} liquidations={s['liquidations']} "
                 + (f"data_end={s['data_end_closes']} " if s["data_end_closes"] else "") +
                 f"fees={s['fees_pct']:.3f}% funding={s['funding_pct']:+.3f}% ({s['seconds']:.1f}s)"]
        lines += [f"  year    {k} {v:+.4f}%" for k, v in self.by_year().items()]
        lines += [f"  quarter {k} {v:+.4f}%" for k, v in self.by_quarter().items()]
        return "\n".join(lines)


def load_context(store, start: int, end: int, markets: tuple, warmup_hours: int, symbols: list[str] | None = None,
                 top: int | None = None) -> tuple[NeutralContext, list[str], dict | None]:
    """Load the run's series: the topN research universe of [start, end] (or `symbols`), each symbol's perp bars and,
    when asked, its spot bars scaled by the contract multiplier; the last bar each instrument may hold a position in
    (its last traded hour when that is a delisting, else the window end)."""
    from heartless.data import symbols_inventory as SI
    from heartless.data import universe as U

    members_by_month = None
    membership = None
    if top:
        res = U.research_universe(store, start, end, int(top))
        if res["unranked"]:
            raise ValueError(f"months {', '.join(res['unranked'])} cannot be ranked from the universe database")
        members_by_month = {m: set(v) for m, v in res["months"].items()}
        syms = sorted(res["symbols"])
        months = U.member_months(U.membership_windows(res["members"], start, end))
        membership = {"rule": f"top{int(top)} (docs/UNIVERSE.md); entries only in member months",
                      "member_months": months, "member_months_total": sum(months.values())}
    else:
        syms = sorted(dict.fromkeys(s.upper() for s in symbols or []))
        if not syms:
            raise ValueError("give symbols or top")
    inv = {r["symbol"]: r for r in store.load_symbol_inventory()}
    ends = {"perp": SI.data_end(store, SI.FUT_TABLE), "spot": SI.data_end(store, SI.SPOT_TABLE)}
    lo, hi = start - (int(warmup_hours) + 1) * MS_H, end - MS_H + 1  # bars that close by the window end
    series, last_bar, delisted, spot_sym, skipped = {}, {}, {}, {}, []
    for sym in syms:
        row = inv.get(sym) or {}
        for market in markets:
            if market == "perp":
                bars = load_bars(store, sym, lo, hi)
            elif market == "spot":
                ss = row.get("spot_symbol")
                if not ss:
                    continue
                spot_sym[sym] = ss
                bars = load_bars(store, ss, lo, hi, SI.SPOT_TABLE)
                if bars is not None:
                    mult = float(row.get("multiplier") or 1.0)
                    for a in (bars.open, bars.high, bars.low, bars.close):
                        a *= mult
            else:
                raise ValueError(f"unknown market {market!r} (perp, spot)")
            if bars is None:
                if market == "perp":
                    skipped.append(sym)
                continue
            if not (bars.qv > 0).any():
                continue
            end_k = int(np.searchsorted(bars.close_time, end + 1, side="right") - 1)
            # the instrument's last traded hour over all stored data (not just the loaded window): a delisting when
            # the store has data for 3 more days (symbols_inventory.symbol_status)
            table = SI.FUT_TABLE if market == "perp" else SI.SPOT_TABLE
            name = sym if market == "perp" else spot_sym[sym]
            lasts = [b["last_trade"] for b in store.kline_block_index(table, name) if b["last_trade"] is not None]
            last = max(lasts) if lasts else None
            gone = (last is not None and ends.get(market) is not None
                    and int(last) < int(ends[market]) - SI.ACTIVE_SLACK_MS)
            lt = int((int(last) - int(bars.t[0])) // MS_H) if gone else end_k
            delisted[(sym, market)] = bool(gone and lt <= end_k)
            last_bar[(sym, market)] = min(lt, end_k)
            series[(sym, market)] = bars
    if membership is not None:
        membership["delisted"] = sorted({s for (s, m), g in delisted.items() if g})
    ctx = NeutralContext(start, end, series, members_by_month, [s for s in syms if s not in skipped], last_bar,
                         delisted, spot_sym)
    return ctx, skipped, membership


def evaluate(strategy: NeutralStrategy, start: int, end: int, universe_db=None, symbols: list[str] | None = None,
             top: int | None = None, capital: float = 1.0, stress: bool = False, exec: str = "taker",
             entry_filter: Callable[[str, int], bool] | None = None, ctx: NeutralContext | None = None,
             unseal=()) -> NeutralResult:
    """Run `strategy` over [start, end] (ms, inclusive) hour by hour (module docstring for the order of events).
    `ctx` may be given directly (tests, synthetic series); otherwise it is loaded from `universe_db`.
    `entry_filter(symbol, T)` can veto entries (the shock layer: shock.evaluator_filter; a filter marked
    `evaluator_policy` stays out of the config hash)."""
    t0 = time.time()
    start, end = int(start), int(end)
    seal.check_window(start, end, unseal)
    costs = LegCosts.of(exec, stress)
    skipped, membership = [], None
    if ctx is None:
        store = ReadOnlyStore(universe_db)
        try:
            ctx, skipped, membership = load_context(store, start, end, tuple(strategy.markets),
                                                    strategy.warmup_hours, symbols, top)
        finally:
            store.close()
    strategy.prepare(ctx)
    book = Book(capital, costs, strategy.leverage, strategy.maintenance)
    # hour closes T (instants T - 1) from the first boundary at or after `start`: as the hourly evaluator, an entry at
    # T = start (decided at the close just before the window) belongs to the window; marks start inside it
    first_T = max(-(-start // MS_H) * MS_H, MS_H)
    last_T = ((end + 1) // MS_H) * MS_H
    day0 = start // MS_D
    daily: dict[int, float] = {d: 0.0 for d in range(day0, (end // MS_D) + 1)}
    equity: dict[int, float] = {}
    prev_eq = book.capital
    positions_out: list[dict] = []

    def closes_at(T: int, pos_list) -> dict:
        out = {}
        for p in pos_list:
            for l in p.legs:
                k = ctx.idx(l.symbol, l.market, T)
                if k is None:  # no bar at T (outside the data): last close at or before
                    b = ctx.series[(l.symbol, l.market)]
                    k = int(min(max((T - MS_H - int(b.t[0])) // MS_H, 0), ctx.last_bar.get((l.symbol, l.market), b.n - 1)))
                out[(l.symbol, l.market)] = float(ctx.series[(l.symbol, l.market)].close[k])
        return out

    def data_ended(symbol: str, market: str, T: int) -> bool:
        """The instrument's last bar closed at or before T (only the close step reads this; entries do not)."""
        b = ctx.series[(symbol, market)]
        return (int(T) - MS_H - int(b.t[0])) // MS_H >= ctx.last_bar.get((symbol, market), 10**12)

    def finish(pos: Position, T: int, reason: str, slip: dict | None = None) -> None:
        book.close_position(pos, T, closes_at(T, [pos]), reason, slip)
        pct = 100.0 / book.capital
        positions_out.append({"key": pos.key, "opened": _iso(pos.opened), "closed": _iso(T), "reason": reason,
                              "pnl_pct": pos.pnl * pct, "rebalances": pos.rebalances,
                              "legs": [{"symbol": l.symbol, "market": l.market, "pnl_pct": l.cash * pct,
                                        "fees_pct": l.fees * pct, "funding_pct": l.funding * pct} for l in pos.legs],
                              "info": pos.info})

    for T in range(first_T, last_T + 1, MS_H):
        bar_open = T - MS_H
        # 1) settlements at this bar's open, for every perp leg open since a close <= the settlement
        for pos in list(book.open.values()):
            for l in pos.legs:
                if not l.alive or l.market != "perp":
                    continue
                b = ctx.series[(l.symbol, "perp")]
                k = (bar_open - int(b.t[0])) // MS_H
                if 1 <= k <= ctx.last_bar.get((l.symbol, "perp"), b.n - 1) and math.isfinite(b.funding[k]):
                    book.settle_funding(l, float(b.funding[k]), float(b.close[k - 1]))
        # 2) liquidations inside the bar (before anything at its close)
        for pos in list(book.open.values()):
            hit = None
            for l in pos.legs:
                if not l.alive or l.market != "perp":
                    continue
                k = ctx.idx(l.symbol, "perp", T)
                if k is None:
                    continue
                b = ctx.series[(l.symbol, "perp")]
                if (l.qty < 0 and b.high[k] >= l.liq) or (l.qty > 0 and b.low[k] <= l.liq):
                    hit = l
                    book.liquidate(pos, l, T)
            if hit is not None and pos.key in book.open:
                if strategy.on_liquidation(ctx, T, pos, hit) == "close":
                    finish(pos, T, "liquidation")
        # 3) delistings and the window end: positions on an instrument at its last bar close there. An instrument whose
        #    stored data stops before the window end without being a delisting (a market built to an earlier end, a
        #    delisting inside the inventory's 3-day slack) also closes the position at its last bar ("data_end"):
        #    held on, its leg would keep a frozen price while the other leg moves, an unhedged PnL that never existed.
        #    A position entered at that last close (entries never know a bar is the last one) closes at the next hour
        #    close, when the missing bar shows the instrument has stopped: that leg at its last close (closes_at)
        #    The instruments settled here take no entry at this close (step 4): the settlement ends their trading, and
        #    a position held into it would otherwise reopen at the close it was forced out at
        settled: set[tuple[str, str]] = set()
        for pos in list(book.open.values()):
            ends = [(l.symbol, l.market) for l in pos.legs if l.alive and data_ended(l.symbol, l.market, T)]
            dl = {k: costs.delist_mult for k in ends if ctx.delisted.get(k)}
            if dl:
                finish(pos, T, "delisting", dl)
            elif T == last_T:
                finish(pos, T, "end")
            elif ends:
                finish(pos, T, "data_end")
            settled.update(ends)  # a position with a stopped leg is closed by one of the three branches above
        # 4) the strategy at the close: exits (decision times), rebalances (every close), entries (decision times)
        decide = T < last_T and strategy.is_decision(ctx, T)
        if decide:
            for pos in list(book.open.values()):
                why = strategy.exit_reason(ctx, T, pos)
                if why:
                    finish(pos, T, why)
        if T < last_T:
            for pos in list(book.open.values()):
                q = strategy.rebalance(ctx, T, pos)
                if q is not None:
                    book.set_quantities(pos, q, closes_at(T, [pos]))
        if decide:
            free = strategy.max_positions - len(book.open)
            if free > 0:
                members = ctx.members(T)
                cands = []
                for c in strategy.candidates(ctx, T, book) or []:
                    if c.key in book.open:
                        continue
                    syms = {s.symbol for s in c.legs}
                    if not syms <= members or not all(strategy.eligible(ctx, s, T) for s in syms):
                        continue
                    if any(ctx.idx(s.symbol, s.market, T) is None for s in c.legs):
                        continue  # no bar at T: the instrument does not trade (yet, or any more)
                    # not refused at an instrument's last bar: that it is the last one is only known an hour later
                    # (step 3 then closes the position as a delisting / data_end) - unless step 3 has just settled a
                    # position on it at this close (its trading ended there)
                    if settled and any((s.symbol, s.market) in settled for s in c.legs):
                        continue
                    if entry_filter is not None and not all(entry_filter(s, T) for s in syms):
                        continue
                    cands.append(c)
                cands.sort(key=lambda c: (-c.score, c.key))
                for c in cands[:free]:
                    closes = {(s.symbol, s.market): ctx.close(s.symbol, s.market, T) for s in c.legs}
                    book.open_position(c, T, closes)
        # 5) the day end: mark both legs, the day's net PnL
        if (T % MS_D == 0 or T == last_T) and T - 1 >= start:
            prices = closes_at(T, list(book.open.values()))
            eq = book.equity(prices)
            d = (T - 1) // MS_D
            daily[d] = daily.get(d, 0.0) + eq - prev_eq
            equity[d] = eq
            prev_eq = eq
    for d in daily:  # a day without an hour close in the window (cannot happen for whole days) keeps the last value
        equity.setdefault(d, prev_eq)
    cfg = {"tool": "neutral", **strategy.config(), "universe": f"top{int(top)}" if top else sorted(ctx.symbols),
           "exec": exec}
    if entry_filter is not None and not getattr(entry_filter, "evaluator_policy", False):
        # a filter that is a choice of the configuration is part of its identity; the shock layer is not (the
        # evaluator's own policy from F-pre-B on, as in the hourly evaluator: a carry / pairs configuration rerun with
        # the layer keeps its hash and N, and the trial records the layer in `extra`)
        cfg["entry_filter"] = getattr(entry_filter, "__name__", type(entry_filter).__name__)
    return NeutralResult(strategy.name, strategy.family, cfg, start, end, float(capital),
                         {d: daily[d] for d in sorted(daily)}, {d: equity[d] for d in sorted(daily)}, positions_out,
                         book.liquidations, book.fees, book.funding, book.min_cash, list(ctx.symbols), skipped,
                         membership, bool(stress), time.time() - t0)


def record(result: NeutralResult, family: str | None = None, extra: dict | None = None, ledger=None) -> dict:
    """Append the run to the trial ledger as tool "neutral" (n = closed positions, t_day = the gating HAC t of the
    daily net PnL, ROADMAP section 6 2026-10-16; the plain daily t, the weekly-block t and the other neutral statistics
    in the line); the configuration excludes the window and costs, so a stress, VALID or FORWARD rerun keeps N."""
    from heartless.learning import trials

    fam = (family or result.family or "").strip()
    s = result.stats()
    return trials.record("neutral", fam, result.config, (result.start, result.end), n=s["positions"], avg_r=None,
                         t_day=s["t_hac"], extra={"stress": result.stress, "evaluator": "neutral",
                                                  "neutral": {k: s[k] for k in ("days", "t_hac", "hac_lag_days",
                                                                                "t_daily", "weekly_t", "annualized_pct",
                                                                                "max_drawdown_pct", "liquidations",
                                                                                "data_end_closes")},
                                              **(extra or {})}, ledger=ledger)


# --- registered gates (docs/results/wave_f_registration.json "gates", ROADMAP 7.6) -----------------------------------
PREREG = seal.REPO_ROOT / "docs" / "results" / "wave_f_registration.json"
# the registered gates as of 2026-10-16, used when the registration file is not readable (an installed package)
DEFAULT_GATES = {
    "train_neutral": {"annualized_net_return_gt": 0.02, "max_drawdown_lt": 0.03, "every_year_net_positive": True,
                      "stress_annualized_net_return_gt": 0.0, "forced_liquidations": 0,
                      "t_hac14_daily_net_ge": "max(3, sqrt(2 ln N_family))"},
    "valid_neutral": {"every_quarter_net_positive": True, "t_hac14_daily_net_ge": 1.0,
                      "quarters": [["2026-01-13", "2026-03-31"], ["2026-04-01", "2026-06-30"],
                                   ["2026-07-01", "2026-09-30"]]},
    "forward_neutral": {"span_net_positive": True, "span_blocks": 3, "t_hac14_daily_net_ge": 1.0},
}


def registered_gate(name: str) -> dict:
    """One neutral gate of the Wave F registration (train_neutral, valid_neutral, forward_neutral)."""
    import json

    try:
        gates = json.loads(PREREG.read_text(encoding="utf-8")).get("gates") or {}
    except (OSError, ValueError):
        gates = {}
    g = gates.get(name)
    if not isinstance(g, dict):
        g = DEFAULT_GATES[name]
    if "t_hac14_daily_net_ge" not in g:  # a gate without the amended statistic would silently gate on nothing
        raise ValueError(f"registered gate {name} has no t_hac14_daily_net_ge (ROADMAP section 6, 2026-10-16)")
    return g


def _day_ms(text: str) -> int:
    return int(datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _verdict(checks: dict[str, bool], values: dict) -> dict:
    return {"passed": bool(checks) and all(checks.values()), "checks": checks, "values": values}


def train_gate(result: NeutralResult, n_family: int, stress: NeutralResult | None = None) -> dict:
    """The registered TRAIN gate of a neutral candidate (7.6 train_neutral): HAC t of the daily net PnL >=
    max(3, sqrt(2 ln N)), annualized net return above 2% and max drawdown under 3% of C, every calendar year net
    positive, the stressed run's annualized net return above 0 (a missing stress run fails that check) and no forced
    liquidation. The plain daily t never gates (ROADMAP section 6, 2026-10-16)."""
    from heartless.learning import trials

    g = registered_gate("train_neutral")
    s = result.stats()
    bar = trials.threshold(n_family)
    years = result.by_year()
    st = stress.stats() if stress is not None else None
    checks = {f"HAC t >= {bar:.2f}": s["t_hac"] >= bar,
              f"annualized > {g['annualized_net_return_gt']:.0%}": s["annualized_pct"] > g["annualized_net_return_gt"] * 100,
              f"max drawdown < {g['max_drawdown_lt']:.0%}": s["max_drawdown_pct"] < g["max_drawdown_lt"] * 100,
              "every year net positive": bool(years) and all(v > 0 for v in years.values()),
              "stress annualized > 0": st is not None and st["annualized_pct"] > g["stress_annualized_net_return_gt"] * 100,
              "no forced liquidation": s["liquidations"] <= int(g.get("forced_liquidations", 0))
                                       and (st is None or st["liquidations"] <= int(g.get("forced_liquidations", 0)))}
    return _verdict(checks, {"t_hac": s["t_hac"], "t_bar": round(bar, 3), "n_family": int(n_family),
                             "t_daily": s["t_daily"], "weekly_t": s["weekly_t"], "annualized_pct": s["annualized_pct"],
                             "max_drawdown_pct": s["max_drawdown_pct"], "by_year": years,
                             "stress_annualized_pct": None if st is None else st["annualized_pct"]})


def confirmation_gate(result: NeutralResult, kind: str) -> dict:
    """The registered confirmation gate (7.6): kind "valid" = every registered VALID-F quarter net positive and HAC t
    >= 1.0; kind "forward" = the registered span net positive and HAC t >= 1.0. Stress is reported, never gating."""
    if kind not in ("valid", "forward"):
        raise ValueError(f"confirmation kind {kind!r}: valid or forward")
    g = registered_gate(f"{kind}_neutral")
    s = result.stats()
    need = float(g["t_hac14_daily_net_ge"])
    checks = {f"HAC t >= {need:g}": s["t_hac"] >= need}
    values: dict = {"t_hac": s["t_hac"], "t_daily": s["t_daily"], "weekly_t": s["weekly_t"], "total_pct": s["total_pct"]}
    if kind == "valid":
        q = {}
        for a, b in g["quarters"]:
            lo, hi = _day_ms(a) // MS_D, _day_ms(b) // MS_D
            q[f"{a}..{b}"] = round(sum(p for d, p in result.daily.items() if lo <= d <= hi) / result.capital * 100, 6)
        values["quarters"] = q
        checks["every quarter net positive"] = all(v > 0 for v in q.values())
    else:
        checks["span net positive"] = s["total_pct"] > 0
    return _verdict(checks, values)


def synthetic_context(start: int, series: dict[tuple[str, str], dict], members_by_month: dict | None = None,
                      end: int | None = None) -> NeutralContext:
    """A context from plain arrays (tests and examples): series[(symbol, market)] = {"close": [...], optional "high",
    "low", "open", "funding", "qv"}, hourly bars opening at `start` - nothing is read from a database."""
    out, last = {}, {}
    n_max = 0
    for (sym, market), cols in series.items():
        c = np.asarray(cols["close"], dtype=float)
        n = len(c)
        n_max = max(n_max, n)
        t = start + np.arange(n, dtype=np.int64) * MS_H

        def col(name, default):
            return np.asarray(cols[name], dtype=float) if name in cols else np.full(n, default, dtype=float)

        out[(sym, market)] = HourlyBars(sym, t, col("open", np.nan) if "open" in cols else c.copy(),
                                        col("high", np.nan) if "high" in cols else c.copy(),
                                        col("low", np.nan) if "low" in cols else c.copy(), c, col("qv", 1e6),
                                        np.full(n, np.nan), col("funding", np.nan), np.full(n, np.nan),
                                        np.full(n, np.nan), np.ones(n, dtype=bool))
        last[(sym, market)] = n - 1
    syms = sorted({s for s, _ in series})
    stop = end if end is not None else start + n_max * MS_H - 1
    return NeutralContext(start, stop, out, members_by_month, syms, last, {k: False for k in out}, {})


def main_cli(settings, args) -> None:
    """`heartless neutral --strategy module:Class`: run a NeutralStrategy over a Wave F period (or --start/--end) on the
    topN research universe (or --symbols) and write the 7.4 statistics; --tag records the run in the trial ledger."""
    import json
    import os
    import sys

    from heartless.data import universe as U
    from heartless.learning import hourly, lab, trials

    try:
        start, end = hourly.resolve_window(args)
        top = U.parse_top(args.universe) if args.universe else None
        symbols = [s.strip().upper() for s in (args.symbols or "").split(",") if s.strip()] or None
        if not top and not symbols:
            raise ValueError("give --universe topN or --symbols")
        strategy = hourly.load_alpha(args.strategy, args.set)
        if not isinstance(strategy, NeutralStrategy):
            raise ValueError(f"{args.strategy} is not a NeutralStrategy")
        tag = (args.tag or "").strip()
        from heartless.strategy import shock as SH

        # F-pre-B (ROADMAP 7.8): the shock layer's entry halt vetoes new positions (rebalances and exits go on); a
        # trial runs with it, and its config hash does not hold it (the same rule as `heartless hourly`)
        SH.check_trial_layer(settings, tag, getattr(strategy, "shock_exempt", False))
        shock_filter, shock_doc = SH.evaluator_filter(settings, strategy, start, end)
        res = evaluate(strategy, start, end, settings.universe_db_path, symbols=symbols, top=top,
                       capital=float(args.capital), stress=bool(args.stress), entry_filter=shock_filter)
        print(res.report())
        if shock_doc is not None:
            shock_doc["blocked_decisions"] = shock_filter.blocked if shock_filter is not None else 0
            print(f"shock layer: {shock_doc['blocked_decisions']} candidate entries blocked; firings "
                  f"{shock_doc['firings']}" + (" (strategy exempt)" if shock_doc["exempt"] else ""), file=sys.stderr)
            if shock_doc["missing"] and not shock_doc["exempt"]:
                print(f"warning: shock layer rules {', '.join(shock_doc['missing'])} not evaluated (no BTC data: "
                      f"SHOCK_CANDLE_DB for S0, UNIVERSE_DB for S1-S3)"
                      + ("; the trial records it under 'shock'" if tag else ""), file=sys.stderr)
        doc = {**res.to_json(), "wave_f_period": trials.wave_f_period(start, end), "argv": list(args.argv),
               "git_commit": seal.git_commit(), "shock": shock_doc,
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if tag:
            doc["trial"] = record(res, tag, extra={"argv": list(args.argv), "shock": shock_doc})
            n_fam = trials.family_tests(tag)
            print(f"trial ledger: {tag} config {doc['trial']['config_hash']} recorded; family N={n_fam}, TRAIN bar "
                  f"t >= {trials.threshold(n_fam):.2f}", file=sys.stderr)
        elif trials.wave_f_period(start, end).startswith("TRAIN"):
            print("note: not recorded in the trial ledger (no --tag FAMILY)", file=sys.stderr)
    except (ValueError, seal.SealedError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    if args.out:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(lab._clean(doc), f, indent=1, default=str)
        print(f"wrote {args.out}", file=sys.stderr)

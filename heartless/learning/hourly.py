"""Hourly directional evaluator (Wave F, F-pre-A; docs/ROADMAP.md 7.4, docs/RESEARCH_TOOLS.md section 5).

Classes that decide on hour closes (liq_cascade, event_funding, AI proposals with >= 1h decisions) are researched on
universe.db's 1h klines and funding of every top-N member, because 1m candles exist only for 20 symbols. The evaluator
reproduces the lab's execution model at hour resolution, so its trades agree with the lab's (lab_equivalence: >= 99%
of trades with the same entry hour and side, mean |R difference| <= 0.02 R on the reference alpha):

* decisions: an HourlyAlpha sees the bars up to and including the decision bar. "The close at T" is the last instant
  of the bar [T - 1h, T); its decision enters at T (the lab fills at the next minute's open, also T). It returns side,
  stop, target (or none) and the longest hold in hours. One position per symbol at a time (an independent account per
  symbol, as the lab); the next decision may come only after the exit bar.
* fills: market orders at the far side of the synthetic book built from the close (a 1 bp spread) plus the slippage
  (the lab's PaperAccount). Stops and targets are rounded to the nearest tick (inferred from the prices as the lab
  does). The R unit is |entry fill - stop| + entry fill x (2 fee + 0.0003), the engine's.
* exits: stop, target and time exit only (no breakeven, no trailing). In every bar after the entry, in this order: a
  gap through the stop at the open (filled at the open's book), a gap through the target at the open, the stop inside
  the bar, the target inside the bar; so the stop comes first when both are touched in the same hour. The time exit is
  at the close of bar entry + max_hold.
* funding (ROADMAP 7.4 time convention): a settlement s is charged when entry close <= s < exit close, i.e. at every
  stored settlement on the open of bars entry + 1 .. exit bar (also when the stop gaps at that open: the backtester
  settles before the bar trades), at the close of the bar before s, as PaperAccount.apply_funding. A position entered
  at the close at s pays s; one closed at the close at s does not. The s rate is usable by decisions from the close at
  s + 1 h (HourlyBars.funding[k] is settled at the open of bar k).
* delisting: a symbol's data ends at its last traded hour (universe.delisting_closes); a position still open then is
  closed at that hour's close with 5x the slippage (10x nominal under stress). No decision is made on that last bar
  (nothing trades after it, as in the lab); entries are never blocked in advance. A delisting before the window
  start counts too: the flat zero-volume files after it never trade (a member month can follow the last trade).
* membership (--universe topN): only decisions whose entry falls in a membership span count; a position open when
  membership ends follows its exit rules. Positions still open at the end of the window are closed at its last close.

The statistics are the lab's (lab.group_stats: avg R, PF, t_day clustered by UTC exit day; years by entry), plus a
daily mark-to-market equity in R. evaluate() refuses sealed windows (seal.check_window) and record() writes the run to
the trial ledger (tool "hourly", the family tag of ROADMAP 7.5; the window and costs stay out of the configuration, so
stress, VALID and FORWARD reruns do not raise N).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from heartless.core.store import Store
from heartless.learning import seal

MS_H = 3_600_000
MS_D = 86_400_000
# ROADMAP 7.4 cost table (docs/results/wave_f_registration.json "costs"), basis points
COST_TABLE = {"perp": {"taker": 5.0, "maker": 2.0, "slippage": 1.5},
              "spot": {"taker": 10.0, "maker": 10.0, "slippage": 3.0}}
STRESS = {"fee": 1.5, "slippage": 2.0}
HALF_SPREAD_BPS = 0.5  # PaperAccount's 1 bp synthetic spread: the book is close x (1 -/+ 0.5 bp)
RISK_BUFFER = 0.0003  # the engine's R unit adds entry x (2 fee + 0.0003)
DELIST_SLIP_MULT = 5.0  # ROADMAP 7.4: delisting close at 5x the run's slippage
RISK_AMOUNT = 50.0  # dollars per 1R: 0.5% of the lab's 10,000 account, so the dollar columns read like the lab's
WARMUP_HOURS = 720
SIM_BUDGET = 1_000_000  # (entries x hours) cells per block of the exit simulation
REASONS = {1: "stop", 2: "target", 3: "stop", 4: "target", 5: "time", 6: "delisting", 7: "end"}


# --- data ------------------------------------------------------------------------------------------------------------
class ReadOnlyStore(Store):
    """Store's readers over a read-only SQLite connection (E2's Store(path, readonly=True): no schema script, no -wal
    / -shm left behind, a running writer's WAL still read). The research databases (universe.db, the long-history
    candles) are shared and an evaluator must never write to them. FileNotFoundError when the file does not exist."""

    def __init__(self, path: str | Path):  # noqa: D107
        super().__init__(path, readonly=True)


def _optional_table(table: str) -> Callable:
    """Loader of an optional hourly series stored like the 1h klines (h1-codec blocks, the value in the close column).
    F-data-h stores OI and the premium index this way; until its tables exist (and Store knows them) the series are
    NaN."""

    def load(store, symbol: str, start: int, end: int):
        from heartless.data.symbols_inventory import load_hourly

        if table not in getattr(store, "_F0_TABLES", ()) or not store.f0_has(table):
            return None
        d = load_hourly(store, symbol, start, end, table)
        return (d["open_time"], d["close"]) if len(d["open_time"]) and "close" in d else None

    return load


# name -> loader(store, symbol, start, end) -> (open_times, values) | None; F-data-h may replace the loaders.
# Contract (no look-ahead): a value keyed at open time t[k] is visible to the decision at the close of bar k
# (t[k] + 1 h), so it must be known by then: an OI snapshot taken at the top of hour H goes on the bar opening at
# H - 1 h or later, a premium-index 1h kline on its own open time (its close is the hour's last value)
OPTIONAL_SERIES: dict[str, Callable] = {"oi": _optional_table("oi_1h"), "premium": _optional_table("premium_1h")}


@dataclass
class HourlyBars:
    """One symbol's hourly bars on a complete grid (bar k opens at t[k]; its close "at T" is t[k] + 1 h). A missing
    hour is filled flat at the previous close with no volume and `stored` False; funding[k] is the rate settled at
    t[k] (NaN without a settlement), usable from the close of bar k on. oi / premium are NaN until F-data-h."""
    symbol: str
    t: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    qv: np.ndarray
    taker_buy_share: np.ndarray
    funding: np.ndarray
    oi: np.ndarray
    premium: np.ndarray
    stored: np.ndarray
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.t)

    @property
    def close_time(self) -> np.ndarray:
        """T of each bar's close (bar open + 1 h): the decision's entry time."""
        return self.t + MS_H

    def upto(self, i: int) -> "HourlyBars":
        """The bars up to and including bar i (views, no copy): what a decision at bar i may see."""
        k = int(i) + 1
        return HourlyBars(self.symbol, self.t[:k], self.open[:k], self.high[:k], self.low[:k], self.close[:k],
                          self.qv[:k], self.taker_buy_share[:k], self.funding[:k], self.oi[:k], self.premium[:k],
                          self.stored[:k])

    def atr(self, n: int = 14) -> np.ndarray:
        """Wilder ATR (indicators.atr, the feature frames' "atr"), cached per object."""
        key = ("atr", n)
        if key not in self._cache:
            from heartless.data import indicators as I

            self._cache[key] = I.atr(self.high, self.low, self.close, n)
        return self._cache[key]

    def tick(self) -> float:
        """The price tick inferred from the stored closes the way the lab infers it from 1m closes
        (lab.infer_symbol_info): the smallest step between distinct closes, normalised to one significant digit."""
        if "tick" not in self._cache:
            self._cache["tick"] = infer_tick(self.close[self.stored])
        return self._cache["tick"]

    def funding_known(self) -> np.ndarray:
        """The last settled rate a decision at each bar's close knows (settlements at or before the bar's open; NaN
        before the first)."""
        key = ("funding_known",)
        if key not in self._cache:
            f = self.funding
            idx = np.where(np.isfinite(f), np.arange(self.n), -1)
            idx = np.maximum.accumulate(idx) if self.n else idx
            self._cache[key] = np.where(idx >= 0, f[np.maximum(idx, 0)], np.nan)
        return self._cache[key]


def infer_tick(close: np.ndarray) -> float:
    """Tick size from prices (lab.infer_symbol_info's rule): 0.01 when the prices never differ."""
    px = np.unique(np.round(np.asarray(close, dtype=float), 8))
    d = np.diff(px)
    d = d[d > 0]
    tick = float(np.round(np.min(d), 8)) if len(d) else 0.01
    if tick > 0:
        mag = 10 ** math.floor(math.log10(tick))
        tick = round(round(tick / mag) * mag, 10)
    return tick


def round_to_tick(x: np.ndarray, tick: float) -> np.ndarray:
    """Nearest multiple of `tick`, half to even (the engine's norm_price for stops and targets)."""
    if not tick or tick <= 0:
        return x
    decimals = max(0, -int(math.floor(math.log10(tick)))) + 1
    with np.errstate(invalid="ignore"):
        # the ratio is rounded first so binary residue (8.0695 / 0.001 = 8069.4999...) halves like the decimal
        # arithmetic of round_step (8069.5 -> 8070, half to even)
        ratio = np.round(np.asarray(x, dtype=float) / tick, 6)
        return np.round(np.rint(ratio) * tick, decimals)


def load_bars(store, symbol: str, start: int, end: int, table: str | None = None) -> HourlyBars | None:
    """`symbol`'s bars opening in [start, end] from universe.db as HourlyBars (None without data). Holes become flat
    bars at the previous close; the optional series are aligned by open time."""
    from heartless.data.symbols_inventory import FUT_TABLE, load_hourly

    raw = load_hourly(store, symbol, int(start), int(end), table or FUT_TABLE)
    ot = raw["open_time"]
    if not len(ot):
        return None
    t = np.arange(int(ot[0]), int(ot[-1]) + MS_H, MS_H, dtype=np.int64)
    pos = ((ot - ot[0]) // MS_H).astype(np.int64)
    stored = np.zeros(len(t), dtype=bool)
    stored[pos] = True
    src = np.maximum.accumulate(np.where(stored, np.arange(len(t)), 0))  # last stored bar at or before each hour

    def place(name: str, fill: float, carry: bool = False) -> np.ndarray:
        if name not in raw:
            return np.full(len(t), np.nan)
        out = np.full(len(t), fill, dtype=float)
        out[pos] = raw[name]
        return out[src] if carry else out

    close = place("close", np.nan, carry=True)
    hole = ~stored
    if "open" in raw:
        o, h, lo = (place(k, np.nan) for k in ("open", "high", "low"))
        o, h, lo = (np.where(hole, close, a) for a in (o, h, lo))
    else:  # spot blocks keep only the close: no intrabar range (spot legs have no stop or liquidation)
        o, h, lo = close.copy(), close.copy(), close.copy()
    extra = {}
    for name, loader in OPTIONAL_SERIES.items():
        series = np.full(len(t), np.nan)
        got = loader(store, symbol, int(start), int(end)) if loader is not None else None
        if got is not None:
            st, val = got
            k = (np.asarray(st, dtype=np.int64) - t[0]) // MS_H
            ok = (k >= 0) & (k < len(t))
            series[k[ok]] = np.asarray(val, dtype=float)[ok]
        extra[name] = series
    return HourlyBars(symbol, t, o, h, lo, close, place("quote_volume", 0.0), place("taker_buy_share", np.nan),
                      place("funding", np.nan), extra["oi"], extra["premium"], stored)


# --- alphas ----------------------------------------------------------------------------------------------------------
@dataclass
class Entry:
    """One decision: side +1 long / -1 short, absolute stop and target (None: none), longest hold in hours."""
    side: int
    stop: float
    target: float | None = None
    max_hold: int = 24


@dataclass
class Decisions:
    """Decisions over every bar of an HourlyBars (side 0: none)."""
    side: np.ndarray
    stop: np.ndarray
    target: np.ndarray
    max_hold: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "Decisions":
        return cls(np.zeros(n, dtype=np.int8), np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype=np.int64))

    def at(self, i: int) -> Entry | None:
        if not self.side[i]:
            return None
        tg = float(self.target[i])
        return Entry(int(self.side[i]), float(self.stop[i]), None if not math.isfinite(tg) else tg,
                     int(self.max_hold[i]))


class HourlyAlpha:
    """An alpha for the hourly evaluator. Subclasses set `name`, `family` (ROADMAP 7.5 tag), `defaults`,
    `max_hold_hours` (the longest hold they can ask for) and implement decide(); decide_all() is the optional
    vectorised twin over every bar, which must give at each bar what decide() gives on the bars cut there
    (audit_lookahead checks it; the evaluator uses decide_all when it returns something)."""
    name: str = "hourly_alpha"
    family: str = ""
    defaults: dict = {}
    warmup_hours: int = WARMUP_HOURS
    max_hold_hours: int = 24
    shock_exempt: bool = False  # ROADMAP 7.8: liq_cascade is not paused by the shock layer

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)} (known: {sorted(self.defaults)})")
        self.params = {**self.defaults, **params}

    def decide(self, h: HourlyBars) -> Entry | None:
        """The decision at the close of h's last bar (h holds nothing later)."""
        raise NotImplementedError

    def decide_all(self, h: HourlyBars) -> Decisions | None:
        return None

    def config(self) -> dict:
        """What identifies this alpha in the trial ledger: name, parameters and the source of its module."""
        return {"alpha": self.name, "params": dict(sorted(self.params.items())), "code": alpha_code_hash(self)}


def alpha_code_hash(obj) -> str:
    """Hash of the source of the module defining `obj`'s class (a code change is a new trial)."""
    try:
        src = inspect.getsource(inspect.getmodule(type(obj)))
    except (OSError, TypeError):
        src = type(obj).__qualname__
    return hashlib.sha1(src.encode()).hexdigest()[:16]


def decide_bars(alpha: HourlyAlpha, bars: HourlyBars, idx=None) -> Decisions:
    """The alpha's decisions over `bars`: decide_all when it has one, else decide() on the bars cut at each index of
    `idx` (all bars by default)."""
    d = alpha.decide_all(bars)
    if d is not None:
        if any(len(a) != bars.n for a in (d.side, d.stop, d.target, d.max_hold)):
            raise ValueError(f"{alpha.name}.decide_all returned arrays of the wrong length")
        return d
    d = Decisions.empty(bars.n)
    for i in (range(bars.n) if idx is None else np.asarray(idx).tolist()):
        e = alpha.decide(bars.upto(i))
        if e is not None and e.side:
            d.side[i] = 1 if e.side > 0 else -1
            d.stop[i] = e.stop
            d.target[i] = np.nan if e.target is None else e.target
            d.max_hold[i] = int(e.max_hold)
    return d


def audit_lookahead(alpha: HourlyAlpha, bars: HourlyBars, idx=None) -> list[dict]:
    """Bars where the decision on the full bars differs from the decision on the bars cut at that bar (the gatekeeper's
    look-ahead audit, ROADMAP 7.4). Compares decide_all(full)[i] with decide(cut) and, when vectorised, with
    decide_all(cut)[-1]; NaNs compare equal. Empty = no look-ahead found at the audited bars."""
    full = decide_bars(alpha, bars, idx)
    out = []
    for i in (range(bars.n) if idx is None else np.asarray(idx).tolist()):
        cut = bars.upto(i)
        views = {"decide": alpha.decide(cut)}
        vec = alpha.decide_all(cut)
        if vec is not None:
            views["decide_all"] = vec.at(cut.n - 1)
        want = full.at(i)
        for how, got in views.items():
            if not _same_entry(want, got):
                out.append({"bar": int(i), "t": int(bars.t[i]), "view": how, "full": want, "cut": got})
    return out


def _same_entry(a: Entry | None, b: Entry | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    def eq(x, y):
        return (x is None and y is None) or (x is not None and y is not None and (x == y or (x != x and y != y)))
    return a.side == b.side and eq(a.stop, b.stop) and eq(a.target, b.target) and a.max_hold == b.max_hold


# --- the reference alpha (evaluator equivalence, ROADMAP 7.12 F-pre-A) --------------------------------------------------
REF_PHASE = 1  # 01:00 UTC: entries never sit on a settlement (00/08/16 UTC), where the lab's next-minute fill differs


class ReferenceAlpha(HourlyAlpha):
    """The deterministic non-signal reference: long at every 24th hour close (the close at T with T / 1 h = phase mod
    24), stop 2 x ATR(14) of the 1h bars under the close, no target, time exit after 12 hours. It predicts nothing; it
    only exercises fills, stops, time exits and funding the same way in both evaluators."""
    name = "hourly_reference"
    family = "evaluator_equivalence"
    defaults = {"phase": REF_PHASE, "stop_atr": 2.0, "hold_h": 12}
    warmup_hours = 48
    max_hold_hours = 12

    def decide(self, h: HourlyBars) -> Entry | None:
        i = h.n - 1
        if i < 0 or ((int(h.t[i]) + MS_H) // MS_H) % 24 != self.params["phase"] or not h.stored[i]:
            return None
        atr = float(h.atr(14)[i])
        if not math.isfinite(atr) or atr <= 0:
            return None
        return Entry(1, float(h.close[i]) - self.params["stop_atr"] * atr, None, int(self.params["hold_h"]))

    def decide_all(self, h: HourlyBars) -> Decisions:
        d = Decisions.empty(h.n)
        atr = h.atr(14)
        on = ((h.close_time // MS_H) % 24 == self.params["phase"]) & h.stored & np.isfinite(atr) & (atr > 0)
        d.side[on] = 1
        d.stop[on] = h.close[on] - self.params["stop_atr"] * atr[on]
        d.max_hold[on] = int(self.params["hold_h"])
        return d


def _engine_reference():
    """The reference alpha as an engine Alpha (timeframe 1h on 1m data), defined lazily so importing this module does
    not pull the strategy package (and the registry does not discover it: it lives outside strategy/alphas)."""
    from heartless.core.models import Side, Signal
    from heartless.strategy.base import Alpha, ok
    from heartless.strategy.spec import ParamSpec

    class ReferenceEngineAlpha(Alpha):
        name = "hourly_reference"
        timeframe = "1h"
        description = "evaluator equivalence reference (not a signal)"
        enabled_by_default = False
        param_specs = [ParamSpec("phase", float(REF_PHASE), 0.0, 23.0, 1.0, integer=True),
                       ParamSpec("stop_atr", 2.0, 0.5, 5.0, 0.1), ParamSpec("hold_h", 12.0, 1.0, 72.0, 1.0, integer=True)]

        def evaluate(self, view, ctx, p):
            h = view.tf("1h")
            if not h.ok:
                return None
            if ((int(h.close_time) + 1) // MS_H) % 24 != int(p.get("phase", REF_PHASE)):
                return None
            atr, close = h.v("atr"), h.v("close")
            if not ok(atr, close) or atr <= 0:
                return None
            return Signal(alpha=self.name, symbol=ctx.symbol, side=Side.LONG, confidence=1.0, reason="reference",
                          stop=close - float(p.get("stop_atr", 2.0)) * atr, take_profit=None, tp1=None,
                          max_hold_bars=int(p.get("hold_h", 12)) * 60, trail_atr_mult=0.0, atr=atr, timeframe="1h",
                          time_exit=True, avoid_funding=False, tags={"ref_price": close})

    ReferenceEngineAlpha.__module__ = __name__
    ReferenceEngineAlpha.__qualname__ = "ReferenceEngineAlpha"
    return ReferenceEngineAlpha


def __getattr__(name: str):
    # `heartless.learning.hourly:ReferenceEngineAlpha` for lab.register_alpha / lab.evaluate(extra_alphas=...)
    if name == "ReferenceEngineAlpha":
        cls = _engine_reference()
        globals()[name] = cls
        return cls
    raise AttributeError(name)


# --- costs and the exit simulation -----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Costs:
    """Per-side fee and slippage as fractions, the synthetic half spread, the engine's R buffer and the delisting
    slippage multiple."""
    fee: float
    slippage: float
    half_spread: float = HALF_SPREAD_BPS / 1e4
    buffer: float = RISK_BUFFER
    delist_mult: float = DELIST_SLIP_MULT

    @classmethod
    def of(cls, market: str = "perp", exec: str = "taker", stress: bool = False) -> "Costs":
        """ROADMAP 7.4 costs. Stress: fees x1.5, slippage x2 (the delisting close then pays 5 x that = 10x nominal)."""
        if exec != "taker":
            raise ValueError(f"exec={exec!r}: the hourly evaluator fills market orders; maker execution is evaluated "
                             f"in the lab (ROADMAP 7.7.2)")
        row = COST_TABLE[market]
        fee = row["taker"] / 1e4 * (STRESS["fee"] if stress else 1.0)
        slip = row["slippage"] / 1e4 * (STRESS["slippage"] if stress else 1.0)
        return cls(fee=fee, slippage=slip)


def simulate_exits(bars: HourlyBars, idx: np.ndarray, side: np.ndarray, stop: np.ndarray, target: np.ndarray,
                   hold: np.ndarray, last: int, delisted: bool, cost: Costs) -> dict[str, np.ndarray]:
    """Exit of an entry at the close of each bar in `idx` (module docstring), vectorised in blocks: exit bar, exit
    fill, reason code (REASONS), entry fill, funding per unit (signed, received > 0) and R. `last` is the last bar a
    position may live in (window end or delisting); `delisted` says it is a delisting."""
    m = len(idx)
    o, h, lo, c = bars.open, bars.high, bars.low, bars.close
    half, slip = cost.half_spread, cost.slippage
    sd = side.astype(float)
    entry = c[idx] * (1 + sd * half) * (1 + sd * slip)
    end = np.minimum(idx + hold, last).astype(np.int64)
    ex_bar = end.copy()
    ex_px = np.empty(m)
    code = np.zeros(m, dtype=np.int8)
    span = np.maximum(end - idx, 0)
    per = max(1, SIM_BUDGET // max(int(span.max()) if m else 1, 1))
    for a in range(0, m, per):
        r = slice(a, min(m, a + per))
        H = int(span[r].max()) if len(span[r]) else 0
        if H <= 0:
            continue
        J = idx[r, None] + np.arange(1, H + 1)[None, :]
        valid = J <= end[r, None]
        Jc = np.minimum(J, bars.n - 1)
        s = sd[r, None]
        st, tg = stop[r, None], target[r, None]
        oj = o[Jc]
        adverse = np.where(s > 0, lo[Jc], h[Jc])
        favour = np.where(s > 0, h[Jc], lo[Jc])
        with np.errstate(invalid="ignore"):
            gap_stop = valid & ((oj - st) * s <= 0)
            gap_tgt = valid & ((oj - tg) * s >= 0)
            in_stop = valid & ((adverse - st) * s <= 0)
            in_tgt = valid & ((favour - tg) * s >= 0)
        hit = gap_stop | gap_tgt | in_stop | in_tgt
        has = hit.any(axis=1)
        k = hit.argmax(axis=1)
        rows = np.flatnonzero(has)
        if len(rows):
            kk = k[rows]
            g = np.arange(a, min(m, a + per))[rows]  # global row numbers
            j = idx[g] + kk + 1
            ss = sd[g]
            first = np.select([gap_stop[rows, kk], gap_tgt[rows, kk], in_stop[rows, kk]], [1, 2, 3], 4).astype(np.int8)
            book = o[j] * (1 - ss * half)  # the open's far-side quote for the closing order
            base = np.where(first == 1, np.where(ss > 0, np.minimum(book, stop[g]), np.maximum(book, stop[g])),
                            np.where(first == 2, book, np.where(first == 3, stop[g], target[g])))
            ex_bar[g] = j
            ex_px[g] = base * (1 - ss * slip)
            code[g] = first
    rest = code == 0
    if rest.any():
        j = end[rest]
        ss = sd[rest]
        timed = (idx[rest] + hold[rest]) <= last
        code[rest] = np.where(timed, 5, 6 if delisted else 7)
        mult = np.where(timed | (not delisted), 1.0, cost.delist_mult)
        ex_px[rest] = c[j] * (1 - ss * half) * (1 - ss * slip * mult)
    # funding: settlements on the opens of bars idx + 1 .. exit bar, charged at the close before each (the mark)
    f = np.where(np.isfinite(bars.funding), bars.funding, 0.0)
    F = np.zeros(bars.n)
    F[1:] = f[1:] * c[:-1]
    C = np.cumsum(F)
    fund = -sd * (C[ex_bar] - C[idx])
    fees = (entry + ex_px) * cost.fee
    risk = np.abs(entry - stop) + entry * (2 * cost.fee + cost.buffer)
    net = (ex_px - entry) * sd - fees + fund
    return {"exit_bar": ex_bar, "exit_px": ex_px, "code": code, "entry_px": entry, "funding": fund, "fees": fees,
            "risk": risk, "r": net / risk}


def take_sequential(idx: np.ndarray, exit_bar: np.ndarray) -> np.ndarray:
    """Positions of the entries one symbol actually takes: the first decision, then the first one after its exit bar,
    and so on (discovery.take_non_overlapping's pointer doubling)."""
    if not len(idx):
        return np.zeros(0, dtype=np.int64)
    from heartless.learning.discovery import take_non_overlapping

    rows = np.arange(len(idx))
    return take_non_overlapping(rows, exit_bar.astype(np.int64), idx.astype(np.int64), np.zeros(len(idx), np.int64),
                                presorted=True)


# --- the evaluation --------------------------------------------------------------------------------------------------
def _iso(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


@dataclass
class HourlyResult:
    alpha: str
    family: str
    config: dict
    start: int
    end: int
    trades: list[dict]
    symbols: list[str]
    skipped: list[str]
    daily_r: dict  # UTC day index -> mark-to-market R change of that day (every day of the window, zeros included)
    membership: dict | None = None
    stress: bool = False
    seconds: float = 0.0
    decisions: int = 0  # candidate decisions (before one-position-per-symbol)

    def summary(self) -> dict:
        from heartless.execution.stats import summarize
        from heartless.learning.lab import group_stats

        o = summarize(self.trades)
        g = group_stats(self.trades)
        days = max((self.end - self.start + 1) / MS_D, 1e-9)
        return {"alpha": self.alpha, "n": o["n"], "win_rate": round(o["win_rate"], 1), "avg_r": round(o["avg_r"], 4),
                "t_stat": round(o["t_stat"], 2), "t_day": g["t_day"], "profit_factor": g["profit_factor"],
                "net": round(o["net"], 1), "fees": round(o["fees"], 1), "funding": round(o["funding"], 1),
                "trades_per_day": round(o["n"] / days, 2), "avg_hold_h": round(o["avg_hold_min"] / 60, 2),
                "months_positive": g["months_positive"], "months": g["months"], "decisions": self.decisions,
                "delisting_closes": sum(1 for t in self.trades if t["exit_reason"] == "delisting"),
                "seconds": round(self.seconds, 1)}

    def _groups(self, key) -> dict:
        from heartless.learning.lab import group_stats

        g: dict[str, list[dict]] = defaultdict(list)
        for t in self.trades:
            g[key(t)].append(t)
        return {k: group_stats(v) for k, v in sorted(g.items())}

    def by_year(self) -> dict:
        return self._groups(lambda t: str(datetime.fromtimestamp(t["entry_time"] / 1000, tz=timezone.utc).year))

    def by_side(self) -> dict:
        return self._groups(lambda t: t["side"])

    def by_year_side(self) -> dict:
        return self._groups(lambda t: f"{datetime.fromtimestamp(t['entry_time'] / 1000, tz=timezone.utc).year} "
                                      f"{t['side']}")

    def by_symbol(self) -> dict:
        return self._groups(lambda t: t["symbol"])

    def exits(self) -> dict:
        g: dict[str, list[float]] = defaultdict(list)
        for t in self.trades:
            g[t["exit_reason"]].append(t["r_multiple"])
        return {k: {"n": len(v), "avg_r": round(float(np.mean(v)), 4)} for k, v in sorted(g.items())}

    def daily(self) -> dict:
        """Mark-to-market equity in R per UTC day (independent per-symbol accounts summed): days, total, mean, the
        daily t (information only; trades overlap days), max drawdown and the worst day."""
        days = sorted(self.daily_r)
        v = np.array([self.daily_r[d] for d in days], dtype=float)
        if not len(v):
            return {"days": 0}
        eq = np.cumsum(v)
        dd = float(np.max(np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:] - eq)) if len(eq) else 0.0
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return {"days": len(v), "total_r": round(float(eq[-1]), 4), "mean_r_per_day": round(float(v.mean()), 5),
                "t_daily": round(float(v.mean() / (sd / math.sqrt(len(v)))), 3) if sd > 0 else 0.0,
                "max_drawdown_r": round(dd, 4), "worst_day_r": round(float(v.min()), 4),
                "first": _iso(days[0] * MS_D)[:10], "last": _iso(days[-1] * MS_D)[:10]}

    def daily_series(self) -> list[list]:
        """[[UTC date, R change, cumulative R], ...]: the daily equity curve."""
        cum, out = 0.0, []
        for d in sorted(self.daily_r):
            cum += self.daily_r[d]
            out.append([_iso(d * MS_D)[:10], round(self.daily_r[d], 6), round(cum, 6)])
        return out

    def to_json(self) -> dict:
        return {"evaluator": "hourly", "alpha": self.alpha, "family": self.family, "config": self.config,
                "window": {"start": _iso(self.start), "end": _iso(self.end), "start_ms": self.start,
                           "end_ms": self.end},
                "stress": self.stress, "summary": self.summary(), "by_year": self.by_year(),
                "by_side": self.by_side(), "by_year_side": self.by_year_side(), "by_symbol": self.by_symbol(),
                "exits": self.exits(), "daily": self.daily(), "daily_equity": self.daily_series(),
                "symbols": self.symbols, "skipped": self.skipped, "membership": self.membership}

    def report(self) -> str:
        s = self.summary()
        lines = [f"[{self.alpha}] hourly {_iso(self.start)[:10]} -> {_iso(self.end)[:10]}{' STRESS' if self.stress else ''}"
                 f"  n={s['n']} win={s['win_rate']}% avgR={s['avg_r']:+.4f} PF={s['profit_factor']:.2f} "
                 f"t_day={s['t_day']:+.2f} trades/day={s['trades_per_day']} hold={s['avg_hold_h']}h "
                 f"symbols={len(self.symbols)} skipped={len(self.skipped)} ({s['seconds']:.1f}s)"]
        for tag, groups in (("year", self.by_year()), ("side", self.by_side())):
            for k, st in groups.items():
                lines.append(f"  {tag:5s} {k:10s} n={st['n']:6d} avgR={st['avg_r']:+.4f} PF={st['profit_factor']:5.2f} "
                             f"t_day={st['t_day']:+.2f}")
        for k, st in self.exits().items():
            lines.append(f"  exit  {k:10s} n={st['n']:6d} avgR={st['avg_r']:+.4f}")
        d = self.daily()
        if d.get("days"):
            lines.append(f"  daily {d['days']} days, total {d['total_r']:+.2f}R, max drawdown {d['max_drawdown_r']:.2f}R")
        if self.membership:
            lines.append(f"  members: {self.membership['member_months_total']} member-months, delisted "
                         f"{len(self.membership.get('delisted') or {})}")
        return "\n".join(lines)


def _daily_mtm(bars: HourlyBars, i: np.ndarray, x: np.ndarray, sim: dict, take: np.ndarray, side: np.ndarray,
               cost: Costs, exit_time: np.ndarray, out: dict) -> None:
    """Add the taken trades' daily mark-to-market R changes into `out` (UTC day index -> R): an open trade is marked at
    every day-end close it spans (mid price, entry fee and funding to date), the exit day takes the rest of its R."""
    if not len(take):
        return
    ct = bars.close_time
    ends = np.flatnonzero(ct % MS_D == 0)  # bars whose close is a UTC day end
    ii, xx = i[take], x[take]
    a = np.searchsorted(ends, ii, side="left")
    b = np.searchsorted(ends, xx, side="left")  # day ends at bars i .. x - 1
    cnt = np.maximum(b - a, 0)
    sd = side[take].astype(float)
    entry, risk = sim["entry_px"][take], sim["risk"][take]
    f = np.where(np.isfinite(bars.funding), bars.funding, 0.0)
    F = np.zeros(bars.n)
    F[1:] = f[1:] * bars.close[:-1]
    C = np.cumsum(F)
    rep = np.repeat(np.arange(len(take)), cnt)
    pos = (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)) + np.repeat(a, cnt)
    d = ends[pos] if len(pos) else np.zeros(0, dtype=np.int64)
    u = ((bars.close[d] - entry[rep]) * sd[rep] - entry[rep] * cost.fee - sd[rep] * (C[d] - C[ii[rep]])) / risk[rep]
    prev = np.zeros(len(u))
    if len(u) > 1:  # the previous day-end mark of the same trade (0 before its first)
        prev[1:] = np.where(rep[1:] == rep[:-1], u[:-1], 0.0)
    # a trade entered at the close T = start (decided just before the window, a member entry of the window) is marked
    # first at a day end before the window's first day: its entry costs belong to that first day, not to a slot that
    # would be dropped (or, as a negative index, wrap to the end of the array)
    days = np.maximum((ct[d] - 1) // MS_D - out["day0"], 0)
    np.add.at(out["acc"], days, u - prev)
    last_u = np.zeros(len(take))
    if len(u):
        lastpos = np.cumsum(cnt) - 1
        has = cnt > 0
        last_u[has] = u[lastpos[has]]
    np.add.at(out["acc"], np.maximum(exit_time // MS_D - out["day0"], 0), sim["r"][take] - last_u)


def evaluate(alpha: HourlyAlpha, start: int, end: int, universe_db, symbols: list[str] | None = None,
             top: int | None = None, stress: bool = False, exec: str = "taker",
             entry_filter: Callable[[str, np.ndarray], np.ndarray] | None = None, unseal=(),
             progress: Callable[[str], None] | None = None) -> HourlyResult:
    """Run `alpha` over [start, end] (ms, inclusive) on universe.db: on `symbols`, or on the topN research universe
    of the window (`top`, membership spans and delistings from heartless.data.universe). `entry_filter(symbol,
    entry_times) -> bool mask` blocks decisions (e.g. the shock layer of F-pre-B; an alpha with shock_exempt is the
    filter's business). Raises SealedError for a sealed window, ValueError for months the universe cannot rank."""
    from heartless.data import universe as U

    t0 = time.time()
    start, end = int(start), int(end)
    seal.check_window(start, end, unseal)
    cost = Costs.of("perp", exec, stress)
    store = ReadOnlyStore(universe_db)
    try:
        membership = None
        if top:
            res = U.research_universe(store, start, end, int(top))
            if res["unranked"]:
                raise ValueError(f"months {', '.join(res['unranked'])} cannot be ranked from {universe_db}")
            windows = U.membership_windows(res["members"], start, end)
            syms = sorted(windows)
        else:
            if not symbols:
                raise ValueError("give symbols or top")
            syms = sorted(dict.fromkeys(s.upper() for s in symbols))
            windows = {s: [(start, end)] for s in syms}
        # every delisting up to the window end, also one before its start: a member month can follow the last traded
        # hour (month m ranks on m-1's volume, so a symbol delisted late in m-1 can still rank, and a window may start
        # mid-month), and the flat zero-volume files the archive keeps publishing after a delisting must not trade
        delist = U.delisting_closes(store, syms, None, end)
        if top:
            months = U.member_months(windows)
            membership = {"rule": f"top{int(top)} (docs/UNIVERSE.md); entries only in member months",
                          "member_months": months, "member_months_total": sum(months.values()),
                          "delisted": {s: _iso(v) for s, v in sorted(delist.items())}}
        trades: list[dict] = []
        skipped: list[str] = []
        day0, day1 = start // MS_D, end // MS_D
        acc = {"acc": np.zeros(day1 - day0 + 2), "day0": day0}
        n_dec = 0
        for k, sym in enumerate(syms):
            spans = [w for w in windows.get(sym, []) if w[1] >= start and w[0] <= end]
            if not spans:
                continue
            lo = spans[0][0] - (int(alpha.warmup_hours) + 1) * MS_H
            hi = min(end, spans[-1][1] + int(alpha.max_hold_hours) * MS_H)
            if sym in delist:
                hi = min(hi, delist[sym])
            bars = load_bars(store, sym, lo, hi - MS_H + 1)  # bars that close by hi
            if bars is None:
                skipped.append(sym)
                continue
            ct = bars.close_time
            last = int(np.searchsorted(ct, min(end, hi) + 1, side="right") - 1)
            if last < 0:
                skipped.append(sym)
                continue
            delisted = sym in delist and ct[last] - 1 >= delist[sym]
            cand = np.flatnonzero(bars.stored & U.member_mask(ct, spans) & (np.arange(bars.n) < last)
                                  & (ct >= start) & (ct <= end))
            if not len(cand):
                continue
            dec = decide_bars(alpha, bars, cand)
            side = dec.side[cand].astype(np.int8)
            stop, target, hold = dec.stop[cand], dec.target[cand].astype(float), dec.max_hold[cand].astype(np.int64)
            if (hold[side != 0] > int(alpha.max_hold_hours)).any():
                raise ValueError(f"{alpha.name}: a decision holds longer than max_hold_hours={alpha.max_hold_hours} "
                                 f"(the bars after a membership span are loaded for that long)")
            # the orders carry tick-rounded levels (the engine's norm_price, nearest tick)
            stop, target = round_to_tick(stop, bars.tick()), round_to_tick(target, bars.tick())
            ref = bars.close[cand] * (1 + side * cost.half_spread)
            with np.errstate(invalid="ignore"):
                ok = (side != 0) & np.isfinite(stop) & ((stop - ref) * side < 0) & (hold >= 1)
                target = np.where(np.isfinite(target) & ((target - ref) * side > 0), target, np.nan)
            if entry_filter is not None and ok.any():
                # only the decisions are asked (a filter that counts what it blocks counts decisions, F-pre-B)
                live = np.flatnonzero(ok)
                ok[live] = np.asarray(entry_filter(sym, ct[cand][live]), dtype=bool)
            sel = np.flatnonzero(ok)
            n_dec += len(sel)
            if not len(sel):
                continue
            i = cand[sel]
            sim = simulate_exits(bars, i, side[sel], stop[sel], target[sel], hold[sel], last, delisted, cost)
            take = take_sequential(i, sim["exit_bar"])
            code = sim["code"][take]
            j = sim["exit_bar"][take]
            # exit instant: inside the bar (its open, the same UTC day) for stops / targets, its close otherwise
            ext = np.where(code <= 4, bars.t[j], ct[j] - 1)
            s_side, s_stop, s_tgt = side[sel], stop[sel], target[sel]
            _daily_mtm(bars, i, sim["exit_bar"], sim, take, s_side, cost, ext, acc)
            for q, p in enumerate(take.tolist()):
                r = float(sim["r"][p])
                scale = RISK_AMOUNT / float(sim["risk"][p])
                tg = float(s_tgt[p])
                trades.append({"symbol": sym, "side": "LONG" if s_side[p] > 0 else "SHORT", "alpha": alpha.name,
                               "entry_time": int(ct[i[p]]), "exit_time": int(ext[q]),
                               "entry_price": float(sim["entry_px"][p]), "exit_price": float(sim["exit_px"][p]),
                               "stop": float(s_stop[p]), "target": tg if math.isfinite(tg) else None,
                               "r_multiple": r, "pnl": r * RISK_AMOUNT, "risk_amount": RISK_AMOUNT,
                               "fees": float(sim["fees"][p]) * scale, "funding": float(sim["funding"][p]) * scale,
                               "exit_reason": REASONS[int(code[q])], "bars_held": int(j[q] - i[p])})
            if progress and (k + 1) % 50 == 0:
                progress(f"hourly {alpha.name}: {k + 1}/{len(syms)} symbols, {len(trades)} trades")
    finally:
        store.close()
    trades.sort(key=lambda t: (t["entry_time"], t["symbol"]))
    daily = {day0 + d: float(v) for d, v in enumerate(acc["acc"]) if day0 + d <= day1}
    cfg = {"tool": "hourly", **alpha.config(), "universe": f"top{int(top)}" if top else syms, "exec": exec}
    if entry_filter is not None and not getattr(entry_filter, "evaluator_policy", False):
        # a filter that is a choice of the configuration is part of its identity; the shock layer is not (it is the
        # evaluator's own policy from F-pre-B on, as the costs are: the trial records it in `extra` instead)
        cfg["entry_filter"] = getattr(entry_filter, "__name__", type(entry_filter).__name__)
    return HourlyResult(alpha.name, alpha.family, cfg, start, end, trades, [s for s in syms if s not in skipped],
                        skipped, daily, membership, bool(stress), time.time() - t0, n_dec)


def record(result: HourlyResult, family: str | None = None, extra: dict | None = None, ledger=None) -> dict:
    """Append the run to the trial ledger as tool "hourly" under `family` (default the alpha's), with the TRAIN
    statistics; the configuration excludes the window and the costs (a stress / VALID / FORWARD rerun keeps N)."""
    from heartless.learning import trials

    fam = (family or result.family or "").strip()
    s = result.summary()
    return trials.record("hourly", fam, result.config, (result.start, result.end), n=s["n"], avg_r=s["avg_r"],
                         t_day=s["t_day"], extra={"stress": result.stress, "evaluator": "hourly", **(extra or {})},
                         ledger=ledger)


# --- equivalence with the lab ----------------------------------------------------------------------------------------
def compare_trades(hourly: list[dict], lab_trades: list[dict]) -> dict:
    """Match two trade lists by (symbol, entry hour, side): shares, |R| differences of the matched trades, exit
    reason / exit hour agreement and a few unmatched examples."""
    def key(t):
        return (t["symbol"], int(t["entry_time"]) // MS_H, t["side"])

    a = {key(t): t for t in hourly}
    b = {key(t): t for t in lab_trades}
    both = sorted(set(a) & set(b))
    diff = np.array([abs(float(a[k]["r_multiple"]) - float(b[k]["r_multiple"] or 0.0)) for k in both])
    same_exit_h = sum(1 for k in both if int(a[k]["exit_time"]) // MS_H == int(b[k]["exit_time"] or 0) // MS_H)
    same_reason = sum(1 for k in both if a[k]["exit_reason"] == _lab_reason(b[k].get("exit_reason")))
    n = max(len(a), len(b), 1)
    return {"n_hourly": len(a), "n_lab": len(b), "matched": len(both), "matched_share": round(len(both) / n, 6),
            "mean_abs_r_diff": round(float(diff.mean()), 6) if len(diff) else None,
            "p99_abs_r_diff": round(float(np.quantile(diff, 0.99)), 6) if len(diff) else None,
            "max_abs_r_diff": round(float(diff.max()), 6) if len(diff) else None,
            "same_exit_hour_share": round(same_exit_h / max(len(both), 1), 6),
            "same_exit_reason_share": round(same_reason / max(len(both), 1), 6),
            "only_hourly": [list(k[:1]) + [_iso(k[1] * MS_H)] for k in sorted(set(a) - set(b))[:10]],
            "only_lab": [list(k[:1]) + [_iso(k[1] * MS_H)] for k in sorted(set(b) - set(a))[:10]]}


def _lab_reason(reason) -> str:
    """The engine's exit reason in this evaluator's words (stop / target / time / end / delisting)."""
    text = str(reason or "")
    if "time exit" in text:
        return "time"
    if text == "backtest end":
        return "end"
    if text == "delisting":
        return "delisting"
    if "SL" in text or "손절" in text:
        return "stop"
    if "TP" in text:
        return "target"
    return text


def lab_equivalence(candle_db, universe_db, symbols: list[str], start: int, end: int, workers: int = 2,
                    progress=None) -> dict:
    """The F-pre-A acceptance check: the reference alpha in this evaluator (universe.db 1h bars) and as an engine
    alpha in the lab (1m candles, PaperAccount fills) on the same symbols and window. An equivalence measurement, not a
    hypothesis: the reference predicts nothing, so it is not recorded in the trial ledger."""
    from heartless.learning import lab
    from heartless.strategy.params import StrategyParams

    t0 = time.time()
    h = evaluate(ReferenceAlpha(), start, end, universe_db, symbols=symbols, progress=progress)
    t1 = time.time()
    res = lab.evaluate(StrategyParams.default(), list(symbols), start, end, str(candle_db),
                       only_alpha=ReferenceAlpha.name, workers=workers,
                       extra_alphas=[f"{__name__}:ReferenceEngineAlpha"])
    t2 = time.time()
    cmp = compare_trades(h.trades, res.trades)
    return {"what": "evaluator equivalence on the deterministic non-signal reference alpha (long at the 01:00 UTC "
                    "close every day, stop 2 ATR(14) 1h, 12 h time exit); not a hypothesis, no ledger line",
            "window": [_iso(start), _iso(end)], "symbols": list(symbols), **cmp,
            "acceptance": {"matched_share_min": 0.99, "mean_abs_r_diff_max": 0.02,
                           "passed": bool(cmp["matched_share"] >= 0.99 and (cmp["mean_abs_r_diff"] or 0.0) <= 0.02)},
            "seconds": {"hourly": round(t1 - t0, 1), "lab": round(t2 - t1, 1)}, "lab_errors": res.errors}


# --- CLI ----------------------------------------------------------------------------------------------------------------
def load_alpha(spec: str, overrides: list[str] | None = None) -> HourlyAlpha:
    """An HourlyAlpha from "module:Class" with `key=value` parameter overrides (numbers parsed as float or int)."""
    import importlib

    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError(f"--alpha expects module:Class, got {spec!r}")
    kw = {}
    for item in overrides or []:
        k, _, v = item.partition("=")
        try:
            num = float(v)
            kw[k.strip()] = int(num) if num.is_integer() and "." not in v else num
        except ValueError:
            kw[k.strip()] = v
    return getattr(importlib.import_module(mod), cls)(**kw)


def resolve_window(args) -> tuple[int, int]:
    """--period (a Wave F partition name: TRAIN-F/A/B, VALID-F, FWD-k) or --start/--end (a date-only end is
    exclusive)."""
    from heartless.learning import lab, trials

    period = getattr(args, "period", None)
    if period:
        name = period.upper()
        if name in trials.WAVE_F:
            a, b = trials.WAVE_F[name]
            return a, b - 1
        if name.startswith("FWD-") and name[4:].isdigit():
            return trials.fwd_span(int(name[4:]))
        raise ValueError(f"unknown period {period!r} (TRAIN-F, TRAIN-A, TRAIN-B, VALID-F, FWD-k)")
    if not (args.start and args.end):
        raise ValueError("give --period or --start and --end")
    return lab.parse_when(args.start), lab.parse_when(args.end, end=True)


def main_cli(settings, args) -> None:
    """`heartless hourly`: evaluate an HourlyAlpha (--alpha module:Class) or run the lab equivalence (--equivalence)."""
    import os
    import sys

    from heartless.data import universe as U
    from heartless.learning import lab, trials

    try:
        start, end = resolve_window(args)
        top = U.parse_top(args.universe) if args.universe else None
        symbols = None
        if not top:
            if args.universe in ("core", "extended"):
                symbols = lab.universe_symbols(args.universe)
            elif args.symbols:
                symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if args.equivalence:
            doc = lab_equivalence(settings.db_path, settings.universe_db_path, symbols or lab.universe_symbols("extended"),
                                  start, end, workers=args.workers or 2,
                                  progress=lambda m: print(m, file=sys.stderr, flush=True))
            print(json.dumps({k: v for k, v in doc.items() if k not in ("only_hourly", "only_lab")}, indent=1))
            doc.update(argv=list(args.argv), git_commit=seal.git_commit(), git_dirty=lab.git_dirty(),
                       generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        else:
            if not args.alpha:
                raise ValueError("--alpha module:Class is required (or --equivalence)")
            alpha = load_alpha(args.alpha, args.set)
            # F-pre-B (ROADMAP 7.8): the shock layer's entry halt is the entry filter (an alpha with shock_exempt takes
            # none); S0 from the candle database's BTC bars when it has them, S1-S3 from universe.db
            shock_filter, shock_doc = None, None
            if getattr(settings, "shock_layer", False):
                from heartless.strategy import shock as SH

                sched = SH.schedule_for(settings, start, end)
                shock_filter = SH.EntryFilter.for_alpha(alpha, sched)
                shock_doc = {"exempt": shock_filter is None, "firings": sched.counts(start, end),
                             "missing": sched.meta.get("missing", [])}
            res = evaluate(alpha, start, end, settings.universe_db_path, symbols=symbols, top=top, stress=args.stress,
                           entry_filter=shock_filter, progress=lambda m: print(m, file=sys.stderr, flush=True))
            print(res.report())
            if shock_doc is not None:
                shock_doc["blocked_decisions"] = shock_filter.blocked if shock_filter is not None else 0
                print(f"shock layer: {shock_doc['blocked_decisions']} decision(s) blocked; firings {shock_doc['firings']}"
                      + (" (alpha exempt)" if shock_doc["exempt"] else ""), file=sys.stderr)
            doc = {**res.to_json(), "wave_f_period": trials.wave_f_period(start, end), "argv": list(args.argv),
                   "git_commit": seal.git_commit(), "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "shock": shock_doc}
            tag = (args.tag or "").strip()
            if tag:
                doc["trial"] = record(res, tag, extra={"argv": list(args.argv), "shock": shock_doc})
                n_fam = trials.family_tests(tag)
                print(f"trial ledger: {tag} config {doc['trial']['config_hash']} recorded; family N={n_fam}, TRAIN bar "
                      f"t_day >= {trials.threshold(n_fam):.2f}", file=sys.stderr)
            elif trials.wave_f_period(start, end).startswith("TRAIN"):
                print("note: not recorded in the trial ledger (no --tag FAMILY)", file=sys.stderr)
            if args.trades_out:
                with open(args.trades_out, "w") as f:
                    json.dump(res.trades, f)
    except (ValueError, seal.SealedError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    if args.out:
        out = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(lab._clean(doc), f, indent=1, default=str)
        print(f"wrote {args.out}", file=sys.stderr)

"""The shock layer (ROADMAP 7.8, docs/SHOCK.md): detect a market shock and pause new entries, one code everywhere.

Detection (fixed definitions, never tuned on performance; docs/results/wave_f_registration.json "shock_protocol")
* S0, every minute: BTC's close moved >= 3% against its close 5 bars earlier -> 15 minutes without new entries (the
  live rule that app._btc_shock_check applied; now also in every backtest).
* S1, every hour: |BTC 1h log return| >= 4 x the population sd of the 720 hourly returns before it.
* S2, every hour: BTC 24h realized volatility (sqrt of the sum of the last 24 squared 1h log returns) >= 2.5 x the
  median of the same statistic over the 720 hours before it.
* S3, every hour: the mean 24h correlation of 1h log returns with BTC over the 40 eligible USDT perpetuals trading at
  that hour with the highest trailing 24h quote volume (point in time from universe.db, the 2026-10-16 amendment: the
  research context's top_n members, context.pit_members; BTC is one of them) is >= 0.80 and >= its mean over the 720
  hours before + 0.25.

Responses
* entry halt: S0 -> 15 minutes, S1-S3 -> 2 hours from the firing's close; no new entry in any decision frame, resting
  entry limits are cancelled, open positions keep their stops, targets and time limits; `shock_exempt` alphas
  (liq_cascade) are not paused. The engine applies it (execution/engine.py), the hourly evaluator takes it as its
  entry_filter (learning/hourly.py).
* shock state: starts at a S1-S3 firing, ends after 24 h without one (state_at).
* kill switch: CUSUM alarm line h 8 -> 6 for 30 days after the latest S1-S3 firing (kill_switch_h; k and the 30-trade
  sum unchanged).
* research window: 120 -> 45 days from the start of a shock state until 45 days after its last firing
  (research_lookback_days; proposal rules and challenger checkpoints unchanged).

Time: every instant is a bar end on the minute grid. A 1m bar with close_time T (= open + 59,999 ms) ends at T + 1
(`instant`); "the hour close H" is a multiple of one hour. A firing at H halts the decisions at instants in
[H, H + span): the bar that closes the hour may not open a trade either (its close is known at H, as BTC's is).

Offline and online: `build_schedule` computes the firings over a window from stored data (BTC 1m for S0, universe.db
1h for S1-S3); `ShockDetector` computes them bar by bar for the live bot. Both call the same rule functions, which use
only exactly rounded arithmetic (math.fsum, math.log / sqrt on scalars, sorting), so they agree to the bit
(tests/test_fpreB_shock.py). The cross-section of S3 is one component (`_CrossSection`) fed hour by hour on both paths.
"""
from __future__ import annotations

import json
import logging
import math
import os
from bisect import bisect_right
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MS_MIN = 60_000
MS_H = 3_600_000
MS_D = 86_400_000
BTC = "BTCUSDT"
VERSION = 2  # bump when a definition below changes (a stored schedule of another version is rebuilt); 2: a schedule
# keeps the firings before its window whose halt reaches into it

# --- fixed definitions (ROADMAP 7.8) ---------------------------------------------------------------------------------
S0_MOVE = 0.03  # BTC move over S0_BARS 1m bars
S0_BARS = 5
S0_HALT_MS = 15 * MS_MIN
S1_K = 4.0
S2_MULT = 2.5
RV_HOURS = 24  # realized volatility over the last 24 hourly returns
S3_TOP = 40
S3_HOURS = 24  # correlation window (hourly returns)
S3_LEVEL = 0.80
S3_JUMP = 0.25
S3_MIN_MEMBERS = 20  # fewer correlations than this at an hour: no value (S3 cannot fire)
CARRY_HOURS = 24  # a close is carried forward at most this long through a gap (context.CARRY_HOURS)
KEEP_HOURS = 49  # bars kept per symbol: the history check reaches 46 hours back, carried closes 47 + the carry
WINDOW_HOURS = 720  # "the prior 30 days" of S1 / S2 / S3
S3_MIN_PRIOR = 360  # S3's prior mean needs at least half of its 720 hours defined (a data hole must not mute it)
HALT_MS = 2 * MS_H  # S1-S3 entry halt
STATE_MS = 24 * MS_H  # a shock state ends after 24 h without a firing
KILL_SWITCH_H = 6.0  # CUSUM alarm line while ...
KILL_SWITCH_MS = 30 * MS_D  # ... within 30 days of the latest firing
RESEARCH_DAYS = 45  # research window while ...
RESEARCH_MS = 45 * MS_D  # ... within 45 days of the latest firing
HOURLY_RULES = ("S1", "S2", "S3")
# a schedule keeps the firings before its window whose halt still reaches into it (an S1-S3 firing up to 2 h - 1 ms,
# an S0 firing up to 15 minutes - 1 ms before `start`, when its BTC 1m bars reach there), so a window's first decisions
# are halted as the online detector and any longer window halt them. S0 is judged from the 6th 1m bar of the data it
# is given; a schedule read from a candle database reads none before `start` (a backtest reads no candle before its own
# first one), so a caller whose first decision is at `start` builds it from S0_LEAD_MS earlier (heartless hourly)
S0_LEAD_MS = S0_HALT_MS + (S0_BARS + 1) * MS_MIN
# history before a window that the hourly rules need: 720 prior values of a statistic over 24 h of returns, + slack
WARMUP_MS = (WINDOW_HOURS + RV_HOURS + 2) * MS_H


def instant(t: int) -> int:
    """The bar end a timestamp stands for: a 1m close_time (open + 59,999) -> open + 60,000; anything else as is."""
    t = int(t)
    return t + 1 if (t + 1) % MS_MIN == 0 else t


# --- the rules (scalar, exactly rounded: offline and online produce the same bits) -----------------------------------
def s0_fires(close: float, close_back: float) -> bool:
    """S0: |close - close 5 bars back| / close 5 bars back >= 3% (the expression of the former live check)."""
    return close_back > 0 and abs(close - close_back) / close_back >= S0_MOVE


def log_return(close: float, prev: float) -> float:
    return math.log(close / prev) if close > 0 and prev > 0 else float("nan")


def _mean(xs) -> float:
    return math.fsum(xs) / len(xs)


def pop_sd(xs) -> float:
    m = _mean(xs)
    return math.sqrt(math.fsum((x - m) * (x - m) for x in xs) / len(xs))


def median(xs) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def realized_vol(rets) -> float:
    """sqrt of the sum of the squared returns (the last RV_HOURS hourly log returns)."""
    return math.sqrt(math.fsum(r * r for r in rets))


def s1_fires(ret: float, prior: list[float]) -> bool:
    """S1: |r| >= 4 x sd of the WINDOW_HOURS returns before it (all finite, else no verdict)."""
    if len(prior) < WINDOW_HOURS or not math.isfinite(ret) or not all(math.isfinite(x) for x in prior):
        return False
    return abs(ret) >= S1_K * pop_sd(prior)


def s2_fires(rv: float, prior: list[float]) -> bool:
    """S2: rv >= 2.5 x the median of the WINDOW_HOURS values before it (all finite, else no verdict)."""
    if len(prior) < WINDOW_HOURS or not math.isfinite(rv) or not all(math.isfinite(x) for x in prior):
        return False
    return rv >= S2_MULT * median(prior)


def s3_fires(cm: float, prior: list[float]) -> bool:
    """S3: cm >= 0.80 and >= the mean of the defined values among the WINDOW_HOURS before it + 0.25."""
    if not math.isfinite(cm) or cm < S3_LEVEL:
        return False
    fin = [x for x in prior if math.isfinite(x)]
    if len(fin) < S3_MIN_PRIOR:
        return False
    return cm >= _mean(fin) + S3_JUMP


def mean_corr(btc: list[float], rows: list[list[float]]) -> float:
    """Mean Pearson correlation of each row's returns with BTC's (rows with no variance are left out); NaN with fewer
    than S3_MIN_MEMBERS correlations. Sums are math.fsum, so the value does not depend on the order of anything."""
    n = len(btc)
    mb = math.fsum(btc) / n
    db = [x - mb for x in btc]
    sbb = math.fsum(d * d for d in db)
    if not sbb > 0:
        return float("nan")
    corrs = []
    for r in rows:
        mr = math.fsum(r) / n
        dr = [x - mr for x in r]
        srr = math.fsum(d * d for d in dr)
        if not srr > 0:
            continue
        corrs.append(math.fsum(a * b for a, b in zip(db, dr)) / math.sqrt(sbb * srr))
    if len(corrs) < S3_MIN_MEMBERS:
        return float("nan")
    return math.fsum(corrs) / len(corrs)


class _CrossSection:
    """S3's hourly cross-section, fed one hour at a time on both paths (offline from universe.db, online by the app).

    The members at hour close H are the research context's point-in-time top_n (context.pit_members, ROADMAP 6절
    2026-10-16, "컨텍스트 top_n 구성원 정의와 같음"): among the eligible perpetuals with a bar in the hour ending at H
    that traded (quote volume > 0) and have a close within CARRY_HOURS before the hour 23 hours earlier, the S3_TOP
    with the largest quote volume over the 24 hours ending at H (summed oldest first, missing hours 0), ties by name.
    BTC is a member like any other symbol. Each member's returns use its closes carried forward at most CARRY_HOURS
    (context._carried); a member without 24 defined returns is left out of the mean, not replaced."""

    def __init__(self) -> None:
        self.bars: dict[str, deque] = {}  # symbol -> (hour close, close, quote volume) of its last stored bars
        self.cur_h: int | None = None
        self.cur: list[str] = []  # the symbols pushed for hour cur_h

    def push(self, h: int, sym: str, close: float, qv: float) -> None:
        if h != self.cur_h:
            self.cur_h, self.cur = h, []
        self.cur.append(sym)
        dq = self.bars.get(sym)
        if dq is None:
            dq = self.bars[sym] = deque(maxlen=KEEP_HOURS)
        dq.append((int(h), float(close), float(qv) if qv == qv else 0.0))

    def members(self, h: int, eligible: Callable[[str], bool] | None = None) -> list[str]:
        """The hour's members, most liquid first."""
        ranked = []
        hi_hist = h - (S3_HOURS - 1) * MS_H  # a close 23 hours back, carried from at most CARRY_HOURS - 1 before it
        lo_hist = hi_hist - (CARRY_HOURS - 1) * MS_H
        for sym in (self.cur if self.cur_h == h else ()):
            dq = self.bars[sym]
            last = dq[-1]
            if last[0] != h or not last[2] > 0:
                continue
            if not any(lo_hist <= t <= hi_hist for t, _, _ in dq):  # no close to carry 23 hours back
                continue
            if eligible is not None and not eligible(sym):
                continue
            vol = 0.0
            for t, _, q in dq:  # the 24 hours ending at h, oldest first (a missing hour adds 0)
                if t > h - S3_HOURS * MS_H:
                    vol = vol + q
            ranked.append((-vol, sym))
        ranked.sort()
        return [sym for _, sym in ranked[:S3_TOP]]

    def returns(self, sym: str, h: int) -> list[float]:
        """The symbol's S3_HOURS hourly log returns ending at h from its closes carried forward <= CARRY_HOURS."""
        dq = self.bars[sym]
        closes = []
        for k in range(S3_HOURS, -1, -1):
            t = h - k * MS_H
            c = float("nan")
            for tb, cb, _ in reversed(dq):
                if tb <= t:
                    if t - tb < CARRY_HOURS * MS_H:
                        c = cb
                    break
            closes.append(c)
        return [log_return(closes[i], closes[i - 1]) for i in range(1, len(closes))]

    def value(self, h: int, btc_rets: list[float], eligible: Callable[[str], bool] | None = None) -> float:
        """The mean correlation with BTC's returns at hour close h over the members."""
        if len(btc_rets) < S3_HOURS or not all(math.isfinite(x) for x in btc_rets):
            return float("nan")
        rows = []
        for sym in self.members(h, eligible):
            r = self.returns(sym, h)
            if all(math.isfinite(x) for x in r):
                rows.append(r)
        return mean_corr(btc_rets[-S3_HOURS:], rows)

    def drop_stale(self, h: int) -> None:
        """Forget symbols without a bar for KEEP_HOURS (a delisted symbol must not hold memory for ever)."""
        for sym in [s for s, dq in self.bars.items() if dq[-1][0] < h - KEEP_HOURS * MS_H]:
            del self.bars[sym]


# --- events and the queries every path shares ------------------------------------------------------------------------
@dataclass
class ShockEvent:
    rule: str  # S0 | S1 | S2 | S3
    t: int  # instant of the firing (minute end for S0, hour close for S1-S3)
    value: float
    threshold: float

    def message(self) -> str:
        if self.rule == "S0":
            return f"BTC 5분 내 {self.value * 100:.1f}% 급변 → 15분간 신규 진입 중단"
        what = {"S1": "BTC 1시간 가격 충격", "S2": "BTC 24시간 변동성 급등", "S3": "상위 40종목 BTC 상관 급등"}[self.rule]
        return f"{what} ({self.rule}: {self.value:.3g} ≥ {self.threshold:.3g}) → 2시간 신규 진입 중단"


class ShockEvents:
    """Firings and the questions every consumer asks of them (engine, hourly evaluator, kill switch, research)."""

    def __init__(self, s0: Iterable[int] = (), hourly: Mapping[str, Iterable[int]] | None = None):
        self.s0: list[int] = sorted(int(t) for t in s0)
        self.hourly: dict[str, list[int]] = {r: sorted(int(t) for t in (hourly or {}).get(r, ())) for r in HOURLY_RULES}
        self._any: list[int] | None = None

    def _all_hourly(self) -> list[int]:
        if self._any is None:
            self._any = sorted({t for r in HOURLY_RULES for t in self.hourly[r]})
        return self._any

    def entry_halt(self, t: int) -> str:
        """Why no new entry may start at time t ('' = none): 'shock S1+S3' (2 h after an hourly firing) or 'shock S0'
        (15 minutes after a BTC 5-minute move)."""
        x = instant(t)
        hit = [r for r in HOURLY_RULES if _within(self.hourly[r], x, HALT_MS)]
        if hit:
            return "shock " + "+".join(hit)
        return "shock S0" if _within(self.s0, x, S0_HALT_MS) else ""

    def entry_halts(self, times) -> np.ndarray:
        """entry_halt(t) != '' for an array of times (vectorised; the hourly evaluator's entry filter)."""
        x = np.asarray(times, dtype=np.int64)
        x = np.where((x + 1) % MS_MIN == 0, x + 1, x)
        out = np.zeros(len(x), dtype=bool)
        for fired, span in [(self.hourly[r], HALT_MS) for r in HOURLY_RULES] + [(self.s0, S0_HALT_MS)]:
            if fired:
                f = np.asarray(fired, dtype=np.int64)
                k = np.searchsorted(f, x, side="right") - 1  # last firing at or before x
                ok = k >= 0
                out[ok] |= x[ok] - f[k[ok]] < span
        return out

    def last_firing(self, t: int) -> int | None:
        """The latest S1-S3 firing at or before t (None: none yet)."""
        a = self._all_hourly()
        i = bisect_right(a, instant(t))
        return a[i - 1] if i else None

    def state_at(self, t: int) -> tuple[int, int] | None:
        """(start, last firing) of the shock state t is in, None outside one. A state starts at a firing and ends
        STATE_MS after its last firing (firings closer than that extend it)."""
        a = self._all_hourly()
        x = instant(t)
        i = bisect_right(a, x)
        if not i or x - a[i - 1] >= STATE_MS:
            return None
        j = i - 1
        while j > 0 and a[j] - a[j - 1] < STATE_MS:
            j -= 1
        return a[j], a[i - 1]

    def kill_switch_h(self, t: int, base: float) -> float:
        """The kill switch's CUSUM alarm line at t: KILL_SWITCH_H for 30 days after the latest firing, else `base`
        (never raised above it)."""
        last = self.last_firing(t)
        return min(float(base), KILL_SWITCH_H) if last is not None and instant(t) - last < KILL_SWITCH_MS else float(base)

    def research_lookback_days(self, t: int, base: int) -> int:
        """The research cycle's window at t: RESEARCH_DAYS from the start of a shock state until 45 days after its
        last firing, else `base` (never lengthened)."""
        last = self.last_firing(t)
        return min(int(base), RESEARCH_DAYS) if last is not None and instant(t) - last < RESEARCH_MS else int(base)

    def counts(self, start: int | None = None, end: int | None = None) -> dict:
        """Firings per rule inside [start, end] and the hours spent in a shock state (reports)."""
        lo, hi = (start if start is not None else -(2**62)), (end if end is not None else 2**62)
        out = {"S0": sum(1 for t in self.s0 if lo <= t <= hi)}
        for r in HOURLY_RULES:
            out[r] = sum(1 for t in self.hourly[r] if lo <= t <= hi)
        a = [t for t in self._all_hourly() if lo <= t <= hi]
        out["hours_fired"] = len(a)
        state_hours = set()
        for f in a:
            for k in range(STATE_MS // MS_H):
                h = f + k * MS_H
                if h <= hi:
                    state_hours.add(h)
        out["shock_state_hours"] = len(state_hours)
        halt_hours = set()
        for f in a:
            halt_hours.update(f + k * MS_H for k in range(HALT_MS // MS_H) if f + k * MS_H <= hi)
        out["halt_hours"] = len(halt_hours)
        return out


def _within(fired: list[int], x: int, span: int) -> bool:
    i = bisect_right(fired, x)
    return bool(i) and x - fired[i - 1] < span


# --- online ----------------------------------------------------------------------------------------------------------
class ShockDetector(ShockEvents):
    """The live shock layer: fed BTC's closed 1m bars (observe_minute) and, at every hour close, BTC's hourly close and
    the cross-section of eligible perpetuals (observe_hour). The app's wiring (F6) is docs/SHOCK.md section 5: prime
    it with WARMUP_MS of stored hourly history before trading, observe each hour before the engines decide on the bar
    that closes it, and give every engine this object as `TradingEngine.shock`."""

    def __init__(self) -> None:
        super().__init__()
        self.m1: deque = deque(maxlen=S0_BARS + 1)
        self.h_last: int | None = None
        self.h_close: float | None = None
        self.rets: deque = deque(maxlen=WINDOW_HOURS + 1)
        self.rvs: deque = deque(maxlen=WINDOW_HOURS + 1)
        self.cms: deque = deque(maxlen=WINDOW_HOURS + 1)
        self.xs = _CrossSection()
        self.events: deque = deque(maxlen=200)

    def observe_minute(self, t: int, close: float) -> ShockEvent | None:
        """BTC's 1m bar that closed at t (close_time or bar end). Returns the S0 event when it fires."""
        x = instant(t)
        self.m1.append(float(close))
        if len(self.m1) <= S0_BARS or not s0_fires(self.m1[-1], self.m1[0]):
            return None
        back = self.m1[0]
        ev = ShockEvent("S0", x, abs(self.m1[-1] - back) / back, S0_MOVE)
        self.s0.append(x)
        self.events.append(ev)
        return ev

    def observe_hour(self, h: int, btc_close: float, xs: Mapping[str, tuple[float, float]] | None = None,
                     eligible: Callable[[str], bool] | None = None) -> list[ShockEvent]:
        """The hour that closed at h (a multiple of one hour, or that hour's last 1m close_time): BTC's close and the
        {symbol: (close, quote volume of the hour)} of every perpetual that traded in it (S3; None skips S3 for the
        hour). `eligible(symbol)`: the instrument rules (USDT perpetual, TRADING, not a stablecoin), as
        universe.eligible_research offline. Returns the hour's events."""
        h = instant(h)
        if self.h_last is not None and h <= self.h_last:
            return []
        if self.h_last is not None and h - self.h_last > MS_H:
            # missing hours: BTC's close is carried flat across them (the stored grid's rule), returns 0 there
            for _ in range((h - self.h_last) // MS_H - 1):
                self._step(0.0)
        r = log_return(float(btc_close), self.h_close) if self.h_close is not None else float("nan")
        self.h_last, self.h_close = h, float(btc_close)
        cm = float("nan")
        if xs is not None:
            for sym, (c, qv) in xs.items():
                self.xs.push(h, sym, c, qv)
            if h % MS_D == 0:
                self.xs.drop_stale(h)
        return self._step(r, h, cm, xs is not None, eligible)

    def _step(self, r: float, h: int | None = None, cm: float = float("nan"), with_xs: bool = False,
              eligible=None) -> list[ShockEvent]:
        self.rets.append(r)
        rets = list(self.rets)
        rv = realized_vol(rets[-RV_HOURS:]) if len(rets) >= RV_HOURS else float("nan")
        self.rvs.append(rv)
        if h is not None and with_xs:
            cm = self.xs.value(h, rets[-S3_HOURS:], eligible)
        self.cms.append(cm)
        if h is None:
            return []
        out = []
        prior = rets[:-1][-WINDOW_HOURS:]
        if s1_fires(r, prior):
            out.append(ShockEvent("S1", h, abs(r), S1_K * pop_sd(prior)))
        rvs = list(self.rvs)
        prv = rvs[:-1][-WINDOW_HOURS:]
        if s2_fires(rv, prv):
            out.append(ShockEvent("S2", h, rv, S2_MULT * median(prv)))
        cms = list(self.cms)
        pcm = cms[:-1][-WINDOW_HOURS:]
        if s3_fires(cm, pcm):
            out.append(ShockEvent("S3", h, cm, _mean([x for x in pcm if math.isfinite(x)]) + S3_JUMP))
        for ev in out:
            self.hourly[ev.rule].append(h)
            self.events.append(ev)
        if out:
            self._any = None
        return out

    def series(self) -> dict[str, list[float]]:
        """The last WINDOW_HOURS + 1 values of each hourly statistic (tests, /health)."""
        return {"ret": list(self.rets), "rv": list(self.rvs), "cm": list(self.cms)}

    def prime(self, universe_db: str | os.PathLike, until: int, since: int | None = None, progress=None) -> int:
        """Warm the detector at start-up from universe.db: BTC's hourly closes and the eligible cross-section of every
        hour closing in [since, until] fed through observe_hour (the members' own history from KEEP_HOURS before), as
        the offline schedule computes them, so the next live hour is judged exactly as build_schedule would
        (tests/test_fpreB_shock.py). `since` defaults to WARMUP_MS + 45 days before `until`: the firings of the last 45
        days are kept, so a shock while the bot was down still lowers the kill switch line and shortens the research
        window. Returns the hours observed (0 without the file or BTC's bars; the layer then judges once live hours
        fill its windows). S0 needs no priming (its halt lasts 15 minutes)."""
        from heartless.core.store import Store

        if not universe_db or not Path(universe_db).exists():
            return 0
        until = instant(until)
        since = until - (WARMUP_MS + RESEARCH_MS) if since is None else instant(since)
        st = Store(universe_db, readonly=True)
        n = 0
        try:
            grid = btc_hourly_from_universe(st, since, until)
            if grid is None:
                return 0
            hours, closes = grid
            h0 = int(hours[0])
            close_at = dict(zip(hours.tolist(), closes.tolist()))
            for h, bars in stored_cross_sections(st, h0 - KEEP_HOURS * MS_H, int(hours[-1]), progress):
                if h < h0:  # history of the members only
                    for sym, (c, q) in bars.items():
                        self.xs.push(h, sym, c, q)
                    if h % MS_D == 0:
                        self.xs.drop_stale(h)
                    continue
                self.observe_hour(h, close_at[h], bars)
                n += 1
        finally:
            st.close()
        return n


# --- offline ---------------------------------------------------------------------------------------------------------
class ShockSchedule(ShockEvents):
    """The firings over a window [start, end] (instants) computed from stored data, with what it was built from."""

    def __init__(self, start: int, end: int, s0: Iterable[int] = (), hourly: Mapping[str, Iterable[int]] | None = None,
                 meta: dict | None = None):
        super().__init__(s0, hourly)
        self.start, self.end = int(start), int(end)
        self.meta = dict(meta or {})

    def covers(self, start: int, end: int) -> bool:
        return self.start <= int(start) and int(end) <= self.end

    def counts(self, start: int | None = None, end: int | None = None) -> dict:
        """ShockEvents.counts over the schedule's own window by default (the firings it keeps before `start` only serve
        the halts reaching into it)."""
        return super().counts(self.start if start is None else start, self.end if end is None else end)

    def to_json(self) -> dict:
        return {"version": VERSION, "start": self.start, "end": self.end, "s0": self.s0,
                "hourly": self.hourly, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "ShockSchedule":
        return cls(d["start"], d["end"], d.get("s0", ()), d.get("hourly"), d.get("meta"))


def s0_firings(close_time, close) -> list[int]:
    """S0 over a symbol's 1m bars (ascending): the bar ends where s0_fires holds against the close 5 bars earlier."""
    ct = np.asarray(close_time, dtype=np.int64)
    c = np.asarray(close, dtype=float)
    if len(c) <= S0_BARS:
        return []
    now, back = c[S0_BARS:], c[:-S0_BARS]
    with np.errstate(divide="ignore", invalid="ignore"):
        # the same IEEE operations as s0_fires, elementwise
        hit = (back > 0) & (np.abs(now - back) / back >= S0_MOVE)
    return [instant(int(t)) for t in ct[S0_BARS:][hit]]


def hourly_firings(hours: np.ndarray, closes: np.ndarray, cms: np.ndarray | None = None,
                   start: int | None = None) -> dict[str, list[int]]:
    """S1-S3 over BTC's hourly closes on a complete hour grid (`hours` = hour closes, ascending, one hour apart; holes
    carried flat) and the S3 statistic per hour (`cms`, NaN where unknown; None: S3 not evaluated). Firings before
    `start` are dropped (they only serve as history)."""
    n = len(hours)
    rets = [float("nan")] + [log_return(float(closes[k]), float(closes[k - 1])) for k in range(1, n)]
    rvs = [realized_vol(rets[k - RV_HOURS + 1:k + 1]) if k >= RV_HOURS else float("nan") for k in range(n)]
    cml = [float(x) for x in cms] if cms is not None else None
    out: dict[str, list[int]] = {r: [] for r in HOURLY_RULES}
    for k in range(n):
        h = int(hours[k])
        if start is not None and h < start:
            continue
        lo = max(0, k - WINDOW_HOURS)
        if s1_fires(rets[k], rets[lo:k]):
            out["S1"].append(h)
        if s2_fires(rvs[k], rvs[lo:k]):
            out["S2"].append(h)
        if cml is not None and s3_fires(cml[k], cml[lo:k]):
            out["S3"].append(h)
    return out


def cross_section_series(store, hours: np.ndarray, btc_closes: np.ndarray, progress=None) -> np.ndarray:
    """S3's statistic at every hour close of `hours` (a complete grid) from universe.db: every eligible perpetual's
    stored bars (universe.eligible_research: USDT perpetual, not a stablecoin) fed hour by hour to the _CrossSection the
    live detector uses, which picks the context's point-in-time members."""
    out = np.full(len(hours), np.nan)
    if not len(hours):
        return out
    h0, h1 = int(hours[0]), int(hours[-1])
    btc_rets = [float("nan")] + [log_return(float(btc_closes[k]), float(btc_closes[k - 1])) for k in range(1, len(hours))]
    xs = _CrossSection()
    pos = {int(h): k for k, h in enumerate(hours)}
    # the members' own history before the first hour: bars closing from KEEP_HOURS before it
    for h, bars in stored_cross_sections(store, h0 - KEEP_HOURS * MS_H, h1, progress):
        for sym, (c, q) in bars.items():  # the order of pushes does not matter
            xs.push(h, sym, c, q)
        k = pos.get(h)
        if k is not None:
            out[k] = xs.value(h, btc_rets[max(0, k - S3_HOURS + 1):k + 1])
        if h % MS_D == 0:
            xs.drop_stale(h)
    return out


def stored_cross_sections(store, h_from: int, h_to: int, progress=None):
    """(hour close, {symbol: (close, quote volume)}) for every hour close in [h_from, h_to] (an hour without bars gives
    {}), from universe.db's 1h bars of the eligible perpetuals (universe.eligible_research), a month at a time."""
    from heartless.data import universe as U
    from heartless.data.symbols_inventory import FUT_TABLE, decode_month

    inv = {r["symbol"]: r for r in store.load_symbol_inventory()}
    ok_inst = {s for s, r in inv.items() if U.eligible_research(r) is None}
    h_from, h_to = int(h_from), int(h_to)
    months = []
    m = _month_start(h_from - MS_H)  # a bar opening at t closes at t + 1 h
    while m <= h_to:
        months.append(m)
        m = _next_month(m)
    for mi, m in enumerate(months):
        per_hour: dict[int, dict[str, tuple[float, float]]] = {}
        for b in store.kline_block_index(FUT_TABLE, month=m):
            sym = b["symbol"]
            if sym not in ok_inst or not b["n"]:
                continue
            blk = store.load_kline_blocks(FUT_TABLE, sym, m, m)
            if not blk:
                continue
            d = decode_month(blk[0][1])
            ot, cl, qv = d["open_time"], d["close"], d["quote_volume"]
            for k in range(len(ot)):
                hc = int(ot[k]) + MS_H
                if h_from <= hc <= h_to:
                    per_hour.setdefault(hc, {})[sym] = (float(cl[k]), float(qv[k]))
        h = max(_month_start(m) + MS_H, h_from)
        end_m = min(_next_month(m), h_to)
        while h <= end_m:
            yield h, per_hour.get(h) or {}
            h += MS_H
        if progress:
            progress(f"shock S3: {mi + 1}/{len(months)} months")


def _month_start(t: int) -> int:
    from datetime import datetime, timezone

    d = datetime.fromtimestamp(t / 1000, tz=timezone.utc)
    return int(datetime(d.year, d.month, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _next_month(m: int) -> int:
    from datetime import datetime, timezone

    d = datetime.fromtimestamp(m / 1000, tz=timezone.utc)
    y, mo = (d.year + 1, 1) if d.month == 12 else (d.year, d.month + 1)
    return int(datetime(y, mo, 1, tzinfo=timezone.utc).timestamp() * 1000)


def btc_hourly_from_universe(store, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray] | None:
    """BTC's hourly closes from universe.db on the complete hour grid of the bars closing in [lo, hi] (holes carried
    flat, as hourly.load_bars does): (hour closes, closes), None without data."""
    from heartless.data.symbols_inventory import FUT_TABLE, load_hourly

    raw = load_hourly(store, BTC, lo - MS_H, hi - MS_H, FUT_TABLE)
    ot = raw["open_time"]
    if not len(ot):
        return None
    return _grid(ot + MS_H, raw["close"])


def btc_hourly_from_minutes(close_time, close) -> tuple[np.ndarray, np.ndarray] | None:
    """BTC's hourly closes from its 1m bars: the close of the bar ending each hour (complete grid, carried flat)."""
    ct = np.asarray(close_time, dtype=np.int64)
    sel = (ct + 1) % MS_H == 0
    if not sel.any():
        return None
    return _grid(ct[sel] + 1, np.asarray(close, dtype=float)[sel])


def _grid(hc: np.ndarray, closes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hours = np.arange(int(hc[0]), int(hc[-1]) + MS_H, MS_H, dtype=np.int64)
    pos = ((hc - hc[0]) // MS_H).astype(np.int64)
    have = np.zeros(len(hours), dtype=bool)
    have[pos] = True
    vals = np.full(len(hours), np.nan)
    vals[pos] = closes
    src = np.maximum.accumulate(np.where(have, np.arange(len(hours)), 0))
    return hours, vals[src]


def build_schedule(start: int, end: int, btc_1m: tuple[np.ndarray, np.ndarray] | None = None,
                   candle_db: str | os.PathLike | None = None, universe_db: str | os.PathLike | None = None,
                   progress=None) -> ShockSchedule:
    """The shock schedule over [start, end] (ms; instants) from stored data, reading nothing after `end`.

    S0: BTC's 1m bars (`btc_1m` = (close_time, close) arrays, or read from `candle_db` inside [start, end] only: a
    backtest reads no candle before its own first one). S1-S3: universe.db's 1h bars (BTC, and the cross-section for
    S3) from WARMUP_MS before `start`; without universe.db S1 / S2 use BTC's 1m bars (the close ending each hour, the
    same value as the 1h kline's; they judge once those bars hold WARMUP_MS of history) and S3 is not evaluated. The
    schedule holds the firings inside the window and those just before it whose halt reaches into it: a window starting
    inside an S1-S3 halt is halted there, as the online detector and every longer window are; for S0 that needs BTC
    bars before `start` (`btc_1m`, or a schedule built from S0_LEAD_MS earlier). meta records the sources and the rules
    that could not be evaluated ("missing")."""
    start, end = int(start), int(end)
    e_inst = instant(end)  # a 1m close_time stands for its bar end: the hour that bar closes is known at `end`
    meta: dict = {"version": VERSION, "missing": []}
    lo = start - WARMUP_MS
    if btc_1m is None and candle_db and Path(candle_db).exists():
        btc_1m = _load_btc_1m(candle_db, start, end)
        meta["candle_db"] = str(candle_db)
    s0: list[int] = []
    if btc_1m is not None and len(btc_1m[0]):
        ct, c = btc_1m
        keep = np.asarray(ct, dtype=np.int64) <= end
        s0 = [t for t in s0_firings(np.asarray(ct)[keep], np.asarray(c)[keep]) if start - S0_HALT_MS < t <= e_inst]
    else:
        meta["missing"].append("S0")
    hourly: dict[str, list[int]] = {r: [] for r in HOURLY_RULES}
    grid = None
    ustore = None
    if universe_db and Path(universe_db).exists():
        from heartless.core.store import Store

        ustore = Store(universe_db, readonly=True)
        meta["universe_db"] = str(universe_db)
    try:
        if ustore is not None:
            grid = btc_hourly_from_universe(ustore, lo, e_inst)  # bars closing by e_inst: data up to `end` only
            meta["hourly_source"] = "universe.db"
        if grid is None and btc_1m is not None and len(btc_1m[0]):
            ct = np.asarray(btc_1m[0], dtype=np.int64)
            keep = ct <= end
            grid = btc_hourly_from_minutes(ct[keep], np.asarray(btc_1m[1])[keep])
            meta["hourly_source"] = "1m"
        if grid is None:
            meta["missing"] += list(HOURLY_RULES)
        else:
            hours, closes = grid
            cms = None
            if ustore is not None:
                cms = cross_section_series(ustore, hours, closes, progress)
            else:
                meta["missing"].append("S3")
            hourly = hourly_firings(hours, closes, cms, start - HALT_MS + 1)  # a halt reaching into the window
            hourly = {r: [h for h in v if h <= e_inst] for r, v in hourly.items()}
            meta["hours"] = [int(hours[0]), int(hours[-1])]
    finally:
        if ustore is not None:
            ustore.close()
    return ShockSchedule(start, end, s0, hourly, meta)


def _load_btc_1m(candle_db, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray] | None:
    from heartless.core.store import Store

    st = Store(candle_db, readonly=True)
    try:
        cts, cls = [], []
        a = lo
        while a <= hi:  # a month at a time: a year of row objects at once is ~0.5 GB
            b = min(hi, a + 30 * MS_D - 1)
            rows = st.load_candles(BTC, start=a, end=b)
            cts.extend(r.close_time for r in rows)
            cls.extend(r.close for r in rows)
            a = b + 1
    finally:
        st.close()
    if not cts:
        return None
    return np.asarray(cts, dtype=np.int64), np.asarray(cls, dtype=float)


# --- the schedule a backtest uses ------------------------------------------------------------------------------------
_CACHE: dict[tuple, ShockSchedule] = {}
_TOLD: set[tuple] = set()


def save_schedule(sched: ShockSchedule, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(sched.to_json()))
    os.replace(tmp, p)  # atomic: concurrent lab workers never read half a file


def load_schedule(path: str | os.PathLike) -> ShockSchedule | None:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if d.get("version") != VERSION:
        return None
    return ShockSchedule.from_json(d)


def schedule_for(settings, start: int, end: int, btc_1m: tuple[np.ndarray, np.ndarray] | None = None,
                 progress=None) -> ShockSchedule:
    """The schedule covering [start, end] for a run with `settings`: SHOCK_SCHEDULE when it covers the window (else
    built and written there), otherwise built from the run's own BTC bars (`btc_1m`) or SHOCK_CANDLE_DB, and
    UNIVERSE_DB, cached per process (the jobs of one lab run load the same span and share one build). Nothing after
    `end` is read (a sealed period after the window never is)."""
    start, end = int(start), int(end)
    path = getattr(settings, "shock_schedule", "") or ""
    if path:
        got = load_schedule(path)
        if got is not None and got.covers(start, end):
            return got
    udb = str(settings.universe_db_path)
    cdb = None if btc_1m is not None else str(settings.shock_candle_db_path)
    key = (udb, cdb, start, end, None if btc_1m is None else (len(btc_1m[0]), int(btc_1m[0][0]) if len(btc_1m[0]) else 0,
                                                             int(btc_1m[0][-1]) if len(btc_1m[0]) else 0))
    got = _CACHE.get(key)
    if got is None:
        got = build_schedule(start, end, btc_1m=btc_1m, candle_db=cdb, universe_db=udb, progress=progress)
        if len(_CACHE) > 8:
            _CACHE.clear()
        _CACHE[key] = got
        if path:
            try:
                save_schedule(got, path)
            except OSError as e:  # a read-only location: the run still has its schedule
                log.warning("cannot write the shock schedule to %s: %s", path, e)
    missing = tuple(got.meta.get("missing") or ())
    if missing and missing not in _TOLD:
        # once per process: a synthetic or live-only data set without universe.db is common (meta keeps the record)
        _TOLD.add(missing)
        log.info("shock layer: %s not evaluated (%s)", ", ".join(missing),
                 "no universe.db" if "S3" in missing else "no BTC data")
    return got


# --- consumers -------------------------------------------------------------------------------------------------------
def is_exempt(alpha_name: str) -> bool:
    """Whether an engine alpha is exempt from the entry halt (Alpha.shock_exempt; unknown names are not)."""
    from heartless.strategy.alphas import ALPHA_BY_NAME

    a = ALPHA_BY_NAME.get(alpha_name)
    return bool(getattr(a, "shock_exempt", False)) if a is not None else False


class EntryFilter:
    """The hourly evaluator's entry filter (learning/hourly.evaluate(entry_filter=)): decisions at halted times are
    blocked; `blocked` counts them. An exempt alpha takes no filter at all (for_alpha).

    `evaluator_policy`: the layer belongs to every evaluation from F-pre-B on (as the costs do), not to a configuration,
    so the evaluators leave it out of the trial config hash and record it in the trial's `extra`: a family's TRAIN,
    stress, VALID and FORWARD runs of one configuration keep one hash (the lab, whose hash never held it, likewise)."""
    __name__ = "shock_layer"
    evaluator_policy = True

    def __init__(self, events: ShockEvents):
        self.events = events
        self.blocked = 0

    def __call__(self, symbol: str, times):
        """A bool mask for an array of decision times (hourly evaluator), a bool for one time (neutral evaluator)."""
        if np.ndim(times) == 0:
            ok = not self.events.entry_halt(int(times))
            self.blocked += 0 if ok else 1
            return ok
        halted = self.events.entry_halts(times)
        self.blocked += int(halted.sum())
        return ~halted

    @classmethod
    def for_alpha(cls, alpha, events: ShockEvents | None) -> "EntryFilter | None":
        if events is None or getattr(alpha, "shock_exempt", False):
            return None
        return cls(events)

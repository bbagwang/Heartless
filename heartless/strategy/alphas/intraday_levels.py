"""Intraday structure / seasonality: on volatile days the UTC-day trend that is already in place before the US session
continues through it (intraday momentum). At the close of the 1h bar ending at `entry_hour` UTC (13:00, just before
the New York open), if price has moved at least `k_adr` average daily ranges away from the 00:00 UTC open and the 1h
ATR is at least `min_atr_bp` of price, join that direction with a post-only entry at the close, a 1h-ATR stop, a
partial at tp1_r, a final target at tp_r and a time stop after `hold_h` hours (around the US close).

All levels come from causal 1h bars: the daily open is the open of the 1h bar that started at 00:00 UTC, the average
daily range (ADR) is the mean high-low of the previous ADR_DAYS complete UTC days.

Research notes (TRAIN 2026-01-13..07-01, 12 symbols; numpy event studies on 1m paths, then the lab):
* Rejected (gross |edge| <= ~0.1R, net <= 0 on TRAIN): prior-UTC-day high/low reactions (first touch, close back
  inside, post-only limit AT the level, breakouts), daily-open retests, session-VWAP +-2/2.5/3 sigma fades, day-range
  extreme fades, 24h-extension fades, opening-range breakouts and fades (UTC, London, New York opens), and returns
  around the 00/08/16 UTC funding settlements (conditional on the settled rate).
* Hour-of-day seasonality: 10:00-11:00 UTC is negative in every TRAIN month and on all 12 symbols (-12bp, day-clustered
  t -3.3), but 12bp is far too small for a ~9-13bp round trip; not traded.
* Kept: the 00:00->13:00 UTC move predicts the 13:00->21:00 move on both sides. Unfiltered (lab, n=757) it made
  +0.11R but only in volatile months (Jan/Feb/Jun +, Mar-May -); it reverses in quiet markets (the same move faded from
  16:00 UTC is the mirror image). Requiring a volatile market (1h ATR >= min_atr_bp) is what makes it consistent:
  min_atr_bp 80/100/120/140 -> TRAIN avgR +0.15/+0.27/+0.39/+0.58 (n 509/329/195/106, monotonic).
* Neighbours (at min_atr_bp=100): k_adr 0.2/0.3/0.4 +0.16/+0.27/+0.29, sl_atr 1.2/1.8 +0.27/+0.22, hold_h 6/10
  +0.28/+0.31, tp_r 4 +0.29, no partial +0.30; entry_hour is the sensitive one (12: +0.19, 14: +0.10, both with only
  3/6 months positive). Market entry instead of post-only: +0.09 vs +0.11 (unfiltered).
* Defaults (min_atr_bp=120, chosen over 100 on VALID): TRAIN n=195 avgR +0.39 t 5.2 PF 2.33 months+ 5/6 symbols+ 12/12,
  --stress +0.37; VALID 2026-07..09 n=41 avgR +0.37 PF 2.85 (min_atr_bp=100: n=62 avgR +0.23 PF 1.92).
* Caveats: the day-clustered t is only ~2.4 (same-day trades on different coins are correlated), most of the profit
  comes from high-volatility weeks, and the alpha is silent in quiet markets (April/May 2026: ~4-13 trades a month).
"""
from __future__ import annotations

import numpy as np

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1h"
MS_H = 3_600_000
MS_D = 86_400_000
ADR_DAYS = 10  # complete UTC days averaged for the daily-range unit (fits the 12-day lab warm-up and the live window)
MIN_ADR_DAYS = 5


def day_levels(open_time: np.ndarray, high: np.ndarray, low: np.ndarray, opn: np.ndarray, close_time: int):
    """Causal UTC-day structure from 1h bars that closed at or before `close_time` (arrays end at the current bar).

    Returns (day_open, day_high, day_low, adr) or None. day_open is the open of the 1h bar that started at 00:00 UTC
    of the current day; adr is the mean high-low range of up to ADR_DAYS previous complete UTC days."""
    n = len(open_time)
    if n == 0:
        return None
    day0 = (int(close_time) + 1) // MS_D * MS_D
    if int(close_time) + 1 == day0:  # the bar closing exactly at midnight belongs to the previous day
        day0 -= MS_D
    j0 = int(np.searchsorted(open_time, day0, side="left"))
    if j0 >= n or int(open_time[j0]) != day0:
        return None  # no bar at 00:00 UTC (data gap): no reliable daily open
    d_open = float(opn[j0])
    d_hi, d_lo = float(np.max(high[j0:])), float(np.min(low[j0:]))
    ranges = []
    for k in range(1, ADR_DAYS + 1):
        a = int(np.searchsorted(open_time, day0 - k * MS_D, side="left"))
        b = int(np.searchsorted(open_time, day0 - (k - 1) * MS_D, side="left"))
        if b - a >= 20:  # (nearly) complete day
            ranges.append(float(np.max(high[a:b]) - np.min(low[a:b])))
    if len(ranges) < MIN_ADR_DAYS:
        return None
    return d_open, d_hi, d_lo, float(np.mean(ranges))


class IntradayLevels(Alpha):
    """Morning-trend continuation into the US session on volatile days (intraday momentum vs the UTC daily open)."""

    name = "intraday_levels"
    timeframe = TF
    enabled_by_default = True  # meets the lab acceptance criteria on TRAIN, --stress and VALID (see module docstring)
    description = ("UTC-day intraday momentum: at 13:00 UTC a move of >= k_adr x ADR from the 00:00 UTC open, in a "
                   "volatile market, is joined into the US session; post-only entry, 1h-ATR stop, partial + target, "
                   "8h time stop")
    param_specs = [
        ParamSpec("entry_hour", 13, 8, 16, 1, integer=True),  # UTC hour of the decision (close of that 1h bar)
        ParamSpec("k_adr", 0.3, 0.1, 1.0, 0.05),  # min |close - daily open| in average daily ranges
        ParamSpec("sl_atr", 1.5, 0.8, 3.0, 0.1),  # stop distance (1h ATR)
        ParamSpec("tp_r", 3.0, 1.0, 6.0, 0.25),  # final target (R)
        ParamSpec("tp1_r", 1.5, 0.5, 3.0, 0.25),  # partial target (R), size from the shared tp1_frac
        ParamSpec("hold_h", 8, 2, 16, 1, integer=True),  # time stop (hours)
        ParamSpec("min_atr_bp", 120, 0, 200, 10, integer=True),  # min 1h ATR in bp of price (volatile market)
    ]
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        h1 = view.tf(TF)
        if not h1.ok:
            return None
        ct = h1.close_time
        hour = ((ct + 1) // MS_H) % 24
        if hour != int(p["entry_hour"]) or hour == 0:
            return None
        close, a = h1.v("close"), h1.v("atr")
        if not ok(close, a) or close <= 0 or a <= 0:
            return None
        if a / close * 1e4 < p["min_atr_bp"]:
            return None  # quiet market: intraday moves are noise that reverts, and costs eat a large share of R
        nbars = (ADR_DAYS + 2) * 24
        open_time = h1.frame.open_time[max(0, h1.idx - nbars + 1): h1.idx + 1]  # causal: ends at the current bar
        lv = day_levels(open_time, h1.arr("high", nbars), h1.arr("low", nbars), h1.arr("open", nbars), ct)
        if lv is None:
            return None
        d_open, d_hi, d_lo, adr = lv
        if not ok(d_open, d_hi, d_lo, adr) or adr <= 0 or d_hi <= d_lo:
            return None
        move = close - d_open
        move_adr = abs(move) / adr
        if move_adr < p["k_adr"]:
            return None
        side = Side.LONG if move > 0 else Side.SHORT
        sg = side.sign
        dpos = (close - d_lo) / (d_hi - d_lo) if side is Side.LONG else (d_hi - close) / (d_hi - d_lo)
        entry = close
        dist = p["sl_atr"] * a
        stop = entry - sg * dist
        tp = entry + sg * p["tp_r"] * dist
        tp1 = entry + sg * p["tp1_r"] * dist if (p["tp1_frac"] > 0 and p["tp1_r"] < p["tp_r"]) else None
        conf = 0.6
        if move_adr >= 2 * p["k_adr"]:
            conf += 0.05
        if dpos >= 0.8:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        pct = move / d_open * 100
        reason = (f"UTC 00시 시가 대비 {pct:+.1f}% ({move_adr:.1f}×일평균범위) 추세가 {hour}시(UTC)까지 유지, "
                  f"고변동성(ATR {a / close * 100:.2f}%) → 미국장 장중 모멘텀 {'롱' if side is Side.LONG else '숏'}, "
                  f"손절 {p['sl_atr']:.1f}×ATR(1h), 최대 {int(p['hold_h'])}시간 보유")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1, entry_style=EntryStyle.LIMIT, limit_price=entry,
                      max_hold_bars=self.bars_to_1m(int(p["hold_h"]), TF), trail_atr_mult=0.0, atr=a, timeframe=TF,
                      tags={"ref_price": entry, "day_open": d_open, "adr": adr, "move_adr": move_adr, "dpos": dpos})

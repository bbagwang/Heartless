"""Daily volatility compression released by a break of the 20-day box (vol_breakout family, ROADMAP P2), entered at
the first 4h close beyond the box in the direction of BTC's week and held for about three days behind a wide
daily-ATR stop.

Setup, at every 4h close (only closed bars):
* compression: on at least one of the comp_days daily bars before the 4h bar's UTC day, the Bollinger(20, 2) width
  was in the lowest comp_q of its last RANK_WINDOW daily values (a coin-relative "quiet" regime; comp_q = 1 turns
  the filter off);
* breakout: the 4h close is the first one beyond the box_d-day high (low) of the closed days before its UTC day;
* direction: BTC's 168h return (ctx.market btc_ret_168h) agrees with the side (btc_filter), optionally the daily
  SMA20 vs SMA50 as well (trend_filter); an entry paying more than fund_max of funding is skipped (the crowded side,
  and exactly the trades the engine's funding-avoidance exit would cut at the next settlement unless 0.3R up).
Market entry, stop sl_atr daily ATR14s (last closed day) away, no target. After max_hold_h hours the engine closes a
trade that is not 0.5R up and lets a winner run under a breakeven stop and a 1.5 daily-ATR trail (Signal.timeframe
"1d") for at most 3x max_hold_h; trail_atr > 0 adds the engine's breakeven-at-1R and chandelier trail before that.

Why this shape (priors before the P2 data): low volatility mean-reverts upwards, and a quiet range collects stops
and leveraged positions at its edges, so the first escape from a multi-week box tends to run for days (liquidation
and stop cascades in perpetual futures); BTC's direction is the dominant common factor of the coins. Stops in
daily ATRs (~10% of price) make the ~13bp round trip ~1% of R. The 1h squeeze_breakout (P1: +0.018R) lost its
edge to costs and noise at a 1-day horizon; this design trades the daily structure instead.
TRAIN screening 2022-2024 (vol_breakout trial ledger, an hourly simulator of the engine's exits): daily-close
breakouts held 3 days beat 2, 5 and 10 days; entering at the first 4h close beyond the daily box beat waiting for
the daily close (earlier, more entries); a 4h-bar box (3.3 days) was much weaker. Compression lifted the average
trade (+0.159R vs +0.130R without it) but halved the count, so the day-clustered t was not higher with it: the
range breakout itself carries most of the effect. Lab results: docs/results/p2_vol_breakout_*.json.
"""
from __future__ import annotations

import math

import numpy as np

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import FrameCursor, MarketView
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec
from heartless.util.timeutil import MS_DAY, TF_MS

TF = "4h"  # decision timeframe (the breakout close)
MTF = "1d"  # box, compression, stop ATR, time stop and trail timeframe
RANK_WINDOW = 60  # daily bars the bandwidth percentile looks back over
SMA_SLOW = 50  # trend_filter: daily SMA20 (bb_mid) vs SMA50


def width_rank(widths: np.ndarray, offset: int, window: int = RANK_WINDOW) -> float:
    """Fraction of the `window` widths ending `offset` bars before the last one that are <= the newest of them; NaN
    unless all of them are known (the alpha never ranks against a warm-up gap)."""
    end = len(widths) - offset
    if offset < 0 or end < window:
        return float("nan")
    seg = widths[end - window:end]
    if not np.isfinite(seg).all():
        return float("nan")
    return float(np.mean(seg <= seg[-1]))


def box(d: FrameCursor, day: int, n: int) -> tuple[float, float]:
    """(high, low) of the n daily bars that closed before UTC day index `day`, read from the last closed daily bar
    `d`: that bar is the day itself (the 4h bar closing at 23:59 closes its day too) or the day before. Anything else
    (a gap in the daily bars) -> NaN."""
    d_day = d.open_time // MS_DAY
    if day == d_day:
        return d.v(f"dc_hi{n}"), d.v(f"dc_lo{n}")  # excludes the bar itself
    if day == d_day + 1:
        return d.v(f"hh{n}"), d.v(f"ll{n}")  # includes the bar
    return float("nan"), float("nan")


class VolBreakoutDbox(Alpha):
    name = "vol_breakout_dbox"
    timeframe = TF
    enabled_by_default = False  # P2 candidate: on only after the pre-registered TRAIN and VALID gates
    description = ("daily volatility compression -> first 4h close beyond the 20-day box with BTC's week, market "
                   "entry, wide daily-ATR stop, ~3-day hold (winners run under a trail)")
    param_specs = [
        ParamSpec("box_d", 20, 10, 50, choices=(10, 20, 50)),  # box length in daily bars (frame Donchian columns)
        ParamSpec("comp_q", 0.25, 0.05, 1.0, 0.05),  # bandwidth percentile that counts as compressed (1 = off)
        ParamSpec("comp_days", 5, 1, 7, 1, integer=True),  # daily bars before the breakout day searched for it
        ParamSpec("btc_filter", 1, 0, 1, 1, integer=True),  # 1: only with the sign of BTC's 168h return
        ParamSpec("trend_filter", 0, 0, 1, 1, integer=True),  # 1: only with the daily SMA20 vs SMA50
        ParamSpec("fund_max", 0.0003, 0.0, 0.001, 0.0001),  # skip entries paying more funding than this (0 = off)
        ParamSpec("side_mode", 0, -1, 1, 1, integer=True),  # 0 both sides, 1 longs only, -1 shorts only
        ParamSpec("sl_atr", 2.0, 1.0, 4.0, 0.25),  # stop distance in daily ATRs
        ParamSpec("trail_atr", 0.0, 0.0, 4.0, 0.5),  # daily chandelier trail before the time stop (0 = none)
        ParamSpec("max_hold_h", 72, 24, 168, 12, integer=True),  # time stop in hours
    ]
    # the breakout bar is a volatility event by construction, and the 5m VOLATILE label (5m ATR rank >= 0.96 or a
    # 4-ATR 3-bar move) fires on it; the default affinity (0.6) would scale the score to 0.39 < entry_threshold and
    # drop exactly those entries, while the 15m/1h trend labels say nothing about a multi-week box. Equal weights.
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    # bandwidth ranks over RANK_WINDOW days for up to comp_days + 1 offsets, each width needing 20 closes: 20 + 60 + 8
    # daily bars, which also covers the SMA50 and the 50-day box
    lookback_days = 90
    max_hold_days = 21  # longest time stop (168 h) x 3: the engine's cap for a winner run past its deadline

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        h4, d = view.tf(TF), view.tf(MTF)
        if not (h4.ok and d.ok):
            return None
        close, prev_close, atr = h4.v("close"), h4.v("close", 1), d.v("atr")
        if not ok(close, prev_close, atr) or atr <= 0 or close <= 0:
            return None
        n = int(p["box_d"])
        day = h4.open_time // MS_DAY  # UTC day of the breakout bar
        prev_day = (h4.open_time - TF_MS[TF]) // MS_DAY
        hi, lo = box(d, day, n)
        prev_hi, prev_lo = box(d, prev_day, n)
        if not ok(hi, lo, prev_hi, prev_lo):
            return None  # warm-up or a gap in the daily bars
        if close > hi and prev_close <= prev_hi:
            side, level = Side.LONG, hi
        elif close < lo and prev_close >= prev_lo:
            side, level = Side.SHORT, lo
        else:
            return None
        sg = side.sign
        q, k = float(p["comp_q"]), int(p["comp_days"])
        comp = float("nan")
        if q < 1.0:
            # the k closed days before the breakout day: offsets 0..k-1 from the last closed daily bar, 1..k when the
            # breakout bar closed that day itself (the 20:00 bar)
            off0 = int(d.open_time // MS_DAY - (day - 1))
            widths = d.arr("bb_width", RANK_WINDOW + off0 + k)
            ranks = [width_rank(widths, j) for j in range(off0, off0 + k)]
            if not all(ok(r) for r in ranks):
                return None
            comp = min(ranks)
            if comp > q + 1e-12:
                return None
        if int(p["btc_filter"]):
            btc = ctx.market.get("btc_ret_168h", float("nan"))
            if not ok(btc) or btc * sg <= 0:
                return None  # unknown or against BTC's week
        if int(p["trend_filter"]):
            m20, closes = d.v("bb_mid"), d.arr("close", SMA_SLOW)
            if len(closes) < SMA_SLOW or not np.isfinite(closes).all() or not ok(m20):
                return None
            if (m20 - float(closes.mean())) * sg <= 0:
                return None
        mode = int(p["side_mode"])
        if (mode == 1 and side is Side.SHORT) or (mode == -1 and side is Side.LONG):
            return None
        fund = ctx.funding_rate
        if p["fund_max"] > 0 and ok(fund) and fund * sg > p["fund_max"]:
            return None
        conf = 0.65
        if conf < p["min_conf"]:
            return None
        stop = close - sg * p["sl_atr"] * atr
        hold = int(p["max_hold_h"])
        up = side is Side.LONG
        comp_txt = f"일봉 밴드폭 {RANK_WINDOW}일 하위 {comp:.0%} 수축 뒤 " if math.isfinite(comp) else ""
        reason = (f"{comp_txt}{n}일 박스 {'상단' if up else '하단'} {level:.6g} 4시간봉 종가 돌파"
                  f"{' (BTC 7일 ' + ('상승' if up else '하락') + ')' if int(p['btc_filter']) else ''} → 시장가 "
                  f"{'롱' if up else '숏'} - 손절 {p['sl_atr']:.2f}일봉ATR · 목표 없음 · {hold}시간 보유"
                  f"(0.5R 이상이면 추적 손절로 연장)")
        # the hold ends one minute before a 4h close, so that close can already decide again
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=None, tp1=None, entry_style=EntryStyle.MARKET, max_hold_bars=hold * 60 - 1,
                      trail_atr_mult=p["trail_atr"], atr=atr, timeframe=MTF,
                      tags={"ref_price": close, "level": level, "box_d": n, "comp_rank": comp, "funding": fund,
                            "btc_ret_168h": ctx.market.get("btc_ret_168h", float("nan")), "stop_atr": p["sl_atr"]})

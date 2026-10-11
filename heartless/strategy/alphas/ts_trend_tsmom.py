"""Daily time-series momentum (ts_trend family, ROADMAP P2): the sign of the volatility-scaled 4-week return, decided
once a day at the 00:00 UTC close and held for a fixed number of days behind a wide ATR stop.

Long when z = (close / close[L] - 1) / (ATR% * sqrt(L)) on the 1d frame exceeds z_min, short below -z_min (L = 28
days by default; ATR% = daily ATR14 / close, so z is roughly the L-day move in units of its own volatility). Market
entry, protective stop sl_atr daily ATRs away, no target. The holding period is the exit: after max_hold days the
engine closes a trade that is not 0.5R up and lets a winner run under a breakeven stop and a 1.5-ATR daily trail for
at most 3x max_hold. trail_atr > 0 adds the engine's breakeven-at-1R and chandelier trail before that.
An entry is skipped while the upcoming funding rate is adverse by more than fund_max (crowded side, expensive carry,
and exactly the trades the engine's funding-avoidance exit would cut at the next settlement unless 0.3R up).

Why this shape (priors before the P2 data, docs/ROADMAP.md section 6): time-series momentum at 1-4 week look-backs
is the documented crypto trend effect (Liu & Tsyvinski 2021); htf_trend (1h, 48h pullback, 8-ATR target) failed out
of sample, so this design holds days to weeks, has no target and pays costs (~13bp a round trip) that are ~1% of R.
Screening on TRAIN 2022-2024 only (ts_trend trial ledger; an hourly simulator of the engine's exits): the 28-day
return beat 20/50-day Donchian breakouts and 7/14/42-day returns, the fixed holding period beat the 3-ATR chandelier
(which cut trends at the forced breakeven), and a BTC 7-day trend filter did not help. Longs carried most of the
edge. Lab results: docs/results/p2_ts_trend_*.json.
"""
from __future__ import annotations

import math

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1d"  # decision, stop ATR, time stop and trail timeframe


def trend_z(close: float, past: float, atr: float, days: int) -> float:
    """Return over `days` daily bars in units of the daily ATR% scaled by sqrt(days) (~ a volatility-scaled z)."""
    if not ok(close, past, atr) or past <= 0 or close <= 0 or atr <= 0 or days <= 0:
        return float("nan")
    return (close / past - 1.0) / (atr / close * math.sqrt(days))


class TsTrendTsmom(Alpha):
    name = "ts_trend_tsmom"
    timeframe = TF
    enabled_by_default = False  # P2 candidate: on only after the pre-registered TRAIN and VALID gates
    description = ("daily time-series momentum: sign of the vol-scaled 4-week return at the 00:00 UTC close, market "
                   "entry, wide daily-ATR stop, fixed holding period (winners run under a trail)")
    param_specs = [
        ParamSpec("lookback_d", 28, 14, 42, 7, integer=True),  # momentum look-back in daily bars
        ParamSpec("z_min", 0.5, 0.0, 1.5, 0.25),  # min |z| of the look-back return
        ParamSpec("side_mode", 0, -1, 1, 1, integer=True),  # 0 both sides, 1 longs only, -1 shorts only
        ParamSpec("fund_max", 0.0003, 0.0, 0.001, 0.0001),  # skip entries paying more funding than this (0 = off)
        ParamSpec("btc_filter", 0, 0, 1, 1, integer=True),  # 1: trade only in the direction of BTC's 168h return
        ParamSpec("sl_atr", 3.0, 2.0, 5.0, 0.5),  # stop distance in daily ATRs
        ParamSpec("trail_atr", 0.0, 0.0, 4.0, 0.5),  # daily chandelier trail before the time stop (0 = none)
        ParamSpec("max_hold", 10, 5, 20, 5, integer=True),  # time stop in days
    ]
    # the 15m/1h regime label says nothing about a 4-week trend; the default affinity would scale a VOLATILE day's
    # score to 0.39 < entry_threshold and drop exactly the breakout days, so every regime weighs the same
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    lookback_days = 45  # longest look-back (42 daily bars) + a few days for the daily ATR14
    max_hold_days = 60  # longest time stop (20 days) x 3: the engine's cap for a winner run past its deadline

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        d = view.tf(TF)
        L = int(p["lookback_d"])
        close, atr, past = d.v("close"), d.v("atr"), d.v("close", L)
        z = trend_z(close, past, atr, L)
        if not ok(z):
            return None  # warm-up: fewer than L + 1 daily bars
        if z > p["z_min"]:
            side = Side.LONG
        elif z < -p["z_min"]:
            side = Side.SHORT
        else:
            return None
        mode = int(p["side_mode"])
        if (mode == 1 and side is Side.SHORT) or (mode == -1 and side is Side.LONG):
            return None
        sg = side.sign
        fund = ctx.funding_rate
        if p["fund_max"] > 0 and ok(fund) and fund * sg > p["fund_max"]:
            return None
        if int(p["btc_filter"]):
            btc = ctx.market.get("btc_ret_168h", float("nan"))
            if not ok(btc) or btc * sg <= 0:
                return None  # unknown or against BTC's week
        conf = 0.65
        if conf < p["min_conf"]:
            return None
        stop = close - sg * p["sl_atr"] * atr
        hold = int(p["max_hold"])
        up = side is Side.LONG
        reason = (f"{L}일 수익률 {'상승' if up else '하락'} 추세(변동성 z {z:+.2f}) → 일봉 마감 시장가 "
                  f"{'롱' if up else '숏'} - 손절 {p['sl_atr']:.1f}일봉ATR · 목표 없음 · {hold}일 보유"
                  f"(0.5R 이상이면 추적 손절로 연장)")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=None, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(hold, TF), trail_atr_mult=p["trail_atr"], atr=atr, timeframe=TF,
                      tags={"ref_price": close, "z": z, "lookback_d": L, "funding": fund, "stop_atr": p["sl_atr"]})

"""Higher-timeframe trend: 1-week time-series momentum, entered on a 2-day pullback against it (1h decisions).

Long when the 168h return is up (vol-scaled z > trend_z) and the last 48h moved DOWN by more than pull_z (vol-scaled);
short mirrored. Post-only entry at the touch, 3 x 1h-ATR stop, far 8-ATR target, 1h chandelier trail, 2-day time stop.
The z-score of an L-bar move is (close / close[L] - 1) / (ATR% * sqrt(L)) on the 1h frame.

Research notes (real Binance futures, 12 symbols; TRAIN 2026-01-13..07-01 was a -35..-69% bear market):
* Plain trend following has no edge beyond the drift here: after demeaning, 1h-based returns show no positive
  autocorrelation at 6-48h and variance ratios < 1 at 2-7 days (multi-day moves revert). Fresh 1h Donchian-20/50
  breakouts (with or without an EMA50/200 filter, 2-3 ATR stops, 1h trail) made ~0..+0.1R on TRAIN, all of it from
  shorts riding the bear drift (longs -0.05..-0.28R); "always short every 24h" alone made +0.12R with these exits.
* Combining the two horizons works on TRAIN: weekly momentum sets the side and a 2-day counter-move (the short-term
  reversal) times the entry. Measured against same-side random entries with identical exits, the timing added about
  +0.3R per trade, positive on both sides in every TRAIN month (also in the flat months Mar/Apr).
* 15m entry triggers (close back over EMA9/EMA21, RSI7 > 50) added nothing; maker entries at the touch fill ~99%.
  Averaging several look-backs, or pullbacks measured as drawdown from the 48h extreme / distance from an EMA, were
  weaker (~+0.1R, longs negative).
* Lab TRAIN (defaults): n=303 avgR +0.32 t 4.5 (day-clustered 2.1) PF 1.99, 6/6 months, 12/12 symbols; under 1.5x fees
  and 2x slippage +0.29R PF 1.91. VALID 2026-07..08: n=102 avgR +0.13 PF 1.22 (t 1.3), 2/2 months, 7/12 symbols.
* Caveats: with trend_z=0 (the first candidate) TRAIN was as good (+0.31R, n=484) but VALID was flat (+0.03R, PF 1.00);
  trend_z=0.25 was chosen on VALID between those two. The pullback look-back is the fragile knob (TRAIN lab: 42 bars
  +0.13R, 54 bars +0.30R; the simulator puts 36 bars at ~0). Shorts carried most of TRAIN (+0.50R vs longs +0.10R).
"""
from __future__ import annotations

import math

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1h"  # decision, stop and trailing timeframe


def vol_z(close: float, past: float, atr: float, bars: int) -> float:
    """Return over `bars` 1h bars in units of the 1h ATR% scaled by sqrt(bars) (~ a volatility-normalised z-score)."""
    if not ok(close, past, atr) or past <= 0 or close <= 0 or atr <= 0 or bars <= 0:
        return float("nan")
    return (close / past - 1.0) / (atr / close * math.sqrt(bars))


class HtfTrend(Alpha):
    name = "htf_trend"
    timeframe = TF
    enabled_by_default = True  # met the TRAIN/stress/VALID acceptance gates (see the caveats in the module docstring)
    description = ("1-week time-series momentum (vol-scaled 168h return) entered on a 2-day pullback against it; "
                   "post-only entry, 3-ATR 1h stop, far target + 1h chandelier trail, ~2 day hold")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry; tp1_frac is unused: no partial)
    param_specs = [
        ParamSpec("week_bars", 168, 96, 336, 12, integer=True),  # trend look-back (1h bars)
        ParamSpec("pull_bars", 48, 12, 96, 6, integer=True),  # pullback look-back (1h bars)
        ParamSpec("trend_z", 0.25, 0.0, 1.5, 0.25),  # min |z| of the week_bars return in the trade direction
        ParamSpec("pull_z", 0.5, 0.0, 1.5, 0.25),  # min |z| of the pull_bars move AGAINST the trend
        ParamSpec("sl_atr", 3.0, 1.5, 4.5, 0.25),  # stop distance in 1h ATRs
        ParamSpec("tp_atr", 8.0, 3.0, 12.0, 0.5),  # final target in 1h ATRs (the trail does most exits)
        ParamSpec("trail_atr", 3.0, 0.0, 4.0, 0.25),  # 1h chandelier trail in ATRs (0 = none before the time stop)
        ParamSpec("max_hold", 48, 12, 96, 6, integer=True),  # time stop in 1h bars
    ]
    # the alpha buys pullbacks, i.e. usually while the short-term regime label points against the weekly trend, so the
    # label is no filter here (TRAIN and VALID disagreed on which regime was best)
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        h1 = view.tf(TF)
        wb, pb = int(p["week_bars"]), int(p["pull_bars"])
        close, atr = h1.v("close"), h1.v("atr")
        past_w, past_p = h1.v("close", wb), h1.v("close", pb)
        if not ok(close, atr, past_w, past_p) or atr <= 0 or close <= 0:
            return None  # warm-up: not enough 1h history yet
        z_w = vol_z(close, past_w, atr, wb)
        z_p = vol_z(close, past_p, atr, pb)
        if not ok(z_w, z_p):
            return None
        if z_w > p["trend_z"] and z_p < -p["pull_z"]:
            side = Side.LONG
        elif z_w < -p["trend_z"] and z_p > p["pull_z"]:
            side = Side.SHORT
        else:
            return None
        sg = side.sign
        dist = p["sl_atr"] * atr
        stop = close - sg * dist
        tp = close + sg * p["tp_atr"] * atr if p["tp_atr"] > p["sl_atr"] else None
        conf = 0.65
        if abs(z_w) >= 1.0:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        up = side is Side.LONG
        reason = (f"1주 {'상승' if up else '하락'} 추세(168시간 z {z_w:+.1f}) 속 {pb}시간 {'조정' if up else '반등'}"
                  f"(z {z_p:+.1f}) 후 추세 방향 지정가 진입 - 손절 {p['sl_atr']:.1f}ATR · 목표 {p['tp_atr']:.0f}ATR · "
                  f"1시간 트레일 {p['trail_atr']:.1f}ATR · 최대 {int(p['max_hold'])}시간 보유")
        limit = ctx.ticker.bid if up else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=None, entry_style=EntryStyle.LIMIT, limit_price=limit or close,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), TF), trail_atr_mult=p["trail_atr"], atr=atr,
                      timeframe=TF, tags={"ref_price": close, "z_week": z_w, "z_pull": z_p, "stop_atr": p["sl_atr"]})

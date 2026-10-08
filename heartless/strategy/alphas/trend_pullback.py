"""Trend pullback on 1h: aligned EMA21/50/200 trend, pullback into the EMA21 zone, resumption bar; post-only entry,
stop beyond the pullback swing, far target + 1h chandelier trail (let the trend run).

Research status (real Binance futures data, see the lab): NO EDGE - shipped disabled.
* The original 5m design lost -0.33R/trade on TRAIN (gross edge negative, costs ~0.15R).
* On this data the EMA/ADX/Donchian trend state does not predict continuation at 1-24h horizons (strong multi-day
  moves tend to mean-revert), so pullback entries have ~zero gross expectancy. Moving to 1h decisions with wide
  swing stops and maker entries cut costs to ~0.04R and brought TRAIN to about break-even (+0.01R), but VALID
  (Jul-Aug 2026) lost -0.20R/trade.
* `ext_max` (skip trends whose 3-day move exceeds ext_max ATR) lifted TRAIN to +0.16R but failed VALID (-0.27R):
  regime-specific overfit, so it is off by default (0) and kept only as a research knob.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1h"  # decision, pullback and trailing timeframe
EXT_LB = 72  # bars (3 days) over which trend extension is measured for ext_max


class TrendPullback(Alpha):
    name = "trend_pullback"
    timeframe = TF
    enabled_by_default = False  # no edge on real data (TRAIN ~break-even, VALID negative)
    description = ("1h EMA21/50/200 trend, pullback into the EMA21 zone, resumption bar; post-only entry, stop beyond "
                   "the swing (>= sl_atr ATR), far target + 1h chandelier trail; optional 3-day extension filter")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry)
    param_specs = [
        ParamSpec("adx_min", 15, 0, 35, 1, integer=True),
        ParamSpec("pullback_atr", 0.3, 0.0, 1.0, 0.1),  # how close (ATR) the pullback low must come to EMA21
        ParamSpec("ext_max", 0.0, 0.0, 6.0, 0.5),  # 0 = off; else skip if the 3-day move in trend direction > ext_max ATR
        ParamSpec("sl_atr", 1.5, 0.8, 3.0, 0.1),  # minimum stop distance (ATR)
        ParamSpec("tp_r", 4.0, 1.2, 6.0, 0.1),  # final target (R); far, the trail does most exits
        ParamSpec("tp1_r", 0.0, 0.0, 1.8, 0.1),  # 0 = no partial: trend trades need their right tail
        ParamSpec("trail_atr", 3.0, 1.0, 4.0, 0.25),
        ParamSpec("max_hold", 24, 6, 48, 1, integer=True),  # in 1h bars
    ]
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 0.9, Regime.VOLATILE: 0.5}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        f = view.tf(TF)
        close, o, atr = f.v("close"), f.v("open"), f.v("atr")
        e9, e21, e50, e200, adx = f.v("ema9"), f.v("ema21"), f.v("ema50"), f.v("ema200"), f.v("adx")
        # warm-up NaNs must block both sides (nan comparisons would otherwise read as a trend)
        if not ok(close, o, atr, e9, e21, e50, e200, adx) or atr <= 0:
            return None
        if adx < p["adx_min"]:
            return None
        if e21 > e50 > e200:
            side = Side.LONG
        elif e21 < e50 < e200:
            side = Side.SHORT
        else:
            return None
        sg = side.sign
        ext = float("nan")
        if p["ext_max"] > 0:
            past = f.v("close", EXT_LB)
            if not ok(past):
                return None
            ext = (close - past) * sg / atr
            if ext > p["ext_max"]:
                return None  # trend already ran far over 3 days
        lows = [f.v("low", k) for k in range(4)]
        highs = [f.v("high", k) for k in range(4)]
        if not ok(*lows, *highs):
            return None
        tip = min(lows[:3]) if side is Side.LONG else max(highs[:3])
        touched = (tip - e21) * sg <= p["pullback_atr"] * atr and (close - e50) * sg > 0
        resume = (close - o) * sg > 0 and (close - e9) * sg > 0
        if not (touched and resume):
            return None
        swing = min(lows) if side is Side.LONG else max(highs)
        dist = max((close - swing) * sg + 0.3 * atr, p["sl_atr"] * atr)
        stop = close - sg * dist
        tp = close + sg * p["tp_r"] * dist
        tp1 = close + sg * p["tp1_r"] * dist if 0 < p["tp1_r"] < p["tp_r"] and p["tp1_frac"] > 0 else None
        conf = 0.65
        if adx >= 25:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"1h {'상승' if side is Side.LONG else '하락'} 추세(EMA21/50/200 정렬, ADX {adx:.0f}) 속 EMA21 눌림목 후 "
                  f"재개 봉 - 지정가 진입, 스윙 너머 손절 {dist / atr:.1f}ATR, 목표 {p['tp_r']:.1f}R·트레일링으로 추세 추종")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1, entry_style=EntryStyle.LIMIT, limit_price=limit or close,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), TF), trail_atr_mult=p["trail_atr"], atr=atr,
                      timeframe=TF, tags={"ref_price": close, "adx": adx, "ext_atr": ext, "stop_atr": dist / atr})

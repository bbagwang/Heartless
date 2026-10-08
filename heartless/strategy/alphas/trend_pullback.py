"""Trend pullback: in an aligned 1h EMA trend, buy the pullback to the EMA21 zone once the bar resumes the trend."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf, targets
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1h"  # decision / pullback timeframe
TREND_TF = "1h"  # trend-alignment timeframe


class TrendPullback(Alpha):
    name = "trend_pullback"
    timeframe = TF
    description = ("1h EMA21/50/200 trend, pullback into the EMA21 zone, resumption bar; post-only entry, "
                   "stop beyond the pullback swing (>= sl_atr ATR), partial at tp1_r, final tp_r, chandelier trail")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry)
    param_specs = [
        ParamSpec("adx_min", 15, 0, 35, 1, integer=True),
        ParamSpec("pullback_atr", 0.3, 0.0, 1.0, 0.1),
        ParamSpec("sl_atr", 1.5, 0.8, 3.0, 0.1),
        ParamSpec("tp_r", 2.0, 1.2, 4.0, 0.1),
        ParamSpec("tp1_r", 1.0, 0.6, 1.8, 0.1),
        ParamSpec("trail_atr", 3.0, 1.0, 4.0, 0.25),
        ParamSpec("max_hold", 24, 6, 48, 1, integer=True),  # in decision-timeframe bars
        ParamSpec("ext_lb", 0, 0, 168, 12, integer=True),  # 0 = off; trend-extension lookback in bars
        ParamSpec("ext_max", 2.5, 1.0, 6.0, 0.5),  # skip if the ext_lb-bar move in trend direction exceeds this (ATR)
    ]
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 0.9, Regime.VOLATILE: 0.5}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        f, h = view.tf(TF), view.tf(TREND_TF)
        close, o, atr = f.v("close"), f.v("open"), f.v("atr")
        e9, e21, e50, adx = f.v("ema9"), f.v("ema21"), f.v("ema50"), f.v("adx")
        h21, h50, h200 = h.v("ema21"), h.v("ema50"), h.v("ema200")
        if not ok(close, o, atr, e9, e21, e50, adx, h21, h50, h200) or atr <= 0:
            return None
        if adx < p["adx_min"]:
            return None
        if h21 > h50 > h200 and e21 > e50:
            side = Side.LONG
        elif h21 < h50 < h200 and e21 < e50:
            side = Side.SHORT
        else:
            return None
        sg = side.sign
        lows = [f.v("low", k) for k in range(4)]
        highs = [f.v("high", k) for k in range(4)]
        if not ok(*lows, *highs):
            return None
        lb = int(p.get("ext_lb", 0))
        if lb > 0:
            past = f.v("close", lb)
            if not ok(past) or (close - past) * sg / atr > p["ext_max"]:
                return None
        ext3 = min(lows[:3]) if side is Side.LONG else max(highs[:3])
        touched = (ext3 - e21) * sg <= p["pullback_atr"] * atr and (close - e50) * sg > 0
        resume = (close - o) * sg > 0 and (close - e9) * sg > 0
        if not (touched and resume):
            return None
        swing = min(lows) if side is Side.LONG else max(highs)
        dist = max((close - swing) * sg + 0.3 * atr, p["sl_atr"] * atr)
        stop = close - sg * dist
        tp, tp1 = targets(side, close, stop, p["tp_r"], p["tp1_r"])
        conf = 0.65
        if adx >= 25:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"1h {'상승' if side is Side.LONG else '하락'} 추세(EMA21/50/200 정렬, ADX {adx:.0f}) 속 EMA21 눌림목 후 "
                  f"재개 봉 - 지정가 진입, 손절 {dist / atr:.1f}ATR")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.LIMIT,
                      limit_price=limit or close, max_hold_bars=self.bars_to_1m(int(p["max_hold"]), TF),
                      trail_atr_mult=p["trail_atr"], atr=atr, timeframe=TF,
                      tags={"ref_price": close, "adx": adx, "stop_atr": dist / atr})

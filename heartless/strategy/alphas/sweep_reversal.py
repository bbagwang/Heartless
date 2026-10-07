"""Liquidity sweep reversal: a wick through the prior N-bar extreme that closes back inside with volume."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf, targets
from heartless.strategy.base import Alpha, Context, ok


class SweepReversal(Alpha):
    name = "sweep_reversal"
    timeframe = "5m"
    description = "Stop-hunt wick beyond prior N-bar high/low, close back inside with a rejection wick and volume (market entry)"

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("5m"):
            return None
        m5, h1 = view.tf("5m"), view.tf("1h")
        L = int(p["lookback"])
        close, o, h, l, atr = m5.v("close"), m5.v("open"), m5.v("high"), m5.v("low"), m5.v("atr")
        prior_hi, prior_lo = m5.v(f"dc_hi{L}"), m5.v(f"dc_lo{L}")
        vol_z, uw, lw, tr = m5.v("vol_z"), m5.v("upper_wick"), m5.v("lower_wick"), m5.v("taker_ratio")
        if not ok(close, o, h, l, atr, prior_hi, prior_lo, vol_z, uw, lw, tr) or atr <= 0:
            return None
        if vol_z < p["vol_z_min"]:
            return None
        h1_adx = h1.v("adx")
        if l < prior_lo - p["sweep_atr"] * atr and close > prior_lo and close > o and lw >= p["wick_min"]:
            if ctx.regime is Regime.TREND_DOWN and ok(h1_adx) and h1_adx > 35:
                return None
            side = Side.LONG
            stop = l - p["sl_buffer_atr"] * atr
        elif h > prior_hi + p["sweep_atr"] * atr and close < prior_hi and close < o and uw >= p["wick_min"]:
            if ctx.regime is Regime.TREND_UP and ok(h1_adx) and h1_adx > 35:
                return None
            side = Side.SHORT
            stop = h + p["sl_buffer_atr"] * atr
        else:
            return None
        entry = close
        if abs(entry - stop) > 2.5 * atr or abs(entry - stop) < 0.4 * atr:
            return None
        tp, tp1 = targets(side, entry, stop, p["tp_r"], p["tp1_r"])
        conf = 0.5
        if vol_z >= 2.0:
            conf += 0.1
        if (side is Side.LONG and tr >= 0.55) or (side is Side.SHORT and tr <= 0.45):
            conf += 0.08
        if ctx.regime is Regime.RANGE:
            conf += 0.1
        rsi = m5.v("rsi14")
        if ok(rsi) and ((side is Side.LONG and rsi < 35) or (side is Side.SHORT and rsi > 65)):
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"{L}봉 {'저점' if side is Side.LONG else '고점'} 유동성 스윕(꼬리 {(lw if side is Side.LONG else uw) * 100:.0f}%) 후 "
                  f"내부 복귀 마감, 거래량 z={vol_z:.1f} → 스탑헌팅 반전")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "5m"), trail_atr_mult=0.0, atr=atr,
                      timeframe="5m", tags={"ref_price": entry, "vol_z": vol_z})

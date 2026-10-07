"""Volatility squeeze (Bollinger inside Keltner) followed by a volume-confirmed Donchian breakout."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf, protective_stop, targets
from heartless.strategy.base import Alpha, Context, ok


class SqueezeBreakout(Alpha):
    name = "squeeze_breakout"
    timeframe = "5m"
    description = "BB/KC squeeze release + Donchian breakout with volume z-score and taker-flow confirmation (market entry)"

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("5m"):
            return None
        m5, m15 = view.tf("5m"), view.tf("15m")
        L = int(p["dc_len"])
        close, o, atr = m5.v("close"), m5.v("open"), m5.v("atr")
        sq_prev, bbw_prev = m5.v("squeeze_bars", 1), m5.v("bb_width_rank", 1)
        dc_hi, dc_lo = m5.v(f"dc_hi{L}"), m5.v(f"dc_lo{L}")
        bb_up, bb_lo, bb_mid = m5.v("bb_up"), m5.v("bb_lo"), m5.v("bb_mid")
        vol_z, tr, body = m5.v("vol_z"), m5.v("taker_ratio"), m5.v("body_ratio")
        hist, hist_prev = m5.v("macd_hist"), m5.v("macd_hist", 1)
        if not ok(close, o, atr, sq_prev, bbw_prev, dc_hi, dc_lo, bb_up, bb_lo, bb_mid, vol_z, tr, body, hist, hist_prev) or atr <= 0:
            return None
        compressed = sq_prev >= p["squeeze_bars_min"] or bbw_prev <= p["bbw_rank_max"]
        if not compressed:
            return None
        if vol_z < p["vol_z_min"] or body < 0.5:
            return None
        if close > dc_hi and close > bb_up and close > o and tr >= p["taker_min"] and hist > hist_prev:
            side = Side.LONG
        elif close < dc_lo and close < bb_lo and close < o and tr <= 1 - p["taker_min"] and hist < hist_prev:
            side = Side.SHORT
        else:
            return None
        # breakout bar must not already be over-extended (>2.5 ATR from the midline)
        if abs(close - bb_mid) > 2.5 * atr:
            return None
        entry = close
        stop = protective_stop(side, entry, atr, p["sl_atr"], bb_mid, min_atr=0.8)
        tp, tp1 = targets(side, entry, stop, p["tp_r"], p["tp1_r"])
        conf = 0.55
        if sq_prev >= 2 * p["squeeze_bars_min"]:
            conf += 0.1
        if vol_z >= 2.5:
            conf += 0.1
        up15 = m15.v("ema21") > m15.v("ema50") if ok(m15.v("ema21"), m15.v("ema50")) else None
        if up15 is not None and ((side is Side.LONG) == up15):
            conf += 0.08
        cvd = m5.v("cvd20")
        if ok(cvd) and ((side is Side.LONG and cvd > 0.05) or (side is Side.SHORT and cvd < -0.05)):
            conf += 0.05
        if ctx.regime is Regime.VOLATILE:
            conf -= 0.1
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"{int(sq_prev)}봉 변동성 스퀴즈 해제 → {L}봉 {'고점' if side is Side.LONG else '저점'} 돌파, "
                  f"거래량 z={vol_z:.1f}, 테이커 매수비율 {tr:.2f}")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "5m"), trail_atr_mult=p["trail_atr"], atr=atr,
                      timeframe="5m", tags={"ref_price": entry, "vol_z": vol_z, "squeeze_bars": sq_prev})

"""Trend pullback: ride the dominant 1h/15m trend after a shallow 5m pullback to the EMA21/VWAP zone."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf, protective_stop, targets
from heartless.strategy.base import Alpha, Context, ok


class TrendPullback(Alpha):
    name = "trend_pullback"
    timeframe = "5m"
    description = "1h/15m trend + ADX filter, 5m pullback to EMA21 with RSI dip, resumption bar entry (post-only limit)"

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("5m"):
            return None
        m5, m15, h1 = view.tf("5m"), view.tf("15m"), view.tf("1h")
        close, atr = m5.v("close"), m5.v("atr")
        ema9, ema21, ema50 = m5.v("ema9"), m5.v("ema21"), m5.v("ema50")
        adx15 = m15.v("adx")
        if not ok(close, atr, ema9, ema21, ema50, adx15) or atr <= 0:
            return None
        if adx15 < p["adx_min"]:
            return None
        up_15 = m15.v("ema21") > m15.v("ema50")
        up_1h = h1.v("ema21") > h1.v("ema50") if ok(h1.v("ema21"), h1.v("ema50")) else up_15
        if up_15 and up_1h and ema21 > ema50 and close > ema50:
            side = Side.LONG
        elif (not up_15) and (not up_1h) and ema21 < ema50 and close < ema50:
            side = Side.SHORT
        else:
            return None
        rsi_now = m5.v("rsi14")
        rsi_recent = [m5.v("rsi14", k) for k in range(0, 4)]
        lows = [m5.v("low", k) for k in range(0, 4)]
        highs = [m5.v("high", k) for k in range(0, 4)]
        if not ok(rsi_now, *rsi_recent, *lows, *highs):
            return None
        o, c = m5.v("open"), close
        if side is Side.LONG:
            touched = min(lows) <= ema21 + p["pullback_atr"] * atr * 0.4 and min(lows) >= ema50 - 0.3 * atr
            dipped = min(rsi_recent) <= p["rsi_low"]
            resume = c > o and c > ema9 and rsi_now > min(rsi_recent) and c > m5.v("high", 1) - 0.1 * atr
            flow_ok = m5.v("taker_ratio3") >= 0.48 if ok(m5.v("taker_ratio3")) else True
            structure = min(lows) - 0.25 * atr
        else:
            touched = max(highs) >= ema21 - p["pullback_atr"] * atr * 0.4 and max(highs) <= ema50 + 0.3 * atr
            dipped = max(rsi_recent) >= 100 - p["rsi_low"]
            resume = c < o and c < ema9 and rsi_now < max(rsi_recent) and c < m5.v("low", 1) + 0.1 * atr
            flow_ok = m5.v("taker_ratio3") <= 0.52 if ok(m5.v("taker_ratio3")) else True
            structure = max(highs) + 0.25 * atr
        if not (touched and dipped and resume and flow_ok):
            return None
        # don't chase: entry must still be near the EMA zone
        if abs(c - ema21) > 1.5 * atr:
            return None
        entry = c
        stop = protective_stop(side, entry, atr, p["sl_atr"], structure)
        tp, tp1 = targets(side, entry, stop, p["tp_r"], p["tp1_r"])
        conf = 0.55
        if adx15 >= 30:
            conf += 0.1
        if ctx.regime in (Regime.TREND_UP, Regime.TREND_DOWN):
            conf += 0.08
        tr3 = m5.v("taker_ratio3")
        if ok(tr3) and ((side is Side.LONG and tr3 > 0.55) or (side is Side.SHORT and tr3 < 0.45)):
            conf += 0.05
        vw = m5.v("vwap")
        if ok(vw) and ((side is Side.LONG and c > vw) or (side is Side.SHORT and c < vw)):
            conf += 0.05
        if ctx.btc_regime is not None and ctx.symbol != "BTCUSDT":
            if (side is Side.LONG and ctx.btc_regime is Regime.TREND_DOWN) or (side is Side.SHORT and ctx.btc_regime is Regime.TREND_UP):
                conf -= 0.1
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"{'상승' if side is Side.LONG else '하락'} 추세(15m ADX {adx15:.0f}, 1h EMA 정렬) 속 5m EMA21 눌림목 "
                  f"(RSI 저점 {min(rsi_recent):.0f}) 후 재개 캔들")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.LIMIT,
                      limit_price=limit or entry, max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "5m"),
                      trail_atr_mult=p["trail_atr"], atr=atr, timeframe="5m",
                      tags={"ref_price": entry, "adx15": adx15, "rsi_min": min(rsi_recent)})

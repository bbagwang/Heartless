"""Funding-rate crowding fade: lean against extreme funding when momentum stalls."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf, targets
from heartless.strategy.base import Alpha, Context, ok


class FundingFade(Alpha):
    name = "funding_fade"
    timeframe = "15m"
    description = "Extreme funding + RSI extreme + 15m stall and 5m Supertrend flip => fade the crowded side (post-only limit)"

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("15m"):
            return None
        m15, m5, h1 = view.tf("15m"), view.tf("5m"), view.tf("1h")
        fr = ctx.funding_rate
        if abs(fr) < p["funding_min"]:
            return None
        close, atr, rsi = m15.v("close"), m15.v("atr"), m15.v("rsi14")
        hh3, ll3, st5 = m15.v("hh10"), m15.v("ll10"), m5.v("st_dir")
        if not ok(close, atr, rsi, hh3, ll3, st5) or atr <= 0:
            return None
        if fr > 0 and rsi >= p["rsi_ext"] and close < hh3 - 0.3 * atr and st5 < 0:
            side = Side.SHORT
        elif fr < 0 and rsi <= 100 - p["rsi_ext"] and close > ll3 + 0.3 * atr and st5 > 0:
            side = Side.LONG
        else:
            return None
        if ctx.oi_change is not None and ctx.oi_change < -0.03:
            return None  # crowd already unwinding, late
        # don't fade a strong 1h trend in the funding direction
        h1_adx = h1.v("adx")
        if ok(h1_adx) and h1_adx > 35 and ((side is Side.SHORT and ctx.regime is Regime.TREND_UP) or
                                            (side is Side.LONG and ctx.regime is Regime.TREND_DOWN)):
            return None
        entry = close
        stop = entry - side.sign * p["sl_atr"] * atr
        tp, tp1 = targets(side, entry, stop, p["tp_r"], p["tp1_r"])
        conf = 0.5
        if abs(fr) >= 2 * p["funding_min"]:
            conf += 0.15
        if ctx.oi_change is not None and ctx.oi_change > 0.02:
            conf += 0.1
        h1_rsi = h1.v("rsi14")
        if ok(h1_rsi) and ((side is Side.SHORT and h1_rsi > 70) or (side is Side.LONG and h1_rsi < 30)):
            conf += 0.1
        if ctx.regime is Regime.RANGE:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"펀딩비 {fr * 100:+.3f}%/8h 극단({'롱' if fr > 0 else '숏'} 과밀) + 15m RSI {rsi:.0f} + "
                  f"고점 정체 & 5m 슈퍼트렌드 전환 → 과밀 포지션 역방향")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.LIMIT,
                      limit_price=limit or entry, max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "15m"),
                      trail_atr_mult=0.0, atr=atr, timeframe="15m",
                      tags={"ref_price": entry, "funding": fr, "oi_change": ctx.oi_change})

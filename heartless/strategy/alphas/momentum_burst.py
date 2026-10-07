"""1-minute momentum burst scalp: consecutive high-volume directional bars aligned with the 5m/15m trend."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf, protective_stop, targets
from heartless.strategy.base import Alpha, Context, ok


class MomentumBurst(Alpha):
    name = "momentum_burst"
    timeframe = "1m"
    description = "N consecutive 1m thrust bars with volume z-score and taker-flow confirmation in trend direction (market entry, tight trail)"

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("1m"):
            return None
        m1, m5, m15 = view.tf("1m"), view.tf("5m"), view.tf("15m")
        n = int(p["bars"])
        atr5 = m5.v("atr")
        closes = [m1.v("close", k) for k in range(n + 1)]
        opens = [m1.v("open", k) for k in range(n)]
        vz = [m1.v("vol_z", k) for k in range(n)]
        trs = [m1.v("taker_ratio", k) for k in range(n)]
        bodies = [m1.v("body_ratio", k) for k in range(n)]
        if not ok(atr5, *closes, *opens, *vz, *trs, *bodies) or atr5 <= 0:
            return None
        rising = all(closes[k] > closes[k + 1] and closes[k] > opens[k] for k in range(n))
        falling = all(closes[k] < closes[k + 1] and closes[k] < opens[k] for k in range(n))
        if not (rising or falling):
            return None
        avg_vz = sum(vz) / n
        avg_tr = sum(trs) / n
        avg_body = sum(bodies) / n
        move = closes[0] - opens[n - 1]
        if avg_vz < p["vol_z_min"] or avg_body < 0.5 or abs(move) < p["move_atr_min"] * atr5:
            return None
        st5, ema21_15, ema50_15 = m5.v("st_dir"), m15.v("ema21"), m15.v("ema50")
        up_bias = (ok(st5) and st5 > 0) or (ok(ema21_15, ema50_15) and ema21_15 > ema50_15)
        dn_bias = (ok(st5) and st5 < 0) or (ok(ema21_15, ema50_15) and ema21_15 < ema50_15)
        rsi7 = m1.v("rsi7")
        if rising and avg_tr >= p["taker_min"] and up_bias:
            if ok(rsi7) and rsi7 > 92:
                return None
            side = Side.LONG
            structure = min(m1.v("low", k) for k in range(n)) - 0.1 * atr5
        elif falling and avg_tr <= 1 - p["taker_min"] and dn_bias:
            if ok(rsi7) and rsi7 < 8:
                return None
            side = Side.SHORT
            structure = max(m1.v("high", k) for k in range(n)) + 0.1 * atr5
        else:
            return None
        ema21_5 = m5.v("ema21")
        if ok(ema21_5) and abs(closes[0] - ema21_5) > 3.0 * atr5:
            return None  # over-extended
        entry = closes[0]
        stop = protective_stop(side, entry, atr5, p["sl_atr"], structure, min_atr=0.5)
        tp, tp1 = targets(side, entry, stop, p["tp_r"], p["tp1_r"])
        conf = 0.5
        if avg_vz >= 3.0:
            conf += 0.1
        cvd = m1.v("cvd20")
        if ok(cvd) and ((side is Side.LONG and cvd > 0.15) or (side is Side.SHORT and cvd < -0.15)):
            conf += 0.1
        if (side is Side.LONG and ctx.regime is Regime.TREND_UP) or (side is Side.SHORT and ctx.regime is Regime.TREND_DOWN):
            conf += 0.1
        imb = ctx.book_imbalance
        if (side is Side.LONG and imb > 0.2) or (side is Side.SHORT and imb < -0.2):
            conf += 0.05
        if ctx.minutes_to_funding < 3:
            conf -= 0.1
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"1m {n}연속 {'양봉' if side is Side.LONG else '음봉'} 모멘텀 버스트 (이동 {abs(move) / atr5:.1f} ATR, "
                  f"거래량 z={avg_vz:.1f}, 테이커 {avg_tr:.2f}) + 상위TF 추세 일치")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=int(p["max_hold"]), trail_atr_mult=p["trail_atr"], atr=atr5, timeframe="1m",
                      tags={"ref_price": entry, "vol_z": avg_vz, "move_atr": abs(move) / atr5})

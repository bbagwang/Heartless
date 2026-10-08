"""Range-regime mean reversion: fade RSI(2)/Bollinger extremes back to the band midline."""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec


class MeanReversion(Alpha):
    name = "mean_reversion"
    timeframe = "5m"
    description = "RSI(2) + Bollinger(2.4σ) extreme with rejection wick in a low-ADX/high-choppiness range, target = band midline"
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry)
    param_specs = [
        ParamSpec("bb_k", 2.4, 1.8, 3.2, 0.1),
        ParamSpec("rsi2_lo", 6, 2, 15, 1, integer=True),
        ParamSpec("adx_max", 20, 14, 28, 1, integer=True),
        ParamSpec("chop_min", 55, 45, 65, 1, integer=True),
        ParamSpec("sl_atr", 1.4, 0.8, 2.5, 0.1),
        ParamSpec("max_hold", 24, 6, 60, 1, integer=True),
    ]
    # prior weight per market regime (0..1); the Thompson-sampling bandit learns the rest
    regime_affinity = {Regime.TREND_UP: 0.35, Regime.TREND_DOWN: 0.35, Regime.RANGE: 1.0, Regime.VOLATILE: 0.6}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("5m"):
            return None
        m5, m15, h1 = view.tf("5m"), view.tf("15m"), view.tf("1h")
        close, o, h, l, atr = m5.v("close"), m5.v("open"), m5.v("high"), m5.v("low"), m5.v("atr")
        mid, sd, rsi2 = m5.v("bb_mid"), m5.v("bb_sd"), m5.v("rsi2")
        adx15, chop15 = m15.v("adx"), m15.v("chop")
        if not ok(close, o, h, l, atr, mid, sd, rsi2, adx15, chop15) or atr <= 0 or sd <= 0:
            return None
        if ctx.regime in (Regime.TREND_UP, Regime.TREND_DOWN) and adx15 >= p["adx_max"]:
            return None
        if adx15 > p["adx_max"] + 6 or chop15 < p["chop_min"] - 8:
            return None
        upper, lower = mid + p["bb_k"] * sd, mid - p["bb_k"] * sd
        rng = max(h - l, 1e-12)
        h1_adx = h1.v("adx")
        h1_up = h1.v("ema21") > h1.v("ema50") if ok(h1.v("ema21"), h1.v("ema50")) else None
        if l < lower and rsi2 <= p["rsi2_lo"] and (close - l) / rng >= 0.3:
            if h1_up is False and ok(h1_adx) and h1_adx > 30:
                return None  # don't catch knives in a strong 1h downtrend
            side = Side.LONG
            structure = l
        elif h > upper and rsi2 >= 100 - p["rsi2_lo"] and (h - close) / rng >= 0.3:
            if h1_up is True and ok(h1_adx) and h1_adx > 30:
                return None
            side = Side.SHORT
            structure = h
        else:
            return None
        entry = close
        stop = entry - side.sign * p["sl_atr"] * atr
        # structure-based stop if it is tighter but still > 0.6 ATR
        alt = structure - side.sign * 0.2 * atr
        if abs(entry - alt) >= 0.6 * atr and abs(entry - alt) < abs(entry - stop):
            stop = alt
        tp = mid
        # Signed reward: the midline must be on the favourable side of entry (a wick can pierce the band while the
        # bar closes past the midline) and far enough away; a TP behind entry is rejected live (-2021) and
        # self-triggers on the next tick in paper/backtest.
        if (tp - entry) * side.sign < 0.6 * abs(entry - stop):  # reward too small or on the wrong side
            return None
        tp1 = entry + (tp - entry) * 0.5
        conf = 0.5
        if rsi2 <= 2 or rsi2 >= 98:
            conf += 0.08
        vw, vsd = m5.v("vwap"), m5.v("vwap_sd")
        if ok(vw, vsd) and vsd > 0:
            dev = (close - vw) / vsd
            if (side is Side.LONG and dev < -1.5) or (side is Side.SHORT and dev > 1.5):
                conf += 0.1
        if chop15 >= 61:
            conf += 0.08
        hurst = m15.v("hurst")
        if ok(hurst) and hurst < 0.45:
            conf += 0.05
        if ctx.regime is Regime.RANGE:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        reason = (f"횡보 레짐(15m ADX {adx15:.0f}, CHOP {chop15:.0f})에서 볼린저 {p['bb_k']:.1f}σ "
                  f"{'하단' if side is Side.LONG else '상단'} 이탈 + RSI(2)={rsi2:.0f} 극단, 중심선 회귀 목표")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.LIMIT,
                      limit_price=limit or entry, max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "5m"),
                      trail_atr_mult=0.0, atr=atr, timeframe="5m", tags={"ref_price": entry, "rsi2": rsi2})

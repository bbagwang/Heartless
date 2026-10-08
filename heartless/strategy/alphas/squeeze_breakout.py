"""1h volatility squeeze (Bollinger inside Keltner) released by a decisive Donchian breakout bar.

Research notes (TRAIN 2026-01..06, 12 symbols, see the research report for numbers):
* 5m squeeze breakouts lose to costs: the move after the break is smaller than the ~13bp round trip and the
  gross edge is negative (late, noisy entries that mean-revert).
* On 1h the picture changes: a fresh close beyond the prior 20-bar extreme after >= 4 bars of squeeze has a
  modest positive drift over the next ~day, and the drift is concentrated in breakouts whose bar is decisive
  (moved >= 1 ATR close-to-close with a body >= 50% of its range). Weak / wicky breaks, or strong breaks without
  a preceding squeeze, have ~no edge. Fading failed breakouts was not better than following real ones.
* Breakouts in the direction of the 1h EMA50/EMA200 trend carried the edge on both sides; counter-trend ones
  were noise.
* The edge needs room: stops of ~2-3 1h ATRs (~1.5-2.5%) and holds of ~1 day; tight stops, partial targets and
  early trailing give it back to noise and costs.
* VERDICT: no validated edge. TRAIN (a -35..-66% bear market) looked strong (n=208, avgR +0.33, PF 1.86, t 4.1,
  +0.29 under stress), but VALID 2026-07..08 was flat (n=69, avgR -0.04, PF 0.96; without the trend filter: n=137,
  avgR -0.03). The TRAIN result leaned on trend persistence that did not carry over, so the alpha ships disabled
  with the best TRAIN configuration as defaults.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

DC_LEN = 20  # Donchian look-back of the breakout level (1h bars)


class SqueezeBreakout(Alpha):
    name = "squeeze_breakout"
    timeframe = "1h"
    description = ("1h BB/KC squeeze -> decisive Donchian-20 breakout bar (>= 1 ATR, body >= 50%) with the 1h "
                   "EMA50/200 trend, wide 1h-ATR stop, ~1 day hold (market entry)")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry; tp1_frac is unused: no partial)
    param_specs = [
        ParamSpec("squeeze_bars_min", 4, 2, 12, 1, integer=True),  # 1h bars in squeeze before the breakout bar
        ParamSpec("ret_atr_min", 1.0, 0.5, 2.0, 0.05),  # breakout bar close-to-close move in 1h ATRs
        ParamSpec("body_min", 0.5, 0.3, 0.8, 0.05),  # breakout bar body / range
        ParamSpec("trend_align", 1, 0, 1, choices=(0, 1)),  # 1 = only breakouts with the 1h EMA50 vs EMA200 trend
        ParamSpec("sl_atr", 3.0, 1.5, 4.0, 0.1),  # stop distance in 1h ATRs
        ParamSpec("tp_atr", 7.0, 3.0, 12.0, 0.5),  # final target in 1h ATRs
        ParamSpec("trail_atr", 0.0, 0.0, 4.0, 0.25),  # chandelier trail in 1h ATRs (0 = none before the time stop)
        ParamSpec("max_hold", 24, 6, 72, 1, integer=True),  # time stop in 1h bars
    ]
    # prior weight per market regime (0..1); the Thompson-sampling bandit learns the rest
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    # TRAIN edge did not survive VALID (see module docstring): off unless the research loop re-enables it
    enabled_by_default = False

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("1h"):
            return None
        h1 = view.tf("1h")
        close, o, hi, lo = h1.v("close"), h1.v("open"), h1.v("high"), h1.v("low")
        prev_close, atr = h1.v("close", 1), h1.v("atr")
        dc_hi, dc_lo = h1.v(f"dc_hi{DC_LEN}"), h1.v(f"dc_lo{DC_LEN}")
        dc_hi_prev, dc_lo_prev = h1.v(f"dc_hi{DC_LEN}", 1), h1.v(f"dc_lo{DC_LEN}", 1)
        sq_prev = h1.v("squeeze_bars", 1)
        if not ok(close, o, hi, lo, prev_close, atr, dc_hi, dc_lo, dc_hi_prev, dc_lo_prev, sq_prev) or atr <= 0:
            return None
        if sq_prev < p["squeeze_bars_min"]:
            return None
        rng = hi - lo
        if rng <= 0 or abs(close - o) / rng < p["body_min"]:
            return None
        # fresh close beyond the prior 20-bar extreme (the previous bar was still inside its range)
        if close > dc_hi and prev_close <= dc_hi_prev and close > o:
            side = Side.LONG
        elif close < dc_lo and prev_close >= dc_lo_prev and close < o:
            side = Side.SHORT
        else:
            return None
        move_atr = (close - prev_close) * side.sign / atr
        if move_atr < p["ret_atr_min"]:
            return None
        e50, e200 = h1.v("ema50"), h1.v("ema200")
        with_trend = ok(e50, e200) and (e50 - e200) * side.sign > 0
        if p["trend_align"] >= 0.5 and not with_trend:
            return None
        entry = close
        stop = entry - side.sign * p["sl_atr"] * atr
        tp = entry + side.sign * p["tp_atr"] * atr
        conf = 0.65
        if sq_prev >= 2 * p["squeeze_bars_min"]:
            conf += 0.05
        if with_trend:
            conf += 0.05
        vol_z = h1.v("vol_z")
        if ok(vol_z) and vol_z >= 1.0:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        level = dc_hi if side is Side.LONG else dc_lo
        trend_txt = ", 1시간 EMA50/200 추세 방향" if with_trend else ""
        reason = (f"1시간봉 {int(sq_prev)}봉 변동성 스퀴즈 후 {DC_LEN}봉 {'고점' if side is Side.LONG else '저점'} "
                  f"{level:.6g} 강한 돌파 (돌파봉 {move_atr:.1f}ATR, 몸통 {abs(close - o) / rng:.0%}{trend_txt}), "
                  f"손절 {p['sl_atr']:.1f}ATR · 목표 {p['tp_atr']:.1f}ATR · 최대 {int(p['max_hold'])}시간 보유")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "1h"), trail_atr_mult=p["trail_atr"], atr=atr,
                      timeframe="1h", tags={"ref_price": entry, "squeeze_bars": sq_prev, "move_atr": move_atr,
                                            "level": level, "with_trend": with_trend})

"""Mean reversion after a multi-hour over-extension, entered on the first 1h exhaustion (reversal) bar.

Research notes (TRAIN 2026-01-13..07-01, 12 symbols; event studies on 1m paths + the production lab):
* The old 5m design (RSI(2) + 2.4 sigma Bollinger wick, target = band midline) lost -0.32R over 3567 trades:
  the midline target was tiny next to the stop, and costs alone were ~0.21R per trade.
* Plain band fades do not revert on this data: 1h Bollinger 2.5-3.0 sigma fades -0.10..-0.14R, 15m 2.5-3.5 sigma
  -0.16..-0.30R, with close-back-inside confirmation -0.08..-0.19R, maker entries only ~0.05R better. Range-edge
  fades (1h Donchian-20 sweep / edge in low 1h & 15m ADX) -0.13..-0.17R; fast 1h shocks (>= 2-3 1h ATR in 1h)
  -0.08..-0.27R; session-VWAP sigma extremes are rare in low-ADX conditions and tiny elsewhere.
* The one reversion effect found is after LARGE multi-hour stretches: a 1h close >= 4.5 1h-ATRs away from the close
  8 hours earlier, on the first 1h bar that closes against the move, reverts ~0.5-1 ATR over the next 4-12h. With a
  2 ATR stop / 2 ATR target / 12h hold it made +0.167R on TRAIN (n=120, t 2.5, PF 1.62, 10/12 symbols, 5/6 months,
  +0.127R under stress) -- but ONLY when run under another alpha name, i.e. without the engine's regime-flip exit.
  The edge grows monotonically with the stretch (>= 4.0: +0.06R, >= 5.0: +0.24R). On VALID that variant was
  negative (n=39, -0.08R, PF 0.72), so it is not a validated edge either way.
* The engine closes "mean_reversion" positions in loss whenever 15m ADX > 32 with the 15m slope against them. After
  such a stretch the 15m ADX is > 32 on every signal (median ~57) and the slope is against the trade on ~90%, so
  ~85% of the trades are scratched within ~1h: TRAIN -0.04R (n=180, PF 0.68). Variants that are "safe" from that
  exit (enter only once 15m ADX <= 32 or the 15m slope turned) lose the edge (-0.11..+0.10R, and the positive
  ones come only from shorts in a bear market: longs -0.08..-0.18R).
* VERDICT: no edge. Shipped disabled; the defaults are the best-found stretch-exhaustion configuration.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec


class MeanReversion(Alpha):
    name = "mean_reversion"
    timeframe = "1h"
    description = ("1h close stretched >= 4.5 1h-ATRs from the close 8h earlier, first 1h bar closing against the "
                   "move (exhaustion) => fade toward the mean; 2 ATR stop / 2 ATR target, 12h time stop (market entry)")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry; tp1_frac is unused: no partial)
    param_specs = [
        ParamSpec("lookback", 8, 4, 24, 1, integer=True),  # stretch window in 1h bars
        ParamSpec("stretch_atr", 4.5, 3.0, 7.0, 0.25),  # |close - close[lookback]| in 1h ATRs
        ParamSpec("sl_atr", 2.0, 1.0, 3.5, 0.1),  # stop distance in 1h ATRs
        ParamSpec("tp_atr", 2.0, 1.0, 4.0, 0.1),  # target distance in 1h ATRs
        ParamSpec("max_hold", 12, 4, 36, 1, integer=True),  # time stop in 1h bars
        ParamSpec("maker_entry", 0, 0, 1, choices=(0, 1)),  # 1 = post-only limit at the touch, 0 = market
    ]
    # the setup fires right after a violent move, which the regime detector labels TREND/VOLATILE: no regime veto
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    # no validated edge (see module docstring): off unless the research loop re-enables it
    enabled_by_default = False

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("1h"):
            return None
        h1 = view.tf("1h")
        lb = int(p["lookback"])
        close, o, atr = h1.v("close"), h1.v("open"), h1.v("atr")
        past = h1.v("close", lb)
        if not ok(close, o, atr, past) or atr <= 0 or close <= 0:
            return None
        stretch = (close - past) / atr
        # still stretched at the close of the first bar that closes against the move (exhaustion)
        if stretch <= -p["stretch_atr"] and close > o:
            side = Side.LONG
        elif stretch >= p["stretch_atr"] and close < o:
            side = Side.SHORT
        else:
            return None
        entry = close
        stop = entry - side.sign * p["sl_atr"] * atr
        tp = entry + side.sign * p["tp_atr"] * atr
        if (entry - stop) * side.sign <= 0 or (tp - entry) * side.sign <= 0 or stop <= 0 or tp <= 0:
            return None
        conf = 0.62
        if abs(stretch) >= p["stretch_atr"] + 1.0:  # the reversion grew with the stretch in research
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        maker = p["maker_entry"] >= 0.5
        touch = (ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask) if maker else 0.0
        move_txt = "급락" if side is Side.LONG else "급등"
        bar_txt = "양봉" if side is Side.LONG else "음봉"
        reason = (f"최근 {lb}시간 {abs(stretch):.1f}ATR {move_txt}(과도한 이탈) 후 첫 1시간 {bar_txt} 마감(소진 신호) → "
                  f"평균 회귀 {'롱' if side is Side.LONG else '숏'}, 손절 {p['sl_atr']:.1f}ATR · "
                  f"목표 {p['tp_atr']:.1f}ATR · 최대 {int(p['max_hold'])}시간 보유")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=None, entry_style=EntryStyle.LIMIT if maker else EntryStyle.MARKET,
                      limit_price=(touch or entry) if maker else None,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "1h"), trail_atr_mult=0.0, atr=atr,
                      timeframe="1h", tags={"ref_price": entry, "stretch_atr": stretch, "lookback": lb})

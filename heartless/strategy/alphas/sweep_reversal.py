"""Liquidity sweep reversal on 15m: a bar trades through the prior 24h (N-bar) high/low, where resting stops sit, and
closes back inside the range; fade the stop run with a post-only entry at the close, the stop beyond the sweep wick
plus a 1h-ATR buffer, a partial at tp1_r and the final target capped at the opposite side of the range.

Research status (real Binance futures data, see the lab): NO EDGE - shipped disabled.
* The original 5m design (20-bar sweep, vol_z >= 1, market entry, 0.3 ATR(5m) stop buffer) lost -0.33R/trade on TRAIN:
  gross -0.11R plus ~0.22R of costs on ~35bp stops.
* Event studies over 5m/15m/1h sweeps of 12-48h extremes, prior UTC-day highs/lows, multi-bar failed breakouts,
  market-structure-shift confirmation, retest limits, sweep depth / rejection / wick / volume, OI, long-short ratios,
  funding and time of day: a sweep followed by a close back inside has ~zero gross expectancy (|G| <= 0.03R) on this
  data; no conditioning gave a market-neutral gross edge above ~0.05R. The few positive subsets were short-only (TRAIN
  fell 34-65%) or came from one crash day (2026-03-29).
* This 15m redesign (1h-ATR stops ~110bp, maker entry) cut costs to ~0.07R/trade: TRAIN -0.07R (n=2082),
  VALID -0.05R (n=798).
* `trend_filter=1` (only sweeps on the 1h-EMA200 trend side) reached TRAIN -0.01R but VALID -0.11R: it only collected
  the bear-market short drift, so it is off by default and kept as a research knob.
"""
from __future__ import annotations

import numpy as np

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "15m"  # decision timeframe (the sweep bar)
ATR_TF = "1h"  # volatility unit for sweep depth, rejection and the stop buffer
MAX_STOP_ATR = 4.0  # skip bars whose stop would sit more than this many 1h ATRs away (news spikes)


class SweepReversal(Alpha):
    name = "sweep_reversal"
    timeframe = TF
    enabled_by_default = False  # no edge on real data (TRAIN -0.07R, VALID -0.05R)
    description = ("15m bar sweeps the prior 12-48h high/low and closes back inside; post-only entry, stop beyond the "
                   "wick + 1h-ATR buffer, target capped at the opposite side of the range")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry)
    param_specs = [
        ParamSpec("lookback", 96, 48, 192, choices=(48, 96, 192)),  # 15m bars defining the liquidity level (12/24/48h)
        ParamSpec("sweep_atr", 0.1, 0.0, 1.5, 0.05),  # min penetration beyond the level (1h ATR)
        ParamSpec("rej_atr", 0.1, 0.0, 1.0, 0.05),  # min close back inside the level (1h ATR)
        ParamSpec("sl_buffer_atr", 1.0, 0.1, 1.5, 0.05),  # stop beyond the sweep extreme (1h ATR)
        ParamSpec("tp_r", 2.0, 1.0, 4.0, 0.1),  # final target (R), capped at the opposite side of the range
        ParamSpec("tp1_r", 1.0, 0.5, 2.0, 0.1),
        ParamSpec("max_hold", 48, 8, 96, 1, integer=True),  # in 15m bars
        ParamSpec("trend_filter", 0, 0, 1, 1, integer=True),  # 1 = only sweeps on the 1h-EMA200 trend side (research)
    ]
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        m, h1 = view.tf(TF), view.tf(ATR_TF)
        n = int(p["lookback"])
        close, h, l = m.v("close"), m.v("high"), m.v("low")
        a = h1.v("atr")
        if not ok(close, h, l, a) or a <= 0 or close <= 0:
            return None
        lows, highs = m.arr("low", n + 1), m.arr("high", n + 1)
        if len(lows) < n + 1 or len(highs) < n + 1:
            return None  # warm-up: not enough history for the level
        prior_lo, prior_hi = float(np.min(lows[:-1])), float(np.max(highs[:-1]))  # the N bars before the current one
        if not ok(prior_lo, prior_hi) or prior_hi <= prior_lo:
            return None
        if l < prior_lo - p["sweep_atr"] * a and close > prior_lo + p["rej_atr"] * a:
            side, level, ext = Side.LONG, prior_lo, l
        elif h > prior_hi + p["sweep_atr"] * a and close < prior_hi - p["rej_atr"] * a:
            side, level, ext = Side.SHORT, prior_hi, h
        else:
            return None
        sg = side.sign
        if int(p["trend_filter"]):
            e200 = h1.v("ema200")
            if not ok(e200) or (close - e200) * sg <= 0:
                return None
        entry = close
        stop = ext - sg * p["sl_buffer_atr"] * a
        dist = (entry - stop) * sg
        if dist <= 0 or dist > MAX_STOP_ATR * a:
            return None
        tp = entry + sg * p["tp_r"] * dist
        tp1 = entry + sg * p["tp1_r"] * dist
        opp = prior_hi if side is Side.LONG else prior_lo
        room = (opp - entry) * sg
        if room < 1.2 * p["tp1_r"] * dist:
            return None  # the opposite side of the range is too close to pay for the risk
        if room < p["tp_r"] * dist:
            tp = opp - sg * 0.05 * a  # the liquidity on the other side of the range is the natural target
        depth = (level - ext) * sg / a
        vol_z = m.v("vol_z")
        conf = 0.6
        if ok(vol_z) and vol_z >= 2.0:
            conf += 0.05
        if (close - level) * sg >= 0.3 * a:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        hours = n * 15 // 60
        reason = (f"{hours}시간 {'저점' if side is Side.LONG else '고점'} 유동성 스윕(돌파 {depth:.2f}ATR) 후 "
                  f"레인지 안으로 복귀 마감 → 스탑헌팅 반전, 손절은 스윕 꼬리 너머")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1 if p["tp1_frac"] > 0 else None, entry_style=EntryStyle.LIMIT,
                      limit_price=entry, max_hold_bars=self.bars_to_1m(int(p["max_hold"]), TF), trail_atr_mult=0.0,
                      atr=a, timeframe=TF, tags={"ref_price": entry, "level": level, "depth_atr": depth})

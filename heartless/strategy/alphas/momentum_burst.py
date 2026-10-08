"""Momentum-burst exhaustion fade (1h): a high-volume 1h burst bar that extended the 1h trend and then stalled for
one bar is faded; stop 1.5 1h ATRs, target 2.5R, ~8h hold (market entry at the next bar open).

Research notes (real Binance futures data; TRAIN 2026-01-13..07-01, VALID 2026-07..08, 12 symbols):
* The original design (3 consecutive 1m thrust bars, market entry with the trend, ~36bp stop, 5 min hold) lost
  -0.40R/trade on TRAIN (n=1325, PF 0.14, every symbol and month negative): costs alone were ~0.28R/trade and the
  gross edge was negative too (-0.12R). Short-horizon continuation after 1m bursts does not survive costs.
* Event studies of 5m/15m/1h bursts (|close-to-close| >= k ATR with volume z >= z): continuation over the next
  15-120 min has a positive mean but a negative median (a few cascade days) and no day-clustered significance;
  bracketed continuation trades were ~0R. Fading at the burst close was ~0R too because the burst keeps running for
  about an hour, but starting the fade one 1h bar later captured a 4-8h reversal (+30bp mean over 8h).
* Lab (production engine) refinements, each with a reason: (1) only bursts in the direction of the 1h EMA50/EMA200
  trend (climax of an extended trend; counter-trend bursts tend to start a reversal and continue) - removing it
  cost ~0.09-0.13R/trade; (2) the decision bar must not extend the burst by more than ft_max ATR (stall
  confirmation; trades whose follow-through bar kept running > 1 ATR lost money, retracing ones made the most).
* An open-interest filter (fade only when 4h OI rose: late chasers) raised TRAIN avgR (+0.42R, n=150) but halved
  the trades and was worse on VALID (+0.18R, n=35 vs +0.24R, n=61 without it), so it is off by default (oi_min=-1)
  and kept as a research knob; the shipped alpha needs no positioning data.
* Result (shipped defaults): TRAIN n=255 avgR +0.35 t=4.9 (day-clustered 2.2) PF 2.05, 5/6 months and 11/12 symbols
  positive, +0.29 under 1.5x fees / 2x slippage; VALID n=61 avgR +0.24 PF 1.72. Caveats: TRAIN profits lean on
  Feb and May 2026 (Jan/Mar about flat, Jun slightly negative), VALID July was flat (August carried it), and the
  trend/stall filters were chosen on TRAIN, so expect live results well below the backtest.
* A post-only entry at the touch (instead of market) missed a few of the best trades and was slightly worse (+0.40R
  vs +0.42R/trade on the OI-filtered variant), so entries stay market orders.
"""
from __future__ import annotations

import math

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1h"  # decision, burst and trailing timeframe
LAG = 1  # the burst bar is LAG bars before the decision bar (wait one hour for the follow-through to finish)


class MomentumBurst(Alpha):
    name = "momentum_burst"
    timeframe = TF
    enabled_by_default = True  # passed the TRAIN acceptance criteria, the cost stress test and VALID
    description = ("Fade a high-volume 1h burst bar (>= 2 ATR, volume z >= 3) that extended the 1h EMA50/200 trend, "
                   "one bar later once it stalled; 1.5 1h-ATR stop, 2.5R target, ~8h hold (market entry)")
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry; tp1_frac is unused: no partial)
    param_specs = [
        ParamSpec("burst_atr", 2.0, 1.0, 4.0, 0.25),  # burst bar close-to-close move in 1h ATRs (ATR before the burst)
        ParamSpec("vol_z_min", 3.0, 1.0, 6.0, 0.25),  # burst bar volume z-score
        ParamSpec("trend_align", 1, 0, 1, choices=(0, 1)),  # 1 = only bursts in the direction of the 1h EMA50 vs EMA200 trend
        ParamSpec("ft_max", 0.5, -1.0, 3.0, 0.25),  # stall: max further move of the decision bar in the burst direction (ATR)
        ParamSpec("oi_min", -1.0, -1.0, 0.05, 0.005),  # min 4h open-interest change; <= -1 = off (research knob, see notes)
        ParamSpec("sl_atr", 1.5, 0.75, 3.0, 0.25),  # stop distance in 1h ATRs
        ParamSpec("tp_r", 2.5, 1.0, 4.0, 0.25),  # target in R
        ParamSpec("max_hold", 8, 2, 24, 1, integer=True),  # in 1h bars
    ]
    # the setup is defined by the burst itself; the regime classifier adds nothing, so no regime is penalised
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        h = view.tf(TF)
        close, atr = h.v("close"), h.v("atr")
        b_close, b_prev, b_atr, b_vz = h.v("close", LAG), h.v("close", LAG + 1), h.v("atr", LAG + 1), h.v("vol_z", LAG)
        if not ok(close, atr, b_close, b_prev, b_atr, b_vz) or atr <= 0 or b_atr <= 0:
            return None
        move = (b_close - b_prev) / b_atr
        if abs(move) < p["burst_atr"] or b_vz < p["vol_z_min"]:
            return None
        burst_dir = 1 if move > 0 else -1
        follow = (close - b_close) * burst_dir / atr
        if follow > p["ft_max"]:
            return None  # the burst is still extending: the climax has not happened yet
        if p["trend_align"]:
            e50, e200 = h.v("ema50"), h.v("ema200")
            if not ok(e50, e200) or (e50 - e200) * burst_dir <= 0:
                return None  # bursts against the trend tend to start a reversal (and continue), not to exhaust it
        oi4 = (ctx.extras or {}).get("oi_chg_4h", math.nan)
        if p["oi_min"] > -1:
            if oi4 is None or not ok(oi4) or oi4 < p["oi_min"]:
                return None  # no evidence that late chasers opened new positions (or no positioning data)
        side = Side.SHORT if burst_dir > 0 else Side.LONG
        sg = side.sign
        dist = p["sl_atr"] * atr
        stop = close - sg * dist
        tp = close + sg * p["tp_r"] * dist
        conf = clamp_conf(0.65)
        if conf < p["min_conf"]:
            return None
        oi_txt = f", 미결제약정 4h {oi4 * 100:+.1f}%" if oi4 is not None and ok(oi4) else ""
        reason = (f"1h 거래량 폭발 {'상승' if burst_dir > 0 else '하락'} 버스트({abs(move):.1f}ATR, 거래량 z={b_vz:.1f}{oi_txt})가 "
                  f"추세 방향으로 과열된 뒤 다음 봉에서 멈춤(추가 {follow:+.1f}ATR) → 과열 소진 역방향 진입, "
                  f"손절 {p['sl_atr']:.2f}ATR, 목표 {p['tp_r']:.1f}R, 최대 {int(p['max_hold'])}시간 보유")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), TF), trail_atr_mult=0.0, atr=atr, timeframe=TF,
                      tags={"ref_price": close, "burst_atr": move, "burst_vol_z": b_vz, "follow_atr": follow,
                            "oi_chg_4h": oi4, "stop_atr": p["sl_atr"]})

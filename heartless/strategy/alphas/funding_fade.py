"""Crowding fade around funding settlements: lean against a leveraged crowd once price rolls over against it.

Crowding is measured RELATIVE to each symbol's own recent history, because in 2026 the absolute funding rate is
pinned near the 0.01%/8h default most of the time (the old |funding| >= 0.04% rule never fired):
  * top-trader long/short POSITION ratio at a 7-day z-score extreme (ctx.extras["top_ls_pos"]),
  * fresh leverage: 24h open-interest build-up (ctx.extras["oi_chg_24h"]),
  * funding leaning the same way: z-score of ctx.funding_rate vs its own 7 days,
  * timing: within `settle_win` minutes of a funding settlement (paying side unwinds around it),
  * trigger: 15m close back through EMA9 + 5m Supertrend turned against the crowd, never against a running trend.
The z-scores need a per-symbol history that the alpha accumulates itself (one sample per evaluated 15m bar; at
least 3 days before it trades), so after a restart / at the start of a backtest it stays silent for 3 days.

Research status (real Binance data, see the lab): TRAIN 2026-01-13..07-01 n=231 avgR +0.23 t 2.6 PF 1.57, also
positive under --stress, BUT VALID 2026-07..09 n=78 avgR -0.31 PF 0.49 -> no robust edge; shipped disabled.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

_WINDOW_MS = 7 * 86_400_000  # z-score look-back of the positioning / funding history
_MIN_SPAN_MS = 3 * 86_400_000  # history must cover this span before the alpha trades
_MAX_GAP_MS = 2 * 86_400_000  # a longer hole (feed outage, restart, new backtest) resets the history
_FUNDING_PERIOD_MIN = 480.0


class _History:
    """Per-symbol rolling samples (one per evaluated 15m bar) of log(top-trader long/short position ratio) and the
    funding rate, so both can be expressed relative to their own recent history (z-scores)."""

    __slots__ = ("ts", "tlp", "fr")

    def __init__(self) -> None:
        self.ts: deque[int] = deque()
        self.tlp: deque[float] = deque()
        self.fr: deque[float] = deque()

    def add(self, ts: int, log_tlp: float, fr: float) -> None:
        if self.ts:
            last = self.ts[-1]
            if ts == last:
                return
            if ts < last or ts - last > _MAX_GAP_MS:
                self.ts.clear()
                self.tlp.clear()
                self.fr.clear()
        self.ts.append(ts)
        self.tlp.append(log_tlp)
        self.fr.append(fr)
        while self.ts[0] < ts - _WINDOW_MS:
            self.ts.popleft()
            self.tlp.popleft()
            self.fr.popleft()

    def ready(self) -> bool:
        return len(self.ts) >= 100 and self.ts[-1] - self.ts[0] >= _MIN_SPAN_MS

    @staticmethod
    def _z(vals: deque) -> float:
        a = np.fromiter(vals, dtype=float, count=len(vals))
        sd = float(a.std())
        return float((a[-1] - a.mean()) / sd) if sd > 1e-12 * max(1.0, abs(float(a.mean()))) else 0.0

    def z_tlp(self) -> float:
        return self._z(self.tlp)

    def z_fr(self) -> float:
        return self._z(self.fr)


class FundingFade(Alpha):
    name = "funding_fade"
    timeframe = "15m"
    description = ("Crowding fade: top-trader position skew at a 7-day extreme + 24h open-interest build-up + funding "
                   "z leaning the same way, near a funding settlement => fade the crowd once 15m/5m price rolls over "
                   "(post-only limit, 1.8x 1h-ATR stop, 3R target, 24h hold)")
    # experimental: positive on TRAIN, negative on VALID -> off unless the research loop re-enables it
    enabled_by_default = False
    # tunable parameters (the shared min_conf / tp1_frac are added by the registry)
    param_specs = [
        ParamSpec("z_min", 1.5, 1.0, 3.0, 0.1),  # |z| of log(top-trader long/short position ratio) vs its 7 days
        ParamSpec("oi_min", 0.02, -0.05, 0.08, 0.005),  # 24h open-interest change that marks fresh leverage
        ParamSpec("fz_min", 0.5, -1.0, 2.0, 0.1),  # funding z (vs 7 days) must lean with the crowd by more than this
        ParamSpec("sl_atr", 1.8, 0.8, 3.0, 0.1),  # stop distance in 1h ATR
        ParamSpec("tp_r", 3.0, 1.5, 5.0, 0.1),
        ParamSpec("tp1_r", 0.0, 0.0, 2.5, 0.1),  # 0 => no partial take profit
        ParamSpec("max_hold", 24, 6, 48, 1, integer=True),  # in 1h bars
        ParamSpec("settle_win", 120, 0, 240, 30, integer=True),  # minutes around a funding settlement (0 = any time)
    ]
    # prior weight per market regime (0..1); the Thompson-sampling bandit learns the rest. Every value keeps the
    # base confidence (0.6) above the 0.55 entry threshold: the counter-trend case is filtered inside evaluate().
    regime_affinity = {Regime.TREND_UP: 0.95, Regime.TREND_DOWN: 0.95, Regime.RANGE: 1.0, Regime.VOLATILE: 0.95}

    def __init__(self) -> None:
        self._hist: dict[str, _History] = {}

    def _observe(self, ctx: Context) -> _History | None:
        ex = ctx.extras or {}
        if ex.get("stale"):
            return None
        tlp = ex.get("top_ls_pos")
        if tlp is None or not ok(tlp) or tlp <= 0 or not ok(ctx.funding_rate):
            return None
        h = self._hist.get(ctx.symbol)
        if h is None:
            h = self._hist[ctx.symbol] = _History()
        h.add(int(ctx.now), math.log(tlp), float(ctx.funding_rate))
        return h

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("15m"):
            return None
        h = self._observe(ctx)  # sample every bar, whether or not it trades
        if h is None or not h.ready():
            return None
        sw = p.get("settle_win", 0)
        mtf = ctx.minutes_to_funding
        if sw and not (mtf <= sw or mtf >= _FUNDING_PERIOD_MIN - sw):
            return None
        oi24 = (ctx.extras or {}).get("oi_chg_24h")
        if oi24 is None or not ok(oi24) or oi24 < p["oi_min"]:
            return None  # no fresh leverage: nobody is trapped
        zt, fz = h.z_tlp(), h.z_fr()
        if zt >= p["z_min"] and fz > p["fz_min"]:
            crowd = Side.LONG
        elif zt <= -p["z_min"] and fz < -p["fz_min"]:
            crowd = Side.SHORT
        else:
            return None
        side = crowd.opposite
        # never lean against a running trend: the crowd is only trapped once price stops going its way
        if (side is Side.LONG and ctx.regime is Regime.TREND_DOWN) or (side is Side.SHORT and ctx.regime is Regime.TREND_UP):
            return None
        m15, m5, h1 = view.tf("15m"), view.tf("5m"), view.tf("1h")
        close, ema9, st5, atr1h = m15.v("close"), m15.v("ema9"), m5.v("st_dir"), h1.v("atr")
        if not ok(close, ema9, st5, atr1h) or atr1h <= 0 or close <= 0:
            return None
        # trigger: the 15m close is back below/above its EMA9 and the 5m Supertrend points our way
        if side is Side.SHORT and not (close < ema9 and st5 < 0):
            return None
        if side is Side.LONG and not (close > ema9 and st5 > 0):
            return None
        entry = close
        stop = entry - side.sign * p["sl_atr"] * atr1h
        r = abs(entry - stop)
        tp = entry + side.sign * p["tp_r"] * r
        tp1 = entry + side.sign * p["tp1_r"] * r if p["tp1_r"] > 0 and p["tp1_frac"] > 0 else None
        conf = 0.6
        if abs(zt) >= p["z_min"] + 1.0:
            conf += 0.1
        if abs(fz) >= p["fz_min"] + 1.0:
            conf += 0.05
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        crowd_kr = "롱" if crowd is Side.LONG else "숏"
        side_kr = "롱" if side is Side.LONG else "숏"
        when_kr = f"펀딩 정산 {mtf:.0f}분 전" if mtf <= _FUNDING_PERIOD_MIN / 2 else f"펀딩 정산 {_FUNDING_PERIOD_MIN - mtf:.0f}분 후"
        reason = (f"상위 트레이더 {crowd_kr} 포지션 과밀(7일 대비 z {zt:+.1f}) + 24h 미결제약정 {oi24 * 100:+.1f}% 증가 + "
                  f"펀딩 {ctx.funding_rate * 100:+.4f}%(7일 대비 z {fz:+.1f}) · {when_kr} → 15m/5m 가격 전환 확인, "
                  f"과밀 포지션 역방향 {side_kr}")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=tp1, entry_style=EntryStyle.LIMIT, limit_price=limit or entry,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "1h"), trail_atr_mult=0.0, atr=atr1h,
                      timeframe="1h",
                      tags={"ref_price": entry, "funding": ctx.funding_rate, "funding_z": fz, "top_pos_z": zt,
                            "oi_chg_24h": oi24, "crowd": crowd.value, "min_to_funding": mtf})

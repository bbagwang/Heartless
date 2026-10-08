"""Positioning: fade a leveraged spike once it rolls over (trapped late entrants unwind).

Setup (evaluated on every closed 15m bar, from ctx.extras = 5-minute Binance futures metrics):
  * a 1h price spike of at least `mv_min` 1h-ATR (close vs the close four 15m bars earlier),
  * fresh leverage piling in at the same time: the 1h open-interest change is a z-score outlier vs its last ~3 days
    (ctx.extras["oi_chg_1h_z"] >= `oiz_min`),
  * optional `chase` filter: the global account long/short ratio or the top-trader position ratio moved WITH the spike
    over the last 4h (ls_acc_chg_4h / top_ls_pos_chg_4h), i.e. the new positions are late trend chasers.
Trigger: within `wait` 15m bars of the (latest) setup bar, a 15m close back through the 15m EMA9 against the spike.
Trade: post-only limit against the spike, stop `sl_buf` 1h-ATR beyond the spike extreme (at least `sl_min` ATR, setups
needing more than 4 ATR are skipped), target `tp_r` R, time stop `max_hold` hours. extras == {} / stale => no signal.

Event study (TRAIN 2026-01-13..07-01, 12 symbols, drift-adjusted 12h forward return in 1h-ATR): spikes > 2 ATR with
OI z > 1 reverse by +0.4..1.1 ATR over 12h on 11-12/12 symbols, while equally large spikes WITHOUT an OI build-up
(z < 0) do not (the same trade loses -0.06R). Entering at the spike close loses (median adverse excursion 1.6 ATR):
the reversal starts only after 1-4h, hence the EMA9 confirmation. The positioning filters carry the edge: without
the OI condition the same trade loses, and requiring the L/S ratios to chase the spike lifts it further.

Research status (lab, production engine, nominal costs): NO ROBUST EDGE - shipped disabled.
  * TRAIN 2026-01-13..07-01 (shipped defaults): n=230 avgR +0.30 t 3.8 (day-clustered t 2.5) PF 1.95, longs +0.36 /
    shorts +0.25, 9/12 symbols and 4/6 months positive (January and March lose), still +0.26R under --stress. Every
    neighbour tried stays positive (mv_min 2.0..2.5, oiz_min 0.5..1.5, chase 0/1, tp_r 2.5..4, max_hold 8..16h,
    sl_buf 0.15..0.5, sl_min 1.0..1.5); only a short confirmation window (wait=8) is weak.
  * VALID 2026-07..09: n=58 avgR +0.005 PF 1.00 (mv_min=2.5 variant: n=40 avgR -0.14) -> fails the VALID bar.
  * Trades cluster on market-wide spike days (5-8 coins on the same day), so the effective sample is ~90 days, not
    230 trades; the TRAIN profit comes from February and April-June.
"""
from __future__ import annotations

from dataclasses import dataclass

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.alphas._common import clamp_conf
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "15m"  # decision timeframe
BAR_MS = 15 * 60_000
SPIKE_BARS = 4  # 4 x 15m = the 1h move
SL_MAX_ATR = 4.0  # a stop wider than this (spike extreme far away) is not worth the risk unit: skip


@dataclass
class _Setup:
    side: Side  # trade direction (against the spike)
    t0: int  # first setup bar close time (ms)
    last: int  # latest qualifying bar close time (ms)
    extreme: float  # spike high (for a short) / low (for a long), extended while waiting
    move: float  # 1h move in ATR at the latest qualifying bar (signed with the spike)
    oiz: float  # 1h open-interest change z-score at that bar
    chase: int  # how many positioning groups (crowd accounts, top-trader positions) chased the spike


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


class Positioning(Alpha):
    name = "positioning"
    timeframe = TF
    description = ("Leveraged-spike fade: 1h move >= mv_min ATR with an open-interest burst (1h OI change z) and "
                   "long/short ratios chasing it => after a 15m close back through EMA9, fade it (post-only limit, "
                   "stop beyond the spike extreme, 3R target, 12h time stop)")
    # no robust edge: TRAIN +0.30R/trade but VALID break-even -> off unless the research loop re-enables it
    enabled_by_default = False
    param_specs = [
        ParamSpec("mv_min", 2.25, 1.0, 4.0, 0.25),  # 1h spike size in 1h-ATR
        ParamSpec("oiz_min", 1.0, -1.0, 4.0, 0.25),  # z-score of the 1h open-interest change (vs ~3 days)
        ParamSpec("chase", 1, 0, 1, 1, integer=True),  # 1 = crowd or top traders must have added with the spike (4h)
        ParamSpec("wait", 24, 2, 32, 1, integer=True),  # 15m bars the setup stays armed for the EMA9 roll-over
        ParamSpec("sl_buf", 0.3, 0.0, 1.0, 0.05),  # stop beyond the spike extreme (1h ATR)
        ParamSpec("sl_min", 1.0, 0.5, 2.5, 0.1),  # minimum stop distance (1h ATR)
        ParamSpec("tp_r", 3.0, 1.0, 5.0, 0.25),
        ParamSpec("max_hold", 12, 2, 36, 1, integer=True),  # in 1h bars
    ]
    # the setup is regime-agnostic (spikes happen everywhere); every value keeps the base confidence 0.62 >= 0.55
    regime_affinity = {Regime.TREND_UP: 0.95, Regime.TREND_DOWN: 0.95, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}

    def __init__(self) -> None:
        self._setups: dict[str, _Setup] = {}

    # --- setup detection --------------------------------------------------------------------------
    def _spike(self, view: MarketView, ctx: Context, p: dict, atr: float) -> _Setup | None:
        ex = ctx.extras or {}
        if not ex or ex.get("stale"):
            return None
        oiz = _f(ex.get("oi_chg_1h_z"))
        if not ok(oiz) or oiz < p["oiz_min"]:
            return None
        m15 = view.tf(TF)
        close, past = m15.v("close"), m15.v("close", SPIKE_BARS)
        if not ok(close, past) or past <= 0:
            return None
        move = (close - past) / atr
        if abs(move) < p["mv_min"]:
            return None
        sg = 1.0 if move > 0 else -1.0
        lsa, tlp = _f(ex.get("ls_acc_chg_4h")), _f(ex.get("top_ls_pos_chg_4h"))
        chase = int(ok(lsa) and lsa * sg > 0) + int(ok(tlp) and tlp * sg > 0)
        if p.get("chase", 0) and chase == 0:
            return None
        if sg > 0:
            highs = [m15.v("high", k) for k in range(SPIKE_BARS)]
            if not ok(*highs):
                return None
            return _Setup(Side.SHORT, int(ctx.now), int(ctx.now), max(highs), move, oiz, chase)
        lows = [m15.v("low", k) for k in range(SPIKE_BARS)]
        if not ok(*lows):
            return None
        return _Setup(Side.LONG, int(ctx.now), int(ctx.now), min(lows), move, oiz, chase)

    def _armed(self, ctx: Context, fresh: _Setup | None, p: dict, high: float, low: float) -> _Setup | None:
        sym, now = ctx.symbol, int(ctx.now)
        s = self._setups.get(sym)
        if s is not None and (now < s.last or now - s.last > int(p["wait"]) * BAR_MS):
            s = None  # expired (or a new backtest restarted the clock)
        if fresh is not None:
            if s is not None and s.side is fresh.side:
                # the spike keeps going: keep the first bar, extend the extreme, re-arm the window
                ext = max(s.extreme, fresh.extreme) if fresh.side is Side.SHORT else min(s.extreme, fresh.extreme)
                s = _Setup(s.side, s.t0, now, ext, fresh.move, max(s.oiz, fresh.oiz), max(s.chase, fresh.chase))
            else:
                s = fresh
        elif s is not None and ok(high, low):
            s.extreme = max(s.extreme, high) if s.side is Side.SHORT else min(s.extreme, low)
        if s is None:
            self._setups.pop(sym, None)
        else:
            self._setups[sym] = s
        return s

    # --- evaluation -------------------------------------------------------------------------------
    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        m15, h1 = view.tf(TF), view.tf("1h")
        close, high, low, ema9 = m15.v("close"), m15.v("high"), m15.v("low"), m15.v("ema9")
        atr = h1.v("atr")
        if not ok(close, ema9, atr) or atr <= 0 or close <= 0:
            return None
        s = self._armed(ctx, self._spike(view, ctx, p, atr), p, high, low)
        if s is None:
            return None
        side, sg = s.side, s.side.sign
        # trigger: the 15m close is back through its EMA9 against the spike
        if (close - ema9) * sg <= 0:
            return None
        self._setups.pop(ctx.symbol, None)  # one trade per setup
        stop = s.extreme - sg * p["sl_buf"] * atr
        dist = (close - stop) * sg
        if dist < p["sl_min"] * atr:
            dist = p["sl_min"] * atr
            stop = close - sg * dist
        if dist > SL_MAX_ATR * atr:
            return None
        tp = close + sg * p["tp_r"] * dist
        conf = 0.62
        if abs(s.move) >= p["mv_min"] + 1.0:
            conf += 0.04
        if s.oiz >= p["oiz_min"] + 1.0:
            conf += 0.03
        if s.chase >= 2:
            conf += 0.04
        conf = clamp_conf(conf)
        if conf < p["min_conf"]:
            return None
        spike_kr = "급등" if s.move > 0 else "급락"
        crowd_kr = "롱" if s.move > 0 else "숏"
        side_kr = "숏" if side is Side.SHORT else "롱"
        who = {0: "", 1: f" + 포지션 비율 {crowd_kr} 추격", 2: f" + 군중·상위 트레이더 모두 {crowd_kr} 추격"}[min(s.chase, 2)]
        waited = (int(ctx.now) - s.t0) / 60_000
        reason = (f"1h {spike_kr} {abs(s.move):.1f}ATR 동안 미결제약정 급증(1h 변화 z {s.oiz:+.1f}){who} → "
                  f"{waited:.0f}분 뒤 15m 종가 EMA9 {'하향' if side is Side.SHORT else '상향'} 이탈로 반전 확인, "
                  f"갇힌 {crowd_kr} 청산을 노린 {side_kr} (손절: {spike_kr} 극단 너머 {dist / atr:.1f}ATR, "
                  f"목표 {p['tp_r']:.1f}R, 최대 {int(p['max_hold'])}시간)")
        limit = ctx.ticker.bid if side is Side.LONG else ctx.ticker.ask
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=tp, tp1=None, entry_style=EntryStyle.LIMIT, limit_price=limit or close,
                      max_hold_bars=self.bars_to_1m(int(p["max_hold"]), "1h"), trail_atr_mult=0.0, atr=atr,
                      timeframe="1h",
                      tags={"ref_price": close, "spike_atr": s.move, "oi_chg_1h_z": s.oiz, "chase": s.chase,
                            "extreme": s.extreme, "stop_atr": dist / atr, "wait_min": waited})

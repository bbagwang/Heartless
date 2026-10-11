"""OI-confirmed momentum against the retail crowd (positioning_flow family, ROADMAP P2): follow a large 24h move that
was made while open interest built up and while the global account long/short ratio leaned against it, decided at
the 4h closes and held for two days behind a wide daily-ATR stop.

At a 4h close (03:59:59.999, 07:59, ... UTC): the 24h move m = (close / close 24 1h-bars back - 1) / (daily ATR14 /
close), i.e. the day's move in units of a typical daily range, and from the positioning snapshot ctx.extras (the same
MetricsSeries.snapshot live and in backtests) the 24h open-interest change in contracts and the 4h change of the
global account long/short ratio. Long when m >= m_min, OI grew by at least oi_min and the account ratio fell over the
last 4h (more retail accounts short into the rally); short when m <= -m_min with the same OI build-up and a rising
account ratio. An entry is skipped while the trade's side already pays more than fund_max funding (ctx.funding_rate,
the upcoming rate): a crowded side, and exactly the trades the engine's funding-avoidance exit cuts at the next
settlement unless 0.3R up. Market entry, protective stop sl_atr daily ATRs away, no target. The holding period is the
exit: the time stop fires after max_hold hours, a minute before a 4h decision bar so that bar can renew the trade; the
engine keeps a trade that is 0.5R up at the deadline under a breakeven stop and a 1.5 x 4h-ATR trail for at most 3x.

Why this shape. The family's prior hypotheses were contrarian: leverage built into a move unwinds (shorts piling into
a drop get squeezed, longs piling into a rally get flushed), a move made while OI collapses (forced deleveraging)
reverts, and top traders diverging from the retail account ratio lead. TRAIN screening (2022-01-13..2024-12-31 only,
positioning_flow trial ledger, an hourly simulator of these exits; the lab matched 695 of its 708 trades) rejected
all three: fading 24h moves of >= 1 daily ATR made with >= 10% OI build-up lost -0.078R per trade at t_day -3.1
(mostly shorts against leveraged rallies, in the 2022 bear market as well), the deleveraging fade was flat (+0.02R,
t_day 0.6) and following the top-trader / retail divergence lost (-0.02R, t_day -1.9). The build-up result read the
other way is the classic price / OI confirmation: a move carried by new positions continues. A 48h hold, the funding
filter and a 2.5-ATR stop each helped in the screen, a stronger build-up (>= 15%) and a 72h hold did not; that
version (retail_filter=0) reached +0.064R at t_day 2.70 in the lab, short of the t_day 3 bar. The retail filter
applies the contrarian reading of the account ratio (the crowd of small accounts is wrong at the margin): leveraged
moves that retail accounts chased lost about -0.14R per trade in the screen, the rest kept the continuation. P1's 15m
spike fade with an OI burst (positioning) failed out of sample at -0.10R, the same mechanism on a short horizon. Lab
results: docs/results/p2_positioning_flow_*.json.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "4h"  # decision, time-stop alignment and post-deadline trail timeframe
MOVE_BARS = 24  # 1h bars of the move (24h)


def move_in_atr(close: float, past: float, atr_d: float) -> float:
    """24h return in units of the daily ATR% (daily ATR14 / close): ~ the day's move against a typical day's range."""
    if not ok(close, past, atr_d) or close <= 0 or past <= 0 or atr_d <= 0:
        return float("nan")
    return (close / past - 1.0) / (atr_d / close)


class PositioningFlowOiMom(Alpha):
    name = "positioning_flow_oimom"
    timeframe = TF
    enabled_by_default = False  # P2 candidate: on only after the pre-registered TRAIN and VALID gates
    description = ("OI-confirmed momentum against the retail crowd: follow a 24h move of >= m_min daily ATRs made "
                   "while open interest grew by >= oi_min and the global account ratio leaned the other way, at the "
                   "4h closes; market entry, wide daily-ATR stop, 48h hold, no target")
    param_specs = [
        ParamSpec("m_min", 1.0, 0.5, 2.0, 0.25),  # |24h move| in daily ATRs
        ParamSpec("oi_min", 0.10, 0.05, 0.25, 0.01),  # 24h open-interest growth (contracts) that counts as a build-up
        ParamSpec("retail_filter", 1, 0, 1, 1, integer=True),  # 1: the 4h account-ratio change must oppose the trade
        ParamSpec("side_mode", 0, -1, 1, 1, integer=True),  # 0 both sides, 1 longs only, -1 shorts only
        ParamSpec("fund_max", 0.0003, 0.0, 0.001, 0.0001),  # skip entries paying more funding than this (0 = off)
        ParamSpec("sl_atr", 2.5, 1.0, 4.0, 0.5),  # stop distance in daily ATRs
        ParamSpec("max_hold", 48, 12, 96, 12, integer=True),  # time stop in hours (a multiple of the 4h decisions)
    ]
    # the 5m/15m/1h regime label says nothing about a two-day positioning trade, and the default affinity would scale
    # a VOLATILE bar's score to 0.39 < entry_threshold, dropping the very bars on which a large move closes
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    lookback_days = 16  # the daily ATR14 (Wilder) needs 14 closed daily bars (+ the open day and a margin)
    max_hold_days = 12  # longest time stop (96 h) x 3: the engine's cap for a winner run past its deadline

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        ex = ctx.extras or {}
        if not ex or ex.get("stale"):
            return None  # no positioning data (or a feed gap): no OI build-up can be confirmed
        oi = ex.get("oi_chg_24h", float("nan"))
        if not ok(oi) or oi < p["oi_min"]:
            return None
        h1, d1 = view.tf("1h"), view.tf("1d")
        close, past, atr_d = h1.v("close"), h1.v("close", MOVE_BARS), d1.v("atr")
        m = move_in_atr(close, past, atr_d)
        if not ok(m) or abs(m) < p["m_min"]:
            return None
        side = Side.LONG if m > 0 else Side.SHORT
        mode = int(p["side_mode"])
        if (mode == 1 and side is Side.SHORT) or (mode == -1 and side is Side.LONG):
            return None
        sg = side.sign
        retail = ex.get("ls_acc_chg_4h", float("nan"))
        if int(p["retail_filter"]) and (not ok(retail) or retail * sg >= 0):
            return None  # retail accounts chased the move (or the ratio is unknown): the crowded continuation fails
        fund = ctx.funding_rate
        if p["fund_max"] > 0 and ok(fund) and fund * sg > p["fund_max"]:
            return None
        conf = 0.65
        if conf < p["min_conf"]:
            return None
        stop = close - sg * p["sl_atr"] * atr_d
        q = view.tf(TF)
        atr4 = q.v("atr")
        if not ok(atr4) or atr4 <= 0:
            atr4 = atr_d / 6 ** 0.5  # the post-deadline trail's fallback ATR (engine) ~ a 4h share of the daily range
        hold = int(p["max_hold"])
        up = side is Side.LONG
        crowd = (f", 개인 계정 롱숏비율 4시간 {retail:+.3f}({'숏' if up else '롱'} 쪽으로 역행)"
                 if int(p["retail_filter"]) else "")
        reason = (f"24시간 {'상승' if up else '하락'} {abs(m):.1f}일봉ATR + 미결제약정 {oi * 100:+.1f}%(신규 포지션 유입)"
                  f"{crowd} → 추세 추종 {'롱' if up else '숏'} - 4시간봉 마감 시장가, 손절 {p['sl_atr']:.1f}일봉ATR · "
                  f"목표 없음 · {hold}시간 보유(0.5R 이상이면 추적 손절로 연장)")
        # the hold ends a minute before a 4h close: the entry fills at the next bar's open, so whole hours would end it
        # on the decision bar itself, and the engine (busy closing the position there) would skip that decision
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=None, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(hold, "1h") - 1, trail_atr_mult=0.0, atr=atr4, timeframe=TF,
                      tags={"ref_price": close, "move_atr": m, "oi_chg_24h": oi, "ls_acc_chg_4h": retail,
                            "funding": fund, "stop_atr": p["sl_atr"], "daily_atr": atr_d})

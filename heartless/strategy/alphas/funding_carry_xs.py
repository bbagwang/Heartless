"""Funding carry and crowding (funding_carry family, ROADMAP P2): lean against the crowded side of the funding
cross-section right after a settlement and collect the funding for a fixed number of settlements.

Short when the symbol's settled funding rate is at least d_min above the market median (crowded, paying longs), long
when it is d_min below (crowded, paying shorts); by default only when open interest grew over the last 24h (fresh
leverage behind the crowded side). Decided at the first hour close after each settlement (00:59 / 08:59 / 16:59 UTC)
on ctx.market alone: settled rates, so the decision is the same live and in a backtest (the running rate of the next
settlement is a stream estimate that the backtester can only approximate). Market entry, protective stop sl_atr daily
ATRs away, no target. The holding period is the exit: the time stop fires after hold_k settlements, one minute before
the decision bar of the k-th next settlement, so that bar can renew a position whose funding is still extreme. The
engine keeps a trade that is 0.5R up at the deadline under a breakeven stop and a 1.5 daily-ATR trail (at most 3x).
Every settlement held over is paid / received by the backtester and counted in the trade's pnl and R.

Why this shape (priors before the P2 data, docs/ROADMAP.md section 6): longs pay shorts when perps trade rich, and an
extreme funding rate marks a crowded, levered side that tends to unwind (crowding reversal), so both effects point the
same way. Binance funding sits at the 0.01%/8h default most of the time (the premium clamp), hence the deviation from
the cross-sectional median instead of a raw rank: ties at the default carry no information. A held funding of
0.03-0.1%/8h is small next to a daily-ATR stop, so the hold is long (3 days by default) to let the carry and the unwind
build up against ~13bp of round-trip costs. funding_fade (15m entries near settlements) failed out of sample.

TRAIN screening (funding_carry trial ledger; an hourly simulator of these exits on the stored ctx.market): the rank
form of the carry (short the top, long the bottom funding ranks) lost because the high-funding shorts kept rising;
this extreme-deviation form was flat to weakly positive (best +0.03R at t_day 1.0 with the OI filter, 2022 negative,
mostly longs), a 1-day hold and a deviation from the symbol's own 7-day funding were weaker, and the funding received
was only ~0.01R per trade. Lab results: docs/results/p2_funding_carry_*.json.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

H_MS = 3_600_000
SETTLE_MS = 8 * H_MS  # Binance USD-M funding: 00:00 / 08:00 / 16:00 UTC
BARS_PER_SETTLE = 480  # 1m bars between two settlements
TF = "1d"  # stop ATR and post-deadline trail timeframe


def decision_hour(now: int) -> bool:
    """True at the close of the first hour after a settlement (e.g. 00:59:59.999 UTC): the settled rate is known."""
    return (int(now) + 1) % SETTLE_MS == H_MS


class FundingCarryXs(Alpha):
    name = "funding_carry_xs"
    timeframe = "1h"  # called at hour closes; decides only at the first one after a settlement
    enabled_by_default = False  # P2 candidate: on only after the pre-registered TRAIN and VALID gates
    description = ("funding crowding / carry: fade a settled funding rate far from the cross-sectional median (with "
                   "rising open interest) right after the settlement, hold a fixed number of settlements behind a "
                   "wide daily-ATR stop and collect the funding")
    param_specs = [
        ParamSpec("d_min", 0.0003, 0.0001, 0.001, 0.0001),  # |settled funding - median| per 8h that counts as crowded
        ParamSpec("oi_filter", 1, 0, 1, 1, integer=True),  # 1: require the 24h open-interest change >= oi_min
        ParamSpec("oi_min", 0.0, -0.1, 0.2, 0.01),
        ParamSpec("side_mode", 0, -1, 1, 1, integer=True),  # 0 both sides, 1 longs only, -1 shorts only
        ParamSpec("sl_atr", 2.5, 1.5, 4.0, 0.5),  # stop distance in daily ATRs
        ParamSpec("hold_k", 9, 3, 12, 3, integer=True),  # time stop in funding settlements (8h each)
    ]
    # the 15m/1h regime label says nothing about a multi-day funding trade; the default affinity would scale a
    # VOLATILE hour's score to 0.39 < entry_threshold and drop exactly the liquidation-driven extremes
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    lookback_days = 16  # the daily ATR14 (Wilder) needs 14 closed daily bars (+ the open day and a margin)
    max_hold_days = 12  # longest time stop (12 settlements = 4 days) x 3: the engine's cap for a winner past it

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed("1h") or not decision_hour(ctx.now):
            return None
        m = ctx.market
        ts = m.get("ts")
        if ts is None or int(ts) != int(ctx.now):
            return None  # no snapshot for this hour close (context unavailable or the symbol missing)
        fund, med = m.get("funding", float("nan")), m.get("funding_median", float("nan"))
        if not ok(fund, med):
            return None
        dev = fund - med
        if dev >= p["d_min"]:
            side = Side.SHORT  # longs crowded and paying
        elif -dev >= p["d_min"]:
            side = Side.LONG  # shorts crowded and paying
        else:
            return None
        mode = int(p["side_mode"])
        if (mode == 1 and side is Side.SHORT) or (mode == -1 and side is Side.LONG):
            return None
        oi = m.get("oi_chg_24h", float("nan"))
        if int(p["oi_filter"]) and (not ok(oi) or oi < p["oi_min"]):
            return None  # no fresh leverage behind the crowded side (or unknown)
        d, h1 = view.tf(TF), view.tf("1h")
        atr, close = d.v("atr"), h1.v("close")
        if not ok(atr, close) or atr <= 0 or close <= 0:
            return None  # warm-up: no daily ATR yet
        conf = 0.65
        if conf < p["min_conf"]:
            return None
        sg = side.sign
        stop = close - sg * p["sl_atr"] * atr
        k = int(p["hold_k"])
        up = side is Side.LONG
        reason = (f"정산 펀딩 {fund * 100:+.4f}% (시장 중앙값 {med * 100:+.4f}%, 차이 {dev * 100:+.4f}%)"
                  f"{f' · 24h 미결제약정 {oi * 100:+.1f}%' if ok(oi) else ''} → "
                  f"{'숏 과밀(숏이 지불)' if up else '롱 과밀(롱이 지불)'} 역방향 시장가 {'롱' if up else '숏'} - "
                  f"손절 {p['sl_atr']:.1f}일봉ATR · 목표 없음 · 펀딩 {k}회({k * 8}시간) 보유 후 시간 청산")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=None, tp1=None, entry_style=EntryStyle.MARKET,
                      # one minute before the decision bar of the k-th next settlement, which may renew the trade
                      max_hold_bars=k * BARS_PER_SETTLE - 1, trail_atr_mult=0.0, atr=atr, timeframe=TF,
                      tags={"ref_price": close, "funding": fund, "funding_median": med, "funding_dev": dev,
                            "funding_rank": m.get("funding_rank", float("nan")), "oi_chg_24h": oi,
                            "stop_atr": p["sl_atr"], "hold_k": k})

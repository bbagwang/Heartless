"""Cross-sectional momentum (xs_rank family, ROADMAP P2): at the 00:00 UTC daily close a symbol goes long when its
return over the look-back ranks in the top fraction q of the context universe (ctx.market ret_{L}h_rank) and short
when it ranks in the bottom q, then holds for a fixed number of hours behind a wide daily-ATR stop.

Each symbol decides from its own rank, so on any day about as many symbols are long as short and the book is close to
market neutral (with q = 0.2 and 19-20 symbols: the top four long, the bottom four short). Market entry, protective
stop sl_atr daily ATRs away (about two holding-period sigmas: a catastrophic stop, not a target), no target; the
holding period is the exit. After hold_h hours the engine closes a trade that is not 0.5R up and lets a winner run
under a breakeven stop and a 1.5-ATR daily trail for at most 3x hold_h. The time stop ends a minute before the daily
close so that close can decide again (as ts_trend_tsmom). An entry is skipped while the upcoming funding is adverse by
more than fund_max (the engine's funding-avoidance exit would close it at the next settlement unless 0.3R up), and,
with vol_rank_min > 0, unless the symbol's 24h quote-volume share ranks above vol_rank_min (moves on high relative
volume carry information and continue; Llorente, Michaely, Saar & Wang 2002).

Why this shape (priors before the P2 data, docs/ROADMAP.md section 6): 1-4 week cross-sectional momentum is the
documented crypto factor (Liu, Tsyvinski & Wu 2022), and ctx.market ranks returns up to 168 h; a time exit of 24-72 h
with a wide stop keeps the ~13bp round trip small next to the move. Screening on TRAIN 2022-2024 only (xs_rank trial
ledger; an hourly simulator of the engine's exits, 18 configurations): lookbacks 24/72/168 h x holds 24/72 h x
momentum / reversal were all at or below zero after costs, momentum beat reversal in every pair, and the gross
spread was ~5-25bp per trade against stops of ~10-15%, i.e. |avg R| <= 0.02. Extremes only (q = 0.1), high
dispersion, a 1.5-ATR stop-loss, skipping BTC downtrends and the volume conditions did not change that. The
default is the pre-registered prior design (168 h rank, 72 h hold). Lab results: docs/results/p2_xs_rank_*.json.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1d"  # decision, stop ATR and post-deadline trail timeframe
MIN_N = 10  # smallest cross-section whose ranks mean anything (the context universe has 20 symbols)


class XsRankMom(Alpha):
    name = "xs_rank_mom"
    timeframe = TF
    enabled_by_default = False  # P2 candidate: on only after the pre-registered TRAIN and VALID gates
    description = ("cross-sectional momentum: top / bottom fifth of the 20-symbol universe by 168h return at the "
                   "00:00 UTC close, market entry, wide daily-ATR stop, 72h time exit (winners run under a trail)")
    param_specs = [
        ParamSpec("lookback_h", 168, 24, 168, choices=(24, 72, 168)),  # ranked return (ctx.market ret_{L}h_rank)
        ParamSpec("q", 0.2, 0.1, 0.3, 0.05),  # long when rank >= 1 - q, short when rank <= q
        ParamSpec("hold_h", 72, 24, 72, choices=(24, 48, 72)),  # time stop in hours
        ParamSpec("sl_atr", 3.0, 1.5, 4.0, 0.5),  # stop distance in daily ATRs
        ParamSpec("fund_max", 0.0003, 0.0, 0.001, 0.0001),  # skip entries paying more funding than this (0 = off)
        ParamSpec("vol_rank_min", 0.0, 0.0, 0.5, choices=(0.0, 0.5)),  # min 24h volume-share rank (0 = off)
    ]
    # the symbol's own 15m regime label says nothing about its rank in the cross-section, and the default affinity
    # (0.6 when VOLATILE) would drop signals on exactly the volatile days and unbalance the long / short book
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 1.0}
    lookback_days = 20  # the daily ATR14 (ranks come from the market context, which has its own history)
    max_hold_days = 9  # longest time stop (72 h) x 3: the engine's cap for a winner run past its deadline

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        m = ctx.market
        L = int(p["lookback_h"])
        rank, n = m.get(f"ret_{L}h_rank", float("nan")), m.get("n", float("nan"))
        if not ok(rank, n) or n < MIN_N:
            return None  # no context, symbol outside the hour's cross-section, or too few symbols
        q = p["q"]
        if rank >= 1.0 - q - 1e-9:
            side = Side.LONG
        elif rank <= q + 1e-9:
            side = Side.SHORT
        else:
            return None
        sg = side.sign
        fund = ctx.funding_rate
        if p["fund_max"] > 0 and ok(fund) and fund * sg > p["fund_max"]:
            return None
        vmin = p["vol_rank_min"]
        if vmin > 0:
            vr = m.get("vol_share_rank", float("nan"))
            if not ok(vr) or vr <= vmin:
                return None
        d = view.tf(TF)
        close, atr = d.v("close"), d.v("atr")
        if not ok(close, atr) or close <= 0 or atr <= 0:
            return None  # warm-up: no daily ATR yet
        conf = 0.65
        if conf < p["min_conf"]:
            return None
        stop = close - sg * p["sl_atr"] * atr
        hold = int(p["hold_h"])
        up = side is Side.LONG
        reason = (f"코인 간 {L}시간 수익률 순위 {'상위' if up else '하위'} {q:.0%}(순위 {rank:.2f}, {int(n)}종목) → "
                  f"일봉 마감 시장가 {'롱' if up else '숏'} - 손절 {p['sl_atr']:.1f}일봉ATR · 목표 없음 · {hold}시간 보유"
                  f"(0.5R 이상이면 추적 손절로 연장)")
        # the hold ends one minute before a daily close so that close (the alpha's only decision) can renew it
        return Signal(alpha=self.name, symbol=ctx.symbol, side=side, confidence=conf, reason=reason, stop=stop,
                      take_profit=None, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=self.bars_to_1m(hold, "1h") - 1, trail_atr_mult=0.0, atr=atr, timeframe=TF,
                      tags={"ref_price": close, "rank": rank, "lookback_h": L, "n": int(n), "funding": fund,
                            "stop_atr": p["sl_atr"]})

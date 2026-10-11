"""A coin's 24h capitulation, bought while the market disperses (discovery_daily family, ROADMAP P2): rule rc899282d
of the multi-year discovery run, hard-coded so it can be gated, tested and switched on like any other alpha.

Rule, at every 1h close (closed bars only; features exactly as the miner computed them, discovery.decision_features):
* c.ret_24h < -6.73%: the coin fell more than 6.7% over the last 24 hours (ctx.market, the hour close's snapshot);
* 15m.hurst < 0.337: the variance-ratio trendiness of the last 100 15m bars is deep in mean-reverting territory, so
  the fall came with whipsaws (bids absorbing the selling) rather than as a one-way slide;
* m.disp_24h > 3.13%: the cross-sectional dispersion of 24h returns is high, so the fall is more idiosyncratic than a
  market-wide beta move.
LONG at market, stop sl_atr 1h ATRs away (the miner's catastrophic stop), no target, time stop after hold_h hours. The
engine closes a trade that is less than 0.5R up at the deadline and lets a winner run under a breakeven stop and a
1.5 1h-ATR trail (Signal.timeframe "1h") for at most 3x hold_h; the miner's template closed every trade at 24 hours.

Provenance (docs/results/p2_discovery_daily_*.json, trial ledger family discovery_daily): `heartless discover --grid
1h,1d --templates all --context`, 20 symbols, mined on 2022-01-13..2023-12-31 and validated on 2024 (both inside
TRAIN), 104,871 rules evaluated (bar t >= 4.81). Mined: n 172, +0.246R, day-clustered t 4.47 (market-excess t 5.18);
2024: n 168, +0.203R, t 2.68. The three rules that passed were all "long after a sharp drop" with a 24h time exit.
Why it can be real: forced sellers (liquidations, stop cascades) pay liquidity providers, and a coin-specific
capitulation whose intraday path already mean-reverts is the textbook case; the 4 ATR stop and a 24h hold make the
~13bp round trip about 1% of R.

The thresholds are the mined values (param_specs keep them tunable by the research loop); confidence, regime affinity,
warm-up and the Signal match the `discovered` alpha running this rule, so the two produce the same trades.
"""
from __future__ import annotations

from heartless.core.models import EntryStyle, Regime, Side, Signal
from heartless.data.features import MarketView
from heartless.strategy.base import Alpha, Context, ok
from heartless.strategy.spec import ParamSpec

TF = "1h"  # decision grid and stop / trail ATR timeframe (discovery.GRID_ATR_TF["1h"])
RULE_ID = "rc899282d"
FEATURES = ["c.ret_24h", "15m.hurst", "m.disp_24h"]  # the rule's conditions, in this order
CONF = 0.66  # the discovered alpha's confidence for one fired rule (0.62 + 0.04)


class DiscoveryDailyCapdisp(Alpha):
    name = "discovery_daily_capdisp"
    timeframe = TF
    enabled_by_default = False  # P2 candidate: on only after the pre-registered TRAIN and VALID gates
    description = ("discovered rule: a coin down > 6.7% in 24h with mean-reverting 15m bars while the market "
                   "disperses -> long at market, 4 1h-ATR stop, no target, 24h time stop")
    param_specs = [
        ParamSpec("ret24_max", -0.067328, -0.20, -0.02),  # c.ret_24h must be below this (24h return)
        ParamSpec("hurst_max", 0.336784, 0.20, 0.50),  # 15m.hurst must be below this (0.5 = random walk)
        ParamSpec("disp_min", 0.031298, 0.0, 0.08),  # m.disp_24h must be above this (cross-sectional std)
        ParamSpec("sl_atr", 4.0, 2.0, 6.0, 0.5),  # stop distance in 1h ATRs
        ParamSpec("hold_h", 24, 8, 48, 4, integer=True),  # time stop in hours (a winner may run 3x longer)
    ]
    # the discovered alpha's weights, kept so the candidate reproduces the evaluated trades: capitulation bars are often
    # labelled VOLATILE by the 5m regime, and 0.66 x 0.8 = 0.53 still clears the 0.50 entry threshold
    regime_affinity = {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 1.0, Regime.VOLATILE: 0.8}
    lookback_days = 22.0  # the miner's warm-up (discovery.WARMUP_MS): the same indicator history as the mined rows

    def evaluate(self, view: MarketView, ctx: Context, p: dict) -> Signal | None:
        if not view.closed(TF):
            return None
        mk = ctx.market
        r24, disp = mk.get("ret_24h", float("nan")), mk.get("disp_24h", float("nan"))
        # cheap pre-check on the context snapshot before the frame features are computed
        if not ok(r24, disp) or not (r24 < p["ret24_max"] and disp > p["disp_min"]):
            return None
        from heartless.learning.discovery import decision_features

        x = decision_features(view, ctx.extras, mk, int(view.cursor_time), FEATURES)
        r24, hurst, disp = (float(v) for v in x)
        # NaN fails every comparison, as in the miner's rule mask
        if not (r24 < p["ret24_max"] and hurst < p["hurst_max"] and disp > p["disp_min"]):
            return None
        cur = view.tf(TF)
        close, atr = cur.v("close"), cur.v("atr")
        if not ok(close, atr) or atr <= 0 or close <= 0:
            return None
        if CONF < p["min_conf"]:
            return None
        stop = close - p["sl_atr"] * atr
        hold = int(p["hold_h"])
        reason = (f"발굴 규칙 {RULE_ID}: 24시간 {r24:+.1%} 급락 · 15분봉 허스트 {hurst:.2f}(평균회귀) · 시장 분산도 "
                  f"{disp:.1%} → 시장가 롱 - 손절 {p['sl_atr']:g} 1시간ATR · 목표 없음 · {hold}시간 보유"
                  f"(0.5R 이상이면 추적 손절로 연장)")
        return Signal(alpha=self.name, symbol=ctx.symbol, side=Side.LONG, confidence=CONF, reason=reason, stop=stop,
                      take_profit=None, tp1=None, entry_style=EntryStyle.MARKET, max_hold_bars=hold * 60,
                      trail_atr_mult=0.0, atr=atr, timeframe=TF,
                      tags={"ref_price": close, "rule_id": RULE_ID, "ret_24h": r24, "hurst_15m": hurst,
                            "disp_24h": disp, "time_exit": True})

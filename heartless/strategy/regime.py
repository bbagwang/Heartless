"""Market regime classification used to weight alphas and size positions."""
from __future__ import annotations

from heartless.core.models import Regime
from heartless.data.features import MarketView
from heartless.strategy.base import ok


def detect_regime(view: MarketView) -> tuple[Regime, dict]:
    h1 = view.tf("1h")
    m15 = view.tf("15m")
    m5 = view.tf("5m")
    info: dict = {}
    if not (h1.ok and m15.ok and m5.ok):
        return Regime.RANGE, info
    adx15, chop15 = m15.v("adx"), m15.v("chop")
    ema21_1h, ema50_1h, close_1h = h1.v("ema21"), h1.v("ema50"), h1.v("close")
    slope15 = m15.v("slope20")
    atr_rank5 = m5.v("atr_rank")
    st15 = m15.v("st_dir")
    hurst15 = m15.v("hurst")
    info.update(adx15=adx15, chop15=chop15, atr_rank5=atr_rank5, slope15=slope15, hurst15=hurst15)
    # recent shock: 3-bar 5m move relative to ATR
    atr5 = m5.v("atr")
    move3 = abs(m5.v("close") - m5.v("close", 3)) / atr5 if ok(atr5) and atr5 > 0 else 0.0
    info["move3_atr"] = move3
    if ok(atr_rank5) and (atr_rank5 >= 0.96 or move3 >= 4.0):
        return Regime.VOLATILE, info
    trend_votes_up = 0
    trend_votes_dn = 0
    if ok(ema21_1h, ema50_1h, close_1h):
        if ema21_1h > ema50_1h and close_1h > ema50_1h:
            trend_votes_up += 1
        if ema21_1h < ema50_1h and close_1h < ema50_1h:
            trend_votes_dn += 1
    if ok(adx15) and adx15 >= 22:
        if ok(slope15) and slope15 > 0:
            trend_votes_up += 1
        elif ok(slope15) and slope15 < 0:
            trend_votes_dn += 1
    if ok(st15):
        if st15 > 0:
            trend_votes_up += 1
        else:
            trend_votes_dn += 1
    if ok(hurst15) and hurst15 > 0.55:
        if ok(slope15) and slope15 > 0:
            trend_votes_up += 1
        elif ok(slope15) and slope15 < 0:
            trend_votes_dn += 1
    info["votes_up"], info["votes_dn"] = trend_votes_up, trend_votes_dn
    strong_range = ok(adx15, chop15) and adx15 < 20 and chop15 > 55
    if trend_votes_up >= 3 and not strong_range:
        return Regime.TREND_UP, info
    if trend_votes_dn >= 3 and not strong_range:
        return Regime.TREND_DOWN, info
    return Regime.RANGE, info


# prior affinity of each alpha to each regime (0..1); the bandit learns the rest
REGIME_AFFINITY: dict[str, dict[Regime, float]] = {
    "trend_pullback": {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 0.55, Regime.VOLATILE: 0.4},
    "squeeze_breakout": {Regime.TREND_UP: 0.9, Regime.TREND_DOWN: 0.9, Regime.RANGE: 0.85, Regime.VOLATILE: 0.5},
    "mean_reversion": {Regime.TREND_UP: 0.35, Regime.TREND_DOWN: 0.35, Regime.RANGE: 1.0, Regime.VOLATILE: 0.6},
    "momentum_burst": {Regime.TREND_UP: 1.0, Regime.TREND_DOWN: 1.0, Regime.RANGE: 0.6, Regime.VOLATILE: 0.7},
    "funding_fade": {Regime.TREND_UP: 0.6, Regime.TREND_DOWN: 0.6, Regime.RANGE: 0.9, Regime.VOLATILE: 0.8},
    "sweep_reversal": {Regime.TREND_UP: 0.7, Regime.TREND_DOWN: 0.7, Regime.RANGE: 1.0, Regime.VOLATILE: 0.8},
}

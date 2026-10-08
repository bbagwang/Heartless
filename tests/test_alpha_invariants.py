"""Design-agnostic invariants for every registered alpha, on a real MarketView over synthetic 1m data.

Replaces the per-design regression tests of the retired 5m trend_pullback / mean_reversion logic (NaN EMAs read as a
trend, a take-profit emitted on the wrong side of entry). Whatever an alpha's setup is, every Signal it emits must be a
bracket the engine can place: stop on the protective side of the reference price, take-profit and partial on the
favourable side (partial inside the final target), a finite positive ATR, a 1m time stop and a ref_price tag. The data
starts cold, so the warm-up NaN path of every alpha is exercised as well.
"""
from __future__ import annotations

import math

import pytest
from synth import synth_candles, synth_symbols

from heartless.core.models import EntryStyle, Regime, Side, Ticker
from heartless.data.features import MarketView
from heartless.strategy.alphas import ALL_ALPHAS
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

SYM = "ETHUSDT"
DAYS = 21


@pytest.fixture(scope="module")
def signals():
    params = StrategyParams.default()
    info = synth_symbols([SYM])[SYM]
    out: dict[str, list] = {a.name: [] for a in ALL_ALPHAS}
    # a calm and a very volatile series (some alphas only trade when the 1h ATR is large)
    for seed, vol in ((3, 0.0008), (5, 0.0022)):
        ca = synth_candles(DAYS * 1440, seed=seed, price=100.0, vol=vol)
        view = MarketView(SYM)
        view.rebuild(ca, live=False)
        close_t, close = ca.view("close_time"), ca.view("close")
        for i in range(0, ca.n, 15):  # every 15m close (no registered alpha decides on a faster timeframe)
            j = min(i + 14, ca.n - 1)
            t = int(close_t[j])
            view.seek(t)
            px = float(close[j])
            tk = Ticker(SYM, bid=px * 0.99995, ask=px * 1.00005, mark=px, last=px, ts=t)
            for regime in (Regime.RANGE, Regime.TREND_UP):
                ctx = Context(symbol=SYM, info=info, ticker=tk, regime=regime, now=t, extras={})
                for a in ALL_ALPHAS:
                    if not view.closed(a.timeframe):
                        continue
                    s = a.evaluate(view, ctx, params.alphas[a.name])
                    if s is not None:
                        out[a.name].append((s, px))
    return out


def test_enough_alphas_fire_for_the_invariants_to_mean_something(signals):
    fired = {a for a, v in signals.items() if v}
    assert len(fired) >= 4, {a: len(v) for a, v in signals.items()}


def test_every_signal_is_a_valid_bracket(signals):
    for name, items in signals.items():
        for s, px in items:
            ref = s.tags.get("ref_price")
            assert s.alpha == name and s.symbol == SYM
            assert ref is not None and math.isfinite(ref) and ref > 0, (name, s.tags)
            assert abs(ref / px - 1) < 0.01, (name, ref, px)  # decided on the just-closed bar, not elsewhere
            sg = s.side.sign
            assert (ref - s.stop) * sg > 0, (name, s.side, ref, s.stop)  # protective stop
            if s.take_profit is not None:
                assert (s.take_profit - ref) * sg > 0, (name, s.side, ref, s.take_profit)
            if s.tp1 is not None:
                assert (s.tp1 - ref) * sg > 0, (name, ref, s.tp1)
                if s.take_profit is not None:
                    assert (s.take_profit - s.tp1) * sg > 0, (name, s.tp1, s.take_profit)
            if s.entry_style is EntryStyle.LIMIT:
                lp = s.limit_price
                assert lp is not None and lp > 0 and (lp - s.stop) * sg > 0, (name, lp, s.stop)
                if s.take_profit is not None:
                    assert (s.take_profit - lp) * sg > 0, (name, lp, s.take_profit)
            assert 0.0 < s.confidence <= 1.0 and s.reason
            assert math.isfinite(s.atr) and s.atr > 0
            assert s.max_hold_bars > 0 and s.timeframe in ("1m", "5m", "15m", "1h")
            assert s.side in (Side.LONG, Side.SHORT)

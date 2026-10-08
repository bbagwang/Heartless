import math

import numpy as np

from heartless.learning.metalabel import META_FEATURES, MetaLabeler, meta_vector, walk_forward


def _sample(rng, informative=True):
    feats = {n: float(rng.normal()) for n in META_FEATURES}
    feats["t.hour_utc"] = float(rng.integers(0, 24))
    side = 1 if rng.random() < 0.5 else -1
    x = meta_vector(feats, 0.7, side)
    # outcome: wins when (signed) 15m.dist_ema50 is positive, if informative; pure noise otherwise
    edge = feats["15m.dist_ema50"] * side if informative else 0.0
    win = rng.random() < 1 / (1 + math.exp(-2.5 * edge))
    return x, (1.5 if win else -1.0)


def test_neutral_until_enough_samples_then_learns_an_informative_feature():
    rng = np.random.default_rng(0)
    ml = MetaLabeler(min_samples=100)
    x, r = _sample(rng)
    assert ml.decide("a", x) == (True, 1.0, None)
    for _ in range(600):
        x, r = _sample(rng)
        ml.update("a", x, r)
    good = meta_vector({**{n: 0.0 for n in META_FEATURES}, "15m.dist_ema50": 2.0, "t.hour_utc": 3.0}, 0.7, 1)
    bad = meta_vector({**{n: 0.0 for n in META_FEATURES}, "15m.dist_ema50": -2.0, "t.hour_utc": 3.0}, 0.7, 1)
    assert ml.predict("a", good) > 0.75 and ml.predict("a", bad) < 0.25
    assert ml.decide("a", bad)[0] is False and ml.decide("a", good)[1] > 1.0


def test_walk_forward_improves_informative_and_does_not_invent_edge_on_noise():
    rng = np.random.default_rng(1)
    info = [("a",) + _sample(rng, True) for _ in range(1500)]
    res = walk_forward(info, min_samples=120)["a"]
    assert res["avg_r_kept"] > res["avg_r_all"] + 0.15
    rng = np.random.default_rng(2)
    noise = [("b",) + _sample(rng, False) for _ in range(1500)]
    res2 = walk_forward(noise, min_samples=120)["b"]
    assert abs(res2["avg_r_kept"] - res2["avg_r_all"]) < 0.12


def test_missing_values_and_persistence(tmp_path):
    from heartless.core.store import Store

    st = Store(tmp_path / "m.db")
    rng = np.random.default_rng(3)
    ml = MetaLabeler(min_samples=50, store=st)
    for _ in range(120):
        x, r = _sample(rng)
        x[0] = math.nan
        ml.update("a", x, r)
    ml._save()
    ml2 = MetaLabeler(min_samples=50, store=st)
    x, _ = _sample(rng)
    assert ml2.predict("a", x) is not None and abs(ml2.predict("a", x) - ml.predict("a", x)) < 1e-6


def test_engine_meta_veto_and_learning_path(tmp_path):
    """With META_LABEL on, the engine stores the context vector at entry and trains the model on close."""
    import asyncio

    from heartless.config import Settings
    from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, SymbolInfo, Ticker
    from heartless.data.candles import CandleArrays
    from heartless.data.features import MarketView
    from heartless.exchange.paper import PaperAccount
    from heartless.execution.engine import TradingEngine
    from heartless.strategy.base import Context
    from heartless.strategy.params import StrategyParams
    from synth import synth_candles

    info = SymbolInfo("BTCUSDT", "BTC", "USDT", 0.01, 0.001, 0.001, 5, 2, 3)
    ca = synth_candles(3000, seed=4)
    view = MarketView("BTCUSDT")
    view.rebuild(ca, live=False)
    t = int(ca.close_time[ca.n - 1])
    view.seek(t)
    px = float(ca.close[ca.n - 1])
    ml = MetaLabeler(min_samples=5)
    acc = PaperAccount("t", 10_000, slippage_bps=0, impact_bps_per_10k=0, spread_bps=0)
    eng = TradingEngine("t", acc, StrategyParams.default(), Settings(_env_file=None), {"BTCUSDT": info}, persist=False,
                        clock=lambda: t, meta=ml)

    async def one(exit_px):
        await acc.on_ticker(Ticker("BTCUSDT", bid=px, ask=px, mark=px, last=px, ts=t))
        sig = Signal("trend_pullback", "BTCUSDT", Side.LONG, 0.8, "t", px * 0.99, px * 1.02, None, EntryStyle.MARKET, None, 0, 0.0,
                     px * 0.004, "15m", {"ref_price": px})
        d = Decision("BTCUSDT", Side.LONG, 0.8, 0.8, ["trend_pullback"], sig, "t", Regime.RANGE, 1.0, 2.0)
        await eng._open(d, view, Context("BTCUSDT", info, acc.tickers["BTCUSDT"], Regime.RANGE), 10_000)
        pos = eng.positions.get("BTCUSDT")
        if pos is None:
            return None
        x = list(pos.extra["meta_x"])
        await acc.on_ticker(Ticker("BTCUSDT", bid=exit_px, ask=exit_px, mark=exit_px, last=exit_px, ts=t))
        return x

    async def run():
        xs = [await one(px * 0.985) for _ in range(5)]  # identical context, always stopped out
        assert all(x is not None and len(x) == len(xs[0]) for x in xs)
        assert len(ml.models["trend_pullback"].r) == 5 and ml.models["trend_pullback"].beta is not None
        # the model has only seen losses in this context: the next identical signal is vetoed
        assert await one(px) is None and eng.stats.skipped.get("meta veto") == 1

    asyncio.run(run())

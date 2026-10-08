"""Alpha discovery: finds a planted effect that persists, rejects pure noise, and shares code with online evaluation."""
import tempfile
from pathlib import Path

import numpy as np

from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data.candles import CandleArrays, resample
from heartless.data.features import FeatureFrame
from heartless.learning import discovery as D

DAY = 86_400_000
START = 1_767_225_600_000  # 2026-01-01 00:00 UTC


def _series(days: int, seed: int, planted: bool) -> list[Candle]:
    """1m random walk; with `planted`, every day 00:00-04:00 UTC drifts up ~1.6% (a persistent seasonal edge)."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    minute = np.arange(n) % 1440
    drift = np.where(planted & (minute < 240), 0.016 / 240, 0.0)
    ret = drift + rng.normal(0, 0.0009, n)
    close = 100 * np.exp(np.cumsum(ret))
    out = []
    prev = 100.0
    for i in range(n):
        c = float(close[i])
        h = max(prev, c) * (1 + abs(rng.normal(0, 0.0003)))
        l = min(prev, c) * (1 - abs(rng.normal(0, 0.0003)))
        v = float(abs(rng.normal(100, 20)) + 1)
        t = START + i * 60_000
        out.append(Candle(t, prev, h, l, c, v, v * c, 10, v * 0.5, t + 59_999))
        prev = c
    return out


def _store(planted: bool, seeds=(1, 2), days=150) -> str:
    path = Path(tempfile.mkdtemp()) / "d.db"
    st = Store(path)
    for i, sd in enumerate(seeds):
        st.save_candles(f"SYM{i}USDT", _series(days, sd, planted))
    st.close()
    return str(path)


def test_finds_a_persistent_planted_effect():
    db = _store(planted=True)
    train = (START + 15 * DAY, START + 105 * DAY)
    valid = (START + 105 * DAY, START + 148 * DAY)
    cfg = D.SearchConfig(min_trades=40, beam=12, depth=2, max_rules=4)
    res = D.mine(db, ["SYM0USDT", "SYM1USDT"], train, valid, "1h", cfg, workers=1)
    assert res["passed"], [r["rule"] for r in res["results"]]
    best = res["passed"][0]
    assert best["side"] == "LONG"
    assert any(c[0] == "t.hour_utc" for c in best["conds"])
    assert best["stats"]["valid"]["avg_r"] > 0.1


def test_rejects_pure_noise():
    db = _store(planted=False, seeds=(7, 8))
    train = (START + 15 * DAY, START + 105 * DAY)
    valid = (START + 105 * DAY, START + 148 * DAY)
    cfg = D.SearchConfig(min_trades=40, beam=12, depth=2, max_rules=6)
    res = D.mine(db, ["SYM0USDT", "SYM1USDT"], train, valid, "1h", cfg, workers=1)
    assert res["tested"] > 1000
    assert res["passed"] == [], "random walks must not produce validated rules"


def test_online_features_equal_batch_features_and_rule_roundtrip():
    ca = CandleArrays("1m", 40 * 1440 + 16)
    ca.extend(_series(40, 3, planted=True))
    frames = {tf: FeatureFrame(tf, resample(ca, tf)) for tf in D.DECISION_TFS}

    class V:
        symbol = "X"

    V.frames = frames
    times = frames["15m"].close_time[-200:].astype(np.int64)
    batch = D.features_at(frames, None, times)
    for j in (0, 77, 199):
        online = D.rule_features_at(V, None, int(times[j]))
        assert np.allclose(online, np.array([batch[n][j] for n in D.FEATURES]), equal_nan=True)
    rule = D.Rule(1, 1, ((D.FEATURES.index("15m.rsi14"), "<", 30.0), (D.FEATURES.index("t.hour_utc"), ">", 3.0)), "15m")
    d = rule.to_dict()
    back = D.rule_from_dict(d)
    assert back.key() == rule.key() and d["id"].startswith("r") and d["sl_atr"] == D.TEMPLATES[1][0]


def test_simulator_is_pessimistic_and_charges_costs():
    # a bar that touches both stop and target resolves as a stop
    o = np.array([100.0, 100.0, 100.0])
    h = np.array([100.0, 103.0, 100.0])
    l = np.array([100.0, 97.0, 100.0])
    c = np.array([100.0, 100.0, 100.0])
    r, ex = D.simulate_template(o, h, l, c, np.array([0]), np.array([2.0]), 1, 1.0, 1.0, 3, D.CostModel())
    assert r[0] < -0.9 and ex[0] == 1
    # flat market times out with a small loss equal to the costs
    h2 = np.array([100.01] * 3)
    l2 = np.array([99.99] * 3)
    r2, _ = D.simulate_template(o, h2, l2, c, np.array([0]), np.array([2.0]), 1, 1.0, 1.0, 3, D.CostModel())
    assert -0.2 < r2[0] < 0

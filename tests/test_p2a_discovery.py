"""P2a discovery: multi-year (symbol x year) datasets, hourly and daily grids, time-exit templates, market-context
features, and a multiple-testing bar over every rule evaluated."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C
from heartless.learning import discovery as D
from heartless.learning import seal

DAY = 86_400_000
START = 1_693_526_400_000  # 2023-09-01 00:00 UTC: the data crosses into 2024 (two dataset chunks per symbol)
SYMS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT")


def _ms(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def _market(days: int, seed: int, planted: bool) -> dict[str, list[Candle]]:
    """1m random walks for five coins. With `planted`, the coin with the worst 24h return at 00:00 UTC drifts up
    about 3% over the following day: a cross-sectional reversal only a context rank feature can see."""
    rng = np.random.default_rng(seed)
    n_sym = len(SYMS)
    rets = rng.normal(0, 0.0006, (days * 1440, n_sym))
    logp = np.zeros(n_sym)
    path = np.empty((days * 1440, n_sym))
    for d in range(days):
        block = rets[d * 1440:(d + 1) * 1440]
        if planted and d > 0:
            last24 = path[d * 1440 - 1] - path[(d - 1) * 1440 - 1] if d > 1 else path[d * 1440 - 1]
            block[:, int(np.argmin(last24))] += 0.03 / 1440
        path[d * 1440:(d + 1) * 1440] = logp + np.cumsum(block, axis=0)
        logp = path[(d + 1) * 1440 - 1]
    out = {}
    for j, sym in enumerate(SYMS):
        close = 50 * np.exp(path[:, j])
        prev = np.concatenate(([50.0], close[:-1]))
        wig = np.abs(rng.normal(0, 2e-4, (len(close), 2)))
        hi = np.maximum(prev, close) * (1 + wig[:, 0])
        lo = np.minimum(prev, close) * (1 - wig[:, 1])
        vol = np.abs(rng.normal(100, 30, len(close))) + 1
        t = START + np.arange(len(close), dtype=np.int64) * 60_000
        out[sym] = [Candle(int(t[i]), float(prev[i]), float(hi[i]), float(lo[i]), float(close[i]), float(vol[i]),
                           float(vol[i] * close[i]), 5, float(vol[i] / 2), int(t[i]) + 59_999) for i in range(len(close))]
    return out


def _db(tmp_path_factory, name: str, planted: bool, seed: int, days: int = 185) -> str:
    path = tmp_path_factory.mktemp(name) / "h.db"
    st = Store(path)
    for sym, bars in _market(days, seed, planted).items():
        st.save_candles(sym, bars)
    st.close()
    return str(path)


@pytest.fixture(scope="module")
def planted_db(tmp_path_factory):
    return _db(tmp_path_factory, "planted", True, 21)


@pytest.fixture(scope="module")
def noise_db(tmp_path_factory):
    return _db(tmp_path_factory, "noise", False, 22)


TRAIN = (START + 40 * DAY, START + 142 * DAY - 1)  # crosses 2024-01-01; >= 500 daily rows (thresholds() minimum)
VALID = (START + 142 * DAY, START + 183 * DAY - 1)
CFG = D.SearchConfig(min_trades=40, beam=10, depth=2, max_rules=4, min_sym_frac=0.0, min_month_frac=0.0)


def _ctx(db: str) -> dict:
    return {"db": db + ".ctx", "universe": list(SYMS), "build": True}


def test_finds_a_planted_cross_sectional_rule_on_the_daily_grid_with_time_exits(planted_db):
    res = D.mine(planted_db, list(SYMS), TRAIN, VALID, "1d", CFG, workers=1, templates="time", context=_ctx(planted_db))
    assert res["rows"]["train"] == 102 * len(SYMS) and res["context"] and res["n_tests"] == res["tested"] > 1000
    assert res["passed"], [r["rule"]["conds"] for r in res["results"]]
    best = res["passed"][0]
    assert best["side"] == "LONG" and best["tp_r"] is None and best["tf"] == "1d"
    assert any(c[0].startswith("c.ret_") and c[0].endswith("_rank") and c[1] == "<" for c in best["conds"]), best
    assert best["stats"]["train"]["t_ex"] >= res["t_bar"] == D.multiple_testing_t(res["tested"])
    back = D.rule_from_dict(best)
    assert D.TEMPLATES[back.tmpl][1] is None and D.TEMPLATES[back.tmpl][2] == best["hold_min"]


def test_pure_noise_yields_nothing_past_the_threshold(noise_db):
    res = D.mine(noise_db, list(SYMS), TRAIN, VALID, "1h,1d", CFG, workers=1, templates="time", context=_ctx(noise_db))
    assert res["tested"] > 5_000 and set(res["rows"]) >= {"1h", "1d"}
    assert res["passed"] == []
    assert all(r["rule"]["stats"]["train"]["t_ex"] < res["t_bar"] for r in res["results"])


def test_the_bar_counts_every_grid_and_prior_trials(monkeypatch):
    sizes = {"1h": 700, "1d": 300}
    built = []

    def fake_build(db, syms, a, b, g, workers, **k):
        built.append((g, k["templates"]))
        return SimpleNamespace(times=np.zeros(3), tf=g)

    monkeypatch.setattr(D, "build_dataset", fake_build)
    monkeypatch.setattr(D, "search", lambda ds, cfg, progress=None: ([], sizes[ds.tf]))
    res = D.mine("x.db", ["A"], TRAIN, VALID, "1h,1d", templates="all", n_prior=1000)
    assert res["tested"] == 1000 and res["n_tests"] == 2000 and res["t_bar"] == D.multiple_testing_t(2000)
    assert [g for g, _ in built] == ["1h", "1h", "1d", "1d"] and {t for _, t in built} == {"all"}
    with pytest.raises(ValueError):
        D.mine("x.db", ["A"], TRAIN, VALID, "4h")


def test_daily_grid_time_exit_simulation_and_chunked_dataset(planted_db):
    ds = D.build_dataset(planted_db, ["AAAUSDT"], TRAIN[0], TRAIN[1], "1d", workers=1, templates="all")
    assert len(ds.times) == 102 and ((ds.times + 1) % DAY == 0).all()  # 23:59:59.999 -> entry at the 00:00 open
    assert ds.templates == (0, 1, 2, 3, 4, 5) and set(ds.R) == {(t, s) for t in range(6) for s in (1, -1)}
    assert (ds.entry_t == ds.times + 1).all()
    # one dataset row per day across the 2023/2024 chunk boundary, exits simulated across it
    days = ds.times // DAY
    assert np.all(np.diff(days) == 1)
    for t, hold in ((3, 480), (4, 1440), (5, 4320)):
        for side in (1, -1):
            dur = ds.EXIT[(t, side)] - ds.entry_t
            ok = ds.EXIT[(t, side)] >= 0
            assert ok.all() and (dur <= (hold - 1) * 60_000).all() and (dur == (hold - 1) * 60_000).any()
            stopped = dur < (hold - 1) * 60_000
            r = ds.R[(t, side)]
            assert (r[stopped] < -0.95).all() and (r[stopped] > -1.2).all()  # only the catastrophic stop ends early


def test_time_exit_simulator_without_target():
    o = np.full(6, 100.0)
    c = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    h, lo = c + 0.5, c - 0.5
    r, ex = D.simulate_template(o, h, lo, c, np.array([0]), np.array([1.0]), 1, 4.0, None, 5, D.CostModel())
    assert ex[0] == 4 and r[0] > 0.8  # held to the close of the last minute, no target in the way
    lo2 = lo.copy()
    lo2[2] = 90.0
    r2, ex2 = D.simulate_template(o, h, lo2, c, np.array([0]), np.array([1.0]), 1, 4.0, None, 5, D.CostModel())
    assert ex2[0] == 2 and -1.1 < r2[0] < -0.95


def _old_take(rows, exit_idx, entry_idx, sym_id):
    """The previous greedy implementation (reference)."""
    if len(rows) == 0:
        return rows
    keep = []
    order = np.lexsort((entry_idx[rows], sym_id[rows]))
    rows = rows[order]
    syms = sym_id[rows]
    for s in np.unique(syms):
        r = rows[syms == s]
        ent = entry_idx[r]
        pos = 0
        while pos < len(r):
            keep.append(r[pos])
            nxt_entry = exit_idx[r[pos]] + 1
            pos = int(np.searchsorted(ent, nxt_entry, side="left")) if nxt_entry > ent[pos] else pos + 1
    return np.array(sorted(keep), dtype=np.int64)


def test_non_overlap_selection_matches_the_previous_greedy_rule():
    rng = np.random.default_rng(5)
    for _ in range(50):
        n = int(rng.integers(1, 400))
        sym = rng.integers(0, 4, n)
        entry = rng.integers(0, 2000, n)
        exit_ = entry + rng.integers(0, 120, n)
        rows = np.flatnonzero(rng.random(n) < 0.6)
        assert np.array_equal(D.take_non_overlapping(rows, exit_, entry, sym), _old_take(rows, exit_, entry, sym))


def test_seal_holds_for_both_windows_and_the_72h_lookahead(tmp_path, monkeypatch):
    s25 = seal.SEALED["2025"][0]
    with pytest.raises(seal.SealedError):
        D.mine(str(tmp_path / "none.db"), ["X"], (_ms(2024, 6, 1), _ms(2025, 2, 1)), VALID, "1h", templates="time")
    with pytest.raises(seal.SealedError):
        D.mine(str(tmp_path / "none.db"), ["X"], TRAIN, (_ms(2025, 6, 1), _ms(2025, 9, 1)), "1d", templates="all",
               context={"db": str(tmp_path / "c.db")})
    assert not (tmp_path / "c.db").exists()  # refused before anything is built
    db = tmp_path / "d.db"
    st = Store(db)
    t0 = _ms(2024, 11, 20)
    st.save_candles("X", [Candle(t0 + i * 60_000, 1, 1.001, 0.999, 1, 1, 1, 1, 1, t0 + i * 60_000 + 59_999)
                          for i in range(45 * 1440)])  # 2024-11-20 .. 2025-01-03
    st.close()
    ends = []
    orig = Store.load_candles
    monkeypatch.setattr(Store, "load_candles", lambda self, *a, **k: ends.append(k.get("end")) or orig(self, *a, **k))
    ds = D.build_dataset(str(db), ["X"], _ms(2024, 12, 20), s25 - 1, "1h", workers=1, templates="time")
    assert len(ds.times) > 200 and ends == [s25 - 1]  # 72h exits of late-December decisions stop at the seal


def test_context_columns_follow_the_last_hour_close(planted_db):
    st = Store(planted_db + ".ctx")
    table = C.ContextTable.load(st, ["AAAUSDT"], TRAIN[0], TRAIN[1])
    st.close()
    T = C.hour_close(C.hour_of(TRAIN[0] + 10 * DAY))
    cols = table.columns("AAAUSDT", np.array([T, T + 15 * 60_000, T + C.MS_H, TRAIN[0] - 10 * DAY]))
    v = table.view("AAAUSDT", T)
    assert cols["c.ret_24h_rank"][0] == v["ret_24h_rank"] == cols["c.ret_24h_rank"][1]  # mid-hour: last close
    assert cols["m.n"][0] == len(SYMS) and np.isnan(cols["m.n"][3])  # before the built window: unknown
    assert len(D.FEATURES_X) == len(D.FEATURES) + len(C.SYMBOL_FEATURES) + len(C.MARKET_FEATURES)

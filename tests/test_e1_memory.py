"""Backtest memory (docs/PORTFOLIO.md section 4.4): the backtester's 1m and 5m frames compute an indicator column only
when the run first reads it, and load_backtester reads a long window's candles a month at a time. Both must leave every
value, trade and equity point exactly as before (eager MarketView.rebuild over the whole history, one query)."""
from __future__ import annotations

import numpy as np
import pytest
from synth import synth_candles, synth_symbols

from heartless.config import Settings
from heartless.core.store import Store
from heartless.data.candles import COLUMNS, CandleArrays
from heartless.data.features import FeatureFrame, MarketView
from heartless.learning import backtester as B
from heartless.strategy.ensemble import ALL_ALPHAS
from heartless.strategy.params import StrategyParams

DAY = 86_400_000
T0 = 1_700_006_400_000
RAW = set(B._RAW_COLUMNS)


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def held(f) -> set[str]:
    """The columns a frame's mapping actually holds (a lazy one reports every name it can provide)."""
    return set(dict.keys(f))


def eager_views(bt: B.Backtester) -> None:
    """The views the backtester built before: MarketView.rebuild(ca, live=False), every column computed up front."""
    for sym, ca in bt.candles.items():
        v = MarketView(sym)
        v.rebuild(ca, live=False)
        bt.views[sym] = v


def test_lazy_frames_hold_the_eager_values_bit_for_bit():
    ca = synth_candles(6000, seed=3)
    bt = B.Backtester(settings(), synth_symbols(["BTCUSDT"]), {"BTCUSDT": ca})
    lazy = bt.views["BTCUSDT"]
    eager = MarketView("BTCUSDT")
    eager.rebuild(ca, live=False)
    assert list(lazy.frames) == list(eager.frames) and lazy.base is ca
    for tf in B._LAZY_TIMEFRAMES:
        assert held(lazy.frames[tf].f) == RAW  # nothing computed up front
    for tf in set(eager.frames) - set(B._LAZY_TIMEFRAMES):
        assert type(lazy.frames[tf]) is FeatureFrame  # 15m .. 1d: built as rebuild builds them
    for tf, ef in eager.frames.items():
        lf = lazy.frames[tf]
        assert lf.n == ef.n and lf.names() == ef.names() and len(lf.f) == len(ef.f)
        assert lf.close_time.tobytes() == ef.close_time.tobytes() and lf.open_time.tobytes() == ef.open_time.tobytes()
        for name in ef.names():
            assert name in lf.f
            assert lf.f[name].tobytes() == ef.f[name].tobytes(), (tf, name)


def test_unknown_columns_behave_as_in_the_eager_frame():
    bt = B.Backtester(settings(), synth_symbols(["BTCUSDT"]), {"BTCUSDT": synth_candles(3000, seed=4)})
    v = bt.views["BTCUSDT"]
    v.seek(int(bt.candles["BTCUSDT"].close_time[2999]))
    f = v.frames["5m"].f
    assert f.get("nope") is None and "nope" not in f
    with pytest.raises(KeyError):
        f["nope"]
    assert np.isnan(v.tf("5m").v("nope")) and v.tf("5m").v("adx") == v.tf("5m").v("adx")
    assert held(f) == RAW | {"adx"}  # an unknown name computes nothing


def test_reads_of_one_bar_share_one_recompute_and_other_symbols_keep_what_was_read(monkeypatch):
    calls = []
    orig = FeatureFrame._compute

    def counting(self, b):
        calls.append(self.tf)
        return orig(self, b)

    syms = ["AAAUSDT", "BBBUSDT"]
    bt = B.Backtester(settings(), synth_symbols(syms), {s: synth_candles(3000, seed=i) for i, s in enumerate(syms)})
    monkeypatch.setattr(FeatureFrame, "_compute", counting)  # _LazyFrame overrides it: only recomputes are counted
    fa = bt.views["AAAUSDT"].frames["5m"].f
    for name in ("adx", "chop", "hurst"):
        fa[name]
    assert calls == ["5m"] and held(fa) == RAW | {"adx", "chop", "hurst"}
    bt._lazy.release()  # the bar is done: the spare computation is dropped
    assert fa.spare is None and not bt._lazy.spared
    fa.get("slope20")
    assert calls == ["5m", "5m"]
    fb = bt.views["BBBUSDT"].frames["5m"].f
    fb["adx"]
    assert calls == ["5m"] * 3 and held(fb) == RAW | {"adx", "chop", "hurst", "slope20"}
    bt._lazy.release()
    # a frame read widely stops being lazy: after _KEEP_ALL_AFTER recomputes the next one keeps every column
    fa.get("ema9")
    bt._lazy.release()
    assert calls == ["5m"] * 4 and fa.computes == 3 and held(fa) == set(fa.names)
    fa.get("ema21")
    assert calls == ["5m"] * 4
    bt._lazy.release()
    assert all(c.spare is None for v in bt.views.values() for c in (fr.f for fr in v.frames.values())
               if isinstance(c, B._LazyColumns))


def _all_alphas() -> StrategyParams:
    p = StrategyParams.default()
    for a in ALL_ALPHAS:
        p.enabled[a.name] = True
    return p


@pytest.mark.parametrize("allocator", [False, True])
def test_backtests_are_identical_to_the_eager_frames(allocator):
    """Every registered alpha on, three symbols, FIFO and the P3 allocator: trades, equity curve, refusals and the
    final equity equal a run on eagerly built views."""
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    params = _all_alphas()

    def run(eager: bool):
        candles = {s: synth_candles(9000, seed=20 + i, price=100.0 * (i + 1)) for i, s in enumerate(syms)}
        bt = B.Backtester(settings(PORTFOLIO_ALLOCATOR=allocator), synth_symbols(syms), candles)
        if eager:
            eager_views(bt)
        return bt, bt.run(params)

    bt_lazy, lazy = run(False)
    _, eager = run(True)

    def strip(trades):
        return [{k: v for k, v in t.items() if k != "position_id"} for t in trades]  # ids are random

    assert lazy.stats["n"] >= 5
    assert strip(lazy.trades) == strip(eager.trades)
    assert lazy.equity_curve == eager.equity_curve and lazy.final_equity == eager.final_equity
    assert lazy.skipped == eager.skipped
    # the run kept only the columns something read, and no spare computation outlives it
    assert not bt_lazy._lazy.spared
    for v in bt_lazy.views.values():
        for tf in B._LAZY_TIMEFRAMES:
            f = v.frames[tf].f
            assert held(f) <= RAW | bt_lazy._lazy.reads.get(tf, set()) and f.spare is None
    assert len(held(bt_lazy.views["BTCUSDT"].frames["1m"].f) - RAW) < 10


def test_a_long_window_is_read_a_month_at_a_time_into_the_same_arrays(tmp_path, monkeypatch):
    st = Store(tmp_path / "c.db")
    src = synth_candles(5 * 1440, seed=9, start_ms=T0)
    st.save_candles("BTCUSDT", [src.candle_at(i) for i in range(src.n)])
    lo, until = T0 + 3600_000 * 7, T0 + 4 * DAY + 3600_000 * 5 - 1
    rows = st.load_candles("BTCUSDT", start=lo, end=until)
    ref = CandleArrays("1m", capacity=len(rows) + 16)
    ref.extend(rows)
    seen = []
    orig = Store.load_candles

    def spy(self, symbol, start=None, end=None, limit=None):
        seen.append((start, end))
        return orig(self, symbol, start, end, limit)

    monkeypatch.setattr(Store, "load_candles", spy)
    monkeypatch.setattr(B, "_LOAD_CHUNK_MS", DAY)  # a "month" of one day: four pieces over 3 days 22 hours
    ca = B._load_1m(st, "BTCUSDT", lo, until)
    assert len(seen) == 4 and seen[0][0] == lo and seen[-1][1] == until
    assert all(b + 1 == a for (_, b), (a, _) in zip(seen, seen[1:]))  # contiguous, no overlap
    assert ca.n == ref.n and ca._cap == ref._cap
    for name in COLUMNS:
        assert ca.view(name).tobytes() == ref.view(name).tobytes()
    # a window shorter than a piece and an open-ended one stay one query
    seen.clear()
    B._load_1m(st, "BTCUSDT", lo, lo + DAY - 1)
    B._load_1m(st, "BTCUSDT", lo, None)
    assert seen == [(lo, lo + DAY - 1), (lo, None)]
    # and load_backtester builds the same backtest from it
    bt = B.load_backtester(settings(HEARTLESS_DATA_DIR=str(tmp_path)), st, synth_symbols(["BTCUSDT"]), lo, until,
                           market=None)
    assert bt.candles["BTCUSDT"].view("close").tobytes() == ref.view("close").tobytes()
    st.close()


def test_the_timeline_union_equals_unique_of_the_concatenation():
    rng = np.random.default_rng(5)
    arrays = [np.unique(rng.integers(0, 10_000, n)).astype(np.int64) * 60_000 + 59_999 for n in (3000, 1, 0, 5000)]
    ref = np.unique(np.concatenate(arrays))
    out = B._union_sorted(arrays)
    assert out.dtype == ref.dtype and out.tobytes() == ref.tobytes()
    assert B._union_sorted([]).size == 0 and B._union_sorted(arrays[:1]).tobytes() == arrays[0].tobytes()

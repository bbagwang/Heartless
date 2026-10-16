"""F7: incremental feature frames are bit-identical to the batch computation.

* IncrementalFrame: after any number of appended bars every column equals FeatureFrame(tf, bars since the anchor), bit
  for bit (same NaN positions, same IEEE-754 values), on random series with NaNs, constant runs, zero volume, session
  gaps and very short histories, including the warm-up boundaries (EMA200 seed, RSI, ADX, MACD signal, Hurst's
  "n + q bars" rule).
* LiveFrames / MarketView(incremental=True), driven like the app (bars appended one by one, holes, the daily trim):
  the 4h / 1d frames equal the batch path's lazily rebuilt frames at every minute; the sliding frames equal the batch
  over their own bars, which are exactly resample()'s bars; a re-anchor checks the frame against the batch and lands on
  what app._rebuild_view builds; edits rebuild; a corrupted frame is caught and the view falls back to batch rebuilds.
"""
import logging
import time

import numpy as np
import pytest

from heartless.core.models import Candle
from heartless.data import incremental as INC
from heartless.data.candles import CandleArrays, resample
from heartless.data.features import LIVE_WINDOW, LONG_TIMEFRAMES, TIMEFRAMES, FeatureFrame, MarketView
from heartless.data.incremental import KEYS, IncrementalFrame, LiveFrames
from heartless.util.timeutil import MS_DAY, MS_MINUTE, TF_MS

START = 1_704_067_200_000  # 2024-01-01 00:00 UTC

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _series(n: int, seed: int, tf: str = "1m", nan_frac: float = 0.0, const_runs: bool = False,
            zero_vol: float = 0.05, start: int = START + 7 * MS_MINUTE, skip: float = 0.0) -> CandleArrays:
    """Random OHLCV bars; optional NaNs in every price/volume column, flat stretches, zero volume and skipped periods."""
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    if const_runs:
        for s in rng.integers(0, n, 6):
            c[s:s + rng.integers(5, 80)] = c[s]
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.001, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.001, n)))
    if const_runs:
        flat = c == np.r_[np.nan, c[:-1]]
        h[flat], lo[flat], o[flat] = c[flat], c[flat], c[flat]
    v = rng.lognormal(3, 1, n)
    v[rng.random(n) < zero_vol] = 0.0
    tb = v * rng.uniform(0, 1, n)
    step = TF_MS[tf]
    gaps = np.cumsum(1 + (rng.random(n) < skip) * rng.integers(1, 2000, n))
    b = CandleArrays(tf, n + 16)
    for k, arr in (("open", o), ("high", h), ("low", lo), ("close", c), ("volume", v), ("quote_volume", v * c),
                   ("taker_buy", tb)):
        getattr(b, k)[:n] = arr
    b.open_time[:n] = start + (gaps - 1) * step
    b.close_time[:n] = b.open_time[:n] + step - 1
    if nan_frac:
        for k in ("open", "high", "low", "close", "volume", "taker_buy"):
            arr = getattr(b, k)[:n]
            arr[rng.random(n) < nan_frac] = np.nan
    b.n = n
    return b


def _same(a, b) -> list[str]:
    assert a.n == b.n and np.array_equal(a.close_time, b.close_time) and np.array_equal(a.open_time, b.open_time)
    assert list(a.f) == list(b.f)
    return [k for k in b.f if not np.array_equal(a.f[k], b.f[k], equal_nan=True)]


def _append(fr: IncrementalFrame, b: CandleArrays, k: int) -> None:
    fr.append(int(b.open_time[k]), float(b.open[k]), float(b.high[k]), float(b.low[k]), float(b.close[k]),
              float(b.volume[k]), float(b.quote_volume[k]), float(b.taker_buy[k]), int(b.close_time[k]))


CASES = [  # (anchor bars, total bars, nan fraction, constant runs, skipped periods, timeframe)
    (300, 700, 0.0, False, 0.0, "1m"),
    (0, 420, 0.0, False, 0.0, "1m"),
    (1, 260, 0.01, True, 0.0, "1m"),
    (150, 900, 0.0, True, 0.01, "1m"),
    (5, 140, 0.03, False, 0.0, "1m"),
    (230, 520, 0.002, True, 0.0, "5m"),
    (50, 400, 0.0, False, 0.02, "1h"),
    (99, 330, 0.0, True, 0.0, "1d"),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_every_appended_row_equals_the_batch_over_the_bars_since_the_anchor(case):
    n0, n1, nan, flat, skip, tf = CASES[case]
    b = _series(n1, 100 + case, tf=tf, nan_frac=nan, const_runs=flat, skip=skip)
    fr = IncrementalFrame(tf, b.slice(0, n0), extra=7)  # small extra: the buffers grow on the way
    assert not _same(fr, FeatureFrame(tf, b.slice(0, n0)))
    checks = 0
    for k in range(n0, n1):
        _append(fr, b, k)
        if k % 23 == 0 or k in (n0, 8, 13, 19, 25, 49, 99, 104, 105, 199, 200, n1 - 1):
            assert _same(fr, FeatureFrame(tf, b.slice(0, k + 1))) == [], (case, k)
            checks += 1
    assert checks > 5 and set(fr.f) == set(KEYS)


def test_only_the_hurst_back_fill_rewrites_rows_of_a_short_frame():
    """The documented exception (IncrementalFrame docstring, red-team F-L12): a frame built from 100 bars publishes
    "hurst" rows 99..103 as NaN; the append that reaches 105 rows fills them (to the batch values) in the shared buffer.
    Nothing else that was handed out changes."""
    b = _series(130, 9)
    fr = IncrementalFrame("1m", b.slice(0, 100), extra=64)  # no reallocation on the way, as live frames are sized
    snaps = []
    for k in range(100, 130):
        if fr.n in (100, 103):
            snaps.append(({c: v.copy() for c, v in fr.f.items()}, dict(fr.f)))
        _append(fr, b, k)
    assert len(snaps) == 2
    touched: set[int] = set()
    for held, views in snaps:
        for c, v in views.items():
            if c != "hurst":
                assert np.array_equal(v, held[c], equal_nan=True), c
                continue
            diff = np.flatnonzero(~((v == held[c]) | (np.isnan(v) & np.isnan(held[c]))))
            assert np.all(np.isnan(held[c][diff])) and np.all(np.isfinite(v[diff]))  # NaN before, the batch value now
            touched |= set(int(x) for x in diff)
    assert touched and touched <= set(range(99, 104))
    assert _same(fr, FeatureFrame("1m", b.slice(0, 130))) == []


def test_snapshots_handed_out_earlier_do_not_change():
    b = _series(400, 7)
    fr = IncrementalFrame("1m", b.slice(0, 200), extra=4)
    held = {k: v.copy() for k, v in fr.f.items()}
    views = dict(fr.f)
    for k in range(200, 400):  # appends and buffer growth
        _append(fr, b, k)
    for k, v in views.items():
        assert np.array_equal(v, held[k], equal_nan=True)


def test_the_column_list_is_the_feature_frames():
    b = _series(30, 3)
    assert tuple(FeatureFrame("1m", b).f) == KEYS
    empty = IncrementalFrame("1m", b.slice(0, 0))
    assert list(empty.f) == ["open", "high", "low", "close", "volume", "quote_volume"] == list(
        FeatureFrame("1m", b.slice(0, 0)).f)


# --- the live frame set --------------------------------------------------------------------------------------------
def _candles(n: int, seed: int, start: int = START, holes=()) -> list[Candle]:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    keep = np.ones(n, dtype=bool)
    for a, k in holes:
        keep[a:a + k] = False
    out, prev = [], float(close[0])
    for i in np.flatnonzero(keep):
        c = float(close[i])
        v = float(abs(rng.normal(50, 15)) + 1) if rng.random() > 0.02 else 0.0
        t = start + int(i) * MS_MINUTE
        out.append(Candle(t, prev, max(prev, c) * 1.0003, min(prev, c) * 0.9997, c, v, v * c, 4, v * 0.5,
                          t + 59_999))
        prev = c
    return out


@pytest.fixture
def small_windows(monkeypatch):
    """Shorter live windows and re-anchor periods, so a test covers several re-anchors in a few simulated days."""
    for tf, n in (("1m", 300), ("5m", 160), ("15m", 90), ("1h", 60)):
        monkeypatch.setitem(LIVE_WINDOW, tf, n)
    for tf, n in (("1m", 97), ("5m", 41), ("15m", 13), ("1h", 7)):
        monkeypatch.setitem(INC.REANCHOR_ROWS, tf, n)


def _ref_bars(ca: CandleArrays, tf: str, first_open: int) -> CandleArrays:
    full = ca if tf == "1m" else resample(ca, tf)
    j = int(np.searchsorted(full.open_time[:full.n], first_open))
    return full.slice(j, full.n)


def _check_sliding(view: MarketView, ca: CandleArrays) -> None:
    for tf in TIMEFRAMES:
        fr = view.frames[tf]
        assert isinstance(fr, IncrementalFrame)
        bars = fr.bars()
        if fr.n:
            ref = _ref_bars(ca, tf, int(fr.open_time[0]))
            assert ref.n == bars.n, tf
            for col in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume",
                        "taker_buy"):
                assert np.array_equal(ref.view(col), bars.view(col)), (tf, col)
        assert _same(fr, FeatureFrame(tf, bars)) == [], tf


def _drive(candles, n0: int, keep_days: int, check_every: int = 37):
    """Feed candles[n0:] one by one like app._process_bar (append, daily trim at the first bar of a UTC day, advance,
    seek) into an incremental view, with a batch-path view as the reference for the 4h / 1d frames."""
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:n0])
    inc = MarketView("BTCUSDT", incremental=True)
    inc.rebuild(ca, live=True)
    ref = MarketView("BTCUSDT")
    ref.rebuild(ca, live=True)
    for tf in TIMEFRAMES:
        assert _same(inc.frames[tf], ref.frames[tf]) == [], tf  # the build is the batch rebuild
    incremental = 0
    for i, c in enumerate(candles[n0:]):
        ca.append(c)
        if c.open_time % MS_DAY == 0:
            ca.trim_before(c.open_time - keep_days * MS_DAY)
        incremental += inc.advance(ca)
        ref.base = ca
        inc.seek(c.close_time)
        ref.seek(c.close_time)
        for tf in LONG_TIMEFRAMES:
            a, b = inc.tf(tf), ref.tf(tf)
            assert inc.closed(tf) == ref.closed(tf) and a.idx == b.idx, (tf, c.close_time)
            if b.frame.n:
                assert _same(a.frame, b.frame) == [], (tf, c.close_time)
        for tf in TIMEFRAMES:
            assert inc.closed(tf) == (tf in inc.just_closed)
        if i % check_every == 0:
            _check_sliding(inc, ca)
    _check_sliding(inc, ca)
    return inc, ca, incremental


def test_live_frames_follow_the_base_with_holes_trims_and_reanchors(small_windows):
    # (1440 - 13, 1): the day-1 00:00 bar is missing, so the trim at day 4 leaves the base starting at 00:01, inside a
    # 4h / 1d period: the long frames take _rebuild_long's re-aggregation branch (red-team F-L8)
    candles = _candles(7 * 1440 + 333, 11, start=START + 13 * MS_MINUTE,
                       holes=[(1440 - 13, 1), (3 * 1440 + 3, 1), (4 * 1440 - 1, 1), (5 * 1440 + 100, 300),
                              (6 * 1440 + 239, 2)])
    inc, ca, steps = _drive(candles, 3 * 1440, keep_days=3)  # the app keeps more history than any live window
    st = inc.live.stats
    assert inc.live.failed == "" and ca.trims >= 3
    assert st["reanchors"] >= 40 and st["verified"] == st["reanchors"] and st["long_rebuilds"] >= 4
    assert st["long_offset_rebuilds"] >= 2  # 4h and 1d after the day-4 trim
    assert steps == len(candles) - 3 * 1440 and st["builds"] == 1


def test_an_edit_after_a_trim_keeps_the_long_frames_until_their_next_close(small_windows):
    """Red-team F-L5: a trim at 00:00, then a revised bar at 00:06 forces a build. The batch path keeps its 4h / 1d
    frames until their next close; the incremental path used to re-anchor them on the trimmed base at once (1d: 25 rows
    against the batch's 28). Compared at every minute by _drive's reference view."""
    candles = _candles(6 * 1440 + 400, 21, start=START)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    n0 = 3 * 1440
    ca.extend(candles[:n0])
    inc = MarketView("BTCUSDT", incremental=True)
    inc.rebuild(ca, live=True)
    ref = MarketView("BTCUSDT")
    ref.rebuild(ca, live=True)
    edits = {4 * 1440 + 6, 4 * 1440 + 200, 5 * 1440 + 3}  # minutes after the day-4 and day-5 trims
    for i in range(n0, len(candles)):
        c = candles[i]
        ca.append(c)
        if c.open_time % MS_DAY == 0:
            ca.trim_before(c.open_time - 2 * MS_DAY)
        if i in edits:  # the exchange revises the last bar (or a late duplicate with other values arrives)
            ca.append(Candle(c.open_time, c.open, c.high * 1.002, c.low, c.close, c.volume, c.quote_volume, c.trades,
                             c.taker_buy_volume, c.close_time))
        inc.advance(ca)
        ref.base = ca
        inc.seek(c.close_time)
        ref.seek(c.close_time)
        for tf in LONG_TIMEFRAMES:
            a, b = inc.tf(tf), ref.tf(tf)
            assert inc.closed(tf) == ref.closed(tf) and a.idx == b.idx, (tf, i)
            if b.frame.n:
                assert a.frame.n == b.frame.n and _same(a.frame, b.frame) == [], (tf, i)
    st = inc.live.stats
    assert ca.edits == len(edits) and st["builds"] == 1 + len(edits) and st["long_kept"] >= 2 * len(edits) - 1


def test_a_reanchor_lands_on_the_per_bar_rebuild(small_windows):
    from heartless.app import Heartless

    candles = _candles(3 * 1440, 12)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:1440])
    v = MarketView("ETHUSDT", incremental=True)
    v.rebuild(ca, live=True)
    seen = 0
    for c in candles[1440:]:
        ca.append(c)
        before = v.live.stats["reanchors"]
        v.advance(ca)
        if v.live.stats["reanchors"] > before:
            legacy = MarketView("ETHUSDT")
            Heartless._rebuild_view(legacy, ca, list(TIMEFRAMES))
            fresh = [tf for tf in TIMEFRAMES if v.frames[tf].rows_appended == 0]
            assert fresh
            for tf in fresh:
                assert _same(v.frames[tf], legacy.frames[tf]) == [], tf
            seen += 1
    assert seen >= 10


def test_edits_rebuild_and_duplicates_do_not(small_windows):
    candles = _candles(2 * 1440, 13)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:1500])
    v = MarketView("X", incremental=True)
    v.rebuild(ca, live=True)
    ca.append(candles[1500])
    assert v.advance(ca)
    ca.append(candles[1500])  # the same bar again (a duplicate message): not an edit
    assert ca.edits == 0 and v.advance(ca) and v.live.stats["builds"] == 1
    c = candles[1500]
    ca.append(Candle(c.open_time, c.open, c.high * 1.01, c.low, c.close, c.volume, c.quote_volume, c.trades,
                     c.taker_buy_volume, c.close_time))  # a revised bar
    assert ca.edits == 1 and not v.advance_cheap(ca)
    assert not v.advance(ca)  # full build
    ref = MarketView("X")
    ref.rebuild(ca, live=True)
    for tf in TIMEFRAMES:
        assert _same(v.frames[tf], ref.frames[tf]) == [], tf
    later = candles[1600]
    ca.append(later)
    old = candles[1550]
    ca.append(old)  # out of order: inserted before the last bar
    assert ca.edits == 2 and not v.advance(ca)
    _check_sliding(v, ca)


def test_cheap_is_false_when_a_reanchor_or_a_long_rebuild_is_due(small_windows):
    candles = _candles(3 * 1440, 14)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:1440])
    v = MarketView("X", incremental=True)
    v.rebuild(ca, live=True)
    for c in candles[1440:]:
        ca.append(c)
        if c.open_time % MS_DAY == 0:
            ca.trim_before(c.open_time - MS_DAY)
        lf = v.live
        want = {tf: lf.frames[tf].rows_appended for tf in INC.REANCHOR_ROWS}
        stats = dict(lf.stats)
        cheap = v.advance_cheap(ca)
        v.advance(ca)
        heavy = (lf.stats["reanchors"] > stats["reanchors"] or lf.stats["long_rebuilds"] > stats["long_rebuilds"])
        if heavy:
            assert not cheap, c.open_time
        if cheap:
            assert not heavy and all(lf.frames[tf].rows_appended >= want[tf] for tf in want)


def test_a_diverging_frame_is_caught_and_the_view_falls_back_to_batch(small_windows, caplog):
    candles = _candles(2 * 1440, 15)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:1440])
    v = MarketView("X", incremental=True)
    v.rebuild(ca, live=True)
    ca.append(candles[1440])
    v.advance(ca)
    v.frames["1m"]._buf["ema21"][v.frames["1m"].n - 1] += 1e-9  # a silent bug
    with caplog.at_level(logging.WARNING, logger="heartless.data.incremental"):
        for c in candles[1441:1441 + 200]:
            ca.append(c)
            v.advance(ca)
            if v.live is None:
                break
    assert v.live is None and not v.incremental
    assert any("diverged" in r.message for r in caplog.records)
    ref = MarketView("X")
    ref.rebuild(ca, live=True)
    for tf in TIMEFRAMES:
        assert _same(v.frames[tf], ref.frames[tf]) == [], tf
    ca.append(candles[1641])
    assert not v.advance(ca) and not v.advance_cheap(ca)  # batch from now on
    v.seek(candles[1641].close_time)
    assert v.tf("4h").ok


def test_an_empty_history_gets_its_frames_from_streamed_bars(small_windows):
    candles = _candles(3 * 1440, 16)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    v = MarketView("X", incremental=True)
    v.rebuild(ca, live=True)
    assert not set(LONG_TIMEFRAMES) & set(v.frames)
    ref = MarketView("X")
    ref.rebuild(ca, live=True)
    for c in candles:
        ca.append(c)
        v.advance(ca)
        ref.base = ca
        v.seek(c.close_time)
        ref.seek(c.close_time)
        for tf in LONG_TIMEFRAMES:
            assert v.closed(tf) == ref.closed(tf)
            if ref.closed(tf):
                assert _same(v.tf(tf).frame, ref.tf(tf).frame) == []
    _check_sliding(v, ca)
    assert v.ready()


def test_the_incremental_step_is_much_cheaper_than_the_batch_rebuild():
    from heartless.app import Heartless

    candles = _candles(3 * 1440, 17)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:2 * 1440])
    v = MarketView("X", incremental=True)
    v.rebuild(ca, live=True)
    legacy = MarketView("X")
    legacy.rebuild(ca, live=True)
    t_inc, t_batch = [], []
    for c in candles[2 * 1440:2 * 1440 + 200]:
        ca.append(c)
        cheap = v.advance_cheap(ca)
        t0 = time.perf_counter()
        v.advance(ca)
        if cheap:
            t_inc.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        Heartless._rebuild_view(legacy, ca, ["1m"])
        t_batch.append(time.perf_counter() - t0)
    assert len(t_inc) > 150
    assert np.median(t_inc) * 10 < np.median(t_batch)


def test_an_unexpected_failure_in_a_step_rebuilds_instead_of_losing_the_bar(small_windows, monkeypatch, caplog):
    candles = _candles(2 * 1440, 18)
    ca = CandleArrays("1m", capacity=len(candles) + 16)
    ca.extend(candles[:1440 + 3])
    v = MarketView("X", incremental=True)
    v.rebuild(ca, live=True)
    real = INC.resample
    calls = {"n": 0}

    def flaky(base, tf, only_complete=True):
        calls["n"] += 1
        if calls["n"] == 1:
            raise MemoryError("simulated")
        return real(base, tf, only_complete)

    monkeypatch.setattr(INC, "resample", flaky)
    with caplog.at_level(logging.WARNING, logger="heartless.data.incremental"):
        for c in candles[1443:1443 + 10]:  # the 5m bar closing in here fails once
            ca.append(c)
            v.advance(ca)
    assert calls["n"] > 1 and v.live is not None and v.live.stats["builds"] == 2
    assert any("step failed" in r.message for r in caplog.records)
    ref = MarketView("X")
    ref.rebuild(ca, live=True)
    _check_sliding(v, ca)
    assert v.frames["5m"].close_time[-1] == ref.frames["5m"].close_time[-1]

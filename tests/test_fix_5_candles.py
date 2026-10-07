"""Regression tests for heartless/data/candles.py resample() (fix round 5).

1. A finished higher-TF group whose *last* base bar is missing must still be emitted (only the
   trailing group may be dropped as incomplete).
2. The output must be sized from the number of groups, not n // bars_per, so a sparse 1m
   series (e.g. only the last minute of each 5m period) does not overflow the buffer.
"""
import numpy as np

from heartless.core.models import Candle
from heartless.data.candles import CandleArrays, resample
from heartless.data.features import TIMEFRAMES
from heartless.util.timeutil import TF_MS

MS_MINUTE = 60_000


def _bar(minute: int) -> Candle:
    ot = minute * MS_MINUTE
    return Candle(ot, 100 + minute, 101 + minute, 99 + minute, 100.5 + minute, 10, 1000, 5, 6, ot + MS_MINUTE - 1)


def _series(minutes) -> CandleArrays:
    ca = CandleArrays("1m")
    for m in minutes:
        ca.append(_bar(m))
    return ca


def test_interior_group_missing_last_minute_is_kept():
    # minutes 0-29 with minute 4 absent: 5m group 0 is finished (groups 1..5 follow) but lacks its
    # final base bar -> must still be emitted.
    ca = _series(m for m in range(30) if m != 4)
    r = resample(ca, "5m")
    assert len(r) == 6
    assert list(r.open_time[:6]) == [g * 5 * MS_MINUTE for g in range(6)]
    assert r.close_time[0] == 5 * MS_MINUTE - 1
    # aggregation over the 4 present bars of group 0 (minutes 0,1,2,3)
    assert r.open[0] == 100 and r.close[0] == 103.5
    assert r.high[0] == 104 and r.low[0] == 99 and r.volume[0] == 40


def test_interior_gaps_do_not_change_group_set_regardless_of_position():
    # whichever minute of a finished period is missing (first, middle, last), the set of emitted
    # 5m groups is identical.
    expected = [g * 5 * MS_MINUTE for g in range(6)]
    for missing in (0, 2, 4, 9, 14, 20):
        r = resample(_series(m for m in range(30) if m != missing), "5m")
        assert list(r.open_time[: len(r)]) == expected, missing


def test_trailing_incomplete_group_still_dropped():
    # minutes 0-28 (minute 29 absent): trailing group 5 is still open -> dropped; groups 0..4 kept.
    r = resample(_series(range(29)), "5m")
    assert len(r) == 5
    assert r.close_time[4] == 25 * MS_MINUTE - 1
    # minute 29 present closes group 5 even if minute 25-28 are missing.
    r2 = resample(_series(list(range(25)) + [29]), "5m")
    assert len(r2) == 6
    # a single still-open group yields an empty result
    assert len(resample(_series(range(3)), "5m")) == 0
    # existing contract: 23 consecutive bars -> 4 complete 5m groups
    r3 = resample(_series(range(23)), "5m")
    assert len(r3) == 4 and r3.close_time[0] == 299999


def test_sparse_base_does_not_overflow_output_capacity():
    # 100 one-minute bars, each the last minute of a distinct 5m group: 100 complete groups from
    # n // bars_per == 20 base-bar budget used to raise ValueError.
    ca = _series(5 * g + 4 for g in range(100))
    r = resample(ca, "5m")
    assert len(r) == 100
    assert all(int(r.close_time[g]) == (5 * g + 5) * MS_MINUTE - 1 for g in range(100))
    assert all(int(r.open_time[g]) == 5 * g * MS_MINUTE for g in range(100))
    assert r.volume[0] == 10 and r.trades[0] == 5


def test_sparse_base_every_timeframe_succeeds():
    # 21 days of 1m bars with a large share of random minutes dropped: resample must succeed for
    # every configured timeframe and emit one bar per finished period that has any base bar.
    rng = np.random.default_rng(5)
    total = 21 * 24 * 60
    keep = rng.random(total) > 0.6
    keep[-1] = True  # make the final minute present so the last groups are closed
    minutes = np.flatnonzero(keep)
    ca = _series(int(m) for m in minutes)
    for tf in TIMEFRAMES:
        if tf == "1m":
            continue
        r = resample(ca, tf)
        groups = np.unique(minutes * MS_MINUTE // TF_MS[tf])
        assert len(r) == len(groups), tf
        assert np.array_equal(r.open_time[: len(r)], groups * TF_MS[tf]), tf
        assert np.all(np.diff(r.open_time[: len(r)]) > 0), tf

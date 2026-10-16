"""F-pre-B vwap_z fix (F0 finding, docs/UNIVERSE.md 5): features.vwap_z is NaN on every bar whose UTC session holds a
single bar with volume (where the old (close - vwap) / vwap_sd was +-1e5 or NaN by rounding) and the old expression
bit for bit everywhere else; the feature frames' columns are untouched, and the batch and incremental frames agree."""
import numpy as np

from heartless.data import indicators as I
from heartless.data.candles import CandleArrays, resample
from heartless.data.features import FeatureFrame, session_volume_bars, vwap_z
from heartless.data.incremental import IncrementalFrame
from synth import synth_candles

D = 86_400_000


def _old(fr) -> np.ndarray:
    """The expression discovery used (discovery._safe_div(close - vwap, vwap_sd))."""
    f = fr.f
    a, b = f["close"] - f["vwap"], f["vwap_sd"]
    out = np.full(len(a), np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def test_session_volume_bars():
    ot = np.array([0, 1, 2, D, D + 1, D + 2, 2 * D], dtype=np.int64)
    v = np.array([0.0, 5.0, 1.0, 2.0, 0.0, 3.0, 0.0])
    assert session_volume_bars(ot, v).tolist() == [0, 1, 2, 1, 1, 2, 0]


def test_vwap_z_fixes_single_bar_sessions_and_keeps_every_other_bar():
    for price in (1234.5678, 0.0123457, 98765.4):
        ca = synth_candles(3 * 1440 + 200, seed=int(price) % 97, price=price)
        for tf in ("15m", "1h"):
            fr = FeatureFrame(tf, resample(ca, tf))
            new, old = vwap_z(fr), _old(fr)
            single = session_volume_bars(fr.open_time, fr.f["volume"]) == 1
            assert single.sum() >= 3  # the first bar of every day
            assert np.isnan(new[single]).all()
            same = ~single
            assert np.array_equal(new[same], old[same], equal_nan=True)  # bit for bit elsewhere
            # at the single bars the old value was rounding noise: huge or undefined
            assert (~np.isfinite(old[single]) | (np.abs(old[single]) > 1e3)).all()
        # the batch indicator is unchanged: vwap_sd is still session_vwap's
        fr = FeatureFrame("1h", resample(ca, "1h"))
        vw, sd = I.session_vwap(fr.open_time, fr.f["high"], fr.f["low"], fr.f["close"], fr.f["volume"])
        assert np.array_equal(fr.f["vwap_sd"], sd, equal_nan=True) and "vwap_z" not in fr.f
    # at chosen bars (-1 = none), as discovery gathers them
    idx = np.array([-1, 0, 5, 30])
    got = vwap_z(fr, idx)
    assert np.isnan(got[0]) and np.array_equal(got[1:], vwap_z(fr)[idx[1:]], equal_nan=True)


def test_batch_and_incremental_frames_agree():
    ca = synth_candles(2 * 1440 + 300, seed=4, price=50.123)
    head = CandleArrays("1m", capacity=ca.n + 16)
    k0 = 1440 + 100
    for i in range(k0):
        head.append(ca.candle_at(i))
    inc = IncrementalFrame("1m", head, extra=ca.n)
    for i in range(k0, ca.n):
        c = ca.candle_at(i)
        inc.append(c.open_time, c.open, c.high, c.low, c.close, c.volume, c.quote_volume, c.taker_buy_volume,
                   c.close_time)
    batch = FeatureFrame("1m", ca)
    assert inc.n == batch.n
    assert np.array_equal(vwap_z(inc), vwap_z(batch), equal_nan=True)

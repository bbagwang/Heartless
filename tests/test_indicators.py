import numpy as np

from heartless.data import indicators as I
from heartless.data.candles import CandleArrays, resample
from heartless.core.models import Candle


def test_sma_ema_rsi_bounds():
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 1000))
    assert abs(I.sma(c, 20)[-1] - c[-20:].mean()) < 1e-9
    e = I.ema(c, 21)
    assert np.isnan(e[:20]).all() and not np.isnan(e[-1])
    r = I.rsi(c, 14)
    assert 0 <= np.nanmin(r) and np.nanmax(r) <= 100


def test_atr_positive_and_causal():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 500))
    h, l = c + 1, c - 1
    a = I.atr(h, l, c, 14)
    assert np.nanmin(a) > 0
    # causality: changing the last bar must not change earlier values
    a2 = I.atr(np.append(h[:-1], h[-1] + 50), l, c, 14)
    assert np.allclose(a[:-1], a2[:-1], equal_nan=True)


def test_supertrend_direction_values():
    rng = np.random.default_rng(2)
    c = 100 + np.cumsum(rng.normal(0.05, 0.5, 400))
    line, d = I.supertrend(c + 0.5, c - 0.5, c)
    assert set(np.unique(d[50:])).issubset({-1, 1})


def test_resample_aggregates_correctly():
    ca = CandleArrays("1m")
    for i in range(23):
        ca.append(Candle(i * 60000, 100 + i, 101 + i, 99 + i, 100.5 + i, 10, 1000, 5, 6, i * 60000 + 59999))
    r = resample(ca, "5m")
    assert len(r) == 4
    assert r.open[0] == 100 and r.high[0] == 105 and r.low[0] == 99 and r.close[0] == 104.5 and r.volume[0] == 50
    assert r.close_time[0] == 299999


def test_candle_arrays_out_of_order_insert_and_replace():
    ca = CandleArrays("1m")
    ca.append(Candle(120000, 1, 1, 1, 1, 1, 1, 1, 1, 179999))
    ca.append(Candle(0, 1, 1, 1, 1, 1, 1, 1, 1, 59999))
    ca.append(Candle(60000, 1, 1, 1, 1, 1, 1, 1, 1, 119999))
    assert list(ca.view("open_time")) == [0, 60000, 120000]
    assert ca.append(Candle(60000, 2, 2, 2, 2, 1, 1, 1, 1, 119999)) is False
    assert ca.close[1] == 2

"""P2a: the faster indicator code is bit-identical to the original implementations.

The reference functions below are verbatim copies of the pre-P2a versions (numpy-scalar loops, unblocked sliding
windows). Every optimised function must reproduce them bit for bit (same NaN positions, same IEEE-754 bits elsewhere)
on random series with NaNs, leading NaNs, constant runs, zero volume and very short inputs, also when the sliding-window
reductions are split into tiny blocks."""
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from heartless.core.models import Candle
from heartless.data import indicators as I
from heartless.data.candles import CandleArrays, resample
from heartless.data.features import FeatureFrame, MarketView


# --- reference implementations (pre-P2a) ---------------------------------------------------------------------------
def ref_ema(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n or n <= 0:
        return out
    alpha = 2.0 / (n + 1)
    seed = np.nanmean(x[:n])
    out[n - 1] = seed
    prev = seed
    for i in range(n, len(x)):
        v = x[i]
        if np.isnan(v):
            out[i] = prev
            continue
        prev = prev + alpha * (v - prev)
        out[i] = prev
    return out


def ref_wilder(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n or n <= 0:
        return out
    first = np.nanmean(x[:n])
    out[n - 1] = first
    prev = first
    a = 1.0 / n
    for i in range(n, len(x)):
        v = x[i]
        if np.isnan(v):
            v = 0.0
        prev = prev + a * (v - prev)
        out[i] = prev
    return out


def ref_supertrend(h, l, c, n=10, mult=3.0):
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    a = I.wilder(I.true_range(h, l, c), n)
    hl2 = (h + l) / 2
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    size = len(c)
    line = np.full(size, np.nan)
    direction = np.zeros(size, dtype=int)
    fub = ub.copy()
    flb = lb.copy()
    start = int(np.argmax(~np.isnan(a))) if (~np.isnan(a)).any() else size
    for i in range(start + 1, size):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        prev_dir = direction[i - 1] if direction[i - 1] != 0 else 1
        if prev_dir == 1:
            direction[i] = -1 if c[i] < flb[i] else 1
        else:
            direction[i] = 1 if c[i] > fub[i] else -1
        line[i] = flb[i] if direction[i] == 1 else fub[i]
    return line, direction


def ref_session_vwap(open_time, h, l, c, v, session_ms=86_400_000):
    tp = (np.asarray(h, float) + np.asarray(l, float) + np.asarray(c, float)) / 3
    v = np.asarray(v, float)
    sess = (np.asarray(open_time) // session_ms)
    n = len(tp)
    vwap = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    cum_pv = cum_v = cum_pv2 = 0.0
    cur = None
    for i in range(n):
        if sess[i] != cur:
            cur = sess[i]
            cum_pv = cum_v = cum_pv2 = 0.0
        cum_pv += tp[i] * v[i]
        cum_pv2 += tp[i] * tp[i] * v[i]
        cum_v += v[i]
        if cum_v > 0:
            m = cum_pv / cum_v
            vwap[i] = m
            var = max(cum_pv2 / cum_v - m * m, 0.0)
            sd[i] = np.sqrt(var)
    return vwap, sd


def ref_rolling_std(x, n):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n or n <= 1:
        return out
    out[n - 1:] = sliding_window_view(x, n).std(axis=1, ddof=0)
    return out


def ref_percentile_rank(x, n=200):
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = sliding_window_view(x, n)
    out[n - 1:] = (w <= w[:, -1][:, None]).mean(axis=1)
    return out


def ref_linreg_slope(x, n=20):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    w = sliding_window_view(x, n)
    t = np.arange(n) - (n - 1) / 2
    denom = (t ** 2).sum()
    slope = (w * t).sum(axis=1) / denom
    level = w.mean(axis=1)
    ok = level != 0
    res = np.full(len(slope), np.nan)
    res[ok] = slope[ok] / level[ok]
    out[n - 1:] = res
    return out


def ref_hurst_proxy(c, n=100):
    c = np.asarray(c, float)
    out = np.full(len(c), np.nan)
    q = 5
    if len(c) < n + q or n <= q:
        return out
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.nan)
    lr[0] = 0.0
    w1 = sliding_window_view(lr, n)
    mu = w1.mean(axis=1)
    var1 = w1.var(axis=1, ddof=1)
    agg = I.rolling_sum(lr, q)
    agg[np.isnan(agg)] = 0.0
    wq = sliding_window_view(agg, n)[:, q - 1:]
    m = q * (n - q + 1) * (1.0 - q / n)
    varq = ((wq - q * mu[:, None]) ** 2).sum(axis=1) / m
    vr = np.divide(varq, var1, out=np.ones(len(var1)), where=var1 > 0)
    out[n - 1:] = 0.5 + 0.5 * np.tanh(np.log(np.maximum(vr, 1e-9)))
    return out


def ref_squeeze_bars(sq):
    cnt = np.zeros(len(sq))
    run = 0
    for i in range(len(sq)):
        run = run + 1 if sq[i] else 0
        cnt[i] = run
    return cnt


# --- helpers ---------------------------------------------------------------------------------------------------------
def assert_bits(a, b, what=""):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    if a.dtype.kind != "f":
        assert np.array_equal(a, b), what
        return
    na, nb = np.isnan(a), np.isnan(b)
    assert np.array_equal(na, nb), f"{what}: NaN positions differ"
    assert np.array_equal(a[~na].view(np.int64), b[~nb].view(np.int64)), f"{what}: values differ in the last bits"


def series(rng, n: int, kind: str) -> np.ndarray:
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    if kind == "nans" and n:
        x[rng.random(n) < 0.05] = np.nan
    elif kind == "leading" and n:
        x[: min(n, int(rng.integers(1, 40)))] = np.nan
    elif kind == "constant" and n:
        a = int(rng.integers(0, n))
        x[a: a + int(rng.integers(1, 60))] = x[a]
    elif kind == "allnan":
        x[:] = np.nan
    return x


KINDS = ("plain", "nans", "leading", "constant", "allnan")
LENGTHS = (0, 1, 2, 5, 13, 14, 15, 99, 100, 104, 105, 106, 199, 200, 201, 777)


@pytest.fixture(params=[I.BLOCK, 7, 1])
def block(request, monkeypatch):
    monkeypatch.setattr(I, "BLOCK", request.param)
    return request.param


# --- tests -------------------------------------------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_recursive_filters_are_bit_identical():
    rng = np.random.default_rng(11)
    for kind in KINDS:
        for n in LENGTHS:
            x = series(rng, n, kind)
            for span in (1, 2, 9, 14, 21, 200):
                assert_bits(I.ema(x, span), ref_ema(x, span), f"ema {kind} n={n} span={span}")
                assert_bits(I.wilder(x, span), ref_wilder(x, span), f"wilder {kind} n={n} span={span}")
            assert_bits(I.ema(x, 0), ref_ema(x, 0))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_supertrend_and_session_vwap_are_bit_identical():
    rng = np.random.default_rng(12)
    for kind in KINDS:
        for n in LENGTHS:
            if n == 0:
                continue
            c = series(rng, n, kind)
            h = c * (1 + np.abs(rng.normal(0, 0.003, n)))
            l = c * (1 - np.abs(rng.normal(0, 0.003, n)))
            for got, want, name in zip(I.supertrend(h, l, c), ref_supertrend(h, l, c), ("line", "dir")):
                assert_bits(got, want, f"supertrend.{name} {kind} n={n}")
            ot = 1_767_225_600_000 + np.arange(n, dtype=np.int64) * 3_600_000 // 7  # several UTC sessions
            v = np.abs(rng.normal(10, 5, n))
            v[rng.random(n) < 0.2] = 0.0  # zero-volume bars (cum_v stays 0 at a session start)
            for got, want, name in zip(I.session_vwap(ot, h, l, c, v), ref_session_vwap(ot, h, l, c, v), ("vwap", "sd")):
                assert_bits(got, want, f"session_vwap.{name} {kind} n={n}")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_blocked_sliding_windows_are_bit_identical(block):
    rng = np.random.default_rng(13)
    for kind in KINDS:
        for n in LENGTHS:
            x = series(rng, n, kind)
            for w in (2, 20, 120):
                assert_bits(I.rolling_std(x, w), ref_rolling_std(x, w), f"rolling_std {kind} n={n} w={w} block={block}")
                assert_bits(I.linreg_slope(x, w), ref_linreg_slope(x, w), f"linreg {kind} n={n} w={w}")
            z = np.nan_to_num(x, nan=0.0)
            for w in (1, 120, 200):
                assert_bits(I.percentile_rank(z, w), ref_percentile_rank(z, w), f"percentile_rank {kind} n={n} w={w}")
            if kind not in ("nans", "allnan"):
                assert_bits(I.hurst_proxy(x, 100), ref_hurst_proxy(x, 100), f"hurst {kind} n={n} block={block}")
                assert_bits(I.hurst_proxy(x, 20), ref_hurst_proxy(x, 20), f"hurst20 {kind} n={n} block={block}")


def test_squeeze_bar_count_matches_the_loop():
    rng = np.random.default_rng(14)
    for n in (0, 1, 2, 50, 400):
        for p in (0.0, 0.3, 0.8, 1.0):
            sq = rng.random(n) < p
            pos = np.arange(n)
            last_off = np.maximum.accumulate(np.where(sq, -1, pos)) if n else pos
            got = np.where(sq, pos - last_off, 0).astype(float)
            assert_bits(got, ref_squeeze_bars(sq), f"squeeze n={n} p={p}")


def _candles(n: int, seed: int) -> CandleArrays:
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    ca = CandleArrays("1m", n + 16)
    prev = 50.0
    for i in range(n):
        c = float(close[i])
        h = max(prev, c) * (1 + abs(rng.normal(0, 4e-4)))
        lo = min(prev, c) * (1 - abs(rng.normal(0, 4e-4)))
        v = 0.0 if i % 211 == 0 else float(abs(rng.normal(80, 30)))
        t = 1_767_225_600_000 + i * 60_000 + (7_200_000 if i > n // 2 else 0)  # one 2h hole
        ca.append(Candle(t, prev, h, lo, c, v, v * c, 3, v * rng.random(), t + 59_999))
        prev = c
    return ca


def test_feature_frames_and_seek_match_a_reference_frame(monkeypatch):
    """A whole multi-timeframe view computed with the optimised code equals one computed with the reference
    indicators patched in (squeeze_bars excepted, compared to its loop above), and seek() picks the same bars as a
    numpy searchsorted."""
    ca = _candles(6000, 5)
    fast = MarketView("X")
    fast.rebuild(ca, live=False)
    for name, fn in (("ema", ref_ema), ("wilder", ref_wilder), ("supertrend", ref_supertrend),
                     ("session_vwap", ref_session_vwap), ("rolling_std", ref_rolling_std),
                     ("percentile_rank", ref_percentile_rank), ("linreg_slope", ref_linreg_slope),
                     ("hurst_proxy", ref_hurst_proxy)):
        monkeypatch.setattr(I, name, fn)
    slow = MarketView("X")
    slow.rebuild(ca, live=False)
    monkeypatch.undo()
    for tf in fast.frames:
        a, b = fast.frames[tf].f, slow.frames[tf].f
        assert set(a) == set(b)
        for k in a:
            assert_bits(a[k], b[k], f"{tf}.{k}")
    ct = ca.view("close_time")
    for t in (int(ct[0]) - 1, int(ct[0]), int(ct[100]) + 5, int(ct[3000]), int(ct[-1]), int(ct[-1]) + 10**9):
        fast.seek(t)
        for tf, fr in fast.frames.items():
            assert fast.tf(tf).idx == int(np.searchsorted(fr.close_time, t, side="right")) - 1
            idx = fast.tf(tf).idx
            assert (tf in fast.just_closed) == (idx >= 0 and int(fr.close_time[idx]) > t - 60_000)


def test_live_window_frame_matches_on_resampled_bars():
    ca = _candles(2500, 6)
    for tf in ("5m", "15m", "1h"):
        bars = resample(ca, tf)
        fr = FeatureFrame(tf, bars)
        assert fr.close_times() == bars.view("close_time").tolist()

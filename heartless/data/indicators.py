"""Vectorised, strictly causal technical indicators on numpy arrays.

Every function returns arrays aligned with the input; warm-up values are NaN.
Nothing here looks at future bars, so the same code is safe for backtests and live trading.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

NAN = np.nan


def _nan(n: int) -> np.ndarray:
    return np.full(n, NAN, dtype=float)


def ema(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n or n <= 0:
        return out
    alpha = 2.0 / (n + 1)
    # seed with SMA of first n values
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


def sma(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n or n <= 0:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n or n <= 1:
        return out
    w = sliding_window_view(x, n)
    out[n - 1:] = w.std(axis=1, ddof=0)
    return out


def rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n or n <= 0:
        return out
    out[n - 1:] = sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n or n <= 0:
        return out
    out[n - 1:] = sliding_window_view(x, n).min(axis=1)
    return out


def rolling_sum(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n or n <= 0:
        return out
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = c[n:] - c[:-n]
    return out


def shift(x: np.ndarray, k: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if k <= 0:
        return x.copy()
    if k < len(x):
        out[k:] = x[:-k]
    return out


def wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder's smoothing (RMA)."""
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
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


def true_range(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    pc = shift(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    return wilder(true_range(h, l, c), n)


def rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    out = _nan(len(c))
    if len(c) <= n:
        return out
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = wilder(up[1:], n)
    ad = wilder(dn[1:], n)
    rs = np.divide(au, ad, out=np.full_like(au, np.inf), where=ad > 0)
    r = 100 - 100 / (1 + rs)
    r[np.isnan(au)] = NAN
    out[1:] = r
    return out


def bollinger(c: np.ndarray, n: int = 20, k: float = 2.0):
    mid = sma(c, n)
    sd = rolling_std(c, n)
    upper = mid + k * sd
    lower = mid - k * sd
    width = np.divide(upper - lower, mid, out=_nan(len(c)), where=mid != 0)
    return mid, upper, lower, width


def keltner(h, l, c, n: int = 20, mult: float = 1.5, atr_n: int | None = None):
    mid = ema(c, n)
    a = atr(h, l, c, atr_n or n)
    return mid, mid + mult * a, mid - mult * a


def macd(c, fast: int = 12, slow: int = 26, signal: int = 9):
    m = ema(c, fast) - ema(c, slow)
    valid = ~np.isnan(m)
    s = _nan(len(c))
    if valid.sum() >= signal:
        idx = np.where(valid)[0]
        s[idx] = ema(m[idx], signal)
    return m, s, m - s


def adx(h, l, c, n: int = 14):
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)
    atr_ = wilder(tr, n)
    pdi = 100 * np.divide(wilder(plus_dm, n), atr_, out=_nan(len(h)), where=atr_ > 0)
    mdi = 100 * np.divide(wilder(minus_dm, n), atr_, out=_nan(len(h)), where=atr_ > 0)
    denom = pdi + mdi
    dx = 100 * np.divide(np.abs(pdi - mdi), denom, out=np.zeros(len(h)), where=denom > 0)
    dx[np.isnan(pdi)] = NAN
    valid = ~np.isnan(dx)
    out = _nan(len(h))
    if valid.sum() >= n:
        idx = np.where(valid)[0]
        out[idx] = wilder(dx[idx], n)
    return out, pdi, mdi


def supertrend(h, l, c, n: int = 10, mult: float = 3.0):
    """Returns (line, direction) with direction +1 for up-trend, -1 for down-trend."""
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    c = np.asarray(c, float)
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    size = len(c)
    line = _nan(size)
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


def donchian(h, l, n: int = 20, exclude_current: bool = True):
    hh = rolling_max(h, n)
    ll = rolling_min(l, n)
    if exclude_current:
        hh = shift(hh, 1)
        ll = shift(ll, 1)
    return hh, ll


def choppiness(h, l, c, n: int = 14) -> np.ndarray:
    tr = true_range(h, l, c)
    s = rolling_sum(tr, n)
    rng = rolling_max(h, n) - rolling_min(l, n)
    out = _nan(len(c))
    ok = (~np.isnan(s)) & (rng > 0) & (s > 0)
    out[ok] = 100 * np.log10(s[ok] / rng[ok]) / np.log10(n)
    return out


def zscore(x: np.ndarray, n: int = 50) -> np.ndarray:
    m = sma(x, n)
    s = rolling_std(x, n)
    out = _nan(len(x))
    ok = (~np.isnan(s)) & (s > 0)
    out[ok] = (np.asarray(x, float)[ok] - m[ok]) / s[ok]
    return out


def percentile_rank(x: np.ndarray, n: int = 200) -> np.ndarray:
    """Fraction of the previous n values that are <= current value (0..1)."""
    x = np.asarray(x, dtype=float)
    out = _nan(len(x))
    if len(x) < n:
        return out
    w = sliding_window_view(x, n)
    cur = w[:, -1][:, None]
    out[n - 1:] = (w <= cur).mean(axis=1)
    return out


def session_vwap(open_time: np.ndarray, h, l, c, v, session_ms: int = 86_400_000):
    """VWAP anchored to the UTC session start plus +/-1 and +/-2 std bands."""
    tp = (np.asarray(h, float) + np.asarray(l, float) + np.asarray(c, float)) / 3
    v = np.asarray(v, float)
    sess = (np.asarray(open_time) // session_ms)
    n = len(tp)
    vwap = _nan(n)
    sd = _nan(n)
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


def cvd_proxy(volume: np.ndarray, taker_buy: np.ndarray, n: int = 20) -> np.ndarray:
    """Rolling net taker flow (buy - sell) over n bars normalised by total volume (-1..1)."""
    volume = np.asarray(volume, float)
    taker_buy = np.asarray(taker_buy, float)
    delta = 2 * taker_buy - volume
    num = rolling_sum(delta, n)
    den = rolling_sum(volume, n)
    out = _nan(len(volume))
    ok = (~np.isnan(den)) & (den > 0)
    out[ok] = num[ok] / den[ok]
    return out


def linreg_slope(x: np.ndarray, n: int = 20) -> np.ndarray:
    """Slope of a least-squares line over the last n values, normalised by the mean level."""
    x = np.asarray(x, float)
    out = _nan(len(x))
    if len(x) < n:
        return out
    w = sliding_window_view(x, n)
    t = np.arange(n) - (n - 1) / 2
    denom = (t ** 2).sum()
    slope = (w * t).sum(axis=1) / denom
    level = w.mean(axis=1)
    ok = level != 0
    res = np.full(len(slope), NAN)
    res[ok] = slope[ok] / level[ok]
    out[n - 1:] = res
    return out


def hurst_proxy(c: np.ndarray, n: int = 100) -> np.ndarray:
    """Cheap variance-ratio based trendiness estimate in [0,1]; >0.5 trending, <0.5 mean-reverting."""
    c = np.asarray(c, float)
    out = _nan(len(c))
    q = 5
    if len(c) < n + q or n <= q:
        return out
    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.nan)
    lr[0] = 0.0
    w1 = sliding_window_view(lr, n)
    mu = w1.mean(axis=1)
    var1 = w1.var(axis=1, ddof=1)
    # q-bar aggregated returns; keep only the n-q+1 overlapping sums that lie fully inside each window
    agg = rolling_sum(lr, q)
    agg[np.isnan(agg)] = 0.0
    wq = sliding_window_view(agg, n)[:, q - 1:]
    # Lo-MacKinlay (1988) small-sample variance-ratio estimator. The plain sample variance of overlapping
    # q-sums is biased low (E[log VR] ~ -0.07 for i.i.d. returns at n=100), which pushed the whole estimate
    # below 0.5 on driftless noise; demeaning by q*mu and dividing by m centres the null at ~0.5.
    m = q * (n - q + 1) * (1.0 - q / n)
    varq = ((wq - q * mu[:, None]) ** 2).sum(axis=1) / m
    vr = np.divide(varq, var1, out=np.ones(len(var1)), where=var1 > 0)
    h = 0.5 + 0.5 * np.tanh(np.log(np.maximum(vr, 1e-9)))  # map VR=1 -> 0.5
    out[n - 1:] = h
    return out

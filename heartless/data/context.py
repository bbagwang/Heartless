"""Market context: hourly cross-sectional features over a fixed universe, computed identically offline and online.

What a snapshot is
* Decision time T is an hour close: T = (h + 1) * 1h - 1 ms for the UTC hour h, i.e. the close_time of the hour's
  last 1m bar. The snapshot for T uses only 1m bars with close_time <= T, funding settlements with funding_time <= T
  and 5m metrics rows usable by T (bucket start + 5 min <= T, as in data/extras.py). Nothing later is read.
* Per symbol of the context universe (CONTEXT_UNIVERSE, default the 20 long-history symbols): returns over
  1/4/24/72/168 h with their cross-sectional ranks and z-scores, 24h quote-volume share and its rank, the settled
  funding rate with rank and z, the 24h open-interest change with rank, the distance from the 24h/168h high and low,
  and whether the close is above its 1h EMA-50 / EMA-200. Market-wide: BTC returns over 1/4/24/168 h and BTC trend
  state, breadth (share of symbols above their EMA-50 / EMA-200), dispersion of 24h returns, median funding, aggregate
  open-interest change and the number of symbols in the cross-section.
* A symbol enters the hour's cross-section when it has a 1m bar inside that hour (others are excluded); a feature
  whose history is too short is NaN and that symbol is left out of the feature's ranks. Ranks and z-scores need at
  least MIN_XS symbols. Ranks are mid-ranks scaled to [0, 1] (0 = lowest); z-scores use the population sd.
* F0 (CONTEXT_TOP_N): with top_n > 0 only the top_n of those symbols with the largest trailing 24h quote volume at
  that hour (24 h of history needed; ties by symbol name) enter the cross-section, so it follows liquidity instead of
  a fixed list. BTC's market-wide features are computed whether or not BTC is a member. top_n = 0 (the default) is the
  fixed-list mode, bit for bit as before.

Why the definitions look the way they do
* Every value depends only on a bounded trailing window (at most GRID_HOURS hourly rows, closes and funding rates
  carried forward at most CARRY_HOURS), so the result does not depend on how much history a process has seen: the
  offline build can be cut into yearly chunks, and the online tracker agrees with it once it holds WARMUP_HOURS of
  history. That is why the EMAs are computed over a fixed trailing window of 4x their span (seeded with the simple
  mean of its first `span` closes) instead of from the beginning of time.
* Offline and online share one implementation (`compute()`): it is vectorised over a leading time axis, the offline
  build calls it for all hours of a year at once and the tracker for one hour. It only uses IEEE-exact operations
  (+ - * / sqrt, comparisons, min/max) and sums across symbols and hours in a fixed sequential order, so both paths
  produce the same bits. Symbols are kept in sorted order, so the result depends on the set of symbols present, not on
  how the universe was listed.

Offline: `build()` (CLI `heartless context build`) writes the market_context table, resumable per calendar year,
into its own SQLite file (CONTEXT_DB, default <data_dir>/context.db; never the candle database). That file records in
its "context.meta" which candle database it was computed from (path and a fingerprint of every universe symbol's
candle / funding / metrics range), the universe and the hour ranges built. A build for another source or universe is
refused until rebuilt explicitly (`--rebuild`); history appended to the same source only invalidates the hours it can
change. Builds hold an inter-process file lock (<CONTEXT_DB>.lock) and readers a shared one, so concurrent lab
workers, discovery runs or agents never build the same years twice or read a table being rewritten.
Readers (lab workers, the backtester, discovery) use `ContextTable`. Online: `MarketContextTracker` (see its
docstring for what the app must call). Alphas read the result as `ctx.market`, a read-only flat mapping of the
symbol's own features plus the market-wide ones and "ts" (empty when no context is available).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - not POSIX: builds run unlocked
    fcntl = None

from heartless.data.extras import AVAILABLE_AFTER_MS

MS_MIN = 60_000
MS_H = 3_600_000
VERSION = "ctx1"  # bump when a definition changes: stored rows of another version are rebuilt

CORE_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
                "BCHUSDT", "BNBUSDT", "SUIUSDT")
EXTRA_SYMBOLS = ("DOTUSDT", "ATOMUSDT", "NEARUSDT", "FILUSDT", "ETCUSDT", "TRXUSDT", "UNIUSDT", "AAVEUSDT")
DEFAULT_UNIVERSE = CORE_SYMBOLS + EXTRA_SYMBOLS
BTC = "BTCUSDT"
MARKET_KEY = "*"  # symbol column of the market-wide rows

RET_HOURS = (1, 4, 24, 72, 168)
BTC_RET_HOURS = (1, 4, 24, 168)
EMA_WINDOWS = ((50, 200), (200, 800))  # (span, trailing window in hours)
GRID_HOURS = 800  # hourly rows one snapshot reads (the longest EMA window)
CARRY_HOURS = 24  # a close / funding rate is carried forward at most this long through a feed gap
WARMUP_HOURS = GRID_HOURS + CARRY_HOURS + 2  # history before the first snapshot needed for exact values
MIN_XS = 3  # minimum symbols for a rank, z-score or market-wide statistic
METRICS_STALE_MS = 30 * MS_MIN  # as MetricsSeries.snapshot: an older metrics row is no information
OI_LOOKBACK_MS = 24 * MS_H

SYMBOL_FEATURES = tuple(
    [f"ret_{k}h" for k in RET_HOURS] + [f"ret_{k}h_rank" for k in RET_HOURS] + [f"ret_{k}h_z" for k in RET_HOURS]
    + ["vol_share_24h", "vol_share_rank", "funding", "funding_rank", "funding_z", "oi_chg_24h", "oi_chg_rank",
       "dist_hi_24h", "dist_lo_24h", "dist_hi_168h", "dist_lo_168h", "above_ema50", "above_ema200"])
MARKET_FEATURES = ("btc_ret_1h", "btc_ret_4h", "btc_ret_24h", "btc_ret_168h", "btc_trend", "breadth_ema50",
                   "breadth_ema200", "disp_24h", "funding_median", "oi_chg_24h_agg", "n")
GRID_FIELDS = ("present", "close", "high", "low", "qvol", "funding", "oi_chg", "oiv_now", "oiv_prev")
EMPTY: Mapping = MappingProxyType({})
DB_NAME = "context.db"  # default CONTEXT_DB file name, next to the candle database


class ContextMismatch(RuntimeError):
    """The context store belongs to another source or universe (or would be the candle database itself)."""


def hour_of(t: int) -> int:
    """UTC hour index of the last hour close at or before t (hour h closes at (h + 1) * 1h - 1)."""
    return (int(t) + 1) // MS_H - 1


def hour_close(h: int) -> int:
    return (int(h) + 1) * MS_H - 1


def universe_list(text: str | None) -> list[str]:
    syms = [s.strip().upper() for s in (text or "").split(",") if s.strip()]
    return sorted(set(syms)) if syms else sorted(DEFAULT_UNIVERSE)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# --- shared feature core (offline: many hours at once, online: one hour) ----------------------------------------------
def _lag(a: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """Rows idx - k of a (NaN where idx - k < 0); a slice when idx is contiguous, a gather otherwise (same values)."""
    n = len(idx)
    out = np.full((n,) + a.shape[1:], np.nan)
    if n == 0:
        return out
    lo, hi = int(idx[0]) - k, int(idx[-1]) - k
    if hi - lo + 1 == n and (n == 1 or bool(np.all(np.diff(idx) == 1))):
        if hi >= 0:
            s = max(lo, 0)
            out[s - lo:] = a[s:hi + 1]
        return out
    j = idx - k
    ok = j >= 0
    out[ok] = a[j[ok]]
    return out


def _ratio(num: np.ndarray, den: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """num / den - 1 where ok and both are finite and den > 0, else NaN."""
    out = np.full(num.shape, np.nan)
    m = ok & np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[m] = num[m] / den[m] - 1.0
    return out


def _seq_sum(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sum over the symbol axis of x where v, in fixed symbol order (no pairwise summation)."""
    s = np.zeros(x.shape[0])
    for j in range(x.shape[1]):
        s = s + np.where(v[:, j], x[:, j], 0.0)
    return s


def _mean_sd(x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = v.sum(axis=1)
    mean = _seq_sum(x, v) / np.maximum(n, 1)
    d = x - mean[:, None]
    sd = np.sqrt(_seq_sum(d * d, v) / np.maximum(n, 1))
    return n, mean, sd


def _rank(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    xv = np.where(v, x, np.nan)
    less = (xv[:, None, :] < xv[:, :, None]).sum(axis=2)
    eq = (xv[:, None, :] == xv[:, :, None]).sum(axis=2)
    n = v.sum(axis=1)[:, None]
    out = np.full(x.shape, np.nan)
    ok = v & (n >= MIN_XS)
    r = (less + 0.5 * (eq - 1)) / np.maximum(n - 1, 1)
    out[ok] = r[ok]
    return out


def _z(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    n, mean, sd = _mean_sd(x, v)
    out = np.full(x.shape, np.nan)
    ok = v & ((n >= MIN_XS) & (sd > 0))[:, None]
    z = (x - mean[:, None]) / np.where(sd > 0, sd, 1.0)[:, None]
    out[ok] = z[ok]
    return out


def _median(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    srt = np.sort(np.where(v, x, np.nan), axis=1)  # NaN sorts last
    n = v.sum(axis=1)
    out = np.full(len(n), np.nan)
    rows = np.arange(len(n))
    mid = n // 2
    odd = (n >= MIN_XS) & (n % 2 == 1)
    even = (n >= MIN_XS) & (n % 2 == 0)
    out[odd] = srt[rows[odd], mid[odd]]
    out[even] = (srt[rows[even], mid[even] - 1] + srt[rows[even], mid[even]]) / 2.0
    return out


def _share(v: np.ndarray, flag: np.ndarray) -> np.ndarray:
    n = v.sum(axis=1)
    hits = (v & (flag == 1.0)).sum(axis=1)
    out = np.full(len(n), np.nan)
    ok = n >= MIN_XS
    out[ok] = hits[ok] / n[ok]
    return out


def _vol24(grid: dict[str, np.ndarray], idx: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Trailing 24h quote volume at the rows idx (oldest hour first), NaN where `ok` (24 h of history) is False."""
    vol = np.zeros((len(idx), grid["close"].shape[1]))
    for k in range(23, -1, -1):  # oldest hour first
        vol = vol + _lag(grid["qvol"], idx, k)
    return np.where(ok, vol, np.nan)


def _top_mask(vol: np.ndarray, top_n: int) -> np.ndarray:
    """Per row, the top_n symbols by `vol` (finite values only; equal volumes go to the lower column, i.e. the name)."""
    # A stable sort of -vol per row puts equal volumes in column order, so a symbol's position is the number of larger
    # volumes plus the equal ones in lower columns: O(S log S) per row instead of S x S comparison matrices (a year of
    # 120 symbols took 300 MB and 4 s that way, quadratic in the universe this mode exists to widen).
    v = np.where(np.isfinite(vol), vol, -np.inf)
    order = np.argsort(-v, axis=1, kind="stable")
    pos = np.empty_like(order)
    np.put_along_axis(pos, order, np.broadcast_to(np.arange(v.shape[1]), v.shape), axis=1)
    return np.isfinite(vol) & (pos < int(top_n))


def members(grid: dict[str, np.ndarray], idx, top_n: int = 0, member_mask: np.ndarray | None = None) -> np.ndarray:
    """[len(idx), symbols] bool: the hour's cross-section (present; with top_n > 0 only the top_n by 24h volume;
    with `member_mask` (point-in-time research stores, pit_members) present and in that mask)."""
    idx = np.asarray(idx, dtype=np.int64)
    pres = grid["present"][idx].astype(bool)
    if member_mask is not None:
        return pres & np.asarray(member_mask, dtype=bool)
    if not top_n or top_n <= 0:
        return pres
    with np.errstate(all="ignore"):
        ok = pres & np.isfinite(_lag(grid["close"], idx, 23))
        return pres & _top_mask(_vol24(grid, idx, ok), top_n)


def compute(grid: dict[str, np.ndarray], idx, btc_col: int = -1, top_n: int = 0,
            member_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot features at the grid rows `idx` (hour indices into the grid, each needs up to GRID_HOURS - 1 rows of
    history before it). grid: GRID_FIELDS -> [rows, symbols] arrays on a contiguous hour axis. Returns
    (symbol features [len(idx), symbols, len(SYMBOL_FEATURES)], market features [len(idx), len(MARKET_FEATURES)]).
    top_n > 0 restricts the cross-section to `members()` (the symbol features of the others are NaN); `member_mask`
    ([len(idx), symbols] bool, the point-in-time selection of pit_members) restricts it the same way to the given
    members instead (top_n is then not used)."""
    idx = np.asarray(idx, dtype=np.int64)
    close = grid["close"]
    k_n, s_n = len(idx), close.shape[1]
    f: dict[str, np.ndarray] = {}
    m: dict[str, np.ndarray] = {}
    with np.errstate(all="ignore"):
        pres = grid["present"][idx].astype(bool)
        c0 = close[idx]

        def has_history(hours: int) -> np.ndarray:
            return pres & np.isfinite(_lag(close, idx, hours - 1))

        for k in RET_HOURS:
            f[f"ret_{k}h"] = _ratio(c0, _lag(close, idx, k), pres)
        for w in (24, 168):
            hi = np.full((k_n, s_n), np.nan)
            lo = np.full((k_n, s_n), np.nan)
            for k in range(w):
                hi = np.fmax(hi, _lag(grid["high"], idx, k))
                lo = np.fmin(lo, _lag(grid["low"], idx, k))
            ok = has_history(w)
            f[f"dist_hi_{w}h"] = _ratio(c0, hi, ok)
            f[f"dist_lo_{w}h"] = _ratio(c0, lo, ok)
        vol = _vol24(grid, idx, has_history(24))
        emas = {}
        for span, win in EMA_WINDOWS:
            acc = np.zeros((k_n, s_n))
            for k in range(span):
                acc = acc + _lag(close, idx, win - 1 - k)
            e = acc / span
            alpha = 2.0 / (span + 1)
            for k in range(span, win):
                e = e + alpha * (_lag(close, idx, win - 1 - k) - e)
            ok = has_history(win) & np.isfinite(e)
            emas[span] = np.where(ok, e, np.nan)
            f[f"above_ema{span}"] = np.where(ok, (c0 > e).astype(float), np.nan)
        btc_ret = {k: f[f"ret_{k}h"] for k in BTC_RET_HOURS}  # the market reference, member or not
        xs = pres
        if member_mask is not None or (top_n and top_n > 0):
            # F0: the hour's liquidity top_n only, or the point-in-time members chosen outside (the fixed-list path
            # never takes this branch)
            xs = pres & (np.asarray(member_mask, dtype=bool) if member_mask is not None else _top_mask(vol, top_n))
            for name in list(f):
                f[name] = np.where(xs, f[name], np.nan)
            vol = np.where(xs, vol, np.nan)
        for k in RET_HOURS:
            r = f[f"ret_{k}h"]
            v = np.isfinite(r)
            f[f"ret_{k}h_rank"] = _rank(r, v)
            f[f"ret_{k}h_z"] = _z(r, v)
        vv = np.isfinite(vol)
        total = _seq_sum(vol, vv)
        share = np.full((k_n, s_n), np.nan)
        ok = vv & ((vv.sum(axis=1) >= MIN_XS) & (total > 0))[:, None]
        share[ok] = (vol / np.where(total > 0, total, 1.0)[:, None])[ok]
        f["vol_share_24h"] = share
        f["vol_share_rank"] = _rank(share, np.isfinite(share))
        fund = np.where(xs, grid["funding"][idx], np.nan)
        fv = np.isfinite(fund)
        f["funding"] = fund
        f["funding_rank"] = _rank(fund, fv)
        f["funding_z"] = _z(fund, fv)
        oi = np.where(xs, grid["oi_chg"][idx], np.nan)
        f["oi_chg_24h"] = oi
        f["oi_chg_rank"] = _rank(oi, np.isfinite(oi))

        # market-wide
        if 0 <= btc_col < s_n:
            for k in BTC_RET_HOURS:
                m[f"btc_ret_{k}h"] = btc_ret[k][:, btc_col]
            c, e50, e200 = c0[:, btc_col], emas[50][:, btc_col], emas[200][:, btc_col]
            up = (c > e50) & (e50 > e200)
            dn = (c < e50) & (e50 < e200)
            known = pres[:, btc_col] & np.isfinite(e50) & np.isfinite(e200)
            m["btc_trend"] = np.where(known, np.where(up, 1.0, np.where(dn, -1.0, 0.0)), np.nan)
        else:
            for k in BTC_RET_HOURS:
                m[f"btc_ret_{k}h"] = np.full(k_n, np.nan)
            m["btc_trend"] = np.full(k_n, np.nan)
        for span, _ in EMA_WINDOWS:
            flag = f[f"above_ema{span}"]
            m[f"breadth_ema{span}"] = _share(np.isfinite(flag), flag)
        r24 = f["ret_24h"]
        n24, _, sd24 = _mean_sd(r24, np.isfinite(r24))
        m["disp_24h"] = np.where(n24 >= MIN_XS, sd24, np.nan)
        m["funding_median"] = _median(fund, fv)
        now, prev = grid["oiv_now"][idx], grid["oiv_prev"][idx]
        okv = xs & np.isfinite(now) & np.isfinite(prev) & (prev > 0)
        num, den = _seq_sum(now, okv), _seq_sum(prev, okv)
        agg = np.full(k_n, np.nan)
        ok = (okv.sum(axis=1) >= MIN_XS) & (den > 0)
        agg[ok] = num[ok] / den[ok] - 1.0
        m["oi_chg_24h_agg"] = agg
        m["n"] = xs.sum(axis=1).astype(float)
    sym = np.stack([f[name] for name in SYMBOL_FEATURES], axis=2) if s_n else np.zeros((k_n, 0, len(SYMBOL_FEATURES)))
    mkt = np.stack([m[name] for name in MARKET_FEATURES], axis=1)
    return sym, mkt


# --- stage A: one symbol's hourly grid columns -------------------------------------------------------------------------
def _raw_minutes(open_time: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, qvol: np.ndarray,
                 base: int, g1: int) -> dict[str, np.ndarray]:
    """Per hour base..g1 (hour indices) from 1m bars sorted by open_time: present (a bar in the hour), high, low, the
    hour's last close (not carried) and the quote volume summed in minute order, as the tracker does."""
    n_ext = g1 - base + 1
    out = {"present": np.zeros(n_ext, dtype=bool), "high": np.full(n_ext, np.nan), "low": np.full(n_ext, np.nan),
           "last_close": np.full(n_ext, np.nan), "qvol": np.zeros(n_ext)}
    ot = np.asarray(open_time, dtype=np.int64)
    hrs = ot // MS_H
    sel = (hrs >= base) & (hrs <= g1)
    if sel.any():
        hh = hrs[sel] - base
        starts = np.concatenate(([0], np.flatnonzero(np.diff(hh)) + 1))
        ends = np.concatenate((starts[1:], [len(hh)]))
        rows = hh[starts]
        out["present"][rows] = True
        out["high"][rows] = np.maximum.reduceat(np.asarray(high, float)[sel], starts)
        out["low"][rows] = np.minimum.reduceat(np.asarray(low, float)[sel], starts)
        out["last_close"][rows] = np.asarray(close, float)[sel][ends - 1]
        q = np.zeros((n_ext, 60))
        q[hh, (ot[sel] // MS_MIN) % 60] = np.asarray(qvol, float)[sel]
        acc = np.zeros(n_ext)
        for minute in range(60):
            acc = acc + q[:, minute]
        out["qvol"] = acc
    return out


def _carried(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """present / close (the last close carried forward at most CARRY_HOURS) / high / low / qvol from raw hourly values."""
    n_ext = len(raw["present"])
    out = {"present": raw["present"], "close": np.full(n_ext, np.nan), "high": raw["high"], "low": raw["low"],
           "qvol": raw["qvol"]}
    if raw["present"].any():
        pos = np.arange(n_ext)
        last = np.maximum.accumulate(np.where(raw["present"], pos, -(10**9)))
        carry = (last >= 0) & (pos - last < CARRY_HOURS)
        out["close"][carry] = raw["last_close"][last[carry]]
    return out


def hourly_columns(open_time: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, qvol: np.ndarray,
                   g0: int, g1: int) -> dict[str, np.ndarray]:
    """present / close (carried forward <= CARRY_HOURS) / high / low / qvol for the hours g0..g1 from 1m bars sorted
    by open_time (bars from hour g0 - CARRY_HOURS on are needed for the carried close). The hour's quote volume is
    summed in minute order, as the tracker does."""
    cols = _carried(_raw_minutes(open_time, high, low, close, qvol, g0 - CARRY_HOURS, g1))
    return {k: v[CARRY_HOURS:] for k, v in cols.items()}


def _raw_hours(open_time: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, qvol: np.ndarray,
               base: int, g1: int) -> dict[str, np.ndarray]:
    """_raw_minutes' layout from 1h bars (universe.db blocks, one row per hour): the hour's own high, low, close and
    quote volume."""
    n_ext = g1 - base + 1
    out = {"present": np.zeros(n_ext, dtype=bool), "high": np.full(n_ext, np.nan), "low": np.full(n_ext, np.nan),
           "last_close": np.full(n_ext, np.nan), "qvol": np.zeros(n_ext)}
    ot = np.asarray(open_time, dtype=np.int64)
    if len(ot):
        h = ot // MS_H - base
        sel = (h >= 0) & (h < n_ext)
        rows = h[sel]
        out["present"][rows] = True
        out["high"][rows] = np.asarray(high, float)[sel]
        out["low"][rows] = np.asarray(low, float)[sel]
        out["last_close"][rows] = np.asarray(close, float)[sel]
        out["qvol"][rows] = np.asarray(qvol, float)[sel]
    return out


def merged_columns(minutes: dict[str, np.ndarray] | None, hours: dict[str, np.ndarray] | None, g0: int,
                   g1: int) -> dict[str, np.ndarray]:
    """hourly_columns' output where each hour comes from the 1m bars when the candle database has one in it and from
    the universe.db 1h bar otherwise (raw values of _raw_minutes / _raw_hours over g0 - CARRY_HOURS .. g1), the close
    then carried over the merged hours. An hour with 1m bars is bit-identical to hourly_columns; an hour from the 1h
    archive differs only in the quote volume's precision (stored to 0.1% relative, symbols_inventory.QV_STEP) and in
    being the exchange's hourly aggregate instead of the sum of the minute files."""
    n_ext = g1 - (g0 - CARRY_HOURS) + 1
    empty = {"present": np.zeros(n_ext, dtype=bool), "high": np.full(n_ext, np.nan), "low": np.full(n_ext, np.nan),
             "last_close": np.full(n_ext, np.nan), "qvol": np.zeros(n_ext)}
    m = minutes if minutes is not None else empty
    hr = hours if hours is not None else empty
    use_h = ~m["present"] & hr["present"]
    raw = {"present": m["present"] | hr["present"]}
    for k in ("high", "low", "last_close", "qvol"):
        raw[k] = np.where(use_h, hr[k], m[k])
    cols = _carried(raw)
    return {k: v[CARRY_HOURS:] for k, v in cols.items()}


def funding_column(times: np.ndarray, rates: np.ndarray, g0: int, g1: int) -> np.ndarray:
    """Settled funding rate as of each hour close (the last settlement <= T, at most CARRY_HOURS old)."""
    T = (np.arange(g0, g1 + 1, dtype=np.int64) + 1) * MS_H - 1
    out = np.full(len(T), np.nan)
    if len(times) == 0:
        return out
    j = np.searchsorted(times, T, side="right") - 1
    ok = j >= 0
    ok[ok] &= times[j[ok]] > T[ok] - CARRY_HOURS * MS_H
    out[ok] = rates[j[ok]]
    return out


def metrics_columns(ts: np.ndarray, oi: np.ndarray, oiv: np.ndarray, g0: int, g1: int) -> dict[str, np.ndarray]:
    """24h open-interest change at each hour close with the semantics of MetricsSeries.snapshot (row usable 5 min
    after its bucket, stale after 30 min, previous row >= 75% of the lookback back), plus the two OI values."""
    T = (np.arange(g0, g1 + 1, dtype=np.int64) + 1) * MS_H - 1
    n = len(T)
    out = {"oi_chg": np.full(n, np.nan), "oiv_now": np.full(n, np.nan), "oiv_prev": np.full(n, np.nan)}
    if len(ts) == 0:
        return out
    i = np.searchsorted(ts, T - AVAILABLE_AFTER_MS, side="right") - 1
    ok = i >= 0
    ok[ok] &= (T[ok] - ts[i[ok]] - AVAILABLE_AFTER_MS) <= METRICS_STALE_MS
    iv = i[ok]
    j = np.searchsorted(ts, ts[iv] - OI_LOOKBACK_MS, side="right") - 1
    good = j >= 0
    good[good] &= (ts[iv[good]] - ts[j[good]]) >= OI_LOOKBACK_MS * 0.75
    rows = np.flatnonzero(ok)[good]
    ii, jj = iv[good], j[good]
    prev = oi[jj]
    chg = np.full(len(rows), np.nan)
    pos = np.isfinite(prev) & (prev > 0)
    chg[pos] = oi[ii][pos] / prev[pos] - 1.0
    out["oi_chg"][rows] = chg
    out["oiv_now"][rows] = oiv[ii]
    out["oiv_prev"][rows] = oiv[jj]
    return out


def _metric_arrays(mrows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ts / oi / oi_value of stored metrics rows, with zero or missing values as NaN (as MetricsSeries.from_rows)."""
    if not mrows:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
    pairs = sorted((int(r["ts"]), _num(r.get("oi")), _num(r.get("oi_value"))) for r in mrows)
    ts = np.array([p[0] for p in pairs], dtype=np.int64)
    return ts, np.array([p[1] for p in pairs]), np.array([p[2] for p in pairs])


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x if x > 0 else math.nan


# --- snapshots and readers -----------------------------------------------------------------------------------------------
def flat(ts: int, mkt, sym=None) -> Mapping:
    """The mapping alphas see as ctx.market: "ts", the market-wide features and (when the symbol is in the hour's
    cross-section) its own features. Values are floats, NaN where undefined."""
    d = {"ts": int(ts)}
    d.update(zip(MARKET_FEATURES, (float(x) for x in mkt)))
    if sym is not None:
        d.update(zip(SYMBOL_FEATURES, (float(x) for x in sym)))
    return MappingProxyType(d)


@dataclass
class Snapshot:
    ts: int
    symbols: tuple[str, ...]
    present: np.ndarray  # [symbols] bool
    sym: np.ndarray  # [symbols, len(SYMBOL_FEATURES)]
    mkt: np.ndarray  # [len(MARKET_FEATURES)]

    def view(self, symbol: str) -> Mapping:
        try:
            j = self.symbols.index(symbol)
        except ValueError:
            return flat(self.ts, self.mkt)
        return flat(self.ts, self.mkt, self.sym[j] if self.present[j] else None)

    def rows(self) -> list[tuple[str, int, bytes]]:
        """market_context rows: the market row and one row per symbol in the hour's cross-section."""
        out = [(MARKET_KEY, int(self.ts), np.asarray(self.mkt, dtype=np.float64).tobytes())]
        for j, s in enumerate(self.symbols):
            if self.present[j]:
                out.append((s, int(self.ts), np.asarray(self.sym[j], dtype=np.float64).tobytes()))
        return out


class ContextTable:
    """Stored context rows of one window, looked up by time: `view(symbol, t)` is the snapshot of the last hour close
    at or before t (EMPTY when that hour is missing). The mapping of one (symbol, hour) is built once and reused."""

    def __init__(self, market: dict[int, np.ndarray], symbols: dict[str, dict[int, np.ndarray]]):
        self.market = market
        self.symbols = symbols
        self._last: dict[str, tuple[int, Mapping]] = {}

    def __len__(self) -> int:
        return len(self.market)

    @classmethod
    def load(cls, store, symbols, start: int, end: int, source: str | None = None) -> "ContextTable | None":
        """Rows of [start, end] (hour closes) for `symbols` from a store holding a current-version table, else None.
        `source` (the candle database path) also refuses a table that records another source. Reads under the shared
        build lock, so a concurrent rebuild never hands out a half-deleted table."""
        if store is None or not store.has_market_context():
            return None
        with build_lock(getattr(store, "path", None), shared=True):
            meta = store.get("context.meta")
            if not meta_ok(meta):
                return None
            recorded = (meta.get("source") or {}).get("path")
            if source is not None and recorded and recorded != _real(source):
                return None
            a, b = int(start), int(end)
            mk = {t: np.frombuffer(blob, dtype=np.float64) for t, blob in store.load_market_context(MARKET_KEY, a, b)}
            per = {s: {t: np.frombuffer(blob, dtype=np.float64) for t, blob in store.load_market_context(s, a, b)}
                   for s in symbols}
        return cls(mk, per)

    def view(self, symbol: str, t: int) -> Mapping:
        T = hour_close(hour_of(t))
        last = self._last.get(symbol)
        if last is not None and last[0] == T:
            return last[1]
        mkt = self.market.get(T)
        if mkt is None:
            v = EMPTY
        else:
            v = flat(T, mkt, self.symbols.get(symbol, {}).get(T))
        self._last[symbol] = (T, v)
        return v

    def columns(self, symbol: str, times: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorised lookup for discovery: every feature (symbol features as "c.<name>", market-wide as "m.<name>")
        at the last hour close at or before each time; NaN where no row exists."""
        times = np.asarray(times, dtype=np.int64)
        T = ((times + 1) // MS_H) * MS_H - 1
        out = {f"c.{n}": np.full(len(T), np.nan) for n in SYMBOL_FEATURES}
        out.update({f"m.{n}": np.full(len(T), np.nan) for n in MARKET_FEATURES})
        for table, prefix, names in ((self.market, "m.", MARKET_FEATURES),
                                     (self.symbols.get(symbol, {}), "c.", SYMBOL_FEATURES)):
            if not table:
                continue
            keys = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
            order = np.argsort(keys)
            keys = keys[order]
            vals = np.stack([table[int(k)] for k in keys])
            pos = np.searchsorted(keys, T)
            hit = pos < len(keys)
            hit[hit] &= keys[pos[hit]] == T[hit]
            for c, n in enumerate(names):
                col = out[prefix + n]
                col[hit] = vals[pos[hit], c]
        return out


# --- metadata / coverage ---------------------------------------------------------------------------------------------
def meta_for(universe, source: dict | None = None, top_n: int = 0) -> dict:
    meta = {"version": VERSION, "universe": sorted(universe), "symbol_features": list(SYMBOL_FEATURES),
            "market_features": list(MARKET_FEATURES), "source": source, "built": []}
    if top_n and top_n > 0:  # recorded only in the liquidity mode, so fixed-list stores keep their metadata
        meta["top_n"] = int(top_n)
    return meta


def _top_of(meta) -> int:
    return int((meta or {}).get("top_n") or 0) if isinstance(meta, dict) else 0


def _real(path) -> str:
    return os.path.realpath(str(path)) if path else ""


def live_source(db_path) -> dict:
    """The `source` a running bot records in its live context store (app.Heartless._ctx_meta): its own candle database,
    flagged live. The bot appends that store hourly and prunes it with its data retention, so a research build or run
    refuses it (source_status), and the bot in turn never touches a store recorded for another candle database: one
    path resolution (Settings.context_db_path) serves both sides without one deleting the other's rows."""
    return {"path": _real(db_path), "live": True}


def default_path(db_path) -> str:
    """<directory of the candle database>/context.db: the CONTEXT_DB default."""
    return str(Path(db_path).resolve().parent / DB_NAME)


def source_info(src, universe) -> dict:
    """What a build reads from the candle database `src`: its resolved path and, per stored universe symbol, the
    [first, last] timestamp of its candles, funding and metrics, with a fingerprint over them."""
    ranges = {}
    for s in stored_symbols(src, universe):
        if hasattr(src, "context_series_bounds"):
            ranges[s] = src.context_series_bounds(s)
        else:  # a duck-typed source (tests): the slower min / max queries
            a, b = src.context_candle_bounds(s)
            ranges[s] = {"candles": [a, b] if a is not None else None}
    fp = hashlib.sha1(json.dumps(ranges, sort_keys=True).encode()).hexdigest()[:16]
    return {"path": _real(getattr(src, "path", "")), "ranges": ranges, "fingerprint": fp}


def source_status(meta, source: dict) -> tuple[str, int | None, str]:
    """How the stored context relates to `source` (source_info of the candle database a run reads):
    ("same", None, ""), ("extended", t, why) when history was only appended (or a series / symbol appeared): every
    hour close <= t reads only rows that existed at build time and stays valid, or ("mismatch", None, why)."""
    old = (meta or {}).get("source") if isinstance(meta, dict) else None
    if not isinstance(old, dict) or not old.get("path"):
        return "mismatch", None, "it records no source candle database (built before source checks existed)"
    if old.get("live"):
        return "mismatch", None, ("it is a running bot's live context store (appended hourly, pruned with the data "
                                  "retention), not a research build")
    if old["path"] != source["path"]:
        return "mismatch", None, f"it was built from {old['path']}, not from {source['path']}"
    if old.get("universe_db") != source.get("universe_db"):
        return "mismatch", None, (f"it was built over the universe database {old.get('universe_db') or '(none)'}, not "
                                  f"{source.get('universe_db') or '(none)'}")
    if old.get("fingerprint") == source["fingerprint"]:
        return "same", None, ""
    o_r, n_r = old.get("ranges") or {}, source["ranges"]
    gone = sorted(set(o_r) - set(n_r))
    if gone:
        return "mismatch", None, f"symbols {', '.join(gone)} are no longer in the source"
    stable: int | None = None
    notes = []
    for sym in sorted(n_r):
        for name, rng in sorted((n_r[sym] or {}).items()):
            prev = (o_r.get(sym) or {}).get(name)
            if prev == rng:
                continue
            if rng is None or (prev is not None and (rng[0] != prev[0] or rng[1] < prev[1])):
                return "mismatch", None, (f"the source's {sym} {name} range changed from {_span(prev)} to {_span(rng)} "
                                          f"(history was replaced, not appended)")
            cut = prev[1] if prev is not None else rng[0] - 1  # appended after prev, or new from rng[0] on
            stable = cut if stable is None else min(stable, cut)
            notes.append(f"{sym} {name}")
    if stable is None:  # same ranges, other key layout: nothing a snapshot reads changed
        return "same", None, ""
    return "extended", stable, f"history appended to {', '.join(notes[:4])}{' ...' if len(notes) > 4 else ''}"


def _span(r) -> str:
    return f"{_iso(r[0])} .. {_iso(r[1])}" if r else "none"


def _clip_built(built: list, until: int) -> list[list[int]]:
    cut = hour_close(hour_of(until))
    return [[a, min(b, cut)] for a, b in built if a <= cut]


@contextlib.contextmanager
def build_lock(db_path, shared: bool = False, progress=None):
    """Inter-process lock on "<context db>.lock": exclusive for a build, shared for a reader. Another process that
    builds or reads the same context store waits (and says so through `progress`) instead of computing the same years
    again or reading a table that is being rewritten. Unlocked where flock is unavailable or the lock file cannot be
    created (a read-only directory can only be read)."""
    if fcntl is None or not db_path or str(db_path) == ":memory:":
        yield
        return
    try:
        fh = open(f"{db_path}.lock", "a+")
    except OSError:
        yield
        return
    try:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(fh.fileno(), mode | fcntl.LOCK_NB)
        except BlockingIOError:
            if progress:
                progress(f"waiting for the market-context lock {db_path}.lock (another process is "
                         f"{'building it' if shared else 'building or reading it'})")
            fcntl.flock(fh.fileno(), mode)
        yield
    finally:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()


def _refuse(dst, why: str) -> ContextMismatch:
    return ContextMismatch(f"context store {getattr(dst, 'path', dst)}: {why}. Point CONTEXT_DB at a separate file for "
                           f"this source / universe, or rebuild this one with `heartless context build --rebuild`.")


def _check_or_reset(dst, meta, source: dict, universe: list[str], rebuild: bool, top_n: int = 0) -> tuple[dict, str]:
    """Under the build lock: the metadata the build continues from (reset to empty when rebuilding, new, or of an
    older definition), or ContextMismatch for another source / universe / replaced history. Returns (meta, note)."""
    note = ""
    if rebuild or not isinstance(meta, dict):
        if dst.has_market_context():
            dst.delete_market_context()
        meta = meta_for(universe, source, top_n)
        note = "rebuilding from scratch" if rebuild else ""
    else:
        status, stable, why = source_status(meta, source)
        rec = meta.get("source") or {}
        if status == "mismatch" and (rec.get("live") or not rec.get("path") or rec["path"] != source["path"]):
            raise _refuse(dst, why)  # before anything is deleted (a live store of an older definition included)
        if meta.get("universe") != universe:
            raise _refuse(dst, f"it was built for the universe {','.join(meta.get('universe') or [])}, this run asks "
                               f"for {','.join(universe)}")
        if _top_of(meta) != int(top_n or 0):
            raise _refuse(dst, f"it was built with the cross-section top_n={_top_of(meta)}, this run asks for "
                               f"top_n={int(top_n or 0)}")
        if not meta_ok(meta):  # the definitions changed in code: the stored rows are of no use to anyone
            if dst.has_market_context():
                dst.delete_market_context()
            note = f"definition changed ({meta.get('version')} -> {VERSION}): rebuilding"
            meta = meta_for(universe, source, top_n)
        elif status == "mismatch":
            raise _refuse(dst, why)
        elif status == "extended":
            # the stored rows of the re-opened hours go as well, before the metadata moves on to the new source: a run
            # that builds only its own window leaves the other re-opened hours to later, and a reader that does not
            # build (CONTEXT_AUTO_BUILD=false) must then find them missing (an empty ctx.market), never values computed
            # from the shorter history
            dst.delete_market_context(after=hour_close(hour_of(stable)))
            meta["built"] = _clip_built(meta.get("built", []), stable)
            note = f"{why}: hours after {_iso(hour_close(hour_of(stable)))} are rebuilt"
    meta["source"] = source
    dst.set("context.meta", meta)
    return meta, note


def meta_ok(meta, universe=None) -> bool:
    if not isinstance(meta, dict) or meta.get("version") != VERSION:
        return False
    if meta.get("symbol_features") != list(SYMBOL_FEATURES) or meta.get("market_features") != list(MARKET_FEATURES):
        return False
    return universe is None or meta.get("universe") == sorted(universe)


def _merge(spans: list) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in sorted([int(x), int(y)] for x, y in spans):
        if out and a <= out[-1][1] + MS_H:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def covered(meta, a: int, b: int) -> bool:
    """Whether the hour closes in [a, b] were all built (per the stored meta)."""
    lo, hi = hour_close(hour_of(a + MS_H - 1)), hour_close(hour_of(b))
    if hi < lo:
        return True
    return any(x <= lo and hi <= y for x, y in (meta or {}).get("built", []))


def year_units(start: int, end: int) -> list[tuple[int, int]]:
    """Calendar-year pieces of [start, end] (the build and resume unit)."""
    units = []
    y = datetime.fromtimestamp(start / 1000, tz=timezone.utc).year
    a = int(start)
    while a <= end:
        nxt = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        units.append((a, min(int(end), nxt - 1)))
        a, y = nxt, y + 1
    return units


def stored_symbols(store, universe) -> list[str]:
    """The universe symbols with stored candles. One indexed lookup per symbol where the store has them: a DISTINCT
    over a multi-year candle table scans its whole index (about 2 s on the 6.5 GB long-history database)."""
    if hasattr(store, "context_candle_bounds"):
        return [s for s in sorted(universe) if store.context_candle_bounds(s)[0] is not None]
    have = set(store.candle_symbols())
    return [s for s in sorted(universe) if s in have]


# --- offline build -----------------------------------------------------------------------------------------------------
def _load_symbol(src, sym: str, g0: int, g1: int) -> dict[str, np.ndarray]:
    lo = (g0 - CARRY_HOURS) * MS_H
    hi = hour_close(g1)
    ot, high, low, close, qv = src.context_candle_columns(sym, lo, hi - MS_MIN + 1)
    cols = hourly_columns(ot, high, low, close, qv, g0, g1)
    fr = [r for r in src.load_funding(sym, lo - MS_H, hi) if r[1] is not None]  # the tracker ignores empty rates too
    ft = np.array([r[0] for r in fr], dtype=np.int64)
    fv = np.array([float(r[1]) for r in fr])
    cols["funding"] = funding_column(ft, fv, g0, g1)
    mrows = src.load_metrics(sym, g0 * MS_H - OI_LOOKBACK_MS - 2 * MS_H, hi) if hasattr(src, "load_metrics") else []
    cols.update(metrics_columns(*_metric_arrays(mrows), g0, g1))
    return cols


def data_span(src, symbols: list[str]) -> tuple[int, int] | None:
    """(first candle open time, close time of the last complete hour) of the stored 1m data over `symbols`."""
    bounds = [src.context_candle_bounds(s) for s in symbols]
    firsts = [a for a, _ in bounds if a is not None]
    lasts = [b for _, b in bounds if b is not None]
    if not firsts:
        return None
    return min(firsts), hour_close(hour_of(max(lasts) + MS_MIN - 1))


def compute_unit(src, universe, a: int, b: int, top_n: int = 0) -> list[Snapshot]:
    """Snapshots for every hour close in [a, b] computed from the source store (one calendar-year unit)."""
    syms = sorted(universe)
    h_a, h_b = hour_of(a + MS_H - 1), hour_of(b)
    if h_b < h_a:
        return []
    g0, g1 = h_a - GRID_HOURS + 1, h_b
    have = set(stored_symbols(src, syms))
    n_rows = g1 - g0 + 1
    grid = {k: (np.zeros((n_rows, len(syms)), dtype=bool) if k == "present" else
                np.zeros((n_rows, len(syms))) if k == "qvol" else np.full((n_rows, len(syms)), np.nan))
            for k in GRID_FIELDS}
    for j, s in enumerate(syms):
        if s not in have:
            continue
        cols = _load_symbol(src, s, g0, g1)
        for k in GRID_FIELDS:
            grid[k][:, j] = cols[k]
    idx = np.arange(h_a - g0, h_b - g0 + 1)
    sym, mkt = compute(grid, idx, syms.index(BTC) if BTC in syms else -1, top_n)
    out = []
    pres = grid["present"][idx]
    xs = members(grid, idx, top_n)
    for r in range(len(idx)):
        if pres[r].any():
            out.append(Snapshot(hour_close(g0 + int(idx[r])), tuple(syms), xs[r].copy(), sym[r], mkt[r]))
    return out


# --- point-in-time research cross-section (ROADMAP section 6, 2026-10-16) ---------------------------------------------
# A research store built with CONTEXT_TOP_N picks, at each hour close, the top_n USDT perpetuals by trailing 24h quote
# volume among EVERY eligible perpetual trading in that hour, from the universe.db 1h blocks (red-team F-M3: the union
# of a window's monthly members held symbols that qualified only years later and lacked liquid ones that never became
# monthly members). The ranking reads the 1h archive for every candidate (one source for all); the features of a
# chosen symbol use its 1m candles where the candle database has them and its 1h bars elsewhere (merged_columns). Such a
# store records the universe PIT_UNIVERSE, so it never mixes with a fixed-list or a 1m top_n store.
PIT_UNIVERSE = ("*",)


def _open_universe(path):
    from heartless.core.store import Store

    if not path or not Path(path).is_file():
        raise ContextMismatch(f"no universe database at {path}: the point-in-time research context (CONTEXT_TOP_N > 0) "
                              f"ranks every USDT perpetual from its 1h blocks; run `heartless universe build` first")
    return Store(path, readonly=True)  # a shared research dataset: only read (E2)


def _pit_candidates(ustore, r0: int, h1: int) -> list[str]:
    """Eligible USDT perpetuals (universe.eligible_research: perpetual, USDT-quoted, not a stablecoin, not an archive
    copy of a settled contract) with a traded 1h bar somewhere in the hours r0..h1."""
    from heartless.data.symbols_inventory import FUT_TABLE
    from heartless.data.universe import eligible_research

    inv = {r["symbol"]: r for r in ustore.load_symbol_inventory()}
    lo, hi = r0 * MS_H, (h1 + 1) * MS_H - 1
    out = set()
    for b in ustore.kline_block_index(FUT_TABLE):
        if not b["n"] or b["first_trade"] is None or b["last_trade"] < lo or b["first_trade"] > hi:
            continue
        row = inv.get(b["symbol"])
        if row is not None and eligible_research(row) is None:
            out.add(b["symbol"])
    return sorted(out)


def _hour_bars(ustore, symbol: str, base: int, h1: int) -> dict[str, np.ndarray]:
    from heartless.data.symbols_inventory import FUT_TABLE, load_hourly

    return load_hourly(ustore, symbol, base * MS_H, h1 * MS_H, FUT_TABLE)


def pit_members(ustore, h_a: int, h_b: int, top_n: int) -> tuple[list[str], np.ndarray]:
    """The point-in-time cross-section of the hour closes h_a..h_b (hour indices): (candidate symbols ascending, a
    [hours, candidates] bool mask). At hour h the members are the top_n candidates by trailing 24h quote volume (the
    hours h - 23 .. h, oldest first, as context._vol24) among those trading in hour h (a 1h bar with volume) with 24 h
    of history (a close within the carry window 23 hours back), ties by name; only universe.db is read, so the choice
    never depends on which symbols the candle database happens to hold. Also the S3 shock rule's cross-section."""
    r0 = h_a - 23 - CARRY_HOURS
    syms = _pit_candidates(ustore, r0, h_b)
    n = h_b - r0 + 1
    pres = np.zeros((n, len(syms)), dtype=bool)
    close = np.full((n, len(syms)), np.nan)
    qvol = np.zeros((n, len(syms)))
    for j, sym in enumerate(syms):
        b = _hour_bars(ustore, sym, r0 - CARRY_HOURS, h_b)
        if not len(b["open_time"]):
            continue
        cols = _carried(_raw_hours(b["open_time"], b["high"], b["low"], b["close"], b["quote_volume"],
                                   r0 - CARRY_HOURS, h_b))
        pres[:, j] = cols["present"][CARRY_HOURS:]
        close[:, j] = cols["close"][CARRY_HOURS:]
        qvol[:, j] = cols["qvol"][CARRY_HOURS:]
    idx = np.arange(h_a - r0, h_b - r0 + 1)
    with np.errstate(all="ignore"):
        ok = pres[idx] & (qvol[idx] > 0) & np.isfinite(_lag(close, idx, 23))
        vol = _vol24({"qvol": qvol, "close": close}, idx, ok)
    return syms, _top_mask(vol, top_n) if len(syms) else np.zeros((len(idx), 0), dtype=bool)


def _load_symbol_pit(src, ustore, sym: str, g0: int, g1: int) -> dict[str, np.ndarray]:
    """A chosen symbol's grid columns: 1m candles where the candle database has them, the 1h block elsewhere
    (merged_columns); funding settlements of the candle database, completed by the 1h blocks' in other hours; 5m
    open-interest metrics from the candle database only (the 1h dataset has none: NaN)."""
    base = g0 - CARRY_HOURS
    lo, hi = base * MS_H, hour_close(g1)
    minutes = None
    if src.context_candle_bounds(sym)[0] is not None:
        ot, high, low, close, qv = src.context_candle_columns(sym, lo, hi - MS_MIN + 1)
        if len(ot):
            minutes = _raw_minutes(ot, high, low, close, qv, base, g1)
    b = _hour_bars(ustore, sym, base, g1)
    hours = (_raw_hours(b["open_time"], b["high"], b["low"], b["close"], b["quote_volume"], base, g1)
             if len(b["open_time"]) else None)
    cols = merged_columns(minutes, hours, g0, g1)
    fr = [(int(t), float(r)) for t, r in src.load_funding(sym, lo - MS_H, hi) if r is not None]
    have = {t // MS_H for t, _ in fr}
    if len(b["open_time"]) and "funding" in b:
        ok = np.isfinite(b["funding"])
        fr += [(int(t), float(r)) for t, r in zip(b["open_time"][ok], b["funding"][ok]) if int(t) // MS_H not in have]
    fr.sort()
    cols["funding"] = funding_column(np.array([t for t, _ in fr], dtype=np.int64), np.array([r for _, r in fr]), g0, g1)
    mrows = src.load_metrics(sym, g0 * MS_H - OI_LOOKBACK_MS - 2 * MS_H, hi) if hasattr(src, "load_metrics") else []
    cols.update(metrics_columns(*_metric_arrays(mrows), g0, g1))
    return cols


def _month_pieces(a: int, b: int) -> list[tuple[int, int]]:
    """Calendar-month pieces of [a, b] (ms)."""
    out = []
    d = datetime.fromtimestamp(int(a) / 1000, tz=timezone.utc)
    lo = int(a)
    y, m = d.year, d.month
    while lo <= b:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        nxt = int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)
        out.append((lo, min(int(b), nxt - 1)))
        lo = nxt
    return out


def compute_unit_pit(src, ustore, a: int, b: int, top_n: int) -> list[Snapshot]:
    """compute_unit for a point-in-time store, one calendar month at a time: a year of 2025 chose 511 different
    symbols at least once (new listings surge into the top 40 for a few hours), and one grid over all of them held
    4 GB. Every snapshot reads at most GRID_HOURS rows before it and the members come from the mask, so a month's
    grid over that month's chosen symbols gives the same bits as the year's grid over the year's."""
    out: list[Snapshot] = []
    for pa, pb in _month_pieces(a, b):
        out += _compute_piece_pit(src, ustore, pa, pb, top_n)
    return out


def _compute_piece_pit(src, ustore, a: int, b: int, top_n: int) -> list[Snapshot]:
    """The cross-section of every hour close in [a, b] from pit_members, the grid over the symbols chosen at least
    once in it (and BTC, the market reference) from _load_symbol_pit."""
    h_a, h_b = hour_of(a + MS_H - 1), hour_of(b)
    if h_b < h_a:
        return []
    g0, g1 = h_a - GRID_HOURS + 1, h_b
    cand, mask = pit_members(ustore, h_a, h_b, top_n)
    syms = [s for s, hit in zip(cand, mask.any(axis=0)) if hit]
    if BTC in cand and BTC not in syms:
        syms = sorted(syms + [BTC])
    col = {s: j for j, s in enumerate(cand)}
    n_rows = g1 - g0 + 1
    grid = {k: (np.zeros((n_rows, len(syms)), dtype=bool) if k == "present" else
                np.zeros((n_rows, len(syms))) if k == "qvol" else np.full((n_rows, len(syms)), np.nan))
            for k in GRID_FIELDS}
    for j, sym in enumerate(syms):
        cols = _load_symbol_pit(src, ustore, sym, g0, g1)
        for k in GRID_FIELDS:
            grid[k][:, j] = cols[k]
    idx = np.arange(h_a - g0, h_b - g0 + 1)
    mm = np.stack([mask[:, col[s]] for s in syms], axis=1) if syms else np.zeros((len(idx), 0), dtype=bool)
    sym_f, mkt = compute(grid, idx, syms.index(BTC) if BTC in syms else -1, 0, member_mask=mm)
    xs = members(grid, idx, member_mask=mm)
    pres = grid["present"][idx]
    return [Snapshot(hour_close(g0 + int(idx[r])), tuple(syms), xs[r].copy(), sym_f[r], mkt[r])
            for r in range(len(idx)) if pres[r].any()]


def pit_span(ustore) -> tuple[int, int] | None:
    """(first hour close, last hour close) the universe.db 1h futures blocks cover."""
    from heartless.data.symbols_inventory import FUT_TABLE, data_end

    last = data_end(ustore, FUT_TABLE)
    if last is None:
        return None
    first = ustore.query(f"SELECT MIN(first_ts) t FROM {FUT_TABLE} WHERE n>0")[0]["t"]
    return hour_close(int(first) // MS_H), hour_close(int(last) // MS_H)


def pit_source_info(src, ustore, universe_db) -> dict:
    """source_info of a point-in-time store: every eligible inventory symbol's candle / funding / metrics range in the
    candle database (the ones it holds), the universe.db path and its 1h data span as one more range entry, so an
    appended universe.db month re-opens only the hours after it."""
    from heartless.data.universe import eligible_research

    ranges: dict = {}
    for r in ustore.load_symbol_inventory():
        if eligible_research(r) is not None:
            continue
        sym = r["symbol"]
        if hasattr(src, "context_series_bounds"):
            rng = src.context_series_bounds(sym)
            if any(v is not None for v in rng.values()):
                ranges[sym] = rng
        else:
            a, b = src.context_candle_bounds(sym)
            if a is not None:
                ranges[sym] = {"candles": [a, b]}
    span = pit_span(ustore)
    ranges["@universe.db"] = {"futures_1h": list(span) if span else None}
    fp = hashlib.sha1(json.dumps(ranges, sort_keys=True).encode()).hexdigest()[:16]
    return {"path": _real(getattr(src, "path", "")), "universe_db": _real(universe_db), "ranges": ranges,
            "fingerprint": fp}


def build(src, dst, start: int | None = None, end: int | None = None, universe=None, rebuild: bool = False,
          progress=None, top_n: int = 0, pit=None) -> dict:
    """Compute and store the market context for the hour closes in [start, end] (default: all stored history),
    calendar year by calendar year, into `dst`, a store of its own (CONTEXT_DB; never the candle database `src`).

    A year already built for the same source, universe and definition is skipped, so an interrupted build resumes
    where it stopped; `rebuild` recomputes everything. A store built from another candle database or for another
    universe raises ContextMismatch unless `rebuild` (history appended to the same source only re-opens the hours it
    can change). The whole build holds the store's exclusive lock and re-reads the metadata once it has it, so a
    process that waited for another one's build skips what that one built.

    `pit` (a universe.db path, with top_n > 0): a point-in-time research store (PIT_UNIVERSE; compute_unit_pit), the
    universe argument is not used."""
    dst_path = _real(getattr(dst, "path", ""))
    if dst is None or dst is src or (dst_path and dst_path == _real(getattr(src, "path", ""))):
        raise ContextMismatch("the market context is never written into the candle database: give it its own file "
                              f"(CONTEXT_DB, default <data_dir>/{DB_NAME})")
    ustore = None
    if pit is not None:
        if not top_n or top_n <= 0:
            raise ValueError("a point-in-time context store needs CONTEXT_TOP_N > 0")
        ustore = _open_universe(pit)
        universe = list(PIT_UNIVERSE)
        span = pit_span(ustore)
        syms = []
        if span is None:
            ustore.close()
            raise RuntimeError(f"the universe database {pit} holds no 1h futures blocks")
        source = pit_source_info(src, ustore, pit)
    else:
        universe = sorted(universe or DEFAULT_UNIVERSE)
        syms = stored_symbols(src, universe)
        if not syms:
            raise RuntimeError("none of the context universe symbols has stored candles")
        span = data_span(src, syms)
        source = source_info(src, universe)
    t0 = time.time()
    first, last = span
    a = max(int(start), first) if start is not None else first
    b = min(int(end), last) if end is not None else last
    stats = {"units": [], "rows": 0, "hours": 0, "seconds": 0.0, "universe": universe, "stored_symbols": syms}
    try:
        _build_units(src, dst, a, b, universe, source, rebuild, progress, top_n, ustore, pit, stats)
    finally:
        if ustore is not None:
            ustore.close()
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def _build_units(src, dst, a: int, b: int, universe: list[str], source: dict, rebuild: bool, progress, top_n: int,
                 ustore, pit, stats: dict) -> None:
    with build_lock(getattr(dst, "path", None), progress=progress):
        meta, note = _check_or_reset(dst, dst.get("context.meta"), source, universe, rebuild, top_n)
        if pit is not None and meta.get("pit") != {"universe_db": _real(pit)}:
            meta["pit"] = {"universe_db": _real(pit)}
            dst.set("context.meta", meta)
        if note and progress:
            progress(f"context: {note}")
        if b < a:
            return
        for ua, ub in year_units(a, b):
            if covered(meta, ua, ub):
                stats["units"].append({"start": _iso(ua), "end": _iso(ub), "skipped": True})
                continue
            u0 = time.time()
            snaps = (compute_unit_pit(src, ustore, ua, ub, top_n) if ustore is not None
                     else compute_unit(src, universe, ua, ub, top_n))
            rows = [r for sn in snaps for r in sn.rows()]
            if rows:
                dst.save_market_context(rows)
            meta["built"] = _merge(meta.get("built", []) + [[hour_close(hour_of(ua + MS_H - 1)),
                                                              hour_close(hour_of(ub))]])
            dst.set("context.meta", meta)
            info = {"start": _iso(ua), "end": _iso(ub), "hours": len(snaps), "rows": len(rows),
                    "seconds": round(time.time() - u0, 1)}
            stats["units"].append(info)
            stats["rows"] += len(rows)
            stats["hours"] += len(snaps)
            if progress:
                progress(f"context {info['start'][:10]} .. {info['end'][:10]}: {info['hours']} hours, {info['rows']} "
                         f"rows ({info['seconds']}s)")


def _source_for(src, universe, pit):
    if pit is None:
        return source_info(src, universe), sorted(universe)
    ustore = _open_universe(pit)
    try:
        return pit_source_info(src, ustore, pit), list(PIT_UNIVERSE)
    finally:
        ustore.close()


def usable(src, dst, universe, top_n: int = 0, pit=None) -> tuple[bool, str, int | None, str]:
    """Whether the stored context in `dst` may serve runs over `src` for `universe` (with `pit`: the point-in-time
    store over that universe.db), without building: (ok, status, stable_until, why) with status "same" / "extended" /
    "mismatch" / "none" (nothing built)."""
    meta = dst.get("context.meta")
    if not isinstance(meta, dict):
        return False, "none", None, "nothing built yet"
    source, universe = _source_for(src, universe, pit)
    status, stable, why = source_status(meta, source)
    if meta.get("universe") != sorted(universe):
        return False, "mismatch", None, (f"built for the universe {','.join(meta.get('universe') or [])}, not "
                                         f"{','.join(sorted(universe))}")
    if _top_of(meta) != int(top_n or 0):
        return False, "mismatch", None, f"built with the cross-section top_n={_top_of(meta)}, not {int(top_n or 0)}"
    if not meta_ok(meta):
        return False, "mismatch", None, f"definition {meta.get('version')} is outdated (current {VERSION})"
    return status != "mismatch", status, stable, why


def ensure(src, dst, start: int, end: int, universe, auto_build: bool = True, progress=None, top_n: int = 0,
           pit=None) -> bool:
    """Make sure the rows of [start, end] exist, building the missing years when `auto_build` (ContextMismatch when
    `dst` belongs to another source or universe). True when a usable table is there for the window afterwards (an
    hour that is still missing reads as an empty ctx.market). Without `auto_build` a store that does not match reads
    as no context (False)."""
    if pit is not None:
        ustore = _open_universe(pit)
        try:
            span = pit_span(ustore)
        finally:
            ustore.close()
    else:
        universe = sorted(universe or DEFAULT_UNIVERSE)
        syms = stored_symbols(src, universe)
        span = data_span(src, syms) if syms else None
    if span is None:
        return False
    a, b = max(int(start), span[0]), min(int(end), span[1])
    ok, status, stable, why = usable(src, dst, universe, top_n, pit)
    meta = dst.get("context.meta")
    if ok and status == "same" and (b < a or covered(meta, a, b)):
        return True  # the common case: no lock, no write
    if not auto_build:
        if not ok and status != "none" and progress:
            progress(f"context store {getattr(dst, 'path', dst)} not used: {why}")
        return ok and (status == "same" or (stable is not None and b <= stable))
    build(src, dst, a, b, universe, progress=progress, top_n=top_n, pit=pit)
    return True


def prepare(db_path: str, cfg: dict, start: int, end: int) -> str | None:
    """For research runs (lab, discovery): build the missing years of [start, end] when cfg["build"] (default True)
    and return the path of the store holding the rows, or None when no usable table exists. cfg: {"db": context store
    path (default <candle db directory>/context.db), "universe": [...], "build": bool, "progress": callable,
    "top_n": the cross-section mode (0 = fixed list), "pit": the universe.db path of a point-in-time research store
    (with top_n > 0; ROADMAP section 6, 2026-10-16)}. Raises
    ContextMismatch for a store of another source / universe and for the candle database itself. The caller caps
    `end` at a sealed period."""
    from heartless.core.store import Store

    ctx_path = str(cfg.get("db") or default_path(db_path))
    if _real(ctx_path) == _real(db_path):
        raise ContextMismatch(f"CONTEXT_DB {ctx_path} is the candle database; the market context needs its own file "
                              f"(default <data_dir>/{DB_NAME})")
    Path(ctx_path).parent.mkdir(parents=True, exist_ok=True)
    src = Store(db_path, readonly=True)  # the source candle database is only read (E2: no schema, no -wal / -shm)
    dst = Store(ctx_path)
    try:
        ok = ensure(src, dst, int(start), int(end), cfg.get("universe") or DEFAULT_UNIVERSE,
                    auto_build=cfg.get("build", True), progress=cfg.get("progress"), top_n=int(cfg.get("top_n") or 0),
                    pit=cfg.get("pit") or None)
    finally:
        dst.close()
        src.close()
    return ctx_path if ok else None


# --- online tracker ----------------------------------------------------------------------------------------------------
class MarketContextTracker:
    """Maintains the market context online from streaming data, with exactly the offline semantics.

    What the app must call (the integration step wires this into app.py):
    * `on_bar(symbol, candle)` for every CLOSED 1m candle of every context-universe symbol (also symbols it does not
      trade: they need only 1m candles, no feature frames). Other symbols are ignored.
    * `on_funding(symbol, funding_time, rate)` for each funding settlement (e.g. when the markPrice stream's next
      funding time advances, with the rate it showed just before), and `on_metrics(symbol, row)` for each polled
      5m metrics row (archive-shaped dict with ts = bucket start, oi, oi_value; see data/extras.rows_from_rest).
    * At an hour close T (the close_time of the hour's last 1m bar): once `ready(T)` (every symbol that traded in the
      previous hour has delivered its bar closing at T) or after a short grace timeout, call `snapshot(T)` BEFORE any
      alpha of that hour decides, then pass `view(symbol, t)` as Context.market for every decision until the next
      hour close. Bars older than the last snapshot hour are ignored (they would change a published snapshot).
    * At start-up, `warm(store, now)` feeds WARMUP_HOURS of stored history so the first snapshot is already exact.
    """

    def __init__(self, universe=None, btc: str = BTC, keep: int = 4, top_n: int = 0):
        self.symbols = tuple(sorted(set(universe or DEFAULT_UNIVERSE)))
        self.top_n = int(top_n or 0)  # F0: the cross-section mode (0 = every symbol with data that hour)
        self._col = {s: j for j, s in enumerate(self.symbols)}
        self._btc_col = self._col.get(btc, -1)
        self._pending: list[dict[int, list]] = [{} for _ in self.symbols]  # hour -> [high, low, close, {minute: qv}]
        self._last_close: list[float] = [math.nan] * len(self.symbols)
        self._last_hour: list[int | None] = [None] * len(self.symbols)
        self._last_bar: list[int] = [-(2**62)] * len(self.symbols)  # close_time of the newest bar seen
        self._funding: list[tuple[list[int], list[float]]] = [([], []) for _ in self.symbols]
        self._metrics: list[tuple[list[int], list[float], list[float]]] = [([], [], []) for _ in self.symbols]
        self._rows: dict[str, list] = {k: [] for k in GRID_FIELDS}
        self._hour: int | None = None  # last finalised hour
        self._snaps: dict[int, Snapshot] = {}
        self._keep = keep

    # --- inputs ----------------------------------------------------------------------------------------------
    def on_bar(self, symbol: str, candle) -> None:
        j = self._col.get(symbol)
        if j is None or not getattr(candle, "closed", True):
            return
        h = int(candle.open_time) // MS_H
        if self._hour is not None and h <= self._hour:
            return  # that hour is already published
        agg = self._pending[j].get(h)
        hi, lo, c = float(candle.high), float(candle.low), float(candle.close)
        minute = (int(candle.open_time) // MS_MIN) % 60
        if agg is None:
            self._pending[j][h] = [hi, lo, c, int(candle.open_time), {minute: float(candle.quote_volume or 0.0)}]
        else:
            agg[0] = max(agg[0], hi)
            agg[1] = min(agg[1], lo)
            if int(candle.open_time) >= agg[3]:
                agg[2], agg[3] = c, int(candle.open_time)
            agg[4][minute] = float(candle.quote_volume or 0.0)
        self._last_bar[j] = max(self._last_bar[j], int(candle.close_time))

    def on_funding(self, symbol: str, funding_time: int, rate: float) -> None:
        j = self._col.get(symbol)
        if j is None or rate is None:
            return
        ts, rs = self._funding[j]
        k = bisect_right(ts, int(funding_time))
        if k and ts[k - 1] == int(funding_time):
            rs[k - 1] = float(rate)
        else:
            ts.insert(k, int(funding_time))
            rs.insert(k, float(rate))

    def on_metrics(self, symbol: str, row) -> None:
        j = self._col.get(symbol)
        if j is None:
            return
        get = row.get if isinstance(row, dict) else (lambda k, d=None: getattr(row, k, d))
        t = int(get("ts"))
        ts, oi, oiv = self._metrics[j]
        k = bisect_right(ts, t)
        if k and ts[k - 1] == t:
            oi[k - 1], oiv[k - 1] = _num(get("oi")), _num(get("oi_value"))
        else:
            ts.insert(k, t)
            oi.insert(k, _num(get("oi")))
            oiv.insert(k, _num(get("oi_value")))

    def warm(self, store, now: int, symbols=None, chunk_hours: int = 24) -> int:
        """Feed the stored history needed for exact snapshots from the next hour close on (1m candles, funding and
        metrics of the last WARMUP_HOURS before `now`) and finalise up to the last hour close <= now. Returns the
        number of candles fed.

        Everything stored up to `now` itself is fed, not just up to that hour close: a bot restarted at 08:30 has
        already missed the live 08:00 settlement and the 08:00..08:29 bars, and without them the next snapshots
        would carry the previous funding rate for 8 h and a partial hour in the 24h volume and the 24h/168h highs
        and lows. The tracker only reads what is due at each hour close, so this adds no look-ahead, and the live
        feed may repeat those rows (a repeated bar or row replaces itself).

        The candles are read `chunk_hours` at a time and the hours they complete are finalised before the next
        chunk (the funding and metrics rows go in first; each hour reads only what is due at its close, so the
        result is the same). Feeding the whole window before finalising held one per-minute aggregate per symbol
        and hour at once (16k for 20 symbols), and the hourly rows allocated in between them kept that memory, about
        90 MB, in the process for good (measured by the Wave C integration replay)."""
        h_end = hour_of(now)
        lo = (h_end - WARMUP_HOURS) * MS_H
        syms = [s for s in (symbols or self.symbols) if s in self._col]
        for s in syms:
            for t, r in store.load_funding(s, lo - MS_H, int(now)):
                self.on_funding(s, t, r)
            if hasattr(store, "load_metrics"):
                for row in store.load_metrics(s, lo - OI_LOOKBACK_MS - 2 * MS_H, int(now)):
                    self.on_metrics(s, row)
        fed = 0
        last = int(now) - MS_MIN + 1  # every bar closed by now
        a = lo
        while a <= last:
            b = min(a + max(1, int(chunk_hours)) * MS_H - 1, last)
            for s in syms:
                for c in store.load_candles(s, start=a, end=b):
                    self.on_bar(s, c)
                    fed += 1
            if b < last and hour_of(b) < h_end:  # b is an hour close: every bar of that hour is in
                self._finalise(hour_of(b))
            a = b + 1
        self._finalise(h_end)
        return fed

    # --- outputs ---------------------------------------------------------------------------------------------
    @property
    def last_hour(self) -> int | None:
        """The last finalised hour index (None before warm-up): its snapshot is fixed and its bars are ignored."""
        return self._hour

    def ready(self, T: int) -> bool:
        """Whether every symbol that had a bar in the hour before T's hour has delivered its bar closing at T."""
        h = hour_of(T)
        for j in range(len(self.symbols)):
            traded_before = (self._last_hour[j] is not None and self._last_hour[j] >= h - 1) or \
                any(hh >= h - 1 for hh in self._pending[j])
            if traded_before and self._last_bar[j] < hour_close(h):
                return False
        return True

    def snapshot(self, T: int) -> Snapshot | None:
        """The snapshot of the hour close at or before T (cached). None for an hour older than the cache."""
        h = hour_of(T)
        if h in self._snaps:
            return self._snaps[h]
        if self._hour is not None and h < self._hour:
            return None
        self._finalise(h)
        grid = {k: np.array(v) for k, v in self._rows.items()}
        if not len(grid["close"]):
            return None
        last = np.array([len(grid["close"]) - 1])
        sym, mkt = compute(grid, last, self._btc_col, self.top_n)
        pres = grid["present"][-1].astype(bool)
        xs = members(grid, last, self.top_n)[0]
        snap = Snapshot(hour_close(h), self.symbols, xs, sym[0], mkt[0]) if pres.any() else None
        if snap is not None:
            self._snaps[h] = snap
            for old in sorted(self._snaps)[:-self._keep]:
                del self._snaps[old]
        return snap

    def view(self, symbol: str, t: int) -> Mapping:
        """ctx.market for a decision at time t: the snapshot of the last hour close <= t (EMPTY when unavailable)."""
        snap = self.snapshot(t)
        return snap.view(symbol) if snap is not None and snap.ts == hour_close(hour_of(t)) else EMPTY

    # --- internals ---------------------------------------------------------------------------------------------
    def _finalise(self, h: int) -> None:
        start = h if self._hour is None else self._hour + 1
        if self._hour is None:
            hours = [hh for p in self._pending for hh in p]
            start = min(hours) if hours else h
        start = max(start, h - GRID_HOURS - CARRY_HOURS)
        for hh in range(start, h + 1):
            self._append_row(hh)
        self._hour = h
        for p in self._pending:
            for old in [k for k in p if k <= h]:
                del p[old]
        self._prune(h)

    def _append_row(self, h: int) -> None:
        T = hour_close(h)
        row = {k: [] for k in GRID_FIELDS}
        for j in range(len(self.symbols)):
            agg = self._pending[j].pop(h, None)
            if agg is not None:
                acc = 0.0
                for minute in sorted(agg[4]):
                    acc = acc + agg[4][minute]
                self._last_close[j], self._last_hour[j] = agg[2], h
                row["present"].append(True)
                row["high"].append(agg[0])
                row["low"].append(agg[1])
                row["qvol"].append(acc)
            else:
                row["present"].append(False)
                row["high"].append(math.nan)
                row["low"].append(math.nan)
                row["qvol"].append(0.0)
            lh = self._last_hour[j]
            row["close"].append(self._last_close[j] if lh is not None and h - lh < CARRY_HOURS else math.nan)
            ts, rs = self._funding[j]
            k = bisect_right(ts, T) - 1
            row["funding"].append(rs[k] if k >= 0 and ts[k] > T - CARRY_HOURS * MS_H else math.nan)
            mts, moi, moiv = self._metrics[j]
            chg = now = prev = math.nan
            i = bisect_right(mts, T - AVAILABLE_AFTER_MS) - 1
            if i >= 0 and T - mts[i] - AVAILABLE_AFTER_MS <= METRICS_STALE_MS:
                jj = bisect_right(mts, mts[i] - OI_LOOKBACK_MS) - 1
                if jj >= 0 and mts[i] - mts[jj] >= OI_LOOKBACK_MS * 0.75:
                    p = moi[jj]
                    if p == p and p > 0:
                        chg = moi[i] / p - 1.0
                    now, prev = moiv[i], moiv[jj]
            row["oi_chg"].append(chg)
            row["oiv_now"].append(now)
            row["oiv_prev"].append(prev)
        for k in GRID_FIELDS:
            self._rows[k].append(row[k])
            if len(self._rows[k]) > GRID_HOURS:
                del self._rows[k][0]

    def _prune(self, h: int) -> None:
        cut_f = hour_close(h) - CARRY_HOURS * MS_H - 8 * MS_H
        cut_m = hour_close(h) - OI_LOOKBACK_MS - 3 * MS_H
        for j in range(len(self.symbols)):
            ts, rs = self._funding[j]
            k = max(0, bisect_right(ts, cut_f) - 1)
            if k:
                del ts[:k], rs[:k]
            mts, moi, moiv = self._metrics[j]
            k = max(0, bisect_right(mts, cut_m) - 1)  # keep the newest row at or before the cut (a gap's anchor)
            if k:
                del mts[:k], moi[:k], moiv[:k]

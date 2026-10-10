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

Offline: `build()` (CLI `heartless context build`) writes the market_context table, resumable per calendar year.
Readers (lab workers, the backtester, discovery) use `ContextTable`. Online: `MarketContextTracker` (see its
docstring for what the app must call). Alphas read the result as `ctx.market`, a read-only flat mapping of the
symbol's own features plus the market-wide ones and "ts" (empty when no context is available).
"""
from __future__ import annotations

import math
import time
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType

import numpy as np

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


def compute(grid: dict[str, np.ndarray], idx, btc_col: int = -1) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot features at the grid rows `idx` (hour indices into the grid, each needs up to GRID_HOURS - 1 rows of
    history before it). grid: GRID_FIELDS -> [rows, symbols] arrays on a contiguous hour axis. Returns
    (symbol features [len(idx), symbols, len(SYMBOL_FEATURES)], market features [len(idx), len(MARKET_FEATURES)])."""
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
        vol = np.zeros((k_n, s_n))
        for k in range(23, -1, -1):  # oldest hour first
            vol = vol + _lag(grid["qvol"], idx, k)
        vol = np.where(has_history(24), vol, np.nan)
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
        fund = np.where(pres, grid["funding"][idx], np.nan)
        fv = np.isfinite(fund)
        f["funding"] = fund
        f["funding_rank"] = _rank(fund, fv)
        f["funding_z"] = _z(fund, fv)
        oi = np.where(pres, grid["oi_chg"][idx], np.nan)
        f["oi_chg_24h"] = oi
        f["oi_chg_rank"] = _rank(oi, np.isfinite(oi))

        # market-wide
        if 0 <= btc_col < s_n:
            for k in BTC_RET_HOURS:
                m[f"btc_ret_{k}h"] = f[f"ret_{k}h"][:, btc_col]
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
        okv = pres & np.isfinite(now) & np.isfinite(prev) & (prev > 0)
        num, den = _seq_sum(now, okv), _seq_sum(prev, okv)
        agg = np.full(k_n, np.nan)
        ok = (okv.sum(axis=1) >= MIN_XS) & (den > 0)
        agg[ok] = num[ok] / den[ok] - 1.0
        m["oi_chg_24h_agg"] = agg
        m["n"] = pres.sum(axis=1).astype(float)
    sym = np.stack([f[name] for name in SYMBOL_FEATURES], axis=2) if s_n else np.zeros((k_n, 0, len(SYMBOL_FEATURES)))
    mkt = np.stack([m[name] for name in MARKET_FEATURES], axis=1)
    return sym, mkt


# --- stage A: one symbol's hourly grid columns -------------------------------------------------------------------------
def hourly_columns(open_time: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, qvol: np.ndarray,
                   g0: int, g1: int) -> dict[str, np.ndarray]:
    """present / close (carried forward <= CARRY_HOURS) / high / low / qvol for the hours g0..g1 from 1m bars sorted
    by open_time (bars from hour g0 - CARRY_HOURS on are needed for the carried close). The hour's quote volume is
    summed in minute order, as the tracker does."""
    base = g0 - CARRY_HOURS
    n_ext = g1 - base + 1
    out = {"present": np.zeros(n_ext, dtype=bool), "close": np.full(n_ext, np.nan), "high": np.full(n_ext, np.nan),
           "low": np.full(n_ext, np.nan), "qvol": np.zeros(n_ext)}
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
        last_close = np.full(n_ext, np.nan)
        last_close[rows] = np.asarray(close, float)[sel][ends - 1]
        q = np.zeros((n_ext, 60))
        q[hh, (ot[sel] // MS_MIN) % 60] = np.asarray(qvol, float)[sel]
        acc = np.zeros(n_ext)
        for minute in range(60):
            acc = acc + q[:, minute]
        out["qvol"] = acc
        pos = np.arange(n_ext)
        last = np.maximum.accumulate(np.where(out["present"], pos, -(10**9)))
        carry = (last >= 0) & (pos - last < CARRY_HOURS)
        out["close"][carry] = last_close[last[carry]]
    return {k: v[CARRY_HOURS:] for k, v in out.items()}


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
    def load(cls, store, symbols, start: int, end: int) -> "ContextTable | None":
        """Rows of [start, end] (hour closes) for `symbols` from a store holding a current-version table, else None."""
        if store is None or not store.has_market_context() or not meta_ok(store.get("context.meta")):
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
def meta_for(universe) -> dict:
    return {"version": VERSION, "universe": sorted(universe), "symbol_features": list(SYMBOL_FEATURES),
            "market_features": list(MARKET_FEATURES), "built": []}


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


def compute_unit(src, universe, a: int, b: int) -> list[Snapshot]:
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
    sym, mkt = compute(grid, idx, syms.index(BTC) if BTC in syms else -1)
    out = []
    pres = grid["present"][idx]
    for r in range(len(idx)):
        if pres[r].any():
            out.append(Snapshot(hour_close(g0 + int(idx[r])), tuple(syms), pres[r].copy(), sym[r], mkt[r]))
    return out


def build(src, dst=None, start: int | None = None, end: int | None = None, universe=None, rebuild: bool = False,
          progress=None) -> dict:
    """Compute and store the market context for the hour closes in [start, end] (default: all stored history),
    calendar year by calendar year. A year already built for the same version and universe is skipped, so an
    interrupted build resumes where it stopped; `rebuild` recomputes everything. `dst` (default `src`) is the store
    that receives the rows, e.g. a separate CONTEXT_DB file next to a read-only research database."""
    dst = dst or src
    universe = sorted(universe or DEFAULT_UNIVERSE)
    syms = stored_symbols(src, universe)
    if not syms:
        raise RuntimeError("none of the context universe symbols has stored candles")
    t0 = time.time()
    first, last = data_span(src, syms)
    a = max(int(start), first) if start is not None else first
    b = min(int(end), last) if end is not None else last
    meta = dst.get("context.meta")
    if rebuild or not meta_ok(meta, universe):
        if dst.has_market_context():
            dst.delete_market_context()
        meta = meta_for(universe)
        dst.set("context.meta", meta)
    stats = {"units": [], "rows": 0, "hours": 0, "seconds": 0.0, "universe": universe, "stored_symbols": syms}
    if b < a:
        return stats
    for ua, ub in year_units(a, b):
        if covered(meta, ua, ub):
            stats["units"].append({"start": _iso(ua), "end": _iso(ub), "skipped": True})
            continue
        u0 = time.time()
        snaps = compute_unit(src, universe, ua, ub)
        rows = [r for sn in snaps for r in sn.rows()]
        if rows:
            dst.save_market_context(rows)
        meta["built"] = _merge(meta.get("built", []) + [[hour_close(hour_of(ua + MS_H - 1)), hour_close(hour_of(ub))]])
        dst.set("context.meta", meta)
        info = {"start": _iso(ua), "end": _iso(ub), "hours": len(snaps), "rows": len(rows),
                "seconds": round(time.time() - u0, 1)}
        stats["units"].append(info)
        stats["rows"] += len(rows)
        stats["hours"] += len(snaps)
        if progress:
            progress(f"context {info['start'][:10]} .. {info['end'][:10]}: {info['hours']} hours, {info['rows']} rows "
                     f"({info['seconds']}s)")
    stats["seconds"] = round(time.time() - t0, 1)
    return stats


def ensure(src, dst, start: int, end: int, universe, auto_build: bool = True, progress=None) -> bool:
    """Make sure the rows of [start, end] exist, building the missing years when `auto_build`. True when a current
    table is there for the window afterwards (an hour that is still missing reads as an empty ctx.market)."""
    universe = sorted(universe or DEFAULT_UNIVERSE)
    syms = stored_symbols(src, universe)
    span = data_span(src, syms) if syms else None
    if span is None:
        return False
    a, b = max(int(start), span[0]), min(int(end), span[1])
    meta = dst.get("context.meta")
    if meta_ok(meta, universe) and (b < a or covered(meta, a, b)):
        return True
    if not auto_build:
        return meta_ok(meta, universe)
    build(src, dst, a, b, universe, progress=progress)
    return True


def prepare(db_path: str, cfg: dict, start: int, end: int) -> str | None:
    """For research runs (lab, discovery): build the missing years of [start, end] when cfg["build"] (default True)
    and return the path of the store holding the rows, or None when no usable table exists. cfg: {"db": context store
    path (default the candle store), "universe": [...], "build": bool, "progress": callable}. The caller caps `end`
    at a sealed period."""
    from heartless.core.store import Store

    ctx_path = str(cfg.get("db") or db_path)
    src = Store(db_path)
    dst = src if ctx_path == str(db_path) else Store(ctx_path)
    try:
        ok = ensure(src, dst, int(start), int(end), cfg.get("universe") or DEFAULT_UNIVERSE,
                    auto_build=cfg.get("build", True), progress=cfg.get("progress"))
    finally:
        if dst is not src:
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

    def __init__(self, universe=None, btc: str = BTC, keep: int = 4):
        self.symbols = tuple(sorted(set(universe or DEFAULT_UNIVERSE)))
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

    def warm(self, store, now: int, symbols=None) -> int:
        """Feed the stored history needed for exact snapshots from the next hour close on (1m candles, funding and
        metrics of the last WARMUP_HOURS before `now`) and finalise up to the last hour close <= now. Returns the
        number of candles fed."""
        from heartless.core.models import Candle

        h_end = hour_of(now)
        lo = (h_end - WARMUP_HOURS) * MS_H
        fed = 0
        for s in (symbols or self.symbols):
            if s not in self._col:
                continue
            for c in store.load_candles(s, start=lo, end=hour_close(h_end) - MS_MIN + 1):
                self.on_bar(s, c)
                fed += 1
            for t, r in store.load_funding(s, lo - MS_H, hour_close(h_end)):
                self.on_funding(s, t, r)
            if hasattr(store, "load_metrics"):
                for row in store.load_metrics(s, lo - OI_LOOKBACK_MS - 2 * MS_H, hour_close(h_end)):
                    self.on_metrics(s, row)
        self._finalise(h_end)
        return fed

    # --- outputs ---------------------------------------------------------------------------------------------
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
        sym, mkt = compute(grid, np.array([len(grid["close"]) - 1]), self._btc_col)
        pres = grid["present"][-1].astype(bool)
        snap = Snapshot(hour_close(h), self.symbols, pres, sym[0], mkt[0]) if pres.any() else None
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

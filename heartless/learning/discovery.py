"""Alpha discovery: mine rule-based alphas from a library of scale-free features, with honest statistics.

Pipeline
1. `build_dataset()` aligns, for every decision time of every symbol, a vector of scale-free features (oscillators,
   distances in ATR units, volatility ranks, flow, positioning from 5-minute metrics, time of day and, optionally,
   the market context of data/context.py: cross-sectional ranks / z-scores of returns, volume share, funding and OI
   change, and market-wide BTC returns, trend, breadth, dispersion) and the realised outcome of fixed exit templates
   simulated on 1-minute bars with the production cost model (entry at the next bar's open, taker fees, slippage,
   pessimistic stop-first fills). Decision grids: every 15m or 1h close, or 00:00 UTC once a day ("1d"). Templates:
   stop k*ATR / target r*stop / timeout, or a pure time exit (8h / 24h / 72h) behind a wide 4 ATR catastrophic stop.
   Long windows are built as independent (symbol, calendar year) jobs, so memory is bounded by one symbol-year.
2. `search()` explores conjunctions of 1-3 threshold conditions (quantile cut points learned on TRAIN only) by
   beam search, long and short, over every template, ranks rules by the day-clustered t-statistic of their R minus
   the same-month same-direction baseline (market drift earns nothing) after removing overlapping trades, and keeps
   a diverse set (low entry overlap).
3. `validate()` re-measures the survivors on a later, untouched window. A rule must clear the multiple-testing bar
   t >= max(3, sqrt(2 ln N)) on TRAIN, N being the number of rules actually evaluated by the run (all grids,
   templates, sides and depths together, plus any prior trials of the family), *and* hold up on VALID.

The very same feature code evaluates a rule online (`decision_features()`), so a discovered rule behaves identically
in the live bot, in the lab and in this miner, for every rule type the miner produces (15m / 1h / 1d grids, stop-target
and time-exit templates, frame, positioning, time and market-context features). Survivors are shipped as data inside
StrategyParams (alphas["discovered"]["rules"]) and traded by the `discovered` alpha; the champion/challenger loop
decides whether a rule set ever reaches real money. The defaults (15m/1h grid, stop/target templates, no market
context) are what the live discovery cycle mines; the other grids, the time-exit templates and the context features
are research options (`heartless discover --grid 1h,1d --templates all --context`) that the `discovered` alpha runs as
well.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from heartless.data.context import MARKET_FEATURES, SYMBOL_FEATURES
from heartless.data.extras import AVAILABLE_AFTER_MS, MetricsSeries

DECISION_TFS = ("15m", "1h")  # feature frames
TF_MS = {"15m": 900_000, "1h": 3_600_000}
GRIDS = ("15m", "1h", "1d")  # decision grids: every 15m / 1h close, or the 1h close at 00:00 UTC
GRID_ATR_TF = {"15m": "15m", "1h": "1h", "1d": "1h"}  # ATR that sizes a grid's stops
DAY_MS = 86_400_000
# exit templates: (stop in ATR of the grid's timeframe, target in R or None for a pure time exit, timeout in minutes)
TEMPLATES: tuple[tuple[float, float | None, int], ...] = (
    (1.0, 1.5, 240), (1.5, 2.0, 480), (2.0, 3.0, 720),  # stop / target / timeout
    (4.0, None, 480), (4.0, None, 1440), (4.0, None, 4320))  # time exit after 8h / 24h / 72h, catastrophic 4 ATR stop
TEMPLATE_SETS = {"classic": (0, 1, 2), "time": (3, 4, 5), "all": (0, 1, 2, 3, 4, 5)}
QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95)
WARMUP_MS = 22 * DAY_MS  # >= the live 1h feature window (520 bars), so mined and live indicator values agree
SIM_BUDGET = 4_000_000  # elements per (entries x minutes) block of the exit simulation
DELIST_SLIP_MULT = 5.0  # exit slippage multiple of a delisting close (ROADMAP 7.4; lab.DELIST_SLIP_MULT)

_FRAME_FEATURES = ("rsi14", "rsi7", "adx", "atr_rank", "bb_width_rank", "vol_z", "taker_ratio3", "cvd20", "hurst", "chop",
                   "body_ratio", "upper_wick", "lower_wick", "st_dir")
_DERIVED = ("slope_atr", "dist_ema21", "dist_ema50", "dist_ema200", "bb_z", "vwap_z", "dc20_pos", "macd_atr", "ret4_atr",
            "squeeze_bars", "pdi_mdi")
_EXTRAS = ("oi_chg_1h", "oi_chg_4h", "oi_chg_24h", "oi_chg_1h_z", "top_ls_pos", "top_ls_pos_chg_4h", "ls_acc",
           "ls_acc_chg_4h", "taker_ratio_1h")
_TIME = ("hour_utc", "min_to_funding")


def feature_names() -> list[str]:
    names = []
    for tf in DECISION_TFS:
        names += [f"{tf}.{n}" for n in _FRAME_FEATURES + _DERIVED]
    names += [f"x.{n}" for n in _EXTRAS]
    names += [f"t.{n}" for n in _TIME]
    return names


FEATURES = feature_names()  # the library the `discovered` alpha evaluates online
# market context features ("c." the symbol's own cross-sectional features, "m." market-wide), appended for --context
CONTEXT_FEATURES = [f"c.{n}" for n in SYMBOL_FEATURES] + [f"m.{n}" for n in MARKET_FEATURES]
FEATURES_X = FEATURES + CONTEXT_FEATURES


# --- feature computation (shared by mining and online evaluation) --------------------------------------------------
def _gather(arr: np.ndarray, idx: np.ndarray, k: int = 0) -> np.ndarray:
    j = idx - k
    out = np.full(len(idx), np.nan)
    ok = (j >= 0) & (j < len(arr))
    out[ok] = arr[j[ok]]
    return out


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan)
    ok = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[ok] = a[ok] / b[ok]
    return out


def frame_features(frame, times: np.ndarray, tf: str) -> dict[str, np.ndarray]:
    """Features of one timeframe as of each time in `times` (the last bar closed at or before t)."""
    idx = np.searchsorted(frame.close_time, times, side="right") - 1
    f = frame.f
    out: dict[str, np.ndarray] = {}
    for n in _FRAME_FEATURES:
        out[f"{tf}.{n}"] = _gather(f[n], idx) if n in f else np.full(len(times), np.nan)
    close = _gather(f["close"], idx)
    atr = _gather(f["atr"], idx)
    out[f"{tf}.slope_atr"] = _safe_div(_gather(f["slope20"], idx) * close, atr)
    for e in ("ema21", "ema50", "ema200"):
        out[f"{tf}.dist_{e}"] = _safe_div(close - _gather(f[e], idx), atr)
    out[f"{tf}.bb_z"] = _safe_div(close - _gather(f["bb_mid"], idx), _gather(f["bb_sd"], idx))
    out[f"{tf}.vwap_z"] = _safe_div(close - _gather(f["vwap"], idx), _gather(f["vwap_sd"], idx))
    hi, lo = _gather(f["dc_hi20"], idx), _gather(f["dc_lo20"], idx)
    out[f"{tf}.dc20_pos"] = _safe_div(close - lo, hi - lo)
    out[f"{tf}.macd_atr"] = _safe_div(_gather(f["macd_hist"], idx), atr)
    out[f"{tf}.ret4_atr"] = _safe_div(close - _gather(f["close"], idx, 4), atr)
    out[f"{tf}.squeeze_bars"] = np.minimum(_gather(f["squeeze_bars"], idx), 40.0)
    out[f"{tf}.pdi_mdi"] = _gather(f["pdi"], idx) - _gather(f["mdi"], idx)
    out[f"_{tf}.close"] = close
    out[f"_{tf}.atr"] = atr
    out[f"_{tf}.idx"] = idx.astype(float)
    return out


def extras_features(ms: MetricsSeries | None, times: np.ndarray) -> dict[str, np.ndarray]:
    """Twin of MetricsSeries.snapshot() for the fields used by rules, bit for bit: what the `discovered` alpha reads
    from ctx.extras (the snapshot) in the bot and the lab is exactly the value a rule was mined on. The ratio, change
    and difference fields are the same IEEE operations vectorised; the 1h taker geometric mean and the OI-change
    z-score run the snapshot's own numpy reductions once per metrics row in use (a cumulative-sum shortcut differed in
    the last bits)."""
    n = len(times)
    out = {f"x.{k}": np.full(n, np.nan) for k in _EXTRAS}
    if ms is None or len(ms) == 0:
        return out
    ts = ms.ts
    i = np.searchsorted(ts, times - AVAILABLE_AFTER_MS, side="right") - 1
    valid = i >= 0
    age = np.full(n, np.inf)
    age[valid] = (times[valid] - ts[i[valid]] - AVAILABLE_AFTER_MS) / 60_000
    valid &= age <= 30
    if not valid.any():
        return out
    iv = i[valid]

    def back(arr: np.ndarray, lookback: int) -> np.ndarray:
        j = np.searchsorted(ts, ts[iv] - lookback, side="right") - 1
        res = np.full(len(iv), np.nan)
        ok = (j >= 0)
        ok[ok] &= (ts[iv[ok]] - ts[j[ok]]) >= lookback * 0.75
        res[ok] = arr[j[ok]]
        return res

    oi = ms.oi[iv]
    chg = {}
    for label, lb in (("1h", 3_600_000), ("4h", 14_400_000), ("24h", 86_400_000)):
        prev = back(ms.oi, lb)
        c = np.full(len(iv), np.nan)
        ok = np.isfinite(prev) & (prev > 0)
        c[ok] = oi[ok] / prev[ok] - 1.0
        chg[label] = c
        out[f"x.oi_chg_{label}"][valid] = c
    tlp = ms.top_ls_positions[iv]
    lsa = ms.ls_accounts[iv]
    out["x.top_ls_pos"][valid] = tlp
    out["x.ls_acc"][valid] = lsa
    out["x.top_ls_pos_chg_4h"][valid] = tlp - back(ms.top_ls_positions, 14_400_000)
    out["x.ls_acc_chg_4h"][valid] = lsa - back(ms.ls_accounts, 14_400_000)
    # geometric mean of the last 12 taker ratios and the z-score of the 1h OI change against the 12-row changes of the
    # last <= 864 rows: both depend only on the usable row, so each distinct row is reduced once, exactly as snapshot()
    tk, oi_all = ms.taker_ls_vol, ms.oi
    rows, inv = np.unique(iv, return_inverse=True)
    first = np.zeros(len(rows), dtype=np.int64)
    first[inv] = np.arange(len(iv))  # one decision of each distinct metrics row (its 1h change is the row's)
    g = np.full(len(rows), np.nan)
    z = np.full(len(rows), np.nan)
    c1h = chg["1h"]
    for k, i in enumerate(rows.tolist()):
        window = tk[max(0, i - 11):i + 1]
        good = window[np.isfinite(window) & (window > 0)]
        if len(good):
            g[k] = float(np.exp(np.mean(np.log(good))))
        seg = oi_all[max(0, i - 864):i + 1]
        cur = float(c1h[first[k]])
        if len(seg) > 60:
            ch = seg[12:] / seg[:-12] - 1.0
            ch = ch[np.isfinite(ch)]
            if len(ch) > 30 and np.std(ch) > 0 and not math.isnan(cur):
                z[k] = float((cur - np.mean(ch)) / np.std(ch))
    out["x.taker_ratio_1h"][valid] = g[inv]
    out["x.oi_chg_1h_z"][valid] = z[inv]
    return out


def time_features(times: np.ndarray) -> dict[str, np.ndarray]:
    t = np.asarray(times, dtype=np.int64)
    hour = ((t // 3_600_000) % 24).astype(float)
    nxt = (t // 28_800_000 + 1) * 28_800_000
    return {"t.hour_utc": hour, "t.min_to_funding": (nxt - t) / 60_000.0}


def features_at(frames: dict, ms: MetricsSeries | None, times: np.ndarray) -> dict[str, np.ndarray]:
    times = np.asarray(times, dtype=np.int64)
    out: dict[str, np.ndarray] = {}
    for tf in DECISION_TFS:
        if tf in frames:
            out.update(frame_features(frames[tf], times, tf))
        else:
            for n in _FRAME_FEATURES + _DERIVED:
                out[f"{tf}.{n}"] = np.full(len(times), np.nan)
    out.update(extras_features(ms, times))
    out.update(time_features(times))
    return out


# --- outcome simulation ----------------------------------------------------------------------------------------------
@dataclass
class CostModel:
    taker_fee: float = 0.0005
    slippage_bps: float = 1.5
    half_spread_bps: float = 0.5
    buffer: float = 0.0003  # extra cost allowance used by the engine's risk definition


def simulate_template(o1: np.ndarray, h1: np.ndarray, l1: np.ndarray, c1: np.ndarray, entry_idx: np.ndarray,
                      atr: np.ndarray, side: int, sl_atr: float, tp_r: float | None, hold_min: int, cost: CostModel,
                      chunk: int | None = None, end_mult: float = 1.0,
                      forced: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """R multiple and exit 1m-index for entries at the open of 1m bar `entry_idx` (vectorised, pessimistic).

    R is net of fees/slippage and expressed against the engine's risk unit (stop distance + round-trip costs).
    `tp_r=None` has no target: the trade leaves at the stop or at the close of its last minute. Entries are simulated
    in blocks of at most SIM_BUDGET (entries x minutes) cells, so a 72h template does not allocate gigabytes.
    `end_mult` != 1 says the data ends at a delisting: a trade the end of the data cuts short (no stop or target hit
    before it) leaves at the last close with end_mult x the slippage (ROADMAP 7.4) and is flagged in `forced`."""
    n = len(entry_idx)
    r_out = np.full(n, np.nan)
    exit_out = np.full(n, -1, dtype=np.int64)
    N1 = len(o1)
    slip = (cost.slippage_bps + cost.half_spread_bps) / 1e4
    ok_all = (entry_idx >= 0) & (entry_idx < N1) & np.isfinite(atr) & (atr > 0)
    steps = np.arange(hold_min)
    chunk = chunk or max(1, min(4000, SIM_BUDGET // max(hold_min, 1)))
    for s in range(0, n, chunk):
        sel = np.arange(s, min(n, s + chunk))
        sel = sel[ok_all[sel]]
        if len(sel) == 0:
            continue
        e = entry_idx[sel]
        entry = o1[e] * (1 + side * slip)
        dist = sl_atr * atr[sel]
        stop = entry - side * dist
        J = e[:, None] + steps[None, :]
        inside = J < N1
        Jc = np.minimum(J, N1 - 1)
        BIG = hold_min + 10
        if side > 0:
            hit_s = (l1[Jc] <= stop[:, None]) & inside
        else:
            hit_s = (h1[Jc] >= stop[:, None]) & inside
        fs = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), BIG)
        if tp_r:
            tp = entry + side * tp_r * dist
            hit_t = ((h1[Jc] >= tp[:, None]) if side > 0 else (l1[Jc] <= tp[:, None])) & inside
            ft = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), BIG)
        else:
            tp = entry
            ft = np.full(len(sel), BIG)
        del J, inside, Jc, hit_s
        last = np.minimum(hold_min - 1, (N1 - 1) - e)
        stop_first = fs <= ft  # same bar -> stop first (pessimistic)
        exit_px = np.where(fs < BIG, stop * (1 - side * cost.slippage_bps / 1e4), np.nan)
        exit_px = np.where(stop_first & (fs < BIG), exit_px, np.where(ft < BIG, tp * (1 - side * cost.slippage_bps / 1e4), np.nan))
        k = np.where(stop_first & (fs < BIG), fs, np.where(ft < BIG, ft, last))
        timeout = ~((fs < BIG) | (ft < BIG))
        exit_px = np.where(timeout, c1[np.minimum(e + last, N1 - 1)] * (1 - side * cost.slippage_bps / 1e4), exit_px)
        if end_mult != 1.0:
            cut = timeout & ((N1 - 1) - e < hold_min - 1)
            exit_px = np.where(cut, c1[np.minimum(e + last, N1 - 1)] * (1 - side * end_mult * cost.slippage_bps / 1e4),
                               exit_px)
            if forced is not None:
                forced[sel] = cut
        fees = (entry + exit_px) * cost.taker_fee
        risk = dist + entry * (2 * cost.taker_fee + cost.buffer)
        r_out[sel] = ((exit_px - entry) * side - fees) / risk
        exit_out[sel] = e + k
    return r_out, exit_out


# --- dataset -----------------------------------------------------------------------------------------------------------
def grid_times(frames: dict, grid: str, start: int, end: int) -> np.ndarray:
    """Decision times of a grid inside [start, end]: frame close times, or the 1h close at 00:00 UTC for "1d"."""
    fr = frames["1h" if grid == "1d" else grid]
    ct = fr.close_time
    t = ct[(ct >= start) & (ct <= end)].astype(np.int64)
    if grid == "1d":
        t = t[(t + 1) % DAY_MS == 0]
    return t


def _months(times: np.ndarray) -> np.ndarray:
    if not len(times):
        return np.zeros(0, dtype=np.int64)
    m = times.astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)  # months since 1970-01
    return (1970 + m // 12) * 100 + m % 12 + 1


@dataclass
class SymbolData:
    symbol: str
    times: np.ndarray  # decision times
    tf: str  # decision grid
    X: np.ndarray  # (n, n_features)
    month: np.ndarray  # yyyymm int
    R: dict  # (template_idx, side) -> R array
    EXIT: dict  # (template_idx, side) -> open time of the exit minute (-1: no trade)
    entry_t: np.ndarray  # open time of the entry minute (-1: no clean entry)
    names: list = field(default_factory=lambda: list(FEATURES))
    FORCED: dict = field(default_factory=dict)  # (template_idx, side) -> exit was a delisting close (delisted only)


@dataclass
class Dataset:
    tf: str
    symbols: list[str]
    X: np.ndarray  # column-major (fast per-feature masks)
    times: np.ndarray
    sym_id: np.ndarray
    month: np.ndarray
    R: dict
    EXIT: dict
    entry_t: np.ndarray
    names: list[str] = field(default_factory=lambda: list(FEATURES))
    templates: tuple = (0, 1, 2)
    FORCED: dict = field(default_factory=dict)  # (template_idx, side) -> delisting close (empty: none in the data)
    _excess: dict = field(default_factory=dict)
    _valid: dict = field(default_factory=dict)
    _sorted: bool | None = None

    @property
    def day(self) -> np.ndarray:
        return self.times // DAY_MS

    @property
    def presorted(self) -> bool:
        """Whether the tradable rows are ordered by (symbol, entry), as merge() produces them."""
        if self._sorted is None:
            ok = self.entry_t >= 0
            s, e = self.sym_id[ok], self.entry_t[ok]
            ds_, de = np.diff(s), np.diff(e)
            self._sorted = bool(np.all((ds_ > 0) | ((ds_ == 0) & (de > 0))))
        return self._sorted

    def valid(self, key: tuple) -> np.ndarray:
        """Rows with a simulated trade for (template, side)."""
        if key not in self._valid:
            self._valid[key] = np.isfinite(self.R[key]) & (self.entry_t >= 0)
        return self._valid[key]

    def excess(self, key: tuple) -> np.ndarray:
        """R minus the unconditional mean R of the same (template, side) in the same calendar month, pooled over
        symbols. Rules are ranked on this, so 'be short during a bear month' (market beta) earns nothing."""
        if key not in self._excess:
            R = self.R[key]
            ex = np.full(len(R), np.nan)
            ok = self.valid(key)
            for mo in np.unique(self.month):
                m = (self.month == mo) & ok
                if m.any():
                    ex[m] = R[m] - R[m].mean()
            self._excess[key] = ex
        return self._excess[key]

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.tf, self.symbols, np.asfortranarray(self.X[mask]), self.times[mask], self.sym_id[mask],
                       self.month[mask], {k: v[mask] for k, v in self.R.items()},
                       {k: v[mask] for k, v in self.EXIT.items()}, self.entry_t[mask], self.names, self.templates,
                       {k: v[mask] for k, v in self.FORCED.items()})


def build_symbol(view, base_1m, ms: MetricsSeries | None, tf: str, start: int, end: int,
                 cost: CostModel | None = None, templates=(0, 1, 2), market=None, end_mult: float = 1.0) -> SymbolData:
    """Decision rows for one symbol from a MarketView-like object rebuilt with live=False over the history.
    `market` (a data.context.ContextTable) adds the context features; the names list follows. `end_mult` != 1: the
    1m data ends at the symbol's delisting (simulate_template), and FORCED marks the delisting closes."""
    cost = cost or CostModel()
    times = grid_times(view.frames, tf, start, end)
    feats = features_at(view.frames, ms, times)
    names = list(FEATURES)
    if market is not None:
        feats.update(market.columns(view.symbol, times))
        names = list(FEATURES_X)
    X = np.column_stack([feats[n] for n in names]) if len(times) else np.zeros((0, len(names)))
    o1, h1, l1, c1 = (base_1m.view(k) for k in ("open", "high", "low", "close"))
    ot1 = base_1m.view("open_time")
    entry_idx = np.searchsorted(ot1, times + 1, side="left").astype(np.int64)
    entry_idx[(entry_idx >= len(ot1))] = -1
    good = entry_idx >= 0
    good[good] &= ot1[entry_idx[good]] == times[good] + 1  # the very next minute must exist (no gap)
    entry_idx[~good] = -1
    entry_t = np.where(entry_idx >= 0, ot1[np.maximum(entry_idx, 0)] if len(ot1) else -1, -1).astype(np.int64)
    atr = feats[f"_{GRID_ATR_TF[tf]}.atr"]
    R, EXIT, FORCED = {}, {}, {}
    for ti in templates:
        sl, tp, hold = TEMPLATES[ti]
        for side in (1, -1):
            forced = np.zeros(len(times), dtype=bool) if end_mult != 1.0 else None
            r, ex = simulate_template(o1, h1, l1, c1, entry_idx, atr, side, sl, tp, hold, cost, end_mult=end_mult,
                                      forced=forced)
            R[(ti, side)] = r
            EXIT[(ti, side)] = np.where(ex >= 0, ot1[np.maximum(ex, 0)] if len(ot1) else -1, -1).astype(np.int64)
            if forced is not None:
                FORCED[(ti, side)] = forced
    return SymbolData(view.symbol, times, tf, X, _months(times), R, EXIT, entry_t, names, FORCED)


def merge(parts: list) -> Dataset:
    """Concatenate (symbol, chunk) parts into one column-major dataset, releasing each part once copied (peak memory
    is the dataset plus one part). Parts of the same symbol share a symbol id."""
    parts = [p for p in parts if p is not None and len(p.times)]
    if not parts:
        raise RuntimeError("no data to build a discovery dataset")
    tf, names = parts[0].tf, parts[0].names
    syms = list(dict.fromkeys(p.symbol for p in parts))
    sid = {s: i for i, s in enumerate(syms)}
    n = sum(len(p.times) for p in parts)
    keys = list(parts[0].R.keys())
    X = np.empty((n, len(names)), order="F")
    times = np.empty(n, dtype=np.int64)
    sym_id = np.empty(n, dtype=np.int64)
    month = np.empty(n, dtype=np.int64)
    entry = np.empty(n, dtype=np.int64)
    R = {k: np.empty(n) for k in keys}
    EXIT = {k: np.empty(n, dtype=np.int64) for k in keys}
    FORCED = {k: np.zeros(n, dtype=bool) for k in keys} if any(p.FORCED for p in parts) else {}
    pos = 0
    for i in range(len(parts)):
        p = parts[i]
        m = len(p.times)
        X[pos:pos + m] = p.X
        times[pos:pos + m] = p.times
        sym_id[pos:pos + m] = sid[p.symbol]
        month[pos:pos + m] = p.month
        entry[pos:pos + m] = p.entry_t
        for k in keys:
            R[k][pos:pos + m] = p.R[k]
            EXIT[k][pos:pos + m] = p.EXIT[k]
            if FORCED and k in p.FORCED:
                FORCED[k][pos:pos + m] = p.FORCED[k]
        pos += m
        parts[i] = None
    tmpls = tuple(sorted({k[0] for k in keys}))
    return Dataset(tf, syms, X, times, sym_id, month, R, EXIT, entry, list(names), tmpls, FORCED)


# --- rules -------------------------------------------------------------------------------------------------------------
@dataclass
class Rule:
    side: int
    tmpl: int
    conds: tuple  # ((feature_index, op, threshold), ...) op: '<' or '>'
    tf: str = "15m"

    def key(self) -> tuple:
        return (self.side, self.tmpl, self.tf, tuple(sorted(self.conds)))

    def mask(self, X: np.ndarray) -> np.ndarray:
        m = np.ones(len(X), dtype=bool)
        for fi, op, thr in self.conds:
            col = X[:, fi]
            m &= (col < thr) if op == "<" else (col > thr)
        return m

    def to_dict(self, names: list[str] = FEATURES, stats: dict | None = None) -> dict:
        sl, tp, hold = TEMPLATES[self.tmpl]
        d = {"side": "LONG" if self.side > 0 else "SHORT", "tf": self.tf, "sl_atr": sl, "tp_r": tp, "hold_min": hold,
             "conds": [[names[fi], op, round(float(thr), 6)] for fi, op, thr in self.conds]}
        d["id"] = "r" + hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
        if stats:
            d["stats"] = stats
        return d


def template_index(sl_atr: float, tp_r, hold_min: int) -> int:
    for i, (sl, tp, hold) in enumerate(TEMPLATES):
        same_tp = (tp is None and not tp_r) or (tp is not None and tp_r is not None and abs(tp - float(tp_r)) < 1e-9)
        if abs(sl - float(sl_atr)) < 1e-9 and same_tp and hold == int(hold_min):
            return i
    return 0


def rule_from_dict(d: dict, names: list[str] = FEATURES_X) -> Rule:
    idx = {n: i for i, n in enumerate(names)}
    conds = tuple((idx[c[0]], c[1], float(c[2])) for c in d["conds"])
    return Rule(1 if d["side"] == "LONG" else -1, template_index(d["sl_atr"], d.get("tp_r"), d["hold_min"]), conds,
                d.get("tf", "15m"))


def take_non_overlapping(rows: np.ndarray, exit_t: np.ndarray, entry_t: np.ndarray, sym_id: np.ndarray,
                         presorted: bool = False) -> np.ndarray:
    """Greedy first-come selection per symbol: a rule holds at most one position per symbol at a time (the next entry
    must start after the previous exit). Entries and exits are times (or any increasing index); `presorted` says the
    rows are already ordered by (symbol, entry).

    The greedy chain p -> next(p) is followed with pointer doubling (log2(trades) vectorised steps) instead of one
    Python step per trade: the k-th kept trade is next^k(first)."""
    if len(rows) == 0:
        return rows
    r = rows if presorted else rows[np.lexsort((entry_t[rows], sym_id[rows]))]
    s = np.asarray(sym_id[r], dtype=np.int64)
    ent = np.asarray(entry_t[r], dtype=np.int64)
    ex = np.asarray(exit_t[r], dtype=np.int64)
    span = int(max(ent.max(), ex.max())) + 1
    base = s - s.min()
    key = base * span + ent  # sorted: symbol blocks, entries ascending inside
    n = len(r)
    # first position whose entry is after this exit; past the block it lands on the next symbol's first entry
    jump = np.append(np.searchsorted(key, base * span + ex, side="right"), n)  # node n: past the end (absorbing)
    chain = np.zeros(1, dtype=np.int64)  # next^k(0) for k < 2^i
    parts = [chain]
    while True:
        step = jump[chain]
        step = step[step < n]
        if not len(step):
            break
        parts.append(step)
        chain = np.concatenate((chain, step))
        jump = jump[jump]
    return np.sort(r[np.concatenate(parts)])


def _clustered_t(values: np.ndarray, clusters: np.ndarray) -> tuple[float, int]:
    """t-statistic of the mean computed on cluster (calendar-day) sums: trades on correlated coins at the same time
    are not independent evidence, so the effective sample size is the number of trading days, not trades."""
    if len(values) < 2:
        return -99.0, 0
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    k = len(uniq)
    if k < 5:
        return -99.0, k
    # cluster-robust t for the per-trade mean: mean / sqrt(sum_c (sum_c - n_c*mean)^2) * ...
    mean = values.mean()
    resid = sums - counts * mean
    var = (resid ** 2).sum() * k / max(k - 1, 1) / (len(values) ** 2)
    if var <= 0:
        return -99.0, k
    return float(mean / math.sqrt(var)), k


def rule_stats(ds: Dataset, rule: Rule, min_trades: int = 1) -> dict:
    key = (rule.tmpl, rule.side)
    m = rule.mask(ds.X) & ds.valid(key)
    rows = take_non_overlapping(np.flatnonzero(m), ds.EXIT[key], ds.entry_t, ds.sym_id, ds.presorted)
    n = len(rows)
    if n < max(min_trades, 2):
        return {"n": n, "avg_r": 0.0, "t": -99.0, "t_ex": -99.0, "avg_ex": 0.0, "pf": 0.0}
    r = ds.R[key][rows]
    ex = ds.excess(key)[rows]
    days = ds.times[rows] // DAY_MS
    delist = {"delist": int(ds.FORCED[key][rows].sum())} if key in ds.FORCED else {}
    t_raw, ndays = _clustered_t(r, days)
    t_ex, _ = _clustered_t(ex, days)
    gw, gl = float(r[r > 0].sum()), float(-r[r <= 0].sum())
    syms = ds.sym_id[rows]
    s_sum = np.bincount(syms, weights=ex)
    s_cnt = np.bincount(syms)
    has = s_cnt > 0
    sym_pos = int((s_sum[has] > 0).sum())
    _, mo_inv = np.unique(ds.month[rows], return_inverse=True)
    m_sum = np.bincount(mo_inv, weights=ex)
    m_cnt = np.bincount(mo_inv)
    full = m_cnt >= 3
    mpos, mtot = int((m_sum[full] > 0).sum()), int(full.sum())
    # stability: thirds of the window in time, measured on excess R
    eo = ex[np.argsort(ds.times[rows], kind="stable")]
    thirds = [eo[int(len(eo) * a / 3):int(len(eo) * (a + 1) / 3)] for a in range(3)]
    thirds_pos = sum(1 for t_ in thirds if len(t_) and t_.mean() > 0)
    return {"n": n, "days": int(ndays), "avg_r": float(r.mean()), "avg_ex": float(ex.mean()), "t": t_raw, "t_ex": t_ex,
            "pf": gw / gl if gl > 0 else 99.0, "win": float((r > 0).mean()), "sym_pos": sym_pos,
            "syms": int(has.sum()), "months_pos": mpos, "months": mtot, "thirds_pos": int(thirds_pos), **delist}


# --- search ------------------------------------------------------------------------------------------------------------
@dataclass
class SearchConfig:
    min_trades: int = 150
    beam: int = 40
    depth: int = 3
    max_rules: int = 8
    max_overlap: float = 0.5  # Jaccard of entry rows between kept rules
    min_sym_frac: float = 0.55
    min_month_frac: float = 0.6


def thresholds(ds: Dataset) -> list[tuple[int, str, float]]:
    conds = []
    for fi, name in enumerate(ds.names):
        col = ds.X[:, fi]
        col = col[np.isfinite(col)]
        if len(col) < 500 or np.nanstd(col) == 0:
            continue
        for q, v in zip(QUANTILES, np.quantile(col, QUANTILES)):
            v = float(np.round(v, 6))
            conds.append((fi, "<" if q < 0.5 else ">", v))
    # drop duplicates
    return sorted(set(conds))


def _score(st: dict, cfg: SearchConfig) -> float:
    if st["n"] < cfg.min_trades or st.get("avg_r", 0.0) <= 0:
        return -99.0
    s = min(st.get("t_ex", -99.0), st.get("t", -99.0) + 1.0)
    s -= 1.0 * max(0, 3 - st.get("thirds_pos", 3))
    if st.get("syms"):
        s -= 1.5 * max(0.0, cfg.min_sym_frac - st["sym_pos"] / max(st["syms"], 1))
    if st.get("months"):
        s -= 1.5 * max(0.0, cfg.min_month_frac - st["months_pos"] / max(st["months"], 1))
    return s


def search(ds: Dataset, cfg: SearchConfig | None = None, progress=None) -> tuple[list[tuple[Rule, dict]], int]:
    """Beam search over rule conjunctions on `ds` (TRAIN). Returns (ranked diverse rules with stats, rules evaluated):
    every distinct rule whose statistics were computed counts, whatever its trade count, side or template."""
    cfg = cfg or SearchConfig()
    cand_conds = thresholds(ds)
    tested = 0
    seen: set = set()
    beam: list[tuple[float, Rule, dict]] = []
    # depth 1: every single condition x side x template
    for side in (1, -1):
        for ti in ds.templates:
            for c in cand_conds:
                rule = Rule(side, ti, (c,), ds.tf)
                st = rule_stats(ds, rule, cfg.min_trades)
                tested += 1
                seen.add(rule.key())
                beam.append((_score(st, cfg), rule, st))
    beam.sort(key=lambda x: -x[0])
    pool = list(beam[: cfg.beam * 4])
    frontier = beam[: cfg.beam]
    if progress:
        progress(f"{ds.tf} depth 1: {tested} rules" if frontier else f"{ds.tf} depth 1: none")
    for depth in range(2, cfg.depth + 1):
        nxt = []
        for _, rule, _ in frontier:
            used = {fi for fi, _, _ in rule.conds}
            for c in cand_conds:
                if c[0] in used:
                    continue
                r2 = Rule(rule.side, rule.tmpl, rule.conds + (c,), ds.tf)
                k = r2.key()
                if k in seen:
                    continue
                seen.add(k)
                st = rule_stats(ds, r2, cfg.min_trades)
                tested += 1
                nxt.append((_score(st, cfg), r2, st))
        nxt.sort(key=lambda x: -x[0])
        frontier = nxt[: cfg.beam]
        pool += nxt[: cfg.beam * 2]
        if progress:
            progress(f"{ds.tf} depth {depth}: {tested} rules evaluated")
    pool.sort(key=lambda x: -x[0])
    kept: list[tuple[Rule, dict, np.ndarray]] = []
    for sc, rule, st in pool:
        if sc <= 0 or len(kept) >= cfg.max_rules:
            continue
        rows = rule.mask(ds.X) & ds.valid((rule.tmpl, rule.side))
        n_rows = int(rows.sum())
        if any(np.count_nonzero(rows & k[2]) / max(1, n_rows + int(k[2].sum()) - np.count_nonzero(rows & k[2]))
               > cfg.max_overlap for k in kept):
            continue
        kept.append((rule, st, rows))
    return [(r, s) for r, s, _ in kept], tested


def multiple_testing_t(n_tested: int) -> float:
    """Approximate family-wise bar for the best of N (roughly Gaussian) t-statistics."""
    return max(3.0, math.sqrt(2.0 * math.log(max(n_tested, 2))))


def validate(train_rules: list[tuple[Rule, dict]], valid: Dataset, n_tested: int, min_valid_trades: int = 40,
             min_valid_avg_r: float = 0.03, min_valid_t: float = 1.5) -> list[dict]:
    """Keep rules that clear the multiple-testing bar on TRAIN (day-clustered t of market-excess R) and hold up on
    the untouched VALID window (positive raw R with day-clustered t >= min_valid_t, and positive excess R)."""
    bar = multiple_testing_t(n_tested)
    out = []
    for rule, st in train_rules:
        sv = rule_stats(valid, rule, 1)
        passed = (st.get("t_ex", -99) >= bar and st.get("t", -99) >= 2.0 and sv["n"] >= min_valid_trades
                  and sv["avg_r"] >= min_valid_avg_r and sv["t"] >= min_valid_t and sv.get("avg_ex", -1) > 0)
        out.append({"rule": rule.to_dict(valid.names, {"train": st, "valid": sv, "t_bar": bar}), "passed": bool(passed)})
    return out


def rule_features_at(view, ms: MetricsSeries | None, t: int) -> np.ndarray:
    """Feature vector at decision time t for online evaluation (same code path as mining)."""
    feats = features_at(view.frames, ms, np.array([t], dtype=np.int64))
    return np.array([feats[n][0] for n in FEATURES])


def decision_features(view, extras: dict | None, market, t: int, names: list[str] | None = None) -> np.ndarray:
    """One decision's feature vector over `names` (default FEATURES_X), computed exactly as build_symbol() computes the
    row at time t, from what an alpha is given: the view's 15m / 1h frames (frame_features), ctx.extras (the
    MetricsSeries.snapshot() dict, whose fields extras_features() reproduces bit for bit; empty or stale: NaN),
    t itself (time_features) and ctx.market (the snapshot of the last hour close at or before t, the row
    ContextTable.columns() looks up; c.* the symbol's own features, m.* the market-wide ones, NaN when missing)."""
    names = FEATURES_X if names is None else names
    feats = features_at(view.frames, None, np.array([t], dtype=np.int64))
    out = np.full(len(names), np.nan)
    ex = extras if extras and not extras.get("stale") else None
    mk = market or {}
    for i, n in enumerate(names):
        if n.startswith("x."):
            v = ex.get(n[2:]) if ex is not None else None
        elif n.startswith(("c.", "m.")):
            v = mk.get(n[2:])
        else:
            col = feats.get(n)
            v = col[0] if col is not None else None
        out[i] = float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else np.nan
    return out


# --- orchestration -----------------------------------------------------------------------------------------------------
class _LightView:
    """Only the decision-timeframe frames (enough for mining; online evaluation uses the bot's full MarketView)."""

    def __init__(self, symbol: str, frames: dict):
        self.symbol = symbol
        self.frames = frames


def _build_job(job: dict) -> SymbolData | None:
    """One (symbol, chunk) of a dataset in a worker process: memory is bounded by one chunk of 1m history."""
    import logging

    logging.disable(logging.WARNING)
    from heartless.core.store import Store
    from heartless.data.candles import CandleArrays, resample
    from heartless.data.features import FeatureFrame
    from heartless.learning import seal

    seal.check_window(job["start"], job["end"], job.get("unseal", ()))
    store = Store(job["db_path"])
    sym, tf = job["symbol"], job["tf"]
    templates = tuple(job.get("templates", (0, 1, 2)))
    hold = max(TEMPLATES[t][2] for t in templates)
    # exits of the last decisions are simulated past the chunk, but never inside a sealed period
    fwd = seal.readable_until(job["end"], job["end"] + hold * 60_000 + 3_600_000, job.get("unseal", ()))
    delist = job.get("delist_close")
    end_mult = 1.0
    if delist is not None and int(delist) < fwd:
        fwd, end_mult = int(delist), DELIST_SLIP_MULT  # nothing trades after the last traded hour
    rows = store.load_candles(sym, start=job["start"] - job.get("warmup_ms", WARMUP_MS), end=fwd)
    mrows = store.load_metrics(sym, job["start"] - 4 * DAY_MS, job["end"]) if hasattr(store, "load_metrics") else []
    store.close()
    if len(rows) < 5000:
        return None
    ca = CandleArrays("1m", capacity=len(rows) + 16)
    ca.extend(rows)
    del rows
    frames = {t: FeatureFrame(t, resample(ca, t)) for t in DECISION_TFS}
    ms = MetricsSeries.from_rows(mrows) if mrows else None
    market = None
    if job.get("context_db"):
        from heartless.data.context import MS_H, ContextTable

        cst = Store(job["context_db"])
        market = ContextTable.load(cst, [sym], job["start"] - MS_H, job["end"], source=job["db_path"])
        cst.close()
        if market is None:  # asked for context features but the table is gone: keep the column layout (all NaN)
            market = ContextTable({}, {})
    cost = CostModel(**job.get("cost", {}))
    return build_symbol(_LightView(sym, frames), ca, ms, tf, job["start"], min(job["end"], fwd), cost, templates,
                        market, end_mult)


def year_chunks(start: int, end: int) -> list[tuple[int, int]]:
    """Calendar-year pieces of [start, end]: the dataset build unit."""
    out = []
    y = datetime.fromtimestamp(start / 1000, tz=timezone.utc).year
    a = int(start)
    while a <= end:
        nxt = int(datetime(y + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        out.append((a, min(int(end), nxt - 1)))
        a, y = nxt, y + 1
    return out


GRID_STEP = {"15m": 900_000, "1h": 3_600_000, "1d": DAY_MS}


def grid_bound(grid: str, a: int, b: int) -> int:
    """Upper bound of a grid's decision times in [a, b] (closes fall at k * step - 1)."""
    step = GRID_STEP[grid]
    return max(0, (b + 1) // step - (-(-(a + 1) // step)) + 1)


class _Assembler:
    """Fills (symbol, chunk) parts into preallocated column-major arrays as they arrive (each part sits at the offset
    of its job, sized by grid_bound) and compacts the gaps at the end, so the build never holds all parts and the
    merged copy at once: peak memory is the dataset plus the parts in flight."""

    def __init__(self, jobs: list[dict], names: list[str], keys: list[tuple]):
        self.jobs, self.names, self.keys = jobs, names, keys
        self.forced = any(j.get("delist_close") is not None for j in jobs)
        self.bounds = [grid_bound(j["tf"], j["start"], j["end"]) for j in jobs]
        self.offsets = np.concatenate(([0], np.cumsum(self.bounds))).astype(np.int64)
        n = int(self.offsets[-1])
        self.X = np.empty((n, len(names)), order="F")
        self.times = np.zeros(n, dtype=np.int64)
        self.month = np.zeros(n, dtype=np.int64)
        self.entry = np.full(n, -1, dtype=np.int64)
        self.R = {k: np.empty(n) for k in keys}
        self.EXIT = {k: np.empty(n, dtype=np.int64) for k in keys}
        self.FORCED = {k: np.zeros(n, dtype=bool) for k in keys} if self.forced else {}
        self.counts = [0] * len(jobs)

    def put(self, i: int, part: SymbolData | None) -> None:
        if part is None or not len(part.times):
            return
        m, a = len(part.times), int(self.offsets[i])
        if m > self.bounds[i] or part.names != self.names:
            raise RuntimeError(f"unexpected dataset part for {part.symbol}: {m} rows (bound {self.bounds[i]})")
        self.X[a:a + m] = part.X
        self.times[a:a + m] = part.times
        self.month[a:a + m] = part.month
        self.entry[a:a + m] = part.entry_t
        for k in self.keys:
            self.R[k][a:a + m] = part.R[k]
            self.EXIT[k][a:a + m] = part.EXIT[k]
            if self.forced:
                self.FORCED[k][a:a + m] = part.FORCED[k] if k in part.FORCED else False
        self.counts[i] = m

    def dataset(self, tf: str, templates: tuple) -> Dataset:
        syms = list(dict.fromkeys(j["symbol"] for j, c in zip(self.jobs, self.counts) if c))
        if not syms:
            raise RuntimeError("no data to build a discovery dataset")
        sid = {s: i for i, s in enumerate(syms)}
        sym_id = np.zeros(len(self.times), dtype=np.int64)
        dst = 0
        for i, (job, m) in enumerate(zip(self.jobs, self.counts)):
            if not m:
                continue
            a = int(self.offsets[i])
            if a != dst:  # close the gap left by a shorter part (numpy handles the overlapping move)
                self.X[dst:dst + m] = self.X[a:a + m]
                for arr in (self.times, self.month, self.entry, *self.R.values(), *self.EXIT.values(),
                            *self.FORCED.values()):
                    arr[dst:dst + m] = arr[a:a + m]
            sym_id[dst:dst + m] = sid[job["symbol"]]
            dst += m
        return Dataset(tf, syms, self.X[:dst], self.times[:dst].copy(), sym_id[:dst].copy(), self.month[:dst].copy(),
                       {k: v[:dst].copy() for k, v in self.R.items()}, {k: v[:dst].copy() for k, v in self.EXIT.items()},
                       self.entry[:dst].copy(), list(self.names), templates,
                       {k: v[:dst].copy() for k, v in self.FORCED.items()})


def member_job_windows(spans, start: int, end: int) -> list[tuple[int, int]]:
    """Decision-time ranges of a symbol's membership spans inside [start, end], cut at calendar years: a decision at
    the close t enters at t + 1, so a span [a, b] holds the decisions in [a - 1, b - 1] (the close before the span's
    first minute is in, the span's own last close, which enters after it, is out)."""
    out = []
    for a, b in sorted(spans):
        a, b = max(int(a), int(start)), min(int(b), int(end))
        if a > b:
            continue
        for ca, cb in year_chunks(a, b):
            out.append((ca - 1, cb - 1))
    return out


def build_dataset(db_path: str, symbols: list[str], start: int, end: int, tf: str = "15m", workers: int = 2,
                  cost: dict | None = None, unseal=(), templates="classic", context_db: str | None = None,
                  windows: dict | None = None, delisted: dict | None = None) -> Dataset:
    """Decision rows of [start, end] for every symbol, built as (symbol, calendar year) jobs. Raises SealedError for
    a window touching a sealed period. `context_db` adds the market-context features (FEATURES_X).

    `windows` = {symbol: membership spans} (a topN universe, docs/ROADMAP.md 7.4) keeps only the decisions whose entry
    falls in a span: one job per (symbol, span x calendar year), so non-member months are never built (a symbol without
    a span has no rows). `delisted` = {symbol: last traded hour close}: that symbol's data ends there and an exit cut
    short by it is a delisting close (5x slippage, FORCED)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from heartless.learning import seal

    if tf not in GRIDS:
        raise ValueError(f"unknown decision grid {tf!r} (one of {', '.join(GRIDS)})")
    seal.check_window(int(start), int(end), unseal)
    unseal = seal.opened(int(start), int(end), unseal)  # exit simulation may only read into a period this window opens
    tmpls = TEMPLATE_SETS[templates] if isinstance(templates, str) else tuple(templates)
    if windows is None:
        ranges = {s: year_chunks(int(start), int(end)) for s in symbols}
    else:
        ranges = {s: member_job_windows(windows.get(s, ()), int(start), int(end)) for s in symbols}
    jobs = [{"db_path": db_path, "symbol": s, "start": a, "end": b, "tf": tf, "cost": cost or {},
             "unseal": sorted(unseal or ()), "templates": tmpls, "context_db": context_db}
            for s in symbols for a, b in ranges[s]]
    if delisted:  # a job whose decisions all come after the symbol's last traded hour has nothing to enter
        jobs = [j for j in jobs if j["symbol"] not in delisted or j["start"] < int(delisted[j["symbol"]])]
    for j in jobs:
        if delisted and j["symbol"] in delisted:
            j["delist_close"] = int(delisted[j["symbol"]])
    if not jobs:
        raise RuntimeError("no data to build a discovery dataset (no symbol has a membership span in the window "
                           "before its last traded hour)")
    asm = _Assembler(jobs, list(FEATURES_X if context_db else FEATURES), [(t, sd) for t in tmpls for sd in (1, -1)])
    if workers <= 1:
        for i, j in enumerate(jobs):
            asm.put(i, _build_job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                 max_tasks_per_child=1 if len(jobs) > len(symbols) else None) as pool:
            futs = {pool.submit(_build_job, j): i for i, j in enumerate(jobs)}
            # as_completed drops its own reference to a yielded future and the dict entry is popped, so each part is
            # freed once it has been copied in
            for fut in as_completed(futs):
                asm.put(futs.pop(fut), fut.result())
                del fut
    return asm.dataset(tf, tmpls)


def parse_grids(tf) -> list[str]:
    out = [g.strip() for g in (tf.split(",") if isinstance(tf, str) else tf) if g.strip()]
    bad = [g for g in out if g not in GRIDS]
    if bad or not out:
        raise ValueError(f"unknown decision grid(s) {bad or tf!r} (choose from {', '.join(GRIDS)})")
    return list(dict.fromkeys(out))


def mine(db_path: str, symbols: list[str], train: tuple[int, int], valid: tuple[int, int], tf: str = "15m",
         cfg: SearchConfig | None = None, workers: int = 2, progress=None, unseal=(), templates="classic",
         context: dict | None = None, n_prior: int = 0, member: dict | None = None) -> dict:
    """Full discovery run: build TRAIN/VALID datasets per grid, beam-search on TRAIN, validate on VALID.

    `tf` is one grid or a comma list ("1h,1d"); the multiple-testing bar uses every rule evaluated over all grids
    plus `n_prior` earlier trials of the family. `context` = {"db", "universe", "build"} adds the market-context
    features (built on demand). Both windows pass the seal check before any data is read (SealedError unless the
    period is in `unseal`). `member` = {"train": {"windows", "delisted"}, "valid": {...}} (a topN universe) keeps
    only member decision rows in each window and closes delisted symbols at their last traded hour (build_dataset);
    the result then reports member-months and delisting closes under "membership"."""
    import time as _t

    from heartless.learning import seal

    seal.check_window(int(train[0]), int(train[1]), unseal)
    seal.check_window(int(valid[0]), int(valid[1]), unseal)
    grids = parse_grids(tf)
    t0 = _t.time()
    ctx_db = None
    if context:
        from heartless.data import context as C

        # each window plus the hour close before it (decisions before its first hour close read that snapshot); two
        # separate builds, so the gap between TRAIN and VALID (the sealed year) is never computed
        ctx_db = (C.prepare(db_path, context, train[0] - C.MS_H, train[1])
                  and C.prepare(db_path, context, valid[0] - C.MS_H, valid[1]))
        if ctx_db is None:
            raise RuntimeError("market context requested but not available (run `heartless context build`)")
    tested, results, rules_all, rows = 0, [], [], {}
    sets = {}
    mem_kw = {"train": {}, "valid": {}}
    if member is not None:
        for phase in ("train", "valid"):
            m = member.get(phase) or {}
            mem_kw[phase] = {"windows": m.get("windows") or {}, "delisted": m.get("delisted") or {}}
    delist_rows: dict = {}
    for g in grids:
        ds_tr = build_dataset(db_path, symbols, train[0], train[1], g, workers, unseal=unseal, templates=templates,
                              context_db=ctx_db, **mem_kw["train"])
        ds_va = build_dataset(db_path, symbols, valid[0], valid[1], g, workers, unseal=unseal, templates=templates,
                              context_db=ctx_db, **mem_kw["valid"])
        rows[g] = {"train": int(len(ds_tr.times)), "valid": int(len(ds_va.times))}
        if member is not None:  # decision rows (any template, any side) whose exit was a delisting close
            delist_rows[g] = {ph: int(np.logical_or.reduce(list(ds.FORCED.values())).sum()) if ds.FORCED else 0
                              for ph, ds in (("train", ds_tr), ("valid", ds_va))}
        if progress:
            progress(f"{g} datasets: train {rows[g]['train']} rows, valid {rows[g]['valid']} rows ({_t.time() - t0:.0f}s)")
        rules, n = search(ds_tr, cfg, progress)
        tested += n
        sets[g] = (rules, ds_va)
        del ds_tr
    n_tests = tested + max(0, int(n_prior))
    for g in grids:
        rules, ds_va = sets[g]
        results += validate(rules, ds_va, n_tests)
        rules_all += rules
    if len(grids) > 1:  # one grid keeps the search ranking; several are merged by TRAIN excess t
        results.sort(key=lambda r: -r["rule"]["stats"]["train"].get("t_ex", -99.0))
    best = max(rules_all, key=lambda rs: rs[1].get("t_ex", -99.0))[1] if rules_all else None
    flat_rows = rows[grids[0]] if len(grids) == 1 else {"train": sum(r["train"] for r in rows.values()),
                                                         "valid": sum(r["valid"] for r in rows.values()), **rows}
    out = {"tf": ",".join(grids), "tested": tested, "n_tests": n_tests, "t_bar": multiple_testing_t(n_tests),
           "results": results, "passed": [r["rule"] for r in results if r["passed"]], "seconds": _t.time() - t0,
           "rows": flat_rows, "templates": templates if isinstance(templates, str) else list(templates),
           "context": bool(context), "best_train": best}
    if member is not None:
        from heartless.data.universe import member_months

        out["membership"] = {
            ph: {"member_months": member_months(mem_kw[ph]["windows"]),
                 "member_months_total": sum(member_months(mem_kw[ph]["windows"]).values()),
                 "delisted": {s: _iso_ms(v) for s, v in sorted(mem_kw[ph]["delisted"].items())}}
            for ph in ("train", "valid")}
        out["membership"]["delisting_close_rows"] = delist_rows
    return out


def _iso_ms(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")

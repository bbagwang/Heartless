"""Alpha discovery: mine rule-based alphas from a library of scale-free features, with honest statistics.

Pipeline
1. `build_dataset()` aligns, for every decision bar (15m or 1h close) of every symbol, a vector of scale-free features
   (oscillators, distances in ATR units, volatility ranks, flow, positioning from 5-minute metrics, time of day) and
   the realised outcome of a fixed exit template (stop k*ATR, target r*stop, timeout) simulated on 1-minute bars
   with the production cost model (entry at the next bar's open, taker fees, slippage, pessimistic stop-first fills).
2. `search()` explores conjunctions of 1-3 threshold conditions (quantile cut points learned on TRAIN only) by
   beam search, ranks rules by the t-statistic of their R-multiples after removing overlapping trades, and keeps a
   diverse set (low entry overlap).
3. `validate()` re-measures the survivors on a later, untouched window. Because thousands of rules were tried, a
   rule must clear a multiple-testing-aware bar on TRAIN (t >= max(3, sqrt(2 ln N))) *and* hold up on VALID.

The very same feature code evaluates a rule online (`rule_features_at()`), so a discovered rule behaves identically in
the live bot, in the lab and in this miner. Survivors are shipped as data inside StrategyParams
(alphas["discovered"]["rules"]) and traded by the `discovered` alpha; the champion/challenger loop decides whether a
rule set ever reaches real money.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from heartless.data.extras import AVAILABLE_AFTER_MS, MetricsSeries

DECISION_TFS = ("15m", "1h")
TF_MS = {"15m": 900_000, "1h": 3_600_000}
# exit templates: (stop in ATR of the decision timeframe, target in R, timeout in minutes)
TEMPLATES: tuple[tuple[float, float, int], ...] = ((1.0, 1.5, 240), (1.5, 2.0, 480), (2.0, 3.0, 720))
QUANTILES = (0.05, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.95)

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


FEATURES = feature_names()


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
    """Vectorised twin of MetricsSeries.snapshot() for the fields used by rules (identical semantics)."""
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
    # geometric mean of the last 12 taker ratios
    tk = ms.taker_ls_vol
    logt = np.where(np.isfinite(tk) & (tk > 0), np.log(np.where(tk > 0, tk, 1.0)), 0.0)
    cnt = (np.isfinite(tk) & (tk > 0)).astype(float)
    cs_l = np.concatenate([[0.0], np.cumsum(logt)])
    cs_c = np.concatenate([[0.0], np.cumsum(cnt)])
    lo = np.maximum(0, iv - 11)
    num = cs_l[iv + 1] - cs_l[lo]
    den = cs_c[iv + 1] - cs_c[lo]
    g = np.full(len(iv), np.nan)
    okd = den > 0
    g[okd] = np.exp(num[okd] / den[okd])
    out["x.taker_ratio_1h"][valid] = g
    # z-score of the 1h OI change vs the 12-row changes over the last <=864 rows (as in snapshot)
    oi_all = ms.oi
    chg_rows = np.full(len(oi_all), np.nan)
    if len(oi_all) > 12:
        chg_rows[12:] = oi_all[12:] / oi_all[:-12] - 1.0
    fin = np.isfinite(chg_rows)
    x = np.where(fin, chg_rows, 0.0)
    cs1 = np.concatenate([[0.0], np.cumsum(x)])
    cs2 = np.concatenate([[0.0], np.cumsum(x * x)])
    csn = np.concatenate([[0.0], np.cumsum(fin.astype(float))])
    lo_seg = np.maximum(0, iv - 864)
    start = lo_seg + 12
    seg_len = iv - lo_seg + 1
    a = np.minimum(start, iv + 1)
    cnt_c = csn[iv + 1] - csn[a]
    s1 = cs1[iv + 1] - cs1[a]
    s2 = cs2[iv + 1] - cs2[a]
    z = np.full(len(iv), np.nan)
    ok = (seg_len > 60) & (cnt_c > 30) & np.isfinite(chg["1h"])
    mean = np.where(cnt_c > 0, s1 / np.maximum(cnt_c, 1), 0.0)
    var = np.where(cnt_c > 0, s2 / np.maximum(cnt_c, 1) - mean * mean, 0.0)
    sd = np.sqrt(np.maximum(var, 0.0))
    ok &= sd > 0
    z[ok] = (chg["1h"][ok] - mean[ok]) / sd[ok]
    out["x.oi_chg_1h_z"][valid] = z
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
                      atr: np.ndarray, side: int, sl_atr: float, tp_r: float, hold_min: int, cost: CostModel,
                      chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """R multiple and exit 1m-index for entries at the open of 1m bar `entry_idx` (vectorised, pessimistic).

    R is net of fees/slippage and expressed against the engine's risk unit (stop distance + round-trip costs)."""
    n = len(entry_idx)
    r_out = np.full(n, np.nan)
    exit_out = np.full(n, -1, dtype=np.int64)
    N1 = len(o1)
    slip = (cost.slippage_bps + cost.half_spread_bps) / 1e4
    ok_all = (entry_idx >= 0) & (entry_idx < N1) & np.isfinite(atr) & (atr > 0)
    steps = np.arange(hold_min)
    for s in range(0, n, chunk):
        sel = np.arange(s, min(n, s + chunk))
        sel = sel[ok_all[sel]]
        if len(sel) == 0:
            continue
        e = entry_idx[sel]
        entry = o1[e] * (1 + side * slip)
        dist = sl_atr * atr[sel]
        stop = entry - side * dist
        tp = entry + side * tp_r * dist
        J = e[:, None] + steps[None, :]
        inside = J < N1
        Jc = np.minimum(J, N1 - 1)
        lo = l1[Jc]
        hi = h1[Jc]
        if side > 0:
            hit_s = (lo <= stop[:, None]) & inside
            hit_t = (hi >= tp[:, None]) & inside
        else:
            hit_s = (hi >= stop[:, None]) & inside
            hit_t = (lo <= tp[:, None]) & inside
        BIG = hold_min + 10
        fs = np.where(hit_s.any(axis=1), hit_s.argmax(axis=1), BIG)
        ft = np.where(hit_t.any(axis=1), hit_t.argmax(axis=1), BIG)
        last = np.minimum(hold_min - 1, (N1 - 1) - e)
        stop_first = fs <= ft  # same bar -> stop first (pessimistic)
        exit_px = np.where(fs < BIG, stop * (1 - side * cost.slippage_bps / 1e4), np.nan)
        exit_px = np.where(stop_first & (fs < BIG), exit_px, np.where(ft < BIG, tp * (1 - side * cost.slippage_bps / 1e4), np.nan))
        k = np.where(stop_first & (fs < BIG), fs, np.where(ft < BIG, ft, last))
        timeout = ~((fs < BIG) | (ft < BIG))
        exit_px = np.where(timeout, c1[np.minimum(e + last, N1 - 1)] * (1 - side * cost.slippage_bps / 1e4), exit_px)
        fees = (entry + exit_px) * cost.taker_fee
        risk = dist + entry * (2 * cost.taker_fee + cost.buffer)
        r_out[sel] = ((exit_px - entry) * side - fees) / risk
        exit_out[sel] = e + k
    return r_out, exit_out


# --- dataset -----------------------------------------------------------------------------------------------------------
@dataclass
class SymbolData:
    symbol: str
    times: np.ndarray  # decision times (tf close times) for each decision tf
    tf: str
    X: np.ndarray  # (n, n_features)
    month: np.ndarray  # yyyymm int
    R: dict  # (template_idx, side) -> R array
    EXIT: dict  # (template_idx, side) -> exit 1m index
    entry_idx: np.ndarray


@dataclass
class Dataset:
    tf: str
    symbols: list[str]
    X: np.ndarray
    times: np.ndarray
    sym_id: np.ndarray
    month: np.ndarray
    R: dict
    EXIT: dict
    entry_idx: np.ndarray
    names: list[str] = field(default_factory=lambda: list(FEATURES))
    _excess: dict = field(default_factory=dict)

    @property
    def day(self) -> np.ndarray:
        return self.times // 86_400_000

    def excess(self, key: tuple) -> np.ndarray:
        """R minus the unconditional mean R of the same (template, side) in the same calendar month, pooled over
        symbols. Rules are ranked on this, so 'be short during a bear month' (market beta) earns nothing."""
        if key not in self._excess:
            R = self.R[key]
            ex = np.full(len(R), np.nan)
            for mo in np.unique(self.month):
                m = (self.month == mo) & np.isfinite(R) & (self.entry_idx >= 0)
                if m.any():
                    ex[m] = R[m] - R[m].mean()
            self._excess[key] = ex
        return self._excess[key]

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(self.tf, self.symbols, self.X[mask], self.times[mask], self.sym_id[mask], self.month[mask],
                       {k: v[mask] for k, v in self.R.items()}, {k: v[mask] for k, v in self.EXIT.items()},
                       self.entry_idx[mask], self.names)


def build_symbol(view, base_1m, ms: MetricsSeries | None, tf: str, start: int, end: int,
                 cost: CostModel | None = None) -> SymbolData:
    """Decision rows for one symbol from a MarketView rebuilt with live=False over the full history."""
    cost = cost or CostModel()
    fr = view.frames[tf]
    ct = fr.close_time
    sel = (ct >= start) & (ct <= end)
    times = ct[sel].astype(np.int64)
    feats = features_at(view.frames, ms, times)
    X = np.column_stack([feats[n] for n in FEATURES]) if len(times) else np.zeros((0, len(FEATURES)))
    o1, h1, l1, c1 = (base_1m.view(k) for k in ("open", "high", "low", "close"))
    ot1 = base_1m.view("open_time")
    entry_idx = np.searchsorted(ot1, times + 1, side="left").astype(np.int64)
    entry_idx[(entry_idx >= len(ot1))] = -1
    good = entry_idx >= 0
    good[good] &= ot1[entry_idx[good]] == times[good] + 1  # the very next minute must exist (no gap)
    entry_idx[~good] = -1
    atr = feats[f"_{tf}.atr"]
    R, EXIT = {}, {}
    for ti, (sl, tp, hold) in enumerate(TEMPLATES):
        for side in (1, -1):
            r, ex = simulate_template(o1, h1, l1, c1, entry_idx, atr, side, sl, tp, hold, cost)
            R[(ti, side)] = r
            EXIT[(ti, side)] = ex
    month = np.array([int(np.datetime64(int(t), "ms").astype("datetime64[M]").astype(str).replace("-", "")) for t in times],
                     dtype=np.int64) if len(times) else np.zeros(0, dtype=np.int64)
    return SymbolData(view.symbol, times, tf, X, month, R, EXIT, entry_idx)


def merge(parts: list[SymbolData]) -> Dataset:
    tf = parts[0].tf
    syms = [p.symbol for p in parts]
    X = np.vstack([p.X for p in parts])
    times = np.concatenate([p.times for p in parts])
    sym_id = np.concatenate([np.full(len(p.times), i, dtype=np.int64) for i, p in enumerate(parts)])
    month = np.concatenate([p.month for p in parts])
    keys = parts[0].R.keys()
    R = {k: np.concatenate([p.R[k] for p in parts]) for k in keys}
    EXIT = {k: np.concatenate([p.EXIT[k] for p in parts]) for k in keys}
    entry = np.concatenate([p.entry_idx for p in parts])
    return Dataset(tf, syms, X, times, sym_id, month, R, EXIT, entry)


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


def rule_from_dict(d: dict, names: list[str] = FEATURES) -> Rule:
    idx = {n: i for i, n in enumerate(names)}
    tmpl = next((i for i, t in enumerate(TEMPLATES) if abs(t[0] - d["sl_atr"]) < 1e-9 and abs(t[1] - d["tp_r"]) < 1e-9
                 and t[2] == int(d["hold_min"])), 0)
    conds = tuple((idx[c[0]], c[1], float(c[2])) for c in d["conds"])
    return Rule(1 if d["side"] == "LONG" else -1, tmpl, conds, d.get("tf", "15m"))


def take_non_overlapping(rows: np.ndarray, exit_idx: np.ndarray, entry_idx: np.ndarray, sym_id: np.ndarray) -> np.ndarray:
    """Greedy first-come selection per symbol: a rule holds at most one position per symbol at a time."""
    if len(rows) == 0:
        return rows
    keep = []
    order = np.lexsort((entry_idx[rows], sym_id[rows]))
    rows = rows[order]
    syms = sym_id[rows]
    for s in np.unique(syms):
        r = rows[syms == s]
        ent = entry_idx[r]
        pos = 0
        while pos < len(r):
            keep.append(r[pos])
            nxt_entry = exit_idx[r[pos]] + 1
            pos = int(np.searchsorted(ent, nxt_entry, side="left"), ) if nxt_entry > ent[pos] else pos + 1
    return np.array(sorted(keep), dtype=np.int64)


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
    m = rule.mask(ds.X)
    R = ds.R[key]
    m &= np.isfinite(R) & (ds.entry_idx >= 0)
    rows = np.flatnonzero(m)
    rows = take_non_overlapping(rows, ds.EXIT[key], ds.entry_idx, ds.sym_id)
    n = len(rows)
    if n < max(min_trades, 2):
        return {"n": n, "avg_r": 0.0, "t": -99.0, "t_ex": -99.0, "avg_ex": 0.0, "pf": 0.0}
    r = R[rows]
    ex = ds.excess(key)[rows]
    days = ds.day[rows]
    t_raw, ndays = _clustered_t(r, days)
    t_ex, _ = _clustered_t(ex, days)
    gw, gl = float(r[r > 0].sum()), float(-r[r <= 0].sum())
    syms = ds.sym_id[rows]
    sym_pos = sum(1 for s_ in np.unique(syms) if ex[syms == s_].mean() > 0)
    months = ds.month[rows]
    mpos = mtot = 0
    for mo in np.unique(months):
        rr = ex[months == mo]
        if len(rr) >= 3:
            mtot += 1
            mpos += rr.mean() > 0
    # stability: thirds of the window in time, measured on excess R
    order = np.argsort(ds.times[rows])
    thirds = [ex[order][i::1][int(len(order) * a / 3):int(len(order) * (a + 1) / 3)] for i, a in ((0, 0), (0, 1), (0, 2))]
    thirds_pos = sum(1 for t_ in thirds if len(t_) and t_.mean() > 0)
    return {"n": n, "days": int(ndays), "avg_r": float(r.mean()), "avg_ex": float(ex.mean()), "t": t_raw, "t_ex": t_ex,
            "pf": gw / gl if gl > 0 else 99.0, "win": float((r > 0).mean()), "sym_pos": int(sym_pos),
            "syms": int(len(np.unique(syms))), "months_pos": int(mpos), "months": int(mtot), "thirds_pos": int(thirds_pos)}


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
    """Beam search over rule conjunctions on `ds` (TRAIN). Returns (ranked diverse rules with stats, rules tested)."""
    cfg = cfg or SearchConfig()
    cand_conds = thresholds(ds)
    tested = 0
    seen: set = set()
    beam: list[tuple[float, Rule, dict]] = []
    # depth 1: every single condition x side x template
    for side in (1, -1):
        for ti in range(len(TEMPLATES)):
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
        progress(f"depth 1: {tested} rules, best t={frontier[0][2]['t']:.2f}" if frontier else "depth 1: none")
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
            progress(f"depth {depth}: {tested} rules tested, best t={frontier[0][2]['t']:.2f}" if frontier else f"depth {depth}: none")
    pool.sort(key=lambda x: -x[0])
    kept: list[tuple[Rule, dict, set]] = []
    for sc, rule, st in pool:
        if sc <= 0 or len(kept) >= cfg.max_rules:
            continue
        rows = set(np.flatnonzero(rule.mask(ds.X) & np.isfinite(ds.R[(rule.tmpl, rule.side)])).tolist())
        if any(len(rows & k[2]) / max(1, len(rows | k[2])) > cfg.max_overlap for k in kept):
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


# --- orchestration -----------------------------------------------------------------------------------------------------
class _LightView:
    """Only the decision-timeframe frames (enough for mining; online evaluation uses the bot's full MarketView)."""

    def __init__(self, symbol: str, frames: dict):
        self.symbol = symbol
        self.frames = frames


def _build_job(job: dict) -> SymbolData | None:
    import logging

    logging.disable(logging.WARNING)
    from heartless.core.store import Store
    from heartless.data.candles import CandleArrays, resample
    from heartless.data.features import FeatureFrame
    from heartless.learning import seal

    seal.check_window(job["start"], job["end"], job.get("unseal", ()))
    store = Store(job["db_path"])
    sym, tf = job["symbol"], job["tf"]
    warm = 14 * 86_400_000
    # exits of the last decisions are simulated up to 13h past the window, but never inside a sealed period
    fwd = seal.readable_until(job["end"], job["end"] + 13 * 3_600_000, job.get("unseal", ()))
    rows = store.load_candles(sym, start=job["start"] - warm, end=fwd)
    mrows = store.load_metrics(sym, job["start"] - 4 * 86_400_000, job["end"]) if hasattr(store, "load_metrics") else []
    store.close()
    if len(rows) < 5000:
        return None
    ca = CandleArrays("1m", capacity=len(rows) + 16)
    ca.extend(rows)
    frames = {t: FeatureFrame(t, resample(ca, t)) for t in DECISION_TFS}
    ms = MetricsSeries.from_rows(mrows) if mrows else None
    cost = CostModel(**job.get("cost", {}))
    return build_symbol(_LightView(sym, frames), ca, ms, tf, job["start"], job["end"], cost)


def build_dataset(db_path: str, symbols: list[str], start: int, end: int, tf: str = "15m", workers: int = 2,
                  cost: dict | None = None, unseal=()) -> Dataset:
    """Decision rows of [start, end] for every symbol. Raises SealedError for a window touching a sealed period."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    from heartless.learning import seal

    seal.check_window(int(start), int(end), unseal)
    unseal = seal.opened(int(start), int(end), unseal)  # exit simulation may only read into a period this window opens
    jobs = [{"db_path": db_path, "symbol": s, "start": int(start), "end": int(end), "tf": tf, "cost": cost or {},
             "unseal": sorted(unseal or ())} for s in symbols]
    if workers <= 1:
        parts = [_build_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            parts = list(pool.map(_build_job, jobs))
    parts = [p for p in parts if p is not None and len(p.times)]
    if not parts:
        raise RuntimeError("no data to build a discovery dataset")
    return merge(parts)


def mine(db_path: str, symbols: list[str], train: tuple[int, int], valid: tuple[int, int], tf: str = "15m",
         cfg: SearchConfig | None = None, workers: int = 2, progress=None, unseal=()) -> dict:
    """Full discovery run: build TRAIN/VALID datasets, beam-search on TRAIN, validate on VALID.

    Both windows pass the seal check before any data is read (SealedError unless the period is in `unseal`)."""
    import time as _t

    from heartless.learning import seal

    seal.check_window(int(train[0]), int(train[1]), unseal)
    seal.check_window(int(valid[0]), int(valid[1]), unseal)
    t0 = _t.time()
    ds_tr = build_dataset(db_path, symbols, train[0], train[1], tf, workers, unseal=unseal)
    ds_va = build_dataset(db_path, symbols, valid[0], valid[1], tf, workers, unseal=unseal)
    if progress:
        progress(f"datasets: train {len(ds_tr.times)} rows, valid {len(ds_va.times)} rows ({_t.time() - t0:.0f}s)")
    rules, tested = search(ds_tr, cfg, progress)
    results = validate(rules, ds_va, tested)
    return {"tf": tf, "tested": tested, "t_bar": multiple_testing_t(tested), "results": results,
            "passed": [r["rule"] for r in results if r["passed"]], "seconds": _t.time() - t0,
            "rows": {"train": int(len(ds_tr.times)), "valid": int(len(ds_va.times))}}

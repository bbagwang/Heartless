"""Derivatives positioning features from 5-minute futures metrics (open interest, long/short ratios, taker ratio).

The same `MetricsSeries.snapshot()` feeds alphas in live trading (rows polled from the REST
/futures/data endpoints) and in backtests (rows from the public archive), so research and production
see identical numbers. A row stamped `ts` is treated as usable only from `ts + 5 minutes` onwards, which
is conservative for both sources and rules out look-ahead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BUCKET_MS = 300_000
AVAILABLE_AFTER_MS = BUCKET_MS  # a row describes the bucket that starts at ts; use it once the bucket closed


@dataclass
class MetricsSeries:
    ts: np.ndarray
    oi: np.ndarray
    oi_value: np.ndarray
    top_ls_accounts: np.ndarray
    top_ls_positions: np.ndarray
    ls_accounts: np.ndarray
    taker_ls_vol: np.ndarray

    @classmethod
    def empty(cls) -> "MetricsSeries":
        z = np.zeros(0)
        return cls(np.zeros(0, dtype=np.int64), z, z, z, z, z, z)

    @classmethod
    def from_rows(cls, rows: list) -> "MetricsSeries":
        """rows: dicts (store) or objects with ts/oi/oi_value/top_ls_accounts/top_ls_positions/ls_accounts/taker_ls_vol."""
        if not rows:
            return cls.empty()

        def col(name: str) -> np.ndarray:
            vals = [(r[name] if isinstance(r, dict) else getattr(r, name)) for r in rows]
            return np.array([float(v) if v is not None else np.nan for v in vals], dtype=float)

        ts = np.array([int(r["ts"] if isinstance(r, dict) else r.ts) for r in rows], dtype=np.int64)
        order = np.argsort(ts, kind="stable")
        out = cls(ts[order], *(col(c)[order] for c in ("oi", "oi_value", "top_ls_accounts", "top_ls_positions",
                                                        "ls_accounts", "taker_ls_vol")))
        # zeros in ratio columns mean "missing" in the archive
        for arr in (out.top_ls_accounts, out.top_ls_positions, out.ls_accounts, out.taker_ls_vol, out.oi, out.oi_value):
            arr[arr <= 0] = np.nan
        return out

    def __len__(self) -> int:
        return int(len(self.ts))

    def merge(self, other: "MetricsSeries") -> "MetricsSeries":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other
        ts = np.concatenate([self.ts, other.ts])
        cols = [np.concatenate([getattr(self, c), getattr(other, c)]) for c in
                ("oi", "oi_value", "top_ls_accounts", "top_ls_positions", "ls_accounts", "taker_ls_vol")]
        uniq, idx = np.unique(ts[::-1], return_index=True)  # keep the newest copy of duplicate buckets
        keep = len(ts) - 1 - idx
        return MetricsSeries(ts[keep], *(c[keep] for c in cols))

    def tail(self, n: int) -> "MetricsSeries":
        if len(self) <= n:
            return self
        return MetricsSeries(self.ts[-n:], self.oi[-n:], self.oi_value[-n:], self.top_ls_accounts[-n:],
                             self.top_ls_positions[-n:], self.ls_accounts[-n:], self.taker_ls_vol[-n:])

    # --- features ----------------------------------------------------------------------------
    def _idx(self, now_ms: int) -> int:
        """Index of the newest row usable at now_ms, or -1."""
        if len(self) == 0:
            return -1
        return int(np.searchsorted(self.ts, now_ms - AVAILABLE_AFTER_MS, side="right")) - 1

    @staticmethod
    def _back(arr: np.ndarray, ts: np.ndarray, i: int, lookback_ms: int) -> float:
        """Value of arr at (ts[i] - lookback), using the newest row at or before that time."""
        j = int(np.searchsorted(ts, ts[i] - lookback_ms, side="right")) - 1
        if j < 0 or ts[i] - ts[j] < lookback_ms * 0.75:
            return math.nan
        return float(arr[j])

    def snapshot(self, now_ms: int) -> dict:
        i = self._idx(now_ms)
        if i < 0:
            return {}
        ts = self.ts
        age_min = (now_ms - int(ts[i]) - AVAILABLE_AFTER_MS) / 60_000
        if age_min > 30:  # stale (feed gap): better no information than old information
            return {"stale": True, "age_min": age_min}
        oi = float(self.oi[i])
        out: dict = {"age_min": age_min, "oi": oi, "oi_value": float(self.oi_value[i])}
        for label, lb in (("1h", 3_600_000), ("4h", 4 * 3_600_000), ("24h", 24 * 3_600_000)):
            prev = self._back(self.oi, ts, i, lb)
            out[f"oi_chg_{label}"] = (oi / prev - 1.0) if prev and not math.isnan(prev) and prev > 0 else math.nan
        tlp = float(self.top_ls_positions[i])
        tla = float(self.top_ls_accounts[i])
        lsa = float(self.ls_accounts[i])
        out["top_ls_pos"] = tlp
        out["top_ls_acc"] = tla
        out["ls_acc"] = lsa
        for name, arr, cur in (("top_ls_pos", self.top_ls_positions, tlp), ("ls_acc", self.ls_accounts, lsa)):
            prev = self._back(arr, ts, i, 4 * 3_600_000)
            out[f"{name}_chg_4h"] = (cur - prev) if not (math.isnan(prev) or math.isnan(cur)) else math.nan
        out["taker_ratio"] = float(self.taker_ls_vol[i])
        lo = max(0, i - 11)
        window = self.taker_ls_vol[lo:i + 1]
        good = window[np.isfinite(window) & (window > 0)]
        # buy/sell ratios are skewed (2.0 and 0.5 are symmetric): average them geometrically
        out["taker_ratio_1h"] = float(np.exp(np.mean(np.log(good)))) if len(good) else math.nan
        # z-score of the 1h OI change against the last ~3 days of 1h changes (crowding build-up detector)
        lo = max(0, i - 864)
        seg = self.oi[lo:i + 1]
        if len(seg) > 60:
            ch = seg[12:] / seg[:-12] - 1.0
            ch = ch[np.isfinite(ch)]
            if len(ch) > 30 and np.std(ch) > 0 and not math.isnan(out["oi_chg_1h"]):
                out["oi_chg_1h_z"] = float((out["oi_chg_1h"] - np.mean(ch)) / np.std(ch))
        return out


def rows_from_rest(oi_hist: list[dict], top_pos: list[dict], top_acc: list[dict], global_acc: list[dict],
                   taker: list[dict]) -> list[dict]:
    """Join the REST /futures/data series (period=5m) into archive-shaped rows keyed by timestamp.

    REST stamps each 5-minute point with its *end* time while the archive stamps the bucket start; both are
    converted to the bucket start so live and backtest rows line up."""
    rows: dict[int, dict] = {}

    def bucket(ts: int) -> int:
        return int(ts) - BUCKET_MS if int(ts) % BUCKET_MS == 0 else int(ts) - int(ts) % BUCKET_MS

    def put(items: list[dict], mapping: dict[str, str]) -> None:
        for it in items or []:
            try:
                b = bucket(int(it["timestamp"]))
            except (KeyError, ValueError, TypeError):
                continue
            row = rows.setdefault(b, {"ts": b, "oi": None, "oi_value": None, "top_ls_accounts": None,
                                      "top_ls_positions": None, "ls_accounts": None, "taker_ls_vol": None})
            for dst, src in mapping.items():
                try:
                    row[dst] = float(it[src])
                except (KeyError, ValueError, TypeError):
                    pass

    put(oi_hist, {"oi": "sumOpenInterest", "oi_value": "sumOpenInterestValue"})
    put(top_pos, {"top_ls_positions": "longShortRatio"})
    put(top_acc, {"top_ls_accounts": "longShortRatio"})
    put(global_acc, {"ls_accounts": "longShortRatio"})
    put(taker, {"taker_ls_vol": "buySellRatio"})
    return [rows[k] for k in sorted(rows)]

"""Column-oriented candle storage with multi-timeframe resampling from 1-minute bars."""
from __future__ import annotations

import numpy as np

from heartless.core.models import Candle
from heartless.util.timeutil import TF_MS

COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy", "close_time")


class CandleArrays:
    """Growable numpy columns for one symbol/timeframe. Only *closed* candles live here."""

    def __init__(self, tf: str = "1m", capacity: int = 4096):
        self.tf = tf
        self.tf_ms = TF_MS[tf]
        self.n = 0
        self._cap = max(capacity, 16)
        self.open_time = np.zeros(self._cap, dtype=np.int64)
        self.close_time = np.zeros(self._cap, dtype=np.int64)
        self.open = np.zeros(self._cap)
        self.high = np.zeros(self._cap)
        self.low = np.zeros(self._cap)
        self.close = np.zeros(self._cap)
        self.volume = np.zeros(self._cap)
        self.quote_volume = np.zeros(self._cap)
        self.trades = np.zeros(self._cap)
        self.taker_buy = np.zeros(self._cap)

    def __len__(self) -> int:
        return self.n

    def _grow(self) -> None:
        new_cap = self._cap * 2
        for name in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy"):
            arr = getattr(self, name)
            new = np.zeros(new_cap, dtype=arr.dtype)
            new[: self.n] = arr[: self.n]
            setattr(self, name, new)
        self._cap = new_cap

    def append(self, c: Candle) -> bool:
        """Append or replace the candle. Returns True if a new bar was added."""
        if self.n > 0:
            last = self.open_time[self.n - 1]
            if c.open_time == last:  # replace (e.g. revised closed bar)
                self._set(self.n - 1, c)
                return False
            if c.open_time < last:
                # out-of-order (backfill into history) -> insert sorted
                idx = int(np.searchsorted(self.open_time[: self.n], c.open_time))
                if idx < self.n and self.open_time[idx] == c.open_time:
                    self._set(idx, c)
                    return False
                self._insert(idx, c)
                return True
        if self.n >= self._cap:
            self._grow()
        self._set(self.n, c)
        self.n += 1
        return True

    def _insert(self, idx: int, c: Candle) -> None:
        if self.n >= self._cap:
            self._grow()
        for name in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy"):
            arr = getattr(self, name)
            arr[idx + 1: self.n + 1] = arr[idx: self.n]
        self.n += 1
        self._set(idx, c)

    def _set(self, i: int, c: Candle) -> None:
        self.open_time[i] = c.open_time
        self.close_time[i] = c.close_time
        self.open[i] = c.open
        self.high[i] = c.high
        self.low[i] = c.low
        self.close[i] = c.close
        self.volume[i] = c.volume
        self.quote_volume[i] = c.quote_volume
        self.trades[i] = c.trades
        self.taker_buy[i] = c.taker_buy_volume

    def extend(self, candles: list[Candle]) -> int:
        added = 0
        for c in candles:
            if c.closed:
                added += self.append(c)
        return added

    def view(self, name: str) -> np.ndarray:
        return getattr(self, name)[: self.n]

    def last(self) -> Candle | None:
        if self.n == 0:
            return None
        return self.candle_at(self.n - 1)

    def candle_at(self, i: int) -> Candle:
        return Candle(int(self.open_time[i]), float(self.open[i]), float(self.high[i]), float(self.low[i]),
                      float(self.close[i]), float(self.volume[i]), float(self.quote_volume[i]), int(self.trades[i]),
                      float(self.taker_buy[i]), int(self.close_time[i]), True)

    def tail(self, n: int) -> "CandleArrays":
        start = max(0, self.n - n)
        return self.slice(start, self.n)

    def slice(self, start: int, end: int) -> "CandleArrays":
        out = CandleArrays(self.tf, capacity=max(end - start, 16))
        cnt = end - start
        for name in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy"):
            getattr(out, name)[:cnt] = getattr(self, name)[start:end]
        out.n = cnt
        return out

    def index_at_or_before(self, close_time: int) -> int:
        """Index of the last bar whose close_time <= given time (or -1)."""
        return int(np.searchsorted(self.close_time[: self.n], close_time, side="right")) - 1

    def trim_before(self, open_time: int) -> None:
        idx = int(np.searchsorted(self.open_time[: self.n], open_time))
        if idx <= 0:
            return
        for name in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy"):
            arr = getattr(self, name)
            arr[: self.n - idx] = arr[idx: self.n]
        self.n -= idx


def resample(base: CandleArrays, tf: str, only_complete: bool = True) -> CandleArrays:
    """Aggregate 1m (or any finer) candles into `tf`.

    Only the trailing group is checked for completeness (enough base bars, or its last base bar
    closing at the group close) and dropped when still open. Every earlier group is unambiguously
    finished because `base` is kept sorted, so it is emitted even when some of its base bars are
    missing (a gap in the 1m store must not punch a hole into the higher timeframe).
    """
    tf_ms = TF_MS[tf]
    n = base.n
    if n == 0:
        return CandleArrays(tf, capacity=16)
    ot = base.open_time[:n]
    groups = ot // tf_ms
    # boundaries where group id changes
    change = np.flatnonzero(np.diff(groups)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    # size from the number of groups, not n // bars_per: a sparse base can have far more groups
    out = CandleArrays(tf, capacity=max(len(starts), 16))
    g_open_time = groups[starts] * tf_ms
    g_close_time = g_open_time + tf_ms - 1
    bars_per = tf_ms // base.tf_ms
    counts = ends - starts
    complete = counts >= bars_per if only_complete else np.ones(len(starts), dtype=bool)
    # a group is also complete if its last base bar closes at the group close
    last_close = base.close_time[:n][ends - 1]
    complete |= last_close >= g_close_time
    # every group followed by a later group is finished regardless of how many base bars it holds
    complete[:-1] = True
    if not complete.any():
        return out
    starts_c = starts[complete]
    ends_c = ends[complete]
    cnt = len(starts_c)
    out.open_time[:cnt] = g_open_time[complete]
    out.close_time[:cnt] = g_close_time[complete]
    out.open[:cnt] = base.open[:n][starts_c]
    out.close[:cnt] = base.close[:n][ends_c - 1]
    # reduceat for max/min/sum over variable-length groups
    hi = np.maximum.reduceat(base.high[:n], starts)
    lo = np.minimum.reduceat(base.low[:n], starts)
    vol = np.add.reduceat(base.volume[:n], starts)
    qv = np.add.reduceat(base.quote_volume[:n], starts)
    tr = np.add.reduceat(base.trades[:n], starts)
    tb = np.add.reduceat(base.taker_buy[:n], starts)
    out.high[:cnt] = hi[complete]
    out.low[:cnt] = lo[complete]
    out.volume[:cnt] = vol[complete]
    out.quote_volume[:cnt] = qv[complete]
    out.trades[:cnt] = tr[complete]
    out.taker_buy[:cnt] = tb[complete]
    out.n = cnt
    return out

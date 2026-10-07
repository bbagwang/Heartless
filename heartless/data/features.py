"""Feature frames: all indicators for one symbol/timeframe, plus a causal multi-timeframe cursor."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from heartless.data import indicators as I
from heartless.data.candles import CandleArrays, resample

TIMEFRAMES = ("1m", "5m", "15m", "1h")
# how many bars of each timeframe to keep for live recomputation (enough for EMA200 + percentiles)
LIVE_WINDOW = {"1m": 1500, "5m": 1000, "15m": 700, "1h": 520}


class FeatureFrame:
    """Dictionary of aligned numpy arrays for one timeframe."""

    def __init__(self, tf: str, bars: CandleArrays):
        self.tf = tf
        self.n = bars.n
        self.open_time = bars.view("open_time")
        self.close_time = bars.view("close_time")
        self.f: dict[str, np.ndarray] = {}
        self._compute(bars)

    def _compute(self, b: CandleArrays) -> None:
        o, h, l, c = b.view("open"), b.view("high"), b.view("low"), b.view("close")
        v, qv, tb, ot = b.view("volume"), b.view("quote_volume"), b.view("taker_buy"), b.view("open_time")
        f = self.f
        f["open"], f["high"], f["low"], f["close"], f["volume"], f["quote_volume"] = o, h, l, c, v, qv
        n = len(c)
        if n == 0:
            return
        f["ema9"] = I.ema(c, 9)
        f["ema21"] = I.ema(c, 21)
        f["ema50"] = I.ema(c, 50)
        f["ema200"] = I.ema(c, 200)
        f["rsi14"] = I.rsi(c, 14)
        f["rsi7"] = I.rsi(c, 7)
        f["rsi2"] = I.rsi(c, 2)
        a = I.atr(h, l, c, 14)
        f["atr"] = a
        f["atr_pct"] = np.divide(a, c, out=np.full(n, np.nan), where=c > 0)
        f["atr_rank"] = I.percentile_rank(np.nan_to_num(a, nan=0.0), 200)
        mid, up, lo, width = I.bollinger(c, 20, 2.0)
        sd = I.rolling_std(c, 20)
        f["bb_mid"], f["bb_up"], f["bb_lo"], f["bb_width"], f["bb_sd"] = mid, up, lo, width, sd
        f["bb_width_rank"] = I.percentile_rank(np.nan_to_num(width, nan=0.0), 120)
        kmid, kup, klo = I.keltner(h, l, c, 20, 1.5)
        f["kc_mid"], f["kc_up"], f["kc_lo"] = kmid, kup, klo
        sq = (up < kup) & (lo > klo)
        f["squeeze"] = sq.astype(float)
        # consecutive squeeze bar count
        cnt = np.zeros(n)
        run = 0
        for i in range(n):
            run = run + 1 if sq[i] else 0
            cnt[i] = run
        f["squeeze_bars"] = cnt
        vw, vsd = I.session_vwap(ot, h, l, c, v)
        f["vwap"], f["vwap_sd"] = vw, vsd
        st, sdir = I.supertrend(h, l, c, 10, 3.0)
        f["st_line"], f["st_dir"] = st, sdir.astype(float)
        adx, pdi, mdi = I.adx(h, l, c, 14)
        f["adx"], f["pdi"], f["mdi"] = adx, pdi, mdi
        m, s, hist = I.macd(c, 12, 26, 9)
        f["macd"], f["macd_sig"], f["macd_hist"] = m, s, hist
        for L in (10, 20, 50):
            hh, ll = I.donchian(h, l, L, exclude_current=True)
            f[f"dc_hi{L}"], f[f"dc_lo{L}"] = hh, ll
            f[f"hh{L}"], f[f"ll{L}"] = I.rolling_max(h, L), I.rolling_min(l, L)
        f["chop"] = I.choppiness(h, l, c, 14)
        f["vol_z"] = I.zscore(v, 50)
        f["vol_sma"] = I.sma(v, 20)
        with np.errstate(divide="ignore", invalid="ignore"):
            tr = np.where(v > 0, tb / np.where(v > 0, v, 1.0), 0.5)
        f["taker_ratio"] = tr
        f["taker_ratio3"] = I.sma(tr, 3)
        f["cvd20"] = I.cvd_proxy(v, tb, 20)
        f["slope20"] = I.linreg_slope(c, 20)
        f["hurst"] = I.hurst_proxy(c, 100)
        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-12)
        f["body_ratio"] = body / rng
        f["ret1"] = np.divide(np.diff(c, prepend=c[0]), np.where(c > 0, c, 1.0))
        f["upper_wick"] = (h - np.maximum(o, c)) / rng
        f["lower_wick"] = (np.minimum(o, c) - l) / rng

    def names(self) -> list[str]:
        return list(self.f.keys())


@dataclass
class FrameCursor:
    """Read-only accessor for a frame at a particular (completed) bar index."""
    frame: FeatureFrame
    idx: int

    @property
    def ok(self) -> bool:
        return self.idx >= 0

    def v(self, name: str, k: int = 0) -> float:
        i = self.idx - k
        if i < 0 or i >= self.frame.n:
            return float("nan")
        arr = self.frame.f.get(name)
        if arr is None:
            return float("nan")
        return float(arr[i])

    def arr(self, name: str, n: int) -> np.ndarray:
        i = self.idx
        if i < 0:
            return np.array([])
        return self.frame.f[name][max(0, i - n + 1): i + 1]

    @property
    def close_time(self) -> int:
        return int(self.frame.close_time[self.idx]) if self.idx >= 0 else 0

    @property
    def open_time(self) -> int:
        return int(self.frame.open_time[self.idx]) if self.idx >= 0 else 0


class MarketView:
    """Multi-timeframe feature view for one symbol with a movable causal cursor."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.frames: dict[str, FeatureFrame] = {}
        self.cursors: dict[str, FrameCursor] = {}
        self.cursor_time: int = 0
        self.just_closed: set[str] = set()
        self.base: CandleArrays | None = None

    def rebuild(self, base_1m: CandleArrays, live: bool = True) -> None:
        self.base = base_1m
        for tf in TIMEFRAMES:
            bars = base_1m if tf == "1m" else resample(base_1m, tf)
            if live:
                bars = bars.tail(LIVE_WINDOW[tf])
            self.frames[tf] = FeatureFrame(tf, bars)

    def seek(self, close_time: int) -> None:
        """Position every timeframe at the last bar completed at or before close_time.

        A timeframe counts as "just closed" when its latest completed bar closed inside the last
        minute ending at close_time (robust to data whose epoch is not aligned to the timeframe).
        """
        self.cursor_time = close_time
        self.just_closed = set()
        for tf, fr in self.frames.items():
            idx = int(np.searchsorted(fr.close_time, close_time, side="right")) - 1
            self.cursors[tf] = FrameCursor(fr, idx)
            if idx >= 0 and int(fr.close_time[idx]) > close_time - 60_000:
                self.just_closed.add(tf)

    def tf(self, tf: str) -> FrameCursor:
        c = self.cursors.get(tf)
        if c is None:
            fr = self.frames.get(tf)
            return FrameCursor(fr, -1) if fr else FrameCursor(FeatureFrame(tf, CandleArrays(tf)), -1)
        return c

    def closed(self, tf: str) -> bool:
        return tf in self.just_closed

    @property
    def price(self) -> float:
        c = self.tf("1m")
        return c.v("close") if c.ok else float("nan")

    def ready(self) -> bool:
        return all(self.tf(tf).ok for tf in ("1m", "5m", "15m")) and self.tf("5m").idx >= 60 and self.tf("15m").idx >= 30

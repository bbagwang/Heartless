"""Feature frames: all indicators for one symbol/timeframe, plus a causal multi-timeframe cursor."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from heartless.data import indicators as I
from heartless.data.candles import CandleArrays, resample, resample_tail
from heartless.util.timeutil import TF_MS

TIMEFRAMES = ("1m", "5m", "15m", "1h")
# how many bars of each timeframe to keep for live recomputation (enough for EMA200 + percentiles)
LIVE_WINDOW = {"1m": 1500, "5m": 1000, "15m": 700, "1h": 520}
# Long-horizon frames for daily strategies: UTC-aligned 4h and 1d bars (resampled by open_time // period, so they open
# at 00/04/08/... and 00:00 UTC) with the same FeatureFrame columns. They are not tails of a fixed bar count: the 1m
# history behind them is sized by the longest Alpha.lookback_days (app history depth, lab / backtester warm-up). They
# are rebuilt only when one of their bars closes (or on first access), never per minute, and seek() does not position
# them on every bar: their cursor is found when an alpha asks for it (MarketView.tf / closed).
LONG_TIMEFRAMES = ("4h", "1d")
ALL_TIMEFRAMES = TIMEFRAMES + LONG_TIMEFRAMES


class FeatureFrame:
    """Dictionary of aligned numpy arrays for one timeframe."""

    def __init__(self, tf: str, bars: CandleArrays):
        self.tf = tf
        self.n = bars.n
        self.open_time = bars.view("open_time")
        self.close_time = bars.view("close_time")
        self.f: dict[str, np.ndarray] = {}
        self._ct: list[int] | None = None
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
        # consecutive squeeze bar count: distance to the last non-squeeze bar (integer valued, so exact as float)
        pos = np.arange(n)
        last_off = np.maximum.accumulate(np.where(sq, -1, pos))
        f["squeeze_bars"] = np.where(sq, pos - last_off, 0).astype(float)
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

    def close_times(self) -> list[int]:
        """close_time as a Python list (built once per frame): seek() bisects it on every bar of a backtest, which is
        several times cheaper than a numpy searchsorted call on a scalar."""
        ct = self._ct
        if ct is None:
            ct = self._ct = self.close_time.tolist()
        return ct


def session_volume_bars(open_time: np.ndarray, volume: np.ndarray, session_ms: int = 86_400_000) -> np.ndarray:
    """Per bar, how many bars with volume its UTC session holds up to and including it: the observations behind
    indicators.session_vwap's running sums at that bar (open_time ascending)."""
    ot = np.asarray(open_time, dtype=np.int64)
    if not len(ot):
        return np.zeros(0, dtype=np.int64)
    sess = ot // int(session_ms)
    c = np.cumsum(np.asarray(volume, dtype=float) > 0)
    first = np.searchsorted(sess, sess, side="left")  # first bar of each bar's session
    return c - np.where(first > 0, c[np.maximum(first - 1, 0)], 0)


def vwap_z(frame: "FeatureFrame", idx=None) -> np.ndarray:
    """(close - session VWAP) / session sd at the bars `idx` of a frame (-1: no bar -> NaN; None: every bar).

    F-pre-B fix of the F0 finding (docs/UNIVERSE.md 5): NaN wherever the z is not defined - no VWAP or sd, an sd of 0,
    and every bar whose session holds a single bar with volume so far. There the true sd is 0 and session_vwap's
    sqrt(E[p^2 v] / E[v] - vwap^2) is rounding noise, so the old expression gave +-1e5 or NaN depending on the price
    level (the first bar of every UTC day, and every 1d bar). Everywhere else it is the old expression bit for bit.
    The frame's columns are not touched (vwap_sd stays as the batch and incremental frames compute it)."""
    n = int(frame.n)
    sel = np.arange(n) if idx is None else np.asarray(idx, dtype=np.int64)
    out = np.full(len(sel), np.nan)
    if not n or not len(sel):
        return out
    f = frame.f
    cnt = session_volume_bars(frame.open_time[:n], f["volume"][:n])
    valid = (sel >= 0) & (sel < n)
    s = np.where(valid, sel, 0)
    close, vw, sd = f["close"][s], f["vwap"][s], f["vwap_sd"][s]
    ok = valid & (cnt[s] >= 2) & np.isfinite(close) & np.isfinite(vw) & np.isfinite(sd) & (sd != 0)
    out[ok] = (close[ok] - vw[ok]) / sd[ok]
    return out


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
    """Multi-timeframe feature view for one symbol with a movable causal cursor.

    `incremental=True` (the live bot, INCREMENTAL_FEATURES): rebuild(live=True) builds data/incremental.LiveFrames and
    advance() then appends each new 1m bar (and each resampled bar it closes) instead of recomputing the frames; the
    backtester and the lab keep the batch path (incremental=False, rebuild(live=False))."""

    def __init__(self, symbol: str, incremental: bool = False):
        self.symbol = symbol
        self.incremental = incremental
        self.live = None  # data.incremental.LiveFrames while incremental updates run
        self.frames: dict[str, FeatureFrame] = {}
        self.cursors: dict[str, FrameCursor] = {}
        self.cursor_time: int = 0
        self.just_closed: set[str] = set()
        self.base: CandleArrays | None = None
        # long frames: the base close time from which the frame is stale (its next bar has closed), and the cursor of
        # the last cursor_time it was asked for
        self._long_due: dict[str, int] = {}
        self._long_cur: dict[str, tuple[int, FrameCursor]] = {}

    def rebuild(self, base_1m: CandleArrays, live: bool = True) -> None:
        self.base = base_1m
        if live and self.incremental:
            from heartless.data.incremental import LiveFrames

            self.live = LiveFrames(self.symbol)
            self._install(self.live.build(base_1m))
            return
        self.live = None
        for tf in TIMEFRAMES:
            if tf == "1m":
                bars = base_1m.tail(LIVE_WINDOW[tf]) if live else base_1m
            else:
                # live: only the 1m tail the frame's last LIVE_WINDOW bars come from (identical bars, bounded cost)
                bars = resample_tail(base_1m, tf, LIVE_WINDOW[tf]) if live else resample(base_1m, tf)
            self.frames[tf] = FeatureFrame(tf, bars)
        self._long_due.clear()
        self._long_cur.clear()
        for tf in LONG_TIMEFRAMES:  # built here so `tf in view.frames` holds (the engine's trail timeframe lookup)
            self.frames.pop(tf, None)
            self._long_frame(tf)

    # --- F7: incremental live frames ---------------------------------------------------------------------------------
    def _install(self, frames: dict) -> None:
        for tf in TIMEFRAMES:
            self.frames[tf] = frames[tf]
        for tf in LONG_TIMEFRAMES:  # like _long_frame: no 4h / 1d frame while the base is empty
            if self.base is not None and self.base.n:
                self.frames[tf] = frames[tf]
            else:
                self.frames.pop(tf, None)
        self._long_due.clear()
        self._long_cur.clear()

    def advance_cheap(self, base_1m: CandleArrays) -> bool:
        """True when advance(base_1m) only appends bars (tens of microseconds each, fine on the event loop)."""
        return self.live is not None and self.live.cheap(base_1m)

    def advance(self, base_1m: CandleArrays) -> bool:
        """Bring the live frames up to base_1m (the bars appended since the last call). Incremental views append; a view
        without LiveFrames (incremental=False, or one whose periodic check failed) rebuilds every base timeframe in
        batch. Returns True when the update was incremental."""
        lf = self.live
        if lf is None:
            self._batch_base_frames(base_1m)
            return False
        done = lf.advance(base_1m)
        self.base = base_1m
        if lf.failed:  # the check against the batch failed: batch rebuilds from now on (logged by LiveFrames)
            self.incremental, self.live = False, None
            self._batch_base_frames(base_1m)
            self._long_due.clear()  # the 4h / 1d frames are rebuilt by _long_frame at their next use
            self._long_cur.clear()
            return False
        self._install(lf.frames)
        return done

    def _batch_base_frames(self, base_1m: CandleArrays) -> None:
        """Every base timeframe recomputed from the 1m tail, as app._rebuild_view does (the non-incremental path)."""
        for tf in TIMEFRAMES:
            bars = base_1m.tail(LIVE_WINDOW[tf]) if tf == "1m" else resample_tail(base_1m, tf, LIVE_WINDOW[tf])
            self.frames[tf] = FeatureFrame(tf, bars)
        self.base = base_1m

    def _long_frame(self, tf: str) -> FeatureFrame | None:
        """The 4h / 1d frame over the whole base, rebuilt once the base holds a bar at or past its next close.

        In a backtest the base is the full loaded history, so the frame is built once; live the base grows by one bar
        a minute and the frame is rebuilt once per period (bars that close later are never visible early: the cursor
        only reaches bars closed at or before cursor_time)."""
        fr = self.frames.get(tf)
        if self.live is not None:  # kept current by advance()
            return fr
        base = self.base
        if base is None or base.n == 0:
            return fr
        last = int(base.close_time[base.n - 1])
        due = self._long_due.get(tf)
        if fr is None or due is None or last >= due:
            period = TF_MS[tf]
            fr = FeatureFrame(tf, resample(base, tf))
            self.frames[tf] = fr
            self._long_due[tf] = ((last + 1) // period + 1) * period - 1  # close of the first group not yet complete
            self._long_cur.pop(tf, None)
        return fr

    def _long_cursor(self, tf: str) -> FrameCursor:
        fr = self._long_frame(tf)
        if fr is None:
            return FrameCursor(FeatureFrame(tf, CandleArrays(tf)), -1)
        hit = self._long_cur.get(tf)
        if hit is not None and hit[0] == self.cursor_time and hit[1].frame is fr:
            return hit[1]
        cur = FrameCursor(fr, bisect_right(fr.close_times(), self.cursor_time) - 1)
        self._long_cur[tf] = (self.cursor_time, cur)
        return cur

    def seek(self, close_time: int) -> None:
        """Position every timeframe at the last bar completed at or before close_time.

        A timeframe counts as "just closed" when its latest completed bar closed inside the last
        minute ending at close_time (robust to data whose epoch is not aligned to the timeframe).
        The long frames (4h, 1d) are only looked at in the minute after one of their UTC period ends: their bars close
        at k * period - 1, so no other minute can hold one of their closes; their cursor is otherwise found on demand.
        """
        self.cursor_time = close_time
        self.just_closed = set()
        frames = self.frames
        for tf in TIMEFRAMES:
            fr = frames.get(tf)
            if fr is None:
                continue
            ct = fr.close_times()
            idx = bisect_right(ct, close_time) - 1  # == searchsorted(close_time, side="right") - 1
            self.cursors[tf] = FrameCursor(fr, idx)
            if idx >= 0 and ct[idx] > close_time - 60_000:
                self.just_closed.add(tf)
        for tf in LONG_TIMEFRAMES:
            # also when the frame does not exist yet: a live view built on an empty history (a failed backfill) gets
            # its first long frame here once streamed bars arrive, or a daily alpha would never be asked on it
            if (close_time + 1) % TF_MS[tf] < 60_000:
                cur = self._long_cursor(tf)
                if cur.idx >= 0 and cur.frame.close_times()[cur.idx] > close_time - 60_000:
                    self.just_closed.add(tf)

    def tf(self, tf: str) -> FrameCursor:
        if tf in LONG_TIMEFRAMES:
            return self._long_cursor(tf)
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

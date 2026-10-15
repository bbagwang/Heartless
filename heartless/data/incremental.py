"""Incremental feature frames for the live bot (F7): every FeatureFrame column updated in O(1) per new bar.

The live bot used to recompute the whole 1m frame (1,500 bars, 61 columns) on every closed minute and the 5m / 15m /
1h frames on each of their closes. `IncrementalFrame` appends one bar at a time instead and reproduces, bit for bit,
`FeatureFrame(tf, bars)` over every bar since its anchor (the bars it was built from):

* recursive filters (EMA, Wilder ATR / RSI / ADX, MACD signal, Supertrend, session VWAP, squeeze run length) keep the
  last value of the very loop in data/indicators.py (same Python float operations in the same order);
* cumulative-sum windows (SMA, rolling sums: Bollinger mid, volume SMA / z, taker ratio, CVD, choppiness, Hurst's
  q-sums) keep the running np.cumsum total, so `(C[i+1] - C[i+1-n]) / n` is the batch expression on the same numbers;
* sliding-window reductions (rolling std / max / min, percentile rank, regression slope, Hurst variances) run the
  same numpy reduction on the last window only (numpy reduces each row of the batch's window view independently;
  tests/test_f7_incremental.py pins the equality, also for NaNs, constant runs and short histories).

The anchor matters: the batch results depend on where the frame starts (EMA seeds, cumulative sums). A frame is
therefore re-anchored periodically (`LiveFrames`, REANCHOR_ROWS): it is rebuilt from the base tail exactly as the
previous per-bar rebuild did, after checking that the incremental columns still equal a batch recomputation over the
bars since the old anchor (a mismatch logs an error and switches the symbol back to per-bar batch rebuilds). Between
anchors a frame is the batch computation over the bars since its anchor, not over the last LIVE_WINDOW bars: an EMA200
then remembers its seed a little less (relative weight (199/201)^rows), which is closer to the backtester (anchored at
the start of the loaded history) than the sliding window was. docs/PERFORMANCE.md has the measurements.

`LiveFrames` turns closed 1m bars into the 5m / 15m / 1h / 4h / 1d bars exactly as candles.resample() does (group
aggregates by np.*.reduceat over the same base bars, the same completeness rule for the trailing group) and appends
them to those frames. The 4h / 1d frames stay anchored at the start of the 1m base like MarketView._long_frame: after
the base is trimmed they are rebuilt from their own cached bars at their next close (the moment the batch path would
rebuild them), so they are bit-identical to that path at all times. The backtester and the lab keep the batch path.
"""
from __future__ import annotations

import logging
import math
import warnings
import zlib

import numpy as np

from heartless.data.candles import CandleArrays, resample, resample_tail
from heartless.data.features import LIVE_WINDOW, LONG_TIMEFRAMES, TIMEFRAMES, FeatureFrame
from heartless.data import indicators as I
from heartless.util.timeutil import TF_MS

log = logging.getLogger(__name__)

NAN = float("nan")
# re-anchor a sliding frame after this many appended rows (spread over symbols, see LiveFrames): 4 h of 1m bars, 8 h of
# 5m / 15m bars, a day of 1h bars. Bounds the frames' growth (+16% / +10% / +5% / +5% rows) and how far the cumulative
# sums run from the anchor, at the cost of one batch rebuild (plus its check) per period.
REANCHOR_ROWS = {"1m": 240, "5m": 96, "15m": 32, "1h": 24}
_RAW = ("open", "high", "low", "close", "volume", "quote_volume")
_HIDDEN = ("taker_buy", "lr", "agg")
_LOG10_14 = np.log10(14)  # the batch divides by np.log10(n) for n = 14 (choppiness)
_T20 = np.arange(20) - (20 - 1) / 2  # linreg_slope(c, 20)
_DENOM20 = (_T20 ** 2).sum()
_HQ, _HN = 5, 100  # hurst_proxy(c, 100), q = 5
_HM = _HQ * (_HN - _HQ + 1) * (1.0 - _HQ / _HN)


def _keys() -> tuple[str, ...]:
    """FeatureFrame's column names in its insertion order (a one-bar frame computes every column)."""
    b = CandleArrays("1m", 16)
    b.n = 1
    b.open[0] = b.high[0] = b.low[0] = b.close[0] = 1.0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tuple(FeatureFrame("1m", b).f)


KEYS = _keys()


def _mx(a: float, b: float) -> float:
    """np.maximum on two floats: a NaN on either side gives NaN."""
    if a != a:
        return a
    if b != b:
        return b
    return a if a >= b else b


def _mn(a: float, b: float) -> float:
    """np.minimum on two floats."""
    if a != a:
        return a
    if b != b:
        return b
    return a if a <= b else b


class _Ema:
    """indicators.ema(x, n) one value at a time (NaN input carries the previous value)."""
    __slots__ = ("n", "a", "i", "prev", "buf")

    def __init__(self, n: int):
        self.n, self.a, self.i, self.prev, self.buf = n, 2.0 / (n + 1), 0, NAN, []

    def init(self, x: np.ndarray, out: np.ndarray) -> "_Ema":
        """State after the batch over x (out = the batch result)."""
        self.i = len(x)
        if self.i >= self.n:
            self.prev, self.buf = float(out[-1]), None
        else:
            self.buf = x.tolist()
        return self

    def push(self, v: float) -> float:
        i = self.i
        self.i = i + 1
        n = self.n
        if i < n - 1:
            self.buf.append(v)
            return NAN
        if i == n - 1:
            self.buf.append(v)
            seed = np.nanmean(np.array(self.buf, dtype=float))
            self.prev, self.buf = float(seed), None
            return self.prev
        if v == v:
            self.prev = self.prev + self.a * (v - self.prev)
        return self.prev


class _Wilder(_Ema):
    """indicators.wilder(x, n) one value at a time (NaN counts as 0 in the recursion, not in the seed)."""
    __slots__ = ()

    def __init__(self, n: int):
        super().__init__(n)
        self.a = 1.0 / n

    def push(self, v: float) -> float:
        i = self.i
        self.i = i + 1
        n = self.n
        if i < n - 1:
            self.buf.append(v)
            return NAN
        if i == n - 1:
            self.buf.append(v)
            seed = np.nanmean(np.array(self.buf, dtype=float))
            self.prev, self.buf = float(seed), None
            return self.prev
        if v != v:
            v = 0.0
        self.prev = self.prev + self.a * (v - self.prev)
        return self.prev


class _Valid:
    """A filter over the non-NaN values of a series only (macd's signal EMA, adx's Wilder): NaN in, NaN out."""
    __slots__ = ("inner",)

    def __init__(self, inner: _Ema):
        self.inner = inner

    def init(self, x: np.ndarray, out: np.ndarray) -> "_Valid":
        ok = ~np.isnan(x)
        self.inner.init(x[ok], out[ok])
        return self

    def push(self, v: float) -> float:
        return self.inner.push(v) if v == v else NAN


class _Cum:
    """Running np.cumsum(np.insert(x, 0, 0.0)) with the last `keep` totals: rolling sums are C[i+1] - C[i+1-n]."""
    __slots__ = ("keep", "c", "i")

    def __init__(self, keep: int):
        self.keep, self.c, self.i = keep, [0.0], 0

    def init(self, x: np.ndarray) -> "_Cum":
        c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
        self.c = c[-self.keep:].tolist()
        self.i = len(x)
        return self

    def push(self, v: float) -> None:
        c = self.c
        c.append(c[-1] + v)
        if len(c) > self.keep:
            del c[0]
        self.i += 1

    def sum(self, n: int) -> float:
        """rolling_sum(x, n) at the last pushed index (NaN while fewer than n values)."""
        if self.i < n:
            return NAN
        c = self.c
        return c[-1] - c[-1 - n]


class IncrementalFrame(FeatureFrame):
    """A FeatureFrame that grows one bar at a time; equal, column by column and bit for bit, to FeatureFrame(tf, bars)
    over every bar appended since it was built. Arrays handed out earlier stay valid snapshots: appends write past
    their end and growing reallocates."""

    def __init__(self, tf: str, bars: CandleArrays, extra: int = 0):  # noqa: D107 - no batch call of the parent
        self.tf = tf
        n = bars.n
        cap = max(n + max(extra, 0), 16)
        self._cap = cap
        self._buf: dict[str, np.ndarray] = {k: np.empty(cap) for k in KEYS + _HIDDEN}
        self._ot = np.empty(cap, dtype=np.int64)
        self._ctb = np.empty(cap, dtype=np.int64)
        self.n = 0  # rows published (FeatureFrame.n); _len counts the rows written, ahead of n until _publish()
        self._len = 0
        self.rows_appended = 0
        self._row: dict[str, float] = {}
        self._ct: list[int] = []
        ff = FeatureFrame(tf, bars)
        self._init_from(bars, ff)

    # --- construction from the batch ----------------------------------------------------------------------------
    def _init_from(self, b: CandleArrays, ff: FeatureFrame) -> None:
        n = b.n
        B = self._buf
        self._ot[:n] = b.view("open_time")
        self._ctb[:n] = b.view("close_time")
        B["taker_buy"][:n] = b.view("taker_buy")
        if n:
            for k in KEYS:
                B[k][:n] = ff.f[k]
        else:
            for k in _RAW:
                B[k][:0] = ff.f[k]
        o, h, l, c = (b.view(k) for k in ("open", "high", "low", "close"))
        v, tb = b.view("volume"), b.view("taker_buy")
        f = ff.f
        self._len = n
        # recursive states
        self.ema = {k: _Ema(p).init(c, f[k]) if n else _Ema(p) for k, p in (("ema9", 9), ("ema21", 21), ("ema50", 50),
                                                                         ("ema200", 200))}
        self.kc_ema = _Ema(20).init(c, f["kc_mid"]) if n else _Ema(20)
        self.ema12 = _Ema(12).init(c, I.ema(c, 12)) if n else _Ema(12)
        self.ema26 = _Ema(26).init(c, I.ema(c, 26)) if n else _Ema(26)
        self.sig = _Valid(_Ema(9)).init(f["macd"], f["macd_sig"]) if n else _Valid(_Ema(9))
        tr = I.true_range(h, l, c) if n else np.zeros(0)
        self.atr14 = _Wilder(14).init(tr, f["atr"]) if n else _Wilder(14)
        self.atr20 = _Wilder(20).init(tr, I.wilder(tr, 20)) if n else _Wilder(20)
        self.atr10 = _Wilder(10).init(tr, I.wilder(tr, 10)) if n else _Wilder(10)
        self.rsi = {}
        d = np.diff(c, prepend=c[0]) if n else np.zeros(0)
        up = np.where(d > 0, d, 0.0)[1:]
        dn = np.where(d < 0, -d, 0.0)[1:]
        for p in (14, 7, 2):
            self.rsi[p] = (_Wilder(p).init(up, I.wilder(up, p)), _Wilder(p).init(dn, I.wilder(dn, p)))
        # adx
        if n:
            hu = np.diff(h, prepend=h[0])
            ld = -np.diff(l, prepend=l[0])
            pdm = np.where((hu > ld) & (hu > 0), hu, 0.0)
            mdm = np.where((ld > hu) & (ld > 0), ld, 0.0)
            self.pdm = _Wilder(14).init(pdm, I.wilder(pdm, 14))
            self.mdm = _Wilder(14).init(mdm, I.wilder(mdm, 14))
            pdi, mdi = f["pdi"], f["mdi"]
            denom = pdi + mdi
            dx = 100 * np.divide(np.abs(pdi - mdi), denom, out=np.zeros(n), where=denom > 0)
            dx[np.isnan(pdi)] = np.nan
            self.adx = _Valid(_Wilder(14)).init(dx, f["adx"])
        else:
            self.pdm, self.mdm, self.adx = _Wilder(14), _Wilder(14), _Valid(_Wilder(14))
        # cumulative sums (shared by every window over the same series)
        self.cum_c = _Cum(21).init(c)
        self.cum_v = _Cum(51).init(v)
        self.cum_tr = _Cum(15).init(tr)
        with np.errstate(divide="ignore", invalid="ignore"):
            trr = np.where(v > 0, tb / np.where(v > 0, v, 1.0), 0.5)
        self.cum_trr = _Cum(4).init(trr)
        self.cum_delta = _Cum(21).init(2 * tb - v)
        # hurst: log returns and their q-sums, kept as hidden columns (windows of 100 / 96)
        if n:
            with np.errstate(divide="ignore", invalid="ignore"):
                L = np.log(np.maximum(c, 1e-12))
            lr = np.diff(L, prepend=np.nan)
            lr[0] = 0.0
            agg = I.rolling_sum(lr, _HQ)
            agg[np.isnan(agg)] = 0.0
            B["lr"][:n] = lr
            B["agg"][:n] = agg
            self.L_prev = float(L[-1])
        else:
            self.L_prev = NAN
        self.cum_lr = _Cum(_HQ + 1).init(B["lr"][:n])
        # supertrend: the batch loop's final bands and direction
        self.st_started, self.st_ub, self.st_lb, self.st_dir = False, NAN, NAN, 0
        if n:
            a10 = I.wilder(tr, 10)
            hl2 = (h + l) / 2
            ubl = (hl2 + 3.0 * a10).tolist()
            lbl = (hl2 - 3.0 * a10).tolist()
            ok = ~np.isnan(a10)
            if ok.any():
                start = int(np.argmax(ok))
                cl = c.tolist()
                fu, fl, dr = ubl[start], lbl[start], 0
                for i in range(start + 1, n):
                    pu, pl, pc = fu, fl, cl[i - 1]
                    u = ubl[i] if (ubl[i] < pu or pc > pu) else pu
                    lo = lbl[i] if (lbl[i] > pl or pc < pl) else pl
                    prev_dir = dr if dr != 0 else 1
                    if prev_dir == 1:
                        dd = -1 if cl[i] < lo else 1
                    else:
                        dd = 1 if cl[i] > u else -1
                    fu, fl, dr = u, lo, dd
                self.st_started, self.st_ub, self.st_lb, self.st_dir = True, fu, fl, dr
        # session vwap: the running sums of the last session in the frame
        self.vw_sess, self.vw_pv, self.vw_v, self.vw_pv2 = None, 0.0, 0.0, 0.0
        if n:
            ot = b.view("open_time")
            sess = ot // 86_400_000
            j = int(np.searchsorted(sess, sess[-1]))
            tpl = ((h[j:] + l[j:] + c[j:]) / 3).tolist()
            for t, vi in zip(tpl, v[j:].tolist()):
                self.vw_pv += t * vi
                self.vw_pv2 += t * t * vi
                self.vw_v += vi
            self.vw_sess = int(sess[-1])
        self.sq_run = float(f["squeeze_bars"][-1]) if n else 0.0
        self.prev_c = float(c[-1]) if n else NAN
        self.prev_h = float(h[-1]) if n else NAN
        self.prev_l = float(l[-1]) if n else NAN
        self._publish()

    def _publish(self) -> None:
        """Refresh the public views (FeatureFrame attributes) after appends. `n` moves last, so a reader on another
        thread never sees a row count its arrays do not hold yet."""
        n = self._len
        B = self._buf
        self.f = {k: B[k][:n] for k in (KEYS if n else _RAW)}
        self.open_time = self._ot[:n]
        self.close_time = self._ctb[:n]
        ct = self._ct
        if len(ct) < n:
            ct.extend(self._ctb[len(ct):n].tolist())
        self.n = n

    def _grow(self) -> None:
        cap, m = self._cap * 2, self._len
        for k, arr in list(self._buf.items()):
            new = np.empty(cap)
            new[:m] = arr[:m]
            self._buf[k] = new
        for name in ("_ot", "_ctb"):
            arr = getattr(self, name)
            new = np.empty(cap, dtype=np.int64)
            new[:m] = arr[:m]
            setattr(self, name, new)
        self._cap = cap

    def close_times(self) -> list[int]:
        return self._ct

    def bars(self) -> CandleArrays:
        """The bars since the anchor (the batch reference of this frame; `trades` is not kept: no feature uses it)."""
        n = self._len
        out = CandleArrays(self.tf, capacity=max(n, 16))
        out.open_time[:n] = self._ot[:n]
        out.close_time[:n] = self._ctb[:n]
        for k in _RAW:
            getattr(out, k)[:n] = self._buf[k][:n]
        out.taker_buy[:n] = self._buf["taker_buy"][:n]
        out.n = n
        return out

    # --- the incremental step -------------------------------------------------------------------------------------
    def append(self, ot: int, o: float, h: float, l: float, c: float, v: float, qv: float, tb: float, ct: int,
               publish: bool = True) -> None:
        """Append one closed bar and compute every column of its row."""
        i = self._len
        if i >= self._cap:
            self._grow()
        B = self._buf
        self._ot[i] = ot
        self._ctb[i] = ct
        B["open"][i], B["high"][i], B["low"][i], B["close"][i], B["volume"][i] = o, h, l, c, v
        B["quote_volume"][i], B["taker_buy"][i] = qv, tb
        cp, hp, lp = (self.prev_c, self.prev_h, self.prev_l) if i else (c, h, l)
        row = self._row
        # moving averages
        e = self.ema
        row["ema9"], row["ema21"] = e["ema9"].push(c), e["ema21"].push(c)
        row["ema50"], row["ema200"] = e["ema50"].push(c), e["ema200"].push(c)
        # rsi over diff(c, prepend=c[0])[1:]
        if i:
            d = c - cp
            up = d if d > 0 else 0.0
            dn = -d if d < 0 else 0.0
            for p, key in ((14, "rsi14"), (7, "rsi7"), (2, "rsi2")):
                wu, wd = self.rsi[p]
                au, ad = wu.push(up), wd.push(dn)
                if au != au:
                    row[key] = NAN
                else:
                    rs = au / ad if ad > 0 else math.inf
                    row[key] = 100 - 100 / (1 + rs)
        else:
            row["rsi14"] = row["rsi7"] = row["rsi2"] = NAN
        # true range and ATRs
        tr = h - l if i == 0 else _mx(h - l, _mx(abs(h - cp), abs(l - cp)))
        a14, a20, a10 = self.atr14.push(tr), self.atr20.push(tr), self.atr10.push(tr)
        row["atr"] = a14
        row["atr_pct"] = a14 / c if c > 0 else NAN
        B["atr"][i] = a14
        row["atr_rank"] = self._prank(B["atr"], i, 200)
        # bollinger
        self.cum_c.push(c)
        if i >= 19:
            mid = self.cum_c.sum(20) / 20
            sd = float(B["close"][i - 19:i + 1].std(ddof=0))
        else:
            mid = sd = NAN
        bup, blo = mid + 2.0 * sd, mid - 2.0 * sd
        width = (bup - blo) / mid if mid != 0 else NAN
        row["bb_mid"], row["bb_up"], row["bb_lo"], row["bb_width"], row["bb_sd"] = mid, bup, blo, width, sd
        B["bb_width"][i] = width
        row["bb_width_rank"] = self._prank(B["bb_width"], i, 120)
        # keltner and squeeze
        kmid = self.kc_ema.push(c)
        kup, klo = kmid + 1.5 * a20, kmid - 1.5 * a20
        row["kc_mid"], row["kc_up"], row["kc_lo"] = kmid, kup, klo
        sq = bup < kup and blo > klo
        row["squeeze"] = 1.0 if sq else 0.0
        self.sq_run = self.sq_run + 1.0 if sq else 0.0
        row["squeeze_bars"] = self.sq_run
        # session vwap
        sess = ot // 86_400_000
        if sess != self.vw_sess:
            self.vw_sess = sess
            self.vw_pv = self.vw_v = self.vw_pv2 = 0.0
        t = (h + l + c) / 3
        self.vw_pv += t * v
        self.vw_pv2 += t * t * v
        self.vw_v += v
        if self.vw_v > 0:
            m = self.vw_pv / self.vw_v
            row["vwap"] = m
            row["vwap_sd"] = math.sqrt(max(self.vw_pv2 / self.vw_v - m * m, 0.0))
        else:
            row["vwap"] = row["vwap_sd"] = NAN
        # supertrend (10, 3.0)
        hl2 = (h + l) / 2
        ub, lb = hl2 + 3.0 * a10, hl2 - 3.0 * a10
        line, dr = NAN, 0
        if self.st_started:
            pu, pl = self.st_ub, self.st_lb
            u = ub if (ub < pu or cp > pu) else pu
            lo = lb if (lb > pl or cp < pl) else pl
            prev_dir = self.st_dir if self.st_dir != 0 else 1
            if prev_dir == 1:
                dr = -1 if c < lo else 1
            else:
                dr = 1 if c > u else -1
            line = lo if dr == 1 else u
            self.st_ub, self.st_lb, self.st_dir = u, lo, dr
        elif a10 == a10:  # the first bar with an ATR: the bands start here, the line the bar after
            self.st_started, self.st_ub, self.st_lb, self.st_dir = True, ub, lb, 0
        row["st_line"], row["st_dir"] = line, float(dr)
        # adx (14)
        hu = h - hp
        ld = -(l - lp)
        sp = self.pdm.push(hu if (hu > ld and hu > 0) else 0.0)
        sm = self.mdm.push(ld if (ld > hu and ld > 0) else 0.0)
        pdi = 100 * (sp / a14) if a14 > 0 else NAN
        mdi = 100 * (sm / a14) if a14 > 0 else NAN
        den = pdi + mdi
        dx = 100 * (abs(pdi - mdi) / den) if den > 0 else 100 * 0.0
        if pdi != pdi:
            dx = NAN
        row["adx"], row["pdi"], row["mdi"] = self.adx.push(dx), pdi, mdi
        # macd (12, 26, 9)
        mac = self.ema12.push(c) - self.ema26.push(c)
        sig = self.sig.push(mac)
        row["macd"], row["macd_sig"], row["macd_hist"] = mac, sig, mac - sig
        # donchian / rolling extremes
        hi_a, lo_a = B["high"], B["low"]
        for L, khh, kll, kdh, kdl in _DC:
            if i >= L - 1:
                hh = float(hi_a[i - L + 1:i + 1].max())
                ll = float(lo_a[i - L + 1:i + 1].min())
            else:
                hh = ll = NAN
            row[kdh] = float(B[khh][i - 1]) if i else NAN
            row[kdl] = float(B[kll][i - 1]) if i else NAN
            row[khh], row[kll] = hh, ll
        # choppiness (14)
        self.cum_tr.push(tr)
        s = self.cum_tr.sum(14)
        if i >= 13:
            rng14 = float(hi_a[i - 13:i + 1].max()) - float(lo_a[i - 13:i + 1].min())
        else:
            rng14 = NAN
        if s == s and rng14 > 0 and s > 0:
            row["chop"] = float(100 * np.log10(s / rng14) / _LOG10_14)
        else:
            row["chop"] = NAN
        # volume
        self.cum_v.push(v)
        if i >= 49:
            vm = self.cum_v.sum(50) / 50
            vs = float(B["volume"][i - 49:i + 1].std(ddof=0))
            row["vol_z"] = (v - vm) / vs if (vs == vs and vs > 0) else NAN
        else:
            row["vol_z"] = NAN
        row["vol_sma"] = self.cum_v.sum(20) / 20 if i >= 19 else NAN
        trr = tb / v if v > 0 else 0.5
        row["taker_ratio"] = trr
        self.cum_trr.push(trr)
        row["taker_ratio3"] = self.cum_trr.sum(3) / 3 if i >= 2 else NAN
        self.cum_delta.push(2 * tb - v)
        num, den20 = self.cum_delta.sum(20), self.cum_v.sum(20)
        row["cvd20"] = num / den20 if (den20 == den20 and den20 > 0) else NAN
        # regression slope (20)
        if i >= 19:
            w = B["close"][i - 19:i + 1]
            slope = (w * _T20).sum() / _DENOM20
            level = w.mean()
            row["slope20"] = float(slope / level) if level != 0 else NAN
        else:
            row["slope20"] = NAN
        # hurst proxy (100): log returns and q-sums first
        Lc = float(np.log(_mx(c, 1e-12)))
        lr = 0.0 if i == 0 else Lc - self.L_prev
        self.L_prev = Lc
        B["lr"][i] = lr
        self.cum_lr.push(lr)
        ag = self.cum_lr.sum(_HQ)
        B["agg"][i] = ag if ag == ag else 0.0
        row["hurst"] = NAN
        if i >= _HN + _HQ - 1:
            row["hurst"] = self._hurst(i)
            if i == _HN + _HQ - 1:  # the batch is all NaN below n + q bars, then fills from bar n - 1 on
                for j in range(_HN - 1, i):
                    B["hurst"][j] = self._hurst(j)
        # candle shape
        rng = _mx(h - l, 1e-12)
        row["body_ratio"] = abs(c - o) / rng
        row["ret1"] = (c - cp) / (c if c > 0 else 1.0)
        row["upper_wick"] = (h - _mx(o, c)) / rng
        row["lower_wick"] = (_mn(o, c) - l) / rng
        for k, val in row.items():
            B[k][i] = val
        self.prev_c, self.prev_h, self.prev_l = c, h, l
        self._len = i + 1
        self.rows_appended += 1
        if publish:
            self._publish()

    @staticmethod
    def _prank(arr: np.ndarray, i: int, n: int) -> float:
        """percentile_rank(nan_to_num(arr), n) at row i."""
        if i < n - 1:
            return NAN
        w = np.nan_to_num(arr[i - n + 1:i + 1], nan=0.0)
        return int(np.count_nonzero(w <= w[-1])) / n

    def _hurst(self, j: int) -> float:
        """hurst_proxy(c, 100) at row j from the hidden log-return and q-sum columns."""
        w1 = self._buf["lr"][j - _HN + 1:j + 1]
        wq = self._buf["agg"][j - _HN + 1 + _HQ - 1:j + 1]
        mu = w1.mean()
        var1 = w1.var(ddof=1)
        varq = ((wq - _HQ * mu) ** 2).sum() / _HM
        vr = varq / var1 if var1 > 0 else 1.0
        return float(0.5 + 0.5 * np.tanh(np.log(_mx(float(vr), 1e-9))))


_DC = tuple((L, f"hh{L}", f"ll{L}", f"dc_hi{L}", f"dc_lo{L}") for L in (10, 20, 50))


# --- live frame set ------------------------------------------------------------------------------------------------
class _Group:
    """The open group of a resampled timeframe: id, bars seen, and whether it was already emitted."""
    __slots__ = ("gid", "cnt", "emitted")

    def __init__(self, gid: int | None = None, cnt: int = 0, emitted: bool = False):
        self.gid, self.cnt, self.emitted = gid, cnt, emitted


def _tail_group(base: CandleArrays, tf: str) -> _Group:
    """The trailing group of `base` as resample() judges it: complete when it holds every base bar of the period or its
    last bar closes at the period close (then it is already a bar of the resampled frame)."""
    n = base.n
    if n == 0:
        return _Group()
    tf_ms = TF_MS[tf]
    ot = base.open_time[:n]
    gid = int(ot[-1]) // tf_ms
    cnt = n - int(np.searchsorted(ot // tf_ms, gid))
    per = max(1, tf_ms // base.tf_ms)
    emitted = cnt >= per or int(base.close_time[n - 1]) >= (gid + 1) * tf_ms - 1
    return _Group(gid, cnt, emitted)


def _frame_equal(a: FeatureFrame, b: FeatureFrame) -> list[str]:
    """Columns that differ (NaN positions or IEEE values), [] when the frames are identical."""
    if a.n != b.n or not np.array_equal(a.close_time, b.close_time):
        return ["<bars>"]
    return [k for k in b.f if k not in a.f or not np.array_equal(a.f[k], b.f[k], equal_nan=True)]


class LiveFrames:
    """Every feature frame of one symbol's live MarketView, advanced from its 1m base bar by bar (F7)."""

    def __init__(self, symbol: str, verify: bool = True):
        self.symbol = symbol
        self.verify = verify
        self.frames: dict[str, IncrementalFrame] = {}
        self.groups: dict[str, _Group] = {}
        self.base: CandleArrays | None = None
        self.last_ot: int | None = None
        self.seen_edits = self.seen_trims = 0
        self.next_reanchor: dict[str, int] = {}
        self.stale_long: set[str] = set()
        self.failed: str = ""  # why incremental updates were given up for this symbol ("" while they run)
        self.stats = {"steps": 0, "bars": 0, "builds": 0, "reanchors": 0, "verified": 0, "long_rebuilds": 0}

    # --- full builds ------------------------------------------------------------------------------------------------
    def build(self, base: CandleArrays) -> dict[str, IncrementalFrame]:
        """Build every frame from `base` exactly as MarketView.rebuild(live=True) does (the 4h / 1d frames over the
        whole base), and the state to continue from it."""
        self.base = base
        self.frames = {}
        self.groups = {}
        for tf in TIMEFRAMES + LONG_TIMEFRAMES:
            self.frames[tf] = self._fresh(tf, base)
            if tf != "1m":
                self.groups[tf] = _tail_group(base, tf)
        self.stale_long.clear()
        self._mark(base)
        self.stats["builds"] += 1
        return self.frames

    def _fresh(self, tf: str, base: CandleArrays) -> IncrementalFrame:
        if tf in LONG_TIMEFRAMES:
            bars = resample(base, tf)
            fr = IncrementalFrame(tf, bars, extra=64)
        else:
            bars = base.tail(LIVE_WINDOW[tf]) if tf == "1m" else resample_tail(base, tf, LIVE_WINDOW[tf])
            every = REANCHOR_ROWS[tf]
            fr = IncrementalFrame(tf, bars, extra=every + 8)
            # spread the symbols' re-anchors over the period (all symbols start together)
            first = every - zlib.crc32(f"{self.symbol}:{tf}".encode()) % every if tf not in self.next_reanchor else every
            self.next_reanchor[tf] = first
        return fr

    def _mark(self, base: CandleArrays) -> None:
        self.last_ot = int(base.open_time[base.n - 1]) if base.n else None
        self.seen_edits, self.seen_trims = base.edits, base.trims

    # --- the per-bar step ---------------------------------------------------------------------------------------------
    def _new_from(self, base: CandleArrays) -> int | None:
        """Index of the first base bar not consumed yet, or None when the base changed in a way that needs a build."""
        if base is not self.base or base.edits != self.seen_edits or self.failed:
            return None
        n = base.n
        if self.last_ot is None:
            return 0 if base.trims == self.seen_trims else None
        if n == 0:
            return None
        j = int(np.searchsorted(base.open_time[:n], self.last_ot))
        if j >= n or int(base.open_time[j]) != self.last_ot:
            return None
        return j + 1

    def _emissions(self, tf: str, base: CandleArrays, j: int) -> int:
        """How many tf bars base bars [j, n) close (the group rule of advance(), without changing the state)."""
        g, tf_ms = self.groups[tf], TF_MS[tf]
        gid, cnt, emitted, out = g.gid, g.cnt, g.emitted, 0
        per = tf_ms // 60_000
        for k in range(j, base.n):
            b = int(base.open_time[k]) // tf_ms
            if b != gid:
                if gid is not None and not emitted and cnt > 0:
                    out += 1
                gid, cnt, emitted = b, 0, False
            cnt += 1
            if not emitted and (cnt >= per or int(base.close_time[k]) >= (b + 1) * tf_ms - 1):
                out += 1
                emitted = True
        return out

    def cheap(self, base: CandleArrays) -> bool:
        """True when advance(base) only appends (no build, re-anchor or 4h / 1d rebuild): well under a millisecond, so
        it can run on the event loop; otherwise the caller runs it in a worker thread."""
        j = self._new_from(base)
        if j is None:
            return False
        new = base.n - j
        if new > 3:
            return False
        if self.frames["1m"].rows_appended + new >= self.next_reanchor["1m"]:
            return False
        stale = self.stale_long | (set(LONG_TIMEFRAMES) if base.trims != self.seen_trims else set())
        for tf in self.groups:
            if tf in REANCHOR_ROWS or tf in stale:
                e = self._emissions(tf, base, j)
                if e and (tf in stale or self.frames[tf].rows_appended + e >= self.next_reanchor[tf]):
                    return False
        return True

    def advance(self, base: CandleArrays) -> bool:
        """Bring every frame up to `base`. Returns True when it was incremental, False when it rebuilt everything (a
        new base, an edited bar). After a failed check (`failed` set) the caller must stop using these frames."""
        j = self._new_from(base)
        if j is None:
            self.build(base)
            return False
        try:
            return self._step(base, j)
        except Exception:  # noqa: BLE001 - an unexpected base shape must cost a rebuild, never the bar's decisions
            log.warning("incremental features of %s: step failed, rebuilding", self.symbol, exc_info=True)
            self.build(base)
            return False

    def _step(self, base: CandleArrays, j: int) -> bool:
        if base.trims != self.seen_trims:
            # the 4h / 1d frames are anchored at the base start: rebuilt at their next close, like the batch path
            self.stale_long.update(LONG_TIMEFRAMES)
        n = base.n
        ot, ct = base.open_time, base.close_time
        o, h, l, c = base.open, base.high, base.low, base.close
        v, qv, tb = base.volume, base.quote_volume, base.taker_buy
        f1 = self.frames["1m"]
        dirty = {"1m"}
        for k in range(j, n):
            bot, bct = int(ot[k]), int(ct[k])
            f1.append(bot, float(o[k]), float(h[k]), float(l[k]), float(c[k]), float(v[k]), float(qv[k]),
                      float(tb[k]), bct, publish=False)
            for tf, g in self.groups.items():
                tf_ms = TF_MS[tf]
                gid = bot // tf_ms
                if gid != g.gid:
                    if g.gid is not None and not g.emitted and g.cnt > 0:
                        self._emit(tf, base, k - g.cnt, k)  # finished: a later group has started
                        dirty.add(tf)
                    g.gid, g.cnt, g.emitted = gid, 0, False
                g.cnt += 1
                if not g.emitted and (g.cnt >= tf_ms // 60_000 or bct >= (gid + 1) * tf_ms - 1):
                    self._emit(tf, base, k + 1 - g.cnt, k + 1)
                    g.emitted = True
                    dirty.add(tf)
        for tf in dirty:
            self.frames[tf]._publish()
        self._mark(base)
        self.stats["steps"] += 1
        self.stats["bars"] += n - j
        for tf in REANCHOR_ROWS:
            if self.frames[tf].rows_appended >= self.next_reanchor[tf]:
                self._reanchor(tf, base)
                if self.failed:
                    return False
        return True

    def _emit(self, tf: str, base: CandleArrays, s: int, e: int) -> None:
        """Append the group of base bars [s, e) to the tf frame (aggregated exactly as resample() does)."""
        if s < 0:
            raise RuntimeError(f"{self.symbol} {tf}: group starts before the base")
        bar = resample(base.slice(s, e), tf, only_complete=False)
        if bar.n != 1:
            raise RuntimeError(f"{self.symbol} {tf}: base bars [{s}, {e}) are not one group")
        fr = self.frames[tf]
        if tf in self.stale_long:
            self._rebuild_long(tf, base, bar)
            return
        fr.append(int(bar.open_time[0]), float(bar.open[0]), float(bar.high[0]), float(bar.low[0]),
                  float(bar.close[0]), float(bar.volume[0]), float(bar.quote_volume[0]), float(bar.taker_buy[0]),
                  int(bar.close_time[0]), publish=False)

    def _rebuild_long(self, tf: str, base: CandleArrays, bar: CandleArrays) -> None:
        """A 4h / 1d frame after the base was trimmed: the batch over its cached bars from the new base start (the same
        bars resample(base) gives when the start is a period boundary) plus the bar that just closed."""
        self.stale_long.discard(tf)
        self.stats["long_rebuilds"] += 1
        fr = self.frames[tf]
        tf_ms = TF_MS[tf]
        first = int(base.open_time[0]) if base.n else 0
        cached = fr.bars()
        if first % tf_ms == 0:
            k = int(np.searchsorted(cached.open_time[:cached.n], first))
            bars = cached.slice(k, cached.n)
            if bars.n >= bars._cap:
                bars._grow()
            for name in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume",
                         "taker_buy"):
                getattr(bars, name)[bars.n] = getattr(bar, name)[0]
            bars.n += 1
        else:  # a start inside a period: the first group is partial, aggregate it again
            bars = resample(base.slice(0, base.n), tf)
            bars = bars.slice(0, int(np.searchsorted(bars.close_time[:bars.n], int(bar.close_time[0]), side="right")))
        self.frames[tf] = IncrementalFrame(tf, bars, extra=64)

    def _reanchor(self, tf: str, base: CandleArrays) -> None:
        """Check the frame against the batch over its own bars, then rebuild it from the base tail (as the per-bar
        rebuild did) so the anchor follows the window."""
        fr = self.frames[tf]
        if self.verify:
            bad = _frame_equal(fr, FeatureFrame(tf, fr.bars()))
            self.stats["verified"] += 1
            if bad:
                self.failed = f"{tf}: {', '.join(bad[:6])}"
                log.error("incremental features of %s diverged from the batch (%s); rebuilding every bar instead",
                          self.symbol, self.failed)
                return
        self.frames[tf] = self._fresh(tf, base)
        if tf != "1m":
            self.groups[tf] = _tail_group(base, tf)
        self.stats["reanchors"] += 1

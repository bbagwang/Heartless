"""Performance instrumentation (F7): the latency budget, clock drift, hot-path profiles and the feature benchmark.

Latency budget: the bot stamps each closed bar on its way from the exchange to an order. Per engine and stage the last
RESERVOIR samples are kept and summarised as p50 / p99 / max (milliseconds):

    close_to_recv      bar close (exchange time, corrected by the measured clock offset) -> kline received
    recv_to_features   kline received -> feature frames updated and positioned (the view is ready)
    feature_update     the feature step alone (incremental append or batch rebuild), sub-millisecond resolution
    close_to_decision  bar close -> the engine finished on_bar for that bar (decided, and sent what it decided)
    close_to_sent      bar close -> an order request leaves (entries and exits decided on that bar)
    sent_to_ack        order request sent -> the exchange's answer (or the paper simulator's) returned

`LATENCY.snapshot()` is what /health and the daily report show (see docs/PERFORMANCE.md for the one-line wiring of the
formatter); `LATENCY.lines()` renders it as short Korean lines. Clock drift: every server-time sample (REST
/fapi/v1/time, and the event time of stream messages) goes through `DRIFT`, which warns when the offset passes
DRIFT_WARN_MS. In a replay all times are virtual, so the stages measure the bot's own processing only.
"""
from __future__ import annotations

import cProfile
import logging
import pstats
import threading
import time
from collections import deque
from typing import Any, Callable

log = logging.getLogger(__name__)

RESERVOIR = 4096  # samples kept per (engine, stage)
STAGES = ("close_to_recv", "recv_to_features", "feature_update", "close_to_decision", "close_to_sent", "sent_to_ack")
STAGE_KO = {"close_to_recv": "봉 마감→수신", "recv_to_features": "수신→피처", "feature_update": "피처 갱신",
            "close_to_decision": "봉 마감→결정",
            "close_to_sent": "봉 마감→주문 전송", "sent_to_ack": "전송→응답"}
DRIFT_WARN_MS = 1000.0  # |local - server| above this is logged (recvWindow is 5000 ms)
DRIFT_RECHECK_MS = 250.0  # a later sample this much further out warns again


def percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method) of an already sorted list."""
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


class Reservoir:
    """The last `size` samples of one quantity."""
    __slots__ = ("vals", "count")

    def __init__(self, size: int = RESERVOIR):
        self.vals: deque[float] = deque(maxlen=size)
        self.count = 0

    def add(self, v: float) -> None:
        self.vals.append(float(v))
        self.count += 1

    def summary(self) -> dict[str, float]:
        s = sorted(self.vals)
        return {"n": self.count, "p50": round(percentile(s, 50), 3), "p99": round(percentile(s, 99), 3),
                "max": round(s[-1], 3) if s else float("nan")}


class LatencyBudget:
    """Per engine and stage latency samples (thread-safe: the feature step may run in a worker thread)."""

    def __init__(self, size: int = RESERVOIR):
        self.size = size
        self._r: dict[tuple[str, str], Reservoir] = {}
        self._lock = threading.Lock()

    def record(self, engine: str, stage: str, ms: float) -> None:
        if ms != ms:
            return
        with self._lock:
            r = self._r.get((engine, stage))
            if r is None:
                r = self._r[(engine, stage)] = Reservoir(self.size)
            r.add(ms)

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        """{engine: {stage: {n, p50, p99, max}}} in milliseconds, stages in STAGES order."""
        with self._lock:
            items = list(self._r.items())
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (eng, stage), r in sorted(items, key=lambda kv: (kv[0][0], STAGES.index(kv[0][1])
                                                            if kv[0][1] in STAGES else 99)):
            out.setdefault(eng, {})[stage] = r.summary()
        return out

    def lines(self) -> list[str]:
        """One line per engine: `paper: 봉 마감→결정 p50 12ms / p99 85ms · 전송→응답 ...`."""
        out = []
        for eng, stages in self.snapshot().items():
            parts = [f"{STAGE_KO.get(st, st)} p50 {_ms(v['p50'])} / p99 {_ms(v['p99'])}" for st, v in stages.items()]
            out.append(f"{eng}: " + " · ".join(parts))
        return out

    def reset(self) -> None:
        with self._lock:
            self._r.clear()


def _ms(v: float) -> str:
    return f"{v:.2f}ms" if v < 10 else f"{v:.0f}ms"


class ClockDrift:
    """Local clock minus exchange clock, from REST time syncs and stream event times."""

    def __init__(self, warn_ms: float = DRIFT_WARN_MS):
        self.warn_ms = warn_ms
        self.last: dict[str, float] = {}
        self.worst = 0.0
        self.warnings = 0
        self._warned_at = 0.0

    def observe(self, offset_ms: float, source: str = "rest", rtt_ms: float | None = None) -> None:
        """offset_ms = local - server (positive: the local clock runs ahead)."""
        self.last[source] = float(offset_ms)
        a = abs(offset_ms)
        self.worst = max(self.worst, a)
        if a > self.warn_ms and a > self._warned_at + DRIFT_RECHECK_MS:
            self._warned_at = a
            self.warnings += 1
            log.warning("clock drift: local clock minus exchange clock is %+.0f ms by %s%s (warning above %.0f ms; "
                        "signed requests carry the measured offset, but check NTP on this host)", offset_ms, source,
                        f" (round trip {rtt_ms:.0f} ms)" if rtt_ms is not None and rtt_ms == rtt_ms else
                        " (stream event times: includes the shortest network delay)" if source != "rest" else "",
                        self.warn_ms)
        elif a < self.warn_ms / 2:
            self._warned_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        return {"offset_ms": {k: round(v, 1) for k, v in self.last.items()}, "worst_ms": round(self.worst, 1),
                "warnings": self.warnings, "warn_ms": self.warn_ms}


LATENCY = LatencyBudget()
DRIFT = ClockDrift()


def reset() -> None:
    """Forget every sample (a replay starts its own budget)."""
    LATENCY.reset()
    DRIFT.__init__(DRIFT.warn_ms)


def exchange_to_local(ts_ms: int | float) -> float:
    """An exchange timestamp on the local clock, using the last measured REST offset (0 before the first sync and in a
    replay, whose clocks are the same virtual clock)."""
    return float(ts_ms) + DRIFT.last.get("rest", 0.0)


def latency_snapshot() -> dict[str, Any]:
    """The accessor /health and the daily report read: the latency budget plus the clock drift."""
    return {"latency": LATENCY.snapshot(), "clock": DRIFT.snapshot()}


# --- event loop ---------------------------------------------------------------------------------------------------
_UVLOOP: Any = False  # the uvloop module once looked up (None: not installed); a failed import is not retried


def _uvloop():
    global _UVLOOP
    if _UVLOOP is False:
        try:
            import uvloop  # type: ignore[import-not-found]
        except ImportError:
            uvloop = None
        _UVLOOP = uvloop
    return _UVLOOP


def loop_factory(enabled: bool = True) -> Callable[[], Any] | None:
    """uvloop's event loop factory when uvloop is installed (an optional speed-up of the socket and timer handling,
    never required), else None (asyncio's default loop). The entry point passes it on:
    `asyncio.run(main(), loop_factory=perf.loop_factory())` (Python 3.12+)."""
    mod = _uvloop() if enabled else None
    return mod.new_event_loop if mod is not None else None


def runtime_info() -> dict[str, str]:
    """Which optional speed-ups this process has (logged at start-up and shown by /health)."""
    import asyncio

    from heartless.exchange.binance_ws import JSON_ENGINE

    try:
        loop = type(asyncio.get_running_loop()).__module__.split(".")[0]
    except RuntimeError:
        loop = "-"
    return {"json": JSON_ENGINE, "loop": loop, "uvloop_installed": str(_uvloop() is not None).lower()}


# --- hot-path profiles ---------------------------------------------------------------------------------------------
class Profiler:
    """cProfile around a block; top() lists the functions that took the most time."""

    def __init__(self) -> None:
        self.prof = cProfile.Profile()
        self.wall = 0.0

    def __enter__(self) -> "Profiler":
        self._t0 = time.perf_counter()
        self.prof.enable()
        return self

    def __exit__(self, *exc) -> None:
        self.prof.disable()
        self.wall = time.perf_counter() - self._t0

    def top(self, n: int = 25, sort: str = "tottime") -> list[dict[str, Any]]:
        st = pstats.Stats(self.prof)
        rows = []
        for (file, line, func), (cc, nc, tt, ct, _callers) in st.stats.items():  # type: ignore[attr-defined]
            short = file.split("/heartless/")[-1] if "/heartless/" in file else file.rsplit("/", 1)[-1]
            rows.append({"function": f"{short}:{line}({func})", "calls": nc, "tottime_s": round(tt, 4),
                         "cumtime_s": round(ct, 4)})
        rows.sort(key=lambda r: -r["tottime_s" if sort == "tottime" else "cumtime_s"])
        return rows[:n]


def format_profile(rows: list[dict[str, Any]], wall: float | None = None) -> str:
    head = f"hot path (top {len(rows)} by own time" + (f", profiled wall {wall:.1f}s)" if wall else ")")
    lines = [head]
    for r in rows:
        lines.append(f"  {r['tottime_s']:9.3f}s own {r['cumtime_s']:9.3f}s cum {r['calls']:>9} calls  {r['function']}")
    return "\n".join(lines)


def timed(fn: Callable, *a, **k) -> tuple[Any, float]:
    """(fn(*a, **k), elapsed milliseconds)."""
    t0 = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t0) * 1e3


# --- feature update benchmark ----------------------------------------------------------------------------------------
def _frame_bytes(view) -> int:
    total = 0
    for fr in view.frames.values():
        buf = getattr(fr, "_buf", None)
        if buf is not None:
            total += sum(a.nbytes for a in buf.values())
        else:
            total += sum(a.nbytes for a in fr.f.values())
    return total


def bench_features(db_path: str, symbol: str, history_days: int = 23, minutes: int = 1440,
                   end_ms: int | None = None, deviation: bool = False) -> dict[str, Any]:
    """Per-bar cost of the live feature update on stored 1m bars (read-only): the per-bar batch rebuild the app did
    before F7 (app._rebuild_view of the timeframes whose period ended, then seek) against MarketView.advance + seek,
    over `minutes` bars after `history_days` of history. Also checks every frame is identical to the batch over its own
    bars at the end."""
    import sqlite3

    import numpy as np

    from heartless.app import Heartless
    from heartless.core.models import Candle
    from heartless.data.candles import CandleArrays
    from heartless.data.features import TIMEFRAMES, FeatureFrame, MarketView
    from heartless.util.timeutil import MS_DAY, MS_MINUTE

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    if end_ms is None:
        end_ms = int(con.execute("SELECT MAX(open_time) FROM candles WHERE symbol=?", (symbol,)).fetchone()[0]) + 1
    lo = end_ms - (history_days * MS_DAY + minutes * MS_MINUTE)
    rows = con.execute("SELECT open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, "
                       "close_time FROM candles WHERE symbol=? AND open_time>=? AND open_time<? ORDER BY open_time",
                       (symbol, lo, end_ms)).fetchall()
    con.close()
    candles = [Candle(int(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], int(r[7]), r[8], int(r[9]), True) for r in rows]
    n0 = len(candles) - minutes
    out: dict[str, Any] = {"symbol": symbol, "bars": len(candles), "updates": minutes}
    for mode in ("batch", "incremental"):
        ca = CandleArrays("1m", capacity=len(candles) + 16)
        ca.extend(candles[:n0])
        view = MarketView(symbol, incremental=mode == "incremental")
        _, build_ms = timed(view.rebuild, ca, True)
        ts: list[float] = []
        for c in candles[n0:]:
            ca.append(c)
            t0 = time.perf_counter()
            if mode == "batch":
                m = c.open_time // MS_MINUTE
                tfs = ["1m"] + [tf for tf, k in (("5m", 5), ("15m", 15), ("1h", 60)) if (m + 1) % k == 0]
                Heartless._rebuild_view(view, ca, tfs)
            else:
                view.advance(ca)
            view.seek(c.close_time)
            ts.append((time.perf_counter() - t0) * 1e3)
        arr = np.array(ts)
        out[mode] = {"build_ms": round(build_ms, 2), "p50_ms": round(float(np.percentile(arr, 50)), 4),
                     "p99_ms": round(float(np.percentile(arr, 99)), 4), "max_ms": round(float(arr.max()), 2),
                     "mean_ms": round(float(arr.mean()), 4), "frame_mb": round(_frame_bytes(view) / 1e6, 2),
                     "cpu_s_per_min_200_symbols": round(float(arr.mean()) * 200 / 1e3, 3)}
        if mode == "incremental":
            lf = view.live
            out[mode]["stats"] = dict(lf.stats) if lf is not None else None
            from heartless.data.incremental import _frame_equal

            out[mode]["identical_to_batch"] = all(
                not _frame_equal(view.frames[tf], FeatureFrame(tf, view.frames[tf].bars())) for tf in TIMEFRAMES)
    out["speedup_p50"] = round(out["batch"]["p50_ms"] / max(out["incremental"]["p50_ms"], 1e-9), 1)
    if deviation:
        out["vs_sliding_window"] = _sliding_deviation(candles, n0, symbol)
    return out


def _sliding_deviation(candles, n0: int, symbol: str) -> dict[str, Any]:
    """How far the incremental (anchored) live values are from the per-bar batch rebuild over the last LIVE_WINDOW
    bars (a sliding anchor) on the newest row of each base frame, per feature: the largest relative difference and
    the share of values that are bit-identical. Not a correctness check (both are exact batch results over different
    first bars); it sizes the anchor effect (EMA seeds, cumulative sums)."""
    import numpy as np

    from heartless.app import Heartless
    from heartless.data.candles import CandleArrays
    from heartless.data.features import TIMEFRAMES, MarketView
    from heartless.util.timeutil import MS_MINUTE

    ca_a = CandleArrays("1m", capacity=len(candles) + 16)
    ca_a.extend(candles[:n0])
    ca_b = CandleArrays("1m", capacity=len(candles) + 16)
    ca_b.extend(candles[:n0])
    inc, legacy = MarketView(symbol, incremental=True), MarketView(symbol)
    inc.rebuild(ca_a, True)
    legacy.rebuild(ca_b, True)
    worst: dict[str, float] = {}
    same = total = 0
    for c in candles[n0:]:
        ca_a.append(c)
        ca_b.append(c)
        inc.advance(ca_a)
        m = c.open_time // MS_MINUTE
        tfs = ["1m"] + [tf for tf, k in (("5m", 5), ("15m", 15), ("1h", 60)) if (m + 1) % k == 0]
        Heartless._rebuild_view(legacy, ca_b, tfs)
        for tf in tfs:
            a, b = inc.frames[tf], legacy.frames[tf]
            if not a.n or not b.n or a.close_time[-1] != b.close_time[-1]:
                continue
            for k, arr in b.f.items():
                x, y = float(a.f[k][-1]), float(arr[-1])
                total += 1
                if x == y or (x != x and y != y):
                    same += 1
                    continue
                rel = abs(x - y) / max(abs(y), 1e-12) if y == y and x == x else float("inf")
                key = f"{tf}.{k}"
                if rel > worst.get(key, 0.0):
                    worst[key] = rel
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:10]
    return {"values": total, "bit_identical_share": round(same / max(total, 1), 6),
            "features_differing": len(worst), "largest_relative": {k: float(f"{v:.3g}") for k, v in top}}


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="python -m heartless.util.perf")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="per-bar feature update cost, batch vs incremental (reads the DB read-only)")
    b.add_argument("--db", required=True)
    b.add_argument("--symbols", default="BTCUSDT")
    b.add_argument("--minutes", type=int, default=1440)
    b.add_argument("--history-days", type=int, default=23)
    b.add_argument("--out", default=None)
    b.add_argument("--deviation", action="store_true", help="also size the anchor effect against the sliding rebuild")
    a = ap.parse_args(argv)
    res = [bench_features(a.db, s.strip().upper(), a.history_days, a.minutes, deviation=a.deviation)
           for s in a.symbols.split(",") if s.strip()]
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()

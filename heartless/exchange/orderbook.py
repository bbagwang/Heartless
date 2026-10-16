"""Local order books (F7): Binance USD-M depth streams -> per-symbol book features for the alphas (ctx.book).

Two stream modes (ORDERBOOK_MODE):

* "partial" (default): `<symbol>@depth20@100ms` pushes the top 20 levels of each side; every message replaces the book.
  No REST snapshots and no sequencing state; depth beyond the 20th level is not seen (BookFeatures.levels says how
  many levels the figures cover).
* "diff": `<symbol>@depth@100ms` diff events maintained on top of a REST snapshot (GET /fapi/v1/depth), following
  Binance's procedure for USD-M futures: buffer the stream, fetch a snapshot, drop events with u < lastUpdateId, the
  first applied event must have U <= lastUpdateId <= u (or directly follow it: pu == lastUpdateId), every later
  event's pu must equal the previous event's u, and
  any break (a lost message, an out-of-order event, a reconnect) discards the book and starts again from a snapshot.
  Quantities are absolute; 0 removes the level.

Features (`BookFeatures`, all from the same book at one instant): spread in bps of the mid, the quote notional resting
within k bps of the mid on each side (k = ORDERBOOK_DEPTH_BPS), the imbalance of those two depths, and the book's age.
A symbol without a usable book reads `EMPTY_BOOK` (NaN figures, source "none").

History has no order book. `bar_book_proxy()` derives a deliberately conservative stand-in from closed 1m bars (wider
spread, thinner depth, no imbalance; docs/PERFORMANCE.md), used by the replay (`heartless/replay/book.py`) and meant for
the backtester, so an alpha sees the same field names everywhere and a rule that needs a tight, deep book is not
flattered on history.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from statistics import median
from typing import Any, Awaitable, Callable

from heartless.util.timeutil import now_ms

log = logging.getLogger(__name__)

NAN = float("nan")
SNAPSHOT_LIMIT = 100  # levels per side of a resync snapshot (request weight 5; 1000 levels cost 20)
SNAPSHOT_MIN_GAP_S = 0.25  # at most ~4 snapshot requests a second across symbols (a reconnect resyncs all of them)
BUFFER_MAX = 5_000  # diff events kept while waiting for a snapshot
PRUNE_LEVELS = 4_000  # a side holding more levels drops those further than PRUNE_FRAC from the mid
PRUNE_FRAC = 0.05


@dataclass(frozen=True, slots=True)
class BookFeatures:
    spread_bps: float = NAN  # (best ask - best bid) / mid, bps
    imbalance: float = NAN  # (depth_bid_k - depth_ask_k) / (depth_bid_k + depth_ask_k), -1..1
    depth_bid_k: float = NAN  # quote notional of the bids within k bps below the mid
    depth_ask_k: float = NAN  # quote notional of the asks within k bps above the mid
    staleness_ms: float = NAN  # age of the book: now - when its last update arrived (proxy: since the bar closed)
    mid: float = NAN
    k_bps: float = NAN
    levels: int = 0  # levels per side the figures could see (partial stream: at most 20)
    source: str = "none"  # "stream", "proxy" or "none"

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


EMPTY_BOOK = BookFeatures()


def book_of(ctx) -> BookFeatures:
    """ctx.book where the context carries one (the live bot and the replay set it), else EMPTY_BOOK (a backtest
    context built before the field existed)."""
    return getattr(ctx, "book", None) or EMPTY_BOOK


def features_from_levels(bids: dict[float, float], asks: dict[float, float], k_bps: float, age_ms: float,
                         source: str = "stream") -> BookFeatures:
    if not bids or not asks:
        return BookFeatures(staleness_ms=age_ms, k_bps=k_bps, source=source)
    bb, ba = max(bids), min(asks)
    if not (bb > 0 and ba > bb):  # crossed or locked: a transient state of an out-of-sync book, not a price
        return BookFeatures(staleness_ms=age_ms, k_bps=k_bps, source=source)
    mid = (bb + ba) / 2
    lo, hi = mid * (1 - k_bps / 1e4), mid * (1 + k_bps / 1e4)
    db = sum(p * q for p, q in bids.items() if p >= lo)
    da = sum(p * q for p, q in asks.items() if p <= hi)
    tot = db + da
    return BookFeatures(spread_bps=(ba - bb) / mid * 1e4, imbalance=(db - da) / tot if tot > 0 else 0.0,
                        depth_bid_k=db, depth_ask_k=da, staleness_ms=age_ms, mid=mid, k_bps=k_bps,
                        levels=min(len(bids), len(asks)), source=source)


def _levels(rows) -> dict[float, float]:
    out: dict[float, float] = {}
    for p, q in rows:
        qf = float(q)
        if qf > 0:
            out[float(p)] = qf
    return out


class LocalBook:
    """One symbol's book (see the module doc for the two modes)."""

    def __init__(self, symbol: str, mode: str = "partial"):
        self.symbol = symbol
        self.mode = mode
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_u = 0  # last applied update id (diff: the chain head; partial: the newest message)
        self.synced = False
        self.first_after_snapshot = False  # diff: the next event must straddle the snapshot's lastUpdateId
        self.buffer: list[dict] = []
        self.snapshot_pending = False
        self.recv_ms = 0  # local time the last applied update arrived
        self.event_ms = 0  # exchange event time of the last applied update
        self.stats = {"updates": 0, "snapshots": 0, "resyncs": 0, "stale": 0, "dropped": 0}

    # --- partial depth ----------------------------------------------------------------------------------------------
    def on_partial(self, ev: dict, recv_ms: int) -> None:
        u = int(ev.get("u") or 0)
        if u and u <= self.last_u:
            self.stats["stale"] += 1
            return
        self.bids, self.asks = _levels(ev.get("b") or ()), _levels(ev.get("a") or ())
        self.last_u = u or self.last_u
        self.synced = True
        self._seen(ev, recv_ms)

    # --- diff depth -------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """Forget the book (a gap, a reconnect): the next event starts a resync."""
        self.bids, self.asks = {}, {}
        self.synced = self.first_after_snapshot = False
        self.last_u = 0

    def on_diff(self, ev: dict, recv_ms: int) -> bool:
        """Apply one diff event. Returns True when the book needs a (new) REST snapshot."""
        if not self.synced:
            self._buffer(ev)
            return not self.snapshot_pending
        U, u, pu = int(ev["U"]), int(ev["u"]), int(ev.get("pu", -1))
        if self.first_after_snapshot:
            if u < self.last_u:
                self.stats["stale"] += 1
                return False
            if U > self.last_u and pu != self.last_u:  # neither straddles nor follows the snapshot id: a gap
                return self._resync(ev)
            self.first_after_snapshot = False
        elif u <= self.last_u:  # already applied (a repeated or late message)
            self.stats["stale"] += 1
            return False
        elif pu != self.last_u:
            return self._resync(ev)
        self._apply(ev, recv_ms)
        return False

    def apply_snapshot(self, snap: dict, recv_ms: int) -> bool:
        """Install a REST snapshot and replay the buffered events. True when the book is in sync; False when the
        buffer does not connect to the snapshot (fetch another one)."""
        self.snapshot_pending = False
        L = int(snap["lastUpdateId"])
        self.stats["snapshots"] += 1
        self.bids, self.asks = _levels(snap.get("bids") or ()), _levels(snap.get("asks") or ())
        self.last_u = L
        self.recv_ms = recv_ms
        self.event_ms = int(snap.get("E") or snap.get("T") or 0)
        events = sorted(self.buffer, key=lambda e: int(e["u"]))
        self.buffer = []
        first = True
        for i, ev in enumerate(events):
            U, u, pu = int(ev["U"]), int(ev["u"]), int(ev.get("pu", -1))
            if u < L:
                continue
            if first:
                if U > L and pu != L:  # the stream started after the snapshot's id: wait for a newer snapshot
                    self.synced = False
                    self.buffer = events[i:]
                    return False
                first = False
            elif pu != self.last_u:
                self.synced = False
                self.buffer = events[i:]
                self.stats["resyncs"] += 1
                return False
            self._apply(ev, recv_ms)
        self.synced = True
        self.first_after_snapshot = first  # no buffered event reached the snapshot: the next live one must
        return True

    def _resync(self, ev: dict) -> bool:
        self.stats["resyncs"] += 1
        log.debug("%s order book out of sequence (u=%s pu=%s, head %s): resync", self.symbol, ev.get("u"),
                  ev.get("pu"), self.last_u)
        self.reset()
        self._buffer(ev)
        return not self.snapshot_pending

    def _buffer(self, ev: dict) -> None:
        self.buffer.append(ev)
        if len(self.buffer) > BUFFER_MAX:
            del self.buffer[: len(self.buffer) - BUFFER_MAX]
            self.stats["dropped"] += 1

    def _apply(self, ev: dict, recv_ms: int) -> None:
        for side, book in (("b", self.bids), ("a", self.asks)):
            for p, q in ev.get(side) or ():
                pf, qf = float(p), float(q)
                if qf > 0:
                    book[pf] = qf
                else:
                    book.pop(pf, None)  # removing a level we never had is normal
        self.last_u = int(ev["u"])
        self._seen(ev, recv_ms)
        if len(self.bids) > PRUNE_LEVELS or len(self.asks) > PRUNE_LEVELS:
            self._prune()

    def _prune(self) -> None:
        if not self.bids or not self.asks:
            return
        mid = (max(self.bids) + min(self.asks)) / 2
        lo, hi = mid * (1 - PRUNE_FRAC), mid * (1 + PRUNE_FRAC)
        self.bids = {p: q for p, q in self.bids.items() if p >= lo}
        self.asks = {p: q for p, q in self.asks.items() if p <= hi}

    def _seen(self, ev: dict, recv_ms: int) -> None:
        self.recv_ms = recv_ms
        self.event_ms = int(ev.get("E") or 0)
        self.stats["updates"] += 1

    def features(self, now: int, k_bps: float) -> BookFeatures:
        if not self.synced:
            return BookFeatures(k_bps=k_bps, source="stream")
        return features_from_levels(self.bids, self.asks, k_bps, float(now - self.recv_ms))


SnapshotFetch = Callable[[str, int], Awaitable[dict]]


class OrderBooks:
    """The local books of the traded symbols, fed by BookStream messages (heartless/exchange/binance_ws.py)."""

    def __init__(self, mode: str = "partial", depth_bps: float = 10.0, fetch_snapshot: SnapshotFetch | None = None,
                 snapshot_limit: int = SNAPSHOT_LIMIT):
        if mode not in ("partial", "diff"):
            raise ValueError(f"order book mode {mode!r}: expected 'partial' or 'diff'")
        self.mode = mode
        self.depth_bps = float(depth_bps)
        self.fetch_snapshot = fetch_snapshot
        self.snapshot_limit = snapshot_limit
        self.books: dict[str, LocalBook] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._gate = asyncio.Lock()
        self._last_snap = 0.0
        self.errors = 0

    def set_symbols(self, symbols: list[str]) -> None:
        want = {s.upper() for s in symbols}
        for s in list(self.books):
            if s not in want:
                self.books.pop(s, None)
                t = self._tasks.pop(s, None)
                if t is not None:
                    t.cancel()
        for s in want:
            self.books.setdefault(s, LocalBook(s, self.mode))

    def on_reset(self, symbols: list[str] | None = None) -> None:
        """A depth connection dropped: diff books are out of sequence from here on, partial books just age."""
        if self.mode != "diff":
            return
        for s in symbols if symbols is not None else list(self.books):
            b = self.books.get(s)
            if b is not None:
                b.reset()
                b.buffer.clear()

    async def on_message(self, ev: dict) -> None:
        sym = ev.get("s")
        book = self.books.get(sym) if sym else None
        if book is None:
            return
        recv = now_ms()
        if self.mode == "partial":
            book.on_partial(ev, recv)
            return
        if book.on_diff(ev, recv):
            self._request_snapshot(book)

    def _request_snapshot(self, book: LocalBook) -> None:
        if self.fetch_snapshot is None or book.snapshot_pending:
            return
        book.snapshot_pending = True
        t = self._tasks.get(book.symbol)
        if t is not None and not t.done():
            return
        self._tasks[book.symbol] = asyncio.create_task(self._snapshot(book), name=f"book-snapshot-{book.symbol}")

    async def _snapshot(self, book: LocalBook) -> None:
        delay = 0.5
        while self.books.get(book.symbol) is book:
            try:
                async with self._gate:  # spaced out: a reconnect asks for every symbol at once
                    loop = asyncio.get_running_loop()
                    wait = self._last_snap + SNAPSHOT_MIN_GAP_S - loop.time()
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last_snap = loop.time()
                snap = await self.fetch_snapshot(book.symbol, self.snapshot_limit)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                self.errors += 1
                log.warning("order book snapshot %s failed: %s", book.symbol, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if book.apply_snapshot(snap, now_ms()) or not book.buffer:
                return
            # the buffered stream does not connect to this snapshot yet: try a newer one shortly
            book.snapshot_pending = True
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)

    def features(self, symbol: str, now: int | None = None) -> BookFeatures:
        """The symbol's book as it is at this call. `now` (the bar close a decision is made on; the replay's ProxyBooks
        picks its bars by it) does not enter staleness_ms: every update was stamped with the local clock on arrival,
        so the age is that clock now minus the stamp. Measured from the bar close instead (the exchange clock, earlier
        than the read by the whole processing delay), a book updated after the close read as negative staleness and
        one that stopped updating seconds before the decision as fresh."""
        book = self.books.get(symbol)
        if book is None:
            return EMPTY_BOOK
        return book.features(now_ms(), self.depth_bps)

    def status(self) -> dict[str, Any]:
        synced = sum(1 for b in self.books.values() if b.synced)
        tot: dict[str, int] = {}
        for b in self.books.values():
            for k, v in b.stats.items():
                tot[k] = tot.get(k, 0) + v
        return {"mode": self.mode, "symbols": len(self.books), "synced": synced, "errors": self.errors, **tot}

    async def close(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        self._tasks.clear()


# --- the historical stand-in ---------------------------------------------------------------------------------------
PROXY_BARS = 15  # closed 1m bars the proxy looks at
PROXY_SPREAD_FRAC = 0.2  # spread = this fraction of the median 1m high-low range (at least one tick)
PROXY_DEPTH_FRAC = 0.1  # depth per side within k bps = this fraction of the median 1m quote volume ...
# ... scaled down by k / range when the minute's range is wider than k (a wide range means a thinner book near the mid)


def bar_book_proxy(high, low, close, quote_volume, close_time: int, now: int, k_bps: float,
                   tick_size: float = 0.0) -> BookFeatures:
    """Conservative order-book stand-in from the last PROXY_BARS closed 1m bars (sequences, oldest first, all closed at
    or before `now`): no imbalance information (0), a spread of PROXY_SPREAD_FRAC of the median range (never below a
    tick), and a depth per side of PROXY_DEPTH_FRAC of the median minute quote volume, scaled by min(1, k / range).
    Live books of liquid USD-M perpetuals are tighter and deeper than this, so a rule that demands a tight, deep book
    can only be stricter on history than live."""
    h, lo, c, qv = (list(x)[-PROXY_BARS:] for x in (high, low, close, quote_volume))
    pts = [(hh, ll, cc, q) for hh, ll, cc, q in zip(h, lo, c, qv) if cc == cc and cc > 0 and hh == hh and ll == ll]
    if not pts:
        return BookFeatures(k_bps=k_bps, source="proxy")
    rng = median((hh - ll) / cc * 1e4 for hh, ll, cc, _ in pts)
    vols = [q for _, _, _, q in pts if q == q]
    qv_med = median(vols) if vols else 0.0
    mid = pts[-1][2]
    tick_bps = tick_size / mid * 1e4 if tick_size > 0 else 0.0
    spread = max(tick_bps, PROXY_SPREAD_FRAC * rng)
    scale = min(1.0, k_bps / rng) if rng > 0 else 1.0
    depth = PROXY_DEPTH_FRAC * qv_med * scale
    return BookFeatures(spread_bps=spread, imbalance=0.0, depth_bid_k=depth, depth_ask_k=depth,
                        staleness_ms=float(max(now - close_time, 0)), mid=mid, k_bps=k_bps, levels=0,
                        source="proxy")


def proxy_from_frame(frame, now: int, k_bps: float, tick_size: float = 0.0) -> BookFeatures:
    """bar_book_proxy over a 1m FeatureFrame's bars closed at or before `now`."""
    if frame is None or not frame.n:
        return BookFeatures(k_bps=k_bps, source="proxy")
    from bisect import bisect_right

    i = bisect_right(frame.close_times(), now)
    if i <= 0:
        return BookFeatures(k_bps=k_bps, source="proxy")
    a = max(0, i - PROXY_BARS)
    f = frame.f
    return bar_book_proxy(f["high"][a:i].tolist(), f["low"][a:i].tolist(), f["close"][a:i].tolist(),
                          f["quote_volume"][a:i].tolist(), int(frame.close_time[i - 1]), now, k_bps, tick_size)

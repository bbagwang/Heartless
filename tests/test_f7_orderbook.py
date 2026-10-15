"""F7: local order books from Binance USD-M depth streams, and their 1m-bar proxy.

Recorded-style synthetic diff messages ({"e": "depthUpdate", "U", "u", "pu", "b", "a"}) drive the Binance procedure:
buffer, snapshot, drop u < lastUpdateId, the first event straddles the snapshot id, then a pu == previous u chain;
a lost, out-of-order or late message is handled (ignored when already applied, a resync otherwise), and a reconnect
discards the book. Features are checked by hand on small books; the proxy only looks at closed bars."""
import asyncio
import json
import math

import numpy as np
import pytest

from heartless.data.candles import CandleArrays
from heartless.data.features import FeatureFrame
from heartless.exchange.binance_ws import BookStream
from heartless.exchange.orderbook import (EMPTY_BOOK, PROXY_DEPTH_FRAC, PROXY_SPREAD_FRAC, BookFeatures, LocalBook,
                                          OrderBooks, bar_book_proxy, book_of, features_from_levels, proxy_from_frame)

SYM = "BTCUSDT"


def ev(U, u, pu, b=(), a=(), E=0):
    return {"e": "depthUpdate", "E": E, "T": E, "s": SYM, "U": U, "u": u, "pu": pu,
            "b": [[str(p), str(q)] for p, q in b], "a": [[str(p), str(q)] for p, q in a]}


def snap(L, b, a):
    return {"lastUpdateId": L, "E": 0, "T": 0, "bids": [[str(p), str(q)] for p, q in b],
            "asks": [[str(p), str(q)] for p, q in a]}


BIDS = [(100.0, 1.0), (99.9, 2.0), (99.0, 5.0)]
ASKS = [(100.1, 1.5), (100.2, 2.0), (101.0, 4.0)]


def test_snapshot_then_buffer_replay_then_live_chain():
    bk = LocalBook(SYM, "diff")
    assert bk.on_diff(ev(90, 95, 89, b=[(100.0, 9)]), 1) is True  # first event: ask for a snapshot
    bk.snapshot_pending = True
    assert bk.on_diff(ev(96, 101, 95, b=[(100.0, 3)]), 2) is False  # buffered, snapshot already asked
    assert bk.on_diff(ev(102, 104, 101, a=[(100.1, 0)]), 3) is False
    assert not bk.synced
    assert bk.apply_snapshot(snap(99, BIDS, ASKS), 10) is True  # u=95 dropped, 96..101 straddles 99, then 102..104
    assert bk.synced and bk.last_u == 104
    assert bk.bids[100.0] == 3.0 and 100.1 not in bk.asks  # absolute quantities, 0 removes
    assert bk.on_diff(ev(105, 107, 104, b=[(99.95, 1)], a=[(100.15, 2)]), 11) is False
    assert bk.bids[99.95] == 1.0 and bk.asks[100.15] == 2.0 and bk.last_u == 107
    assert bk.on_diff(ev(108, 108, 107, a=[(555.0, 0)]), 12) is False  # removing an unknown level is normal
    f = bk.features(20, 10.0)
    assert f.source == "stream" and f.staleness_ms == 8 and f.spread_bps > 0


def test_a_lost_message_triggers_a_resync_and_late_duplicates_are_ignored():
    bk = LocalBook(SYM, "diff")
    bk.on_diff(ev(1, 5, 0), 1)
    bk.snapshot_pending = True
    assert bk.apply_snapshot(snap(3, BIDS, ASKS), 2)
    assert bk.on_diff(ev(6, 8, 5), 3) is False
    assert bk.on_diff(ev(6, 8, 5), 4) is False and bk.stats["stale"] == 1  # the same message twice
    assert bk.on_diff(ev(2, 4, 1), 5) is False and bk.stats["stale"] == 2  # an old message arriving late
    assert bk.on_diff(ev(12, 14, 11), 6) is True  # pu 11 != 8: something was lost
    assert not bk.synced and bk.stats["resyncs"] == 1 and not bk.bids and len(bk.buffer) == 1
    bk.snapshot_pending = True
    assert bk.on_diff(ev(15, 16, 14), 7) is False  # buffered while the snapshot is on its way
    assert bk.apply_snapshot(snap(13, BIDS, ASKS), 8)
    assert bk.synced and bk.last_u == 16


def test_out_of_order_messages_inside_the_buffer_are_sorted_before_the_replay():
    bk = LocalBook(SYM, "diff")
    bk.on_diff(ev(11, 12, 10, b=[(100.0, 7)]), 1)
    bk.snapshot_pending = True
    bk.on_diff(ev(7, 10, 6, b=[(100.0, 4)]), 2)  # arrives after its successor
    assert bk.apply_snapshot(snap(8, BIDS, ASKS), 3)
    assert bk.bids[100.0] == 7.0 and bk.last_u == 12


def test_a_snapshot_older_than_the_stream_start_waits_for_a_newer_one():
    bk = LocalBook(SYM, "diff")
    bk.on_diff(ev(50, 55, 49), 1)
    bk.snapshot_pending = True
    assert bk.apply_snapshot(snap(40, BIDS, ASKS), 2) is False  # U=50 > 40: the events 41..49 are unknown
    assert not bk.synced and len(bk.buffer) == 1
    assert bk.apply_snapshot(snap(52, BIDS, ASKS), 3) is True and bk.last_u == 55


def test_a_gap_inside_the_buffer_needs_another_snapshot():
    bk = LocalBook(SYM, "diff")
    for e in (ev(10, 12, 9), ev(13, 15, 12), ev(20, 22, 19)):  # 16..19 lost
        bk.on_diff(e, 1)
    bk.snapshot_pending = True
    assert bk.apply_snapshot(snap(11, BIDS, ASKS), 2) is False and bk.buffer[0]["u"] == 22
    assert bk.apply_snapshot(snap(21, BIDS, ASKS), 3) is True and bk.last_u == 22


def test_no_buffered_event_reaches_the_snapshot_then_the_first_live_one_must_straddle_it():
    bk = LocalBook(SYM, "diff")
    bk.on_diff(ev(1, 2, 0), 1)
    bk.snapshot_pending = True
    assert bk.apply_snapshot(snap(30, BIDS, ASKS), 2) and bk.first_after_snapshot
    assert bk.on_diff(ev(20, 25, 19), 3) is False and bk.stats["stale"] == 1  # u < 30: older than the snapshot
    assert bk.on_diff(ev(28, 33, 27, b=[(100.0, 2)]), 4) is False  # straddles 30: applied, pu not checked
    assert bk.bids[100.0] == 2.0 and bk.last_u == 33 and not bk.first_after_snapshot
    bk2 = LocalBook(SYM, "diff")
    bk2.on_diff(ev(1, 2, 0), 1)
    bk2.snapshot_pending = True
    bk2.apply_snapshot(snap(30, BIDS, ASKS), 2)
    assert bk2.on_diff(ev(31, 33, 30), 3) is False and bk2.last_u == 33  # directly follows the snapshot (pu == L)
    bk3 = LocalBook(SYM, "diff")
    bk3.on_diff(ev(1, 2, 0), 1)
    bk3.snapshot_pending = True
    bk3.apply_snapshot(snap(30, BIDS, ASKS), 2)
    assert bk3.on_diff(ev(35, 36, 34), 3) is True  # starts after the snapshot id: resync


def test_partial_depth_replaces_the_book_and_ignores_older_messages():
    bk = LocalBook(SYM, "partial")
    bk.on_partial(ev(1, 10, 9, b=BIDS, a=ASKS), 5)
    assert bk.synced and len(bk.bids) == 3
    bk.on_partial(ev(1, 9, 8, b=[(1.0, 1.0)], a=[(2.0, 1.0)]), 6)
    assert len(bk.bids) == 3 and bk.stats["stale"] == 1
    bk.on_partial(ev(11, 12, 10, b=[(100.0, 1.0)], a=[(100.2, 1.0)]), 7)
    assert list(bk.bids) == [100.0] and bk.features(7, 10.0).levels == 1


def test_features_by_hand():
    bids = {100.0: 1.0, 99.95: 2.0, 99.0: 5.0}
    asks = {100.1: 1.5, 100.12: 2.0, 101.0: 4.0}
    f = features_from_levels(bids, asks, 10.0, 250.0)
    mid = (100.0 + 100.1) / 2
    assert f.mid == pytest.approx(mid) and f.spread_bps == pytest.approx(0.1 / mid * 1e4)
    lo, hi = mid * (1 - 1e-3), mid * (1 + 1e-3)
    db = sum(p * q for p, q in bids.items() if p >= lo)
    da = sum(p * q for p, q in asks.items() if p <= hi)
    assert f.depth_bid_k == pytest.approx(db) and f.depth_ask_k == pytest.approx(da)
    assert f.depth_bid_k == pytest.approx(100.0 + 99.95 * 2) and f.depth_ask_k == pytest.approx(100.1 * 1.5 + 100.12 * 2)
    assert f.imbalance == pytest.approx((db - da) / (db + da)) and f.staleness_ms == 250.0 and f.levels == 3
    crossed = features_from_levels({100.2: 1.0}, {100.1: 1.0}, 10.0, 0.0)
    assert math.isnan(crossed.spread_bps) and crossed.source == "stream"
    assert math.isnan(EMPTY_BOOK.spread_bps) and EMPTY_BOOK.source == "none"
    assert set(EMPTY_BOOK.as_dict()) == {"spread_bps", "imbalance", "depth_bid_k", "depth_ask_k", "staleness_ms", "mid",
                                         "k_bps", "levels", "source"}


def test_the_manager_fetches_snapshots_spaced_and_resyncs_after_gaps_and_reconnects():
    calls = []
    snaps = {1: snap(99, BIDS, ASKS), 2: snap(205, BIDS, ASKS)}

    async def fetch(sym, limit):
        calls.append((sym, limit))
        await asyncio.sleep(0)
        return snaps[len(calls)]

    async def main():
        books = OrderBooks("diff", 10.0, fetch_snapshot=fetch)
        books.set_symbols([SYM, "ETHUSDT"])
        await books.on_message(ev(96, 101, 95))
        await books.on_message(ev(102, 104, 101))
        await asyncio.sleep(0.01)
        b = books.books[SYM]
        assert b.synced and b.last_u == 104 and calls == [(SYM, 100)]
        f = books.features(SYM)
        assert f.source == "stream" and f.spread_bps > 0
        await books.on_message(ev(200, 206, 199))  # a gap: resync with a second snapshot
        await asyncio.sleep(0.6)
        assert len(calls) == 2 and b.synced and b.last_u == 206
        books.on_reset([SYM])  # the depth connection dropped
        assert not b.synced and books.features(SYM).source == "stream" and math.isnan(books.features(SYM).spread_bps)
        assert books.features("XRPUSDT") is EMPTY_BOOK
        st = books.status()
        assert st["symbols"] == 2 and st["snapshots"] == 2 and st["resyncs"] == 1
        await books.close()

    asyncio.run(main())


def test_partial_manager_needs_no_snapshot():
    async def main():
        books = OrderBooks("partial", 5.0)
        books.set_symbols([SYM])
        await books.on_message(ev(1, 2, 1, b=BIDS, a=ASKS))
        f = books.features(SYM)
        assert f.source == "stream" and f.k_bps == 5.0 and f.depth_bid_k == pytest.approx(100.0)  # 99.9 is 15 bps out
        books.on_reset([SYM])  # partial books only age
        assert books.books[SYM].synced

    asyncio.run(main())
    with pytest.raises(ValueError):
        OrderBooks("full")


def test_book_stream_names_chunks_and_dispatch():
    got = []
    st = BookStream("wss://x", got.append, mode="partial", speed_ms=100, per_connection=2)
    st.set_symbols(["ethusdt", "BTCUSDT", "SOLUSDT"])
    urls = st.urls()
    assert [c for c, _ in urls] == [["BTCUSDT", "ETHUSDT"], ["SOLUSDT"]]
    assert urls[0][1] == "wss://x/stream?streams=btcusdt@depth20@100ms/ethusdt@depth20@100ms"
    assert BookStream("w", got.append, mode="diff", speed_ms=250).stream_name("BTCUSDT") == "btcusdt@depth"
    assert BookStream("w", got.append, mode="diff", speed_ms=480).stream_name("BTCUSDT") == "btcusdt@depth@500ms"
    msg = {"stream": "btcusdt@depth20@100ms", "data": ev(1, 2, 1, b=BIDS, a=ASKS, E=5)}
    asyncio.run(st._dispatch(json.dumps(msg)))
    asyncio.run(st._dispatch(json.dumps({"data": {"e": "kline"}})))
    asyncio.run(st._dispatch("not json"))
    assert len(got) == 1 and got[0]["u"] == 2 and st.messages == 1


# --- the 1m-bar proxy ------------------------------------------------------------------------------------------------
def test_the_proxy_is_conservative_and_causal():
    h = [101.0, 100.6, 100.4, 100.9]
    lo = [99.0, 99.8, 99.9, 100.1]
    c = [100.0, 100.2, 100.1, 100.5]
    qv = [5e5, 7e5, 6e5, 9e5]
    f = bar_book_proxy(h, lo, c, qv, close_time=1_000, now=1_500, k_bps=10.0, tick_size=0.1)
    ranges = sorted((a - b) / x * 1e4 for a, b, x in zip(h, lo, c))
    med = (ranges[1] + ranges[2]) / 2
    assert f.source == "proxy" and f.imbalance == 0.0 and f.staleness_ms == 500
    assert f.spread_bps == pytest.approx(max(0.1 / 100.5 * 1e4, PROXY_SPREAD_FRAC * med))
    assert f.depth_bid_k == f.depth_ask_k == pytest.approx(PROXY_DEPTH_FRAC * 6.5e5 * min(1.0, 10.0 / med))
    wide = bar_book_proxy(h, lo, c, qv, 1_000, 1_000, k_bps=1000.0)
    assert wide.depth_bid_k == pytest.approx(PROXY_DEPTH_FRAC * 6.5e5)  # never more than the volume fraction
    tick = bar_book_proxy([100.0] * 3, [100.0] * 3, [100.0] * 3, [1.0] * 3, 0, 0, 10.0, tick_size=0.5)
    assert tick.spread_bps == pytest.approx(50.0)  # a flat minute still pays a tick
    assert bar_book_proxy([], [], [], [], 0, 0, 10.0).source == "proxy"


def test_proxy_from_a_frame_uses_only_closed_bars():
    n = 40
    rng = np.random.default_rng(3)
    ca = CandleArrays("1m", n + 16)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    ca.open[:n], ca.close[:n] = c, c
    ca.high[:n], ca.low[:n] = c * 1.001, c * 0.999
    ca.volume[:n] = 10.0
    ca.quote_volume[:n] = rng.uniform(1e5, 1e6, n)
    ca.open_time[:n] = np.arange(n) * 60_000
    ca.close_time[:n] = ca.open_time[:n] + 59_999
    ca.n = n
    fr = FeatureFrame("1m", ca)
    t = int(ca.close_time[25])
    a = proxy_from_frame(fr, t, 10.0)
    b = proxy_from_frame(FeatureFrame("1m", ca.slice(0, 26)), t, 10.0)
    assert a == b and a.staleness_ms == 0
    assert proxy_from_frame(fr, t + 30_000, 10.0).staleness_ms == 30_000
    assert proxy_from_frame(fr, -5, 10.0).source == "proxy" and proxy_from_frame(None, t, 10.0).source == "proxy"


def test_book_of_reads_ctx_book_or_the_empty_book():
    class Ctx:
        pass

    ctx = Ctx()
    assert book_of(ctx) is EMPTY_BOOK
    ctx.book = BookFeatures(spread_bps=1.0, source="stream")
    assert book_of(ctx).spread_bps == 1.0

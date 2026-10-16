"""Red-team fixes of the local order books and the live universe (2026-10-16):
F-M5 depth / imbalance are NaN when the levels do not reach k bps (covered_bps); F-M6 a dropped connection unsyncs
partial books too, a book without updates reads "stale", usable_book() is the guard's check, a failing book read is
logged at WARNING (rate limited); F-L6 the unsynced buffer is bounded; F-L7 stop() closes the books and the depth
stream before the REST client; F-L10 snapshot requests are spaced across symbols; F-M7 universe hysteresis never
re-adds a member another live rule rejects."""
import asyncio
import logging
import math
import tempfile
from pathlib import Path

import pytest

from heartless.config import Settings
from heartless.exchange import orderbook as OB
from heartless.exchange.orderbook import (EMPTY_BOOK, BookFeatures, LocalBook, OrderBooks, bar_book_proxy,
                                          features_from_levels, usable_book)

SYM = "BTCUSDT"


def ev(U, u, pu, b=(), a=(), s=SYM):
    return {"e": "depthUpdate", "E": 0, "T": 0, "s": s, "U": U, "u": u, "pu": pu,
            "b": [[str(p), str(q)] for p, q in b], "a": [[str(p), str(q)] for p, q in a]}


def test_depth_is_unknown_when_the_levels_stop_short_of_k():
    # twenty 0.1-wide levels around 60,000: the book spans 0.3 bp a side, far short of 10 bp
    bids = {60_000.0 - 0.1 * i: 1.0 for i in range(20)}
    asks = {60_000.1 + 0.1 * i: 1.0 for i in range(20)}
    f = features_from_levels(bids, asks, 10.0, 5.0)
    assert f.covered_bps < 1.0 and math.isnan(f.depth_bid_k) and math.isnan(f.depth_ask_k) and math.isnan(f.imbalance)
    assert f.spread_bps > 0 and f.levels == 20 and not usable_book(f)  # the spread is still measured
    # the red-team case: equal notional within 10 bp on both sides, the bid side seen only 0.19 bp deep
    deep = {100.0 - 0.01 * i: 1.0 for i in range(150)}
    asks2 = {100.01 + 0.01 * i: 1.0 for i in range(150)}
    trunc = {p: q for p, q in deep.items() if p >= 99.998}
    full = features_from_levels(deep, asks2, 10.0, 0.0)
    cut = features_from_levels(trunc, asks2, 10.0, 0.0)
    assert full.imbalance == pytest.approx(0.0, abs=0.02) and usable_book(full)
    assert math.isnan(cut.imbalance) and cut.covered_bps < 10.0  # was -0.82 before the fix


def test_levels_beyond_k_cover_the_band_and_only_the_band_is_summed():
    f = features_from_levels({100.0: 1.0, 99.90: 1.0}, {100.02: 1.0, 100.12: 1.0}, 10.0, 0.0)  # mid 100.01
    assert f.covered_bps >= 10.0 and f.depth_bid_k == pytest.approx(100.0) and f.depth_ask_k == pytest.approx(100.02)


def test_partial_books_unsync_on_reset_and_stale_books_read_nan(monkeypatch):
    t = [1_000_000]
    monkeypatch.setattr(OB, "now_ms", lambda: t[0])

    async def main():
        books = OrderBooks("partial", 10.0, max_age_ms=3_000)
        books.set_symbols([SYM])
        lv_b = [(100.0, 1.0), (99.5, 2.0)]
        lv_a = [(100.1, 1.0), (100.6, 2.0)]
        await books.on_message(ev(1, 2, 1, b=lv_b, a=lv_a))
        assert usable_book(books.features(SYM))
        t[0] += 2_000
        assert books.features(SYM).source == "stream" and usable_book(books.features(SYM))
        t[0] += 1_500  # 3.5 s without an update
        f = books.features(SYM)
        assert f.source == "stale" and math.isnan(f.spread_bps) and f.staleness_ms == 3_500 and not usable_book(f)
        await books.on_message(ev(3, 4, 3, b=lv_b, a=lv_a))
        assert usable_book(books.features(SYM))
        books.on_reset(None)  # the connection dropped: no figures until the stream delivers again
        f = books.features(SYM)
        assert not books.books[SYM].synced and math.isnan(f.spread_bps) and not usable_book(f)
        t[0] += 600_000
        assert math.isnan(books.features(SYM).spread_bps)  # never the old spread, however long it waits

    asyncio.run(main())


def test_usable_book_rules():
    assert not usable_book(EMPTY_BOOK) and not usable_book(BookFeatures(source="stream"))
    proxy = bar_book_proxy([101.0] * 15, [99.0] * 15, [100.0] * 15, [1e6] * 15, 0, 60_000, 10.0)
    assert proxy.source == "proxy" and proxy.covered_bps == 10.0 and usable_book(proxy)
    ok = BookFeatures(spread_bps=1.0, imbalance=0.1, depth_bid_k=1e5, depth_ask_k=1e5, staleness_ms=100.0, mid=100.0,
                      k_bps=10.0, levels=50, source="stream", covered_bps=12.0)
    assert usable_book(ok) and not usable_book(BookFeatures(**{**ok.as_dict(), "staleness_ms": 3_001.0}))


def _snap(L, bids, asks):
    return {"lastUpdateId": L, "E": 0, "T": 0, "bids": [[str(p), str(q)] for p, q in bids],
            "asks": [[str(p), str(q)] for p, q in asks]}


def test_a_diff_book_is_covered_only_as_far_as_its_snapshot_reached():
    """Adversarial review 2026-10-16: a diff book is known level by level only inside the band its snapshot returned.
    One later order 50 bp away on each side made a 0.5 bp snapshot read as 50 bp covered, its depth within 10 bp (the
    0.5 bp held) as measured and the book as usable."""
    bk = LocalBook(SYM, "diff")
    bids = [(60_000.0 - 0.1 * i, 5.0) for i in range(30)]  # 30 levels of 0.1: 0.5 bp a side, cut at the limit
    asks = [(60_000.1 + 0.1 * i, 5.0) for i in range(30)]
    bk.on_diff(ev(9, 11, 8, b=[(60_000.0, 5.0)]), 1)
    assert bk.apply_snapshot(_snap(10, bids, asks), 2, limit=30)
    assert bk.features(3, 10.0).covered_bps < 1.0 and not usable_book(bk.features(3, 10.0))
    bk.on_diff(ev(12, 12, 11, b=[(59_700.0, 1.0)], a=[(60_300.0, 1.0)]), 4)  # one far order each side
    f = bk.features(5, 10.0)
    assert f.covered_bps < 1.0 and math.isnan(f.depth_bid_k) and math.isnan(f.imbalance) and not usable_book(f)
    assert f.spread_bps > 0  # the touch is still known
    # a snapshot reaching past k: covered, and a far level beyond the band does not move it
    wide_b = [(round(100.0 - 0.01 * i, 2), 1.0) for i in range(15)]  # 100.00 .. 99.86: 14.5 bp below the mid
    wide_a = [(round(100.01 + 0.01 * i, 2), 1.0) for i in range(15)]  # 100.01 .. 100.15
    bk2 = LocalBook(SYM, "diff")
    bk2.on_diff(ev(9, 11, 8), 1)
    assert bk2.apply_snapshot(_snap(10, wide_b, wide_a), 2, limit=15)
    g = bk2.features(3, 10.0)
    assert 14.0 < g.covered_bps < 15.0 and usable_book(g)
    # the price moves up 10 bp (asks taken, bids raised): the ask side now reaches only ~4.5 bp of known book
    bk2.on_diff(ev(12, 12, 11, b=[(100.10, 1.0)], a=[(round(100.01 + 0.01 * i, 2), 0.0) for i in range(10)]), 4)
    bk2.on_diff(ev(13, 13, 12, a=[(101.0, 1.0)]), 5)  # a far ask beyond the band
    h = bk2.features(6, 10.0)
    assert h.covered_bps < 10.0 and math.isnan(h.depth_ask_k) and not usable_book(h)
    # a side the snapshot returned whole (fewer levels than asked for) is known all the way
    bk3 = LocalBook(SYM, "diff")
    bk3.on_diff(ev(9, 11, 8), 1)
    assert bk3.apply_snapshot(_snap(10, [(100.0, 1.0), (99.95, 2.0)], [(100.01, 1.0), (100.05, 2.0)]), 2, limit=1000)
    w = bk3.features(3, 10.0)
    assert w.covered_bps >= 10.0 and w.depth_bid_k == pytest.approx(100.0 + 99.95 * 2) and usable_book(w)
    # without the requested limit a side counts as cut at its last level (the previous reading)
    bk4 = LocalBook(SYM, "diff")
    bk4.on_diff(ev(9, 11, 8), 1)
    assert bk4.apply_snapshot(_snap(10, [(100.0, 1.0), (99.95, 2.0)], [(100.01, 1.0), (100.05, 2.0)]), 2)
    assert bk4.features(3, 10.0).covered_bps < 10.0
    bk4.reset()
    assert bk4.known_bid is None and bk4.known_ask is None


def test_the_books_pass_the_snapshot_limit(monkeypatch):
    t = [1_000_000]
    monkeypatch.setattr(OB, "now_ms", lambda: t[0])
    asked = []

    async def fetch(sym, limit):
        asked.append(limit)
        return _snap(6, [(100.0, 1.0), (99.95, 2.0)], [(100.01, 1.0), (100.05, 2.0)])  # a whole (short) book

    async def main():
        books = OrderBooks("diff", 10.0, fetch_snapshot=fetch, snapshot_limit=1000)
        books.set_symbols([SYM])
        await books.on_message(ev(5, 9, 4))
        for _ in range(50):
            await asyncio.sleep(0.01)
            if books.books[SYM].synced:
                break
        f = books.features(SYM)
        await books.close()
        return f

    f = asyncio.run(main())
    assert asked == [1000] and f.covered_bps >= 10.0 and usable_book(f)


def test_the_unsynced_buffer_is_bounded():
    bk = LocalBook(SYM, "diff")
    for i in range(3 * OB.BUFFER_MAX):
        bk.on_diff(ev(10 * i + 1, 10 * i + 5, 10 * i, b=[(100.0, 1.0)]), i)
    assert len(bk.buffer) == OB.BUFFER_MAX == 600 and not bk.synced
    assert bk.buffer[-1]["u"] == 10 * (3 * OB.BUFFER_MAX - 1) + 5  # the newest are kept


def test_snapshot_requests_are_spaced_across_symbols(monkeypatch):
    monkeypatch.setattr(OB, "SNAPSHOT_MIN_GAP_S", 0.15)
    times: list[tuple[str, float]] = []

    async def fetch(sym, limit):
        times.append((sym, asyncio.get_running_loop().time()))
        return {"lastUpdateId": 6, "E": 0, "T": 0, "bids": [["100", "1"]], "asks": [["100.1", "1"]]}

    async def main():
        books = OrderBooks("diff", 10.0, fetch_snapshot=fetch)
        syms = [SYM, "ETHUSDT", "SOLUSDT"]
        books.set_symbols(syms)
        for s in syms:  # a reconnect: every book asks for a snapshot at once
            await books.on_message(ev(5, 9, 4, s=s))
        await asyncio.sleep(0.6)
        await books.close()

    asyncio.run(main())
    assert sorted(s for s, _ in times) == sorted([SYM, "ETHUSDT", "SOLUSDT"])
    gaps = [b - a for (_, a), (_, b) in zip(times, times[1:])]
    assert len(gaps) == 2 and all(g >= 0.15 - 0.01 for g in gaps), gaps


def _app(**kw):
    from heartless.app import Heartless

    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ALWAYS_INCLUDE="BTCUSDT", ARCHIVE_BACKFILL=False, **kw)
    s.data_dir.mkdir(exist_ok=True)
    return Heartless(s)


def test_stop_closes_the_books_and_the_depth_stream_first():
    app = _app(ORDERBOOK_STREAM=True, ORDERBOOK_MODE="diff")
    order = []

    async def books_close():
        order.append("books")

    async def stream_stop():
        order.append("stream")

    real_close = app.rest.close

    async def rest_close():
        order.append("rest")
        await real_close()

    app.books.close = books_close
    app.book_stream.stop = stream_stop
    app.rest.close = rest_close
    asyncio.run(app.stop())
    assert order == ["stream", "books", "rest"]


def test_a_failing_book_read_warns_once_a_minute(caplog):
    app = _app(ORDERBOOK_STREAM=True)

    def boom(sym, now):
        raise RuntimeError("broken book")

    app.books.features = boom
    with caplog.at_level(logging.WARNING, logger="heartless.app"):
        assert app._f7_book(SYM, 1) is EMPTY_BOOK
        assert app._f7_book(SYM, 2) is EMPTY_BOOK
    warns = [r for r in caplog.records if "order book features failed" in r.getMessage()]
    assert len(warns) == 1 and warns[0].levelno == logging.WARNING
    app.store.close()
    asyncio.run(app.rest.close())


class _Rest:
    def __init__(self, rows: dict[str, tuple[float, float]]):
        self.rows = rows  # symbol -> (quote volume, last price)

    async def ticker_24h(self):
        return [{"symbol": s, "quoteVolume": str(v), "lastPrice": str(p)} for s, (v, p) in self.rows.items()]

    async def close(self):
        pass


def test_hysteresis_does_not_re_add_a_member_another_rule_rejects():
    from heartless.core.models import SymbolInfo

    app = _app(UNIVERSE_SIZE=4, UNIVERSE_MIN_QUOTE_VOLUME=80e6)
    syms = ["BTCUSDT", "ETHUSDT", "FINEUSDT", "COARSEUSDT"]
    app.symbols = {s: SymbolInfo(s, s[:-4], "USDT", 0.2 if s == "COARSEUSDT" else 0.01, 0.001, 0.001, 5.0, 2, 3)
                   for s in syms}

    async def nothing(*a, **k):
        return set()

    app._backfill = nothing
    app._build_views = lambda syms: None
    app.rest = _Rest({"BTCUSDT": (9e9, 60_000.0), "ETHUSDT": (5e9, 3_000.0), "FINEUSDT": (1e9, 10.0),
                      "COARSEUSDT": (2e8, 1_000.0)})  # tick 0.2 at 1,000 = 2 bp: eligible
    asyncio.run(app.refresh_universe(initial=True))
    assert "COARSEUSDT" in app.universe
    app.rest.rows["COARSEUSDT"] = (79e6, 1_000.0)  # just under the volume floor, nothing else wrong: kept
    asyncio.run(app.refresh_universe())
    assert "COARSEUSDT" in app.universe
    app.rest.rows["COARSEUSDT"] = (79e6, 100.0)  # the price fell: tick 0.2 = 20 bp > 10 bp
    asyncio.run(app.refresh_universe())
    assert "COARSEUSDT" not in app.universe  # the red-team repro re-added it
    app.store.close()
    asyncio.run(app.rest.close())

"""F-pre-B book guard (ROADMAP 7.11): the vectorised 1m-bar proxy equals exchange/orderbook.bar_book_proxy bar by bar,
the guard's spread / depth rules against the symbol's own prior 24 hours (stream samples and proxy alike), and the
engine holding an entry in a backtest only while the guard is on."""
import numpy as np

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.data.candles import CandleArrays
from heartless.data.features import FeatureFrame
from heartless.exchange import orderbook as OB
from heartless.exchange.orderbook import BookFeatures, bar_book_proxy, proxy_from_frame
from heartless.execution.engine import BOOK_MAX_AGE_MS, BookGuard, book_proxy_series, usable_book, usable_stream_book
from heartless.strategy.base import Context
from synth import synth_candles
from tests.test_fpreB_support import Scripted, backtest

MIN = 60_000


def test_vectorised_proxy_equals_bar_book_proxy():
    ca = synth_candles(1600, seed=3, price=123.4567)
    fr = FeatureFrame("1m", ca)
    ct = fr.close_time
    now = int(ct[-1])
    spread, depth = book_proxy_series(fr, now, 10.0, 0.0001, 60)
    assert len(spread) == len(depth) == 60
    for j in range(60):
        i = len(ct) - 60 + j  # the bar the proxy is computed at
        a = max(0, i + 1 - 15)
        ref = bar_book_proxy(fr.f["high"][a:i + 1], fr.f["low"][a:i + 1], fr.f["close"][a:i + 1],
                             fr.f["quote_volume"][a:i + 1], int(ct[i]), int(ct[i]), 10.0, 0.0001)
        assert spread[j] == ref.spread_bps and depth[j] == ref.depth_bid_k + ref.depth_ask_k
    # bars after `now` are never read
    s2, _ = book_proxy_series(fr, int(ct[-11]), 10.0, 0.0001, 50)
    assert np.array_equal(s2, spread[:-10][-50:])


class _V:
    def __init__(self, frame):
        self.frames = {"1m": frame}


def _ctx(book=None):
    from heartless.core.models import Regime, Ticker
    from tests.test_fpreB_support import synth_symbols

    return Context("XUSDT", synth_symbols(["XUSDT"])["XUSDT"], Ticker("XUSDT"), Regime.RANGE,
                   book=book or BookFeatures())


def _stream(spread, depth_side, age=50.0, covered=12.0):
    return BookFeatures(spread_bps=spread, imbalance=0.0, depth_bid_k=depth_side, depth_ask_k=depth_side,
                        staleness_ms=age, k_bps=10.0, source="stream", covered_bps=covered)


def test_guard_rules_on_stream_samples():
    g = BookGuard(Settings(_env_file=None))
    t0 = 1_700_006_400_000
    for k in range(1440):
        g.observe(_ctx(_stream(1.0 + 0.001 * (k % 3), 5e5)), t0 + k * MIN)
    now = t0 + 1440 * MIN
    ok = _stream(4.0, 4e5)
    assert g.check(None, _ctx(ok), now, 0.01) == ""  # 4 bp: over 3x the median but under the 5 bp floor
    wide = _stream(6.0, 4e5)
    assert g.check(None, _ctx(wide), now, 0.01) == "book guard: spread"
    thin = _stream(1.0, 1.2e5)
    assert g.check(None, _ctx(thin), now, 0.01) == "book guard: depth"  # 2.4e5 < 25% of 1e6
    # a fresh stream (under 720 prior minutes) judges nothing
    g2 = BookGuard(Settings(_env_file=None))
    for k in range(700):
        g2.observe(_ctx(_stream(1.0, 5e5)), t0 + k * MIN)
    assert g2.check(None, _ctx(wide), t0 + 700 * MIN, 0.01) == ""
    # one sample per minute
    assert len(g.samples["XUSDT"]) == 1440


def _calm_then_wild(n=2000, wild_from=1985, k_range=0.0003, wild=0.009):
    """Calm 1m bars (range ~3 bp) then the last bars with a ~90 bp range: the proxy spread jumps ~30x."""
    ca = CandleArrays("1m", capacity=n + 16)
    t0, p = 1_700_006_400_000, 100.0
    for k in range(n):
        r = wild if k >= wild_from else k_range
        o = p
        p = p * (1 + (0.0001 if k % 2 else -0.0001))
        ca.append(Candle(t0 + k * MIN, o, max(o, p) * (1 + r / 2), min(o, p) * (1 - r / 2), p, 1000.0, 1000.0 * p, 10,
                         500.0, t0 + k * MIN + MIN - 1))
    return ca


def test_guard_on_the_proxy_and_in_a_backtest(monkeypatch, tmp_path):
    ca = _calm_then_wild()
    fr = FeatureFrame("1m", ca)
    g = BookGuard(Settings(_env_file=None))
    ct = fr.close_time
    assert g.check(_V(fr), _ctx(), int(ct[1980]), 0.01) == ""
    assert g.check(_V(fr), _ctx(), int(ct[-1]), 0.01) == "book guard: spread"
    # the backtest: a decision in the calm trades, one in the wild bars is held (and only with the guard on)
    times = [int(ct[1900]), int(ct[1995])]
    on, _ = backtest(monkeypatch, tmp_path, Scripted(times, hold=2), {"XUSDT": ca}, shock=None, BOOK_GUARD=True)
    assert [int(t["entry_time"]) - 1 for t in on.trades] == [times[0]] and on.skipped.get("book guard") == 1
    off, _ = backtest(monkeypatch, tmp_path, Scripted(times, hold=2), {"XUSDT": ca}, shock=None, BOOK_GUARD=False)
    assert sorted(int(t["entry_time"]) - 1 for t in off.trades) == times and not off.skipped.get("book guard")


def test_live_incremental_frames_give_the_same_proxy():
    """The live bot's 1m frame is an IncrementalFrame (F7): the guard reads the same proxy series from it as from the
    batch frame over the same bars."""
    from heartless.data.incremental import IncrementalFrame

    ca = synth_candles(1700, seed=8, price=77.7)
    head = CandleArrays("1m", capacity=ca.n + 16)
    for i in range(1500):
        head.append(ca.candle_at(i))
    inc = IncrementalFrame("1m", head, extra=ca.n)
    for i in range(1500, ca.n):
        c = ca.candle_at(i)
        inc.append(c.open_time, c.open, c.high, c.low, c.close, c.volume, c.quote_volume, c.taker_buy_volume,
                   c.close_time)
    batch = FeatureFrame("1m", ca)
    now = int(batch.close_time[-1])
    a, b = book_proxy_series(inc, now, 10.0, 0.001, 1441), book_proxy_series(batch, now, 10.0, 0.001, 1441)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    g = BookGuard(Settings(_env_file=None))
    assert g.figures(_V(inc), _ctx(), now, 0.001)[:2] == g.figures(_V(batch), _ctx(), now, 0.001)[:2]


def test_an_unusable_stream_book_falls_back_to_the_proxy():
    """Red-team F-M5 / F-M6 (orderbook.usable_book): a stale stream book, or one whose levels stop short of k (NaN
    depth, e.g. BTC's 20 partial levels), is no book: the guard judges the 1m-bar proxy instead (current and prior),
    and such books never enter the stream samples."""
    ca = _calm_then_wild()
    fr = FeatureFrame("1m", ca)
    now = int(fr.close_time[-1])
    stale = _stream(1.0, 5e5, age=BOOK_MAX_AGE_MS + 1)
    uncovered = BookFeatures(spread_bps=1.0, staleness_ms=50.0, source="stream")  # depth / imbalance NaN
    assert usable_stream_book(_stream(1.0, 5e5)) and not usable_stream_book(stale)
    assert not usable_stream_book(uncovered) and not usable_stream_book(BookFeatures(source="proxy"))
    g = BookGuard(Settings(_env_file=None))
    for b in (stale, uncovered, BookFeatures()):
        assert g.check(_V(fr), _ctx(b), now, 0.01) == "book guard: spread"  # the proxy's wild last bars
        g.observe(_ctx(b), now)
    assert g.samples == {}
    # a usable stream book is judged on its own samples (none yet: no verdict)
    assert g.check(_V(fr), _ctx(_stream(1.0, 5e5)), now, 0.01) == ""


def _unusable_books():
    return [BookFeatures(),  # no book (EMPTY_BOOK, or the app's fallback when the book raises)
            BookFeatures(k_bps=10.0, source="stream"),  # unsynced (every figure NaN)
            _stream(1.0, 5e5, age=BOOK_MAX_AGE_MS + 1),  # stale
            BookFeatures(spread_bps=1.0, staleness_ms=50.0, source="stream"),  # levels short of +-k (NaN depth)
            BookFeatures(k_bps=10.0, source="stale"),  # the red-team orderbook's stale marker (wave-f-merge)
            BookFeatures(k_bps=10.0, source="proxy"),  # a proxy without bars
            _stream(1.0, 5e5, covered=9.5),  # depth figures but levels reaching only 9.5 bp (coverage flag below k)
            _stream(1.0, 5e5, covered=float("nan"))]  # depth figures without a known coverage


def test_missing_book_clause_with_the_stream_on():
    """ROADMAP 7.11 missing-book clause (6절 2026-10-16, red-team F-M6): with ORDERBOOK_STREAM on, an entry without a
    usable book - none, unsynced, stale, NaN figures, levels short of +-k - is held, not judged on the 1m-bar proxy (the
    stand-in cannot see the live book that is missing); a defined proxy, which the backtester and the replay serve under
    ORDERBOOK_STREAM, is judged as before. With the stream off nothing changes."""
    fr = FeatureFrame("1m", _calm_then_wild())
    calm = int(fr.close_time[1980])
    g = BookGuard(Settings(_env_file=None, ORDERBOOK_STREAM=True, BOOK_GUARD=True))
    for b in _unusable_books():
        assert not usable_book(b)
        assert g.check(_V(fr), _ctx(b), calm, 0.01) == "book guard: no usable book", b
    assert g.held == len(_unusable_books())
    # usable: a fresh synced stream book (judged on its own samples: none yet, no verdict) and a defined proxy
    assert usable_book(_stream(1.0, 5e5)) and g.check(_V(fr), _ctx(_stream(1.0, 5e5)), calm, 0.01) == ""
    proxy = proxy_from_frame(fr, calm, 10.0, 0.01)
    assert usable_book(proxy) and g.check(_V(fr), _ctx(proxy), calm, 0.01) == ""
    wild = proxy_from_frame(fr, int(fr.close_time[-1]), 10.0, 0.01)
    assert g.check(_V(fr), _ctx(wild), int(fr.close_time[-1]), 0.01) == "book guard: spread"
    # the stream off (the default): no book is the normal case, the proxy judges (calm bars: not held)
    off = BookGuard(Settings(_env_file=None, BOOK_GUARD=True))
    assert all(off.check(_V(fr), _ctx(b), calm, 0.01) == "" for b in _unusable_books()) and off.held == 0


def test_backtest_under_the_stream_judges_the_proxy_book(monkeypatch, tmp_path):
    """The backtester with ORDERBOOK_STREAM on serves the 1m-bar proxy as ctx.book: the missing-book clause holds
    nothing there (the proxy is defined once 15 bars exist), the trades equal those with the stream off."""
    ca = _calm_then_wild()
    ct = FeatureFrame("1m", ca).close_time
    times = [int(ct[1900]), int(ct[1995])]
    runs = {}
    for on in (False, True):
        res, _ = backtest(monkeypatch, tmp_path, Scripted(times, hold=2), {"XUSDT": ca}, shock=None, BOOK_GUARD=True,
                          ORDERBOOK_STREAM=on)
        runs[on] = ([int(t["entry_time"]) for t in res.trades], res.skipped.get("book guard"))
    assert runs[True] == runs[False] and runs[True][0] == [times[0] + 1] and runs[True][1] == 1


def test_usable_book_matches_the_orderbook_rule():
    """engine.usable_book is orderbook.usable_book (red-team F-M5 / F-M6, wired at the F-pre-B integration) on every
    kind of book, including the coverage flag (covered_bps >= k_bps); no ctx.book at all is no book."""
    books = _unusable_books() + [_stream(1.0, 5e5), _stream(1.0, 5e5, age=BOOK_MAX_AGE_MS),
                                 _stream(1.0, 5e5, covered=10.0),
                                 proxy_from_frame(FeatureFrame("1m", _calm_then_wild()), 1_700_006_400_000 + 1980 * MIN,
                                                  10.0, 0.01)]
    for b in books:
        assert usable_book(b) == OB.usable_book(b), b
    assert BOOK_MAX_AGE_MS == OB.MAX_BOOK_AGE_MS and not usable_book(None) and not usable_stream_book(None)
    assert usable_book(_stream(1.0, 5e5, covered=10.0)) and not usable_book(_stream(1.0, 5e5, covered=9.99))


def test_a_book_from_the_stream_reaches_the_guard_with_its_coverage():
    """A real stream book (features_from_levels): levels reaching +-12 bp are usable and judged on their samples; the
    same book cut at +-6 bp (covered_bps < k) is held under ORDERBOOK_STREAM, and is never sampled."""
    mid = 100.0
    bids = {round(mid - 0.01 * k, 2): 5_000.0 for k in range(1, 13)}  # 1..12 bp below the mid
    asks = {round(mid + 0.01 * k, 2): 5_000.0 for k in range(1, 13)}
    full = OB.features_from_levels(bids, asks, 10.0, 20.0)
    cut = OB.features_from_levels({p: q for p, q in bids.items() if p >= 99.94},
                                  {p: q for p, q in asks.items() if p <= 100.06}, 10.0, 20.0)
    assert full.covered_bps >= 10.0 and usable_stream_book(full)
    assert cut.covered_bps < 10.0 and not usable_book(cut) and not usable_stream_book(cut)
    fr = FeatureFrame("1m", _calm_then_wild())
    calm = int(fr.close_time[1980])
    g = BookGuard(Settings(_env_file=None, ORDERBOOK_STREAM=True, BOOK_GUARD=True))
    assert g.check(_V(fr), _ctx(full), calm, 0.01) == ""
    assert g.check(_V(fr), _ctx(cut), calm, 0.01) == "book guard: no usable book"
    g.observe(_ctx(cut), calm)
    assert g.samples == {}
    g.observe(_ctx(full), calm)
    assert len(g.samples["XUSDT"]) == 1

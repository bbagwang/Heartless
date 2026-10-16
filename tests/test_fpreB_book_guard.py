"""F-pre-B book guard (ROADMAP 7.11): the vectorised 1m-bar proxy equals exchange/orderbook.bar_book_proxy bar by bar,
the guard's spread / depth rules against the symbol's own prior 24 hours (stream samples and proxy alike), and the
engine holding an entry in a backtest only while the guard is on."""
import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.data.candles import CandleArrays
from heartless.data.features import FeatureFrame
from heartless.exchange.orderbook import BookFeatures, bar_book_proxy
from heartless.execution.engine import BookGuard, book_proxy_series
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


def test_guard_rules_on_stream_samples():
    g = BookGuard(Settings(_env_file=None))
    t0 = 1_700_006_400_000
    for k in range(1440):
        g.observe(_ctx(BookFeatures(spread_bps=1.0 + 0.001 * (k % 3), depth_bid_k=5e5, depth_ask_k=5e5,
                                    source="stream")), t0 + k * MIN)
    now = t0 + 1440 * MIN
    ok = BookFeatures(spread_bps=4.0, depth_bid_k=4e5, depth_ask_k=4e5, source="stream")
    assert g.check(None, _ctx(ok), now, 0.01) == ""  # 4 bp: over 3x the median but under the 5 bp floor
    wide = BookFeatures(spread_bps=6.0, depth_bid_k=4e5, depth_ask_k=4e5, source="stream")
    assert g.check(None, _ctx(wide), now, 0.01) == "book guard: spread"
    thin = BookFeatures(spread_bps=1.0, depth_bid_k=1.2e5, depth_ask_k=1.2e5, source="stream")
    assert g.check(None, _ctx(thin), now, 0.01) == "book guard: depth"  # 2.4e5 < 25% of 1e6
    # a fresh stream (under 720 prior minutes) judges nothing
    g2 = BookGuard(Settings(_env_file=None))
    for k in range(700):
        g2.observe(_ctx(BookFeatures(spread_bps=1.0, depth_bid_k=5e5, depth_ask_k=5e5, source="stream")), t0 + k * MIN)
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

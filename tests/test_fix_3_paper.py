"""Regression tests for the PaperAccount fixes (intrabar ordering, gap stops, maker trade-through,
tick-mode book pricing, tick-mode funding settlement)."""
import pytest

from heartless.core.models import Candle, Ticker
from heartless.exchange.paper import PaperAccount


def _acc(**kw):
    kw.setdefault("initial_balance", 1000)
    kw.setdefault("slippage_bps", 0)
    kw.setdefault("impact_bps_per_10k", 0)
    kw.setdefault("spread_bps", 0)
    return PaperAccount(**kw)


def _bar(o, h, l, c, t0=60000):
    return Candle(t0, o, h, l, c, 1, 1, 1, 1, t0 + 59999)


def _record(acc):
    fills = []

    async def on_fill(f):
        fills.append(f)

    acc.on_fill(on_fill)
    return fills


# --- 1) intrabar ordering: adverse extreme first for an open position -------------------------------
async def test_long_stop_evaluated_before_target_in_bullish_bar():
    acc = _acc()
    fills = _record(acc)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    await acc.market_order("X", "BUY", 1)
    await acc.place_stop("X", "SELL", 98, close_position=True)
    await acc.place_take_profit("X", "SELL", 102, qty=1)
    # both levels inside a bar that closes UP: pessimistic path visits the low (stop) first
    await acc.on_bar("X", _bar(100, 103, 97, 101))
    assert [f.kind for f in fills] == ["ENTRY", "SL"]
    assert fills[-1].price == pytest.approx(98.0)
    assert acc.positions["X"].qty == 0
    assert acc.wallet < 1000


async def test_short_stop_evaluated_before_target_in_bearish_bar():
    acc = _acc()
    fills = _record(acc)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    await acc.market_order("X", "SELL", 1)
    await acc.place_stop("X", "BUY", 102, close_position=True)
    await acc.place_take_profit("X", "BUY", 98, qty=1)
    await acc.on_bar("X", _bar(100, 103, 97, 99))
    assert [f.kind for f in fills] == ["ENTRY", "SL"]
    assert fills[-1].price == pytest.approx(102.0)
    assert acc.wallet < 1000


async def test_resting_entry_filled_at_last_extreme_cannot_exit_in_same_bar():
    acc = _acc()
    fills = _record(acc)

    async def brackets(f):
        if f.kind == "ENTRY":
            await acc.place_stop("X", "SELL", 96, close_position=True)
            await acc.place_take_profit("X", "SELL", 101.5, qty=1)

    acc.on_fill(brackets)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    r = await acc.limit_order("X", "BUY", 1, 98)
    assert r.status == "NEW"
    # bearish bar: conventional path O-H-L-C -> the entry fills at the low, the high was visited before it
    await acc.on_bar("X", _bar(100, 102, 97.9, 99))
    assert [f.kind for f in fills] == ["ENTRY"]
    assert fills[0].price == pytest.approx(98.0) and fills[0].maker
    assert acc.positions["X"].qty == pytest.approx(1.0)
    assert len([a for a in acc.algos.values() if a.status == "NEW"]) == 2


# --- 2) a stop placed while the market is already through it fills at the market, not the trigger ----
async def test_stop_placed_on_gap_open_fills_at_open_not_trigger():
    acc = _acc()
    fills = _record(acc)

    async def brackets(f):
        if f.kind == "ENTRY":
            await acc.place_stop("X", "SELL", 99.0, close_position=True)
            await acc.place_take_profit("X", "SELL", 101.0, qty=1)

    acc.on_fill(brackets)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    r = await acc.limit_order("X", "BUY", 1, 99.995)
    assert r.status == "NEW"
    await acc.on_bar("X", _bar(98.5, 98.8, 98.0, 98.2))
    assert [(f.kind, f.price) for f in fills] == [("ENTRY", 99.995), ("SL", pytest.approx(98.5))]
    assert acc.positions["X"].qty == 0
    # loss booked at the open (1.495/unit), not at a trigger the bar never revisited (0.995/unit)
    assert 1000 - acc.wallet == pytest.approx(1.495 + 99.995 * 0.0002 + 98.5 * 0.0005, abs=1e-9)
    assert all(a.status != "NEW" or a.kind == "TAKE_PROFIT_MARKET" for a in acc.algos.values())


async def test_stop_placed_at_open_inside_bar_still_fills_at_trigger_intrabar():
    """Control: a bracket placed at the open that is NOT yet through keeps the intrabar trigger fill."""
    acc = _acc()
    fills = _record(acc)

    async def brackets(f):
        if f.kind == "ENTRY":
            await acc.place_stop("X", "SELL", 99.0, close_position=True)

    acc.on_fill(brackets)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    await acc.limit_order("X", "BUY", 1, 99.995)
    await acc.on_bar("X", _bar(99.5, 99.8, 98.0, 98.2))
    assert [(f.kind, f.price) for f in fills] == [("ENTRY", 99.995), ("SL", pytest.approx(99.0))]


# --- 3) maker orders need the opposite side of the book to trade through them ----------------------
async def test_bar_mode_resting_limit_requires_trade_through():
    acc = _acc(spread_bps=1.0)  # default spread: ask = px * (1 + 0.5 bps)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    bid = acc.tickers["X"].bid
    r = await acc.limit_order("X", "BUY", 1, bid)
    assert r.status == "NEW"
    # the low only touches the limit: still resting
    await acc.on_bar("X", _bar(100, 100.5, bid, 100.2))
    assert (await acc.query_order("X", r.order_id)).status == "NEW"
    assert acc.fees_paid == 0
    # the low trades through by more than half a spread: maker fill at the limit price
    await acc.on_bar("X", _bar(100.2, 100.3, bid * (1 - 1e-4), 100.1, 120000))
    q = await acc.query_order("X", r.order_id)
    assert q.status == "FILLED" and q.avg_price == pytest.approx(bid)
    assert acc.fees_paid == pytest.approx(bid * 0.0002)


async def test_bar_mode_resting_sell_limit_requires_trade_through():
    acc = _acc(spread_bps=1.0)
    await acc.on_bar("X", _bar(100, 100, 100, 100, 0))
    ask = acc.tickers["X"].ask
    r = await acc.limit_order("X", "SELL", 1, ask)
    assert r.status == "NEW"
    await acc.on_bar("X", _bar(100, ask, 99.5, 99.8))
    assert (await acc.query_order("X", r.order_id)).status == "NEW"
    await acc.on_bar("X", _bar(99.8, ask * (1 + 1e-4), 99.7, 99.9, 120000))
    assert (await acc.query_order("X", r.order_id)).status == "FILLED"
    assert acc.positions["X"].qty == pytest.approx(-1.0)


# --- 5) tick mode: trigger on mark, price on the book ----------------------------------------------
async def test_tick_mode_stop_fills_at_bid_not_mark():
    acc = _acc()
    fills = _record(acc)
    await acc.on_ticker(Ticker("X", bid=100, ask=100.1, mark=100.05, ts=1))
    await acc.market_order("X", "BUY", 1)
    await acc.place_stop("X", "SELL", 99, close_position=True)
    # mark triggers the stop, but the best bid is far below it
    await acc.on_ticker(Ticker("X", bid=98.0, ask=98.1, mark=98.95, ts=2))
    assert [f.kind for f in fills] == ["ENTRY", "SL"]
    assert fills[-1].price == pytest.approx(98.0)
    assert acc.realized == pytest.approx(98.0 - 100.1)


async def test_tick_mode_stop_never_better_than_trigger():
    acc = _acc()
    fills = _record(acc)
    await acc.on_ticker(Ticker("X", bid=100, ask=100.1, mark=100.05, ts=1))
    await acc.market_order("X", "BUY", 1)
    await acc.place_stop("X", "SELL", 99, close_position=True)
    await acc.on_ticker(Ticker("X", bid=99.3, ask=99.4, mark=98.95, ts=2))
    assert fills[-1].kind == "SL" and fills[-1].price == pytest.approx(99.0)


async def test_tick_mode_take_profit_fills_at_book():
    acc = _acc()
    fills = _record(acc)
    await acc.on_ticker(Ticker("X", bid=100, ask=100.1, mark=100.05, ts=1))
    await acc.market_order("X", "SELL", 1)
    await acc.place_take_profit("X", "BUY", 99, qty=1)
    await acc.on_ticker(Ticker("X", bid=99.2, ask=99.3, mark=98.9, ts=2))
    assert fills[-1].kind == "TP" and fills[-1].price == pytest.approx(99.3)


async def test_tick_mode_resting_limit_needs_the_book_not_the_mark():
    acc = _acc()
    await acc.on_ticker(Ticker("X", bid=100, ask=100.1, mark=100.05, ts=1))
    r = await acc.limit_order("X", "BUY", 1, 99.0)
    assert r.status == "NEW"
    # mark dips through our bid while the book never traded below 99.5 -> no fill
    await acc.on_ticker(Ticker("X", bid=99.5, ask=99.6, mark=98.9, ts=2))
    assert (await acc.query_order("X", r.order_id)).status == "NEW"
    # best ask reaches our bid -> maker fill at our price even though the mark sits above it
    await acc.on_ticker(Ticker("X", bid=98.9, ask=99.0, mark=99.2, ts=3))
    q = await acc.query_order("X", r.order_id)
    assert q.status == "FILLED" and q.avg_price == 99.0
    assert acc.fees_paid == pytest.approx(99.0 * 0.0002)


async def test_mark_only_tick_needs_strict_trade_through():
    acc = _acc()
    await acc.on_ticker(Ticker("X", bid=100, ask=100.1, mark=100.05, ts=1))
    r = await acc.limit_order("X", "BUY", 1, 99.0)
    await acc.on_ticker(Ticker("X", mark=99.0, ts=2))  # no book: touching is not enough
    assert (await acc.query_order("X", r.order_id)).status == "NEW"
    await acc.on_ticker(Ticker("X", mark=98.99, ts=3))
    assert (await acc.query_order("X", r.order_id)).status == "FILLED"


# --- 4) tick mode settles funding on the markPrice next_funding_time rollover ----------------------
async def test_tick_mode_funding_settles_once_per_boundary():
    acc = _acc()
    fills = _record(acc)
    T1, T2 = 28_800_000, 57_600_000
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=28_000_000, funding_rate=0.0005, next_funding_time=T1))
    await acc.market_order("X", "BUY", 10)
    w = acc.wallet
    # still before settlement: nothing happens (rate drifts, T unchanged)
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=28_500_000, funding_rate=0.0004, next_funding_time=T1))
    assert acc.wallet == w and acc.funding_paid == 0
    # first tick after T1 is a mark update that already quotes the NEXT period: the pre-flip rate is paid
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=T1 + 500, funding_rate=-0.0001, next_funding_time=T2))
    assert acc.wallet == pytest.approx(w - 10 * 100 * 0.0004)
    assert acc.funding_paid == pytest.approx(-0.4)
    f = [x for x in fills if x.kind == "FUNDING"]
    assert len(f) == 1 and f[0].fee == pytest.approx(0.4) and f[0].ts == T1 and f[0].symbol == "X"
    # a stale book tick (still carrying T1) after settlement must not settle again
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=T1 + 900, funding_rate=0.0004, next_funding_time=T1))
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=T1 + 1200, funding_rate=-0.0001, next_funding_time=T2))
    assert acc.funding_paid == pytest.approx(-0.4)
    # next boundary: the negative rate pays the long
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=T2 + 1, funding_rate=0.0, next_funding_time=T2 + 28_800_000))
    assert acc.funding_paid == pytest.approx(-0.4 + 10 * 100 * 0.0001)
    assert len([x for x in fills if x.kind == "FUNDING"]) == 2


async def test_tick_mode_funding_ignored_without_funding_info_or_position():
    acc = _acc()
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=1))
    await acc.market_order("X", "BUY", 1)
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=10**12))
    assert acc.funding_paid == 0
    acc2 = _acc()
    await acc2.on_ticker(Ticker("Y", bid=100, ask=100, mark=100, ts=1, funding_rate=0.001, next_funding_time=100))
    await acc2.on_ticker(Ticker("Y", bid=100, ask=100, mark=100, ts=200, funding_rate=0.001, next_funding_time=300))
    assert acc2.funding_paid == 0 and acc2.wallet == 1000
    acc.reset()
    assert not acc._funding_next and not acc._funding_settled

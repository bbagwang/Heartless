import asyncio

import pytest

from heartless.core.models import Candle, Ticker
from heartless.exchange.paper import PaperAccount


async def test_market_entry_tp_sl_cycle():
    acc = PaperAccount(initial_balance=1000, taker_fee=0.0005)
    fills = []

    async def on_fill(f):
        fills.append(f)

    acc.on_fill(on_fill)
    await acc.on_ticker(Ticker("BTCUSDT", bid=100, ask=100.1, mark=100.05, ts=1))
    r = await acc.market_order("BTCUSDT", "BUY", 1, client_id="HLE1")
    assert r.status == "FILLED" and r.avg_price > 100.1  # pays ask + slippage
    await acc.place_stop("BTCUSDT", "SELL", 95, close_position=True)
    await acc.place_take_profit("BTCUSDT", "SELL", 110, qty=0.5)
    await acc.on_ticker(Ticker("BTCUSDT", bid=110.2, ask=110.3, mark=110.2, ts=2))
    assert abs(acc.positions["BTCUSDT"].qty - 0.5) < 1e-9
    await acc.on_ticker(Ticker("BTCUSDT", bid=94.9, ask=95.0, mark=94.9, ts=3))
    assert acc.positions["BTCUSDT"].qty == 0
    assert [f.kind for f in fills] == ["ENTRY", "TP", "SL"]
    # 0.5 * (110.2-100.11..) - 0.5*(100.11-94.9..) minus fees ~ +2.3
    assert acc.wallet > 1000


async def test_post_only_rejected_when_crossing_and_fills_when_touched():
    acc = PaperAccount(initial_balance=1000)
    await acc.on_ticker(Ticker("ETHUSDT", bid=100, ask=100.1, mark=100.05, ts=1))
    r = await acc.limit_order("ETHUSDT", "BUY", 1, 100.2)
    assert r.status == "EXPIRED"
    r = await acc.limit_order("ETHUSDT", "BUY", 1, 99.5)
    assert r.status == "NEW"
    await acc.on_ticker(Ticker("ETHUSDT", bid=99.4, ask=99.5, mark=99.45, ts=2))
    q = await acc.query_order("ETHUSDT", r.order_id)
    assert q.status == "FILLED" and q.avg_price == 99.5
    assert acc.fees_paid == pytest.approx(99.5 * 0.0002)


async def test_bar_driven_stop_fills_at_trigger_not_extreme():
    acc = PaperAccount(initial_balance=1000, slippage_bps=0, impact_bps_per_10k=0, spread_bps=0)
    await acc.on_bar("X", Candle(0, 100, 101, 99, 100, 1, 1, 1, 1, 59999))
    await acc.market_order("X", "SELL", 2)
    await acc.place_stop("X", "BUY", 103, close_position=True)
    await acc.on_bar("X", Candle(60000, 100, 106, 99.5, 105, 1, 1, 1, 1, 119999))
    assert acc.positions["X"].qty == 0
    # loss = 2 * (103 - 100) = 6 plus fees, not 2 * (106 - 100)
    assert 1000 - acc.wallet == pytest.approx(6 + 2 * 100 * 0.0005 + 2 * 103 * 0.0005, abs=1e-6)


async def test_reduce_only_rejected_without_position():
    acc = PaperAccount(initial_balance=1000)
    await acc.on_ticker(Ticker("X", bid=10, ask=10.01, mark=10, ts=1))
    r = await acc.market_order("X", "SELL", 1, reduce_only=True)
    assert r.status == "REJECTED"


async def test_funding_payment():
    acc = PaperAccount(initial_balance=1000)
    await acc.on_ticker(Ticker("X", bid=100, ask=100, mark=100, ts=1))
    await acc.market_order("X", "BUY", 10)
    w = acc.wallet
    await acc.apply_funding("X", 0.0001, 100, 2)
    assert acc.wallet == pytest.approx(w - 10 * 100 * 0.0001)

"""LiveAccount behaviour against a fake REST client (no network)."""
import asyncio

from heartless.core.models import Fill
from heartless.exchange.binance_rest import BinanceError
from heartless.exchange.live import LiveAccount


class FakeRest:
    ws_base = "wss://fake"

    def __init__(self, reject_close_position=False):
        self.calls = []
        self.reject = reject_close_position

    async def new_algo_order(self, symbol, side, type_, trigger_price, **kw):
        self.calls.append(kw)
        if self.reject and kw.get("close_position"):
            raise BinanceError(-1102, "Mandatory parameter 'quantity' was not sent", 400)
        return {"algoId": 123, "clientAlgoId": kw.get("client_algo_id")}

    async def new_order(self, *a, **kw):
        return {"orderId": 1, "clientOrderId": kw.get("client_id"), "status": "FILLED", "executedQty": "1", "avgPrice": "100"}


def test_stop_falls_back_to_quantity_reduce_only():
    acc = LiveAccount(FakeRest(reject_close_position=True))
    aid = asyncio.run(acc.place_stop("BTCUSDT", "SELL", 95.0, qty=0.5, close_position=True, client_id="HLS1"))
    assert aid == "123"
    assert acc.rest.calls[-1]["quantity"] == 0.5 and acc.rest.calls[-1]["reduce_only"] is True


def test_stop_uses_close_position_when_accepted():
    acc = LiveAccount(FakeRest())
    asyncio.run(acc.place_stop("BTCUSDT", "SELL", 95.0, qty=0.5, close_position=True))
    assert acc.rest.calls[-1]["close_position"] is True and len(acc.rest.calls) == 1


def test_user_stream_fill_classification():
    acc = LiveAccount(FakeRest())
    fills: list[Fill] = []

    async def h(f):
        fills.append(f)

    acc.on_fill(h)
    ev = {"e": "ORDER_TRADE_UPDATE", "o": {"s": "BTCUSDT", "c": "HLSABC", "S": "SELL", "o": "MARKET", "ot": "STOP_MARKET",
                                           "x": "TRADE", "X": "FILLED", "l": "0.5", "L": "94.9", "n": "0.02", "N": "USDT",
                                           "T": 1, "i": 99, "R": True, "m": False}}
    asyncio.run(acc._on_user_event(ev))
    assert len(fills) == 1 and fills[0].kind == "SL" and fills[0].reduce_only and fills[0].qty == 0.5
    ev["o"].update({"ot": "MARKET", "c": "HLEXYZ", "R": False, "S": "BUY"})
    asyncio.run(acc._on_user_event(ev))
    assert fills[-1].kind == "ENTRY"
    # non-trade execution types are ignored
    ev["o"]["x"] = "NEW"
    asyncio.run(acc._on_user_event(ev))
    assert len(fills) == 2

"""Regression tests for heartless/exchange/live.py hardening (no network):

* transport errors / 5xx / -1007 on a new-order POST never escape and never masquerade as a definitive REJECTED,
* hedge mode never sends reduceOnly together with positionSide,
* query_order only maps "order does not exist" to UNKNOWN and raises transient failures,
* cancel paths swallow transport errors,
* FUNDING_FEE ACCOUNT_UPDATE events are booked as FUNDING fills with the paper sign convention.
"""
import asyncio

import httpx
import pytest

import heartless.exchange.live as live
from heartless.config import Settings
from heartless.core.models import Fill, Position, PositionStatus, Side
from heartless.exchange.binance_rest import BinanceError
from heartless.exchange.live import LiveAccount
from heartless.execution.engine import TradingEngine
from heartless.strategy.params import StrategyParams
from synth import synth_symbols

FILLED = {"orderId": 7, "clientOrderId": "HLC1", "status": "FILLED", "executedQty": "0.5", "avgPrice": "100.5"}


class FakeRest:
    ws_base = "wss://fake"

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.new_order_error: Exception | None = None
        self.query_responses: list = []  # dicts or exceptions, consumed in order (last one repeats)
        self.cancel_error: Exception | None = None
        self.income_rows: list[dict] = []
        self.income_error: Exception | None = None
        self.reject_close_position = False
        self.dual = False
        self.switch_error: Exception | None = None
        self.positions: list[dict] = []

    # helpers
    def count(self, name):
        return sum(1 for n, _ in self.calls if n == name)

    def last(self, name):
        return [kw for n, kw in self.calls if n == name][-1]

    # orders
    async def new_order(self, symbol, side, type_, **kw):
        self.calls.append(("new_order", {"symbol": symbol, "side": side, "type": type_, **kw}))
        if self.new_order_error is not None:
            raise self.new_order_error
        return {"orderId": 1, "clientOrderId": kw.get("client_id"), "status": "FILLED" if type_ == "MARKET" else "NEW",
                "executedQty": kw.get("quantity") if type_ == "MARKET" else "0", "avgPrice": "100"}

    async def new_algo_order(self, symbol, side, type_, trigger_price, **kw):
        self.calls.append(("new_algo_order", {"symbol": symbol, "side": side, "type": type_, **kw}))
        if self.reject_close_position and kw.get("close_position"):
            raise BinanceError(-1102, "Mandatory parameter 'quantity' was not sent", 400)
        return {"algoId": 123, "clientAlgoId": kw.get("client_algo_id")}

    async def query_order(self, symbol, order_id=None, client_id=None):
        self.calls.append(("query_order", {"symbol": symbol, "order_id": order_id, "client_id": client_id}))
        if not self.query_responses:
            raise BinanceError(-2013, "Order does not exist.", 400)
        r = self.query_responses[0] if len(self.query_responses) == 1 else self.query_responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def cancel_order(self, symbol, order_id=None, client_id=None):
        self.calls.append(("cancel_order", {}))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {}

    async def cancel_algo_order(self, symbol, algo_id=None, client_algo_id=None):
        self.calls.append(("cancel_algo_order", {}))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {}

    async def cancel_all_orders(self, symbol):
        self.calls.append(("cancel_all_orders", {}))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {}

    async def cancel_all_algo_orders(self, symbol):
        self.calls.append(("cancel_all_algo_orders", {}))
        if self.cancel_error is not None:
            raise self.cancel_error
        return {}

    async def income_history(self, income_type=None, start=None, end=None, limit=1000):
        self.calls.append(("income_history", {"income_type": income_type, "start": start, "end": end}))
        if self.income_error is not None:
            raise self.income_error
        return list(self.income_rows)

    # start() plumbing
    async def sync_time(self):
        return None

    async def get_position_mode(self):
        return self.dual

    async def set_position_mode(self, dual):
        self.calls.append(("set_position_mode", {"dual": dual}))
        if self.switch_error is not None:
            raise self.switch_error
        self.dual = dual

    async def position_risk(self, symbol=None):
        return list(self.positions)

    async def account(self):
        return {"totalWalletBalance": "1000", "totalUnrealizedProfit": "0", "availableBalance": "1000"}

    async def create_listen_key(self):
        raise RuntimeError("no network in tests")

    async def close_listen_key(self):
        return None


def _acc(hedge=False):
    rest = FakeRest()
    acc = LiveAccount(rest)
    acc.hedge_mode = hedge
    return rest, acc


def _collect(acc):
    fills: list[Fill] = []

    async def h(f):
        fills.append(f)

    acc.on_fill(h)
    return fills


def _funding_event(bc="-0.5", positions=({"s": "BTCUSDT", "pa": "1", "ep": "100", "up": "0", "mt": "isolated"},), ts=1):
    return {"e": "ACCOUNT_UPDATE", "E": ts, "T": ts,
            "a": {"m": "FUNDING_FEE", "B": [{"a": "USDT", "wb": "999.5", "cw": "999.5", "bc": bc}], "P": list(positions)}}


# --- finding 2: hedge mode ------------------------------------------------------------------------------------

def test_hedge_mode_sends_position_side_and_never_reduce_only():
    rest, acc = _acc(hedge=True)
    asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
    kw = rest.last("new_order")
    assert not kw["reduce_only"] and kw["position_side"] == "LONG" and kw["client_id"] == "HLC1"
    asyncio.run(acc.market_order("BTCUSDT", "BUY", 0.5, client_id="HLE1"))  # entry leg
    kw = rest.last("new_order")
    assert not kw["reduce_only"] and kw["position_side"] == "LONG"
    asyncio.run(acc.limit_order("BTCUSDT", "BUY", 0.5, 99.0, reduce_only=True, client_id="HLC2"))
    kw = rest.last("new_order")
    assert not kw["reduce_only"] and kw["position_side"] == "SHORT" and kw["time_in_force"] == "GTX"
    asyncio.run(acc.place_take_profit("BTCUSDT", "SELL", 110.0, 0.25, client_id="HLT1"))
    kw = rest.last("new_algo_order")
    assert kw["type"] == "TAKE_PROFIT_MARKET" and not kw.get("reduce_only") and kw["position_side"] == "LONG"
    rest.reject_close_position = True
    asyncio.run(acc.place_stop("BTCUSDT", "SELL", 95.0, qty=0.5, close_position=True, client_id="HLS1"))
    kw = rest.last("new_algo_order")
    assert kw["quantity"] == 0.5 and not kw.get("reduce_only") and kw["position_side"] == "LONG"


def test_one_way_mode_still_sends_reduce_only_without_position_side():
    rest, acc = _acc(hedge=False)
    asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
    kw = rest.last("new_order")
    assert kw["reduce_only"] is True and kw["position_side"] is None
    asyncio.run(acc.place_take_profit("BTCUSDT", "SELL", 110.0, 0.25, client_id="HLT1"))
    kw = rest.last("new_algo_order")
    assert kw["reduce_only"] is True and kw["position_side"] is None
    rest.reject_close_position = True
    asyncio.run(acc.place_stop("BTCUSDT", "SELL", 95.0, qty=0.5, close_position=True, client_id="HLS1"))
    kw = rest.last("new_algo_order")
    assert kw["quantity"] == 0.5 and kw["reduce_only"] is True and kw["position_side"] is None


def test_start_rereads_position_mode_when_switch_to_one_way_fails():
    async def run(switch_error):
        rest = FakeRest()
        rest.dual = True
        rest.switch_error = switch_error
        acc = LiveAccount(rest)
        await acc.start()
        acc._stream_task.cancel()
        return acc.hedge_mode

    assert asyncio.run(run(None)) is False
    assert asyncio.run(run(BinanceError(-4068, "Position side cannot be changed if there exists open orders.", 400))) is True


# --- findings 1 & 3: uncertain order sends ----------------------------------------------------------------------

def test_market_order_timeout_is_resolved_by_client_id_lookup():
    rest, acc = _acc()
    rest.new_order_error = httpx.ReadTimeout("read timed out")
    rest.query_responses = [dict(FILLED)]
    res = asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
    assert res.status == "FILLED" and res.filled_qty == 0.5 and res.avg_price == 100.5 and res.order_id == "7"
    assert rest.count("query_order") == 1 and rest.last("query_order")["client_id"] == "HLC1"


def test_market_order_timeout_with_order_never_received_is_rejected(monkeypatch):
    monkeypatch.setattr(live, "UNKNOWN_RECHECK_DELAY", 0)
    rest, acc = _acc()
    rest.new_order_error = httpx.ReadTimeout("read timed out")
    rest.query_responses = [BinanceError(-2013, "Order does not exist.", 400)]
    res = asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
    assert res.status == "REJECTED" and res.raw.get("uncertain") is True
    assert rest.count("query_order") == 2  # confirmed twice before giving up on the order


def test_market_order_timeout_with_failed_lookup_is_unknown_not_rejected():
    for lookup_error in (httpx.ReadTimeout("again"), BinanceError(-1003, "Too many requests", 429),
                         BinanceError(-1, "bad gateway", 502)):
        rest, acc = _acc()
        rest.new_order_error = httpx.RemoteProtocolError("connection reset")
        rest.query_responses = [lookup_error]
        res = asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
        assert res.status == "UNKNOWN" and res.client_id == "HLC1", lookup_error
        assert rest.count("query_order") == 1


def test_market_order_without_client_id_is_unknown_after_timeout():
    rest, acc = _acc()
    rest.new_order_error = httpx.ReadTimeout("read timed out")
    res = asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5))
    assert res.status == "UNKNOWN" and rest.count("query_order") == 0


def test_connect_error_means_the_order_was_never_sent():
    rest, acc = _acc()
    rest.new_order_error = httpx.ConnectError("connection refused")
    res = asyncio.run(acc.market_order("BTCUSDT", "BUY", 0.5, client_id="HLE1"))
    assert res.status == "REJECTED" and rest.count("query_order") == 0


def test_binance_5xx_and_unknown_status_codes_are_not_rejections():
    for err in (BinanceError(-1007, "Timeout waiting for response from backend server. Send status unknown; "
                                    "execution status unknown.", 504),
                BinanceError(-1, "<html>502 Bad Gateway</html>", 502),
                BinanceError(-1001, "Internal error; unable to process your request.", 500),
                BinanceError(-1000, "An unknown error occurred while processing the request.", 200)):
        rest, acc = _acc()
        rest.new_order_error = err
        rest.query_responses = [dict(FILLED)]
        res = asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
        assert res.status == "FILLED" and res.filled_qty == 0.5, err
        assert rest.count("query_order") == 1


def test_4xx_application_errors_are_definitive_rejections():
    for err in (BinanceError(-1102, "Mandatory parameter 'quantity' was not sent", 400),
                BinanceError(-2022, "ReduceOnly Order is rejected.", 200),
                BinanceError(-1003, "Too many requests", 429)):
        rest, acc = _acc()
        rest.new_order_error = err
        rest.query_responses = [dict(FILLED)]  # must not even be consulted
        res = asyncio.run(acc.market_order("BTCUSDT", "SELL", 0.5, reduce_only=True, client_id="HLC1"))
        assert res.status == "REJECTED" and res.raw["code"] == err.code, err
        assert rest.count("query_order") == 0


def test_limit_order_keeps_gtx_expiry_and_resolves_timeouts():
    rest, acc = _acc()
    rest.new_order_error = BinanceError(-5022, "Due to the order could not be executed as maker, the Post Only "
                                               "order will be rejected.", 400)
    res = asyncio.run(acc.limit_order("BTCUSDT", "BUY", 0.5, 99.0, client_id="HLE1"))
    assert res.status == "EXPIRED"
    rest.new_order_error = httpx.WriteTimeout("write timed out")
    rest.query_responses = [{"orderId": 9, "clientOrderId": "HLE1", "status": "NEW", "executedQty": "0", "avgPrice": "0"}]
    res = asyncio.run(acc.limit_order("BTCUSDT", "BUY", 0.5, 99.0, client_id="HLE1"))
    assert res.status == "NEW" and res.order_id == "9"


# --- finding 4: query_order ----------------------------------------------------------------------------------

def test_query_order_only_maps_missing_orders_to_unknown():
    rest, acc = _acc()
    rest.query_responses = [BinanceError(-2013, "Order does not exist.", 400)]
    res = asyncio.run(acc.query_order("BTCUSDT", client_id="HLE1"))
    assert res.status == "UNKNOWN" and res.raw["code"] == -2013
    rest.query_responses = [BinanceError(-1003, "Too many requests", 429)]
    with pytest.raises(BinanceError):
        asyncio.run(acc.query_order("BTCUSDT", client_id="HLE1"))
    rest.query_responses = [BinanceError(-1, "bad gateway", 502)]
    with pytest.raises(BinanceError):
        asyncio.run(acc.query_order("BTCUSDT", client_id="HLE1"))
    rest.query_responses = [httpx.ReadTimeout("read timed out")]
    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(acc.query_order("BTCUSDT", client_id="HLE1"))
    rest.query_responses = [dict(FILLED)]
    assert asyncio.run(acc.query_order("BTCUSDT", client_id="HLC1")).status == "FILLED"


# --- finding 1: cancel paths ---------------------------------------------------------------------------------

def test_cancel_paths_swallow_transport_errors():
    rest, acc = _acc()
    rest.cancel_error = httpx.ReadTimeout("read timed out")
    assert asyncio.run(acc.cancel_order("BTCUSDT", client_id="HLE1")) is False
    assert asyncio.run(acc.cancel_algo("BTCUSDT", "123")) is False
    asyncio.run(acc.cancel_all("BTCUSDT"))  # must not raise
    rest.cancel_error = BinanceError(-2011, "Unknown order sent.", 400)
    assert asyncio.run(acc.cancel_order("BTCUSDT", client_id="HLE1")) is False
    rest.cancel_error = None
    assert asyncio.run(acc.cancel_order("BTCUSDT", client_id="HLE1")) is True


# --- findings 5/6/7: funding ---------------------------------------------------------------------------------

def test_isolated_funding_fee_event_emits_funding_fill():
    rest, acc = _acc()
    fills = _collect(acc)
    asyncio.run(acc._on_user_event(_funding_event("-0.5")))
    assert len(fills) == 1
    f = fills[0]
    assert f.kind == "FUNDING" and f.symbol == "BTCUSDT" and f.fee == 0.5 and f.qty == 0.0 and f.order_side == "FUNDING"
    assert f.price == 100.0 and f.ts == 1
    assert rest.count("income_history") == 0
    assert acc.cached_positions["BTCUSDT"].qty == 1.0  # the regular position refresh still happens
    # funding received -> negative fee (paper convention: fee = -payment)
    asyncio.run(acc._on_user_event(_funding_event("0.3", ts=3_600_001)))
    assert fills[-1].fee == pytest.approx(-0.3)
    # other ACCOUNT_UPDATE reasons never produce fills
    ev = _funding_event("-0.5")
    ev["a"]["m"] = "ORDER"
    asyncio.run(acc._on_user_event(ev))
    assert len(fills) == 2


def test_cross_margin_funding_event_uses_income_ledger_and_dedups():
    rest, acc = _acc()
    fills = _collect(acc)
    rest.income_rows = [{"symbol": "BTCUSDT", "incomeType": "FUNDING_FEE", "income": "-0.5", "time": 1, "tranId": 11},
                        {"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": "0.2", "time": 1, "tranId": 12}]
    ev = _funding_event("-0.3", positions=())
    asyncio.run(acc._on_user_event(ev))
    assert rest.count("income_history") == 1 and rest.last("income_history")["income_type"] == "FUNDING_FEE"
    assert {(f.symbol, round(f.fee, 6)) for f in fills} == {("BTCUSDT", 0.5), ("ETHUSDT", -0.2)}
    assert all(f.kind == "FUNDING" for f in fills)
    asyncio.run(acc._on_user_event(ev))  # a second event for the same settlement must not double count
    assert len(fills) == 2
    rest.income_error = httpx.ReadTimeout("read timed out")
    asyncio.run(acc._on_user_event(ev))  # ledger failure is logged, never raised into the user stream
    assert len(fills) == 2


def test_directly_booked_funding_is_not_double_counted_by_the_ledger():
    rest, acc = _acc()
    fills = _collect(acc)
    ts = 5 * 3_600_000 + 1000
    asyncio.run(acc._on_user_event(_funding_event("-0.5", ts=ts)))
    rest.income_rows = [{"symbol": "BTCUSDT", "income": "-0.5", "time": ts, "tranId": 21},
                        {"symbol": "ETHUSDT", "income": "-0.1", "time": ts, "tranId": 22}]
    asyncio.run(acc._on_user_event(_funding_event("-0.6", positions=(), ts=ts + 5)))
    assert [(f.symbol, round(f.fee, 6)) for f in fills] == [("BTCUSDT", 0.5), ("ETHUSDT", 0.1)]


def test_live_funding_fill_reaches_engine_position():
    rest, acc = _acc()
    syms = synth_symbols(["BTCUSDT"])
    eng = TradingEngine("live", acc, StrategyParams.default(), Settings(_env_file=None), syms, store=None, bus=None,
                        persist=False)
    pos = Position(id="P1", engine="live", symbol="BTCUSDT", side=Side.LONG, qty=1.0, entry_price=100.0, entry_time=0,
                   stop=95.0, take_profit=110.0, tp1=None, initial_stop=95.0, alpha="trend_pullback", alphas=["trend_pullback"],
                   reason="t", confidence=0.5, regime="RANGE", risk_amount=5.0, r_unit=5.0, notional=100.0, leverage=5,
                   params_version="v", status=PositionStatus.OPEN, filled_qty=1.0, original_qty=1.0)
    eng.positions["BTCUSDT"] = pos  # the engine subscribed to account fills in its constructor
    asyncio.run(acc._on_user_event(_funding_event("-0.5")))
    assert pos.funding == pytest.approx(-0.5)  # paid funding lowers net pnl, as in paper/backtest
    assert pos.net_pnl() == pytest.approx(-0.5)

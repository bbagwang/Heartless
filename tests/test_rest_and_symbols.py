import asyncio

from heartless.core.models import SymbolInfo
from heartless.exchange.binance_rest import BinanceRest, _fmt
from heartless.exchange.symbols import min_qty_for_notional, norm_price, norm_qty, parse_exchange_info


def test_signature_and_params():
    r = BinanceRest("key", "secret")
    p = r._sign({"symbol": "BTCUSDT", "side": "BUY", "x": None})
    assert "x" not in p and p["recvWindow"] == 5000 and len(p["signature"]) == 64
    asyncio.run(r.close())


def test_fmt_numbers():
    assert _fmt(0.00001230) == "0.0000123"
    assert _fmt(12345.0) == "12345"
    assert _fmt(1e-8) == "0.00000001"


def test_algo_order_requires_conditional_type():
    r = BinanceRest("key", "secret")
    try:
        asyncio.run(r.new_order("BTCUSDT", "SELL", "STOP_MARKET", quantity=1))
        assert False, "conditional orders must not go through /fapi/v1/order"
    except ValueError:
        pass
    finally:
        asyncio.run(r.close())


def test_exchange_info_parsing_and_rounding():
    data = {"symbols": [{"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT", "pricePrecision": 2,
                         "quantityPrecision": 3, "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1,
                         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                                     {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                     {"filterType": "MIN_NOTIONAL", "notional": "100"},
                                     {"filterType": "MAX_NUM_ALGO_ORDERS", "limit": "10"}]}]}
    infos = parse_exchange_info(data)
    info = infos["BTCUSDT"]
    assert info.tick_size == 0.1 and info.step_size == 0.001 and info.min_notional == 100
    assert norm_price(info, 100.04) == 100.0 and norm_price(info, 100.06) == 100.1
    assert norm_price(info, 100.01, "SELL") == 100.1 and norm_price(info, 100.09, "BUY") == 100.0
    assert norm_qty(info, 0.0015) == 0.001 and norm_qty(info, 0.0004) == 0.0
    assert min_qty_for_notional(info, 50_000) == 0.002

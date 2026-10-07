"""Exchange-info parsing and price/quantity normalisation."""
from __future__ import annotations

from heartless.core.models import SymbolInfo
from heartless.util.mathutil import round_step, round_up_step


def parse_exchange_info(data: dict) -> dict[str, SymbolInfo]:
    out: dict[str, SymbolInfo] = {}
    for s in data.get("symbols", []):
        filters = {f["filterType"]: f for f in s.get("filters", [])}
        pf = filters.get("PRICE_FILTER", {})
        lf = filters.get("LOT_SIZE", {})
        mn = filters.get("MIN_NOTIONAL", {})
        algo = filters.get("MAX_NUM_ALGO_ORDERS", {})
        info = SymbolInfo(
            symbol=s["symbol"], base=s.get("baseAsset", ""), quote=s.get("quoteAsset", ""),
            tick_size=float(pf.get("tickSize", 0.01) or 0.01), step_size=float(lf.get("stepSize", 0.001) or 0.001),
            min_qty=float(lf.get("minQty", 0.0) or 0.0), min_notional=float(mn.get("notional", 5.0) or 5.0),
            price_precision=int(s.get("pricePrecision", 2)), quantity_precision=int(s.get("quantityPrecision", 3)),
            max_algo_orders=int(algo.get("limit", algo.get("maxNumAlgoOrders", 10)) or 10),
            status=s.get("status", "TRADING"), contract_type=s.get("contractType", "PERPETUAL"),
            onboard_date=int(s.get("onboardDate", 0) or 0),
        )
        out[info.symbol] = info
    return out


def norm_price(info: SymbolInfo, price: float, side_hint: str | None = None) -> float:
    """Round to tick size. side_hint 'BUY' rounds down (never pay more), 'SELL' rounds up."""
    if side_hint == "SELL":
        return round_up_step(price, info.tick_size)
    if side_hint == "BUY":
        return round_step(price, info.tick_size, "down")
    return round_step(price, info.tick_size, "nearest")


def norm_qty(info: SymbolInfo, qty: float) -> float:
    q = round_step(qty, info.step_size, "down")
    return q if q >= info.min_qty else 0.0


def min_qty_for_notional(info: SymbolInfo, price: float) -> float:
    if price <= 0:
        return 0.0
    q = round_up_step(info.min_notional / price, info.step_size)
    return max(q, info.min_qty)

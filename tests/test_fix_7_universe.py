"""Regression tests for heartless/data/universe.py: ALWAYS_INCLUDE symbols must pass the same
instrument checks (USDT quote, PERPETUAL contract, TRADING status) as ordinary candidates."""
import logging

from heartless.data.universe import select_universe
from heartless.exchange.symbols import parse_exchange_info
from heartless.util.timeutil import MS_DAY, now_ms

OLD = now_ms() - 400 * MS_DAY


def _sym(symbol: str, base: str, quote: str = "USDT", status: str = "TRADING", contract: str = "PERPETUAL",
         onboard: int = OLD) -> dict:
    return {"symbol": symbol, "baseAsset": base, "quoteAsset": quote, "pricePrecision": 2, "quantityPrecision": 3,
            "status": status, "contractType": contract, "onboardDate": onboard,
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"}]}


def _infos():
    return parse_exchange_info({"symbols": [
        _sym("BTCUSDT", "BTC"),
        _sym("ETHUSDT", "ETH"),
        _sym("SOLUSDT", "SOL"),
        _sym("XRPUSDT", "XRP"),
        _sym("DOGEUSDT", "DOGE"),
        # always-include candidates that must be rejected
        _sym("OLDUSDT", "OLD", status="SETTLING"),                      # delisting
        _sym("BTCUSDT_260327", "BTC", contract="CURRENT_QUARTER"),      # delivery contract
        _sym("BTCUSDC", "BTC", quote="USDC"),                           # USDC-margined perp
        _sym("PAUSEDUSDT", "PAUSED", status="BREAK"),                   # halted
    ]})


def _tickers(**qv):
    return [{"symbol": s, "quoteVolume": str(v)} for s, v in qv.items()]


def test_always_include_respects_instrument_filters(caplog):
    infos = _infos()
    tickers = _tickers(BTCUSDT=9e9, ETHUSDT=8e9, SOLUSDT=7e9, XRPUSDT=6e9, DOGEUSDT=5e9,
                       OLDUSDT=9e9, BTCUSDT_260327=9e9, BTCUSDC=9e9, PAUSEDUSDT=9e9)
    always = ["BTCUSDT", "OLDUSDT", "BTCUSDT_260327", "BTCUSDC", "PAUSEDUSDT", "MISSINGUSDT"]
    with caplog.at_level(logging.WARNING, logger="heartless.data.universe"):
        uni = select_universe(infos, tickers, size=4, min_quote_volume=1e6, always=always)
    # the valid always symbol is kept and seeded first
    assert uni[0] == "BTCUSDT"
    # none of the ineligible always symbols leak into the universe
    for bad in ("OLDUSDT", "BTCUSDT_260327", "BTCUSDC", "PAUSEDUSDT", "MISSINGUSDT"):
        assert bad not in uni
    # their slots go to the next-best eligible candidates and the size cap is honoured
    assert uni == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    assert len(uni) == len(set(uni))
    skipped = {rec.getMessage().split()[1] for rec in caplog.records if "always-include" in rec.getMessage()}
    assert skipped == {"OLDUSDT", "BTCUSDT_260327", "BTCUSDC", "PAUSEDUSDT", "MISSINGUSDT"}


def test_always_include_keeps_volume_and_age_exemptions():
    """Fix must not remove the existing exemptions: a TRADING USDT perp in `always` is kept even when
    it is too young or too illiquid to qualify as an ordinary candidate."""
    infos = parse_exchange_info({"symbols": [
        _sym("BTCUSDT", "BTC"),
        _sym("ETHUSDT", "ETH"),
        _sym("NEWUSDT", "NEW", onboard=now_ms() - 1 * MS_DAY),   # too young
        _sym("THINUSDT", "THIN"),                                 # too illiquid
    ]})
    tickers = _tickers(BTCUSDT=9e9, ETHUSDT=8e9, NEWUSDT=7e9, THINUSDT=10.0)
    uni = select_universe(infos, tickers, size=10, min_quote_volume=1e6, always=["NEWUSDT", "THINUSDT"])
    assert uni[:2] == ["NEWUSDT", "THINUSDT"]
    assert set(uni) == {"NEWUSDT", "THINUSDT", "BTCUSDT", "ETHUSDT"}
    # without the always exemption they are excluded as before
    uni2 = select_universe(infos, tickers, size=10, min_quote_volume=1e6, always=[])
    assert uni2 == ["BTCUSDT", "ETHUSDT"]


def test_ordinary_candidates_still_filtered():
    """The shared predicate must keep rejecting non-USDT / non-perpetual / non-trading candidates."""
    infos = _infos()
    tickers = _tickers(BTCUSDT=1e9, ETHUSDT=1e9, SOLUSDT=1e9, XRPUSDT=1e9, DOGEUSDT=1e9,
                       OLDUSDT=9e9, BTCUSDT_260327=9e9, BTCUSDC=9e9, PAUSEDUSDT=9e9)
    uni = select_universe(infos, tickers, size=20, min_quote_volume=1e6, always=[])
    assert set(uni) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"}

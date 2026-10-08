"""ALGO_UPDATE: a protective stop rejected/expired by the exchange is re-armed immediately."""
import asyncio

from heartless.config import Settings
from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, SymbolInfo, Ticker
from heartless.data.features import MarketView
from heartless.exchange.live import LiveAccount
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams

INFO = SymbolInfo("ETHUSDT", "ETH", "USDT", 0.01, 0.001, 0.001, 5.0, 2, 3)


def _engine():
    acc = PaperAccount("t", 10_000, slippage_bps=0, impact_bps_per_10k=0, spread_bps=0)
    eng = TradingEngine("t", acc, StrategyParams.default(), Settings(_env_file=None), {"ETHUSDT": INFO},
                        persist=False, clock=lambda: 10_000_000)
    return acc, eng


async def _open(acc, eng):
    await acc.on_ticker(Ticker("ETHUSDT", bid=3000, ask=3000, mark=3000, last=3000, ts=1))
    sig = Signal("trend_pullback", "ETHUSDT", Side.LONG, 0.8, "t", 2950.0, 3100.0, None, EntryStyle.MARKET, None, 0, 0.0,
                 20.0, "15m", {"ref_price": 3000.0})
    d = Decision("ETHUSDT", Side.LONG, 0.8, 0.8, ["trend_pullback"], sig, "t", Regime.RANGE, 1.0, 2.0)
    await eng._open(d, MarketView("ETHUSDT"), Context("ETHUSDT", INFO, acc.tickers["ETHUSDT"], Regime.RANGE), 10_000)
    return eng.positions["ETHUSDT"]


def test_rejected_stop_is_rearmed():
    async def run():
        acc, eng = _engine()
        pos = await _open(acc, eng)
        old = pos.sl_algo_id
        acc.algos[old].status = "REJECTED"  # what the exchange did
        await acc._emit_algo_event({"symbol": "ETHUSDT", "algo_id": old, "status": "REJECTED", "order_type": "STOP_MARKET"})
        assert pos.sl_algo_id and pos.sl_algo_id != old
        assert acc.algos[pos.sl_algo_id].status == "NEW" and acc.algos[pos.sl_algo_id].trigger == pos.stop

    asyncio.run(run())


def test_unrelated_or_benign_events_are_ignored():
    async def run():
        acc, eng = _engine()
        pos = await _open(acc, eng)
        sid = pos.sl_algo_id
        await acc._emit_algo_event({"symbol": "ETHUSDT", "algo_id": sid, "status": "TRIGGERED"})
        await acc._emit_algo_event({"symbol": "BTCUSDT", "algo_id": sid, "status": "REJECTED"})
        await acc._emit_algo_event({"symbol": "ETHUSDT", "algo_id": "other", "status": "EXPIRED"})
        assert pos.sl_algo_id == sid

    asyncio.run(run())


def test_live_account_parses_algo_update_both_spellings():
    class R:
        ws_base = "wss://x"

    got = []

    async def h(ev):
        got.append(ev)

    acc = LiveAccount(R())
    acc.on_algo_event(h)
    asyncio.run(acc._on_user_event({"e": "ALGO_UPDATE", "o": {"aid": 7, "caid": "HLSx", "s": "ETHUSDT", "X": "EXPIRED", "o": "STOP_MARKET"}}))
    asyncio.run(acc._on_user_event({"e": "ALGO_UPDATE", "o": {"algoId": 8, "clientAlgoId": "HLSy", "symbol": "ETHUSDT",
                                                              "algoStatus": "rejected", "orderType": "STOP_MARKET"}}))
    assert [g["algo_id"] for g in got] == ["7", "8"] and got[1]["status"] == "REJECTED" and got[0]["client_id"] == "HLSx"

"""Regression tests: float-residue rounding, dust flattening and the post-deadline holding cap."""
import asyncio

from heartless.config import Settings
from heartless.core.models import Candle, Decision, EntryStyle, Regime, Side, Signal, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.exchange.paper import PaperAccount
from heartless.exchange.symbols import norm_qty
from heartless.execution.engine import TradingEngine
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.mathutil import round_step, round_up_step

INFO = SymbolInfo("BTCUSDT", "BTC", "USDT", tick_size=0.1, step_size=0.0001, min_qty=0.0001, min_notional=5.0,
                  price_precision=1, quantity_precision=4)


def test_round_step_tolerates_float_residue():
    assert round_step(0.0564 - 0.0282, 0.0001) == 0.0282
    assert round_step(0.00019999, 0.0001) == 0.0001  # genuinely below the next step still rounds down
    assert round_up_step(0.1 + 0.2, 0.1) == 0.3  # 0.30000000000000004 must not round up to 0.4
    assert norm_qty(INFO, 0.0564 - 0.0282) == 0.0282


def _engine():
    s = Settings(_env_file=None, DAILY_LOSS_LIMIT_PCT=1000, WEEKLY_LOSS_LIMIT_PCT=1000, MAX_DRAWDOWN_HALT_PCT=1000)
    acc = PaperAccount("t", 10_000, slippage_bps=0, impact_bps_per_10k=0, spread_bps=0)
    eng = TradingEngine("t", acc, StrategyParams.default(), s, {"BTCUSDT": INFO}, persist=False, clock=lambda: 1)
    return acc, eng


def _decision(side=Side.SHORT, entry=100_000.0, stop=100_500.0, tp=99_000.0, tp1=99_500.0, max_hold=0, trail=0.0):
    sig = Signal("mean_reversion", "BTCUSDT", side, 0.8, "test", stop, tp, tp1, EntryStyle.MARKET, None, max_hold, trail,
                 200.0, "5m", {"ref_price": entry})
    return Decision("BTCUSDT", side, 0.8, 0.8, ["mean_reversion"], sig, "test", Regime.RANGE, 1.0, 2.0)


def _ctx(px):
    t = Ticker("BTCUSDT", bid=px, ask=px, mark=px, last=px, ts=1)
    return Context("BTCUSDT", INFO, t, Regime.RANGE, now=1)


async def _open(acc, eng, px=100_000.0, **kw):
    await acc.on_ticker(Ticker("BTCUSDT", bid=px, ask=px, mark=px, last=px, ts=1))
    view = MarketView("BTCUSDT")
    await eng._open(_decision(entry=px, **kw), view, _ctx(px), 10_000.0)
    return eng.positions["BTCUSDT"], view


def test_brackets_cover_the_whole_position_and_nothing_is_left_after_both_targets():
    async def run():
        acc, eng = _engine()
        pos, _ = await _open(acc, eng)
        assert pos.status.value == "OPEN"
        tp1_qty = pos.extra.get("tp1_qty", 0.0)
        runner_algo = acc.algos[pos.tp_algo_id]
        assert abs(tp1_qty + runner_algo.qty - pos.qty) < 1e-12  # no uncovered residue
        await acc.on_ticker(Ticker("BTCUSDT", bid=99_450, ask=99_450, mark=99_450, last=99_450, ts=2))  # TP1
        await acc.on_ticker(Ticker("BTCUSDT", bid=98_900, ask=98_900, mark=98_900, last=98_900, ts=3))  # TP
        assert "BTCUSDT" not in eng.positions and eng.closed and eng.closed[-1].exit_reason.startswith("익절")
        assert acc.positions["BTCUSDT"].qty == 0

    asyncio.run(run())


def test_dust_left_by_an_exit_is_flattened():
    async def run():
        acc, eng = _engine()
        pos, _ = await _open(acc, eng, tp1=None)
        # simulate an exchange-side fill that leaves one step behind
        full = pos.qty
        await acc.market_order("BTCUSDT", "BUY", round(full - 0.0001, 4), reduce_only=True, client_id="HLTx")
        assert "BTCUSDT" not in eng.positions, "a one-step leftover must be flattened, not managed forever"
        assert acc.positions["BTCUSDT"].qty == 0
        assert "잔량 정리" in eng.closed[-1].exit_reason

    asyncio.run(run())


def test_winner_past_three_deadlines_is_closed():
    async def run():
        acc, eng = _engine()
        pos, view = await _open(acc, eng, tp=None, tp1=None, max_hold=10)
        px = 99_000.0  # +2R for the short: survives the first deadline
        await acc.on_ticker(Ticker("BTCUSDT", bid=px, ask=px, mark=px, last=px, ts=2))
        for _ in range(35):
            if "BTCUSDT" not in eng.positions:
                break
            await eng._manage(eng.positions["BTCUSDT"], view, _ctx(px))
        assert "BTCUSDT" not in eng.positions
        assert eng.closed[-1].exit_reason == "최대 보유시간 초과 청산"

    asyncio.run(run())

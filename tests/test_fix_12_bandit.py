"""Regression tests for heartless/learning/bandit.py (round 12).

The paper champion and the live engine share one ``AlphaBandit`` and trade the same params on the same bars, so each
market outcome used to be booked twice (1.0 paper + 1.5 live = 2.5x evidence from one real event). ``live_attached``
gates paper evidence while a live engine shares the bandit; default behaviour (flag off) is unchanged.
"""
from __future__ import annotations

import asyncio

import pytest

from heartless.config import Settings
from heartless.core.models import AccountState, Position, PositionStatus, Regime, Side
from heartless.exchange.base import Account, OrderResult
from heartless.exchange.paper import PaperAccount
from heartless.execution.engine import TradingEngine
from heartless.learning.bandit import PRIOR_A, PRIOR_B, AlphaBandit
from heartless.strategy.params import StrategyParams
from synth import synth_symbols

SYM = "BTCUSDT"
T0 = 1_700_000_000_000
ALPHA = "trend_pullback"


def _evidence(b: AlphaBandit, regime) -> float:
    arm = b.arm(ALPHA, regime)
    return arm.a + arm.b - PRIOR_A - PRIOR_B


# --- bandit level -----------------------------------------------------------------------------------------------

def test_default_behaviour_unchanged_without_live_attached():
    b = AlphaBandit([ALPHA], seed=1)
    assert b.live_attached is False
    b.update(ALPHA, Regime.RANGE, 2.0)  # paper
    assert _evidence(b, Regime.RANGE) == pytest.approx(2.0)
    assert _evidence(b, "ALL") == pytest.approx(2.0)
    b2 = AlphaBandit([ALPHA], seed=1)
    b2.update(ALPHA, Regime.RANGE, 2.0, live=True)
    assert _evidence(b2, "ALL") == pytest.approx(3.0)
    assert b2.arm(ALPHA, "ALL").n == 1


def test_paper_evidence_ignored_while_live_attached_and_resumes_after():
    b = AlphaBandit([ALPHA], seed=1)
    b.live_attached = True
    b.update(ALPHA, Regime.RANGE, 2.0)  # paper duplicate: must be a no-op on both arms
    for key in (Regime.RANGE, "ALL"):
        arm = b.arm(ALPHA, key)
        assert (arm.a, arm.b, arm.n, arm.sum_r, arm.updated) == (PRIOR_A, PRIOR_B, 0, 0.0, 0)
    b.update(ALPHA, Regime.RANGE, 2.0, live=True)  # the live outcome is booked, at live weight
    assert _evidence(b, "ALL") == pytest.approx(3.0)
    assert b.arm(ALPHA, "ALL").n == 1
    # live stopped -> paper champion learns again at weight 1.0
    b.live_attached = False
    b.update(ALPHA, "RANGE", -1.0)
    arm = b.arm(ALPHA, "ALL")
    assert arm.n == 2 and arm.b > PRIOR_B


def test_gate_does_not_persist_paper_duplicates_to_the_store():
    class Store:
        def __init__(self):
            self.saved = []

        def load_alpha_stats(self, engine):
            return []

        def save_alpha_stat(self, *row):
            self.saved.append(row)

    st = Store()
    b = AlphaBandit([ALPHA], store=st, engine="shared")
    b.live_attached = True
    b.update(ALPHA, Regime.RANGE, 1.0)
    assert st.saved == []
    b.update(ALPHA, Regime.RANGE, 1.0, live=True)
    assert len(st.saved) == 2  # regime arm + ALL arm, once


# --- engine level: paper champion + live engine share one bandit --------------------------------------------------

class FakeLive(Account):
    """Live-shaped account: everything the close path touches is a no-op."""
    is_paper = False

    async def get_state(self):
        return AccountState(10_000, 10_000, 10_000, 0.0, T0)

    async def get_positions(self):
        return {}

    async def open_orders(self, symbol=None):
        return []

    async def open_algo_orders(self, symbol=None):
        return []

    async def market_order(self, symbol, side, qty, reduce_only=False, client_id=""):
        return OrderResult("1", client_id, "FILLED", qty, 100.0)

    async def limit_order(self, symbol, side, qty, price, post_only=True, reduce_only=False, client_id=""):
        return OrderResult("1", client_id, "NEW")

    async def cancel_order(self, symbol, order_id="", client_id=""):
        return True

    async def query_order(self, symbol, order_id="", client_id=""):
        return OrderResult(order_id, client_id, "FILLED")

    async def place_stop(self, symbol, side, trigger_price, qty=None, close_position=False, client_id=""):
        return "A1"

    async def place_take_profit(self, symbol, side, trigger_price, qty, client_id=""):
        return "A2"

    async def cancel_algo(self, symbol, algo_id):
        return True

    async def cancel_all(self, symbol):
        return None


def _open_position(eng: TradingEngine, qty=1.0, entry=100.0, stop=99.0) -> Position:
    pos = Position(id=f"P-{eng.name}", engine=eng.name, symbol=SYM, side=Side.LONG, qty=qty, entry_price=entry, entry_time=T0,
                   stop=stop, take_profit=None, tp1=None, initial_stop=stop, alpha=ALPHA, alphas=[ALPHA], reason="t",
                   confidence=0.7, regime="RANGE", risk_amount=1.0, r_unit=abs(entry - stop), notional=qty * entry, leverage=5,
                   params_version="v", status=PositionStatus.OPEN, original_qty=qty, filled_qty=qty)
    eng.positions[SYM] = pos
    return pos


def _engines(bandit: AlphaBandit):
    s = Settings(_env_file=None)
    symbols = synth_symbols([SYM])
    pa = PaperAccount("paper", 10_000.0, s.taker_fee, s.maker_fee)
    pa.set_symbols(symbols)
    live = FakeLive()
    live.set_symbols(symbols)
    params = StrategyParams.default()
    paper_eng = TradingEngine("paper", pa, params, s, symbols, bandit=bandit, persist=False)
    live_eng = TradingEngine("live", live, params, s, symbols, bandit=bandit, persist=False)
    assert paper_eng.bandit is live_eng.bandit is paper_eng.ensemble.bandit is live_eng.ensemble.bandit
    return paper_eng, live_eng


async def _close_same_trade(first: TradingEngine, second: TradingEngine, exit_price=102.0) -> float:
    r = None
    for eng in (first, second):
        pos = _open_position(eng)
        await eng._finalize(pos, exit_price, T0 + 60_000, "test")
        assert pos.status is PositionStatus.CLOSED
        r = pos.r_multiple if r is None else r
        assert pos.r_multiple == pytest.approx(r)  # same market outcome in both engines
    return r


@pytest.mark.parametrize("live_first", [False, True])
def test_shared_bandit_books_one_market_event_once_when_live_attached(live_first):
    bandit = AlphaBandit([ALPHA], seed=3)
    bandit.live_attached = True  # what app._start_live must set while the live engine shares the bandit
    paper_eng, live_eng = _engines(bandit)
    order = (live_eng, paper_eng) if live_first else (paper_eng, live_eng)
    r = asyncio.run(_close_same_trade(*order))
    assert r > 0
    for key in (Regime.RANGE, "ALL"):
        arm = bandit.arm(ALPHA, key)
        assert arm.n == 1  # one real event, one observation
        assert arm.a + arm.b - PRIOR_A - PRIOR_B == pytest.approx(1.5 * min(r, 3.0))  # live weight only
    # the posterior is exactly what a lone live engine would have produced
    ref = AlphaBandit([ALPHA], seed=3)
    ref.update(ALPHA, "RANGE", r, live=True)
    assert bandit.arm(ALPHA, "ALL").a == pytest.approx(ref.arm(ALPHA, "ALL").a)
    assert bandit.arm(ALPHA, "ALL").b == pytest.approx(ref.arm(ALPHA, "ALL").b)


def test_paper_only_still_learns_at_paper_weight():
    bandit = AlphaBandit([ALPHA], seed=3)
    paper_eng, _ = _engines(bandit)
    pos = _open_position(paper_eng)
    asyncio.run(paper_eng._finalize(pos, 102.0, T0 + 60_000, "test"))
    arm = bandit.arm(ALPHA, "ALL")
    assert arm.n == 1
    assert arm.a + arm.b - PRIOR_A - PRIOR_B == pytest.approx(1.0 * min(pos.r_multiple, 3.0))

"""P0b signal-level pieces: the exit_on_regime_change / entry_ttl_bars fields on Signal -> Decision -> Position
(including rows persisted before the flag existed), which alphas opt in, and the shared sizing rule used by entries
and re-quotes (RiskManager.size / size_at)."""
from __future__ import annotations

import json
import math

import pytest
from synth import synth_symbols

from heartless.config import Settings
from heartless.core.models import Decision, Position, PositionStatus, Regime, Side, Signal, Ticker
from heartless.learning.bandit import AlphaBandit
from heartless.strategy.base import Alpha, Context
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.strategy.risk import RiskManager
from tests.test_alpha_funding_fade import short_setup, warmed
from tests.test_alpha_invariants import signals  # noqa: F401  (module fixture: every shipped alpha on synthetic data)
from tests.test_alpha_mean_reversion import long_bar, run

SYM = "BTCUSDT"
OPTED_IN = {"mean_reversion", "funding_fade"}  # fades whose premise a strong counter-trend invalidates


def _position(**over) -> Position:
    kw = dict(id="P1", engine="paper", symbol=SYM, side=Side.LONG, qty=1.0, entry_price=100.0, entry_time=1,
              stop=98.0, take_profit=None, tp1=None, initial_stop=98.0, alpha="mean_reversion",
              alphas=["mean_reversion"], reason="t", confidence=0.7, regime="RANGE", risk_amount=2.13, r_unit=2.0,
              notional=100.0, leverage=5, params_version="v", status=PositionStatus.OPEN)
    kw.update(over)
    return Position(**kw)


# --- fields and persistence ------------------------------------------------------------------------------------

def test_defaults_keep_the_old_behaviour():
    sig = Signal(alpha="x", symbol=SYM, side=Side.LONG, confidence=0.7, reason="r", stop=98.0, take_profit=None,
                 tp1=None)
    assert sig.exit_on_regime_change is False and sig.entry_ttl_bars is None
    d = Decision(symbol=SYM, side=Side.LONG, score=0.7, confidence=0.7, alphas=["x"], primary=sig, reason="r",
                 regime=Regime.RANGE)
    assert d.exit_on_regime_change is False and d.entry_ttl_bars is None
    pos = _position(alpha="x")
    assert pos.exit_on_regime_change is False and pos.entry_ttl_bars is None


def test_position_row_round_trip_keeps_both_fields():
    pos = _position(alpha="htf_trend", exit_on_regime_change=True, entry_ttl_bars=3)
    back = Position.from_row(json.loads(json.dumps(pos.to_row(), default=str)))
    assert back.exit_on_regime_change is True and back.entry_ttl_bars == 3
    off = Position.from_row(_position(alpha="mean_reversion", exit_on_regime_change=False).to_row())
    assert off.exit_on_regime_change is False  # an explicit value always wins over the legacy name rule


@pytest.mark.parametrize("alpha,expected", [("mean_reversion", True), ("funding_fade", True), ("htf_trend", False),
                                            ("adopted", False)])
def test_rows_persisted_before_the_flag_existed_keep_their_old_exit(alpha, expected):
    row = _position(alpha=alpha).to_row()
    del row["exit_on_regime_change"], row["entry_ttl_bars"]
    pos = Position.from_row(row)
    assert pos.exit_on_regime_change is expected and pos.entry_ttl_bars is None


class _Fixed(Alpha):
    """An alpha that emits one prepared signal (registered name, so params and affinity exist)."""

    def __init__(self, name: str, sig: Signal):
        self.name = name
        self.timeframe = "1h"
        self._sig = sig

    def evaluate(self, view, ctx, p):
        return self._sig


class _View:
    price = 100.0

    def closed(self, tf: str) -> bool:
        return True


@pytest.mark.parametrize("flag,ttl", [(True, 4), (False, None)])
def test_ensemble_takes_both_fields_from_the_primary_signal(flag, ttl):
    sig = Signal(alpha="htf_trend", symbol=SYM, side=Side.LONG, confidence=0.9, reason="r", stop=98.0,
                 take_profit=104.0, tp1=None, tags={"ref_price": 100.0}, exit_on_regime_change=flag, entry_ttl_bars=ttl)
    params = StrategyParams.default()
    ens = Ensemble(params, AlphaBandit(list(params.alphas), seed=1), alphas=[_Fixed("htf_trend", sig)],
                   only_alpha="htf_trend")
    info = synth_symbols([SYM])[SYM]
    ctx = Context(symbol=SYM, info=info, ticker=Ticker(SYM, bid=99.99, ask=100.01, mark=100.0), regime=Regime.RANGE)
    d = ens.decide(_View(), ctx)
    assert d is not None and d.primary is sig
    assert d.exit_on_regime_change is flag and d.entry_ttl_bars == ttl


# --- which alphas opt in ---------------------------------------------------------------------------------------

def test_mean_reversion_and_funding_fade_opt_into_the_regime_exit():
    mr = run(long_bar())
    assert mr is not None and mr.exit_on_regime_change is True and mr.entry_ttl_bars is None
    a, now = warmed()
    ff = short_setup(a, now)
    assert ff is not None and ff.exit_on_regime_change is True and ff.entry_ttl_bars is None


def test_no_other_shipped_alpha_opts_in_and_none_sets_a_ttl(signals):  # noqa: F811
    fired = 0
    for name, items in signals.items():
        for s, _ in items:
            fired += 1
            assert s.exit_on_regime_change is (name in OPTED_IN), name
            assert s.entry_ttl_bars is None, name
    assert fired > 0


# --- the shared sizing rule ------------------------------------------------------------------------------------

def _rm(**env) -> RiskManager:
    return RiskManager(Settings(_env_file=None, **env))


def _decision(stop: float, size_mult: float = 1.0) -> Decision:
    sig = Signal(alpha="x", symbol=SYM, side=Side.LONG, confidence=0.7, reason="r", stop=stop, take_profit=None,
                 tp1=None)
    return Decision(symbol=SYM, side=Side.LONG, score=0.7, confidence=0.7, alphas=["x"], primary=sig, reason="r",
                    regime=Regime.RANGE, size_mult=size_mult)


def test_size_is_size_at_with_the_decision_budget():
    rm, info = _rm(), synth_symbols([SYM])[SYM]
    for size_mult in (0.36, 1.0, 1.4, 5.0):
        a = rm.size(_decision(98.0, size_mult), 10_000.0, info, 100.0, open_notional=1_000.0)
        b = rm.size_at(98.0, 100.0, 10_000.0, info, rm.budget_pct(size_mult), open_notional=1_000.0)
        assert a == b
    assert rm.budget_pct(5.0) == 1.25 and rm.budget_pct(0.01) == 0.1  # clamped to [0.1, MAX_RISK_PER_TRADE_PCT]
    # reference values of the unchanged rule: 0.5% of 10k over (2 + 13 bp of 100) per unit
    r = rm.size(_decision(98.0), 10_000.0, info, 100.0)
    assert r.qty == pytest.approx(math.floor(50.0 / 2.13 * 1000) / 1000) and r.risk_pct == 0.5
    assert r.risk_amount == pytest.approx(r.qty * 2.13)


def test_size_at_lets_the_filled_part_consume_the_budget_and_the_caps():
    rm, info = _rm(), synth_symbols([SYM])[SYM]
    whole = rm.size_at(98.0, 100.0, 10_000.0, info, 0.5)
    part = rm.size_at(98.0, 101.0, 10_000.0, info, 0.5, filled_risk=20.0, filled_notional=940.0)
    assert 0 < part.qty and 20.0 + part.risk_amount <= 50.0 + 1e-9
    assert 20.0 + (part.qty + 0.001) * (3.0 + 101.0 * 0.0013) > 50.0  # the largest remainder that fits
    assert part.qty < whole.qty
    spent = rm.size_at(98.0, 101.0, 10_000.0, info, 0.5, filled_risk=50.0, lift_to_min=False)
    assert spent.qty == 0 and spent.reason == "below min notional"
    # notional cap (MAX_POSITION_LEVERAGE 5 -> 50k) counts the filled notional
    tight = rm.size_at(99.9, 100.0, 10_000.0, info, 0.5, filled_notional=49_000.0)
    assert tight.qty * 100.0 <= 1_000.0 + 1e-6


def test_size_at_never_lifts_to_the_exchange_minimum_without_permission():
    info = synth_symbols([SYM])[SYM]
    info.min_notional = 500.0  # 5 units at 100, while the 5 USDT budget (0.5% of 1k) fits only 2.347
    rm = _rm()
    lifted = rm.size_at(98.0, 100.0, 1_000.0, info, 0.5)
    assert lifted.qty == pytest.approx(5.0)  # a new entry may exceed the budget (bounded by 2.5x and the hard cap)
    assert lifted.risk_pct == pytest.approx(5.0 * 2.13 / 1_000.0 * 100)
    strict = rm.size_at(98.0, 100.0, 1_000.0, info, 0.5, lift_to_min=False)
    assert strict.qty == 0 and strict.reason == "below min notional"


def test_entries_blocked_is_the_account_wide_part_of_can_open():
    rm, t = _rm(), 1_700_000_000_000
    assert rm.entries_blocked(t) == "" and rm.can_open(_decision(98.0), [], t, 10_000.0) == (True, "")
    rm.register_market_shock(t, 15)
    assert rm.entries_blocked(t) == "market shock cool-off" == rm.can_open(_decision(98.0), [], t, 10_000.0)[1]
    assert rm.entries_blocked(t + 15 * 60_000) == ""
    rm.state.halted_until, rm.state.halt_reason = t + 1, "daily"
    assert rm.entries_blocked(t) == "halted:daily" == rm.can_open(_decision(98.0), [], t, 10_000.0)[1]
    rm.pause("owner")
    assert rm.entries_blocked(t) == "paused" == rm.can_open(_decision(98.0), [], t, 10_000.0)[1]

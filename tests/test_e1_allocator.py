"""P3 portfolio allocator (heartless/execution/allocator.py): signal batching per bar close, ranking by expected value,
risk-budget limits with BTC-beta weighting, and the same code path in the joint backtester and the live app."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from synth import synth_candles, synth_symbols

from heartless.config import Settings
from heartless.core.bus import EventBus
from heartless.core.models import Decision, EntryStyle, Position, PositionStatus, Regime, Side, Signal, Ticker
from heartless.exchange.paper import PaperAccount
from heartless.execution.allocator import (BetaEstimator, Candidate, PortfolioAllocator, SAME_REASON, TOTAL_REASON,
                                           hourly_closes)
from heartless.execution.engine import TradingEngine
from heartless.learning.backtester import Backtester
from heartless.strategy.base import Alpha, Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_HOUR, MS_MINUTE

T0 = 1_700_006_400_000
H = MS_HOUR


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


# --- beta -------------------------------------------------------------------------------------------------------------

def _walk(n: int, seed: int, beta: float | None = None, base: np.ndarray | None = None, noise: float = 0.002):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.004, n) if base is None else beta * base + rng.normal(0, noise, n)
    return r, 100 * np.exp(np.cumsum(r))


def test_beta_is_the_slope_on_btc_hourly_returns_and_is_bounded():
    n = 720
    rb, pb = _walk(n, 1)
    est = BetaEstimator(hours=720)
    hours = [T0 + i * H for i in range(n)]
    est.prime("BTCUSDT", hours, pb)
    for name, b in (("ALT", 1.6), ("HI", 4.0), ("NEG", -1.0)):
        _, p = _walk(n, 2, beta=b, base=rb)
        est.prime(name, hours, p)
    assert est.raw_beta("ALT") == pytest.approx(1.6, abs=0.1)
    assert est.beta("ALT") == pytest.approx(est.raw_beta("ALT"))
    assert est.beta("HI") == 2.5 and est.beta("NEG") == 0.5  # bounded
    assert est.beta("BTCUSDT") == 1.0 and est.beta("UNKNOWN") == 1.0


def test_beta_needs_a_week_of_aligned_hours_and_observe_is_idempotent():
    rb, pb = _walk(300, 3)
    _, pa = _walk(300, 4, beta=2.0, base=rb)
    est = BetaEstimator(hours=720)
    for i in range(100):
        est.observe("BTCUSDT", T0 + i * H, pb[i])
        est.observe("ALT", T0 + i * H, pa[i])
        est.observe("ALT", T0 + i * H, 1e9)  # a second engine reporting the same hour: ignored
    assert est.beta("ALT") == 1.0  # < 168 aligned returns
    for i in range(100, 300):
        est.observe("BTCUSDT", T0 + i * H, pb[i])
        est.observe("ALT", T0 + i * H, pa[i])
    assert est.beta("ALT") == pytest.approx(2.0, abs=0.15)
    # priming older hours afterwards merges them in front (a fresh process seeds from the store after its first bars)
    late = BetaEstimator(hours=720)
    late.observe("ALT", T0 + 299 * H, pa[299])
    late.prime("ALT", [T0 + i * H for i in range(299)], pa[:299])
    assert [t for t, _ in late._bars["ALT"]] == [T0 + i * H for i in range(300)]


def test_hourly_closes_take_the_last_minute_of_each_hour():
    ot = np.arange(T0, T0 + 3 * H, MS_MINUTE, dtype=np.int64)
    cl = np.arange(len(ot), dtype=float) + 1
    hours, closes = hourly_closes(ot, cl)
    assert hours == [T0, T0 + H, T0 + 2 * H] and closes == [60.0, 120.0, 180.0]
    assert hourly_closes(ot, cl, before=T0 + 2 * H)[0] == [T0, T0 + H]
    assert hourly_closes(ot, cl, limit=1)[0] == [T0 + 2 * H]


# --- ranking ----------------------------------------------------------------------------------------------------------

def _cand(sym: str, alpha: str = "htf_trend", conf: float = 0.65, regime: Regime = Regime.RANGE, er=None) -> Candidate:
    sig = Signal(alpha=alpha, symbol=sym, side=Side.LONG, confidence=conf, reason="t", stop=90.0, take_profit=None,
                 tp1=None)
    d = Decision(symbol=sym, side=Side.LONG, score=conf, confidence=conf, alphas=[alpha], primary=sig, reason="t",
                 regime=regime)
    return Candidate(d, None, None, meta=None if er is None else ([], True, 1.0, er))


def test_rank_expected_r_first_then_bandit_score_then_symbol(monkeypatch):
    from heartless.learning.bandit import AlphaBandit
    from heartless.strategy import evidence

    monkeypatch.setattr(evidence, "verdict", lambda a: "passed" if a == "good" else "failed")
    monkeypatch.setattr(evidence, "expected_r", lambda a: 0.12 if a == "good" else 0.0)
    bandit = AlphaBandit(["good", "htf_trend", "momentum_burst"], seed=1)
    for _ in range(20):
        bandit.update("momentum_burst", "RANGE", 1.5)  # a winning record: posterior mean above neutral
    alloc = PortfolioAllocator(settings())
    ranked = alloc.rank([_cand("ZZZUSDT"), _cand("AAAUSDT"), _cand("MMMUSDT", alpha="good"),
                         _cand("BBBUSDT", alpha="momentum_burst"), _cand("CCCUSDT", er=0.30)], bandit)
    assert [c.symbol for c in ranked] == ["CCCUSDT", "MMMUSDT", "BBBUSDT", "AAAUSDT", "ZZZUSDT"]
    assert [c.source for c in ranked] == ["meta", "evidence", "bandit", "bandit", "bandit"]
    assert ranked[1].value == 0.12 and ranked[2].value > ranked[3].value == pytest.approx(0.5 * 0.65)
    # equal values: the lower BTC beta first (less of the side's budget), then the symbol
    alloc.beta.beta = lambda s: {"ZZZUSDT": 0.8}.get(s, 1.2)  # type: ignore[method-assign]
    ranked = alloc.rank([_cand("AAAUSDT"), _cand("ZZZUSDT"), _cand("BBBUSDT")], bandit)
    assert [c.symbol for c in ranked] == ["ZZZUSDT", "AAAUSDT", "BBBUSDT"]


# --- limits -----------------------------------------------------------------------------------------------------------

def _pos(sym: str, side: Side, risk: float, status=PositionStatus.OPEN, stop_frac: float = 1.0, qty_frac: float = 1.0):
    entry = 100.0
    stop = entry - side.sign * 2.0 * stop_frac
    return Position(id=sym, engine="e", symbol=sym, side=side, qty=1.0 * qty_frac, entry_price=entry, entry_time=0,
                    stop=stop, take_profit=None, tp1=None, initial_stop=entry - side.sign * 2.0, alpha="a", alphas=["a"],
                    reason="", confidence=0.6, regime="RANGE", risk_amount=risk, r_unit=2.0, notional=100.0,
                    leverage=3, params_version="v", status=status, filled_qty=1.0, original_qty=1.0)


def test_open_risk_follows_the_current_stop_and_the_part_still_held():
    a = PortfolioAllocator
    assert a.open_risk(_pos("A", Side.LONG, 50.0)) == 50.0
    assert a.open_risk(_pos("A", Side.LONG, 50.0, stop_frac=0.5)) == 25.0  # trailed halfway to breakeven
    assert a.open_risk(_pos("A", Side.LONG, 50.0, stop_frac=-0.1)) == 0.0  # stop beyond breakeven
    assert a.open_risk(_pos("A", Side.SHORT, 50.0, qty_frac=0.5)) == 25.0  # half taken at TP1
    assert a.open_risk(_pos("A", Side.LONG, 50.0, status=PositionStatus.PENDING)) == 50.0
    assert a.open_risk(_pos("A", Side.LONG, 50.0, status=PositionStatus.CLOSED)) == 0.0


def test_total_and_beta_weighted_same_direction_budgets():
    alloc = PortfolioAllocator(settings(MAX_TOTAL_RISK_PCT=2.5, MAX_SAME_DIRECTION_RISK_PCT=1.5))
    alloc.beta.beta = lambda s: {"ALT": 2.0}.get(s, 1.0)  # type: ignore[method-assign]
    eq = 10_000.0
    book = [_pos("BTCUSDT", Side.LONG, 50.0), _pos("ETHUSDT", Side.LONG, 50.0)]  # 1.0% long at beta 1
    assert alloc.check("SOLUSDT", Side.LONG, 50.0, book, eq) == (True, "")  # 1.5% long
    assert alloc.check("ALT", Side.LONG, 50.0, book, eq) == (False, SAME_REASON)  # 1.0 + 0.5 x 2.0 = 2.0% > 1.5%
    assert alloc.check("ALT", Side.SHORT, 50.0, book, eq) == (True, "")  # the other side has its own budget
    book += [_pos("XRPUSDT", Side.SHORT, 50.0), _pos("ADAUSDT", Side.SHORT, 50.0)]  # 2.0% total
    assert alloc.check("DOGEUSDT", Side.SHORT, 50.0, book, eq) == (True, "")  # 2.5% total
    assert alloc.check("DOGEUSDT", Side.SHORT, 60.0, book, eq) == (False, TOTAL_REASON)
    ex = alloc.exposure(book, eq)
    assert ex["total_pct"] == pytest.approx(2.0) and ex["long_pct"] == pytest.approx(1.0)


# --- the engine: batching -------------------------------------------------------------------------------------------

class _View:
    frames: dict = {}

    def __init__(self, price):
        self.price = price

    def ready(self):
        return True

    def closed(self, tf):
        return tf == "1h"

    def tf(self, name):
        return SimpleNamespace(ok=False, v=lambda *a: float("nan"), open_time=0)


class _Always(Alpha):
    name = "htf_trend"
    timeframe = "1h"
    param_specs = []

    def __init__(self, conf: dict[str, float]):
        self.conf = conf

    def evaluate(self, view, ctx, p):
        px = view.price
        return Signal(alpha=self.name, symbol=ctx.symbol, side=Side.LONG, confidence=self.conf.get(ctx.symbol, 0.7),
                      reason="t", stop=px - 2.0, take_profit=None, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=60, atr=2.0, timeframe="1h", tags={"ref_price": px})


SYMS = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]


def _engine(alloc_on: bool, conf: dict[str, float], **kw):
    clock = SimpleNamespace(t=T0)
    acc = PaperAccount(name="p", initial_balance=10_000.0, clock=lambda: clock.t)
    syms = synth_symbols(SYMS)
    acc.set_symbols(syms)
    params = StrategyParams.default()
    params.enabled["htf_trend"] = True
    eng = TradingEngine("paper", acc, params, settings(PORTFOLIO_ALLOCATOR=alloc_on, MAX_POSITIONS=2, **kw), syms,
                        bus=EventBus(), clock=lambda: clock.t, persist=False)
    eng.ensemble.alphas = [_Always(conf)]
    return eng, acc


async def _bar_all(eng, acc):
    for s in SYMS:
        tk = Ticker(s, bid=99.99, ask=100.01, mark=100.0, last=100.0, ts=T0)
        await acc.on_ticker(tk)
        ctx = Context(symbol=s, info=eng.symbols[s], ticker=tk, regime=Regime.RANGE, now=T0)
        await eng.on_bar(_View(100.0), ctx)


async def test_fifo_lets_the_first_processed_symbols_take_the_slots():
    eng, acc = _engine(False, {"DDDUSDT": 0.95, "CCCUSDT": 0.9})
    await _bar_all(eng, acc)
    assert sorted(eng.positions) == ["AAAUSDT", "BBBUSDT"] and eng.queued_entries == 0
    assert await eng.flush_entries() == 0


async def test_allocator_queues_a_bar_close_and_gives_the_slots_to_the_best_ranked():
    eng, acc = _engine(True, {"DDDUSDT": 0.95, "CCCUSDT": 0.9})
    await _bar_all(eng, acc)
    assert eng.positions == {} and eng.queued_entries == 4  # nothing opens before the whole close is seen
    assert await eng.flush_entries() == 2
    assert sorted(eng.positions) == ["CCCUSDT", "DDDUSDT"]  # bandit mean x confidence, the higher confidence first
    assert eng.stats.skipped.get("max positions") == 2 and eng.queued_entries == 0
    snap = eng.snapshot()
    assert snap["risk_budget"]["total_pct"] == pytest.approx(
        sum(p.risk_amount for p in eng.positions.values()) / 100.0)


async def test_allocator_risk_budget_skips_what_does_not_fit_and_tries_the_next():
    # 0.5% risk per trade, total budget 0.9%: only one position fits even though MAX_POSITIONS allows two
    eng, acc = _engine(True, {"DDDUSDT": 0.95, "CCCUSDT": 0.9}, MAX_TOTAL_RISK_PCT=0.9)
    await _bar_all(eng, acc)
    assert await eng.flush_entries() == 1
    assert list(eng.positions) == ["DDDUSDT"] and eng.stats.skipped.get(TOTAL_REASON) == 3


# --- the joint backtester runs the same code ------------------------------------------------------------------------

class _Hourly(Alpha):
    name = "htf_trend"
    timeframe = "1h"
    param_specs = []

    def evaluate(self, view, ctx, p):
        h = view.tf("1h")
        close, atr = h.v("close"), h.v("atr")
        if not (atr > 0) or h.v("close", 1) >= close:
            return None
        return Signal(alpha=self.name, symbol=ctx.symbol, side=Side.LONG, confidence=0.7, reason="t",
                      stop=close - 1.5 * atr, take_profit=close + 2.0 * atr, tp1=None, entry_style=EntryStyle.MARKET,
                      max_hold_bars=180, atr=atr, timeframe="1h", tags={"ref_price": close})


def _bt(monkeypatch, syms, **kw):
    import heartless.strategy.ensemble as E

    monkeypatch.setattr(E, "ALL_ALPHAS", [_Hourly()])
    s = settings(**kw)
    candles = {name: synth_candles(4000, seed=7 + i, price=100.0 * (i + 1)) for i, name in enumerate(syms)}
    return Backtester(s, synth_symbols(syms), candles)


def test_single_symbol_backtest_is_identical_with_the_allocator_when_its_budgets_never_bind(monkeypatch):
    """With one symbol a batch is one candidate: an allocator whose budgets never bind must reproduce FIFO trade for
    trade, i.e. queueing and flushing at the step end changes nothing about when or how an entry is sent."""
    params = StrategyParams.default()
    fifo = _bt(monkeypatch, ["BTCUSDT"]).run(params, only_alpha="htf_trend")
    batched = _bt(monkeypatch, ["BTCUSDT"], PORTFOLIO_ALLOCATOR=True, MAX_TOTAL_RISK_PCT=100.0,
                  MAX_SAME_DIRECTION_RISK_PCT=100.0).run(params, only_alpha="htf_trend")
    assert fifo.stats["n"] >= 5

    def strip(trades):
        return [{k: v for k, v in t.items() if k != "position_id"} for t in trades]  # ids are random

    assert strip(batched.trades) == strip(fifo.trades) and batched.final_equity == fifo.final_equity


def test_joint_backtest_with_the_allocator_flushes_every_step_and_respects_the_budgets(monkeypatch):
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    params = StrategyParams.default()
    seen: list[float] = []
    orig = PortfolioAllocator.check

    def check(self, symbol, side, risk_amount, positions, equity):
        ok, why = orig(self, symbol, side, risk_amount, positions, equity)
        if ok:
            ex = self.exposure(positions, equity)
            seen.append(ex["total_pct"] + risk_amount / equity * 100)
        return ok, why

    monkeypatch.setattr(PortfolioAllocator, "check", check)
    bt = _bt(monkeypatch, syms, PORTFOLIO_ALLOCATOR=True, MAX_TOTAL_RISK_PCT=1.0)
    res = bt.run(params, only_alpha="htf_trend")
    assert res.stats["n"] > 0 and seen and max(seen) <= 1.0 + 1e-9
    assert res.skipped.get(TOTAL_REASON, 0) > 0
    fifo = _bt(monkeypatch, syms).run(params, only_alpha="htf_trend")
    assert TOTAL_REASON not in fifo.skipped


# --- the live app: batching wiring ----------------------------------------------------------------------------------

class _Eng:
    def __init__(self):
        self.allocator = object()
        self.flushed = 0
        self.queued_entries = 1
        self.name = "e"

    async def flush_entries(self):
        self.flushed += 1
        return 0


def _app():
    from heartless.app import Heartless

    app = Heartless.__new__(Heartless)
    app.engines = {"paper": _Eng()}
    app.views = {s: None for s in ("AAAUSDT", "BBBUSDT", "CCCUSDT")}
    app._prime_beta = lambda eng: None
    return app


def test_app_flushes_once_every_traded_symbol_reported_the_close():
    app, t = _app(), T0 + MS_MINUTE - 1
    eng = app.engines["paper"]
    asyncio.run(app._entry_batch_done("AAAUSDT", t))
    asyncio.run(app._entry_batch_done("BBBUSDT", t))
    assert eng.flushed == 0 and app._batch_t == t
    asyncio.run(app._entry_batch_done("BBBUSDT", t - MS_MINUTE))  # an older (gap-filled) bar rides along
    asyncio.run(app._entry_batch_done("CCCUSDT", t))
    assert eng.flushed == 1 and app._batch_t is None


def test_app_flushes_an_incomplete_batch_on_a_later_close_or_after_the_grace():
    from heartless.app import ENTRY_BATCH_GRACE_MS

    app, t = _app(), T0 + MS_MINUTE - 1
    eng = app.engines["paper"]
    asyncio.run(app._entry_batch_done("AAAUSDT", t))
    asyncio.run(app._entry_batch_roll(t + MS_MINUTE))  # CCC/BBB never came: the next close admits t's batch first
    assert eng.flushed == 1 and app._batch_t is None
    asyncio.run(app._entry_batch_done("AAAUSDT", t + MS_MINUTE))
    asyncio.run(app._entry_batch_due(t + MS_MINUTE + ENTRY_BATCH_GRACE_MS))
    assert eng.flushed == 1
    asyncio.run(app._entry_batch_due(t + MS_MINUTE + ENTRY_BATCH_GRACE_MS + 1))
    assert eng.flushed == 2


def test_app_batching_is_inert_with_the_allocator_off():
    app = _app()
    app.engines["paper"].allocator = None
    asyncio.run(app._entry_batch_done("AAAUSDT", T0))
    assert app._batch_t is None


def test_full_app_replay_with_the_allocator_passes_every_invariant(tmp_path):
    """The real orchestrator on the virtual-time loop (tests/test_p5_replay.py) with PORTFOLIO_ALLOCATOR on: entries go
    through the per-close batches (every traded symbol's bar, then flush) and every minute's invariants hold."""
    from heartless.data import features
    from heartless.replay.runner import ReplayConfig, run_replay
    from tests.test_p5_replay import NO_LEARNING, SEED_DAYS, SMALL_FRAMES, START, build_source, busy_params

    flushed: list[int] = []
    orig = TradingEngine.flush_entries

    async def counting(self):
        n = await orig(self)
        flushed.append(n)
        return n

    with pytest.MonkeyPatch.context() as mp:
        for tf, n in SMALL_FRAMES.items():
            mp.setitem(features.LIVE_WINDOW, tf, n)
        mp.setattr(TradingEngine, "flush_entries", counting)
        cfg = ReplayConfig(source_db=build_source(tmp_path / "src.db", 1), start_ms=START,
                           end_ms=START + MS_HOUR * 24 - 1, work_dir=str(tmp_path / "bot"), seed_days=SEED_DAYS,
                           params=busy_params(), settings={**NO_LEARNING, "PORTFOLIO_ALLOCATOR": True})
        s = run_replay(cfg)
    assert s["passed"] and s["invariants"]["violations_total"] == 0, s["report"]
    assert flushed and sum(flushed) == s["trades"]["opened"]["paper"] >= 1

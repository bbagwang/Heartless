"""Regression tests for heartless/learning/research.py (fix round 14).

1. graduation_status() must see the whole 14-day equity window, not just the newest 5000 one-minute rows.
2. A crashed research worker (BrokenProcessPool) must not permanently break the self-improvement loop.
3. The promotion drawdown gate must be scale-free (challenger and champion have different equity bases).
"""
import asyncio
from concurrent.futures.process import BrokenProcessPool

import pytest

import heartless.learning.research as research_mod
from heartless.config import Settings
from heartless.core.models import TradeRecord
from heartless.core.store import Store
from heartless.execution.stats import objective, summarize
from heartless.learning.research import ChallengerSlot, ResearchManager
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, now_ms

MS_MIN = 60_000


class _StubApp:
    """Minimal stand-in for heartless.app.Heartless as seen by ResearchManager."""

    def __init__(self, settings: Settings, store: Store):
        self.s = settings
        self.store = store
        self.mode = "paper"
        self.engines: dict = {}
        self.challengers: list[ChallengerSlot] = []
        self.symbols: dict = {}
        self.universe: list[str] = []
        self.params = StrategyParams.default()
        self.events: list[tuple[str, dict]] = []
        self.promoted: list = []
        self.retired: list = []

    async def emit(self, kind: str, payload: dict) -> None:
        self.events.append((kind, payload))

    async def promote(self, slot, st_ch, st_champ) -> None:
        self.promoted.append((slot.name, st_ch, st_champ))

    async def retire_challenger(self, slot, reason: str, silent: bool = False) -> None:
        self.retired.append((slot.name, reason))


def _env(tmp_path, **overrides):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), TELEGRAM_BOT_TOKEN="", WEB_ENABLED=False,
                        **overrides)
    store = Store(tmp_path / "heartless.db")
    app = _StubApp(settings, store)
    return app, ResearchManager(app)


def _trade(engine: str, idx: int, r: float, risk: float, exit_time: int) -> TradeRecord:
    pnl = r * risk
    return TradeRecord(position_id=f"{engine}-{idx}", engine=engine, symbol="BTCUSDT", side="LONG", alpha="trend_pullback",
                       alphas=["trend_pullback"], regime="trend", entry_time=exit_time - 5 * MS_MIN, exit_time=exit_time,
                       entry_price=100.0, exit_price=100.0 + pnl / 10, qty=10.0, notional=1000.0 * risk / 100.0, pnl=pnl,
                       gross=pnl, fees=0.0, funding=0.0, r_multiple=r, risk_amount=risk, exit_reason="tp", confidence=0.6,
                       params_version="v0", bars_held=5, max_fav_r=max(r, 0.0), max_adv_r=min(r, 0.0))


# --- 1. graduation drawdown gate sees the whole window -----------------------------------------------

def test_graduation_drawdown_sees_full_14_day_window(tmp_path):
    app, rm = _env(tmp_path)
    now = now_ms()
    n = 14 * 1440  # one persisted equity row per minute for 14 days
    start = now - n * MS_MIN + MS_MIN
    app.store.execute("BEGIN")
    for i in range(n):
        eq = 10_000.0
        if 3 * 1440 <= i < 4 * 1440:  # 30% drawdown on day 3 (~11 days ago), fully recovered afterwards
            eq = 7_000.0
        app.store.save_equity("paper", start + i * MS_MIN, eq, eq)
    app.store.execute("COMMIT")

    gs = rm.graduation_status()
    assert gs["max_dd_pct"] == pytest.approx(30.0)
    assert gs["ok"] is False  # 30% >> GRADUATION_MAX_DD_PCT default 8%


def test_graduation_drawdown_still_flat_without_drawdown(tmp_path):
    app, rm = _env(tmp_path)
    now = now_ms()
    app.store.execute("BEGIN")
    for i in range(2 * 1440):
        app.store.save_equity("paper", now - i * MS_MIN, 10_000.0 - i, 10_000.0 - i)  # rising monotonically in time
    app.store.execute("COMMIT")
    assert rm.graduation_status()["max_dd_pct"] == pytest.approx(0.0)


# --- 2. crashed worker does not permanently break the loop --------------------------------------------

class _DummyPool:
    def __init__(self):
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


def test_broken_process_pool_is_discarded_and_recreated(tmp_path, monkeypatch):
    app, rm = _env(tmp_path)

    async def run():
        loop = asyncio.get_running_loop()
        dummy = _DummyPool()
        rm.pool = dummy

        def boom(executor, fn, *args):
            fut = loop.create_future()
            fut.set_exception(BrokenProcessPool("A process in the process pool was terminated abruptly"))
            return fut

        monkeypatch.setattr(loop, "run_in_executor", boom)
        assert await rm.cycle(force=True) is None
        assert rm.running is False
        assert rm.pool is None, "broken executor must be discarded"
        assert dummy.shutdown_calls == [(False, True)]
        assert rm.research_failures == 1
        assert "terminated abruptly" in rm.last_error
        assert any(kind == "error" for kind, _ in app.events)

        # the next cycle lazily builds a fresh executor and runs on it
        created: list = []

        class _FakeExecutor:
            def __init__(self, *a, **k):
                created.append(self)

            def shutdown(self, wait=True, cancel_futures=False):
                pass

        monkeypatch.setattr(research_mod, "ProcessPoolExecutor", _FakeExecutor)

        def ok(executor, fn, *args):
            assert executor is created[-1]
            fut = loop.create_future()
            fut.set_result({"error": "no candles"})
            return fut

        monkeypatch.setattr(loop, "run_in_executor", ok)
        res = await rm.cycle(force=True)
        assert res == {"error": "no candles"}
        assert len(created) == 1 and rm.pool is created[0]
        assert rm.research_failures == 1

        rm.shutdown()
        assert rm.pool is None

    asyncio.run(run())


# --- 3. promotion drawdown gate is scale-free --------------------------------------------------------

def _install(app: _StubApp, champ_rs: list[float], ch_rs: list[float], champ_risk: float, ch_risk: float) -> ChallengerSlot:
    now = now_ms()
    started = now - 2 * MS_DAY
    app.store.delete_trades("paper")
    app.store.delete_trades("challenger-1")
    for i, r in enumerate(champ_rs):
        app.store.save_trade(_trade("paper", i, r, champ_risk, started + (i + 1) * 30 * MS_MIN))
    for i, r in enumerate(ch_rs):
        app.store.save_trade(_trade("challenger-1", i, r, ch_risk, started + (i + 1) * 30 * MS_MIN))
    slot = ChallengerSlot(name="challenger-1", params=StrategyParams.default(), started=started, source="trend_pullback")
    app.challengers = [slot]
    app.engines = {"paper": object(), "challenger-1": object()}
    app.promoted.clear()
    app.retired.clear()
    return slot


def test_promotion_drawdown_gate_is_scale_free(tmp_path):
    app, rm = _env(tmp_path)
    champ_rs = [1.2, -1.0] * 15  # avgR +0.1, R-drawdown 1.0
    ch_rs = [2.0, -1.0] * 15  # avgR +0.5, R-drawdown 1.0: a genuinely better challenger

    # sanity: every non-drawdown gate passes, so the drawdown gate alone decides
    st_ch, st_champ = summarize([{"pnl": r, "r_multiple": r, "exit_time": i} for i, r in enumerate(ch_rs)]), \
        summarize([{"pnl": r, "r_multiple": r, "exit_time": i} for i, r in enumerate(champ_rs)])
    assert objective(st_ch, 10) > objective(st_champ, 10) + 0.5 and st_ch["avg_r"] > st_champ["avg_r"] + 0.05

    # champion wallet grown to 2x the challenger's equity base: identical R, double the USDT drawdown
    _install(app, champ_rs, ch_rs, champ_risk=200.0, ch_risk=100.0)
    asyncio.run(rm.evaluate_challengers(force=True))
    assert [p[0] for p in app.promoted] == ["challenger-1"]

    # champion in drawdown / smaller than the challenger: the USDT comparison used to reject this (200 > 100 * 1.2)
    _install(app, champ_rs, ch_rs, champ_risk=100.0, ch_risk=200.0)
    asyncio.run(rm.evaluate_challengers(force=True))
    assert [p[0] for p in app.promoted] == ["challenger-1"], "same R-sequences must give the same verdict at any scale"
    assert app.promoted[0][1]["max_dd_r"] == pytest.approx(1.0)
    assert app.promoted[0][2]["max_dd_r"] == pytest.approx(1.0)


def test_promotion_rejects_worse_r_drawdown_hidden_by_smaller_equity_base(tmp_path):
    app, rm = _env(tmp_path)
    champ_rs = [1.2, -1.0] * 15  # R-drawdown 1.0
    # better average but a 4R losing streak: R-drawdown 4.0 > 1.2 x champion
    ch_rs = [2.0, -1.0] * 12 + [-1.0, -1.0, -1.0] + [2.0, 2.0, 2.0]
    st_ch = summarize([{"pnl": r, "r_multiple": r, "exit_time": i} for i, r in enumerate(ch_rs)])
    st_champ = summarize([{"pnl": r, "r_multiple": r, "exit_time": i} for i, r in enumerate(champ_rs)])
    assert objective(st_ch, 10) > objective(st_champ, 10) + 0.5 and st_ch["avg_r"] > st_champ["avg_r"] + 0.05

    # champion sized 4x larger: in USDT the challenger's 400 drawdown is under 1.2 x 400, which used to pass
    _install(app, champ_rs, ch_rs, champ_risk=400.0, ch_risk=100.0)
    asyncio.run(rm.evaluate_challengers(force=True))
    assert app.promoted == []
    assert app.retired == []  # not retired either: enough trades but not 2x promotion_min_trades

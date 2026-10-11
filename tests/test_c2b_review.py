"""C2-B review fixes. (1) A replay keeps its live context store in its work dir, whatever CONTEXT_DB the environment
names (the replayed bot appends to that store and prunes it at the virtual time). (2) A research cycle due right at
start-up waits for the start-up fill of the live context store, so its backtests do not read the hours still missing.
(3) An older stored challenger records the base it was given at its first load, so a restart after a champion change
does not fold that change's old values into its diff. (4) A kill-switch trip on an alpha a challenger inherited from its
base does not count as a judged trial of the challenger's change. Synthetic data only."""
from __future__ import annotations

import asyncio

from heartless.app import Heartless, live_context_path
from heartless.learning import research as R
from heartless.learning.research import ResearchManager
from heartless.replay.runner import ReplayConfig, _settings
from heartless.util.timeutil import MS_DAY, MS_HOUR, now_ms
from tests.test_c2b_rebase import _App
from tests.test_fix_14_research import _DummyPool, _env
from tests.test_p4_app import _boot, _make, _rec


# --- (1) the replay's context store --------------------------------------------------------------------------------------
def test_a_replay_keeps_its_live_context_store_in_its_work_dir(tmp_path, monkeypatch):
    offline = tmp_path / "longhist" / "context.db"
    monkeypatch.setenv("CONTEXT_DB", str(offline))  # as the lab commands of docs/RESEARCH_TOOLS.md export it
    work = tmp_path / "work"
    for extra in ({}, {"CONTEXT_DB": str(offline)}):
        s = _settings(ReplayConfig(source_db="unused", start_ms=0, end_ms=1, settings=extra), work, ["BTCUSDT"])
        assert live_context_path(s) == work / "context.db", extra
        app = Heartless(s)
        assert app.research.worker_settings()["CONTEXT_DB"] == str(work / "context.db")
        app._prune(1_780_000_000_000)  # the retention prune at the virtual time touches only the work dir's store
        asyncio.run(app.stop())
    assert not offline.exists()


# --- (2) the research cycle and the start-up context fill -----------------------------------------------------------------
def test_the_research_cycle_waits_for_the_start_up_context_fill(tmp_path, monkeypatch):
    app, rm = _env(tmp_path)

    async def run():
        loop = asyncio.get_running_loop()
        gate = asyncio.Event()

        async def fill():
            await gate.wait()
            return 72

        submitted: list[bool] = []

        def submit(executor, fn, *args):
            submitted.append(app._ctx_fill_task.done())
            fut = loop.create_future()
            fut.set_result({"error": "no candles"})
            return fut

        monkeypatch.setattr(loop, "run_in_executor", submit)
        rm.pool = _DummyPool()
        app._ctx_fill_task = asyncio.create_task(fill())
        cyc = asyncio.create_task(rm.cycle(force=True))
        for _ in range(20):
            await asyncio.sleep(0)
        assert submitted == [] and rm.running and not cyc.done()  # the worker would load a context with holes
        gate.set()
        assert await cyc == {"error": "no candles"} and submitted == [True]
        # a finished (or failed) fill does not hold a cycle up
        assert await rm.cycle(force=True) == {"error": "no candles"} and submitted == [True, True]
        # the bot stops while the cycle waits: nothing is submitted, the cycle ends quietly
        app._ctx_fill_task = asyncio.create_task(asyncio.Event().wait())
        cyc = asyncio.create_task(rm.cycle(force=True))
        for _ in range(5):
            await asyncio.sleep(0)
        rm.stopped = True
        app._ctx_fill_task.cancel()
        assert await cyc is None and submitted == [True, True] and rm.running is False

    asyncio.run(run())


# --- (3) an older stored challenger keeps the base of its first load ------------------------------------------------------
def test_an_older_challenger_keeps_the_base_recorded_at_its_first_load(tmp_path):
    app = _App(tmp_path)
    champ0 = app.params
    v = champ0.alphas["trend_pullback"]["sl_atr"]
    old = champ0.with_alpha("trend_pullback", {"sl_atr": v + 0.5}, version="vold")
    started = now_ms() - 10 * MS_DAY
    app.store.save_params_version("vold", started, "challenger", "trend_pullback", "pre-C2-B", old.to_dict(),
                                  {"slot": "challenger-1", "started": started})  # stored without base / diff
    (slot,) = ResearchManager(app).load_challengers()
    assert slot.base == champ0.version and set(slot.diff["alphas"]) == {"trend_pullback"}
    row = next(r for r in app.store.load_params_versions(role="challenger") if r["id"] == "vold")
    assert row["metrics"]["base"] == champ0.version and row["metrics"]["diff"] == slot.diff
    assert row["metrics"]["slot"] == "challenger-1" and row["metrics"]["started"] == started  # the rest is kept
    # another change is promoted (mean_reversion re-tuned), then the bot restarts
    lb = champ0.alphas["mean_reversion"]["lookback"]
    app.params = champ0.with_alpha("mean_reversion", {"lookback": lb + 1}, version="vlater")
    rm = ResearchManager(app)
    (again,) = rm.load_challengers()
    # derived again from the new champion, the diff would hold the challenger's stale mean_reversion values: its
    # promotion would undo the promoted re-tune
    assert again.diff == slot.diff and again.base == champ0.version and "mean_reversion" not in again.diff["alphas"]
    assert rm.promotion_blockers(again) == []
    new = R.apply_diff(app.params, rm.slot_diff(again))
    assert new.alphas["mean_reversion"]["lookback"] == lb + 1 and new.alphas["trend_pullback"]["sl_atr"] == v + 0.5


# --- (4) the retry cooldown and kill-switch trips -------------------------------------------------------------------------
def _lose(app, slot, alpha: str, n: int = 14) -> None:
    for i in range(n):  # n full losses in a row: the challenger's kill switch trips on `alpha`
        app.store.save_trade(_rec(slot.name, i, alpha, -1.0, slot.started + (i + 2) * MS_HOUR, slot.params.version))


async def test_a_kill_switch_trip_on_an_inherited_alpha_is_no_trial_of_the_change():
    app, candles, _ = _make()
    await _boot(app, candles)
    app.params.enabled["trend_pullback"] = True  # the champion trades it: a challenger inherits it from its base

    def change():
        p = app.params.clone(note="enable squeeze_breakout")
        p.enabled["squeeze_breakout"] = True
        return p

    slot = await app.install_challenger(change(), source="squeeze_breakout")
    assert slot.diff["enabled"] == {"squeeze_breakout": True}
    _lose(app, slot, "trend_pullback")
    (trip,) = await app.killswitch.check(slot.name)
    assert trip["alpha"] == "trend_pullback" and slot not in app.challengers
    # the inherited alpha lost, not the change: the same change may be tried again at once
    slot2 = await app.install_challenger(change(), source="squeeze_breakout")
    assert slot2 is not None
    # a trip on the alpha the change switches on judged the change: no retry within CHALLENGER_RETRY_DAYS
    _lose(app, slot2, "squeeze_breakout")
    (trip,) = await app.killswitch.check(slot2.name)
    assert trip["alpha"] == "squeeze_breakout" and slot2 not in app.challengers
    assert await app.install_challenger(change(), source="squeeze_breakout") is None
    await app.stop()

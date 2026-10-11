"""C2-B: a challenger is a CHANGE to the champion it was created from (base + diff). A promotion applies only that change
onto the current champion; it is refused (the challenger retired, with the reason logged) when the change would
re-enable an alpha the kill switch switched off after the base, or when a later champion change set a key it touches
differently. Plus the retry cooldown for a change that was judged and retired. Synthetic records only."""
from __future__ import annotations

import asyncio

from heartless.config import Settings
from heartless.core.store import Store
from heartless.learning import research as R
from heartless.learning.research import ChallengerSlot, ResearchManager
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, now_ms
from tests.test_p4_app import _boot, _capture, _make, _rec


def _strong(app, engine: str, start: int, alpha: str, version: str = "v", days: int = 90) -> None:
    """A record that clears any checkpoint bar: three trades a day, every day of the trial, +0.33R mean (the days
    differ, so the day-clustered t is defined)."""
    i = 0
    for d in range(days):
        for k, r in enumerate((1.0, 1.0, -0.5) if d % 3 else (1.0, -0.5, -0.5)):
            app.store.save_trade(_rec(engine, i, alpha, r, start + d * MS_DAY + (k + 2) * MS_HOUR, version))
            i += 1


# --- diff mechanics ---------------------------------------------------------------------------------------------------
def test_params_diff_and_apply_onto_a_newer_champion():
    base = StrategyParams.default()
    ch = base.with_alpha("trend_pullback", {"sl_atr": base.alphas["trend_pullback"]["sl_atr"] + 0.5})
    ch.enabled["trend_pullback"] = True
    diff, old = R.params_diff(base, ch)
    assert diff["enabled"] == {"trend_pullback": True} and list(diff["alphas"]) == ["trend_pullback"]
    assert set(diff["alphas"]["trend_pullback"]) == {"sl_atr"} and diff["ensemble"] == {}
    assert old["enabled"] == {"trend_pullback": False}
    assert old["alphas"]["trend_pullback"]["sl_atr"] == base.alphas["trend_pullback"]["sl_atr"]
    # the champion moved on in the meantime (another alpha switched on): the change applies on top, keeping that
    newer = base.clone()
    newer.enabled["squeeze_breakout"] = True
    assert R.diff_conflicts(diff, old, base.to_dict(), newer) == []
    out = R.apply_diff(newer, diff, version="vnew", note="n")
    assert out.version == "vnew" and out.enabled["trend_pullback"] and out.enabled["squeeze_breakout"]
    assert out.alphas["trend_pullback"]["sl_atr"] == ch.alphas["trend_pullback"]["sl_atr"]
    assert out.alphas["mean_reversion"] == newer.alphas["mean_reversion"]


def test_diff_conflicts_detect_a_later_change_of_a_touched_alpha():
    base = StrategyParams.default()
    base.enabled["trend_pullback"] = True
    v = base.alphas["trend_pullback"]
    ch = base.with_alpha("trend_pullback", {"sl_atr": v["sl_atr"] + 0.5})
    diff, old = R.params_diff(base, ch)
    later = base.with_alpha("trend_pullback", {"sl_atr": v["sl_atr"] - 0.25})  # another promotion re-tuned it
    assert any("trend_pullback.sl_atr" in c for c in R.diff_conflicts(diff, old, base.to_dict(), later))
    other = list(k for k in v if k != "sl_atr")[0]
    later2 = base.with_alpha("trend_pullback", {other: v[other]})  # unchanged values: no conflict
    assert R.diff_conflicts(diff, old, base.to_dict(), later2) == []
    killed = base.clone()
    killed.enabled["trend_pullback"] = False  # switched off since the base (kill switch / prune)
    assert R.diff_conflicts(diff, old, base.to_dict(), killed) == ["trend_pullback: switched off since the base"]
    same = R.apply_diff(base, diff)  # the champion already carries exactly this change: nothing to undo
    assert R.diff_conflicts(diff, old, base.to_dict(), same) == []


# --- the manager refuses a change that would undo a later champion change ----------------------------------------------
class _App:
    def __init__(self, tmp_path):
        self.s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
        self.store = Store(tmp_path / "h.db")
        self.params = StrategyParams.default()
        self.engines = {"paper": object()}
        self.challengers: list[ChallengerSlot] = []
        self.promoted, self.retired = [], []

    async def promote(self, slot, decision):
        self.promoted.append((slot, decision))
        self.challengers.remove(slot)

    async def retire_challenger(self, slot, reason, silent=False, decision=None):
        self.retired.append((slot, reason, decision))
        self.challengers.remove(slot)


def _slot(app, base: StrategyParams, params: StrategyParams, start: int, name="challenger-1") -> ChallengerSlot:
    diff, old = R.params_diff(base, params)
    slot = ChallengerSlot(name=name, params=params, started=start, source="test", base=base.version,
                          base_created=base.created, diff=diff, base_values=old, base_params=base.to_dict())
    app.challengers.append(slot)
    app.engines[name] = object()
    return slot


def test_a_change_reenabling_an_alpha_killed_after_its_base_is_retired_not_promoted(tmp_path):
    app = _App(tmp_path)
    start = now_ms() // MS_DAY * MS_DAY - 95 * MS_DAY
    base = app.params.clone(version="vbase")
    base.created = start - MS_HOUR
    ch = base.clone(version="vch")
    ch.enabled["trend_pullback"] = True
    slot = _slot(app, base, ch, start)
    # after the base: trend_pullback was switched on by another promotion and then off again by the kill switch
    on = base.clone(version="von")
    on.enabled["trend_pullback"] = True
    app.store.save_params_version("von", start + 5 * MS_DAY, "retired", "other", "promoted", on.to_dict())
    off = on.clone(version="voff")
    off.enabled["trend_pullback"] = False
    app.store.save_params_version("voff", start + 9 * MS_DAY, "champion", "kill-switch", "kill switch: trend_pullback off",
                                  off.to_dict(), {"decisions": [{"kind": "kill_switch", "alpha": "trend_pullback"}]})
    app.params = off  # the current champion has it off, just like the base: no value conflict, the history decides
    _strong(app, "challenger-1", start, "trend_pullback")
    rm = ResearchManager(app)
    assert rm.promotion_blockers(slot) == ["kill switch switched trend_pullback off in voff after the challenger's base vbase"]
    asyncio.run(rm.evaluate_challengers(force=True))
    assert app.promoted == []
    (s, reason, d), = app.retired
    assert s is slot and d["verdict"] == "retire" and d["blockers"] and "kill switch" in d["rule"]
    assert "승격 기준은 넘었으나" in reason and "voff" in reason
    # a kill-switch trip BEFORE the base does not block: switching back on through a promotion is the way back
    (tmp_path / "b").mkdir()
    app2 = _App(tmp_path / "b")
    app2.store.save_params_version("voff", start - 2 * MS_DAY, "retired", "kill-switch", "k", off.to_dict(),
                                   {"decisions": [{"kind": "kill_switch", "alpha": "trend_pullback"}]})
    slot2 = _slot(app2, base, ch, start)
    assert ResearchManager(app2).promotion_blockers(slot2) == []


def test_a_change_conflicting_with_a_later_promotion_is_retired_with_the_reason(tmp_path):
    app = _App(tmp_path)
    start = now_ms() // MS_DAY * MS_DAY - 95 * MS_DAY
    base = app.params.clone(version="vbase")
    base.enabled["trend_pullback"] = True
    app.params = base
    v = base.alphas["trend_pullback"]["sl_atr"]
    slot = _slot(app, base, base.with_alpha("trend_pullback", {"sl_atr": v + 0.5}, version="vch"), start)
    app.params = base.with_alpha("trend_pullback", {"sl_atr": v - 0.25}, version="vlater")  # a later promotion
    _strong(app, "challenger-1", start, "trend_pullback")
    rm = ResearchManager(app)
    asyncio.run(rm.evaluate_challengers(force=True))
    assert app.promoted == []
    (s, reason, d), = app.retired
    assert d["blockers"] == [f"trend_pullback.sl_atr: {v!r} -> {v - 0.25!r} since the base"]
    assert "conflicts with the current champion" in d["rule"] and "trend_pullback.sl_atr" in reason


def test_check_conflicts_frees_the_slot_right_after_a_champion_change(tmp_path):
    app = _App(tmp_path)
    base = app.params.clone(version="vbase")
    base.enabled["trend_pullback"] = True
    app.params = base
    v = base.alphas["trend_pullback"]["sl_atr"]
    s1 = _slot(app, base, base.with_alpha("trend_pullback", {"sl_atr": v + 0.5}, version="v1"), now_ms())
    s2 = _slot(app, base, base.clone(version="v2"), now_ms(), name="challenger-2")
    s2.params.enabled["discovered"] = True
    s2.diff, s2.base_values = R.params_diff(base, s2.params)
    killed = base.clone(version="vkilled")
    killed.enabled["trend_pullback"] = False
    app.params = killed
    rm = ResearchManager(app)
    retired = asyncio.run(rm.check_conflicts())
    assert retired == [s1] and app.challengers == [s2]
    (_, reason, d), = app.retired
    assert d["kind"] == "rebase" and d["blockers"] == ["trend_pullback: switched off since the base"]


def test_an_older_stored_challenger_takes_the_current_champion_as_its_base(tmp_path):
    app = _App(tmp_path)
    p = app.params.clone(version="vold")
    p.enabled["squeeze_breakout"] = True
    app.store.save_params_version("vold", now_ms(), "challenger", "x", "x", p.to_dict(), {"slot": "challenger-1",
                                                                                          "started": now_ms()})
    (slot,) = ResearchManager(app).load_challengers()
    assert slot.base == app.params.version and slot.diff["enabled"] == {"squeeze_breakout": True}


# --- end to end on the real orchestrator -----------------------------------------------------------------------------
async def test_promotion_applies_only_the_change_onto_the_current_champion():
    app, candles, _ = _make()
    await _boot(app, candles)
    sent = _capture(app)
    params = app.params.clone(note="enable squeeze_breakout")
    params.enabled["squeeze_breakout"] = True
    slot = await app.install_challenger(params, source="squeeze_breakout")
    row = next(r for r in app.store.load_params_versions(role="challenger") if r["id"] == params.version)
    assert row["metrics"]["base"] == app.params.version and row["metrics"]["diff"]["enabled"] == {"squeeze_breakout": True}
    base_version = app.params.version
    # the champion changes after the challenger's base (a non-conflicting change: another alpha's value)
    later = app.params.with_alpha("mean_reversion", {"lookback": app.params.alphas["mean_reversion"]["lookback"] + 1},
                                  note="later champion")
    app.store.save_params_version(later.version, later.created, "champion", "test", later.note, later.to_dict())
    app.params = later
    slot.started = now_ms() - 91 * MS_DAY
    _strong(app, slot.name, slot.started, "squeeze_breakout", params.version)
    await app.research.evaluate_challengers(force=True)
    new = app.params
    assert new.version not in (later.version, base_version)
    assert new.enabled["squeeze_breakout"] is True  # the challenger's change ...
    assert new.alphas["mean_reversion"] == later.alphas["mean_reversion"]  # ... on top of the later champion
    stored = app.store.load_params_versions(role="champion", limit=1)[0]
    assert stored["metrics"]["base"] == base_version and stored["metrics"]["diff"]["enabled"] == {"squeeze_breakout": True}
    assert f"applied onto {later.version}" in stored["note"] and "90-day checkpoint" in stored["note"]
    assert any("변경분만 적용" in m for m in sent)
    await app.stop()


async def test_a_judged_change_is_not_retried_within_the_cooldown():
    app, candles, _ = _make()
    await _boot(app, candles)
    params = app.params.clone(note="enable squeeze_breakout")
    params.enabled["squeeze_breakout"] = True
    slot = await app.install_challenger(params, source="squeeze_breakout")
    d = {"verdict": "retire", "rule": "trial over: not promoted at the last checkpoint (90 days)", "path": "absolute"}
    await app.retire_challenger(slot, reason="시험 기간 종료", decision=d)
    again = app.params.clone(note="the same change again")
    again.enabled["squeeze_breakout"] = True
    assert await app.install_challenger(again, source="squeeze_breakout") is None
    other = app.params.clone(note="another change")
    other.enabled["mean_reversion"] = True
    assert await app.install_challenger(other, source="mean_reversion") is not None
    # a replacement (no judgement) or a conflict retirement does not count; after the cooldown the change may return
    rows = app.store.get(app.TRIED_KEY)
    rows[-1]["at"] = now_ms() - (app.s.challenger_retry_days + 1) * MS_DAY
    app.store.set(app.TRIED_KEY, rows)
    third = app.params.clone(note="after the cooldown")
    third.enabled["squeeze_breakout"] = True
    assert await app.install_challenger(third, source="squeeze_breakout") is not None
    await app.stop()

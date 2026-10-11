"""C2-B: replay determinism with a challenger. A challenger's Thompson-sampling RNG is seeded from its slot name and its
(virtual) start, not from its version id (wall time + random bits): the same 3-day synthetic replay run twice, with a
challenger installed mid-run as the research cycle would, gives identical trades for the champion and the challenger.
Also covers the learning fields of the replay summary (docs/REPLAY.md)."""
from __future__ import annotations

import pytest

from heartless.core.store import Store
from heartless.data import features
from heartless.replay.runner import ReplayConfig, run_replay
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_MINUTE
from tests.test_p5_replay import NO_LEARNING, SEED_DAYS, SMALL_FRAMES, START, build_source, busy_params

COLS = ("engine", "symbol", "side", "alpha", "entry_time", "exit_time", "entry_price", "exit_price", "qty", "pnl",
        "r_multiple", "exit_reason")


@pytest.fixture
def small_frames():
    with pytest.MonkeyPatch.context() as mp:
        for tf, n in SMALL_FRAMES.items():
            mp.setitem(features.LIVE_WINDOW, tf, n)
        yield


async def _install(app) -> None:
    p = StrategyParams.from_dict(busy_params()).clone(note="replay challenger")  # a fresh version id every run
    p = p.with_alpha("trend_pullback", {"min_conf": 0.5}, version=p.version, note=p.note)
    slot = await app.install_challenger(p, source="trend_pullback")
    assert slot is not None


def _run(tmp, name: str, src: str) -> tuple[dict, list[tuple]]:
    cfg = ReplayConfig(source_db=src, start_ms=START, end_ms=START + 3 * MS_DAY - 1, work_dir=str(tmp / name),
                       symbols=["ETHUSDT"], seed_days=SEED_DAYS, params=busy_params(), settings=NO_LEARNING,
                       commands=[], inject=[(START + 10 * MS_MINUTE, _install)])
    s = run_replay(cfg)
    st = Store(tmp / name / "heartless.db")
    rows = [tuple(r[c] for c in COLS) for r in st.query("SELECT * FROM trades ORDER BY engine, entry_time, position_id")]
    st.close()
    return s, rows


def test_the_same_replay_with_a_challenger_gives_identical_trades(tmp_path, small_frames):
    src = build_source(tmp_path / "src.db", 3, ["ETHUSDT"])
    s1, t1 = _run(tmp_path, "a", src)
    s2, t2 = _run(tmp_path, "b", src)
    for s in (s1, s2):
        assert s["passed"], s["report"]
        assert s["invariants"]["violations_total"] == 0
    assert s1["learning"]["challengers_installed"][0]["version"] != s2["learning"]["challengers_installed"][0]["version"]
    ch = [r for r in t1 if r[0] == "challenger-1"]
    assert ch and [r for r in t1 if r[0] == "paper"], "both engines must trade for the comparison to mean anything"
    assert t1 == t2
    # the learning fields of the summary (docs/REPLAY.md)
    learn = s1["learning"]
    assert sorted(learn["champion_enabled"]) == sorted(a for a, on in busy_params()["enabled"].items() if on)
    assert learn["kill_switch_trips"] == s2["learning"]["kill_switch_trips"]
    assert learn["challenger_decisions"] and learn["challenger_decisions"][0]["verdict"] == "hold"
    assert learn["challenger_decisions"][0]["slot"] == "challenger-1"
    assert s1["context"]["snapshots"] == s1["context"]["hours_expected"]

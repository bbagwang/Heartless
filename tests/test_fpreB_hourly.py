"""F-pre-B hook of the shock layer in the hourly evaluator: the entry filter blocks decisions at halted hour closes and
counts decisions only, an exempt alpha takes none, and `heartless hourly` applies the layer when SHOCK_LAYER is on."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.learning import hourly as Hr
from heartless.strategy import shock as SH
from tests.test_fpreA_hourly import H, T0, Plan, ledger, make_db  # noqa: F401  (ledger: autouse fixture)


def test_entry_filter_blocks_halted_decisions_only(tmp_path):
    n = 24 * 10
    c = np.round(100 + np.sin(np.arange(n) / 5), 4)
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": c}})
    T = T0 + 50 * H
    plan = tuple((t, 1, 90.0, None, 2) for t in (T, T + H, T + 3 * H))
    flt = SH.EntryFilter(SH.ShockEvents(hourly={"S1": [T]}))
    res = Hr.evaluate(Plan(plan=plan), T0, T0 + n * H - 1, db, symbols=["AUSDT"], entry_filter=flt)
    assert [t["entry_time"] for t in res.trades] == [T + 3 * H]
    free = Hr.evaluate(Plan(plan=plan), T0, T0 + n * H - 1, db, symbols=["AUSDT"])
    assert [t["entry_time"] for t in free.trades] == [T, T + 3 * H]  # T + 1 h is inside the first trade
    # the layer is the evaluator's policy, not the configuration's: one config hash with and without it
    assert flt.blocked == 2 and res.config == free.config and "entry_filter" not in res.config
    # a filter of the configuration's own is part of its identity

    def odd_hours(sym, times):
        return (np.asarray(times) // H) % 2 == 1

    own = Hr.evaluate(Plan(plan=plan), T0, T0 + n * H - 1, db, symbols=["AUSDT"], entry_filter=odd_hours)
    assert own.config["entry_filter"] == "odd_hours"


def test_hourly_cli_applies_the_layer(tmp_path, capsys):
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": np.round(100 + np.sin(np.arange(240) / 3), 4)}})
    out = tmp_path / "r.json"
    args = SimpleNamespace(period=None, start="2024-01-02", end="2024-01-09", universe=None, symbols="AUSDT",
                           equivalence=False, alpha="heartless.learning.hourly:ReferenceAlpha", set=["hold_h=6"],
                           stress=False, tag=None, trades_out=None, out=str(out), workers=1, argv=["heartless", "hourly"])
    from heartless.learning import trials

    configs = {}
    for on in (True, False):
        SH._CACHE.clear()
        s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=db, SHOCK_LAYER=on)
        Hr.main_cli(s, SimpleNamespace(**{**vars(args), "tag": "fpreb_test" if on else None}))
        doc = json.loads(out.read_text())
        configs[on] = doc["config"]
        assert "entry_filter" not in doc["config"]
        if on:
            # no BTC anywhere in these databases: nothing can be evaluated, nothing is blocked, the run says so
            assert doc["shock"]["missing"] == ["S0", "S1", "S2", "S3"] and doc["shock"]["blocked_decisions"] == 0
            assert doc["trial"]["shock"] == doc["shock"]  # the ledger line records the layer
            err = capsys.readouterr().err
            assert "shock layer: 0 decision(s) blocked" in err and "S0, S1, S2, S3 not evaluated" in err
            assert doc["trial"]["config_hash"] == trials.config_hash(doc["config"])
        else:
            assert doc["shock"] is None and "trial" not in doc
    assert configs[True] == configs[False]  # one configuration, with or without the layer
    # ... so a trial runs with the layer: SHOCK_LAYER=false with --tag is refused before anything is evaluated or recorded
    n_lines = trials.family_tests("fpreb_test")
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=db, SHOCK_LAYER=False)
    with pytest.raises(SystemExit) as e:
        Hr.main_cli(s, SimpleNamespace(**{**vars(args), "tag": "fpreb_test", "out": None}))
    assert e.value.code == 2 and "SHOCK_LAYER=false runs only without --tag" in capsys.readouterr().err
    assert trials.family_tests("fpreb_test") == n_lines


def test_an_exempt_alpha_may_record_without_the_layer(tmp_path, capsys):
    """A shock_exempt alpha takes no entry filter either way, so its trial does not depend on SHOCK_LAYER."""
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": np.round(100 + np.sin(np.arange(240) / 3), 4)}})
    out = tmp_path / "r.json"
    args = SimpleNamespace(period=None, start="2024-01-02", end="2024-01-09", universe=None, symbols="AUSDT",
                           equivalence=False, alpha="tests.test_fpreB_hourly:ExemptReference", set=["hold_h=6"],
                           stress=False, tag="fpreb_exempt", trades_out=None, out=str(out), workers=1,
                           argv=["heartless", "hourly"])
    Hr.main_cli(Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=db, SHOCK_LAYER=False), args)
    doc = json.loads(out.read_text())
    assert doc["trial"]["family"] == "fpreb_exempt" and doc["shock"] is None


class ExemptReference(Hr.ReferenceAlpha):
    shock_exempt = True

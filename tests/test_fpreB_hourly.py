"""F-pre-B hook of the shock layer in the hourly evaluator: the entry filter blocks decisions at halted hour closes and
counts decisions only, an exempt alpha takes none, and `heartless hourly` applies the layer when SHOCK_LAYER is on."""
import json
from types import SimpleNamespace

import numpy as np

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
    assert flt.blocked == 2 and res.config["entry_filter"] == "shock_layer"
    free = Hr.evaluate(Plan(plan=plan), T0, T0 + n * H - 1, db, symbols=["AUSDT"])
    assert [t["entry_time"] for t in free.trades] == [T, T + 3 * H]  # T + 1 h is inside the first trade
    assert "entry_filter" not in free.config


def test_hourly_cli_applies_the_layer(tmp_path, capsys):
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": np.round(100 + np.sin(np.arange(240) / 3), 4)}})
    out = tmp_path / "r.json"
    args = SimpleNamespace(period=None, start="2024-01-02", end="2024-01-09", universe=None, symbols="AUSDT",
                           equivalence=False, alpha="heartless.learning.hourly:ReferenceAlpha", set=["hold_h=6"],
                           stress=False, tag=None, trades_out=None, out=str(out), workers=1, argv=["heartless", "hourly"])
    for on in (True, False):
        SH._CACHE.clear()
        s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=db, SHOCK_LAYER=on)
        Hr.main_cli(s, args)
        doc = json.loads(out.read_text())
        if on:
            # no BTC anywhere in these databases: nothing can be evaluated, nothing is blocked, the run says so
            assert doc["shock"]["missing"] == ["S0", "S1", "S2", "S3"] and doc["shock"]["blocked_decisions"] == 0
            assert doc["config"]["entry_filter"] == "shock_layer"
            assert "shock layer: 0 decision(s) blocked" in capsys.readouterr().err
        else:
            assert doc["shock"] is None and "entry_filter" not in doc["config"]

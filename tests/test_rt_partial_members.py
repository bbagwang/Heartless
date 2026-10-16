"""Red-team F-H1 (ROADMAP section 6, 2026-10-16): a topN lab / discover run checks the 1m coverage of every
member-month and of every membership span's warm-up and is refused (blocked_data) when more than 2% of either is
missing. --allow-partial runs it anyway, loudly, recorded in --out, never with --tag and never in a FORWARD command."""
import json
from types import SimpleNamespace

import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import universe as U
from heartless.util.timeutil import MS_DAY
from tests.test_f0_universe import M, make_universe_db

MIN = 60_000


def _minutes(store: Store, sym: str, a: int, b: int, every: int = 1) -> None:
    store.save_candles(sym, [Candle(t, 1, 1, 1, 1, 1, 1, 1, 1, t + MIN - 1) for t in range(a, b, every * MIN)])


@pytest.fixture
def dbs(tmp_path):
    """universe.db of tests/test_f0_universe (top2: December BIG + GONE, January BIG + MID) and a candle database in
    which BIG is complete with its warm-up, GONE has 1m candles only in October (outside its member month) and MID has
    none."""
    udb = make_universe_db(tmp_path / "universe.db")
    udb.close()
    cdb = tmp_path / "candles.db"
    st = Store(cdb)
    _minutes(st, "BIGUSDT", M("2023-12") - 13 * MS_DAY, M("2024-02"))
    _minutes(st, "GONEUSDT", M("2023-10"), M("2023-10") + 5 * MS_DAY)
    st.close()
    return tmp_path / "universe.db", cdb


def test_a_member_with_candles_outside_its_member_months_is_missing(dbs):
    udb, cdb = dbs
    res = U.resolve_for_run(udb, cdb, M("2023-12"), M("2024-02") - 1, 2, coverage=True)
    cov = res["coverage"]
    # BIG December + January, GONE December (until its last traded hour), MID January
    assert cov["member_months"] == 4
    assert [(m["symbol"], m["month"]) for m in cov["missing_member_months"]] == [("GONEUSDT", "2023-12"),
                                                                                 ("MIDUSDT", "2024-01")]
    gone = cov["missing_member_months"][0]
    assert gone["minutes"] == 0 and gone["expected"] == 10 * 1440  # the month after the delisting is not expected
    assert cov["missing_share"] == 0.5 and cov["warmups"] == 3
    assert {w["symbol"] for w in cov["missing_warmups"]} == {"GONEUSDT", "MIDUSDT"}  # BIG has its 12 days
    assert "2 of 4 member-months" in U.coverage_problem(cov)
    # without the coverage flag the resolution is unchanged (FORWARD's member lookup, older callers)
    assert "coverage" not in U.resolve_for_run(udb, cdb, M("2023-12"), M("2024-02") - 1, 2)


def test_holes_inside_a_member_month_count_and_the_threshold_is_two_percent(dbs, tmp_path):
    udb, _ = dbs
    cdb = tmp_path / "holes.db"
    st = Store(cdb)
    _minutes(st, "BIGUSDT", M("2023-11") - 13 * MS_DAY, M("2023-12"))
    _minutes(st, "BIGUSDT", M("2023-12"), M("2024-01"), every=2)  # half of December's minutes
    st.close()
    cov = U.resolve_for_run(udb, cdb, M("2023-11"), M("2024-01") - 1, 1, coverage=True)["coverage"]
    assert cov["member_months"] == 2 and [m["month"] for m in cov["missing_member_months"]] == ["2023-12"]
    assert cov["missing_member_months"][0]["minutes"] == 31 * 720
    ok = {"member_months": 60, "missing_member_months": [{}], "missing_share": 1 / 60, "warmups": 10,
          "missing_warmups": [], "missing_warmup_share": 0.0}
    assert U.coverage_problem(ok) is None and U.MAX_MISSING_SHARE == 0.02
    assert "2 of 60 member-months" in U.coverage_problem({**ok, "missing_member_months": [{}, {}], "missing_share": 2 / 60})
    assert "warm-up" in U.coverage_problem({**ok, "missing_warmups": [{}], "missing_warmup_share": 0.1})


def _settings(tmp_path, udb, cdb, monkeypatch):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=str(udb), CONTEXT_AUTO_BUILD=False)
    monkeypatch.setattr(settings.__class__, "db_path", property(lambda self: cdb))
    from heartless.learning import lab

    monkeypatch.setattr(lab, "UNIVERSES", dict(lab.UNIVERSES))
    return settings


def _args(**kw):
    base = dict(universe="top2", years=None, start="2023-12-01", end="2024-02-01", tag=None, allow_partial=False)
    return SimpleNamespace(**{**base, **kw})


def test_lab_refuses_a_partial_member_set_unless_allowed_and_untagged(dbs, tmp_path, monkeypatch, capsys):
    from heartless import cli

    udb, cdb = dbs
    settings = _settings(tmp_path, udb, cdb, monkeypatch)
    with pytest.raises(SystemExit) as e:
        cli._top_universe(settings, _args(), "lab")
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "blocked_data" in err and "2 of 4 member-months" in err and "GONEUSDT 2023-12" in err
    with pytest.raises(SystemExit):
        cli._top_universe(settings, _args(allow_partial=True, tag="fam"), "lab")
    assert "never recorded in the trial ledger" in capsys.readouterr().err
    res = cli._top_universe(settings, _args(allow_partial=True), "lab")
    assert res["partial"] and res["allow_partial"] and "PARTIAL MEMBER SET" in capsys.readouterr().err
    out = tmp_path / "r.json"
    out.write_text("{}")
    cli._note_top_universe(SimpleNamespace(out=str(out)), res)
    doc = json.loads(out.read_text())["universe_resolution"]
    assert doc["partial"] is True and doc["allow_partial"] is True
    assert doc["coverage"]["missing_share"] == 0.5 and len(doc["coverage"]["missing_member_months"]) == 2
    # discover resolves over --train: the same refusal
    with pytest.raises(SystemExit):
        cli._top_universe(settings, SimpleNamespace(universe="top2", train="2023-12-01:2024-02-01",
                                                    valid="2024-03-01:2024-04-01", tag=None, allow_partial=False),
                          "discover")


def test_a_complete_member_set_runs_without_the_flag(dbs, tmp_path, monkeypatch, capsys):
    from heartless import cli

    udb, cdb = dbs
    st = Store(cdb)
    _minutes(st, "GONEUSDT", M("2023-12") - 12 * MS_DAY, M("2023-12") + 11 * MS_DAY)
    _minutes(st, "MIDUSDT", M("2024-01") - 12 * MS_DAY, M("2024-02"))
    st.close()
    res = cli._top_universe(_settings(tmp_path, udb, cdb, monkeypatch), _args(), "lab")
    assert not res["partial"] and res["coverage"]["missing_share"] == 0.0 and res["coverage"]["missing_warmups"] == []
    assert "blocked_data" not in capsys.readouterr().err


def test_a_forward_command_never_carries_allow_partial():
    from heartless.learning import forward

    for bad in ("--allow-partial", "--allow-p"):
        with pytest.raises(forward.ForwardError):
            forward.normalize_command(f"lab --alpha b --universe top40 {bad}")



def test_relistings_inside_member_spans_are_reported():
    spans = {"AUSDT": [(100, 200), (400, 500)], "BUSDT": [(100, 200)]}
    relisted = {"AUSDT": [50, 150, 450, 600], "BUSDT": [250], "CUSDT": [150]}
    assert U.relisted_in_spans(relisted, spans) == {"AUSDT": [150, 450]}


def test_the_resolution_lists_relisted_members(dbs):
    udb, cdb = dbs
    res = U.resolve_for_run(udb, cdb, M("2023-12"), M("2024-02") - 1, 2, coverage=True)
    assert res["relisted_in_member_months"] == {}  # no relisting in the fixture's member months

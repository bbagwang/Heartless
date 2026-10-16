"""F-pre-A lab and discover on a topN universe (docs/ROADMAP.md 7.4): only entries made while the symbol is a member,
positions running on through membership loss, delisting closes at 5x slippage, the --out reporting, and the lab's
context top_n."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.store import Store
from heartless.data import universe as U
from heartless.learning import discovery as D
from heartless.learning import lab
from tests.test_p0_lab import DAY, INLINE, Timer, _ms, timer_db  # noqa: F401 (fixture)

MIN = 60_000


def test_member_jobs_follow_the_spans():
    a, b = _ms(2023, 12, 1), _ms(2024, 3, 1) - 1
    spans = {"A": [(_ms(2023, 12, 1), _ms(2024, 1, 1) - 1), (_ms(2024, 2, 1), _ms(2024, 3, 1) - 1)],
             "B": [(_ms(2023, 12, 15), _ms(2024, 2, 1) - 1)]}
    jobs = lab.member_jobs(["A", "B"], spans, a, b, None, 7 * DAY, 7 * DAY)
    by = {(j["symbol"], j["start"]): j for j in jobs}
    ja, ja2 = by[("A", _ms(2023, 12, 1))], by[("A", _ms(2024, 2, 1))]
    # A's December span ends before the window: it runs on to the next span (or the window end), not further
    assert ja["first"] and not ja["final"] and ja["lead_ms"] == MIN
    assert ja["runoff_end"] == _ms(2024, 2, 1) - MIN and ja["flatten_is_end"] is False
    assert ja2["first"] and ja2["final"]
    # B's span crosses a calendar year: two chunks; the second re-creates held positions through a burn-in that
    # starts inside the span, and the last chunk's run-off may reach the window end
    jb1, jb2 = by[("B", _ms(2023, 12, 15))], by[("B", _ms(2024, 1, 1))]
    assert jb1["first"] and not jb1["final"] and "runoff_end" not in jb1
    assert not jb2["first"] and jb2["burn_in_ms"] == 7 * DAY and jb2["runoff_end"] == b
    assert jb2["flatten_is_end"] is False  # 7 days of run-off end before the window end
    short = lab.member_jobs(["B"], {"B": [(_ms(2023, 12, 29), _ms(2024, 1, 15) - 1)]}, a, b, None, 7 * DAY, 7 * DAY)
    assert short[1]["burn_in_ms"] == _ms(2024, 1, 1) - _ms(2023, 12, 29) + MIN  # never before the span
    assert lab.member_jobs(["C"], {}, a, b, None, 0, 0) == []


def test_lab_counts_only_member_entries_and_keeps_positions_through_membership_loss(timer_db):
    """BTC is a member on 12-29 .. 12-30 and from 2024-01-01: no entry on 12-31; the position opened on 12-30 runs to
    its 25 h time stop on 12-31; the decision at the close before 01-01 00:00 fills at the span's first minute."""
    db, params = timer_db
    a, b = _ms(2023, 12, 29), _ms(2024, 1, 3) - 1
    spans = {"BTCUSDT": [(a, _ms(2023, 12, 31) - 1), (_ms(2024, 1, 1), b)]}
    res = lab.evaluate(params, ["BTCUSDT"], a, b, db, pool=INLINE, warmup_ms=DAY, member_windows=spans)
    # the unfiltered reference also simulates the close before the window (a span's lead minute)
    whole = lab.evaluate(params, ["BTCUSDT"], a - MIN, b, db, pool=INLINE, warmup_ms=DAY, chunk_days=0)
    assert res.errors == {} and res.trades
    assert all(U.member_mask([t["entry_time"]], spans["BTCUSDT"])[0] for t in res.trades)
    assert not [t for t in res.trades if _ms(2023, 12, 31) <= t["entry_time"] < _ms(2024, 1, 1)]
    carried = [t for t in res.trades if t["entry_time"] < _ms(2023, 12, 31) <= t["exit_time"]]
    assert carried and not carried[0]["boundary"] and carried[0]["exit_reason"] != lab.BOUNDARY_REASON
    assert any(t["entry_time"] == _ms(2024, 1, 1) for t in res.trades)
    assert any(_ms(2023, 12, 31) <= t["entry_time"] < _ms(2024, 1, 1) for t in whole.trades)  # what was dropped
    # the first span's trades are the unfiltered run's (nothing before them differs)
    first = lambda r: [(t["entry_time"], t["exit_time"], round(t["r_multiple"], 12)) for t in r.trades
                       if t["entry_time"] < _ms(2023, 12, 31)]
    assert first(res) == first(whole)
    m = res.to_json()["membership"]
    assert m["member_months"] == {"BTCUSDT": 2} and m["jobs"] == 2 and m["delisting_closes"] == []
    assert "member-months" in res.report()


def test_delisting_close_reprices_the_end_flatten(timer_db):
    """The symbol's trading ends at 2024-01-01 11:59:59.999: the position open then leaves at that close with 5x the
    run's slippage, the same fill as a run ending there except for 4 more slippages; nothing is entered later."""
    db, params = timer_db
    a = _ms(2023, 12, 29)
    gone = _ms(2024, 1, 1) + 12 * 3_600_000 - 1
    ref = lab.evaluate(params, ["BTCUSDT"], a, gone, db, pool=INLINE, warmup_ms=DAY, chunk_days=0)
    res = lab.evaluate(params, ["BTCUSDT"], a, _ms(2024, 1, 3) - 1, db, pool=INLINE, warmup_ms=DAY, chunk_days=0,
                       delisted={"BTCUSDT": gone})
    last_ref, last = ref.trades[-1], res.trades[-1]
    assert last_ref["exit_reason"] == lab.BOUNDARY_REASON and last["exit_reason"] == lab.DELIST_REASON
    assert last["entry_time"] == last_ref["entry_time"] and last["exit_time"] == last_ref["exit_time"] == gone
    assert last["exit_price"] == pytest.approx(last_ref["exit_price"] * (1 - 4 * 1.5e-4), rel=1e-12)
    assert last["pnl"] < last_ref["pnl"] and last["r_multiple"] < last_ref["r_multiple"]
    assert last["delisting"]["r_before"] == pytest.approx(last_ref["r_multiple"], rel=1e-12)
    assert last["delisting"]["slippage_bps"] == pytest.approx(7.5) and not last["boundary"]
    assert max(t["entry_time"] for t in res.trades) <= gone and len(res.trades) == len(ref.trades)
    # stress: the doubled slippage x5 = 10x nominal
    st = lab.evaluate(params, ["BTCUSDT"], a, _ms(2024, 1, 3) - 1, db, pool=INLINE, warmup_ms=DAY, chunk_days=0,
                      delisted={"BTCUSDT": gone}, settings=dict(lab.STRESS_SETTINGS))
    assert st.trades[-1]["delisting"]["slippage_bps"] == pytest.approx(15.0)


def test_delisting_close_arithmetic():
    d = {"side": "SHORT", "entry_price": 100.0, "exit_price": 110.0, "notional": 1000.0, "gross": -100.0,
         "fees": 1.05, "pnl": -101.05, "risk_amount": 50.0, "r_multiple": -2.021}
    lab.delisting_close(d, 1.5, 0.0005, 123)
    new = 110.0 * (1 + 4 * 1.5e-4)
    q = 10.0
    assert d["exit_price"] == pytest.approx(new) and d["gross"] == pytest.approx(-100.0 - (new - 110.0) * q)
    assert d["fees"] == pytest.approx(1.05 + (new - 110.0) * q * 0.0005)
    assert d["pnl"] == pytest.approx(d["gross"] - d["fees"]) and d["r_multiple"] == pytest.approx(d["pnl"] / 50.0)
    assert d["exit_reason"] == "delisting" and d["delisting"] == {"close": 123, "r_before": -2.021, "slippage_bps": 7.5}


def test_lab_cli_top_universe_reports_membership_and_passes_context_top_n(timer_db, tmp_path, monkeypatch, capsys):
    from heartless import cli
    from tests.test_f0_universe import make_universe_db

    db, params = timer_db
    udb = make_universe_db(tmp_path / "universe.db")
    udb.close()
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=str(tmp_path / "universe.db"),
                        CONTEXT_AUTO_BUILD=False, CONTEXT_TOP_N=40)
    monkeypatch.setattr(settings.__class__, "db_path", property(lambda self: db))
    # the 1m candles are BTC's under the universe's BIG name
    st = Store(db)
    st.execute("UPDATE candles SET symbol='BIGUSDT'")
    st.close()
    seen = {}
    real = lab.evaluate

    def capture(*a, **k):
        seen.update(k)
        return real(*a, **{**k, "pool": INLINE, "warmup_ms": DAY, "context": None})

    monkeypatch.setattr(lab, "evaluate", capture)
    monkeypatch.setattr(lab, "UNIVERSES", dict(lab.UNIVERSES))
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(params.to_dict()))
    out = tmp_path / "r.json"
    args = SimpleNamespace(symbols=None, universe="top2", years=None, start="2023-12-01", end="2024-01-03",
                           split="train", unseal=[], params=str(pfile), set=[], stress=False, meta=False,
                           chunk_days=None, alpha=None, workers=1, json=False, trades_out=None, out=str(out),
                           by_side=False, tag=None, allow_partial=True,  # MID has no 1m candles (partial)
                           argv=["heartless", "lab", "--universe", "top2"])
    top = cli._top_universe(settings, args, "lab")
    lab.main_cli(settings, args)
    cli._note_top_universe(args, top)
    err = capsys.readouterr().err
    assert "member-months" in err and "delisted in the window: GONEUSDT" in err
    assert seen["context"]["top_n"] == 40  # a store built with CONTEXT_TOP_N is asked for in its own mode
    assert seen["member_windows"] == {"BIGUSDT": [(lab.parse_when("2023-12-01"), lab.parse_when("2024-01-03",
                                                                                                  end=True))]}
    assert seen["delisted"]["GONEUSDT"] < lab.parse_when("2023-12-11")
    doc = json.loads(out.read_text())
    assert doc["membership"]["member_months"] == {"BIGUSDT": 2} and doc["membership"]["delisting_closes"] == []
    r = doc["universe_resolution"]
    assert r["member_months"] == {"BIGUSDT": 2} and r["skipped_member_months"] == 2  # GONE December, MID January
    assert set(r["delisted"]) == {"GONEUSDT"} and doc["wave_f_period"] == "TRAIN-F"  # crosses TRAIN-A / TRAIN-B


def test_lab_context_top_n_store_is_accepted(timer_db, tmp_path):
    """A context store built in top-N mode (CONTEXT_TOP_N) serves a lab run that asks for that mode and is refused in
    the fixed mode the lab used to ask for."""
    from heartless.data import context as C

    db, _ = timer_db
    ctx_path = tmp_path / "ctx.db"
    src, dst = Store(db), Store(ctx_path)
    C.build(src, dst, _ms(2023, 12, 29), _ms(2024, 1, 2) - 1, universe=["BTCUSDT"], top_n=1)
    src.close()
    dst.close()
    cfg = {"db": str(ctx_path), "universe": ["BTCUSDT"], "build": True, "top_n": 1}
    assert lab.prepare_context(db, cfg, _ms(2023, 12, 30), _ms(2024, 1, 1) - 1) == str(ctx_path)
    with pytest.raises(C.ContextMismatch, match="top_n"):
        lab.prepare_context(db, {**cfg, "top_n": 0}, _ms(2023, 12, 30), _ms(2024, 1, 1) - 1)


# --- discover ---------------------------------------------------------------------------------------------------------
def test_member_job_windows_shift_to_decisions():
    a, b = _ms(2024, 3, 1), _ms(2024, 5, 1) - 1
    w = D.member_job_windows([(a, b)], _ms(2024, 1, 1), _ms(2025, 1, 1) - 1)
    assert w == [(a - 1, b - 1)]  # the close before March enters on March 1; April 30's last close enters in May
    w = D.member_job_windows([(_ms(2023, 12, 1), _ms(2024, 2, 1) - 1)], _ms(2023, 12, 15), _ms(2025, 1, 1) - 1)
    assert w == [(_ms(2023, 12, 15) - 1, _ms(2024, 1, 1) - 2), (_ms(2024, 1, 1) - 1, _ms(2024, 2, 1) - 2)]


@pytest.fixture(scope="module")
def walk_db(tmp_path_factory):
    from heartless.core.models import Candle

    rng = np.random.default_rng(4)
    start = _ms(2024, 1, 1)
    n = 75 * 1440
    close = 50 * np.exp(np.cumsum(rng.normal(0, 6e-4, n)))
    prev = np.concatenate(([50.0], close[:-1]))
    hi, lo = np.maximum(prev, close) * 1.0002, np.minimum(prev, close) * 0.9998
    path = tmp_path_factory.mktemp("walk") / "w.db"
    st = Store(path)
    t = start + np.arange(n, dtype=np.int64) * MIN
    st.save_candles("AUSDT", [Candle(int(t[i]), float(prev[i]), float(hi[i]), float(lo[i]), float(close[i]), 10.0,
                                     10.0 * float(close[i]), 5, 5.0, int(t[i]) + MIN - 1) for i in range(n)])
    st.close()
    return str(path)


def test_discover_dataset_keeps_member_rows_only(walk_db):
    """Member in February only: the rows are the full build's decisions whose entry (close + 1 ms) is in February,
    with the same outcomes."""
    a, b = _ms(2024, 1, 25), _ms(2024, 3, 10) - 1
    feb = (_ms(2024, 2, 1), _ms(2024, 3, 1) - 1)
    full = D.build_dataset(walk_db, ["AUSDT"], a, b, "1h", workers=1, templates=(0, 3))
    mem = D.build_dataset(walk_db, ["AUSDT"], a, b, "1h", workers=1, templates=(0, 3), windows={"AUSDT": [feb]})
    keep = U.member_mask(full.times + 1, [feb])
    assert keep.sum() == 29 * 24 and np.array_equal(mem.times, full.times[keep])
    assert mem.times[0] == feb[0] - 1 and mem.times[-1] == feb[1] - 3_600_000
    for k in full.R:  # the outcomes come from the same 1m bars (exits run past the span: membership loss is no exit)
        # (the ATR behind the stops starts from a later warm-up: equal up to the Wilder filter's start-up residue)
        assert np.allclose(mem.R[k], full.R[k][keep], rtol=1e-9, atol=1e-12, equal_nan=True)
        assert np.array_equal(mem.EXIT[k], full.EXIT[k][keep])
    assert mem.FORCED == {} and full.FORCED == {}
    with pytest.raises(RuntimeError, match="membership span"):
        D.build_dataset(walk_db, ["AUSDT"], a, b, "1h", workers=1, windows={"AUSDT": []})


def test_discover_delisting_close_marks_rows_cut_by_the_end_of_trading(walk_db):
    """The data 'end' at 2024-02-20 12:00 (a delisting): a 24 h time-exit row decided after 02-19 12:00 leaves at the
    last close with 5x slippage and is flagged; earlier rows are unchanged."""
    a, b = _ms(2024, 2, 10), _ms(2024, 2, 25) - 1
    gone = _ms(2024, 2, 20) + 12 * 3_600_000 - 1
    full = D.build_dataset(walk_db, ["AUSDT"], a, b, "1h", workers=1, templates=(4,))
    dl = D.build_dataset(walk_db, ["AUSDT"], a, b, "1h", workers=1, templates=(4,), delisted={"AUSDT": gone})
    assert dl.times[-1] == gone and dl.entry_t[-1] == -1  # the last traded hour's close: nothing trades after it
    key = (4, 1)
    forced = dl.FORCED[key]
    cut = dl.times + 1 + 1440 * MIN > gone + 1
    ok = np.isfinite(dl.R[key])
    assert forced[ok & cut].any() and not forced[~cut].any()
    early = ~cut
    n_early = int(early.sum())
    assert np.array_equal(dl.R[key][early], full.R[key][:n_early], equal_nan=True)
    st = D.rule_stats(dl, D.Rule(1, 4, ((0, ">", -1e18),), "1h"))
    assert "delist" in st and st["delist"] >= 0
    # the same exit price with one slippage instead of five: a bit more R
    o1 = np.array([1.0])
    r5, _ = D.simulate_template(o1, o1, o1, o1, np.array([0]), np.array([0.01]), 1, 4.0, None, 5, D.CostModel(),
                                end_mult=5.0, forced=np.zeros(1, bool))
    r1, _ = D.simulate_template(o1, o1, o1, o1, np.array([0]), np.array([0.01]), 1, 4.0, None, 5, D.CostModel())
    assert r5[0] < r1[0]


def test_discover_skips_member_spans_after_the_last_traded_hour(walk_db):
    """Review fix: a member span that starts after the symbol's delisting (month m ranks on m-1, so a symbol delisted
    late in m-1 can be a member of m) has no decision to enter: its jobs are dropped before any build."""
    feb = (_ms(2024, 2, 1), _ms(2024, 3, 1) - 1)
    with pytest.raises(RuntimeError, match="last traded hour"):
        D.build_dataset(walk_db, ["AUSDT"], _ms(2024, 1, 25), _ms(2024, 3, 10) - 1, "1h", workers=1,
                        windows={"AUSDT": [feb]}, delisted={"AUSDT": _ms(2024, 1, 20) - 1})


def test_discover_mine_reports_membership(walk_db):
    tr = (_ms(2024, 1, 20), _ms(2024, 2, 20) - 1)
    va = (_ms(2024, 2, 20), _ms(2024, 3, 14) - 1)
    member = {"train": {"windows": {"AUSDT": [(_ms(2024, 2, 1), tr[1])]}, "delisted": {}},
              "valid": {"windows": {"AUSDT": [(_ms(2024, 3, 1), va[1])]}, "delisted": {}}}
    res = D.mine(walk_db, ["AUSDT"], tr, va, "1h", D.SearchConfig(min_trades=20, beam=3, depth=1), workers=1,
                 member=member)
    assert res["rows"]["train"] == 19 * 24 and res["rows"]["valid"] == 13 * 24
    m = res["membership"]
    assert m["train"]["member_months"] == {"AUSDT": 1} and m["valid"]["member_months_total"] == 1
    assert m["delisting_close_rows"] == {"1h": {"train": 0, "valid": 0}}


def test_member_windows_for_a_frozen_list(tmp_path):
    """discover's VALID membership: the TRAIN-frozen list, each symbol only in the months it is top N there (BIG
    January and February; GONE, delisted in December, nowhere; NEW is not in the frozen list)."""
    from tests.test_f0_universe import M, make_universe_db

    st = make_universe_db(tmp_path / "universe.db")
    st.close()
    res = U.member_windows_for(tmp_path / "universe.db", ["BIGUSDT", "GONEUSDT"], M("2024-01"), M("2024-03") - 1, 2)
    assert res["windows"] == {"BIGUSDT": [(M("2024-01"), M("2024-03") - 1)], "GONEUSDT": []}
    assert res["delisted"] == {}
    dec = U.member_windows_for(tmp_path / "universe.db", ["GONEUSDT"], M("2023-12"), M("2024-01") - 1, 2)
    assert dec["windows"]["GONEUSDT"] == [(M("2023-12"), M("2024-01") - 1)] and "GONEUSDT" in dec["delisted"]
    # review fix: GONE is a December member but stopped trading on 12-10; a window starting after that keeps its span
    # and now reports the earlier delisting, so the span is not traded on the flat files that follow the last trade
    late = U.member_windows_for(tmp_path / "universe.db", ["GONEUSDT"], M("2023-12") + 14 * DAY, M("2024-01") - 1, 2)
    assert late["windows"]["GONEUSDT"] == [(M("2023-12") + 14 * DAY, M("2024-01") - 1)]
    assert late["delisted"]["GONEUSDT"] < M("2023-12") + 14 * DAY
    res = U.resolve_for_run(tmp_path / "universe.db", tmp_path / "no_candles.db", M("2023-12") + 14 * DAY,
                            M("2024-01") - 1, 2)
    assert "GONEUSDT" in res["research"] and res["delisted"]["GONEUSDT"] == late["delisted"]["GONEUSDT"]
    assert U.member_months({"A": [(M("2024-01") + 5, M("2024-03") - 1)]}) == {"A": 2}
    with pytest.raises(ValueError, match="cannot be ranked"):
        U.member_windows_for(tmp_path / "universe.db", ["BIGUSDT"], M("2023-10"), M("2023-11") - 1, 2)

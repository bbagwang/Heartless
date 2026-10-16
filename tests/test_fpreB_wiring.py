"""F-pre-B integration (wave-f-merge): the connections F-pre-B and the red-team fixes left to the owners of each
other's files.

- book guard: engine.usable_book is exchange/orderbook.usable_book, coverage flag included (test_fpreB_book_guard)
- S3: the offline schedule takes each hour's members from context.pit_members (one definition with the research
  context); the live detector's incremental choice is pinned to it (test_fpreB_shock)
- neutral evaluator: the shock layer is evaluator policy (outside the config hash, as in the hourly evaluator), and
  `heartless neutral` applies it like `heartless hourly` (schedule from S0_LEAD_MS before the window, --tag refused
  with SHOCK_LAYER=false, the layer recorded on the result and the trial line)
- lab: the trial configuration hashes the execution mode the run trades (params exec, then EXEC_MODE), the stress run
  fills maker limits through the whole spread, --tag needs the shock layer, the F-pre-B switches reach every job and
  mark the ledger line, a job without BTC reads S0 from the run's own candle database
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from synth import synth_candles

from heartless.config import Settings
from heartless.core.store import Store
from heartless.learning import lab, trials
from heartless.learning import neutral as N
from heartless.strategy import shock as SH
from heartless.strategy.params import StrategyParams
from tests.test_fpreA_neutral import PerpPair, Scripted, _close, _ctx, _run, ledger, neutral_db  # noqa: F401
from tests.test_fpreB_shock import _btc_minutes, _universe

H = SH.MS_H
DAY = SH.MS_D


# --- neutral evaluator -----------------------------------------------------------------------------------------------
def test_the_neutral_shock_filter_is_evaluator_policy():
    """A halted close vetoes the entry; the layer stays out of the config hash (a carry / pairs configuration rerun with
    the layer keeps its hash and N), a filter of the configuration's own is part of it."""
    strat = Scripted(enter=_close(2), leave=_close(10))
    flt = SH.EntryFilter(SH.ShockEvents(hourly={"S1": [_close(2)]}))
    halted = N.evaluate(strat, _ctx().start, _ctx().end, ctx=_ctx(), capital=10_000.0, entry_filter=flt)
    free = _run(Scripted(enter=_close(2), leave=_close(10)), _ctx())
    assert len(free.positions) == 1 and halted.positions == [] and flt.blocked == 1
    assert halted.config == free.config and "entry_filter" not in halted.config
    assert trials.config_hash(halted.config) == trials.config_hash(free.config)

    def own_rule(sym, T):
        return True

    mine = N.evaluate(strat, _ctx().start, _ctx().end, ctx=_ctx(), capital=10_000.0, entry_filter=own_rule)
    assert mine.config["entry_filter"] == "own_rule" and len(mine.positions) == 1


def _neutral_args(out, tag):
    return SimpleNamespace(strategy="tests.test_fpreA_neutral:PerpPair", set=["hours=5"], period=None,
                           start="2024-01-02", end="2024-01-05", universe=None, symbols="AUSDT,1000XUSDT",
                           capital=1000.0, stress=False, tag=tag, out=str(out), argv=["heartless", "neutral"])


def test_the_neutral_cli_applies_the_shock_layer(neutral_db, tmp_path, monkeypatch, capsys):
    from heartless.learning import hourly

    out = tmp_path / "n.json"
    start, end = hourly.resolve_window(_neutral_args(out, None))
    asked = []

    def schedule(settings, a, b, *x, **k):  # an S1 firing at the window's first close halts it and the next one
        asked.append((a, b))
        return SH.ShockSchedule(a, b, hourly={"S1": [start]}, meta={"missing": []})

    monkeypatch.setattr(SH, "schedule_for", schedule)
    on = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=neutral_db)
    N.main_cli(on, _neutral_args(out, "pairs_neutral"))
    doc = json.loads(out.read_text())
    assert asked == [(start - SH.S0_LEAD_MS, end)]
    assert doc["positions"][0]["opened"] == N._iso(start + 2 * H)  # the first entry after the 2-hour halt
    assert doc["shock"]["blocked_decisions"] == 2 and doc["shock"]["firings"]["S1"] == 1
    assert doc["trial"]["shock"] == doc["shock"] and "entry_filter" not in doc["config"]
    assert "shock layer: 2 candidate entries blocked" in capsys.readouterr().err
    # without the layer (untagged): the first close is taken, the configuration hash is the same
    off = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=neutral_db, SHOCK_LAYER=False)
    N.main_cli(off, _neutral_args(out, None))
    free = json.loads(out.read_text())
    assert free["shock"] is None and free["positions"][0]["opened"] == N._iso(start)
    assert trials.config_hash(free["config"]) == doc["trial"]["config_hash"]
    # a trial runs with the layer: --tag with SHOCK_LAYER=false is refused before anything is evaluated or recorded
    n_lines = len(trials.load())
    with pytest.raises(SystemExit) as e:
        N.main_cli(off, _neutral_args(tmp_path / "x.json", "pairs_neutral"))
    assert e.value.code == 2 and "SHOCK_LAYER=false runs only without --tag" in capsys.readouterr().err
    assert len(trials.load()) == n_lines and not (tmp_path / "x.json").exists()


def test_an_exempt_neutral_strategy_takes_no_filter(neutral_db, tmp_path):
    out = tmp_path / "n.json"
    args = SimpleNamespace(**{**vars(_neutral_args(out, "pairs_neutral")),
                              "strategy": "tests.test_fpreB_wiring:ExemptPair"})
    N.main_cli(Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=neutral_db, SHOCK_LAYER=False),
               args)
    assert json.loads(out.read_text())["trial"]["family"] == "pairs_neutral"
    filt, doc = SH.evaluator_filter(Settings(_env_file=None, UNIVERSE_DB=neutral_db), ExemptPair(), 0, DAY)
    assert filt is None and doc["exempt"] is True


class ExemptPair(PerpPair):
    shock_exempt = True


# --- S3 --------------------------------------------------------------------------------------------------------------
def test_the_offline_s3_takes_its_members_from_context_pit_members(tmp_path, monkeypatch):
    """cross_section_series asks context.pit_members for every hour it evaluates (a month at a time) and uses exactly
    those members: removing one member there changes the statistic, and the untouched result equals the live
    detector's incremental cross-section at every hour."""
    from heartless.data import context as C

    ct, c = _btc_minutes()
    udb, _ = _universe(tmp_path, ct, c)
    hours, hc = SH.btc_hourly_from_minutes(ct, c)
    calls = []
    real = C.pit_members

    def spy(store, h_a, h_b, top_n):
        calls.append((h_a, h_b, top_n))
        return real(store, h_a, h_b, top_n)

    monkeypatch.setattr(C, "pit_members", spy)
    st = Store(udb, readonly=True)
    try:
        series = SH.cross_section_series(st, hours, hc)
        # every hour of the grid is covered once, in chunks of at most PIT_CHUNK_HOURS, with S3's top 40
        covered = [h for a, b, _ in calls for h in range(a, b + 1)]
        assert covered == [int(h) // H - 1 for h in hours] and all(n == SH.S3_TOP for _, _, n in calls)
        assert max(b - a + 1 for a, b, _ in calls) <= SH.PIT_CHUNK_HOURS and len(calls) >= 2
        # the live detector's own member choice gives the same bits
        xs, live = SH._CrossSection(), []
        btc = [float("nan")] + [SH.log_return(float(hc[k]), float(hc[k - 1])) for k in range(1, len(hours))]
        for h, bars in SH.stored_cross_sections(st, int(hours[0]) - SH.KEEP_HOURS * H, int(hours[-1])):
            for sym, (cl, q) in bars.items():
                xs.push(h, sym, cl, q)
            k = int((h - hours[0]) // H)
            if h >= hours[0]:
                live.append(xs.value(h, btc[max(0, k - SH.S3_HOURS + 1):k + 1]))
            if h % DAY == 0:
                xs.drop_stale(h)
        assert np.isfinite(series).sum() > 100
        assert all((math.isnan(a) and math.isnan(b)) or a == b for a, b in zip(series, live))

        def fewer(store, h_a, h_b, top_n):  # drop the most liquid member wherever it is chosen
            syms, mask = real(store, h_a, h_b, top_n)
            mask = mask.copy()
            j = syms.index("BTCUSDT")
            mask[:, j] = False
            return syms, mask

        monkeypatch.setattr(C, "pit_members", fewer)
        other = SH.cross_section_series(st, hours, hc)
    finally:
        st.close()
    fin = np.isfinite(series) & np.isfinite(other)
    assert fin.any() and not np.array_equal(series[fin], other[fin])


# --- lab -------------------------------------------------------------------------------------------------------------
def test_stress_settings_fill_maker_limits_through_the_whole_spread():
    s = Settings(_env_file=None, **lab.STRESS_SETTINGS)
    assert s.maker_through_spread == 1.0 and Settings(_env_file=None).maker_through_spread == 0.5
    assert (s.taker_fee, s.maker_fee, s.backtest_slippage_bps) == (0.00075, 0.0003, 3.0)


def test_the_lab_trial_config_hashes_the_exec_mode_the_run_trades():
    params = StrategyParams.default()
    base = trials.lab_config(params.to_dict(), "htf_trend", ["BTCUSDT"])
    taker = lab.run_config(Settings(_env_file=None), params, "htf_trend", ["BTCUSDT"])
    assert taker == base and "exec" not in taker  # taker runs hash exactly as before
    m1 = lab.run_config(Settings(_env_file=None, EXEC_MODE="maker_m1"), params, "htf_trend", ["BTCUSDT"])
    assert m1["exec"] == {"htf_trend": "maker_m1"} and trials.config_hash(m1) != trials.config_hash(base)
    named = lab.run_config(Settings(_env_file=None, EXEC_MODE="xs_momentum:maker_m2"), params, "htf_trend", ["BTCUSDT"])
    assert named == base  # a mode for another alpha leaves this run taker
    # the parameter set's exec comes first (the engine's precedence)
    params.exec = {"htf_trend": "maker_m2"}
    both = lab.run_config(Settings(_env_file=None, EXEC_MODE="maker_m1"), params, "htf_trend", ["BTCUSDT"])
    assert both["exec"] == {"htf_trend": "maker_m2"}
    # every enabled alpha of a joint run
    joint = StrategyParams.default()
    joint.enabled = {"htf_trend": True, "xs_momentum": True}
    cfg = lab.run_config(Settings(_env_file=None, EXEC_MODE="htf_trend:maker_m1"), joint, None, ["BTCUSDT"])
    assert cfg["alphas"] == ["htf_trend", "xs_momentum"] and cfg["exec"] == {"htf_trend": "maker_m1"}


def _lab_args(**kw):
    base = dict(symbols="BTCUSDT", years=None, start="2024-01-01", end="2024-02-01", split="train", unseal=[],
                params=None, set=[], stress=False, meta=False, chunk_days=None, alpha="htf_trend", workers=1,
                json=False, trades_out=None, out=None, by_side=False, tag=None, argv=["heartless", "lab"])
    base.update(kw)
    return SimpleNamespace(**base)


def _stub(start, end):
    return lab.LabResult("v-test", None, start, end, {**lab.summarize([])}, {}, {}, {}, chunks=[(start, end)])


def test_the_lab_cli_hands_the_engine_settings_to_the_jobs_and_records_them(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), BOOK_GUARD=False, EXEC_MODE="maker_m1",
                        UNIVERSE_DB=str(tmp_path / "u.db"))
    Store(settings.db_path).close()
    seen = []
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k: seen.append(k) or _stub(start, end))
    out = tmp_path / "r.json"
    lab.main_cli(settings, _lab_args(tag="fpreb_lab", stress=True, out=str(out)))
    job = seen[-1]["settings"]
    assert job["SHOCK_LAYER"] is True and job["BOOK_GUARD"] is False and job["EXEC_MODE"] == "maker_m1"
    assert job["UNIVERSE_DB"] == str(tmp_path / "u.db") and "SHOCK_CANDLE_DB" not in job
    assert job["MAKER_THROUGH_SPREAD"] == 1.0 and job["TAKER_FEE"] == lab.STRESS_SETTINGS["TAKER_FEE"]
    line = trials.load()[-1]
    assert line["family"] == "fpreb_lab" and line["exec"] == {"htf_trend": "maker_m1"}
    assert line["engine"] == {"SHOCK_LAYER": True, "BOOK_GUARD": False, "EXEC_MODE": "maker_m1"}
    # no BTC anywhere here: the run says which rules it could not evaluate, and the line keeps it
    assert line["shock"]["missing"] == ["S0", "S1", "S2", "S3"] and line["shock"]["exempt"] is False
    assert "S0, S1, S2, S3 not evaluated" in capsys.readouterr().err
    doc = json.loads(out.read_text())
    assert doc["engine"] == job_engine(job) and doc["shock"] == line["shock"]
    assert doc["settings_overrides"] == lab.STRESS_SETTINGS  # the cost stress only, as before
    # a trial runs with the shock layer: --tag with SHOCK_LAYER=false is refused before any job
    off = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), SHOCK_LAYER=False)
    n_jobs, n_lines = len(seen), len(trials.load())
    with pytest.raises(SystemExit) as e:
        lab.main_cli(off, _lab_args(tag="fpreb_lab"))
    assert e.value.code == 2 and "SHOCK_LAYER=false runs only without --tag" in capsys.readouterr().err
    assert len(seen) == n_jobs and len(trials.load()) == n_lines
    lab.main_cli(off, _lab_args())  # untagged runs without the layer still run
    assert seen[-1]["settings"]["SHOCK_LAYER"] is False and len(trials.load()) == n_lines


def job_engine(job):
    return {k: job[k] for k in ("SHOCK_LAYER", "BOOK_GUARD", "EXEC_MODE", "UNIVERSE_DB")}


def test_a_lab_job_without_btc_reads_s0_from_the_runs_candle_database(tmp_path, monkeypatch):
    """A job's Settings come from the environment (no .env); without SHOCK_CANDLE_DB its S0 reads the BTC bars of the
    candle database the run evaluates, not <data_dir>/heartless.db of whatever the environment says."""
    db = tmp_path / "c.db"
    st = Store(db)
    start = 1_704_067_200_000 - 3 * DAY  # 2023-12-29
    for sym, seed in (("BTCUSDT", 3), ("ETHUSDT", 4)):
        ca = synth_candles(5 * 1440, seed=seed, start_ms=start)
        st.save_candles(sym, [ca.candle_at(i) for i in range(ca.n)])
    st.close()
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path / "elsewhere"))
    monkeypatch.delenv("SHOCK_CANDLE_DB", raising=False)
    got = []
    real = SH.schedule_for

    def spy(settings, a, b, btc_1m=None, progress=None):
        got.append((str(settings.shock_candle_db_path), btc_1m is None))
        return real(settings, a, b, btc_1m=btc_1m, progress=progress)

    monkeypatch.setattr(SH, "schedule_for", spy)
    SH._CACHE.clear()
    inline = SimpleNamespace(map=map)
    res = lab.evaluate(StrategyParams.default(), ["ETHUSDT"], start + DAY, start + 4 * DAY, str(db), only_alpha="htf_trend",
                       pool=inline, settings={"SHOCK_LAYER": True}, warmup_ms=DAY, chunk_days=0)
    assert res.errors == {} and got == [(str(db), True)]
    # SHOCK_CANDLE_DB, when given, is kept
    got.clear()
    SH._CACHE.clear()
    lab.evaluate(StrategyParams.default(), ["ETHUSDT"], start + DAY, start + 4 * DAY, str(db), only_alpha="htf_trend",
                 pool=inline, settings={"SHOCK_LAYER": True, "SHOCK_CANDLE_DB": str(tmp_path / "other.db")},
                 warmup_ms=DAY, chunk_days=0)
    assert got == [(str(tmp_path / "other.db"), True)]

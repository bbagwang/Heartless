"""F-pre-A hourly directional evaluator (heartless/learning/hourly.py) on synthetic universe databases: fills and R
like the lab, stop before target in one hour, gaps, time exits, one position per symbol, the funding time convention,
delisting closes, membership windows, daily equity, the look-ahead audit, the ledger line, and the reference alpha
matching the lab on synthetic 1m data."""
import math

import numpy as np
import pytest

from heartless.core.store import Store
from heartless.data import symbols_inventory as SI
from heartless.learning import hourly as Hr
from heartless.learning import lab, seal, trials
from tests.test_f0_universe import _inv

H = 3_600_000
D = 86_400_000
M = SI._month_ms
T0 = M("2024-01")
HALF, SLIP, FEE = 0.5e-4, 1.5e-4, 5e-4


@pytest.fixture
def registry_guard():
    """lab.register_alpha mutates the process's registry; real lab runs do it in their own worker processes, the
    in-process tests here restore it so later tests see the shipped 16 alphas."""
    from heartless.strategy import alphas as A
    from heartless.strategy import params as P

    saved = (list(A.ALL_ALPHAS), dict(A.ALPHA_BY_NAME), dict(P.ALPHA_SPECS))
    yield
    A.ALL_ALPHAS[:] = saved[0]
    A.ALPHA_BY_NAME.clear()
    A.ALPHA_BY_NAME.update(saved[1])
    P.ALPHA_SPECS.clear()
    P.ALPHA_SPECS.update(saved[2])


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "p2_trials.jsonl"
    monkeypatch.setattr(trials, "LEDGER", path)
    return path


def put_symbol(st: Store, symbol: str, start: int, c, o=None, h=None, lo=None, qv=None, funding=None) -> None:
    """Hourly blocks of `symbol` from bar arrays (prices with at most 4 decimals are stored exactly); funding =
    {bar index: rate settled at that bar's open}."""
    c = np.asarray(c, float)
    n = len(c)
    o = c.copy() if o is None else np.asarray(o, float)
    h = np.maximum(o, c) if h is None else np.asarray(h, float)
    lo = np.minimum(o, c) if lo is None else np.asarray(lo, float)
    qv = np.full(n, 1e6) if qv is None else np.asarray(qv, float)
    t = start + np.arange(n, dtype=np.int64) * H
    blocks = []
    for m in sorted({SI.month_str(int(x)) for x in t}):
        sel = [i for i in range(n) if SI.month_str(int(t[i])) == m]
        rows = [[str(int(t[i])), f"{o[i]:.4f}", f"{h[i]:.4f}", f"{lo[i]:.4f}", f"{c[i]:.4f}", "1",
                 str(int(t[i]) + H - 1), f"{qv[i]:.4f}", "10", "0", f"{qv[i] * 0.5:.4f}", "0"] for i in sel]
        fr = [(int(t[i]), f"{funding[i]:.8f}") for i in sel if funding and i in funding]
        blob, s = SI.encode_month(rows, m, fr)
        blocks.append((symbol, M(m), s["n"], s["first_ts"], s["last_ts"], s["first_trade"], s["last_trade"],
                       s["med_dqv"], s["days_traded"], blob))
    st.save_kline_blocks(SI.FUT_TABLE, blocks)


def make_db(path, symbols: dict, first_month="2023-11", last_month="2024-03") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    st = Store(path)
    st.save_symbol_inventory([_inv(s, first_month, last_month) for s in symbols])
    for s, kw in symbols.items():
        put_symbol(st, s, **kw)
    st.refresh_inventory_trades()
    st.close()
    return str(path)


class Plan(Hr.HourlyAlpha):
    """Decides exactly what `plan` says: {close T: (side, stop, target or None, hold)}."""
    name = "plan"
    family = "test_hourly"
    defaults = {"plan": ()}
    max_hold_hours = 72
    warmup_hours = 2

    def decide(self, h):
        for T, side, stop, target, hold in self.params["plan"]:
            if int(h.close_time[-1]) == T:
                return Hr.Entry(side, stop, target, hold)
        return None


def _bar_close(k):
    return T0 + (k + 1) * H


def _flat(n=24 * 10, price=100.0):
    return np.full(n, price)


def _one(tmp_path, plan, n=24 * 10, **bars):
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": bars.pop("c", _flat(n)), **bars}})
    return Hr.evaluate(Plan(plan=tuple(plan)), T0, T0 + n * H - 1, db, symbols=["AUSDT"])


def test_target_fill_and_r_like_the_engine(tmp_path):
    c = _flat()
    h = c.copy()
    h[12] = 103.5
    r = _one(tmp_path, [(_bar_close(10), 1, 98.0, 103.0, 10)], c=c, h=h)
    t = r.trades[0]
    entry = 100.0 * (1 + HALF) * (1 + SLIP)
    ex = 103.0 * (1 - SLIP)
    risk = (entry - 98.0) + entry * (2 * FEE + 0.0003)
    assert t["entry_price"] == pytest.approx(entry, rel=1e-14) and t["exit_price"] == pytest.approx(ex, rel=1e-14)
    assert t["r_multiple"] == pytest.approx(((ex - entry) - (entry + ex) * FEE) / risk, rel=1e-12)
    assert t["exit_reason"] == "target" and t["entry_time"] == _bar_close(10) and t["bars_held"] == 2
    assert t["pnl"] == pytest.approx(t["r_multiple"] * Hr.RISK_AMOUNT)


def test_stop_comes_first_when_both_are_touched_in_one_hour(tmp_path):
    c = _flat()
    h, lo = c.copy(), c.copy()
    h[13], lo[13] = 104.0, 97.0
    t = _one(tmp_path, [(_bar_close(10), 1, 98.0, 103.0, 10)], c=c, h=h, lo=lo).trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(98.0 * (1 - SLIP), rel=1e-14)
    # a short: the high is its adverse extreme
    t = _one(tmp_path / "s", [(_bar_close(10), -1, 102.0, 97.0, 10)], c=c, h=h, lo=lo).trades[0]
    assert t["exit_reason"] == "stop" and t["side"] == "SHORT"
    assert t["exit_price"] == pytest.approx(102.0 * (1 + SLIP), rel=1e-14)


def test_a_gap_through_the_stop_fills_at_the_open_book(tmp_path):
    c = _flat()
    c[3] = 100.01  # the prices show a 0.01 tick (stops are rounded to the inferred tick)
    o = c.copy()
    o[12], c[12] = 95.0, 95.0
    t = _one(tmp_path, [(_bar_close(10), 1, 98.0, None, 10)], c=c, o=o).trades[0]
    assert t["exit_reason"] == "stop" and t["exit_price"] == pytest.approx(95.0 * (1 - HALF) * (1 - SLIP), rel=1e-14)


def test_time_exit_and_one_position_at_a_time(tmp_path):
    """Hold 5 h from the close of bar 10: out at the close of bar 15. A decision at bar 15 (the exit bar) is not
    taken; one at bar 16 is."""
    plan = [(_bar_close(10), 1, 90.0, None, 5), (_bar_close(12), 1, 90.0, None, 5),
            (_bar_close(15), 1, 90.0, None, 5), (_bar_close(16), 1, 90.0, None, 5)]
    r = _one(tmp_path, plan)
    assert [t["entry_time"] for t in r.trades] == [_bar_close(10), _bar_close(16)]
    t = r.trades[0]
    assert t["exit_reason"] == "time" and t["exit_time"] == _bar_close(15) - 1 and t["bars_held"] == 5
    assert t["exit_price"] == pytest.approx(100.0 * (1 - HALF) * (1 - SLIP), rel=1e-14)
    assert r.decisions == 4
    with pytest.raises(ValueError, match="max_hold_hours"):
        _one(tmp_path / "x", [(_bar_close(10), 1, 90.0, None, 500)])


def test_funding_time_convention(tmp_path):
    """Settlements on the opens of bars 11 (= the entry close), 14 and 18 (= the exit close of a 7 h hold): 11 and 14
    are charged at the close before them, 18 is not."""
    n = 24 * 4
    c = 100.0 + np.arange(n) * 0.01
    fund = {11: 0.0004, 14: 0.0002, 18: 0.0008}
    t = _one(tmp_path, [(_bar_close(10), 1, 90.0, None, 7)], n=n, c=c, funding=fund).trades[0]
    entry = c[10] * (1 + HALF) * (1 + SLIP)
    ex = c[17] * (1 - HALF) * (1 - SLIP)
    fund_unit = -(0.0004 * c[10] + 0.0002 * c[13])  # a long pays a positive rate
    risk = (entry - 90.0) + entry * (2 * FEE + 0.0003)
    assert t["r_multiple"] == pytest.approx(((ex - entry) - (entry + ex) * FEE + fund_unit) / risk, rel=1e-12)
    assert t["funding"] == pytest.approx(fund_unit / risk * Hr.RISK_AMOUNT, rel=1e-12)
    # the bars a decision sees: the rate settled at the open of bar 11 is known from the close of bar 11
    st = Hr.ReadOnlyStore(tmp_path / "u.db")
    bars = Hr.load_bars(st, "AUSDT", T0, T0 + n * H)
    st.close()
    assert math.isnan(bars.funding_known()[10]) and bars.funding_known()[11] == 0.0004


def test_delisting_close_and_no_decision_on_the_last_traded_bar(tmp_path):
    """BUSDT trades until bar 40 and then only publishes flat zero-volume files (the archive's habit); the store's
    data run on to the end of March with AUSDT, so BUSDT is delisted: a position open at bar 40 leaves at its close with
    5x the slippage, and the decision at bar 40 itself is not taken."""
    n_a = 24 * 90
    qv = np.full(24 * 20, 1e6)
    qv[41:] = 0.0
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": _flat(n_a)},
                                     "BUSDT": {"start": T0, "c": _flat(24 * 20, 50.0), "qv": qv}})
    plan = [(_bar_close(36), 1, 45.0, None, 24)]
    r = Hr.evaluate(Plan(plan=tuple(plan)), T0, T0 + 30 * D - 1, db, symbols=["BUSDT"])
    t = r.trades[0]
    assert t["exit_reason"] == "delisting" and t["exit_time"] == _bar_close(40) - 1
    assert t["exit_price"] == pytest.approx(50.0 * (1 - HALF) * (1 - 5 * SLIP), rel=1e-14)
    # under stress the delisting slippage is 5 x the doubled slippage (10x nominal)
    rs = Hr.evaluate(Plan(plan=tuple(plan)), T0, T0 + 30 * D - 1, db, symbols=["BUSDT"], stress=True)
    assert rs.trades[0]["exit_price"] == pytest.approx(50.0 * (1 - HALF) * (1 - 10 * SLIP), rel=1e-14)
    assert rs.summary()["delisting_closes"] == 1 and rs.stress
    r2 = Hr.evaluate(Plan(plan=((_bar_close(40), 1, 45.0, None, 5),)), T0, T0 + 30 * D - 1, db, symbols=["BUSDT"])
    assert r2.trades == [] and r2.decisions == 0


def test_top_universe_counts_entries_only_in_member_months(tmp_path):
    """top1: AUSDT leads December's volume (member in January), BUSDT January's (member in February). A daily 48 h
    hold entered on the last day of January keeps running into February (membership lost, exit rules apply)."""
    start = M("2023-12")
    n = 24 * (31 + 31 + 29)
    qa, qb = np.full(n, 1e6), np.full(n, 1e5)
    qa[24 * 31:24 * 62], qb[24 * 31:24 * 62] = 1e5, 1e6  # January: B leads
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": start, "c": _flat(n), "qv": qa},
                                     "BUSDT": {"start": start, "c": _flat(n, 50.0), "qv": qb}},
                 first_month="2023-10", last_month="2024-02")

    class Daily(Hr.HourlyAlpha):
        name, family, max_hold_hours, warmup_hours = "daily", "test_hourly", 48, 2

        def decide(self, h):
            T = int(h.close_time[-1])
            return Hr.Entry(1, float(h.close[-1]) * 0.5, None, 48) if (T // H) % 24 == 1 else None

    r = Hr.evaluate(Daily(), M("2024-01"), M("2024-03") - 1, db, top=1)
    months = {(t["symbol"], SI.month_str(t["entry_time"])) for t in r.trades}
    assert months == {("AUSDT", "2024-01"), ("BUSDT", "2024-02")}
    last_a = max((t for t in r.trades if t["symbol"] == "AUSDT"), key=lambda t: t["entry_time"])
    assert SI.month_str(last_a["exit_time"]) == "2024-02" and last_a["exit_reason"] == "time"
    assert r.membership["member_months"] == {"AUSDT": 1, "BUSDT": 1}
    assert r.config["universe"] == "top1"


def test_daily_equity_is_continuous(tmp_path):
    n = 24 * 6
    c = 100.0 + np.sin(np.arange(n) / 5.0)
    plan = [(_bar_close(30), 1, 95.0, None, 40), (_bar_close(80), -1, 105.0, None, 30)]
    r = _one(tmp_path, plan, n=n, c=np.round(c, 4))
    assert len(r.daily_r) == 6 and r.daily()["days"] == 6
    assert sum(r.daily_r.values()) == pytest.approx(sum(t["r_multiple"] for t in r.trades), abs=1e-12)
    assert r.daily_r[T0 // D] == 0.0  # before the first entry
    series = r.daily_series()
    assert series[-1][2] == pytest.approx(sum(t["r_multiple"] for t in r.trades), abs=1e-6)
    # the first trade (day 1 to day 2) is marked at the day-1 close, not booked at its exit only
    assert r.daily_r[T0 // D + 1] != 0.0 and r.trades[0]["exit_time"] // D == T0 // D + 2


def test_statistics_are_the_labs(tmp_path):
    n = 24 * 40
    rng = np.random.default_rng(5)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), 4)
    plan = [(_bar_close(k), 1 if k % 2 else -1, float(c[k]) * (0.97 if k % 2 else 1.03), None, 6)
            for k in range(20, n - 10, 13)]
    r = _one(tmp_path, plan, n=n, c=c)
    g = lab.group_stats(r.trades)
    s = r.summary()
    assert s["n"] == g["n"] and s["t_day"] == g["t_day"] and s["avg_r"] == pytest.approx(g["avg_r"], abs=1e-4)
    assert set(r.by_side()) == {"LONG", "SHORT"} and set(r.by_year()) == {"2024"}
    assert sum(v["n"] for v in r.by_symbol().values()) == len(r.trades)
    doc = r.to_json()
    assert doc["evaluator"] == "hourly" and doc["summary"]["n"] == len(r.trades) and doc["daily"]["days"] == 40


def test_audit_lookahead_catches_a_peeking_alpha(tmp_path):
    n = 24 * 5
    rng = np.random.default_rng(1)
    c = np.round(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))), 4)
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": c}})
    st = Hr.ReadOnlyStore(db)
    bars = Hr.load_bars(st, "AUSDT", T0, T0 + n * H)
    st.close()

    class Peek(Hr.HourlyAlpha):
        name = "peek"

        def decide(self, h):
            return None

        def decide_all(self, h):
            d = Hr.Decisions.empty(h.n)
            up = np.zeros(h.n, dtype=bool)
            up[:-1] = h.close[1:] > h.close[:-1]  # the NEXT close: look-ahead
            d.side[up] = 1
            d.stop[up] = h.close[up] * 0.9
            d.max_hold[up] = 3
            return d

    assert len(Hr.audit_lookahead(Peek(), bars, range(10, n))) > 10
    ref = Hr.ReferenceAlpha()
    assert Hr.audit_lookahead(ref, bars, range(20, n)) == []
    assert any(ref.decide(bars.upto(i)) for i in range(20, n))


def test_ledger_line_reruns_and_refusals(tmp_path, monkeypatch):
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": _flat()}})
    plan = ((_bar_close(10), 1, 98.0, None, 5),)
    r = Hr.evaluate(Plan(plan=plan), T0, T0 + 10 * D - 1, db, symbols=["AUSDT"])
    e = Hr.record(r)
    assert e["tool"] == "hourly" and e["family"] == "test_hourly" and e["wave_f"] == "TRAIN-B" and e["exec"] == "taker"
    assert e["config"]["tool"] == "hourly" and e["config"]["alpha"] == "plan" and "start" not in e["config"]
    Hr.record(Hr.evaluate(Plan(plan=plan), T0, T0 + 10 * D - 1, db, symbols=["AUSDT"], stress=True))
    assert trials.family_tests("test_hourly") == 1  # a stress rerun of the same configuration
    Hr.record(Hr.evaluate(Plan(plan=((_bar_close(11), 1, 98.0, None, 5),)), T0, T0 + 10 * D - 1, db,
                          symbols=["AUSDT"]))
    assert trials.family_tests("test_hourly") == 2
    with pytest.raises(ValueError, match="lab"):
        Hr.evaluate(Plan(plan=plan), T0, T0 + D, db, symbols=["AUSDT"], exec="maker")
    monkeypatch.setattr(seal, "SEALED", {"X": (T0 + D, T0 + 2 * D)})
    with pytest.raises(seal.SealedError):
        Hr.evaluate(Plan(plan=plan), T0, T0 + 3 * D, db, symbols=["AUSDT"])
    with pytest.raises(ValueError, match="unknown parameter"):
        Plan(nope=1)


def test_hourly_cli_writes_the_result_and_the_ledger_line(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from heartless.config import Settings

    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": np.round(100 + np.sin(np.arange(240) / 3), 4)}})
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=db)
    out = tmp_path / "r.json"
    args = SimpleNamespace(period=None, start="2024-01-02", end="2024-01-09", universe=None, symbols="AUSDT",
                           equivalence=False, alpha="heartless.learning.hourly:ReferenceAlpha", set=["hold_h=6"],
                           stress=False, tag="evaluator_test", trades_out=None, out=str(out), workers=1,
                           argv=["heartless", "hourly"])
    Hr.main_cli(settings, args)
    import json

    doc = json.loads(out.read_text())
    assert doc["config"]["params"]["hold_h"] == 6 and doc["wave_f_period"] == "TRAIN-B"
    assert doc["trial"]["family"] == "evaluator_test" and doc["summary"]["n"] == 7
    assert "trial ledger: evaluator_test" in capsys.readouterr().err
    assert Hr.resolve_window(SimpleNamespace(period="FWD-02")) == trials.fwd_span(2)
    assert Hr.resolve_window(SimpleNamespace(period="TRAIN-F")) == (trials.WAVE_F["TRAIN-F"][0],
                                                                    trials.WAVE_F["TRAIN-F"][1] - 1)


def test_tick_inference_and_rounding():
    assert Hr.infer_tick(np.array([100.0, 100.01, 100.05, 101.0])) == 0.01
    assert Hr.infer_tick(np.array([0.14601, 0.14602, 0.1461])) == 1e-05
    assert Hr.infer_tick(np.array([5.0, 5.0])) == 0.01
    assert list(Hr.round_to_tick(np.array([8.0695, 8.0694, 8.0705, np.nan]), 0.001)[:3]) == [8.07, 8.069, 8.07]


def test_reference_alpha_matches_the_lab_on_synthetic_1m_data(tmp_path, registry_guard):
    """The acceptance check in miniature: 1m candles in a candle store, their exact hourly aggregate in a universe
    store; the reference alpha in both evaluators gives the same trades (entry hour, side) and nearly the same R."""
    from synth import synth_candles

    from heartless.core.models import Candle
    from tests.test_p0_lab import INLINE

    days = 20
    ca = synth_candles(days * 1440, seed=11, start_ms=T0)
    rows = []
    for i in range(ca.n):
        k = ca.candle_at(i)
        o, h, l, c = (round(x, 4) for x in (k.open, k.high, k.low, k.close))
        rows.append(Candle(k.open_time, o, max(o, h, c), min(o, l, c), c, k.volume, k.quote_volume, k.trades,
                           k.taker_buy_volume, k.close_time))
    cdb = tmp_path / "c.db"
    st = Store(cdb)
    st.save_candles("AUSDT", rows)
    st.close()
    o = np.array([r.open for r in rows]).reshape(-1, 60)[:, 0]
    h = np.array([r.high for r in rows]).reshape(-1, 60).max(axis=1)
    lo = np.array([r.low for r in rows]).reshape(-1, 60).min(axis=1)
    c = np.array([r.close for r in rows]).reshape(-1, 60)[:, -1]
    udb = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": c, "o": o, "h": h, "lo": lo}})
    a, b = T0 + 12 * D, T0 + days * D - 1
    hres = Hr.evaluate(Hr.ReferenceAlpha(), a, b, udb, symbols=["AUSDT"])
    from heartless.strategy.params import StrategyParams

    lres = lab.evaluate(StrategyParams.default(), ["AUSDT"], a, b, str(cdb), only_alpha="hourly_reference",
                        pool=INLINE, extra_alphas=["heartless.learning.hourly:ReferenceEngineAlpha"])
    cmp = Hr.compare_trades(hres.trades, lres.trades)
    assert cmp["n_hourly"] == 8 and cmp["matched_share"] == 1.0
    assert cmp["mean_abs_r_diff"] < 0.02 and cmp["same_exit_reason_share"] == 1.0
    # the port is registered only in the process that ran the jobs (here this one, INLINE) and outside the shipped
    # registry: it is not an alpha module of heartless/strategy/alphas
    from heartless.strategy import alphas as A

    assert A.ALPHA_BY_NAME["hourly_reference"].__module__ == "heartless.learning.hourly"
    assert lab.register_alpha("heartless.learning.hourly:ReferenceEngineAlpha") == "hourly_reference"  # idempotent
    assert sum(a.name == "hourly_reference" for a in A.ALL_ALPHAS) == 1


def test_optional_hourly_series_plug_in(tmp_path, monkeypatch):
    """OI and the premium index 1h (F-data-h) arrive through OPTIONAL_SERIES loaders, aligned by open time; absent,
    the columns are NaN."""
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": _flat(48)}})
    st = Hr.ReadOnlyStore(db)
    try:
        bars = Hr.load_bars(st, "AUSDT", T0, T0 + 47 * H)
        assert np.isnan(bars.oi).all() and np.isnan(bars.premium).all()
        monkeypatch.setitem(Hr.OPTIONAL_SERIES, "oi", lambda store, s, a, b: (np.array([T0 + 5 * H, T0 + 7 * H]),
                                                                              np.array([1.5, 2.5])))
        bars = Hr.load_bars(st, "AUSDT", T0, T0 + 47 * H)
    finally:
        st.close()
    assert bars.oi[5] == 1.5 and bars.oi[7] == 2.5 and np.isnan(bars.oi[6]) and np.isnan(bars.premium).all()

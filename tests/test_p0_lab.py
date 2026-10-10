"""P0 lab periods: year windows, (symbol x chunk) jobs with boundary flags, per-year / per-side breakdowns, result
files, and --set values that the ParamSpec grid would silently change."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from heartless.config import Settings
from heartless.core.models import EntryStyle, Side, Signal
from heartless.core.store import Store
from heartless.learning import lab
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from synth import synth_candles

DAY = 86_400_000
HOUR = 3_600_000
INLINE = SimpleNamespace(map=map)  # run lab jobs in-process so the patched ensemble applies


def _ms(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def test_parse_when_year_window_and_chunk_windows():
    assert lab.parse_when("2025-01-01", end=True) == _ms(2025, 1, 1) - 1  # date-only end is exclusive
    assert lab.parse_when("2025-01-01") == _ms(2025, 1, 1)
    assert lab.parse_when("2025-01-01T06:30", end=True) == _ms(2025, 1, 1, 6, 30)
    a, b, ys = lab.year_window("2024,2022,2023")
    assert (a, b, ys) == (_ms(2022, 1, 1), _ms(2025, 1, 1) - 1, [2022, 2023, 2024])
    with pytest.raises(ValueError):
        lab.year_window("2022,2024")
    w = lab.chunk_windows(_ms(2022, 1, 13), _ms(2025, 1, 1) - 1)
    assert w == [(_ms(2022, 1, 13), _ms(2023, 1, 1) - 1), (_ms(2023, 1, 1), _ms(2024, 1, 1) - 1),
                 (_ms(2024, 1, 1), _ms(2025, 1, 1) - 1)]
    assert lab.chunk_windows(0, 100 * DAY - 1, 45) == [(0, 45 * DAY - 1), (45 * DAY, 90 * DAY - 1), (90 * DAY, 100 * DAY - 1)]
    assert lab.chunk_windows(5, 10, 0) == [(5, 10)]
    assert lab.chunk_windows(_ms(2026, 1, 13), _ms(2026, 7, 1)) == [(_ms(2026, 1, 13), _ms(2026, 7, 1))]


class Timer(Alpha):
    """Deterministic test alpha: LONG at every 4h close, wide bracket, 25h time stop, so a position is usually open
    at any given instant (in particular at a chunk boundary)."""
    name = "trend_pullback"
    timeframe = "1m"

    def evaluate(self, view, ctx, p):
        if (view.cursor_time + 1) % (4 * HOUR):
            return None
        px = ctx.ticker.mark
        return Signal(self.name, ctx.symbol, Side.LONG, 0.95, "timer", px * 0.85, px * 1.3, None, EntryStyle.MARKET, None,
                      1500, 0.0, px * 0.01, "1m", {"ref_price": px})


@pytest.fixture
def timer_db(tmp_path, monkeypatch):
    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Timer()], only_alpha))
    db = tmp_path / "t.db"
    st = Store(db)
    ca = synth_candles(6 * 1440, seed=3, start_ms=_ms(2023, 12, 28))  # 2023-12-28 .. 2024-01-02 23:59
    st.save_candles("BTCUSDT", [ca.candle_at(i) for i in range(ca.n)])
    st.close()
    params = StrategyParams.default()
    params.enabled["trend_pullback"] = True
    return str(db), params


def _keys(res):
    return [(t["symbol"], t["entry_time"], t["exit_time"], round(t["r_multiple"], 12), t["exit_reason"]) for t in res.trades]


def test_chunk_handoff_reproduces_the_continuous_run(timer_db):
    """Run-off lets the position open at the year end close naturally; the next chunk's burn-in re-creates it so it
    blocks the same entries, and its replica is dropped there: the chunked trade list equals the continuous one."""
    db, params = timer_db
    a, b = _ms(2023, 12, 29), _ms(2024, 1, 3) - 1
    whole = lab.evaluate(params, ["BTCUSDT"], a, b, db, pool=INLINE, chunk_days=0, warmup_ms=DAY)
    split = lab.evaluate(params, ["BTCUSDT"], a, b, db, pool=INLINE, warmup_ms=DAY)  # default: calendar years
    assert whole.errors == {} and split.errors == {} and len(split.chunks) == 2
    assert _keys(split) == _keys(whole) and len(whole.trades) >= 4
    assert split.boundary_trades == [] and set(split.by_year()) == {"2023", "2024"}
    crossing = [t for t in split.trades if t["entry_time"] < _ms(2024, 1, 1) < t["exit_time"]]
    assert len(crossing) == 1 and crossing[0]["chunk"] == "2023"  # owned by the chunk it was opened in


def test_pure_cut_flags_boundary_trades_and_never_double_counts(timer_db):
    db, params = timer_db
    a, b = _ms(2023, 12, 29), _ms(2024, 1, 3) - 1
    whole = lab.evaluate(params, ["BTCUSDT"], a, b, db, pool=INLINE, chunk_days=0, warmup_ms=DAY)
    split = lab.evaluate(params, ["BTCUSDT"], a, b, db, pool=INLINE, warmup_ms=DAY, burn_in_ms=0, runoff_ms=0)
    assert whole.errors == {} and split.errors == {}
    assert len(whole.chunks) == 1 and split.chunks == [(a, _ms(2024, 1, 1) - 1), (_ms(2024, 1, 1), b)]
    assert not any(t["boundary"] for t in whole.trades)
    bnd = split.boundary_trades
    assert len(bnd) == 1 and split.summary()["boundary"] == 1
    t = bnd[0]
    assert t["exit_time"] <= _ms(2024, 1, 1) - 1 and t["exit_reason"] == lab.BOUNDARY_REASON and t["chunk"] == "2023"
    # the same position in the unchunked run: same entry, closed later -> counted once, in the chunk it was opened in
    twin = [w for w in whole.trades if w["entry_time"] == t["entry_time"]]
    assert len(twin) == 1 and twin[0]["exit_time"] > t["exit_time"]
    keys = [(x["symbol"], x["entry_time"]) for x in split.trades]
    assert len(keys) == len(set(keys))
    for x in split.trades:
        lo, hi = split.chunks[0] if x["chunk"] == "2023" else split.chunks[1]
        assert lo - 60_000 <= x["entry_time"] <= hi and x["exit_time"] <= hi
    # the first chunk is a prefix of the unchunked run: every trade closed before the boundary is identical
    early = lambda trades: [(x["entry_time"], x["exit_time"], round(x["r_multiple"], 12)) for x in trades
                            if x["exit_time"] < _ms(2024, 1, 1) and x["exit_reason"] != lab.BOUNDARY_REASON]
    assert early(split.trades) == early(whole.trades) and early(whole.trades)
    assert set(split.by_year()) == {"2023", "2024"}
    excl = split.summary_excl_boundary()
    assert excl["n"] == split.summary()["n"] - 1
    assert sum(v["n"] for v in split.by_year_side().values()) == len(split.trades)
    rep = split.report(by_side=True)
    assert "boundary trades 1" in rep and "year  2023" in rep and "side  LONG" in rep and "ys    2024 LONG" in rep


def test_symbol_listed_inside_the_warmup_starts_after_a_full_warmup(timer_db):
    db, params = timer_db
    res = lab.evaluate(params, ["BTCUSDT"], _ms(2023, 12, 28), _ms(2024, 1, 3) - 1, db, pool=INLINE, chunk_days=0,
                       warmup_ms=2 * DAY)
    assert res.jobs[0]["eff_start"] == "2023-12-30 00:00"
    assert res.trades and min(t["entry_time"] for t in res.trades) >= _ms(2023, 12, 30)


def test_lab_cli_out_file_has_the_full_breakdown(timer_db, tmp_path, monkeypatch, capsys):
    db, params = timer_db
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    monkeypatch.setattr(settings.__class__, "db_path", property(lambda self: db))
    real = lab.evaluate
    monkeypatch.setattr(lab, "evaluate", lambda *a, **k: real(*a, **{**k, "pool": INLINE, "warmup_ms": DAY}))
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(params.to_dict()))
    out = tmp_path / "results" / "r.json"
    args = SimpleNamespace(symbols="BTCUSDT", years=None, start="2023-12-29", end="2024-01-03", split="train", unseal=[],
                           params=str(pfile), set=["trend_pullback.tp1_frac=0.25"], stress=True, meta=False,
                           chunk_days=None, alpha=None, workers=1, json=False, trades_out=None, out=str(out),
                           by_side=True, argv=["heartless", "lab", "--out", str(out)])
    lab.main_cli(settings, args)
    cap = capsys.readouterr()
    assert "side  LONG" in cap.out and "trend_pullback.tp1_frac=0.25" in cap.err  # off-grid override is announced
    doc = json.loads(out.read_text())
    for k in ("summary", "summary_excl_boundary", "by_year", "by_alpha", "by_symbol", "by_month", "by_side", "by_year_side",
              "overrides", "override_adjustments", "settings_overrides", "params_version", "git_commit", "seal", "argv",
              "chunks", "jobs", "window"):
        assert k in doc, k
    assert doc["seal"]["clean"] is True and doc["settings_overrides"]["TAKER_FEE"] == lab.STRESS_SETTINGS["TAKER_FEE"]
    assert doc["override_adjustments"][0]["effective"] == 0.2 and doc["summary"]["boundary"] == 0
    assert set(doc["by_year"]) == {"2023", "2024"} and "2024 LONG" in doc["by_year_side"]


def test_apply_overrides_warns_when_the_grid_changes_a_value(capsys):
    base = StrategyParams.default()
    notes = []
    p = lab.apply_overrides(base, ["htf_trend.trend_z=0.3", "htf_trend.sl_atr=3.25", "htf_trend.max_hold=500",
                                   "sweep_reversal.lookback=100", "ensemble.entry_threshold=0.55"], notes)
    err = capsys.readouterr().err
    assert "htf_trend.trend_z=0.3" in err and "runs as htf_trend.trend_z=0.25" in err
    assert "htf_trend.max_hold=96" in err and "outside [12, 96]" in err
    assert "sweep_reversal.lookback=96" in err
    assert "sl_atr" not in err  # on the grid: silent
    assert p.alphas["htf_trend"]["trend_z"] == 0.25 and p.alphas["htf_trend"]["max_hold"] == 96
    assert {n["key"] for n in notes} == {"htf_trend.trend_z", "htf_trend.max_hold", "sweep_reversal.lookback"}
    # the value stored is the value the worker runs (from_dict clips idempotently)
    assert StrategyParams.from_dict(p.to_dict()).alphas["htf_trend"] == p.alphas["htf_trend"]
    lab.apply_overrides(base, ["intraday_levels.tp_r=1.0", "intraday_levels.tp1_r=1.5"])
    assert "intraday_levels.tp1_r=1.5" in capsys.readouterr().err  # partial target must stay below tp_r
    lab.apply_overrides(base, ["htf_trend.no_such_param=3"])
    assert "no parameter 'no_such_param'" in capsys.readouterr().err

"""P1 preparation: the seal ledger is sandboxed in tests, lab result files say whether the code was committed, and
`heartless backtest --start/--end` runs one shared account over a stored-history window (seal-checked, offline, no
reads past the window end) for the joint-portfolio report."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import heartless.cli as cli
from heartless.config import Settings
from heartless.core.models import EntryStyle, Side, Signal
from heartless.core.store import Store
from heartless.learning import backtester as B
from heartless.learning import lab, seal
from heartless.strategy.base import Alpha
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from synth import synth_candles, synth_symbols

DAY = 86_400_000
HOUR = 3_600_000
REAL_LEDGER = seal.REPO_ROOT / "docs" / "results" / "seal_ledger.jsonl"


def _ms(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def _save(store: Store, sym: str, start: int, days: float, seed: int) -> None:
    ca = synth_candles(int(days * 1440), seed=seed, start_ms=start)
    store.save_candles(sym, [ca.candle_at(i) for i in range(ca.n)])


# --- (b) seal ledger sandbox ------------------------------------------------------------------------------------
def test_tests_never_write_the_real_seal_ledger():
    """conftest redirects seal.LEDGER for every test: an unseal recorded by a test lands in its tmp dir."""
    before = REAL_LEDGER.read_bytes() if REAL_LEDGER.exists() else None
    assert seal.LEDGER != REAL_LEDGER
    entry = seal.record_unseal("2025", ["pytest"])
    assert json.loads(Path(seal.LEDGER).read_text())["argv"] == entry["argv"] == ["pytest"]
    assert (REAL_LEDGER.read_bytes() if REAL_LEDGER.exists() else None) == before


# --- (a) git_dirty in lab result files ----------------------------------------------------------------------------
def _git(cwd: Path, *a: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=cwd, check=True, capture_output=True)


def test_git_dirty_tracks_the_package_only(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "heartless").mkdir(parents=True)
    (repo / "docs" / "results").mkdir(parents=True)
    (repo / "heartless" / "m.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    monkeypatch.setattr(seal, "REPO_ROOT", repo)
    assert lab.git_dirty() is False
    (repo / "docs" / "results" / "p1_x.json").write_text("{}")  # result files written by a batch do not count
    assert lab.git_dirty() is False
    (repo / "heartless" / "m.py").write_text("x = 2\n")  # an uncommitted code change does
    assert lab.git_dirty() is True
    _git(repo, "checkout", "--", "heartless/m.py")
    (repo / "heartless" / "new.py").write_text("")  # so does an untracked module
    assert lab.git_dirty() is True
    monkeypatch.setattr(seal, "REPO_ROOT", tmp_path / "not-a-repo")
    assert lab.git_dirty() is None


def test_lab_result_file_records_git_dirty_next_to_the_commit(tmp_path, monkeypatch):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    Store(settings.db_path).close()
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k:
                        lab.LabResult("v", None, start, end, lab.summarize([]), {}, {}, {}, chunks=[(start, end)]))
    monkeypatch.setattr(lab, "git_dirty", lambda: True)
    out = tmp_path / "r.json"
    args = SimpleNamespace(symbols="BTCUSDT", years="2022", start=None, end=None, split="train", unseal=[], params=None,
                           set=[], stress=False, meta=False, chunk_days=None, alpha=None, workers=1, json=False,
                           trades_out=None, out=str(out), by_side=False, argv=["heartless", "lab"])
    lab.main_cli(settings, args)
    doc = json.loads(out.read_text())
    assert doc["git_dirty"] is True and "git_commit" in doc
    keys = list(doc)
    assert keys.index("git_dirty") == keys.index("git_commit") + 1


# --- (c) load_backtester window support ---------------------------------------------------------------------------
def _spy_store(monkeypatch) -> dict:
    seen: dict = {"candles": [], "funding": [], "metrics": []}
    oc, of, om = Store.load_candles, Store.load_funding, Store.load_metrics

    def c(self, symbol, start=None, end=None, limit=None):
        seen["candles"].append((start, end))
        return oc(self, symbol, start, end, limit)

    def f(self, symbol, start, end):
        seen["funding"].append((start, end))
        return of(self, symbol, start, end)

    def m(self, symbol, start=None, end=None):
        seen["metrics"].append((start, end))
        return om(self, symbol, start, end)

    monkeypatch.setattr(Store, "load_candles", c)
    monkeypatch.setattr(Store, "load_funding", f)
    monkeypatch.setattr(Store, "load_metrics", m)
    return seen


def test_load_backtester_window_warms_up_before_and_reads_nothing_after(tmp_path, monkeypatch):
    st = Store(tmp_path / "w.db")
    _save(st, "BTCUSDT", _ms(2024, 1, 1), 5, seed=1)
    seen = _spy_store(monkeypatch)
    since, until = _ms(2024, 1, 3), _ms(2024, 1, 4) - 1
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    bt = B.load_backtester(s, st, synth_symbols(["BTCUSDT"]), since, until, warmup_ms=DAY, lookahead=False)
    ca = bt.candles["BTCUSDT"]
    assert int(ca.view("open_time")[0]) == since - DAY and int(ca.view("close_time")[-1]) == until
    assert seen["candles"] == [(since - DAY, until)]
    assert max(e for _, e in seen["funding"] + seen["metrics"]) == until
    # the defaults are unchanged: no warm-up, next settlement's funding read one interval past `until`
    seen["funding"].clear()
    bt = B.load_backtester(s, st, synth_symbols(["BTCUSDT"]), since, until)
    assert int(bt.candles["BTCUSDT"].view("open_time")[0]) == since
    assert seen["funding"] == [(since - B.FUNDING_INTERVAL, until + B.FUNDING_INTERVAL)]
    st.close()


# --- (c) backtest --start/--end: joint portfolio -------------------------------------------------------------------
def test_backtest_window_refuses_the_sealed_period_and_has_no_unseal(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as e:
        cli.main(["backtest", "--start", "2024-12-01", "--end", "2025-01-02"])
    assert e.value.code == 2 and "sealed period 2025" in capsys.readouterr().err
    assert not (tmp_path / "heartless.db").exists()  # refused before the database is opened
    with pytest.raises(SystemExit) as e:
        cli.main(["backtest", "--start", "2024-12-01", "--end", "2025-01-02", "--unseal", "2025"])
    assert e.value.code == 2  # argparse: no such option
    # a date-only end is exclusive: --end 2025-01-01 stops at 2024-12-31 23:59:59.999 and passes the seal
    with pytest.raises(SystemExit) as e:
        cli.main(["backtest", "--start", "2024-12-01", "--end", "2025-01-01", "--symbols", "BTCUSDT"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "sealed" not in err and "no stored candles" in err
    # an open end defaults to the last stored candle and is seal-checked as well
    st = Store(tmp_path / "heartless.db")
    _save(st, "BTCUSDT", _ms(2024, 12, 30), 3, seed=2)  # runs into 2025
    st.close()
    with pytest.raises(SystemExit) as e:
        cli.main(["backtest", "--start", "2024-12-31"])
    assert e.value.code == 2 and "sealed period 2025" in capsys.readouterr().err


class Burst(Alpha):
    """Test alpha: LONG on every symbol at each 2h close, out after 90 minutes, so the symbols always compete for
    the same-direction limit at the same instant."""
    name = "htf_trend"
    timeframe = "1m"

    def evaluate(self, view, ctx, p):
        if (view.cursor_time + 1) % (2 * HOUR):
            return None
        px = ctx.ticker.mark
        return Signal(self.name, ctx.symbol, Side.LONG, 0.95, "burst", px * 0.85, px * 1.3, None, EntryStyle.MARKET,
                      None, 90, 0.0, px * 0.01, "1m", {"ref_price": px})


def test_backtest_window_runs_one_shared_account_with_production_risk(tmp_path, monkeypatch, capsys):
    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Burst()], only_alpha))
    monkeypatch.setattr(lab, "WARMUP_MS", DAY)
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    syms = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT"]
    st = Store(tmp_path / "heartless.db")
    for k, s in enumerate(syms):
        _save(st, s, _ms(2023, 3, 1), 4, seed=10 + k)  # 03-01 .. 03-04 23:59
    st.close()
    params = StrategyParams.default()
    params.enabled = {a: a == "htf_trend" for a in params.enabled}
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(params.to_dict()))
    out = tmp_path / "res" / "joint.json"
    seen = _spy_store(monkeypatch)
    cli.main(["backtest", "--start", "2023-03-02", "--end", "2023-03-04", "--symbols", ",".join(syms + ["SUIUSDT"]),
              "--params", str(pfile), "--out", str(out)])
    printed = capsys.readouterr().out
    doc = json.loads(out.read_text())
    start, end = _ms(2023, 3, 2), _ms(2023, 3, 4) - 1
    assert doc["window"]["start_ms"] == start and doc["window"]["end_ms"] == end
    assert all(e is not None and e <= end for _, e in seen["candles"] + seen["funding"] + seen["metrics"])
    assert min(s for s, _ in seen["candles"]) == start - DAY  # warm-up before the window
    assert doc["symbols_loaded"] == syms and doc["symbols_missing"] == ["SUIUSDT"]
    assert doc["enabled"] == ["htf_trend"] and doc["params_source"] == f"file {pfile}"
    defaults = Settings(_env_file=None)
    assert doc["risk_settings"] == {k: getattr(defaults, k) for k in cli.JOINT_RISK_KEYS}
    assert doc["risk_settings"]["max_positions"] == 5 and doc["risk_settings"]["max_same_direction"] == 3
    sm = doc["summary"]
    assert sm["n"] > 10 and set(doc["by_alpha"]) == {"htf_trend"} and doc["by_alpha"]["htf_trend"]["n"] == sm["n"]
    for k in ("return_pct", "max_dd_pct", "profit_factor", "avg_r", "t_day", "boundary", "final_equity"):
        assert k in sm
    # four symbols fire LONG at the same close: the shared account takes three, the fourth is rejected
    assert doc["rejected"].get("max same-direction positions", 0) >= sm["n"] // 4
    assert doc["seal"]["clean"] and "git_commit" in doc and "git_dirty" in doc
    assert sum(m["n"] for m in doc["by_month"].values()) == sm["n"] and start <= lab.parse_when(doc["first_entry"]) <= end
    assert doc["drawdown_pause"] == {"latched": False, "refused_signals": 0, "hourly_crossing": None}
    assert min(doc["equity_daily"]) == "2023-03-02"
    assert "Joint backtest" in printed and "rejected" in printed and "4/5 symbols" in printed


def test_backtest_window_reports_when_the_drawdown_pause_latched(tmp_path, monkeypatch, capsys):
    """A latched -15% drawdown pause ends trading for the rest of the window (live it waits for the owner's /resume):
    the result says when, so a short loss-capped year is not mistaken for a mild one."""
    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Burst()], only_alpha))
    monkeypatch.setattr(lab, "WARMUP_MS", DAY)
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_DRAWDOWN_HALT_PCT", "0.3")  # the shipped 15% would need a long losing stretch
    st = Store(tmp_path / "heartless.db")
    for k, s in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT"]):
        _save(st, s, _ms(2023, 3, 1), 4, seed=20 + k)
    st.close()
    params = StrategyParams.default()
    params.enabled = {a: a == "htf_trend" for a in params.enabled}
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(params.to_dict()))
    cli.main(["backtest", "--start", "2023-03-02", "--end", "2023-03-04", "--params", str(pfile), "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["risk_settings"]["max_drawdown_halt_pct"] == 0.3 and doc["rejected"].get("paused", 0) > 0
    dp = doc["drawdown_pause"]
    assert dp["latched"] and dp["refused_signals"] == doc["rejected"]["paused"]
    assert dp["hourly_crossing"] is None or doc["last_entry"] <= dp["hourly_crossing"]


def test_backtest_window_without_params_uses_shipped_defaults_read_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(lab, "WARMUP_MS", DAY)
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    st = Store(tmp_path / "heartless.db")
    _save(st, "AAAUSDT", _ms(2023, 3, 1), 2, seed=4)
    st.close()
    cli.main(["backtest", "--start", "2023-03-02", "--end", "2023-03-03", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["params_source"] == "shipped defaults (no stored champion)"
    assert doc["enabled"] == sorted(a for a, on in StrategyParams.default().enabled.items() if on)
    st = Store(tmp_path / "heartless.db")
    assert st.load_params_versions() == []  # nothing persisted (load_champion would have saved a default)
    st.close()


def test_backtest_window_infers_symbol_filters_from_the_newest_closes(tmp_path, monkeypatch):
    """The joint run infers the quantity step from the median of the last closes, like the lab: a symbol that fell
    from ~300 to ~30 over the window must get the step of a ~30 coin (0.1), not that of the older ~300 prices (0.01).
    With the tail reversed, the "last" closes were the oldest part of it, about a month before the window end."""
    st = Store(tmp_path / "w.db")
    t0 = _ms(2023, 3, 1)
    old = synth_candles(4 * 1440, seed=5, start_ms=t0, price=300.0)  # 03-01 .. 03-04
    new = synth_candles(5 * 1440, seed=6, start_ms=t0 + 4 * DAY, price=30.0)  # 03-05 .. 03-09
    st.save_candles("AAAUSDT", [old.candle_at(i) for i in range(old.n)] + [new.candle_at(i) for i in range(new.n)])
    seen: list = []
    orig = lab.infer_symbol_info
    monkeypatch.setattr(lab, "infer_symbol_info", lambda sym, close, *a: seen.append(close) or orig(sym, close, *a))
    lo, hi = t0, _ms(2023, 3, 10) - 1
    infos = cli._window_symbol_infos(st, ["AAAUSDT", "BBBUSDT"], lo, hi)
    assert set(infos) == {"AAAUSDT"}  # a symbol without stored candles gets no filters (and is not loaded)
    closes = [c.close for c in st.load_candles("AAAUSDT", start=lo, end=hi)]
    assert list(seen[0]) == closes  # chronological, the newest close last
    assert infos["AAAUSDT"] == orig("AAAUSDT", np.array(closes))
    assert infos["AAAUSDT"].step_size == 0.1  # ~30 USDT coin; the old ~300 prices would give 0.01
    st.close()

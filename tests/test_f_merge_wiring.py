"""Wave F merge: the small connections the F0 / F7 / F-pre-A / F4 packages left to the owners of other files, and the
places where Wave E (E1 lazy backtest frames, E2 read-only sources and research warm-up) meets them.

- F7: uvloop loop factory at the entry point, Context.book as a field, the backtester's 1m-bar book proxy behind
  ORDERBOOK_STREAM (lazy 1m columns untouched, trades unchanged), the latency line of the daily report
- F4: lab / discover hand the trial-ledger configuration hash to seal.check_window; trial-ledger FORWARD labels from
  forward.period_label; the research warm-up steps past a FORWARD block (it used to read seal.SEALED only)
- E2 x F0 / F-pre-A: the research universe database, the 1m-candle presence check and the evaluators' ReadOnlyStore
  open their files read-only (no schema, no -wal / -shm)
- merge verification: the replay source, the replay's seed copy and the F7 feature bench were the last plain
  `mode=ro` openers (a WAL source got -wal / -shm beside it); they use E2's readonly_uri too
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from synth import synth_candles, synth_symbols

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.exchange.orderbook import EMPTY_BOOK, proxy_from_frame
from heartless.learning import backtester as B
from heartless.learning import discovery as D
from heartless.learning import forward, lab, seal, trials
from heartless.strategy.base import Context
from heartless.strategy.ensemble import ALL_ALPHAS
from heartless.strategy.params import StrategyParams

DAY = 86_400_000


def _ms(y, m, d):
    from datetime import datetime, timezone

    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


# --- F7 ------------------------------------------------------------------------------------------------------------
def test_run_entry_point_passes_the_uvloop_factory(monkeypatch):
    """cli._run hands perf.loop_factory() to asyncio.run (None without uvloop: asyncio's own loop)."""
    import heartless.cli as cli
    from heartless.util import perf

    sentinel = object()
    seen = {}

    def fake_run(coro, **kw):
        coro.close()  # never started: no app is built
        seen.update(kw)

    monkeypatch.setattr(perf, "loop_factory", lambda enabled=True: sentinel)
    monkeypatch.setattr(cli.asyncio, "run", fake_run)
    cli._run(Settings(_env_file=None))
    assert seen == {"loop_factory": sentinel}


def test_context_book_is_a_field_defaulting_to_the_empty_book():
    from heartless.core.models import Regime, Ticker

    ctx = Context(symbol="BTCUSDT", info=synth_symbols(["BTCUSDT"])["BTCUSDT"], ticker=Ticker("BTCUSDT"),
                  regime=Regime.RANGE)
    assert ctx.book is EMPTY_BOOK and "book" in Context.__dataclass_fields__


def _all_alphas() -> StrategyParams:
    p = StrategyParams.default()
    for a in ALL_ALPHAS:
        p.enabled[a.name] = True
    return p


def test_backtester_book_proxy_only_with_the_stream_setting(monkeypatch):
    """ORDERBOOK_STREAM on: every decision's ctx.book is the replay's 1m-bar proxy at that bar; off: EMPTY_BOOK. No
    shipped alpha reads the book, so trades are the same, and the proxy reads bar columns only (the lazy 1m frame
    computes no indicator for it)."""
    seen: list = []
    orig = B.TradingEngine.on_bar

    async def spy(self, view, ctx):
        seen.append((view, ctx.now, ctx.book))
        return await orig(self, view, ctx)

    monkeypatch.setattr(B.TradingEngine, "on_bar", spy)

    def run(on: bool):
        seen.clear()
        bt = B.Backtester(Settings(_env_file=None, ORDERBOOK_STREAM=on, ORDERBOOK_DEPTH_BPS=12.0),
                          synth_symbols(["BTCUSDT"]), {"BTCUSDT": synth_candles(6000, seed=11)})
        res = bt.run(_all_alphas())
        return bt, res, list(seen)

    bt_off, off, seen_off = run(False)
    bt_on, on, seen_on = run(True)
    assert seen_off and all(b is EMPTY_BOOK for _, _, b in seen_off)
    view = bt_on.views["BTCUSDT"]
    tick = bt_on.symbols["BTCUSDT"].tick_size
    for k in (0, 1, len(seen_on) // 2, len(seen_on) - 1):
        _, t, book = seen_on[k]
        assert book.source == "proxy" and book == proxy_from_frame(view.frames["1m"], t, 12.0, tick)
    assert seen_on[-1][2].spread_bps > 0 and seen_on[-1][2].k_bps == 12.0

    def strip(trades):
        return [{k: v for k, v in t.items() if k != "position_id"} for t in trades]

    assert off.stats["n"] >= 5 and strip(on.trades) == strip(off.trades) and on.final_equity == off.final_equity
    held = lambda bt: set(dict.keys(bt.views["BTCUSDT"].frames["1m"].f))  # noqa: E731
    assert held(bt_on) == held(bt_off)


def test_daily_report_latency_line():
    from heartless.notify import formatter as F

    st = {"engines": {}, "primary": None, "mode": "paper", "params_version": "v-test", "health": {"latency": "paper: 봉 마감→결정 p50 1.20ms"}}
    rep = {"stats": {}}
    assert "지연 예산: paper: 봉 마감→결정 p50 1.20ms" in F.fmt_daily(rep, rep, st, "UTC")
    for health in ({}, {"latency": "-"}):
        assert "지연 예산" not in F.fmt_daily(rep, rep, {**st, "health": health}, "UTC")


# --- F4 ------------------------------------------------------------------------------------------------------------
def _stub_result(start, end):
    return lab.LabResult("v-test", None, start, end, {**lab.summarize([])}, {}, {}, {}, chunks=[(start, end)])


def _lab_args(**kw):
    base = dict(symbols="BTCUSDT", years=None, start=None, end=None, split="train", unseal=[], params=None, set=[],
                stress=False, meta=False, chunk_days=None, alpha=None, workers=1, json=False, trades_out=None,
                out=None, by_side=False, argv=["heartless", "lab"])
    base.update(kw)
    return SimpleNamespace(**base)


def _refuse(h, a, b, entries=None):
    raise seal.SealedError(f"configuration {h} has an open FORWARD registration FR-0001")


def test_lab_hands_its_ledger_config_hash_to_the_seal(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path))
    Store(settings.db_path).close()
    monkeypatch.setattr(trials, "LEDGER", tmp_path / "trials.jsonl")
    seen = []
    monkeypatch.setattr(forward, "check_candidate", lambda h, a, b, entries=None: seen.append((h, a, b)))
    monkeypatch.setattr(lab, "evaluate", lambda params, symbols, start, end, *a, **k: _stub_result(start, end))
    lab.main_cli(settings, _lab_args(alpha="htf_trend", start="2024-01-01", end="2024-02-01", tag="fam"))
    rec = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
    assert seen and seen[-1] == (rec["config_hash"], _ms(2024, 1, 1), _ms(2024, 2, 1) - 1)
    # an open FORWARD registration of this configuration refuses the run before any job
    monkeypatch.setattr(forward, "check_candidate", _refuse)
    ran = []
    monkeypatch.setattr(lab, "evaluate", lambda *a, **k: ran.append(1))
    with pytest.raises(SystemExit) as e:
        lab.main_cli(settings, _lab_args(alpha="htf_trend", start="2024-01-01", end="2024-02-01"))
    assert e.value.code == 2 and "open FORWARD registration" in capsys.readouterr().err and not ran


def test_discover_hands_its_ledger_config_hash_to_the_seal(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    Store(tmp_path / "heartless.db").close()
    monkeypatch.setattr(trials, "LEDGER", tmp_path / "trials.jsonl")
    seen = []
    monkeypatch.setattr(forward, "check_candidate", lambda h, a, b, entries=None: seen.append((h, a, b)))
    mined = []
    monkeypatch.setattr(D, "mine", lambda *a, **k: mined.append(1) or {
        "tested": 3, "t_bar": 3.0, "passed": [], "results": [], "seconds": 0.0, "rows": {"train": 0}})
    argv = ["discover", "--symbols", "BTCUSDT", "--train", "2024-01-15:2024-06-01", "--valid", "2024-06-01:2024-09-01"]
    cli.main(argv + ["--tag", "fam"])
    rec = json.loads((tmp_path / "trials.jsonl").read_text().splitlines()[-1])
    assert [h for h, _, _ in seen] == [rec["config_hash"]] * 2  # TRAIN and VALID windows
    monkeypatch.setattr(forward, "check_candidate", _refuse)
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and "open FORWARD registration" in capsys.readouterr().err and len(mined) == 1


def test_trial_ledger_forward_labels_come_from_the_forward_tools():
    assert (trials.FWD_ORIGIN, trials.FWD_BLOCK_MS) == (seal.FORWARD_ORIGIN_MS, seal.FORWARD_BLOCK_MS)
    o, blk = seal.FORWARD_ORIGIN_MS, seal.FORWARD_BLOCK_MS
    for a, b in ((o, o + DAY), (o + blk, o + 2 * blk - 1), (o + 5 * DAY, o + 3 * blk + 2 * DAY)):
        assert trials.wave_f_period(a, b) == forward.period_label(a, b)
    assert trials.wave_f_period(o, o + DAY) == "FWD-01" and trials.wave_f_period(o, o + 2 * blk) == "FWD-01..FWD-03"
    assert trials.wave_f_period(o - DAY, o + DAY) == "other"  # starts before FWD-01: not a FORWARD run
    assert trials.wave_f_period(_ms(2024, 2, 1), _ms(2024, 3, 1)) == "TRAIN-B"
    for k in (1, 2, 12):
        assert trials.fwd_span(k) == (seal.block_span(k)[0], seal.block_span(k)[1] - 1)
        assert trials.fwd_block(seal.block_span(k)[0]) == k
    assert trials.fwd_block(o - 1) == 0


def test_research_warmup_steps_past_a_forward_block():
    """E2's warm-up start reads the end of each sealed period it overlaps; FORWARD blocks are not in seal.SEALED (F4),
    which raised KeyError for a caller without the live-loop declaration."""
    from heartless.learning.optimizer import warmup_start

    a1, b1 = seal.block_span(1)
    since = b1 + 5 * DAY  # inside FWD-02; a 20-day warm-up reaches back into FWD-01 (neither opened in this repo)
    assert not {1, 2} & seal.forward_spent()
    assert warmup_start(since, 20 * DAY) == since  # every block it touches is sealed: no warm-up at all
    assert warmup_start(since, 20 * DAY, seal.live_loop("research_cycle")) == since - 20 * DAY
    assert warmup_start(a1 - DAY, 10 * DAY) == a1 - 11 * DAY  # before FWD-01: unchanged


# --- E2 read-only sources x F0 / F-pre-A -----------------------------------------------------------------------------
def _digest(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_universe_resolution_opens_its_databases_read_only(tmp_path):
    from tests.test_f0_universe import M, make_universe_db

    from heartless.data import universe as U

    st = make_universe_db(tmp_path / "universe.db")
    st.close()
    cdb = tmp_path / "candles.db"
    cs = Store(cdb)
    cs.save_candles("BIGUSDT", [Candle(M("2024-01"), 1, 1, 1, 1, 1, 1, 1, 1, M("2024-01") + 59_999)])
    cs.close()
    before = {p: _digest(p) for p in (tmp_path / "universe.db", cdb)}
    res = U.resolve_for_run(tmp_path / "universe.db", cdb, M("2023-12"), M("2024-02") - 1, 2)
    win = U.member_windows_for(tmp_path / "universe.db", res["symbols"], M("2024-01"), M("2024-03") - 1, 2)
    assert res["symbols"] == ["BIGUSDT"] and win["windows"]["BIGUSDT"]
    assert {p: _digest(p) for p in before} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["candles.db", "universe.db"]  # no -wal / -shm left
    with pytest.raises(ValueError, match="no universe database"):
        U.member_windows_for(tmp_path / "missing.db", ["BIGUSDT"], M("2024-01"), M("2024-03") - 1, 2)
    assert not (tmp_path / "missing.db").exists()


def test_evaluator_read_only_store_is_the_e2_read_only_store(tmp_path):
    from heartless.learning import hourly

    db = tmp_path / "u.db"
    Store(db).close()
    st = hourly.ReadOnlyStore(db)
    assert st.readonly and st.f0_has("candles") is True
    with pytest.raises(sqlite3.OperationalError):
        st.execute("CREATE TABLE x (a)")
    st.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["u.db"]
    with pytest.raises(FileNotFoundError):
        hourly.ReadOnlyStore(tmp_path / "missing.db")


# --- merge verification: the remaining raw mode=ro openers (replay source, replay seed copy, F7 bench) -------------------
def _wal_source(path, days: int = 2, syms=("BTCUSDT", "ETHUSDT")) -> int:
    """A WAL candle store closed cleanly (no -wal left), like the shared longhist DB between writers."""
    st = Store(path)
    t0 = _ms(2026, 3, 1)
    for i, s in enumerate(syms):
        ca = synth_candles(days * 1440, seed=7 + i, start_ms=t0, price=100.0 * (i + 1))
        st.save_candles(s, [ca.candle_at(k) for k in range(ca.n)])
    st.close()
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    con.close()  # the last writer-capable connection: removes -wal / -shm
    assert not Path(f"{path}-wal").exists()
    return t0


def test_replay_source_seed_copy_and_feature_bench_leave_no_side_files(tmp_path):
    """A plain `file:...?mode=ro` open of a WAL database creates -wal / -shm beside it (and fails in a read-only
    folder): the replay smoke left them next to the shared longhist DB. All three now use E2's readonly_uri."""
    from heartless.replay.market import ReplaySource
    from heartless.replay.runner import ReplayConfig, _prepare_store
    from heartless.util import perf

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "heartless.db"
    t0 = _wal_source(src)
    before = _digest(src)
    rs = ReplaySource(str(src), ["BTCUSDT"], floor_ms=t0, ceil_ms=t0 + 2 * DAY)
    assert rs.ticker_24h("BTCUSDT", t0 + DAY)["count"] == 1440
    work = tmp_path / "work"
    work.mkdir()
    _prepare_store(ReplayConfig(source_db=str(src), start_ms=t0 + DAY, end_ms=t0 + 2 * DAY - 1, seed_days=1), work,
                   ["BTCUSDT", "ETHUSDT"])
    con = sqlite3.connect(work / "heartless.db")
    assert con.execute("SELECT COUNT(*) FROM candles").fetchone()[0] == 2 * 1440
    con.close()
    res = perf.bench_features(str(src), "ETHUSDT", history_days=1, minutes=30)
    assert res["updates"] == 30
    assert sorted(p.name for p in src_dir.iterdir()) == ["heartless.db"] and _digest(src) == before

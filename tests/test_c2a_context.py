"""C2-A context store: its own file with source / universe metadata, refusal of a foreign store, appended history,
and the inter-process build lock."""
import multiprocessing
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C
from heartless.learning import lab, trials

H = C.MS_H
DAY = 86_400_000
START = 1_701_734_400_000  # 2023-12-05 00:00 UTC: the history crosses the 2023/2024 year boundary
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


@pytest.fixture(autouse=True)
def _sandboxed_trial_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "LEDGER", tmp_path / "p2_trials.jsonl")


def _bars(sym_k: int, a: int, n: int, seed: int) -> list[Candle]:
    rng = np.random.default_rng(seed)
    close = (100 + 30 * sym_k) * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    out, prev = [], float(close[0])
    for i in range(n):
        c = float(close[i])
        t = a + i * 60_000
        out.append(Candle(t, prev, max(prev, c) * 1.0002, min(prev, c) * 0.9998, c, 10.0, 10.0 * c, 3, 5.0,
                          t + 59_999))
        prev = c
    return out


def _make_db(path, days: int = 40, seed: int = 1, syms=SYMS) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    st = Store(path)
    for k, sym in enumerate(syms):
        st.save_candles(sym, _bars(k, START, days * 1440, seed + k))
        st.save_funding(sym, [(START + j * 8 * H, 1e-4 * (k + 1), 0.0) for j in range(days * 3)])
        st.save_metrics(sym, [SimpleNamespace(ts=START + j * 300_000, oi=1e6 + j, oi_value=5e7 + j,
                                              top_ls_accounts=1.0, top_ls_positions=1.0, ls_accounts=1.0,
                                              taker_ls_vol=1.0) for j in range(days * 288)])
    st.close()
    return str(path)


@pytest.fixture
def src_db(tmp_path):
    return _make_db(tmp_path / "data" / "heartless.db")


def _rows(store) -> dict:
    rows = store.query("SELECT symbol, ts, data FROM market_context")
    return {(r["symbol"], r["ts"]): bytes(r["data"]) for r in rows}


def test_the_context_store_defaults_to_its_own_file_and_never_the_candle_database(src_db, tmp_path):
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "data"))
    assert s.context_db_path == tmp_path / "data" / "context.db" != s.db_path
    assert Settings(_env_file=None, CONTEXT_DB=str(tmp_path / "x.db")).context_db_path == tmp_path / "x.db"
    src = Store(src_db)
    with pytest.raises(C.ContextMismatch, match="never written into the candle database"):
        C.build(src, src, universe=SYMS)
    with pytest.raises(C.ContextMismatch):
        C.build(src, None, universe=SYMS)
    with pytest.raises(C.ContextMismatch):
        C.build(src, Store(src_db), universe=SYMS)  # another handle on the same file
    src.close()
    with pytest.raises(C.ContextMismatch, match="is the candle database"):
        C.prepare(src_db, {"db": src_db, "universe": list(SYMS)}, START + 30 * DAY, START + 31 * DAY)
    # no "db": the default file next to the candle database
    path = C.prepare(src_db, {"universe": list(SYMS)}, START + 30 * DAY, START + 31 * DAY)
    assert path == C.default_path(src_db) == str((tmp_path / "data" / "context.db").resolve())
    assert not Store(src_db).has_market_context() and Store(path).has_market_context()


def test_the_metadata_records_source_universe_and_built_hours(src_db, tmp_path):
    src, dst = Store(src_db), Store(tmp_path / "c.db")
    C.build(src, dst, START + 30 * DAY, START + 35 * DAY - 1, universe=SYMS)
    meta = dst.get("context.meta")
    assert meta["universe"] == sorted(SYMS) and meta["version"] == C.VERSION
    assert meta["source"]["path"] == os.path.realpath(src_db) and len(meta["source"]["fingerprint"]) == 16
    rng = meta["source"]["ranges"]["BTCUSDT"]
    assert rng["candles"] == [START, START + (40 * 1440 - 1) * 60_000]
    assert rng["funding"] == [START, START + (40 * 3 - 1) * 8 * H] and rng["metrics"][0] == START
    assert meta["built"] == [[START + 30 * DAY + H - 1, START + 35 * DAY - 1]]  # hour closes inside the window
    assert C.usable(src, dst, SYMS)[:2] == (True, "same")
    src.close()
    dst.close()


def test_a_store_of_another_source_or_universe_is_refused_until_rebuilt(src_db, tmp_path):
    other_db = _make_db(tmp_path / "other" / "heartless.db", seed=7)
    ctx = tmp_path / "shared_ctx.db"
    src, other, dst = Store(src_db), Store(other_db), Store(ctx)
    C.build(src, dst, START + 30 * DAY, START + 32 * DAY, universe=SYMS)
    before = _rows(dst)
    with pytest.raises(C.ContextMismatch, match="built from .*heartless.db.*--rebuild"):
        C.build(other, dst, START + 30 * DAY, START + 32 * DAY, universe=SYMS)
    with pytest.raises(C.ContextMismatch, match="universe"):
        C.build(src, dst, START + 30 * DAY, START + 32 * DAY, universe=SYMS[:3])
    assert _rows(dst) == before  # nothing was touched
    assert not C.ensure(other, dst, START + 30 * DAY, START + 32 * DAY, SYMS, auto_build=False)
    with pytest.raises(C.ContextMismatch):
        C.ensure(other, dst, START + 30 * DAY, START + 32 * DAY, SYMS)
    # the research entry points refuse too (lab before any job runs, discovery before any dataset is built)
    with pytest.raises(C.ContextMismatch):
        lab.evaluate(lab_params(), ["BTCUSDT"], START + 30 * DAY, START + 31 * DAY, other_db,
                     pool=SimpleNamespace(map=map), context={"db": str(ctx), "universe": list(SYMS)})
    # a reader of another source does not get the rows either
    assert C.ContextTable.load(dst, SYMS, START, START + 40 * DAY, source=other_db) is None
    assert C.ContextTable.load(dst, SYMS, START, START + 40 * DAY, source=src_db) is not None
    # an explicit rebuild takes the store over
    C.build(other, dst, START + 30 * DAY, START + 32 * DAY, universe=SYMS, rebuild=True)
    assert dst.get("context.meta")["source"]["path"] == os.path.realpath(other_db) and _rows(dst) != before
    # a store from before source checks existed (no source recorded) is refused as well
    meta = dst.get("context.meta")
    meta.pop("source")
    dst.set("context.meta", meta)
    with pytest.raises(C.ContextMismatch, match="records no source"):
        C.build(other, dst, START + 30 * DAY, START + 32 * DAY, universe=SYMS)
    for s in (src, other, dst):
        s.close()


def lab_params():
    from heartless.strategy.params import StrategyParams

    return StrategyParams.default()


def test_appended_history_rebuilds_only_the_hours_it_can_change(tmp_path):
    db = _make_db(tmp_path / "grow.db", days=40)
    src, dst = Store(db), Store(tmp_path / "c.db")
    C.build(src, dst, universe=SYMS)
    old_end = dst.get("context.meta")["built"][-1][1]
    before = _rows(dst)
    # five more days for every symbol (candles, funding, metrics): appended, nothing replaced
    full = {k: _bars(k, START, 45 * 1440, 1 + k) for k in range(len(SYMS))}
    for k, sym in enumerate(SYMS):
        src.save_candles(sym, full[k][40 * 1440:])
        src.save_funding(sym, [(START + j * 8 * H, 1e-4 * (k + 1), 0.0) for j in range(120, 135)])
        src.save_metrics(sym, [SimpleNamespace(ts=START + j * 300_000, oi=1e6 + j, oi_value=5e7 + j,
                                               top_ls_accounts=1.0, top_ls_positions=1.0, ls_accounts=1.0,
                                               taker_ls_vol=1.0) for j in range(40 * 288, 45 * 288)])
    ok, status, stable, _ = C.usable(src, dst, SYMS)
    assert ok and status == "extended" and C.hour_close(C.hour_of(stable)) <= old_end
    assert C.ensure(src, dst, START, START + 45 * DAY, SYMS)
    meta = dst.get("context.meta")
    assert meta["built"][-1][1] == START + 45 * DAY - 1 and C.usable(src, dst, SYMS)[1] == "same"
    after = _rows(dst)
    assert all(after[k] == v for k, v in before.items() if k[1] <= C.hour_close(C.hour_of(stable)))
    fresh = Store(tmp_path / "fresh.db")
    C.build(src, fresh, universe=SYMS)
    assert _rows(fresh) == after  # identical to building the extended source from scratch
    # replacing history (here: an earlier first candle) is not an extension: refused
    src.save_candles("BTCUSDT", _bars(0, START - DAY, 10, 99))
    assert C.usable(src, dst, SYMS)[1] == "mismatch"
    with pytest.raises(C.ContextMismatch, match="replaced"):
        C.ensure(src, dst, START, START + 45 * DAY, SYMS)
    for s in (src, dst, fresh):
        s.close()


def _build_in_child(db: str, ctx: str, out: str) -> None:
    src, dst = Store(db), Store(ctx)
    msgs = []
    stats = C.build(src, dst, universe=SYMS, progress=msgs.append)
    built = [u for u in stats["units"] if not u.get("skipped")]
    with open(out, "w") as f:
        f.write(f"{len(built)} {time.time()} {int(any('waiting' in m for m in msgs))}")


def _load_in_child(ctx: str, out: str) -> None:
    table = C.ContextTable.load(Store(ctx), SYMS, START, START + 40 * DAY)
    with open(out, "w") as f:
        f.write(f"{len(table) if table is not None else -1} {time.time()}")


def test_builds_take_an_inter_process_lock_and_never_duplicate_work(src_db, tmp_path):
    ctx = str(tmp_path / "locked.db")
    spawn = multiprocessing.get_context("spawn")
    # a build waits while another process holds the lock, then re-reads the metadata and builds what is missing
    with C.build_lock(ctx):
        p1 = spawn.Process(target=_build_in_child, args=(src_db, ctx, str(tmp_path / "p1")))
        p1.start()
        time.sleep(2.0)
        assert p1.is_alive() and not (tmp_path / "p1").exists()
        released = time.time()
    p1.join(120)
    n_built, finished, waited = (tmp_path / "p1").read_text().split()
    assert int(n_built) == 2 and float(finished) >= released and waited == "1"  # two calendar years
    expected = _rows(Store(ctx))
    # two more builds racing on a fresh store: exactly one computes the years, the other finds them built
    ctx2 = str(tmp_path / "race.db")
    procs = [spawn.Process(target=_build_in_child, args=(src_db, ctx2, str(tmp_path / f"r{i}"))) for i in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    counts = sorted(int((tmp_path / f"r{i}").read_text().split()[0]) for i in range(2))
    assert counts == [0, 2] and _rows(Store(ctx2)) == expected
    # a reader waits for a build in progress (shared lock against the builder's exclusive one)
    with C.build_lock(ctx2):
        rd = spawn.Process(target=_load_in_child, args=(ctx2, str(tmp_path / "rd")))
        rd.start()
        time.sleep(1.5)
        assert rd.is_alive()
        released = time.time()
    rd.join(60)
    hours, finished = (tmp_path / "rd").read_text().split()
    assert int(hours) > 0 and float(finished) >= released


def test_context_cli_reports_the_source_and_refuses_a_foreign_store(src_db, tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    data = tmp_path / "data"
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(data))
    monkeypatch.setenv("CONTEXT_DB", "")  # (an empty variable also overrides a local .env): the default file
    monkeypatch.setenv("CONTEXT_UNIVERSE", ",".join(SYMS))
    cli.main(["context", "status"])
    assert "not built" in capsys.readouterr().out
    cli.main(["context", "build", "--start", "2024-01-05", "--end", "2024-01-07"])
    out = capsys.readouterr().out
    assert "48 hours" in out and str(data / "context.db") in out
    cli.main(["context", "status"])
    out = capsys.readouterr().out
    assert "usable (same source" in out and os.path.realpath(src_db) in out and "current" in out
    monkeypatch.setenv("CONTEXT_UNIVERSE", ",".join(SYMS[:3]))
    with pytest.raises(SystemExit) as e:
        cli.main(["context", "build"])
    assert e.value.code == 2 and "--rebuild" in capsys.readouterr().err
    cli.main(["context", "status"])
    assert "NOT usable" in capsys.readouterr().out
    monkeypatch.setenv("CONTEXT_DB", str(data / "heartless.db"))
    with pytest.raises(SystemExit) as e:
        cli.main(["context", "build"])
    assert e.value.code == 2 and "candle database" in capsys.readouterr().err

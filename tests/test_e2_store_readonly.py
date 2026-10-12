"""E2: Store(path, readonly=True) and the readers of source candle databases. A read-only open creates no table, no
-wal / -shm file and writes nothing; lab, discovery, the context build and the research worker read their source
candle database that way, and `heartless discover` keeps its results out of it. Synthetic data only."""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store, readonly_uri
from heartless.learning import discovery as D
from heartless.learning import lab
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_MINUTE

START = 1_704_067_200_000  # 2024-01-01 00:00 UTC


def _walk(start: int, n: int, seed: int = 1) -> list[Candle]:
    import numpy as np

    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    out, prev = [], float(close[0])
    for i, c in enumerate(close):
        t = start + i * MS_MINUTE
        out.append(Candle(t, prev, max(prev, c) * 1.0004, min(prev, c) * 0.9996, float(c), 10.0, 10.0 * c, 5, 5.0,
                          t + 59_999))
        prev = float(c)
    return out


def _candle_db(path: Path, minutes: int = 3 * 1440, symbols=("BTCUSDT",)) -> Path:
    """A WAL-mode candle database closed cleanly: no -wal / -shm file left beside it."""
    st = Store(path)
    for i, s in enumerate(symbols):
        st.save_candles(s, _walk(START, minutes, seed=i + 1))
    st.close()
    assert not _side_files(path)
    return path


def _side_files(path: Path) -> list[str]:
    return sorted(p.name for p in path.parent.iterdir() if p.name.startswith(path.name + "-"))


def _schema(path: Path) -> list[str]:
    con = sqlite3.connect(readonly_uri(path), uri=True)
    try:
        return sorted(r[0] for r in con.execute("SELECT name FROM sqlite_master"))
    finally:
        con.close()


def test_a_readonly_store_reads_without_creating_anything(tmp_path):
    db = _candle_db(tmp_path / "heartless.db")
    before = (db.stat().st_size, db.stat().st_mtime_ns, _schema(db))
    st = Store(db, readonly=True)
    assert st.readonly and st.candle_symbols() == ["BTCUSDT"] and len(st.load_candles("BTCUSDT")) == 3 * 1440
    assert st.get("missing", "default") == "default" and st.market_context_stats()["hours"] == 0
    assert not _side_files(db)  # immutable open: no -wal / -shm for the shared-memory index
    for write in (lambda: st.set("k", 1), lambda: st.save_candles("ETHUSDT", _walk(START, 3)),
                  lambda: st.execute("CREATE TABLE x (a)")):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            write()
    st.close()
    assert (db.stat().st_size, db.stat().st_mtime_ns, _schema(db)) == before and not _side_files(db)


def test_a_readonly_store_creates_no_table_in_a_foreign_database(tmp_path):
    db = tmp_path / "plain.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE candles (symbol TEXT, open_time INTEGER)")
    con.commit()
    con.close()
    st = Store(db, readonly=True)  # a read-write Store would add its whole schema
    assert st.query("SELECT COUNT(*) n FROM candles")[0]["n"] == 0
    st.close()
    assert _schema(db) == ["candles"] and not _side_files(db)


def test_a_missing_database_is_never_created(tmp_path):
    with pytest.raises(FileNotFoundError):
        Store(tmp_path / "nope.db", readonly=True)
    assert not (tmp_path / "nope.db").exists()


def test_with_a_live_writer_the_reader_sees_its_commits(tmp_path):
    db = _candle_db(tmp_path / "heartless.db")
    writer = Store(db)  # e.g. the running bot: its WAL is open, so the reader must not be immutable
    writer.set("k", 1)
    assert "immutable" not in readonly_uri(db)
    reader = Store(db, readonly=True)
    assert reader.get("k") == 1
    writer.set("k", 2)
    writer.save_candles("ETHUSDT", _walk(START, 10))
    assert reader.get("k") == 2 and "ETHUSDT" in reader.candle_symbols()
    reader.close()
    writer.close()


def test_uncheckpointed_transactions_are_read(tmp_path):
    db = _candle_db(tmp_path / "heartless.db")
    con = sqlite3.connect(db)
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("INSERT INTO kv(key, value) VALUES('w', '7')")
    con.commit()  # committed into the -wal file only, the connection still open
    assert (tmp_path / "heartless.db-wal").stat().st_size > 0
    st = Store(db, readonly=True)
    assert st.get("w") == 7
    st.close()
    con.close()


def test_the_lab_reads_its_source_database_read_only(tmp_path):
    db = _candle_db(tmp_path / "heartless.db", minutes=20 * 1440)
    before = (db.stat().st_mtime_ns, _schema(db))
    res = lab.evaluate(StrategyParams.default(), ["BTCUSDT"], START + 13 * MS_DAY, START + 19 * MS_DAY, str(db),
                       pool=SimpleNamespace(map=map))
    assert res.errors == {}
    assert (db.stat().st_mtime_ns, _schema(db)) == before and not _side_files(db)
    # a job whose database does not exist reports it instead of creating an empty one
    out = lab._run_symbol({"symbol": "BTCUSDT", "start": START, "end": START + MS_DAY, "db_path": str(tmp_path / "x.db")})
    assert "no candle database" in out["error"] and not (tmp_path / "x.db").exists()


def test_discovery_reads_its_source_database_read_only(tmp_path):
    db = _candle_db(tmp_path / "heartless.db", minutes=30 * 1440)
    before = (db.stat().st_mtime_ns, _schema(db))
    ds = D.build_dataset(str(db), ["BTCUSDT"], START + 14 * MS_DAY, START + 28 * MS_DAY, "1h", workers=1)
    assert len(ds.times) > 0
    assert (db.stat().st_mtime_ns, _schema(db)) == before and not _side_files(db)
    assert D._build_job({"db_path": str(tmp_path / "x.db"), "symbol": "BTCUSDT", "tf": "1h", "start": START,
                         "end": START + MS_DAY}) is None and not (tmp_path / "x.db").exists()


def test_the_discover_cli_writes_nothing_into_the_candle_database(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    db = _candle_db(tmp_path / "heartless.db")
    before = (db.stat().st_mtime_ns, _schema(db))
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(D, "mine", lambda *a, **k: {"tested": 3, "t_bar": 3.0, "passed": [], "results": [],
                                                     "seconds": 0.0, "rows": {"train": 0}})
    out = tmp_path / "res.json"
    cli.main(["discover", "--symbols", "BTCUSDT", "--train", "2024-01-01:2024-01-02", "--valid",
              "2024-01-02:2024-01-03", "--out", str(out)])
    assert "tested 3 rules" in capsys.readouterr().out and out.exists()  # results: stdout and --out
    assert (db.stat().st_mtime_ns, _schema(db)) == before and not _side_files(db)
    st = Store(db, readonly=True)
    assert st.get("discovery.last") is None  # it used to be written into the candle database
    st.close()


def test_the_cli_readers_refuse_a_missing_candle_database(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONTEXT_DB", str(tmp_path / "ctx.db"))
    for argv in (["discover", "--symbols", "BTCUSDT", "--train", "2024-01-01:2024-01-02", "--valid",
                  "2024-01-02:2024-01-03"], ["context", "build"], ["lab", "--years", "2024"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 2 and "no candle database" in capsys.readouterr().err, argv
    assert not (tmp_path / "heartless.db").exists()


def test_the_context_build_reads_its_source_read_only(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    db = _candle_db(tmp_path / "heartless.db", minutes=2 * 1440, symbols=("BTCUSDT", "ETHUSDT"))
    before = (db.stat().st_mtime_ns, _schema(db))
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CONTEXT_DB", str(tmp_path / "ctx.db"))
    monkeypatch.setenv("CONTEXT_UNIVERSE", "BTCUSDT,ETHUSDT")
    cli.main(["context", "build"])
    assert "hours" in capsys.readouterr().out and (tmp_path / "ctx.db").exists()
    assert (db.stat().st_mtime_ns, _schema(db)) == before and not _side_files(db)


def test_the_research_worker_opens_the_candle_database_read_only(tmp_path, monkeypatch):
    from heartless.learning import optimizer as O

    db = _candle_db(tmp_path / "heartless.db", minutes=2 * 1440)
    seen = []
    orig = O.load_backtester

    def spy(settings, store, *a, **k):
        seen.append(store.readonly)
        return orig(settings, store, *a, **k)

    monkeypatch.setattr(O, "load_backtester", spy)
    O.run_research_cycle(str(db), {"HEARTLESS_DATA_DIR": str(tmp_path)}, StrategyParams.default().to_dict(), {},
                         lookback_days=1, n_candidates=1, alphas=[])
    assert seen == [True] and not _side_files(db)
    assert os.path.getsize(db) > 0

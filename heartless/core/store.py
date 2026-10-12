"""SQLite persistence (WAL mode). Single-process, thread-safe through a lock."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from heartless.core.models import Candle, Position, TradeRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL, open_time INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL, quote_volume REAL,
    trades INTEGER, taker_buy_volume REAL, close_time INTEGER,
    PRIMARY KEY (symbol, open_time)
);
CREATE TABLE IF NOT EXISTS funding (
    symbol TEXT NOT NULL, funding_time INTEGER NOT NULL, rate REAL, mark REAL,
    PRIMARY KEY (symbol, funding_time)
);
CREATE TABLE IF NOT EXISTS metrics (
    symbol TEXT NOT NULL, ts INTEGER NOT NULL, oi REAL, oi_value REAL, top_ls_accounts REAL, top_ls_positions REAL,
    ls_accounts REAL, taker_ls_vol REAL,
    PRIMARY KEY (symbol, ts)
);
CREATE TABLE IF NOT EXISTS positions (
    id TEXT PRIMARY KEY, engine TEXT, symbol TEXT, side TEXT, status TEXT,
    entry_time INTEGER, exit_time INTEGER, data TEXT
);
CREATE INDEX IF NOT EXISTS idx_positions_engine_status ON positions(engine, status);
CREATE TABLE IF NOT EXISTS trades (
    position_id TEXT PRIMARY KEY, engine TEXT, symbol TEXT, side TEXT, alpha TEXT, alphas TEXT, regime TEXT,
    entry_time INTEGER, exit_time INTEGER, entry_price REAL, exit_price REAL, qty REAL, notional REAL,
    pnl REAL, gross REAL, fees REAL, funding REAL, r_multiple REAL, risk_amount REAL, exit_reason TEXT,
    confidence REAL, params_version TEXT, bars_held INTEGER, max_fav_r REAL, max_adv_r REAL, reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_engine_time ON trades(engine, exit_time);
CREATE TABLE IF NOT EXISTS equity (
    engine TEXT NOT NULL, ts INTEGER NOT NULL, balance REAL, equity REAL,
    PRIMARY KEY (engine, ts)
);
CREATE TABLE IF NOT EXISTS params_versions (
    id TEXT PRIMARY KEY, created INTEGER, role TEXT, source TEXT, note TEXT, params TEXT, metrics TEXT
);
CREATE TABLE IF NOT EXISTS alpha_stats (
    engine TEXT, alpha TEXT, regime TEXT, a REAL, b REAL, n INTEGER, sum_r REAL, updated INTEGER,
    PRIMARY KEY (engine, alpha, regime)
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, level TEXT, topic TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS research_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, alpha TEXT, summary TEXT
);
"""

# --- P4 (self-improvement hardening) tables ---------------------------------------------------------------
# fill_quality: one row per entry / exit fill of a persisted engine with the reference price the engine decided on,
# the signed slippage against it and the fee against the fee model (execution-quality monitor).
SCHEMA_P4 = """
CREATE TABLE IF NOT EXISTS fill_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT, engine TEXT, ts INTEGER, symbol TEXT, position_id TEXT, kind TEXT,
    side TEXT, maker INTEGER, qty REAL, price REAL, ref REAL, notional REAL, slip_bps REAL, fee REAL, model_fee REAL
);
CREATE INDEX IF NOT EXISTS idx_fill_quality_engine_ts ON fill_quality(engine, ts);
"""


def readonly_uri(path: str | Path) -> str:
    """SQLite URI of a read-only connection to `path` (E2).

    mode=ro never creates the file, a table, an index or a journal mode change. A WAL database opened that way alone
    still creates (and leaves behind) -wal and -shm files for its shared-memory index, and fails in a read-only
    directory, so when no -wal (or rollback -journal) file sits next to the database - no process has it open for
    writing, and every committed transaction is in the main file - it is also opened immutable=1: no lock, no
    side file. With such a file (the live bot, a fetch, or a writer that left transactions un-checkpointed) the normal
    shared-memory read path is used, which reads the WAL's committed content and the writer's later commits. An
    immutable reader assumes nobody writes while it is open: start a writer on a database only after its readers end.

    SQLite resolves a symlinked database file and keeps the writer's -wal / -journal beside the real file, so the side
    files are looked for there (and beside the given name): looking only beside a symlink missed the running bot's WAL
    and opened its database immutable, blind to every commit not yet checkpointed into the main file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"database {p} does not exist (read-only open never creates it)")
    real = p.resolve()
    uri = real.as_uri() + "?mode=ro"
    if not any(Path(f"{base}{sfx}").exists() for base in {p, real} for sfx in ("-wal", "-journal")):
        uri += "&immutable=1"
    return uri


class Store:
    def __init__(self, path: str | Path, readonly: bool = False):
        """`readonly=True` (E2) opens an existing database through readonly_uri for readers of source candle databases
        (lab, discovery, the context build, the research worker): no table, index, -wal or -shm file is created and
        every write raises sqlite3.OperationalError. Default: read-write in WAL mode with the schema created."""
        self.path = str(path)
        self.readonly = bool(readonly)
        self._lock = threading.RLock()
        if self.readonly:
            self._conn = sqlite3.connect(readonly_uri(path), uri=True, check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA temp_store=MEMORY")
            return
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.executescript(SCHEMA_P4)

    # --- generic -----------------------------------------------------------------------------
    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable]) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(sql, rows)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def query(self, sql: str, params: Iterable = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- kv ----------------------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM kv WHERE key=?", (key,))
        if not rows:
            return default
        try:
            return json.loads(rows[0]["value"])
        except Exception:
            return rows[0]["value"]

    def set(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value, default=str)))

    # --- candles -----------------------------------------------------------------------------
    def save_candles(self, symbol: str, candles: Iterable[Candle]) -> int:
        rows = [(symbol, c.open_time, c.open, c.high, c.low, c.close, c.volume, c.quote_volume, c.trades,
                 c.taker_buy_volume, c.close_time) for c in candles if c.closed]
        if not rows:
            return 0
        self.executemany(
            "INSERT OR REPLACE INTO candles(symbol,open_time,open,high,low,close,volume,quote_volume,trades,"
            "taker_buy_volume,close_time) VALUES(?,?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def load_candles(self, symbol: str, start: int | None = None, end: int | None = None,
                     limit: int | None = None) -> list[Candle]:
        sql = "SELECT * FROM candles WHERE symbol=?"
        params: list[Any] = [symbol]
        if start is not None:
            sql += " AND open_time>=?"
            params.append(start)
        if end is not None:
            sql += " AND open_time<=?"
            params.append(end)
        sql += " ORDER BY open_time DESC" if limit else " ORDER BY open_time ASC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.query(sql, params)
        out = [Candle(open_time=r["open_time"], open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                      volume=r["volume"], quote_volume=r["quote_volume"], trades=r["trades"],
                      taker_buy_volume=r["taker_buy_volume"], close_time=r["close_time"], closed=True) for r in rows]
        if limit:
            out.reverse()
        return out

    def candle_range(self, symbol: str) -> tuple[int | None, int | None, int]:
        r = self.query("SELECT MIN(open_time) a, MAX(open_time) b, COUNT(*) n FROM candles WHERE symbol=?", (symbol,))[0]
        return r["a"], r["b"], r["n"]

    def candle_symbols(self) -> list[str]:
        return [r["symbol"] for r in self.query("SELECT DISTINCT symbol FROM candles")]

    def prune_candles(self, before_open_time: int) -> None:
        self.execute("DELETE FROM candles WHERE open_time<?", (before_open_time,))

    # --- funding -----------------------------------------------------------------------------
    def save_funding(self, symbol: str, rows: Iterable[tuple[int, float, float]]) -> None:
        self.executemany("INSERT OR REPLACE INTO funding(symbol,funding_time,rate,mark) VALUES(?,?,?,?)",
                         [(symbol, t, r, m) for t, r, m in rows])

    def load_funding(self, symbol: str, start: int, end: int) -> list[tuple[int, float]]:
        return [(r["funding_time"], r["rate"]) for r in self.query(
            "SELECT funding_time, rate FROM funding WHERE symbol=? AND funding_time>=? AND funding_time<=? "
            "ORDER BY funding_time", (symbol, start, end))]

    def funding_range(self, symbol: str) -> tuple[int | None, int | None]:
        r = self.query("SELECT MIN(funding_time) a, MAX(funding_time) b FROM funding WHERE symbol=?", (symbol,))[0]
        return r["a"], r["b"]

    # --- metrics (open interest / long-short ratios, 5m) ------------------------------------
    def save_metrics(self, symbol: str, rows) -> int:
        data = [(symbol, r.ts, r.oi, r.oi_value, r.top_ls_accounts, r.top_ls_positions, r.ls_accounts, r.taker_ls_vol)
                for r in rows]
        if data:
            self.executemany("INSERT OR REPLACE INTO metrics(symbol,ts,oi,oi_value,top_ls_accounts,top_ls_positions,"
                             "ls_accounts,taker_ls_vol) VALUES(?,?,?,?,?,?,?,?)", data)
        return len(data)

    def load_metrics(self, symbol: str, start: int | None = None, end: int | None = None) -> list[dict]:
        sql = "SELECT * FROM metrics WHERE symbol=?"
        params: list[Any] = [symbol]
        if start is not None:
            sql += " AND ts>=?"
            params.append(start)
        if end is not None:
            sql += " AND ts<=?"
            params.append(end)
        sql += " ORDER BY ts"
        return [dict(r) for r in self.query(sql, params)]

    def metrics_range(self, symbol: str) -> tuple[int | None, int | None]:
        r = self.query("SELECT MIN(ts) a, MAX(ts) b FROM metrics WHERE symbol=?", (symbol,))[0]
        return r["a"], r["b"]

    # --- coverage (gap scans run inside SQLite so multi-year tables never load into memory) --------
    _SERIES = {"candles": "open_time", "funding": "funding_time", "metrics": "ts"}

    def series_stats(self, table: str, symbol: str, start: int | None = None,
                     end: int | None = None) -> tuple[int | None, int | None, int]:
        """(first, last, rows) of one symbol's series in `table` (candles / funding / metrics) within [start, end]."""
        col = self._SERIES[table]
        r = self.query(f"SELECT MIN({col}) a, MAX({col}) b, COUNT(*) n FROM {table} WHERE symbol=? AND {col}>=? "
                       f"AND {col}<=?", (symbol, -(2**62) if start is None else start, 2**62 if end is None else end))[0]
        return r["a"], r["b"], r["n"]

    def series_gaps(self, table: str, symbol: str, step_ms: int, min_missing: int, start: int | None = None,
                    end: int | None = None) -> list[tuple[int, int, int]]:
        """Interior holes of a fixed-step series: (first missing ts, last missing ts, missing steps) for every hole
        of at least `min_missing` steps between consecutive stored rows within [start, end]."""
        col = self._SERIES[table]
        rows = self.query(
            f"SELECT prev, t FROM (SELECT {col} t, LAG({col}) OVER (ORDER BY {col}) prev FROM {table} "
            f"WHERE symbol=? AND {col}>=? AND {col}<=?) WHERE prev IS NOT NULL AND t - prev >= ? ORDER BY t",
            (symbol, -(2**62) if start is None else start, 2**62 if end is None else end, (min_missing + 1) * step_ms))
        return [(r["prev"] + step_ms, r["t"] - step_ms, (r["t"] - r["prev"]) // step_ms - 1) for r in rows]

    # --- positions / trades ------------------------------------------------------------------
    def save_position(self, p: Position) -> None:
        self.execute(
            "INSERT INTO positions(id,engine,symbol,side,status,entry_time,exit_time,data) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, exit_time=excluded.exit_time, data=excluded.data",
            (p.id, p.engine, p.symbol, p.side.value, p.status.value, p.entry_time, p.exit_time,
             json.dumps(p.to_row(), default=str)))

    def load_open_positions(self, engine: str) -> list[Position]:
        rows = self.query("SELECT data FROM positions WHERE engine=? AND status IN ('PENDING','OPEN','CLOSING')", (engine,))
        return [Position.from_row(json.loads(r["data"])) for r in rows]

    def delete_positions(self, engine: str) -> None:
        self.execute("DELETE FROM positions WHERE engine=?", (engine,))

    def save_trade(self, t: TradeRecord) -> None:
        d = t.to_dict()
        cols = ",".join(d.keys())
        qs = ",".join("?" for _ in d)
        self.execute(f"INSERT OR REPLACE INTO trades({cols}) VALUES({qs})", list(d.values()))

    def load_trades(self, engine: str | None = None, since: int | None = None, until: int | None = None,
                    limit: int | None = None, alpha: str | None = None) -> list[dict]:
        sql = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []
        if engine:
            sql += " AND engine=?"
            params.append(engine)
        if since is not None:
            sql += " AND exit_time>=?"
            params.append(since)
        if until is not None:
            sql += " AND exit_time<?"
            params.append(until)
        if alpha:
            sql += " AND alpha=?"
            params.append(alpha)
        sql += " ORDER BY exit_time DESC"
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        out = []
        for r in self.query(sql, params):
            d = dict(r)
            try:
                d["alphas"] = json.loads(d.get("alphas") or "[]")
            except Exception:
                d["alphas"] = []
            out.append(d)
        return out

    def delete_trades(self, engine: str) -> None:
        self.execute("DELETE FROM trades WHERE engine=?", (engine,))

    # --- equity ------------------------------------------------------------------------------
    def save_equity(self, engine: str, ts: int, balance: float, equity: float) -> None:
        self.execute("INSERT OR REPLACE INTO equity(engine,ts,balance,equity) VALUES(?,?,?,?)",
                     (engine, ts, balance, equity))

    def load_equity(self, engine: str, since: int | None = None, limit: int = 2000) -> list[tuple[int, float, float]]:
        sql = "SELECT ts,balance,equity FROM equity WHERE engine=?"
        params: list[Any] = [engine]
        if since is not None:
            sql += " AND ts>=?"
            params.append(since)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = self.query(sql, params)
        return [(r["ts"], r["balance"], r["equity"]) for r in reversed(rows)]

    def delete_equity(self, engine: str) -> None:
        self.execute("DELETE FROM equity WHERE engine=?", (engine,))

    # --- params ------------------------------------------------------------------------------
    def save_params_version(self, vid: str, created: int, role: str, source: str, note: str, params: dict,
                            metrics: dict | None = None) -> None:
        self.execute(
            "INSERT INTO params_versions(id,created,role,source,note,params,metrics) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET role=excluded.role, note=excluded.note, metrics=excluded.metrics",
            (vid, created, role, source, note, json.dumps(params), json.dumps(metrics or {})))

    def set_params_role(self, vid: str, role: str) -> None:
        self.execute("UPDATE params_versions SET role=? WHERE id=?", (role, vid))

    def load_params_versions(self, role: str | None = None, limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM params_versions"
        params: list[Any] = []
        if role:
            sql += " WHERE role=?"
            params.append(role)
        sql += " ORDER BY created DESC LIMIT ?"
        params.append(limit)
        out = []
        for r in self.query(sql, params):
            d = dict(r)
            d["params"] = json.loads(d["params"])
            d["metrics"] = json.loads(d["metrics"] or "{}")
            out.append(d)
        return out

    # --- alpha stats (bandit posteriors) -----------------------------------------------------
    def save_alpha_stat(self, engine: str, alpha: str, regime: str, a: float, b: float, n: int, sum_r: float,
                        updated: int) -> None:
        self.execute(
            "INSERT INTO alpha_stats(engine,alpha,regime,a,b,n,sum_r,updated) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(engine,alpha,regime) DO UPDATE SET a=excluded.a,b=excluded.b,n=excluded.n,"
            "sum_r=excluded.sum_r,updated=excluded.updated", (engine, alpha, regime, a, b, n, sum_r, updated))

    def load_alpha_stats(self, engine: str) -> list[dict]:
        return [dict(r) for r in self.query("SELECT * FROM alpha_stats WHERE engine=?", (engine,))]

    # --- events / research -------------------------------------------------------------------
    def log_event(self, ts: int, level: str, topic: str, message: Any) -> None:
        self.execute("INSERT INTO events(ts,level,topic,message) VALUES(?,?,?,?)",
                     (ts, level, topic, json.dumps(message, default=str)))
        if ts % 97 == 0:  # occasional cleanup
            self.execute("DELETE FROM events WHERE id < (SELECT MAX(id) FROM events) - 5000")

    def load_events(self, limit: int = 100, topic: str | None = None) -> list[dict]:
        sql = "SELECT * FROM events"
        params: list[Any] = []
        if topic:
            sql += " WHERE topic=?"
            params.append(topic)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        out = []
        for r in self.query(sql, params):
            d = dict(r)
            try:
                d["message"] = json.loads(d["message"])
            except Exception:
                pass
            out.append(d)
        return out

    def save_research_run(self, ts: int, alpha: str, summary: dict) -> None:
        self.execute("INSERT INTO research_runs(ts,alpha,summary) VALUES(?,?,?)", (ts, alpha, json.dumps(summary, default=str)))

    def load_research_runs(self, limit: int = 20) -> list[dict]:
        out = []
        for r in self.query("SELECT * FROM research_runs ORDER BY id DESC LIMIT ?", (limit,)):
            d = dict(r)
            d["summary"] = json.loads(d["summary"])
            out.append(d)
        return out

    # --- P4: retention, archived challenger runs, decision records, fill quality ----------------------------
    def prune_funding(self, before_ts: int) -> None:
        self.execute("DELETE FROM funding WHERE funding_time<?", (before_ts,))

    def prune_metrics(self, before_ts: int) -> None:
        self.execute("DELETE FROM metrics WHERE ts<?", (before_ts,))

    def prune_equity(self, before_ts: int) -> None:
        self.execute("DELETE FROM equity WHERE ts<?", (before_ts,))

    def archive_trades(self, engine: str) -> int:
        """Keep the trades of a finished challenger run under a unique engine name ("<slot>@<params version>")
        instead of deleting them when its slot is reused: the record stays auditable, and the next run in the slot
        (queried by its exact engine name) starts from nothing. Returns the number of rows archived."""
        cur = self.execute("UPDATE trades SET engine = engine || '@' || COALESCE(NULLIF(params_version, ''), 'run') "
                           "WHERE engine=?", (engine,))
        return int(cur.rowcount or 0)

    def record_params_decision(self, vid: str, decision: dict, role: str | None = None, note: str | None = None) -> None:
        """Attach a promotion / retirement / kill-switch decision (its inputs, thresholds and verdict) to a stored
        parameter version: merged into its metrics under "decisions", optionally with a new role and note."""
        rows = self.query("SELECT metrics, note, role FROM params_versions WHERE id=?", (vid,))
        if not rows:
            return
        try:
            metrics = json.loads(rows[0]["metrics"] or "{}")
        except Exception:  # noqa: BLE001
            metrics = {}
        metrics.setdefault("decisions", []).append(decision)
        self.execute("UPDATE params_versions SET metrics=?, note=?, role=? WHERE id=?",
                     (json.dumps(metrics, default=str), note if note is not None else rows[0]["note"],
                      role or rows[0]["role"], vid))

    def update_params_metrics(self, vid: str, fields: dict) -> None:
        """Merge `fields` into the metrics of a stored parameter version (its other metrics are kept)."""
        rows = self.query("SELECT metrics FROM params_versions WHERE id=?", (vid,))
        if not rows:
            return
        try:
            metrics = json.loads(rows[0]["metrics"] or "{}")
        except Exception:  # noqa: BLE001
            metrics = {}
        metrics.update(fields)
        self.execute("UPDATE params_versions SET metrics=? WHERE id=?", (json.dumps(metrics, default=str), vid))

    def params_versions_since(self, since: int, source: str | None = None) -> list[dict]:
        """Parameter versions created after `since` (optionally of one source, e.g. "kill-switch"), newest first: the
        champion changes a challenger's promotion must not undo (C2-B)."""
        sql = "SELECT * FROM params_versions WHERE created>?"
        params: list[Any] = [int(since)]
        if source is not None:
            sql += " AND source=?"
            params.append(source)
        out = []
        for r in self.query(sql + " ORDER BY created DESC", params):
            d = dict(r)
            d["params"] = json.loads(d["params"])
            d["metrics"] = json.loads(d["metrics"] or "{}")
            out.append(d)
        return out

    def save_fill_quality(self, row: dict) -> None:
        cols = ("engine", "ts", "symbol", "position_id", "kind", "side", "maker", "qty", "price", "ref", "notional",
                "slip_bps", "fee", "model_fee")
        self.execute(f"INSERT INTO fill_quality({','.join(cols)}) VALUES({','.join('?' for _ in cols)})",
                     [int(row[c]) if c == "maker" else row.get(c) for c in cols])

    def load_fill_quality(self, engine: str, limit: int = 200) -> list[dict]:
        """The newest `limit` fill rows of an engine in the order they were observed (fill timestamps come from the
        account and need not be monotonic: a paper stop carries its ticker's time)."""
        rows = self.query("SELECT * FROM fill_quality WHERE engine=? ORDER BY id DESC LIMIT ?", (engine, limit))
        return [dict(r) for r in reversed(rows)]

    def prune_fill_quality(self, before_ts: int) -> None:
        self.execute("DELETE FROM fill_quality WHERE ts<?", (before_ts,))

    # --- market context (P2a, heartless/data/context.py) --------------------------------------------------------------
    # Created on first write instead of in SCHEMA, so opening a research database only to read it never changes its
    # schema. One row per (symbol, hour close); symbol "*" holds the market-wide features. `data` is a float64 array in
    # the feature order recorded under the kv key "context.meta".
    _CONTEXT_DDL = ("CREATE TABLE IF NOT EXISTS market_context (symbol TEXT NOT NULL, ts INTEGER NOT NULL, "
                    "data BLOB NOT NULL, PRIMARY KEY (symbol, ts)) WITHOUT ROWID")

    def has_market_context(self) -> bool:
        return bool(self.query("SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_context'"))

    def save_market_context(self, rows: Iterable[tuple[str, int, bytes]]) -> None:
        self.execute(self._CONTEXT_DDL)
        self.executemany("INSERT OR REPLACE INTO market_context(symbol,ts,data) VALUES(?,?,?)", rows)

    def load_market_context(self, symbol: str, start: int, end: int) -> list[tuple[int, bytes]]:
        if not self.has_market_context():
            return []
        rows = self.query("SELECT ts, data FROM market_context WHERE symbol=? AND ts>=? AND ts<=? ORDER BY ts",
                          (symbol, int(start), int(end)))
        return [(r["ts"], r["data"]) for r in rows]

    def delete_market_context(self, after: int | None = None) -> None:
        """Every row, or only those of the hour closes after `after` (hours re-opened by appended source history)."""
        if self.has_market_context():
            if after is None:
                self.execute("DELETE FROM market_context")
            else:
                self.execute("DELETE FROM market_context WHERE ts>?", (int(after),))

    def market_context_stats(self) -> dict:
        """First / last hour close and row counts of the stored context (market rows and all rows)."""
        if not self.has_market_context():
            return {"first": None, "last": None, "hours": 0, "rows": 0}
        r = self.query("SELECT MIN(ts) a, MAX(ts) b, COUNT(*) n FROM market_context WHERE symbol='*'")[0]
        total = self.query("SELECT COUNT(*) n FROM market_context")[0]["n"]
        return {"first": r["a"], "last": r["b"], "hours": r["n"], "rows": total}

    def context_series_bounds(self, symbol: str) -> dict[str, list[int] | None]:
        """[first, last] timestamp of a symbol's candles / funding / metrics (None when it has no rows), read from the
        primary-key indexes: the source fingerprint the market-context metadata records, cheap enough to compare on
        every lab or discovery run."""
        out: dict[str, list[int] | None] = {}
        for table, col in self._SERIES.items():
            lo = self.query(f"SELECT {col} t FROM {table} WHERE symbol=? ORDER BY {col} ASC LIMIT 1", (symbol,))
            hi = self.query(f"SELECT {col} t FROM {table} WHERE symbol=? ORDER BY {col} DESC LIMIT 1", (symbol,))
            out[table] = [int(lo[0]["t"]), int(hi[0]["t"])] if lo and hi else None
        return out

    # --- C2-B: the live context store (the running bot's hourly snapshots, appended and pruned) ---------------------
    _MS_H = 3_600_000

    @classmethod
    def _merge_hour_spans(cls, spans) -> list[list[int]]:
        """Sorted, merged [first, last] hour-close spans (adjacent hours join), like data/context.py _merge."""
        out: list[list[int]] = []
        for a, b in sorted([int(x), int(y)] for x, y in spans):
            if out and a <= out[-1][1] + cls._MS_H:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    def add_market_context(self, rows: Iterable[tuple[str, int, bytes]], start: int, end: int, meta: dict,
                           key: str = "context.meta") -> dict:
        """Store snapshot rows of the hour closes [start, end] and record that span as built in the context meta under
        `key` (the stored meta when it is a dict, else `meta`; its other fields are kept). Returns the meta written."""
        rows = list(rows)
        with self._lock:
            if rows:
                self.save_market_context(rows)
            cur = self.get(key)
            m = dict(cur) if isinstance(cur, dict) else dict(meta)
            m["built"] = self._merge_hour_spans(list(m.get("built") or []) + [[int(start), int(end)]])
            self.set(key, m)
            return m

    def prune_market_context(self, before_ts: int, key: str = "context.meta") -> int:
        """Drop the context rows of hour closes before `before_ts` (the data retention) and clip the built spans of the
        meta under `key` to what is left. Returns the number of rows deleted."""
        if not self.has_market_context():
            return 0
        with self._lock:
            n = int(self.execute("DELETE FROM market_context WHERE ts<?", (int(before_ts),)).rowcount or 0)
            meta = self.get(key)
            if isinstance(meta, dict) and meta.get("built"):
                first = -(-(int(before_ts) + 1) // self._MS_H) * self._MS_H - 1  # first hour close >= before_ts
                meta["built"] = [[max(int(a), first), int(b)] for a, b in meta["built"] if int(b) >= first]
                self.set(key, meta)
        return n

    def context_candle_bounds(self, symbol: str) -> tuple[int | None, int | None]:
        """(first, last) candle open_time of a symbol from the primary-key index (no row count, unlike candle_range)."""
        lo = self.query("SELECT open_time t FROM candles WHERE symbol=? ORDER BY open_time ASC LIMIT 1", (symbol,))
        hi = self.query("SELECT open_time t FROM candles WHERE symbol=? ORDER BY open_time DESC LIMIT 1", (symbol,))
        return (lo[0]["t"] if lo else None), (hi[0]["t"] if hi else None)

    def context_candle_columns(self, symbol: str, start: int, end: int, chunk: int = 200_000):
        """open_time, high, low, close, quote_volume of 1m candles with open_time in [start, end] as numpy arrays.

        Read in chunks into arrays (no Candle objects): the context build scans years of 1m history for 20 symbols."""
        import numpy as np

        parts = []
        with self._lock:
            cur = self._conn.cursor()
            cur.row_factory = None
            cur.execute("SELECT open_time, high, low, close, quote_volume FROM candles WHERE symbol=? AND open_time>=? "
                        "AND open_time<=? ORDER BY open_time", (symbol, int(start), int(end)))
            while True:
                rows = cur.fetchmany(chunk)
                if not rows:
                    break
                parts.append(np.array(rows, dtype=float))
            cur.close()
        arr = np.concatenate(parts) if parts else np.zeros((0, 5))
        arr[:, 4] = np.nan_to_num(arr[:, 4], nan=0.0)
        return arr[:, 0].astype(np.int64), arr[:, 1].copy(), arr[:, 2].copy(), arr[:, 3].copy(), arr[:, 4].copy()

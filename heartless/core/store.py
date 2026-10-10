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


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        with self._lock:
            self._conn.executescript(SCHEMA)

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

"""Replay market: the public Binance market data (REST + websocket) served from a stored history in virtual time.

* `ReplaySource` reads a Heartless store (read-only): 1m candles, funding settlements and 5m positioning metrics.
* `ReplayRest` is a `BinanceRest` whose `request()` answers the public GET endpoints the orchestrator uses from that
  store with Binance-shaped JSON (klines, exchangeInfo, 24h tickers, funding history, premium index, open-interest and
  long/short series) and refuses everything else, so the app's own parsing code runs unchanged. Only data that exists
  at the current virtual time is ever returned: closed candles, settled funding, metrics buckets that closed.
* `ReplayMarketStream` is a `MarketStream` that, instead of a websocket, emits every stored 1m candle in virtual time:
  four price points inside the minute (open, first extreme, second extreme, close; bullish bars O-L-H-C, bearish
  O-H-L-C) as bookTicker + markPrice@arr messages with a consistent bid/ask/mark, then the closed kline at the bar's
  close time. Messages go through the stream's real `_dispatch()` as JSON, exactly like websocket frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from typing import Any, Callable

import numpy as np

from heartless.core.models import SymbolInfo
from heartless.exchange.binance_rest import BinanceError, BinanceRest
from heartless.exchange.binance_ws import MarketStream
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE, now_ms

log = logging.getLogger(__name__)

FUNDING_INTERVAL = 8 * MS_HOUR
METRICS_BUCKET = 5 * MS_MINUTE
# price points inside a 1m bar: (offset from open_time in ms, which price); the last one is the close at close_time
PRICE_POINTS = ((1_000, "open"), (20_000, "first"), (40_000, "second"), (59_999, "close"))
_CANDLE_COLS = "open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, close_time"
_CHUNK_MS = MS_DAY


def _s(x: float) -> str:
    """Binance sends prices/quantities as strings; repr round-trips the float exactly."""
    return repr(float(x))


class ReplaySource:
    """Read-only access to a stored history, chunked per symbol for minute-by-minute emission."""

    def __init__(self, db_path: str, symbols: list[str] | None = None, spread_bps: float = 1.0,
                 floor_ms: int | None = None, ceil_ms: int | None = None):
        """Nothing outside [floor_ms, ceil_ms] is ever served (candles, funding, metrics, the funding estimate). The
        replay passes the span its seal check covered (seed start .. end): the bot asks for HISTORY_DAYS of history on
        its own, the funding estimate looks at the next settlement, and the source store may hold a sealed period
        right before or after the window."""
        self.path = str(db_path)
        self.floor = int(floor_ms) if floor_ms is not None else -(2**62)
        self.ceil = int(ceil_ms) if ceil_ms is not None else 2**62
        self._conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        stored = [r[0] for r in self._conn.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")]
        want = [s.upper() for s in symbols] if symbols else stored
        missing = [s for s in want if s not in stored]
        if missing:
            raise ValueError(f"replay source {self.path} has no candles for {', '.join(missing)}")
        self.symbols = want
        self.half_spread = spread_bps / 2e4
        self._chunk: dict[str, tuple[int, int, np.ndarray, np.ndarray]] = {}  # sym -> (lo, hi, open_times, rows)
        self._funding: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._first: dict[str, int] = {}
        for s in self.symbols:
            r = self._conn.execute("SELECT MIN(open_time) FROM candles WHERE symbol=?", (s,)).fetchone()
            self._first[s] = int(r[0])
            rows = self._conn.execute("SELECT funding_time, rate, mark FROM funding WHERE symbol=? AND funding_time>=? "
                                      "AND funding_time<=? ORDER BY funding_time", (s, self.floor, self.ceil)).fetchall()
            self._funding[s] = (np.array([r[0] for r in rows], dtype=np.int64), np.array([r[1] for r in rows], dtype=float),
                                np.array([r[2] or 0.0 for r in rows], dtype=float))

    def close(self) -> None:
        self._conn.close()

    # --- candles -------------------------------------------------------------------------------------
    def coverage(self, symbol: str, start: int, end: int) -> tuple[int | None, int | None, int]:
        r = self._conn.execute("SELECT MIN(open_time), MAX(open_time), COUNT(*) FROM candles WHERE symbol=? AND "
                               "open_time>=? AND open_time<=?", (symbol, start, end)).fetchone()
        return r[0], r[1], int(r[2])

    def first_bar(self, symbol: str) -> int:
        return self._first[symbol]

    def bar(self, symbol: str, open_time: int) -> np.ndarray | None:
        """The stored 1m bar opening at `open_time` as a row (columns of _CANDLE_COLS), or None (gap)."""
        ch = self._chunk.get(symbol)
        if ch is None or not (ch[0] <= open_time < ch[1]):
            lo = open_time - open_time % _CHUNK_MS
            rows = self._conn.execute(f"SELECT {_CANDLE_COLS} FROM candles WHERE symbol=? AND open_time>=? AND "
                                      f"open_time<? ORDER BY open_time", (symbol, lo, lo + _CHUNK_MS)).fetchall()
            arr = np.array(rows, dtype=float).reshape(-1, 10)
            ch = (lo, lo + _CHUNK_MS, arr[:, 0].astype(np.int64), arr)
            self._chunk[symbol] = ch
        i = int(np.searchsorted(ch[2], open_time))
        if i < len(ch[2]) and ch[2][i] == open_time:
            return ch[3][i]
        return None

    def kline_rows(self, symbol: str, start: int | None, end: int | None, limit: int, now: int) -> list[list]:
        """REST /fapi/v1/klines rows for closed bars only (close_time <= now)."""
        now = min(now, self.ceil)
        limit = max(1, min(int(limit or 500), 1500))
        hi = now if end is None else min(int(end), now)
        if start is not None:
            rows = self._conn.execute(f"SELECT {_CANDLE_COLS} FROM candles WHERE symbol=? AND open_time>=? AND "
                                      f"open_time<=? AND close_time<=? ORDER BY open_time LIMIT ?",
                                      (symbol, max(int(start), self.floor), hi, now, limit)).fetchall()
        else:
            rows = self._conn.execute(f"SELECT {_CANDLE_COLS} FROM candles WHERE symbol=? AND open_time>=? AND "
                                      f"open_time<=? AND close_time<=? ORDER BY open_time DESC LIMIT ?",
                                      (symbol, self.floor, hi, now, limit)).fetchall()
            rows.reverse()
        return [[int(r[0]), _s(r[1]), _s(r[2]), _s(r[3]), _s(r[4]), _s(r[5]), int(r[9]), _s(r[6]), int(r[7] or 0),
                 _s(r[8] or 0.0), _s((r[8] or 0.0) * r[4]), "0"] for r in rows]

    def last_close(self, symbol: str, now: int) -> float:
        now = min(now, self.ceil)
        r = self._conn.execute("SELECT close FROM candles WHERE symbol=? AND open_time>=? AND close_time<=? ORDER BY "
                               "open_time DESC LIMIT 1", (symbol, self.floor, now)).fetchone()
        return float(r[0]) if r else 0.0

    def ticker_24h(self, symbol: str, now: int) -> dict | None:
        now = min(now, self.ceil)
        lo = max(now - MS_DAY, self.floor)
        r = self._conn.execute("SELECT SUM(quote_volume), SUM(volume), MAX(high), MIN(low), COUNT(*) FROM candles "
                               "WHERE symbol=? AND open_time>=? AND close_time<=?", (symbol, lo, now)).fetchone()
        if not r or not r[4]:
            return None
        first = self._conn.execute("SELECT open FROM candles WHERE symbol=? AND open_time>=? ORDER BY open_time LIMIT 1",
                                   (symbol, lo)).fetchone()
        last = self.last_close(symbol, now)
        op = float(first[0]) if first else last
        return {"symbol": symbol, "priceChange": _s(last - op), "priceChangePercent": _s((last / op - 1) * 100 if op else 0),
                "weightedAvgPrice": _s(r[0] / r[1] if r[1] else last), "lastPrice": _s(last), "lastQty": "0",
                "openPrice": _s(op), "highPrice": _s(r[2]), "lowPrice": _s(r[3]), "volume": _s(r[1]),
                "quoteVolume": _s(r[0]), "openTime": now - MS_DAY, "closeTime": now, "firstId": 0, "lastId": 0,
                "count": int(r[4])}

    def symbol_info(self, symbol: str, before_ms: int) -> SymbolInfo:
        """Exchange filters inferred from the 30 days before `before_ms` (stored history carries no exchangeInfo)."""
        from heartless.learning.lab import infer_symbol_info

        rows = self._conn.execute("SELECT close FROM candles WHERE symbol=? AND open_time>=? AND open_time<? ORDER BY "
                                  "open_time", (symbol, max(before_ms - 30 * MS_DAY, self.floor), before_ms)).fetchall()
        if not rows:
            rows = self._conn.execute("SELECT close FROM candles WHERE symbol=? AND open_time>=? ORDER BY open_time "
                                      "LIMIT 43200", (symbol, self.floor)).fetchall()
        info = infer_symbol_info(symbol, np.array([r[0] for r in rows], dtype=float))
        info.onboard_date = self._first[symbol]
        return info

    # --- funding -------------------------------------------------------------------------------------
    def funding_rows(self, symbol: str, start: int | None, end: int | None, limit: int, now: int) -> list[dict]:
        t, r, m = self._funding.get(symbol, (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)))
        now = min(now, self.ceil)
        hi = now if end is None else min(int(end), now)
        lo_i = int(np.searchsorted(t, int(start))) if start is not None else 0
        hi_i = int(np.searchsorted(t, hi, side="right"))
        limit = max(1, min(int(limit or 100), 1000))
        idx = range(lo_i, min(hi_i, lo_i + limit)) if start is not None else range(max(lo_i, hi_i - limit), hi_i)
        return [{"symbol": symbol, "fundingTime": int(t[i]), "fundingRate": _s(r[i]),
                 "markPrice": _s(m[i] or self.last_close(symbol, int(t[i])))} for i in idx]

    def funding_upcoming(self, symbol: str, ts: int) -> tuple[float, int]:
        """(r, T) as the markPrice stream shows them at `ts`: T is the next settlement, r the rate for it.

        Binance's r is a running estimate converging to the charged rate; the stored history only has the charged
        values, so r is blended linearly from the last settled rate to the next charged one, reaching it in the last
        minute of the window (the paper account settles with the last r it saw before T, as with the real stream)."""
        nxt = (ts // FUNDING_INTERVAL + 1) * FUNDING_INTERVAL
        t, r, _ = self._funding.get(symbol, (np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)))
        if len(t) == 0:
            return 0.0, nxt
        i = int(np.searchsorted(t, ts, side="right"))  # first settlement strictly after ts
        prev = float(r[i - 1]) if i >= 1 else 0.0
        if i >= len(t) or abs(int(t[i]) - nxt) > MS_MINUTE:
            return prev, nxt  # next settlement not stored (end of history / gap): keep the last settled rate
        w = min(1.0, (ts - (nxt - FUNDING_INTERVAL)) / (FUNDING_INTERVAL - MS_MINUTE))
        return (1 - w) * prev + w * float(r[i]), nxt

    # --- positioning metrics -------------------------------------------------------------------------
    def metrics_rows(self, symbol: str, limit: int, now: int, start: int | None = None,
                     end: int | None = None) -> list[sqlite3.Row]:
        """Archive rows (bucket start ts) whose 5m bucket has closed at `now`, newest `limit`, ascending."""
        hi = min(now, self.ceil) - METRICS_BUCKET
        if end is not None:
            hi = min(hi, int(end) - METRICS_BUCKET)
        limit = max(1, min(int(limit or 30), 500))
        sql = ("SELECT ts, oi, oi_value, top_ls_accounts, top_ls_positions, ls_accounts, taker_ls_vol FROM metrics "
               "WHERE symbol=? AND ts<=? AND ts>=?")
        params: list[Any] = [symbol, hi, self.floor]
        if start is not None:
            sql += " AND ts>=?"
            params.append(int(start) - METRICS_BUCKET)
            sql += " ORDER BY ts LIMIT ?"
            params.append(limit)
            return self._conn.execute(sql, params).fetchall()
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return list(reversed(self._conn.execute(sql, params).fetchall()))


def _bad(code: int, msg: str, status: int = 400) -> BinanceError:
    return BinanceError(code, f"replay: {msg}", status)


class ReplayRest(BinanceRest):
    """Public Binance REST answered from a ReplaySource at the current virtual time; private endpoints refuse."""

    def __init__(self, source: ReplaySource, infos: dict[str, SymbolInfo], latency_ms: int = 0):
        super().__init__("", "", False)
        self.source = source
        self.infos = infos
        self.latency_ms = latency_ms
        self.calls: dict[str, int] = {}
        self._routes: dict[str, Callable[[dict], Any]] = {
            "/fapi/v1/time": lambda p: {"serverTime": now_ms()},
            "/fapi/v1/exchangeInfo": self._exchange_info,
            "/fapi/v1/ticker/24hr": self._ticker_24h,
            "/fapi/v1/klines": self._klines,
            "/fapi/v1/fundingRate": self._funding_rate,
            "/fapi/v1/premiumIndex": self._premium_index,
            "/fapi/v1/ticker/bookTicker": self._book_ticker,
            "/fapi/v1/openInterest": self._open_interest,
            "/futures/data/openInterestHist": lambda p: self._metrics(p, "oi"),
            "/futures/data/topLongShortPositionRatio": lambda p: self._metrics(p, "top_ls_positions"),
            "/futures/data/topLongShortAccountRatio": lambda p: self._metrics(p, "top_ls_accounts"),
            "/futures/data/globalLongShortAccountRatio": lambda p: self._metrics(p, "ls_accounts"),
            "/futures/data/takerlongshortRatio": lambda p: self._metrics(p, "taker_ls_vol"),
        }

    async def request(self, method: str, path: str, params: dict[str, Any] | None = None, signed: bool = False,
                      retries: int = 3) -> Any:
        self.calls[path] = self.calls.get(path, 0) + 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if signed or method != "GET":
            raise _bad(-2015, f"{method} {path} is a private endpoint (no account in a replay)", 401)
        route = self._routes.get(path)
        if route is None:
            raise _bad(-1121, f"{path} is not served", 404)
        return route({k: v for k, v in (params or {}).items() if v is not None})

    # --- routes --------------------------------------------------------------------------------------
    def _known(self, p: dict) -> str:
        sym = str(p.get("symbol", "")).upper()
        if sym not in self.infos:
            raise _bad(-1121, f"Invalid symbol {sym}")
        return sym

    def _listed(self, now: int) -> list[str]:
        return [s for s in self.infos if self.source.first_bar(s) <= now]

    def _exchange_info(self, p: dict) -> dict:
        now = now_ms()
        out = []
        for sym in self._listed(now):
            i = self.infos[sym]
            out.append({"symbol": sym, "pair": sym, "contractType": i.contract_type, "deliveryDate": 4133404800000,
                        "onboardDate": i.onboard_date, "status": i.status, "baseAsset": i.base, "quoteAsset": i.quote,
                        "marginAsset": i.quote, "pricePrecision": i.price_precision,
                        "quantityPrecision": i.quantity_precision, "underlyingType": "COIN",
                        "filters": [{"filterType": "PRICE_FILTER", "tickSize": _s(i.tick_size), "minPrice": _s(i.tick_size),
                                     "maxPrice": "1000000"},
                                    {"filterType": "LOT_SIZE", "stepSize": _s(i.step_size), "minQty": _s(i.min_qty),
                                     "maxQty": "100000000"},
                                    {"filterType": "MARKET_LOT_SIZE", "stepSize": _s(i.step_size), "minQty": _s(i.min_qty),
                                     "maxQty": "100000000"},
                                    {"filterType": "MAX_NUM_ALGO_ORDERS", "limit": i.max_algo_orders},
                                    {"filterType": "MIN_NOTIONAL", "notional": _s(i.min_notional)}],
                        "orderTypes": ["LIMIT", "MARKET", "STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"],
                        "timeInForce": ["GTC", "IOC", "FOK", "GTX"]})
        return {"timezone": "UTC", "serverTime": now, "rateLimits": [], "assets": [], "symbols": out}

    def _ticker_24h(self, p: dict) -> Any:
        now = now_ms()
        if "symbol" in p:
            return self.source.ticker_24h(self._known(p), now) or {}
        return [t for t in (self.source.ticker_24h(s, now) for s in self._listed(now)) if t]

    def _klines(self, p: dict) -> list[list]:
        sym = self._known(p)
        if p.get("interval", "1m") != "1m":
            raise _bad(-1120, f"interval {p.get('interval')} not stored (1m only)")
        return self.source.kline_rows(sym, p.get("startTime"), p.get("endTime"), int(p.get("limit", 500)), now_ms())

    def _funding_rate(self, p: dict) -> list[dict]:
        sym = self._known(p)
        return self.source.funding_rows(sym, p.get("startTime"), p.get("endTime"), int(p.get("limit", 100)), now_ms())

    def _premium_one(self, sym: str, now: int) -> dict:
        px = self.source.last_close(sym, now)
        rate, nxt = self.source.funding_upcoming(sym, now)
        return {"symbol": sym, "markPrice": _s(px), "indexPrice": _s(px), "estimatedSettlePrice": _s(px),
                "lastFundingRate": _s(rate), "interestRate": "0.0001", "nextFundingTime": nxt, "time": now}

    def _premium_index(self, p: dict) -> Any:
        now = now_ms()
        if "symbol" in p:
            return self._premium_one(self._known(p), now)
        return [self._premium_one(s, now) for s in self._listed(now)]

    def _book_one(self, sym: str, now: int) -> dict:
        px = self.source.last_close(sym, now)
        h = self.source.half_spread
        return {"symbol": sym, "bidPrice": _s(px * (1 - h)), "bidQty": "1", "askPrice": _s(px * (1 + h)), "askQty": "1",
                "time": now}

    def _book_ticker(self, p: dict) -> Any:
        now = now_ms()
        if "symbol" in p:
            return self._book_one(self._known(p), now)
        return [self._book_one(s, now) for s in self._listed(now)]

    def _open_interest(self, p: dict) -> dict:
        sym = self._known(p)
        rows = self.source.metrics_rows(sym, 1, now_ms())
        oi = rows[-1][1] if rows and rows[-1][1] else 0.0
        return {"symbol": sym, "openInterest": _s(oi), "time": now_ms()}

    def _metrics(self, p: dict, column: str) -> list[dict]:
        sym = self._known(p)
        if p.get("period", "5m") != "5m":
            raise _bad(-1120, f"period {p.get('period')} not stored (5m only)")
        rows = self.source.metrics_rows(sym, int(p.get("limit", 30)), now_ms(), p.get("startTime"), p.get("endTime"))
        out = []
        for ts, oi, oi_value, top_acc, top_pos, ls_acc, taker in rows:
            stamp = int(ts) + METRICS_BUCKET  # REST stamps the bucket end (heartless.data.extras.rows_from_rest)
            if column == "oi":
                if oi:
                    out.append({"symbol": sym, "sumOpenInterest": _s(oi), "sumOpenInterestValue": _s(oi_value or 0.0),
                                "timestamp": stamp})
            elif column == "taker_ls_vol":
                if taker:
                    out.append({"buySellRatio": _s(taker), "buyVol": "0", "sellVol": "0", "timestamp": stamp})
            else:
                v = {"top_ls_positions": top_pos, "top_ls_accounts": top_acc, "ls_accounts": ls_acc}[column]
                if v:
                    out.append({"symbol": sym, "longShortRatio": _s(v), "longAccount": _s(v / (1 + v)),
                                "shortAccount": _s(1 / (1 + v)), "timestamp": stamp})
        return out


class ReplayMarketStream(MarketStream):
    """MarketStream fed from a ReplaySource in virtual time instead of a websocket (see the module docstring)."""

    def __init__(self, source: ReplaySource, on_kline, on_book, on_mark, start_ms: int, end_ms: int):
        super().__init__("replay://", on_kline, on_book, on_mark)
        self.source = source
        self.start_ms = start_ms - start_ms % MS_MINUTE
        self.end_ms = end_ms  # last emitted bar closes at or before this instant
        self.finished = asyncio.Event()
        self.bars_emitted = 0
        self.messages = 0
        self.minutes = 0
        self.gaps = 0  # (symbol, minute) pairs without a stored bar

    async def _send(self, stream: str, data: Any) -> None:
        self.messages += 1
        self.last_msg_ts = now_ms() / 1000
        await self._dispatch(json.dumps({"stream": stream, "data": data}))

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        sleep_until = getattr(loop, "sleep_until_ms", None)
        if sleep_until is None:
            raise RuntimeError("ReplayMarketStream needs the replay's VirtualTimeLoop")
        self.connected = True
        h = self.source.half_spread
        try:
            m = self.start_ms
            while not self._stop and m + MS_MINUTE - 1 <= self.end_ms:
                syms = list(self.symbols)
                bars = {}
                for s in syms:
                    row = self.source.bar(s, m)
                    if row is None:
                        self.gaps += 1
                    else:
                        bars[s] = row
                for off, which in PRICE_POINTS:
                    await sleep_until(m + off)
                    if self._stop:
                        return
                    ts = m + off
                    marks = []
                    for s, row in bars.items():
                        o, hi, lo, c = row[1], row[2], row[3], row[4]
                        if which == "open":
                            px = o
                        elif which == "close":
                            px = c
                        else:  # bullish bar: low first, bearish: high first (the backtester's flat-path convention)
                            first = lo if c >= o else hi
                            px = first if which == "first" else (hi if first == lo else lo)
                        await self._send(f"{s.lower()}@bookTicker",
                                         {"e": "bookTicker", "u": ts, "s": s, "b": _s(px * (1 - h)), "B": "10",
                                          "a": _s(px * (1 + h)), "A": "10", "T": ts, "E": ts})
                        rate, nxt = self.source.funding_upcoming(s, ts)
                        marks.append({"e": "markPriceUpdate", "E": ts, "s": s, "p": _s(px), "i": _s(px), "P": _s(px),
                                      "r": _s(rate), "T": nxt})
                    if marks:
                        await self._send("!markPrice@arr@1s", marks)
                for s, row in bars.items():
                    await self._send(f"{s.lower()}@kline_1m", {
                        "e": "kline", "E": int(row[9]), "s": s,
                        "k": {"t": int(row[0]), "T": int(row[9]), "s": s, "i": "1m", "f": 0, "L": 0, "o": _s(row[1]),
                              "c": _s(row[4]), "h": _s(row[2]), "l": _s(row[3]), "v": _s(row[5]), "n": int(row[7]),
                              "x": True, "q": _s(row[6]), "V": _s(row[8]), "Q": _s(row[8] * row[4]), "B": "0"}})
                    self.bars_emitted += 1
                self.minutes += 1
                m += MS_MINUTE
            await sleep_until(m + 500)  # let the last bar's processing finish at its own instant
        finally:
            self.connected = False
            self.finished.set()


def symbol_infos(source: ReplaySource, start_ms: int) -> dict[str, SymbolInfo]:
    return {s: source.symbol_info(s, start_ms) for s in source.symbols}

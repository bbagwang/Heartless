"""F0: the exchange-wide USD-M symbol inventory and the 1h research datasets, from the public archive.

Why it exists: every research run used to start from a hand-written symbol list (the 20 long-history coins), which
only holds coins that survived until today. The archive keeps the history of every contract it ever listed, delisted
ones included, so the inventory comes from the bucket listing itself: one row per USD-M symbol with its first and last
monthly kline file, quote asset and contract kind. The 1h datasets built on it (futures, spot, funding) are small
enough to hold every symbol's whole lifetime, which is what a survivorship-free research universe needs
(heartless/data/universe.py ranks them month by month).

What the archive does not say directly
* Delisting: the archive keeps publishing flat zero-volume monthly files for many delisted contracts (SRMUSDT until
  2024-05, 1.5 years after its last trade), so the last FILE is not the delisting month. The 1h build records the last
  hour with trades (`last_trade`); the inventory report uses it once the data is there.
* "...SETTLED" symbols (BNXUSDTSETTLED, AERGOUSDTSETTLED) are archive copies of a contract that settled while its name
  was reused: the same history as the original symbol over the same months, so they are kept as kind "settled" and never
  enter a universe (they would count one market twice). Dated delivery contracts (BTCUSDT_240329) are kind "delivery".
* Multiplier contracts (1000PEPEUSDT, 1000000BOBUSDT, 1MBABYDOGEUSDT) quote 1000 / 1e6 units of the asset; the spot
  market of the asset (PEPEUSDT) is matched with that multiplier.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import re
import time
import zipfile
import zlib
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import numpy as np

from heartless.data.archive import SPOT, UM, monthly_funding_path, monthly_kline_path

QUOTES = ("FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USD1", "BTC", "U")  # matched longest first
_MULT = re.compile(r"^(1000000|100000|10000|1000|1M)(?=[A-Z])")
_DELIVERY = re.compile(r"^[A-Z0-9]+_\d{6}$")
FUTURES_KLINES = f"{UM}/monthly/klines/"
SPOT_KLINES = f"{SPOT}/monthly/klines/"


def parse_symbol(symbol: str) -> dict:
    """Contract kind, quote, base, multiplier and underlying asset of a USD-M archive symbol name."""
    sym = symbol.upper()
    kind = "perpetual"
    core = sym
    if _DELIVERY.match(sym):
        kind, core = "delivery", sym.split("_", 1)[0]
    elif sym.endswith("SETTLED"):
        kind = "settled"
        while core.endswith("SETTLED"):
            core = core[: -len("SETTLED")].rstrip("_")
    quote = next((q for q in sorted(QUOTES, key=len, reverse=True) if core.endswith(q) and len(core) > len(q)), "")
    base = core[: len(core) - len(quote)] if quote else core
    mult = 1.0
    asset = base
    m = _MULT.match(base)
    if m:
        mult = 1e6 if m.group(1) == "1M" else float(m.group(1))
        asset = base[m.end():]
    return {"symbol": sym, "kind": kind, "quote": quote, "base": base, "asset": asset, "multiplier": mult}


def month_of(key: str) -> str | None:
    """"YYYY-MM" of a monthly kline zip key (None for checksum files and other objects)."""
    m = re.search(r"-(\d{4}-\d{2})\.zip$", key)
    return m.group(1) if m else None


def _months_between(a: str, b: str) -> list[str]:
    y, m = map(int, a.split("-"))
    yb, mb = map(int, b.split("-"))
    out = []
    while (y, m) <= (yb, mb):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _file_summary(keys: list[tuple[str, int]]) -> dict:
    files = {}
    for key, size in keys:
        mo = month_of(key)
        if mo is not None:
            files[mo] = int(size)
    months = sorted(files)
    if not months:
        return {"first_month": None, "last_month": None, "months": 0, "missing_months": [], "files": {}}
    missing = [m for m in _months_between(months[0], months[-1]) if m not in files]
    return {"first_month": months[0], "last_month": months[-1], "months": len(months), "missing_months": missing,
            "files": {m: files[m] for m in months}}


async def list_inventory(archive, interval: str = "1h", spot: bool = True,
                         progress: Callable[[str], None] | None = None) -> list[dict]:
    """Every USD-M symbol of the archive with its monthly `interval` kline files (first / last month, gaps, sizes), and
    for the perpetuals the spot market of the same asset quoted in USDT when the archive has one. One bucket listing
    for the symbol names, one per symbol for its files (no object is downloaded)."""
    say = progress or (lambda m: None)
    prefixes, _ = await archive.list_prefix(FUTURES_KLINES)
    symbols = sorted(p[len(FUTURES_KLINES):].strip("/") for p in prefixes)
    say(f"inventory: {len(symbols)} USD-M symbols in the archive listing")
    spot_names: set[str] = set()
    if spot:
        sp, _ = await archive.list_prefix(SPOT_KLINES)
        spot_names = {p[len(SPOT_KLINES):].strip("/") for p in sp}
        say(f"inventory: {len(spot_names)} spot symbols in the archive listing")

    async def one(sym: str) -> dict:
        row = parse_symbol(sym)
        _, keys = await archive.list_prefix(f"{FUTURES_KLINES}{sym}/{interval}/", delimiter=None)
        row.update(_file_summary(keys))
        row["spot_symbol"] = None
        row["spot"] = _file_summary([])
        cand = f"{row['asset']}USDT"
        if spot and row["kind"] == "perpetual" and row["quote"] == "USDT" and cand in spot_names:
            _, skeys = await archive.list_prefix(f"{SPOT_KLINES}{cand}/{interval}/", delimiter=None)
            summary = _file_summary(skeys)
            if summary["months"]:
                row["spot_symbol"], row["spot"] = cand, summary
        return row

    rows: list[dict] = []
    batch = 64
    for i in range(0, len(symbols), batch):
        rows += await asyncio.gather(*(one(s) for s in symbols[i:i + batch]))
        say(f"inventory: listed {min(i + batch, len(symbols))}/{len(symbols)} symbols")
    return rows


# --- compact monthly blocks ----------------------------------------------------------------------------------------------
# One block holds a symbol's hourly bars of one calendar month. The archive files for every USD-M perpetual since 2022
# are about 0.6 GB (futures) + 0.56 GB (spot) as zipped CSV, far too much to keep for a ranking and a basis series, so a
# block keeps what research at hourly resolution reads, in a form that compresses well:
# * prices exactly: the decimal strings as integers on one decimal scale per block (value = int / 10**k, which is the
#   same float64 as parsing the string), close as differences, open against the previous close, high / low as the
#   distance beyond the body, so a quiet hour costs a byte or two;
# * quote volume to 0.1% (log-quantised) and the taker-buy share of it to 0.001: volumes are only ranked, summed or
#   used as ratios, and their last digits are most of the CSV's entropy;
# * the funding rate settled at the bar's OPEN (futures only), exactly, NaN in hours without a settlement. Keyed by
#   the bar's open so a month's block needs only that month's funding file.
# Base volume, trade count and close_time are dropped (derivable or unused). About 6 KB per active futures month and
# 2.5 KB per spot month (close and quote volume only). Layout: zlib(len(header) | JSON header | byte-shuffled arrays).
CODEC = "h1"
QV_STEP = 1e-3  # relative resolution of the stored quote volume
_QK = 1.0 / math.log1p(QV_STEP)
MS_H = 3_600_000
MS_D = 86_400_000
FUT_TABLE, SPOT_TABLE = "klines_1h", "spot_klines_1h"


def _month_ms(month: str) -> int:
    y, m = map(int, month.split("-"))
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _next_month_ms(month: str) -> int:
    y, m = map(int, month.split("-"))
    return _month_ms(f"{y + (m == 12):04d}-{m % 12 + 1:02d}")


def month_str(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m")


def _decimal_ints(values: list[str]) -> tuple[int | None, list[int]]:
    """Exact integers of decimal strings (plain or exponent notation, e.g. funding "8.3E-7") on their common, smallest
    sufficient scale k: value = int / 10**k, which is the same float64 as float(string). (None, []) when they do not fit
    losslessly (more than 12 decimals, 2**53 and beyond, or not a finite number)."""
    parsed = []
    k = 0
    try:
        for v in values:
            d = Decimal(v.strip()).normalize()
            exp = d.as_tuple().exponent
            if not isinstance(exp, int):
                return None, []
            k = max(k, -exp)
            parsed.append(d)
    except InvalidOperation:
        return None, []
    if k > 12:
        return None, []
    out = [int(d.scaleb(k)) for d in parsed]
    if out and max(abs(x) for x in out) >= 2**53:
        return None, []
    return k, out


def _narrow(a: np.ndarray) -> np.ndarray:
    m = int(np.abs(a).max()) if len(a) else 0
    for dt in (np.int8, np.int16, np.int32):
        if m < np.iinfo(dt).max:
            return a.astype(dt)
    return a.astype(np.int64)


def _shuffle(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize).T.tobytes()


def _unshuffle(b: bytes, dtype: str, n: int) -> np.ndarray:
    dt = np.dtype(dtype)
    return np.frombuffer(np.frombuffer(b, dtype=np.uint8).reshape(dt.itemsize, n).T.tobytes(), dtype=dt)


def _pack(arrays: dict[str, np.ndarray], meta: dict) -> bytes:
    hdr = {"v": CODEC, **meta, "a": [[name, a.dtype.str, len(a)] for name, a in arrays.items()]}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    return zlib.compress(len(h).to_bytes(4, "little") + h + b"".join(_shuffle(a) for a in arrays.values()), 9)


def _unpack(blob: bytes) -> tuple[dict, dict[str, np.ndarray]]:
    raw = zlib.decompress(blob)
    hl = int.from_bytes(raw[:4], "little")
    hdr = json.loads(raw[4:4 + hl])
    pos = 4 + hl
    arrays = {}
    for name, dtype, n in hdr["a"]:
        size = np.dtype(dtype).itemsize * n
        arrays[name] = _unshuffle(raw[pos:pos + size], dtype, n)
        pos += size
    return hdr, arrays


def _csv_rows(blob: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next((n for n in z.namelist() if n.endswith(".csv")), None)
        if name is None:
            return []
        text = z.read(name).decode("utf-8", errors="replace")
    return [r for r in csv.reader(io.StringIO(text)) if r and r[0].strip().isdigit()]


def kline_rows(blob: bytes, month: str) -> list[list[str]]:
    """The month's archive kline rows as strings, open_time normalised to ms (2025+ spot files use microseconds),
    sorted, one per hour, restricted to bars that open inside the month."""
    a, b = _month_ms(month), _next_month_ms(month)
    by_t: dict[int, list[str]] = {}
    for r in _csv_rows(blob):
        if len(r) < 11:
            continue
        t = int(r[0])
        if t > 10**14:
            t //= 1000
        if a <= t < b:
            by_t[t - t % MS_H] = r
    return [[str(t)] + by_t[t][1:] for t in sorted(by_t)]


def funding_rows(blob: bytes | None) -> list[tuple[int, str]]:
    """(settlement time snapped to the hour as data/archive.parse_funding does, rate string) of a funding file."""
    out = []
    for r in _csv_rows(blob) if blob else []:
        if len(r) < 3:
            continue
        t = int(r[0])
        t = t - (t % MS_H) if t % MS_H < 60_000 else t
        out.append((t, r[2].strip()))
    return out


def encode_month(rows: list[list[str]], month: str, funding: list[tuple[int, str]] | None = None,
                 spot: bool = False) -> tuple[bytes, dict]:
    """Block bytes and summary (n, first_ts, last_ts, first_trade, last_trade, med_dqv, days_traded) of one month of
    hourly archive rows (kline_rows). med_dqv is the median over the month's calendar days of the daily quote volume,
    days without bars counting as 0, so a symbol listed late in the month or delisted early in it ranks low."""
    m0 = _month_ms(month)
    n = len(rows)
    t = np.array([int(r[0]) for r in rows], dtype=np.int64)
    qv = np.array([float(r[7]) for r in rows]) if n else np.zeros(0)
    arrays: dict[str, np.ndarray] = {"t": _narrow(np.diff((t - m0) // MS_H, prepend=0))}
    meta: dict = {"m": month, "n": n}
    cols = (4,) if spot else (1, 2, 3, 4)
    k, ints = _decimal_ints([r[j] for j in cols for r in rows])
    if k is None:  # cannot be held exactly as integers: raw float64 (never seen in the archive so far)
        meta["k"] = None
        for j, name in zip(cols, ("o", "h", "l", "c")[-len(cols):]):
            arrays[name + "64"] = np.array([float(r[j]) for r in rows])
    else:
        meta["k"] = k
        p = np.array(ints, dtype=np.int64).reshape(len(cols), n)
        c = p[-1]
        arrays["c"] = _narrow(np.diff(c, prepend=0))
        if not spot:
            o, h, lo = p[0], p[1], p[2]
            prev = np.concatenate((c[:1], c[:-1]))
            arrays["o"] = _narrow(o - prev)
            arrays["h"] = _narrow(h - np.maximum(o, c))
            arrays["l"] = _narrow(np.minimum(o, c) - lo)
    pos = qv > 0
    arrays["q"] = _narrow(np.where(pos, np.round(np.log1p(np.where(pos, qv, 0.0)) * _QK), 0).astype(np.int64))
    if not spot:
        tbq = np.array([float(r[10]) for r in rows]) if n else np.zeros(0)
        share = np.where(pos, tbq / np.where(pos, qv, 1.0), np.nan)
        arrays["tb"] = np.where(np.isfinite(share), np.round(np.clip(share, 0, 1) * 1000), -1).astype(np.int16)
        fr = sorted((ft, v) for ft, v in (funding or []) if m0 <= ft < _next_month_ms(month))
        kf, fints = _decimal_ints([v for _, v in fr])
        if kf is None:
            meta["kf"] = None
            arrays["fi"] = np.array([(ft - m0) // MS_H for ft, _ in fr], dtype=np.int16)
            arrays["f64"] = np.array([float(v) for _, v in fr])
        else:
            meta["kf"] = kf
            arrays["fi"] = np.array([(ft - m0) // MS_H for ft, _ in fr], dtype=np.int16)
            arrays["f"] = _narrow(np.array(fints, dtype=np.int64))
    blob = _pack(arrays, meta)
    days = (_next_month_ms(month) - m0) // MS_D
    daily = np.zeros(days)
    if n:
        np.add.at(daily, (t - m0) // MS_D, qv)
    traded = t[pos]
    summary = {"n": n, "first_ts": int(t[0]) if n else None, "last_ts": int(t[-1]) if n else None,
               "first_trade": int(traded[0]) if len(traded) else None,
               "last_trade": int(traded[-1]) if len(traded) else None,
               "med_dqv": float(np.median(daily)), "days_traded": int((daily > 0).sum())}
    return blob, summary


def decode_month(blob: bytes) -> dict[str, np.ndarray]:
    """Columns of a block: open_time (int64 ms), close, quote_volume and, for futures blocks, open, high, low,
    taker_buy_share (NaN when the hour had no volume) and funding (rate settled at the bar's open, NaN otherwise)."""
    hdr, a = _unpack(blob)
    m0 = _month_ms(hdr["m"])
    t = m0 + np.cumsum(a["t"].astype(np.int64)) * MS_H
    out: dict[str, np.ndarray] = {"open_time": t}
    k = hdr.get("k")
    if k is None:
        for name in ("o", "h", "l", "c"):
            if name + "64" in a:
                out[{"o": "open", "h": "high", "l": "low", "c": "close"}[name]] = a[name + "64"].astype(float)
    else:
        scale = float(10 ** k)
        c = np.cumsum(a["c"].astype(np.int64))
        out["close"] = c / scale
        if "o" in a:
            prev = np.concatenate((c[:1], c[:-1]))
            o = prev + a["o"].astype(np.int64)
            out["open"] = o / scale
            out["high"] = (np.maximum(o, c) + a["h"].astype(np.int64)) / scale
            out["low"] = (np.minimum(o, c) - a["l"].astype(np.int64)) / scale
    q = a["q"].astype(np.int64)
    out["quote_volume"] = np.where(q > 0, np.expm1(q / _QK), 0.0)
    if "tb" in a:
        tb = a["tb"].astype(float)
        out["taker_buy_share"] = np.where(tb >= 0, tb / 1000.0, np.nan)
        fund = np.full(len(t), np.nan)
        hours = (t - m0) // MS_H
        fi = a["fi"].astype(np.int64)
        vals = a["f"].astype(np.int64) / float(10 ** hdr["kf"]) if "f" in a else a.get("f64", np.zeros(0))
        pos = np.searchsorted(hours, fi)
        ok = (pos < len(hours)) & (hours[np.minimum(pos, max(len(hours) - 1, 0))] == fi) if len(hours) else fi < 0
        fund[pos[ok]] = vals[ok]
        out["funding"] = fund
    return out


def load_hourly(store, symbol: str, start: int | None = None, end: int | None = None,
                table: str = FUT_TABLE) -> dict[str, np.ndarray]:
    """A symbol's stored hourly bars with open_time in [start, end] (ms) as concatenated columns (decode_month)."""
    lo = None if start is None else _month_ms(month_str(start))
    parts = [decode_month(b) for _, b in store.load_kline_blocks(table, symbol, lo, end)]
    if not parts:
        return {"open_time": np.zeros(0, dtype=np.int64)}
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    cols = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    sel = np.ones(len(cols["open_time"]), dtype=bool)
    if start is not None:
        sel &= cols["open_time"] >= int(start)
    if end is not None:
        sel &= cols["open_time"] <= int(end)
    return {k: v[sel] for k, v in cols.items()}


# --- inventory rows in the store -------------------------------------------------------------------------------------------
def inventory_rows(listed: list[dict], now: int | None = None) -> list[dict]:
    """symbol_inventory rows of a list_inventory() result."""
    now = int(time.time() * 1000) if now is None else int(now)
    out = []
    for r in listed:
        sp = r.get("spot") or {}
        out.append({"symbol": r["symbol"], "kind": r["kind"], "quote": r["quote"], "base": r["base"],
                    "asset": r["asset"], "multiplier": r["multiplier"], "first_month": r["first_month"],
                    "last_month": r["last_month"], "months": r["months"], "missing_months": r["missing_months"],
                    "spot_symbol": r.get("spot_symbol"), "spot_first_month": sp.get("first_month"),
                    "spot_last_month": sp.get("last_month"), "spot_months": sp.get("months") or 0, "updated": now})
    return out


def archive_months(first: str | None, last: str | None, missing: list[str], start: int, end: int) -> list[str]:
    """The months [first, last] the archive lists for a symbol (minus its missing ones) that overlap [start, end]."""
    if not first or not last:
        return []
    lo, hi = max(first, month_str(start)), min(last, month_str(end))
    if hi < lo:
        return []
    gone = set(missing or [])
    return [m for m in _months_between(lo, hi) if m not in gone]


async def refresh_inventory(store, archive, progress=None) -> list[dict]:
    rows = inventory_rows(await list_inventory(archive, progress=progress))
    store.save_symbol_inventory(rows)
    return store.load_symbol_inventory()


# --- builds ----------------------------------------------------------------------------------------------------------------
async def _build(store, archive, table: str, plan: dict[str, list[str]], fetch, progress=None,
                 symbols_at_once: int = 8) -> dict:
    """Download and store the planned (symbol -> months) blocks of `table`, symbol by symbol (each symbol's blocks are
    saved together, so an interrupted build resumes at the first symbol not finished). `fetch(symbol, month)` returns
    (block, summary) or None when the archive did not serve the month."""
    say = progress or (lambda m: None)
    stats = {"symbols": 0, "blocks": 0, "empty": 0, "failed": [], "bytes": 0}
    todo = sorted(plan.items())
    done = 0
    t0 = time.time()
    if todo:  # one request first: the archive remembers which endpoint answers before the burst of parallel requests
        try:
            await archive.get(monthly_kline_path(todo[0][0], todo[0][1][0], market="spot" if table == SPOT_TABLE
                                                 else "um"))
        except Exception:  # noqa: BLE001 - the real requests below report their own failures
            pass

    async def one(sym: str, months: list[str]) -> None:
        rows = []
        for mo, res in zip(months, await asyncio.gather(*(fetch(sym, mo) for mo in months), return_exceptions=True)):
            if isinstance(res, BaseException):
                stats["failed"].append(f"{sym} {mo}: {type(res).__name__}")
                continue
            if res is None:
                rows.append((sym, _month_ms(mo), 0, None, None, None, None, 0.0, 0, b""))
                stats["empty"] += 1
                continue
            blob, s = res
            rows.append((sym, _month_ms(mo), s["n"], s["first_ts"], s["last_ts"], s["first_trade"], s["last_trade"],
                         s["med_dqv"], s["days_traded"], blob))
            stats["bytes"] += len(blob)
        stats["blocks"] += store.save_kline_blocks(table, rows)
        stats["symbols"] += 1

    for i in range(0, len(todo), symbols_at_once):
        part = todo[i:i + symbols_at_once]
        await asyncio.gather(*(one(s, m) for s, m in part))
        done += len(part)
        if progress and (done % (symbols_at_once * 8) == 0 or done == len(todo)):
            say(f"{table}: {done}/{len(todo)} symbols, {stats['blocks']} blocks, {stats['bytes'] / 1e6:.1f} MB, "
                f"{time.time() - t0:.0f}s")
    return stats


def _plan(store, table: str, wanted: dict[str, list[str]], rebuild: bool) -> dict[str, list[str]]:
    have = set() if rebuild else {(r["symbol"], month_str(r["month"])) for r in store.kline_block_index(table)}
    plan = {s: [m for m in months if (s, m) not in have] for s, months in wanted.items()}
    return {s: m for s, m in plan.items() if m}


async def build_futures(store, archive, start: int, end: int, quotes=("USDT",), symbols=None, funding: bool = True,
                        rebuild: bool = False, progress=None) -> dict:
    """1h futures blocks (with the month's funding settlements) for every perpetual of the inventory quoted in `quotes`
    over the archive months of its lifetime that overlap [start, end], plus the month before `start` (the ranking of a
    month reads the previous one). Resumable; `rebuild` downloads again."""
    from heartless.data.universe import prev_month

    start = _month_ms(prev_month(month_str(start)))
    inv = [r for r in store.load_symbol_inventory(kind="perpetual") if r["quote"] in quotes]
    if symbols:
        inv = [r for r in inv if r["symbol"] in set(symbols)]
    wanted = {r["symbol"]: archive_months(r["first_month"], r["last_month"], r["missing_months"], start, end)
              for r in inv}
    plan = _plan(store, FUT_TABLE, wanted, rebuild)
    if progress:
        progress(f"{FUT_TABLE}: {sum(len(v) for v in plan.values())} months of {len(plan)} symbols to download "
                 f"({sum(len(v) for v in wanted.values())} in the window)")

    async def fetch(sym: str, month: str):
        kb, fb = await asyncio.gather(archive.get(monthly_kline_path(sym, month)),
                                      archive.get(monthly_funding_path(sym, month)) if funding else _none())
        if kb is None:
            return None
        return encode_month(kline_rows(kb, month), month, funding_rows(fb))

    stats = await _build(store, archive, FUT_TABLE, plan, fetch, progress)
    store.refresh_inventory_trades(FUT_TABLE)
    return stats


async def build_spot(store, archive, start: int, end: int, symbols=None, rebuild: bool = False, progress=None) -> dict:
    """1h spot blocks (close, quote volume) for every USDT perpetual of the inventory with a spot USDT market of the same
    asset, over the months both markets were listed that overlap [start, end] (spot history before the perpetual
    existed is of no use to a basis). Stored under the spot symbol."""
    inv = [r for r in store.load_symbol_inventory(kind="perpetual") if r["quote"] == "USDT" and r["spot_symbol"]]
    if symbols:
        inv = [r for r in inv if r["symbol"] in set(symbols) or r["spot_symbol"] in set(symbols)]
    wanted: dict[str, list[str]] = {}
    for r in inv:
        perp = set(archive_months(r["first_month"], r["last_month"], r["missing_months"], start, end))
        spot = archive_months(r["spot_first_month"], r["spot_last_month"], [], start, end)
        wanted.setdefault(r["spot_symbol"], [])
        wanted[r["spot_symbol"]] = sorted(set(wanted[r["spot_symbol"]]) | {m for m in spot if m in perp})
    plan = _plan(store, SPOT_TABLE, wanted, rebuild)
    if progress:
        progress(f"{SPOT_TABLE}: {sum(len(v) for v in plan.values())} months of {len(plan)} spot symbols to download")

    async def fetch(sym: str, month: str):
        kb = await archive.get(monthly_kline_path(sym, month, market="spot"))
        return None if kb is None else encode_month(kline_rows(kb, month), month, spot=True)

    return await _build(store, archive, SPOT_TABLE, plan, fetch, progress)


async def _none():
    return None


# --- basis (F2 carry input; read-only, no statistics) --------------------------------------------------------------------
def basis_series(store, symbol: str, start: int | None = None, end: int | None = None,
                 row: dict | None = None) -> dict[str, np.ndarray]:
    """Hourly perpetual-vs-spot basis of a USDT perpetual from the stored 1h blocks, one row per hour close T where both
    markets have a bar closing at T:
    ts = T (bar open + 1 h), perp / spot = the two closes (spot times the contract multiplier, 1000 for 1000PEPEUSDT),
    basis = (perp - spot) / spot, funding = the rate settled at T (NaN when no settlement at T), funding_last = the
    last rate settled at or before T, perp_qv / spot_qv = the hour's quote volumes (0 in an hour without trades, e.g.
    the flat files the archive keeps publishing after a delisting). Empty arrays when the symbol has no spot market or
    no stored blocks. `row`: the
    symbol's inventory row when the caller has it (read from the store otherwise)."""
    r = row if row is not None else next((x for x in store.load_symbol_inventory() if x["symbol"] == symbol), None)
    empty = {k: np.zeros(0) for k in ("ts", "perp", "spot", "basis", "funding", "funding_last", "perp_qv", "spot_qv")}
    if r is None or not r.get("spot_symbol"):
        return empty
    # a day before `start`, so funding_last is known from the first row on (settlements are at most 8 h apart)
    fut = load_hourly(store, symbol, None if start is None else start - MS_D, end, FUT_TABLE)
    spot = load_hourly(store, r["spot_symbol"], None if start is None else start - MS_H, end, SPOT_TABLE)
    if not len(fut["open_time"]) or not len(spot["open_time"]):
        return empty
    t_f = fut["open_time"] + MS_H
    common, i_f, i_s = np.intersect1d(t_f, spot["open_time"] + MS_H, return_indices=True)
    # funding settled at T is stored on the bar that OPENS at T: shift by one bar via the time key
    f_t = fut["open_time"][np.isfinite(fut["funding"])]
    f_v = fut["funding"][np.isfinite(fut["funding"])]
    at = np.searchsorted(f_t, common)
    hit = (at < len(f_t)) & (f_t[np.minimum(at, max(len(f_t) - 1, 0))] == common) if len(f_t) else np.zeros(len(common), bool)
    fund = np.full(len(common), np.nan)
    fund[hit] = f_v[at[hit]]
    last_i = np.searchsorted(f_t, common, side="right") - 1
    f_last = np.where(last_i >= 0, f_v[np.maximum(last_i, 0)] if len(f_v) else np.nan, np.nan)
    perp = fut["close"][i_f]
    sp = spot["close"][i_s] * float(r.get("multiplier") or 1.0)
    sel = np.ones(len(common), dtype=bool)
    if start is not None:
        sel &= common >= int(start)
    if end is not None:
        sel &= common <= int(end)
    with np.errstate(all="ignore"):
        basis = (perp - sp) / sp
    return {"ts": common[sel], "perp": perp[sel], "spot": sp[sel], "basis": basis[sel], "funding": fund[sel],
            "funding_last": f_last[sel], "perp_qv": fut["quote_volume"][i_f][sel],
            "spot_qv": spot["quote_volume"][i_s][sel]}


# --- reports ---------------------------------------------------------------------------------------------------------------
ACTIVE_SLACK_MS = 3 * MS_D  # a symbol with trades in the last 3 days of the stored data counts as listed at its end


def _day(ms) -> str | None:
    return None if ms is None else datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def data_end(store, table: str = FUT_TABLE) -> int | None:
    """The last hour open stored in `table` (the end of the inventory's knowledge about trading)."""
    if not store.f0_has(table):
        return None
    r = store.query(f"SELECT MAX(last_ts) t FROM {table} WHERE n>0")[0]["t"]
    return None if r is None else int(r)


def symbol_status(row: dict, end: int | None, window_start: int | None, built: bool = False) -> str:
    """active / delisted (no trades in the last ACTIVE_SLACK_MS of the stored data) / delisted_before_window (every
    archive month is older than the built window, or its built months are all flat files without a trade) / no_data
    (not built)."""
    if row.get("last_trade") is not None and end is not None:
        return "active" if row["last_trade"] >= end - ACTIVE_SLACK_MS else "delisted"
    if window_start is not None and row.get("last_month") and _month_ms(row["last_month"]) < window_start:
        return "delisted_before_window"
    return "delisted_before_window" if built else "no_data"


def inventory_report(store) -> dict:
    """Counts of the inventory (kinds, quotes) and, for USDT perpetuals, listings and delistings per year, plus one
    compact line per symbol: the exchange-wide symbol list research uses instead of a hand-written one."""
    from heartless.data.universe import listed_at

    inv = store.load_symbol_inventory()
    end = data_end(store)
    idx = store.kline_block_index(FUT_TABLE)
    w0 = min((r["month"] for r in idx), default=None)
    built = {r["symbol"] for r in idx if r["n"]}
    by_kind: dict[str, int] = {}
    by_quote: dict[str, int] = {}
    listed: dict[str, int] = {}
    delisted: dict[str, int] = {}
    status_n: dict[str, int] = {}
    lines = []
    for r in inv:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        by_quote[r["quote"] or "?"] = by_quote.get(r["quote"] or "?", 0) + 1
        st = symbol_status(r, end, w0, r["symbol"] in built)
        la = listed_at(r)
        if r["kind"] == "perpetual" and r["quote"] == "USDT":
            status_n[st] = status_n.get(st, 0) + 1
            if la is not None:
                y = _day(la)[:4]
                listed[y] = listed.get(y, 0) + 1
            if st == "delisted":
                y = _day(r["last_trade"])[:4]
                delisted[y] = delisted.get(y, 0) + 1
            elif st == "delisted_before_window":
                delisted["before_window"] = delisted.get("before_window", 0) + 1
        lines.append([r["symbol"], r["kind"], r["quote"], r["first_month"], r["last_month"], _day(la),
                      _day(r.get("last_trade")), st, r.get("spot_symbol")])
    return {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "Binance public archive bucket listing data/futures/um/monthly/klines/ (1h files) + the 1h blocks",
            "window": [_day(w0), _day(end)], "data_end": _day(end),
            "counts": {"symbols": len(inv), "by_kind": dict(sorted(by_kind.items())),
                       "by_quote": dict(sorted(by_quote.items())),
                       "perpetual_usdt": sum(1 for r in inv if r["kind"] == "perpetual" and r["quote"] == "USDT"),
                       "perpetual_usdt_status": dict(sorted(status_n.items())),
                       "with_spot_usdt_market": sum(1 for r in inv if r.get("spot_symbol"))},
            "perpetual_usdt_listed_per_year": dict(sorted(listed.items())),
            "perpetual_usdt_delisted_per_year": dict(sorted(delisted.items())),
            "notes": ["listed = first hour with trades when the archive's first 1h month is inside the built window, "
                      "else the start of that first month (the 1h archive starts 2020-01: earlier listings count there)",
                      "delisted = no trades in the last 3 days of the stored data (the archive keeps publishing flat "
                      "files after a delisting, so the last file month is not used)",
                      "kind settled = an archive copy of a contract settled while its name was reused (same history as "
                      "the original symbol); delivery = dated quarterly contract; neither enters a universe"],
            "columns": ["symbol", "kind", "quote", "first_month", "last_month", "listed", "last_trade", "status",
                        "spot_symbol"],
            "symbols": lines}


def spot_coverage(store) -> dict:
    """Spot side of the carry datasets: perpetuals with a spot USDT market in the archive, those with stored spot
    blocks, the first stored month, and a sanity check of the pairing (the median |basis| over the hours both markets
    traded; above 5% the spot market is probably another asset with the same ticker, or a redenominated one). Data
    checks only, no performance statistic."""
    inv = [r for r in store.load_symbol_inventory(kind="perpetual") if r["quote"] == "USDT" and r.get("spot_symbol")]
    idx: dict[str, list[dict]] = {}
    for b in store.kline_block_index(SPOT_TABLE):
        idx.setdefault(b["symbol"], []).append(b)
    per = {}
    first_months: dict[str, int] = {}
    for r in inv:
        blocks = [b for b in idx.get(r["spot_symbol"], []) if b["n"]]
        if not blocks:
            per[r["symbol"]] = {"spot_symbol": r["spot_symbol"], "stored_months": 0}
            continue
        fm = month_str(blocks[0]["month"])
        first_months[fm[:4]] = first_months.get(fm[:4], 0) + 1
        bs = basis_series(store, r["symbol"], row=r)
        both = np.isfinite(bs["basis"]) & (bs["perp_qv"] > 0) & (bs["spot_qv"] > 0)
        b = bs["basis"][both]
        med = float(np.median(np.abs(b))) if len(b) else None
        per[r["symbol"]] = {"spot_symbol": r["spot_symbol"], "multiplier": r["multiplier"], "first_month": fm,
                            "last_month": month_str(blocks[-1]["month"]), "stored_months": len(blocks),
                            "hours_both_traded": int(len(b)), "median_abs_basis": med,
                            "paired": med is not None and med < 0.05}
    stored = [s for s, v in per.items() if v["stored_months"]]
    return {"perpetuals_with_spot_market": len(inv), "with_stored_spot": len(stored),
            "paired_ok": sum(1 for s in stored if per[s].get("paired")),
            "suspect_pairs": sorted(s for s in stored if per[s]["median_abs_basis"] is not None
                                    and not per[s]["paired"]),
            "unverified_pairs": sorted(s for s in stored if per[s]["median_abs_basis"] is None),
            "first_spot_month_per_year": dict(sorted(first_months.items())), "symbols": per}


# --- CLI: heartless universe ... --------------------------------------------------------------------------------------------
def _when(text: str | None, end: bool = False) -> int | None:
    if not text:
        return None
    dt = datetime.fromisoformat(text)
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    ms = int(dt.timestamp() * 1000)
    return ms - 1 if end and len(text) <= 10 else ms  # a date-only end is exclusive


def dumps_report(doc: dict) -> str:
    """JSON with one line per top-level key, and one line per element for the long lists / maps (the per-symbol and
    per-month tables), so a report file diffs line by line when the inventory changes."""
    def one(v) -> str:
        return json.dumps(v, default=str, ensure_ascii=False)

    parts = []
    for k, v in doc.items():
        if isinstance(v, list) and len(v) > 8:
            parts.append(f" {one(k)}: [\n" + ",\n".join(f"  {one(x)}" for x in v) + "\n ]")
        elif isinstance(v, dict) and len(v) > 8:
            parts.append(f" {one(k)}: {{\n" + ",\n".join(f"  {one(kk)}: {one(vv)}" for kk, vv in v.items()) + "\n }")
        else:
            parts.append(f" {one(k)}: {one(v)}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def _write(path: str | None, doc: dict) -> None:
    if not path:
        return
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(dumps_report(doc))
    print(f"wrote {path}")


def main_cli(settings, args) -> None:
    """heartless universe inventory | build [--spot] | show | resolve | status (docs/UNIVERSE.md)."""
    import resource
    import sys

    from heartless.core.store import Store
    from heartless.data.archive import BinanceArchive
    from heartless.data import universe as U

    path = settings.universe_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    say = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    try:
        if args.action in ("inventory", "build"):
            async def run() -> dict:
                out: dict = {}
                async with BinanceArchive(concurrency=12) as arch:
                    t0 = time.time()
                    if args.action == "inventory" or not store.load_symbol_inventory():
                        await refresh_inventory(store, arch, progress=say)
                        out["inventory_seconds"] = round(time.time() - t0, 1)
                    if args.action == "build":
                        start = _when(args.start) or _month_ms("2022-01")
                        end = _when(args.end, end=True) or int(time.time() * 1000)
                        syms = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
                        t1 = time.time()
                        if args.spot:
                            out["spot"] = await build_spot(store, arch, start, end, syms, args.rebuild, say)
                        else:
                            out["futures"] = await build_futures(store, arch, start, end, symbols=syms,
                                                                 rebuild=args.rebuild, progress=say)
                        out["build_seconds"] = round(time.time() - t1, 1)
                    out["downloaded_mb"] = round(arch.downloaded_bytes / 1e6, 1)
                return out

            res = asyncio.run(run())
            for k in ("futures", "spot"):
                if k in res:
                    st = res[k]
                    print(f"{k}: {st['symbols']} symbols, {st['blocks']} blocks ({st['empty']} not served), "
                          f"{st['bytes'] / 1e6:.1f} MB stored, {len(st['failed'])} failed"
                          + (f": {', '.join(st['failed'][:10])}" if st["failed"] else ""))
            print(f"downloaded {res['downloaded_mb']} MB; {res.get('build_seconds', res.get('inventory_seconds'))}s; "
                  f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB -> {path}")
            if args.spot:
                cov = spot_coverage(store)
                print(f"spot: {cov['perpetuals_with_spot_market']} perpetuals with a spot USDT market, "
                      f"{cov['with_stored_spot']} stored, {cov['paired_ok']} pass the pairing check; suspect: "
                      f"{', '.join(cov['suspect_pairs']) or '-'}; no hour both traded: "
                      f"{', '.join(cov['unverified_pairs']) or '-'}")
                _write(args.out, cov)
            else:
                rep = inventory_report(store)
                c = rep["counts"]
                print(f"inventory: {c['symbols']} symbols ({c['by_kind']}), {c['perpetual_usdt']} USDT perpetuals "
                      f"{c['perpetual_usdt_status']}; delisted per year {rep['perpetual_usdt_delisted_per_year']}")
                _write(args.out, rep)
        elif args.action == "show":
            if not args.month:
                raise ValueError("show needs --month YYYY-MM")
            ranked = U.month_ranking(store, args.month)
            if not ranked:
                raise ValueError(f"no ranking for {args.month}: build the 1h data of {U.prev_month(args.month)} first")
            end = data_end(store)
            print(f"{args.month}: {len(ranked)} ranked USDT perpetuals by median daily quote volume of "
                  f"{U.prev_month(args.month)} (top {args.top})")
            print(f"{'#':>3s} {'symbol':16s} {'med daily qv':>16s} {'days':>4s} {'listed':10s} {'last trade':10s} spot")
            for r in ranked[:args.top]:
                gone = r["last_trade"] is not None and end is not None and r["last_trade"] < end - ACTIVE_SLACK_MS
                print(f"{r['rank']:>3d} {r['symbol']:16s} {r['med_dqv']:>16,.0f} {r['days_traded']:>4d} "
                      f"{_day(r['listed'])} {(_day(r['last_trade']) or '-') + (' (delisted)' if gone else '')} "
                      f"{r['spot_symbol'] or '-'}")
        elif args.action == "resolve":
            start, end = _when(args.start), _when(args.end, end=True)
            if start is None or end is None:
                raise ValueError("resolve needs --start and --end")
            res = U.research_universe(store, start, end, args.top)
            last = data_end(store)
            gone = {s for s, t in res["last_trade"].items()
                    if t is not None and last is not None and t <= end and t < last - ACTIVE_SLACK_MS}
            print(f"top{args.top} {args.start} .. {args.end}: {len(res['symbols'])} symbols over {len(res['months'])} "
                  f"ranked months ({len(gone)} stopped trading inside the window); unranked months: "
                  f"{', '.join(res['unranked']) or '-'}")
            print(", ".join(res["symbols"]))
            _write(args.out, res)
        else:  # status
            for t in (FUT_TABLE, SPOT_TABLE):
                st = store.kline_block_stats(t)
                print(f"{t}: {st['symbols']} symbols, {st['blocks']} blocks ({st['empty']} not served), "
                      f"{st['rows']:,} hourly rows, {st['bytes'] / 1e6:.1f} MB")
            print(f"inventory: {len(store.load_symbol_inventory())} symbols; data end {_day(data_end(store)) or '-'}; "
                  f"file {path} ({path.stat().st_size / 1e6:.1f} MB)")
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    finally:
        store.close()

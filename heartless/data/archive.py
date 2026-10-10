"""Bulk history from Binance's public data archive (data.binance.vision).

The archive publishes USDⓈ-M futures 1m klines (monthly + daily zips), funding rates (monthly) and 5-minute
"metrics" (open interest, top-trader and global long/short ratios, taker buy/sell ratio). It costs no API
weight, so the bot uses it for bulk backfill and the research tools use it for months of real history.

Some networks block the data.binance.vision CDN hostname; the same bucket is reachable through the regional
S3 endpoint, so both are tried in order and the first one that answers is remembered.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from heartless.core.models import Candle
from heartless.util.timeutil import MS_DAY, MS_MINUTE

log = logging.getLogger(__name__)

ARCHIVE_BASES = (
    "https://data.binance.vision",
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision",
)
UM = "data/futures/um"


@dataclass(slots=True)
class MetricsRow:
    ts: int  # ms, start of the 5-minute bucket
    oi: float  # open interest (contracts)
    oi_value: float  # open interest in USDT
    top_ls_accounts: float  # top-trader long/short ratio by accounts
    top_ls_positions: float  # top-trader long/short ratio by positions
    ls_accounts: float  # global long/short account ratio
    taker_ls_vol: float  # taker buy/sell volume ratio


def _utc(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _month_start(dt: datetime) -> datetime:
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _next_month(dt: datetime) -> datetime:
    return (dt.replace(day=28) + timedelta(days=4)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def plan_kline_files(symbol: str, start_ms: int, end_ms: int, now_ms: int, interval: str = "1m",
                     daily: bool = False) -> list[tuple[str, int, int]]:
    """Archive paths covering [start_ms, end_ms]: monthly zips for complete past months, daily zips otherwise
    (`daily` forces daily zips: some monthly files lack days that their daily files have, e.g. SOLUSDT 2022-02).

    Returns (path, file_start_ms, file_end_ms). Daily files are published the next day, so days on or after
    the current UTC date are skipped (the caller fetches the tail via REST).
    """
    out: list[tuple[str, int, int]] = []
    start = _utc(start_ms).replace(hour=0, minute=0, second=0, microsecond=0)
    end = _utc(end_ms)
    today = _utc(now_ms).replace(hour=0, minute=0, second=0, microsecond=0)
    this_month = _month_start(today)
    cur = start
    while cur <= end and cur < today:
        ms = _month_start(cur)
        nm = _next_month(cur)
        if not daily and cur == ms and nm <= this_month and nm - timedelta(milliseconds=1) <= end + timedelta(days=31):
            # whole month available as one monthly file (it may extend past `end`; callers filter by time)
            out.append((f"{UM}/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{cur:%Y-%m}.zip", _ms(cur), _ms(nm) - 1))
            cur = nm
            continue
        out.append((f"{UM}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{cur:%Y-%m-%d}.zip", _ms(cur),
                    _ms(cur + timedelta(days=1)) - 1))
        cur = cur + timedelta(days=1)
    return out


def _rows_from_zip(blob: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        name = next((n for n in z.namelist() if n.endswith(".csv")), None)
        if name is None:
            return []
        text = z.read(name).decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if rows and rows[0] and not rows[0][0].strip().lstrip("-").replace(".", "").isdigit():
        rows = rows[1:]  # header line (newer files have one, older ones do not)
    return [r for r in rows if r]


def parse_klines(blob: bytes) -> list[Candle]:
    out = []
    for r in _rows_from_zip(blob):
        try:
            ot = int(r[0])
            if ot > 10**14:  # some 2025+ spot files use microseconds; futures use ms, but be safe
                ot //= 1000
            ct = int(r[6])
            if ct > 10**14:
                ct //= 1000
            out.append(Candle(open_time=ot, open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]),
                              volume=float(r[5]), close_time=ct, quote_volume=float(r[7]), trades=int(float(r[8])),
                              taker_buy_volume=float(r[9]), closed=True))
        except (ValueError, IndexError):
            continue
    return out


def parse_funding(blob: bytes) -> list[tuple[int, float, float]]:
    out = []
    for r in _rows_from_zip(blob):
        try:
            t = int(r[0])
            # calc_time is a few ms after the settlement instant; snap to the funding boundary (hour)
            t = t - (t % 3_600_000) if t % 3_600_000 < 60_000 else t
            out.append((t, float(r[2]), 0.0))
        except (ValueError, IndexError):
            continue
    return out


def parse_metrics(blob: bytes) -> list[MetricsRow]:
    out = []
    for r in _rows_from_zip(blob):
        try:
            dt = datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            ts = _ms(dt)
            ts -= ts % 300_000

            def f(i: int) -> float:
                try:
                    return float(r[i]) if r[i] not in ("", None) else 0.0
                except (ValueError, IndexError):
                    return 0.0

            out.append(MetricsRow(ts=ts, oi=f(2), oi_value=f(3), top_ls_accounts=f(4), top_ls_positions=f(5),
                                  ls_accounts=f(6), taker_ls_vol=f(7)))
        except (ValueError, IndexError):
            continue
    return out


class BinanceArchive:
    def __init__(self, bases: tuple[str, ...] = ARCHIVE_BASES, concurrency: int = 6, timeout: float = 60.0,
                 verify_checksum: bool = False):
        self.bases = list(bases)
        self._good: str | None = None
        self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._sem = asyncio.Semaphore(concurrency)
        self.verify_checksum = verify_checksum
        self.downloaded_bytes = 0

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "BinanceArchive":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def get(self, path: str) -> bytes | None:
        """Download one archive object. Returns None when it does not exist (404 / 403 for missing keys)."""
        bases = ([self._good] if self._good else []) + [b for b in self.bases if b != self._good]
        last_err: Exception | None = None
        async with self._sem:
            for base in bases:
                for attempt in range(3):
                    try:
                        r = await self._client.get(f"{base}/{path}")
                    except (httpx.TransportError, httpx.TimeoutException) as e:
                        last_err = e
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    if r.status_code == 200:
                        self._good = base
                        blob = r.content
                        self.downloaded_bytes += len(blob)
                        if self.verify_checksum and not await self._checksum_ok(base, path, blob):
                            log.warning("checksum mismatch for %s, retrying", path)
                            continue
                        return blob
                    if r.status_code in (403, 404):
                        self._good = base  # the endpoint works; the object is simply missing
                        return None
                    if r.status_code in (429, 500, 502, 503, 504):
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    break  # other status: try the next base
        if last_err is not None:
            raise last_err
        return None

    async def _checksum_ok(self, base: str, path: str, blob: bytes) -> bool:
        try:
            r = await self._client.get(f"{base}/{path}.CHECKSUM")
            if r.status_code != 200:
                return True
            expected = r.text.split()[0].strip().lower()
            return hashlib.sha256(blob).hexdigest() == expected
        except Exception:  # noqa: BLE001
            return True

    # --- datasets ------------------------------------------------------------------------------
    async def klines(self, symbol: str, start_ms: int, end_ms: int, now_ms: int, interval: str = "1m",
                     daily: bool = False) -> list[Candle]:
        files = plan_kline_files(symbol, start_ms, end_ms, now_ms, interval, daily)

        async def one(item: tuple[str, int, int]) -> list[Candle]:
            path, f_start, f_end = item
            blob = await self.get(path)
            if blob is None and "/monthly/" in path:
                # monthly file not published yet (first days of a month): fall back to its daily files
                days = []
                d = _utc(f_start)
                while _ms(d) <= f_end and _ms(d) <= end_ms:
                    days.append(f"{UM}/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{d:%Y-%m-%d}.zip")
                    d += timedelta(days=1)
                blobs = await asyncio.gather(*(self.get(p) for p in days))
                return [c for b in blobs if b for c in parse_klines(b)]
            return parse_klines(blob) if blob else []

        chunks = await asyncio.gather(*(one(f) for f in files))
        out = [c for ch in chunks for c in ch if start_ms <= c.open_time <= end_ms]
        out.sort(key=lambda c: c.open_time)
        dedup: list[Candle] = []
        for c in out:
            if dedup and dedup[-1].open_time == c.open_time:
                dedup[-1] = c
            else:
                dedup.append(c)
        return dedup

    async def funding(self, symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float, float]]:
        months = []
        cur = _month_start(_utc(start_ms))
        while _ms(cur) <= end_ms:
            months.append(f"{UM}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{cur:%Y-%m}.zip")
            cur = _next_month(cur)
        blobs = await asyncio.gather(*(self.get(p) for p in months))
        rows = [r for b in blobs if b for r in parse_funding(b)]
        return sorted({r[0]: r for r in rows if start_ms <= r[0] <= end_ms}.values())

    async def metrics(self, symbol: str, start_ms: int, end_ms: int, now_ms: int) -> list[MetricsRow]:
        days = []
        d = _utc(start_ms).replace(hour=0, minute=0, second=0, microsecond=0)
        today = _utc(now_ms).replace(hour=0, minute=0, second=0, microsecond=0)
        while _ms(d) <= end_ms and d < today:
            days.append(f"{UM}/daily/metrics/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip")
            d += timedelta(days=1)
        blobs = await asyncio.gather(*(self.get(p) for p in days))
        rows = [r for b in blobs if b for r in parse_metrics(b)]
        return sorted({r.ts: r for r in rows if start_ms <= r.ts <= end_ms}.values(), key=lambda r: r.ts)


FUNDING_STEP_MS = 8 * 3_600_000  # regular funding settlement interval
MIN_GAP_MIN = 5  # candle holes of at least this many minutes are reported (and re-requested from the archive)
MAX_SPAN_MS = 366 * MS_DAY  # at most about one year of 1m klines is held in memory per archive request
HOLES_KEY = "archive.holes.{}"  # kv: published candle ranges the archive was asked for and does not have
_MAX_HOLES = 500
DAILY_RETRY_MS = 31 * MS_DAY  # holes up to this long are retried from daily files before they count as archive holes


def year_spans(start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """[start, end] cut at UTC calendar-year boundaries, ascending (the unit of a resumable long fetch)."""
    out = []
    a = start_ms
    while a <= end_ms:
        nxt = _ms(datetime(_utc(a).year + 1, 1, 1, tzinfo=timezone.utc))
        out.append((a, min(end_ms, nxt - 1)))
        a = nxt
    return out


def _pieces(a: int, b: int, span: int) -> list[tuple[int, int]]:
    return [(x, min(b, x + span - 1)) for x in range(a, b + 1, span)]


def candle_missing(store, symbol: str, start_ms: int, end_ms: int, min_gap_min: int = MIN_GAP_MIN) -> list[tuple[int, int]]:
    """Open-time ranges of [start_ms, end_ms] without stored 1m candles: the part before the first / after the last
    stored bar and every interior hole of at least `min_gap_min` minutes, in ascending order."""
    first, last, _ = store.series_stats("candles", symbol, start_ms, end_ms)
    if first is None:
        return [(start_ms, end_ms)]
    out: list[tuple[int, int]] = []
    if start_ms < first - MS_MINUTE:
        out.append((start_ms, first - MS_MINUTE))
    out += [(a, b) for a, b, _ in store.series_gaps("candles", symbol, MS_MINUTE, min_gap_min, start_ms, end_ms)]
    if last + MS_MINUTE < end_ms:
        out.append((last + MS_MINUTE, end_ms))
    return out


def _covered(r: tuple[int, int], known: list) -> bool:
    return any(k[0] <= r[0] and r[1] <= k[1] for k in known)


async def sync_symbol(store, archive: BinanceArchive, symbol: str, start_ms: int, end_ms: int, now_ms: int,
                      metrics: bool = True, funding: bool = True, retry_holes: bool = False,
                      max_span_ms: int = MAX_SPAN_MS) -> dict:
    """Fill the store with archive data for one symbol over [start_ms, end_ms], fetching only what is missing.

    Missing candle ranges (head, tail and interior holes, e.g. a bot that was offline or an earlier interrupted run)
    are requested in ascending pieces of at most `max_span_ms`, each saved before the next is downloaded, so an
    interrupted run resumes where it stopped. Ranges the archive turns out not to have (exchange outages, days before
    a listing) are remembered in the kv table and not re-requested unless `retry_holes`; they stay in `gaps` /
    the coverage report. Funding and metrics are only synced from the first stored candle on (nothing exists before
    a listing)."""
    res: dict = {"symbol": symbol, "candles": 0, "funding": 0, "metrics": 0, "gaps": [], "known_holes": 0}
    key = HOLES_KEY.format(symbol)
    stored_holes = [tuple(h) for h in store.get(key, []) or []]
    known = [] if retry_holes else stored_holes
    todo: list[tuple[int, int]] = []
    for r in candle_missing(store, symbol, start_ms, end_ms):
        if _covered(r, known):
            res["known_holes"] += 1
        else:
            todo.append(r)
    monthly: list[tuple[int, int]] = []  # spans served by monthly zips in this pass
    for a, b in todo:
        for pa, pb in _pieces(a, b, max_span_ms):
            monthly += [(fa, fb) for path, fa, fb in plan_kline_files(symbol, pa, pb, now_ms) if "/monthly/" in path]
            rows = await archive.klines(symbol, pa, pb, now_ms)
            if rows:
                res["candles"] += store.save_candles(symbol, rows)
    if todo:
        published = _ms(_utc(now_ms).replace(hour=0, minute=0, second=0, microsecond=0)) - MS_DAY

        def requested_and_published(r: tuple[int, int]) -> bool:
            return r[1] < published and any(a <= r[0] and r[1] <= b for a, b in todo)

        remaining = candle_missing(store, symbol, start_ms, end_ms)
        retry = [r for r in remaining if requested_and_published(r) and r[1] - r[0] <= DAILY_RETRY_MS
                 and any(fa <= r[1] and r[0] <= fb for fa, fb in monthly)]
        for a, b in retry:  # a monthly file can lack days its daily files have: ask day by day before giving up
            rows = await archive.klines(symbol, a, b, now_ms, daily=True)
            if rows:
                res["candles"] += store.save_candles(symbol, rows)
        if retry:
            remaining = candle_missing(store, symbol, start_ms, end_ms)
        # what is still missing inside a requested, published range is a hole of the archive itself; remembered holes
        # of this window that have since been filled are forgotten
        holes = {r for r in remaining if requested_and_published(r)}
        keep = {h for h in stored_holes if not (start_ms <= h[0] and h[1] <= end_ms)
                or any(r[0] <= h[1] and h[0] <= r[1] for r in remaining)}
        merged = sorted(keep | holes)[-_MAX_HOLES:]
        if merged != sorted(stored_holes):
            store.set(key, [list(h) for h in merged])
    first, last, _ = store.series_stats("candles", symbol, start_ms, end_ms)
    res["first"], res["last"] = first, last
    if first is None:
        return res
    res["gaps"] = store.series_gaps("candles", symbol, MS_MINUTE, MIN_GAP_MIN, start_ms, end_ms)
    lo = max(start_ms, first)
    if funding:
        _, _, f_rows = store.series_stats("funding", symbol, lo, end_ms)
        if f_rows < 0.95 * (end_ms - lo) / (8 * 3_600_000) - 1:
            fr = await archive.funding(symbol, lo, end_ms)
            if fr:
                store.save_funding(symbol, fr)
                res["funding"] = len(fr)
    if metrics:
        m_first, m_last, _ = store.series_stats("metrics", symbol, lo, end_ms)
        if m_first is None:
            m_todo = [(lo, end_ms)]
        else:
            m_todo = [(lo, m_first - 300_000)] if m_first - lo >= MS_DAY else []
            m_todo += [(a, b) for a, b, _ in store.series_gaps("metrics", symbol, 300_000, MS_DAY // 300_000, lo, end_ms)]
            if end_ms - m_last >= MS_DAY:
                m_todo.append((m_last + 300_000, end_ms))
        for a, b in m_todo:
            for pa, pb in _pieces(a, b, max_span_ms):
                mr = await archive.metrics(symbol, pa, pb, now_ms)
                if mr:
                    res["metrics"] += store.save_metrics(symbol, mr)
    return res


# --- coverage report -----------------------------------------------------------------------------------------------
def _iso(ms: int | None) -> str | None:
    return None if ms is None else _utc(ms).strftime("%Y-%m-%d %H:%M")


def coverage(store, symbols: list[str], start_ms: int | None = None, end_ms: int | None = None,
             min_gap_min: int = MIN_GAP_MIN, metrics_gap_days: float = 1.0) -> dict:
    """Per symbol: candle range, rows vs expected minutes (overall and per year), holes of >= `min_gap_min` minutes,
    funding rows/range, metrics rows/range and metrics holes of >= `metrics_gap_days` days. Read-only."""
    out: dict = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "window": [_iso(start_ms), _iso(end_ms)], "min_gap_min": min_gap_min,
                 "metrics_gap_days": metrics_gap_days, "symbols": {}}
    for sym in symbols:
        first, last, rows = store.series_stats("candles", sym, start_ms, end_ms)
        known = list(store.get(HOLES_KEY.format(sym), []) or [])
        entry: dict = {"first": _iso(first), "last": _iso(last), "rows": rows, "expected": 0, "missing": 0,
                       "missing_pct": 0.0, "gaps": [], "n_gaps": 0, "gap_minutes": 0, "largest_gap": None, "by_year": {}}
        if first is not None:
            expected = (last - first) // MS_MINUTE + 1
            gaps = store.series_gaps("candles", sym, MS_MINUTE, min_gap_min, start_ms, end_ms)
            entry.update(expected=expected, missing=expected - rows,
                         missing_pct=round((expected - rows) / expected * 100, 4) if expected else 0.0,
                         gaps=[{"start": _iso(a), "end": _iso(b), "minutes": m, "archive_hole": _covered((a, b), known)}
                               for a, b, m in gaps],
                         n_gaps=len(gaps), gap_minutes=sum(g[2] for g in gaps))
            if gaps:
                big = max(gaps, key=lambda g: g[2])
                entry["largest_gap"] = {"start": _iso(big[0]), "end": _iso(big[1]), "minutes": big[2]}
            if start_ms is not None and first > start_ms:
                entry["head_missing_min"] = (first - start_ms) // MS_MINUTE
            if end_ms is not None and last < end_ms - MS_MINUTE:
                entry["tail_missing_min"] = (end_ms - last) // MS_MINUTE
            for ya, yb in year_spans(first, last):
                _, _, yr = store.series_stats("candles", sym, ya, yb)
                y_exp = (yb - ya) // MS_MINUTE + 1
                entry["by_year"][str(_utc(ya).year)] = {"rows": yr, "expected": y_exp,
                                                        "missing_pct": round((y_exp - yr) / y_exp * 100, 4)}
        f_first, f_last, f_rows = store.series_stats("funding", sym, start_ms, end_ms)
        m_first, m_last, m_rows = store.series_stats("metrics", sym, start_ms, end_ms)
        m_gaps = store.series_gaps("metrics", sym, 300_000, int(metrics_gap_days * MS_DAY // 300_000), start_ms, end_ms)
        # settlements are 8h apart (some symbols temporarily 4h): a step of more than 8h means a missing settlement,
        # which a backtest would silently charge at the previous rate
        f_gaps = store.series_gaps("funding", sym, FUNDING_STEP_MS, 1, start_ms, end_ms)
        entry["funding"] = {"rows": f_rows, "first": _iso(f_first), "last": _iso(f_last), "n_gaps": len(f_gaps),
                            "gaps": [{"start": _iso(a), "end": _iso(b), "settlements": m} for a, b, m in f_gaps]}
        entry["metrics"] = {"rows": m_rows, "first": _iso(m_first), "last": _iso(m_last), "n_gaps": len(m_gaps),
                            "gaps": [{"start": _iso(a), "end": _iso(b), "days": round(m * 300_000 / MS_DAY, 2)}
                                     for a, b, m in m_gaps]}
        out["symbols"][sym] = entry
    return out


def coverage_table(cov: dict) -> str:
    """Compact text table of a coverage() result."""
    lines = [f"{'symbol':10s} {'first':16s} {'last':16s} {'rows':>10s} {'miss%':>7s} {'gaps':>5s} {'gap_min':>8s} "
             f"{'largest gap':>28s} {'funding':>8s} {'f_gaps':>6s} {'metrics':>9s} {'m_gaps':>6s}"]
    for sym, e in cov["symbols"].items():
        big = e.get("largest_gap")
        big_s = f"{big['minutes']}m @ {big['start']}" if big else "-"
        lines.append(f"{sym:10s} {e['first'] or '-':16s} {e['last'] or '-':16s} {e['rows']:>10,d} {e['missing_pct']:>7.3f} "
                     f"{e['n_gaps']:>5d} {e['gap_minutes']:>8d} {big_s:>28s} {e['funding']['rows']:>8,d} "
                     f"{e['funding'].get('n_gaps', 0):>6d} {e['metrics']['rows']:>9,d} {e['metrics']['n_gaps']:>6d}")
    return "\n".join(lines)

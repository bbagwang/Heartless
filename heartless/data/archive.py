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


def plan_kline_files(symbol: str, start_ms: int, end_ms: int, now_ms: int, interval: str = "1m") -> list[tuple[str, int, int]]:
    """Archive paths covering [start_ms, end_ms]: monthly zips for complete past months, daily zips otherwise.

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
        if cur == ms and nm <= this_month and nm - timedelta(milliseconds=1) <= end + timedelta(days=31):
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
    async def klines(self, symbol: str, start_ms: int, end_ms: int, now_ms: int, interval: str = "1m") -> list[Candle]:
        files = plan_kline_files(symbol, start_ms, end_ms, now_ms, interval)

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


async def sync_symbol(store, archive: BinanceArchive, symbol: str, start_ms: int, end_ms: int, now_ms: int,
                      metrics: bool = True, funding: bool = True) -> dict:
    """Fill the store with archive data for one symbol, skipping ranges that are already stored."""
    res = {"symbol": symbol, "candles": 0, "funding": 0, "metrics": 0}
    lo, hi, n = store.candle_range(symbol)
    ranges: list[tuple[int, int]] = []
    if lo is None:
        ranges.append((start_ms, end_ms))
    else:
        if start_ms < lo - 60_000:
            ranges.append((start_ms, lo - 60_000))
        if hi + 60_000 < end_ms:
            ranges.append((hi + 60_000, end_ms))
    for a, b in ranges:
        rows = await archive.klines(symbol, a, b, now_ms)
        if rows:
            res["candles"] += store.save_candles(symbol, rows)
    if funding:
        f_lo, f_hi = store.funding_range(symbol)
        if f_lo is None or f_lo > start_ms + 8 * 3_600_000 or (f_hi or 0) < end_ms - 40 * 86_400_000:
            fr = await archive.funding(symbol, start_ms, end_ms)
            if fr:
                store.save_funding(symbol, fr)
                res["funding"] = len(fr)
    if metrics:
        m_lo, m_hi = store.metrics_range(symbol)
        m_start = start_ms if m_lo is None or m_lo > start_ms + 86_400_000 else (m_hi or start_ms) + 300_000
        if m_start < end_ms:
            mr = await archive.metrics(symbol, m_start, end_ms, now_ms)
            if mr:
                store.save_metrics(symbol, mr)
                res["metrics"] = len(mr)
    return res

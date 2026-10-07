from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

MS_MINUTE = 60_000
MS_HOUR = 3_600_000
MS_DAY = 86_400_000

TF_MS = {"1m": MS_MINUTE, "3m": 3 * MS_MINUTE, "5m": 5 * MS_MINUTE, "15m": 15 * MS_MINUTE,
         "30m": 30 * MS_MINUTE, "1h": MS_HOUR, "4h": 4 * MS_HOUR, "1d": MS_DAY}


def now_ms() -> int:
    return int(time.time() * 1000)


def floor_ms(ts: int, step: int) -> int:
    return ts - (ts % step)


def fmt_ts(ts_ms: int | float | None, tz: str = "UTC", fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if ts_ms is None:
        return "-"
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    try:
        dt = dt.astimezone(ZoneInfo(tz))
    except Exception:
        pass
    return dt.strftime(fmt)


def utc_day_start(ts_ms: int | None = None) -> int:
    ts_ms = now_ms() if ts_ms is None else ts_ms
    return floor_ms(ts_ms, MS_DAY)


def local_day_start(ts_ms: int, tz: str) -> int:
    """Start of the local calendar day containing ts_ms, as epoch ms."""
    try:
        z = ZoneInfo(tz)
    except Exception:
        z = timezone.utc
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=z)
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


def next_local_time(hour: int, minute: int, tz: str, after_ms: int | None = None) -> int:
    """Epoch ms of the next occurrence of hour:minute in tz strictly after after_ms."""
    after_ms = now_ms() if after_ms is None else after_ms
    try:
        z = ZoneInfo(tz)
    except Exception:
        z = timezone.utc
    dt = datetime.fromtimestamp(after_ms / 1000, tz=z)
    cand = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cand <= dt:
        cand = cand + timedelta(days=1)
    return int(cand.timestamp() * 1000)


def human_duration(ms: float) -> str:
    s = int(ms // 1000)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"

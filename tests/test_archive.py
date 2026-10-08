"""Offline tests for the Binance public-archive downloader (no network)."""
import asyncio
import io
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from heartless.core.store import Store
from heartless.data.archive import (BinanceArchive, parse_funding, parse_klines, parse_metrics, plan_kline_files,
                                    sync_symbol)

DAY = 86_400_000


def _ms(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def _zip(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text)
    return buf.getvalue()


def _kline_csv(start_ms: int, n: int, header: bool = True) -> str:
    lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"] if header else []
    for i in range(n):
        t = start_ms + i * 60_000
        lines.append(f"{t},100.0,101.0,99.0,100.5,10.0,{t + 59_999},1005.0,7,6.0,603.0,0")
    return "\n".join(lines)


def test_plan_uses_monthly_for_complete_past_months_and_daily_otherwise():
    now = _ms(2026, 10, 7, 12)
    files = plan_kline_files("BTCUSDT", _ms(2026, 7, 25), now, now)
    names = [f[0].rsplit("/", 1)[1] for f in files]
    assert names[0] == "BTCUSDT-1m-2026-07-25.zip"  # partial July -> daily files
    assert "BTCUSDT-1m-2026-08.zip" in names and "BTCUSDT-1m-2026-09.zip" in names
    assert names[-1] == "BTCUSDT-1m-2026-10-06.zip"  # today's file is not published yet
    assert all("2026-10-07" not in n for n in names)
    assert sum(1 for n in names if n.startswith("BTCUSDT-1m-2026-07-")) == 7


def test_parse_klines_with_and_without_header():
    for header in (True, False):
        rows = parse_klines(_zip("x.csv", _kline_csv(_ms(2026, 9, 1), 3, header)))
        assert len(rows) == 3 and rows[0].open_time == _ms(2026, 9, 1) and rows[0].taker_buy_volume == 6.0
        assert rows[1].close_time == rows[1].open_time + 59_999 and rows[0].closed


def test_parse_funding_snaps_calc_time_to_the_hour():
    blob = _zip("f.csv", "calc_time,funding_interval_hours,last_funding_rate\n1788220800005,8,0.00008482\n1788249600000,8,-0.0001\n")
    rows = parse_funding(blob)
    assert rows[0][0] == 1788220800000 and abs(rows[0][1] - 0.00008482) < 1e-12 and rows[1][1] == -0.0001


def test_parse_metrics_columns():
    blob = _zip("m.csv", "create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
                         "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
                         "2026-10-06 01:05:00,BTCUSDT,94432.19,8119648067.72,1.18,1.76,1.08,0.91\n"
                         "2026-10-06 01:10:00,BTCUSDT,94500.00,8120000000.00,1.17,1.75,,0.76\n")
    rows = parse_metrics(blob)
    assert len(rows) == 2 and rows[0].oi == 94432.19 and rows[0].top_ls_positions == 1.76 and rows[0].taker_ls_vol == 0.91
    assert rows[0].ts == _ms(2026, 10, 6, 1, 5) and rows[1].ls_accounts == 0.0  # empty cell -> 0


class FakeArchive(BinanceArchive):
    """Serves synthetic monthly/daily files; the September monthly file is 'not published' to test the fallback."""

    def __init__(self):
        super().__init__()
        self.requested: list[str] = []

    async def get(self, path: str):
        self.requested.append(path)
        name = path.rsplit("/", 1)[1]
        if "/klines/" in path:
            stem = name[len("BTCUSDT-1m-"):-4]
            if len(stem) == 7:  # monthly YYYY-MM
                if stem == "2026-09":
                    return None
                y, m = map(int, stem.split("-"))
                start = _ms(y, m, 1)
                end = _ms(y + (m == 12), m % 12 + 1, 1)
                return _zip("k.csv", _kline_csv(start, (end - start) // 60_000))
            y, m, d = map(int, stem.split("-"))
            return _zip("k.csv", _kline_csv(_ms(y, m, d), 1440))
        if "/fundingRate/" in path:
            return _zip("f.csv", "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(
                f"{_ms(2026, 9, 1) + i * 8 * 3_600_000},8,0.0001" for i in range(90)))
        if "/metrics/" in path:
            day = name[len("BTCUSDT-metrics-"):-4]
            return _zip("m.csv", "create_time,symbol,a,b,c,d,e,f\n" + "\n".join(
                f"{day} {h:02d}:00:00,BTCUSDT,1,2,3,4,5,6" for h in range(24)))
        return None


def test_sync_symbol_fills_store_and_falls_back_to_daily_when_monthly_missing():
    async def run():
        st = Store(Path(tempfile.mkdtemp()) / "a.db")
        arch = FakeArchive()
        now = _ms(2026, 10, 3, 12)
        res = await sync_symbol(st, arch, "BTCUSDT", _ms(2026, 8, 30), now, now)
        lo, hi, n = st.candle_range("BTCUSDT")
        assert lo == _ms(2026, 8, 30) and hi == _ms(2026, 10, 2, 23, 59)  # through the last published daily file
        assert n == (hi - lo) // 60_000 + 1 and res["candles"] == n  # no gaps despite the missing monthly file
        assert any(p.endswith("BTCUSDT-1m-2026-09-15.zip") for p in arch.requested)  # daily fallback used
        assert res["funding"] > 0 and res["metrics"] > 0
        # a second sync downloads nothing new
        arch.requested.clear()
        res2 = await sync_symbol(st, arch, "BTCUSDT", _ms(2026, 8, 30), now, now, metrics=False, funding=False)
        assert res2["candles"] == 0
        await arch.close()

    asyncio.run(run())

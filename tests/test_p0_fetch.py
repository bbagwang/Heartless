"""P0 long-history fetch: gap-aware, resumable archive sync, chunked `heartless fetch --start/--end`, coverage report."""
import asyncio
import io
import json
import zipfile
from datetime import datetime, timezone

import pytest

from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data.archive import BinanceArchive, HOLES_KEY, candle_missing, coverage, coverage_table, sync_symbol, year_spans

MIN = 60_000
DAY = 86_400_000
NOW = int(datetime(2026, 10, 9, tzinfo=timezone.utc).timestamp() * 1000)


def _ms(y, m, d, hh=0, mm=0):
    return int(datetime(y, m, d, hh, mm, tzinfo=timezone.utc).timestamp() * 1000)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


class FakeArchive(BinanceArchive):
    """Serves synthetic monthly/daily kline, funding and metrics files without network access.

    `holes`: open-time ranges the archive does not have; `listed`: first minute the symbol exists;
    `fail_once`: path fragments that raise a transport error the first time they are requested."""

    def __init__(self, holes=(), listed=None, fail_once=(), monthly_lacks=()):
        super().__init__()
        self.holes = list(holes)
        self.listed = listed
        self.fail_once = set(fail_once)
        self.monthly_lacks = list(monthly_lacks)  # ranges missing from monthly zips only (their daily zips have them)
        self.requested: list[str] = []

    def _has(self, t: int) -> bool:
        if self.listed is not None and t < self.listed:
            return False
        return not any(a <= t <= b for a, b in self.holes)

    def _klines(self, start: int, end: int, monthly: bool = False) -> bytes | None:
        rows = [f"{t},100,101,99,100.5,10,{t + 59_999},1000,5,4,400,0" for t in range(start, end, MIN)
                if self._has(t) and not (monthly and any(a <= t <= b for a, b in self.monthly_lacks))]
        return _zip("open_time,open,high,low,close,volume,close_time,qv,n,tb,tbq,ignore\n" + "\n".join(rows)) if rows else None

    async def get(self, path: str):
        self.requested.append(path)
        for frag in list(self.fail_once):
            if frag in path:
                self.fail_once.discard(frag)
                raise RuntimeError(f"connection reset ({frag})")
        name = path.rsplit("/", 1)[1]
        if "/klines/" in path:
            stem = name.split("-1m-")[1][:-4]
            if len(stem) == 7:
                y, m = map(int, stem.split("-"))
                return self._klines(_ms(y, m, 1), _ms(y + (m == 12), m % 12 + 1, 1), monthly=True)
            y, m, d = map(int, stem.split("-"))
            return self._klines(_ms(y, m, d), _ms(y, m, d) + DAY)
        if "/fundingRate/" in path:
            y, m = map(int, name.split("-fundingRate-")[1][:-4].split("-"))
            a, b = _ms(y, m, 1), _ms(y + (m == 12), m % 12 + 1, 1)
            rows = [f"{t + 3},8,0.0001" for t in range(a, b, 8 * 3_600_000) if self._has(t)]
            return _zip("calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows)) if rows else None
        if "/metrics/" in path:
            day = name.split("-metrics-")[1][:-4]
            d0 = _ms(*map(int, day.split("-")))
            rows = [f"{datetime.fromtimestamp(t / 1000, timezone.utc):%Y-%m-%d %H:%M:%S},X,1,2,3,4,5,6"
                    for t in range(d0, d0 + DAY, 300_000) if self._has(t)]
            return _zip("create_time,symbol,a,b,c,d,e,f\n" + "\n".join(rows)) if rows else None
        return None

    def kline_requests(self) -> list[str]:
        return [p for p in self.requested if "/klines/" in p]


def _candles(start: int, end: int) -> list[Candle]:
    return [Candle(t, 100, 101, 99, 100.5, 10, 1000, 5, 4, t + 59_999) for t in range(start, end + 1, MIN)]


def test_year_spans_cut_at_calendar_years():
    spans = year_spans(_ms(2022, 6, 1), _ms(2024, 2, 1) - 1)
    assert spans == [(_ms(2022, 6, 1), _ms(2023, 1, 1) - 1), (_ms(2023, 1, 1), _ms(2024, 1, 1) - 1),
                     (_ms(2024, 1, 1), _ms(2024, 2, 1) - 1)]


def test_interior_ranges_are_fetched_when_newer_history_is_already_stored(tmp_path):
    """The old [lo, hi] skip ignored everything between the oldest and newest stored bar: an ascending chunked fetch
    into a DB that already holds newer data would silently skip whole chunks."""
    async def run():
        st = Store(tmp_path / "a.db")
        st.save_candles("BTCUSDT", _candles(_ms(2024, 1, 5), _ms(2024, 1, 10, 23, 59)))  # e.g. bot-stored recent data
        arch = FakeArchive()
        r1 = await sync_symbol(st, arch, "BTCUSDT", _ms(2023, 12, 28), _ms(2024, 1, 1) - 1, NOW, metrics=False, funding=False)
        assert r1["candles"] == 4 * 1440 and r1["gaps"] == []
        # bounded: only files of the requested chunk, nothing from the stored January range
        assert all("2023-12-" in p for p in arch.kline_requests())
        arch.requested.clear()
        r2 = await sync_symbol(st, arch, "BTCUSDT", _ms(2024, 1, 1), _ms(2024, 2, 1) - 1, NOW, metrics=False, funding=False)
        assert r2["candles"] == (31 - 6) * 1440  # Jan 1-4 and Jan 11-31; the stored Jan 5-10 is not re-saved
        assert candle_missing(st, "BTCUSDT", _ms(2023, 12, 28), _ms(2024, 2, 1) - 1) == []
        _, _, n = st.candle_range("BTCUSDT")
        assert n == (31 + 4) * 1440
        await arch.close()

    asyncio.run(run())


def test_archive_holes_are_reported_remembered_and_retried_on_request(tmp_path):
    hole = (_ms(2024, 1, 3, 10, 0), _ms(2024, 1, 3, 10, 29))

    async def run():
        st = Store(tmp_path / "a.db")
        arch = FakeArchive(holes=[hole])
        a, b = _ms(2024, 1, 1), _ms(2024, 1, 8) - 1
        r1 = await sync_symbol(st, arch, "BTCUSDT", a, b, NOW, metrics=False, funding=False)
        assert r1["gaps"] == [(hole[0], hole[1], 30)]  # reported, not silently ignored
        assert st.get(HOLES_KEY.format("BTCUSDT")) == [[hole[0], hole[1]]]
        arch.requested.clear()
        r2 = await sync_symbol(st, arch, "BTCUSDT", a, b, NOW, metrics=False, funding=False)
        assert arch.kline_requests() == [] and r2["known_holes"] == 1 and r2["gaps"] == r1["gaps"]
        r3 = await sync_symbol(st, arch, "BTCUSDT", a, b, NOW, metrics=False, funding=False, retry_holes=True)
        # one daily request: the hole was served by a daily zip already, so there is no second (daily) retry
        assert len([p for p in arch.kline_requests() if p.endswith("2024-01-03.zip")]) == 1 and r3["candles"] == 0
        await arch.close()

    asyncio.run(run())


def test_days_missing_from_a_monthly_zip_are_taken_from_daily_zips_and_stale_holes_forgotten(tmp_path):
    """Seen on the real archive: SOLUSDT/XRPUSDT/... 2022-02 monthly zips stop at 02-25 while the daily zips exist."""
    lacks = (_ms(2024, 1, 1), _ms(2024, 1, 2, 23, 59))

    async def run():
        st = Store(tmp_path / "a.db")
        a, b = _ms(2024, 1, 1), _ms(2024, 2, 1) - 1
        arch = FakeArchive(monthly_lacks=[lacks])
        res = await sync_symbol(st, arch, "BTCUSDT", a, b, NOW, metrics=False, funding=False)
        assert res["candles"] == 31 * 1440 and res["gaps"] == [] and st.get(HOLES_KEY.format("BTCUSDT")) is None
        assert any(p.endswith("BTCUSDT-1m-2024-01-01.zip") for p in arch.kline_requests())
        # a hole the archive really had is remembered; once published and retried it is filled and forgotten
        hole = (_ms(2024, 2, 3), _ms(2024, 2, 3, 5, 59))
        gone = FakeArchive(holes=[hole])
        await sync_symbol(st, gone, "BTCUSDT", _ms(2024, 2, 1), _ms(2024, 2, 6) - 1, NOW, metrics=False, funding=False)
        assert st.get(HOLES_KEY.format("BTCUSDT")) == [list(hole)]
        res = await sync_symbol(st, FakeArchive(), "BTCUSDT", _ms(2024, 2, 1), _ms(2024, 2, 6) - 1, NOW, metrics=False,
                                funding=False, retry_holes=True)
        assert res["candles"] == 360 and res["gaps"] == [] and st.get(HOLES_KEY.format("BTCUSDT")) == []
        await arch.close()
        await gone.close()

    asyncio.run(run())


def test_interrupted_long_fetch_resumes_where_it_stopped(tmp_path):
    async def run():
        st = Store(tmp_path / "a.db")
        arch = FakeArchive(fail_once=["1m-2023-11.zip"])
        a, b = _ms(2023, 10, 1), _ms(2023, 12, 1) - 1
        with pytest.raises(RuntimeError):
            await sync_symbol(st, arch, "BTCUSDT", a, b, NOW, metrics=False, funding=False, max_span_ms=31 * DAY)
        lo, hi, n = st.candle_range("BTCUSDT")
        assert (lo, hi, n) == (a, _ms(2023, 11, 1) - MIN, 31 * 1440)  # October was saved before November failed
        arch.requested.clear()
        res = await sync_symbol(st, arch, "BTCUSDT", a, b, NOW, metrics=False, funding=False, max_span_ms=31 * DAY)
        assert res["candles"] == 30 * 1440 and not any("2023-10" in p for p in arch.kline_requests())
        await arch.close()

    asyncio.run(run())


def test_listing_inside_the_window_limits_funding_and_metrics_and_is_remembered(tmp_path):
    listed = _ms(2024, 1, 3)

    async def run():
        st = Store(tmp_path / "a.db")
        arch = FakeArchive(listed=listed)
        a, b = _ms(2024, 1, 1), _ms(2024, 1, 6) - 1
        res = await sync_symbol(st, arch, "SUIUSDT", a, b, NOW)
        assert res["first"] == listed and res["gaps"] == []
        assert not any("metrics-2024-01-0" + d in p for p in arch.requested for d in ("1", "2"))  # nothing before listing
        assert res["metrics"] == 3 * 288 and res["funding"] == 9
        assert st.series_stats("funding", "SUIUSDT")[0] >= listed
        arch.requested.clear()
        res2 = await sync_symbol(st, arch, "SUIUSDT", a, b, NOW)
        assert arch.requested == [] and res2["known_holes"] == 1  # the pre-listing range is a known hole now
        await arch.close()

    asyncio.run(run())


def test_coverage_reports_rows_gaps_funding_and_metrics(tmp_path):
    st = Store(tmp_path / "c.db")
    a = _ms(2023, 12, 31, 12)
    rows = [c for c in _candles(a, a + 2 * DAY - MIN)
            if not (a + 100 * MIN <= c.open_time < a + 103 * MIN or a + 1000 * MIN <= c.open_time < a + 1010 * MIN)]
    st.save_candles("BTCUSDT", rows)
    st.save_funding("BTCUSDT", [(a + i * 8 * 3_600_000, 0.0001, 0.0) for i in range(6)])
    from heartless.data.archive import MetricsRow

    ms = [MetricsRow(t, 1, 2, 3, 4, 5, 6) for t in range(a, a + 2 * DAY, 300_000) if not a + 6 * 3_600_000 <= t < a + 33 * 3_600_000]
    st.save_metrics("BTCUSDT", ms)
    cov = coverage(st, ["BTCUSDT"])
    e = cov["symbols"]["BTCUSDT"]
    assert e["expected"] == 2 * 1440 and e["missing"] == 13 and e["rows"] == 2 * 1440 - 13
    assert e["n_gaps"] == 1 and e["gaps"][0]["minutes"] == 10 and e["gap_minutes"] == 10  # the 3-minute hole is below 5m
    assert e["largest_gap"]["start"] == "2024-01-01 04:40"
    assert set(e["by_year"]) == {"2023", "2024"} and e["by_year"]["2023"]["rows"] == 717
    assert e["funding"]["rows"] == 6 and e["metrics"]["n_gaps"] == 1 and e["metrics"]["gaps"][0]["days"] >= 1.0
    table = coverage_table(cov)
    assert "BTCUSDT" in table and "10m @ 2024-01-01 04:40" in table
    win = coverage(st, ["BTCUSDT"], a - 60 * MIN, a + 3 * DAY)["symbols"]["BTCUSDT"]
    assert win["head_missing_min"] == 60 and win["tail_missing_min"] == 1440 + 1
    assert coverage(st, ["NOPEUSDT"])["symbols"]["NOPEUSDT"]["rows"] == 0


def test_fetch_cli_long_range_runs_ascending_year_chunks_and_coverage_cli(tmp_path, monkeypatch, capsys):
    import heartless.cli as cli
    import heartless.data.archive as archive_mod

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "now_ms", lambda: NOW)
    fakes = []

    def make():
        fakes.append(FakeArchive(holes=[(_ms(2024, 1, 2, 5, 0), _ms(2024, 1, 2, 5, 9))]))
        return fakes[-1]

    monkeypatch.setattr(archive_mod, "BinanceArchive", make)
    cli.main(["fetch", "--symbols", "BTCUSDT", "--start", "2023-12-30", "--end", "2024-01-04", "--no-metrics"])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("BTCUSDT ")]
    assert lines[0].startswith("BTCUSDT 2023: +2880 candles") and lines[1].startswith("BTCUSDT 2024: +4310 candles")
    assert "1 gap(s) >= 5m (10 min, largest 10m at 2024-01-02 05:00)" in lines[1]
    st = Store(tmp_path / "heartless.db")
    lo, hi, n = st.candle_range("BTCUSDT")
    assert lo == _ms(2023, 12, 30) and hi == _ms(2024, 1, 3, 23, 59)  # --end is exclusive
    st.close()
    # rerun: everything is stored, the hole is known -> no kline downloads at all
    cli.main(["fetch", "--symbols", "BTCUSDT", "--start", "2023-12-30", "--end", "2024-01-04", "--no-metrics"])
    assert fakes[-1].kline_requests() == []
    capsys.readouterr()
    cli.main(["coverage"])
    out = capsys.readouterr().out
    cov = json.loads((tmp_path / "coverage.json").read_text())
    assert cov["symbols"]["BTCUSDT"]["n_gaps"] == 1 and cov["symbols"]["BTCUSDT"]["gaps"][0]["archive_hole"] is True
    assert "BTCUSDT" in out and "wrote" in out


def test_coverage_reports_missing_funding_settlements(tmp_path):
    """ROADMAP P0 asks for gaps of every series: a missing settlement would be charged at the previous rate."""
    st = Store(tmp_path / "f.db")
    a = _ms(2024, 1, 1)
    st.save_candles("BTCUSDT", _candles(a, a + 3 * DAY - MIN))
    h8 = 8 * 3_600_000
    st.save_funding("BTCUSDT", [(a + i * h8, 0.0001, 0.0) for i in range(9) if i not in (4, 5)])
    st.save_funding("SOLUSDT", [(a + i * h8 // 2, 0.0001, 0.0) for i in range(12)])  # 4h settlements: no gap
    cov = coverage(st, ["BTCUSDT", "SOLUSDT"])
    f = cov["symbols"]["BTCUSDT"]["funding"]
    assert f["rows"] == 7 and f["n_gaps"] == 1
    assert f["gaps"][0] == {"start": "2024-01-02 08:00", "end": "2024-01-02 16:00", "settlements": 2}
    assert cov["symbols"]["SOLUSDT"]["funding"]["n_gaps"] == 0
    assert " 1 " in coverage_table(cov).splitlines()[1]

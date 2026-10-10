"""P5 findings fixed in the orchestrator:

* the in-memory 1m history per symbol grew without bound (every bar appended, nothing dropped): memory and the cost of
  every 5m/15m/1h resample rose for the whole process lifetime. It is now trimmed daily to the span the live frames can
  reach, and the frames are bit-identical to the untrimmed ones;
* a symbol whose 24h volume hovers around the entry threshold flapped in and out of the universe at hourly refreshes;
  members now stay until they fall below UNIVERSE_KEEP_FRACTION of the threshold."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
from synth import synth_candles

from heartless.app import LIVE_HISTORY_DAYS, UNIVERSE_KEEP_FRACTION, Heartless
from heartless.config import Settings
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.util.timeutil import MS_DAY

START = 1_772_323_200_000  # 2026-03-01 00:00 UTC
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT"]


def _app(**kw) -> Heartless:
    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ALWAYS_INCLUDE="BTCUSDT", **kw)
    s.data_dir.mkdir(exist_ok=True)
    return Heartless(s)


def test_trimmed_history_is_bounded_and_yields_identical_frames():
    app = _app()
    days = LIVE_HISTORY_DAYS + 9
    full = synth_candles(days * 1440, seed=3, start_ms=START - days * MS_DAY)
    trimmed = full.slice(0, full.n)
    app._trim_history(trimmed, START)
    assert trimmed.n == LIVE_HISTORY_DAYS * 1440 and int(trimmed.open_time[0]) == START - LIVE_HISTORY_DAYS * MS_DAY
    a, b = MarketView("X"), MarketView("X")
    Heartless._rebuild_view(a, full, ["1m", "5m", "15m", "1h"])
    Heartless._rebuild_view(b, trimmed, ["1m", "5m", "15m", "1h"])
    for tf in ("1m", "5m", "15m", "1h"):
        fa, fb = a.frames[tf], b.frames[tf]
        assert fa.n == fb.n and np.array_equal(fa.close_time, fb.close_time), tf
        for name, arr in fa.f.items():
            assert np.array_equal(arr, fb.f[name], equal_nan=True), (tf, name)


def test_daily_trim_runs_in_the_bar_path_and_keeps_memory_flat():
    app = _app(HISTORY_DAYS=3)
    days = LIVE_HISTORY_DAYS + 2
    ca = synth_candles((days + 3) * 1440, seed=4, start_ms=START - days * MS_DAY)
    live = ca.slice(0, days * 1440)
    live._grow()  # room for the replayed days, so a capacity change would show
    cap = live._cap
    app.candles["BTCUSDT"] = live
    sizes = []
    for i in range(days * 1440, ca.n):  # three more days arrive bar by bar (no view: the trim happens before it)
        asyncio.run(app._process_bar("BTCUSDT", ca.candle_at(i)))
        if int(ca.open_time[i]) % MS_DAY == 0:
            sizes.append(live.n)
    assert len(sizes) == 3 and all(n == LIVE_HISTORY_DAYS * 1440 + 1 for n in sizes)  # span + the day's first bar
    assert live.n <= (LIVE_HISTORY_DAYS + 1) * 1440 and live._cap == cap  # no regrowth


class _Rest:
    def __init__(self, vols: dict[str, float]):
        self.vols = vols

    async def ticker_24h(self):
        return [{"symbol": s, "quoteVolume": str(v)} for s, v in self.vols.items()]


def test_universe_members_stay_until_well_below_the_entry_threshold():
    app = _app(UNIVERSE_SIZE=4, UNIVERSE_MIN_QUOTE_VOLUME=80e6)
    from heartless.core.models import SymbolInfo

    app.symbols = {s: SymbolInfo(s, s[:-4], "USDT", 0.01, 0.001, 0.001, 5.0, 2, 3) for s in SYMS}

    async def nothing(*a, **k):
        return set()

    app._backfill = nothing
    app._build_views = lambda syms: None
    app.rest = _Rest({"BTCUSDT": 9e9, "ETHUSDT": 5e9, "SOLUSDT": 1e9, "AVAXUSDT": 81e6, "LINKUSDT": 50e6})
    asyncio.run(app.refresh_universe(initial=True))
    assert app.universe == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
    app.rest.vols["AVAXUSDT"] = 79e6  # just under the entry threshold: stays (no flapping)
    asyncio.run(app.refresh_universe())
    assert "AVAXUSDT" in app.universe
    app.rest.vols["AVAXUSDT"] = 80e6 * UNIVERSE_KEEP_FRACTION * 0.99  # clearly below: leaves
    asyncio.run(app.refresh_universe())
    assert "AVAXUSDT" not in app.universe
    app.rest.vols["AVAXUSDT"] = 79e6  # a non-member needs the full threshold to (re)enter
    asyncio.run(app.refresh_universe())
    assert "AVAXUSDT" not in app.universe and "LINKUSDT" not in app.universe

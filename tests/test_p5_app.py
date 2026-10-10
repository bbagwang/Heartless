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


class _FundingRest:
    def __init__(self, rows: list[tuple[int, float]]):
        self.rows = rows
        self.calls: list[tuple[str, int, int]] = []

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        self.calls.append((symbol, start, end))
        return [{"symbol": symbol, "fundingTime": t, "fundingRate": str(r), "markPrice": "100"}
                for t, r in self.rows if start <= t <= end]


def test_funding_settlements_keep_flowing_into_the_store_after_startup():
    from heartless.util import timeutil

    app = _app()
    app.universe = ["BTCUSDT", "ETHUSDT"]
    eight = 8 * 3_600_000
    app.store.save_funding("BTCUSDT", [(START - eight, 0.0001, 100.0), (START, 0.0002, 100.0)])
    app.rest = _FundingRest([(START + eight, 0.0003), (START + 2 * eight, -0.0001)])
    prev = timeutil.set_clock(lambda: START + 2 * eight + 60_000)
    try:
        asyncio.run(app._refresh_funding())
    finally:
        timeutil.set_clock(prev)
    assert app.store.load_funding("BTCUSDT", START, START + 3 * eight) == [
        (START, 0.0002), (START + eight, 0.0003), (START + 2 * eight, -0.0001)]
    assert app.rest.calls == [("BTCUSDT", START + 1, START + 2 * eight + 60_000)]  # ETHUSDT has no backfill yet
    # nothing due: no request
    prev = timeutil.set_clock(lambda: START + 2 * eight + 120_000)
    try:
        asyncio.run(app._refresh_funding())
    finally:
        timeutil.set_clock(prev)
    assert len(app.rest.calls) == 1


class _MetricsRest:
    def __init__(self, stamps: list[int]):
        self.stamps = stamps

    async def open_interest_hist(self, symbol, period="5m", limit=30):
        return [{"symbol": symbol, "sumOpenInterest": "1000", "sumOpenInterestValue": "5000000", "timestamp": t}
                for t in self.stamps]

    async def top_long_short_ratio(self, symbol, period="5m", limit=30):
        return [{"longShortRatio": "1.5", "timestamp": t} for t in self.stamps]

    top_long_short_account_ratio = top_long_short_ratio
    global_long_short_account_ratio = top_long_short_ratio

    async def taker_long_short_ratio(self, symbol, period="5m", limit=30):
        return [{"buySellRatio": "0.9", "timestamp": t} for t in self.stamps[:-1]]  # newest bucket not yet published


def test_polled_metrics_are_persisted_for_the_research_backtests():
    app = _app()
    app.universe = ["BTCUSDT"]
    bucket = 300_000
    app.rest = _MetricsRest([START + bucket, START + 2 * bucket])  # REST stamps the bucket end
    asyncio.run(app._poll_metrics())
    rows = app.store.load_metrics("BTCUSDT")
    assert [r["ts"] for r in rows] == [START, START + bucket]  # stored by bucket start, as the archive does
    assert rows[0]["oi"] == 1000 and rows[0]["top_ls_positions"] == 1.5 and rows[0]["taker_ls_vol"] == 0.9
    assert rows[1]["taker_ls_vol"] is None
    app.rest.stamps = [START + 2 * bucket]  # the next poll re-reads the bucket, now complete
    app.rest.taker_long_short_ratio = lambda *a, **k: _ret([{"buySellRatio": "1.1", "timestamp": START + 2 * bucket}])
    asyncio.run(app._poll_metrics())
    assert app.store.load_metrics("BTCUSDT")[-1]["taker_ls_vol"] == 1.1


async def _ret(v):
    return v


class _LaggingMetricsRest(_MetricsRest):
    """Every series answers its newest `limit` points, but the taker series lags one bucket behind the others (or
    answers nothing at all), so a poll's oldest/newest bucket comes from only some of the endpoints."""

    def __init__(self, stamps: list[int], taker: list[int]):
        super().__init__(stamps)
        self.taker = taker

    async def taker_long_short_ratio(self, symbol, period="5m", limit=30):
        return [{"buySellRatio": "0.9", "timestamp": t} for t in self.taker]


def test_a_partial_poll_never_erases_stored_metrics():
    """Each poll re-reads the newest buckets. A bucket that only some series answered in this poll (a lagging or empty
    series) was written as a whole row with the other columns NULL, erasing the values an earlier poll had stored, and
    with a series that lags by one bucket that happened to every bucket as it left the poll window."""
    app = _app()
    app.universe = ["BTCUSDT"]
    b = 300_000
    stamps = [START + i * b for i in range(1, 7)]
    app.rest = _LaggingMetricsRest(stamps, [t - b for t in stamps])
    asyncio.run(app._poll_metrics())
    full = {r["ts"]: r for r in app.store.load_metrics("BTCUSDT")}
    assert full[START]["oi"] == 1000 and full[START]["taker_ls_vol"] == 0.9  # stamp START + b: every series
    app.rest = _LaggingMetricsRest([t + b for t in stamps], stamps)  # one bucket later: START + b only from taker
    asyncio.run(app._poll_metrics())
    rows = {r["ts"]: r for r in app.store.load_metrics("BTCUSDT")}
    assert rows[START]["oi"] == 1000 and rows[START]["top_ls_positions"] == 1.5 and rows[START]["taker_ls_vol"] == 0.9
    app.rest = _LaggingMetricsRest([t + b for t in stamps], [])  # the taker series answers nothing this time
    asyncio.run(app._poll_metrics())
    rows = {r["ts"]: r for r in app.store.load_metrics("BTCUSDT")}
    assert all(rows[t - b]["taker_ls_vol"] == 0.9 for t in stamps)  # nothing stored is lost
    assert rows[START + 5 * b]["taker_ls_vol"] == 0.9 and rows[START + 6 * b]["taker_ls_vol"] is None


def test_a_trade_closed_seconds_after_midnight_counts_for_the_new_day():
    """The engines' daily counters (the "today" line of close messages) were reset only when the 5 s scheduler noticed
    the new day: a stop filled in between was booked into the old day's counter and then wiped. The day now rolls
    before any tick, bar or owner close reaches an engine."""
    from synth import synth_symbols

    from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, Ticker
    from heartless.strategy.base import Context
    from heartless.util import timeutil

    clock = {"t": START + MS_DAY - 60_000}  # 23:59 of day 1 (UTC)
    prev = timeutil.set_clock(lambda: clock["t"])
    try:
        app = _app(TIMEZONE="UTC")
        app.symbols = synth_symbols(["BTCUSDT"])
        asyncio.run(_open_long(app, Ticker, Signal, Decision, EntryStyle, Regime, Side, Context))
        eng = app.engines["paper"]
        clock["t"] = START + MS_DAY + 1_000  # 00:00:01 of day 2: the open tick of the new day hits the stop
        tk = Ticker("BTCUSDT", bid=96.99, ask=97.01, mark=97.0, last=97.0, ts=clock["t"])
        asyncio.run(app._tick("BTCUSDT", tk))
        assert "BTCUSDT" not in eng.positions
        app._roll_day(START + MS_DAY + 5_000)  # the scheduler's own check, seconds later
        today = sum(t["pnl"] for t in app.store.load_trades(engine="paper", since=START + MS_DAY))
        assert today < 0 and eng.stats.realized_today == today and eng.stats.trades_today == 1
    finally:
        timeutil.set_clock(prev)


async def _open_long(app, Ticker, Signal, Decision, EntryStyle, Regime, Side, Context) -> None:
    await app._create_engines()
    eng = app.engines["paper"]
    tk = Ticker("BTCUSDT", bid=99.99, ask=100.01, mark=100.0, last=100.0, ts=START)
    await app.paper_accounts["paper"].on_ticker(tk)
    sig = Signal(alpha="htf_trend", symbol="BTCUSDT", side=Side.LONG, confidence=0.7, reason="t", stop=98.0,
                 take_profit=None, tp1=None, entry_style=EntryStyle.MARKET, atr=2.0, timeframe="1h")
    d = Decision(symbol="BTCUSDT", side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"], primary=sig,
                 reason="t", regime=Regime.RANGE)

    class _View:
        frames: dict = {}
        price = 100.0

    await eng._open(d, _View(), Context(symbol="BTCUSDT", info=app.symbols["BTCUSDT"], ticker=tk, regime=Regime.RANGE,
                                        now=START), await eng.equity())
    assert eng.positions["BTCUSDT"].status.value == "OPEN"


async def _long(eng, sym: str, px: float) -> None:
    """Open a MARKET long on `eng` (its account a PaperAccount) at `px`, stop 2 % below."""
    from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, Ticker
    from heartless.strategy.base import Context
    from heartless.util.timeutil import now_ms

    tk = Ticker(sym, bid=px * 0.9999, ask=px * 1.0001, mark=px, last=px, ts=now_ms())
    await eng.account.on_ticker(tk)
    sig = Signal(alpha="htf_trend", symbol=sym, side=Side.LONG, confidence=0.7, reason="t", stop=px * 0.98,
                 take_profit=None, tp1=None, entry_style=EntryStyle.MARKET, atr=px * 0.02, timeframe="1h")
    d = Decision(symbol=sym, side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"], primary=sig, reason="t",
                 regime=Regime.RANGE)

    class _View:
        frames: dict = {}
        price = px

    await eng._open(d, _View(), Context(symbol=sym, info=eng.symbols[sym], ticker=tk, regime=Regime.RANGE,
                                        now=now_ms()), await eng.equity())
    assert eng.positions[sym].status.value == "OPEN"


def _around_midnight(body) -> None:
    """Run `body(app, loop)` on the replay's virtual clock from 23:59:56 of day 1 (UTC), with the parts of the
    scheduler pass that would reach the network or start a research worker stubbed out."""
    from synth import synth_symbols

    from heartless.replay.clock import run_virtual

    async def main(loop):
        app = _app(TIMEZONE="UTC")
        app.symbols = synth_symbols(["BTCUSDT", "ETHUSDT"])

        async def nothing(*a, **k):
            return None

        app.refresh_universe = nothing
        app.research.due = app.research.discovery_due = lambda now: False
        await app._create_engines()
        try:
            await body(app, loop)
        finally:
            for t in app.tasks:
                t.cancel()

    run_virtual(main, START + MS_DAY - 4_000)


def _today(app, engine: str) -> float:
    return sum(t["pnl"] for t in app.store.load_trades(engine=engine, since=START + MS_DAY))


def test_a_trade_the_scheduler_books_after_midnight_counts_for_the_new_day():
    """The scheduler rolled the day late in its pass, after the paper reconcile (which books trades): a position
    finalized there in the first pass after midnight went into the old day's counter and was wiped."""
    async def body(app, loop):
        paper = app.engines["paper"]
        await _long(paper, "BTCUSDT", 100.0)
        app.paper_accounts["paper"].positions["BTCUSDT"].qty = 0.0  # gone from the simulator: the reconcile books it
        app.tasks.append(asyncio.create_task(app._scheduler()))
        await loop.sleep_until_ms(START + MS_DAY + 2_000)  # the first pass runs at 00:00:01
        assert "BTCUSDT" not in paper.positions
        assert _today(app, "paper") < 0 and paper.stats.realized_today == _today(app, "paper")

    _around_midnight(body)


def test_a_mode_switch_flatten_after_midnight_counts_for_the_new_day():
    """/mode paper flattens the live engine directly (not through close_all): right after midnight its closes went into
    the old day's counter, which the next scheduler pass then wiped."""
    from heartless.exchange.paper import PaperAccount
    from heartless.execution.engine import TradingEngine

    async def body(app, loop):
        acc = PaperAccount("live", 10_000.0)
        acc.set_symbols(app.symbols)
        live = TradingEngine("live", acc, app.params, app.s, app.symbols, app.store, app.bus, app.bandit, notify=True)
        await live.start()
        app.engines["live"], app.mode = live, "live"
        await _long(live, "ETHUSDT", 200.0)
        await live.close_position("ETHUSDT", "t")  # a trade on day 1
        await _long(live, "ETHUSDT", 200.0)
        app.tasks.append(asyncio.create_task(app._scheduler()))  # its first pass (00:00:01) rolls the day
        await loop.sleep_until_ms(START + MS_DAY + 500)
        assert await app.set_mode("paper") == "페이퍼 모드로 전환했습니다"  # flattens at 00:00:00.5
        assert _today(app, "live") < 0 and live.stats.realized_today == _today(app, "live")
        assert live.stats.trades_today == 1

    _around_midnight(body)

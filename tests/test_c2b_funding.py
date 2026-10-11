"""C2-B: funding settlements per symbol. The app takes each symbol's next settlement time from the exchange
(premiumIndex nextFundingTime, the markPrice stream) and its interval from fundingInfo (a symbol it does not list settles
every 8 h) instead of assuming 8 h, so a 4-hourly symbol's settlements reach the store, the market context and the
engines on time. The replay market derives the same schedule from the stored settlements (SOLUSDT settled every 2-4 h
during November 2022)."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from heartless.app import Heartless
from heartless.config import Settings
from heartless.core.models import Candle, Ticker
from heartless.core.store import Store
from heartless.data import context as C
from heartless.replay.market import FUNDING_INTERVAL, ReplayRest, ReplaySource
from heartless.util import timeutil
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

START = 1_772_323_200_000  # 2026-03-01 00:00 UTC
H4 = 4 * MS_HOUR


class _Rest:
    """premiumIndex / fundingInfo / funding history of a 4-hourly SOLUSDT and an 8-hourly BTCUSDT."""

    def __init__(self, settled: dict[str, list[tuple[int, float]]], interval: dict[str, int]):
        self.settled, self.interval = settled, interval
        self.calls: list[tuple] = []

    def _next(self, sym: str, now: int) -> int:
        iv = self.interval[sym]
        return (now // iv + 1) * iv

    async def premium_index(self, symbol=None):
        now = timeutil.now_ms()
        self.calls.append(("premiumIndex",))
        return [{"symbol": s, "nextFundingTime": self._next(s, now)} for s in self.settled]

    async def request(self, method, path, params=None, signed=False, retries=3):
        assert path == "/fapi/v1/fundingInfo"
        self.calls.append(("fundingInfo",))
        return [{"symbol": s, "fundingIntervalHours": iv // MS_HOUR} for s, iv in self.interval.items()
                if iv != 8 * MS_HOUR]

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        self.calls.append(("funding", symbol, start, end))
        now = timeutil.now_ms()
        return [{"fundingTime": t, "fundingRate": repr(r), "markPrice": "0"} for t, r in self.settled[symbol]
                if (start or 0) <= t <= min(end or now, now)]


def _app(tmp: Path) -> Heartless:
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ARCHIVE_BACKFILL=False, CONTEXT_UNIVERSE="BTCUSDT,SOLUSDT")
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return Heartless(s)


def _at(t: int, coro):
    prev = timeutil.set_clock(lambda: t)
    try:
        return asyncio.run(coro)
    finally:
        timeutil.set_clock(prev)


def test_a_four_hourly_settlement_is_stored_and_reaches_the_context_on_time():
    app = _app(Path(tempfile.mkdtemp()))
    app.universe = ["BTCUSDT", "SOLUSDT"]
    settled = {"SOLUSDT": [(START + k * H4, 1e-4 * (k + 1)) for k in range(4)],
               "BTCUSDT": [(START + k * 8 * MS_HOUR, 2e-4) for k in range(2)]}
    app.rest = _Rest(settled, {"SOLUSDT": H4, "BTCUSDT": 8 * MS_HOUR})
    for sym in settled:
        app.store.save_funding(sym, [(START, settled[sym][0][1], 0.0)])
    fed: list[tuple] = []
    app.context.on_funding = lambda sym, t, r: fed.append((sym, t, r))
    _at(START + H4 + 3 * MS_MINUTE, app._funding_tick())  # 04:03, ten minutes' refresh grid
    assert app.funding_interval == {"SOLUSDT": H4} and app.funding_next["SOLUSDT"] == START + 2 * H4
    assert app._funding_interval_ms("BTCUSDT") == 8 * MS_HOUR
    # SOLUSDT's 04:00 settlement is stored and handed to the market context; BTCUSDT has nothing due until 08:00
    assert app.store.load_funding("SOLUSDT", START, START + MS_DAY)[-1] == (START + H4, 2e-4)
    assert fed == [("SOLUSDT", START + H4, 2e-4)]
    assert [c for c in app.rest.calls if c[0] == "funding"] == [("funding", "SOLUSDT", START + 1, START + H4 + 3 * MS_MINUTE)]
    # nothing due at 04:13
    app.rest.calls.clear()
    _at(START + H4 + 13 * MS_MINUTE, app._funding_tick())
    assert [c for c in app.rest.calls if c[0] == "funding"] == []
    # 08:01: both settle
    _at(START + 2 * H4 + MS_MINUTE, app._funding_tick())
    assert {c[1] for c in app.rest.calls if c[0] == "funding"} == {"SOLUSDT", "BTCUSDT"}
    assert app.store.funding_range("SOLUSDT")[1] == app.store.funding_range("BTCUSDT")[1] == START + 2 * H4
    app.store.close()


def test_an_expected_settlement_that_never_shows_is_not_asked_for_again():
    app = _app(Path(tempfile.mkdtemp()))
    app.universe = ["SOLUSDT"]
    app.rest = _Rest({"SOLUSDT": [(START, 1e-4)]}, {"SOLUSDT": H4})  # the 04:00 row never appears
    app.store.save_funding("SOLUSDT", [(START, 1e-4, 0.0)])
    for minutes, asked in ((2, True), (12, True), (22, False), (32, False)):
        app.rest.calls.clear()
        _at(START + H4 + minutes * MS_MINUTE, app._funding_tick())
        assert bool([c for c in app.rest.calls if c[0] == "funding"]) is asked, minutes
    app.rest.calls.clear()
    _at(START + 2 * H4 + MS_MINUTE, app._funding_tick())  # the next expectation is asked for as usual
    assert [c for c in app.rest.calls if c[0] == "funding"]
    app.store.close()


def test_without_a_schedule_the_eight_hour_rule_and_the_context_fallback_apply():
    app = _app(Path(tempfile.mkdtemp()))
    assert app._funding_due("BTCUSDT", START, START + 8 * MS_HOUR - 1) is None
    assert app._funding_due("BTCUSDT", START, START + 8 * MS_HOUR) == START + 8 * MS_HOUR
    # the engines' minutes to funding: the stream's T, else the exchange's schedule, else the symbol's interval grid
    from heartless.core.models import SymbolInfo

    app.symbols = {"SOLUSDT": SymbolInfo("SOLUSDT", "SOL", "USDT", 0.01, 0.001, 0.001, 5.0, 2, 3)}
    app.tickers["SOLUSDT"] = Ticker("SOLUSDT", mark=100.0)
    now = START + MS_HOUR
    assert app._context("SOLUSDT", now).minutes_to_funding == 7 * 60
    app.funding_interval["SOLUSDT"] = H4
    assert app._context("SOLUSDT", now).minutes_to_funding == 3 * 60
    app.funding_next["SOLUSDT"] = START + 2 * MS_HOUR
    assert app._context("SOLUSDT", now).minutes_to_funding == 60
    app.store.close()


# --- the replay market serves a consistent per-symbol schedule ----------------------------------------------------------
def _source(tmp: Path) -> ReplaySource:
    st = Store(tmp / "src.db")
    n = 3 * 1440
    for sym in ("BTCUSDT", "SOLUSDT"):
        st.save_candles(sym, [Candle(START + i * MS_MINUTE, 100.0, 100.5, 99.5, 100.0, 10.0, 1000.0, 3, 5.0,
                                     START + i * MS_MINUTE + 59_999) for i in range(n)])
    st.save_funding("BTCUSDT", [(START + k * 8 * MS_HOUR, 1e-4, 0.0) for k in range(10)])
    # SOLUSDT: 8-hourly, then every 2 h from 16:00 on day 1 (like November 2022), back to 8 h at 00:00 day 2
    sol = [START, START + 8 * MS_HOUR, START + 16 * MS_HOUR] + \
          [START + 16 * MS_HOUR + k * 2 * MS_HOUR for k in range(1, 5)] + [START + 32 * MS_HOUR, START + 40 * MS_HOUR]
    st.save_funding("SOLUSDT", [(t, 1e-4 * (i + 1), 0.0) for i, t in enumerate(sol)])
    st.close()
    return ReplaySource(str(tmp / "src.db"), ["BTCUSDT", "SOLUSDT"])


def test_the_replay_schedule_follows_the_stored_settlements(tmp_path):
    src = _source(tmp_path)
    try:
        assert src.funding_schedule("BTCUSDT", START + MS_HOUR) == (START + 8 * MS_HOUR, FUNDING_INTERVAL)
        assert src.funding_schedule("SOLUSDT", START + 17 * MS_HOUR) == (START + 18 * MS_HOUR, 2 * MS_HOUR)
        assert src.funding_schedule("SOLUSDT", START + 25 * MS_HOUR) == (START + 32 * MS_HOUR, 8 * MS_HOUR)
        # past the stored history: the last interval continues
        assert src.funding_schedule("SOLUSDT", START + 41 * MS_HOUR) == (START + 48 * MS_HOUR, 8 * MS_HOUR)
        # the stream's estimate blends over the 2-hour period and is charged at its end
        r_mid, t_mid = src.funding_upcoming("SOLUSDT", START + 17 * MS_HOUR)
        assert t_mid == START + 18 * MS_HOUR and 3e-4 < r_mid < 4e-4
        r_end, _ = src.funding_upcoming("SOLUSDT", START + 18 * MS_HOUR - 30_000)
        assert r_end == 4e-4
        rest = ReplayRest(src, {s: None for s in src.symbols})
        prev = timeutil.set_clock(lambda: START + 17 * MS_HOUR)
        try:
            prem = asyncio.run(rest.premium_index())
            info = asyncio.run(rest.request("GET", "/fapi/v1/fundingInfo"))
        finally:
            timeutil.set_clock(prev)
        assert {p["symbol"]: p["nextFundingTime"] for p in prem} == {"BTCUSDT": START + 24 * MS_HOUR,
                                                                     "SOLUSDT": START + 18 * MS_HOUR}
        assert [(i["symbol"], i["fundingIntervalHours"]) for i in info] == [("SOLUSDT", 2)]
    finally:
        src.close()


def test_a_replayed_bot_puts_a_short_interval_settlement_into_the_next_snapshot(tmp_path, monkeypatch):
    """End to end in a replay: SOLUSDT settles every 4 h. The bot stores each settlement within the 10-minute refresh
    and the hour-close snapshots after it are bit-identical to the offline build (with the 8-hour assumption the
    04:00 and 12:00 settlements arrived 4 h late and the snapshots in between carried the old rate)."""
    from heartless.data import features
    from heartless.replay.runner import ReplayConfig, context_digest, run_replay

    for tf, n in {"1m": 240, "5m": 240, "15m": 240, "1h": 260}.items():
        monkeypatch.setitem(features.LIVE_WINDOW, tf, n)
    seed_days, hours = 14, 14
    start = START + 30 * MS_DAY
    t0 = start - seed_days * MS_DAY
    n = (seed_days * 24 + hours) * 60
    rng = np.random.default_rng(3)
    src = Store(tmp_path / "src.db")
    syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    for k, sym in enumerate(syms):
        close = (100 + 30 * k) * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
        opn = np.concatenate(([close[0]], close[:-1]))
        vol = np.abs(rng.normal(40, 10, n))
        src.save_candles(sym, [Candle(t0 + i * MS_MINUTE, float(opn[i]), float(max(opn[i], close[i]) * 1.0002),
                                      float(min(opn[i], close[i]) * 0.9998), float(close[i]), float(vol[i]),
                                      float(vol[i] * close[i]), 3, float(vol[i] / 2), t0 + i * MS_MINUTE + 59_999)
                                for i in range(n)])
        step = H4 if sym == "SOLUSDT" else 8 * MS_HOUR
        src.save_funding(sym, [(t, float(rng.normal(1e-4, 3e-4)), 0.0) for t in range(t0, t0 + n * MS_MINUTE, step)])
        src.save_metrics(sym, [SimpleNamespace(ts=t, oi=1e6 + 50 * j, oi_value=1e8 + j, top_ls_accounts=1.1,
                                               top_ls_positions=1.2, ls_accounts=1.3, taker_ls_vol=1.0)
                               for j, t in enumerate(range(t0, t0 + n * MS_MINUTE, 300_000))])
    cfg = ReplayConfig(source_db=str(tmp_path / "src.db"), start_ms=start, end_ms=start + hours * MS_HOUR - 1,
                       work_dir=str(tmp_path / "bot"), symbols=["BTCUSDT", "ETHUSDT"], seed_days=seed_days,
                       seed_metrics=True, commands=[],
                       settings={"RESEARCH_INTERVAL_MINUTES": 10**9, "DISCOVERY_INTERVAL_HOURS": 10**8,
                                 "HISTORY_DAYS": 12, "CONTEXT_UNIVERSE": ",".join(syms)})
    s = run_replay(cfg)
    assert s["passed"], s["report"]
    bot = Store(tmp_path / "bot" / "heartless.db")
    sol = [t for t, _ in bot.load_funding("SOLUSDT", start, start + hours * MS_HOUR)]
    assert sol == [start + k * H4 for k in range(4)]  # 00, 04, 08, 12 h: every settlement in the window
    bot.close()
    dst = Store(tmp_path / "ctx.db")
    C.build(src, dst, universe=syms)
    table = C.ContextTable.load(dst, syms, start - MS_HOUR, start + hours * MS_HOUR)
    digests = s["context"]["digests"]
    assert len(digests) == hours + 1
    for ts, digest in digests.items():
        T = int(ts)
        rows = [(C.MARKET_KEY, T, table.market[T].tobytes())]
        rows += [(sym, T, table.symbols[sym][T].tobytes()) for sym in syms if T in table.symbols.get(sym, {})]
        assert context_digest(rows) == digest, C._iso(T)
    assert s["market"]["rest_calls"].get("/fapi/v1/fundingInfo", 0) >= hours * 6 - 1

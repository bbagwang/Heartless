"""C2-A long-horizon support: UTC 4h / 1d frames in MarketView, Alpha.lookback_days in the lab / backtester warm-up
and in the live 1m history depth, and the bounded-cost live resample."""
import asyncio
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from heartless.app import LIVE_HISTORY_DAYS, Heartless
from heartless.config import Settings
from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data.candles import CandleArrays, resample, resample_tail
from heartless.data.features import LIVE_WINDOW, LONG_TIMEFRAMES, TIMEFRAMES, FeatureFrame, MarketView
from heartless.learning import lab, trials
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.base import Alpha, required_lookback_days
from heartless.strategy.ensemble import Ensemble
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

START = 1_704_067_200_000  # 2024-01-01 00:00 UTC
INLINE = SimpleNamespace(map=map)


@pytest.fixture(autouse=True)
def _sandboxed_trial_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "LEDGER", tmp_path / "p2_trials.jsonl")


def _candles(n: int, seed: int, start: int = START, holes=()) -> CandleArrays:
    """1m random walk with optional holes [(first minute, minutes)]."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    keep = np.ones(n, dtype=bool)
    for a, k in holes:
        keep[a:a + k] = False
    ca = CandleArrays("1m", capacity=n + 16)
    prev = float(close[0])
    for i in np.flatnonzero(keep):
        c = float(close[i])
        v = float(abs(rng.normal(50, 15)) + 1)
        t = start + int(i) * MS_MINUTE
        ca.append(Candle(t, prev, max(prev, c) * 1.0003, min(prev, c) * 0.9997, c, v, v * c, 4, v * 0.5, t + 59_999))
        prev = c
    return ca


def _same_frame(a: FeatureFrame, b: FeatureFrame) -> bool:
    if a.n != b.n or not np.array_equal(a.close_time, b.close_time) or set(a.f) != set(b.f):
        return False
    return all(np.array_equal(a.f[k], b.f[k], equal_nan=True) for k in a.f)


# --- resample_tail ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("tf", ["5m", "15m", "1h", "4h", "1d"])
def test_resample_tail_equals_the_tail_of_the_full_resample(tf):
    ca = _candles(9 * 1440 + 17, 1, start=START + 7 * MS_MINUTE,  # unaligned start, an open last group, holes
                  holes=[(500, 3), (2000, 200), (6000, 1500)])
    for n in (1, 5, 30, 200, 520, 5000):
        a, b = resample(ca, tf).tail(n), resample_tail(ca, tf, n)
        assert a.n == b.n, (tf, n)
        for col in ("open_time", "close_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                    "taker_buy"):
            assert np.array_equal(a.view(col), b.view(col)), (tf, n, col)


def test_the_live_rebuild_is_unchanged_with_a_long_history():
    """app._rebuild_view now resamples only the needed 1m tail: frames identical to resampling everything."""
    ca = _candles(70 * 1440, 2, holes=[(30 * 1440, 90)])
    fast = MarketView("X")
    Heartless._rebuild_view(fast, ca, list(TIMEFRAMES))
    for tf in TIMEFRAMES:
        bars = ca if tf == "1m" else resample(ca, tf)
        assert _same_frame(fast.frames[tf], FeatureFrame(tf, bars.tail(LIVE_WINDOW[tf]))), tf
    live = MarketView("X")
    live.rebuild(ca, live=True)
    for tf in TIMEFRAMES:
        assert _same_frame(live.frames[tf], fast.frames[tf]), tf


# --- 4h / 1d frames ---------------------------------------------------------------------------------------------------
def test_long_frames_are_utc_aligned_with_the_same_columns():
    ca = _candles(20 * 1440 + 300, 3, start=START + 13 * MS_MINUTE)
    v = MarketView("X")
    v.rebuild(ca, live=False)
    assert set(LONG_TIMEFRAMES) <= set(v.frames)
    for tf, period in (("4h", 4 * MS_HOUR), ("1d", MS_DAY)):
        fr = v.frames[tf]
        assert _same_frame(fr, FeatureFrame(tf, resample(ca, tf)))
        assert np.all(fr.open_time % period == 0) and np.all((fr.close_time + 1) % period == 0)
        assert set(fr.f) == set(v.frames["1h"].f)
    assert v.frames["1d"].n == 20 and v.frames["4h"].n == 20 * 6 + 1  # the open last group is not a bar


def test_long_cursors_are_causal_and_closed_only_in_the_minute_after_a_period_end():
    ca = _candles(6 * 1440, 4, holes=[(2 * 1440 - 1, 1)])  # the last minute of day 2 is missing
    v = MarketView("X")
    v.rebuild(ca, live=False)
    calls = []
    orig = v._long_cursor
    v._long_cursor = lambda tf: calls.append(tf) or orig(tf)
    ct = ca.view("close_time")
    for t in ct.tolist():
        v.seek(t)
        for tf in LONG_TIMEFRAMES:
            fr = v.frames[tf]
            idx = int(np.searchsorted(fr.close_time, t, side="right")) - 1
            just = idx >= 0 and int(fr.close_time[idx]) > t - MS_MINUTE
            assert v.closed(tf) == just == (tf in v.just_closed), (tf, t)
    # seek() itself looked at the long frames only in boundary minutes: 6 per day for 4h, 1 per day for 1d (minus the
    # missing 23:59 bar of day 2, whose day is still a bar: it closed by the next bar after the gap)
    n_seek = sum(1 for c in calls if c == "1d")
    assert n_seek == 5 and calls.count("4h") == 6 * 6 - 1
    v.seek(int(START + 3 * MS_DAY - 1))  # the close of day 3
    assert v.closed("1d") and v.tf("1d").idx == 2 and v.tf("1d").close_time == START + 3 * MS_DAY - 1
    v.seek(int(START + 3 * MS_DAY + 5 * MS_HOUR - 1))
    assert not v.closed("1d") and v.tf("1d").idx == 2 and v.tf("4h").close_time == START + 3 * MS_DAY + 4 * MS_HOUR - 1
    assert v.tf("4h").v("close") == float(ca.close[ca.index_at_or_before(START + 3 * MS_DAY + 4 * MS_HOUR - 1)])


def test_live_long_frames_are_rebuilt_once_per_period_not_per_minute(monkeypatch):
    """The live path: the base grows a bar a minute (app appends and seeks); a long frame is rebuilt when a period
    has closed and someone looks, never in between, and always equals a fresh resample of the base."""
    import heartless.data.features as F

    full = _candles(5 * 1440, 5)
    ca = full.slice(0, 2 * 1440 + 30)
    v = MarketView("X")
    v.rebuild(ca, live=True)
    built = []
    orig = F.FeatureFrame

    class Counting(orig):
        def __init__(self, tf, bars):
            built.append(tf)
            super().__init__(tf, bars)

    monkeypatch.setattr(F, "FeatureFrame", Counting)
    closes = {"4h": 0, "1d": 0}
    for i in range(ca.n, full.n):
        ca.append(full.candle_at(i))
        t = int(ca.close_time[ca.n - 1])
        v.seek(t)
        if (t + 1) % (4 * MS_HOUR) == 0:
            closes["4h"] += 1
            assert v.closed("4h") and _same_frame(v.frames["4h"], orig("4h", resample(ca, "4h")))
        if (t + 1) % MS_DAY == 0:
            closes["1d"] += 1
            assert v.closed("1d") and _same_frame(v.tf("1d").frame, orig("1d", resample(ca, "1d")))
        else:
            assert not v.closed("1d")
    assert closes == {"4h": 18, "1d": 3}  # 2 days 00:30 .. 5 days
    assert built.count("1d") == closes["1d"] and built.count("4h") == closes["4h"]  # one rebuild per period
    assert set(built) <= set(LONG_TIMEFRAMES)  # base frames are app._rebuild_view's business (not touched here)


# --- lookback_days: lab / backtester warm-up -----------------------------------------------------------------------
@pytest.fixture
def long_alpha(monkeypatch):
    """trend_pullback declares a 60-day lookback and is replaced in the ensemble by a probe recording what it sees."""
    monkeypatch.setattr(ALPHA_BY_NAME["trend_pullback"], "lookback_days", 60.0, raising=False)

    class Probe(Alpha):
        name = "trend_pullback"
        timeframe = "1h"
        seen: list = []

        def evaluate(self, view, ctx, p):
            d = view.tf("1d")
            Probe.seen.append((int(view.cursor_time), d.idx + 1, d.close_time))
            return None

    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Probe()], only_alpha))
    Probe.seen = []
    return Probe


def test_warmup_covers_the_longest_lookback_of_the_evaluated_alphas(long_alpha):
    p = StrategyParams.default()
    assert required_lookback_days([p]) == 0.0 and lab.warmup_for(p) == lab.WARMUP_MS  # existing alphas: unchanged
    assert lab.warmup_for(p, "trend_pullback") == 62 * MS_DAY  # an isolation run evaluates it even when off
    p.enabled["trend_pullback"] = True
    assert required_lookback_days([p]) == 60.0 and lab.warmup_for(p.to_dict()) == 62 * MS_DAY
    assert lab.warmup_for(p, "htf_trend") == lab.WARMUP_MS  # another alpha alone
    assert lab.warmup_for(p, base=90 * MS_DAY) == 90 * MS_DAY
    assert required_lookback_days([StrategyParams.default(), p]) == 60.0


def test_lab_jobs_get_the_lookback_warmup_also_when_chunked(long_alpha, tmp_path):
    p = StrategyParams.default()
    p.enabled["trend_pullback"] = True
    jobs = []
    pool = SimpleNamespace(map=lambda fn, js: [jobs.append(j) or {"symbol": j["symbol"], "chunk": j["label"],
                                                                   "start": j["start"], "end": j["end"],
                                                                   "trades": []} for j in js])
    y23 = START - 365 * MS_DAY  # 2023-01-01: 400 days cross one calendar year boundary (and stay before the seal)
    lab.evaluate(p, ["AUSDT", "BUSDT"], y23, y23 + 400 * MS_DAY, str(tmp_path / "none.db"), pool=pool)
    assert len(jobs) == 4 and {j["warmup_ms"] for j in jobs} == {62 * MS_DAY}  # 2 symbols x 2 calendar years
    jobs.clear()
    lab.evaluate(p, ["AUSDT"], START, START + 100 * MS_DAY, str(tmp_path / "none.db"), pool=pool, chunk_days=30)
    assert len(jobs) == 4 and {j["warmup_ms"] for j in jobs} == {62 * MS_DAY}
    jobs.clear()
    lab.evaluate(StrategyParams.default(), ["AUSDT"], START, START + 10 * MS_DAY, str(tmp_path / "none.db"), pool=pool)
    assert jobs[0]["warmup_ms"] == lab.WARMUP_MS


@pytest.fixture(scope="module")
def hist_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("c2a") / "heartless.db"
    st = Store(path)
    ca = _candles(110 * 1440, 6)
    st.save_candles("AUSDT", [ca.candle_at(i) for i in range(ca.n)])
    st.close()
    return str(path)


def test_the_alpha_sees_its_lookback_from_the_first_decision_in_lab_and_backtest(long_alpha, hist_db, tmp_path,
                                                                                monkeypatch, capsys):
    p = StrategyParams.default()
    p.enabled["trend_pullback"] = True
    a, b = START + 90 * MS_DAY, START + 95 * MS_DAY - 1
    lab.evaluate(p, ["AUSDT"], a, b, hist_db, pool=INLINE)
    first = long_alpha.seen[0]
    assert first[0] >= a and first[1] >= 60 and first[2] < first[0] + 1  # 60+ closed days, none in the future
    # with the default 12-day warm-up the same decision would have seen 12 days at most
    long_alpha.seen = []
    lab.evaluate(p, ["AUSDT"], a, b, hist_db, pool=INLINE, warmup_ms=lab.WARMUP_MS)
    assert long_alpha.seen[0][1] >= 60  # warmup_ms is a floor: the lookback still wins
    # the joint-portfolio backtest CLI warms up the same way
    import json

    import heartless.cli as cli

    data = Path(hist_db).parent
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps(p.to_dict()))
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(data))
    monkeypatch.setenv("CONTEXT_DB", str(tmp_path / "ctx.db"))
    long_alpha.seen = []
    cli.main(["backtest", "--start", "2024-03-31", "--end", "2024-04-02", "--params", str(pf), "--json"])
    capsys.readouterr()
    assert long_alpha.seen and long_alpha.seen[0][1] >= 60


# --- live 1m history depth ------------------------------------------------------------------------------------------
def _app(**kw) -> Heartless:
    tmp = Path(tempfile.mkdtemp())
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                 ALWAYS_INCLUDE="BTCUSDT", **kw)
    s.data_dir.mkdir(exist_ok=True)
    return Heartless(s)


def test_live_history_covers_the_longest_lookback_of_champion_and_challengers(long_alpha, monkeypatch):
    app = _app()
    assert app._alpha_history_days() == 0
    ch = StrategyParams.default()
    ch.enabled["trend_pullback"] = True
    app.challengers = [SimpleNamespace(params=ch, name="challenger-1")]
    assert app._alpha_history_days() == 62
    # the daily trim keeps the lookback (and only then)
    days = 70
    ca = _candles(days * 1440, 7, start=START - days * MS_DAY)
    app._trim_history(ca, START)
    assert int(ca.open_time[0]) == START - 62 * MS_DAY
    app.challengers = []
    app._trim_history(ca, START + MS_DAY)
    assert int(ca.open_time[0]) == START + MS_DAY - LIVE_HISTORY_DAYS * MS_DAY
    # views are built with that depth from the store, the retention prune keeps it
    app.challengers = [SimpleNamespace(params=ch, name="challenger-1")]
    now = START + 80 * MS_DAY
    monkeypatch.setattr("heartless.app.now_ms", lambda: now)
    full = _candles(80 * 1440, 8)
    app.store.save_candles("BTCUSDT", [full.candle_at(i) for i in range(full.n)])
    app._build_views(["BTCUSDT"])
    assert int(app.candles["BTCUSDT"].open_time[0]) == now - 62 * MS_DAY
    assert app.views["BTCUSDT"].frames["1d"].n >= 61
    cut = []
    monkeypatch.setattr(app.store, "prune_candles", cut.append)
    app._prune(now)
    assert cut == [now - (max(app.s.data_retention_days, 62) + 5) * MS_DAY]


def test_a_new_long_lookback_challenger_gets_its_history_once(long_alpha, monkeypatch):
    app = _app(HISTORY_DAYS=21)
    now = START + 80 * MS_DAY
    monkeypatch.setattr("heartless.app.now_ms", lambda: now)
    full = _candles(80 * 1440, 9)
    app.store.save_candles("BTCUSDT", [full.candle_at(i) for i in range(full.n)])
    app.universe = ["BTCUSDT"]
    app._build_views(["BTCUSDT"])
    assert int(app.candles["BTCUSDT"].open_time[0]) == now - 21 * MS_DAY
    fetched = []

    async def backfill(symbols):
        fetched.append(list(symbols))
        return set()

    monkeypatch.setattr(app, "_backfill", backfill)
    asyncio.run(app._ensure_history_depth())
    assert fetched == []  # nothing enabled needs more than HISTORY_DAYS
    ch = StrategyParams.default()
    ch.enabled["trend_pullback"] = True
    app.challengers = [SimpleNamespace(params=ch, name="challenger-1")]
    asyncio.run(app._ensure_history_depth())
    assert fetched == [["BTCUSDT"]] and int(app.candles["BTCUSDT"].open_time[0]) == now - 62 * MS_DAY
    asyncio.run(app._ensure_history_depth())
    assert fetched == [["BTCUSDT"]]  # covered now: no second fetch
    # the scheduler hook runs it at most hourly, in the background
    calls = []

    async def ensure():
        calls.append(1)

    monkeypatch.setattr(app, "_ensure_history_depth", ensure)

    async def ticks():
        app._history_depth_tick(now)
        app._history_depth_tick(now + 60_000)
        await asyncio.sleep(0)
        app._history_depth_tick(now + MS_HOUR)
        await asyncio.sleep(0)

    asyncio.run(ticks())
    assert calls == [1, 1]


def test_an_alpha_on_the_daily_timeframe_is_evaluated_once_per_utc_day():
    """Alpha.timeframe = "1d": the ensemble asks it only in the minute its daily bar closed (00:00 UTC decisions)."""
    from heartless.core.models import Regime, SymbolInfo, Ticker
    from heartless.learning.bandit import AlphaBandit
    from heartless.strategy.base import Context

    class Daily(Alpha):
        name = "trend_pullback"
        timeframe = "1d"
        lookback_days = 3.0
        seen: list = []

        def evaluate(self, view, ctx, p):
            Daily.seen.append((int(view.cursor_time), view.tf("1d").idx))
            return None

    ca = _candles(5 * 1440 + 90, 10)
    v = MarketView("X")
    v.rebuild(ca, live=False)
    p = StrategyParams.default()
    ens = Ensemble(p, AlphaBandit(list(p.alphas)), alphas=[Daily()], only_alpha="trend_pullback")
    ctx = Context("X", SymbolInfo("X", "X", "USDT", 0.01, 0.001, 0.001, 5, 2, 3), Ticker("X"), Regime.RANGE)
    for t in ca.view("close_time").tolist():
        v.seek(t)
        ens.evaluate_signals(v, ctx)
    assert Daily.seen == [(START + (d + 1) * MS_DAY - 1, d) for d in range(5)]


def test_chunk_handoff_covers_the_longest_hold(monkeypatch, tmp_path):
    """Alpha.max_hold_days lengthens the burn-in / run-off of chunked lab runs (multi-week daily trades)."""
    p = StrategyParams.default()
    assert lab.handoff_for(p, None, lab.BURN_IN_MS, lab.RUNOFF_MS) == (lab.BURN_IN_MS, lab.RUNOFF_MS)
    monkeypatch.setattr(ALPHA_BY_NAME["htf_trend"], "max_hold_days", 20.0, raising=False)
    assert lab.handoff_for(p, "htf_trend", lab.BURN_IN_MS, lab.RUNOFF_MS) == (21 * MS_DAY, 21 * MS_DAY)
    assert lab.handoff_for(p, None, lab.BURN_IN_MS, lab.RUNOFF_MS) == (lab.BURN_IN_MS, lab.RUNOFF_MS)  # htf_trend off
    jobs = []
    pool = SimpleNamespace(map=lambda fn, js: [jobs.append(j) or {"symbol": j["symbol"], "chunk": j["label"],
                                                                   "start": j["start"], "end": j["end"],
                                                                   "trades": []} for j in js])
    lab.evaluate(p, ["AUSDT"], START, START + 90 * MS_DAY, str(tmp_path / "none.db"), only_alpha="htf_trend",
                 pool=pool, chunk_days=30)
    assert len(jobs) == 3 and {(j["burn_in_ms"], j["runoff_ms"]) for j in jobs} == {(21 * MS_DAY, 21 * MS_DAY)}

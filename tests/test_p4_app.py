"""P4 on the real orchestrator (offline, fake Binance REST, synthetic bars): truthful messages while no alpha is enabled,
a challenger that clears the absolute bar is promoted and starts trading, a losing alpha trips the kill switch (and the
result survives a restart), the execution-quality monitor, and archived challenger runs."""
from __future__ import annotations

import json
import tempfile
from collections import deque
from pathlib import Path

import pytest
from synth import synth_candles

from heartless.app import NO_ALPHA_TEXT, Heartless
from heartless.config import Settings
from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, Ticker, TradeRecord
from heartless.notify import formatter as F
from heartless.notify.telegram import TelegramBot
from heartless.replay.invariants import _RE_DAILY_TODAY, _RE_DAILY_WEEK
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, now_ms

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
N_HIST = 1500


def _exchange_info():
    return {"symbols": [{"symbol": s, "baseAsset": s[:-4], "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
                         "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1,
                         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                     {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]} for s in SYMS]}


class FakeRest:
    def __init__(self, candles):
        self.candles = candles
        self.used_weight = 0
        self.ws_base = "wss://fake"
        self.time_offset = 0

    async def sync_time(self): return None
    async def exchange_info(self): return _exchange_info()
    async def ticker_24h(self): return [{"symbol": s, "quoteVolume": "500000000"} for s in SYMS]

    async def klines_range(self, symbol, start, end, interval="1m"):
        ca = self.candles[symbol]
        return [ca.candle_at(i) for i in range(ca.n) if start <= ca.open_time[i] <= end]

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        return []

    async def open_interest_hist(self, symbol, period="15m", limit=20): return []
    async def close(self): return None


def _make(tmp: Path | None = None, candles=None, **settings):
    tmp = tmp or Path(tempfile.mkdtemp())
    if candles is None:
        start = (now_ms() - (N_HIST + 200) * 60_000) // 60_000 * 60_000
        candles = {s: synth_candles(N_HIST + 100, seed=i, start_ms=start, price=100 * (i + 1)) for i, s in enumerate(SYMS)}
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), HISTORY_DAYS=2, UNIVERSE_SIZE=3,
                 ALWAYS_INCLUDE=",".join(SYMS), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="", ARCHIVE_BACKFILL=False,
                 **settings)
    s.data_dir.mkdir(exist_ok=True)
    app = Heartless(s)
    app.rest = FakeRest({k: v.slice(0, N_HIST) for k, v in candles.items()})
    return app, candles, tmp


async def _boot(app, candles):
    await app._load_symbols()
    await app.refresh_universe(initial=True)
    await app._backfill(app.universe)
    app._build_views(app.universe)
    await app._create_engines()
    await app.killswitch.check_all()  # as run() does right after the engines exist
    for i, s in enumerate(SYMS):  # a quote per symbol (the monitor's close reference, the engines' entry touch)
        c = candles[s].candle_at(N_HIST - 1)
        t = app.tickers.setdefault(s, Ticker(s))
        t.bid, t.ask, t.mark, t.last, t.ts = c.close * 0.99995, c.close * 1.00005, c.close, c.close, now_ms()
        await app._tick(s, t)


def _capture(app) -> list[str]:
    """The production Telegram bot on the app's bus with the HTTP send replaced by a list."""
    app.s.telegram_owner_id = 1
    bot = TelegramBot(app)
    sent: list[str] = []
    bot.send = lambda text, keyboard=None, chat_id=None: sent.append(text)
    return sent


def _rec(engine: str, i: int, alpha: str, r: float, exit_time: int, version: str = "v") -> TradeRecord:
    return TradeRecord(position_id=f"{engine}-{alpha}-{i}", engine=engine, symbol="BTCUSDT", side="LONG", alpha=alpha,
                       alphas=[alpha], regime="RANGE", entry_time=exit_time - MS_HOUR, exit_time=exit_time,
                       entry_price=100.0, exit_price=100.0 + r, qty=1.0, notional=100.0, pnl=r * 50.0, gross=r * 50.0,
                       fees=0.0, funding=0.0, r_multiple=r, risk_amount=50.0, exit_reason="x", confidence=0.6,
                       params_version=version, bars_held=60, max_fav_r=0.0, max_adv_r=0.0)


# --- truthful messaging --------------------------------------------------------------------------------------------
async def test_messages_state_that_nothing_trades_and_what_starts_trading():
    app, candles, _ = _make()
    await _boot(app, candles)
    assert app.enabled_alphas() == []  # shipped defaults since the P1 judgement
    text = app.startup_summary()
    assert NO_ALPHA_TEXT in text and "활성 알파 없음" in text and "거래하지 않고 연구·발굴·챌린저 검증만 진행" in text
    assert "거래 시작 조건" in text and "거래 30건 이상·청산일 14일 이상" in text and "t ≥ 1.5" in text
    assert "PF ≥ 1.15" in text and "최대낙폭 ≤ 10R" in text and "30일·t ≥ 2" in text
    assert "OOS 판정: 13개 중 탈락 13, 통과 0" in text and "킬 스위치 켜짐" in text  # P1: 9, P2 candidates: 4
    assert "리서치 사이클이 통과 알파를 켭니다" not in text  # the old, untrue claim
    st = app.status()
    assert st["learning"]["no_alpha"] is True and st["learning"]["evidence_counts"]["failed"] == 13
    status = F.fmt_status(st, "UTC")
    assert NO_ALPHA_TEXT in status and "거래 시작 조건" in status
    daily = F.fmt_daily(app.pnl_report("today"), app.pnl_report("week"), st, "UTC")
    assert NO_ALPHA_TEXT in daily and "거래 시작 조건" in daily
    # the replay checker still parses the report's PnL lines
    assert _RE_DAILY_TODAY.search(daily) and _RE_DAILY_WEEK.search(daily)
    json.dumps(st, default=str)  # the web dashboard serves it as JSON
    await app.stop()


async def test_messages_name_the_enabled_alphas_and_their_evidence():
    app, candles, _ = _make()
    app.params.enabled["htf_trend"] = True
    await _boot(app, candles)
    text = app.startup_summary()
    assert "활성 알파: htf_trend" in text and "OOS 탈락" in text and NO_ALPHA_TEXT not in text
    st = app.status()
    assert st["learning"]["active_evidence"] == {"htf_trend": "failed"}
    assert [k["alpha"] for k in st["kill_switch"]] == ["htf_trend"]
    assert "킬 스위치 <code>htf_trend</code>" in F.fmt_status(st, "UTC")
    assert "OOS 탈락" in F.fmt_alphas(app.bandit.snapshot(), app.params, st["kill_switch"])
    await app.stop()


# --- promotion through the absolute path ----------------------------------------------------------------------------
async def test_a_challenger_clearing_the_absolute_bar_is_promoted_and_trades():
    app, candles, _ = _make()
    await _boot(app, candles)
    sent = _capture(app)
    params = app.params.clone(note="enable trend_pullback")
    params.enabled["trend_pullback"] = True  # failed out of sample (P1): the 30-day / t >= 2 bar applies
    slot = await app.install_challenger(params, source="trend_pullback")
    assert slot is not None and app.research.trial_days(slot) == 30
    slot.started = now_ms() - 34 * MS_DAY
    pattern = ([1.0, -0.4] * 3 + [-1.0, 0.5]) * 8  # 64 trades on 32 days, mean +0.16R, t_day ~3.8
    for i, r in enumerate(pattern):
        app.store.save_trade(_rec(slot.name, i, "trend_pullback", r, slot.started + (i // 2 + 1) * MS_DAY + i % 2 * MS_HOUR,
                                  params.version))
    old = app.params.version
    await app.research.evaluate_challengers(force=True)
    assert app.params.enabled["trend_pullback"] is True and app.params.version != old
    assert app.engines["paper"].params is app.params and slot.name not in app.engines
    assert app.enabled_alphas() == ["trend_pullback"]
    row = app.store.load_params_versions(role="champion", limit=1)[0]
    assert row["id"] == app.params.version and "absolute path" in row["note"] and "n=64" in row["note"]
    d = row["metrics"]["decisions"][0]
    assert d["verdict"] == "promote" and d["path"] == "absolute" and d["failed_alphas"] == ["trend_pullback"]
    retired = next(r for r in app.store.load_params_versions(limit=10) if r["id"] == params.version)
    assert retired["role"] == "retired" and retired["metrics"]["decisions"][0]["verdict"] == "promote"
    assert app.killswitch.epochs["trend_pullback"] >= now_ms() - 60_000  # a fresh live record starts now
    msg = next(m for m in sent if "파라미터 승격" in m)
    assert "절대 경로" in msg and "n=64" in msg and "OOS 탈락 알파 trend_pullback" in msg
    assert NO_ALPHA_TEXT not in app.startup_summary()
    await app.stop()


async def test_identical_candidates_are_not_installed_twice():
    app, candles, _ = _make()
    await _boot(app, candles)
    p = app.params.clone(note="a")
    p.enabled["squeeze_breakout"] = True
    s1 = await app.install_challenger(p, source="squeeze_breakout")
    q = p.clone(note="the same experiment again")
    assert s1 is not None and await app.install_challenger(q, source="squeeze_breakout") is None
    assert len(app.challengers) == 1
    await app.stop()


# --- kill switch -----------------------------------------------------------------------------------------------------
async def test_a_losing_alpha_trips_the_kill_switch_and_stays_off_after_a_restart():
    tmp = Path(tempfile.mkdtemp())
    champ = StrategyParams.default().clone(version="vlive", note="htf_trend on")
    champ.enabled["htf_trend"] = True
    champ.created = now_ms() - 10 * MS_DAY
    app, candles, _ = _make(tmp)
    app.store.set_params_role(app.params.version, "retired")  # the default champion the first start stored
    app.store.save_params_version(champ.version, champ.created, "champion", "test", champ.note, champ.to_dict())
    await app.stop()
    app, candles, _ = _make(tmp, candles)
    assert app.params.version == "vlive" and app.enabled_alphas() == ["htf_trend"]
    await _boot(app, candles)
    sent = _capture(app)
    eng = app.engines["paper"]
    t = now_ms() - 5 * MS_DAY
    for i in range(11):  # eleven full losses: CUSUM -8.25 sd (sd floor 1R)
        rec = _rec("paper", i, "htf_trend", -1.0, t + i * MS_HOUR)
        app.store.save_trade(rec)
        await app.bus.publish("position_closed", {"engine": "paper", "trade": rec, "notify": True})
    assert app.params.enabled["htf_trend"] is False and app.params.version != "vlive"
    assert eng.params is app.params and app.enabled_alphas() == []
    row = app.store.load_params_versions(role="champion", limit=1)[0]
    assert row["source"] == "kill-switch" and row["note"].startswith("kill switch: htf_trend off")
    d = row["metrics"]["decisions"][0]
    assert d["kind"] == "kill_switch" and d["n"] == 11 and d["cusum"] == pytest.approx(-8.25)
    msg = next(m for m in sent if "킬 스위치" in m)
    assert "htf_trend 끔" in msg and "CUSUM -8.25σ" in msg and "챌린저 승격 경로" in msg
    assert app.store.load_events(topic="kill_switch")
    version = app.params.version
    await app.stop()
    # restart: the switched-off champion is what loads; nothing re-enables the alpha by itself
    app2, _, _ = _make(tmp, candles)
    await _boot(app2, candles)
    assert app2.params.version == version and app2.enabled_alphas() == []
    assert "htf_trend" not in app2.killswitch.epochs and NO_ALPHA_TEXT in app2.startup_summary()
    await app2.stop()


async def test_a_trip_stored_before_a_crash_is_acted_on_at_startup():
    tmp = Path(tempfile.mkdtemp())
    champ = StrategyParams.default().clone(version="vlive2", note="htf_trend on")
    champ.enabled["htf_trend"] = True
    champ.created = now_ms() - 10 * MS_DAY
    app, candles, _ = _make(tmp)
    app.store.set_params_role(app.params.version, "retired")  # the default champion the first start stored
    app.store.save_params_version(champ.version, champ.created, "champion", "test", champ.note, champ.to_dict())
    t = now_ms() - 5 * MS_DAY
    for i in range(30):  # a bleed the previous process stored but never acted on: last-30 sum -21R
        app.store.save_trade(_rec("paper", i, "htf_trend", -0.7, t + i * MS_HOUR))
    await app.stop()
    app, _, _ = _make(tmp, candles)
    await _boot(app, candles)  # run() checks every engine right after creating them
    assert app.enabled_alphas() == [] and app.params.version != "vlive2"
    assert "last 30 sum" in app.store.load_params_versions(role="champion", limit=1)[0]["note"]
    await app.stop()


# --- execution quality ------------------------------------------------------------------------------------------------
async def test_fill_quality_is_measured_against_the_decision_reference():
    app, candles, tmp = _make()
    await _boot(app, candles)
    eng = app.engines["paper"]
    sym = "ETHUSDT"
    t = app.tickers[sym]
    ctx = app._context(sym, now_ms())
    sig = Signal(alpha="htf_trend", symbol=sym, side=Side.LONG, confidence=0.7, reason="t", stop=t.ask * 0.97,
                 take_profit=t.ask * 1.06, tp1=None, entry_style=EntryStyle.MARKET, atr=t.ask * 0.01, timeframe="1h")
    await eng._open(Decision(symbol=sym, side=Side.LONG, score=0.7, confidence=0.7, alphas=["htf_trend"], primary=sig,
                             reason="t", regime=Regime.RANGE), app.views[sym], ctx, await eng.equity())
    pos = eng.positions[sym]
    (entry,) = list(app.fill_rows["paper"])
    assert entry["kind"] == "ENTRY" and entry["ref"] == pytest.approx(t.ask) and entry["side"] == "BUY"
    acc = app.paper_accounts["paper"]
    model = acc.slippage_bps + acc.impact_bps_per_10k * entry["ref"] * entry["qty"] / 10_000
    assert entry["slip_bps"] == pytest.approx(model, rel=1e-6)  # the paper fill IS the slippage model
    assert entry["fee"] == pytest.approx(entry["model_fee"]) and entry["position_id"] == pos.id
    # the stop is hit: the fill is measured against the stop trigger
    t.bid = t.ask = t.mark = pos.stop * 0.999
    await app._tick(sym, t)
    rows = list(app.fill_rows["paper"])
    assert [r["kind"] for r in rows] == ["ENTRY", "SL"] and rows[1]["ref"] == pytest.approx(pos.stop)
    assert rows[1]["slip_bps"] > 0 and sym not in eng.positions
    st = app.status()
    q = st["exec_quality"]["paper"]
    assert q["fills"] == 2 and q["taker_fills"] == 2 and q["suggest"] is False
    assert "체결 품질(paper" in F.fmt_status(st, "UTC")
    # rows are persisted: a restart resumes the rolling window
    await app.stop()
    app2, _, _ = _make(tmp, candles)
    await _boot(app2, candles)
    assert [r["kind"] for r in app2.fill_rows["paper"]] == ["ENTRY", "SL"]
    await app2.stop()


async def test_the_daily_report_suggests_a_higher_backtest_slippage_but_never_changes_it():
    app, candles, _ = _make()
    await _boot(app, candles)
    rows = [{"slip_bps": 4.0, "maker": False, "notional": 1000.0, "fee": 0.5, "model_fee": 0.5} for _ in range(60)]
    app.fill_rows["paper"] = deque(rows, maxlen=200)
    st = app.status()
    assert st["exec_quality"]["paper"]["suggest"] is True
    daily = F.fmt_daily(app.pnl_report("today"), app.pnl_report("week"), st, "UTC")
    assert "BACKTEST_SLIPPAGE_BPS(1.5bp)의 2배" in daily and "자동 변경하지 않음" in daily
    assert app.s.backtest_slippage_bps == 1.5
    await app.stop()


# --- challenger runs are archived, not deleted ------------------------------------------------------------------------
async def test_a_reused_slot_archives_the_previous_runs_trades():
    app, candles, _ = _make()
    await _boot(app, candles)
    p1 = app.params.clone(note="one")
    p1.enabled["squeeze_breakout"] = True
    s1 = await app.install_challenger(p1, source="squeeze_breakout")
    app.store.save_trade(_rec(s1.name, 0, "squeeze_breakout", 0.7, now_ms() - 1000, p1.version))
    await app.retire_challenger(s1, reason="test", decision={"verdict": "retire", "rule": "test"})
    p2 = app.params.clone(note="two")
    p2.enabled["mean_reversion"] = True
    s2 = await app.install_challenger(p2, source="mean_reversion")
    assert s2.name == s1.name
    assert app.store.load_trades(engine=s2.name) == []
    (kept,) = app.store.load_trades(engine=f"{s1.name}@{p1.version}")
    assert kept["r_multiple"] == 0.7
    row = next(r for r in app.store.load_params_versions(limit=10) if r["id"] == p1.version)
    assert row["role"] == "retired" and row["metrics"]["decisions"][0]["reason"] == "test"
    await app.stop()


async def test_a_live_fill_arriving_after_the_rest_result_keeps_the_decision_reference():
    """Live: the engine books a market order's REST result first (entry price overwritten, a closed position gone) and
    the user-stream fill reaches the account handlers afterwards; the reference recorded when the order was sent is
    what the fill is measured against."""
    from types import SimpleNamespace

    from heartless.core.models import Fill

    app, candles, _ = _make()
    await _boot(app, candles)

    class StreamAccount:
        def __init__(self):
            self.handlers = []

        def on_fill(self, h):
            self.handlers.append(h)

        async def market_order(self, symbol, side, qty, reduce_only=False, client_id=""):
            return SimpleNamespace(status="FILLED")

        async def limit_order(self, symbol, side, qty, price, post_only=True, reduce_only=False, client_id=""):
            return SimpleNamespace(status="NEW")

    acc = StreamAccount()
    pos = SimpleNamespace(id="P1", side=Side.LONG, status=None, entry_style="MARKET", limit_price=None, filled_qty=0.0,
                          entry_price=100.1, stop=97.0, tp1=None, tp1_done=False, take_profit=None, extra={})
    holder = app._watch_fills(acc, "live")
    holder["engine"] = SimpleNamespace(positions={"ETHUSDT": pos})
    app.tickers["ETHUSDT"] = Ticker("ETHUSDT", bid=99.9, ask=100.1, mark=100.0)
    await acc.market_order("ETHUSDT", "BUY", 1.0, client_id="HLELIVE1")
    pos.entry_price, pos.filled_qty = 100.2, 1.0  # the engine already booked the REST fill
    await acc.handlers[0](Fill("ETHUSDT", "BUY", 1.0, 100.2, 0.05, now_ms(), "HLELIVE1", "1", kind="ENTRY"))
    app.tickers["ETHUSDT"].bid = 99.95
    await acc.market_order("ETHUSDT", "SELL", 1.0, reduce_only=True, client_id="HLCLIVE1")
    holder["engine"].positions.clear()  # finalized from the REST result before the stream fill arrived
    app.tickers["ETHUSDT"].bid = 99.0
    await acc.handlers[0](Fill("ETHUSDT", "SELL", 1.0, 99.90, 0.05, now_ms(), "HLCLIVE1", "2", reduce_only=True,
                               kind="CLOSE"))
    entry, close = app.fill_rows["live"]
    assert entry["kind"] == "ENTRY" and entry["ref"] == pytest.approx(100.1) and entry["position_id"] == "P1"
    assert entry["slip_bps"] == pytest.approx((100.2 - 100.1) / 100.1 * 1e4)
    assert close["kind"] == "CLOSE" and close["ref"] == pytest.approx(99.95) and close["position_id"] == "P1"
    assert close["slip_bps"] == pytest.approx((99.95 - 99.90) / 99.95 * 1e4)
    await app.stop()

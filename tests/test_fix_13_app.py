"""Regression tests for the orchestrator fixes in heartless/app.py (round 13).

Everything runs offline against a fake Binance REST and a stubbed user stream. Covered:
  * the owner's /kill or /pause and the paper wallet survive a process restart (and apply to engines created later)
  * /mode live is atomic: a failure in LiveAccount.start() or TradingEngine.start() publishes nothing and is reported
  * /mode paper keeps the live engine (and mode) while anything is still open on the engine or on the exchange
  * restored paper positions get their simulated SL/TP1/TP back; interrupted (CLOSING) rows are recovered or booked
  * a bar that arrives through the gap fill rebuilds the 5m/15m/1h frames it completed
  * HEARTLESS_MODE precedence: a changed .env wins, an unchanged .env never undoes the persisted /mode live
  * challengers keep their slot until they had a fair trial; a failed first backfill is retried and repaired
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from heartless.app import CHALLENGER_MIN_AGE_MS, Heartless
from heartless.config import Settings
from heartless.core.models import Position, PositionStatus, Side, Ticker
from heartless.exchange.binance_rest import BinanceError
from heartless.exchange.paper import PaperPosition
from heartless.util.timeutil import MS_MINUTE, now_ms
from synth import synth_candles

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
N_HIST = 1500


def _exchange_info():
    return {"symbols": [{"symbol": s, "baseAsset": s[:-4], "quoteAsset": "USDT", "pricePrecision": 2, "quantityPrecision": 3,
                         "status": "TRADING", "contractType": "PERPETUAL", "onboardDate": 1,
                         "filters": [{"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                                     {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                                     {"filterType": "MIN_NOTIONAL", "notional": "5"}]} for s in SYMS]}


class FakeRest:
    """Public market-data surface plus the private endpoints LiveAccount / reconcile touch; behaviour is scripted."""

    def __init__(self, candles):
        self.candles = candles
        self.used_weight = 0
        self.ws_base = "wss://fake"
        self.time_offset = 0
        self.fail_klines: dict[str, int] = {}  # symbol -> remaining klines_range calls that raise
        self.account_script: list = []  # per account() call: an exception to raise, or None
        self.order_script: list = []  # per new_order() call: an exception to raise, or None (-> FILLED)
        self.positions: list[dict] = []  # position_risk rows
        self.algo_orders: list[dict] = []
        self.calls: list[str] = []

    async def sync_time(self): return None
    async def exchange_info(self): return _exchange_info()
    async def ticker_24h(self): return [{"symbol": s, "quoteVolume": "500000000"} for s in SYMS]

    async def klines_range(self, symbol, start, end, interval="1m"):
        if self.fail_klines.get(symbol, 0) > 0:
            self.fail_klines[symbol] -= 1
            raise RuntimeError("klines: 503 service unavailable")
        ca = self.candles[symbol]
        return [ca.candle_at(i) for i in range(ca.n) if start <= ca.open_time[i] <= end]

    async def funding_rate_history(self, symbol, start=None, end=None, limit=1000):
        return [{"fundingTime": t, "fundingRate": "0.0001", "markPrice": "100"} for t in range(start or 0, end or 0, 8 * 3600_000)]

    async def open_interest_hist(self, symbol, period="15m", limit=20): return []
    async def close(self): return None

    # --- private endpoints used by LiveAccount --------------------------------------------------
    async def get_position_mode(self): return False

    async def account(self):
        self.calls.append("account")
        step = self.account_script.pop(0) if self.account_script else None
        if isinstance(step, Exception):
            raise step
        return {"totalWalletBalance": "10000", "totalUnrealizedProfit": "0", "availableBalance": "10000"}

    async def position_risk(self, symbol=None): return list(self.positions)
    async def open_orders(self, symbol=None): return []
    async def open_algo_orders(self, symbol=None): return list(self.algo_orders)

    async def new_order(self, symbol, side, type_, quantity=None, **kw):
        self.calls.append("new_order")
        step = self.order_script.pop(0) if self.order_script else None
        if isinstance(step, Exception):
            raise step
        return {"orderId": len(self.calls), "clientOrderId": kw.get("client_id") or "", "status": "FILLED",
                "executedQty": str(quantity), "avgPrice": "100"}

    async def new_algo_order(self, symbol, side, type_, trigger_price, **kw):
        aid = f"A{len(self.calls)}"
        self.calls.append("new_algo_order")
        self.algo_orders.append({"algoId": aid, "symbol": symbol, "side": side, "orderType": type_, "triggerPrice": trigger_price})
        return {"algoId": aid}

    async def cancel_algo_order(self, symbol, algo_id=None, **kw):
        self.algo_orders = [a for a in self.algo_orders if a["algoId"] != str(algo_id)]
        return {}

    async def cancel_order(self, symbol, order_id=None, client_id=None): return {}
    async def query_order(self, symbol, order_id=None, client_id=None): raise BinanceError(-2013, "Order does not exist", 400)
    async def cancel_all_orders(self, symbol): return {}
    async def cancel_all_algo_orders(self, symbol): return {}
    async def create_listen_key(self): return "lk"
    async def close_listen_key(self): return None


class DummyStream:
    """Stand-in for the user data websocket: never connects, never leaves a task behind."""

    def __init__(self, rest, on_event):
        self.connected = False
        self.stopped = False

    async def run(self): return None

    async def stop(self): self.stopped = True


def _make_app(tmp: Path | None = None, candles=None, **env):
    tmp = tmp or Path(tempfile.mkdtemp())
    now = now_ms()
    start = (now - (N_HIST + 400) * MS_MINUTE) // MS_MINUTE * MS_MINUTE
    candles = candles or {s: synth_candles(N_HIST + 300, seed=i, start_ms=start, price=100 * (i + 1)) for i, s in enumerate(SYMS)}
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), HISTORY_DAYS=5, UNIVERSE_SIZE=3,
                        ALWAYS_INCLUDE="BTCUSDT,ETHUSDT,SOLUSDT", WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="", **env)
    settings.data_dir.mkdir(exist_ok=True)
    app = Heartless(settings)
    app.rest = FakeRest(dict(candles))
    return app, candles


async def _bootstrap(app, candles):
    for s in SYMS:  # only the first N_HIST bars exist for "backfill"; the rest arrive as live bars
        app.rest.candles[s] = candles[s].slice(0, N_HIST)
    await app._load_symbols()
    await app.refresh_universe(initial=True)
    await app._backfill(app.universe)
    app._build_views(app.universe)
    await app._create_engines()


def _position(sym: str, qty: float, status: PositionStatus, engine: str = "paper", entry: float = 100.0, stop: float = 99.0,
              tp: float | None = 102.0, tp1: float | None = 101.0) -> Position:
    return Position(id=f"P-{sym}-{status.value}", engine=engine, symbol=sym, side=Side.LONG, qty=qty, entry_price=entry,
                    entry_time=now_ms() - 10 * MS_MINUTE, stop=stop, take_profit=tp, tp1=tp1, initial_stop=stop,
                    alpha="trend_pullback", alphas=["trend_pullback"], reason="test", confidence=0.6, regime="TREND",
                    risk_amount=qty * (entry - stop), r_unit=entry - stop, notional=qty * entry, leverage=5,
                    params_version="v-test", atr=0.5, trail_atr_mult=2.0, status=status, filled_qty=qty, original_qty=qty,
                    extra={"tp1_frac": 0.5})


# --- kill switch / wallet flush across a restart ---------------------------------------------------------------
async def test_kill_switch_and_paper_wallet_survive_restart():
    tmp = Path(tempfile.mkdtemp())
    app, candles = _make_app(tmp)
    await _bootstrap(app, candles)
    await app.kill()
    assert app.store.get("paused") is True and app.engines["paper"].risk.state.paused
    app.paper_accounts["paper"].wallet = 9_876.5  # a fill the 30 s scheduler tick has not persisted yet
    await app.stop()

    app2, _ = _make_app(tmp, candles)
    await _bootstrap(app2, candles)
    eng = app2.engines["paper"]
    assert eng.risk.state.paused is True and "/resume" in eng.risk.state.halt_reason
    assert app2.status()["paused"] is True
    assert "일시정지" in app2.startup_summary()
    assert app2.paper_accounts["paper"].wallet == pytest.approx(9_876.5)
    # an engine created while the pause is in force inherits it too
    slot = await app2.install_challenger(app2.params.clone(note="t"), source="trend_pullback")
    assert slot is not None and app2.engines[slot.name].risk.state.paused is True
    await app2.resume()
    assert not eng.risk.state.paused and not app2.engines[slot.name].risk.state.paused
    assert app2.store.get("paused") is False and app2.store.get("halt_reason") == ""
    assert "일시정지" not in app2.startup_summary()
    await app2.stop()


# --- /mode live atomicity and /mode paper never orphaning a position -------------------------------------------
async def test_mode_switches_are_atomic_and_report_failures(monkeypatch):
    monkeypatch.setattr("heartless.exchange.live.UserStream", DummyStream)
    app, candles = _make_app(BINANCE_API_KEY="k", BINANCE_API_SECRET="s")
    await _bootstrap(app, candles)
    rest = app.rest

    # 1) LiveAccount.start() fails (bad key): nothing is published, nothing persisted, the owner gets the reason
    rest.account_script = [BinanceError(-2015, "Invalid API-key, IP, or permissions for action.", 401)]
    msg = await app.set_mode("live")
    assert "실패" in msg and "-2015" in msg
    assert app.mode == "paper" and app.live_account is None and "live" not in app.engines
    assert app.store.get("mode") is None and app.primary_engine.name == "paper" and app.engines["paper"].notify

    # 2) TradingEngine.start() fails (second account() call): same invariants, the half-started account is stopped
    rest.account_script = [None, BinanceError(-1003, "Too many requests", 429)]
    msg = await app.set_mode("live")
    assert "실패" in msg
    assert app.mode == "paper" and app.live_account is None and "live" not in app.engines and app.store.get("mode") is None

    # 3) a clean retry actually creates the live engine
    assert "라이브" in await app.set_mode("live")
    assert app.mode == "live" and "live" in app.engines and app.live_account is not None
    assert app.store.get("mode") == "live" and app.primary_engine.name == "live" and app.engines["paper"].notify is False

    # 4) live -> paper while a reduce-only close is rejected and the exchange still holds the position
    eng = app.engines["live"]
    eng.positions["BTCUSDT"] = _position("BTCUSDT", 0.01, PositionStatus.OPEN, engine="live")
    rest.positions = [{"symbol": "BTCUSDT", "positionAmt": "0.01", "entryPrice": "100", "unRealizedProfit": "0",
                       "leverage": "5", "markPrice": "100"}]
    rest.order_script = [BinanceError(-2022, "ReduceOnly Order is rejected.", 400)]
    msg = await app.set_mode("paper")
    assert "BTCUSDT" in msg and "라이브 모드 유지" in msg
    assert app.mode == "live" and app.engines.get("live") is eng and app.live_account is not None
    assert eng.entries_enabled is True and app.store.get("mode") == "live"
    assert eng.positions["BTCUSDT"].status is PositionStatus.OPEN and eng.positions["BTCUSDT"].sl_algo_id  # re-protected

    # 5) once the close fills and the exchange is flat the switch goes through
    rest.positions = []
    msg = await app.set_mode("paper")
    assert "페이퍼" in msg
    assert app.mode == "paper" and "live" not in app.engines and app.live_account is None
    assert app.store.get("mode") == "paper" and app.engines["paper"].notify is True
    assert eng.closed and eng.closed[-1].symbol == "BTCUSDT"
    await app.stop()


# --- paper restart: brackets re-armed, CLOSING rows recovered / booked -----------------------------------------
async def test_paper_restart_rearms_brackets_and_recovers_closing_rows():
    tmp = Path(tempfile.mkdtemp())
    app, candles = _make_app(tmp)
    await _bootstrap(app, candles)
    eng, acc = app.engines["paper"], app.paper_accounts["paper"]
    # an OPEN position with SL / TP1 / TP in the simulator, exactly as a filled entry leaves it
    pos = _position("BTCUSDT", 0.1, PositionStatus.OPEN)
    eng.positions["BTCUSDT"] = pos
    acc.positions["BTCUSDT"] = PaperPosition(qty=0.1, entry=100.0, leverage=5)
    await eng._place_brackets(pos)
    eng._save(pos)
    old_ids = {pos.sl_algo_id, pos.tp_algo_id, pos.extra["tp1_algo_id"]}
    assert len(old_ids) == 3 and all(old_ids)
    # a close interrupted mid-flight: persisted CLOSING, quantity still held by the simulator
    closing = _position("ETHUSDT", 0.2, PositionStatus.CLOSING, entry=200.0, stop=198.0, tp=204.0, tp1=None)
    closing.exit_reason = "시간 초과 청산(time stop)"
    app.store.save_position(closing)
    # a CLOSING row whose quantity already left the simulator (the exit fill was booked, _finalize never ran)
    app.store.save_position(_position("SOLUSDT", 0.0, PositionStatus.CLOSING, entry=300.0, stop=297.0, tp=306.0, tp1=None))
    await app.stop()

    app2, _ = _make_app(tmp, candles)
    await _bootstrap(app2, candles)
    eng2, acc2 = app2.engines["paper"], app2.paper_accounts["paper"]
    live_algos = {a.algo_id: a for a in acc2.algos.values() if a.status == "NEW"}
    btc = eng2.positions["BTCUSDT"]
    new_ids = {btc.sl_algo_id, btc.tp_algo_id, btc.extra["tp1_algo_id"]}
    assert btc.status is PositionStatus.OPEN and len(new_ids) == 3 and new_ids <= set(live_algos)
    assert sorted(a.kind for a in live_algos.values() if a.symbol == "BTCUSDT") == ["STOP_MARKET", "TAKE_PROFIT_MARKET", "TAKE_PROFIT_MARKET"]
    # the fresh simulator numbers algos from 1 again (ids can repeat across processes): re-arming one restored
    # position must never cancel the brackets just placed for another one
    eth = eng2.positions["ETHUSDT"]
    assert eth.status is PositionStatus.OPEN and eth.sl_algo_id in live_algos and eth.tp_algo_id in live_algos
    assert sorted(a.kind for a in live_algos.values() if a.symbol == "ETHUSDT") == ["STOP_MARKET", "TAKE_PROFIT_MARKET"]
    assert len(live_algos) == 5
    assert acc2.positions["ETHUSDT"].qty == pytest.approx(0.2)
    # the rows carry the new ids / status, so a further restart does not lose them again
    rows = {p.symbol: p for p in app2.store.load_open_positions("paper")}
    assert rows["BTCUSDT"].status is PositionStatus.OPEN and rows["BTCUSDT"].sl_algo_id == btc.sl_algo_id
    assert rows["ETHUSDT"].status is PositionStatus.OPEN and rows["SOLUSDT"].status is PositionStatus.CLOSING
    # the flat CLOSING row is booked by the reconcile pass and stops occupying its symbol / a max_positions slot
    assert "SOLUSDT" in eng2.positions and len(eng2.open_positions()) == 3
    await app2._paper_reconcile()
    assert "SOLUSDT" not in eng2.positions and "SOLUSDT" not in {p.symbol for p in app2.store.load_open_positions("paper")}
    assert eng2.closed[-1].symbol == "SOLUSDT"
    # a mark through the take-profit closes the restored position at TP (not via the software backstop)
    await app2._tick("BTCUSDT", Ticker("BTCUSDT", bid=102.49, ask=102.51, mark=102.5, last=102.5, ts=now_ms()))
    assert "BTCUSDT" not in eng2.positions
    assert eng2.closed[-1].symbol == "BTCUSDT" and eng2.closed[-1].exit_reason == "익절(TP)"
    assert acc2.positions["BTCUSDT"].qty == 0
    await app2.stop()


# --- gap-filled bars rebuild the higher timeframes they complete -----------------------------------------------
async def test_gap_filled_bars_rebuild_higher_timeframes():
    app, candles = _make_app()
    await _bootstrap(app, candles)
    full = candles["BTCUSDT"]
    app.rest.candles["BTCUSDT"] = full  # the gap fill can fetch the skipped bar
    i59 = next(i for i in range(N_HIST, full.n - 2) if (int(full.open_time[i]) // MS_MINUTE) % 60 == 59)
    for i in range(N_HIST, i59):
        await app._process_bar("BTCUSDT", full.candle_at(i))
    view = app.views["BTCUSDT"]
    hour_close = int(full.close_time[i59])
    assert int(view.frames["1h"].close_time[-1]) < hour_close
    # the stream skipped the :59 bar; the :00 bar arrives and the gap fill brings :59 in with it
    await app._process_bar("BTCUSDT", full.candle_at(i59 + 1))
    assert app.candles["BTCUSDT"].n == i59 + 2
    assert int(view.frames["1h"].close_time[-1]) == hour_close
    assert int(view.frames["15m"].close_time[-1]) == hour_close
    assert int(view.frames["5m"].close_time[-1]) == hour_close
    assert view.tf("1h").close_time == hour_close and "BTCUSDT" in app.regimes
    await app.stop()


# --- HEARTLESS_MODE precedence -------------------------------------------------------------------------------------
async def test_env_mode_change_wins_but_unchanged_env_keeps_persisted_mode():
    tmp = Path(tempfile.mkdtemp())

    async def start(**env) -> Heartless:
        s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp), WEB_ENABLED=False, TELEGRAM_BOT_TOKEN="",
                     BINANCE_API_KEY="k", BINANCE_API_SECRET="s", **env)
        s.data_dir.mkdir(exist_ok=True)
        return Heartless(s)

    async def dispose(app: Heartless) -> None:
        await app.rest.close()
        app.store.close()

    # fresh install with the shipped .env (HEARTLESS_MODE=paper); later the owner goes live via Telegram
    a = await start(HEARTLESS_MODE="paper")
    assert a.mode == "paper" and a.mode_notice == ""
    a.store.set("mode", "live")
    await dispose(a)
    # crash-restart with the unchanged .env: the persisted live mode is kept (its positions stay managed), with a warning
    b = await start(HEARTLESS_MODE="paper")
    assert b.mode == "live" and b.store.get("mode") == "live" and "무시" in b.mode_notice
    await dispose(b)
    # a database from before this rule (no record of the previous env) behaves the same
    c = await start(HEARTLESS_MODE="paper")
    c.store.execute("DELETE FROM kv WHERE key='mode.env'")
    await dispose(c)
    c = await start(HEARTLESS_MODE="paper")
    assert c.mode == "live" and "무시" in c.mode_notice
    await dispose(c)
    # env unset: no disagreement at all
    d = await start()
    assert d.mode == "live" and d.mode_notice == ""
    await dispose(d)
    # the operator now sets HEARTLESS_MODE=paper to stop live trading: a changed env wins and is persisted
    e = await start(HEARTLESS_MODE="paper")
    assert e.mode == "paper" and e.store.get("mode") == "paper" and "변경" in e.mode_notice
    assert "변경" in e.startup_summary()
    await dispose(e)
    # ... and an explicit change back to live is honoured as well
    f = await start(HEARTLESS_MODE="live")
    assert f.mode == "live" and f.store.get("mode") == "live"
    await dispose(f)


# --- challenger eviction needs a fair trial ---------------------------------------------------------------------
async def test_challenger_eviction_requires_a_fair_trial():
    app, candles = _make_app()
    await _bootstrap(app, candles)
    p = app.params
    s1 = await app.install_challenger(p.clone(note="c1"), source="trend_pullback")
    s2 = await app.install_challenger(p.clone(note="c2"), source="mean_reversion")
    assert {s1.name, s2.name} == {"challenger-1", "challenger-2"}
    # both slots are fresh (no trades, age ~0): a third candidate must not evict either of them
    assert await app.install_challenger(p.clone(note="c3"), source="breakout") is None
    assert app.challengers == [s1, s2] and {s1.name, s2.name} <= set(app.engines)
    # once a challenger has had its trial period it may be replaced; the younger one keeps its slot
    s1.started -= CHALLENGER_MIN_AGE_MS + 1
    s3 = await app.install_challenger(p.clone(note="c3"), source="breakout")
    assert s3 is not None and s3.name == s1.name
    assert s1 not in app.challengers and s2 in app.challengers and s3 in app.challengers
    await app.stop()


# --- a failed first backfill is retried and repaired --------------------------------------------------------------
async def test_failed_backfill_is_retried_and_repairs_history():
    app, candles = _make_app()
    for s in SYMS:
        app.rest.candles[s] = candles[s].slice(0, N_HIST)
    app.rest.fail_klines = {"SOLUSDT": 1}
    await app._load_symbols()
    await app.refresh_universe(initial=True)
    assert await app._backfill(app.universe) == {"SOLUSDT"}
    assert app._backfill_failed == {"SOLUSDT"}
    app._build_views(app.universe)
    await app._create_engines()
    assert app.candles["SOLUSDT"].n == 0 and app.candles["BTCUSDT"].n == N_HIST
    app._update_health(now_ms())
    assert app.health["backfill_pending"] == ["SOLUSDT"] and "SOLUSDT" in app.health["short_history"]
    # live bars start arriving before the retry, so the store holds only a few recent bars for the symbol
    for i in range(N_HIST, N_HIST + 3):
        await app._process_bar("SOLUSDT", candles["SOLUSDT"].candle_at(i))
    assert app.candles["SOLUSDT"].n == 3
    await app._retry_backfill()
    assert app._backfill_failed == set()
    assert app.candles["SOLUSDT"].n == N_HIST + 3 and app.views["SOLUSDT"].frames["1m"].n > 1000
    app._update_health(now_ms())
    assert app.health["backfill_pending"] == [] and "SOLUSDT" not in app.health["short_history"]
    await app.stop()


# --- startup: which alphas trade ------------------------------------------------------------------------------
def test_startup_summary_names_the_enabled_alphas_or_warns_when_none():
    app, _ = _make_app()
    active = app.enabled_alphas()
    assert active == [a for a, on in app.params.enabled.items() if on]
    assert f"활성 알파: {', '.join(active)}" in app.startup_summary()
    for a in app.params.enabled:
        app.params.enabled[a] = False
    assert app.enabled_alphas() == [] and "활성 알파 없음" in app.startup_summary()

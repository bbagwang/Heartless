"""F-pre-B execution modes (ROADMAP 7.7.2): the conservative maker fill model of PaperAccount in bar mode and tick mode
(trade-through, never in the placement bar, the 20x volume rule, stress), and the engine's maker_m1 / maker_m2 entries
(limit price, validity, sizing for the limit, no market fallback, alpha limits kept) with taker unchanged."""
import asyncio

import pytest

from heartless.config import EXEC_MODES, Settings, parse_exec_modes
from heartless.core.models import Candle, EntryStyle, Side
from heartless.exchange.paper import PaperAccount, maker_eligible_from
from heartless.exchange.symbols import norm_price
from heartless.learning import trials
from heartless.learning.backtester import BarPaperAccount
from synth import synth_candles, synth_symbols
from tests.test_fpreB_support import ALPHA, Scripted, backtest

MIN = 60_000
T0 = 1_700_006_400_000


def bar(k: int, o: float, h: float, lo: float, c: float, qv: float = 1e9) -> Candle:
    t = T0 + k * MIN
    return Candle(t, o, h, lo, c, qv / c, qv, 10, qv / c / 2, t + MIN - 1)


def _account(cls=PaperAccount, **kw):
    clock = {"t": T0 + MIN - 1}
    acc = cls(clock=lambda: clock["t"], **kw)
    acc.set_symbols(synth_symbols(["XUSDT"])) if hasattr(acc, "set_symbols") else None
    return acc, clock


def run(coro):
    return asyncio.run(coro)


def test_eligible_from_skips_the_placement_bar():
    # decided at a bar's close (close_time = open + 59,999): the next bar is the placement bar
    assert maker_eligible_from(T0 + MIN - 1) == T0 + 2 * MIN
    # sent a second into a bar (live): the bar after it
    assert maker_eligible_from(T0 + MIN + 1000) == T0 + 2 * MIN


async def _bar_mode(limit=99.0, through=0.5, qv_ok=True, mult=20.0, fill_rule="maker"):
    acc, clock = _account(maker_through=through, maker_volume_mult=mult)
    await acc.on_bar("XUSDT", bar(0, 100, 100.2, 99.9, 100))
    fills = []
    acc.on_fill(lambda f: fills.append(f) or asyncio.sleep(0))
    res = await acc.limit_order("XUSDT", "BUY", 1.0, limit, post_only=True, client_id="E1", fill_rule=fill_rule)
    assert res.status == "NEW"
    deep = limit * (1 - 0.0005)  # well through the limit
    await acc.on_bar("XUSDT", bar(1, 100, 100, deep, 99.5))  # the placement bar
    await acc.on_bar("XUSDT", bar(2, 99.5, 99.6, deep, 99.2, qv=1e9 if qv_ok else 20 * limit - 1))
    await acc.on_bar("XUSDT", bar(3, 99.2, 99.3, deep, 99.1))
    return acc, fills


def test_bar_mode_maker_fill_rules():
    acc, fills = run(_bar_mode())
    assert len(fills) == 1 and fills[0].ts == T0 + 2 * MIN and fills[0].maker and fills[0].price == 99.0
    assert fills[0].fee == pytest.approx(99.0 * 0.0002)
    # the bar after placement lacks volume (quote volume < 20 x notional): the next one fills
    acc, fills = run(_bar_mode(qv_ok=False))
    assert len(fills) == 1 and fills[0].ts == T0 + 3 * MIN
    # a plain post-only limit (taker mode, an alpha's own limit) keeps the old rule: it may fill in the next bar
    acc, fills = run(_bar_mode(fill_rule=""))
    assert len(fills) == 1 and fills[0].ts == T0 + MIN


def test_trade_through_by_half_or_the_whole_spread():
    async def go(low, through):
        acc, clock = _account(maker_through=through)
        await acc.on_bar("XUSDT", bar(0, 100, 100.2, 99.9, 100))
        fills = []
        acc.on_fill(lambda f: fills.append(f) or asyncio.sleep(0))
        await acc.limit_order("XUSDT", "BUY", 1.0, 99.0, client_id="E1", fill_rule="maker")
        for k in (1, 2):
            await acc.on_bar("XUSDT", bar(k, 99.5, 99.6, low, 99.4))
        return fills

    half, full = 0.5e-4, 1e-4
    assert not run(go(99.0, 0.5))  # a touch does not fill
    assert run(go(99.0 / (1 + half) - 1e-6, 0.5))  # through by half the 1 bp spread
    assert not run(go(99.0 / (1 + half) - 1e-6, 1.0))  # stress needs the whole spread
    assert run(go(99.0 / (1 + full) - 1e-6, 1.0))


def test_tick_mode_settles_maker_orders_at_bar_close_like_bar_mode():
    from heartless.core.models import Ticker

    async def go():
        acc, clock = _account()
        acc.tickers["XUSDT"] = Ticker("XUSDT", bid=99.99, ask=100.01, mark=100.0, last=100.0, ts=T0 + MIN - 1)
        fills = []
        acc.on_fill(lambda f: fills.append(f) or asyncio.sleep(0))
        await acc.limit_order("XUSDT", "SELL", 1.0, 101.0, client_id="E1", fill_rule="maker")
        # ticks through the limit never fill a maker order in tick mode
        await acc.on_ticker(Ticker("XUSDT", bid=101.5, ask=101.6, mark=101.5, last=101.5, ts=T0 + 2 * MIN + 5))
        assert not fills
        hi = 101.0 * (1 + 0.5e-4) + 0.01
        assert await acc.on_bar_close("XUSDT", bar(1, 100, hi, 100, 100.5)) == 0  # the placement bar
        assert await acc.on_bar_close("XUSDT", bar(2, 100.5, hi, 100.4, 100.6, qv=10.0)) == 0  # volume
        assert await acc.on_bar_close("XUSDT", bar(3, 100.6, hi, 100.5, 100.7)) == 1
        return fills

    fills = run(go())
    assert fills[0].price == 101.0 and fills[0].maker and fills[0].ts == T0 + 4 * MIN - 1


# --- engine ----------------------------------------------------------------------------------------------------------
def test_exec_mode_setting():
    assert EXEC_MODES == ("taker", "maker_m1", "maker_m2")
    assert parse_exec_modes("maker_m2") == ("maker_m2", {})
    assert parse_exec_modes("htf_trend:maker_m1, ts_trend_tsmom:maker_m2") == (
        "taker", {"htf_trend": "maker_m1", "ts_trend_tsmom": "maker_m2"})
    with pytest.raises(ValueError):
        parse_exec_modes("htf_trend:maker")
    with pytest.raises(ValueError):
        Settings(_env_file=None, EXEC_MODE="post_only")
    # the trial ledger hashes the mode under `exec` (taker = the old hash)
    p = {"enabled": {"htf_trend": True}, "alphas": {"htf_trend": {}}, "ensemble": {}}
    taker = trials.config_hash(trials.lab_config(p, "htf_trend", ["BTCUSDT"]))
    maker = trials.config_hash(trials.lab_config(p, "htf_trend", ["BTCUSDT"], exec="maker_m1"))
    assert taker == trials.config_hash(trials.lab_config(p, "htf_trend", ["BTCUSDT"], exec="taker")) != maker


def _limit_calls(monkeypatch):
    calls = []
    orig = PaperAccount.limit_order

    async def spy(self, symbol, side, qty, price, post_only=True, reduce_only=False, client_id="", fill_rule=""):
        res = await orig(self, symbol, side, qty, price, post_only, reduce_only, client_id, fill_rule)
        calls.append({"symbol": symbol, "side": side, "qty": qty, "price": price, "rule": fill_rule, "status": res.status,
                      "t": self.now()})
        return res

    monkeypatch.setattr(PaperAccount, "limit_order", spy)
    return calls


def _decisions(ca, tf_min=15, first=900, n=12, every=4):
    ct = ca.view("close_time")
    picks = [k for k in range(first, ca.n - 400) if (int(ct[k]) + 1) % (tf_min * MIN) == 0][::every][:n]
    return [int(ct[k]) for k in picks]


def test_maker_m1_rests_at_the_decision_close_and_is_cancelled_after_one_bar(monkeypatch, tmp_path):
    ca = synth_candles(4000, seed=21, price=2000.0)
    times = _decisions(ca)
    calls = _limit_calls(monkeypatch)
    res, _ = backtest(monkeypatch, tmp_path, Scripted(times, tf="15m", hold=60), {"ETHUSDT": ca}, shock=None,
                      EXEC_MODE="maker_m1")
    ct, close = ca.view("close_time").tolist(), ca.view("close").tolist()
    info = synth_symbols(["ETHUSDT"])["ETHUSDT"]
    assert len(calls) == len(times) and all(c["rule"] == "maker" for c in calls)
    for c, t in zip(calls, times):
        assert c["t"] == t and c["price"] == norm_price(info, close[ct.index(t)], "BUY")
    assert res.trades and len(res.trades) < len(times)  # some never traded through within 15 minutes
    for tr in res.trades:
        t = max(x for x in times if x < tr["entry_time"])
        # not in the placement bar (opening at t + 1), and within the validity: the 15th bar after it is the last
        assert t + 1 + MIN <= tr["entry_time"] <= t + 1 + 14 * MIN
        assert tr["entry_price"] == norm_price(info, close[ct.index(t)], "BUY")
        # maker fee on the entry (2 bp), taker on the time exit (5 bp)
        exit_notional = tr["notional"] / tr["entry_price"] * tr["exit_price"]
        assert tr["fees"] == pytest.approx(tr["notional"] * 0.0002 + exit_notional * 0.0005, rel=1e-6)


def test_maker_m2_retest_limit_validity_and_stop_rule(monkeypatch, tmp_path):
    ca = synth_candles(4000, seed=22, price=2000.0)
    times = _decisions(ca, tf_min=60, every=2, n=6)
    calls = _limit_calls(monkeypatch)
    cancels = []
    from heartless.execution.engine import TradingEngine

    orig = TradingEngine._cancel_pending

    async def spy(self, pos, reason):
        cancels.append((self.clock() - pos.pending_since, reason))
        await orig(self, pos, reason)

    monkeypatch.setattr(TradingEngine, "_cancel_pending", spy)
    res, _ = backtest(monkeypatch, tmp_path, Scripted(times, tf="1h", hold=60, side=Side.SHORT), {"ETHUSDT": ca},
                      shock=None, EXEC_MODE="maker_m2")
    ct, close = ca.view("close_time").tolist(), ca.view("close").tolist()
    info = synth_symbols(["ETHUSDT"])["ETHUSDT"]
    assert calls and {c["t"] for c in calls} <= set(times)
    for c in calls:
        px = close[ct.index(c["t"])]
        assert c["side"] == "SELL" and c["price"] == norm_price(info, px + 0.25 * 0.004 * px, "SELL")
    # 3 decision bars of 1h, capped at 240 minutes: cancelled 180 minutes after placement when unfilled
    ttl = [age for age, why in cancels if "유효 시간" in why]
    assert ttl and all(age == 180 * MIN for age in ttl)
    # a retest limit beyond the stop drops the signal (no order at all)
    calls.clear()
    res2, _ = backtest(monkeypatch, tmp_path, Scripted(times, tf="1h", stop_atr=0.2), {"ETHUSDT": ca}, shock=None,
                       EXEC_MODE="maker_m2")
    assert not calls and res2.skipped.get("maker") == len(times) and not res2.trades


def test_alpha_limits_keep_their_price_and_crossing_limits_are_dropped(monkeypatch, tmp_path):
    ca = synth_candles(4000, seed=23, price=2000.0)
    times = _decisions(ca, n=6)
    calls = _limit_calls(monkeypatch)
    backtest(monkeypatch, tmp_path, Scripted(times, tf="15m", style=EntryStyle.LIMIT, limit_off=0.5, ttl=8),
             {"ETHUSDT": ca}, shock=None, EXEC_MODE=f"{ALPHA}:maker_m2")
    ct, close = ca.view("close_time").tolist(), ca.view("close").tolist()
    info = synth_symbols(["ETHUSDT"])["ETHUSDT"]
    assert calls and {c["t"] for c in calls} <= set(times)
    for c in calls:
        px = close[ct.index(c["t"])]
        assert c["price"] == norm_price(info, px - 0.5 * 0.004 * px, "BUY")  # the alpha's limit, not close - 0.25 ATR
    # a limit through the book would cross: dropped, never sent as a market order
    calls.clear()
    markets = []
    orig = BarPaperAccount.market_order

    async def mspy(self, symbol, side, qty, reduce_only=False, client_id=""):
        if not reduce_only:
            markets.append(symbol)
        return await orig(self, symbol, side, qty, reduce_only, client_id)

    monkeypatch.setattr(BarPaperAccount, "market_order", mspy)
    res, _ = backtest(monkeypatch, tmp_path, Scripted(times, tf="15m", style=EntryStyle.LIMIT, limit_off=-0.5, ttl=8),
                      {"ETHUSDT": ca}, shock=None, EXEC_MODE="maker_m1")
    assert calls and all(c["status"] == "EXPIRED" for c in calls) and not markets and not res.trades
    assert res.skipped.get("maker cross") == len(calls)


def test_taker_mode_sends_no_maker_orders(monkeypatch, tmp_path):
    ca = synth_candles(4000, seed=24, price=2000.0)
    times = _decisions(ca, n=6)
    calls = _limit_calls(monkeypatch)
    res, _ = backtest(monkeypatch, tmp_path, Scripted(times, tf="15m"), {"ETHUSDT": ca}, shock=None)
    assert not calls and len(res.trades) == len(times)
    assert all(t["entry_time"] == x + 1 for t, x in zip(sorted(res.trades, key=lambda t: t["entry_time"]), times))

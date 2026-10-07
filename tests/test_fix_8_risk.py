"""Regression tests for heartless/strategy/risk.py fixes (round 8):
1. RiskState persistence (daily halt / DD pause / cool-downs / owner pause survive a restart),
2. one-shot max-drawdown pause (/resume is not undone on the next equity tick),
3. min-notional bump bounded by max_risk_per_trade_pct (incl. costs),
4. weekly anchor rolls over on Monday 00:00 UTC.
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from heartless.config import Settings
from heartless.core.models import Decision, EntryStyle, Regime, Side, Signal, SymbolInfo
from heartless.core.store import Store
from heartless.strategy.risk import RiskManager
from heartless.util.timeutil import MS_DAY


def _settings():
    return Settings(_env_file=None, RISK_PER_TRADE_PCT=0.5, MAX_POSITION_LEVERAGE=5, EXCHANGE_LEVERAGE=10)


def _decision(stop=98.0, entry=100.0, size_mult=1.0):
    sig = Signal("a", "BTCUSDT", Side.LONG, 0.7, "r", stop, 104.0, 102.0, EntryStyle.MARKET, None, 0, 0.0, 1.0, "5m",
                 {"ref_price": entry})
    return Decision("BTCUSDT", Side.LONG, 0.7, 0.7, ["a"], sig, "r", Regime.RANGE, size_mult, 2.0)


def _ms(*ymdhms) -> int:
    return int(datetime(*ymdhms, tzinfo=timezone.utc).timestamp() * 1000)


def _store() -> Store:
    return Store(Path(tempfile.mkdtemp()) / "risk.db")


T0 = _ms(2026, 10, 7, 10, 0, 0)  # Wednesday 10:00 UTC


# --- 1. persistence ----------------------------------------------------------------------------
def test_daily_halt_survives_reconstruction_from_store():
    s = _settings()
    store = _store()
    rm = RiskManager(s, store, "paper")
    rm.observe_equity(10_000, T0)
    events = rm.observe_equity(9_650, T0 + 60_000)  # -3.5% > daily_loss_limit_pct (3%)
    assert events and rm.state.halted_until == T0 - T0 % MS_DAY + MS_DAY
    assert store.get("paper.risk")["halted_until"] == rm.state.halted_until  # written through immediately

    # "process restart": a fresh RiskManager over the same store, first observe_equity at the drawn-down equity
    rm2 = RiskManager(s, store, "paper")
    assert rm2.observe_equity(9_650, T0 + 120_000) == []  # no duplicate halt event
    assert rm2.state.halted_until == rm.state.halted_until
    assert rm2.state.day_start_equity == 10_000  # day anchor NOT re-based to the drawn-down equity
    ok, why = rm2.can_open(_decision(), [], T0 + 130_000, 9_650)
    assert not ok and why.startswith("halted:")

    # the halt still expires naturally after UTC midnight
    rm2.observe_equity(9_650, T0 + MS_DAY)
    assert rm2.state.halted_until == 0 and rm2.can_open(_decision(), [], T0 + MS_DAY, 9_650)[0]


def test_owner_pause_dd_pause_cooldowns_and_risk_scale_survive_restart():
    s = _settings()
    store = _store()
    rm = RiskManager(s, store, "paper")
    rm.observe_equity(10_000, T0)
    rm.observe_equity(9_200, T0 + MS_DAY)  # next day: -8% on the week -> risk_scale 0.5
    assert rm.state.risk_scale == 0.5
    rm.register_loss("ETHUSDT", T0 + MS_DAY)
    rm.register_market_shock(T0 + MS_DAY, 15)

    rm2 = RiskManager(s, store, "paper")
    assert rm2.state.risk_scale == 0.5
    assert rm2.state.symbol_cooldown == {"ETHUSDT": rm.state.symbol_cooldown["ETHUSDT"]}
    assert rm2.state.shock_until == rm.state.shock_until
    assert rm2.state.week_start_equity == 10_000 and rm2.state.peak_equity == 10_000
    assert rm2.observe_equity(9_200, T0 + MS_DAY + 30_000) == [] and rm2.state.risk_scale == 0.5

    # the owner's /pause is stored by app.pause() under the global "paused" key and must be honoured on restart
    store.set("paused", True)
    rm3 = RiskManager(s, store, "paper")
    assert rm3.state.paused and rm3.can_open(_decision(), [], T0 + MS_DAY + 60_000, 9_200) == (False, "paused")

    # a max-drawdown pause is persisted inside the per-engine blob even when the global flag is off
    store.set("paused", False)
    rm4 = RiskManager(s, store, "paper")
    assert not rm4.state.paused
    rm4.observe_equity(8_400, T0 + MS_DAY + 90_000)
    assert rm4.state.paused and store.get("paper.risk")["paused"] is True
    assert RiskManager(s, store, "paper").state.paused


def test_dict_roundtrip_ignores_unknown_and_malformed_keys():
    s = _settings()
    rm = RiskManager(s)
    rm.observe_equity(10_000, T0)
    rm.observe_equity(9_650, T0 + 60_000)
    rm.register_loss("BTCUSDT", T0 + 60_000)
    d = rm.to_dict()
    assert d == rm.dump() and isinstance(d["symbol_cooldown"], dict)

    rm2 = RiskManager(s)
    rm2.load(d)
    assert rm2.state == rm.state
    rm3 = RiskManager(s)
    rm3.restore(d)
    assert rm3.state == rm.state

    rm2.restore({"unknown_field": 1, "halted_until": "not-a-number", "risk_scale": "0.5", "symbol_cooldown": None})
    assert rm2.state.halted_until == rm.state.halted_until  # malformed value skipped
    assert rm2.state.risk_scale == 0.5 and rm2.state.symbol_cooldown == {}
    rm2.restore(None)  # tolerated
    rm2.load(None)  # no store attached -> no-op


def test_no_store_means_no_persistence_side_effects():
    rm = RiskManager(_settings())
    rm.observe_equity(10_000, T0)
    rm.save()  # no store -> nothing to do, no error
    rm.pause("x")
    rm.resume()
    assert not rm.state.paused


# --- 2. one-shot max-drawdown pause ------------------------------------------------------------
def test_dd_pause_is_one_shot_after_owner_resume():
    s = _settings()
    rm = RiskManager(s)
    rm.observe_equity(10_000, T0)
    t1 = T0 + 2 * MS_DAY  # later day so the daily limit is not involved
    events = rm.observe_equity(8_400, t1)  # -16% from peak -> paused
    assert rm.state.paused and any("고점" in e for e in events)

    # what app.resume() currently does: poke the two fields
    rm.state.paused = False
    rm.state.halt_reason = ""
    assert rm.observe_equity(8_400, t1 + 30_000) == []  # NOT re-paused, no new event
    assert not rm.state.paused
    assert rm.observe_equity(8_300, t1 + 60_000) == [] and not rm.state.paused

    # re-arms only after a new equity high
    rm.observe_equity(10_500, t1 + 90_000)
    events = rm.observe_equity(8_900, t1 + 120_000)  # -15.2% from the new peak
    assert rm.state.paused and any("고점" in e for e in events)


def test_resume_with_equity_rebases_the_drawdown_peak():
    s = _settings()
    rm = RiskManager(s)
    rm.observe_equity(10_000, T0)
    t1 = T0 + 2 * MS_DAY
    rm.observe_equity(8_400, t1)
    assert rm.state.paused
    rm.resume(8_400)
    assert not rm.state.paused and rm.state.halt_reason == "" and rm.state.peak_equity == 8_400
    assert rm.observe_equity(8_400, t1 + 30_000) == [] and not rm.state.paused
    # -13.1% from the new base: the (independent) daily loss halt fires, but the drawdown pause does not
    events = rm.observe_equity(7_300, t1 + 60_000)
    assert not any("고점" in e for e in events) and not rm.state.paused
    events = rm.observe_equity(7_100, t1 + 90_000)  # -15.5% from the new base -> fires again
    assert rm.state.paused and any("고점" in e for e in events)


# --- 3. min-notional bump bounded by the hard cap ---------------------------------------------
def test_min_notional_bump_never_exceeds_max_risk_per_trade():
    s = _settings()
    rm = RiskManager(s)
    info = SymbolInfo("BTCUSDT", "BTC", "USDT", 0.1, 0.001, 0.001, 100, 5, 1, 3)  # minNotional 100
    # equity 150, stop 3% away, size_mult 2.5 -> risk_pct capped at 1.25% (1.875 USDT); the 1.0 minimum would
    # lose 3.13 USDT (2.09%) at the stop -> must be refused
    res = rm.size(_decision(stop=97.0, size_mult=2.5), equity=150, info=info, entry_price=100.0)
    assert res.qty == 0 and res.reason == "below min notional"
    # a 1% stop: the minimum loses 1.13 USDT <= 1.875 hard cap -> bumped to the minimum
    res = rm.size(_decision(stop=99.0), equity=150, info=info, entry_price=100.0)
    assert res.qty == 1.0 and res.reason == ""
    assert res.risk_amount <= 150 * s.max_risk_per_trade_pct / 100
    assert abs(res.risk_pct - res.risk_amount / 150 * 100) < 1e-9  # reported risk reflects the bumped size
    # the 2.5x-budget slack still applies when it is the tighter bound (weekly 50% risk scale)
    rm.state.risk_scale = 0.5  # budget 0.375 USDT -> 2.5x = 0.9375 < 1.13 loss at the minimum
    res = rm.size(_decision(stop=99.0), equity=150, info=info, entry_price=100.0)
    assert res.qty == 0 and res.reason == "below min notional"


# --- 4. weekly anchor boundary -----------------------------------------------------------------
def test_week_key_rolls_over_on_monday_utc():
    rm = RiskManager(_settings())
    rm.observe_equity(1, _ms(2026, 10, 3, 23, 59, 59))  # Saturday
    k_sat = rm.state.week_key
    rm.observe_equity(1, _ms(2026, 10, 4, 0, 0, 0))  # Sunday 00:00
    k_sun = rm.state.week_key
    rm.observe_equity(1, _ms(2026, 10, 4, 23, 59, 59))  # Sunday 23:59:59
    k_sun_late = rm.state.week_key
    rm.observe_equity(1, _ms(2026, 10, 5, 0, 0, 0))  # Monday 00:00
    k_mon = rm.state.week_key
    assert k_sat == k_sun == k_sun_late
    assert k_mon != k_sun and k_mon == _ms(2026, 10, 5, 0, 0, 0)
    assert datetime.fromtimestamp(k_mon / 1000, timezone.utc).weekday() == 0


def test_weekly_loss_limit_covers_monday_to_sunday():
    s = _settings()
    rm = RiskManager(s)
    rm.observe_equity(10_000, _ms(2026, 9, 28, 0, 0, 0))  # Monday
    rm.observe_equity(9_310, _ms(2026, 10, 3, 12, 0, 0))  # Saturday: -6.9%, no breach
    assert rm.state.risk_scale == 1.0
    events = rm.observe_equity(9_110, _ms(2026, 10, 4, 12, 0, 0))  # Sunday: -8.9% Mon-Sun -> breach
    assert rm.state.risk_scale == 0.5 and any("주간" in e for e in events)
    rm.observe_equity(9_110, _ms(2026, 10, 5, 0, 0, 0))  # Monday: new week, scale back to 1
    assert rm.state.risk_scale == 1.0 and rm.state.week_start_equity == 9_110

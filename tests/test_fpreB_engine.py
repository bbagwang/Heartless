"""F-pre-B engine: the shock layer's entry halt in the engine and the backtester (S0 from the run's own BTC bars, a
given schedule, shock_exempt alphas, resting limits cancelled, open positions untouched) and the backtester's delisting
close (5x slippage, 10x under stress, never repriced again by the lab)."""
import numpy as np
import pytest

from heartless.core.models import EntryStyle
from heartless.execution.engine import TradingEngine
from heartless.learning import lab
from heartless.learning.backtester import DELIST_REASON, DELIST_SLIP_MULT, stored_delistings
from heartless.strategy import shock as SH
from heartless.strategy.alphas import ALPHA_BY_NAME
from synth import synth_candles
from tests.test_fpreB_support import ALPHA, Scripted, backtest

MIN, H = 60_000, 3_600_000


def _eth(n=3000):
    return synth_candles(n, seed=11, price=2000.0)


def _times(ca, first=600, every=60, n=30):
    ct = ca.view("close_time")
    return [int(ct[k]) for k in range(first, min(ca.n - 40, first + every * n), every)]


def _entries(res):
    """The decision close of every trade: a market entry fills at the next bar's open (close_time + 1)."""
    return sorted(int(t["entry_time"]) - 1 for t in res.trades)


def test_a_given_schedule_halts_new_entries_two_hours(monkeypatch, tmp_path):
    ca = _eth()
    times = _times(ca)
    base, _ = backtest(monkeypatch, tmp_path, Scripted(times), {"ETHUSDT": ca}, shock=None)
    assert len(base.trades) == len(times)
    fire = (times[5] + 1) // H * H + H  # an hour close between two decisions
    ev = SH.ShockEvents(hourly={"S2": [fire]})
    res, _ = backtest(monkeypatch, tmp_path, Scripted(times), {"ETHUSDT": ca}, shock=ev)
    halted = [t for t in times if fire <= t + 1 < fire + 2 * H]
    assert halted and res.skipped.get("shock halt") == len(halted)
    # every other decision trades exactly as before
    assert _entries(base) == times
    assert _entries(res) == [t for t in times if t not in halted]


def test_shock_exempt_alphas_are_not_halted(monkeypatch, tmp_path):
    ca = _eth()
    times = _times(ca)
    ev = SH.ShockEvents(hourly={"S1": [(times[3] + 1) // H * H + H]})
    base, _ = backtest(monkeypatch, tmp_path, Scripted(times), {"ETHUSDT": ca}, shock=None)
    monkeypatch.setattr(type(ALPHA_BY_NAME[ALPHA]), "shock_exempt", True)
    res, _ = backtest(monkeypatch, tmp_path, Scripted(times), {"ETHUSDT": ca}, shock=ev)
    assert _entries(res) == _entries(base) and not res.skipped.get("shock halt")


def test_resting_limits_are_cancelled_and_open_positions_untouched(monkeypatch, tmp_path):
    ca = _eth()
    ct = ca.view("close_time")
    t_lim, t_open = int(ct[700]), int(ct[900])
    reasons = []
    orig = TradingEngine._cancel_pending

    async def spy(self, pos, reason):
        reasons.append((self.clock(), reason))
        await orig(self, pos, reason)

    monkeypatch.setattr(TradingEngine, "_cancel_pending", spy)
    # a retest limit 5 ATR below the close (never reached; stop 8 ATR) resting for 120 bars; S0 fires 10 minutes later
    s0 = t_lim + 1 + 10 * MIN
    ev = SH.ShockEvents(s0=[s0])
    backtest(monkeypatch, tmp_path, Scripted([t_lim], style=EntryStyle.LIMIT, limit_off=5.0, ttl=120, stop_atr=8.0),
             {"ETHUSDT": ca}, shock=ev)
    assert reasons and reasons[0][0] == s0 - 1 and "shock S0" in reasons[0][1]
    # a position opened before an hourly firing keeps its time exit: same trade with and without the halt
    alpha = Scripted([t_open], hold=90)
    base, _ = backtest(monkeypatch, tmp_path, alpha, {"ETHUSDT": ca}, shock=None)
    ev2 = SH.ShockEvents(hourly={"S1": [(t_open + 1) // H * H + H]})
    res, _ = backtest(monkeypatch, tmp_path, Scripted([t_open], hold=90), {"ETHUSDT": ca}, shock=ev2)
    assert len(res.trades) == len(base.trades) == 1
    keep = ("entry_time", "exit_time", "exit_reason", "entry_price", "exit_price", "pnl")
    assert {k: res.trades[0][k] for k in keep} == {k: base.trades[0][k] for k in keep}


def test_s0_from_the_runs_own_btc_bars(monkeypatch, tmp_path):
    """SHOCK_LAYER on (the default) and no databases: S0 is computed from the backtest's BTCUSDT bars; a 4% gap halts
    every symbol's entries for 15 minutes from the bar that shows it."""
    btc, eth = synth_candles(3000, seed=5, price=30000.0), _eth()
    k = 1500
    for name in ("open", "high", "low", "close"):
        getattr(btc, name)[k:btc.n] *= 0.96
    ct = eth.view("close_time")
    times = [int(ct[j]) for j in range(k - 6, k + 30, 3)]
    script = dict(hold=1, symbols=("ETHUSDT",))
    res, bt = backtest(monkeypatch, tmp_path, Scripted(times, **script), {"BTCUSDT": btc, "ETHUSDT": eth})
    sched = bt._shock_events()
    assert sched.meta["missing"] == ["S3"] and sched.s0 == [int(ct[j]) + 1 for j in range(k, k + 5)]
    halted = [j for j in range(k - 10, k + 40) if sched.entry_halt(int(ct[j]))]
    assert halted == list(range(k, k + 4 + 15))  # the five S0 bars, then 15 minutes after the last of them
    blocked = {t for t in times if sched.entry_halt(t)}
    entered = set(_entries(res))
    assert blocked and not entered & blocked and res.skipped.get("shock halt") == len(blocked)
    assert entered == set(times) - blocked
    off, _ = backtest(monkeypatch, tmp_path, Scripted(times, **script), {"BTCUSDT": btc, "ETHUSDT": eth},
                      SHOCK_LAYER=False)
    assert not off.skipped.get("shock halt") and set(_entries(off)) == set(times)


def test_layer_off_or_none_is_the_old_backtest(monkeypatch, tmp_path):
    ca = _eth()
    times = _times(ca)
    a, bt = backtest(monkeypatch, tmp_path, Scripted(times), {"ETHUSDT": ca}, SHOCK_LAYER=False)
    assert bt._shock_events() is None
    b, _ = backtest(monkeypatch, tmp_path, Scripted(times), {"ETHUSDT": ca}, shock=None)
    keep = ("entry_time", "exit_time", "exit_reason", "entry_price", "exit_price", "pnl", "fees")
    assert [{k: t[k] for k in keep} for t in a.trades] == [{k: t[k] for k in keep} for t in b.trades]


# --- delisting close -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("slip", [1.5, 3.0])
def test_delisting_close_at_the_last_traded_hour(monkeypatch, tmp_path, slip):
    ca = _eth()
    ct = ca.view("close_time")
    t_in = int(ct[800])
    d = int(ct[[k for k in range(900, 1100) if (int(ct[k]) + 1) % H == 0][0]])  # an hour's last minute
    k = int(np.searchsorted(ct, d))
    ca.quote_volume[k + 1:ca.n] = 0.0  # after the delisting the archive's files are flat and empty
    later = [int(ct[k]) for k in range(1200, 1300, 20)]
    res, _ = backtest(monkeypatch, tmp_path, Scripted([t_in] + later, hold=600, stop_atr=40.0), {"ETHUSDT": ca}, shock=None,
                      delisted={"ETHUSDT": d}, BACKTEST_SLIPPAGE_BPS=slip)
    assert len(res.trades) == 1  # nothing after the delisting
    t = res.trades[0]
    assert t["exit_reason"] == DELIST_REASON == lab.DELIST_REASON and int(t["exit_time"]) == d
    close = float(ca.close[k])
    usual = min(slip + 0.3 * t["notional"] / 10_000.0 * (close / t["entry_price"]), 25.0)
    bps = usual + (DELIST_SLIP_MULT - 1) * slip
    assert t["exit_price"] == pytest.approx(close * (1 - 0.5e-4) * (1 - bps / 1e4), rel=1e-9)
    # the lab re-prices only trades its end flatten closed: this one is never charged twice
    assert t["exit_reason"] != lab.BOUNDARY_REASON


def test_load_backtester_reads_delistings_from_the_universe_db(tmp_path):
    from heartless.config import Settings
    from tests.test_fpreA_hourly import T0, make_db

    n = 24 * 60
    db = make_db(tmp_path / "u.db", {"AUSDT": {"start": T0, "c": np.full(n, 10.0)},
                                     "BUSDT": {"start": T0, "c": np.full(24 * 20, 5.0)}}, "2024-01", "2024-02")
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=db)
    got = stored_delistings(s, ["AUSDT", "BUSDT"], T0, T0 + n * H)
    assert got == {"BUSDT": T0 + 24 * 20 * H - 1}
    assert stored_delistings(Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path / "x")), ["BUSDT"], T0, None) == {}


def test_a_symbol_still_trading_after_its_listed_delisting_is_not_closed(monkeypatch, tmp_path):
    """A stale inventory (or a relisting) must not stop a live symbol: bars with volume a day past the close win."""
    ca = _eth()
    ct = ca.view("close_time")
    d = int(ct[[k for k in range(900, 1100) if (int(ct[k]) + 1) % H == 0][0]])
    res, bt = backtest(monkeypatch, tmp_path, Scripted([int(ct[800])], hold=1500, stop_atr=40.0), {"ETHUSDT": ca},
                       shock=None, delisted={"ETHUSDT": d})
    assert bt._delist_bars() == {} and res.trades[0]["exit_reason"] != DELIST_REASON
    # the archive's flat zero-volume files after a real delisting do not count as trading
    k = int(np.searchsorted(ct, d))
    ca.quote_volume[k + 1:ca.n] = 0.0
    ca.volume[k + 1:ca.n] = 0.0
    _, bt2 = backtest(monkeypatch, tmp_path, Scripted([]), {"ETHUSDT": ca}, shock=None, delisted={"ETHUSDT": d})
    assert bt2._delist_bars() == {"ETHUSDT": d}

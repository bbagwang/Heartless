"""F-pre-B shock layer (heartless/strategy/shock.py, ROADMAP 7.8): the fixed rules S0-S3, the halt / state / kill
switch / research-window queries, and the offline schedule built from stored data agreeing bit for bit with the online
detector fed bar by bar (the registration's offline = online equivalence)."""
import math

import numpy as np
import pytest

from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import symbols_inventory as SI
from heartless.strategy import shock as SH
from tests.test_f0_universe import _inv
from tests.test_fpreA_hourly import put_symbol

H, MIN, D = SH.MS_H, SH.MS_MIN, SH.MS_D
T0 = SI._month_ms("2024-01")


# --- the rules -------------------------------------------------------------------------------------------------------
def test_rule_thresholds():
    assert SH.s0_fires(103.0, 100.0) and SH.s0_fires(97.0, 100.0)
    assert not SH.s0_fires(102.9, 100.0) and not SH.s0_fires(103.0, 0.0)
    prior = [0.01 if k % 2 else -0.01 for k in range(720)]  # population sd exactly 0.01
    assert SH.pop_sd(prior) == 0.01
    assert SH.s1_fires(0.0401, prior) and SH.s1_fires(-0.0401, prior) and not SH.s1_fires(0.0399, prior)
    assert not SH.s1_fires(0.5, prior[1:])  # 719 prior returns: no verdict
    assert not SH.s1_fires(0.5, prior[:-1] + [float("nan")])
    flat = [0.01] * 720
    assert SH.s2_fires(0.0251, flat) and not SH.s2_fires(0.0249, flat)
    assert SH.median([3.0, 1.0, 2.0]) == 2.0 and SH.median([4.0, 1.0, 2.0, 3.0]) == 2.5
    half = [0.5] * 720
    assert SH.s3_fires(0.81, half) and not SH.s3_fires(0.79, [0.0] * 720)  # below the 0.80 level
    assert not SH.s3_fires(0.84, [0.6] * 720)  # under the prior mean + 0.25
    assert SH.s3_fires(0.81, [float("nan")] * 360 + half[:360])  # half of the window defined is enough
    assert not SH.s3_fires(0.81, [float("nan")] * 361 + half[:359])


def test_mean_corr():
    rng = np.random.default_rng(1)
    btc = rng.normal(0, 0.01, 24).tolist()
    same = [[2 * x + 0.5 for x in btc] for _ in range(SH.S3_MIN_MEMBERS)]
    assert SH.mean_corr(btc, same) == pytest.approx(1.0, abs=1e-12)
    anti = [[-x for x in btc] for _ in range(SH.S3_MIN_MEMBERS)]
    assert SH.mean_corr(btc, anti) == pytest.approx(-1.0, abs=1e-12)
    assert math.isnan(SH.mean_corr(btc, same[:-1]))  # too few members
    flat = [[1.0] * 24] * 5  # no variance: left out
    assert SH.mean_corr(btc, same + flat) == SH.mean_corr(btc, same)
    # the order of the rows does not change a bit (fsum)
    rows = [rng.normal(0, 0.01, 24).tolist() for _ in range(30)]
    assert SH.mean_corr(btc, rows) == SH.mean_corr(btc, rows[::-1])


# --- queries ---------------------------------------------------------------------------------------------------------
def test_entry_halt_windows_in_bar_close_times():
    h = T0 + 100 * H
    m = T0 + 10 * H + 7 * MIN
    ev = SH.ShockEvents(s0=[m], hourly={"S1": [h], "S3": [h]})
    # the 1m bar that closes the firing hour (close_time h - 1) is already halted, two hours of bars after it too
    assert ev.entry_halt(h - 1) == "shock S1+S3"
    assert ev.entry_halt(h + 2 * H - MIN - 1) == "shock S1+S3"
    assert ev.entry_halt(h + 2 * H - 1) == "" and ev.entry_halt(h - MIN - 1) == ""
    assert ev.entry_halt(m - 1) == "shock S0" and ev.entry_halt(m + 15 * MIN - MIN - 1) == "shock S0"
    assert ev.entry_halt(m + 15 * MIN - 1) == ""
    times = np.arange(T0, T0 + 103 * H, MIN, dtype=np.int64) - 1
    assert np.array_equal(ev.entry_halts(times), np.array([bool(ev.entry_halt(int(t))) for t in times]))
    # hourly evaluator decision times are hour closes (open + 1 h)
    assert ev.entry_halts(np.array([h, h + H, h + 2 * H])).tolist() == [True, True, False]


def test_state_kill_switch_and_research_window():
    f1, f2 = T0 + 10 * H, T0 + 30 * H  # 20 h apart: one state
    ev = SH.ShockEvents(hourly={"S2": [f1, f2]})
    assert ev.state_at(f1 - H) is None
    assert ev.state_at(f1 + 5 * H) == (f1, f1)
    assert ev.state_at(f2 + 23 * H) == (f1, f2)
    assert ev.state_at(f2 + 24 * H) is None
    assert ev.kill_switch_h(f1 - H, 8.0) == 8.0
    assert ev.kill_switch_h(f2 + 29 * D, 8.0) == 6.0 and ev.kill_switch_h(f2 + 30 * D, 8.0) == 8.0
    assert ev.kill_switch_h(f2, 5.0) == 5.0  # never raised
    assert ev.research_lookback_days(f1 - H, 120) == 120
    assert ev.research_lookback_days(f1, 120) == 45 and ev.research_lookback_days(f2 + 44 * D, 120) == 45
    assert ev.research_lookback_days(f2 + 45 * D, 120) == 120
    assert ev.research_lookback_days(f2, 30) == 30  # never lengthened
    c = ev.counts()
    assert (c["S2"], c["hours_fired"], c["halt_hours"]) == (2, 2, 4)
    assert c["shock_state_hours"] == 20 + 24


def test_entry_filter_for_the_evaluators():
    ev = SH.ShockEvents(hourly={"S1": [T0 + 5 * H]})

    class A:
        shock_exempt = False

    class Exempt:
        shock_exempt = True

    f = SH.EntryFilter.for_alpha(A(), ev)
    assert f.__name__ == "shock_layer" and SH.EntryFilter.for_alpha(Exempt(), ev) is None
    assert f("X", np.array([T0 + 4 * H, T0 + 5 * H, T0 + 6 * H, T0 + 7 * H])).tolist() == [True, False, False, True]
    assert f.blocked == 2
    assert f("X", T0 + 5 * H) is False and f("X", T0 + 9 * H) is True and f.blocked == 3  # scalar (neutral evaluator)


# --- offline = online ------------------------------------------------------------------------------------------------
N_DAYS = 34
N_SYM = 25


def _btc_minutes():
    """34 days of BTC 1m closes (4 decimals: stored exactly in the 1h blocks): a calm walk, a 4% 3-minute drop on day 9
    (S0), a +6% hour on day 33 (S1 and S2)."""
    rng = np.random.default_rng(7)
    n = N_DAYS * 1440
    r = rng.normal(0, 0.0004, n)
    r[9 * 1440 + 300:9 * 1440 + 303] = -0.0137  # ~ -4% over 3 minutes
    r[9 * 1440 + 303:9 * 1440 + 363] = 0.04 / 60
    k = 33 * 1440 + 600
    r[k:k + 60] = 0.06 / 60
    c = np.round(30000.0 * np.exp(np.cumsum(r)), 4)
    ct = T0 + np.arange(1, n + 1, dtype=np.int64) * MIN - 1
    return ct, c


def _universe(tmp_path, ct, c):
    """universe.db: BTC's hourly bars (the closes ending each hour) and 25 perpetuals that move on their own until the
    last 30 hours, then with BTC (S3)."""
    hours, hc = SH.btc_hourly_from_minutes(ct, c)
    rng = np.random.default_rng(3)
    n = len(hours)
    btc_r = np.diff(np.log(hc), prepend=np.log(hc[0]))
    st = Store(tmp_path / "u.db")
    syms = [f"S{k:02d}USDT" for k in range(N_SYM)]
    st.save_symbol_inventory([_inv(s, "2023-10", "2024-02") for s in ["BTCUSDT"] + syms])
    first_open = int(hours[0]) - H
    hi = np.round(hc * 1.001, 4)
    put_symbol(st, "BTCUSDT", first_open, hc, h=hi, lo=np.round(hc * 0.999, 4), qv=np.full(n, 5e9))
    for k, s in enumerate(syms):
        own = rng.normal(0, 0.006, n)
        follow = btc_r * 1.2 + rng.normal(0, 0.0005, n)
        r = np.where(np.arange(n) >= n - 30, follow, own)
        px = np.round(10.0 * (k + 1) * np.exp(np.cumsum(r)), 4)
        put_symbol(st, s, first_open, px, qv=np.round(1e7 * (1 + k) * (1 + 0.3 * rng.random(n)), 4))
    st.refresh_inventory_trades()
    st.close()
    return str(tmp_path / "u.db"), syms


def test_offline_schedule_equals_the_online_detector(tmp_path):
    ct, c = _btc_minutes()
    udb, syms = _universe(tmp_path, ct, c)
    start, end = T0 + 32 * D, int(ct[-1])
    off = SH.build_schedule(start, end, btc_1m=(ct, c), universe_db=udb)
    assert off.meta["missing"] == [] and off.meta["hourly_source"] == "universe.db"
    # online: every 1m bar, and at each hour close the hour's BTC close and the cross-section from the same blocks
    st = Store(udb, readonly=True)
    bars = {s: SI.load_hourly(st, s, None, None) for s in ["BTCUSDT"] + syms}  # every perpetual, BTC included
    st.close()
    xs_at: dict[int, dict] = {}
    for s, d in bars.items():
        for t, cl, q in zip(d["open_time"], d["close"], d["quote_volume"]):
            xs_at.setdefault(int(t) + H, {})[s] = (float(cl), float(q))
    det = SH.ShockDetector()
    cms = {}
    for t, px in zip(ct.tolist(), c.tolist()):
        det.observe_minute(t, px)
        if (t + 1) % H == 0:
            det.observe_hour(t, px, xs_at.get(t + 1, {}))
            cms[t + 1] = det.cms[-1]
    # the schedule holds the window's firings and those before it whose halt reaches into it
    on_s0 = [x for x in det.s0 if start - SH.S0_HALT_MS < x <= end + 1]
    on_h = {r: [x for x in det.hourly[r] if x > start - SH.HALT_MS] for r in SH.HOURLY_RULES}
    assert off.s0 == on_s0 and off.hourly == on_h
    assert len(off.s0) == 0  # the S0 drop is on day 9, before this window
    assert off.hourly["S1"] and off.hourly["S2"] and off.hourly["S3"], off.hourly
    # the S3 statistic itself, bit for bit, at every hour of the window
    hours, hc = SH.btc_hourly_from_minutes(ct, c)
    st = Store(udb, readonly=True)
    series = SH.cross_section_series(st, hours, hc)
    st.close()
    for k, h in enumerate(hours.tolist()):
        if h >= T0 + 2 * D:
            a, b = series[k], cms[h]
            assert (math.isnan(a) and math.isnan(b)) or a == b, (h, a, b)
    # the same schedule over the whole span finds the S0 drop
    full = SH.build_schedule(T0 + 8 * D, end, btc_1m=(ct, c), universe_db=udb)
    assert len(full.s0) >= 1 and all(T0 + 9 * D <= x < T0 + 9 * D + 7 * H for x in full.s0)
    assert full.s0 == [x for x in det.s0 if x > T0 + 8 * D - SH.S0_HALT_MS]


def test_a_primed_detector_continues_as_the_offline_schedule(tmp_path):
    """The app's start-up: ShockDetector.prime reads universe.db up to `until` (no 1m bars), then live hours follow;
    the firings equal the offline schedule's, and so does the S3 statistic of every live hour."""
    ct, c = _btc_minutes()
    udb, syms = _universe(tmp_path, ct, c)
    start, end = T0 + 32 * D, int(ct[-1])
    off = SH.build_schedule(start, end, btc_1m=(ct, c), universe_db=udb)
    mid = T0 + 33 * D
    det = SH.ShockDetector()
    assert SH.ShockDetector().prime(tmp_path / "none.db", mid) == 0
    n = det.prime(udb, until=mid, since=start - SH.WARMUP_MS)
    assert n == (mid - (start - SH.WARMUP_MS)) // H + 1 and det.h_last == mid
    st = Store(udb, readonly=True)
    live = dict(SH.stored_cross_sections(st, mid + H, end + 1))
    hours, hc = SH.btc_hourly_from_minutes(ct, c)
    series = SH.cross_section_series(st, hours, hc)
    st.close()
    btc_at = dict(zip(hours.tolist(), hc.tolist()))
    for h in range(mid + H, end + 2, H):
        det.observe_hour(h, btc_at[h], live.get(h, {}))
        k = int((h - hours[0]) // H)
        assert (math.isnan(series[k]) and math.isnan(det.cms[-1])) or series[k] == det.cms[-1]
    assert {r: [x for x in det.hourly[r] if x > start - SH.HALT_MS] for r in SH.HOURLY_RULES} == off.hourly
    assert off.hourly["S1"] and off.hourly["S3"]
    # the default history keeps 45 days of firings (kill switch 30 days, research window 45 days)
    again = SH.ShockDetector()
    assert again.prime(udb, until=end + 1) > 0 and again.last_firing(end + 1) == off.last_firing(end + 1)


def test_schedule_from_a_candle_database_and_without_universe(tmp_path):
    ct, c = _btc_minutes()
    db = Store(tmp_path / "c.db")
    db.save_candles("BTCUSDT", [Candle(int(t) - 59_999, float(p), float(p), float(p), float(p), 1.0, float(p), 1, 0.5,
                                       int(t)) for t, p in zip(ct, c)])
    db.close()
    end = int(ct[-1])
    a = SH.build_schedule(T0 + 32 * D, end, candle_db=tmp_path / "c.db")
    b = SH.build_schedule(T0 + 32 * D, end, btc_1m=(ct, c))
    assert a.meta["missing"] == ["S3"] and a.meta["hourly_source"] == "1m" and a.s0 == b.s0
    # no candle earlier than S0_READ_MS before the window start is read: two days of bars hold no 720-hour history, so
    # S1 / S2 cannot judge
    assert a.hourly["S1"] == [] and b.hourly["S1"]
    a2 = SH.build_schedule(T0 + D, end, candle_db=tmp_path / "c.db")
    assert {r: [h for h in v if h > T0 + 32 * D - SH.HALT_MS] for r, v in a2.hourly.items()} == b.hourly
    # nothing after `end` is read: a window ending before the +6% hour has no S1
    early = SH.build_schedule(T0 + 32 * D, T0 + 33 * D, btc_1m=(ct, c))
    assert early.hourly["S1"] == []
    none = SH.build_schedule(T0 + 32 * D, int(ct[-1]))
    assert none.meta["missing"] == ["S0", "S1", "S2", "S3"]


def test_a_window_starting_inside_a_halt_is_halted(tmp_path):
    """A firing just before a window halts the window's first decisions as it does online and in any longer window:
    the schedule keeps the S1-S3 firings up to 2 h and the S0 firings up to 15 minutes before `start` (they are history
    only: counts() reports the window itself)."""
    ct, c = _btc_minutes()
    udb, _ = _universe(tmp_path, ct, c)
    end = int(ct[-1])
    full = SH.build_schedule(T0 + 32 * D, end, btc_1m=(ct, c), universe_db=udb)
    f = full.hourly["S1"][0]
    for k in (1, 2):  # windows starting 1 h and 2 h after the S1 firing
        cut = SH.build_schedule(f + k * H, end, btc_1m=(ct, c), universe_db=udb)
        assert ("S1" in cut.entry_halt(f + k * H)) == (k == 1) == ("S1" in full.entry_halt(f + k * H))
        assert cut.counts()["S1"] == 0 and cut.counts(f, end)["S1"] == (1 if k == 1 else 0)
        for t in range(f + k * H, f + 3 * H, MIN):  # every decision of the window from its start, minute by minute
            assert cut.entry_halt(t) == full.entry_halt(t), (k, t)
    det = SH.ShockDetector()
    det.prime(udb, until=f + H, since=f + H - SH.WARMUP_MS)
    assert det.entry_halt(f + H) == full.entry_halt(f + H) == SH.build_schedule(f + H, end, universe_db=udb).entry_halt(f + H)
    # S0 read from a candle database: the 15 minutes after a firing that precedes the window
    db = Store(tmp_path / "c.db")
    db.save_candles("BTCUSDT", [Candle(int(t) - 59_999, float(p), float(p), float(p), float(p), 1.0, float(p), 1, 0.5,
                                       int(t)) for t, p in zip(ct, c)])
    db.close()
    s0 = SH.build_schedule(T0 + 8 * D, end, candle_db=tmp_path / "c.db")
    m = s0.s0[0]
    last = s0.s0[-1]  # the drop fires on consecutive minutes
    for t0 in (m + MIN, last + MIN, last + 14 * MIN, last + 15 * MIN):
        late = SH.build_schedule(t0, end, candle_db=tmp_path / "c.db")
        assert late.entry_halt(t0) == s0.entry_halt(t0) == ("shock S0" if t0 < last + 15 * MIN else ""), t0
        assert late.counts()["S0"] == sum(1 for x in s0.s0 if x >= t0)


def test_schedule_file_and_process_cache(tmp_path):
    from heartless.config import Settings

    ct, c = _btc_minutes()
    path = tmp_path / "shock.json"
    s = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), SHOCK_SCHEDULE=str(path))
    SH._CACHE.clear()
    a = SH.schedule_for(s, T0 + 32 * D + 5 * H, int(ct[-1]), btc_1m=(ct, c))
    assert path.exists() and a.start == T0 + 32 * D + 5 * H
    b = SH.load_schedule(path)
    assert b.hourly == a.hourly and b.s0 == a.s0 and b.covers(T0 + 32 * D + 5 * H, int(ct[-1]))
    SH._CACHE.clear()
    assert SH.schedule_for(s, T0 + 32 * D + 5 * H, int(ct[-1])).hourly == a.hourly  # read back, nothing rebuilt


# --- S3 membership: the research context's point-in-time top_n ------------------------------------------------------
def _feed(xs, bars: dict[str, dict], h_from: int, h_to: int):
    by_h: dict[int, list] = {}
    for s, d in bars.items():
        for t, c, q in zip(d["open_time"], d["close"], d["quote_volume"]):
            by_h.setdefault(int(t) + H, []).append((s, float(c), float(q)))
    out = {}
    for h in range(h_from, h_to + H, H):
        for s, c, q in by_h.get(h, []):
            xs.push(h, s, c, q)
        out[h] = xs.members(h)
    return out


def test_s3_members_follow_the_context_rule():
    xs = SH._CrossSection()
    h = T0 + 100 * H
    bars = {}
    for k in range(45):  # 45 symbols trading every hour, volume rising with k; BTC the most liquid
        s = "BTCUSDT" if k == 44 else f"Z{k:02d}USDT"
        t = np.arange(h - 60 * H, h, H, dtype=np.int64)
        bars[s] = {"open_time": t, "close": np.full(len(t), 10.0 + k), "quote_volume": np.full(len(t), 1e6 * (k + 1))}
    bars["Z43USDT"]["quote_volume"][-1] = 0.0  # did not trade in the last hour: out
    bars["TIEAUSDT"] = dict(bars["Z42USDT"])  # same volume as Z42: the name decides
    young = np.arange(h - 20 * H, h, H, dtype=np.int64)  # 20 hours of history: no close 23 hours back
    bars["YNGUSDT"] = {"open_time": young, "close": np.full(20, 5.0), "quote_volume": np.full(20, 1e12)}
    gap = np.array([h - 47 * H, h - H], dtype=np.int64)  # a close 46 hours before the close then a gap: carried
    bars["GAPUSDT"] = {"open_time": gap, "close": np.array([5.0, 5.0]), "quote_volume": np.array([1e12, 1e12])}
    far = np.array([h - 48 * H, h - H], dtype=np.int64)  # one hour older: beyond the carry
    bars["FARUSDT"] = {"open_time": far, "close": np.array([5.0, 5.0]), "quote_volume": np.array([2e12, 2e12])}
    m = _feed(xs, bars, h - 60 * H + H, h)[h]
    assert len(m) == SH.S3_TOP and m[0] == "GAPUSDT" and m[1] == "BTCUSDT"
    assert "YNGUSDT" not in m and "FARUSDT" not in m and "Z43USDT" not in m
    assert m.index("TIEAUSDT") < m.index("Z42USDT")
    # members without 24 defined returns stay members but leave the mean (GAPUSDT's carry ends 24 hours on)
    assert any(not math.isfinite(x) for x in xs.returns("GAPUSDT", h))


def test_s3_members_equal_context_pit_members_when_available(tmp_path):
    """After the merge with the point-in-time research context (context.pit_members), S3's members must be the same
    set at every hour (ROADMAP 7.8: "the context's top_n member definition")."""
    from heartless.data import context as C

    if not hasattr(C, "pit_members"):
        pytest.skip("context.pit_members is not in this tree yet")
    rng = np.random.default_rng(9)
    st = Store(tmp_path / "u.db")
    syms = [f"P{k:02d}USDT" for k in range(48)]
    st.save_symbol_inventory([_inv(s, "2023-10", "2024-02") for s in syms])
    n = 24 * 6
    for k, s in enumerate(syms):
        qv = np.round(rng.lognormal(14, 1, n), 4)
        qv[rng.random(n) < 0.05] = 0.0
        start = T0 + int(rng.integers(0, 40)) * H  # staggered listings
        put_symbol(st, s, start, np.round(10 + rng.normal(0, 0.1, n).cumsum(), 4), qv=qv)
    st.refresh_inventory_trades()
    st.close()
    us = Store(tmp_path / "u.db", readonly=True)
    bars = {s: SI.load_hourly(us, s, None, None) for s in syms}
    ha, hb = T0 // H + 60, T0 // H + n - 2
    cands, mask = C.pit_members(us, ha, hb, SH.S3_TOP)
    us.close()
    ours = _feed(SH._CrossSection(), bars, T0 + H, (hb + 1) * H)
    for i, hh in enumerate(range(ha, hb + 1)):
        want = {cands[j] for j in np.flatnonzero(mask[i])}
        assert set(ours[(hh + 1) * H]) == want, hh

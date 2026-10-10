"""P2a market context: the offline build and the online tracker produce identical snapshots, without look-ahead."""
import math

import numpy as np
import pytest

from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C

H = C.MS_H
DAY = 86_400_000
START = 1_701_734_400_000  # 2023-12-05 00:00 UTC: the history crosses the 2023/2024 year boundary
SYMS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "LATEUSDT", "GAPUSDT")


def _history(days: int = 40, seed: int = 3) -> dict:
    """1m bars, funding and metrics for six symbols with the awkward cases: a late listing, a 30h outage, a missing
    hour, zero-volume minutes, a funding gap and a stale metrics stretch."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    common = rng.normal(0, 0.0008, n)
    out = {}
    for k, sym in enumerate(SYMS):
        ret = 0.6 * common + rng.normal(0.00002 * (k - 2), 0.0009, n)
        close = (100 + 40 * k) * np.exp(np.cumsum(ret))
        keep = np.ones(n, dtype=bool)
        if sym == "LATEUSDT":
            keep[: 9 * 1440 + 17] = False  # listed on day 9, mid-hour
        if sym == "GAPUSDT":
            keep[20 * 1440: 20 * 1440 + 30 * 60] = False  # a 30-hour outage (longer than the carry window)
            keep[30 * 1440 + 120: 30 * 1440 + 180] = False  # one whole missing hour
        if sym == "DOGEUSDT":
            keep[rng.random(n) < 0.01] = False  # scattered missing minutes
        bars = []
        prev = float(close[0])
        for i in np.flatnonzero(keep):
            c = float(close[i])
            hi = max(prev, c) * (1 + abs(float(rng.normal(0, 3e-4))))
            lo = min(prev, c) * (1 - abs(float(rng.normal(0, 3e-4))))
            v = 0.0 if i % 389 == 0 else float(abs(rng.normal(50, 20)))
            t = START + int(i) * 60_000
            bars.append(Candle(t, prev, hi, lo, c, v, v * c, 5, v * 0.5, t + 59_999))
            prev = c
        funding = [(START + j * 8 * H, float(rng.normal(1e-4, 2e-4))) for j in range(days * 3)
                   if not (sym == "ETHUSDT" and 40 <= j < 46)]  # a two-day funding hole
        metrics = []
        if sym != "SOLUSDT":  # one symbol without positioning data
            for j in range(days * 288):
                if sym == "BTCUSDT" and 3000 <= j < 3020:
                    continue  # 100 minutes without rows: stale snapshot
                metrics.append({"ts": START + j * 300_000, "oi": float(1e6 * (1 + 0.1 * math.sin(j / 500 + k))),
                                "oi_value": float(5e7 * (1 + 0.1 * math.sin(j / 400 + k))), "top_ls_accounts": 1.0,
                                "top_ls_positions": 1.0, "ls_accounts": 1.0, "taker_ls_vol": 1.0})
        out[sym] = {"bars": bars, "funding": funding, "metrics": metrics}
    return out


@pytest.fixture(scope="module")
def hist():
    return _history()


@pytest.fixture(scope="module")
def db(hist, tmp_path_factory):
    path = tmp_path_factory.mktemp("ctx") / "h.db"
    st = Store(path)
    for sym, d in hist.items():
        st.save_candles(sym, d["bars"])
        st.save_funding(sym, [(t, r, 0.0) for t, r in d["funding"]])
        from types import SimpleNamespace

        st.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"]])
    st.close()
    return path


def _replay(hist, tracker: C.MarketContextTracker, until: int | None = None) -> dict[int, C.Snapshot]:
    """Feed everything in time order like the live bot (all symbols' bars of a minute, then funding settlements and
    metrics rows as they become known) and take a snapshot at every hour close."""
    events = []
    for sym, d in hist.items():
        events += [(b.close_time, 0, sym, b) for b in d["bars"]]
        events += [(t, 1, sym, r) for t, r in d["funding"]]
        events += [(m["ts"] + 300_000 - 1, 2, sym, m) for m in d["metrics"]]  # usable once the bucket closed
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    snaps = {}
    hour = None
    for t, kind, sym, x in events:
        if until is not None and t > until:
            break
        h = C.hour_of(t)
        if hour is not None and h > hour:
            for hh in range(hour, h):
                s = tracker.snapshot(C.hour_close(hh))
                if s is not None:
                    snaps[s.ts] = s
        hour = h if hour is None else max(hour, h)
        if kind == 0:
            tracker.on_bar(sym, x)
        elif kind == 1:
            tracker.on_funding(sym, t, x)
        else:
            tracker.on_metrics(sym, x)
    if until is None and hour is not None:  # the data ends on an hour close: that hour is complete as well
        s = tracker.snapshot(C.hour_close(hour))
        if s is not None:
            snaps[s.ts] = s
    return snaps


def _bits_equal(a, b) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.isnan(a), np.isnan(b)
    return bool(np.array_equal(na, nb) and np.array_equal(a[~na].view(np.int64), b[~nb].view(np.int64)))


def test_offline_build_equals_online_tracker_bit_for_bit(hist, db, tmp_path):
    st = Store(db)
    out = Store(tmp_path / "ctx.db")
    stats = C.build(st, out, universe=SYMS)
    assert [u.get("skipped", False) for u in stats["units"]] == [False, False]  # two calendar years
    table = C.ContextTable.load(out, SYMS, START, START + 41 * DAY)
    online = _replay(hist, C.MarketContextTracker(SYMS))
    assert len(table) == len(online) > 900
    checked = defined = 0
    seen = np.zeros(len(C.SYMBOL_FEATURES), dtype=bool)
    seen_m = np.zeros(len(C.MARKET_FEATURES), dtype=bool)
    for T, snap in online.items():
        seen_m |= np.isfinite(snap.mkt)
        mkt = table.market[T]
        assert _bits_equal(mkt, snap.mkt), C._iso(T)
        for j, sym in enumerate(snap.symbols):
            row = table.symbols[sym].get(T)
            assert (row is not None) == bool(snap.present[j]), (sym, C._iso(T))
            if row is not None:
                assert _bits_equal(row, snap.sym[j]), (sym, C._iso(T))
                checked += 1
                defined += int(np.isfinite(row).all())
                seen |= np.isfinite(row)
        for sym in SYMS:  # what the alphas see
            a, b = dict(table.view(sym, T)), dict(snap.view(sym))
            assert a.keys() == b.keys()
            assert all(_bits_equal(a[k], b[k]) for k in a)
    # the last week has every feature defined (the 800h EMA window included), and each one was compared somewhere
    assert checked > 5000 and defined > 300 and seen.all() and seen_m.all()
    st.close()
    out.close()


def test_snapshot_is_free_of_look_ahead_and_chunking(hist, db, tmp_path):
    """The row for T is the same when nothing after T exists, and when the build is cut into different units."""
    st = Store(db)
    T = C.hour_close(C.hour_of(START + 35 * DAY + 5 * H))
    full = C.compute_unit(st, SYMS, START, START + 40 * DAY)
    one = {s.ts: s for s in full}[T]
    cut = Store(tmp_path / "cut.db")
    for sym, d in hist.items():
        cut.save_candles(sym, [b for b in d["bars"] if b.close_time <= T])
        cut.save_funding(sym, [(t, r, 0.0) for t, r in d["funding"] if t <= T])
        from types import SimpleNamespace

        cut.save_metrics(sym, [SimpleNamespace(**m) for m in d["metrics"] if m["ts"] + 300_000 <= T])
    alone = {s.ts: s for s in C.compute_unit(cut, SYMS, T - 3 * H, T)}[T]
    assert _bits_equal(one.mkt, alone.mkt) and _bits_equal(one.sym, alone.sym)
    # a partial hour at the data end is not built: hours are complete or absent
    assert C.data_span(cut, list(SYMS))[1] == T
    # the tracker that only ever saw data up to T agrees as well
    tr = C.MarketContextTracker(SYMS)
    snaps = _replay(hist, tr, until=T)
    last = tr.snapshot(T)
    assert last is not None and _bits_equal(last.sym, one.sym) and _bits_equal(last.mkt, one.mkt)
    assert max(snaps) < T
    st.close()
    cut.close()


def test_feature_semantics(hist, db):
    st = Store(db)
    snaps = {s.ts: s for s in C.compute_unit(st, SYMS, START, START + 40 * DAY)}
    T = max(snaps)
    s = snaps[T]
    i = {n: k for k, n in enumerate(C.SYMBOL_FEATURES)}
    m = dict(zip(C.MARKET_FEATURES, s.mkt))
    btc = s.symbols.index("BTCUSDT")
    closes = {}
    for sym, d in hist.items():
        cs = [b for b in d["bars"] if b.close_time <= T]
        closes[sym] = cs
    c_now = closes["BTCUSDT"][-1].close
    c_24 = [b for b in closes["BTCUSDT"] if b.close_time <= T - 24 * H][-1].close
    assert m["btc_ret_24h"] == c_now / c_24 - 1.0 == s.sym[btc, i["ret_24h"]]
    assert m["n"] == s.present.sum() == len(SYMS)
    for k in C.RET_HOURS:
        r = s.sym[:, i[f"ret_{k}h_rank"]]
        assert np.nanmin(r) == 0.0 and np.nanmax(r) == 1.0
        z = s.sym[:, i[f"ret_{k}h_z"]]
        assert abs(np.nanmean(z)) < 1e-9 and abs(np.nanstd(z) - 1.0) < 1e-9
    assert (s.sym[:, i["dist_hi_24h"]] <= 0).all() and (s.sym[:, i["dist_lo_168h"]] >= 0).all()
    assert abs(np.nansum(s.sym[:, i["vol_share_24h"]]) - 1.0) < 1e-12
    assert np.isnan(s.sym[s.symbols.index("SOLUSDT"), i["oi_chg_24h"]])  # no metrics -> unknown, not 0
    assert m["breadth_ema200"] == np.nanmean(s.sym[:, i["above_ema200"]])
    assert m["btc_trend"] in (-1.0, 0.0, 1.0)
    # the late listing has no 168h return for its first week, and the cross-section shrinks without it
    late = s.symbols.index("LATEUSDT")
    first_late = min(t for t, sn in snaps.items() if sn.present[late])
    early = snaps[first_late + 24 * H]
    assert np.isnan(early.sym[late, i["ret_168h"]]) and np.isfinite(early.sym[late, i["ret_1h"]])
    assert np.isnan(early.sym[late, i["ret_168h_rank"]])
    # the 30h outage: absent from the cross-section; in the first hour back the 1h return is unknown (the last close
    # is older than the 24h carry window) while the 24h return reaches back to the close before the outage
    gap = s.symbols.index("GAPUSDT")
    t_gap = START + 20 * DAY + 10 * H
    assert not snaps[C.hour_close(C.hour_of(t_gap))].present[gap]
    back = snaps[C.hour_close(C.hour_of(START + 20 * DAY + 31 * H))]
    assert back.present[gap] and np.isnan(back.sym[gap, i["ret_1h"]]) and np.isfinite(back.sym[gap, i["ret_24h"]])
    assert np.isnan(back.sym[gap, i["above_ema50"]])  # the EMA window holds a gap
    st.close()


def test_tracker_ignores_late_bars_ready_and_warm(hist, db):
    tr = C.MarketContextTracker(SYMS)
    T = C.hour_close(C.hour_of(START + 36 * DAY))
    st = Store(db)
    fed = tr.warm(st, T)
    assert fed > 6 * C.WARMUP_HOURS * 50
    snap = tr.snapshot(T)
    ref = {s.ts: s for s in C.compute_unit(st, SYMS, T - 2 * H, T)}[T]
    assert _bits_equal(snap.sym, ref.sym) and _bits_equal(snap.mkt, ref.mkt)
    # a late bar of the published hour changes nothing
    b = hist["BTCUSDT"]["bars"]
    late = [x for x in b if x.close_time == T][0]
    tr.on_bar("BTCUSDT", Candle(late.open_time, late.open, late.high * 2, late.low, late.close * 3, 1e9, 1e12, 1, 1,
                                late.close_time))
    assert tr.snapshot(T) is snap and _bits_equal(tr.snapshot(T).sym, ref.sym)
    # ready() waits for every symbol that traded in the previous hour
    nxt = T + H
    assert not tr.ready(nxt)
    for sym in SYMS:
        for x in hist[sym]["bars"]:
            if T < x.close_time <= nxt:
                tr.on_bar(sym, x)
                if sym == "BTCUSDT" and x.close_time == nxt:
                    assert not tr.ready(nxt)
    assert tr.ready(nxt)
    v = tr.view("ETHUSDT", nxt + 30 * 60_000)
    assert v["ts"] == nxt and "ret_24h_rank" in v and "breadth_ema50" in v
    assert tr.view("XRPUSDT", nxt)["ts"] == nxt and "ret_24h" not in tr.view("XRPUSDT", nxt)  # not in the universe
    with pytest.raises(TypeError):
        v["ts"] = 0  # read-only
    st.close()


def test_tracker_warmed_mid_hour_after_a_settlement_matches_the_offline_build(hist, db):
    """A bot restarted at 08:30 has missed the live 08:00 settlement and the 08:00..08:29 bars: warm() must take them
    from the store, or the next snapshots carry the previous funding rate for 8 h and a partial hour of volume."""
    st = Store(db)
    now = START + 37 * DAY + 8 * H + 30 * 60_000
    assert any(t == START + 37 * DAY + 8 * H for t, _ in hist["BTCUSDT"]["funding"])
    offline = {s.ts: s for s in C.compute_unit(st, SYMS, now - H, now + 31 * H)}
    tr = C.MarketContextTracker(SYMS)
    tr.warm(st, now)
    events = []  # the live feed from `now` on
    for sym, d in hist.items():
        events += [(b.close_time, 0, sym, b) for b in d["bars"] if b.close_time > now]
        events += [(t, 1, sym, r) for t, r in d["funding"] if t > now]
        events += [(m["ts"] + 300_000 - 1, 2, sym, m) for m in d["metrics"] if m["ts"] + 300_000 - 1 > now]
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    compared = []
    hour = C.hour_of(now)
    for t, kind, sym, x in events:
        if t > now + 31 * H:
            break
        while t > C.hour_close(hour + 1):  # the next hour close has passed: publish it before feeding later data
            hour += 1
            snap, ref = tr.snapshot(C.hour_close(hour)), offline[C.hour_close(hour)]
            assert _bits_equal(snap.sym, ref.sym) and _bits_equal(snap.mkt, ref.mkt), C._iso(ref.ts)
            compared.append(ref.ts)
        if kind == 0:
            tr.on_bar(sym, x)
        elif kind == 1:
            tr.on_funding(sym, t, x)
        else:
            tr.on_metrics(sym, x)
    assert len(compared) == 31 and compared[0] == START + 37 * DAY + 9 * H - 1
    st.close()


def test_store_table_is_lazy_and_build_resumes(db, tmp_path):
    st = Store(db)
    assert not st.has_market_context()  # opening / reading never creates it
    assert C.ContextTable.load(st, SYMS, START, START + DAY) is None
    out = Store(tmp_path / "c.db")
    first = C.build(out_src := st, out, end=START + 30 * DAY, universe=SYMS)
    assert [u.get("skipped", False) for u in first["units"]] == [False, False]
    again = C.build(out_src, out, end=START + 30 * DAY, universe=SYMS)
    assert all(u["skipped"] for u in again["units"]) and again["rows"] == 0
    more = C.build(out_src, out, universe=SYMS)  # the second year grew: rebuilt, the first year is kept
    assert [u.get("skipped", False) for u in more["units"]] == [True, False]
    assert C.ensure(st, out, START, START + 40 * DAY, SYMS)
    # another universe invalidates the stored rows
    assert C.ContextTable.load(out, SYMS, START, START + DAY) is not None
    assert not C.ensure(st, out, START, START + 40 * DAY, SYMS[:4], auto_build=False)
    st.close()
    out.close()


def test_context_cli_build_and_status(hist, tmp_path, monkeypatch, capsys):
    import heartless.cli as cli

    data = tmp_path / "data"
    data.mkdir()
    st = Store(data / "heartless.db")
    for sym in SYMS[:3]:
        st.save_candles(sym, hist[sym]["bars"][: 3 * 1440])
    st.close()
    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(data))
    monkeypatch.setenv("CONTEXT_DB", str(tmp_path / "derived" / "ctx.db"))
    monkeypatch.setenv("CONTEXT_UNIVERSE", ",".join(SYMS))
    cli.main(["context", "build"])
    out = capsys.readouterr().out
    assert "72 hours" in out and "3/6 universe symbols stored" in out
    cli.main(["context", "build"])
    assert "0 hours" in capsys.readouterr().out  # resumed: nothing left to build
    cli.main(["context", "status"])
    out = capsys.readouterr().out
    assert "72 hours" in out and "current" in out
    assert not Store(data / "heartless.db").has_market_context()  # the candle database was only read

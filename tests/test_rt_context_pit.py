"""Registered amendment 2026-10-16 (ROADMAP section 6, red-team F-M3): a research context store built with
CONTEXT_TOP_N chooses, at every hour close, the top N USDT perpetuals by trailing 24h quote volume among EVERY
eligible perpetual trading in that hour (universe.db 1h blocks), not among the union of a window's monthly members.
A chosen symbol's features use its 1m candles where the candle database has them and its 1h bars elsewhere."""
import numpy as np
import pytest

from heartless.core.models import Candle
from heartless.core.store import Store
from heartless.data import context as C
from heartless.data import symbols_inventory as SI
from tests.test_f0_universe import M, _inv

H = 3_600_000
DAY = 86_400_000


def _rows(month: str, price: float, qv_of, days: int | None = None) -> list[list[str]]:
    """Hourly archive rows of `month` (open_time, open, high, low, close, volume, close_time, quote_volume, ...):
    a slow drift around `price`, quote volume qv_of(open_time)."""
    a, b = SI._month_ms(month), SI._next_month_ms(month)
    out = []
    for i, t in enumerate(range(a, b, H)):
        if days is not None and (t - a) // DAY + 1 > days:
            break
        c = price * (1 + 0.01 * np.sin(i / 7))
        o = price * (1 + 0.01 * np.sin((i - 1) / 7))
        q = float(qv_of(t))
        out.append([str(t), f"{o:.4f}", f"{max(o, c) * 1.002:.4f}", f"{min(o, c) * 0.998:.4f}", f"{c:.4f}",
                    f"{q / c:.3f}", str(t + H - 1), f"{q:.5f}", "10" if q > 0 else "0", f"{q / c / 2:.3f}",
                    f"{q * 0.55:.5f}", "0"])
    return out


SPEC = {  # symbol -> (price, quote volume per hour as a function of the bar open)
    "AAAUSDT": (100.0, lambda t: 1e6),
    "BBBUSDT": (5.0, lambda t: 5e6 if t >= M("2024-01") + 19 * DAY else 1e5),  # liquid from 2024-01-20
    "DDDUSDT": (2.0, lambda t: 3e6 if t < M("2024-01") + 9 * DAY else 0.0),  # stops trading 2024-01-10 (flat bars)
    "BTCUSDT": (40_000.0, lambda t: 2e6),
    "USDCUSDT": (1.0, lambda t: 1e9),  # a stablecoin perpetual: never a candidate
    "AAAUSDTSETTLED": (100.0, lambda t: 1e9),  # a settled archive copy: never a candidate
}


def make_udb(path, months=("2024-01", "2024-02")) -> None:
    st = Store(path)
    st.save_symbol_inventory([_inv(s, months[0], months[-1], kind="settled" if s.endswith("SETTLED") else "perpetual")
                              for s in SPEC])
    blocks = []
    for sym, (price, qv) in SPEC.items():
        for mo in months:
            rows = _rows(mo, price, qv)
            fund = [(int(r[0]), "0.00010000") for r in rows if int(r[0]) % (8 * H) == 0]
            blob, s = SI.encode_month(rows, mo, fund)
            blocks.append((sym, M(mo), s["n"], s["first_ts"], s["last_ts"], s["first_trade"], s["last_trade"],
                           s["med_dqv"], s["days_traded"], blob))
    st.save_kline_blocks(SI.FUT_TABLE, blocks)
    st.refresh_inventory_trades()
    st.close()


@pytest.fixture
def udb(tmp_path):
    make_udb(tmp_path / "universe.db")
    return tmp_path / "universe.db"


def _members_at(udb, when: int, top_n: int) -> list[str]:
    st = Store(udb, readonly=True)
    h = C.hour_of(when)
    syms, mask = C.pit_members(st, h, h, top_n)
    st.close()
    return sorted(s for s, m in zip(syms, mask[0]) if m)


def test_the_cross_section_is_point_in_time_over_every_eligible_perpetual(udb):
    # 2024-01-05: DDD (3e6/h) and BTC (2e6/h); AAA (1e6) and BBB (1e5) are below them
    assert _members_at(udb, M("2024-01") + 4 * DAY + 12 * H - 1, 2) == ["BTCUSDT", "DDDUSDT"]
    # 2024-01-15: DDD no longer trades (flat bars), AAA takes its place; BBB is not liquid yet
    assert _members_at(udb, M("2024-01") + 14 * DAY + 12 * H - 1, 2) == ["AAAUSDT", "BTCUSDT"]
    assert _members_at(udb, M("2024-01") + 14 * DAY + 12 * H - 1, 3) == ["AAAUSDT", "BBBUSDT", "BTCUSDT"]
    # 2024-01-25: BBB's trailing 24 h are 5e6 an hour: it leads; the union of the window's monthly members would have
    # held it from the start of the window (it ranks only from 2024-02 by January's median), and DDD until the end
    assert _members_at(udb, M("2024-01") + 24 * DAY + 12 * H - 1, 2) == ["BBBUSDT", "BTCUSDT"]
    # the stablecoin and the settled copy never enter, however large their volume
    st = Store(udb, readonly=True)
    syms, _ = C.pit_members(st, C.hour_of(M("2024-01") + 5 * DAY), C.hour_of(M("2024-01") + 6 * DAY), 10)
    st.close()
    assert "USDCUSDT" not in syms and "AAAUSDTSETTLED" not in syms


def test_24_hours_of_history_and_the_switch_hour(udb):
    assert _members_at(udb, M("2024-01") + 10 * H - 1, 2) == []  # nobody has 24 h of bars yet
    # the trailing 24 h of BBB cross BTC's 2e6 an hour once enough 5e6 hours are in the window: (5e6 k + 1e5 (24 - k))
    # > 48e6 for k >= 10, so 10 hours after the switch at 2024-01-20 00:00
    switch = M("2024-01") + 19 * DAY
    assert "BBBUSDT" not in _members_at(udb, switch + 9 * H - 1, 1)
    assert _members_at(udb, switch + 10 * H - 1, 1) == ["BBBUSDT"]


def _candles(sym: str, a: int, b: int, price_of) -> list[Candle]:
    return [Candle(t, price_of(t), price_of(t) * 1.001, price_of(t) * 0.999, price_of(t), 10.0, 777.0, 5, 5.0,
                   t + 59_999) for t in range(a, b, 60_000)]


def test_features_use_1m_candles_where_stored_and_1h_bars_elsewhere(udb, tmp_path):
    cdb = tmp_path / "c.db"
    st = Store(cdb)
    day15 = M("2024-01") + 14 * DAY
    st.save_candles("AAAUSDT", _candles("AAAUSDT", day15, day15 + DAY, lambda t: 123.0))  # 1m only on 2024-01-15
    st.close()
    src, us = Store(cdb, readonly=True), Store(udb, readonly=True)
    g0, g1 = C.hour_of(M("2024-01") + 10 * DAY), C.hour_of(M("2024-01") + 20 * DAY)
    cols = C._load_symbol_pit(src, us, "AAAUSDT", g0, g1)
    hours = SI.load_hourly(us, "AAAUSDT", g0 * H, g1 * H)
    by_t = dict(zip(hours["open_time"].tolist(), hours["close"].tolist()))
    qv_t = dict(zip(hours["open_time"].tolist(), hours["quote_volume"].tolist()))
    for r, h in enumerate(range(g0, g1 + 1)):
        t = h * H
        if day15 <= t < day15 + DAY:  # the candle database's minutes: last 1m close, the minutes' summed volume
            assert cols["close"][r] == 123.0 and cols["qvol"][r] == pytest.approx(60 * 777.0)
        else:  # the 1h archive bar (its quote volume to 0.1%, symbols_inventory.QV_STEP)
            assert cols["close"][r] == by_t[t] and cols["qvol"][r] == qv_t[t] and cols["present"][r]
            assert qv_t[t] == pytest.approx(1e6, rel=1e-3)
    # funding: the 1h blocks' settlements fill in where the candle database has none
    assert np.nanmax(cols["funding"]) == pytest.approx(1e-4)
    assert np.isnan(cols["oi_chg"]).all()  # no metrics in either source
    src.close()
    us.close()


def test_merged_hours_equal_the_1m_path_bit_for_bit_where_1m_exists():
    rng = np.random.default_rng(3)
    g0, g1 = 1000, 1100
    base = g0 - C.CARRY_HOURS
    ot = np.array([t for t in range(base * H, (g1 + 1) * H, 60_000) if not (1040 * H <= t < 1050 * H)], dtype=np.int64)
    n = len(ot)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    hi, lo, qv = c * 1.001, c * 0.999, rng.lognormal(3, 1, n)
    m = C._raw_minutes(ot, hi, lo, c, qv, base, g1)
    ref = C.hourly_columns(ot, hi, lo, c, qv, g0, g1)
    merged = C.merged_columns(m, None, g0, g1)
    for k in ref:
        assert np.array_equal(ref[k], merged[k], equal_nan=True), k
    # a 1h bar in the 1m hole fills it; every other hour keeps the 1m values bit for bit
    hot = np.arange(base, g1 + 1, dtype=np.int64) * H
    hrs = C._raw_hours(hot, np.full(len(hot), 2.0), np.full(len(hot), 1.0), np.full(len(hot), 1.5),
                       np.full(len(hot), 9.0), base, g1)
    both = C.merged_columns(m, hrs, g0, g1)
    hole = (np.arange(g0, g1 + 1) >= 1040) & (np.arange(g0, g1 + 1) < 1050)
    assert both["present"].all() and np.all(both["close"][hole] == 1.5) and np.all(both["qvol"][hole] == 9.0)
    for k in ("high", "low", "qvol", "present"):
        assert np.array_equal(both[k][~hole], ref[k][~hole], equal_nan=True), k


def test_an_injected_member_mask_is_the_top_n_computation():
    rng = np.random.default_rng(5)
    rows, s_n = C.GRID_HOURS + 60, 6
    close = 100 * np.exp(np.cumsum(rng.normal(0, 2e-3, (rows, s_n)), axis=0))
    grid = {"present": np.ones((rows, s_n), dtype=bool), "close": close, "high": close * 1.002,
            "low": close * 0.998, "qvol": rng.lognormal(10, 1, (rows, s_n)), "funding": rng.normal(1e-4, 1e-5, (rows, s_n)),
            "oi_chg": rng.normal(0, 0.01, (rows, s_n)), "oiv_now": rng.lognormal(15, 1, (rows, s_n)),
            "oiv_prev": rng.lognormal(15, 1, (rows, s_n))}
    idx = np.arange(rows - 50, rows)
    a_sym, a_mkt = C.compute(grid, idx, 0, top_n=3)
    mask = C.members(grid, idx, 3)
    b_sym, b_mkt = C.compute(grid, idx, 0, 0, member_mask=mask)
    assert np.array_equal(a_sym, b_sym, equal_nan=True) and np.array_equal(a_mkt, b_mkt, equal_nan=True)
    assert np.array_equal(C.members(grid, idx, member_mask=mask), mask) and mask.sum(axis=1).max() == 3


def test_pit_stores_build_resume_and_refuse_other_modes(udb, tmp_path):
    cdb = tmp_path / "c.db"
    st = Store(cdb)
    st.save_candles("AAAUSDT", _candles("AAAUSDT", M("2024-01") + 14 * DAY, M("2024-01") + 15 * DAY, lambda t: 101.0))
    st.close()
    src = Store(cdb, readonly=True)
    dst = Store(tmp_path / "ctx.db")
    a, b = M("2024-02"), M("2024-02") + 3 * DAY - 1
    stats = C.build(src, dst, a, b, top_n=2, pit=str(udb))
    meta = dst.get("context.meta")
    assert meta["universe"] == ["*"] and meta["top_n"] == 2 and meta["pit"]["universe_db"].endswith("universe.db")
    assert stats["hours"] == 72 and stats["rows"] == 72 * 3  # the market row and two members an hour
    t = C.ContextTable.load(dst, ["AAAUSDT", "BBBUSDT", "BTCUSDT", "DDDUSDT"], a, b)
    assert len(t) == 72 and not t.symbols["DDDUSDT"] and len(t.symbols["BBBUSDT"]) == 72
    assert len(t.symbols["BTCUSDT"]) == 72 and not t.symbols["AAAUSDT"]  # AAA (1e6) is third
    v = t.view("BBBUSDT", a + 5 * H)
    # two members: ranks need MIN_XS = 3, so they are NaN; the symbol's own and BTC's market features are there
    assert v["n"] == 2.0 and np.isfinite(v["ret_24h"]) and np.isfinite(v["btc_ret_24h"]) and np.isnan(v["ret_24h_rank"])
    assert C.usable(src, dst, ["AAAUSDT"], 2, pit=str(udb))[:2] == (True, "same")
    assert C.ensure(src, dst, a, b, None, auto_build=False, top_n=2, pit=str(udb))
    # a 1m top_n store is another mode: refused both ways
    assert C.usable(src, dst, ["AAAUSDT"], 2)[:2] == (False, "mismatch")
    with pytest.raises(C.ContextMismatch):
        C.build(src, dst, a, b, ["AAAUSDT"], top_n=2)
    fixed = Store(tmp_path / "fixed.db")
    C.build(src, fixed, M("2024-01") + 14 * DAY, M("2024-01") + 15 * DAY - 1, ["AAAUSDT"], top_n=2)
    with pytest.raises(C.ContextMismatch):
        C.build(src, fixed, a, b, top_n=2, pit=str(udb))
    with pytest.raises(C.ContextMismatch, match="no universe database"):
        C.build(src, Store(tmp_path / "x.db"), a, b, top_n=2, pit=str(tmp_path / "missing.db"))
    with pytest.raises(ValueError):
        C.build(src, Store(tmp_path / "y.db"), a, b, top_n=0, pit=str(udb))
    # prepare() takes the universe.db path as cfg["pit"] (lab / discover pass it when CONTEXT_TOP_N > 0)
    path = C.prepare(str(cdb), {"db": str(tmp_path / "ctx2.db"), "top_n": 2, "pit": str(udb)}, a, b)
    assert path is not None and Store(path).get("context.meta")["universe"] == ["*"]
    for x in (src, dst, fixed):
        x.close()


def test_an_appended_universe_month_reopens_only_later_hours(udb, tmp_path):
    cdb = tmp_path / "c.db"
    Store(cdb).close()
    us = Store(udb, readonly=True)
    src = Store(cdb, readonly=True)
    before = C.pit_source_info(src, us, udb)
    us.close()
    make_udb(tmp_path / "universe.db", months=("2024-03",))  # one more month appended
    us = Store(udb, readonly=True)
    after = C.pit_source_info(src, us, udb)
    us.close()
    status, stable, why = C.source_status({"source": before}, after)
    assert status == "extended" and stable == before["ranges"]["@universe.db"]["futures_1h"][1]
    other = {**after, "universe_db": "/elsewhere/universe.db"}
    assert C.source_status({"source": before}, other)[0] == "mismatch"
    src.close()


def test_month_pieces_give_the_bits_of_one_grid_over_the_whole_window(udb, tmp_path):
    """compute_unit_pit works a calendar month at a time (memory: a 2025 year chose 511 symbols); a snapshot reads only
    GRID_HOURS rows before it and its members come from the mask, so the pieces equal one grid over the window."""
    cdb = tmp_path / "c.db"
    st = Store(cdb)
    st.save_candles("AAAUSDT", _candles("AAAUSDT", M("2024-01") + 25 * DAY, M("2024-02") + 5 * DAY,
                                        lambda t: 100.0 + (t // 60_000) % 37 / 10))
    st.close()
    src, us = Store(cdb, readonly=True), Store(udb, readonly=True)
    a, b = M("2024-01") + 18 * DAY, M("2024-02") + 10 * DAY - 1
    pieces = C.compute_unit_pit(src, us, a, b, 2)
    whole = C._compute_piece_pit(src, us, a, b, 2)
    assert len(pieces) == len(whole) == (b + 1 - a) // H
    # AAA is a member only until BBB overtakes it on 2024-01-20: the February grid lacks its column, the whole one not
    assert "AAAUSDT" in whole[0].symbols and "AAAUSDT" not in pieces[-1].symbols
    for p, w in zip(pieces, whole):
        assert p.ts == w.ts
        pr, wr = dict((s, blob) for s, _, blob in p.rows()), dict((s, blob) for s, _, blob in w.rows())
        assert pr == wr, p.ts  # the same members with the same bytes, market row included
    src.close()
    us.close()


def test_a_symbol_added_inside_the_stored_span_reopens_the_hours_it_changes(udb, tmp_path):
    """Adversarial review 2026-10-16: the ranking reads every eligible perpetual's 1h bars, so a perpetual whose blocks
    appear inside the span a store already holds (added to the inventory later) changes the members of those hours.
    The fingerprint held only universe.db's global span, which such an addition leaves unchanged: the store read as
    "same" and kept serving the old members. Each symbol's own span is tracked now."""
    cdb = tmp_path / "c.db"
    Store(cdb).close()
    src = Store(cdb, readonly=True)
    dst = Store(tmp_path / "ctx.db")
    a, b = M("2024-02"), M("2024-02") + 3 * DAY - 1
    C.build(src, dst, a, b, top_n=2, pit=str(udb))
    assert C.usable(src, dst, ["AAAUSDT"], 2, pit=str(udb))[:2] == (True, "same")
    us = Store(udb, readonly=True)
    before = C.pit_source_info(src, us, udb)
    us.close()
    # EEE: 9e6 an hour from 2024-02-01 12:00 on, inside the stored span (it outranks every member)
    st = Store(udb)
    st.save_symbol_inventory([_inv("EEEUSDT", "2024-02", "2024-02")])
    first = M("2024-02") + 12 * H
    rows = [r for r in _rows("2024-02", 3.0, lambda t: 9e6) if int(r[0]) >= first]
    blob, s = SI.encode_month(rows, "2024-02", [])
    st.save_kline_blocks(SI.FUT_TABLE, [("EEEUSDT", M("2024-02"), s["n"], s["first_ts"], s["last_ts"], s["first_trade"],
                                         s["last_trade"], s["med_dqv"], s["days_traded"], blob)])
    st.refresh_inventory_trades()
    st.close()
    us = Store(udb, readonly=True)
    after = C.pit_source_info(src, us, udb)
    assert C.pit_span(us) == tuple(before["ranges"]["@universe.db"]["futures_1h"])  # the global span did not move
    h = C.hour_of(first + 36 * H)
    syms, mask = C.pit_members(us, h, h, 2)
    assert "EEEUSDT" in [x for x, m in zip(syms, mask[0]) if m]  # the hour's members did change
    us.close()
    status, stable, why = C.source_status({"source": before}, after)
    assert status == "extended" and stable < first + H - 1 and "EEEUSDT" in why  # before its first hour close
    assert C.usable(src, dst, ["AAAUSDT"], 2, pit=str(udb))[1] != "same"
    # the next build re-opens those hours: EEE is a member once it has 24 h of bars, as a fresh build has it
    C.build(src, dst, a, b, top_n=2, pit=str(udb))
    fresh = Store(tmp_path / "fresh.db")
    C.build(src, fresh, a, b, top_n=2, pit=str(udb))
    names = ["AAAUSDT", "BBBUSDT", "BTCUSDT", "DDDUSDT", "EEEUSDT"]
    got, want = C.ContextTable.load(dst, names, a, b), C.ContextTable.load(fresh, names, a, b)
    assert len(got.symbols["EEEUSDT"]) == len(want.symbols["EEEUSDT"]) > 0
    assert {k: len(v) for k, v in got.symbols.items()} == {k: len(v) for k, v in want.symbols.items()}
    src.close()
    dst.close()
    fresh.close()

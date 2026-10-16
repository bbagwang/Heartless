"""F-pre-A two-leg neutral frame (heartless/learning/neutral.py) on synthetic series: the funding time convention,
liquidation detection before a rebalance, the rebalance arithmetic, cost application, daily equity continuity, the
7.4 statistics, delisting closes, membership and the ledger line."""
import math

import numpy as np
import pytest

from heartless.learning import neutral as N
from heartless.learning import seal, trials

H = 3_600_000
D = 86_400_000
T0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC (TRAIN-B)


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    path = tmp_path / "p2_trials.jsonl"
    monkeypatch.setattr(trials, "LEDGER", path)
    return path


class Scripted(N.NeutralStrategy):
    """Opens one carry position (spot long + perp short, `notional` each) at the close `enter`, closes it at `leave`;
    `rebal`: at that close, scale both legs by `frac`."""
    name = "scripted_carry"
    family = "carry_neutral"
    markets = ("perp", "spot")
    defaults = {"enter": 0, "leave": 0, "notional": 1000.0, "rebal": 0, "frac": 1.0, "symbol": "AUSDT"}

    def candidates(self, ctx, T, book):
        if T != self.params["enter"]:
            return []
        s, q = self.params["symbol"], self.params["notional"]
        return [N.Candidate(s, 1.0, [N.LegSpec(s, "spot", 1, q), N.LegSpec(s, "perp", -1, q)])]

    def exit_reason(self, ctx, T, pos):
        return "rule" if T == self.params["leave"] else None

    def rebalance(self, ctx, T, pos):
        if T != self.params["rebal"]:
            return None
        return [leg.qty * self.params["frac"] for leg in pos.legs]


def _ctx(n=72, perp=None, spot=None, funding=None, high=None, low=None):
    c = np.full(n, 100.0) if perp is None else np.asarray(perp, float)
    series = {("AUSDT", "perp"): {"close": c, "high": c if high is None else high, "low": c if low is None else low,
                                  "funding": np.full(n, np.nan) if funding is None else funding},
              ("AUSDT", "spot"): {"close": c.copy() if spot is None else np.asarray(spot, float)}}
    return N.synthetic_context(T0, series)


def _close(k):
    """The close of bar k (the bar opening at T0 + k h)."""
    return T0 + (k + 1) * H


def _run(strategy, ctx, capital=10_000.0, stress=False, end=None):
    return N.evaluate(strategy, T0, end if end is not None else ctx.end, ctx=ctx, capital=capital, stress=stress)


def test_costs_follow_the_table_and_the_flat_round_trip_is_about_41bp():
    """Constant prices: the only PnL is the cost of four fills (spot 10 bp + 3 bp, perp 5 bp + 1.5 bp, each with half
    of the 1 bp synthetic spread), i.e. the registration's ~39 bp plus the four half spreads."""
    ctx = _ctx()
    r = _run(Scripted(enter=_close(2), leave=_close(10)), ctx)
    c = N.LegCosts.of()
    assert c.fee == {"perp": 5e-4, "spot": 10e-4} and c.slippage == {"perp": 1.5e-4, "spot": 3e-4}
    buy_s, sell_s = c.fill("spot", 100.0, True), c.fill("spot", 100.0, False)
    sell_p, buy_p = c.fill("perp", 100.0, False), c.fill("perp", 100.0, True)
    qs, qp = 1000.0 / buy_s, 1000.0 / sell_p
    want = (-qs * buy_s * 10e-4 - qs * (buy_s - 100) + qs * (sell_s - 100) - qs * sell_s * 10e-4  # spot
            - qp * sell_p * 5e-4 + qp * (sell_p - 100) - qp * (buy_p - 100) - qp * buy_p * 5e-4)  # perp short
    total = sum(r.daily.values())
    assert total == pytest.approx(want, rel=1e-12, abs=1e-9)
    assert -total / 1000.0 * 1e4 == pytest.approx(41.0, abs=0.1)  # bp of one leg's notional
    assert r.positions[0]["reason"] == "rule" and r.positions[0]["pnl_pct"] == pytest.approx(want / 100, abs=1e-8)
    # stress: fees x1.5 and slippage x2
    s = N.LegCosts.of(stress=True)
    assert s.fee["perp"] == pytest.approx(7.5e-4) and s.slippage["spot"] == pytest.approx(6e-4)
    rs = _run(Scripted(enter=_close(2), leave=_close(10)), _ctx(), stress=True)
    assert sum(rs.daily.values()) < total
    with pytest.raises(ValueError):
        N.LegCosts.of(exec="maker")


def test_funding_time_convention():
    """Settlements on the opens of bars 3, 6 and 11 with the position entered at the close of bar 2 (= the open of
    bar 3) and closed at the close of bar 10 (= the open of bar 11): 3 and 6 are charged, 11 is not; each at the close
    before the settlement (a short receives a positive rate)."""
    n = 24
    perp = 100.0 + np.arange(n)  # a distinct close per bar shows which close is the mark
    f = np.full(n, np.nan)
    f[[3, 6, 11]] = [0.001, 0.002, 0.004]
    ctx = _ctx(n, perp=perp, spot=perp, funding=f)
    r = _run(Scripted(enter=_close(2), leave=_close(10)), ctx)
    leg_q = 1000.0 / N.LegCosts.of().fill("perp", perp[2], False)
    want = leg_q * (perp[2] * 0.001 + perp[5] * 0.002)
    assert r.funding == pytest.approx(want, abs=1e-8)
    assert r.positions[0]["legs"][1]["funding_pct"] == pytest.approx(want / 10_000 * 100, abs=1e-8)
    # a decision at the close of bar 2 cannot know the rate settled at the open of bar 3 (usable from its close)
    b = ctx.bars("AUSDT", "perp")
    assert math.isnan(b.funding_known()[2]) and b.funding_known()[3] == 0.001 and b.funding_known()[5] == 0.001


def test_liquidation_is_detected_inside_the_bar_before_the_rebalance():
    """2x short at 100: liquidation price 100 x 1.5 / 1.015. Bar 5 closes at 140 (a rebalance at that close would
    re-strike the margin) but its high reaches 148 first: the leg is liquidated, its whole margin lost, the spot leg
    closed at that close."""
    n = 12
    c = np.full(n, 100.0)
    c[5:] = 140.0
    hi = c.copy()
    hi[5] = 148.0
    ctx = _ctx(n, perp=c, spot=c, high=hi)
    liq = 100.0 * 1.5 / 1.015
    assert liq == pytest.approx(147.783, abs=1e-3)
    r = _run(Scripted(enter=_close(2), leave=_close(9), rebal=_close(5), frac=1.0), ctx)
    assert len(r.liquidations) == 1 and r.liquidations[0]["bar_close"] == N._iso(_close(5))
    assert r.liquidations[0]["liq"] == pytest.approx(liq, abs=1e-8)
    pos = r.positions[0]
    assert pos["reason"] == "liquidation" and pos["rebalances"] == 0
    perp_leg = pos["legs"][1]
    px = N.LegCosts.of().fill("perp", 100.0, False)
    q = 1000.0 / px
    # the leg's cash flows: the fee, the fill below the close, the margin (|q| x strike 100 / 2) that never comes back
    assert perp_leg["pnl_pct"] == pytest.approx((-q * px * 5e-4 + q * (px - 100.0) - q * 100.0 / 2) / 10_000 * 100,
                                                abs=1e-8)
    # the same path with the rebalance one bar earlier (close 140 >= 130 at bar 4): the margin is restored at that
    # close (strike 140, liquidation 140 x 1.5 / 1.015 = 206.9) and the 148 high of bar 5 liquidates nothing
    c2 = c.copy()
    c2[4] = 140.0
    r2 = _run(Scripted(enter=_close(2), leave=_close(9), rebal=_close(4), frac=1.0),
              _ctx(n, perp=c2, spot=c2, high=hi))
    assert r2.liquidations == [] and r2.positions[0]["reason"] == "rule" and r2.positions[0]["rebalances"] == 1


def test_rebalance_arithmetic_restores_the_margin_and_costs_only_the_traded_part():
    """At the close 130 (strike 100) the strategy cuts both legs by 30%: the perp leg is re-struck at 130 with margin
    = remaining notional / 2 and a new liquidation price; equity drops by exactly the cost of the two traded parts."""
    n = 10
    c = np.full(n, 100.0)
    c[4:] = 130.0
    ctx = _ctx(n, perp=c, spot=c)
    st = Scripted(enter=_close(1), leave=0, rebal=_close(4), frac=0.7)
    costs = N.LegCosts.of()
    book = N.Book(10_000.0, costs, 2.0, 0.015)
    pos = book.open_position(st.candidates(ctx, _close(1), book)[0], _close(1),
                             {("AUSDT", "spot"): 100.0, ("AUSDT", "perp"): 100.0})
    spot, perp = pos.legs
    prices = {("AUSDT", "spot"): 130.0, ("AUSDT", "perp"): 130.0}
    eq0, qs, qp = book.equity(prices), spot.qty, perp.qty
    assert perp.margin == pytest.approx(abs(qp) * 100.0 / 2) and perp.liq == pytest.approx(100 * 1.5 / 1.015)
    book.set_quantities(pos, [qs * 0.7, qp * 0.7], prices)
    d_s, d_p = qs * 0.3, abs(qp) * 0.3  # spot sold, perp bought back
    sell_s, buy_p = costs.fill("spot", 130.0, False), costs.fill("perp", 130.0, True)
    cost = d_s * sell_s * 10e-4 + d_s * (130 - sell_s) + d_p * buy_p * 5e-4 + d_p * (buy_p - 130)
    assert book.equity(prices) == pytest.approx(eq0 - cost, rel=1e-12)
    assert perp.entry == 130.0 and perp.qty == pytest.approx(qp * 0.7)
    assert perp.margin == pytest.approx(abs(perp.qty) * 130.0 / 2, abs=1e-8)
    assert perp.liq == pytest.approx(130.0 * 1.5 / 1.015, abs=1e-8)
    # an unchanged quantity still re-strikes (a pure margin top-up through the cash pool), at no cost
    cash, eq1 = book.cash, book.equity(prices)
    book.set_quantities(pos, [spot.qty, perp.qty], prices)
    assert book.equity(prices) == pytest.approx(eq1, rel=1e-12) and book.cash == pytest.approx(cash, rel=1e-12)
    with pytest.raises(ValueError, match="long only"):
        book.set_quantities(pos, [-1.0, perp.qty], prices)
    # the long side's liquidation price (a pairs leg)
    long_leg = N.Leg("B", "perp", 10.0, 100.0, margin=500.0)
    assert book.liquidation_price(long_leg) == pytest.approx((1000 - 500) / (10 * 0.985))


def test_daily_equity_is_continuous_and_marks_both_legs():
    """Three days, the position open across two day ends while the prices drift: equity[d] - equity[d-1] = daily[d],
    the days without a position are 0, the sum of the days is the final equity minus C, the day ends mark both
    legs."""
    n = 72
    perp = 100.0 * np.exp(np.linspace(0, 0.05, n))
    spot = perp * 0.999
    f = np.full(n, np.nan)
    f[8::8] = 0.0001
    ctx = _ctx(n, perp=perp, spot=spot, funding=f)
    r = _run(Scripted(enter=_close(10), leave=_close(40)), ctx, end=T0 + 3 * D - 1)
    days = sorted(r.daily)
    assert len(days) == 3 and days[0] == T0 // D
    eq = [r.equity[d] for d in days]
    assert eq[0] - 10_000.0 == pytest.approx(r.daily[days[0]], abs=1e-9)
    for a, b in zip(days, days[1:]):
        assert r.equity[b] - r.equity[a] == pytest.approx(r.daily[b], abs=1e-9)
    assert sum(r.daily.values()) == pytest.approx(eq[-1] - 10_000.0, abs=1e-9)
    assert r.daily[days[2]] == 0.0  # flat after the close at bar 40 (day 2)
    # the day-1 mark: spot qty x spot close + perp margin + unrealized, plus cash
    c = N.LegCosts.of()
    qs = 1000.0 / c.fill("spot", spot[10], True)
    assert r.positions[0]["legs"][0]["pnl_pct"] != 0.0 and qs > 0
    st = r.stats()
    v = np.array([r.daily[d] for d in days]) / 10_000 * 100
    assert st["days"] == 3 and st["mean_daily_pct"] == pytest.approx(v.mean(), abs=1e-6)
    assert st["annualized_pct"] == pytest.approx(v.mean() * 365, abs=1e-3)
    # the plain daily t is informational; the gating "t" is the HAC t (ROADMAP section 6, 2026-10-16)
    assert st["t_daily"] == pytest.approx(v.mean() / (v.std(ddof=1) / math.sqrt(3)), abs=1e-3)
    assert st["t"] == st["t_hac"] and st["hac_lag_days"] == 14


def test_statistics_years_quarters_drawdown_and_weekly_t():
    days = list(range(T0 // D - 3, T0 // D + 25))  # 2023-12-29 .. 2024-01-25
    pnl = {d: (1.0 if i % 3 else -2.5) for i, d in enumerate(days)}
    eq, e = {}, 100.0
    for d in days:
        e += pnl[d]
        eq[d] = e
    r = N.NeutralResult("x", "carry_neutral", {}, days[0] * D, (days[-1] + 1) * D - 1, 100.0, pnl, eq, [], [], 0.0,
                        0.0, 100.0, [], [])
    v = np.array([pnl[d] for d in days])
    st = r.stats()
    assert st["annualized_pct"] == pytest.approx(v.mean() * 365, abs=1e-3)
    curve = 100 + np.cumsum(v)
    peak = np.maximum.accumulate(np.concatenate(([100.0], curve)))[1:]
    assert st["max_drawdown_pct"] == pytest.approx(float(np.max((peak - curve) / peak)) * 100, abs=1e-4)
    assert r.by_year() == {"2023": pytest.approx(sum(v[:3])), "2024": pytest.approx(sum(v[3:]))}
    assert set(r.by_quarter()) == {"2023Q4", "2024Q1"}
    w = v[:28].reshape(4, 7).sum(axis=1)
    assert st["weekly_t"] == pytest.approx(w.mean() / (w.std(ddof=1) / 2), abs=1e-3)
    assert len(r.daily_series()) == len(days)


def test_delisting_closes_the_position_with_5x_slippage_on_that_leg():
    n = 20
    ctx = _ctx(n)
    ctx.last_bar[("AUSDT", "perp")] = 8
    ctx.delisted[("AUSDT", "perp")] = True
    r = _run(Scripted(enter=_close(2), leave=_close(15)), ctx)
    pos = r.positions[0]
    assert pos["reason"] == "delisting" and pos["closed"] == N._iso(_close(8))
    c = N.LegCosts.of()
    q = 1000.0 / c.fill("perp", 100.0, False)
    buy5 = c.fill("perp", 100.0, True, 5.0)
    assert buy5 == pytest.approx(100 * (1 + 0.5e-4) * (1 + 7.5e-4))
    perp_cash = -q * c.fill("perp", 100.0, False) * 5e-4 + q * (c.fill("perp", 100.0, False) - 100) \
        - q * buy5 * 5e-4 - q * (buy5 - 100)
    assert pos["legs"][1]["pnl_pct"] == pytest.approx(perp_cash / 10_000 * 100, rel=1e-9)


def test_an_entry_on_an_instruments_last_bar_is_taken_and_closed_when_the_next_bar_is_missing():
    """Regression (2026-10-16): the frame refused entries on an instrument's last traded bar, which only the bar that
    never comes can tell (future knowledge; ROADMAP 7.4: entries are never blocked in advance). The entry at that close
    is now the same as on an instrument that trades on; the next hour close finds the bar missing and closes the
    position there: the stopped leg at its last close (5x the slippage for a delisting), the other leg at that next
    close. A position opened before the last bar still closes at the last bar itself."""
    n = 20
    px = 100.0 + np.arange(n)  # a distinct close per bar shows which close each leg used

    def ctx_(last=None, delisted=False):
        ctx = _ctx(n, perp=px, spot=px)
        if last is not None:
            ctx.last_bar[("AUSDT", "perp")] = last
            ctx.delisted[("AUSDT", "perp")] = delisted
        return ctx

    # the decision at the close of bar 8 does not depend on whether bar 8 is the perp's last one
    live = _run(Scripted(enter=_close(8), leave=_close(12)), ctx_())
    dead = _run(Scripted(enter=_close(8), leave=_close(12)), ctx_(8, True))
    assert len(live.positions) == len(dead.positions) == 1
    assert live.positions[0]["opened"] == dead.positions[0]["opened"] == N._iso(_close(8))
    assert live.positions[0]["reason"] == "rule" and live.positions[0]["closed"] == N._iso(_close(12))
    pos = dead.positions[0]
    assert pos["reason"] == "delisting" and pos["closed"] == N._iso(_close(9))
    c = N.LegCosts.of()
    # perp short sold and bought back at bar 8's close (its last), the buy with 5x the slippage
    sell, buy5 = c.fill("perp", px[8], False), c.fill("perp", px[8], True, 5.0)
    q = 1000.0 / sell
    perp_cash = -q * sell * 5e-4 + q * (sell - px[8]) - q * buy5 * 5e-4 - q * (buy5 - px[8])
    assert pos["legs"][1]["pnl_pct"] == pytest.approx(perp_cash / 10_000 * 100, rel=1e-9)
    # spot long bought at bar 8's close and sold at bar 9's (the hour the stop shows), nominal costs
    buy, sold = c.fill("spot", px[8], True), c.fill("spot", px[9], False)
    qs = 1000.0 / buy
    spot_cash = -qs * buy * 10e-4 - qs * buy + qs * sold - qs * sold * 10e-4
    assert pos["legs"][0]["pnl_pct"] == pytest.approx(spot_cash / 10_000 * 100, rel=1e-9)
    assert sum(dead.daily.values()) == pytest.approx(perp_cash + spot_cash, rel=1e-9)
    # a position opened before the last bar closes at the last bar itself, as before
    early = _run(Scripted(enter=_close(2), leave=_close(12)), ctx_(8, True)).positions
    assert [(p["reason"], p["closed"]) for p in early] == [("delisting", N._iso(_close(8)))]
    # stored data that just stops (no delisting): the same timing at nominal costs
    de = _run(Scripted(enter=_close(8), leave=_close(12)), ctx_(8, False)).positions
    assert [(p["reason"], p["closed"]) for p in de] == [("data_end", N._iso(_close(9)))]
    # after the last bar there is no bar at T: nothing is entered
    assert _run(Scripted(enter=_close(9), leave=_close(12)), ctx_(8, True)).positions == []


def test_a_leg_whose_data_ends_early_closes_the_position_there():
    """Review fix: the spot series stops after bar 39 without a delisting (a market built to an earlier end, or a
    delisting inside the inventory's 3-day slack) while the perp trades on and rises 20%. The position used to stay
    open with a frozen spot price, booking an unhedged perp loss; it now closes at the spot's last bar ("data_end",
    nominal costs) and the later perp move does not reach the book."""
    n = 96
    perp = np.full(n, 100.0)
    perp[50:] = 120.0
    ctx = N.synthetic_context(T0, {("AUSDT", "perp"): {"close": perp, "funding": np.full(n, np.nan)},
                                   ("AUSDT", "spot"): {"close": np.full(40, 100.0)}}, end=T0 + n * H - 1)
    r = N.evaluate(Scripted(enter=_close(10)), T0, T0 + n * H - 1, ctx=ctx, capital=10_000.0)
    pos = r.positions[0]
    assert pos["reason"] == "data_end" and pos["closed"] == N._iso(_close(39))
    flat = _run(Scripted(enter=_close(10), leave=_close(39)), _ctx(n=96))  # the same round trip at constant prices
    assert r.stats()["total_pct"] == pytest.approx(flat.stats()["total_pct"], abs=1e-9)
    # the scripted strategy enters once: one position
    assert len(r.positions) == 1
    # visible in the statistics and the report (a lagging market build is not a silently flat tail)
    assert r.stats()["data_end_closes"] == 1 and "data_end=1" in r.report()
    assert flat.stats()["data_end_closes"] == 0 and "data_end" not in flat.report()


def test_entries_only_for_members_and_positions_survive_membership_loss():
    """Members: AUSDT in January only. An entry proposed in February is refused; a January position stays open into
    February until its own exit."""
    n = 24 * 40
    ctx = _ctx(n)
    ctx.members_by_month = {"2024-01": {"AUSDT"}, "2024-02": set()}
    feb = T0 + 31 * D
    r = _run(Scripted(enter=feb + 5 * H, leave=feb + 20 * H), ctx)
    assert r.positions == []
    r = _run(Scripted(enter=feb - 5 * H, leave=feb + 20 * H), ctx)
    assert r.positions[0]["reason"] == "rule" and r.positions[0]["closed"] == N._iso(feb + 20 * H)


def test_ranking_fills_free_slots_by_score():
    class Many(N.NeutralStrategy):
        name, markets, max_positions = "many", ("perp",), 2

        def candidates(self, ctx, T, book):
            if T != _close(1):
                return []
            return [N.Candidate(s, sc, [N.LegSpec(s, "perp", -1, 100.0), N.LegSpec(o, "perp", 1, 100.0)])
                    for s, o, sc in (("AUSDT", "BUSDT", 0.1), ("BUSDT", "AUSDT", 0.9), ("CUSDT", "AUSDT", 0.5))]

    c = {"close": np.full(10, 50.0)}
    ctx = N.synthetic_context(T0, {(s, "perp"): dict(c) for s in ("AUSDT", "BUSDT", "CUSDT")})
    r = N.evaluate(Many(), T0, ctx.end, ctx=ctx, capital=1000.0)
    assert sorted(p["key"] for p in r.positions) == ["BUSDT", "CUSDT"]
    assert all(p["reason"] == "end" for p in r.positions)


def test_ledger_line_and_reruns_do_not_raise_n(monkeypatch):
    ctx = _ctx()
    st = Scripted(enter=_close(2), leave=_close(10))
    r = _run(st, ctx)
    e = N.record(r)
    assert e["tool"] == "neutral" and e["family"] == "carry_neutral" and e["wave_f"] == "TRAIN-B"
    assert e["exec"] == "taker" and e["neutral"]["liquidations"] == 0 and e["n"] == 1
    assert e["t_day"] == round(r.stats()["t_hac"], 3) and {"t_hac", "t_daily", "weekly_t"} <= set(e["neutral"])
    assert "window" not in e["config"] and e["config"]["leverage"] == 2.0
    rs = _run(Scripted(enter=_close(2), leave=_close(10)), _ctx(), stress=True)
    N.record(rs)
    rv = N.evaluate(Scripted(enter=_close(2), leave=_close(10)), trials.WAVE_F["VALID-F"][0], trials.WAVE_F["VALID-F"][0]
                    + 3 * D - 1, ctx=N.synthetic_context(trials.WAVE_F["VALID-F"][0], {
                        ("AUSDT", "perp"): {"close": np.full(72, 100.0)}, ("AUSDT", "spot"): {"close": np.full(72, 100.0)}}))
    assert N.record(rv)["wave_f"] == "VALID-F"
    assert trials.family_tests("carry_neutral") == 1  # stress and VALID reruns of one configuration
    N.record(_run(Scripted(enter=_close(3), leave=_close(10)), _ctx()))
    assert trials.family_tests("carry_neutral") == 2
    # a sealed window is refused before anything runs
    monkeypatch.setattr(seal, "SEALED", {"FWD-02": trials.fwd_span(2)})
    with pytest.raises(seal.SealedError):
        N.evaluate(st, trials.fwd_span(2)[0], trials.fwd_span(2)[0] + D, ctx=ctx)


def test_unknown_parameters_are_refused():
    with pytest.raises(ValueError, match="unknown parameter"):
        Scripted(nope=1)


# --- loading from universe.db and the CLI --------------------------------------------------------------------------------
class PerpPair(N.NeutralStrategy):
    """Short AUSDT / long the spot leg of 1000XUSDT for 10 hours from the first decision (a loader test, no rule)."""
    name = "perp_pair_test"
    family = "pairs_neutral"
    markets = ("perp", "spot")
    defaults = {"hours": 10}

    def candidates(self, ctx, T, book):
        if book.closed or book.open:
            return []
        return [N.Candidate("X", 1.0, [N.LegSpec("1000XUSDT", "spot", 1, 100.0), N.LegSpec("AUSDT", "perp", -1, 100.0)])]

    def exit_reason(self, ctx, T, pos):
        return "hold" if T - pos.opened >= self.params["hours"] * H else None


def _spot_block(symbol, month, closes, start):
    from heartless.data import symbols_inventory as SI

    rows = [[str(start + i * H), "0", "0", "0", f"{c:.6f}", "1", str(start + i * H + H - 1), "1000.0", "1", "0", "0",
             "0"] for i, c in enumerate(closes)]
    blob, s = SI.encode_month(rows, month, spot=True)
    return (symbol, SI._month_ms(month), s["n"], s["first_ts"], s["last_ts"], s["first_trade"], s["last_trade"],
            s["med_dqv"], s["days_traded"], blob)


@pytest.fixture
def neutral_db(tmp_path):
    from heartless.core.store import Store
    from heartless.data import symbols_inventory as SI
    from tests.test_f0_universe import _inv
    from tests.test_fpreA_hourly import make_db

    n = 24 * 31
    qv = np.full(24 * 10, 1e6)
    qv[41:] = 0.0  # BUSDT stops trading after bar 40 and is delisted (the store runs on with AUSDT)
    path = tmp_path / "u.db"
    make_db(path, {"AUSDT": {"start": T0, "c": np.full(n, 100.0)}, "1000XUSDT": {"start": T0, "c": np.full(n, 100.0)},
                   "BUSDT": {"start": T0, "c": np.full(24 * 10, 50.0), "qv": qv}})
    st = Store(path)
    row = _inv("1000XUSDT", "2023-11", "2024-03", spot="XUSDT")
    st.save_symbol_inventory([row])
    st.save_kline_blocks(SI.SPOT_TABLE, [_spot_block("XUSDT", "2024-01", np.full(n, 0.1), T0)])
    st.refresh_inventory_trades()
    st.close()
    return str(path)


def test_load_context_scales_spot_and_finds_delistings(neutral_db):
    st = N.ReadOnlyStore(neutral_db)
    try:
        ctx, skipped, _ = N.load_context(st, T0 + D, T0 + 20 * D - 1, ("perp", "spot"), 24,
                                         symbols=["AUSDT", "1000XUSDT", "BUSDT"])
    finally:
        st.close()
    assert ctx.close("1000XUSDT", "spot", T0 + 2 * D) == pytest.approx(100.0)  # 0.1 x the contract multiplier 1000
    assert ctx.spot_symbol == {"1000XUSDT": "XUSDT"} and skipped == []
    b = ctx.bars("BUSDT", "perp")
    assert ctx.delisted[("BUSDT", "perp")] and int(b.t[ctx.last_bar[("BUSDT", "perp")]]) == T0 + 40 * H
    assert not ctx.delisted[("AUSDT", "perp")] and ctx.idx("BUSDT", "perp", T0 + 42 * H) is None
    assert ("AUSDT", "spot") not in ctx.series  # no spot market


def test_neutral_cli_runs_a_strategy_and_records_it(neutral_db, tmp_path, capsys):
    import json
    from types import SimpleNamespace

    from heartless.config import Settings

    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=neutral_db)
    out = tmp_path / "n.json"
    args = SimpleNamespace(strategy="tests.test_fpreA_neutral:PerpPair", set=["hours=5"], period=None,
                           start="2024-01-02", end="2024-01-05", universe=None, symbols="AUSDT,1000XUSDT",
                           capital=1000.0, stress=False, tag="pairs_neutral", out=str(out),
                           argv=["heartless", "neutral"])
    N.main_cli(settings, args)
    doc = json.loads(out.read_text())
    assert doc["evaluator"] == "neutral" and doc["config"]["params"] == {"hours": 5}
    assert doc["trial"]["tool"] == "neutral" and doc["wave_f_period"] == "TRAIN-B"
    assert doc["stats"]["days"] == 3 and doc["positions"][0]["reason"] == "hold"
    assert "trial ledger: pairs_neutral" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        N.main_cli(settings, SimpleNamespace(**{**vars(args), "strategy": "tests.test_fpreA_hourly:Plan"}))

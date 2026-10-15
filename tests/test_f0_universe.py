"""F0 universes by rule: the survivorship-free research universe (monthly liquidity ranks from the previous month,
delisted symbols kept, topN for lab / discover with skipped symbols reported) and the live selection (listing age,
volume, tick, spread; ALWAYS_INCLUDE as a data subscription)."""
import json
from types import SimpleNamespace

import pytest

from heartless.config import Settings
from heartless.core.store import Store
from heartless.data import symbols_inventory as SI
from heartless.data import universe as U
from heartless.exchange.symbols import parse_exchange_info
from heartless.util.timeutil import MS_DAY, now_ms
from tests.test_f0_inventory import _zip, kline_csv

M = SI._month_ms


def _inv(symbol, first_month, last_month, kind="perpetual", quote="USDT", asset=None, spot=None):
    p = SI.parse_symbol(symbol)
    return {"symbol": symbol, "kind": kind, "quote": quote, "base": p["base"], "asset": asset or p["asset"],
            "multiplier": p["multiplier"], "first_month": first_month, "last_month": last_month, "months": 0,
            "missing_months": [], "spot_symbol": spot, "spot_first_month": None, "spot_last_month": None,
            "spot_months": 0, "updated": 0}


def _block(symbol, month, qv, days=None, zero=False):
    rows = SI.kline_rows(_zip(kline_csv(month, 10.0, qv, days=days, zero=zero)), month)
    blob, s = SI.encode_month(rows, month)
    return (symbol, M(month), s["n"], s["first_ts"], s["last_ts"], s["first_trade"], s["last_trade"], s["med_dqv"],
            s["days_traded"], blob)


def make_universe_db(path) -> Store:
    """Hourly volumes per symbol and month (quote volume per hour):
    BIG    2023-10..2024-03  1e7          always first
    MID    2023-10..2024-03  1e6 (2e7 in 2024-01: ranks first in 2024-02 only)
    GONE   2023-10..2023-12  5e6, last trade 2023-12-10, flat files after (delisted)
    NEW    listed 2024-01-05 with 9e7: January's median is high, but on 2024-02-01 it is 27 days old (< 30), so it
           ranks only from 2024-03
    STBL   a stablecoin perpetual (USDCUSDT) with the largest volume; never ranked
    COPY   a settled archive copy with volume; never ranked
    """
    st = Store(path)
    st.save_symbol_inventory([
        _inv("BIGUSDT", "2023-10", "2024-03", spot="BIGUSDT"), _inv("MIDUSDT", "2023-10", "2024-03"),
        _inv("GONEUSDT", "2023-10", "2024-03"), _inv("NEWUSDT", "2024-01", "2024-03"),
        _inv("USDCUSDT", "2023-10", "2024-03"), _inv("BIGUSDTSETTLED", "2023-10", "2024-03", kind="settled"),
        _inv("BIGUSDC", "2023-10", "2024-03", quote="USDC")])
    blocks = []
    for m in ("2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"):
        blocks += [_block("BIGUSDT", m, 1e7), _block("MIDUSDT", m, 2e7 if m == "2024-01" else 1e6),
                   _block("USDCUSDT", m, 1e9), _block("BIGUSDTSETTLED", m, 5e8), _block("BIGUSDC", m, 5e8)]
        if m < "2023-12":
            blocks.append(_block("GONEUSDT", m, 5e6))
        elif m == "2023-12":
            blocks.append(_block("GONEUSDT", m, 5e6, days=range(1, 11)))
        else:
            blocks.append(_block("GONEUSDT", m, 0, zero=True))
    blocks += [_block("NEWUSDT", "2024-01", 9e7, days=range(5, 32)), _block("NEWUSDT", "2024-02", 9e7),
               _block("NEWUSDT", "2024-03", 9e7)]
    st.save_kline_blocks(SI.FUT_TABLE, blocks)
    st.refresh_inventory_trades()
    return st


@pytest.fixture
def udb(tmp_path):
    st = make_universe_db(tmp_path / "universe.db")
    yield st
    st.close()


def _syms(ranked):
    return [r["symbol"] for r in ranked]


def test_ranking_reads_only_the_previous_month(udb):
    assert _syms(U.month_ranking(udb, "2023-12")) == ["BIGUSDT", "GONEUSDT", "MIDUSDT"]
    assert _syms(U.month_ranking(udb, "2024-01")) == ["BIGUSDT", "MIDUSDT"]  # GONE stopped on 12-10: median 0
    assert _syms(U.month_ranking(udb, "2024-02")) == ["MIDUSDT", "BIGUSDT"]  # MID's January volume, used in February
    assert _syms(U.month_ranking(udb, "2024-03")) == ["NEWUSDT", "BIGUSDT", "MIDUSDT"]
    # a change inside month m never changes the ranking of m (no look-ahead), a change in m-1 does
    before = U.month_ranking(udb, "2024-02")
    udb.save_kline_blocks(SI.FUT_TABLE, [_block("BIGUSDT", "2024-02", 1e9)])
    assert U.month_ranking(udb, "2024-02") == before
    udb.save_kline_blocks(SI.FUT_TABLE, [_block("BIGUSDT", "2024-01", 1e9)])
    assert _syms(U.month_ranking(udb, "2024-02"))[0] == "BIGUSDT"


def test_listing_age_and_instrument_rules(udb):
    inv = {r["symbol"]: r for r in udb.load_symbol_inventory()}
    assert U.listed_at(inv["NEWUSDT"]) == M("2024-01") + 4 * MS_DAY  # first traded hour inside the stored months
    assert U.listed_at(inv["BIGUSDT"]) == M("2023-10")  # listed before the stored data: start of its first month
    # NEW traded most of January but was listed 2024-01-05: 27 days before 2024-02-01 -> not ranked in February
    assert "NEWUSDT" not in _syms(U.month_ranking(udb, "2024-02"))
    assert _syms(U.month_ranking(udb, "2024-02", min_age_days=5))[0] == "NEWUSDT"
    for s in ("USDCUSDT", "BIGUSDTSETTLED", "BIGUSDC"):
        assert U.eligible_research(inv[s]) is not None and all(s not in _syms(U.month_ranking(udb, m))
                                                                for m in ("2023-11", "2024-03"))


def test_research_universe_keeps_delisted_members_and_reports_unranked_months(udb):
    res = U.research_universe(udb, M("2023-11"), M("2024-04") - 1, top=2)
    assert res["months"] == {"2023-11": ["BIGUSDT", "GONEUSDT"], "2023-12": ["BIGUSDT", "GONEUSDT"],
                             "2024-01": ["BIGUSDT", "MIDUSDT"], "2024-02": ["MIDUSDT", "BIGUSDT"],
                             "2024-03": ["NEWUSDT", "BIGUSDT"]}
    assert set(res["symbols"]) == {"BIGUSDT", "GONEUSDT", "MIDUSDT", "NEWUSDT"}
    assert res["members"]["GONEUSDT"] == ["2023-11", "2023-12"] and res["unranked"] == []
    assert res["last_trade"]["GONEUSDT"] < M("2023-12") + 10 * MS_DAY  # the delisting stays visible
    assert U.research_universe(udb, M("2023-10"), M("2023-11") - 1, 2)["unranked"] == ["2023-10"]


def test_resolve_for_run_reports_symbols_without_1m_candles(udb, tmp_path):
    from heartless.core.models import Candle

    cdb = tmp_path / "candles.db"
    cs = Store(cdb)
    cs.save_candles("BIGUSDT", [Candle(M("2024-01"), 1, 1, 1, 1, 1, 1, 1, 1, M("2024-01") + 59_999)])
    # GONE has 1m candles, but only before the window: it would only come back as a lab error, so it is skipped
    cs.save_candles("GONEUSDT", [Candle(M("2023-10"), 1, 1, 1, 1, 1, 1, 1, 1, M("2023-10") + 59_999)])
    cs.close()
    res = U.resolve_for_run(udb.path, cdb, M("2023-12"), M("2024-02") - 1, 2)
    assert res["research"] == ["BIGUSDT", "GONEUSDT", "MIDUSDT"] or set(res["research"]) == {"BIGUSDT", "GONEUSDT",
                                                                                             "MIDUSDT"}
    assert res["symbols"] == ["BIGUSDT"] and res["skipped"] == ["GONEUSDT", "MIDUSDT"]
    assert res["windows"] == {"BIGUSDT": [(M("2023-12"), M("2024-02") - 1)]} and res["member_share"] == 1.0
    with pytest.raises(ValueError, match="cannot be ranked"):
        U.resolve_for_run(udb.path, cdb, M("2023-10"), M("2023-12") - 1, 2)
    with pytest.raises(ValueError, match="no universe database"):
        U.resolve_for_run(tmp_path / "missing.db", cdb, M("2023-12"), M("2024-02") - 1, 2)
    assert U.parse_top("top40") == 40 and U.parse_top("core") is None
    with pytest.raises(ValueError):
        U.parse_top("top0")


def test_universe_cli_show_resolve_status(udb, tmp_path, monkeypatch, capsys):
    from heartless import cli

    monkeypatch.setenv("HEARTLESS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("UNIVERSE_DB", str(udb.path))
    cli.main(["universe", "show", "--month", "2024-03", "--top", "2"])
    out = capsys.readouterr().out
    assert "NEWUSDT" in out and "BIGUSDT" in out and "MIDUSDT" not in out
    cli.main(["universe", "show", "--month", "2024-01", "--top", "5"])
    assert "GONEUSDT" not in capsys.readouterr().out
    res_file = tmp_path / "res.json"
    cli.main(["universe", "resolve", "--start", "2023-11-01", "--end", "2024-04-01", "--top", "2", "--out",
              str(res_file)])
    out = capsys.readouterr().out
    assert "4 symbols" in out and "(1 stopped trading inside the window)" in out
    assert json.loads(res_file.read_text())["members"]["NEWUSDT"] == ["2024-03"]
    cli.main(["universe", "status"])
    assert "klines_1h: 7 symbols" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["universe", "show", "--month", "2023-10"])  # the previous month is not built


def test_lab_universe_topn_runs_the_stored_members_and_records_the_skipped(udb, tmp_path, monkeypatch, capsys):
    """lab --universe top2 over 2023-12-29 .. 2024-01-03: members BIG, GONE (December) and BIG, MID (January); only BIG
    has 1m candles, so the run evaluates BIG and the result file lists GONE and MID as skipped."""
    from heartless import cli
    from heartless.learning import lab
    from tests.test_p0_lab import INLINE, Timer, _ms
    from synth import synth_candles
    from heartless.strategy.ensemble import Ensemble
    from heartless.strategy.params import StrategyParams

    orig = Ensemble.__init__
    monkeypatch.setattr(Ensemble, "__init__", lambda self, params, bandit, alphas=None, only_alpha=None:
                        orig(self, params, bandit, [Timer()], only_alpha))
    db = tmp_path / "t.db"
    st = Store(db)
    ca = synth_candles(6 * 1440, seed=3, start_ms=_ms(2023, 12, 28))
    st.save_candles("BIGUSDT", [ca.candle_at(i) for i in range(ca.n)])
    st.close()
    params = StrategyParams.default()
    params.enabled["trend_pullback"] = True
    pfile = tmp_path / "p.json"
    pfile.write_text(json.dumps(params.to_dict()))
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=str(udb.path),
                        CONTEXT_AUTO_BUILD=False)
    monkeypatch.setattr(settings.__class__, "db_path", property(lambda self: db))
    real = lab.evaluate
    monkeypatch.setattr(lab, "evaluate", lambda *a, **k: real(*a, **{**k, "pool": INLINE, "warmup_ms": MS_DAY}))
    monkeypatch.setattr(lab, "UNIVERSES", dict(lab.UNIVERSES))
    out = tmp_path / "r.json"
    args = SimpleNamespace(symbols=None, universe="top2", years=None, start="2023-12-29", end="2024-01-03",
                           split="train", unseal=[], params=str(pfile), set=[], stress=False, meta=False,
                           chunk_days=None, alpha=None, workers=1, json=False, trades_out=None, out=str(out),
                           by_side=False, argv=["heartless", "lab", "--universe", "top2"])
    top = cli._top_universe(settings, args, "lab")
    lab.main_cli(settings, args)
    cli._note_top_universe(args, top)
    assert "skipped 2 without 1m candles: GONEUSDT, MIDUSDT" in capsys.readouterr().err
    doc = json.loads(out.read_text())
    assert doc["universe"] == "top2" and doc["symbols"] == ["BIGUSDT"]
    assert doc["universe_resolution"]["skipped"] == ["GONEUSDT", "MIDUSDT"]
    assert set(doc["universe_resolution"]["research"]) == {"BIGUSDT", "GONEUSDT", "MIDUSDT"}
    # the membership spans travel with the result: BIG was top 2 in December and January, clipped to the window
    assert doc["universe_resolution"]["windows"] == {"BIGUSDT": [[lab.parse_when("2023-12-29"),
                                                                  lab.parse_when("2024-01-03", end=True)]]}
    assert doc["universe_resolution"]["members"]["GONEUSDT"] == ["2023-12"]
    # a topN universe needs the window up front
    with pytest.raises(SystemExit):
        cli._top_universe(settings, SimpleNamespace(universe="top2", years=None, start=None, end=None), "lab")
    assert cli._universe_name("TOP40") == "top40" and cli._universe_name("extended") == "extended"


# --- live selection -----------------------------------------------------------------------------------------------------
OLD = now_ms() - 400 * MS_DAY


def _sym(symbol, base, tick="0.01", onboard=OLD, status="TRADING", contract="PERPETUAL", quote="USDT"):
    return {"symbol": symbol, "baseAsset": base, "quoteAsset": quote, "pricePrecision": 2, "quantityPrecision": 3,
            "status": status, "contractType": contract, "onboardDate": onboard,
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": tick},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "notional": "5"}]}


def _live():
    infos = parse_exchange_info({"symbols": [
        _sym("BTCUSDT", "BTC"), _sym("ETHUSDT", "ETH"), _sym("AAAUSDT", "AAA"),
        _sym("YOUNGUSDT", "YOUNG", onboard=now_ms() - 20 * MS_DAY),  # 20 days: under the 30-day rule
        _sym("COARSEUSDT", "COARSE", tick="0.01"),  # price 1.0: a tick is 100 bp
        _sym("WIDEUSDT", "WIDE"), _sym("USDCUSDT", "USDC"), _sym("THINUSDT", "THIN"),
    ]})
    tickers = [{"symbol": s, "quoteVolume": str(v), "lastPrice": str(p)} for s, v, p in (
        ("BTCUSDT", 9e9, 60000), ("ETHUSDT", 5e9, 3000), ("AAAUSDT", 3e8, 20), ("YOUNGUSDT", 8e9, 5),
        ("COARSEUSDT", 7e9, 1.0), ("WIDEUSDT", 6e9, 50), ("USDCUSDT", 9e9, 1.0), ("THINUSDT", 1e6, 3))]
    book = [{"symbol": s, "bidPrice": str(b), "askPrice": str(a)} for s, b, a in (
        ("BTCUSDT", 60000, 60000.1), ("ETHUSDT", 3000, 3000.01), ("AAAUSDT", 20, 20.002), ("WIDEUSDT", 50, 50.1))]
    return infos, tickers, book


def test_live_selection_rules_are_about_the_symbol_not_its_name():
    infos, tickers, book = _live()
    sel = U.select_live_universe(infos, tickers, 10, 1e8, reference=["BTCUSDT"], book_tickers=book)
    assert sel.trade == ["BTCUSDT", "ETHUSDT", "AAAUSDT"]
    assert sel.rejected["YOUNGUSDT"].startswith("listed 20.0d ago")
    assert sel.rejected["COARSEUSDT"].startswith("tick 100.0bp") and sel.rejected["WIDEUSDT"].startswith("spread 20.0bp")
    assert sel.rejected["USDCUSDT"] == "stablecoin" and sel.rejected["THINUSDT"].startswith("24h quote volume")
    # without book tickers the spread rule cannot be applied (the app does not fetch them yet)
    assert "WIDEUSDT" in U.select_live_universe(infos, tickers, 10, 1e8).trade


def test_always_include_is_a_data_subscription_not_a_trading_guarantee():
    infos, tickers, book = _live()
    sel = U.select_live_universe(infos, tickers, 10, 1e8, reference=["THINUSDT", "BTCUSDT", "MISSINGUSDT"])
    assert sel.reference == ["THINUSDT", "BTCUSDT"] and "THINUSDT" not in sel.trade
    assert sel.subscribed[:2] == ["THINUSDT", "BTCUSDT"] and "THINUSDT" in sel.subscribed
    # the list interface app.py uses: reference symbols first (subscribed), capped at the universe size
    assert U.select_universe(infos, tickers, 3, 1e8, ["THINUSDT"]) == ["THINUSDT", "BTCUSDT", "WIDEUSDT"]


def test_live_selection_from_settings():
    infos, tickers, book = _live()
    s = Settings(_env_file=None, UNIVERSE_SIZE=2, UNIVERSE_MIN_QUOTE_VOLUME=1e8, UNIVERSE_MIN_AGE_DAYS=10,
                 UNIVERSE_MAX_TICK_BPS=200)
    sel = U.live_selection(s, infos, tickers, book)
    assert sel.trade == ["BTCUSDT", "YOUNGUSDT"] and sel.reference == ["BTCUSDT"]
    assert Settings(_env_file=None).always_include_list == ["BTCUSDT"]


# --- review: a relisting is aged from its new first trade (live: the exchange's new onboardDate) ---------------------------
def test_relisting_counts_the_listing_age_from_the_new_first_trade(tmp_path):
    """BIG trades throughout. BACK (listed 2023-10) stops after November, flat December file, trades again from
    2024-01-10 with the largest volume: 22 days old on 2024-02-01 (the PUMPUSDT 2025-08 case), ranked from 2024-03.
    INSIDE pauses inside October (trades on the 1st..3rd and from the 20th): 12 days old on 2023-11-01. OLD was listed
    before the stored months (first archive month 2023-01), its first stored months are flat files, it trades from
    2024-01-08: a return, not a fresh listing of 2023-01."""
    st = Store(tmp_path / "u.db")
    st.save_symbol_inventory([_inv("BIGUSDT", "2023-10", "2024-03"), _inv("BACKUSDT", "2023-10", "2024-03"),
                              _inv("INSIDEUSDT", "2023-10", "2024-03"), _inv("OLDUSDT", "2023-01", "2024-03")])
    blocks = []
    for m in ("2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03"):
        blocks.append(_block("BIGUSDT", m, 1e6))
        blocks.append(_block("BACKUSDT", m, 9e7, days=range(10, 32) if m == "2024-01" else None,
                             zero=m == "2023-12"))
        blocks.append(_block("INSIDEUSDT", m, 5e5, days=[1, 2, 3, *range(20, 32)] if m == "2023-10" else None))
        blocks.append(_block("OLDUSDT", m, 8e7, days=range(8, 32) if m == "2024-01" else None, zero=m < "2024-01"))
    st.save_kline_blocks(SI.FUT_TABLE, blocks)
    st.refresh_inventory_trades()
    rel = U.relistings(st)
    assert rel == {"BACKUSDT": [M("2024-01") + 9 * MS_DAY], "INSIDEUSDT": [M("2023-10") + 19 * MS_DAY],
                   "OLDUSDT": [M("2024-01") + 7 * MS_DAY]}
    assert "INSIDEUSDT" not in _syms(U.month_ranking(st, "2023-11"))
    assert "INSIDEUSDT" in _syms(U.month_ranking(st, "2023-12"))
    feb = _syms(U.month_ranking(st, "2024-02"))
    assert "BACKUSDT" not in feb and "OLDUSDT" not in feb and feb[0] == "BIGUSDT"
    assert _syms(U.month_ranking(st, "2024-03"))[:2] == ["BACKUSDT", "OLDUSDT"]
    # without the relisting rule BACK would have ranked first in February on its original 2023-10 listing
    assert _syms(U.month_ranking(st, "2024-02", relisted={}))[0] == "BACKUSDT"
    inv = {r["symbol"]: r for r in st.load_symbol_inventory()}
    assert U.listed_as_of(inv["BACKUSDT"], M("2024-02"), rel) == M("2024-01") + 9 * MS_DAY
    assert U.listed_as_of(inv["BACKUSDT"], M("2024-01"), rel) == M("2023-10")  # not yet known at the start of January
    st.close()


# --- review: the union of monthly members carries a look-ahead outside each member's months ---------------------------
def test_membership_windows_and_member_trades():
    members = {"AUSDT": ["2024-01", "2024-02", "2024-04"], "BUSDT": ["2024-03"]}
    w = U.membership_windows(members, M("2024-01") + 5 * MS_DAY, M("2024-05") - 1)
    assert w == {"AUSDT": [(M("2024-01") + 5 * MS_DAY, M("2024-03") - 1), (M("2024-04"), M("2024-05") - 1)],
                 "BUSDT": [(M("2024-03"), M("2024-04") - 1)]}
    trades = [{"symbol": "AUSDT", "entry_time": M("2024-02") + 1}, {"symbol": "AUSDT", "entry_time": M("2024-03") + 1},
              {"symbol": "BUSDT", "entry_time": M("2024-03") + 1}, {"symbol": "BUSDT", "entry_time": M("2024-02")},
              {"symbol": "CUSDT", "entry_time": M("2024-03")}]
    # A in March (not a member) and B in February (a member only from March: picked by its February surge) drop out
    assert U.member_trades(trades, w) == [trades[0], trades[2]]


def test_discover_resolves_the_universe_over_train_only(udb, tmp_path, monkeypatch, capsys):
    """VALID must not choose the symbols mined on TRAIN: with TRAIN = December 2023 the top-2 universe is BIG and GONE,
    although NEW (top in March 2024, inside VALID) would join a TRAIN+VALID resolution."""
    from heartless import cli
    from heartless.core.models import Candle
    from heartless.learning import lab

    db = tmp_path / "c.db"
    cs = Store(db)
    for sym in ("BIGUSDT", "GONEUSDT", "NEWUSDT"):
        cs.save_candles(sym, [Candle(M("2023-12") + 60_000 * i, 1, 1, 1, 1, 1, 1, 1, 1, M("2023-12") + 60_000 * i
                                     + 59_999) for i in range(3)])
    cs.close()
    settings = Settings(_env_file=None, HEARTLESS_DATA_DIR=str(tmp_path), UNIVERSE_DB=str(udb.path))
    monkeypatch.setattr(settings.__class__, "db_path", property(lambda self: db))
    monkeypatch.setattr(lab, "UNIVERSES", dict(lab.UNIVERSES))
    res = cli._top_universe(settings, SimpleNamespace(universe="top2", train="2023-12-01:2024-01-01",
                                                      valid="2024-03-01:2024-04-01"), "discover")
    assert set(res["research"]) == {"BIGUSDT", "GONEUSDT"} and lab.UNIVERSES["top2"] == ["BIGUSDT", "GONEUSDT"]
    err = capsys.readouterr().err
    assert "TRAIN window" in err and "look-ahead" in err
    whole = U.resolve_for_run(udb.path, db, M("2023-12"), M("2024-04") - 1, 2)
    assert "NEWUSDT" in whole["research"]  # what the old TRAIN+VALID resolution let in


def test_relisting_inside_a_month_at_exactly_the_gap(tmp_path):
    """No trade from 2023-10-03 13:00 to 2023-10-10 11:00: exactly 7 days between two traded hours, but only six whole
    calendar days without a trade. The block must still be decoded and the relisting found."""
    month = "2023-10"
    rows = SI.kline_rows(_zip(kline_csv(month, 10.0, 1e6)), month)
    a, b = M(month) + 2 * MS_DAY + 12 * 3_600_000, M(month) + 9 * MS_DAY + 12 * 3_600_000
    for r in rows:
        if a < int(r[0]) < b:
            r[7] = r[10] = "0"
    blob, s = SI.encode_month(rows, month)
    st = Store(tmp_path / "u.db")
    st.save_symbol_inventory([_inv("GAPUSDT", "2023-10", "2023-11")])
    st.save_kline_blocks(SI.FUT_TABLE, [("GAPUSDT", M(month), s["n"], s["first_ts"], s["last_ts"], s["first_trade"],
                                         s["last_trade"], s["med_dqv"], s["days_traded"], blob)])
    st.refresh_inventory_trades()
    assert s["days_traded"] == 31 - 6 and U.relistings(st) == {"GAPUSDT": [b]}
    st.close()

"""F0 symbol inventory and 1h research datasets: archive listing, symbol parsing, the compact monthly block codec,
resumable builds (futures with funding, spot), the basis helper and the inventory report. No network: a fake archive
serves zipped CSVs and bucket listings."""
import asyncio
import io
import json
import zipfile
from datetime import datetime, timezone

import httpx
import numpy as np
import pytest

from heartless.core.store import Store
from heartless.data import symbols_inventory as SI
from heartless.data.archive import BinanceArchive, monthly_funding_path, monthly_kline_path

H = 3_600_000
DAY = 86_400_000


def _ms(y, m, d=1, hh=0):
    return int(datetime(y, m, d, hh, tzinfo=timezone.utc).timestamp() * 1000)


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("x.csv", text)
    return buf.getvalue()


def kline_csv(month: str, price: float, qv: float, days: range | None = None, decimals: int = 4,
              header: bool = True, micro: bool = False, zero: bool = False) -> str:
    """Hourly archive rows of `month`: a slow drift around `price`, quote volume `qv` per hour (0 when `zero`), only on
    `days` (1-based) when given."""
    a, b = SI._month_ms(month), SI._next_month_ms(month)
    lines = ["open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,"
             "taker_buy_quote_volume,ignore"] if header else []
    i = 0
    for t in range(a, b, H):
        day = (t - a) // DAY + 1
        if days is not None and day not in days:
            continue
        c = price * (1 + 0.01 * np.sin(i / 7))
        o = price * (1 + 0.01 * np.sin((i - 1) / 7))
        hi, lo = max(o, c) * 1.002, min(o, c) * 0.998
        q = 0.0 if zero else qv * (1 + 0.3 * np.sin(i / 5))
        ts = t * 1000 if micro else t
        f = f"{{:.{decimals}f}}"
        lines.append(f"{ts},{f.format(o)},{f.format(hi)},{f.format(lo)},{f.format(c)},{q / c:.3f},{ts + H - 1},"
                     f"{q:.5f},{0 if zero else 10},{q / c / 2:.3f},{q * 0.55:.5f},0")
        i += 1
    return "\n".join(lines)


def funding_csv(month: str, rate: str = "0.00010000", every_h: int = 8) -> str:
    a, b = SI._month_ms(month), SI._next_month_ms(month)
    rows = [f"{t + 3},{every_h},{rate}" for t in range(a, b, every_h * H)]
    return "calc_time,funding_interval_hours,last_funding_rate\n" + "\n".join(rows)


class FakeArchive(BinanceArchive):
    def __init__(self, objects: dict[str, bytes], listing: dict[str, list[str]] | None = None):
        super().__init__()
        self.objects = objects
        self.listing = listing or {}
        self.requested: list[str] = []

    async def get(self, path: str):
        self.requested.append(path)
        return self.objects.get(path)

    async def list_prefix(self, prefix, delimiter="/", bases=()):
        if prefix in self.listing:  # directory listing
            return [prefix + s + "/" for s in self.listing[prefix]], []
        keys = [(k, len(v)) for k, v in self.objects.items() if k.startswith(prefix)]
        return [], keys + [(k + ".CHECKSUM", 10) for k, _ in keys]


def test_parse_symbol_kinds_quotes_and_multipliers():
    p = SI.parse_symbol
    assert p("BTCUSDT") == {"symbol": "BTCUSDT", "kind": "perpetual", "quote": "USDT", "base": "BTC", "asset": "BTC",
                            "multiplier": 1.0}
    assert p("BTCUSDT_240329")["kind"] == "delivery" and p("BTCUSDT_240329")["quote"] == "USDT"
    for s in ("BNXUSDTSETTLED", "AERGOUSDTSETTLEDSETTLED", "ICPUSDT_SETTLED"):
        r = p(s)
        assert r["kind"] == "settled" and r["quote"] == "USDT", r
    assert p("ICPUSDT_SETTLED")["base"] == "ICP"
    assert (p("1000PEPEUSDT")["asset"], p("1000PEPEUSDT")["multiplier"]) == ("PEPE", 1000.0)
    assert (p("1000000BOBUSDT")["asset"], p("1000000BOBUSDT")["multiplier"]) == ("BOB", 1e6)
    assert (p("1MBABYDOGEUSDT")["asset"], p("1MBABYDOGEUSDT")["multiplier"]) == ("BABYDOGE", 1e6)
    assert (p("1INCHUSDT")["asset"], p("1INCHUSDT")["multiplier"]) == ("1INCH", 1.0)  # a coin name, not a multiplier
    assert p("ETHBTC")["quote"] == "BTC" and p("BTCUSDC")["quote"] == "USDC" and p("BTCUSD1")["quote"] == "USD1"
    assert p("BTCU")["quote"] == "U" and p("ETHFDUSD")["quote"] == "FDUSD"


def test_list_prefix_follows_continuation_tokens():
    pages = {
        None: ("<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'><IsTruncated>true</IsTruncated>"
               "<NextContinuationToken>tok1</NextContinuationToken><CommonPrefixes><Prefix>p/AUSDT/</Prefix>"
               "</CommonPrefixes><Contents><Key>p/x.zip</Key><Size>5</Size></Contents></ListBucketResult>"),
        "tok1": ("<ListBucketResult xmlns='http://s3.amazonaws.com/doc/2006-03-01/'><IsTruncated>false</IsTruncated>"
                 "<CommonPrefixes><Prefix>p/BUSDT/</Prefix></CommonPrefixes></ListBucketResult>"),
    }
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        seen.append(q)
        assert q["list-type"] == "2" and q["prefix"] == "p/" and q["delimiter"] == "/"
        return httpx.Response(200, content=pages[q.get("continuation-token")].encode())

    async def run():
        arch = BinanceArchive()
        await arch._client.aclose()
        arch._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        res = await arch.list_prefix("p/", bases=("https://bucket.example",))
        await arch.close()
        return res

    prefixes, keys = asyncio.run(run())
    assert prefixes == ["p/AUSDT/", "p/BUSDT/"] and keys == [("p/x.zip", 5)]
    assert [q.get("continuation-token") for q in seen] == [None, "tok1"]


def test_block_codec_is_exact_for_prices_and_close_for_volumes():
    month = "2024-02"
    for price, dec in ((42150.1, 1), (0.0012345, 7), (3.1, 6)):
        rows = SI.kline_rows(_zip(kline_csv(month, price, 1e6, decimals=dec)), month)
        blob, s = SI.encode_month(rows, month, SI.funding_rows(_zip(funding_csv(month, "-0.00012345"))))
        d = SI.decode_month(blob)
        assert len(d["open_time"]) == len(rows) == 29 * 24
        for j, name in ((1, "open"), (2, "high"), (3, "low"), (4, "close")):
            assert all(d[name][i] == float(r[j]) for i, r in enumerate(rows)), name  # the same float64 as the CSV
        qv = np.array([float(r[7]) for r in rows])
        assert np.max(np.abs(d["quote_volume"] / qv - 1)) <= SI.QV_STEP / 2 + 1e-12
        assert np.allclose(d["taker_buy_share"], 0.55, atol=6e-4)
        f = d["funding"]
        assert np.isfinite(f).sum() == 29 * 3 and np.all(f[np.isfinite(f)] == -0.00012345)
        assert np.all((d["open_time"][np.isfinite(f)] - SI._month_ms(month)) % (8 * H) == 0)  # at the bar's open
        assert s["n"] == 29 * 24 and s["days_traded"] == 29 and s["first_trade"] == SI._month_ms(month)
    # funding rates in exponent notation (the archive writes small rates as e.g. 8.3E-7) stay exact
    rows = SI.kline_rows(_zip(kline_csv(month, 3.1, 1e6)), month)
    d = SI.decode_month(SI.encode_month(rows, month, SI.funding_rows(_zip(funding_csv(month, "8.3E-7"))))[0])
    assert np.all(d["funding"][np.isfinite(d["funding"])] == float("8.3E-7"))
    # spot: microsecond timestamps, no header, close and quote volume only
    rows = SI.kline_rows(_zip(kline_csv(month, 2.5, 5e5, header=False, micro=True)), month)
    d = SI.decode_month(SI.encode_month(rows, month, spot=True)[0])
    assert d["open_time"][0] == SI._month_ms(month) and set(d) == {"open_time", "close", "quote_volume"}
    # more than 12 decimals: stored as float64, still exact
    rows = SI.kline_rows(_zip(kline_csv(month, 0.5, 1e3, decimals=14)), month)
    d = SI.decode_month(SI.encode_month(rows, month)[0])
    assert all(d["close"][i] == float(r[4]) for i, r in enumerate(rows))


def test_median_daily_volume_counts_missing_days_as_zero():
    month = "2024-03"  # 31 days
    full = SI.encode_month(SI.kline_rows(_zip(kline_csv(month, 10, 1000.0)), month), month)[1]
    late = SI.encode_month(SI.kline_rows(_zip(kline_csv(month, 10, 1000.0, days=range(20, 32))), month), month)[1]
    early_gone = SI.encode_month(SI.kline_rows(_zip(kline_csv(month, 10, 1000.0, days=range(1, 10))), month), month)[1]
    flat = SI.encode_month(SI.kline_rows(_zip(kline_csv(month, 10, 1000.0, zero=True)), month), month)[1]
    assert full["med_dqv"] > 20_000 and full["days_traded"] == 31
    assert late["med_dqv"] == 0.0 and late["days_traded"] == 12  # listed on the 20th: 19 empty days
    assert early_gone["med_dqv"] == 0.0 and early_gone["last_trade"] < SI._month_ms(month) + 9 * DAY
    assert flat["med_dqv"] == 0.0 and flat["first_trade"] is None and flat["n"] == 31 * 24


def _world():
    """Archive objects + listings: a perpetual with a 1000x spot pair, a listing inside the window, a symbol delisted
    in the window whose flat files continue, a settled copy, a delivery contract and a USDC perpetual."""
    objs = {}
    listing = {SI.FUTURES_KLINES: ["AAAUSDT", "1000PEPEUSDT", "NEWUSDT", "GONEUSDT", "GONEUSDTSETTLED",
                                   "AAAUSDT_240329", "AAAUSDC"],
               SI.SPOT_KLINES: ["AAAUSDT", "PEPEUSDT", "GONEUSDT", "OTHERBTC"]}
    months = ["2023-11", "2023-12", "2024-01", "2024-02"]
    for m in months:
        objs[monthly_kline_path("AAAUSDT", m)] = _zip(kline_csv(m, 100.0, 5e6))
        objs[monthly_funding_path("AAAUSDT", m)] = _zip(funding_csv(m))
        objs[monthly_kline_path("1000PEPEUSDT", m)] = _zip(kline_csv(m, 1.234, 2e6, decimals=7))
        objs[monthly_kline_path("AAAUSDT", m, market="spot")] = _zip(kline_csv(m, 99.9, 1e6, header=False))
        objs[monthly_kline_path("PEPEUSDT", m, market="spot")] = _zip(kline_csv(m, 0.001233, 1e6, decimals=8))
        objs[monthly_kline_path("AAAUSDC", m)] = _zip(kline_csv(m, 100.0, 1e6))
        objs[monthly_kline_path("AAAUSDT_240329", m)] = _zip(kline_csv(m, 101.0, 1e5))
        objs[monthly_kline_path("GONEUSDTSETTLED", m)] = _zip(kline_csv(m, 5.0, 1e6))
    objs[monthly_kline_path("GONEUSDT", "2023-11")] = _zip(kline_csv("2023-11", 5.0, 3e6))
    objs[monthly_kline_path("GONEUSDT", "2023-12")] = _zip(kline_csv("2023-12", 5.0, 3e6, days=range(1, 11)))
    objs[monthly_kline_path("GONEUSDT", "2024-01")] = _zip(kline_csv("2024-01", 5.0, 0, zero=True))  # flat file
    # its spot market keeps trading after the delisting, far from the perpetual's frozen price
    objs[monthly_kline_path("GONEUSDT", "2023-12", market="spot")] = _zip(kline_csv("2023-12", 5.0, 1e5))
    objs[monthly_kline_path("GONEUSDT", "2024-01", market="spot")] = _zip(kline_csv("2024-01", 50.0, 1e5))
    objs[monthly_kline_path("NEWUSDT", "2024-01")] = _zip(kline_csv("2024-01", 7.0, 9e6, days=range(15, 32)))
    objs[monthly_kline_path("NEWUSDT", "2024-02")] = _zip(kline_csv("2024-02", 7.0, 9e6))
    return objs, listing


@pytest.fixture
def built(tmp_path):
    objs, listing = _world()
    st = Store(tmp_path / "u.db")
    arch = FakeArchive(objs, listing)

    async def run():
        await SI.refresh_inventory(st, arch)
        fut = await SI.build_futures(st, arch, SI._month_ms("2023-12"), SI._month_ms("2024-02") + 10 * DAY)
        spot = await SI.build_spot(st, arch, SI._month_ms("2023-12"), SI._month_ms("2024-02") + 10 * DAY)
        return fut, spot

    fut, spot = asyncio.run(run())
    yield st, arch, fut, spot
    st.close()


def test_inventory_rows_months_spot_pairs_and_kinds(built):
    st, _, _, _ = built
    inv = {r["symbol"]: r for r in st.load_symbol_inventory()}
    assert set(inv) == {"AAAUSDT", "1000PEPEUSDT", "NEWUSDT", "GONEUSDT", "GONEUSDTSETTLED", "AAAUSDT_240329",
                        "AAAUSDC"}
    assert (inv["AAAUSDT"]["first_month"], inv["AAAUSDT"]["last_month"], inv["AAAUSDT"]["months"]) == \
        ("2023-11", "2024-02", 4)
    assert inv["AAAUSDT"]["spot_symbol"] == "AAAUSDT" and inv["1000PEPEUSDT"]["spot_symbol"] == "PEPEUSDT"
    assert inv["NEWUSDT"]["spot_symbol"] is None and inv["AAAUSDC"]["spot_symbol"] is None  # USDC perps: no pairing
    assert inv["GONEUSDTSETTLED"]["kind"] == "settled" and inv["AAAUSDT_240329"]["kind"] == "delivery"
    assert inv["NEWUSDT"]["first_month"] == "2024-01" and inv["NEWUSDT"]["missing_months"] == []


def test_futures_build_includes_the_month_before_start_and_resumes(built):
    st, arch, fut, _ = built
    idx = st.kline_block_index(SI.FUT_TABLE)
    got = {(r["symbol"], SI.month_str(r["month"])) for r in idx}
    # USDT perpetuals only, from 2023-11 (the month before the 2023-12 start: the ranking of December reads it)
    assert {s for s, _ in got} == {"AAAUSDT", "1000PEPEUSDT", "NEWUSDT", "GONEUSDT"}
    assert ("AAAUSDT", "2023-11") in got and ("AAAUSDT", "2024-02") in got and fut["failed"] == []
    inv = {r["symbol"]: r for r in st.load_symbol_inventory()}
    assert inv["GONEUSDT"]["last_trade"] < SI._month_ms("2023-12") + 10 * DAY  # flat January file ignored
    assert inv["NEWUSDT"]["first_trade"] == SI._month_ms("2024-01") + 14 * DAY
    arch.requested.clear()
    again = asyncio.run(SI.build_futures(st, arch, SI._month_ms("2023-12"), SI._month_ms("2024-02") + 10 * DAY))
    assert again["blocks"] == 0 and arch.requested == []  # nothing left to download


def test_spot_build_and_basis_series(built):
    st, _, _, spot = built
    assert {r["symbol"] for r in st.kline_block_index(SI.SPOT_TABLE)} == {"AAAUSDT", "PEPEUSDT", "GONEUSDT"}
    # the window's months only (the futures build adds the month before for the ranking; a basis does not need it)
    assert {SI.month_str(r["month"]) for r in st.kline_block_index(SI.SPOT_TABLE)} == {"2023-12", "2024-01", "2024-02"}
    b = SI.basis_series(st, "1000PEPEUSDT", SI._month_ms("2024-01"), SI._month_ms("2024-02") - 1)
    assert len(b["ts"]) == 31 * 24 and b["ts"][0] == SI._month_ms("2024-01")
    assert np.allclose(b["spot"], b["perp"], rtol=0.01)  # 1000 x 0.001233 against 1.234: paired via the multiplier
    assert np.allclose(b["basis"], (b["perp"] - b["spot"]) / b["spot"]) and np.all(np.abs(b["basis"]) < 0.01)
    a = SI.basis_series(st, "AAAUSDT", SI._month_ms("2024-01"), SI._month_ms("2024-01") + 2 * DAY)
    settled = a["ts"][np.isfinite(a["funding"])]
    assert len(settled) == 7 and np.all(settled % (8 * H) == 0)  # T in [01-01 00:00, 01-03 00:00]: 2 days + 1
    assert np.all(a["funding_last"] == 0.0001)  # known from the first row on (preloaded a day back)
    assert SI.basis_series(st, "NEWUSDT")["ts"].size == 0  # no spot market
    cov = SI.spot_coverage(st)
    # the pairing check reads only hours both markets traded: GONE's frozen January perpetual does not count
    g = SI.basis_series(st, "GONEUSDT")
    assert np.all(g["perp_qv"][g["ts"] > SI._month_ms("2024-01") + H] == 0) and np.median(np.abs(g["basis"])) > 0.5
    assert cov["with_stored_spot"] == 3 and cov["paired_ok"] == 3 and cov["suspect_pairs"] == []
    assert cov["symbols"]["GONEUSDT"]["hours_both_traded"] == 10 * 24


def test_inventory_report_counts_listings_and_delistings(built):
    st, _, _, _ = built
    rep = SI.inventory_report(st)
    c = rep["counts"]
    assert c["symbols"] == 7 and c["perpetual_usdt"] == 4 and c["by_kind"] == {"delivery": 1, "perpetual": 5,
                                                                                 "settled": 1}
    assert c["perpetual_usdt_status"] == {"active": 3, "delisted": 1}
    assert rep["perpetual_usdt_delisted_per_year"] == {"2023": 1}
    assert rep["perpetual_usdt_listed_per_year"] == {"2023": 3, "2024": 1}
    line = {r[0]: r for r in rep["symbols"]}["GONEUSDT"]
    assert line[7] == "delisted" and line[6] == "2023-12-10"
    text = SI.dumps_report(rep)  # the report file: one line per symbol, same content
    assert json.loads(text) == json.loads(json.dumps(rep))
    big = SI.dumps_report({"symbols": [[i, "x"] for i in range(20)], "n": 1})
    assert len(big.splitlines()) == 20 + 5 and json.loads(big)["symbols"][19] == [19, "x"]

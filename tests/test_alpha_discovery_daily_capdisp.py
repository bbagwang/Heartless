"""discovery_daily_capdisp: the hard-coded discovered rule rc899282d (24h capitulation with mean-reverting 15m bars
while the market disperses -> long, 4 1h-ATR stop, 24h time stop). Registration and the mined thresholds, the Signal
geometry on a real MarketView over synthetic 1m data, every gate, causality (the decision at an hour close does not
change when later bars are appended) and equality with the `discovered` alpha running the same rule (the alpha that
was evaluated in the lab), hour close by hour close."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from heartless.core.models import EntryStyle, Regime, Side, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.learning.discovery import WARMUP_MS, decision_features
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.alphas.discovered import Discovered
from heartless.strategy.alphas.discovery_daily_capdisp import FEATURES, RULE_ID, DiscoveryDailyCapdisp
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
DAYS = 30  # the last day is the capitulation day; the decision is at its 23:00-23:59 hour close
INFO = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                  price_precision=2, quantity_precision=3)
MINED = {"ret24_max": -0.067328, "hurst_max": 0.336784, "disp_min": 0.031298}  # rule rc899282d as mined
CAPIT = {"ret_24h": -0.10, "disp_24h": 0.05}  # a context snapshot that satisfies both market conditions
T_CAPIT = DAY0 + DAYS * MS_DAY - 1  # the hour close ending the capitulation day


def params(**over) -> dict:
    p = dict(StrategyParams.default().alphas["discovery_daily_capdisp"])
    p.update(over)
    return p


def ctx(price: float, t: int, market: dict | None = None) -> Context:
    return Context(symbol=SYM, info=INFO, ticker=Ticker(SYM, bid=price, ask=price, mark=price, last=price, ts=t),
                   regime=Regime.VOLATILE, now=t, market=dict(CAPIT if market is None else market))


def _candles(logret: np.ndarray, start: int = DAY0) -> CandleArrays:
    """1m bars from per-minute log returns (open = previous close, a 1bp wick each side)."""
    close = 100.0 * np.exp(np.cumsum(logret))
    n = len(close)
    ca = CandleArrays("1m", capacity=n + 16)
    o = np.r_[100.0, close[:-1]]
    ca.open_time[:n] = start + np.arange(n, dtype=np.int64) * MS_MINUTE
    ca.close_time[:n] = ca.open_time[:n] + MS_MINUTE - 1
    ca.open[:n], ca.close[:n] = o, close
    ca.high[:n], ca.low[:n] = np.maximum(o, close) * 1.0001, np.minimum(o, close) * 0.9999
    ca.volume[:n], ca.quote_volume[:n], ca.trades[:n], ca.taker_buy[:n] = 10.0, 10.0 * close, 10, 5.0
    ca.n = n
    return ca


def _path(extra_days: int = 0, seed: int = 7) -> np.ndarray:
    """DAYS - 1 days of minute noise, then a capitulation day: -8% over 24 hours with every 15m bar reversing the last
    one by about 1.2% (a strongly mean-reverting 15m path, Hurst proxy ~0.05), then `extra_days` of a further slide."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.0006, (DAYS + extra_days) * 1440)
    s = (DAYS - 1) * 1440
    m = np.arange(1440)
    r[s:s + 1440] = -0.08 / 1440 + 0.012 * np.where((m // 15) % 2 == 0, 1.0, -1.0) / 15 + rng.normal(0, 0.0002, 1440)
    if extra_days:
        r[DAYS * 1440:] -= 0.0003  # what comes after must not change the decision
    return r


def _view(ca: CandleArrays) -> MarketView:
    v = MarketView(SYM)
    v.rebuild(ca, live=False)
    return v


def _decide(view: MarketView, t: int, alpha=None, market: dict | None = None, **over):
    view.seek(t)
    px = view.tf("1h").v("close")
    return (alpha or DiscoveryDailyCapdisp()).evaluate(view, ctx(px, t, market), params(**over))


@pytest.fixture(scope="module")
def view() -> MarketView:
    return _view(_candles(_path()))


# --- registration -------------------------------------------------------------------------------------------------
def test_registered_1h_off_by_default_with_the_mined_rule_and_the_miners_warm_up():
    a = ALPHA_BY_NAME["discovery_daily_capdisp"]
    assert isinstance(a, DiscoveryDailyCapdisp) and a.timeframe == "1h" and a.enabled_by_default is False
    assert StrategyParams.default().enabled["discovery_daily_capdisp"] is False
    p = params()
    assert {k: p[k] for k in MINED} == MINED and p["sl_atr"] == 4.0 and p["hold_h"] == 24
    for s in a.param_specs:
        assert s.clip(s.default) == s.default
    assert a.lookback_days * MS_DAY == WARMUP_MS  # the same indicator history as the mined rows
    # a winner may run 3x the longest hold: inside the lab's 7-day chunk hand-off, so no max_hold_days is needed
    assert a.max_hold_days is None and 3 * {s.name: s for s in a.param_specs}["hold_h"].hi <= 7 * 24
    assert a.regime_affinity == Discovered.regime_affinity


def test_the_thresholds_are_those_of_the_rule_the_discovery_run_validated():
    path = Path(__file__).resolve().parents[1] / "docs" / "results" / "p2_discovery_daily_discover.json"
    if not path.exists():
        pytest.skip("discovery result file not present")
    rule = next(r["rule"] for r in json.loads(path.read_text())["results"] if r["rule"]["id"] == RULE_ID)
    assert [c[0] for c in rule["conds"]] == FEATURES and [c[1] for c in rule["conds"]] == ["<", "<", ">"]
    assert [c[2] for c in rule["conds"]] == [MINED["ret24_max"], MINED["hurst_max"], MINED["disp_min"]]
    assert (rule["side"], rule["tf"], rule["sl_atr"], rule["tp_r"], rule["hold_min"]) == ("LONG", "1h", 4.0, None, 1440)


# --- signal and gates ---------------------------------------------------------------------------------------------
def test_a_capitulation_with_mean_reverting_15m_bars_goes_long_at_market_behind_a_4_atr_stop(view):
    s = _decide(view, T_CAPIT)
    assert s is not None and s.side is Side.LONG and s.alpha == "discovery_daily_capdisp"
    h1 = view.tf("1h")
    close, atr = h1.v("close"), h1.v("atr")
    assert s.tags["ref_price"] == close and s.atr == atr and s.stop == pytest.approx(close - 4.0 * atr)
    assert s.take_profit is None and s.tp1 is None and s.entry_style is EntryStyle.MARKET
    assert s.max_hold_bars == 24 * 60 and s.trail_atr_mult == 0.0 and s.timeframe == "1h"
    assert s.confidence == pytest.approx(0.66) and s.tags["rule_id"] == RULE_ID and s.tags["time_exit"] is True
    assert s.tags["hurst_15m"] < 0.2 and s.tags["ret_24h"] == CAPIT["ret_24h"] and s.reason
    s2 = _decide(view, T_CAPIT, sl_atr=2.5, hold_h=12)
    assert s2.stop == pytest.approx(close - 2.5 * atr) and s2.max_hold_bars == 12 * 60


def test_every_condition_gates(view):
    assert _decide(view, T_CAPIT) is not None
    assert _decide(view, T_CAPIT, market={"ret_24h": -0.05, "disp_24h": 0.05}) is None  # not a capitulation
    assert _decide(view, T_CAPIT, market={"ret_24h": -0.10, "disp_24h": 0.02}) is None  # a market-wide move
    assert _decide(view, T_CAPIT, market={}) is None  # no context: unknown is no trade
    assert _decide(view, T_CAPIT, market={"ret_24h": float("nan"), "disp_24h": 0.05}) is None
    h = _decide(view, T_CAPIT).tags["hurst_15m"]
    assert _decide(view, T_CAPIT, hurst_max=h) is None and _decide(view, T_CAPIT, hurst_max=h + 1e-9) is not None
    assert _decide(view, T_CAPIT, min_conf=0.7) is None
    view.seek(T_CAPIT - 15 * MS_MINUTE)  # a 15m close inside the hour: the rule decides at hour closes only
    assert not view.closed("1h")
    px = view.tf("15m").v("close")
    assert DiscoveryDailyCapdisp().evaluate(view, ctx(px, T_CAPIT - 15 * MS_MINUTE), params()) is None


def test_a_random_walk_hour_rarely_qualifies_and_warm_up_gives_nothing(view):
    fired = [t for t in range(DAY0 + 3 * MS_DAY - 1, T_CAPIT - MS_DAY, MS_HOUR) if _decide(view, t) is not None]
    assert len(fired) < 0.25 * (DAYS - 4) * 24  # the 15m Hurst proxy of noise sits near 0.5
    cold = _view(_candles(_path()[: 20 * 60]))  # 20 hours: no 100-bar 15m window yet
    assert all(_decide(cold, DAY0 + k * MS_HOUR - 1) is None for k in range(1, 20))


# --- causality and equality with the evaluated alpha ----------------------------------------------------------------
def test_decision_identical_with_and_without_future_data():
    full = _candles(_path(extra_days=2))
    cut = DAYS * 1440  # through the bar closing at T_CAPIT
    head = CandleArrays("1m", capacity=cut + 16)
    head.extend([full.candle_at(i) for i in range(cut)])
    for t in (T_CAPIT, T_CAPIT - 6 * MS_HOUR):
        a, b = _decide(_view(head), t, hurst_max=0.5), _decide(_view(full), t, hurst_max=0.5)
        assert a is not None and b is not None
        assert (a.stop, a.atr, a.max_hold_bars, a.tags, a.reason) == (b.stop, b.atr, b.max_hold_bars, b.tags, b.reason)


def test_the_hurst_condition_reads_the_15m_frame_as_the_miner_does(view):
    view.seek(T_CAPIT)
    x = decision_features(view, {}, CAPIT, T_CAPIT, FEATURES)
    assert list(x[[0, 2]]) == [CAPIT["ret_24h"], CAPIT["disp_24h"]]
    assert _decide(view, T_CAPIT).tags["hurst_15m"] == x[1] and math.isfinite(x[1])


def test_same_signals_as_the_discovered_alpha_running_the_rule(view):
    """Hour close by hour close over the month, both with the mined thresholds except a median Hurst bar (so that
    both outcomes occur often): the candidate fires exactly when `discovered` fires, with the same bracket."""
    over = {"hurst_max": 0.48}
    rule = {"id": RULE_ID, "side": "LONG", "tf": "1h", "sl_atr": 4.0, "tp_r": None, "hold_min": 1440,
            "conds": [["c.ret_24h", "<", MINED["ret24_max"]], ["15m.hurst", "<", over["hurst_max"]],
                      ["m.disp_24h", ">", MINED["disp_min"]]]}
    disc, cand = Discovered(), DiscoveryDailyCapdisp()
    pd = {**StrategyParams.default().alphas["discovered"], "rules": [rule]}
    fired = 0
    for t in range(DAY0 + 2 * MS_DAY - 1, T_CAPIT + 1, MS_HOUR):
        for market in (CAPIT, {"ret_24h": -0.05, "disp_24h": 0.05}):
            view.seek(t)
            c = ctx(view.tf("1h").v("close"), t, market)
            a, b = disc.evaluate(view, c, pd), cand.evaluate(view, c, params(**over))
            assert (a is None) == (b is None), t
            if a is None:
                continue
            fired += 1
            assert (a.side, a.stop, a.take_profit, a.tp1, a.entry_style, a.max_hold_bars, a.trail_atr_mult, a.atr,
                    a.timeframe, a.confidence) == (b.side, b.stop, b.take_profit, b.tp1, b.entry_style, b.max_hold_bars,
                                                   b.trail_atr_mult, b.atr, b.timeframe, b.confidence)
            assert a.tags["ref_price"] == b.tags["ref_price"]
    assert 20 <= fired < (DAYS - 2) * 24


@pytest.mark.parametrize("field", ["hold_h"])
def test_integer_params_stay_integer(field):
    spec = {s.name: s for s in DiscoveryDailyCapdisp.param_specs}[field]
    assert isinstance(spec.clip(spec.default + 0.4), int)

"""P2 gatekeeper checks (docs/RESEARCH_P2.md): a causality sweep over the four TRAIN-gate candidates and the evidence
registry entries of the P2 judgement against docs/results/p2_decisions.json.

Causality: each candidate's own tests compare one decision with and without later bars. The sweep here repeats that at
many decision closes per alpha, always cutting the history exactly at the decision bar: a feature that read a later
bar (a negative offset, a frame bar that has not closed, a centred window) can only show up as a difference at the
cut, never at a decision that both histories already contain. Thresholds are relaxed so the decisions fire often;
ctx.market / ctx.extras are fixed inputs here (their own causality is tested with the context store and MetricsSeries).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from heartless.core.models import Regime, SymbolInfo, Ticker
from heartless.data.candles import CandleArrays
from heartless.data.features import MarketView
from heartless.learning import trials
from heartless.strategy import evidence as E
from heartless.strategy.alphas import ALPHA_BY_NAME
from heartless.strategy.base import Context
from heartless.strategy.params import StrategyParams
from heartless.util.timeutil import MS_DAY, MS_HOUR, MS_MINUTE

ROOT = Path(__file__).resolve().parents[1]
SYM = "ETHUSDT"
DAY0 = 1_704_067_200_000  # 2024-01-01 00:00 UTC
INFO = SymbolInfo(SYM, "ETH", "USDT", tick_size=0.01, step_size=0.001, min_qty=0.001, min_notional=5.0,
                  price_precision=2, quantity_precision=3)

# alpha -> (history days, decision period, first decision day, relaxed params, ctx.market, ctx.extras)
CASES = {
    "ts_trend_tsmom": (70, MS_DAY, 32, {"z_min": 0.25, "fund_max": 0.0}, {}, {}),
    "positioning_flow_oimom": (40, 4 * MS_HOUR, 18, {"m_min": 0.5, "retail_filter": 0, "fund_max": 0.0}, {},
                               {"oi_chg_24h": 0.2, "ls_acc_chg_4h": 0.0, "age_min": 1.0}),
    "vol_breakout_dbox": (70, 4 * MS_HOUR, 25, {"comp_q": 1.0, "btc_filter": 0, "fund_max": 0.0}, {}, {}),
    "discovery_daily_capdisp": (26, MS_HOUR, 23, {"hurst_max": 0.5}, {"ret_24h": -0.10, "disp_24h": 0.05}, {}),
}


def _path(days: int, seed: int) -> CandleArrays:
    """1m bars: minute noise (~2.3% a day) on alternating 12-day up / down drifts of 1.5% a day (trends, reversals and
    box breaks for every candidate), with an intraday swing so the daily range is not just the drift."""
    rng = np.random.default_rng(seed)
    n = days * 1440
    m = np.arange(n)
    drift = np.where((m // (12 * 1440)) % 2 == 0, 0.015, -0.015) / 1440
    r = drift + rng.normal(0.0, 0.0006, n) + 0.004 * np.sin(2 * np.pi * m / 1440) / 1440 * 2 * np.pi
    close = 100.0 * np.exp(np.cumsum(r))
    o = np.r_[100.0, close[:-1]]
    ca = CandleArrays("1m", capacity=n + 16)
    ca.open_time[:n] = DAY0 + m.astype(np.int64) * MS_MINUTE
    ca.close_time[:n] = ca.open_time[:n] + MS_MINUTE - 1
    ca.open[:n], ca.close[:n] = o, close
    ca.high[:n], ca.low[:n] = np.maximum(o, close) * 1.0002, np.minimum(o, close) * 0.9998
    ca.volume[:n], ca.quote_volume[:n], ca.trades[:n], ca.taker_buy[:n] = 10.0, 10.0 * close, 10, 5.0
    ca.n = n
    return ca


def _decide(alpha, view: MarketView, t: int, params: dict, market: dict, extras: dict):
    view.seek(t)
    px = view.tf("1m").v("close")
    ctx = Context(symbol=SYM, info=INFO, ticker=Ticker(SYM, bid=px, ask=px, mark=px, last=px, ts=t),
                  regime=Regime.VOLATILE, now=t, funding_rate=0.0, extras=dict(extras), market=dict(market))
    return alpha.evaluate(view, ctx, params)


def _key(s) -> tuple | None:
    if s is None:
        return None
    return (s.side, s.stop, s.atr, s.max_hold_bars, s.timeframe, s.confidence, s.entry_style, s.trail_atr_mult,
            json.dumps(s.tags, sort_keys=True, default=str))


@pytest.mark.parametrize("name", sorted(CASES))
def test_decisions_cut_at_the_decision_bar_equal_the_full_history(name):
    days, period, first_day, over, market, extras = CASES[name]
    alpha = ALPHA_BY_NAME[name]
    params = {**StrategyParams.default().alphas[name], **over}
    full_ca = _path(days, seed=sum(map(ord, name)))
    full = MarketView(SYM)
    full.rebuild(full_ca, live=False)
    times = list(range(DAY0 + first_day * MS_DAY + period - 1, DAY0 + days * MS_DAY - 1, period))
    decided = {t: _key(_decide(alpha, full, t, params, market, extras)) for t in times}
    fired = [t for t in times if decided[t] is not None]
    quiet = [t for t in times if decided[t] is None]
    assert len(fired) >= 5, f"{name}: the relaxed settings should fire on the synthetic path ({len(fired)})"
    picks = [fired[i] for i in np.linspace(0, len(fired) - 1, 5).astype(int)] + quiet[:: max(1, len(quiet) // 2)][:2]
    for t in sorted(set(picks)):
        n_head = int(np.searchsorted(full_ca.close_time[:full_ca.n], t, side="right"))  # bars closed at or before t
        head = MarketView(SYM)
        head.rebuild(full_ca.slice(0, n_head), live=False)
        assert _key(_decide(alpha, head, t, params, market, extras)) == decided[t], (name, t)


# --- evidence registry: the P2 judgement ---------------------------------------------------------------------------
def _p2() -> dict:
    return json.loads((ROOT / "docs" / "results" / "p2_decisions.json").read_text())


def test_p2_decision_table_applies_the_preregistered_gates():
    d = _p2()
    assert d["valid_gate"] == {"avg_r_gt": 0.0, "t_day_ge": 1.0}
    assert d["train_gate"]["avg_r_ge"] == 0.05 and d["train_gate"]["years_positive_ge"] == 2
    for fam, f in d["families"].items():
        assert f["threshold"] == pytest.approx(trials.threshold(f["n_family"]), abs=1e-3), fam
    for name, c in d["candidates"].items():
        tr, va = c["train"], c["valid"]
        f = d["families"][c["family"]]
        train_ok = (tr["t_day"] >= f["threshold"] and tr["avg_r"] >= 0.05 and tr["years_positive"] >= 2
                    and tr["stress_avg_r"] > 0)
        assert c["checks"]["train"] == train_ok, name
        if va is None:  # not run: only a candidate that cleared TRAIN is evaluated on VALID
            assert not train_ok and c["verdict"] == "failed", name
            continue
        assert train_ok, f"{name}: evaluated on VALID without clearing TRAIN"
        assert va["config_hash"] == tr["config_hash"], f"{name}: VALID ran another configuration"
        valid_ok = va["avg_r"] > 0 and va["t_day"] >= 1.0
        assert c["checks"]["valid"] == valid_ok, name
        assert c["verdict"] == ("passed" if train_ok and valid_ok else "failed"), name


def test_p2_results_and_ledger_back_the_decision_table():
    d = _p2()
    ledger = {(e["config_hash"], e["utc"]) for e in trials.load()}
    for name, c in d["candidates"].items():
        for part in ("train", "valid"):
            r = c[part]
            if r is None:
                continue
            res = json.loads((ROOT / r["file"]).read_text())
            s = res["summary"]
            assert (s["n"], s["avg_r"], s["t_day"]) == (r["n"], r["avg_r"], r["t_day"]), (name, part)
            assert res["trial"]["config_hash"] == r["config_hash"] and res["alpha"] == name, (name, part)
            assert (res["trial"]["config_hash"], res["trial"]["utc"]) in ledger, (name, part)
            assert res["seal"]["clean"] and not res["seal"]["unsealed"], (name, part)
            assert res["trial"]["period"] == part.upper(), (name, part)


def test_registry_carries_the_p2_verdicts_and_expected_r():
    d = _p2()
    reg = E.registry()["alphas"]
    for name, c in d["candidates"].items():
        e = reg[name]
        assert e["judgement"] == "p2" and e["verdict"] == c["verdict"] and e["family"] == c["family"], name
        assert e["param_schema"] == E.param_fingerprint(name), name
        tr, va = c["train"], c["valid"]
        assert e["train"]["n"] == tr["n"] and e["train"]["avg_r"] == tr["avg_r"] and e["train"]["t_day"] == tr["t_day"]
        if va is not None:
            assert e["valid"]["n"] == va["n"] and e["valid"]["avg_r"] == va["avg_r"]
            assert e["valid"]["t_day"] == va["t_day"]
            # the kill switch expects the lower of the two means (evidence.expected_r reads avg_r)
            assert e["avg_r"] == min(tr["avg_r"], va["avg_r"]), name
        assert E.verdict(name) == c["verdict"]
        assert E.expected_r(name) == (e["avg_r"] if c["verdict"] == "passed" else 0.0), name
        # only what passed both gates ships switched on
        assert ALPHA_BY_NAME[name].enabled_by_default is (c["verdict"] == "passed"), name
        assert StrategyParams.default().enabled[name] is (c["verdict"] == "passed"), name

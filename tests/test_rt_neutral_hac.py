"""Registered amendment 2026-10-16 (ROADMAP section 6): the neutral gates use the Newey-West HAC t of the daily net
PnL (Bartlett kernel, fixed lag 14 days); the plain daily t and the weekly-block t are informational. Synthetic AR(1)
series show why: with positive autocorrelation the plain t overstates the evidence, and at lag 0 the two agree."""
import json
import math

import numpy as np
import pytest

from heartless.execution.stats import hac_t
from heartless.learning import neutral as N
from heartless.learning import trials

D = 86_400_000
TRAIN_START = trials.WAVE_F["TRAIN-F"][0]
VALID_START = trials.WAVE_F["VALID-F"][0]


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "LEDGER", tmp_path / "p2_trials.jsonl")


def _ar1(n: int, rho: float, mean: float, seed: int, sd: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sd, n)
    x = np.empty(n)
    x[0] = e[0] / math.sqrt(max(1 - rho * rho, 1e-12))
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return x + mean


def _result(pnl_pct: np.ndarray, start: int = TRAIN_START, capital: float = 100.0, liquidations=()) -> N.NeutralResult:
    """A NeutralResult whose daily net PnL is `pnl_pct` percent of C on consecutive UTC days from `start`."""
    d0 = start // D
    daily = {d0 + i: float(v) / 100 * capital for i, v in enumerate(pnl_pct)}
    eq, e = {}, capital
    for d in sorted(daily):
        e += daily[d]
        eq[d] = e
    return N.NeutralResult("x", "carry_neutral", {"rule": "x"}, d0 * D, (d0 + len(pnl_pct)) * D - 1, capital, daily, eq,
                           [], list(liquidations), 0.0, 0.0, capital, [], [])


def _plain_t(v: np.ndarray) -> float:
    return float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))))


def test_hac_t_equals_the_plain_t_at_lag_0():
    for rho in (0.0, 0.5, 0.9):
        v = _ar1(500, rho, 0.05, seed=1)
        assert hac_t(list(v), 0) == pytest.approx(_plain_t(v), rel=1e-12)
    r = _result(_ar1(400, 0.3, 0.02, seed=2))
    st = r.stats()
    assert st["t_daily"] == pytest.approx(hac_t(list(r.daily_pct()), 0), abs=1e-3)


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.9])
def test_hac_t_is_smaller_than_the_plain_t_under_positive_autocorrelation(rho):
    v = _ar1(1_460, rho, 0.08, seed=3)  # four years of days, a positive mean
    plain, hac = _plain_t(v), hac_t(list(v), N.HAC_LAG_DAYS)
    assert plain > 0 and 0 < hac < plain
    # roughly the long-run variance ratio (1 + rho) / (1 - rho), truncated at lag 14
    assert plain / hac > 1.3
    st = _result(v).stats()
    assert st["t"] == st["t_hac"] == pytest.approx(hac, abs=1e-3)
    assert st["t_daily"] == pytest.approx(plain, abs=1e-3) and st["t"] < st["t_daily"]


def test_zero_edge_false_passes_shrink_with_the_hac_t():
    """A zero-edge AR(1) carry series (rho 0.7) passes t >= 3 far more often on the plain daily t than on the HAC t:
    the red-team estimate was about 10% against the nominal 0.1% (ROADMAP section 6)."""
    plain = hac = 0
    runs = 200
    for k in range(runs):
        v = _ar1(1_460, 0.7, 0.0, seed=100 + k)
        plain += _plain_t(v) >= 3.0
        hac += hac_t(list(v), N.HAC_LAG_DAYS) >= 3.0
    assert plain >= 8 and hac < plain / 3


def test_the_ledger_line_records_the_hac_t():
    r = _result(_ar1(60, 0.7, 0.05, seed=4))
    e = N.record(r, "carry_neutral")
    st = r.stats()
    assert e["t_day"] == st["t_hac"] and e["neutral"]["t_daily"] == st["t_daily"]
    assert e["neutral"]["hac_lag_days"] == 14 and e["neutral"]["weekly_t"] == st["weekly_t"]
    assert "HAC14 t=" in r.report()


def test_the_registration_names_the_hac_t_in_every_neutral_gate():
    reg = json.loads(N.PREREG.read_text(encoding="utf-8"))["gates"]
    for name in ("train_neutral", "valid_neutral", "forward_neutral"):
        assert "t_hac14_daily_net_ge" in reg[name] and "t_daily_net_ge" not in reg[name]
        assert N.registered_gate(name)["t_hac14_daily_net_ge"] == N.DEFAULT_GATES[name]["t_hac14_daily_net_ge"]
    from heartless.learning import forward

    assert "t_hac14_daily_net_ge" in forward.forward_gates()["forward_neutral"]


def test_train_gate_gates_on_the_hac_t_not_the_plain_t():
    days = (trials.WAVE_F["TRAIN-F"][1] - TRAIN_START) // D
    v = _ar1(days, 0.8, 0.008, seed=5, sd=0.05)  # a positive drift with long runs: plain t 6.5, HAC t 2.5
    r = _result(v)
    st = r.stats()
    stress = _result(v - 0.002)
    g = N.train_gate(r, n_family=1, stress=stress)
    key = "HAC t >= 3.00"
    assert st["t_daily"] >= 3.0 > st["t_hac"] > 0  # the plain t would have passed
    assert g["checks"][key] is False and not g["passed"]
    assert g["values"]["t_hac"] == st["t_hac"] and g["values"]["t_daily"] == st["t_daily"]
    assert N.train_gate(r, n_family=1, stress=None)["checks"]["stress annualized > 0"] is False
    big = N.train_gate(r, n_family=10_000, stress=stress)
    assert f"HAC t >= {trials.threshold(10_000):.2f}" in big["checks"]


def test_train_gate_checks_every_registered_condition():
    days = (trials.WAVE_F["TRAIN-F"][1] - TRAIN_START) // D
    v = np.full(days, 0.01) + _ar1(days, 0.0, 0.0, seed=6, sd=0.002)  # 3.65% a year, tiny noise
    r, stress = _result(v), _result(v - 0.005)
    g = N.train_gate(r, n_family=5, stress=stress)
    assert g["passed"], g
    assert set(g["values"]["by_year"]) == {"2022", "2023", "2024", "2025"}
    bad_year = v.copy()
    bad_year[:300] = -0.01  # 2022 loses
    assert not N.train_gate(_result(bad_year), 5, stress)["checks"]["every year net positive"]
    assert not N.train_gate(_result(v, liquidations=[{"x": 1}]), 5, stress)["checks"]["no forced liquidation"]
    assert not N.train_gate(_result(v * 0.3), 5, stress)["checks"]["annualized > 2%"]
    assert not N.train_gate(r, 5, _result(v - 0.02))["checks"]["stress annualized > 0"]


def test_confirmation_gates_valid_quarters_and_forward_span():
    days = (trials.WAVE_F["VALID-F"][1] - VALID_START) // D
    v = np.full(days, 0.01) + _ar1(days, 0.0, 0.0, seed=7, sd=0.002)
    g = N.confirmation_gate(_result(v, start=VALID_START), "valid")
    assert g["passed"] and len(g["values"]["quarters"]) == 3
    w = v.copy()
    w[110:150] = -0.05  # the second quarter loses
    gv = N.confirmation_gate(_result(w, start=VALID_START), "valid")
    assert not gv["checks"]["every quarter net positive"]
    f = N.confirmation_gate(_result(v[:90], start=trials.fwd_span(1)[0]), "forward")
    assert f["passed"] and f["checks"]["span net positive"]
    assert not N.confirmation_gate(_result(-v[:90], start=trials.fwd_span(1)[0]), "forward")["passed"]
    with pytest.raises(ValueError):
        N.confirmation_gate(_result(v), "train")
